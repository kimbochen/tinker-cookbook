"""On-policy Qwen3.6 distillation from domain-specific Caesar and geometry teachers.

Install the evaluation dependency with `pip install -e '.[qwen36-puzzles]'`.
Run with TINKER_API_KEY set and access to both teacher checkpoints:

    python -m tinker_cookbook.recipes.distillation.on_policy_multi_teacher_qwen36

Use `dry_run=True` to validate data and rendering without creating a paid session.
See the distillation README for settings and dataset provenance.
"""

import asyncio
import json
import random
from datetime import datetime
from functools import cache, partial
from pathlib import Path
from typing import Literal

import chz
import tinker
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.distillation.qwen36_puzzles import scoring
from tinker_cookbook.recipes.distillation.qwen36_puzzles.puzzles import (
    DATA_DIR,
    PuzzleRow,
    load_split,
)
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import register_tokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
RENDERER = model_info.get_recommended_renderer_name(MODEL)
CONTEXT_LENGTH = 32768
EVAL_SAMPLES = 3
DOMAINS = ("caesar_cipher", "simple_geometry")
TEACHERS = (
    "tinker://02c4683b-edf3-526b-991d-90c9a0c1b183:train:0/sampler_weights/final",
    "tinker://231067a0-323a-5424-b0f2-62353750d220:train:0/sampler_weights/final",
)


@cache
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


@cache
def renderer() -> renderers.Renderer:
    return renderers.get_renderer(RENDERER, tokenizer(), model_name=MODEL)


def checked_prompt(row: PuzzleRow, max_tokens: int) -> tinker.ModelInput:
    prompt = renderer().build_generation_prompt(row["prompt"])
    expected = tokenizer().apply_chat_template(
        row["prompt"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if prompt.to_ints() != expected:
        raise ValueError("Cookbook renderer differs from the pinned HF thinking template")
    if prompt.length + max_tokens > CONTEXT_LENGTH:
        raise ValueError("Prompt plus response budget exceeds the 32,768-token context")
    return prompt


@chz.chz
class PuzzleDatasetBuilder(RLDatasetBuilder):
    domain: str
    data_dir: Path
    groups_per_batch: int
    max_steps: int
    max_tokens: int
    seed: int

    async def __call__(self) -> tuple[PromptOnlyDataset, None]:
        rows = load_split(self.domain, "train", self.data_dir)
        random.Random(self.seed + DOMAINS.index(self.domain)).shuffle(rows)
        # Match benchmark indexing, including wrapping for a larger step budget.
        selected = [rows[i % len(rows)] for i in range(self.max_steps * self.groups_per_batch)]
        prefix = selected[0]["prompt"][:-1]
        for row in selected:
            messages = row["prompt"]
            if messages[:-1] != prefix or messages[-1]["role"] != "user":
                raise ValueError("Expected a common conversation prefix and final user question")
            checked_prompt(row, self.max_tokens)
        return PromptOnlyDataset(
            prompts=[row["prompt"][-1]["content"] for row in selected],
            batch_size=self.groups_per_batch,
            group_size=1,
            renderer=renderer(),
            tokenizer=tokenizer(),
            max_prompt_tokens=None,
            convo_prefix=prefix,
            dataset_name=self.domain,
        ), None  # Evaluate separately: prompt-only environments intentionally have zero reward.


class DevEvaluator(SamplingClientEvaluator):
    def __init__(self, data_dir: Path, max_tokens: int, concurrency: int):
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.work = {
            domain: [
                (row, checked_prompt(row, max_tokens))
                for row in load_split(domain, "dev", data_dir)
            ]
            for domain in DOMAINS
        }

    async def __call__(
        self, sampling_client: tinker.SamplingClient, *, domains: tuple[str, ...] = DOMAINS
    ) -> dict[str, float]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate(
            row: PuzzleRow, prompt: tinker.ModelInput
        ) -> list[tuple[float, int, bool]]:
            async with semaphore:
                result = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=EVAL_SAMPLES,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=self.max_tokens,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        # Match the benchmark: leave seed unset for multi-response groups.
                        stop=renderer().get_stop_sequences(),
                    ),
                )
            if len(result.sequences) != EVAL_SAMPLES:
                raise ValueError(f"Expected {EVAL_SAMPLES} evaluation responses per prompt")
            return [
                (
                    scoring.score(
                        tokenizer().decode(sequence.tokens, skip_special_tokens=True), row["label"]
                    ),
                    len(sequence.tokens),
                    sequence.stop_reason == "length",
                )
                for sequence in result.sequences
            ]

        metrics = {}
        for domain in domains:
            work = self.work[domain]
            groups = await asyncio.gather(*(evaluate(row, prompt) for row, prompt in work))
            results = [result for group in groups for result in group]
            metrics[f"{domain}/correct"] = sum(score for score, _, _ in results)
            metrics[f"{domain}/accuracy"] = metrics[f"{domain}/correct"] / len(results)
            metrics[f"{domain}/avg@3"] = metrics[f"{domain}/accuracy"]
            metrics[f"{domain}/pass@3"] = sum(
                any(score == 1.0 for score, _, _ in group) for group in groups
            ) / len(groups)
            metrics[f"{domain}/mean_tokens"] = sum(n for _, n, _ in results) / len(results)
            metrics[f"{domain}/truncated"] = sum(truncated for _, _, truncated in results)
            metrics[f"{domain}/n"] = len(groups)
            metrics[f"{domain}/samples"] = len(results)
        return metrics


@chz.chz
class CLIConfig:
    data_dir: Path = DATA_DIR
    log_path: str = chz.field(
        default_factory=lambda: cli_utils.runs_path(
            "distillation", "qwen36-multi-teacher-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
    )
    caesar_teacher: str = TEACHERS[0]
    geometry_teacher: str = TEACHERS[1]
    groups_per_domain: int = 64
    max_steps: int = 20
    max_tokens: int = 30720
    learning_rate: float = 1e-5
    lora_rank: int = 64
    seed: int = 20260921  # Prompt ordering; training and multi-response eval sampling are unseeded.
    eval_every: int = 7  # Completed updates; 0 disables teacher and student evaluation.
    save_every: int = 10
    eval_concurrency: int = 32
    dry_run: bool = False


class RunEvaluator(SamplingClientEvaluator):
    """Evaluate teacher baselines once and students through the cookbook's hook."""

    def __init__(self, cli: CLIConfig):
        self.cli = cli
        self.dev = DevEvaluator(cli.data_dir, cli.max_tokens, cli.eval_concurrency)
        self.next_step = 0
        self.baselines: dict[str, dict[str, float]] = {}

    def record(
        self, role: str, step: int, metrics: dict[str, float], checkpoint: str | None = None
    ) -> None:
        row = {"role": role, "completed_updates": step, "metrics": metrics}
        if checkpoint is not None:
            row["checkpoint"] = checkpoint
        with (Path(self.cli.log_path).expanduser() / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    async def student(
        self, client: tinker.SamplingClient, step: int, checkpoint: str | None = None
    ) -> dict[str, float]:
        metrics = await self.dev(client)
        for domain in DOMAINS:
            baseline = self.baselines[domain][f"{domain}/accuracy"]
            metrics[f"{domain}/teacher_accuracy"] = baseline
            metrics[f"{domain}/student_minus_teacher"] = metrics[f"{domain}/accuracy"] - baseline
        self.record("student", step, metrics, checkpoint)
        return metrics

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        step = self.next_step
        self.next_step += self.cli.eval_every
        if step:
            return await self.student(sampling_client, step)

        # First hook: each frozen teacher is evaluated once on its own full dev split.
        metrics = {}
        service = tinker.ServiceClient()
        status: Literal["success", "errored", "interrupted"] = "errored"
        try:
            for domain, checkpoint in zip(
                DOMAINS, (self.cli.caesar_teacher, self.cli.geometry_teacher), strict=True
            ):
                client = await service.create_sampling_client_async(
                    base_model=MODEL, model_path=checkpoint
                )
                values = await self.dev(client, domains=(domain,))
                self.baselines[domain] = values
                self.record(f"teacher/{domain}", 0, values, checkpoint)
                metrics.update({f"teacher/{domain}/{key}": value for key, value in values.items()})
            status = "success"
        except (asyncio.CancelledError, KeyboardInterrupt):
            status = "interrupted"
            raise
        finally:
            await service.close(status)
        metrics.update(await self.student(sampling_client, 0))
        return metrics

    async def final(self) -> None:
        # The upstream loop stops before invoking the hook at the final update.
        checkpoint = checkpoint_utils.get_last_checkpoint(
            str(Path(self.cli.log_path).expanduser()), "sampler_path"
        )
        if (
            checkpoint is None
            or checkpoint.name != "final"
            or checkpoint.batch != self.cli.max_steps
        ):
            raise RuntimeError("Expected the cookbook's final sampler checkpoint for evaluation")
        service = tinker.ServiceClient()
        status: Literal["success", "errored", "interrupted"] = "errored"
        try:
            client = await service.create_sampling_client_async(model_path=checkpoint.sampler_path)
            await self.student(client, self.cli.max_steps, checkpoint.sampler_path)
            status = "success"
        except (asyncio.CancelledError, KeyboardInterrupt):
            status = "interrupted"
            raise
        finally:
            await service.close(status)


def build_config(cli: CLIConfig) -> train_on_policy.Config:
    for name in ("groups_per_domain", "max_steps", "max_tokens", "lora_rank", "eval_concurrency"):
        if getattr(cli, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < cli.learning_rate < float("inf") or cli.eval_every < 0 or cli.save_every <= 0:
        raise ValueError("Require finite positive learning_rate, eval_every >= 0, save_every > 0")
    teachers = (cli.caesar_teacher, cli.geometry_teacher)
    if any(
        not path.startswith("tinker://") or "/sampler_weights/" not in path for path in teachers
    ):
        raise ValueError("Teachers must be Tinker sampler checkpoint paths")
    return train_on_policy.Config(
        recipe_name="on_policy_multi_teacher_qwen36",
        model_name=MODEL,
        renderer_name=RENDERER,
        lora_rank=cli.lora_rank,
        learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens,
        temperature=1.0,
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        loss_fn="importance_sampling",
        num_substeps=1,
        dataset_configs=[
            DistillationDatasetConfig(
                dataset_builder=PuzzleDatasetBuilder(
                    domain=domain,
                    data_dir=cli.data_dir,
                    groups_per_batch=cli.groups_per_domain,
                    max_steps=cli.max_steps,
                    max_tokens=cli.max_tokens,
                    seed=cli.seed,
                ),
                teacher_config=TeacherConfig(base_model=MODEL, load_checkpoint_path=teacher),
                groups_per_batch=cli.groups_per_domain,
            )
            for domain, teacher in zip(DOMAINS, teachers, strict=True)
        ],
        evaluator_builders=[partial(RunEvaluator, cli)] if cli.eval_every else [],
        max_steps=cli.max_steps,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        log_path=cli.log_path,
    )


async def cli_main(cli: CLIConfig) -> None:
    cli = chz.replace(cli, log_path=str(Path(cli.log_path).expanduser()))
    config = build_config(cli)
    output = Path(config.log_path)
    if not cli.dry_run and output.exists():
        raise FileExistsError(f"Choose a fresh log_path; already exists: {output}")
    register_tokenizer(MODEL, tokenizer)
    # Validate data, prompt rendering and context limits before creating a paid session.
    for spec in config.dataset_configs:
        await spec.dataset_builder()
    evaluator = config.evaluator_builders[0]() if config.evaluator_builders else None
    if cli.dry_run:
        print(
            f"Validated {cli.max_steps} updates × {2 * cli.groups_per_domain} prompts; "
            f"teachers: {dict(zip(DOMAINS, (cli.caesar_teacher, cli.geometry_teacher)))}"
        )
        return
    output.mkdir(parents=True)
    (output / "recipe.json").write_text(json.dumps(chz.asdict(cli), default=str, indent=2) + "\n")
    if evaluator is not None:
        # Reuse one evaluator so the hook and final evaluation share teacher baselines.
        config = chz.replace(config, evaluator_builders=[lambda: evaluator])
    await train_on_policy.main(config)
    if evaluator is not None:
        await evaluator.final()


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
