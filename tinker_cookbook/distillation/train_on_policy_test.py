"""Offline tests for the on-policy trainer's session lifecycle."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.distillation import train_on_policy as train


@pytest.fixture
def lifecycle(tmp_path: Path) -> Iterator[tuple[train.Config, MagicMock]]:
    config = train.Config(
        learning_rate=1e-4,
        dataset_configs=[],
        model_name="test-model",
        recipe_name="test",
        max_tokens=32,
        log_path=str(tmp_path),
    )
    mocks = MagicMock()
    mocks.service.close = AsyncMock()
    mocks.service.create_lora_training_client_async = AsyncMock()
    mocks.checkpoint.save_final_async = AsyncMock()
    mocks.loop = AsyncMock()
    mocks.dataset.__len__.return_value = 1
    mocks.log.get_logger_url.return_value = None
    with (
        patch.object(train.ml_log, "setup_logging", return_value=mocks.log),
        patch.object(train.tinker, "ServiceClient", return_value=mocks.service) as factory,
        patch.object(train.checkpoint_utils, "get_last_checkpoint", return_value=None),
        patch.object(train.checkpoint_utils, "CheckpointManager", return_value=mocks.checkpoint),
        patch.object(train.model_info, "warn_if_renderer_not_recommended"),
        patch.object(train, "get_tokenizer"),
        patch.object(train, "CompositeDataset", return_value=mocks.dataset),
        patch.object(train, "do_sync_training", mocks.loop),
    ):
        mocks.factory = factory
        yield config, mocks


@pytest.mark.asyncio
async def test_success_closes_session_after_final_checkpoint(
    lifecycle: tuple[train.Config, MagicMock],
) -> None:
    config, mocks = lifecycle

    async def save_final(*, loop_state: dict[str, int]) -> None:
        assert loop_state == {"batch": 1}
        mocks.service.close.assert_not_called()

    mocks.checkpoint.save_final_async.side_effect = save_final
    await train.main(config)
    mocks.checkpoint.save_final_async.assert_awaited_once()
    mocks.service.close.assert_awaited_once_with("success")
    mocks.log.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["setup", "training", "checkpoint"])
async def test_failure_closes_session_and_preserves_error(
    lifecycle: tuple[train.Config, MagicMock], stage: str
) -> None:
    config, mocks = lifecycle
    failure = RuntimeError(f"{stage} failed")
    operations = {
        "setup": mocks.service.create_lora_training_client_async,
        "training": mocks.loop,
        "checkpoint": mocks.checkpoint.save_final_async,
    }
    operations[stage].side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        await train.main(config)
    assert caught.value is failure
    mocks.service.close.assert_awaited_once_with("errored")
    mocks.log.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_interruption_closes_session(
    lifecycle: tuple[train.Config, MagicMock], error_type: type[BaseException]
) -> None:
    config, mocks = lifecycle
    mocks.loop.side_effect = error_type()
    with pytest.raises(error_type):
        await train.main(config)
    mocks.service.close.assert_awaited_once_with("interrupted")
    mocks.checkpoint.save_final_async.assert_not_called()
    mocks.log.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("training_fails", [False, True])
async def test_close_failure_still_closes_logger(
    lifecycle: tuple[train.Config, MagicMock], training_fails: bool
) -> None:
    config, mocks = lifecycle
    cleanup_error = RuntimeError("close failed")
    training_error = ValueError("training failed")
    mocks.service.close.side_effect = cleanup_error
    if training_fails:
        mocks.loop.side_effect = training_error
    expected = training_error if training_fails else cleanup_error
    with pytest.raises(type(expected)) as caught:
        await train.main(config)
    assert caught.value is expected
    mocks.log.close.assert_called_once()


@pytest.mark.asyncio
async def test_client_creation_failure_closes_logger(
    lifecycle: tuple[train.Config, MagicMock],
) -> None:
    config, mocks = lifecycle
    mocks.factory.side_effect = RuntimeError("client creation failed")
    with pytest.raises(RuntimeError, match="client creation failed"):
        await train.main(config)
    mocks.service.close.assert_not_called()
    mocks.log.close.assert_called_once()
