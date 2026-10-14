"""reasoning_gym exact-match scoring for thinking-mode responses.

The rule used by the teacher GRPO runs: take the text after the last `</think>`
(a response cut off mid-thought scores 0), read its `<answer>`, and score 1 when
reasoning_gym's `score_answer` returns 1. In OPD this score is diagnostic only;
the training signal is the teachers' token log probabilities.

Requires `reasoning-gym==0.1.25`.
"""

import json

import reasoning_gym
from reasoning_gym.utils import extract_answer

_SCORERS = {}


def score_reply(reply, label):
    """Score a final reply that no longer contains the thinking block."""
    row = json.loads(label) if isinstance(label, str) else label
    answer = extract_answer(reply or "")
    if answer is None:
        return 0.0
    scorer = _SCORERS.setdefault(row["task"], reasoning_gym.get_score_answer_fn(row["task"]))
    try:
        return float(scorer(answer, row) >= 1.0)
    except Exception:
        return 0.0


def score(response, label):
    """Score a full response, thinking included."""
    text = (response or "").replace("<|im_end|>", "")
    if "</think>" not in text:
        return 0.0
    return score_reply(text.rsplit("</think>", 1)[1], label)
