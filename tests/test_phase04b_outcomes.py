from __future__ import annotations

import pytest

from reasonbench.evaluation.outcomes import derive_terminal_outcomes


@pytest.mark.parametrize(
    ("correct", "finish_reason", "expected"),
    [
        (True, "eos", (True, False, False, False)),
        (False, "eos", (True, False, True, True)),
        (True, "max_new_tokens", (False, True, False, True)),
        (False, "max_new_tokens", (False, True, False, True)),
        (False, "answer_reserve", (False, True, False, True)),
    ],
)
def test_frozen_phase04b_outcome_truth_table(
    correct: bool,
    finish_reason: str,
    expected: tuple[bool, bool, bool, bool],
) -> None:
    outcome = derive_terminal_outcomes(correct=correct, finish_reason=finish_reason)
    assert tuple(outcome[key] for key in (
        "normal_completion",
        "noncompletion",
        "wrong_completion",
        "needs_intervention",
    )) == expected
    assert not (outcome["normal_completion"] and outcome["noncompletion"])
    assert not outcome["wrong_completion"] or outcome["needs_intervention"]
    assert not outcome["noncompletion"] or outcome["needs_intervention"]


def test_phase04b_outcome_mapping_rejects_unreviewed_finish_reason() -> None:
    with pytest.raises(ValueError, match="Unknown finish_reason"):
        derive_terminal_outcomes(correct=False, finish_reason="unknown_boundary")
