"""Frozen, auditable terminal outcomes for early-risk analyses."""

from __future__ import annotations

from typing import Any

# These are protocol boundaries rather than model decisions.  Keep this mapping
# deliberately small: an unseen finish reason must be reviewed instead of being
# silently folded into the operational target.
NONCOMPLETION_REASONS = frozenset({"max_new_tokens", "answer_reserve"})
NORMAL_COMPLETION_REASONS = frozenset({"eos"})
KNOWN_FINISH_REASONS = NONCOMPLETION_REASONS | NORMAL_COMPLETION_REASONS


def derive_terminal_outcomes(*, correct: bool, finish_reason: str) -> dict[str, bool]:
    """Return mutually auditable endpoint labels for one trajectory.

    ``needs_intervention`` encodes the deployment decision: a wrong answer or
    a forced boundary warrants another attempt.  It intentionally differs from
    correctness alone, because a correct-looking answer at a hard cap did not
    complete under the declared protocol.
    """

    if finish_reason not in KNOWN_FINISH_REASONS:
        raise ValueError(
            "Unknown finish_reason for Phase 4b endpoint mapping: "
            f"{finish_reason!r}; add an explicit protocol mapping before analysis."
        )
    noncompletion = finish_reason in NONCOMPLETION_REASONS
    normal_completion = finish_reason in NORMAL_COMPLETION_REASONS
    wrong_completion = (not bool(correct)) and normal_completion
    needs_intervention = (not bool(correct)) or noncompletion
    return {
        "normal_completion": normal_completion,
        "noncompletion": noncompletion,
        "wrong_completion": wrong_completion,
        "needs_intervention": needs_intervention,
    }


def outcome_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convenience wrapper used by reports without duplicating endpoint logic."""

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                **record,
                **derive_terminal_outcomes(
                    correct=bool(record["correct"]),
                    finish_reason=str(record["finish_reason"]),
                ),
            }
        )
    return rows
