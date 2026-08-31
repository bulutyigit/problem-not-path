"""Transparent three-action adaptive-compute routing utilities."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd

ARMS = ("short", "medium", "long")


def policy_digest(payload: dict) -> str:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "policy_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_policy_digest(payload: dict) -> bool:
    declared = str(payload.get("policy_digest", ""))
    return bool(declared) and declared == policy_digest(payload)


def choose_arm(
    uncertainty_score: float,
    *,
    short_max: float,
    medium_max: float,
    breakthrough_probability: float | None = None,
    breakthrough_continue_min: float | None = None,
) -> str:
    """Choose short/medium/long from frozen continuous-score thresholds.

    When a separately frozen breakthrough forecast is available, a high-U512
    prefix predicted to break through soon receives medium rather than long.
    Missing forecasts never silently activate this branch.
    """
    if not 0.0 <= short_max < medium_max <= 1.0:
        raise ValueError("Require 0 <= short_max < medium_max <= 1")
    score = float(uncertainty_score)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("uncertainty_score must be finite and in [0, 1]")
    if score <= short_max:
        return "short"
    if score <= medium_max:
        return "medium"
    if breakthrough_continue_min is None:
        return "long"
    if breakthrough_probability is None or not np.isfinite(breakthrough_probability):
        raise ValueError("Frozen policy requires a finite breakthrough probability")
    return "medium" if breakthrough_probability >= breakthrough_continue_min else "long"


def materialize_actions(
    frame: pd.DataFrame,
    thresholds: dict,
    *,
    breakthrough_column: str | None = None,
) -> pd.Series:
    values = []
    for _, row in frame.iterrows():
        probability = None if breakthrough_column is None else float(row[breakthrough_column])
        values.append(
            choose_arm(
                float(row["uncertainty_score"]),
                short_max=float(thresholds["short_max"]),
                medium_max=float(thresholds["medium_max"]),
                breakthrough_probability=probability,
                breakthrough_continue_min=thresholds.get("breakthrough_continue_min"),
            )
        )
    return pd.Series(values, index=frame.index, dtype="string")


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, *, anchor: int = 512) -> dict:
    if len(frame) != len(actions):
        raise ValueError("frame and actions must have identical lengths")
    correct = np.asarray(
        [bool(frame.at[index, f"{arm}_correct"]) for index, arm in actions.items()],
        dtype=float,
    )
    reasoning_tokens = np.asarray(
        [anchor + int(frame.at[index, f"{arm}_reasoning_tokens"]) for index, arm in actions.items()],
        dtype=float,
    )
    has_answer_tokens = all(f"{arm}_answer_tokens" in frame.columns for arm in ARMS)
    has_total_tokens = all(f"{arm}_total_generated_tokens" in frame.columns for arm in ARMS)
    answer_tokens = np.asarray(
        [
            int(frame.at[index, f"{arm}_answer_tokens"])
            if has_answer_tokens
            else 0
            for index, arm in actions.items()
        ],
        dtype=float,
    )
    total_tokens = np.asarray(
        [
            int(frame.at[index, f"{arm}_total_generated_tokens"])
            if has_total_tokens
            else reasoning_tokens[position] + answer_tokens[position]
            for position, (index, arm) in enumerate(actions.items())
        ],
        dtype=float,
    )
    rates = actions.value_counts(normalize=True)
    return {
        "accuracy": float(correct.mean()),
        "mean_realized_reasoning_tokens": float(reasoning_tokens.mean()),
        "mean_final_answer_tokens": float(answer_tokens.mean()),
        "mean_total_generated_tokens": float(total_tokens.mean()),
        "route_rates": {arm: float(rates.get(arm, 0.0)) for arm in ARMS},
    }


def fit_three_action_thresholds(
    frame: pd.DataFrame,
    *,
    max_accuracy_gap: float = 0.05,
    threshold_grid: tuple[float, ...] = tuple(np.linspace(0.05, 0.95, 19)),
    breakthrough_column: str | None = None,
    breakthrough_grid: tuple[float, ...] = tuple(np.linspace(0.1, 0.9, 9)),
    anchor: int = 512,
) -> tuple[dict, pd.DataFrame]:
    """Select the lowest-total-token transparent policy near fixed-long accuracy."""
    required = {
        "uncertainty_score",
        *(f"{arm}_correct" for arm in ARMS),
        *(f"{arm}_reasoning_tokens" for arm in ARMS),
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Development table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Development table is empty")
    fixed_long = evaluate_actions(frame, pd.Series("long", index=frame.index), anchor=anchor)
    target_accuracy = fixed_long["accuracy"] - max_accuracy_gap
    rows: list[dict] = []
    breakthrough_thresholds: tuple[float | None, ...] = (
        tuple(float(value) for value in breakthrough_grid)
        if breakthrough_column is not None
        else (None,)
    )
    for short_max, medium_max in combinations(threshold_grid, 2):
        for breakthrough_min in breakthrough_thresholds:
            thresholds = {
                "short_max": float(short_max),
                "medium_max": float(medium_max),
                "breakthrough_continue_min": breakthrough_min,
            }
            actions = materialize_actions(
                frame,
                thresholds,
                breakthrough_column=breakthrough_column,
            )
            metrics = evaluate_actions(frame, actions, anchor=anchor)
            rows.append(
                {
                    **thresholds,
                    **metrics,
                    **{f"route_rate_{arm}": metrics["route_rates"][arm] for arm in ARMS},
                    "meets_accuracy_target": metrics["accuracy"] >= target_accuracy,
                }
            )
    candidates = pd.DataFrame(rows).drop(columns="route_rates")
    feasible = candidates[candidates["meets_accuracy_target"]]
    ranked = feasible if not feasible.empty else candidates
    ranked = ranked.sort_values(
        ["mean_total_generated_tokens", "accuracy", "short_max", "medium_max"],
        ascending=[True, False, True, True],
    ) if not feasible.empty else ranked.sort_values(
        ["accuracy", "mean_total_generated_tokens", "short_max", "medium_max"],
        ascending=[False, True, True, True],
    )
    winner = ranked.iloc[0].to_dict()
    winner["fixed_long_accuracy"] = fixed_long["accuracy"]
    winner["fixed_long_mean_realized_reasoning_tokens"] = fixed_long[
        "mean_realized_reasoning_tokens"
    ]
    winner["fixed_long_mean_final_answer_tokens"] = fixed_long[
        "mean_final_answer_tokens"
    ]
    winner["fixed_long_mean_total_generated_tokens"] = fixed_long[
        "mean_total_generated_tokens"
    ]
    winner["accuracy_target"] = target_accuracy
    winner["selection_status"] = (
        "minimum_compute_within_accuracy_gap" if not feasible.empty else "no_candidate_met_accuracy_gap"
    )
    return winner, candidates
