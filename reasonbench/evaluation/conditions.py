"""Paired descriptive comparisons for modes, budgets, and models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clustered_mean_interval(
    frame: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    grouped = list(frame.groupby("problem_id", sort=False))
    point = float(frame[value_column].mean())
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[index][1] for index in sampled], ignore_index=True)
        draws.append(float(sample[value_column].mean()))
    return {
        "value": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def summarize_conditions(
    frame: pd.DataFrame,
    group_columns: list[str],
    repetitions: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Summarize accuracy, length, entropy, and geometry by experimental condition."""

    metrics = [
        "correct",
        "trajectory_token_count",
        "generated_tokens",
        "elapsed_seconds",
        "peak_allocated_gib",
        "reasoning_boundary_forced",
        "normalized_entropy_mean",
        "surprisal_mean",
        "geometry_mean_relative_velocity",
        "geometry_mean_cosine_drift",
    ]
    result: dict[str, Any] = {"group_columns": group_columns, "groups": {}}
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = "|".join(
            f"{column}={value}" for column, value in zip(group_columns, keys, strict=True)
        )
        group_result: dict[str, Any] = {
            "trajectories": len(group),
            "problems": int(group["problem_id"].nunique()),
        }
        for metric in metrics:
            if metric not in group:
                continue
            clean = group[np.isfinite(pd.to_numeric(group[metric], errors="coerce"))].copy()
            if clean.empty:
                continue
            clean[metric] = pd.to_numeric(clean[metric])
            group_result[metric] = _clustered_mean_interval(
                clean,
                metric,
                repetitions=repetitions,
                seed=seed,
            )
        result["groups"][label] = group_result
    return result


def paired_condition_difference(
    frame: pd.DataFrame,
    condition_column: str,
    left: Any,
    right: Any,
    value_column: str,
    repetitions: int = 2000,
    seed: int = 20260728,
) -> dict[str, float | int]:
    """Compute a paired right-minus-left contrast within problem, dataset, and seed."""

    keys = ["problem_id", "dataset", "seed"]
    left_frame = frame[frame[condition_column] == left][keys + [value_column]].rename(
        columns={value_column: "left_value"}
    )
    right_frame = frame[frame[condition_column] == right][keys + [value_column]].rename(
        columns={value_column: "right_value"}
    )
    paired = left_frame.merge(right_frame, on=keys, validate="one_to_one")
    paired["difference"] = paired["right_value"].astype(float) - paired["left_value"].astype(float)
    grouped = list(paired.groupby("problem_id", sort=False))
    if not grouped:
        raise ValueError("No paired observations were found")
    point = float(paired["difference"].mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[index][1] for index in sampled], ignore_index=True)
        draws.append(float(sample["difference"].mean()))
    return {
        "paired_trajectories": len(paired),
        "paired_problems": int(paired["problem_id"].nunique()),
        "difference": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }
