"""Difficulty-conditioned summaries for repeated stochastic reasoning trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reasonbench.features.extractor import trajectory_directories
from reasonbench.storage import read_json

TOKEN_SIGNALS = (
    "normalized_entropy",
    "surprisal",
    "top1_top2_probability_margin",
    "top1_top2_logit_margin",
    "relative_l2_step",
    "cosine_drift",
)


def validate_difficulty_design(
    frame: pd.DataFrame,
    *,
    expected_models: Iterable[str],
    problems_per_level: int = 20,
    seeds_per_problem: int = 4,
) -> dict[str, Any]:
    """Require an exactly balanced, problem/seed-paired MATH design."""

    required = {"dataset", "model_key", "problem_id", "seed", "level", "correct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Difficulty design is missing columns: {missing}")
    if set(frame["dataset"].dropna()) != {"math"}:
        raise ValueError("Difficulty experiments must contain only MATH trajectories")
    models = tuple(sorted(str(value) for value in frame["model_key"].unique()))
    expected = tuple(sorted(expected_models))
    if models != expected:
        raise ValueError(f"Expected models {expected}, observed {models}")
    observed_levels = set(frame["level"].dropna().astype(int))
    if observed_levels != set(range(1, 6)):
        raise ValueError(f"Expected MATH levels 1-5, observed {sorted(observed_levels)}")

    cells: dict[str, Any] = {}
    reference_pairs: set[tuple[str, int]] | None = None
    for model in models:
        model_frame = frame[frame["model_key"] == model]
        pairs = set(zip(model_frame["problem_id"], model_frame["seed"], strict=True))
        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise ValueError("Models do not share the exact same problem/seed pairs")
        for level in range(1, 6):
            cell = model_frame[model_frame["level"] == level]
            problems = int(cell["problem_id"].nunique())
            trajectories = len(cell)
            if problems != problems_per_level or trajectories != problems * seeds_per_problem:
                raise ValueError(
                    f"Unbalanced {model} level {level}: "
                    f"problems={problems}, trajectories={trajectories}"
                )
            seed_counts = cell.groupby("problem_id")["seed"].nunique()
            if not (seed_counts == seeds_per_problem).all():
                raise ValueError(f"Incomplete seed replication for {model} level {level}")
            cells[f"{model}|level={level}"] = {
                "problems": problems,
                "trajectories": trajectories,
            }
    return {
        "models": list(models),
        "problems": int(frame["problem_id"].nunique()),
        "trajectories": len(frame),
        "problems_per_level": problems_per_level,
        "seeds_per_problem": seeds_per_problem,
        "cells": cells,
    }


def _bootstrap_interval(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    clean = values[np.isfinite(values)]
    if not len(clean):
        return np.nan, np.nan, np.nan
    draws = clean[rng.integers(0, len(clean), size=(repetitions, len(clean)))].mean(axis=1)
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def difficulty_metric_summary(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    repetitions: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    """Summarize metrics by model and level with problem-cluster intervals."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (model, level), cell in frame.groupby(["model_key", "level"], sort=True):
        for metric in metrics:
            if metric not in cell:
                continue
            problem_values = (
                cell.assign(_value=pd.to_numeric(cell[metric], errors="coerce"))
                .groupby("problem_id")["_value"]
                .mean()
                .to_numpy(dtype=float)
            )
            mean, low, high = _bootstrap_interval(problem_values, repetitions, rng)
            rows.append(
                {
                    "model_key": model,
                    "level": int(level),
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "problems": len(problem_values),
                }
            )
    return pd.DataFrame(rows)


def level_trends(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    repetitions: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Estimate per-level slopes and paired model differences in those slopes."""

    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"models": {}, "model_slope_differences": {}}
    problem_table = frame.groupby(["model_key", "problem_id", "level"], as_index=False).mean(
        numeric_only=True
    )
    for model, model_frame in problem_table.groupby("model_key", sort=True):
        result["models"][model] = {}
        grouped = list(model_frame.groupby("problem_id", sort=False))
        for metric in metrics:
            if metric not in model_frame:
                continue
            clean = model_frame[["level", metric]].dropna()
            if clean["level"].nunique() < 2:
                result["models"][model][metric] = {
                    "slope_per_level": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
                continue
            point = float(np.polyfit(clean["level"], clean[metric], 1)[0])
            draws = np.full(repetitions, np.nan, dtype=float)
            for index in range(repetitions):
                sampled = rng.integers(0, len(grouped), size=len(grouped))
                sample = pd.concat([grouped[position][1] for position in sampled])
                sample = sample[["level", metric]].dropna()
                if sample["level"].nunique() >= 2:
                    draws[index] = np.polyfit(sample["level"], sample[metric], 1)[0]
            finite_draws = draws[np.isfinite(draws)]
            if not len(finite_draws):
                finite_draws = np.array([point])
            result["models"][model][metric] = {
                "slope_per_level": point,
                "ci_low": float(np.quantile(finite_draws, 0.025)),
                "ci_high": float(np.quantile(finite_draws, 0.975)),
            }
    models = sorted(result["models"])
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            label = f"{right}_minus_{left}"
            result["model_slope_differences"][label] = {}
            for metric in metrics:
                if metric not in problem_table:
                    continue
                left_table = problem_table.loc[
                    problem_table["model_key"] == left,
                    ["problem_id", "level", metric],
                ].rename(columns={metric: "left_value"})
                right_table = problem_table.loc[
                    problem_table["model_key"] == right,
                    ["problem_id", "level", metric],
                ].rename(columns={metric: "right_value"})
                paired = left_table.merge(
                    right_table,
                    on=["problem_id", "level"],
                    how="inner",
                    validate="one_to_one",
                ).dropna(subset=["left_value", "right_value"])
                if paired["level"].nunique() < 2:
                    continue
                point = float(
                    np.polyfit(paired["level"], paired["right_value"], 1)[0]
                    - np.polyfit(paired["level"], paired["left_value"], 1)[0]
                )
                difference = np.full(repetitions, np.nan, dtype=float)
                for index in range(repetitions):
                    sampled = paired.iloc[
                        rng.integers(0, len(paired), size=len(paired))
                    ]
                    if sampled["level"].nunique() >= 2:
                        difference[index] = (
                            np.polyfit(sampled["level"], sampled["right_value"], 1)[0]
                            - np.polyfit(sampled["level"], sampled["left_value"], 1)[0]
                        )
                difference = difference[np.isfinite(difference)]
                if not len(difference):
                    difference = np.array([point])
                result["model_slope_differences"][label][metric] = {
                    "difference": point,
                    "ci_low": float(np.quantile(difference, 0.025)),
                    "ci_high": float(np.quantile(difference, 0.975)),
                }
    return result


def seed_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per model/problem with success count across decoding seeds."""

    return (
        frame.groupby(["model_key", "problem_id", "level", "category"], dropna=False)
        .agg(successes=("correct", "sum"), seeds=("seed", "nunique"))
        .reset_index()
    )


def binned_token_dynamics(
    run_directories: Iterable[str | Path],
    *,
    bins: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize reasoning-token time and extract entropy-spike diagnostics."""

    rows: list[pd.DataFrame] = []
    spike_rows: list[dict[str, Any]] = []
    for directory in trajectory_directories(run_directories):
        metadata = read_json(directory / "metadata.json")
        tokens = pd.read_parquet(directory / "token_metrics.parquet")
        segment = "thinking" if (tokens["segment"] == "thinking").any() else "solution"
        tokens = tokens[tokens["segment"] == segment].sort_values("token_index").reset_index()
        if tokens.empty:
            continue
        token_bins = np.minimum(np.arange(len(tokens)) * bins // len(tokens), bins - 1)
        scoped = tokens[list(TOKEN_SIGNALS)].apply(pd.to_numeric, errors="coerce")
        scoped["time_bin"] = token_bins
        binned = scoped.groupby("time_bin", as_index=False).mean()
        for key, value in {
            "run_id": metadata["run_id"],
            "model_key": metadata["model_key"],
            "problem_id": metadata["problem_id"],
            "seed": metadata["seed"],
            "level": metadata.get("level"),
            "category": metadata.get("category"),
            "correct": bool(metadata["verification"]["correct"]),
        }.items():
            binned[key] = value
        rows.append(binned)
        entropy = scoped["normalized_entropy"].to_numpy(dtype=float)
        differences = np.diff(entropy)
        if len(differences) and np.isfinite(differences).any():
            position = int(np.nanargmax(differences))
            spike_rows.append(
                {
                    "run_id": metadata["run_id"],
                    "model_key": metadata["model_key"],
                    "problem_id": metadata["problem_id"],
                    "seed": metadata["seed"],
                    "level": metadata.get("level"),
                    "correct": bool(metadata["verification"]["correct"]),
                    "maximum_entropy_rise": float(differences[position]),
                    "spike_relative_position": float((position + 1) / len(entropy)),
                    "entropy_after_spike": float(entropy[position + 1]),
                    "final_entropy": float(entropy[-1]),
                    "recovered_after_spike": bool(entropy[-1] < entropy[position + 1]),
                }
            )
    if not rows:
        raise ValueError("No token trajectories were available for difficulty analysis")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(spike_rows)


def token_dynamics_summary(
    binned: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    """Build problem-clustered uncertainty bands for normalized token trajectories."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    keys = ["model_key", "level", "correct", "time_bin"]
    for group_keys, cell in binned.groupby(keys, sort=True):
        for signal in TOKEN_SIGNALS:
            values = cell.groupby("problem_id")[signal].mean().to_numpy(dtype=float)
            mean, low, high = _bootstrap_interval(values, repetitions, rng)
            rows.append(
                {
                    **dict(zip(keys, group_keys, strict=True)),
                    "signal": signal,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)
