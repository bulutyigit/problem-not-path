#!/usr/bin/env python
"""Plot cumulative spectral features at homogeneous relative reasoning intervals."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasonbench.features.extractor import _analysis_window, trajectory_directories
from reasonbench.features.spectral import summarize_spectrum
from reasonbench.storage import ensure_directory, read_json

# Eight equal intervals let every eligible trajectory contribute a 12.5%, ...,
# 100% point, regardless of how many reasoning tokens it ultimately used.
RELATIVE_PROGRESS = tuple(index / 8 for index in range(1, 9))
MINIMUM_REASONING_TOKENS = 512
FEATURES = {
    "spectral_normalized_entropy_entropy": "Entropy spectral entropy",
    "spectral_normalized_entropy_high_energy_ratio": "Entropy high-frequency energy",
    "spectral_surprisal_entropy": "Surprisal spectral entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    return parser.parse_args()


def _spectral_rows(directory: Path) -> list[dict[str, object]]:
    metadata = read_json(directory / "metadata.json")
    token_frame = pd.read_parquet(directory / "token_metrics.parquet")
    reasoning = _analysis_window(token_frame)
    full_tokens = len(reasoning)
    if full_tokens < MINIMUM_REASONING_TOKENS:
        return []

    rows: list[dict[str, object]] = []
    for progress in RELATIVE_PROGRESS:
        prefix_tokens = min(full_tokens, max(64, round(full_tokens * progress)))
        window = reasoning.iloc[:prefix_tokens]
        entropy = summarize_spectrum(
            window["normalized_entropy"].to_numpy(),
            prefix="spectral_normalized_entropy",
        )
        surprisal = summarize_spectrum(
            window["surprisal"].to_numpy(),
            prefix="spectral_surprisal",
        )
        rows.append(
            {
                "run_id": metadata["run_id"],
                "model_key": metadata["model_key"],
                "problem_id": metadata["problem_id"],
                "correct": bool(metadata["verification"]["correct"]),
                "terminal_outcome": (
                    "Correct" if metadata["verification"]["correct"] else "Incorrect"
                ),
                "relative_progress": progress,
                "progress_percent": progress * 100,
                "prefix_tokens": prefix_tokens,
                "full_trajectory_token_count": full_tokens,
                **{feature: entropy.get(feature, surprisal.get(feature)) for feature in FEATURES},
            }
        )
    return rows


def _mean_interval(
    frame: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean with a complete-problem clustered bootstrap interval."""

    observed = frame[["problem_id", value_column]].dropna(subset=[value_column]).copy()
    if observed.empty:
        return math.nan, math.nan, math.nan
    point = float(observed[value_column].mean())
    if repetitions <= 0:
        return point, math.nan, math.nan
    groups = [group for _, group in observed.groupby("problem_id", sort=False)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sample = pd.concat(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        draws.append(float(sample[value_column].mean()))
    low, high = np.quantile(draws, (0.025, 0.975))
    return point, float(low), float(high)


def _summarize(observations: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    long = observations.melt(
        id_vars=[
            "run_id",
            "model_key",
            "problem_id",
            "terminal_outcome",
            "relative_progress",
            "progress_percent",
            "prefix_tokens",
            "full_trajectory_token_count",
        ],
        value_vars=list(FEATURES),
        var_name="feature",
        value_name="value",
    )
    long["feature_label"] = long["feature"].map(FEATURES)
    rows: list[dict[str, object]] = []
    group_columns = ["model_key", "terminal_outcome", "relative_progress", "feature", "feature_label"]
    for offset, (keys, group) in enumerate(long.groupby(group_columns, observed=True, sort=True)):
        point, low, high = _mean_interval(
            group, "value", repetitions, seed=20260806 + offset
        )
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "mean": point,
                "ci_low": low,
                "ci_high": high,
                "trajectories": int(group["run_id"].nunique()),
                "problems": int(group["problem_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output_path: Path) -> None:
    models = sorted(summary["model_key"].unique())
    feature_items = list(FEATURES.items())
    figure, axes = plt.subplots(
        len(feature_items),
        len(models),
        figsize=(15, 10),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    colors = {"Correct": "#2a9d8f", "Incorrect": "#e76f51"}
    x_values = [progress * 100 for progress in RELATIVE_PROGRESS]
    display_ticks = [12.5, 25.0, 50.0, 75.0, 100.0]
    for row, (feature, label) in enumerate(feature_items):
        for column, model in enumerate(models):
            axis = axes[row, column]
            scoped = summary[(summary["model_key"] == model) & (summary["feature"] == feature)]
            for outcome in ("Correct", "Incorrect"):
                group = scoped[scoped["terminal_outcome"] == outcome].sort_values("relative_progress")
                if group.empty:
                    continue
                x = group["relative_progress"].to_numpy(float) * 100
                axis.plot(x, group["mean"], marker="o", color=colors[outcome], label=outcome)
                axis.fill_between(
                    x,
                    group["ci_low"].to_numpy(float),
                    group["ci_high"].to_numpy(float),
                    color=colors[outcome],
                    alpha=0.16,
                )
            axis.set_title(model if row == 0 else "")
            axis.set_xlim(min(x_values) - 2, max(x_values) + 2)
            axis.set_xticks(display_ticks)
            if row == len(feature_items) - 1:
                axis.set_xticklabels([f"{value:g}%" for value in display_ticks])
            else:
                axis.tick_params(labelbottom=False)
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=22)
    axes[-1, len(models) // 2].set_xlabel("Relative reasoning progress")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    figure.suptitle(
        "Cumulative spectral evolution over homogeneous relative reasoning intervals\n"
        "Each trajectory contributes at 12.5% increments through its own reasoning endpoint",
        y=0.99,
    )
    figure.subplots_adjust(left=0.30, right=0.98, bottom=0.09, top=0.79, hspace=0.24, wspace=0.18)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 0:
        raise ValueError("bootstrap-repetitions must be non-negative")
    output_dir = ensure_directory(args.output_dir)
    sns.set_theme(style="whitegrid", context="talk")
    rows = [row for directory in trajectory_directories([args.generation_dir]) for row in _spectral_rows(directory)]
    if not rows:
        raise ValueError(f"No trajectories with at least {MINIMUM_REASONING_TOKENS} reasoning tokens")
    observations = pd.DataFrame(rows).sort_values(
        ["model_key", "problem_id", "relative_progress"]
    )
    summary = _summarize(observations, args.bootstrap_repetitions)
    observations.to_parquet(output_dir / "phase04_relative_spectral_observations.parquet", index=False)
    summary.to_parquet(output_dir / "phase04_relative_spectral_summary.parquet", index=False)
    _plot(summary, output_dir / "phase04_relative_spectral_progress_by_correctness.png")


if __name__ == "__main__":
    main()
