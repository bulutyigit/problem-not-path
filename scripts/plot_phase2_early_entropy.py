#!/usr/bin/env python
"""Create descriptive fixed-prefix entropy figures from a completed Phase 2 run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PREFIXES = (16, 32, 64, 128, 256, 512, 1024, 2048)
RNG = np.random.default_rng(20260804)


def interval(frame: pd.DataFrame, value: str) -> tuple[float, float, float]:
    """Problem-cluster bootstrap interval, avoiding seed pseudoreplication."""

    clustered = frame.groupby("problem_id", as_index=False)[value].mean()[value].to_numpy()
    point = float(clustered.mean())
    draws = [RNG.choice(clustered, len(clustered), replace=True).mean() for _ in range(1000)]
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prefix in PREFIXES:
        frame = pd.read_parquet(args.features_dir / f"features_prefix_{prefix}.parquet")
        active = frame[frame["full_trajectory_token_count"] >= prefix].copy()
        for correct, group in active.groupby("correct"):
            mean, low, high = interval(group, "normalized_entropy_mean")
            rows.append(
                {
                    "prefix": prefix,
                    "correct": correct,
                    "level": "all",
                    "mean": mean,
                    "low": low,
                    "high": high,
                    "trajectories": len(group),
                    "problems": group.problem_id.nunique(),
                }
            )
        for level, level_frame in active.groupby("level"):
            for correct, group in level_frame.groupby("correct"):
                if group.problem_id.nunique() >= 2:
                    mean, low, high = interval(group, "normalized_entropy_mean")
                    rows.append(
                        {
                            "prefix": prefix,
                            "correct": correct,
                            "level": str(level),
                            "mean": mean,
                            "low": low,
                            "high": high,
                            "trajectories": len(group),
                            "problems": group.problem_id.nunique(),
                        }
                    )
    summary = pd.DataFrame(rows)
    summary.to_parquet(args.output_dir / "fixed_prefix_entropy_summary.parquet", index=False)
    sns.set_theme(style="whitegrid", context="talk")
    colors = {False: "#4C78A8", True: "#F58518"}
    labels = {False: "Incorrect terminal answer", True: "Correct terminal answer"}
    fig, axis = plt.subplots(figsize=(11, 6))
    for correct in (False, True):
        scoped = summary[(summary.level == "all") & (summary.correct == correct)].sort_values(
            "prefix"
        )
        axis.plot(
            scoped.prefix, scoped["mean"], marker="o", color=colors[correct], label=labels[correct]
        )
        axis.fill_between(scoped.prefix, scoped.low, scoped.high, color=colors[correct], alpha=0.16)
    axis.set(
        xscale="log",
        xlabel="Observed reasoning tokens (fixed prefix)",
        ylabel="Mean normalized predictive entropy",
        title="Early entropy by terminal correctness",
    )
    axis.set_xticks(PREFIXES)
    axis.set_xticklabels(PREFIXES)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "early_entropy_by_correctness.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), sharey=True)
    for axis, level in zip(axes, map(str, range(1, 6)), strict=True):
        for correct in (False, True):
            scoped = summary[(summary.level == level) & (summary.correct == correct)].sort_values(
                "prefix"
            )
            if scoped.empty:
                continue
            axis.plot(
                scoped.prefix,
                scoped["mean"],
                marker="o",
                color=colors[correct],
                label=labels[correct],
            )
            axis.fill_between(
                scoped.prefix, scoped.low, scoped.high, color=colors[correct], alpha=0.16
            )
        axis.set_xscale("log")
        axis.set_xticks((16, 64, 256, 1024))
        axis.set_xticklabels((16, 64, 256, 1024))
        axis.set_title(f"Level {level}")
        axis.set_xlabel("Tokens")
    axes[0].set_ylabel("Mean normalized entropy")
    axes[0].legend(fontsize=10)
    fig.suptitle("Fixed-prefix entropy conditional on MATH difficulty", y=1.03)
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "early_entropy_by_level_and_correctness.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    active = (
        summary[summary.level == "all"]
        .pivot(index="prefix", columns="correct", values="trajectories")
        .fillna(0)
    )
    fig, axis = plt.subplots(figsize=(11, 5))
    for correct in (False, True):
        axis.plot(
            active.index,
            active.get(correct, pd.Series(0, index=active.index)),
            marker="o",
            color=colors[correct],
            label=labels[correct],
        )
    axis.set(
        xscale="log",
        xlabel="Fixed prefix",
        ylabel="Trajectories still active",
        title="Risk set retained at each prefix",
    )
    axis.set_xticks(PREFIXES)
    axis.set_xticklabels(PREFIXES)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "early_entropy_risk_set.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
