#!/usr/bin/env python3
"""Relate Phase 4B reasoning length and early uncertainty to MATH difficulty."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from reasonbench.evaluation.compute_extension import (
    UNCERTAINTY_BLOCKS,
    UNCERTAINTY_FEATURES,
    fit_percentile_references,
    score_uncertainty_components,
)

MODEL_LABELS = {
    "gemma4_e4b_mlx_4bit": "Gemma 4 E4B",
    "ministral3_3b_mlx_4bit": "Ministral 3 3B",
    "qwen35_4b_mlx_4bit": "Qwen 3.5 4B",
}
MODEL_COLORS = {
    "gemma4_e4b_mlx_4bit": "#2878B5",
    "ministral3_3b_mlx_4bit": "#F28E2B",
    "qwen35_4b_mlx_4bit": "#3A9D5D",
}
BLOCK_LABELS = {
    "predictive_ambiguity": "Predictive ambiguity",
    "temporal_instability": "Temporal instability",
    "geometry_instability": "Geometry instability",
    "spectral_instability": "Spectral instability",
}
KEYS = ("model_key", "problem_id", "seed")
LEVELS = (1, 2, 3, 4, 5)
ANCHOR = 512
TOKEN_CAP = 16_384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-features", type=Path, required=True)
    parser.add_argument("--prefix-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    return parser.parse_args()


def _validate(frame: pd.DataFrame, *, name: str) -> None:
    required = {
        *KEYS,
        "level",
        "correct",
        "research_split",
        "full_trajectory_token_count",
        *UNCERTAINTY_FEATURES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError(f"{name} contains duplicate model/problem/seed rows")
    invalid_levels = set(pd.to_numeric(frame["level"], errors="coerce").dropna()) - set(LEVELS)
    if invalid_levels:
        raise ValueError(f"{name} contains invalid MATH levels: {sorted(invalid_levels)}")


def _add_uncertainty_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    references = fit_percentile_references(result)
    components = score_uncertainty_components(result, references)
    result[components.columns] = components
    result["uncertainty_index_u512"] = result["uncertainty_score"]
    return result


def _bootstrap_spearman(
    frame: pd.DataFrame,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, float]:
    scoped = frame[["level", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(scoped) < 4 or scoped[metric].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    estimate, p_value = spearmanr(scoped["level"], scoped[metric])
    random = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(len(scoped))
    for _ in range(repetitions):
        sampled = scoped.iloc[random.choice(indices, size=len(indices), replace=True)]
        if sampled[metric].nunique() < 2 or sampled["level"].nunique() < 2:
            continue
        rho, _ = spearmanr(sampled["level"], sampled[metric])
        if np.isfinite(rho):
            values.append(float(rho))
    if not values:
        return float(estimate), np.nan, np.nan, float(p_value)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(estimate), float(low), float(high), float(p_value)


def _trend_table(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "full": (full, ("full_trajectory_token_count",)),
        "fixed_512": (
            prefix,
            (
                "uncertainty_index_u512",
                *(f"uncertainty_block__{block}" for block in UNCERTAINTY_BLOCKS),
            ),
        ),
    }
    for cohort, (frame, metric_names) in metrics.items():
        for model_index, (model_key, group) in enumerate(frame.groupby("model_key", sort=True)):
            for metric_index, metric in enumerate(metric_names):
                rho, low, high, p_value = _bootstrap_spearman(
                    group,
                    metric,
                    repetitions=repetitions,
                    seed=17_041 + 101 * model_index + metric_index,
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "model_key": model_key,
                        "model": MODEL_LABELS.get(model_key, model_key),
                        "metric": metric,
                        "n": len(group),
                        "spearman_rho": rho,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value_descriptive": p_value,
                    }
                )
    return pd.DataFrame(rows)


def _summary_table(full: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = {
        "full": (full, ("full_trajectory_token_count",)),
        "fixed_512": (
            prefix,
            (
                "uncertainty_index_u512",
                *(f"uncertainty_block__{block}" for block in UNCERTAINTY_BLOCKS),
                *UNCERTAINTY_FEATURES,
            ),
        ),
    }
    for cohort, (frame, metrics) in cohorts.items():
        for (model_key, level), group in frame.groupby(["model_key", "level"], sort=True):
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                rows.append(
                    {
                        "cohort": cohort,
                        "model_key": model_key,
                        "model": MODEL_LABELS.get(model_key, model_key),
                        "level": int(level),
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                    }
                )
    return pd.DataFrame(rows)


def _jitter(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    categorical_positions = frame["level"].to_numpy(dtype=float) - 1.0
    return categorical_positions + random.uniform(-0.13, 0.13, len(frame))


def _plot_reasoning_length(full: pd.DataFrame, trends: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, constrained_layout=True)
    model_order = [key for key in MODEL_LABELS if key in set(full["model_key"])]
    for model_index, (axis, model_key) in enumerate(zip(axes, model_order, strict=True)):
        group = full[full["model_key"] == model_key].copy()
        sns.boxplot(
            data=group,
            x="level",
            y="full_trajectory_token_count",
            order=LEVELS,
            color=MODEL_COLORS[model_key],
            width=0.55,
            showfliers=False,
            boxprops={"alpha": 0.22},
            ax=axis,
        )
        for correct, color, marker, label in (
            (True, "#2074B4", "o", "Correct"),
            (False, "#D9534F", "X", "Incorrect"),
        ):
            scoped = group[group["correct"].astype(bool) == correct]
            axis.scatter(
                _jitter(scoped, seed=910 + model_index + int(correct)),
                scoped["full_trajectory_token_count"],
                c=color,
                marker=marker,
                s=38,
                alpha=0.72,
                linewidths=0.4,
                edgecolors="white",
                label=label,
                zorder=3,
            )
        trend = trends[
            (trends["model_key"] == model_key)
            & (trends["metric"] == "full_trajectory_token_count")
        ].iloc[0]
        axis.axhline(TOKEN_CAP, color="#6A3D9A", linestyle="--", linewidth=1.2, alpha=0.8)
        axis.set(
            title=(
                f"{MODEL_LABELS[model_key]}\n"
                f"Spearman ρ={trend.spearman_rho:.2f} "
                f"[{trend.ci_low:.2f}, {trend.ci_high:.2f}]"
            ),
            xlabel="MATH difficulty level",
            ylabel="Reasoning-token count" if model_index == 0 else "",
            yscale="log",
        )
        axis.set_xticks(range(5), [str(level) for level in LEVELS])
        axis.grid(True, which="both", axis="y", alpha=0.22)
    axes[0].legend(frameon=True, fontsize=10, loc="upper left")
    figure.suptitle(
        "Phase 4B: reasoning length by MATH difficulty\n"
        "Boxes show the interquartile range; points are individual trajectories",
        fontsize=18,
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_uncertainty_index(prefix: pd.DataFrame, trends: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, constrained_layout=True)
    model_order = [key for key in MODEL_LABELS if key in set(prefix["model_key"])]
    for model_index, (axis, model_key) in enumerate(zip(axes, model_order, strict=True)):
        group = prefix[prefix["model_key"] == model_key].copy()
        sns.boxplot(
            data=group,
            x="level",
            y="uncertainty_index_u512",
            order=LEVELS,
            color=MODEL_COLORS[model_key],
            width=0.55,
            showfliers=False,
            boxprops={"alpha": 0.22},
            ax=axis,
        )
        for correct, color, marker, label in (
            (True, "#2074B4", "o", "Correct"),
            (False, "#D9534F", "X", "Incorrect"),
        ):
            scoped = group[group["correct"].astype(bool) == correct]
            axis.scatter(
                _jitter(scoped, seed=1_310 + model_index + int(correct)),
                scoped["uncertainty_index_u512"],
                c=color,
                marker=marker,
                s=38,
                alpha=0.72,
                linewidths=0.4,
                edgecolors="white",
                label=label,
                zorder=3,
            )
        trend = trends[
            (trends["model_key"] == model_key)
            & (trends["metric"] == "uncertainty_index_u512")
        ].iloc[0]
        counts = group.groupby("level").size().reindex(LEVELS, fill_value=0)
        axis.set(
            title=(
                f"{MODEL_LABELS[model_key]}\n"
                f"Spearman ρ={trend.spearman_rho:.2f} "
                f"[{trend.ci_low:.2f}, {trend.ci_high:.2f}]"
            ),
            xlabel="MATH difficulty level",
            ylabel="Early uncertainty index U512" if model_index == 0 else "",
            ylim=(-0.03, 1.03),
        )
        axis.set_xticks(
            range(5),
            [f"{level}\nn={int(counts.loc[level])}" for level in LEVELS],
        )
    axes[0].legend(frameon=True, fontsize=10, loc="upper left")
    figure.suptitle(
        "Phase 4B: early uncertainty by MATH difficulty\n"
        "Fixed ≥512-token cohort; model-relative train-ECDF index (higher = more uncertain)",
        fontsize=18,
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_uncertainty_components(prefix: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharex=True, sharey=True)
    for axis_index, (axis, block) in enumerate(
        zip(axes.flat, UNCERTAINTY_BLOCKS, strict=True)
    ):
        metric = f"uncertainty_block__{block}"
        for model_key, group in prefix.groupby("model_key", sort=True):
            summary = group.groupby("level")[metric].agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
            )
            x = summary.index.to_numpy(dtype=float)
            axis.plot(
                x,
                summary["median"],
                marker="o",
                linewidth=2.1,
                color=MODEL_COLORS.get(model_key),
                label=MODEL_LABELS.get(model_key, model_key),
            )
            axis.fill_between(
                x,
                summary["q25"],
                summary["q75"],
                color=MODEL_COLORS.get(model_key),
                alpha=0.13,
            )
        axis.set(title=BLOCK_LABELS[block], xticks=range(1, 6), ylim=(-0.03, 1.03))
        axis.set_xlabel("MATH difficulty level" if axis_index >= 2 else "")
        axis.set_ylabel("Model-relative uncertainty percentile" if axis_index % 2 == 0 else "")
    axes[0, 0].legend(fontsize=10, frameon=True)
    figure.suptitle(
        "Phase 4B: early uncertainty components across difficulty\n"
        "First 512 reasoning tokens; lines are medians and bands are interquartile ranges",
        fontsize=18,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    trends: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Phase 4B difficulty analysis",
        "",
        "## Design",
        "",
        f"- Reasoning-length analysis: all {len(full)} Phase 4B trajectories.",
        (
            f"- Early-uncertainty analysis: {len(prefix)} trajectories that generated at least "
            f"{ANCHOR} reasoning tokens. Features use only the first {ANCHOR} tokens."
        ),
        (
            "- U512 is a model-conditional, training-ECDF index in [0, 1]: features are "
            "averaged within four conceptual blocks, then the four blocks receive equal weight."
        ),
        "- U512 is a relative uncertainty index, not a calibrated probability of failure.",
        "- Spearman intervals are trajectory-bootstrap descriptive intervals; Phase 4B has one seed.",
        "",
        "## Fixed-512 cohort sizes",
        "",
        "| Model | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    counts = prefix.groupby(["model_key", "level"]).size().unstack(fill_value=0)
    for model_key, row in counts.iterrows():
        values = [str(int(row.get(level, 0))) for level in LEVELS]
        lines.append(f"| {MODEL_LABELS.get(model_key, model_key)} | " + " | ".join(values) + " |")
    lines.extend(["", "## Descriptive difficulty trends", ""])
    for metric, label in (
        ("full_trajectory_token_count", "difficulty vs. reasoning length"),
        ("uncertainty_index_u512", "difficulty vs. early uncertainty U512"),
    ):
        lines.append(f"### {label}")
        lines.append("")
        scoped = trends[trends["metric"] == metric]
        for row in scoped.itertuples(index=False):
            lines.append(
                f"- {row.model}: Spearman rho = {row.spearman_rho:.3f} "
                f"(95% bootstrap interval {row.ci_low:.3f} to {row.ci_high:.3f}; n={row.n})."
            )
        lines.append("")
    lines.extend(
        [
            "### difficulty vs. U512 blocks",
            "",
            "| Block | Model | Spearman rho | 95% bootstrap interval | n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for block in UNCERTAINTY_BLOCKS:
        metric = f"uncertainty_block__{block}"
        scoped = trends[trends["metric"] == metric]
        for row in scoped.itertuples(index=False):
            lines.append(
                f"| {BLOCK_LABELS[block]} | {row.model} | {row.spearman_rho:.3f} | "
                f"{row.ci_low:.3f} to {row.ci_high:.3f} | {row.n} |"
            )
    lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- Difficulty labels are properties of problems; the plots are associative, not causal.",
            "- Correctness markers diagnose overlap but do not adjust the difficulty trend for outcome.",
            "- The fixed-512 restriction prevents shorter trajectories from masquerading as 512-token prefixes.",
            (
                "- The fixed-512 cohort is length-selected: especially at easier levels, trajectories "
                "that ended before 512 tokens are absent, so its difficulty trend can show selection bias."
            ),
            "- Because only one seed was run, uncertainty intervals do not capture between-seed variation.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    full = pd.read_parquet(args.full_features)
    prefix = pd.read_parquet(args.prefix_features)
    _validate(full, name="full feature table")
    _validate(prefix, name="prefix feature table")

    expected = full[list(KEYS)].sort_values(list(KEYS)).reset_index(drop=True)
    observed = prefix[list(KEYS)].sort_values(list(KEYS)).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError("Full and prefix feature tables do not contain identical trajectories")

    full = full.copy()
    prefix = prefix[prefix["full_trajectory_token_count"] >= ANCHOR].copy()
    prefix = _add_uncertainty_percentiles(prefix)
    trends = _trend_table(
        full,
        prefix,
        repetitions=args.bootstrap_repetitions,
    )
    summary = _summary_table(full, prefix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output_dir / "phase04b_difficulty_summary.parquet", index=False)
    summary.to_csv(args.output_dir / "phase04b_difficulty_summary.csv", index=False)
    trends.to_parquet(args.output_dir / "phase04b_difficulty_trends.parquet", index=False)
    trends.to_csv(args.output_dir / "phase04b_difficulty_trends.csv", index=False)
    prefix.to_parquet(args.output_dir / "phase04b_fixed_512_uncertainty.parquet", index=False)
    _plot_reasoning_length(
        full,
        trends,
        args.output_dir / "phase04b_reasoning_length_by_difficulty.png",
    )
    _plot_uncertainty_index(
        prefix,
        trends,
        args.output_dir / "phase04b_early_uncertainty_by_difficulty.png",
    )
    _plot_uncertainty_components(
        prefix,
        args.output_dir / "phase04b_uncertainty_components_by_difficulty.png",
    )
    _write_report(
        full,
        prefix,
        trends,
        args.output_dir / "phase04b_difficulty_report.md",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
