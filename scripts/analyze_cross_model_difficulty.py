#!/usr/bin/env python
"""Compare three models on the same level-balanced MATH problem/seed pairs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ConstantInputWarning, spearmanr

from reasonbench.evaluation.conditions import paired_condition_difference
from reasonbench.evaluation.difficulty import (
    binned_token_dynamics,
    difficulty_metric_summary,
    level_trends,
    seed_consistency,
    token_dynamics_summary,
    validate_difficulty_design,
)
from reasonbench.storage import ensure_directory, write_json_atomic

MODELS = ("gemma4_e4b", "qwen35_4b", "ministral3_3b")
METRICS = (
    "correct",
    "trajectory_token_count",
    "elapsed_seconds",
    "normalized_entropy_mean",
    "surprisal_mean",
    "top1_top2_probability_margin_mean",
    "geometry_mean_relative_velocity",
    "geometry_mean_cosine_drift",
)
PREFIXES = (16, 32, 64, 128, 256, 512, 1024, 2048)
TOKEN_USAGE_PREFIXES = (16, 32, 64, 128, 256)
MODEL_LABELS = {
    "gemma4_e4b": "Gemma 4 E4B",
    "qwen35_4b": "Qwen 3.5 4B",
    "ministral3_3b": "Ministral 3 3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument(
        "--gemma-phase2-features-dir",
        type=Path,
        help=(
            "Optional Phase 2 feature directory. When supplied, the correctness-stratified "
            "Gemma plot uses all four Phase 2 seeds instead of the one Phase 3 seed."
        ),
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seeds-per-problem", type=int, default=4)
    parser.add_argument("--time-bins", type=int, default=20)
    return parser.parse_args()


def _plot_metric_trends(summary: pd.DataFrame, output: Path) -> None:
    specifications = {
        "correct": "Correctness",
        "trajectory_token_count": "Reasoning tokens",
        "normalized_entropy_mean": "Normalized entropy",
        "geometry_mean_relative_velocity": "Relative hidden velocity",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    for axis, (metric, title) in zip(axes.flat, specifications.items(), strict=True):
        scoped = summary[summary["metric"] == metric]
        for model, group in scoped.groupby("model_key"):
            group = group.sort_values("level")
            axis.plot(group["level"], group["mean"], marker="o", label=model)
            axis.fill_between(group["level"], group["ci_low"], group["ci_high"], alpha=0.12)
        axis.set(title=title, xlabel="MATH difficulty level", xticks=range(1, 6))
        if metric == "correct":
            axis.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Matched cross-model difficulty response")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_interaction_heatmap(trends: dict, output: Path) -> None:
    rows = []
    for model, metrics in trends["models"].items():
        rows.append(
            {
                "model_key": model,
                **{metric: value["slope_per_level"] for metric, value in metrics.items()},
            }
        )
    frame = pd.DataFrame(rows).set_index("model_key")
    standardized = frame.sub(frame.mean()).div(frame.std().replace(0, np.nan))
    figure, axis = plt.subplots(figsize=(12, 5))
    sns.heatmap(standardized, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Standardized model-specific difficulty slopes")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_token_entropy(token_summary: pd.DataFrame, output: Path) -> None:
    scoped = token_summary[
        (token_summary["signal"] == "normalized_entropy") & token_summary["correct"]
    ].copy()
    scoped["progress"] = (scoped["time_bin"] + 0.5) / scoped["time_bin"].nunique()
    grid = sns.relplot(
        data=scoped,
        x="progress",
        y="mean",
        hue="model_key",
        col="level",
        col_wrap=3,
        kind="line",
        height=4,
    )
    grid.set_axis_labels("Normalized reasoning progress", "Normalized entropy")
    grid.set_titles("MATH level {col_name}")
    grid.figure.suptitle("Cross-model entropy dynamics on correct trajectories", y=1.03)
    grid.figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(grid.figure)


def _cross_model_prefix_entropy(features_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Plot leakage-safe early entropy curves separately for every model family."""

    rows = []
    for prefix in PREFIXES:
        frame = pd.read_parquet(features_dir / f"features_prefix_{prefix}.parquet")
        active = frame[frame["full_trajectory_token_count"] >= prefix]
        for (model, level, correct), group in active.groupby(["model_key", "level", "correct"]):
            problem_means = group.groupby("problem_id")["normalized_entropy_mean"].mean()
            rows.append({"model_key": model, "level": level, "correct": correct, "prefix_length": prefix, "mean": problem_means.mean(), "trajectories": len(group), "problems": len(problem_means)})
    summary = pd.DataFrame(rows)
    summary.to_parquet(output_dir / "cross_model_fixed_prefix_entropy.parquet", index=False)
    grid = sns.relplot(data=summary, x="prefix_length", y="mean", hue="correct", col="model_key", row="level", kind="line", marker="o", facet_kws={"sharey": True}, height=2.3, aspect=1.35)
    grid.set(xscale="log", xlabel="Observed reasoning tokens", ylabel="Mean normalized entropy")
    grid.set_titles("{row_name} · {row_var} | {col_name}")
    grid.figure.suptitle("Fixed-prefix entropy by terminal correctness and model", y=1.01)
    grid.figure.savefig(output_dir / "cross_model_early_entropy_by_level.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    risk = summary.groupby(["model_key", "correct", "prefix_length"], as_index=False)["trajectories"].sum()
    grid = sns.relplot(data=risk, x="prefix_length", y="trajectories", hue="correct", col="model_key", kind="line", marker="o", height=3.5, aspect=1.25)
    grid.set(xscale="log", xlabel="Fixed prefix", ylabel="Trajectories still active")
    grid.figure.suptitle("Risk set retained at each fixed prefix", y=1.04)
    grid.figure.savefig(output_dir / "cross_model_early_entropy_risk_set.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    return summary


def _spearman_interval(
    frame: pd.DataFrame,
    *,
    entropy_column: str,
    token_column: str,
    repetitions: int,
    seed: int,
    minimum_problems: int = 3,
) -> dict[str, float]:
    """Estimate a problem-clustered Spearman association with a bootstrap interval."""

    observed = frame[["problem_id", "level", entropy_column, token_column]].dropna().copy()
    observed = observed[observed[token_column] > 0]
    entropy = observed[entropy_column].to_numpy(dtype=float)
    log_tokens = np.log(observed[token_column].to_numpy(dtype=float))
    levels = observed["level"].to_numpy()
    problem_ids = observed["problem_id"].to_numpy()
    unique_problem_ids = pd.unique(problem_ids)
    token_count_varies = bool(np.ptp(log_tokens) > 0) if len(log_tokens) else False
    if len(unique_problem_ids) < minimum_problems:
        return {
            "trajectories": len(observed),
            "problems": len(unique_problem_ids),
            "token_count_varies": token_count_varies,
            "raw_spearman_rho": float("nan"),
            "raw_ci_low": float("nan"),
            "raw_ci_high": float("nan"),
            "level_adjusted_spearman_rho": float("nan"),
            "level_adjusted_ci_low": float("nan"),
            "level_adjusted_ci_high": float("nan"),
        }

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            return float(spearmanr(left, right).statistic)

    raw = correlation(entropy, log_tokens)
    entropy_adjusted = entropy - pd.Series(entropy).groupby(levels).transform("mean").to_numpy(
        dtype=float
    )
    token_adjusted = log_tokens - pd.Series(log_tokens).groupby(levels).transform("mean").to_numpy(
        dtype=float
    )
    level_adjusted = correlation(entropy_adjusted, token_adjusted)

    rng = np.random.default_rng(seed)
    group_indices = [np.flatnonzero(problem_ids == problem_id) for problem_id in unique_problem_ids]
    raw_draws: list[float] = []
    adjusted_draws: list[float] = []
    for _ in range(repetitions):
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[index] for index in sampled_groups])
        sampled_levels = levels[indices]
        sampled_entropy = entropy[indices]
        sampled_tokens = log_tokens[indices]
        raw_value = correlation(sampled_entropy, sampled_tokens)
        if np.isfinite(raw_value):
            raw_draws.append(raw_value)
        sampled_entropy_adjusted = sampled_entropy - pd.Series(sampled_entropy).groupby(
            sampled_levels
        ).transform("mean").to_numpy(dtype=float)
        sampled_token_adjusted = sampled_tokens - pd.Series(sampled_tokens).groupby(
            sampled_levels
        ).transform("mean").to_numpy(dtype=float)
        adjusted_value = correlation(sampled_entropy_adjusted, sampled_token_adjusted)
        if np.isfinite(adjusted_value):
            adjusted_draws.append(adjusted_value)

    def interval(value: float, draws: list[float]) -> tuple[float, float, float]:
        values = np.asarray(draws, dtype=float)
        return (
            value,
            float(np.quantile(values, 0.025)) if len(values) else float("nan"),
            float(np.quantile(values, 0.975)) if len(values) else float("nan"),
        )

    raw_value, raw_low, raw_high = interval(raw, raw_draws)
    adjusted_value, adjusted_low, adjusted_high = interval(level_adjusted, adjusted_draws)
    return {
        "trajectories": len(observed),
        "problems": len(unique_problem_ids),
        "token_count_varies": token_count_varies,
        "raw_spearman_rho": raw_value,
        "raw_ci_low": raw_low,
        "raw_ci_high": raw_high,
        "level_adjusted_spearman_rho": adjusted_value,
        "level_adjusted_ci_low": adjusted_low,
        "level_adjusted_ci_high": adjusted_high,
    }


def _plot_early_entropy_token_scatter(
    observations: pd.DataFrame,
    associations: pd.DataFrame,
    output: Path,
) -> None:
    """Plot early entropy against the total reasoning tokens eventually used."""

    models = [model for model in MODELS if model in set(observations["model_key"])]
    figure, axes = plt.subplots(
        len(models),
        len(TOKEN_USAGE_PREFIXES),
        figsize=(21, 10),
        sharey=True,
        constrained_layout=True,
    )
    palette = {"Correct": "#4c78a8", "Incorrect": "#e45756"}
    for row, model in enumerate(models):
        for column, prefix in enumerate(TOKEN_USAGE_PREFIXES):
            axis = axes[row, column]
            scoped = observations[
                (observations["model_key"] == model)
                & (observations["prefix_length"] == prefix)
            ]
            sns.scatterplot(
                data=scoped,
                x="early_entropy",
                y="total_reasoning_tokens",
                hue="terminal_outcome",
                palette=palette,
                alpha=0.74,
                s=42,
                linewidth=0,
                legend=False,
                ax=axis,
            )
            if scoped["early_entropy"].nunique() > 1:
                coefficients = np.polyfit(
                    scoped["early_entropy"], np.log10(scoped["total_reasoning_tokens"]), 1
                )
                x_values = np.linspace(
                    scoped["early_entropy"].min(), scoped["early_entropy"].max(), 100
                )
                axis.plot(
                    x_values,
                    10 ** np.polyval(coefficients, x_values),
                    color="#2f2f2f",
                    linewidth=1.5,
                    zorder=1,
                )
            association = associations[
                (associations["model_key"] == model)
                & (associations["prefix_length"] == prefix)
            ].iloc[0]
            axis.set_yscale("log")
            axis.set_title(
                f"{MODEL_LABELS.get(model, model)}\nfirst {prefix} tokens · "
                f"ρ={association['raw_spearman_rho']:.2f}",
                fontsize=11,
            )
            if row == len(models) - 1:
                axis.set_xlabel("Mean normalized entropy at prefix")
            else:
                axis.set_xlabel("")
            if column == 0:
                axis.set_ylabel("Total reasoning tokens\n(log scale)")
            else:
                axis.set_ylabel("")
    figure.suptitle(
        "Does early uncertainty predict a longer reasoning trajectory? "
        "(blue: correct; red: incorrect)",
        fontsize=17,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_early_entropy_token_associations(associations: pd.DataFrame, output: Path) -> None:
    """Show raw and difficulty-adjusted early entropy/token associations."""

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True, constrained_layout=True)
    specifications = (
        ("raw_spearman_rho", "raw_ci_low", "raw_ci_high", "Unadjusted association"),
        (
            "level_adjusted_spearman_rho",
            "level_adjusted_ci_low",
            "level_adjusted_ci_high",
            "Within-MATH-level association",
        ),
    )
    for axis, (value_column, low_column, high_column, title) in zip(
        axes, specifications, strict=True
    ):
        for model, group in associations.groupby("model_key", sort=False):
            group = group.sort_values("prefix_length")
            values = group[value_column].to_numpy(dtype=float)
            low = group[low_column].to_numpy(dtype=float)
            high = group[high_column].to_numpy(dtype=float)
            axis.errorbar(
                group["prefix_length"],
                values,
                yerr=np.vstack((values - low, high - values)),
                marker="o",
                capsize=3,
                label=MODEL_LABELS.get(model, model),
            )
        axis.axhline(0, color="black", linewidth=1, alpha=0.65)
        axis.set_xscale("log", base=2)
        axis.set_xticks(TOKEN_USAGE_PREFIXES)
        axis.set_xticklabels([str(prefix) for prefix in TOKEN_USAGE_PREFIXES])
        axis.set(
            xlabel="Observed reasoning tokens",
            ylabel="Spearman ρ: early entropy vs. total tokens",
            title=title,
        )
    axes[0].legend(fontsize=9)
    figure.suptitle(
        "Early uncertainty and eventual reasoning-token use", fontsize=17
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_early_entropy_token_associations_by_correctness(
    associations: pd.DataFrame,
    output: Path,
) -> None:
    """Compare entropy/token associations for correct and incorrect trajectories."""

    models = [model for model in MODELS if model in set(associations["model_key"])]
    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(14.5, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    specifications = (
        ("raw_spearman_rho", "raw_ci_low", "raw_ci_high", "Unadjusted association"),
        (
            "level_adjusted_spearman_rho",
            "level_adjusted_ci_low",
            "level_adjusted_ci_high",
            "Within-MATH-level association",
        ),
    )
    colors = {"Correct": "#4c78a8", "Incorrect": "#e45756"}
    for row, model in enumerate(models):
        model_frame = associations[associations["model_key"] == model]
        source = str(model_frame["source"].iloc[0])
        for column, (value_column, low_column, high_column, title) in enumerate(
            specifications
        ):
            axis = axes[row, column]
            for outcome, color in colors.items():
                group = model_frame[model_frame["terminal_outcome"] == outcome].sort_values(
                    "prefix_length"
                )
                usable = group[np.isfinite(group[value_column])]
                if not usable.empty:
                    values = usable[value_column].to_numpy(dtype=float)
                    low = usable[low_column].to_numpy(dtype=float)
                    high = usable[high_column].to_numpy(dtype=float)
                    axis.errorbar(
                        usable["prefix_length"],
                        values,
                        yerr=np.vstack((values - low, high - values)),
                        marker="o",
                        capsize=3,
                        color=color,
                        label=outcome,
                    )
                else:
                    support = int(group["problems"].max()) if not group.empty else 0
                    if support < 10:
                        reason = f"n={support} (<10)"
                    elif not bool(group["token_count_varies"].any()):
                        reason = f"constant token count (n={support})"
                    else:
                        reason = f"not estimable (n={support})"
                    axis.text(
                        0.98,
                        0.08 if outcome == "Correct" else 0.17,
                        f"{outcome}: {reason}",
                        transform=axis.transAxes,
                        color=color,
                        ha="right",
                        fontsize=9,
                    )
            axis.axhline(0, color="black", linewidth=1, alpha=0.65)
            axis.set_xscale("log", base=2)
            axis.set_xticks(TOKEN_USAGE_PREFIXES)
            axis.set_xticklabels([str(prefix) for prefix in TOKEN_USAGE_PREFIXES])
            if row == 0:
                axis.set_title(title)
            if row == len(models) - 1:
                axis.set_xlabel("Observed reasoning tokens")
            if column == 0:
                axis.set_ylabel(
                    f"{MODEL_LABELS.get(model, model)} ({source})\n"
                    "Spearman ρ: entropy vs. total tokens"
                )
            low_support = model_frame[
                (model_frame["terminal_outcome"] == "Incorrect")
                & (model_frame["prefix_length"] == TOKEN_USAGE_PREFIXES[0])
            ]
            if not low_support.empty and int(low_support["problems"].iloc[0]) < 10:
                axis.text(
                    0.98,
                    0.25,
                    "Incorrect: "
                    f"{int(low_support['trajectories'].iloc[0])} trajectories / "
                    f"{int(low_support['problems'].iloc[0])} questions",
                    transform=axis.transAxes,
                    color=colors["Incorrect"],
                    ha="right",
                    fontsize=8.5,
                )
    handles = [
        plt.Line2D([], [], marker="o", color=color, label=outcome)
        for outcome, color in colors.items()
    ]
    axes[0, 0].legend(handles=handles, fontsize=9, loc="lower left")
    figure.suptitle(
        "Does early entropy relate to token use differently for correct and incorrect trajectories?",
        fontsize=16,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _early_entropy_token_usage_analysis(
    features_dir: Path,
    output_dir: Path,
    *,
    repetitions: int,
    gemma_phase2_features_dir: Path | None = None,
) -> pd.DataFrame:
    """Relate fixed-prefix entropy to the final token cost without peeking past the prefix."""

    observations: list[pd.DataFrame] = []
    associations: list[dict[str, object]] = []
    for prefix in TOKEN_USAGE_PREFIXES:
        frame = pd.read_parquet(features_dir / f"features_prefix_{prefix}.parquet")
        frame["source"] = "Phase 3 · seed 11"
        if gemma_phase2_features_dir is not None:
            phase2_gemma = pd.read_parquet(
                gemma_phase2_features_dir / f"features_prefix_{prefix}.parquet"
            )
            phase2_gemma = phase2_gemma[phase2_gemma["model_key"] == "gemma4_e4b"].copy()
            phase2_gemma["source"] = "Phase 2 · 4 seeds"
            frame = pd.concat(
                [frame[frame["model_key"] != "gemma4_e4b"], phase2_gemma],
                ignore_index=True,
            )
        active = frame[frame["full_trajectory_token_count"] >= prefix].copy()
        active = active.rename(
            columns={
                "normalized_entropy_mean": "early_entropy",
                "full_trajectory_token_count": "total_reasoning_tokens",
            }
        )
        active["prefix_length"] = prefix
        active["terminal_outcome"] = np.where(active["correct"], "Correct", "Incorrect")
        observations.append(
            active[
                [
                    "model_key",
                    "problem_id",
                    "level",
                    "correct",
                    "prefix_length",
                    "early_entropy",
                    "total_reasoning_tokens",
                    "terminal_outcome",
                    "source",
                ]
            ]
        )
        for (model, source), model_frame in active.groupby(["model_key", "source"], sort=False):
            associations.append(
                {
                    "model_key": model,
                    "source": source,
                    "prefix_length": prefix,
                    **_spearman_interval(
                        model_frame,
                        entropy_column="early_entropy",
                        token_column="total_reasoning_tokens",
                        repetitions=repetitions,
                        seed=20260805 + prefix,
                    ),
                }
            )
    observation_frame = pd.concat(observations, ignore_index=True)
    association_frame = pd.DataFrame(associations)
    observation_frame.to_parquet(
        output_dir / "early_entropy_token_usage_observations.parquet", index=False
    )
    association_frame.to_parquet(
        output_dir / "early_entropy_token_usage_associations.parquet", index=False
    )
    outcome_associations: list[dict[str, object]] = []
    for (model, source, outcome, prefix), group in observation_frame.groupby(
        ["model_key", "source", "terminal_outcome", "prefix_length"], sort=False
    ):
        outcome_associations.append(
            {
                "model_key": model,
                "source": source,
                "terminal_outcome": outcome,
                "prefix_length": prefix,
                **_spearman_interval(
                    group,
                    entropy_column="early_entropy",
                    token_column="total_reasoning_tokens",
                    repetitions=repetitions,
                    seed=20260820 + int(prefix),
                    minimum_problems=3,
                ),
            }
        )
    outcome_association_frame = pd.DataFrame(outcome_associations)
    outcome_association_frame.to_parquet(
        output_dir / "early_entropy_token_usage_by_correctness.parquet", index=False
    )
    _plot_early_entropy_token_scatter(
        observation_frame,
        association_frame,
        output_dir / "early_entropy_vs_token_usage.png",
    )
    _plot_early_entropy_token_associations(
        association_frame,
        output_dir / "early_entropy_token_usage_associations.png",
    )
    _plot_early_entropy_token_associations_by_correctness(
        outcome_association_frame,
        output_dir
        / (
            "early_entropy_token_usage_by_correctness_with_phase2_gemma.png"
            if gemma_phase2_features_dir is not None
            else "early_entropy_token_usage_by_correctness.png"
        ),
    )
    return association_frame


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    sns.set_theme(style="whitegrid", context="talk")
    frame = pd.read_parquet(args.features)
    design = validate_difficulty_design(
        frame,
        expected_models=MODELS,
        seeds_per_problem=args.seeds_per_problem,
    )
    summary = difficulty_metric_summary(
        frame,
        METRICS,
        repetitions=args.bootstrap_repetitions,
    )
    trends = level_trends(frame, METRICS, repetitions=args.bootstrap_repetitions)
    consistency = seed_consistency(frame)
    binned, spikes = binned_token_dynamics(args.run_dir, bins=args.time_bins)
    token_summary = token_dynamics_summary(
        binned,
        repetitions=args.bootstrap_repetitions,
    )
    prefix_entropy = _cross_model_prefix_entropy(args.features_dir, output_dir)
    entropy_token_usage = _early_entropy_token_usage_analysis(
        args.features_dir,
        output_dir,
        repetitions=args.bootstrap_repetitions,
        gemma_phase2_features_dir=args.gemma_phase2_features_dir,
    )
    contrasts = {}
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            label = f"{right}_minus_{left}"
            contrasts[label] = {
                metric: paired_condition_difference(
                    frame,
                    condition_column="model_key",
                    left=left,
                    right=right,
                    value_column=metric,
                    repetitions=args.bootstrap_repetitions,
                )
                for metric in METRICS
            }
    summary.to_parquet(output_dir / "cross_model_difficulty_summary.parquet", index=False)
    consistency.to_parquet(output_dir / "cross_model_seed_consistency.parquet", index=False)
    binned.to_parquet(output_dir / "cross_model_token_dynamics_binned.parquet", index=False)
    token_summary.to_parquet(output_dir / "cross_model_token_dynamics_summary.parquet", index=False)
    spikes.to_parquet(output_dir / "cross_model_entropy_spikes.parquet", index=False)
    write_json_atomic(output_dir / "difficulty_design_validation.json", design)
    write_json_atomic(output_dir / "level_trends.json", trends)
    write_json_atomic(output_dir / "paired_model_contrasts.json", contrasts)
    _plot_metric_trends(summary, output_dir / "cross_model_difficulty_trends.png")
    _plot_interaction_heatmap(trends, output_dir / "model_difficulty_interactions.png")
    _plot_token_entropy(token_summary, output_dir / "cross_model_entropy_dynamics.png")
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "inconclusive",
            "next_decision": "continue",
            "summary": "Matched three-model MATH difficulty comparison completed.",
            "metrics": {
                "trajectories": len(frame),
                "problems": int(frame["problem_id"].nunique()),
                "models": len(MODELS),
                "levels": 5,
                "paired_model_contrasts": len(contrasts),
                "fixed_prefix_entropy_rows": len(prefix_entropy),
                "early_entropy_token_usage_rows": len(entropy_token_usage),
            },
            "warnings": [
                "Raw hidden-state coordinates are not compared across model families.",
                "GSM8K is reserved for later out-of-domain confirmation.",
                "Fixed-prefix entropy comparisons are descriptive; Phase 4 evaluates held-out early failure after its matched 16K rerun.",
                *(
                    [
                        "Phase 3 uses one fixed decoding seed; its seed-consistency table is "
                        "descriptive only. Stability claims are scoped to Phase 2."
                    ]
                    if args.seeds_per_problem == 1
                    else []
                ),
            ],
        },
    )
    print(output_dir / "phase_summary.json")


if __name__ == "__main__":
    main()
