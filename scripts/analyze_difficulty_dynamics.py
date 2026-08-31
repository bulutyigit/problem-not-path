#!/usr/bin/env python
"""Analyze Gemma reasoning dynamics across the five MATH difficulty levels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasonbench.evaluation.difficulty import (
    binned_token_dynamics,
    difficulty_metric_summary,
    level_trends,
    seed_consistency,
    token_dynamics_summary,
    validate_difficulty_design,
)
from reasonbench.evaluation.failure_dynamics import (
    bootstrap_residualized_associations,
    problem_seed_instability,
    residualized_cross_block_correlations,
)
from reasonbench.evaluation.metrics import paired_clustered_metric_difference
from reasonbench.evaluation.predictor import evaluate_one
from reasonbench.storage import ensure_directory, write_json_atomic

METRICS = (
    "correct",
    "trajectory_token_count",
    "elapsed_seconds",
    "normalized_entropy_mean",
    "normalized_entropy_slope",
    "surprisal_mean",
    "top1_top2_probability_margin_mean",
    "geometry_mean_relative_velocity",
    "geometry_mean_cosine_drift",
    "geometry_trajectory_efficiency",
)
EXPECTED_MODEL = "gemma4_e4b"
PREFIX_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048)
ONSET_PREFIXES = (16, 32, 64, 128)
PREFIX_FEATURE_SETS = (
    "early_baseline",
    "early_confidence",
    "early_geometry",
    "early_spectral",
    "early_full_without_spectral",
    "early_full",
)
FAILURE_SUMMARY_COLUMNS = (
    "prefix_length",
    "feature_set",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "coverage",
    "trajectories",
    "problems",
    "calibration_applied",
)
FAILURE_IMPROVEMENT_COLUMNS = (
    "prefix_length",
    "feature_set",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "coverage",
)
ONSET_UNCERTAINTY_FEATURES = (
    "normalized_entropy_mean",
    "normalized_entropy_max_rise",
    "surprisal_mean",
    "top1_top2_probability_margin_mean",
)
GEOMETRY_ASSOCIATION_FEATURES = (
    "geometry_normalized_path_length",
    "geometry_mean_relative_velocity",
    "geometry_velocity_variance",
    "geometry_mean_cosine_drift",
    "geometry_cosine_drift_variance",
    "geometry_trajectory_efficiency",
    "geometry_turning_angle_mean",
    "geometry_turning_angle_variance",
)
SPECTRAL_ASSOCIATION_FEATURES = (
    "spectral_normalized_entropy_entropy",
    "spectral_normalized_entropy_low_energy_ratio",
    "spectral_normalized_entropy_high_energy_ratio",
    "spectral_surprisal_entropy",
    "spectral_surprisal_high_energy_ratio",
    "spectral_top1_top2_logit_margin_entropy",
    "spectral_relative_l2_step_entropy",
    "spectral_relative_l2_step_high_energy_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--time-bins", type=int, default=20)
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _plot_metric_grid(summary: pd.DataFrame, output: Path) -> None:
    selected = {
        "correct": "Correctness",
        "trajectory_token_count": "Reasoning tokens",
        "elapsed_seconds": "Generation time (s)",
        "normalized_entropy_mean": "Mean normalized entropy",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (metric, title) in zip(axes.flat, selected.items(), strict=True):
        scoped = summary[summary["metric"] == metric].sort_values("level")
        axis.plot(scoped["level"], scoped["mean"], marker="o", color="#2468a2")
        axis.fill_between(scoped["level"], scoped["ci_low"], scoped["ci_high"], alpha=0.2)
        axis.set(title=title, xlabel="MATH difficulty level", xticks=range(1, 6))
        if metric == "correct":
            axis.set_ylim(0, 1)
    figure.suptitle("Gemma outcomes and reasoning dynamics by MATH difficulty")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_token_curves(token_summary: pd.DataFrame, output: Path) -> None:
    signals = {
        "normalized_entropy": "Normalized entropy",
        "surprisal": "Sampled-token surprisal",
        "top1_top2_probability_margin": "Top-1/Top-2 probability margin",
        "relative_l2_step": "Relative hidden-state step",
    }
    figure, axes = plt.subplots(2, 2, figsize=(17, 12))
    for axis, (signal, title) in zip(axes.flat, signals.items(), strict=True):
        scoped = token_summary[(token_summary["signal"] == signal) & token_summary["correct"]]
        for level, group in scoped.groupby("level"):
            group = group.sort_values("time_bin")
            x = (group["time_bin"].to_numpy() + 0.5) / group["time_bin"].nunique()
            axis.plot(x, group["mean"], label=f"Level {int(level)}")
            axis.fill_between(x, group["ci_low"], group["ci_high"], alpha=0.12)
        axis.set(title=title, xlabel="Normalized reasoning progress", ylabel="Mean")
    axes[0, 0].legend(ncol=2, fontsize=10)
    figure.suptitle("Correct-trajectory token dynamics by difficulty")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_correctness_entropy(token_summary: pd.DataFrame, output: Path) -> None:
    scoped = token_summary[token_summary["signal"] == "normalized_entropy"].copy()
    scoped["progress"] = (scoped["time_bin"] + 0.5) / scoped["time_bin"].nunique()
    grid = sns.relplot(
        data=scoped,
        x="progress",
        y="mean",
        hue="correct",
        col="level",
        col_wrap=3,
        kind="line",
        facet_kws={"sharey": True},
        height=4,
    )
    grid.set_axis_labels("Normalized reasoning progress", "Normalized entropy")
    grid.set_titles("MATH level {col_name}")
    grid.figure.suptitle("Entropy dynamics: correct versus incorrect", y=1.03)
    grid.figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(grid.figure)


def _plot_heatmap(summary: pd.DataFrame, output: Path) -> None:
    scoped = summary.pivot(index="metric", columns="level", values="mean")
    standardized = scoped.sub(scoped.mean(axis=1), axis=0).div(
        scoped.std(axis=1).replace(0, np.nan), axis=0
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(standardized, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Within-metric standardized difficulty profile")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_seed_consistency(consistency: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.countplot(data=consistency, x="successes", hue="level", palette="viridis", ax=axis)
    axis.set(xlabel="Correct seeds out of four", ylabel="Problems", title="Seed consistency")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_spikes(spikes: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(
        data=spikes,
        x="level",
        y="maximum_entropy_rise",
        hue="correct",
        ax=axes[0],
    )
    sns.boxplot(
        data=spikes,
        x="level",
        y="spike_relative_position",
        hue="correct",
        ax=axes[1],
    )
    axes[0].set_title("Largest local entropy rise")
    axes[1].set_title("Timing of largest entropy rise")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _display_name(feature: str) -> str:
    return (
        feature.replace("mean__", "mean · ")
        .replace("sd__", "seed SD · ")
        .replace("geometry_", "geometry · ")
        .replace("spectral_", "spectral · ")
        .replace("normalized_entropy", "entropy")
        .replace("top1_top2_", "margin ")
        .replace("relative_l2_step", "hidden step")
        .replace("_", " ")
    )


def _plot_failure_prefixes(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    labels = {
        "early_baseline": "difficulty + category + observed tokens",
        "early_confidence": "+ confidence dynamics",
        "early_geometry": "+ geometry",
        "early_spectral": "+ spectral",
        "early_full_without_spectral": "+ confidence + geometry",
        "early_full": "+ all trajectory features",
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("auprc", "brier"),
        ("Failure AUPRC", "Failure Brier score (lower is better)"),
        strict=True,
    ):
        scoped = summary[summary["metric"] == metric]
        for feature_set, group in scoped.groupby("feature_set", sort=False):
            group = group.sort_values("prefix_length")
            axis.plot(
                group["prefix_length"],
                group["value"],
                marker="o",
                label=labels.get(feature_set, feature_set),
            )
            axis.fill_between(
                group["prefix_length"],
                group["ci_low"],
                group["ci_high"],
                alpha=0.12,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(scoped["prefix_length"].unique()))
        axis.set_xticklabels(
            [str(int(value)) for value in sorted(scoped["prefix_length"].unique())]
        )
        axis.set_xlabel("Observed reasoning tokens")
        axis.set_ylabel(title)
        axis.set_title(title)
    axes[0].legend(fontsize=9)
    figure.suptitle("Gemma terminal-failure prediction from partial trajectories")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_incremental_prefix_value(improvements: pd.DataFrame, output: Path) -> None:
    scoped = improvements[improvements["metric"] == "auprc"].copy()
    if scoped.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6))
    for feature_set, group in scoped.groupby("feature_set", sort=False):
        group = group.sort_values("prefix_length")
        axis.plot(
            group["prefix_length"],
            group["value"],
            marker="o",
            label=feature_set.replace("early_", "").replace("_", " "),
        )
        axis.fill_between(
            group["prefix_length"],
            group["ci_low"],
            group["ci_high"],
            alpha=0.14,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xscale("log", base=2)
    ticks = sorted(scoped["prefix_length"].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(int(value)) for value in ticks])
    axis.set(
        xlabel="Observed reasoning tokens",
        ylabel="AUPRC improvement over difficulty/category/token baseline",
        title="Incremental early-warning value",
    )
    axis.legend(fontsize=9)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _failure_prediction_analysis(
    features_dir: Path | None,
    output_dir: Path,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if features_dir is None:
        warnings.append(
            "Prefix feature directory was not provided; failure prediction was skipped."
        )
        return pd.DataFrame(), pd.DataFrame(), warnings
    summary_rows: list[dict[str, object]] = []
    improvement_rows: list[dict[str, object]] = []
    prediction_dir = ensure_directory(output_dir / "failure_predictions")
    prefix_paths = sorted(features_dir.glob("features_prefix_*.parquet"))
    if not prefix_paths:
        warnings.append(
            "No fixed-prefix feature tables were found; failure prediction was skipped."
        )
        return pd.DataFrame(), pd.DataFrame(), warnings
    for feature_path in prefix_paths:
        prefix = int(feature_path.stem.rsplit("_", maxsplit=1)[-1])
        frame = pd.read_parquet(feature_path)
        eligible = frame[frame["full_trajectory_token_count"] >= prefix].copy()
        coverage = len(eligible) / len(frame) if len(frame) else 0.0
        eligible["terminal_correct"] = eligible["correct"].astype(bool)
        eligible["correct"] = ~eligible["terminal_correct"]
        train_classes = eligible.loc[eligible["research_split"] == "train", "correct"].nunique()
        test_classes = eligible.loc[eligible["research_split"] == "test", "correct"].nunique()
        if eligible.empty or train_classes < 2 or test_classes < 2:
            warnings.append(
                f"Prefix {prefix} lacked both terminal-failure classes in train or test; skipped."
            )
            continue
        evaluated = {}
        feature_sets = (
            tuple(
                feature_set
                for feature_set in PREFIX_FEATURE_SETS
                if feature_set not in {"early_spectral", "early_full"}
            )
            if prefix < 64
            else PREFIX_FEATURE_SETS
        )
        for feature_set in feature_sets:
            result = evaluate_one(
                eligible,
                feature_set=feature_set,
                model_name="logistic_regression",
                bootstrap_repetitions=repetitions,
            )
            evaluated[feature_set] = result
            predictions = result.predictions.rename(columns={"correct": "terminal_failure"})
            predictions.to_parquet(
                prediction_dir / f"predictions_{feature_set}_prefix_{prefix}.parquet",
                index=False,
            )
            for metric, interval in result.metrics.items():
                summary_rows.append(
                    {
                        "prefix_length": prefix,
                        "feature_set": feature_set,
                        "metric": metric,
                        **interval,
                        "coverage": coverage,
                        "trajectories": len(eligible),
                        "problems": int(eligible["problem_id"].nunique()),
                        "calibration_applied": result.calibration_applied,
                    }
                )
        baseline = evaluated["early_baseline"]
        for feature_set, result in evaluated.items():
            if feature_set == "early_baseline":
                continue
            for metric in ("auroc", "auprc", "brier"):
                difference = paired_clustered_metric_difference(
                    baseline.predictions,
                    result.predictions,
                    metric=metric,
                    repetitions=repetitions,
                )
                improvement_rows.append(
                    {
                        "prefix_length": prefix,
                        "feature_set": feature_set,
                        "metric": metric,
                        **difference,
                        "coverage": coverage,
                    }
                )
    # Preserve the schema when class support is insufficient at every prefix.
    # This is a valid empirical outcome, not an analysis failure: downstream
    # reporting can then mark failure prediction as underpowered.
    summary = pd.DataFrame(summary_rows, columns=FAILURE_SUMMARY_COLUMNS)
    improvements = pd.DataFrame(improvement_rows, columns=FAILURE_IMPROVEMENT_COLUMNS)
    summary.to_parquet(output_dir / "failure_prediction_by_prefix.parquet", index=False)
    improvements.to_parquet(output_dir / "failure_prediction_improvements.parquet", index=False)
    _plot_failure_prefixes(summary, output_dir / "failure_prediction_by_prefix.png")
    _plot_incremental_prefix_value(
        improvements,
        output_dir / "failure_prediction_incremental_value.png",
    )
    return summary, improvements, warnings


def _plot_cross_block(matrix: pd.DataFrame, output: Path) -> None:
    if matrix.empty:
        return
    figure, axis = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        matrix.rename(index=_display_name, columns=_display_name),
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=axis,
    )
    axis.set(
        xlabel="Spectral feature",
        ylabel="Geometry feature",
        title="Geometry–spectral associations after level/category/length control",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_instability_associations(associations: pd.DataFrame, output: Path) -> None:
    scoped = associations.dropna(subset=["association"]).sort_values("association")
    if scoped.empty:
        return
    figure, axis = plt.subplots(figsize=(12, max(8, 0.35 * len(scoped))))
    positions = np.arange(len(scoped))
    colors = ["#d97732" if "spectral" in feature else "#336b9b" for feature in scoped["feature"]]
    for position, (_, row), color in zip(positions, scoped.iterrows(), colors, strict=True):
        axis.errorbar(
            row["association"],
            position,
            xerr=[
                [row["association"] - row["ci_low"]],
                [row["ci_high"] - row["association"]],
            ],
            fmt="none",
            ecolor=color,
            capsize=3,
            alpha=0.9,
        )
    axis.scatter(scoped["association"], positions, c=colors, s=45, zorder=3)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([_display_name(feature) for feature in scoped["feature"]], fontsize=9)
    axis.set(
        xlabel="Residualized Spearman association with seed instability",
        title="Geometry and spectral signatures of problem-level seed instability",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _geometry_spectral_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry = [feature for feature in GEOMETRY_ASSOCIATION_FEATURES if feature in frame]
    spectral = [feature for feature in SPECTRAL_ASSOCIATION_FEATURES if feature in frame]
    aggregate_features = ["trajectory_token_count", *geometry, *spectral]
    problem_table = problem_seed_instability(frame, aggregate_features)
    problem_table.to_parquet(output_dir / "problem_seed_instability.parquet", index=False)
    association_features = [
        f"{prefix}__{feature}" for feature in [*geometry, *spectral] for prefix in ("mean", "sd")
    ]
    associations = bootstrap_residualized_associations(
        problem_table,
        "seed_instability",
        association_features,
        repetitions=repetitions,
    )
    associations.to_parquet(output_dir / "seed_instability_associations.parquet", index=False)
    matrix = residualized_cross_block_correlations(
        problem_table,
        [f"mean__{feature}" for feature in geometry],
        [f"mean__{feature}" for feature in spectral],
    )
    matrix.to_parquet(output_dir / "geometry_spectral_correlations.parquet")
    _plot_cross_block(matrix, output_dir / "geometry_spectral_correlations.png")
    _plot_instability_associations(
        associations,
        output_dir / "seed_instability_associations.png",
    )
    return problem_table, associations


def _plot_onset_uncertainty(associations: pd.DataFrame, output: Path) -> None:
    if associations.empty:
        return
    targets = (
        ("failure_rate", "Terminal failure rate\n(controlling level + category)"),
        ("level", "MATH difficulty level\n(controlling failure rate + category)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for axis, (target, title) in zip(axes, targets, strict=True):
        scoped = associations[associations["target"] == target].copy()
        values = scoped.pivot(index="feature", columns="prefix_length", values="association")
        values = values.reindex(
            index=[f"mean__{feature}" for feature in ONSET_UNCERTAINTY_FEATURES],
            columns=ONSET_PREFIXES,
        )
        annotations = values.copy().astype(object)
        interval_lookup = scoped.set_index(["feature", "prefix_length"])
        for feature in values.index:
            for prefix in values.columns:
                value = values.loc[feature, prefix]
                if not np.isfinite(value):
                    annotations.loc[feature, prefix] = "NA"
                    continue
                interval = interval_lookup.loc[(feature, prefix)]
                excludes_zero = bool(
                    (interval["ci_low"] > 0 and interval["ci_high"] > 0)
                    or (interval["ci_low"] < 0 and interval["ci_high"] < 0)
                )
                annotations.loc[feature, prefix] = f"{value:.2f}{'*' if excludes_zero else ''}"
        sns.heatmap(
            values.rename(index=_display_name),
            annot=annotations.to_numpy(),
            fmt="",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            ax=axis,
        )
        axis.set(
            xlabel="Observed thinking tokens",
            ylabel="First-token uncertainty signal",
            title=title,
        )
    figure.suptitle(
        "Information in the first thinking tokens (* problem-bootstrap CI excludes zero)"
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _reasoning_onset_analysis(
    features_dir: Path | None,
    output_dir: Path,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if features_dir is None:
        warnings.append("Prefix feature directory was not provided; onset analysis was skipped.")
        return pd.DataFrame(), warnings
    association_tables = []
    problem_tables = []
    for prefix in ONSET_PREFIXES:
        feature_path = features_dir / f"features_prefix_{prefix}.parquet"
        if not feature_path.exists():
            warnings.append(f"Missing onset feature table for prefix {prefix}.")
            continue
        frame = pd.read_parquet(feature_path)
        all_problem_count = int(frame["problem_id"].nunique())
        frame = frame[frame["full_trajectory_token_count"] >= prefix].copy()
        active_seed_counts = frame.groupby("problem_id")["seed"].nunique()
        complete_problem_ids = active_seed_counts[active_seed_counts == 4].index
        frame = frame[frame["problem_id"].isin(complete_problem_ids)].copy()
        if len(complete_problem_ids) < all_problem_count:
            warnings.append(
                f"Onset prefix {prefix} uses {len(complete_problem_ids)}/{all_problem_count} "
                "problems with all four seeds still active."
            )
        if not len(complete_problem_ids):
            continue
        available = [feature for feature in ONSET_UNCERTAINTY_FEATURES if feature in frame]
        if not available:
            warnings.append(f"Prefix {prefix} has no onset uncertainty features.")
            continue
        problem_table = problem_seed_instability(frame, available)
        problem_table["prefix_length"] = prefix
        problem_table["complete_problem_coverage"] = (
            len(complete_problem_ids) / all_problem_count if all_problem_count else 0.0
        )
        problem_tables.append(problem_table)
        aggregated_features = [f"mean__{feature}" for feature in available]
        failure_associations = bootstrap_residualized_associations(
            problem_table,
            "failure_rate",
            aggregated_features,
            controls=("level", "category"),
            repetitions=repetitions,
            seed=20260728 + prefix,
        )
        difficulty_associations = bootstrap_residualized_associations(
            problem_table,
            "level",
            aggregated_features,
            controls=("category", "failure_rate"),
            repetitions=repetitions,
            seed=20260728 + prefix + 1,
        )
        failure_associations["prefix_length"] = prefix
        difficulty_associations["prefix_length"] = prefix
        association_tables.extend((failure_associations, difficulty_associations))
    associations = (
        pd.concat(association_tables, ignore_index=True) if association_tables else pd.DataFrame()
    )
    problems = pd.concat(problem_tables, ignore_index=True) if problem_tables else pd.DataFrame()
    associations.to_parquet(
        output_dir / "reasoning_onset_uncertainty_associations.parquet",
        index=False,
    )
    problems.to_parquet(output_dir / "reasoning_onset_problem_features.parquet", index=False)
    _plot_onset_uncertainty(
        associations,
        output_dir / "reasoning_onset_uncertainty_associations.png",
    )
    return associations, warnings


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    _style()
    frame = pd.read_parquet(args.features)
    design = validate_difficulty_design(frame, expected_models=[EXPECTED_MODEL])
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
    summary.to_parquet(output_dir / "difficulty_metric_summary.parquet", index=False)
    consistency.to_parquet(output_dir / "seed_consistency.parquet", index=False)
    binned.to_parquet(output_dir / "token_dynamics_binned.parquet", index=False)
    token_summary.to_parquet(output_dir / "token_dynamics_summary.parquet", index=False)
    spikes.to_parquet(output_dir / "entropy_spike_events.parquet", index=False)
    write_json_atomic(output_dir / "difficulty_design_validation.json", design)
    write_json_atomic(output_dir / "level_trends.json", trends)
    _plot_metric_grid(summary, output_dir / "difficulty_outcomes.png")
    _plot_token_curves(token_summary, output_dir / "token_dynamics.png")
    _plot_correctness_entropy(token_summary, output_dir / "correctness_entropy_dynamics.png")
    _plot_heatmap(summary, output_dir / "difficulty_metric_heatmap.png")
    _plot_seed_consistency(consistency, output_dir / "seed_consistency.png")
    _plot_spikes(spikes, output_dir / "entropy_spike_timing.png")
    problem_instability, instability_associations = _geometry_spectral_analysis(
        frame,
        output_dir,
        repetitions=args.bootstrap_repetitions,
    )
    failure_summary, failure_improvements, failure_warnings = _failure_prediction_analysis(
        args.features_dir,
        output_dir,
        repetitions=args.bootstrap_repetitions,
    )
    onset_associations, onset_warnings = _reasoning_onset_analysis(
        args.features_dir,
        output_dir,
        repetitions=args.bootstrap_repetitions,
    )
    entropy_trend = trends["models"][EXPECTED_MODEL]["normalized_entropy_mean"]
    primary = (
        failure_improvements[
            (
                (
                    (failure_improvements.get("prefix_length") < 64)
                    & (failure_improvements.get("feature_set") == "early_full_without_spectral")
                )
                | (
                    (failure_improvements.get("prefix_length") >= 64)
                    & (failure_improvements.get("feature_set") == "early_full")
                )
            )
            & (failure_improvements.get("metric") == "auprc")
        ]
        if not failure_improvements.empty
        else pd.DataFrame()
    )
    positive_prefixes = int((primary["ci_low"] > 0).sum()) if not primary.empty else 0
    finish_reason_counts = (
        {str(key): int(value) for key, value in frame["finish_reason"].value_counts().items()}
        if "finish_reason" in frame
        else {}
    )
    warnings = [
        "Phase 2 is exploratory; hypotheses and feature blocks must be frozen before Phase 3.",
        "Token predictive entropy is not identified as epistemic uncertainty.",
        "Four seeds are repeated observations; problem_id is the independent bootstrap unit.",
        "Fixed-prefix estimates are conditional on trajectories that remain active at each prefix.",
        *failure_warnings,
        *onset_warnings,
    ]
    if set(finish_reason_counts) == {"max_new_tokens"}:
        warnings.append(
            "All trajectories report max_new_tokens; audit finish-reason instrumentation before interpretation."
        )
    association_candidates = instability_associations[
        ((instability_associations["ci_low"] > 0) & (instability_associations["ci_high"] > 0))
        | ((instability_associations["ci_low"] < 0) & (instability_associations["ci_high"] < 0))
    ]
    onset_candidates = (
        onset_associations[
            ((onset_associations["ci_low"] > 0) & (onset_associations["ci_high"] > 0))
            | ((onset_associations["ci_low"] < 0) & (onset_associations["ci_high"] < 0))
        ]
        if not onset_associations.empty
        else pd.DataFrame(
            columns=[
                "target",
                "feature",
                "prefix_length",
                "association",
                "ci_low",
                "ci_high",
            ]
        )
    )
    next_decision = "freeze_hypotheses" if not failure_summary.empty else "insufficient_failures"
    freeze_payload = {
        "status": (
            "frozen_at_phase_02_completion"
            if next_decision == "freeze_hypotheses"
            else "candidate_only_not_frozen"
        ),
        "primary_question": (
            "Can partial Gemma reasoning trajectories predict terminal failure "
            "beyond difficulty, category, problem surface features, and observed tokens?"
        ),
        "positive_class": "terminal_failure",
        "independent_unit": "problem_id",
        "fixed_prefixes": list(PREFIX_LENGTHS),
        "reasoning_onset_prefixes": list(ONSET_PREFIXES),
        "primary_estimand": (
            "failure AUPRC: confidence+geometry minus baseline at 16/32 tokens; "
            "early_full minus baseline from 64 tokens onward"
        ),
        "feature_blocks": list(PREFIX_FEATURE_SETS),
        "phase_03_confirmation": (
            "Apply frozen feature blocks without reselection to the previously unseen "
            "Qwen and Ministral model families on the matched MATH panel."
        ),
        "dataset_confirmation": ("Test the frozen model on held-out GSM8K only after Phase 3."),
        "positive_primary_prefixes": [
            int(value) for value in primary.loc[primary["ci_low"] > 0, "prefix_length"].tolist()
        ],
        "seed_instability_association_candidates": association_candidates[
            ["feature", "association", "ci_low", "ci_high"]
        ].to_dict(orient="records"),
        "reasoning_onset_association_candidates": onset_candidates[
            ["target", "feature", "prefix_length", "association", "ci_low", "ci_high"]
        ].to_dict(orient="records"),
        "interpretation_boundaries": warnings,
    }
    if next_decision == "freeze_hypotheses":
        write_json_atomic(output_dir / "hypothesis_freeze.json", freeze_payload)
    else:
        write_json_atomic(output_dir / "hypothesis_freeze_candidates.json", freeze_payload)
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "hypothesis_candidate" if positive_prefixes else "inconclusive",
            "next_decision": next_decision,
            "summary": (
                "Exploratory Gemma partial-trajectory failure prediction and "
                "reasoning-onset, geometry/spectral seed-instability analysis completed."
            ),
            "metrics": {
                "trajectories": len(frame),
                "problems": int(frame["problem_id"].nunique()),
                "levels": 5,
                "seeds_per_problem": 4,
                "entropy_slope_per_level": entropy_trend["slope_per_level"],
                "entropy_spike_events": len(spikes),
                "prefix_prediction_rows": len(failure_summary),
                "positive_full_vs_baseline_prefixes": positive_prefixes,
                "seed_instability_problems": len(problem_instability),
                "geometry_spectral_instability_associations": len(instability_associations),
                "reasoning_onset_associations": len(onset_associations),
                "reasoning_onset_ci_excluding_zero": len(onset_candidates),
                "finish_reason_counts": finish_reason_counts,
            },
            "warnings": warnings,
        },
    )
    print(output_dir / "phase_summary.json")


if __name__ == "__main__":
    main()
