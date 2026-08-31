#!/usr/bin/env python
"""Build publication-oriented Phase 1 figures from committed local trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasonbench.features.extractor import trajectory_directories
from reasonbench.storage import (
    ensure_directory,
    read_json,
    sha256_file,
    write_json_atomic,
)

MODE_ORDER = ("non_reasoning", "reasoning")
DATASET_ORDER = ("gsm8k", "math")
MODE_COLORS = {"non_reasoning": "#4472A8", "reasoning": "#D97745"}
DATASET_COLORS = {"gsm8k": "#2F6B9A", "math": "#C96C3B"}

PAIRED_METRICS = (
    ("correct", "Correctness", "proportion"),
    ("trajectory_token_count", "Analysis-window tokens", "tokens"),
    ("elapsed_seconds", "Generation time", "seconds"),
    ("normalized_entropy_mean", "Normalized entropy", "scalar"),
    ("surprisal_mean", "Surprisal", "scalar"),
    ("top1_top2_logit_margin_mean", "Top-1/Top-2 logit margin", "scalar"),
    ("geometry_mean_relative_velocity", "Relative hidden velocity", "scalar"),
    ("geometry_trajectory_efficiency", "Hidden trajectory efficiency", "scalar"),
)

TOKEN_SIGNALS = (
    ("normalized_entropy", "Normalized entropy"),
    ("surprisal", "Surprisal"),
    ("top1_top2_logit_margin", "Top-1/Top-2 logit margin"),
    ("relative_l2_step", "Relative hidden-state step"),
)

CORRELATION_FEATURES = (
    "trajectory_token_count",
    "elapsed_seconds",
    "normalized_entropy_mean",
    "normalized_entropy_std",
    "normalized_entropy_slope",
    "normalized_entropy_autocorr_lag1",
    "surprisal_mean",
    "surprisal_std",
    "surprisal_slope",
    "top1_top2_logit_margin_mean",
    "top1_top2_logit_margin_std",
    "top1_top2_probability_margin_mean",
    "hidden_norm_mean",
    "relative_l2_step_mean",
    "relative_l2_step_std",
    "cosine_drift_mean",
    "geometry_normalized_path_length",
    "geometry_mean_relative_velocity",
    "geometry_trajectory_efficiency",
    "geometry_turning_angle_mean",
    "spectral_normalized_entropy_low_energy_ratio",
    "spectral_normalized_entropy_entropy",
    "spectral_surprisal_entropy",
    "spectral_relative_l2_step_high_energy_ratio",
)

CORRECTNESS_FEATURES = (
    ("normalized_entropy_mean", "Entropy"),
    ("surprisal_mean", "Surprisal"),
    ("top1_top2_logit_margin_mean", "Logit margin"),
    ("geometry_mean_relative_velocity", "Hidden velocity"),
    ("geometry_mean_cosine_drift", "Cosine drift"),
    ("geometry_trajectory_efficiency", "Trajectory efficiency"),
    ("spectral_normalized_entropy_entropy", "Entropy spectral entropy"),
    ("spectral_relative_l2_step_high_energy_ratio", "Hidden-step high-frequency ratio"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(figure: plt.Figure, output: Path) -> None:
    ensure_directory(output.parent)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan, np.nan, np.nan
    sampled = clean[
        rng.integers(0, len(clean), size=(repetitions, len(clean)))
    ].mean(axis=1)
    return (
        float(clean.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def _problem_interval(
    frame: pd.DataFrame,
    value: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = (
        frame.assign(_value=pd.to_numeric(frame[value], errors="coerce"))
        .groupby("problem_id")["_value"]
        .mean()
        .to_numpy(dtype=float)
    )
    return _bootstrap_interval(values, repetitions=repetitions, rng=rng)


def _paired_effect(
    frame: pd.DataFrame,
    metric: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    keys = ["problem_id", "seed"]
    left = frame[frame["model_mode"] == MODE_ORDER[0]][keys + [metric]].rename(
        columns={metric: "left"}
    )
    right = frame[frame["model_mode"] == MODE_ORDER[1]][keys + [metric]].rename(
        columns={metric: "right"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    right_values = pd.to_numeric(paired["right"], errors="coerce").astype(float)
    left_values = pd.to_numeric(paired["left"], errors="coerce").astype(float)
    paired["difference"] = right_values - left_values
    problem_values = paired.groupby("problem_id")["difference"].mean().to_numpy(dtype=float)
    return _bootstrap_interval(problem_values, repetitions=repetitions, rng=rng)


def _paired_effect_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for metric, label, unit in PAIRED_METRICS:
        for dataset in DATASET_ORDER:
            point, low, high = _paired_effect(
                frame[frame["dataset"] == dataset],
                metric,
                repetitions=repetitions,
                rng=rng,
            )
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "unit": unit,
                    "dataset": dataset,
                    "difference": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    effects = pd.DataFrame(rows)
    effects.to_parquet(figure_data_dir / "paired_effects.parquet", index=False)
    figure, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    for axis, (metric, label, unit) in zip(axes.flat, PAIRED_METRICS, strict=True):
        scoped = effects[effects["metric"] == metric]
        for position, dataset in enumerate(DATASET_ORDER):
            row = scoped[scoped["dataset"] == dataset].iloc[0]
            axis.errorbar(
                row["difference"],
                position,
                xerr=[[row["difference"] - row["ci_low"]], [row["ci_high"] - row["difference"]]],
                fmt="o",
                capsize=4,
                markersize=8,
                color=DATASET_COLORS[dataset],
            )
            axis.annotate(
                f"{row['difference']:+.3g}",
                (row["difference"], position),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10,
            )
        axis.axvline(0, color="0.25", linewidth=1)
        axis.set_yticks(range(len(DATASET_ORDER)), [name.upper() for name in DATASET_ORDER])
        axis.set_title(label)
        axis.set_xlabel(f"Reasoning − non-reasoning ({unit})")
    figure.suptitle("Dataset-specific paired effects of reasoning mode", y=1.03)
    _save(figure, output_dir / "paired_effect_forest.png")
    return {
        f"{row.dataset}|{row.metric}": {
            "difference": float(row.difference),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
        }
        for row in effects.itertuples()
    }


def _paired_problem_accuracy_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
) -> None:
    problem = (
        frame.groupby(["dataset", "problem_id", "model_mode"], as_index=False)["correct"]
        .mean()
        .pivot(index=["dataset", "problem_id"], columns="model_mode", values="correct")
        .reset_index()
    )
    problem.to_parquet(figure_data_dir / "paired_problem_accuracy.parquet", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, dataset in zip(axes, DATASET_ORDER, strict=True):
        scoped = problem[problem["dataset"] == dataset]
        counts = (
            scoped.groupby(list(MODE_ORDER), as_index=False)
            .size()
            .rename(columns={"size": "problems"})
        )
        axis.scatter(
            counts[MODE_ORDER[0]],
            counts[MODE_ORDER[1]],
            s=50 + counts["problems"] * 24,
            color=DATASET_COLORS[dataset],
            alpha=0.78,
            edgecolor="white",
            linewidth=1,
        )
        for row in counts.itertuples(index=False):
            axis.annotate(
                str(row.problems),
                (getattr(row, MODE_ORDER[0]), getattr(row, MODE_ORDER[1])),
                ha="center",
                va="center",
                fontsize=10,
            )
        axis.plot([0, 1], [0, 1], color="0.35", linewidth=1)
        improved = int((scoped[MODE_ORDER[1]] > scoped[MODE_ORDER[0]]).sum())
        unchanged = int((scoped[MODE_ORDER[1]] == scoped[MODE_ORDER[0]]).sum())
        worsened = int((scoped[MODE_ORDER[1]] < scoped[MODE_ORDER[0]]).sum())
        axis.set_title(
            f"{dataset.upper()}  improved {improved} · unchanged {unchanged} · worsened {worsened}"
        )
        axis.set_xlim(-0.07, 1.07)
        axis.set_ylim(-0.07, 1.07)
        axis.set_xlabel("Non-reasoning success rate across four seeds")
    axes[0].set_ylabel("Reasoning success rate across four seeds")
    figure.suptitle("Problem-level paired accuracy")
    _save(figure, output_dir / "paired_problem_accuracy.png")


def _math_level_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> None:
    math = frame[frame["dataset"] == "math"].copy()
    rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        for level in range(1, 6):
            cell = math[(math["model_mode"] == mode) & (math["level"] == level)]
            for metric in ("correct", "trajectory_token_count", "elapsed_seconds"):
                point, low, high = _problem_interval(
                    cell,
                    metric,
                    repetitions=repetitions,
                    rng=rng,
                )
                rows.append(
                    {
                        "model_mode": mode,
                        "level": level,
                        "metric": metric,
                        "mean": point,
                        "ci_low": low,
                        "ci_high": high,
                        "problems": int(cell["problem_id"].nunique()),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_parquet(figure_data_dir / "math_level_response.parquet", index=False)
    specifications = (
        ("correct", "Correctness", (0, 1.02)),
        ("trajectory_token_count", "Analysis-window tokens", None),
        ("elapsed_seconds", "Generation time (seconds)", None),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, (metric, label, limits) in zip(axes, specifications, strict=True):
        for mode in MODE_ORDER:
            scoped = summary[(summary["metric"] == metric) & (summary["model_mode"] == mode)]
            axis.errorbar(
                scoped["level"],
                scoped["mean"],
                yerr=[scoped["mean"] - scoped["ci_low"], scoped["ci_high"] - scoped["mean"]],
                marker="o",
                capsize=4,
                linewidth=2,
                color=MODE_COLORS[mode],
                label=mode.replace("_", " "),
            )
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("MATH difficulty level")
        axis.set_ylabel(label)
        if limits:
            axis.set_ylim(*limits)
    axes[0].legend(loc="lower left")
    figure.suptitle("MATH difficulty response by reasoning mode")
    _save(figure, output_dir / "math_level_response.png")


def _load_binned_token_dynamics(
    run_directories: list[Path],
    *,
    bins: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for directory in trajectory_directories(run_directories):
        metadata = read_json(directory / "metadata.json")
        tokens = pd.read_parquet(
            directory / "token_metrics.parquet",
            columns=["token_index", "segment", *[name for name, _ in TOKEN_SIGNALS]],
        )
        segment = "thinking" if (tokens["segment"] == "thinking").any() else "solution"
        tokens = tokens[tokens["segment"] == segment].sort_values("token_index")
        if tokens.empty:
            continue
        tokens = tokens.assign(
            time_bin=np.minimum(np.arange(len(tokens)) * bins // len(tokens), bins - 1)
        )
        binned = tokens.groupby("time_bin", as_index=False)[
            [name for name, _ in TOKEN_SIGNALS]
        ].mean()
        for key, value in {
            "run_id": metadata["run_id"],
            "dataset": metadata["dataset"],
            "model_mode": metadata["model_mode"],
            "problem_id": metadata["problem_id"],
            "seed": metadata["seed"],
            "correct": bool(metadata["verification"]["correct"]),
            "analysis_segment": segment,
        }.items():
            binned[key] = value
        rows.append(binned)
    if not rows:
        raise ValueError("No token trajectories were available")
    return pd.concat(rows, ignore_index=True)


def _token_dynamics_figure(
    binned: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
    *,
    bins: int,
    repetitions: int,
    rng: np.random.Generator,
) -> None:
    rows: list[dict[str, Any]] = []
    for (dataset, mode, time_bin), cell in binned.groupby(
        ["dataset", "model_mode", "time_bin"], sort=True
    ):
        for signal, _label in TOKEN_SIGNALS:
            values = cell.groupby("problem_id")[signal].mean().to_numpy(dtype=float)
            point, low, high = _bootstrap_interval(
                values,
                repetitions=repetitions,
                rng=rng,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model_mode": mode,
                    "time_bin": int(time_bin),
                    "progress": (int(time_bin) + 0.5) / bins,
                    "signal": signal,
                    "mean": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    summary = pd.DataFrame(rows)
    binned.to_parquet(figure_data_dir / "token_dynamics_binned.parquet", index=False)
    summary.to_parquet(figure_data_dir / "token_dynamics_summary.parquet", index=False)
    figure, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True, constrained_layout=True)
    for row_index, (signal, label) in enumerate(TOKEN_SIGNALS):
        for column_index, dataset in enumerate(DATASET_ORDER):
            axis = axes[row_index, column_index]
            for mode in MODE_ORDER:
                scoped = summary[
                    (summary["signal"] == signal)
                    & (summary["dataset"] == dataset)
                    & (summary["model_mode"] == mode)
                ].sort_values("time_bin")
                axis.plot(
                    scoped["progress"],
                    scoped["mean"],
                    color=MODE_COLORS[mode],
                    linewidth=2,
                    label=mode.replace("_", " "),
                )
                axis.fill_between(
                    scoped["progress"],
                    scoped["ci_low"],
                    scoped["ci_high"],
                    color=MODE_COLORS[mode],
                    alpha=0.16,
                )
            axis.set_title(f"{dataset.upper()} · {label}")
            axis.set_ylabel("Problem-cluster mean")
            if row_index == len(TOKEN_SIGNALS) - 1:
                axis.set_xlabel("Normalized analysis-window progress")
    axes[0, 0].legend(loc="best")
    figure.suptitle(
        "Token-level dynamics across the reasoning/solution analysis window",
        y=1.035,
    )
    _save(figure, output_dir / "token_dynamics_by_mode_dataset.png")


def _seed_transition_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
) -> None:
    success = (
        frame.groupby(["dataset", "problem_id", "model_mode"], as_index=False)["correct"]
        .sum()
        .pivot(index=["dataset", "problem_id"], columns="model_mode", values="correct")
        .reset_index()
    )
    success.to_parquet(figure_data_dir / "seed_success_transitions.parquet", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, dataset in zip(axes, DATASET_ORDER, strict=True):
        scoped = success[success["dataset"] == dataset]
        matrix = pd.crosstab(scoped[MODE_ORDER[0]], scoped[MODE_ORDER[1]]).reindex(
            index=range(5), columns=range(5), fill_value=0
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=axis,
        )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Reasoning: correct seeds out of four")
        axis.set_ylabel("Non-reasoning: correct seeds out of four")
    figure.suptitle("Problem-level seed-consistency transitions")
    _save(figure, output_dir / "seed_consistency_transition.png")


def _compute_frontier_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> None:
    rows: list[dict[str, Any]] = []
    for (dataset, mode), cell in frame.groupby(["dataset", "model_mode"], sort=True):
        row: dict[str, Any] = {"dataset": dataset, "model_mode": mode}
        for metric in ("correct", "trajectory_token_count", "elapsed_seconds"):
            point, low, high = _problem_interval(
                cell,
                metric,
                repetitions=repetitions,
                rng=rng,
            )
            row[f"{metric}_mean"] = point
            row[f"{metric}_low"] = low
            row[f"{metric}_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_parquet(figure_data_dir / "accuracy_compute_frontier.parquet", index=False)
    figure, axis = plt.subplots(figsize=(10, 7))
    markers = {"non_reasoning": "o", "reasoning": "s"}
    for row in summary.itertuples():
        axis.errorbar(
            row.trajectory_token_count_mean,
            row.correct_mean,
            xerr=[
                [row.trajectory_token_count_mean - row.trajectory_token_count_low],
                [row.trajectory_token_count_high - row.trajectory_token_count_mean],
            ],
            yerr=[
                [row.correct_mean - row.correct_low],
                [row.correct_high - row.correct_mean],
            ],
            marker=markers[row.model_mode],
            markersize=10,
            capsize=4,
            linewidth=1.5,
            color=DATASET_COLORS[row.dataset],
        )
        axis.annotate(
            f"{row.dataset.upper()} · {row.model_mode.replace('_', ' ')}\n{row.elapsed_seconds_mean:.0f}s",
            (row.trajectory_token_count_mean, row.correct_mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )
    axis.set_xlabel("Mean analysis-window tokens")
    axis.set_ylabel("Correctness")
    axis.set_ylim(0, 1.03)
    axis.set_title("Accuracy–compute frontier (labels show mean wall time)")
    _save(figure, output_dir / "accuracy_compute_frontier.png")


def _correlation_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
) -> None:
    available = [feature for feature in CORRELATION_FEATURES if feature in frame]
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    strata = frame[["dataset", "model_mode"]].astype(str).agg("|".join, axis=1)
    standardized = numeric.copy()
    for feature in available:
        standardized[feature] = numeric[feature].groupby(strata).transform(
            lambda values: (values - values.mean()) / max(float(values.std()), 1e-12)
        )
    usable = [
        feature
        for feature in available
        if standardized[feature].notna().sum() > 1
        and float(standardized[feature].std(skipna=True)) > 0
    ]
    correlation = standardized[usable].corr(method="spearman").fillna(0.0).copy()
    for index in range(len(correlation)):
        correlation.iat[index, index] = 1.0
    correlation.to_parquet(figure_data_dir / "feature_spearman_correlation.parquet")
    grid = sns.clustermap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(15, 15),
        dendrogram_ratio=0.12,
        cbar_pos=(0.12, 0.96, 0.18, 0.015),
        cbar_kws={"label": "Spearman correlation", "orientation": "horizontal"},
    )
    grid.ax_heatmap.set_xticklabels(
        [label.get_text().replace("_", " ") for label in grid.ax_heatmap.get_xticklabels()],
        rotation=55,
        ha="right",
        fontsize=8,
    )
    grid.ax_heatmap.set_yticklabels(
        [label.get_text().replace("_", " ") for label in grid.ax_heatmap.get_yticklabels()],
        fontsize=8,
    )
    grid.figure.suptitle("Within-dataset/mode feature redundancy", y=1.02)
    grid.figure.savefig(output_dir / "feature_correlation_cluster.png", bbox_inches="tight")
    grid.figure.savefig(output_dir / "feature_correlation_cluster.pdf", bbox_inches="tight")
    plt.close(grid.figure)


def _correctness_effect(
    frame: pd.DataFrame,
    feature: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    scale = float(values.std())
    if not np.isfinite(scale) or scale <= 0:
        return np.nan, np.nan, np.nan
    working = pd.DataFrame(
        {
            "problem_id": frame["problem_id"],
            "correct": frame["correct"].astype(bool),
            "value": (values - values.mean()) / scale,
        }
    ).dropna()
    if working["correct"].nunique() < 2:
        return np.nan, np.nan, np.nan
    point = float(
        working.loc[working["correct"], "value"].mean()
        - working.loc[~working["correct"], "value"].mean()
    )
    problem_ids = working["problem_id"].drop_duplicates().tolist()
    aggregates = []
    for problem_id in problem_ids:
        problem = working[working["problem_id"] == problem_id]
        aggregates.append(
            (
                float(problem.loc[~problem["correct"], "value"].sum()),
                int((~problem["correct"]).sum()),
                float(problem.loc[problem["correct"], "value"].sum()),
                int(problem["correct"].sum()),
            )
        )
    aggregate = np.asarray(aggregates, dtype=float)
    sampled = rng.integers(
        0,
        len(aggregate),
        size=(repetitions, len(aggregate)),
    )
    totals = aggregate[sampled].sum(axis=1)
    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    draws = totals[valid, 2] / totals[valid, 3] - totals[valid, 0] / totals[valid, 1]
    if not len(draws):
        return point, np.nan, np.nan
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _stratified_correctness_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    figure_data_dir: Path,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for mode in MODE_ORDER:
            cell = frame[(frame["dataset"] == dataset) & (frame["model_mode"] == mode)]
            for feature, label in CORRECTNESS_FEATURES:
                point, low, high = _correctness_effect(
                    cell,
                    feature,
                    repetitions=repetitions,
                    rng=rng,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model_mode": mode,
                        "feature": feature,
                        "label": label,
                        "difference": point,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    effects = pd.DataFrame(rows)
    effects.to_parquet(figure_data_dir / "stratified_correctness_effects.parquet", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, constrained_layout=True)
    labels = [label for _feature, label in CORRECTNESS_FEATURES]
    for axis, dataset, mode in zip(
        axes.flat,
        ["gsm8k", "gsm8k", "math", "math"],
        ["non_reasoning", "reasoning", "non_reasoning", "reasoning"],
        strict=True,
    ):
        scoped = effects[
            (effects["dataset"] == dataset) & (effects["model_mode"] == mode)
        ].set_index("label").reindex(labels).reset_index()
        positions = np.arange(len(scoped))
        axis.errorbar(
            scoped["difference"],
            positions,
            xerr=[
                scoped["difference"] - scoped["ci_low"],
                scoped["ci_high"] - scoped["difference"],
            ],
            fmt="o",
            capsize=4,
            color=MODE_COLORS[mode],
        )
        axis.axvline(0, color="0.3", linewidth=1)
        axis.set_yticks(positions, scoped["label"])
        axis.set_title(f"{dataset.upper()} · {mode.replace('_', ' ')}")
        axis.set_xlabel("Correct − incorrect standardized mean")
    figure.suptitle("Correctness-associated features after dataset/mode stratification")
    _save(figure, output_dir / "correctness_features_stratified.png")


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    if args.time_bins < 5:
        raise ValueError("time-bins must be at least 5")
    output_dir = ensure_directory(args.output_dir)
    figure_data_dir = ensure_directory(output_dir / "figure_data")
    _style()
    frame = pd.read_parquet(args.features)
    expected = {
        "rows": 800,
        "problems": 100,
        "datasets": set(DATASET_ORDER),
        "modes": set(MODE_ORDER),
    }
    if len(frame) != expected["rows"] or frame["problem_id"].nunique() != expected["problems"]:
        raise ValueError("Phase 1 extended analysis requires the complete 800-trajectory panel")
    if set(frame["dataset"]) != expected["datasets"] or set(frame["model_mode"]) != expected["modes"]:
        raise ValueError("Unexpected Phase 1 datasets or model modes")
    rng = np.random.default_rng(args.seed)
    paired_effects = _paired_effect_figure(
        frame,
        output_dir,
        figure_data_dir,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    _paired_problem_accuracy_figure(frame, output_dir, figure_data_dir)
    _math_level_figure(
        frame,
        output_dir,
        figure_data_dir,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    binned = _load_binned_token_dynamics(args.run_dir, bins=args.time_bins)
    _token_dynamics_figure(
        binned,
        output_dir,
        figure_data_dir,
        bins=args.time_bins,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    _seed_transition_figure(frame, output_dir, figure_data_dir)
    _compute_frontier_figure(
        frame,
        output_dir,
        figure_data_dir,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    _correlation_figure(frame, output_dir, figure_data_dir)
    _stratified_correctness_figure(
        frame,
        output_dir,
        figure_data_dir,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    finish_counts = frame["finish_reason"].value_counts(dropna=False).to_dict()
    write_json_atomic(
        output_dir / "extended_analysis_summary.json",
        {
            "features_path": str(args.features),
            "features_sha256": sha256_file(args.features),
            "trajectories": len(frame),
            "problems": int(frame["problem_id"].nunique()),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "time_bins": args.time_bins,
            "paired_effects": paired_effects,
            "finish_reason_counts": finish_counts,
            "figures": sorted(path.name for path in output_dir.glob("*.png")),
            "interpretation_boundaries": [
                "Dataset-specific effects are primary because the mode effect is heterogeneous.",
                "Correctness-associated feature effects are descriptive, not causal.",
                "Reasoning trajectories use the thinking segment; non-reasoning trajectories use the solution segment.",
                "Token predictive entropy is not identified as epistemic uncertainty.",
                "The all-max_new_tokens finish-reason pattern requires instrumentation audit.",
            ],
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
