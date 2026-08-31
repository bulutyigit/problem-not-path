#!/usr/bin/env python
"""Analyze Phase 4 prefix dynamics, length associations, and correctness separation."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ConstantInputWarning, spearmanr

from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

PREFIXES = (16, 32, 64, 128, 256, 512)
FEATURES = (
    "normalized_entropy_mean",
    "normalized_entropy_std",
    "normalized_entropy_slope",
    "normalized_entropy_max_rise",
    "surprisal_mean",
    "top1_top2_probability_margin_mean",
    "geometry_mean_relative_velocity",
    "geometry_mean_cosine_drift",
    "spectral_normalized_entropy_entropy",
    "spectral_normalized_entropy_high_energy_ratio",
    "spectral_surprisal_entropy",
)
FEATURE_LABELS = {
    "normalized_entropy_mean": "Entropy mean",
    "normalized_entropy_std": "Entropy volatility",
    "normalized_entropy_slope": "Entropy slope",
    "normalized_entropy_max_rise": "Largest entropy rise",
    "surprisal_mean": "Surprisal mean",
    "top1_top2_probability_margin_mean": "Top-1/2 margin",
    "geometry_mean_relative_velocity": "Hidden velocity",
    "geometry_mean_cosine_drift": "Cosine drift",
    "spectral_normalized_entropy_entropy": "Entropy spectral entropy",
    "spectral_normalized_entropy_high_energy_ratio": "Entropy high-frequency energy",
    "spectral_surprisal_entropy": "Surprisal spectral entropy",
}
LIMIT_REASONS = {"max_new_tokens", "answer_reserve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


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
            [groups[index] for index in rng.integers(0, len(groups), size=len(groups))],
            ignore_index=True,
        )
        draws.append(float(sample[value_column].mean()))
    low, high = np.quantile(draws, (0.025, 0.975))
    return point, float(low), float(high)


def _level_adjusted_spearman(frame: pd.DataFrame, x: str, y: str) -> float:
    observed = frame[[x, y, "level", "category"]].dropna(subset=[x, y]).copy()
    if len(observed) < 3 or observed[x].nunique() < 2 or observed[y].nunique() < 2:
        return math.nan
    ranked_x = observed[x].rank().to_numpy(float)
    ranked_y = observed[y].rank().to_numpy(float)
    controls = [
        pd.factorize(observed[control].astype(str), sort=False)[0]
        for control in ("level", "category")
    ]

    def two_way_residual(values: np.ndarray) -> np.ndarray:
        """Remove additive level/category effects without dense linear algebra."""

        residual = values.astype(float, copy=True)
        # Alternating projections converge to the additive two-way fixed-effect
        # residual.  This also remains defined for rank-deficient bootstrap
        # resamples where a level/category combination is absent.
        for _ in range(8):
            for codes in controls:
                counts = np.bincount(codes)
                means = np.bincount(codes, weights=residual) / counts
                residual -= means[codes]
        return residual

    residual_x = two_way_residual(ranked_x)
    residual_y = two_way_residual(ranked_y)
    if np.std(residual_x) < 1e-12 or np.std(residual_y) < 1e-12:
        return math.nan
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _association_interval(
    frame: pd.DataFrame,
    feature: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    columns = ["problem_id", "level", "category", feature, "full_trajectory_token_count"]
    observed = frame[columns].dropna(subset=[feature, "full_trajectory_token_count"]).copy()
    observed["log_total_tokens"] = np.log1p(observed["full_trajectory_token_count"])
    if len(observed) < 5 or observed[feature].nunique() < 2:
        return {
            "trajectories": len(observed),
            "problems": int(observed["problem_id"].nunique()),
            "raw_rho": math.nan,
            "raw_ci_low": math.nan,
            "raw_ci_high": math.nan,
            "adjusted_rho": math.nan,
            "adjusted_ci_low": math.nan,
            "adjusted_ci_high": math.nan,
        }

    def estimates(sample: pd.DataFrame) -> tuple[float, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            raw = float(spearmanr(sample[feature], sample["log_total_tokens"]).statistic)
        return raw, _level_adjusted_spearman(sample, feature, "log_total_tokens")

    raw, adjusted = estimates(observed)
    groups = [group for _, group in observed.groupby("problem_id", sort=False)]
    rng = np.random.default_rng(seed)
    draws = [
        estimates(
            pd.concat(
                [groups[index] for index in rng.integers(0, len(groups), len(groups))],
                ignore_index=True,
            )
        )
        for _ in range(repetitions)
    ]
    raw_draws = np.asarray([value[0] for value in draws if np.isfinite(value[0])])
    adjusted_draws = np.asarray([value[1] for value in draws if np.isfinite(value[1])])
    return {
        "trajectories": len(observed),
        "problems": int(observed["problem_id"].nunique()),
        "raw_rho": raw,
        "raw_ci_low": float(np.quantile(raw_draws, 0.025)) if len(raw_draws) else math.nan,
        "raw_ci_high": float(np.quantile(raw_draws, 0.975)) if len(raw_draws) else math.nan,
        "adjusted_rho": adjusted,
        "adjusted_ci_low": (
            float(np.quantile(adjusted_draws, 0.025)) if len(adjusted_draws) else math.nan
        ),
        "adjusted_ci_high": (
            float(np.quantile(adjusted_draws, 0.975)) if len(adjusted_draws) else math.nan
        ),
    }


def _load_observations(features_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for prefix in PREFIXES:
        path = features_dir / f"features_prefix_{prefix}.parquet"
        frame = pd.read_parquet(path)
        active = frame[frame["full_trajectory_token_count"] >= prefix].copy()
        available = [feature for feature in FEATURES if feature in active]
        missing = sorted(set(FEATURES) - set(available))
        if missing:
            raise ValueError(f"Prefix {prefix} is missing Phase 4 dynamics features: {missing}")
        active["prefix_length"] = prefix
        active["terminal_outcome"] = np.where(active["correct"], "Correct", "Incorrect")
        active["capped_16k"] = active["finish_reason"].isin(LIMIT_REASONS)
        rows.append(active)
        manifests.append({"prefix": prefix, "path": str(path), "sha256": sha256_file(path)})
    return pd.concat(rows, ignore_index=True), manifests


def _standardized_long(observations: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "run_id",
        "model_key",
        "problem_id",
        "level",
        "category",
        "correct",
        "terminal_outcome",
        "prefix_length",
        "full_trajectory_token_count",
        "capped_16k",
    ]
    long = observations.melt(
        id_vars=identifiers,
        value_vars=list(FEATURES),
        var_name="feature",
        value_name="value",
    )
    long["feature_label"] = long["feature"].map(FEATURE_LABELS)
    fixed_ids = set(
        observations.loc[observations["full_trajectory_token_count"] >= 512, "run_id"]
    )
    cohorts = []
    for cohort, cohort_frame in (
        ("at_risk", long),
        ("fixed_512", long[long["run_id"].isin(fixed_ids)]),
    ):
        scoped = cohort_frame.copy()
        # Spectral summaries require at least 64 samples, while ordinary
        # token-level/geometry features are available at 16.  Anchor each
        # feature at its own first valid prefix so the 16/32 spectral NaNs do
        # not erase the otherwise valid 64–512-token trajectory.
        valid = scoped.dropna(subset=["value"])
        first_valid = valid.groupby(["model_key", "feature"], observed=True)[
            "prefix_length"
        ].transform("min")
        baseline = valid.loc[valid["prefix_length"].eq(first_valid)].groupby(
            ["model_key", "feature"], observed=True
        )["value"].agg(["mean", "std"])
        scoped = scoped.join(baseline, on=["model_key", "feature"])
        scoped["z_from_first_valid_prefix"] = (scoped["value"] - scoped["mean"]) / scoped[
            "std"
        ].replace(0, np.nan)
        scoped["cohort"] = cohort
        cohorts.append(scoped.drop(columns=["mean", "std"]))
    return pd.concat(cohorts, ignore_index=True)


def _summarize_dynamics(long: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    group_columns = [
        "cohort",
        "model_key",
        "terminal_outcome",
        "prefix_length",
        "feature",
        "feature_label",
    ]
    for offset, (keys, group) in enumerate(long.groupby(group_columns, observed=True, sort=True)):
        point, low, high = _mean_interval(
            group,
            "z_from_first_valid_prefix",
            repetitions,
            seed=20260805 + offset,
        )
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "mean_z": point,
                "ci_low": low,
                "ci_high": high,
                "trajectories": len(group),
                "problems": int(group["problem_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _correctness_separation(long: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    scoped = long[long["cohort"] == "fixed_512"]
    rows = []
    columns = ["model_key", "prefix_length", "feature", "feature_label"]
    for offset, (keys, group) in enumerate(scoped.groupby(columns, observed=True, sort=True)):
        correct = group[group["terminal_outcome"] == "Correct"].dropna(
            subset=["z_from_first_valid_prefix"]
        )
        incorrect = group[group["terminal_outcome"] == "Incorrect"].dropna(
            subset=["z_from_first_valid_prefix"]
        )
        if correct.empty or incorrect.empty:
            point = low = high = math.nan
        else:
            point = float(
                incorrect["z_from_first_valid_prefix"].mean()
                - correct["z_from_first_valid_prefix"].mean()
            )
            rng = np.random.default_rng(20260805 + offset)
            correct_groups = [item for _, item in correct.groupby("problem_id", sort=False)]
            incorrect_groups = [item for _, item in incorrect.groupby("problem_id", sort=False)]
            draws = []
            for _ in range(repetitions):
                correct_sample = pd.concat(
                    [
                        correct_groups[index]
                        for index in rng.integers(0, len(correct_groups), len(correct_groups))
                    ],
                    ignore_index=True,
                )
                incorrect_sample = pd.concat(
                    [
                        incorrect_groups[index]
                        for index in rng.integers(0, len(incorrect_groups), len(incorrect_groups))
                    ],
                    ignore_index=True,
                )
                draws.append(
                    float(
                        incorrect_sample["z_from_first_valid_prefix"].mean()
                        - correct_sample["z_from_first_valid_prefix"].mean()
                    )
                )
            low, high = (
                (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
                if draws
                else (math.nan, math.nan)
            )
        rows.append(
            {
                **dict(zip(columns, keys, strict=True)),
                "incorrect_minus_correct_z": point,
                "ci_low": low,
                "ci_high": high,
                "correct_trajectories": len(correct),
                "incorrect_trajectories": len(incorrect),
                "correct_problems": int(correct["problem_id"].nunique()),
                "incorrect_problems": int(incorrect["problem_id"].nunique()),
                "interval_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def _plot_movement(summary: pd.DataFrame, output_path: Path, cohort: str) -> None:
    scoped = summary[summary["cohort"] == cohort]
    models = sorted(scoped["model_key"].unique())
    outcomes = ("Correct", "Incorrect")
    figure, axes = plt.subplots(
        len(models), 2, figsize=(13, 3.8 * len(models)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    limit = float(np.nanmax(np.abs(scoped["mean_z"])))
    for row, model in enumerate(models):
        for column, outcome in enumerate(outcomes):
            data = scoped[
                (scoped["model_key"] == model)
                & (scoped["terminal_outcome"] == outcome)
            ].pivot(index="feature_label", columns="prefix_length", values="mean_z")
            sns.heatmap(
                data,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                ax=axes[row, column],
                cbar=row == 0 and column == 1,
            )
            axes[row, column].set_title(f"{model} · {outcome}")
            axes[row, column].set_xlabel("Observed tokens")
            axes[row, column].set_ylabel("")
    cohort_label = "fixed ≥512-token cohort" if cohort == "fixed_512" else "at-risk cohort"
    figure.suptitle(
        f"Feature movement from each feature's first valid prefix ({cohort_label})"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_evolution(
    summary: pd.DataFrame,
    output_path: Path,
    features: tuple[str, ...],
    title: str,
) -> None:
    scoped = summary[
        (summary["cohort"] == "fixed_512") & summary["feature"].isin(features)
    ]
    spectral_panel = all(feature.startswith("spectral_") for feature in features)
    models = sorted(scoped["model_key"].unique())
    figure, axes = plt.subplots(
        len(features),
        len(models),
        figsize=(4.8 * len(models), 3.2 * len(features)),
        sharex=True,
        squeeze=False,
    )
    colors = {"Correct": "#2a9d8f", "Incorrect": "#e76f51"}
    for row, feature in enumerate(features):
        for column, model in enumerate(models):
            axis = axes[row, column]
            data = scoped[(scoped["feature"] == feature) & (scoped["model_key"] == model)]
            for outcome in ("Correct", "Incorrect"):
                group = data[data["terminal_outcome"] == outcome].sort_values("prefix_length")
                if group.empty:
                    continue
                x = group["prefix_length"].to_numpy(float)
                y = group["mean_z"].to_numpy(float)
                axis.plot(x, y, marker="o", color=colors[outcome], label=outcome)
                axis.fill_between(
                    x,
                    group["ci_low"].to_numpy(float),
                    group["ci_high"].to_numpy(float),
                    color=colors[outcome],
                    alpha=0.16,
                )
            axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            axis.set_xscale("log", base=2)
            # A model/outcome cell can be empty in a small bootstrap smoke run.
            # Fixing the positive domain keeps Matplotlib from attempting to
            # derive logarithmic limits from its default (0, 1) interval.
            axis.set_xlim(min(PREFIXES) * 0.9, max(PREFIXES) * 1.1)
            axis.set_xticks(PREFIXES)
            axis.set_xticklabels(PREFIXES)
            axis.set_title(model if row == 0 else "")
            # A row shares one feature across models; repeating long spectral
            # labels on every column makes adjacent panels collide.  Horizontal
            # row labels avoid long vertical spectral labels overlapping rows.
            if column == 0:
                axis.set_ylabel(
                    FEATURE_LABELS[feature],
                    rotation=0 if spectral_panel else 90,
                    ha="right" if spectral_panel else "center",
                    va="center",
                    labelpad=16 if spectral_panel else 4,
                )
            else:
                axis.set_ylabel("")
            axis.grid(alpha=0.2)
    axes[-1, len(models) // 2].set_xlabel("Observed reasoning tokens")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=2,
        frameon=False,
    )
    reference = "64-token baseline" if spectral_panel else "16-token baseline"
    figure.suptitle(
        f"{title} (fixed ≥512-token cohort; {reference})", y=0.985
    )
    if spectral_panel:
        figure.subplots_adjust(
            left=0.36,
            right=0.98,
            bottom=0.08,
            top=0.79,
            hspace=0.20,
            wspace=0.24,
        )
    else:
        figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_correctness_gap(
    separation: pd.DataFrame, output_path: Path, spectral_only: bool
) -> None:
    scoped = separation
    if spectral_only:
        scoped = scoped[scoped["feature"].str.startswith("spectral_")]
    if scoped.empty:
        figure, axis = plt.subplots(figsize=(7, 3))
        axis.text(0.5, 0.5, "No finite correctness contrasts available", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    gap = scoped.pivot_table(
        index=["model_key", "feature_label"],
        columns="prefix_length",
        values="incorrect_minus_correct_z",
    )
    significance = scoped.pivot_table(
        index=["model_key", "feature_label"],
        columns="prefix_length",
        values="interval_excludes_zero",
    )
    models = sorted(scoped["model_key"].unique())
    figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    axes = np.atleast_1d(axes)
    finite_gaps = gap.to_numpy(float)
    finite_gaps = finite_gaps[np.isfinite(finite_gaps)]
    if not len(finite_gaps):
        figure, axis = plt.subplots(figsize=(7, 3))
        axis.text(0.5, 0.5, "No finite correctness contrasts available", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    limit = float(np.max(np.abs(finite_gaps)))
    for axis, model in zip(axes, models, strict=True):
        data = gap.loc[model]
        marked = significance.loc[model].reindex(
            index=data.index, columns=data.columns, fill_value=False
        )
        annotations = np.empty(data.shape, dtype=object)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = data.iloc[row, column]
                star = "*" if bool(marked.iloc[row, column]) else ""
                annotations[row, column] = "" if pd.isna(value) else f"{value:.2f}{star}"
        sns.heatmap(
            data,
            cmap="vlag",
            center=0,
            vmin=-limit,
            vmax=limit,
            ax=axis,
            cbar=axis is axes[-1],
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 11},
        )
        axis.set_title(model)
        axis.set_xlabel("Observed tokens")
        axis.set_ylabel("")
    title = (
        "Spectral separation: incorrect − correct (* pointwise, uncorrected clustered-bootstrap CI excludes 0)"
        if spectral_only
        else "Feature separation: incorrect − correct (* pointwise, uncorrected clustered-bootstrap CI excludes 0)"
    )
    figure.suptitle(title, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _length_associations(observations: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for model_offset, (model, model_frame) in enumerate(
        observations.groupby("model_key", sort=True)
    ):
        for prefix in PREFIXES:
            prefix_frame = model_frame[model_frame["prefix_length"] == prefix]
            for feature_offset, feature in enumerate(FEATURES):
                rows.append(
                    {
                        "model_key": model,
                        "prefix_length": prefix,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        **_association_interval(
                            prefix_frame,
                            feature,
                            repetitions,
                            seed=20260805
                            + model_offset * 1000
                            + prefix
                            + feature_offset,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _plot_length_associations(associations: pd.DataFrame, output_path: Path) -> None:
    models = sorted(associations["model_key"].unique())
    figure, axes = plt.subplots(2, len(models), figsize=(5.3 * len(models), 11), sharey=True)
    axes = np.asarray(axes).reshape(2, len(models))
    specifications = (("raw_rho", "Unadjusted"), ("adjusted_rho", "Level/category adjusted"))
    limit = float(
        np.nanmax(
            np.abs(associations[["raw_rho", "adjusted_rho"]].to_numpy(float))
        )
    )
    for row, (metric, _label) in enumerate(specifications):
        for column, model in enumerate(models):
            data = associations[associations["model_key"] == model].pivot(
                index="feature_label", columns="prefix_length", values=metric
            )
            sns.heatmap(
                data,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                ax=axes[row, column],
                cbar=column == len(models) - 1,
            )
            axes[row, column].set_title(model if row == 0 else "", pad=14)
            axes[row, column].set_xlabel("Observed tokens")
            axes[row, column].set_ylabel("")
    figure.suptitle(
        "Early feature association with eventual reasoning length (Spearman ρ)\n"
        "Top: unadjusted · Bottom: adjusted for MATH level and category",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=4.5)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    sns.set_theme(style="whitegrid", context="talk")
    observations, manifests = _load_observations(args.features_dir)
    long = _standardized_long(observations)
    summary = _summarize_dynamics(long, args.bootstrap_repetitions)
    separation = _correctness_separation(long, args.bootstrap_repetitions)
    associations = _length_associations(observations, args.bootstrap_repetitions)
    long.to_parquet(output_dir / "phase04_prefix_feature_observations.parquet", index=False)
    summary.to_parquet(output_dir / "phase04_feature_dynamics.parquet", index=False)
    separation.to_parquet(
        output_dir / "phase04_correctness_separation.parquet", index=False
    )
    associations.to_parquet(
        output_dir / "phase04_feature_length_associations.parquet", index=False
    )
    _plot_movement(
        summary,
        output_dir / "phase04_feature_movement.png",
        "fixed_512",
    )
    _plot_movement(
        summary,
        output_dir / "phase04_feature_movement_at_risk.png",
        "at_risk",
    )
    _plot_evolution(
        summary,
        output_dir / "phase04_uncertainty_evolution_by_correctness.png",
        (
            "normalized_entropy_mean",
            "normalized_entropy_std",
            "normalized_entropy_slope",
            "normalized_entropy_max_rise",
        ),
        "Uncertainty evolution by terminal correctness",
    )
    _plot_evolution(
        summary,
        output_dir / "phase04_spectral_evolution_by_correctness.png",
        (
            "spectral_normalized_entropy_entropy",
            "spectral_normalized_entropy_high_energy_ratio",
            "spectral_surprisal_entropy",
        ),
        "Spectral evolution by terminal correctness",
    )
    _plot_correctness_gap(
        separation, output_dir / "phase04_feature_correctness_separation.png", False
    )
    _plot_correctness_gap(
        separation, output_dir / "phase04_spectral_correctness_separation.png", True
    )
    _plot_length_associations(
        associations, output_dir / "phase04_feature_length_associations.png"
    )
    capped = int(
        observations.loc[observations["prefix_length"] == 16, "capped_16k"].sum()
    )
    write_json_atomic(
        output_dir / "dynamics_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "descriptive",
            "summary": (
                "Prefix-censored uncertainty, geometry, and spectral features were "
                "tracked from 16 to 512 tokens and stratified by terminal correctness."
            ),
            "metrics": {
                "prefixes": list(PREFIXES),
                "features": len(FEATURES),
                "trajectories": int(
                    observations.loc[observations["prefix_length"] == 16, "run_id"].nunique()
                ),
                "capped_at_16k": capped,
                "association_rows": len(associations),
            },
            "warnings": [
                "Exact-length correlations treat 16K-capped trajectories at their observed lower bound; held-out threshold prediction is the censoring-robust length analysis.",
                "At-risk curves change composition with prefix; fixed-512 cohort figures are reported to separate feature movement from risk-set attrition.",
            ],
            "input_tables": manifests,
        },
    )


if __name__ == "__main__":
    main()
