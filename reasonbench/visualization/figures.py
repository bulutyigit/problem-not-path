"""Publication-oriented static figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasonbench.storage import ensure_directory


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def plot_condition_accuracy(
    feature_frame: pd.DataFrame,
    condition_column: str,
    output_path: str | Path,
) -> Path:
    """Plot correctness with problem-bootstrap uncertainty approximated by problem means."""

    _style()
    problem_means = (
        feature_frame.groupby([condition_column, "dataset", "problem_id"], dropna=False)["correct"]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=problem_means,
        x=condition_column,
        y="correct",
        hue="dataset",
        errorbar=("ci", 95),
        ax=axis,
    )
    axis.set_ylabel("Correctness rate")
    axis.set_xlabel(condition_column.replace("_", " ").title())
    axis.set_ylim(0, 1)
    axis.set_title("Correctness by experimental condition")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_condition_profile(
    feature_frame: pd.DataFrame,
    condition_column: str,
    output_path: str | Path,
) -> Path:
    """Plot outcomes, compute, and intervention-quality diagnostics."""

    _style()
    frame = feature_frame.copy()
    frame["parse_success"] = frame["parse_status"] != "missing"
    frame["truncated"] = frame["finish_reason"].isin({"max_new_tokens", "answer_reserve"})
    is_budget_phase = (
        "assigned_reasoning_budget" in frame
        and frame["assigned_reasoning_budget"].notna().any()
        and "reasoning_boundary_forced" in frame
    )
    specifications = [
        ("correct", "Correctness rate", (0, 1)),
        ("trajectory_token_count", "Included reasoning tokens", None),
        ("parse_success", "Final-answer parse rate", (0, 1)),
        ("truncated", "Truncation rate", (0, 1)),
    ]
    if is_budget_phase:
        specifications = [
            ("correct", "Correctness rate", (0, 1)),
            ("trajectory_token_count", "Included reasoning tokens", None),
            ("elapsed_seconds", "Generation time (seconds)", None),
            ("reasoning_boundary_forced", "Forced thought-boundary rate", (0, 1)),
            ("parse_success", "Final-answer parse rate", (0, 1)),
            ("truncated", "Final-answer truncation rate", (0, 1)),
        ]
    columns = 3 if is_budget_phase else 2
    figure, axes = plt.subplots(2, columns, figsize=(7.5 * columns, 10))
    for axis, (metric, label, limits) in zip(
        axes.flat,
        specifications,
        strict=True,
    ):
        problem_means = (
            frame.groupby(
                [condition_column, "dataset", "problem_id"],
                dropna=False,
            )[metric]
            .mean()
            .reset_index()
        )
        sns.barplot(
            data=problem_means,
            x=condition_column,
            y=metric,
            hue="dataset",
            errorbar=("ci", 95),
            ax=axis,
        )
        axis.set_xlabel(condition_column.replace("_", " ").title())
        axis.set_ylabel(label)
        if limits is not None:
            axis.set_ylim(*limits)
    figure.suptitle("Condition outcome and data-quality profile")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_correctness_feature_profile(
    feature_frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot selected normalized trajectory features by final correctness."""

    _style()
    candidates = [
        "normalized_entropy_mean",
        "surprisal_mean",
        "top1_top2_logit_margin_mean",
        "geometry_mean_relative_velocity",
        "geometry_mean_cosine_drift",
        "geometry_trajectory_efficiency",
    ]
    columns = [column for column in candidates if column in feature_frame]
    melted = feature_frame.melt(
        id_vars=["model_key", "correct"],
        value_vars=columns,
        var_name="feature",
        value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    melted["standardized_value"] = melted.groupby(
        ["model_key", "feature"],
        sort=False,
    )["value"].transform(lambda values: (values - values.mean()) / max(values.std(), 1e-12))
    figure, axis = plt.subplots(figsize=(14, 7))
    sns.pointplot(
        data=melted,
        x="feature",
        y="standardized_value",
        hue="correct",
        dodge=0.25,
        errorbar=("ci", 95),
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Within-model standardized trajectory feature")
    axis.set_ylabel("Standardized mean with 95% interval")
    axis.set_xticklabels(
        [label.get_text().replace("_", " ") for label in axis.get_xticklabels()],
        rotation=30,
        ha="right",
    )
    axis.set_title("Correct versus incorrect trajectory profiles")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_ablation(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Plot AUROC and clustered confidence intervals for feature ablations."""

    _style()
    rows = []
    for result in results:
        auroc = result["metrics"]["auroc"]
        rows.append(
            {
                "feature_set": result["feature_set"],
                "model_name": result["model_name"],
                "value": auroc["value"],
                "low": auroc["ci_low"],
                "high": auroc["ci_high"],
            }
        )
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(12, 7))
    feature_order = list(frame["feature_set"].drop_duplicates())
    feature_positions = {feature_set: index for index, feature_set in enumerate(feature_order)}
    model_groups = list(frame.groupby("model_name", sort=True))
    for model_index, (model_name, group) in enumerate(model_groups):
        offset = (model_index - (len(model_groups) - 1) / 2) * 0.14
        positions = (
            np.asarray(
                [feature_positions[value] for value in group["feature_set"]],
                dtype=float,
            )
            + offset
        )
        axis.errorbar(
            positions,
            group["value"],
            yerr=[group["value"] - group["low"], group["high"] - group["value"]],
            fmt="o",
            capsize=4,
            label=model_name.replace("_", " "),
        )
    axis.set_xticks(np.arange(len(feature_order)), feature_order, rotation=35, ha="right")
    axis.set_ylabel("AUROC")
    axis.set_title("Correctness-prediction feature ablations")
    axis.legend()
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_calibration_comparison(
    prediction_sets: dict[str, pd.DataFrame],
    output_path: str | Path,
    bins: int = 10,
) -> Path:
    """Plot equal-width reliability curves for selected predictors."""

    _style()
    figure, axis = plt.subplots(figsize=(8, 7))
    edges = np.linspace(0, 1, bins + 1)
    for label, frame in prediction_sets.items():
        rows = []
        probabilities = frame["probability"].to_numpy(dtype=float)
        labels = frame["correct"].to_numpy(dtype=float)
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            mask = (
                (probabilities >= lower) & (probabilities <= upper)
                if upper == 1
                else (probabilities >= lower) & (probabilities < upper)
            )
            if mask.any():
                rows.append((probabilities[mask].mean(), labels[mask].mean()))
        if rows:
            values = np.asarray(rows)
            axis.plot(values[:, 0], values[:, 1], marker="o", label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted correctness")
    axis.set_ylabel("Observed correctness")
    axis.set_title("Held-out calibration")
    axis.legend()
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_logistic_effects(
    fitted_predictor: Any,
    output_path: str | Path,
    maximum_features: int = 25,
) -> Path:
    """Plot the largest absolute standardized Logistic Regression coefficients."""

    _style()
    pipeline = fitted_predictor.estimator
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    coefficients = classifier.coef_[0]
    order = np.argsort(np.abs(coefficients))[-maximum_features:]
    frame = pd.DataFrame(
        {
            "feature": [
                name.replace("numeric__", "").replace("categorical__", "") for name in names[order]
            ],
            "coefficient": coefficients[order],
        }
    ).sort_values("coefficient")
    figure, axis = plt.subplots(figsize=(10, 9))
    colors = ["#b44" if value < 0 else "#2878b5" for value in frame["coefficient"]]
    axis.barh(frame["feature"], frame["coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Standardized Logistic Regression coefficient")
    axis.set_title("Largest full-model feature effects")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_transfer_matrix(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Plot AUROC for directed transfer experiments."""

    _style()
    sources = sorted({result["source_value"] for result in results})
    targets = sorted({result["target_value"] for result in results})
    matrix = pd.DataFrame(index=sources, columns=targets, dtype=float)
    for result in results:
        matrix.loc[result["source_value"], result["target_value"]] = result["metrics"]["auroc"][
            "value"
        ]
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0.5, vmax=1.0, cmap="viridis", ax=axis)
    axis.set_xlabel("Target")
    axis.set_ylabel("Source")
    axis.set_title("Zero-shot transfer AUROC")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_early_prediction(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Plot prediction performance versus observed token prefix."""

    _style()
    frame = pd.DataFrame(
        [
            {
                "scope": result.get("scope", "pooled"),
                "prefix_length": result["prefix_length"],
                "auroc": result["metrics"]["auroc"]["value"],
                "low": result["metrics"]["auroc"]["ci_low"],
                "high": result["metrics"]["auroc"]["ci_high"],
            }
            for result in results
        ]
    ).sort_values("prefix_length")
    figure, axis = plt.subplots(figsize=(10, 6))
    for scope, group in frame.groupby("scope", sort=True):
        group = group.sort_values("prefix_length")
        axis.plot(group["prefix_length"], group["auroc"], marker="o", label=scope)
        if scope == "pooled":
            axis.fill_between(
                group["prefix_length"],
                group["low"],
                group["high"],
                alpha=0.2,
            )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Observed reasoning tokens")
    axis.set_ylabel("AUROC")
    axis.set_title("Early correctness prediction")
    axis.legend(title="Evaluation scope")
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_early_compute_savings(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot fixed-prefix coverage and the upper-bound remaining-token opportunity."""

    _style()
    frame = pd.DataFrame(results)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for scope, group in frame.groupby("scope", sort=True):
        group = group.sort_values("prefix_length")
        axes[0].plot(
            group["prefix_length"],
            group["coverage"],
            marker="o",
            label=scope,
        )
        axes[1].plot(
            group["prefix_length"],
            group["upper_bound_compute_savings_fraction"],
            marker="o",
            label=scope,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Observed reasoning tokens")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Eligible trajectory coverage")
    axes[0].set_title("Fixed-prefix coverage")
    axes[1].set_ylabel("Remaining-token fraction")
    axes[1].set_title("Upper-bound compute opportunity")
    axes[1].legend(title="Evaluation scope", bbox_to_anchor=(1.02, 1))
    output = Path(output_path)
    ensure_directory(output.parent)
    figure.savefig(output)
    plt.close(figure)
    return output
