#!/usr/bin/env python
"""Evaluate held-out correctness prediction at fixed observed-token prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from reasonbench.evaluation.metrics import paired_clustered_metric_difference
from reasonbench.evaluation.predictor import evaluate_one
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic
from reasonbench.visualization import plot_early_compute_savings

DEFAULT_FEATURE_SETS = (
    "early_baseline",
    "early_confidence",
    "early_dynamic_uncertainty",
    "early_transition",
    "early_geometry",
    "early_spectral",
    "early_full_without_spectral",
    "early_full",
)
PRIMARY_BASELINE = "early_baseline"
PRIMARY_SIGNAL = "early_full"
PRIMARY_PREFIX = 128
FEATURE_LABELS = {
    "early_baseline": "Difficulty + context",
    "early_confidence": "+ predictive distribution",
    "early_dynamic_uncertainty": "+ dynamic uncertainty",
    "early_transition": "+ belief change",
    "early_geometry": "+ geometry",
    "early_spectral": "+ spectral",
    "early_full_without_spectral": "All except spectral",
    "early_full": "All early signals",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", action="append", type=int, required=True)
    parser.add_argument("--feature-set", action="append", default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--target-column",
        default="correct",
        help="Binary endpoint to predict; Phase 4b uses needs_intervention.",
    )
    parser.add_argument(
        "--power-audit",
        type=Path,
        help="Label-only power audit required before Phase 4b confirmatory metrics.",
    )
    return parser.parse_args()


def _serializable_result(
    *,
    scope: str,
    prefix: int,
    feature_set: str,
    eligible: pd.DataFrame,
    total_rows: int,
    result,
    target_column: str,
) -> dict:
    split_problem_counts = {
        split: int(eligible.loc[eligible["research_split"] == split, "problem_id"].nunique())
        for split in ("train", "validation", "test")
    }
    remaining = np.maximum(
        eligible["full_trajectory_token_count"].to_numpy(dtype=float) - prefix,
        0.0,
    )
    total = float(eligible["full_trajectory_token_count"].sum())
    return {
        "scope": scope,
        "prefix_length": prefix,
        "feature_set": feature_set,
        "feature_label": FEATURE_LABELS.get(feature_set, feature_set),
        "feature_columns": result.feature_columns,
        "trajectories": len(eligible),
        "problems": int(eligible["problem_id"].nunique()),
        "split_problem_counts": split_problem_counts,
        "coverage": len(eligible) / total_rows if total_rows else 0.0,
        "average_remaining_tokens": float(remaining.mean()),
        "upper_bound_compute_savings_fraction": (
            float(remaining.sum() / total) if total > 0 else 0.0
        ),
        "calibration_applied": result.calibration_applied,
        "target_column": target_column,
        "nan_to_num_interventions": result.nan_to_num_interventions,
        "metrics": result.metrics,
    }


def _evaluate_prefix(
    features_dir: Path,
    prefix: int,
    feature_sets: tuple[str, ...],
    repetitions: int,
    target_column: str,
):
    feature_path = features_dir / f"features_prefix_{prefix}.parquet"
    frame = pd.read_parquet(feature_path)
    results: list[dict] = []
    contrasts: list[dict] = []
    warnings: list[str] = []
    predictions: list[dict] = []
    groups = [("pooled", frame)]
    groups.extend(
        (str(model_key), group.copy()) for model_key, group in frame.groupby("model_key", sort=True)
    )
    for scope, scoped_frame in groups:
        eligible = scoped_frame[scoped_frame["full_trajectory_token_count"] >= prefix].copy()
        train = eligible[eligible["research_split"] == "train"]
        test = eligible[eligible["research_split"] == "test"]
        if (
            target_column not in eligible
            or eligible.empty
            or train[target_column].nunique() < 2
            or test[target_column].nunique() < 2
        ):
            warnings.append(
                f"{scope} at prefix {prefix} lacked both outcome classes in train or test."
            )
            continue
        evaluated = {}
        for feature_set in feature_sets:
            result = evaluate_one(
                eligible,
                feature_set=feature_set,
                model_name="logistic_regression",
                bootstrap_repetitions=repetitions,
                seed=20260728 + prefix,
                target_column=target_column,
            )
            evaluated[feature_set] = result
            prediction_path = f"predictions_{scope}_prefix_{prefix}_{feature_set}.parquet"
            prediction_frame = result.predictions.copy()
            if target_column == "correct":
                # Legacy endpoint: probability is P(correct), operational risk
                # is its complement for backwards-compatible Phase 4 outputs.
                prediction_frame["estimated_error_risk"] = 1.0 - prediction_frame["probability"]
            else:
                prediction_frame[f"estimated_{target_column}_risk"] = prediction_frame[
                    "probability"
                ]
            predictions.append({"path": prediction_path, "frame": prediction_frame})
            results.append(
                _serializable_result(
                    scope=scope,
                    prefix=prefix,
                    feature_set=feature_set,
                    eligible=eligible,
                    total_rows=len(scoped_frame),
                    result=result,
                    target_column=target_column,
                )
            )
        if PRIMARY_BASELINE in evaluated and PRIMARY_SIGNAL in evaluated:
            for metric in ("auroc", "auprc", "brier", "log_loss", "ece"):
                difference = paired_clustered_metric_difference(
                    evaluated[PRIMARY_BASELINE].predictions,
                    evaluated[PRIMARY_SIGNAL].predictions,
                    metric=metric,
                    repetitions=repetitions,
                    seed=20260728 + prefix,
                    target_column=target_column,
                )
                contrasts.append(
                    {
                        "scope": scope,
                        "prefix_length": prefix,
                        "metric": metric,
                        "contrast": "early_full_minus_early_baseline",
                        **difference,
                    }
                )
    input_table = {
        "prefix_length": prefix,
        "path": str(feature_path),
        "sha256": sha256_file(feature_path),
        "rows": len(frame),
    }
    return input_table, results, contrasts, warnings, predictions


def _target_display_name(target_column: str) -> str:
    return target_column.replace("_", " ")


def _output_prefix(target_column: str) -> str:
    """Keep historical filenames stable while making Phase 4b self-describing."""

    return "early" if target_column == "correct" else f"early_{target_column}"


def _plot_primary_auroc(results: list[dict], output_path: Path, target_column: str) -> None:
    frame = pd.DataFrame(results)
    frame = frame[frame["feature_set"].isin({PRIMARY_BASELINE, PRIMARY_SIGNAL})]
    scopes = sorted(frame["scope"].unique(), key=lambda value: (value != "pooled", value))
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, scope in zip(axes, scopes, strict=True):
        scoped = frame[frame["scope"] == scope]
        for feature_set, group in scoped.groupby("feature_set", sort=True):
            group = group.sort_values("prefix_length")
            values = np.array([row["auroc"]["value"] for row in group["metrics"]])
            low = np.array([row["auroc"]["ci_low"] for row in group["metrics"]])
            high = np.array([row["auroc"]["ci_high"] for row in group["metrics"]])
            axis.errorbar(
                group["prefix_length"],
                values,
                yerr=np.vstack([np.maximum(values - low, 0), np.maximum(high - values, 0)]),
                marker="o",
                capsize=3,
                label=FEATURE_LABELS[feature_set],
            )
        axis.axhline(0.5, color="black", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(scoped["prefix_length"].unique()))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_title(scope)
        axis.set_xlabel("Observed reasoning tokens")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out AUROC")
    axes[-1].legend(frameon=False)
    figure.suptitle(
        f"Early {_target_display_name(target_column)} prediction: baseline vs. trajectory signals"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_ablation_gain(results: list[dict], output_path: Path, target_column: str) -> None:
    frame = pd.DataFrame(results)
    frame = frame[frame["scope"].ne("pooled")]
    baseline = frame[frame["feature_set"] == PRIMARY_BASELINE][
        ["scope", "prefix_length", "metrics"]
    ].copy()
    baseline["baseline_auroc"] = baseline["metrics"].map(lambda metrics: metrics["auroc"]["value"])
    merged = frame.merge(
        baseline[["scope", "prefix_length", "baseline_auroc"]],
        on=["scope", "prefix_length"],
        validate="many_to_one",
    )
    merged = merged[merged["feature_set"] != PRIMARY_BASELINE]
    merged["delta_auroc"] = (
        merged["metrics"].map(lambda metrics: metrics["auroc"]["value"]) - merged["baseline_auroc"]
    )
    scopes = sorted(merged["scope"].unique())
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, scope in zip(axes, scopes, strict=True):
        scoped = merged[merged["scope"] == scope]
        for feature_set, group in scoped.groupby("feature_set", sort=True):
            group = group.sort_values("prefix_length")
            axis.plot(
                group["prefix_length"],
                group["delta_auroc"],
                marker="o",
                label=FEATURE_LABELS.get(feature_set, feature_set),
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(scoped["prefix_length"].unique()))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_title(scope)
        axis.set_xlabel("Observed reasoning tokens")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("ΔAUROC versus difficulty/context baseline")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle(
        f"Which early signal families add held-out {_target_display_name(target_column)} information?"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_primary_gain(contrasts: list[dict], output_path: Path, target_column: str) -> None:
    frame = pd.DataFrame(contrasts)
    frame = frame[frame["metric"].eq("auroc")].copy()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for scope, group in frame.groupby("scope", sort=True):
        group = group.sort_values("prefix_length")
        values = group["value"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        axis.errorbar(
            group["prefix_length"],
            values,
            yerr=np.vstack([np.maximum(values - low, 0), np.maximum(high - values, 0)]),
            marker="o",
            capsize=3,
            label=scope,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(frame["prefix_length"].unique()))
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Observed reasoning tokens")
    axis.set_ylabel("Held-out ΔAUROC: all early signals − baseline")
    axis.set_title(
        f"Incremental early {_target_display_name(target_column)} information beyond difficulty/context"
    )
    axis.grid(alpha=0.2)
    axis.legend(title="Model", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_reliability_and_risk_coverage(
    predictions: list[dict],
    output_path: Path,
    target_column: str,
) -> None:
    """Describe calibration and operational triage at the primary prefix only."""

    primary = [
        row["frame"]
        for row in predictions
        if f"_prefix_{PRIMARY_PREFIX}_{PRIMARY_SIGNAL}.parquet" in row["path"]
        and "pooled" in row["path"]
    ]
    if not primary:
        return
    frame = pd.concat(primary, ignore_index=True)
    labels = frame[target_column].astype(int).to_numpy()
    probabilities = frame["probability"].to_numpy(float)
    edges = np.linspace(0.0, 1.0, 6)
    centres: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if mask.any():
            centres.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    order = np.argsort(-probabilities)
    ordered_labels = labels[order]
    coverage = np.linspace(1 / len(frame), 1.0, len(frame))
    risk = np.cumsum(ordered_labels) / np.arange(1, len(frame) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].plot(centres, observed, marker="o")
    for x, y, count in zip(centres, observed, counts, strict=True):
        axes[0].annotate(f"n={count}", (x, y), xytext=(4, 4), textcoords="offset points")
    axes[0].set(xlabel="Predicted intervention risk", ylabel="Observed intervention rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title("Pooled held-out reliability (five bins)")
    axes[1].plot(coverage, risk, color="#c44e52")
    axes[1].set(xlabel="Coverage retained after taking highest-risk trajectories first", ylabel="Intervention rate among retained trajectories")
    axes[1].set_title("Risk–coverage diagnostic")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Primary {PRIMARY_PREFIX}-token {_target_display_name(target_column)} risk score; descriptive, held-out"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    prefixes = sorted(set(args.prefix_length))
    feature_sets = tuple(dict.fromkeys(args.feature_set or DEFAULT_FEATURE_SETS))
    output_prefix = _output_prefix(args.target_column)
    missing_primary = {PRIMARY_BASELINE, PRIMARY_SIGNAL} - set(feature_sets)
    if missing_primary:
        raise ValueError(
            f"Early evaluation requires primary baseline and signal sets: {sorted(missing_primary)}"
        )
    if args.power_audit is not None:
        power_audit = read_json(args.power_audit)
        if not bool(power_audit.get("gate_passed", False)):
            write_json_atomic(
                output_dir / "early_summary.json",
                {
                    "technical_status": "passed",
                    "scientific_outcome": "underpowered",
                    "target_column": args.target_column,
                    "summary": (
                        "The frozen label-only power gate did not pass; no feature-performance "
                        "or predictor metrics were computed. Run the pre-registered challenge "
                        "panel before opening held-out discrimination results."
                    ),
                    "power_audit": str(args.power_audit),
                },
            )
            return
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        evaluated = joblib.Parallel(n_jobs=min(args.workers, len(prefixes)))(
            joblib.delayed(_evaluate_prefix)(
                args.features_dir,
                prefix,
                feature_sets,
                args.bootstrap_repetitions,
                args.target_column,
            )
            for prefix in prefixes
        )
    input_tables = []
    results: list[dict] = []
    contrasts: list[dict] = []
    warnings: list[str] = []
    for input_table, prefix_results, prefix_contrasts, prefix_warnings, predictions in evaluated:
        input_tables.append(input_table)
        results.extend(prefix_results)
        contrasts.extend(prefix_contrasts)
        warnings.extend(prefix_warnings)
        for prediction in predictions:
            prediction["frame"].to_parquet(output_dir / prediction["path"], index=False)
    if not results:
        raise ValueError("No eligible held-out early-prediction evaluations were produced")
    results_name = (
        "early_prediction_results.json"
        if args.target_column == "correct"
        else f"{output_prefix}_prediction_results.json"
    )
    contrasts_name = (
        "early_primary_contrasts.json"
        if args.target_column == "correct"
        else f"{output_prefix}_primary_contrasts.json"
    )
    manifest_name = (
        "early_input_manifest.json"
        if args.target_column == "correct"
        else f"{output_prefix}_input_manifest.json"
    )
    write_json_atomic(output_dir / results_name, results)
    write_json_atomic(output_dir / contrasts_name, contrasts)
    write_json_atomic(
        output_dir / manifest_name,
            {
                "tables": input_tables,
                "feature_sets": list(feature_sets),
                "target_column": args.target_column,
                "spectral_availability_note": (
                    "At 16 and 32 tokens, early_spectral is baseline-equivalent because "
                    "the spectral feature block is all missing by design."
                ),
            },
    )
    _plot_primary_auroc(
        results,
        output_dir
        / (
            "early_failure_baseline_vs_signals.png"
            if args.target_column == "correct"
            else f"{output_prefix}_baseline_vs_signals.png"
        ),
        args.target_column,
    )
    _plot_ablation_gain(
        results,
        output_dir
        / (
            "early_failure_feature_ablation.png"
            if args.target_column == "correct"
            else f"{output_prefix}_feature_ablation.png"
        ),
        args.target_column,
    )
    _plot_primary_gain(
        contrasts,
        output_dir
        / (
            "early_failure_primary_gain.png"
            if args.target_column == "correct"
            else f"{output_prefix}_primary_gain.png"
        ),
        args.target_column,
    )
    full_results = [row for row in results if row["feature_set"] == PRIMARY_SIGNAL]
    plot_early_compute_savings(
        full_results,
        output_dir
        / (
            "early_compute_opportunity.png"
            if args.target_column == "correct"
            else f"{output_prefix}_compute_opportunity.png"
        ),
    )
    _plot_reliability_and_risk_coverage(
        predictions,
        output_dir
        / (
            "early_risk_reliability_and_coverage.png"
            if args.target_column == "correct"
            else f"{output_prefix}_reliability_and_coverage.png"
        ),
        args.target_column,
    )
    model_prefix_auroc = [
        row
        for row in contrasts
        if row["metric"] == "auroc" and row["scope"] != "pooled" and row["prefix_length"] <= 512
    ]
    positive_model_prefixes = [row for row in model_prefix_auroc if row["ci_low"] > 0]
    useful_scopes = sorted({row["scope"] for row in positive_model_prefixes})
    primary_candidates = [
        row
        for row in contrasts
        if row["metric"] == "auroc"
        and row["scope"] == "pooled"
        and row["prefix_length"] == PRIMARY_PREFIX
    ]
    if len(primary_candidates) != 1:
        raise ValueError(f"Expected one pooled primary AUROC contrast at prefix {PRIMARY_PREFIX}")
    primary_contrast = primary_candidates[0]
    primary_positive = primary_contrast["ci_low"] > 0
    write_json_atomic(
        output_dir / "early_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "positive" if primary_positive else "limited",
            "target_column": args.target_column,
            "candidate_for_stopping": len(useful_scopes) >= 2,
            "summary": (
                "Problem-held-out fixed-prefix prediction compared early trajectory signals "
                "against a difficulty/context baseline. Saved probabilities represent "
                f"P({args.target_column}); legacy correctness analyses additionally retain "
                "estimated_error_risk = 1 - P(correct)."
            ),
            "metrics": {
                "evaluated_model_prefix_feature_rows": len(results),
                "confirmatory_prefix": PRIMARY_PREFIX,
                "confirmatory_pooled_delta_auroc": primary_contrast,
                "confirmatory_positive": primary_positive,
                "model_prefix_auroc_contrasts": len(model_prefix_auroc),
                "positive_model_prefix_contrasts": len(positive_model_prefixes),
                "positive_model_scopes": useful_scopes,
            },
            "warnings": [
                "Every prefix analysis is conditional on trajectories still active at that prefix; coverage is reported.",
                "One decoding seed supports problem-held-out prediction but not sampling-stability claims.",
                "The pooled 128-token ΔAUROC is the confirmatory test; other prefixes and model-specific contrasts are timing/heterogeneity analyses with pointwise intervals.",
                "Early spectral features at 16 and 32 tokens are baseline-equivalent because their spectral block is structurally unavailable.",
                *warnings,
            ],
        },
    )


if __name__ == "__main__":
    main()
