#!/usr/bin/env python
"""Predict whether reasoning will cross future token thresholds from early prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasonbench.evaluation.metrics import paired_clustered_metric_difference
from reasonbench.evaluation.predictor import evaluate_one
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

DEFAULT_PREFIXES = (16, 32, 64, 128, 256, 512)
DEFAULT_THRESHOLDS = (1024, 2048, 4096, 8192)
FEATURE_SETS = (
    "early_baseline",
    "early_confidence",
    "early_geometry",
    "early_spectral",
    "early_full",
)
BASELINE = "early_baseline"
SIGNAL = "early_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", action="append", type=int, default=[])
    parser.add_argument("--threshold", action="append", type=int, default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU workers for independent prefix/threshold evaluations (default: 1).",
    )
    return parser.parse_args()


def _evaluate(
    frame: pd.DataFrame,
    *,
    prefix: int,
    threshold: int,
    scope: str,
    repetitions: int,
) -> tuple[list[dict], list[dict], list[tuple[str, pd.DataFrame]]]:
    eligible = frame[frame["full_trajectory_token_count"] >= prefix].copy()
    eligible["correct"] = eligible["full_trajectory_token_count"] >= threshold
    train = eligible[eligible["research_split"] == "train"]
    test = eligible[eligible["research_split"] == "test"]
    if eligible.empty or train["correct"].nunique() < 2 or test["correct"].nunique() < 2:
        return [], [], []
    results: list[dict] = []
    fitted = {}
    prediction_frames = []
    for feature_set in FEATURE_SETS:
        result = evaluate_one(
            eligible,
            feature_set=feature_set,
            model_name="logistic_regression",
            bootstrap_repetitions=repetitions,
            seed=20260805 + prefix + threshold,
        )
        fitted[feature_set] = result
        filename = f"length_predictions_{scope}_p{prefix}_t{threshold}_{feature_set}.parquet"
        prediction_frames.append((filename, result.predictions))
        results.append(
            {
                "scope": scope,
                "prefix_length": prefix,
                "future_threshold": threshold,
                "feature_set": feature_set,
                "feature_columns": result.feature_columns,
                "trajectories": len(eligible),
                "problems": int(eligible["problem_id"].nunique()),
                "positive_rate": float(eligible["correct"].mean()),
                "coverage": len(eligible) / len(frame) if len(frame) else 0.0,
                "metrics": result.metrics,
            }
        )
    contrasts = []
    for metric in ("auroc", "auprc", "brier", "log_loss", "ece"):
        contrasts.append(
            {
                "scope": scope,
                "prefix_length": prefix,
                "future_threshold": threshold,
                "metric": metric,
                "contrast": "early_full_minus_early_baseline",
                **paired_clustered_metric_difference(
                    fitted[BASELINE].predictions,
                    fitted[SIGNAL].predictions,
                    metric=metric,
                    repetitions=repetitions,
                    seed=20260805 + prefix + threshold,
                ),
            }
        )
    return results, contrasts, prediction_frames


def _evaluate_task(
    frame: pd.DataFrame,
    *,
    prefix: int,
    threshold: int,
    scope: str,
    repetitions: int,
) -> tuple[int, int, str, list[dict], list[dict], list[tuple[str, pd.DataFrame]]]:
    """Run one independent held-out duration-prediction task."""

    rows, contrasts, predictions = _evaluate(
        frame,
        prefix=prefix,
        threshold=threshold,
        scope=scope,
        repetitions=repetitions,
    )
    return prefix, threshold, scope, rows, contrasts, predictions


def _metric_frame(results: list[dict], feature_set: str, metric: str) -> pd.DataFrame:
    rows = []
    for row in results:
        if row["feature_set"] == feature_set:
            rows.append(
                {
                    "scope": row["scope"],
                    "prefix_length": row["prefix_length"],
                    "future_threshold": row["future_threshold"],
                    "value": row["metrics"][metric]["value"],
                }
            )
    return pd.DataFrame(rows)


def _plot_auroc(results: list[dict], output_path: Path) -> None:
    frame = _metric_frame(results, SIGNAL, "auroc")
    scopes = sorted(frame["scope"].unique(), key=lambda value: (value != "pooled", value))
    figure, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, scope in zip(axes, scopes, strict=True):
        data = frame[frame["scope"] == scope].pivot(
            index="future_threshold", columns="prefix_length", values="value"
        )
        sns.heatmap(data, vmin=0.5, vmax=1.0, cmap="viridis", annot=True, fmt=".2f", ax=axis)
        axis.set_title(scope)
        axis.set_xlabel("Observed tokens")
        axis.set_ylabel("Will exceed token threshold")
    figure.suptitle("Held-out prediction of eventual reasoning duration (all early signals)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_gain(contrasts: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(contrasts)
    frame = frame[frame["metric"] == "auroc"]
    scopes = sorted(frame["scope"].unique(), key=lambda value: (value != "pooled", value))
    figure, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    limit = max(0.1, float(np.nanmax(np.abs(frame["value"]))))
    for axis, scope in zip(axes, scopes, strict=True):
        data = frame[frame["scope"] == scope].pivot(
            index="future_threshold", columns="prefix_length", values="value"
        )
        sns.heatmap(
            data,
            vmin=-limit,
            vmax=limit,
            center=0,
            cmap="vlag",
            annot=True,
            fmt=".2f",
            ax=axis,
        )
        axis.set_title(scope)
        axis.set_xlabel("Observed tokens")
        axis.set_ylabel("Will exceed token threshold")
    figure.suptitle("Added duration information: all early signals − baseline (ΔAUROC)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    prefixes = tuple(sorted(set(args.prefix_length or DEFAULT_PREFIXES)))
    thresholds = tuple(sorted(set(args.threshold or DEFAULT_THRESHOLDS)))
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    results: list[dict] = []
    contrasts: list[dict] = []
    warnings: list[str] = []
    manifests = []
    tasks: list[tuple[pd.DataFrame, int, int, str]] = []
    for prefix in prefixes:
        path = args.features_dir / f"features_prefix_{prefix}.parquet"
        frame = pd.read_parquet(path)
        manifests.append({"prefix": prefix, "path": str(path), "sha256": sha256_file(path)})
        scopes = [("pooled", frame), *frame.groupby("model_key", sort=True)]
        for scope, scoped in scopes:
            for threshold in thresholds:
                if threshold <= prefix:
                    continue
                tasks.append((scoped.copy(), prefix, threshold, str(scope)))
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        task_outputs = joblib.Parallel(n_jobs=min(args.workers, len(tasks)))(
            joblib.delayed(_evaluate_task)(
                scoped,
                prefix=prefix,
                threshold=threshold,
                scope=scope,
                repetitions=args.bootstrap_repetitions,
            )
            for scoped, prefix, threshold, scope in tasks
        )
    for prefix, threshold, scope, rows, paired, predictions in task_outputs:
        if not rows:
            warnings.append(
                f"{scope}, prefix {prefix}, threshold {threshold}: both classes were not available in train/test."
            )
        results.extend(rows)
        contrasts.extend(paired)
        for filename, prediction in predictions:
            prediction.to_parquet(output_dir / filename, index=False)
    if not results:
        raise ValueError("No eligible held-out duration evaluations were produced")
    write_json_atomic(output_dir / "early_length_prediction_results.json", results)
    write_json_atomic(output_dir / "early_length_primary_contrasts.json", contrasts)
    _plot_auroc(results, output_dir / "early_length_prediction_auroc.png")
    _plot_gain(contrasts, output_dir / "early_length_prediction_gain.png")
    write_json_atomic(
        output_dir / "length_prediction_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "descriptive",
            "summary": (
                "Problem-held-out models tested whether 16–512-token trajectory signals "
                "predict crossing future 1K, 2K, 4K, and 8K reasoning thresholds."
            ),
            "metrics": {
                "prefixes": list(prefixes),
                "future_thresholds": list(thresholds),
                "evaluation_rows": len(results),
                "contrast_rows": len(contrasts),
            },
            "warnings": [
                "Threshold outcomes below the 16K generation cap are observable even for cap-censored trajectories.",
                "Exact eventual length remains right-censored at 16K; threshold prediction is the inferential duration analysis.",
                *warnings,
            ],
            "input_tables": manifests,
        },
    )


if __name__ == "__main__":
    main()
