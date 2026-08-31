#!/usr/bin/env python
"""Run within-model correctness-prediction ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from reasonbench.evaluation.metrics import paired_clustered_metric_difference
from reasonbench.evaluation.predictor import evaluate_feature_sets
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic
from reasonbench.visualization import (
    plot_ablation,
    plot_calibration_comparison,
    plot_logistic_effects,
)

FEATURE_SETS = [
    "constant",
    "difficulty",
    "length",
    "mean_confidence",
    "dynamic_uncertainty",
    "geometry",
    "spectral",
    "full_without_spectral",
    "full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _serializable_result(result) -> dict:
    return {
        "feature_set": result.feature_set,
        "model_name": result.model_name,
        "feature_columns": result.feature_columns,
        "metrics": result.metrics,
        "calibration_applied": result.calibration_applied,
    }


def _evaluate_model(model_key: str, model_frame: pd.DataFrame, repetitions: int):
    evaluated = evaluate_feature_sets(
        model_frame,
        feature_sets=FEATURE_SETS,
        bootstrap_repetitions=repetitions,
    )
    logistic = {
        result.feature_set: result
        for result in evaluated
        if result.model_name == "logistic_regression"
    }
    difference = paired_clustered_metric_difference(
        logistic["length"].predictions,
        logistic["full"].predictions,
        metric="auroc",
        repetitions=repetitions,
    )
    return model_key, evaluated, difference


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    frame = pd.read_parquet(args.features)
    write_json_atomic(
        output_dir / "prediction_input_manifest.json",
        {
            "features_path": str(args.features),
            "features_sha256": sha256_file(args.features),
            "rows": len(frame),
        },
    )
    model_keys = args.model_key or sorted(frame["model_key"].unique())
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        model_evaluations = joblib.Parallel(n_jobs=min(args.workers, len(model_keys)))(
            joblib.delayed(_evaluate_model)(
                str(model_key),
                frame[frame["model_key"] == model_key].copy(),
                args.bootstrap_repetitions,
            )
            for model_key in model_keys
        )
    all_results: dict = {"models": {}}
    improvements = []
    for model_key, evaluated, difference in model_evaluations:
        model_root = ensure_directory(output_dir / model_key)
        serialized = [_serializable_result(result) for result in evaluated]
        all_results["models"][model_key] = serialized
        for result in evaluated:
            result.predictions.to_parquet(
                model_root / f"predictions_{result.model_name}_{result.feature_set}.parquet",
                index=False,
            )
            joblib.dump(
                result.pipeline,
                model_root / f"pipeline_{result.model_name}_{result.feature_set}.joblib",
            )
        logistic = {
            result.feature_set: result
            for result in evaluated
            if result.model_name == "logistic_regression"
        }
        improvements.append({"model_key": model_key, **difference})
        write_json_atomic(model_root / "ablation_results.json", serialized)
        plot_ablation(serialized, model_root / "ablation_auroc.png")
        plot_calibration_comparison(
            {
                "difficulty + length": logistic["length"].predictions,
                "full trajectory": logistic["full"].predictions,
            },
            model_root / "calibration.png",
        )
        plot_logistic_effects(
            logistic["full"].pipeline,
            model_root / "full_logistic_effects.png",
        )
    write_json_atomic(output_dir / "all_ablation_results.json", all_results)
    write_json_atomic(output_dir / "primary_improvements.json", improvements)
    positive = [item for item in improvements if item["ci_low"] > 0]
    negative = [item for item in improvements if item["ci_high"] < 0]
    if len(positive) >= 2:
        outcome = "positive"
        decision = "run_early_prediction"
    elif len(negative) == len(model_keys):
        outcome = "negative"
        decision = "run_early_prediction"
    else:
        outcome = "limited"
        decision = "run_early_prediction"
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": outcome,
            "next_decision": decision,
            "summary": "Within-model correctness-prediction ablations completed.",
            "metrics": {
                "models": len(model_keys),
                "trajectories": len(frame),
                "positive_models": len(positive),
            },
            "warnings": [],
        },
    )


if __name__ == "__main__":
    main()
