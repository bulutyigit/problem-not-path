#!/usr/bin/env python
"""Measure spectral-feature contribution and finalize the Phase 6 summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from reasonbench.evaluation.metrics import paired_clustered_metric_difference
from reasonbench.storage import (
    ensure_directory,
    read_json,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--early-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _evaluate_model_directory(model_directory: Path, repetitions: int):
    import pandas as pd

    without_path = model_directory / "predictions_logistic_regression_full_without_spectral.parquet"
    full_path = model_directory / "predictions_logistic_regression_full.parquet"
    if not without_path.exists() or not full_path.exists():
        return None
    without = pd.read_parquet(without_path)
    full = pd.read_parquet(full_path)
    increment = {
        "model_key": model_directory.name,
        "auroc": paired_clustered_metric_difference(
            without,
            full,
            metric="auroc",
            repetitions=repetitions,
        ),
        "auprc": paired_clustered_metric_difference(
            without,
            full,
            metric="auprc",
            repetitions=repetitions,
        ),
        "brier": paired_clustered_metric_difference(
            without,
            full,
            metric="brier",
            repetitions=repetitions,
        ),
    }
    input_files = [
        {"path": str(without_path), "sha256": sha256_file(without_path)},
        {"path": str(full_path), "sha256": sha256_file(full_path)},
    ]
    return increment, input_files


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    model_directories = sorted(path for path in args.prediction_root.iterdir() if path.is_dir())
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        evaluated = joblib.Parallel(n_jobs=min(args.workers, len(model_directories)))(
            joblib.delayed(_evaluate_model_directory)(
                model_directory,
                args.bootstrap_repetitions,
            )
            for model_directory in model_directories
        )
    increments = []
    input_files = []
    for evaluation in evaluated:
        if evaluation is None:
            continue
        increment, model_input_files = evaluation
        increments.append(increment)
        input_files.extend(model_input_files)
    if not increments:
        raise ValueError("No paired Phase 5 spectral prediction files were found")
    write_json_atomic(output_dir / "spectral_increment_results.json", increments)
    write_json_atomic(
        output_dir / "spectral_input_manifest.json",
        {"prediction_files": input_files},
    )
    early = read_json(args.early_results_dir / "early_summary.json")
    spectral_positive = [item for item in increments if item["auroc"]["ci_low"] > 0]
    early_candidate = bool(early.get("candidate_for_stopping"))
    if early_candidate:
        decision = "candidate_for_stopping"
    elif any(
        not row.get("calibration_applied", False)
        for row in read_json(args.early_results_dir / "early_prediction_results.json")
        if row.get("scope") == "pooled"
    ):
        decision = "insufficient_calibration"
    else:
        decision = "signal_too_late"
    outcome = "positive" if early_candidate or spectral_positive else "limited"
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": outcome,
            "next_decision": decision,
            "summary": (
                "Fixed-prefix correctness prediction and paired spectral-feature "
                "increment analyses completed."
            ),
            "metrics": {
                "early_candidate": early_candidate,
                "spectral_positive_models": len(spectral_positive),
                "spectral_models": len(increments),
            },
            "warnings": early.get("warnings", []),
        },
    )


if __name__ == "__main__":
    main()
