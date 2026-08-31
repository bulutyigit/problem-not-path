#!/usr/bin/env python
"""Evaluate directed transfer across every observed model and dataset domain."""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import joblib
import pandas as pd

from reasonbench.evaluation.transfer import evaluate_transfer_direction
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic
from reasonbench.visualization import plot_transfer_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _without_predictions(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "predictions"}


def _evaluate_direction(
    frame: pd.DataFrame,
    source_column: str,
    source: str,
    target: str,
    repetitions: int,
) -> dict:
    return evaluate_transfer_direction(
        frame,
        source_column=source_column,
        source_value=source,
        target_value=target,
        bootstrap_repetitions=repetitions,
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    frame = pd.read_parquet(args.features)
    write_json_atomic(
        output_dir / "transfer_input_manifest.json",
        {
            "features_path": str(args.features),
            "features_sha256": sha256_file(args.features),
            "rows": len(frame),
        },
    )
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    model_directions = list(permutations(sorted(frame["model_key"].unique()), 2))
    dataset_directions = list(permutations(sorted(frame["dataset"].unique()), 2))
    tasks = [("model_key", str(source), str(target)) for source, target in model_directions] + [
        ("dataset", str(source), str(target)) for source, target in dataset_directions
    ]
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        evaluated = joblib.Parallel(n_jobs=min(args.workers, len(tasks)))(
            joblib.delayed(_evaluate_direction)(
                frame,
                source_column,
                source,
                target,
                args.bootstrap_repetitions,
            )
            for source_column, source, target in tasks
        )
    model_results = []
    dataset_results = []
    for (source_column, source, target), result in zip(tasks, evaluated, strict=True):
        result["predictions"].to_parquet(
            output_dir / f"predictions_{'model' if source_column == 'model_key' else 'dataset'}_"
            f"{source}_to_{target}.parquet",
            index=False,
        )
        target_results = model_results if source_column == "model_key" else dataset_results
        target_results.append(_without_predictions(result))
    write_json_atomic(output_dir / "model_transfer_results.json", model_results)
    write_json_atomic(output_dir / "dataset_transfer_results.json", dataset_results)
    plot_transfer_matrix(model_results, output_dir / "model_transfer_auroc.png")
    transferable = [
        result for result in model_results if result["metrics"]["auroc"]["ci_low"] > 0.5
    ]
    transferable_sources = {result["source_value"] for result in transferable}
    robust_transfer = len(transferable) >= 2 and len(transferable_sources) >= 2
    if dataset_results:
        summary = "Cross-model and cross-dataset transfer evaluation completed."
        warnings: list[str] = []
    else:
        summary = "Matched-MATH cross-model transfer evaluation completed."
        warnings = [
            "Dataset transfer was not estimated from the MATH-only Phase 3 panel; "
            "GSM8K remains a separately versioned out-of-domain confirmation."
        ]
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": "positive" if robust_transfer else "limited",
            "next_decision": "run_early_prediction",
            "summary": summary,
            "metrics": {
                "model_directions": len(model_results),
                "dataset_directions": len(dataset_results),
                "transferable_model_directions": len(transferable),
                "transferable_model_sources": len(transferable_sources),
            },
            "warnings": warnings,
        },
    )


if __name__ == "__main__":
    main()
