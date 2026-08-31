#!/usr/bin/env python
"""Evaluate grouped logistic breakthrough forecasts and hazard models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from reasonbench.evaluation.predictor import evaluate_one
from reasonbench.storage import ensure_directory, write_json_atomic

FEATURE_SETS = (
    "early_baseline",
    "early_confidence",
    "early_dynamic_uncertainty",
    "early_transition",
    "early_geometry",
    "early_spectral",
    "early_full_without_spectral",
    "early_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-table", type=Path, required=True)
    parser.add_argument("--hazard-table", type=Path, required=True)
    parser.add_argument("--eventual-success-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def _class_audit(frame: pd.DataFrame, target: str) -> dict:
    return {
        str(split): {
            "rows": len(group),
            "problems": int(group["problem_id"].nunique()),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in group[target].value_counts().items()
            },
        }
        for split, group in frame.groupby("research_split", dropna=False)
    }


def _viable(frame: pd.DataFrame, target: str) -> tuple[bool, str | None]:
    for split in ("train", "test"):
        block = frame[frame["research_split"] == split]
        if block.empty:
            return False, f"{split} split is empty"
        if block[target].nunique() < 2:
            return False, f"{split} split has only one {target} class"
    return True, None


def _evaluate_grid(
    frame: pd.DataFrame,
    *,
    target: str,
    name: str,
    output_dir: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> list[dict]:
    viable, reason = _viable(frame, target)
    if not viable:
        return [
            {
                "analysis": name,
                "status": "underpowered",
                "reason": reason,
                "class_audit": _class_audit(frame, target),
            }
        ]
    records: list[dict] = []
    for feature_set in FEATURE_SETS:
        result = evaluate_one(
            frame,
            feature_set=feature_set,
            model_name="logistic_regression",
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
            target_column=target,
        )
        stem = f"{name}_{feature_set}"
        result.predictions.to_parquet(output_dir / f"{stem}_predictions.parquet", index=False)
        joblib.dump(result.pipeline, output_dir / f"{stem}_model.joblib")
        records.append(
            {
                "analysis": name,
                "status": "estimated",
                "feature_set": feature_set,
                "target": target,
                "feature_count": len(result.feature_columns),
                "features": result.feature_columns,
                "metrics": result.metrics,
                "calibration_applied": result.calibration_applied,
                "class_audit": _class_audit(frame, target),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap-repetitions must be positive")
    output = ensure_directory(args.output_dir)
    horizon = pd.read_parquet(args.horizon_table)
    hazard = pd.read_parquet(args.hazard_table)
    eventual_success = pd.read_parquet(args.eventual_success_table)
    results: list[dict] = []

    # Original baseline retained exactly as eventual terminal correctness, not
    # the rejected composite needs_intervention endpoint.
    results.extend(
        _evaluate_grid(
            eventual_success,
            target="correct",
            name="eventual_success",
            output_dir=output,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    )
    for forecast_horizon, frame in horizon.groupby("forecast_horizon", sort=True):
        results.extend(
            _evaluate_grid(
                frame,
                target="breakthrough_within_horizon",
                name=f"breakthrough_horizon_{int(forecast_horizon)}",
                output_dir=output,
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed + int(forecast_horizon),
            )
        )
    results.extend(
        _evaluate_grid(
            hazard,
            target="breakthrough_in_bin",
            name="discrete_time_hazard",
            output_dir=output,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + 10_000,
        )
    )
    estimated = [record for record in results if record["status"] == "estimated"]
    primary = next(
        (
            record
            for record in estimated
            if record["analysis"] == "breakthrough_horizon_256"
            and record.get("feature_set") == "early_full"
        ),
        None,
    )
    primary_auprc = (
        primary["metrics"]["auprc"]["value"] if primary is not None else math.nan
    )
    summary = {
        "technical_status": "passed" if estimated else "incomplete",
        "scientific_outcome": "underpowered" if not estimated else "pending_interpretation",
        "primary_estimand": "P(breakthrough within 256 tokens | prefix features)",
        "primary_feature_set": "early_full",
        "primary_test_auprc": primary_auprc,
        "grouping_unit": "problem_id",
        "row_level_random_split_used": False,
        "surface_text_model_included": False,
        "sequence_encoder_included": False,
        "results": results,
    }
    write_json_atomic(output / "breakthrough_forecast_summary.json", summary)
    print(output / "breakthrough_forecast_summary.json")


if __name__ == "__main__":
    main()
