"""Zero-shot feature transfer across models or datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reasonbench.evaluation.metrics import clustered_bootstrap
from reasonbench.evaluation.predictor import feature_columns, fit_predictor


def evaluate_transfer_direction(
    frame: pd.DataFrame,
    source_column: str,
    source_value: str,
    target_value: str,
    feature_set: str = "transfer",
    bootstrap_repetitions: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Fit on source training problems and evaluate target test problems."""

    source = frame[
        (frame[source_column] == source_value) & (frame["research_split"] == "train")
    ].copy()
    source_validation = frame[
        (frame[source_column] == source_value) & (frame["research_split"] == "validation")
    ].copy()
    target = frame[
        (frame[source_column] == target_value) & (frame["research_split"] == "test")
    ].copy()
    if source.empty or target.empty:
        raise ValueError(
            f"Transfer direction {source_value} -> {target_value} has an empty source or target"
        )
    columns = feature_columns(frame, feature_set)
    pipeline = fit_predictor(
        frame,
        train=source,
        validation=source_validation,
        feature_set=feature_set,
        model_name="logistic_regression",
        seed=seed,
    )
    probabilities = pipeline.predict_proba(target)
    predictions = target[["run_id", "problem_id", "dataset", "model_key", "correct"]].copy()
    predictions["probability"] = probabilities
    metrics = clustered_bootstrap(
        predictions,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return {
        "source_column": source_column,
        "source_value": source_value,
        "target_value": target_value,
        "feature_set": feature_set,
        "source_trajectories": len(source),
        "source_problems": int(source["problem_id"].nunique()),
        "target_trajectories": len(target),
        "target_problems": int(target["problem_id"].nunique()),
        "feature_columns": columns,
        "calibration_applied": pipeline.calibrator is not None,
        "metrics": metrics,
        "predictions": predictions,
    }
