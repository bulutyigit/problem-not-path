"""Classification metrics and problem-clustered bootstrap."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(
            float(labels[mask].mean()) - float(probabilities[mask].mean())
        )
    return value if total else math.nan


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute discrimination and calibration metrics."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    if len(np.unique(labels)) < 2:
        auroc = math.nan
        auprc = math.nan
    else:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(labels, probabilities),
    }


def clustered_bootstrap(
    predictions: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = 20260728,
    metric_function: Callable[[np.ndarray, np.ndarray], dict[str, float]] = classification_metrics,
    target_column: str = "correct",
) -> dict[str, dict[str, float]]:
    """Bootstrap complete problem clusters with replacement."""

    required = {"problem_id", target_column, "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    point = metric_function(
        predictions[target_column].to_numpy(),
        predictions["probability"].to_numpy(),
    )
    groups = list(predictions.groupby("problem_id", sort=False))
    if not groups:
        raise ValueError("No problem groups are available for bootstrap")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in point}
    for _ in range(repetitions):
        sampled_indices = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[index][1] for index in sampled_indices], ignore_index=True)
        metrics = metric_function(
            sample[target_column].to_numpy(),
            sample["probability"].to_numpy(),
        )
        for metric, value in metrics.items():
            if np.isfinite(value):
                draws[metric].append(value)
    result: dict[str, dict[str, float]] = {}
    for metric, value in point.items():
        values = np.asarray(draws[metric], dtype=float)
        result[metric] = {
            "value": value,
            "ci_low": float(np.quantile(values, 0.025)) if len(values) else math.nan,
            "ci_high": float(np.quantile(values, 0.975)) if len(values) else math.nan,
        }
    return result


def paired_clustered_metric_difference(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    metric: str,
    repetitions: int = 2000,
    seed: int = 20260728,
    target_column: str = "correct",
) -> dict[str, float]:
    """Bootstrap a paired right-minus-left metric difference by problem."""

    keys = ["run_id", "problem_id", target_column]
    left = left_predictions[keys + ["probability"]].rename(
        columns={"probability": "left_probability"}
    )
    right = right_predictions[keys + ["probability"]].rename(
        columns={"probability": "right_probability"}
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Prediction rows do not align for paired metric comparison")

    def difference(sample: pd.DataFrame) -> float:
        left_metric = classification_metrics(
            sample[target_column].to_numpy(),
            sample["left_probability"].to_numpy(),
        )[metric]
        right_metric = classification_metrics(
            sample[target_column].to_numpy(),
            sample["right_probability"].to_numpy(),
        )[metric]
        return float(right_metric - left_metric)

    point = difference(paired)
    groups = list(paired.groupby("problem_id", sort=False))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[index][1] for index in sampled], ignore_index=True)
        value = difference(sample)
        if np.isfinite(value):
            draws.append(value)
    return {
        "value": point,
        "ci_low": float(np.quantile(draws, 0.025)) if draws else math.nan,
        "ci_high": float(np.quantile(draws, 0.975)) if draws else math.nan,
    }
