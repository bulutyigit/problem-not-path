"""Scalar sequence summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import theilslopes

# Theil--Sen enumerates O(n²) point pairs. A bounded, deterministic grid keeps
# the robust trend feature practical for 8k-token trajectories while preserving
# coverage of the entire observed prefix.
MAX_THEIL_SEN_POINTS = 512


def _safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return math.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _robust_slope(array: np.ndarray, positions: np.ndarray) -> float:
    """Compute a bounded-cost Theil--Sen slope on an evenly spaced grid."""

    if len(array) > MAX_THEIL_SEN_POINTS:
        selected = np.linspace(
            0,
            len(array) - 1,
            num=MAX_THEIL_SEN_POINTS,
            dtype=np.intp,
        )
        array = array[selected]
        positions = positions[selected]
    return float(theilslopes(array, positions).slope)


def summarize_scalar(values: Any, prefix: str) -> dict[str, float]:
    """Return robust descriptive and dynamic features for one scalar trajectory."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"{prefix}_{name}": math.nan for name in _feature_names()}
    positions = np.arange(len(array), dtype=np.float64)
    if len(array) >= 2:
        slope = float(np.polyfit(positions, array, deg=1)[0])
        robust_slope = _robust_slope(array, positions)
        differences = np.diff(array)
        max_rise = float(np.max(differences))
        max_fall = float(np.min(differences))
    else:
        slope = robust_slope = max_rise = max_fall = math.nan
    window = max(1, min(64, len(array) // 4 or 1))
    result = {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "initial_mean": float(np.mean(array[:window])),
        "final_mean": float(np.mean(array[-window:])),
        "slope": slope,
        "robust_slope": robust_slope,
        "max_rise": max_rise,
        "max_fall": max_fall,
        "autocorr_lag1": _safe_autocorrelation(array, 1),
        "autocorr_lag4": _safe_autocorrelation(array, 4),
    }
    return {f"{prefix}_{key}": value for key, value in result.items()}


def _feature_names() -> tuple[str, ...]:
    return (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p10",
        "p25",
        "p75",
        "p90",
        "initial_mean",
        "final_mean",
        "slope",
        "robust_slope",
        "max_rise",
        "max_fall",
        "autocorr_lag1",
        "autocorr_lag4",
    )
