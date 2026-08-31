from __future__ import annotations

import math

import numpy as np
import pandas as pd

from reasonbench.evaluation.metrics import (
    classification_metrics,
    clustered_bootstrap,
    paired_clustered_metric_difference,
)
from reasonbench.features.scalar import MAX_THEIL_SEN_POINTS, summarize_scalar
from reasonbench.features.spectral import summarize_spectrum


def test_scalar_summary_detects_positive_slope() -> None:
    summary = summarize_scalar(np.arange(100, dtype=float), "signal")
    assert summary["signal_mean"] == 49.5
    assert summary["signal_slope"] > 0.99
    assert summary["signal_robust_slope"] > 0.99


def test_scalar_summary_bounds_theil_sen_work_for_long_sequences() -> None:
    values = np.arange(MAX_THEIL_SEN_POINTS * 4, dtype=float)
    summary = summarize_scalar(values, "signal")
    assert summary["signal_robust_slope"] == 1.0


def test_spectral_summary_separates_low_frequency_signal() -> None:
    positions = np.arange(256)
    signal = np.sin(2 * np.pi * 4 * positions / 256)
    summary = summarize_spectrum(signal, "spectral")
    assert summary["spectral_low_energy_ratio"] > 0.9
    assert 0 < summary["spectral_dominant_frequency"] <= 0.1


def test_short_spectrum_returns_missing_features() -> None:
    summary = summarize_spectrum([1.0, 2.0], "spectral")
    assert all(math.isnan(value) for value in summary.values())


def test_spectral_summary_accepts_63_finite_transition_values_at_prefix_64() -> None:
    values = np.linspace(0.0, 1.0, 63)
    summary = summarize_spectrum(values, "spectral")
    assert all(math.isfinite(value) for value in summary.values())
    too_short = summarize_spectrum(values[:-1], "spectral")
    assert all(math.isnan(value) for value in too_short.values())


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [f"run_{index}" for index in range(8)],
            "problem_id": [f"problem_{index // 2}" for index in range(8)],
            "correct": [0, 0, 0, 1, 1, 0, 1, 1],
            "probability": [0.1, 0.2, 0.3, 0.65, 0.7, 0.4, 0.8, 0.9],
        }
    )


def test_classification_and_clustered_bootstrap() -> None:
    frame = _predictions()
    metrics = classification_metrics(frame["correct"], frame["probability"])
    assert metrics["auroc"] > 0.8
    intervals = clustered_bootstrap(frame, repetitions=100, seed=3)
    assert intervals["auroc"]["ci_low"] <= intervals["auroc"]["value"]
    assert intervals["auroc"]["ci_high"] >= intervals["auroc"]["value"]


def test_paired_metric_difference_aligns_run_ids() -> None:
    left = _predictions()
    right = left.copy()
    right["probability"] = [0.05, 0.1, 0.15, 0.75, 0.8, 0.3, 0.9, 0.95]
    difference = paired_clustered_metric_difference(
        left,
        right,
        metric="brier",
        repetitions=100,
        seed=4,
    )
    assert difference["value"] < 0


def test_clustered_metrics_accept_a_parameterized_operational_target() -> None:
    frame = _predictions()
    frame["needs_intervention"] = 1 - frame["correct"]
    frame["probability"] = 1 - frame["probability"]
    intervals = clustered_bootstrap(
        frame,
        repetitions=50,
        seed=9,
        target_column="needs_intervention",
    )
    assert intervals["auroc"]["value"] > 0.8
