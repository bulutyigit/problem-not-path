from __future__ import annotations

import numpy as np
import pandas as pd

from reasonbench.evaluation.failure_dynamics import (
    bootstrap_residualized_associations,
    problem_seed_instability,
    residualized_cross_block_correlations,
)
from reasonbench.evaluation.predictor import feature_columns


def _trajectory_frame() -> pd.DataFrame:
    rows = []
    for problem_index in range(20):
        level = problem_index % 5 + 1
        for seed_index, seed in enumerate((11, 23, 37, 53)):
            mixed = problem_index % 2 == 0
            correct = seed_index < 2 if mixed else True
            rows.append(
                {
                    "problem_id": f"math_{problem_index}",
                    "seed": seed,
                    "correct": correct,
                    "level": level,
                    "category": "algebra" if problem_index % 3 else "geometry",
                    "trajectory_token_count": 100 + level,
                    "observed_token_count": 128,
                    "normalized_entropy_mean": 0.1 * level + 0.2 * mixed,
                    "geometry_trajectory_efficiency": 1.0 - 0.2 * mixed,
                    "spectral_normalized_entropy_entropy": 0.1 + 0.3 * mixed,
                }
            )
    return pd.DataFrame(rows)


def test_problem_seed_instability_is_maximal_for_two_of_four_successes() -> None:
    problem_table = problem_seed_instability(
        _trajectory_frame(),
        [
            "trajectory_token_count",
            "geometry_trajectory_efficiency",
            "spectral_normalized_entropy_entropy",
        ],
    )

    assert set(problem_table["seed_instability"].round(6)) == {0.0, 1.0}
    assert (problem_table.loc[problem_table["correct_seeds"] == 2, "seed_instability"] == 1).all()


def test_controlled_geometry_spectral_and_instability_associations_are_finite() -> None:
    problem_table = problem_seed_instability(
        _trajectory_frame(),
        [
            "trajectory_token_count",
            "geometry_trajectory_efficiency",
            "spectral_normalized_entropy_entropy",
        ],
    )
    associations = bootstrap_residualized_associations(
        problem_table,
        "seed_instability",
        ["mean__geometry_trajectory_efficiency"],
        repetitions=20,
        seed=7,
    )
    matrix = residualized_cross_block_correlations(
        problem_table,
        ["mean__geometry_trajectory_efficiency"],
        ["mean__spectral_normalized_entropy_entropy"],
    )

    assert np.isfinite(associations.loc[0, "association"])
    assert np.isfinite(matrix.iloc[0, 0])


def test_early_feature_blocks_preserve_the_baseline_and_separate_modalities() -> None:
    frame = _trajectory_frame()
    baseline = set(feature_columns(frame, "early_baseline"))
    geometry = set(feature_columns(frame, "early_geometry"))
    spectral = set(feature_columns(frame, "early_spectral"))
    full = set(feature_columns(frame, "early_full"))

    assert {"level", "category", "observed_token_count"} <= baseline
    assert "geometry_trajectory_efficiency" in geometry - baseline
    assert "spectral_normalized_entropy_entropy" in spectral - baseline
    assert geometry | spectral <= full


def test_early_blocks_is_low_dimensional_and_fails_closed_on_missing_columns() -> None:
    import pandas as pd

    from reasonbench.evaluation.predictor import EARLY_BLOCKS_SUMMARY_COLUMNS

    frame = _trajectory_frame()
    for column in EARLY_BLOCKS_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.5
    columns = feature_columns(frame, "early_blocks")

    assert set(EARLY_BLOCKS_SUMMARY_COLUMNS) <= set(columns)
    assert len(columns) <= 25
    assert len(columns) < len(feature_columns(frame, "early_full"))

    broken = frame.drop(columns=[EARLY_BLOCKS_SUMMARY_COLUMNS[0]])
    try:
        feature_columns(broken, "early_blocks")
    except ValueError as error:
        assert EARLY_BLOCKS_SUMMARY_COLUMNS[0] in str(error)
    else:  # pragma: no cover - guard
        raise AssertionError("early_blocks must fail closed when frozen columns are missing")
