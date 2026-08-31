from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reasonbench.storage import read_json


def _prefix_features(prefix: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(9)
    split_by_problem = {
        **{f"p{index:02d}": "train" for index in range(18)},
        **{f"p{index:02d}": "validation" for index in range(18, 24)},
        **{f"p{index:02d}": "test" for index in range(24, 32)},
    }
    for problem_id, split in split_by_problem.items():
        correct = int(problem_id[1:]) % 2 == 0
        entropy = 0.2 if correct else 0.8
        rows.append(
            {
                "run_id": f"run-{problem_id}",
                "experiment_id": "phase_04_fixture",
                "phase_id": "phase_04",
                "problem_id": problem_id,
                "research_split": split,
                "correct": correct,
                "normal_completion": True,
                "noncompletion": False,
                "wrong_completion": not correct,
                "needs_intervention": not correct,
                "parse_status": "boxed",
                "finish_reason": "eos",
                "boundary_status": "think_tag",
                "prefix_length": prefix,
                "dataset": "math",
                "model_key": "fixture_model",
                "level": int(problem_id[1:]) % 5 + 1,
                "category": "algebra",
                "problem_character_count": 80,
                "problem_token_proxy_count": 20,
                "problem_numeric_count": 2,
                "problem_operator_count": 1,
                "problem_equation_count": 1,
                "trajectory_token_count": prefix,
                "full_trajectory_token_count": 600,
                "observed_token_count": prefix,
                "assigned_reasoning_budget": np.nan,
                "normalized_entropy_mean": entropy + rng.normal(0, 0.01),
                "normalized_entropy_std": 0.1,
                "normalized_entropy_slope": 0.01 * entropy,
                "surprisal_mean": 4 * entropy,
                "top1_top2_probability_margin_mean": 1 - entropy,
                "geometry_mean_relative_velocity": entropy / 2,
                "cosine_drift_mean": entropy / 3,
                "spectral_normalized_entropy_low_energy_ratio": 1 - entropy,
            }
        )
    return pd.DataFrame(rows)


def test_early_prediction_compares_baseline_and_signal_on_held_out_problems(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "analysis"
    features_dir.mkdir()
    _prefix_features(128).to_parquet(features_dir / "features_prefix_128.parquet", index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_early_prediction.py"),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(output_dir),
            "--prefix-length",
            "128",
            "--bootstrap-repetitions",
            "20",
            "--workers",
            "1",
        ],
        cwd=project_root,
        check=True,
    )

    results = read_json(output_dir / "early_prediction_results.json")
    contrasts = read_json(output_dir / "early_primary_contrasts.json")
    summary = read_json(output_dir / "early_summary.json")
    feature_sets = {row["feature_set"] for row in results}
    test_problem_counts = {
        row["split_problem_counts"]["test"] for row in results if row["scope"] == "fixture_model"
    }

    assert {"early_baseline", "early_full", "early_spectral"} <= feature_sets
    assert test_problem_counts == {8}
    assert any(row["metric"] == "auroc" for row in contrasts)
    assert summary["technical_status"] == "passed"
    assert (output_dir / "early_failure_baseline_vs_signals.png").exists()
    assert (output_dir / "early_failure_feature_ablation.png").exists()
    assert (output_dir / "early_failure_primary_gain.png").exists()


def test_early_prediction_saves_direct_intervention_risk_for_parameterized_target(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "analysis"
    features_dir.mkdir()
    _prefix_features(128).to_parquet(features_dir / "features_prefix_128.parquet", index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_early_prediction.py"),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(output_dir),
            "--prefix-length",
            "128",
            "--target-column",
            "needs_intervention",
            "--bootstrap-repetitions",
            "20",
        ],
        cwd=project_root,
        check=True,
    )
    results = read_json(output_dir / "early_needs_intervention_prediction_results.json")
    assert {row["target_column"] for row in results} == {"needs_intervention"}
    prediction = pd.read_parquet(
        output_dir / "predictions_pooled_prefix_128_early_full.parquet"
    )
    assert "estimated_needs_intervention_risk" in prediction
    assert (output_dir / "early_needs_intervention_reliability_and_coverage.png").exists()
