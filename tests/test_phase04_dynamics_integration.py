from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reasonbench.storage import read_json

PREFIXES = (16, 32, 64, 128, 256, 512)


def _frame(prefix: int) -> pd.DataFrame:
    rows = []
    lengths = (700, 1500, 2500, 5000, 9000, 16384)
    for index in range(42):
        total = lengths[index % len(lengths)]
        correct = index % 2 == 0
        split = "train" if index < 24 else "validation" if index < 30 else "test"
        signal = np.log1p(total) / 10 + (0.15 if not correct else 0)
        movement = np.log2(prefix / 16 + 1) / 10
        rows.append(
            {
                "run_id": f"run-{index}",
                "experiment_id": "fixture",
                "phase_id": "phase_04",
                "problem_id": f"problem-{index}",
                "research_split": split,
                "correct": correct,
                "parse_status": "boxed",
                "finish_reason": "max_new_tokens" if total == 16384 else "eos",
                "boundary_status": "think_tag",
                "prefix_length": prefix,
                "dataset": "math",
                "model_key": "fixture_model",
                "level": index % 5 + 1,
                "category": "algebra" if index % 2 else "geometry",
                "problem_character_count": 80 + index,
                "problem_token_proxy_count": 20 + index % 5,
                "problem_numeric_count": 2,
                "problem_operator_count": 1,
                "problem_equation_count": 1,
                "trajectory_token_count": prefix,
                "full_trajectory_token_count": total,
                "observed_token_count": prefix,
                "assigned_reasoning_budget": np.nan,
                "normalized_entropy_mean": signal + movement,
                "normalized_entropy_std": 0.1 + movement,
                "normalized_entropy_slope": movement,
                "normalized_entropy_max_rise": signal / 5 + movement,
                "surprisal_mean": 2 * signal + movement,
                "top1_top2_probability_margin_mean": 1 - signal / 2 - movement,
                "geometry_mean_relative_velocity": signal / 3 + movement,
                "geometry_mean_cosine_drift": signal / 4 + movement,
                "spectral_normalized_entropy_entropy": signal / 5 + movement,
                "spectral_normalized_entropy_high_energy_ratio": signal / 6 + movement,
                "spectral_surprisal_entropy": signal / 7 + movement,
            }
        )
    return pd.DataFrame(rows)


def test_phase04_length_and_dynamics_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    features = tmp_path / "features"
    analysis = tmp_path / "analysis"
    features.mkdir()
    for prefix in PREFIXES:
        _frame(prefix).to_parquet(features / f"features_prefix_{prefix}.parquet", index=False)

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_early_length.py"),
            "--features-dir",
            str(features),
            "--output-dir",
            str(analysis),
            "--threshold",
            "1024",
            "--threshold",
            "4096",
            "--bootstrap-repetitions",
            "10",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_phase04_dynamics.py"),
            "--features-dir",
            str(features),
            "--output-dir",
            str(analysis),
            "--bootstrap-repetitions",
            "10",
        ],
        cwd=root,
        check=True,
    )

    assert read_json(analysis / "length_prediction_summary.json")["technical_status"] == "passed"
    assert read_json(analysis / "dynamics_summary.json")["metrics"]["prefixes"][-1] == 512
    assert (analysis / "early_length_prediction_auroc.png").exists()
    assert (analysis / "phase04_feature_movement.png").exists()
    assert (analysis / "phase04_feature_movement_at_risk.png").exists()
    assert (analysis / "phase04_uncertainty_evolution_by_correctness.png").exists()
    assert (analysis / "phase04_spectral_evolution_by_correctness.png").exists()
    assert (analysis / "phase04_spectral_correctness_separation.png").exists()
