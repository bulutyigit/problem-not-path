from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from reasonbench.storage import write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_difficulty_analysis_module():
    spec = importlib.util.spec_from_file_location(
        "difficulty_analysis",
        PROJECT_ROOT / "scripts" / "analyze_difficulty_dynamics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_prediction_handles_insufficient_terminal_failures(tmp_path: Path) -> None:
    module = _load_difficulty_analysis_module()
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "analysis"
    features_dir.mkdir()
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "full_trajectory_token_count": 64,
                "correct": True,
                "research_split": split,
            }
            for split in ("train", "train", "test", "test")
        ]
    ).to_parquet(features_dir / "features_prefix_16.parquet", index=False)

    summary, improvements, warnings = module._failure_prediction_analysis(
        features_dir,
        output_dir,
        repetitions=2,
    )

    assert summary.empty
    assert improvements.empty
    assert list(summary.columns) == list(module.FAILURE_SUMMARY_COLUMNS)
    assert list(improvements.columns) == list(module.FAILURE_IMPROVEMENT_COLUMNS)
    assert warnings == [
        "Prefix 16 lacked both terminal-failure classes in train or test; skipped."
    ]


def test_phase_02_analysis_writes_tables_figures_and_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "generation"
    feature_rows = []
    token_rows = [
        {
            "token_index": index,
            "segment": "thinking",
            "normalized_entropy": 0.1 + index * 0.01,
            "surprisal": 0.2 + index * 0.02,
            "top1_top2_probability_margin": 0.5 - index * 0.01,
            "top1_top2_logit_margin": 2.0 - index * 0.02,
            "relative_l2_step": 0.01 * index,
            "cosine_drift": 0.001 * index,
        }
        for index in range(8)
    ]
    for level in range(1, 6):
        for problem_index in range(20):
            problem_id = f"math_{level}_{problem_index}"
            for seed in (11, 23, 37, 53):
                run_id = f"run_{level}_{problem_index}_{seed}"
                correct = (problem_index + seed) % (level + 1) != 0
                trajectory = run_root / run_id
                trajectory.mkdir(parents=True)
                write_json_atomic(
                    trajectory / "metadata.json",
                    {
                        "run_id": run_id,
                        "model_key": "gemma4_e4b",
                        "problem_id": problem_id,
                        "seed": seed,
                        "level": level,
                        "category": "algebra",
                        "verification": {"correct": correct},
                    },
                )
                pd.DataFrame(token_rows).to_parquet(
                    trajectory / "token_metrics.parquet",
                    index=False,
                )
                write_json_atomic(trajectory / "complete.json", {"complete": True})
                feature_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": "math",
                        "model_key": "gemma4_e4b",
                        "problem_id": problem_id,
                        "research_split": (
                            "train"
                            if problem_index < 12
                            else "validation"
                            if problem_index < 16
                            else "test"
                        ),
                        "seed": seed,
                        "level": level,
                        "category": "algebra",
                        "correct": correct,
                        "trajectory_token_count": 100 + level,
                        "full_trajectory_token_count": 2048 + level,
                        "observed_token_count": 100 + level,
                        "elapsed_seconds": 2.0 + level,
                        "normalized_entropy_mean": 0.1 * level,
                        "normalized_entropy_slope": 0.001 * level,
                        "surprisal_mean": 0.2 * level,
                        "top1_top2_probability_margin_mean": 0.5 / level,
                        "geometry_mean_relative_velocity": 0.01 * level,
                        "geometry_mean_cosine_drift": 0.001 * level,
                        "geometry_trajectory_efficiency": 1.0 / level,
                        "geometry_normalized_path_length": 0.5 * level,
                        "geometry_velocity_variance": 0.002 * level,
                        "geometry_cosine_drift_variance": 0.0002 * level,
                        "geometry_turning_angle_mean": 0.1 * level,
                        "geometry_turning_angle_variance": 0.01 * level,
                        "spectral_normalized_entropy_entropy": 0.2 * level,
                        "spectral_normalized_entropy_low_energy_ratio": 1.0 / level,
                        "spectral_normalized_entropy_high_energy_ratio": 0.05 * level,
                        "spectral_surprisal_entropy": 0.15 * level,
                        "spectral_surprisal_high_energy_ratio": 0.03 * level,
                        "spectral_top1_top2_logit_margin_entropy": 0.12 * level,
                        "spectral_relative_l2_step_entropy": 0.1 * level,
                        "spectral_relative_l2_step_high_energy_ratio": 0.02 * level,
                        "finish_reason": "eos",
                    }
                )
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    features = features_dir / "features_full.parquet"
    feature_frame = pd.DataFrame(feature_rows)
    feature_frame.to_parquet(features, index=False)
    for prefix in (16, 32, 64, 128, 256, 512, 1024, 2048):
        prefix_frame = feature_frame.copy()
        prefix_frame["trajectory_token_count"] = prefix
        prefix_frame["observed_token_count"] = prefix
        prefix_frame["prefix_length"] = prefix
        prefix_frame.to_parquet(
            features_dir / f"features_prefix_{prefix}.parquet",
            index=False,
        )
    output = tmp_path / "analysis"
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_difficulty_dynamics.py"),
            "--features",
            str(features),
            "--features-dir",
            str(features_dir),
            "--run-dir",
            str(run_root),
            "--output-dir",
            str(output),
            "--bootstrap-repetitions",
            "10",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for filename in (
        "phase_summary.json",
        "difficulty_design_validation.json",
        "difficulty_metric_summary.parquet",
        "token_dynamics_summary.parquet",
        "difficulty_outcomes.png",
        "token_dynamics.png",
        "entropy_spike_timing.png",
        "failure_prediction_by_prefix.parquet",
        "failure_prediction_by_prefix.png",
        "geometry_spectral_correlations.parquet",
        "geometry_spectral_correlations.png",
        "seed_instability_associations.parquet",
        "seed_instability_associations.png",
        "hypothesis_freeze.json",
        "reasoning_onset_uncertainty_associations.parquet",
        "reasoning_onset_problem_features.parquet",
        "reasoning_onset_uncertainty_associations.png",
    ):
        assert (output / filename).exists()
    onset = pd.read_parquet(output / "reasoning_onset_uncertainty_associations.parquet")
    assert set(onset["target"]) == {"failure_rate", "level"}
    assert set(onset["prefix_length"]) == {16, 32, 64, 128}
