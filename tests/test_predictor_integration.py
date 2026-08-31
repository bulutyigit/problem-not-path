from __future__ import annotations

import numpy as np
import pandas as pd

from reasonbench.evaluation.predictor import evaluate_one, feature_columns


def _feature_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(17)
    split_by_problem = {
        **{f"p{index:02d}": "train" for index in range(18)},
        **{f"p{index:02d}": "validation" for index in range(18, 24)},
        **{f"p{index:02d}": "test" for index in range(24, 32)},
    }
    for problem_id, split in split_by_problem.items():
        base = int(problem_id[1:]) % 2
        for seed in (11, 23):
            correct = bool(base)
            entropy = 0.25 + 0.5 * (1 - int(correct)) + rng.normal(0, 0.03)
            rows.append(
                {
                    "run_id": f"{problem_id}_{seed}",
                    "experiment_id": "fixture",
                    "phase_id": "phase_03",
                    "problem_id": problem_id,
                    "research_split": split,
                    "correct": correct,
                    "parse_status": "boxed",
                    "finish_reason": "eos",
                    "boundary_status": "think_tag",
                    "prefix_length": None,
                    "dataset": "gsm8k",
                    "model_key": "fixture_model",
                    "level": np.nan,
                    "category": None,
                    "problem_character_count": 50 + int(problem_id[1:]),
                    "problem_token_proxy_count": 10,
                    "problem_numeric_count": 2,
                    "problem_operator_count": 1,
                    "problem_equation_count": 0,
                    "trajectory_token_count": 128 + seed,
                    "generated_tokens": 140 + seed,
                    "assigned_reasoning_budget": 512,
                    "normalized_entropy_mean": entropy,
                    "normalized_entropy_std": 0.1,
                    "surprisal_mean": entropy * 4,
                    "top1_probability_mean": 1 - entropy / 2,
                    "top5_probability_mass_mean": 1 - entropy / 5,
                    "probability_tail_mass_mean": entropy / 5,
                    "effective_vocabulary_size_mean": 10 + entropy * 10,
                    "sampled_token_regret_mean": entropy / 3,
                    "successive_kl_divergence_mean": entropy / 4,
                    "successive_js_divergence_mean": entropy / 8,
                    "geometry_mean_relative_velocity": entropy / 2,
                    "spectral_normalized_entropy_low_energy_ratio": 1 - entropy,
                }
            )
    return pd.DataFrame(rows)


def test_predictor_uses_disjoint_problem_splits_and_calibration() -> None:
    result = evaluate_one(
        _feature_frame(),
        feature_set="full",
        model_name="logistic_regression",
        bootstrap_repetitions=100,
        seed=5,
    )
    assert result.calibration_applied
    assert result.metrics["auroc"]["value"] > 0.9
    assert set(result.predictions["problem_id"]) == {f"p{index:02d}" for index in range(24, 32)}


def test_constant_baseline_is_supported() -> None:
    result = evaluate_one(
        _feature_frame(),
        feature_set="constant",
        model_name="constant",
        bootstrap_repetitions=20,
        seed=5,
    )
    assert result.feature_columns == []
    assert not result.calibration_applied


def test_early_baseline_uses_only_prefix_available_context() -> None:
    frame = _feature_frame()
    frame["observed_token_count"] = 64
    columns = feature_columns(frame, "early_baseline")

    assert "model_key" in columns
    assert "observed_token_count" in columns
    assert "trajectory_token_count" not in columns
    assert "full_trajectory_token_count" not in columns


def test_transition_ablation_uses_only_distribution_shift_features() -> None:
    frame = _feature_frame()
    frame["observed_token_count"] = 64
    columns = feature_columns(frame, "early_transition")

    assert "successive_kl_divergence_mean" in columns
    assert "successive_js_divergence_mean" in columns
    assert "top1_probability_mean" not in columns
