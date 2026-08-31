from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from reasonbench.datasets.loader import ProblemRecord
from reasonbench.datasets.splits import write_problem_bundle
from reasonbench.evaluation.breakthrough import (
    AnchorProbe,
    BreakthroughLabel,
    build_longitudinal_tables,
    derive_breakthrough_label,
    horizon_outcome,
)
from reasonbench.evaluation.breakthrough_controller import (
    ARMS,
    CONTROLLER_FEATURES,
    BreakthroughAwareController,
    artifact_digest,
    fit_arm_models,
    select_compute_penalty,
    verify_artifact_digest,
)
from reasonbench.evaluation.predictor import EARLY_BLOCKS_SUMMARY_COLUMNS
from reasonbench.evaluation.compute_extension import (
    UNCERTAINTY_BLOCKS,
    UNCERTAINTY_FEATURES,
    UNCERTAINTY_SCORE_VERSION,
    UNCERTAINTY_SIGNS,
    UNCERTAINTY_TRANSFORMS,
    assign_balanced_uncertainty_strata,
    fit_percentile_references,
    paired_budget_effects,
    score_uncertainty_components,
    score_uncertainty_rows,
    validate_compute_extension_protocol,
)
from reasonbench.storage import read_json, sha256_file, write_json_atomic
from scripts.build_breakthrough_probe_manifest import _balanced_problem_ids
from scripts.generate_breakthrough_probes import _branch_seed
from scripts.validate_uncertainty_extensions import _nested_token_path_flags


def test_breakthrough_requires_adjacent_stability() -> None:
    probes = [
        AnchorProbe(64, 1, 4),
        AnchorProbe(128, 3, 4),
        AnchorProbe(256, 2, 4),
        AnchorProbe(512, 4, 4),
        AnchorProbe(1024, 3, 4),
    ]

    label = derive_breakthrough_label(probes, threshold=0.75)

    assert label.event_observed
    assert label.interval_lower == 256
    assert label.interval_upper == 512
    assert label.stable_anchor == 512
    assert label.stability_anchor == 1024


def test_breakthrough_without_stable_pair_is_right_censored() -> None:
    label = derive_breakthrough_label(
        [AnchorProbe(64, 0, 4), AnchorProbe(128, 3, 4)],
        threshold=0.75,
    )

    assert not label.event_observed
    assert label.interval_upper is None
    assert label.censoring_time == 128


def test_probe_cohort_selection_is_balanced_and_outcome_agnostic() -> None:
    records = [
        {
            "problem_id": f"level_{level}_{index}",
            "level": level,
            # These values must have no effect because the selection helper
            # receives only protocol-level identity and difficulty fields.
            "correct": index % 2 == 0,
            "generated_tokens": index * 100,
        }
        for level in range(1, 6)
        for index in range(10)
    ]

    selected = _balanced_problem_ids(records, problem_count=20, selection_seed=7)

    assert len(selected) == 20
    assert {
        level: sum(problem_id.startswith(f"level_{level}_") for problem_id in selected)
        for level in range(1, 6)
    } == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}


def test_uncertainty_index_uses_training_ecdfs_and_balanced_rank_strata() -> None:
    rows = []
    for index in range(8):
        row = {
            "run_id": f"run_{index}",
            "model_key": "gemma4_e4b_mlx_4bit",
            "research_split": "train" if index < 6 else "test",
            "level": 3,
        }
        for feature_index, feature in enumerate(UNCERTAINTY_FEATURES):
            increasing = 0.1 + feature_index * 0.01 + index * 0.005
            if UNCERTAINTY_SIGNS[feature] < 0:
                increasing = 1.0 - increasing
            if UNCERTAINTY_TRANSFORMS[feature] == "absolute":
                increasing = (-1.0 if index % 2 else 1.0) * increasing
            row[feature] = increasing
        rows.append(row)
    frame = pd.DataFrame(rows)
    references = fit_percentile_references(frame)
    components = score_uncertainty_components(frame, references)
    frame["uncertainty_score"] = score_uncertainty_rows(frame, references)
    frame["uncertainty_stratum"] = assign_balanced_uncertainty_strata(frame)

    assert frame.loc[7, "uncertainty_score"] > frame.loc[0, "uncertainty_score"]
    assert frame["uncertainty_score"].between(0.0, 1.0).all()
    assert set(frame["uncertainty_stratum"]) == {"low", "high"}
    assert (frame["uncertainty_stratum"] == "low").sum() == 4
    assert (frame["uncertainty_stratum"] == "high").sum() == 4
    assert set(references["gemma4_e4b_mlx_4bit"]) == set(UNCERTAINTY_FEATURES)
    block_columns = [f"uncertainty_block__{block}" for block in UNCERTAINTY_BLOCKS]
    pd.testing.assert_series_equal(
        components[block_columns].mean(axis=1),
        components["uncertainty_score"],
        check_names=False,
    )
    for block, features in UNCERTAINTY_BLOCKS.items():
        feature_columns = [f"uncertainty_feature_percentile__{feature}" for feature in features]
        pd.testing.assert_series_equal(
            components[feature_columns].mean(axis=1),
            components[f"uncertainty_block__{block}"],
            check_names=False,
        )


def test_paired_budget_effects_reports_high_minus_low_interaction() -> None:
    frame = pd.DataFrame(
        [
            {
                "problem_id": "p1",
                "source_run_id": "r1",
                "branch_index": 0,
                "uncertainty_stratum": "high",
                "short_correct": False,
                "medium_correct": False,
                "long_correct": True,
            },
            {
                "problem_id": "p2",
                "source_run_id": "r2",
                "branch_index": 0,
                "uncertainty_stratum": "low",
                "short_correct": False,
                "medium_correct": False,
                "long_correct": False,
            },
        ]
    )

    summary = paired_budget_effects(frame)

    assert summary["strata"]["high"]["contrasts"]["long_minus_medium"] == 1.0
    assert summary["strata"]["low"]["contrasts"]["long_minus_medium"] == 0.0
    assert summary["interactions_high_minus_low"]["long_minus_medium"] == 1.0


def test_nested_token_path_check_detects_divergent_medium_arm() -> None:
    by_arm = {
        "short": {
            "continuation_token_ids": [1, 2, 90],
            "reasoning_continuation_token_count": 2,
        },
        "medium": {
            "continuation_token_ids": [1, 8, 3, 90],
            "reasoning_continuation_token_count": 3,
        },
        "long": {
            "continuation_token_ids": [1, 8, 3, 4, 90],
            "reasoning_continuation_token_count": 4,
        },
    }

    short_nested, medium_nested = _nested_token_path_flags(by_arm)

    assert not short_nested
    assert medium_nested


def test_horizon_outcome_preserves_interval_censoring() -> None:
    label = BreakthroughLabel(True, 256, 512, 512, 1024, 512, 1024)

    assert horizon_outcome(label, prefix=128, horizon=128) == 0
    assert horizon_outcome(label, prefix=128, horizon=512) == 1
    assert horizon_outcome(label, prefix=384, horizon=256) is None
    assert horizon_outcome(label, prefix=512, horizon=128) is None


def _feature_row(prefix: int) -> dict:
    return {
        "run_id": "run_1",
        "problem_id": "math_1",
        "model_key": "gemma4_e4b",
        "dataset": "math",
        "research_split": "train",
        "seed": 11,
        "prefix_length": prefix,
        "observed_token_count": prefix,
        "correct": True,
        "normalized_entropy_mean": 0.4,
    }


def test_longitudinal_tables_never_treat_ambiguous_interval_as_negative(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features"
    features.mkdir()
    for prefix in (128, 256, 512):
        pd.DataFrame([_feature_row(prefix)]).to_parquet(
            features / f"features_prefix_{prefix}.parquet", index=False
        )
    labels = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "event_observed": True,
                "interval_lower": 256,
                "interval_upper": 512,
                "event_time_proxy": 512,
                "censoring_time": 1024,
                "stable_anchor": 512,
                "stability_anchor": 1024,
            }
        ]
    )

    horizon, hazard = build_longitudinal_tables(features, labels, horizons=(128, 512))

    small = horizon[horizon["forecast_horizon"] == 128]
    assert list(small["forecast_token"]) == [128]
    assert list(small["breakthrough_within_horizon"]) == [0]
    large = horizon[horizon["forecast_horizon"] == 512]
    assert list(large["forecast_token"]) == [128, 256]
    assert set(large["breakthrough_within_horizon"]) == {1}
    assert hazard.groupby("run_id")["breakthrough_in_bin"].sum().to_dict() == {"run_1": 1}
    assert hazard.loc[hazard["breakthrough_in_bin"] == 1, "forecast_token"].tolist() == [256]
    assert 512 not in set(hazard["forecast_token"])


def test_probe_validator_checks_exact_branch_keys_and_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "probe_manifest.json"
    write_json_atomic(
        manifest,
        {
            "trajectories": [{"run_id": "run_1"}],
            "pilot_run_ids": ["run_1"],
        },
    )
    manifest_sha = sha256_file(manifest)
    probe_root = tmp_path / "probes"
    summary = probe_root / "run_1" / "trajectory_probe_summary.json"
    write_json_atomic(
        summary,
        {
            "run_id": "run_1",
            "problem_id": "math_1",
            "model_key": "gemma4_e4b",
            "dataset": "math",
            "research_split": "test",
            "level": 1,
            "category": "algebra",
            "seed": 11,
            "event_observed": False,
            "interval_lower": 64,
            "interval_upper": None,
            "event_time_proxy": None,
            "censoring_time": 64,
            "stable_anchor": None,
            "stability_anchor": None,
            "probe_manifest_sha256": manifest_sha,
            "probes": [{"anchor": 64, "successes": 1, "continuations": 2, "success_rate": 0.5}],
        },
    )
    for branch_index in range(2):
        branch = probe_root / "run_1" / "anchor_64" / f"branch_{branch_index:02d}"
        result_path = branch / "result.json"
        write_json_atomic(
            result_path,
            {
                "source_run_id": "run_1",
                "anchor": 64,
                "branch_index": branch_index,
                "probe_manifest_sha256": manifest_sha,
            },
        )
        write_json_atomic(
            branch / "branch_complete.json",
            {
                "result_sha256": sha256_file(result_path),
                "result_size_bytes": result_path.stat().st_size,
            },
        )
    output = tmp_path / "validated"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_breakthrough_probes.py"),
            "--probe-dir",
            str(probe_root),
            "--probe-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--pilot-only",
        ],
        cwd=root,
        check=True,
    )

    validation = read_json(output / "breakthrough_probe_validation.json")
    assert validation["valid"]
    assert validation["expected_branches"] == 2
    assert validation["missing_branches"] == 0
    assert (output / "breakthrough_label_sensitivity.parquet").exists()


def test_uncertainty_extension_validator_requires_exact_paired_arms(
    tmp_path: Path,
) -> None:
    records = []
    for index, stratum in enumerate(("low", "high")):
        records.append(
            {
                "run_id": f"run_{index}",
                "problem_id": f"problem_{index}",
                "model_key": "gemma4_e4b_mlx_4bit",
                "level": index + 1,
                "research_split": "test",
                "generated_prefix_sha256": f"prefix_{index}",
                "generated_prefix_token_count": 512,
                "uncertainty_score": float(index),
                "uncertainty_stratum": stratum,
                "eligible": True,
            }
        )
    manifest_payload = {
        "source_probe_manifest_sha256": "probe_sha",
        "protocol": {
            "primary_anchor": 512,
            "continuations_per_arm": 4,
            "max_total_generated_tokens": 25600,
            "nominal_prefix_and_boundary_overhead_reserve": 512,
            "final_answer_reserve": 512,
            "budget_semantics": "target_total_reasoning_tokens_including_anchor",
            "paired_branch_seeds": True,
            "nested_token_paths_required": True,
            "arms": {
                "short": {
                    "target_total_reasoning_tokens": 1024,
                    "reasoning_continuation_budget": 512,
                },
                "medium": {
                    "target_total_reasoning_tokens": 4096,
                    "reasoning_continuation_budget": 3584,
                },
                "long": {
                    "target_total_reasoning_tokens": 24576,
                    "reasoning_continuation_budget": 24064,
                },
            },
        },
        "uncertainty_score": {"version": UNCERTAINTY_SCORE_VERSION},
        "records": records,
        "eligible_run_ids": [record["run_id"] for record in records],
        "pilot_eligible_run_ids": [record["run_id"] for record in records],
    }
    canonical = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_payload["extension_digest"] = hashlib.sha256(canonical).hexdigest()
    manifest = tmp_path / "extension_manifest.json"
    write_json_atomic(manifest, manifest_payload)
    manifest_sha = sha256_file(manifest)
    extension_root = tmp_path / "extensions"
    for record in records:
        arm_payloads = {
            "short": (1024, 512, [101, 102]),
            "medium": (4096, 3584, [101, 102, 103]),
            "long": (24576, 24064, [101, 102, 103, 104]),
        }
        for branch_index in range(4):
            for arm, (target, budget, reasoning_ids) in arm_payloads.items():
                branch = extension_root / record["run_id"] / arm / f"branch_{branch_index:02d}"
                result_path = branch / "result.json"
                write_json_atomic(
                    result_path,
                    {
                        "source_run_id": record["run_id"],
                        "problem_id": record["problem_id"],
                        "model_key": record["model_key"],
                        "level": record["level"],
                        "research_split": "test",
                        "anchor": 512,
                        "budget_arm": arm,
                        "branch_index": branch_index,
                        "branch_seed": _branch_seed(record["run_id"], 512, branch_index),
                        "generated_prefix_sha256": record["generated_prefix_sha256"],
                        "generated_prefix_token_count": 512,
                        "reasoning_continuation_budget": budget,
                        "target_total_reasoning_tokens": target,
                        "budget_semantics": ("target_total_reasoning_tokens_including_anchor"),
                        "final_answer_reserve": 512,
                        "max_total_generated_tokens": 25600,
                        "extension_manifest_sha256": manifest_sha,
                        "uncertainty_score_version": UNCERTAINTY_SCORE_VERSION,
                        "uncertainty_score": record["uncertainty_score"],
                        "uncertainty_stratum": record["uncertainty_stratum"],
                        "continuation_token_count": len(reasoning_ids),
                        "continuation_token_ids": reasoning_ids,
                        "reasoning_continuation_token_count": len(reasoning_ids),
                        "finish_reason": "eos",
                        "verification": {
                            "correct": arm == "long" and record["uncertainty_stratum"] == "high"
                        },
                        "reused_short_branch": arm == "short",
                    },
                )
                write_json_atomic(
                    branch / "branch_complete.json",
                    {
                        "result_sha256": sha256_file(result_path),
                        "result_size_bytes": result_path.stat().st_size,
                    },
                )
    output = tmp_path / "validated_extensions"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_uncertainty_extensions.py"),
            "--extension-dir",
            str(extension_root),
            "--extension-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--pilot-only",
            "--bootstrap-repetitions",
            "10",
        ],
        cwd=root,
        check=True,
    )

    validation = read_json(output / "uncertainty_extension_validation.json")
    assert validation["valid"]
    assert validation["inferential_claim_allowed"] is False
    assert validation["nested_token_path_mismatches"] == 0
    assert validation["effects"]["interactions_high_minus_low"]["long_minus_medium"] == 1.0


def test_breakthrough_forecast_evaluator_uses_grouped_logistic_models(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(30):
        split = "train" if index < 18 else "validation" if index < 24 else "test"
        for prefix in (64, 128):
            rows.append(
                {
                    "run_id": f"run_{index}",
                    "problem_id": f"problem_{index}",
                    "dataset": "math",
                    "model_key": "gemma4_e4b" if index % 2 else "qwen35_4b",
                    "research_split": split,
                    "correct": index % 2 == 0,
                    "forecast_token": prefix,
                    "forecast_time_bin": f"t_{prefix}",
                    "time_log1p": 4.0,
                    "observed_token_count": prefix,
                    "level": index % 5 + 1,
                    "category": "algebra",
                    "problem_character_count": 100,
                    "problem_token_proxy_count": 20,
                    "problem_numeric_count": 2,
                    "problem_operator_count": 1,
                    "problem_equation_count": 1,
                    "normalized_entropy_mean": 0.2 + 0.1 * (index % 2),
                    "breakthrough_within_horizon": (index + prefix // 64) % 2,
                    "breakthrough_in_bin": (index + prefix // 64) % 2,
                    "forecast_horizon": 256,
                    "hazard_interval_width": 64,
                }
            )
    frame = pd.DataFrame(rows)
    eventual = tmp_path / "eventual.parquet"
    horizon = tmp_path / "horizon.parquet"
    hazard = tmp_path / "hazard.parquet"
    frame.to_parquet(eventual, index=False)
    frame.to_parquet(horizon, index=False)
    frame.to_parquet(hazard, index=False)
    output = tmp_path / "forecast"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_breakthrough_forecasts.py"),
            "--eventual-success-table",
            str(eventual),
            "--horizon-table",
            str(horizon),
            "--hazard-table",
            str(hazard),
            "--output-dir",
            str(output),
            "--bootstrap-repetitions",
            "10",
        ],
        cwd=root,
        check=True,
    )

    summary = read_json(output / "breakthrough_forecast_summary.json")
    assert summary["technical_status"] == "passed"
    assert summary["row_level_random_split_used"] is False
    assert summary["sequence_encoder_included"] is False
    assert any(
        result.get("analysis") == "breakthrough_horizon_256"
        and result.get("feature_set") == "early_full"
        for result in summary["results"]
    )


def _phase05_response_rows() -> pd.DataFrame:
    rows = []
    for index in range(24):
        high_breakthrough = index % 2 == 0
        rows.append(
            {
                "problem_id": f"p{index // 2}",
                "source_run_id": f"r{index}",
                "model_key": (
                    "gemma4_e4b_mlx_4bit" if index % 3 else "ministral3_3b_mlx_4bit"
                ),
                "level": index % 6 + 1,
                "uncertainty_score": 0.8 if high_breakthrough else 0.2,
                "eventual_success_probability": 0.7 if index % 4 else 0.3,
                "breakthrough_probability_within_512": 0.8 if high_breakthrough else 0.2,
                "short_correct": not high_breakthrough,
                "medium_correct": index % 7 != 0,
                "long_correct": index % 11 != 0,
            }
        )
    return pd.DataFrame(rows)


def test_phase05_controller_uses_forecasts_and_penalizes_compute() -> None:
    frame = _phase05_response_rows()
    models = fit_arm_models(frame, seed=5)
    controller = BreakthroughAwareController(
        arm_models=models,
        token_costs={"short": 1024.0, "medium": 4096.0, "long": 24576.0},
        compute_penalty=0.2,
    )

    decisions = controller.choose(frame)

    assert set(decisions["selected_arm"]).issubset(ARMS)
    assert set(CONTROLLER_FEATURES).issubset(frame.columns)
    probabilities = decisions.filter(like="predicted_correct_").to_numpy(float)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_phase05_penalty_selection_and_digest_are_deterministic() -> None:
    frame = _phase05_response_rows()
    # Aligned predictions keep at least one grid policy inside the frozen
    # accuracy gap; the deliberately infeasible case is covered by the
    # fail-closed test below.
    predictions = pd.DataFrame(
        {
            "predicted_correct_short": [0.9 if value else 0.2 for value in frame["short_correct"]],
            "predicted_correct_medium": 0.8,
            "predicted_correct_long": 0.81,
        }
    )
    penalty, grid = select_compute_penalty(
        frame,
        predictions,
        token_costs={"short": 1024.0, "medium": 4096.0, "long": 24576.0},
        maximum_accuracy_gap=0.05,
    )
    payload = {"schema_version": "test", "compute_penalty": penalty}
    payload["artifact_digest"] = artifact_digest(payload)

    assert 0.0 <= penalty <= 1.0
    assert len(grid) == 101
    assert verify_artifact_digest(payload)
    payload["compute_penalty"] = penalty + 0.01
    assert not verify_artifact_digest(payload)


def test_phase05_extension_protocol_requires_equal_answer_reserve() -> None:
    protocol = {
        "protocol_schema_version": "phase05_breakthrough_controller_v1",
        "primary_anchor": 512,
        "continuations_per_arm": 1,
        "max_total_generated_tokens": 29696,
        "final_answer_reserve": 4096,
        "budget_semantics": "target_total_reasoning_tokens_including_anchor",
        "paired_branch_seeds": True,
        "nested_token_paths_required": True,
        "arms": {
            "short": {
                "target_total_reasoning_tokens": 1024,
                "reasoning_continuation_budget": 512,
            },
            "medium": {
                "target_total_reasoning_tokens": 4096,
                "reasoning_continuation_budget": 3584,
            },
            "long": {
                "target_total_reasoning_tokens": 24576,
                "reasoning_continuation_budget": 24064,
            },
        },
    }

    validate_compute_extension_protocol(protocol)
    protocol["final_answer_reserve"] = 512
    with pytest.raises(ValueError, match="4,096-token"):
        validate_compute_extension_protocol(protocol)


def test_phase05_fit_script_freezes_crossfit_controller(tmp_path: Path) -> None:
    feature_rows = []
    horizon_rows = []
    eventual_rows = []
    pair_rows = []
    model_keys = ("gemma4_e4b_mlx_4bit", "ministral3_3b_mlx_4bit")
    for problem_index in range(20):
        for model_index, model_key in enumerate(model_keys):
            run_id = f"run_{problem_index}_{model_index}"
            base = {
                "run_id": run_id,
                "problem_id": f"problem_{problem_index}",
                "dataset": "math",
                "model_key": model_key,
                "research_split": "train",
                "correct": (problem_index + model_index) % 2 == 0,
                "forecast_token": 512,
                "forecast_time_bin": "t_512",
                "time_log1p": 6.24,
                "observed_token_count": 512,
                "level": problem_index % 5 + 1,
                "category": "algebra",
                "normalized_entropy_mean": 0.1 + 0.02 * ((problem_index + model_index) % 5),
                "normalized_entropy_std": 0.03 + 0.01 * (problem_index % 3),
                **{
                    column: 0.1 + 0.03 * ((problem_index + model_index + offset) % 7)
                    for offset, column in enumerate(EARLY_BLOCKS_SUMMARY_COLUMNS)
                },
            }
            feature_rows.append(base)
            eventual_rows.append(base)
            horizon_rows.append(
                {
                    **base,
                    "forecast_horizon": 512,
                    "breakthrough_within_horizon": (problem_index + model_index) % 2,
                }
            )
            for branch_index in range(2):
                pair_rows.append(
                    {
                        "problem_id": base["problem_id"],
                        "source_run_id": run_id,
                        "model_key": model_key,
                        "level": base["level"],
                        "branch_index": branch_index,
                        "uncertainty_score": 0.2 + 0.6 * ((problem_index % 4) / 3),
                        "short_correct": (problem_index + branch_index) % 3 == 0,
                        "medium_correct": (problem_index + branch_index) % 4 != 0,
                        "long_correct": (problem_index + branch_index) % 7 != 0,
                        "short_total_generated_tokens": 1500,
                        "medium_total_generated_tokens": 5000,
                        "long_total_generated_tokens": 25000,
                    }
                )
    validation = tmp_path / "validation.json"
    write_json_atomic(
        validation,
        {"valid": True, "stage": "labeling_cohort", "completed_trajectories": 40},
    )
    paths = {
        "horizon": tmp_path / "horizon.parquet",
        "eventual": tmp_path / "eventual.parquet",
        "features": tmp_path / "features.parquet",
        "pairs": tmp_path / "pairs.parquet",
    }
    pd.DataFrame(horizon_rows).to_parquet(paths["horizon"], index=False)
    pd.DataFrame(eventual_rows).to_parquet(paths["eventual"], index=False)
    pd.DataFrame(feature_rows).to_parquet(paths["features"], index=False)
    pd.DataFrame(pair_rows).to_parquet(paths["pairs"], index=False)
    output = tmp_path / "policy"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "fit_phase05_breakthrough_controller.py"),
            "--probe-validation",
            str(validation),
            "--horizon-table",
            str(paths["horizon"]),
            "--eventual-success-table",
            str(paths["eventual"]),
            "--prefix-features",
            str(paths["features"]),
            "--development-pairs",
            str(paths["pairs"]),
            "--output-dir",
            str(output),
            "--folds",
            "2",
            # Synthetic arm outcomes are weakly predictable; a wide gap keeps
            # a feasible policy on the grid so the freeze mechanics under test
            # can run. Infeasibility itself is covered by the dedicated
            # fail-closed unit test.
            "--maximum-accuracy-gap",
            "0.4",
        ],
        cwd=root,
        check=True,
    )

    policy = read_json(output / "phase05_frozen_policy.json")
    assert policy["status"] == "frozen_for_external_evaluation"
    assert policy["harp_outcomes_opened"] is False
    assert verify_artifact_digest(policy)
    assert (output / "phase05_controller.joblib").exists()


def test_phase05_harp_preparation_balances_levels_and_removes_math_overlap(
    tmp_path: Path,
) -> None:
    math_record = ProblemRecord(
        problem_id="math_1",
        dataset="math",
        source_repository="test",
        source_split="test",
        source_index=0,
        problem="What is one plus one?",
        reference_answer="2",
        reference_solution="1+1=2",
        level=1,
        category="Algebra",
        research_split="train",
    )
    math_path, _ = write_problem_bundle([math_record], tmp_path / "math", "math_sample")
    harp_rows = [
        {
            "problem": "What is one plus one?",
            "answer": "2",
            "solution_0": "2",
            "year": 2000,
            "contest": "AMC",
            "number": 1,
            "level": 1,
            "subject": "algebra",
        }
    ]
    for level in range(1, 7):
        for index in range(3):
            harp_rows.append(
                {
                    "problem": f"Unique HARP level {level} problem {index}: compute {level}+{index}.",
                    "answer": str(level + index),
                    "solution_0": f"The result is {level + index}.",
                    "year": 2001 + level,
                    "contest": "AIME",
                    "number": index + 1,
                    "level": level,
                    "subject": "number_theory",
                }
            )
    archive = tmp_path / "HARP.jsonl.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "HARP.jsonl",
            "\n".join(json.dumps(row) for row in harp_rows) + "\n",
        )
    output = tmp_path / "harp"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "prepare_phase05_harp.py"),
            "--harp-jsonl-or-zip",
            str(archive),
            "--math-bundle",
            str(math_path),
            "--output-dir",
            str(output),
            "--sample-size",
            "12",
        ],
        cwd=root,
        check=True,
    )

    manifest = read_json(output / "dataset_manifest.json")
    assert manifest["dataset"] == "harp"
    assert manifest["sample_size"] == 12
    assert set(manifest["levels"].values()) == {2}
    overlap = read_json(output / "math_overlap_audit.json")
    assert overlap["exact_or_near_duplicates_removed"] == 1


def test_phase05_penalty_selection_fails_closed_when_no_policy_is_feasible() -> None:
    frame = _phase05_response_rows().copy()
    frame["short_correct"] = False
    frame["medium_correct"] = False
    frame["long_correct"] = True
    predictions = pd.DataFrame(
        {
            "predicted_correct_short": 0.99,
            "predicted_correct_medium": 0.5,
            "predicted_correct_long": 0.01,
        },
        index=frame.index,
    )
    with pytest.raises(RuntimeError, match="no feasible compute penalty"):
        select_compute_penalty(
            frame,
            predictions,
            token_costs={"short": 1024.0, "medium": 4096.0, "long": 24576.0},
            maximum_accuracy_gap=0.0,
        )
