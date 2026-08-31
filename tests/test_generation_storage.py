from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reasonbench.config import ModelConfig, SamplingConfig
from reasonbench.generation.engine import GenerationResult
from reasonbench.generation.storage import (
    materialize_reused_trajectory,
    trajectory_is_complete,
    write_trajectory,
)
from reasonbench.instrumentation.recorder import TOKEN_METRIC_SCHEMA_VERSION, TokenSignal
from reasonbench.storage import deterministic_run_id, read_json, sha256_file
from scripts.generate import _completed_trajectory_is_compatible, _resume_identity, _reusable_pairs


def _signal(index: int, token_id: int) -> TokenSignal:
    return TokenSignal(
        token_index=index,
        token_id=token_id,
        token_text=str(token_id),
        entropy=1.0,
        normalized_entropy=0.1,
        top1_top2_logit_margin=2.0,
        top1_top2_probability_margin=0.3,
        top1_probability=0.5,
        top5_probability_mass=0.8,
        probability_tail_mass=0.2,
        effective_vocabulary_size=10.0,
        sampled_logprob=-0.4,
        sampled_token_regret=0.2,
        surprisal=0.4,
        successive_kl_divergence=None if index == 0 else 0.03,
        successive_js_divergence=None if index == 0 else 0.01,
        hidden_norm=3.0,
        relative_l2_step=None if index == 0 else 0.1,
        cosine_drift=None if index == 0 else 0.01,
        segment="thinking",
    )


def test_trajectory_commit_contains_payload_hashes(tmp_path: Path) -> None:
    root = tmp_path / "trajectory"
    result = GenerationResult(
        generated_text="Reasoning </think> \\boxed{4}",
        reasoning_text="Reasoning",
        final_response_text="\\boxed{4}",
        boundary_status="think_tag",
        generated_token_ids=[1, 2],
        signals=[_signal(0, 1), _signal(1, 2)],
        hidden_state_indices=[0, 1],
        hidden_states=[
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
        ],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )
    write_trajectory(root, {"run_id": "run_1"}, result)
    assert trajectory_is_complete(root)
    completion = read_json(root / "complete.json")
    assert set(completion["files"]) == {
        "metadata.json",
        "token_metrics.parquet",
        "hidden_states.npz",
    }
    for filename, record in completion["files"].items():
        path = root / filename
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_file(path)


def test_materialized_reuse_is_complete_and_preserves_payload_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    result = GenerationResult(
        generated_text="Reasoning \\boxed{4}",
        reasoning_text="Reasoning",
        final_response_text="\\boxed{4}",
        boundary_status="complete",
        generated_token_ids=[1, 2],
        signals=[_signal(0, 1), _signal(1, 2)],
        hidden_state_indices=[],
        hidden_states=[],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )
    write_trajectory(source, {"run_id": "reused_run"}, result)
    destination = materialize_reused_trajectory(source, tmp_path / "destination")
    assert trajectory_is_complete(destination)
    assert read_json(destination / "complete.json") == read_json(source / "complete.json")


def test_corrupt_completed_payload_is_not_reusable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    result = GenerationResult(
        generated_text="Reasoning \\boxed{4}",
        reasoning_text="Reasoning",
        final_response_text="\\boxed{4}",
        boundary_status="complete",
        generated_token_ids=[1, 2],
        signals=[_signal(0, 1), _signal(1, 2)],
        hidden_state_indices=[],
        hidden_states=[],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )
    write_trajectory(source, {"run_id": "corrupt"}, result)
    with (source / "token_metrics.parquet").open("ab") as handle:
        handle.write(b"corruption")
    assert trajectory_is_complete(source)
    with pytest.raises(ValueError, match="corrupt"):
        materialize_reused_trajectory(source, tmp_path / "destination")


def test_local_resume_requires_exact_immutable_metadata(tmp_path: Path) -> None:
    config = type(
        "ExperimentFixture",
        (),
        {
            "experiment_id": "phase_04b_fixture",
            "phase_id": "phase_04b",
            "model": ModelConfig(
                key="gemma4_e4b",
                model_id="fixture",
                revision="abc123",
                adapter="gemma4",
                max_new_tokens=16384,
                sampling=SamplingConfig(temperature=0.6, top_p=0.95, top_k=20),
            ),
            "prompt_version": "v1",
            "config_hash": lambda self: "fixture-config-hash",
        },
    )()
    problem = SimpleNamespace(dataset="math", problem_id="problem_1", research_split="test")
    run_id = deterministic_run_id(
        config.experiment_id,
        config.model.key,
        problem.dataset,
        problem.problem_id,
        11,
    )
    metadata = _resume_identity(
        config,
        dataset_bundle_sha256="dataset-hash",
        problem=problem,
        seed=11,
        run_id=run_id,
    )
    result = GenerationResult(
        generated_text="Reasoning \\boxed{4}",
        reasoning_text="Reasoning",
        final_response_text="\\boxed{4}",
        boundary_status="complete",
        generated_token_ids=[1, 2],
        signals=[_signal(0, 1), _signal(1, 2)],
        hidden_state_indices=[],
        hidden_states=[],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )
    trajectory = tmp_path / "trajectory"
    write_trajectory(trajectory, metadata, result)
    assert _completed_trajectory_is_compatible(
        trajectory,
        config,
        dataset_bundle_sha256="dataset-hash",
        problem=problem,
        seed=11,
    )
    changed_problem = SimpleNamespace(
        dataset="math", problem_id="other_problem", research_split="test"
    )
    assert not _completed_trajectory_is_compatible(
        trajectory,
        config,
        dataset_bundle_sha256="dataset-hash",
        problem=changed_problem,
        seed=11,
    )


def test_external_reuse_requires_exact_generation_settings(tmp_path: Path) -> None:
    config = type(
        "ExperimentFixture",
        (),
        {
            "model": ModelConfig(
                key="gemma4_e4b",
                model_id="fixture",
                revision="abc123",
                adapter="gemma4",
                reasoning_budget=8192,
                reasoning_budget_policy="external_hard_cap",
                final_answer_reserve=512,
                max_new_tokens=8704,
                sampling=SamplingConfig(temperature=0.6, top_p=0.95, top_k=20),
            ),
            "prompt_version": "v1",
            "config_hash": lambda self: "fixture-config-hash",
        },
    )()
    trajectory = tmp_path / "trajectories" / "run"
    result = GenerationResult(
        generated_text="Reasoning \\boxed{4}",
        reasoning_text="Reasoning",
        final_response_text="\\boxed{4}",
        boundary_status="complete",
        generated_token_ids=[1, 2],
        signals=[_signal(0, 1), _signal(1, 2)],
        hidden_state_indices=[],
        hidden_states=[],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )
    write_trajectory(
        trajectory,
        {
            "run_id": "run",
            "model_key": "gemma4_e4b",
            "config_hash": "fixture-config-hash",
            "token_metric_schema_version": TOKEN_METRIC_SCHEMA_VERSION,
            "model_revision": "abc123",
            "model_mode": "reasoning",
            "assigned_reasoning_budget": 8192,
            "reasoning_budget_policy": "external_hard_cap",
            "final_answer_reserve": 512,
            "max_new_tokens": 8704,
            "prompt_version": "v1",
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
            },
            "dataset": "gsm8k",
            "problem_id": "problem_1",
            "seed": 11,
        },
        result,
    )
    assert _reusable_pairs(config, [tmp_path]) == {("gsm8k", "problem_1", 11)}
    changed = type(
        "ExperimentFixture",
        (),
        {
            "model": replace(config.model, final_answer_reserve=256),
            "prompt_version": "v1",
        },
    )()
    assert not _reusable_pairs(changed, [tmp_path])
