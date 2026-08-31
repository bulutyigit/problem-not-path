from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reasonbench.config import experiment_from_dict, load_experiment_config
from reasonbench.exceptions import ConfigurationError
from reasonbench.runtime import source_tree_revision
from reasonbench.storage import deterministic_run_id, read_json, write_json_atomic


def _experiment_payload() -> dict:
    return {
        "experiment_id": "phase_01_test",
        "phase_id": "phase_01",
        "model": {
            "key": "test_model",
            "model_id": "organization/test-model",
            "adapter": "gemma4",
            "dtype": "bfloat16",
            "mode": "reasoning",
            "max_new_tokens": 128,
        },
        "datasets": [{"name": "gsm8k", "sample_size": 10}],
        "seeds": [11, 23],
    }


def test_experiment_hash_is_stable() -> None:
    first = experiment_from_dict(_experiment_payload())
    second = experiment_from_dict(_experiment_payload())
    assert first.config_hash() == second.config_hash()


@pytest.mark.parametrize(
    "filename",
    [
        "phase_02_gemma4_math_difficulty.yaml",
        "phase_03_gemma4.yaml",
        "phase_03_qwen35.yaml",
        "phase_03_ministral3.yaml",
        "phase_04_gemma4_16k.yaml",
        "phase_04_qwen35_16k.yaml",
        "phase_04_ministral3_16k.yaml",
    ],
)
def test_difficulty_phases_use_one_hundred_balanced_math_problems(filename: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project_root / "configs" / "experiments" / filename)

    assert len(config.datasets) == 1
    assert config.datasets[0].name == "math"
    assert config.datasets[0].sample_size == 100
    assert config.datasets[0].levels == (1, 2, 3, 4, 5)
    assert config.datasets[0].nested_base_sample_size == 50
    expected_limit = 16384 if filename.startswith("phase_04_") else 8192
    assert config.model.max_new_tokens == expected_limit
    expected_seeds = 1 if filename.startswith(("phase_03_", "phase_04_")) else 4
    assert len(config.seeds) == expected_seeds
    assert (
        sum(dataset.sample_size for dataset in config.datasets) * len(config.seeds)
        == 100 * expected_seeds
    )


@pytest.mark.parametrize(
    ("baseline_name", "extended_name"),
    [
        ("phase_03_gemma4.yaml", "phase_04_gemma4_16k.yaml"),
        ("phase_03_qwen35.yaml", "phase_04_qwen35_16k.yaml"),
        ("phase_03_ministral3.yaml", "phase_04_ministral3_16k.yaml"),
    ],
)
def test_phase_04_changes_only_phase_identity_and_generation_cap(
    baseline_name: str,
    extended_name: str,
) -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    baseline = load_experiment_config(config_root / baseline_name)
    extended = load_experiment_config(config_root / extended_name)

    assert replace(extended.model, max_new_tokens=8192) == baseline.model
    assert extended.datasets == baseline.datasets
    assert extended.seeds == baseline.seeds
    assert extended.prompt_version == baseline.prompt_version
    assert extended.output_subdirectory == baseline.output_subdirectory
    assert extended.phase_id == "phase_04"


@pytest.mark.parametrize(
    "filename",
    [
        "phase_04b_gemma4_16k.yaml",
        "phase_04b_qwen35_16k.yaml",
        "phase_04b_ministral3_16k.yaml",
    ],
)
def test_phase_04b_configs_freeze_two_seed_three_model_16k_panel(filename: str) -> None:
    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "configs" / "experiments" / filename
    )
    assert config.phase_id == "phase_04b"
    assert config.model.max_new_tokens == 16384
    assert config.seeds == (11, 23)
    assert config.datasets[0].levels == (1, 2, 3, 4, 5)
    assert config.datasets[0].nested_base_sample_size is None
    assert config.model.key in {"gemma4_e4b", "qwen35_4b", "ministral3_3b"}


@pytest.mark.parametrize(
    "filename",
    [
        "phase_04b_gemma4_mlx_4bit_16k.yaml",
        "phase_04b_qwen35_mlx_4bit_16k.yaml",
        "phase_04b_ministral3_mlx_4bit_16k.yaml",
    ],
)
def test_mlx_phase_04b_configs_freeze_separate_int4_panel(filename: str) -> None:
    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "configs" / "experiments" / filename
    )
    assert config.phase_id == "phase_04b"
    assert config.model.backend == "mlx_vlm"
    assert config.model.dtype == "int4"
    assert config.model.source_model_id
    assert config.model.model_id.startswith("mlx-community/")
    assert config.model.max_new_tokens == 16384
    assert config.seeds == (11,)
    assert config.datasets[0].levels == (1, 2, 3, 4, 5)
    assert config.model.key.endswith("_mlx_4bit")


def test_primary_configuration_rejects_non_bf16() -> None:
    payload = _experiment_payload()
    payload["model"]["dtype"] = "float16"
    with pytest.raises(ConfigurationError, match="bfloat16"):
        experiment_from_dict(payload)


def test_mlx_configuration_requires_int4() -> None:
    payload = _experiment_payload()
    payload["model"].update({"backend": "mlx_vlm", "dtype": "bfloat16"})
    with pytest.raises(ConfigurationError, match="int4"):
        experiment_from_dict(payload)
    payload["model"]["dtype"] = "int4"
    assert experiment_from_dict(payload).model.backend == "mlx_vlm"


def test_assigned_budget_requires_explicit_external_policy() -> None:
    payload = _experiment_payload()
    payload["model"].update(
        {
            "reasoning_budget": 64,
            "final_answer_reserve": 32,
            "max_new_tokens": 96,
        }
    )
    with pytest.raises(ConfigurationError, match="external_hard_cap"):
        experiment_from_dict(payload)
    payload["model"]["reasoning_budget_policy"] = "external_hard_cap"
    assert experiment_from_dict(payload).model.reasoning_budget == 64


def test_deterministic_run_id_changes_with_seed() -> None:
    first = deterministic_run_id("experiment", "model", "gsm8k", "problem", 11)
    repeated = deterministic_run_id("experiment", "model", "gsm8k", "problem", 11)
    changed = deterministic_run_id("experiment", "model", "gsm8k", "problem", 23)
    assert first == repeated
    assert first != changed


def test_atomic_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    value = {"status": "passed", "metrics": {"count": 3}}
    write_json_atomic(path, value)
    assert read_json(path) == value
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_source_tree_hash_ignores_notebook_outputs(tmp_path: Path) -> None:
    notebook_path = tmp_path / "notebook.ipynb"
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["value = 1\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json_atomic(notebook_path, payload)
    first = source_tree_revision(tmp_path)
    payload["cells"][0]["execution_count"] = 3
    payload["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["1\n"], "name": "stdout"}]
    write_json_atomic(notebook_path, payload)
    assert source_tree_revision(tmp_path) == first
