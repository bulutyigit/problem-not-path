from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from reasonbench.generation.engine import GenerationResult
from reasonbench.generation.storage import write_trajectory
from reasonbench.instrumentation.recorder import TokenSignal
from reasonbench.storage import read_json, sha256_file, write_json_atomic


def _result() -> GenerationResult:
    signal = TokenSignal(
        token_index=0,
        token_id=1,
        token_text="x",
        entropy=1.0,
        normalized_entropy=0.1,
        top1_top2_logit_margin=2.0,
        top1_top2_probability_margin=0.3,
        top1_probability=0.5,
        top5_probability_mass=0.8,
        probability_tail_mass=0.2,
        effective_vocabulary_size=2.7,
        sampled_logprob=-0.4,
        sampled_token_regret=0.2,
        surprisal=0.4,
        successive_kl_divergence=None,
        successive_js_divergence=None,
        hidden_norm=3.0,
        relative_l2_step=None,
        cosine_drift=None,
        segment="thinking",
    )
    return GenerationResult(
        generated_text="x",
        reasoning_text="x",
        final_response_text="",
        boundary_status="complete",
        generated_token_ids=[1],
        signals=[signal],
        hidden_state_indices=[],
        hidden_states=[],
        finish_reason="eos",
        inserted_boundary_token_count=0,
        reasoning_boundary_forced=False,
        reasoning_stage_token_count=None,
    )


def _write_panel(root: Path, *, omit_last: bool = False) -> tuple[Path, Path]:
    manifest = root / "dataset_manifest.json"
    payload = {
        "problem_ids_by_split": {"train": ["p1"], "validation": [], "test": []},
        "frozen_generation_seeds": [11, 23],
        "accepted_model_configs": [
            {"model_key": "gemma4_e4b", "experiment_id": "gemma"},
            {"model_key": "ministral3_3b", "experiment_id": "ministral"},
        ],
    }
    write_json_atomic(manifest, payload)
    manifest_hash = sha256_file(manifest)
    run_dir = root / "generation"
    rows = [
        ("gemma4_e4b", "gemma", 11),
        ("gemma4_e4b", "gemma", 23),
        ("ministral3_3b", "ministral", 11),
        ("ministral3_3b", "ministral", 23),
    ]
    for index, (model_key, experiment_id, seed) in enumerate(rows):
        if omit_last and index == len(rows) - 1:
            continue
        write_trajectory(
            run_dir / f"trajectory_{index}",
            {
                "run_id": f"{model_key}-{seed}",
                "model_key": model_key,
                "experiment_id": experiment_id,
                "problem_id": "p1",
                "seed": seed,
                "dataset_bundle_sha256": manifest_hash,
                "verification": {"correct": True, "extraction_status": "boxed"},
                "finish_reason": "eos",
                "dataset": "math",
            },
            _result(),
        )
    return manifest, run_dir


def _run(root: Path, manifest: Path, run_dir: Path, expected: int) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = root / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_generation.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--expected-trajectories",
            str(expected),
            "--expected-panel-manifest",
            str(manifest),
            "--minimum-completion-rate",
            "1.0",
        ],
        cwd=project_root,
        check=True,
    )
    return read_json(output_dir / "generation_validation.json")


def test_strict_panel_validation_accepts_exact_model_problem_seed_grid(tmp_path: Path) -> None:
    manifest, run_dir = _write_panel(tmp_path)
    validation = _run(tmp_path, manifest, run_dir, expected=4)
    assert validation["valid"]
    assert validation["strict_panel"]["missing_trajectory_keys"] == 0
    assert validation["strict_panel"]["dataset_manifest_hash_mismatches"] == 0


def test_strict_panel_validation_rejects_missing_model_problem_seed_cell(tmp_path: Path) -> None:
    manifest, run_dir = _write_panel(tmp_path, omit_last=True)
    validation = _run(tmp_path, manifest, run_dir, expected=4)
    assert not validation["valid"]
    assert validation["strict_panel"]["missing_trajectory_keys"] == 1
