from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_runpod_pipeline import _slug

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "scripts" / "run_runpod_pipeline.py"


def test_log_slug_removes_path_separators() -> None:
    assert _slug("shard 1/2") == "shard_1_2"


def run_dry_phase(
    tmp_path: Path,
    phase: str,
    *extra_arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            phase,
            "--project-root",
            str(PROJECT_ROOT),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--dry-run",
            *extra_arguments,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_phase_00_dry_run_contains_readiness_commands(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_00")

    assert "prepare_datasets.py" in result.stdout
    assert "smoke_test_models.py" in result.stdout
    assert "gemma4_e4b_reasoning.yaml" in result.stdout
    assert "qwen35_4b_reasoning.yaml" in result.stdout
    assert "ministral3_3b_reasoning.yaml" in result.stdout
    assert "create_phase_report.py" in result.stdout


def test_phase_01_dry_run_is_resumable_and_complete(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_01")

    assert result.stdout.count("generate.py") == 2
    assert result.stdout.count("--resume") == 2
    assert "--expected-trajectories 800" in result.stdout
    assert "--paired-left non_reasoning --paired-right reasoning" in result.stdout
    assert "[6/6] write phase report" in result.stdout


def test_phase_01_two_workers_use_disjoint_shard_directories(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_01", "--generation-workers", "2")

    assert result.stdout.count("generate.py") == 4
    assert result.stdout.count("--shard-count 2") == 4
    assert result.stdout.count("--shard-index 0") == 2
    assert result.stdout.count("--shard-index 1") == 2
    assert "generation/reasoning/shard_00" in result.stdout
    assert "generation/reasoning/shard_01" in result.stdout
    assert "generation/non_reasoning/shard_00" in result.stdout
    assert "generation/non_reasoning/shard_01" in result.stdout
    assert result.stdout.count("--reuse-run-dir") == 4
    assert "[6/6] write phase report" in result.stdout


def test_phase_03_four_workers_use_four_disjoint_shards(tmp_path: Path) -> None:
    result = run_dry_phase(
        tmp_path,
        "phase_03",
        "--generation-workers",
        "4",
        "--generation-only",
    )

    assert result.stdout.count("--shard-count 4") == 12
    for index in range(4):
        assert result.stdout.count(f"--shard-index {index}") == 3
    assert "generation/qwen35/shard_03" in result.stdout
    assert "generation/ministral3/shard_03" in result.stdout


def test_phase_02_dry_run_generates_gemma_difficulty_study(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_02")

    assert result.stdout.count("generate.py") == 1
    assert "phase_02_gemma4_math_difficulty.yaml" in result.stdout
    assert "phase_01/generation/reasoning" in result.stdout
    assert "--expected-trajectories 400" in result.stdout
    assert "analyze_difficulty_dynamics.py" in result.stdout
    assert "--features-dir" in result.stdout
    for prefix in (16, 32, 64, 128, 256, 512, 1024, 2048):
        assert f"--prefix-length {prefix}" in result.stdout
    assert "[5/5] write phase report" in result.stdout


def test_phase_03_dry_run_materializes_phase_02_gemma_reuse(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_03")

    assert result.stdout.count("generate.py") == 3
    assert "phase_02/generation/gemma4" in result.stdout
    assert result.stdout.count("--materialize-reuse-run-dir") == 1
    assert "--expected-trajectories 300" in result.stdout
    assert "analyze_hidden_pca.py" in result.stdout
    assert "analyze_cross_model_difficulty.py" in result.stdout
    assert "--features-dir" in result.stdout
    for prefix in (16, 32, 64, 128, 256, 512, 1024, 2048):
        assert f"--prefix-length {prefix}" in result.stdout
    assert "analyze cross-model difficulty and hidden PCA" in result.stdout
    assert "require Phase 2 hypothesis freeze or candidate record" in result.stdout
    assert "[7/7] write phase report" in result.stdout


def test_generation_only_stops_before_cpu_feature_extraction(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_03", "--generation-only")

    assert "validate_generation.py" in result.stdout
    assert "extract_features.py" not in result.stdout
    assert "run_local_phase_analysis.py" in result.stdout


def test_phase_04_reruns_matched_panel_at_16k(tmp_path: Path) -> None:
    phase_04 = run_dry_phase(tmp_path, "phase_04")

    assert phase_04.stdout.count("generate.py") == 3
    assert "phase_04_gemma4_16k.yaml" in phase_04.stdout
    assert "phase_04_qwen35_16k.yaml" in phase_04.stdout
    assert "phase_04_ministral3_16k.yaml" in phase_04.stdout
    assert "--expected-trajectories 300" in phase_04.stdout
    assert "analyze_cap_extension.py" in phase_04.stdout
    assert "evaluate_early_prediction.py" in phase_04.stdout
    assert "evaluate_early_length.py" in phase_04.stdout
    assert "analyze_phase04_dynamics.py" in phase_04.stdout
    assert "finalize_phase_04.py" in phase_04.stdout
    assert "length_prediction_summary.json" in phase_04.stdout
    assert "dynamics_summary.json" in phase_04.stdout
    assert "phase_03/features/features_full.parquet" in phase_04.stdout
    for prefix in (16, 32, 64, 128, 256, 512):
        assert f"--prefix-length {prefix}" in phase_04.stdout


def test_phase_04b_schedules_three_models_two_seeds(tmp_path: Path) -> None:
    phase_04b = run_dry_phase(tmp_path, "phase_04b", "--generation-only")

    assert phase_04b.stdout.count("generate.py") == 3
    assert "phase_04b_gemma4_16k.yaml" in phase_04b.stdout
    assert "phase_04b_qwen35_16k.yaml" in phase_04b.stdout
    assert "phase_04b_ministral3_16k.yaml" in phase_04b.stdout
    assert "--expected-trajectories 600" in phase_04b.stdout
    assert "datasets_v2" in phase_04b.stdout
    assert "prepare_phase04b_datasets.py" in phase_04b.stdout
    assert "smoke_test_models.py" in phase_04b.stdout
    assert "preflight_phase04b_generation.py" in phase_04b.stdout
    assert "--expected-panel-manifest" in phase_04b.stdout
    for prefix in (100, 120, 500, 768, 1024, 1536, 2048, 4096, 8192):
        assert f"--prefix-length {prefix}" not in phase_04b.stdout


def test_phase_04b_dense_prefix_grid_is_extracted_locally(tmp_path: Path) -> None:
    phase_04b = run_dry_phase(tmp_path, "phase_04b")

    for prefix in (
        16,
        32,
        64,
        100,
        120,
        128,
        256,
        500,
        512,
        768,
        1024,
        1536,
        2048,
        4096,
        8192,
    ):
        assert f"--prefix-length {prefix}" in phase_04b.stdout


def test_phase_04b_four_worker_preflight_matches_paid_generation(tmp_path: Path) -> None:
    phase_04b = run_dry_phase(
        tmp_path,
        "phase_04b",
        "--generation-workers",
        "4",
        "--generation-only",
    )

    assert "preflight_phase04b_generation.py" in phase_04b.stdout
    assert phase_04b.stdout.count("--generation-workers 4") == 1
    assert phase_04b.stdout.count("--shard-count 4") == 12


def test_phase_04c_builds_sparse_probe_cohort_and_validates_all_models(
    tmp_path: Path,
) -> None:
    result = run_dry_phase(
        tmp_path,
        "phase_04c",
        "--generation-workers",
        "2",
        "--probe-problem-count",
        "20",
    )

    assert "build_breakthrough_probe_manifest.py" in result.stdout
    assert "--problem-count 20" in result.stdout
    assert result.stdout.count("generate_breakthrough_probes.py") == 6
    assert "phase_04b_gemma4_16k.yaml" in result.stdout
    assert "phase_04b_qwen35_16k.yaml" in result.stdout
    assert "phase_04b_ministral3_16k.yaml" in result.stdout
    assert "validate_breakthrough_probes.py" in result.stdout
    assert "breakthrough_probe_manifest.json" in result.stdout
    assert "Phase 4c GPU work is complete" in result.stdout


def test_phase_04c_pilot_reuses_the_full_frozen_manifest(tmp_path: Path) -> None:
    result = run_dry_phase(
        tmp_path,
        "phase_04c",
        "--generation-workers",
        "2",
        "--probe-problem-count",
        "20",
        "--probe-pilot-only",
    )

    assert "--problem-count 20" in result.stdout
    # One flag per model shard plus one on strict pilot validation.
    assert result.stdout.count("--pilot-only") == 7
    assert "validate_breakthrough_probes.py" in result.stdout
    assert "audit_breakthrough_pilot.py" in result.stdout


def test_cpu_analysis_phases_use_two_workers_by_default(tmp_path: Path) -> None:
    phase_05 = run_dry_phase(tmp_path, "phase_05")
    phase_06 = run_dry_phase(tmp_path, "phase_06")

    assert "train_predictors.py" in phase_05.stdout
    assert "phase_04/features/features_full.parquet" in phase_05.stdout
    assert "--workers 2" in phase_05.stdout
    assert "evaluate_early_prediction.py" in phase_06.stdout
    assert "evaluate_spectral_increment.py" in phase_06.stdout
    assert "phase_04/generation" in phase_06.stdout
    assert "phase_05/analysis" in phase_06.stdout
    assert phase_06.stdout.count("--workers 2") == 2


def test_phase_07_verifies_hashes_by_default(tmp_path: Path) -> None:
    result = run_dry_phase(tmp_path, "phase_07")

    assert "build_final_report.py" in result.stdout
    assert "--verify-hashes" in result.stdout
    assert "validate_final_report.py" in result.stdout
