#!/usr/bin/env python
"""Run the quantized Phase 4B panel end-to-end on an Apple-Silicon Mac."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.storage import read_json, write_json_atomic

MODEL_SPECS = {
    "gemma4": (
        "configs/models/gemma4_e4b_mlx_4bit_reasoning.yaml",
        "configs/experiments/phase_04b_gemma4_mlx_4bit_16k.yaml",
    ),
    "qwen35": (
        "configs/models/qwen35_4b_mlx_4bit_reasoning.yaml",
        "configs/experiments/phase_04b_qwen35_mlx_4bit_16k.yaml",
    ),
    "ministral3": (
        "configs/models/ministral3_3b_mlx_4bit_reasoning.yaml",
        "configs/experiments/phase_04b_ministral3_mlx_4bit_16k.yaml",
    ),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=project_root / "artifacts" / "mac_mlx",
    )
    parser.add_argument(
        "--historical-bundle",
        type=Path,
        default=(
            project_root
            / "runpod_backups"
            / "rk679teh4b19ak"
            / "artifacts"
            / "shared"
            / "datasets"
            / "math_sample.jsonl"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        dest="models",
        help="Run only selected model(s); repeat this flag. Defaults to all three.",
    )
    parser.add_argument("--rebuild-datasets", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-max-new-tokens", type=int, default=64)
    parser.add_argument("--maximum-allocated-gib", type=float, default=36.0)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def _run(project_root: Path, script: str, *arguments: object) -> None:
    command = [
        sys.executable,
        "-u",
        str(project_root / "scripts" / script),
        *map(str, arguments),
    ]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def _expected_trajectories(project_root: Path, models: list[str]) -> int:
    total = 0
    for model_name in models:
        experiment_path = project_root / MODEL_SPECS[model_name][1]
        config = load_experiment_config(experiment_path)
        total += sum(dataset.sample_size for dataset in config.datasets) * len(config.seeds)
    return total


def _write_pipeline_checkpoint(
    phase_root: Path,
    project_root: Path,
    *,
    selected_models: list[str],
    status: str,
    current_model: str | None = None,
) -> None:
    """Write a human-readable model-level checkpoint from durable child progress."""

    model_states: dict[str, dict[str, object]] = {}
    config_hashes: dict[str, str] = {}
    for model_name, (_, experiment) in MODEL_SPECS.items():
        config = load_experiment_config(project_root / experiment)
        config_hashes[model_name] = config.config_hash()
        progress_path = (
            phase_root
            / "generation"
            / f"{model_name}_mlx_4bit"
            / "generation_progress.json"
        )
        if progress_path.exists():
            try:
                progress = read_json(progress_path)
            except Exception as exc:
                model_states[model_name] = {
                    "status": "unreadable",
                    "error": str(exc),
                    "progress_path": str(progress_path),
                }
            else:
                model_states[model_name] = {
                    "status": progress.get("status", "unknown"),
                    "completed_trajectories": progress.get("completed_trajectories", 0),
                    "expected_trajectories": progress.get(
                        "global_expected_trajectories",
                        progress.get("expected_trajectories", 0),
                    ),
                    "current_trajectory": progress.get("current_trajectory"),
                    "updated_at": progress.get("updated_at"),
                    "progress_path": str(progress_path),
                }
        else:
            model_states[model_name] = {
                "status": "not_started",
                "completed_trajectories": 0,
                "expected_trajectories": sum(
                    dataset.sample_size for dataset in config.datasets
                )
                * len(config.seeds),
                "progress_path": str(progress_path),
            }
    write_json_atomic(
        phase_root / "pipeline_checkpoint.json",
        {
            "schema_version": 1,
            "backend_profile": "mlx_4bit",
            "status": status,
            "current_model": current_model,
            "selected_models": selected_models,
            "expected_full_panel_trajectories": _expected_trajectories(
                project_root, list(MODEL_SPECS)
            ),
            "config_hashes": config_hashes,
            "models": model_states,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    datasets = artifacts / "shared" / "datasets_v2"
    phase = artifacts / "phase_04b"
    smoke = phase / "preflight" / "smoke"
    models = args.models or list(MODEL_SPECS)
    expected_trajectories = _expected_trajectories(root, models)

    if args.rebuild_datasets or not (datasets / "dataset_manifest.json").exists():
        if not args.historical_bundle.exists():
            raise FileNotFoundError(
                "Historical MATH bundle is required to preserve the frozen held-out panel: "
                f"{args.historical_bundle}"
            )
        _run(
            root,
            "prepare_phase04b_datasets.py",
            "--output-dir",
            datasets,
            "--backend-profile",
            "mlx_4bit",
            "--historical-bundle",
            args.historical_bundle,
        )

    if not args.skip_smoke:
        smoke_arguments: list[object] = [
            "--project-root",
            root,
            "--output-dir",
            smoke,
            "--max-new-tokens",
            args.smoke_max_new_tokens,
            "--maximum-allocated-gib",
            args.maximum_allocated_gib,
        ]
        for model_name in models:
            smoke_arguments.extend(("--model-config", MODEL_SPECS[model_name][0]))
        _run(root, "smoke_test_mlx_models.py", *smoke_arguments)
    readiness = smoke / "model_readiness.json"
    if not readiness.exists():
        raise FileNotFoundError(
            f"MLX readiness manifest is missing: {readiness}. Run without --skip-smoke first."
        )

    _write_pipeline_checkpoint(
        phase,
        root,
        selected_models=models,
        status="ready",
    )

    for model_name in models:
        _, experiment = MODEL_SPECS[model_name]
        generation_arguments: list[object] = [
            "--project-root",
            root,
            "--config",
            experiment,
            "--datasets-dir",
            datasets,
            "--readiness-manifest",
            readiness,
            "--output-dir",
            phase / "generation" / f"{model_name}_mlx_4bit",
            "--resume",
        ]
        representative = phase / "preflight" / "representative" / model_name
        if representative.exists():
            generation_arguments.extend(("--materialize-reuse-run-dir", representative))
        _write_pipeline_checkpoint(
            phase,
            root,
            selected_models=models,
            status="generating",
            current_model=model_name,
        )
        try:
            _run(root, "generate.py", *generation_arguments)
        except BaseException:
            _write_pipeline_checkpoint(
                phase,
                root,
                selected_models=models,
                status="interrupted",
                current_model=model_name,
            )
            raise
        _write_pipeline_checkpoint(
            phase,
            root,
            selected_models=models,
            status="generating",
        )

    final_status = "selected_models_complete"
    if len(models) == len(MODEL_SPECS):
        stage = "validation"
        try:
            _write_pipeline_checkpoint(
                phase,
                root,
                selected_models=models,
                status="validating",
            )
            _run(
                root,
                "validate_generation.py",
                "--run-dir",
                phase / "generation",
                "--output-dir",
                phase,
                "--expected-trajectories",
                expected_trajectories,
                "--minimum-completion-rate",
                0.98,
                "--expected-panel-manifest",
                datasets / "dataset_manifest.json",
            )
            final_status = "generation_complete"
            if args.analysis:
                stage = "analysis"
                _write_pipeline_checkpoint(
                    phase,
                    root,
                    selected_models=models,
                    status="analyzing",
                )
                _run(
                    root,
                    "run_local_phase_analysis.py",
                    "phase_04b",
                    "--project-root",
                    root,
                    "--artifacts-root",
                    artifacts,
                    "--bootstrap-repetitions",
                    args.bootstrap_repetitions,
                    "--expected-trajectories",
                    expected_trajectories,
                )
                final_status = "complete"
        except BaseException:
            _write_pipeline_checkpoint(
                phase,
                root,
                selected_models=models,
                status=f"{stage}_failed",
            )
            raise
    else:
        print(
            "Selected-model generation is complete. Strict full-panel validation waits "
            "until all three model directories are present.",
            flush=True,
        )
    _write_pipeline_checkpoint(
        phase,
        root,
        selected_models=models,
        status=(
            final_status
        ),
    )


if __name__ == "__main__":
    main()
