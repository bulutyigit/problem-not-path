#!/usr/bin/env python
"""Checkpointed Apple-Silicon runner for Phase 4C-P and Phase 4C-U."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reasonbench.storage import write_json_atomic

MODEL_SPECS = {
    "gemma4": "configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml",
    "qwen35": "configs/experiments/phase_04c_qwen35_mlx_4bit_25k.yaml",
    "ministral3": "configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml",
}
STAGES = (
    "status",
    "prepare",
    "pilot",
    "validate-pilot",
    "labeling",
    "validate-labeling",
    "extension-pilot",
    "validate-extension-pilot",
    "extension-full",
    "validate-extension-full",
)


def _print_status(phase4c: Path, selected_models: list[str]) -> None:
    checkpoint = phase4c / "pipeline_checkpoint.json"
    checkpoint_payload: dict[str, object] = {}
    if checkpoint.exists():
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        print("Pipeline checkpoint:")
        print(json.dumps(checkpoint_payload, indent=2))
    else:
        print("Pipeline checkpoint: not created")
    manifest_path = phase4c / "manifests" / "uncertainty_extension_manifest.json"
    expected_by_model: dict[str, int] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage = checkpoint_payload.get("stage")
        run_ids = set(
            manifest.get(
                "eligible_run_ids" if stage == "extension-full" else "pilot_eligible_run_ids",
                [],
            )
        )
        for record in manifest.get("records", []):
            if record.get("run_id") not in run_ids:
                continue
            model_key = str(record["model_key"])
            expected_by_model[model_key] = expected_by_model.get(model_key, 0) + 12
    print("\nPer-model progress:")
    for model in selected_models:
        model_root = phase4c / "uncertainty" / "models" / model
        progress = model_root / "extension_progress.json"
        if not progress.exists():
            print(f"- {model}: not started")
            continue
        payload = json.loads(progress.read_text(encoding="utf-8"))
        completed = sum(1 for _ in model_root.rglob("branch_complete.json"))
        expected = expected_by_model.get(str(payload.get("model_key")), payload.get("expected_branches", 0))
        status = payload.get("status", "unknown")
        current = payload.get("current_budget_arm")
        suffix = f", current arm={current}" if current else ""
        if status == "complete" and completed < expected:
            status = "checkpointed; waiting for its turn"
        print(f"- {model}: {status}, {completed}/{expected} branches{suffix}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--artifacts-root", type=Path, default=root / "artifacts" / "mac_mlx")
    parser.add_argument("--model", action="append", choices=tuple(MODEL_SPECS), dest="models")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--maximum-trajectories", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--approve-full-cohort",
        action="store_true",
        help="Required for labeling or full extension after manual pilot review.",
    )
    return parser.parse_args()


def _run(root: Path, script: str, *arguments: object) -> None:
    command = [sys.executable, "-u", str(root / "scripts" / script), *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _model_arguments(args: argparse.Namespace, *, pilot_only: bool) -> list[object]:
    values: list[object] = [
        "--resume",
        "--shard-count",
        args.shard_count,
        "--shard-index",
        args.shard_index,
    ]
    if pilot_only:
        values.append("--pilot-only")
    if args.maximum_trajectories is not None:
        values.extend(("--maximum-trajectories", args.maximum_trajectories))
    if args.deterministic:
        values.append("--deterministic")
    return values


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    phase4b = artifacts / "phase_04b"
    phase4c = artifacts / "phase_04c"
    generation = phase4b / "generation"
    readiness = phase4b / "preflight" / "smoke" / "model_readiness.json"
    probe_manifest = phase4c / "manifests" / "breakthrough_probe_manifest.json"
    extension_manifest = phase4c / "manifests" / "uncertainty_extension_manifest.json"
    selected_models = args.models or list(MODEL_SPECS)
    checkpoint = phase4c / "pipeline_checkpoint.json"
    phase4c.mkdir(parents=True, exist_ok=True)
    (phase4c / "manifests").mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        _print_status(phase4c, selected_models)
        return
    if (
        args.stage
        in {
            "labeling",
            "validate-labeling",
            "extension-full",
            "validate-extension-full",
        }
        and not args.approve_full_cohort
    ):
        raise RuntimeError(
            "Full-cohort stages require --approve-full-cohort after manual pilot review"
        )

    write_json_atomic(
        checkpoint,
        {
            "schema_version": 1,
            "stage": args.stage,
            "status": "running",
            "selected_models": selected_models,
            "probe_manifest": str(probe_manifest),
            "extension_manifest": str(extension_manifest),
        },
    )
    try:
        if args.stage == "prepare":
            _run(
                root,
                "build_breakthrough_probe_manifest.py",
                "--generation-dir",
                generation,
                "--output",
                probe_manifest,
                "--problem-count",
                20,
                "--base-seed",
                11,
            )
            _run(
                root,
                "build_uncertainty_extension_manifest.py",
                "--probe-manifest",
                probe_manifest,
                "--generation-dir",
                generation,
                "--prefix-features",
                phase4b / "features" / "features_prefix_512.parquet",
                "--output",
                extension_manifest,
            )
        elif args.stage == "pilot":
            if not probe_manifest.exists():
                raise FileNotFoundError("Run the prepare stage before the pilot")
            for model in selected_models:
                _run(
                    root,
                    "generate_breakthrough_probes.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model],
                    "--readiness-manifest",
                    readiness,
                    "--base-run-dir",
                    generation,
                    "--probe-manifest",
                    probe_manifest,
                    "--output-dir",
                    phase4c / "probes" / "models" / model,
                    *_model_arguments(args, pilot_only=True),
                )
        elif args.stage == "validate-pilot":
            probe_arguments: list[object] = []
            for model in selected_models:
                probe_arguments.extend(("--probe-dir", phase4c / "probes" / "models" / model))
            _run(
                root,
                "validate_breakthrough_probes.py",
                *probe_arguments,
                "--probe-manifest",
                probe_manifest,
                "--output-dir",
                phase4c / "probes",
                "--pilot-only",
            )
            _run(
                root,
                "audit_breakthrough_pilot.py",
                "--phase-dir",
                phase4c / "probes",
                "--output-dir",
                phase4c / "probes",
            )
        elif args.stage == "labeling":
            for model in selected_models:
                _run(
                    root,
                    "generate_breakthrough_probes.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model],
                    "--readiness-manifest",
                    readiness,
                    "--base-run-dir",
                    generation,
                    "--probe-manifest",
                    probe_manifest,
                    "--output-dir",
                    phase4c / "probes" / "models" / model,
                    *_model_arguments(args, pilot_only=False),
                )
        elif args.stage == "validate-labeling":
            probe_arguments = []
            for model in selected_models:
                probe_arguments.extend(("--probe-dir", phase4c / "probes" / "models" / model))
            _run(
                root,
                "validate_breakthrough_probes.py",
                *probe_arguments,
                "--probe-manifest",
                probe_manifest,
                "--output-dir",
                phase4c / "probes",
            )
        elif args.stage == "extension-pilot":
            if not extension_manifest.exists():
                raise FileNotFoundError("Run the prepare stage before the extension pilot")
            for model in selected_models:
                _run(
                    root,
                    "generate_uncertainty_extensions.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model],
                    "--readiness-manifest",
                    readiness,
                    "--base-run-dir",
                    generation,
                    "--extension-manifest",
                    extension_manifest,
                    "--output-dir",
                    phase4c / "uncertainty" / "models" / model,
                    *_model_arguments(args, pilot_only=True),
                )
        elif args.stage == "validate-extension-pilot":
            extension_arguments: list[object] = []
            for model in selected_models:
                extension_arguments.extend(
                    (
                        "--extension-dir",
                        phase4c / "uncertainty" / "models" / model,
                    )
                )
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *extension_arguments,
                "--extension-manifest",
                extension_manifest,
                "--output-dir",
                phase4c / "uncertainty" / "validation_pilot",
                "--pilot-only",
            )
        elif args.stage == "extension-full":
            for model in selected_models:
                _run(
                    root,
                    "generate_uncertainty_extensions.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model],
                    "--readiness-manifest",
                    readiness,
                    "--base-run-dir",
                    generation,
                    "--extension-manifest",
                    extension_manifest,
                    "--output-dir",
                    phase4c / "uncertainty" / "models" / model,
                    *_model_arguments(args, pilot_only=False),
                )
        elif args.stage == "validate-extension-full":
            extension_arguments = []
            for model in selected_models:
                extension_arguments.extend(
                    (
                        "--extension-dir",
                        phase4c / "uncertainty" / "models" / model,
                    )
                )
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *extension_arguments,
                "--extension-manifest",
                extension_manifest,
                "--output-dir",
                phase4c / "uncertainty" / "validation_full",
            )
    except BaseException:
        write_json_atomic(
            checkpoint,
            {
                "schema_version": 1,
                "stage": args.stage,
                "status": "failed_or_interrupted",
                "selected_models": selected_models,
            },
        )
        raise
    write_json_atomic(
        checkpoint,
        {
            "schema_version": 1,
            "stage": args.stage,
            "status": "complete",
            "selected_models": selected_models,
        },
    )


if __name__ == "__main__":
    main()
