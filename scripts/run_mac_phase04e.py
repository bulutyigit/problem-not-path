#!/usr/bin/env python
"""Checkpointed local runner for the held-out Phase 4E U512 routing experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reasonbench.storage import ensure_directory, write_json_atomic

MODEL_SPECS = {
    "gemma4": {
        "key": "gemma4_e4b_mlx_4bit",
        "prefix": "configs/experiments/phase_04e_gemma4_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_04e_gemma4_mlx_4bit_extension.yaml",
    },
    "ministral3": {
        "key": "ministral3_3b_mlx_4bit",
        "prefix": "configs/experiments/phase_04e_ministral3_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_04e_ministral3_mlx_4bit_extension.yaml",
    },
}
STAGES = (
    "status",
    "prepare-dataset",
    "generate-prefixes",
    "extract-prefix-features",
    "freeze-policy",
    "generate-branches",
    "validate-branches",
    "analyze-policy",
    "analyze-explanatory",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--artifacts-root", type=Path, default=root / "artifacts" / "mac_mlx")
    parser.add_argument("--model", action="append", choices=tuple(MODEL_SPECS), dest="models")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument(
        "--approve-heldout-run",
        action="store_true",
        help="Required before any held-out model generation is started.",
    )
    return parser.parse_args()


def _run(root: Path, script: str, *arguments: object) -> None:
    command = [sys.executable, "-u", str(root / "scripts" / script), *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _status(phase: Path, selected: list[str]) -> None:
    checkpoint_path = phase / "pipeline_checkpoint.json"
    if checkpoint_path.exists():
        print(json.dumps(json.loads(checkpoint_path.read_text(encoding="utf-8")), indent=2))
    else:
        print("No Phase 4E checkpoint exists yet.")
    for model in selected:
        name = MODEL_SPECS[model]["key"]
        prefix_progress = phase / "generation" / model / "generation_progress.json"
        branch_progress = phase / "branches" / model / "extension_progress.json"
        print(f"\n{model} ({name})")
        for label, path in (("prefixes", prefix_progress), ("branches", branch_progress)):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if label == "prefixes":
                    complete = payload.get("completed_trajectories", 0)
                    expected = payload.get("global_expected_trajectories", 0)
                else:
                    complete = payload.get("completed_branches", 0)
                    expected = payload.get("expected_branches", 0)
                print(f"- {label}: {payload.get('status', 'unknown')} ({complete}/{expected})")
            else:
                print(f"- {label}: not started")


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard-count/shard-index")
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    phase = ensure_directory(artifacts / "phase_04e")
    datasets = phase / "datasets"
    generation = phase / "generation"
    features = phase / "features"
    manifests = ensure_directory(phase / "manifests")
    branches = phase / "branches"
    validation = phase / "validation"
    analysis = phase / "analysis"
    selected = args.models or list(MODEL_SPECS)
    if args.stage == "status":
        _status(phase, selected)
        return
    if args.stage in {"generate-prefixes", "generate-branches"} and not args.approve_heldout_run:
        raise RuntimeError("Held-out generation requires --approve-heldout-run")

    checkpoint = phase / "pipeline_checkpoint.json"
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "running", "models": selected})
    try:
        if args.stage == "prepare-dataset":
            _run(
                root,
                "prepare_phase04e_challenge.py",
                "--challenge-bundle",
                artifacts / "shared" / "datasets_v2" / "high_difficulty_challenge_50.jsonl",
                "--source-dataset-manifest",
                artifacts / "shared" / "datasets_v2" / "dataset_manifest.json",
                "--output-dir",
                datasets,
            )
        elif args.stage == "generate-prefixes":
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            if not (datasets / "dataset_manifest.json").exists():
                raise FileNotFoundError("Run prepare-dataset before prefix generation")
            for model in selected:
                _run(
                    root,
                    "generate.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model]["prefix"],
                    "--datasets-dir",
                    datasets,
                    "--readiness-manifest",
                    readiness,
                    "--output-dir",
                    generation / model,
                    "--resume",
                )
        elif args.stage == "extract-prefix-features":
            _run(
                root,
                "extract_features.py",
                "--run-dir",
                generation,
                "--output-dir",
                features,
                "--prefix-length",
                512,
                "--workers",
                1,
            )
        elif args.stage == "freeze-policy":
            source_manifest = manifests / "source_prefix_manifest.json"
            extension_manifest = manifests / "u512_policy_extension_manifest.json"
            source_args: list[object] = [
                "--generation-dir",
                generation,
                "--dataset-manifest",
                datasets / "dataset_manifest.json",
                "--output",
                source_manifest,
                "--base-seed",
                29,
            ]
            for model in selected:
                source_args.extend(("--model-key", MODEL_SPECS[model]["key"]))
            _run(root, "build_phase04e_source_manifest.py", *source_args)
            _run(
                root,
                "build_uncertainty_extension_manifest.py",
                "--probe-manifest",
                source_manifest,
                "--generation-dir",
                generation,
                "--prefix-features",
                features / "features_prefix_512.parquet",
                "--reference-prefix-features",
                artifacts / "phase_04b" / "features" / "features_prefix_512.parquet",
                "--output",
                extension_manifest,
                "--continuations",
                4,
            )
        elif args.stage == "generate-branches":
            extension_manifest = manifests / "u512_policy_extension_manifest.json"
            if not extension_manifest.exists():
                raise FileNotFoundError("Run freeze-policy before generating branches")
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            for model in selected:
                _run(
                    root,
                    "generate_uncertainty_extensions.py",
                    "--project-root",
                    root,
                    "--config",
                    MODEL_SPECS[model]["extension"],
                    "--readiness-manifest",
                    readiness,
                    "--base-run-dir",
                    generation,
                    "--extension-manifest",
                    extension_manifest,
                    "--output-dir",
                    branches / model,
                    "--resume",
                    "--shard-count",
                    args.shard_count,
                    "--shard-index",
                    args.shard_index,
                )
        elif args.stage == "validate-branches":
            extension_manifest = manifests / "u512_policy_extension_manifest.json"
            validation_args: list[object] = []
            for model in selected:
                validation_args.extend(("--extension-dir", branches / model))
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *validation_args,
                "--extension-manifest",
                extension_manifest,
                "--output-dir",
                validation,
                "--bootstrap-repetitions",
                args.bootstrap_repetitions,
            )
        elif args.stage == "analyze-policy":
            _run(
                root,
                "evaluate_phase04e_routing.py",
                "--pairs",
                validation / "uncertainty_extension_pairs.parquet",
                "--validation",
                validation / "uncertainty_extension_validation.json",
                "--extension-manifest",
                manifests / "u512_policy_extension_manifest.json",
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                args.bootstrap_repetitions,
            )
        elif args.stage == "analyze-explanatory":
            _run(
                root,
                "analyze_phase04e_explanatory.py",
                "--pairs",
                validation / "uncertainty_extension_pairs.parquet",
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                args.bootstrap_repetitions,
            )
    except BaseException:
        write_json_atomic(checkpoint, {"stage": args.stage, "status": "failed_or_interrupted", "models": selected})
        raise
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "complete", "models": selected})


if __name__ == "__main__":
    main()
