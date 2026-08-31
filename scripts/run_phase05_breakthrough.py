#!/usr/bin/env python
"""Checkpointed Phase 5 breakthrough-aware controller pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reasonbench.storage import ensure_directory, write_json_atomic

MODELS = {
    "gemma4": {
        "key": "gemma4_e4b_mlx_4bit",
        "development_probe": "configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml",
        "prefix": "configs/experiments/phase_05_gemma4_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_05_gemma4_mlx_4bit_extension.yaml",
    },
    "ministral3": {
        "key": "ministral3_3b_mlx_4bit",
        "development_probe": "configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml",
        "prefix": "configs/experiments/phase_05_ministral3_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_05_ministral3_mlx_4bit_extension.yaml",
    },
}
STAGES = (
    "status",
    "complete-development-probes",
    "validate-development-probes",
    "build-development-tables",
    "fit-freeze-controller",
    "prepare-harp",
    "generate-harp-prefixes",
    "extract-harp-prefix-features",
    "build-harp-extension-manifest",
    "freeze-harp-routing",
    "generate-harp-arms",
    "validate-harp-arms",
    "evaluate-harp",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--artifacts-root", type=Path, default=root / "artifacts" / "mac_mlx")
    parser.add_argument("--harp-source", type=Path)
    parser.add_argument("--model", action="append", choices=tuple(MODELS), dest="models")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--approve-external-run", action="store_true")
    return parser.parse_args()


def _run(root: Path, script: str, *arguments: object) -> None:
    command = [sys.executable, "-u", str(root / "scripts" / script), *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _status(phase: Path) -> None:
    checkpoint = phase / "pipeline_checkpoint.json"
    print(checkpoint.read_text(encoding="utf-8") if checkpoint.exists() else "No Phase 5 checkpoint.")
    paths = {
        "development labels": phase / "development_labels" / "breakthrough_probe_validation.json",
        "frozen controller": phase / "policy" / "phase05_frozen_policy.json",
        "HARP dataset": phase / "datasets" / "dataset_manifest.json",
        "frozen HARP routing": phase / "manifests" / "harp_routing_manifest.json",
        "HARP validation": phase / "validation" / "uncertainty_extension_validation.json",
        "HARP report": phase / "analysis" / "phase05_harp_report.md",
    }
    for label, path in paths.items():
        print(f"- {label}: {'ready' if path.exists() else 'missing'} ({path})")


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard-count/shard-index")
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    phase = ensure_directory(artifacts / "phase_05_breakthrough")
    selected = args.models or list(MODELS)
    if args.stage == "status":
        _status(phase)
        return
    expensive = {"complete-development-probes", "generate-harp-prefixes", "generate-harp-arms"}
    if args.stage in expensive and not args.approve_external_run:
        raise RuntimeError(f"{args.stage} requires --approve-external-run")
    checkpoint = phase / "pipeline_checkpoint.json"
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "running", "models": selected})

    development_probe_root = artifacts / "phase_04c" / "probes" / "models"
    development_labels = phase / "development_labels"
    tables = phase / "development_tables"
    policy = phase / "policy"
    datasets = phase / "datasets"
    generation = phase / "generation"
    features = phase / "features"
    manifests = ensure_directory(phase / "manifests")
    branches = phase / "branches"
    validation = phase / "validation"
    analysis = phase / "analysis"
    readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
    probe_manifest = artifacts / "phase_04c" / "manifests" / "breakthrough_probe_manifest.json"
    try:
        if args.stage == "complete-development-probes":
            for model in selected:
                spec = MODELS[model]
                _run(
                    root,
                    "generate_breakthrough_probes.py",
                    "--project-root", root,
                    "--config", spec["development_probe"],
                    "--readiness-manifest", readiness,
                    "--base-run-dir", artifacts / "phase_04b" / "generation",
                    "--probe-manifest", probe_manifest,
                    "--output-dir", development_probe_root / model,
                    "--resume",
                    "--shard-count", args.shard_count,
                    "--shard-index", args.shard_index,
                )
        elif args.stage == "validate-development-probes":
            command: list[object] = []
            for model in selected:
                command.extend(("--probe-dir", development_probe_root / model))
                command.extend(("--model-key", MODELS[model]["key"]))
            _run(
                root,
                "validate_breakthrough_probes.py",
                *command,
                "--probe-manifest", probe_manifest,
                "--output-dir", development_labels,
            )
        elif args.stage == "build-development-tables":
            _run(
                root,
                "build_breakthrough_tables.py",
                "--features-dir", artifacts / "phase_04b" / "features",
                "--labels", development_labels / "breakthrough_labels.parquet",
                "--output-dir", tables,
            )
        elif args.stage == "fit-freeze-controller":
            _run(
                root,
                "fit_phase05_breakthrough_controller.py",
                "--probe-validation", development_labels / "breakthrough_probe_validation.json",
                "--horizon-table", tables / "breakthrough_horizon_table.parquet",
                "--eventual-success-table", tables / "eventual_success_table.parquet",
                "--prefix-features", artifacts / "phase_04b" / "features" / "features_prefix_512.parquet",
                "--development-pairs", artifacts / "phase_04c" / "uncertainty" / "validation_answer_remediated" / "uncertainty_extension_pairs.parquet",
                "--output-dir", policy,
            )
        elif args.stage == "prepare-harp":
            if args.harp_source is None:
                raise ValueError("prepare-harp requires --harp-source HARP.jsonl.zip")
            _run(
                root,
                "prepare_phase05_harp.py",
                "--harp-jsonl-or-zip", args.harp_source,
                "--math-bundle", artifacts / "shared" / "datasets_v2" / "math_sample.jsonl",
                "--math-bundle", artifacts / "shared" / "datasets_v2" / "high_difficulty_challenge_50.jsonl",
                "--output-dir", datasets,
            )
        elif args.stage == "generate-harp-prefixes":
            for model in selected:
                _run(
                    root,
                    "generate.py",
                    "--project-root", root,
                    "--config", MODELS[model]["prefix"],
                    "--datasets-dir", datasets,
                    "--readiness-manifest", readiness,
                    "--output-dir", generation / model,
                    "--resume",
                    "--shard-count", args.shard_count,
                    "--shard-index", args.shard_index,
                )
        elif args.stage == "extract-harp-prefix-features":
            _run(
                root,
                "extract_features.py",
                "--run-dir", generation,
                "--output-dir", features,
                "--prefix-length", 512,
                "--workers", 1,
            )
        elif args.stage == "build-harp-extension-manifest":
            source = manifests / "harp_source_prefix_manifest.json"
            source_args: list[object] = [
                "--generation-dir", generation,
                "--dataset-manifest", datasets / "dataset_manifest.json",
                "--output", source,
                "--base-seed", 41,
                "--dataset-schema-version", "phase05_harp_external_cohort_v1",
                "--output-schema-version", "phase05_harp_source_prefix_manifest_v1",
                "--phase-label", "Phase 5",
            ]
            for model in selected:
                source_args.extend(("--model-key", MODELS[model]["key"]))
            _run(root, "build_phase04e_source_manifest.py", *source_args)
            _run(
                root,
                "build_uncertainty_extension_manifest.py",
                "--probe-manifest", source,
                "--generation-dir", generation,
                "--prefix-features", features / "features_prefix_512.parquet",
                "--reference-prefix-features", artifacts / "phase_04b" / "features" / "features_prefix_512.parquet",
                "--output", manifests / "harp_extension_manifest.json",
                "--continuations", 1,
                "--final-answer-reserve", 4096,
                "--maximum-total-generated-tokens", 29696,
                "--protocol-schema-version", "phase05_breakthrough_controller_v1",
            )
        elif args.stage == "freeze-harp-routing":
            _run(
                root,
                "apply_phase05_breakthrough_controller.py",
                "--policy", policy / "phase05_frozen_policy.json",
                "--forecasters", policy / "phase05_forecasters.joblib",
                "--controller", policy / "phase05_controller.joblib",
                "--prefix-features", features / "features_prefix_512.parquet",
                "--extension-manifest", manifests / "harp_extension_manifest.json",
                "--dataset-manifest", datasets / "dataset_manifest.json",
                "--output", manifests / "harp_routing_manifest.json",
            )
        elif args.stage == "generate-harp-arms":
            if not (manifests / "harp_routing_manifest.json").exists():
                raise RuntimeError("Freeze HARP routing before any arm generation")
            for model in selected:
                _run(
                    root,
                    "generate_uncertainty_extensions.py",
                    "--project-root", root,
                    "--config", MODELS[model]["extension"],
                    "--readiness-manifest", readiness,
                    "--base-run-dir", generation,
                    "--extension-manifest", manifests / "harp_extension_manifest.json",
                    "--output-dir", branches / model,
                    "--resume",
                    "--shard-count", args.shard_count,
                    "--shard-index", args.shard_index,
                )
        elif args.stage == "validate-harp-arms":
            branch_args: list[object] = []
            for model in selected:
                branch_args.extend(("--extension-dir", branches / model))
                branch_args.extend(("--model-key", MODELS[model]["key"]))
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *branch_args,
                "--extension-manifest", manifests / "harp_extension_manifest.json",
                "--output-dir", validation,
                "--bootstrap-repetitions", args.bootstrap_repetitions,
            )
        elif args.stage == "evaluate-harp":
            _run(
                root,
                "evaluate_phase05_breakthrough_controller.py",
                "--pairs", validation / "uncertainty_extension_pairs.parquet",
                "--validation", validation / "uncertainty_extension_validation.json",
                "--routing-manifest", manifests / "harp_routing_manifest.json",
                "--output-dir", analysis,
                "--bootstrap-repetitions", args.bootstrap_repetitions,
            )
    except BaseException:
        write_json_atomic(checkpoint, {"stage": args.stage, "status": "failed_or_interrupted", "models": selected})
        raise
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "complete", "models": selected})


if __name__ == "__main__":
    main()
