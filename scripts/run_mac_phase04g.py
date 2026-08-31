#!/usr/bin/env python
"""Checkpointed runner for Phase 4G continuous-score three-action routing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reasonbench.storage import ensure_directory, write_json_atomic

MODELS = {
    "gemma4": {
        "key": "gemma4_e4b_mlx_4bit",
        "prefix": "configs/experiments/phase_04g_gemma4_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_04g_gemma4_mlx_4bit_extension.yaml",
        "development_extension": "configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml",
    },
    "ministral3": {
        "key": "ministral3_3b_mlx_4bit",
        "prefix": "configs/experiments/phase_04g_ministral3_mlx_4bit_prefix.yaml",
        "extension": "configs/experiments/phase_04g_ministral3_mlx_4bit_extension.yaml",
        "development_extension": "configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml",
    },
}
STAGES = (
    "status", "fit-policy", "prepare-dataset", "generate-prefixes", "extract-prefix-features",
    "score-prefixes", "freeze-routing", "generate-branches", "validate-branches", "analyze-routing",
    "remediate-answers", "validate-remediated", "analyze-remediated",
    "remediate-development-answers", "validate-development-remediated",
    "refit-policy-remediated", "freeze-routing-remediated",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--artifacts-root", type=Path, default=root / "artifacts" / "mac_mlx")
    parser.add_argument("--approve-heldout-run", action="store_true")
    parser.add_argument("--allow-underpowered-policy", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


def _run(root: Path, script: str, *args: object) -> None:
    command = [sys.executable, "-u", str(root / "scripts" / script), *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _status(phase: Path) -> None:
    checkpoint = phase / "pipeline_checkpoint.json"
    if checkpoint.exists():
        print(json.dumps(json.loads(checkpoint.read_text(encoding="utf-8")), indent=2))
    else:
        print("No Phase 4G checkpoint exists yet.")
    for model in MODELS:
        print(f"\n{model}")
        for label, path, complete_key, expected_key in (
            ("prefixes", phase / "generation" / model / "generation_progress.json", "completed_trajectories", "global_expected_trajectories"),
            ("branches", phase / "branches" / model / "extension_progress.json", "completed_branches", "expected_branches"),
            ("remediated answers", phase / "branches_answer_remediated" / model / "answer_remediation_progress.json", "completed_branches", "expected_branches"),
        ):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                print(f"- {label}: {payload.get('status', 'unknown')} ({payload.get(complete_key, 0)}/{payload.get(expected_key, 0)})")
            else:
                print(f"- {label}: not started")
    development_root = phase.parent / "phase_04c" / "uncertainty" / "models_answer_remediated"
    print("\nremediated development answers")
    for model in MODELS:
        path = development_root / model / "answer_remediation_progress.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"- {model}: {payload.get('status', 'unknown')} "
                f"({payload.get('completed_branches', 0)}/{payload.get('expected_branches', 0)})"
            )
        else:
            print(f"- {model}: not started")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    phase = ensure_directory(artifacts / "phase_04g")
    datasets, generation, features = phase / "datasets", phase / "generation", phase / "features"
    manifests, branches = ensure_directory(phase / "manifests"), phase / "branches"
    remediated_branches = phase / "branches_answer_remediated"
    validation, analysis, policy_dir = phase / "validation", phase / "analysis", phase / "policy"
    remediated_validation = phase / "validation_answer_remediated"
    remediated_analysis = phase / "analysis_answer_remediated"
    remediated_policy_dir = phase / "policy_answer_remediated"
    remediated_routing = manifests / "routing_manifest_answer_remediated.json"
    development = artifacts / "phase_04c" / "uncertainty"
    remediated_development_models = development / "models_answer_remediated"
    remediated_development_validation = development / "validation_answer_remediated"
    checkpoint = phase / "pipeline_checkpoint.json"
    if args.stage == "status":
        _status(phase)
        return
    generation_stages = {
        "generate-prefixes",
        "generate-branches",
        "remediate-answers",
        "remediate-development-answers",
    }
    if args.stage in generation_stages and not args.approve_heldout_run:
        raise RuntimeError("Held-out generation requires --approve-heldout-run")
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "running"})
    try:
        if args.stage == "fit-policy":
            model_args = []
            for spec in MODELS.values():
                model_args.extend(("--model-key", spec["key"]))
            _run(root, "fit_phase04g_policy.py", "--development-pairs", artifacts / "phase_04c" / "uncertainty" / "validation_full" / "uncertainty_extension_pairs.parquet", "--output-dir", policy_dir, *model_args)
        elif args.stage == "prepare-dataset":
            _run(root, "prepare_phase04g_challenge.py", "--challenge-bundle", artifacts / "shared" / "datasets_v2" / "high_difficulty_challenge_50.jsonl", "--phase04b-manifest", artifacts / "shared" / "datasets_v2" / "dataset_manifest.json", "--phase04e-manifest", artifacts / "phase_04e" / "datasets" / "dataset_manifest.json", "--output-dir", datasets)
        elif args.stage == "generate-prefixes":
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            for model, spec in MODELS.items():
                _run(root, "generate.py", "--project-root", root, "--config", spec["prefix"], "--datasets-dir", datasets, "--readiness-manifest", readiness, "--output-dir", generation / model, "--resume")
        elif args.stage == "extract-prefix-features":
            _run(root, "extract_features.py", "--run-dir", generation, "--output-dir", features, "--prefix-length", 512, "--workers", 1)
        elif args.stage == "score-prefixes":
            source = manifests / "source_prefix_manifest.json"
            source_args = ["--generation-dir", generation, "--dataset-manifest", datasets / "dataset_manifest.json", "--output", source, "--base-seed", 37, "--dataset-schema-version", "phase04g_heldout_challenge_dataset_v1", "--output-schema-version", "phase04g_source_prefix_manifest_v1", "--phase-label", "Phase 4G"]
            for spec in MODELS.values():
                source_args.extend(("--model-key", spec["key"]))
            _run(root, "build_phase04e_source_manifest.py", *source_args)
            _run(root, "build_uncertainty_extension_manifest.py", "--probe-manifest", source, "--generation-dir", generation, "--prefix-features", features / "features_prefix_512.parquet", "--reference-prefix-features", artifacts / "phase_04b" / "features" / "features_prefix_512.parquet", "--output", manifests / "extension_manifest.json", "--continuations", 4)
        elif args.stage == "freeze-routing":
            extra = ["--allow-underpowered-policy"] if args.allow_underpowered_policy else []
            _run(root, "apply_phase04g_policy.py", "--policy", policy_dir / "phase04g_policy.json", "--extension-manifest", manifests / "extension_manifest.json", "--output", manifests / "routing_manifest.json", *extra)
        elif args.stage == "generate-branches":
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            for model, spec in MODELS.items():
                _run(root, "generate_uncertainty_extensions.py", "--project-root", root, "--config", spec["extension"], "--readiness-manifest", readiness, "--base-run-dir", generation, "--extension-manifest", manifests / "extension_manifest.json", "--output-dir", branches / model, "--resume")
        elif args.stage == "validate-branches":
            extension_args = []
            for model in MODELS:
                extension_args.extend(("--extension-dir", branches / model))
            _run(root, "validate_uncertainty_extensions.py", *extension_args, "--extension-manifest", manifests / "extension_manifest.json", "--output-dir", validation, "--bootstrap-repetitions", args.bootstrap_repetitions)
        elif args.stage == "analyze-routing":
            _run(root, "evaluate_phase04g_routing.py", "--pairs", validation / "uncertainty_extension_pairs.parquet", "--validation", validation / "uncertainty_extension_validation.json", "--routing-manifest", manifests / "routing_manifest.json", "--output-dir", analysis, "--bootstrap-repetitions", args.bootstrap_repetitions)
        elif args.stage == "remediate-answers":
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            for model, spec in MODELS.items():
                _run(
                    root,
                    "remediate_phase04g_answers.py",
                    "--project-root", root,
                    "--config", spec["extension"],
                    "--readiness-manifest", readiness,
                    "--base-run-dir", generation,
                    "--source-extension-dir", branches / model,
                    "--output-dir", remediated_branches / model,
                    "--final-answer-token-limit", 4096,
                    "--resume",
                )
        elif args.stage == "remediate-development-answers":
            readiness = artifacts / "phase_04b" / "preflight" / "smoke" / "model_readiness.json"
            for model, spec in MODELS.items():
                _run(
                    root,
                    "remediate_phase04g_answers.py",
                    "--project-root", root,
                    "--config", spec["development_extension"],
                    "--readiness-manifest", readiness,
                    "--base-run-dir", artifacts / "phase_04b" / "generation",
                    "--source-extension-dir", development / "models" / model,
                    "--output-dir", remediated_development_models / model,
                    "--final-answer-token-limit", 4096,
                    "--resume",
                )
        elif args.stage == "validate-development-remediated":
            extension_args = []
            source_args = []
            model_args = []
            for model, spec in MODELS.items():
                extension_args.extend(("--extension-dir", remediated_development_models / model))
                source_args.extend(("--source-extension-dir", development / "models" / model))
                model_args.extend(("--model-key", spec["key"]))
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *extension_args,
                *source_args,
                *model_args,
                "--extension-manifest", artifacts / "phase_04c" / "manifests" / "uncertainty_extension_manifest.json",
                "--output-dir", remediated_development_validation,
                "--require-answer-remediation",
                "--final-answer-token-limit", 4096,
                "--bootstrap-repetitions", args.bootstrap_repetitions,
            )
        elif args.stage == "refit-policy-remediated":
            model_args = []
            for spec in MODELS.values():
                model_args.extend(("--model-key", spec["key"]))
            _run(
                root,
                "fit_phase04g_policy.py",
                "--development-pairs", remediated_development_validation / "uncertainty_extension_pairs.parquet",
                "--output-dir", remediated_policy_dir,
                *model_args,
            )
        elif args.stage == "freeze-routing-remediated":
            extra = ["--allow-underpowered-policy"] if args.allow_underpowered_policy else []
            _run(
                root,
                "apply_phase04g_policy.py",
                "--policy", remediated_policy_dir / "phase04g_policy.json",
                "--extension-manifest", manifests / "extension_manifest.json",
                "--output", remediated_routing,
                *extra,
            )
        elif args.stage == "validate-remediated":
            extension_args = []
            source_args = []
            for model in MODELS:
                extension_args.extend(("--extension-dir", remediated_branches / model))
                source_args.extend(("--source-extension-dir", branches / model))
            _run(
                root,
                "validate_uncertainty_extensions.py",
                *extension_args,
                *source_args,
                "--extension-manifest", manifests / "extension_manifest.json",
                "--output-dir", remediated_validation,
                "--require-answer-remediation",
                "--final-answer-token-limit", 4096,
                "--bootstrap-repetitions", args.bootstrap_repetitions,
            )
        elif args.stage == "analyze-remediated":
            _run(
                root,
                "evaluate_phase04g_routing.py",
                "--pairs", remediated_validation / "uncertainty_extension_pairs.parquet",
                "--validation", remediated_validation / "uncertainty_extension_validation.json",
                "--routing-manifest", remediated_routing,
                "--output-dir", remediated_analysis,
                "--bootstrap-repetitions", args.bootstrap_repetitions,
            )
    except BaseException:
        write_json_atomic(checkpoint, {"stage": args.stage, "status": "failed_or_interrupted"})
        raise
    write_json_atomic(checkpoint, {"stage": args.stage, "status": "complete"})


if __name__ == "__main__":
    main()
