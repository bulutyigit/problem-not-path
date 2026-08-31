#!/usr/bin/env python
"""Freeze and validate the new Phase 4E source-prefix cohort before scoring it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.storage import read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", action="append", required=True)
    parser.add_argument("--base-seed", type=int, default=29)
    parser.add_argument(
        "--dataset-schema-version",
        default="phase04e_heldout_challenge_dataset_v1",
    )
    parser.add_argument(
        "--output-schema-version",
        default="phase04e_source_prefix_manifest_v1",
    )
    parser.add_argument("--phase-label", default="Phase 4E")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_manifest = read_json(args.dataset_manifest)
    if dataset_manifest.get("schema_version") != args.dataset_schema_version:
        raise RuntimeError(
            f"{args.phase_label} requires dataset schema {args.dataset_schema_version}"
        )
    expected_ids = {str(value) for value in dataset_manifest["problem_ids"]}
    model_keys = tuple(args.model_key)
    if len(set(model_keys)) != len(model_keys):
        raise ValueError("model-key values must be unique")
    indexed: dict[tuple[str, str], dict] = {}
    for marker in sorted(args.generation_dir.rglob("complete.json")):
        trajectory = marker.parent
        if not verify_trajectory_payload(trajectory):
            raise RuntimeError(f"Corrupt Phase 4E source trajectory: {trajectory}")
        metadata = read_json(trajectory / "metadata.json")
        model_key = str(metadata.get("model_key"))
        problem_id = str(metadata.get("problem_id"))
        if model_key not in model_keys or problem_id not in expected_ids:
            continue
        if int(metadata.get("seed", -1)) != args.base_seed:
            raise RuntimeError(f"Unexpected base seed in {trajectory}")
        key = (model_key, problem_id)
        if key in indexed:
            raise RuntimeError(f"Duplicate Phase 4E source trajectory: {key}")
        indexed[key] = {
            "run_id": str(metadata["run_id"]),
            "model_key": model_key,
            "problem_id": problem_id,
            "dataset": str(metadata["dataset"]),
            "research_split": str(metadata["research_split"]),
            "level": int(metadata["level"]),
            "category": metadata.get("category"),
            "seed": int(metadata["seed"]),
            "config_hash": str(metadata["config_hash"]),
            "model_revision": str(metadata["model_revision"]),
            "dataset_bundle_sha256": str(metadata["dataset_bundle_sha256"]),
            "source_complete_sha256": sha256_file(marker),
        }
    missing = [
        f"{model_key}:{problem_id}"
        for model_key in model_keys
        for problem_id in sorted(expected_ids)
        if (model_key, problem_id) not in indexed
    ]
    if missing:
        raise RuntimeError(f"Missing Phase 4E source trajectories: {missing}")
    records = [
        indexed[(model_key, problem_id)]
        for problem_id in sorted(expected_ids)
        for model_key in model_keys
    ]
    # A given problem must preserve the same immutable metadata across models.
    for problem_id in sorted(expected_ids):
        signatures = {
            (
                indexed[(model_key, problem_id)]["dataset"],
                indexed[(model_key, problem_id)]["research_split"],
                indexed[(model_key, problem_id)]["level"],
                indexed[(model_key, problem_id)]["category"],
            )
            for model_key in model_keys
        }
        if len(signatures) != 1:
            raise RuntimeError(f"Model metadata disagreement for {problem_id}")
    payload = {
        "schema_version": args.output_schema_version,
        "purpose": "outcome_blind_source_prefixes_for_frozen_compute_policy",
        "source_generation_directory": str(args.generation_dir),
        "source_dataset_manifest": str(args.dataset_manifest),
        "source_dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "base_seed": args.base_seed,
        "models": list(model_keys),
        "problem_count": len(expected_ids),
        "trajectory_count": len(records),
        "problem_ids": sorted(expected_ids),
        "trajectories": records,
        "pilot_run_ids": [],
        "selection_policy": {
            "outcome_independent_metadata_only": True,
            "held_out_from_phase04b_and_phase04c": True,
            "forbidden_selection_fields": [
                "correct",
                "finish_reason",
                "generated_tokens",
                "uncertainty",
                "trajectory_length",
            ],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["selection_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
