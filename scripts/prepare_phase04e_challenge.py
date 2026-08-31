#!/usr/bin/env python
"""Freeze a fresh, high-difficulty MATH cohort for adaptive-compute testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasonbench.datasets.loader import build_problem_sample
from reasonbench.datasets.splits import read_problem_bundle, write_problem_bundle
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-bundle", type=Path, required=True)
    parser.add_argument("--source-dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20260818)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 10 or args.sample_size % 2:
        raise ValueError("sample-size must be an even integer of at least 10")
    records = read_problem_bundle(args.challenge_bundle)
    if not records:
        raise ValueError("Challenge bundle is empty")
    if {record.dataset for record in records} != {"math"}:
        raise ValueError("Phase 4E expects an immutable MATH challenge bundle")
    if {record.level for record in records} != {4, 5}:
        raise ValueError("Phase 4E challenge cohort must contain only MATH levels 4 and 5")
    if {record.research_split for record in records} != {"challenge"}:
        raise ValueError("Challenge records must retain research_split='challenge'")

    selected = build_problem_sample(
        records,
        sample_size=args.sample_size,
        seed=args.selection_seed,
        levels=(4, 5),
    )
    output = ensure_directory(args.output_dir)
    data_path, split_path = write_problem_bundle(selected, output, "math_sample")
    source_manifest_sha = sha256_file(args.source_dataset_manifest)
    selected_ids = {record.problem_id for record in selected}
    source_manifest = json.loads(args.source_dataset_manifest.read_text(encoding="utf-8"))
    historical_ids = set(source_manifest.get("problem_ids_by_split", {}).get("train", []))
    historical_ids |= set(source_manifest.get("problem_ids_by_split", {}).get("validation", []))
    historical_ids |= set(source_manifest.get("problem_ids_by_split", {}).get("test", []))
    overlap = selected_ids & historical_ids
    if overlap:
        raise RuntimeError(f"Held-out challenge selection overlaps Phase 4B: {sorted(overlap)}")
    manifest = {
        "schema_version": "phase04e_heldout_challenge_dataset_v1",
        "purpose": "frozen_test_only_adaptive_compute_policy_evaluation",
        "challenge_bundle": str(args.challenge_bundle),
        "challenge_bundle_sha256": sha256_file(args.challenge_bundle),
        "source_phase04b_dataset_manifest": str(args.source_dataset_manifest),
        "source_phase04b_dataset_manifest_sha256": source_manifest_sha,
        "selection_seed": args.selection_seed,
        "sample_size": len(selected),
        "levels": {"4": sum(record.level == 4 for record in selected), "5": sum(record.level == 5 for record in selected)},
        "problem_ids": [record.problem_id for record in selected],
        "data_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "phase04b_overlap_count": len(overlap),
        "selection_outcome_blind": True,
        "forbidden_selection_fields": [
            "model_output",
            "correctness",
            "finish_reason",
            "uncertainty",
            "reasoning_length",
        ],
    }
    write_json_atomic(output / "dataset_manifest.json", manifest)
    print(output / "dataset_manifest.json")


if __name__ == "__main__":
    main()
