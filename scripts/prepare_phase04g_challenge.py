#!/usr/bin/env python
"""Freeze a fresh Phase 4G MATH cohort excluding all earlier panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.datasets.loader import build_problem_sample
from reasonbench.datasets.splits import read_problem_bundle, write_problem_bundle
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-bundle", type=Path, required=True)
    parser.add_argument("--phase04b-manifest", type=Path, required=True)
    parser.add_argument("--phase04e-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 10 or args.sample_size % 2:
        raise ValueError("sample-size must be an even integer of at least 10")
    records = read_problem_bundle(args.challenge_bundle)
    phase04b = read_json(args.phase04b_manifest)
    phase04e = read_json(args.phase04e_manifest)
    historical_ids = set(phase04e["problem_ids"])
    for values in phase04b.get("problem_ids_by_split", {}).values():
        historical_ids.update(values)
    available = [record for record in records if record.problem_id not in historical_ids]
    selected = build_problem_sample(
        available,
        sample_size=args.sample_size,
        seed=args.selection_seed,
        levels=(4, 5),
    )
    selected_ids = {record.problem_id for record in selected}
    overlap = selected_ids & historical_ids
    if overlap:
        raise RuntimeError(f"Phase 4G selection overlaps historical panels: {sorted(overlap)}")
    output = ensure_directory(args.output_dir)
    data_path, split_path = write_problem_bundle(selected, output, "math_sample")
    manifest = {
        "schema_version": "phase04g_heldout_challenge_dataset_v1",
        "purpose": "fresh_test_only_three_action_routing_evaluation",
        "challenge_bundle": str(args.challenge_bundle),
        "challenge_bundle_sha256": sha256_file(args.challenge_bundle),
        "phase04b_manifest_sha256": sha256_file(args.phase04b_manifest),
        "phase04e_manifest_sha256": sha256_file(args.phase04e_manifest),
        "selection_seed": args.selection_seed,
        "sample_size": len(selected),
        "levels": {
            "4": sum(record.level == 4 for record in selected),
            "5": sum(record.level == 5 for record in selected),
        },
        "problem_ids": [record.problem_id for record in selected],
        "data_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "historical_overlap_count": 0,
        "selection_outcome_blind": True,
    }
    write_json_atomic(output / "dataset_manifest.json", manifest)
    print(output / "dataset_manifest.json")


if __name__ == "__main__":
    main()
