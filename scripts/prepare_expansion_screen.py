#!/usr/bin/env python
"""Wave 3 step 1: fresh 150-problem screening bundle + short-budget screen configs.

Amendment: docs/protocol_amendments/2026-08-20-phase-04c-cohort-expansion.md
Downloads the pinned MATH revision recorded in the Phase 4b dataset manifest,
excludes every already-used problem ID (Phase 4b bundle + any --exclude-bundle),
draws a level-balanced sample, assigns stratified research splits BEFORE any
outcome exists, and writes screening configs (3 seeds, 3,072 tokens, no
hidden-state capture).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reasonbench.datasets.loader import build_problem_sample, load_problem_records
from reasonbench.datasets.splits import (
    assign_stratified_research_splits,
    read_problem_bundle,
    write_problem_bundle,
)
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic

MODEL_CONFIGS = {
    "gemma4_e4b_mlx_4bit": "phase_04b_gemma4_mlx_4bit_16k.yaml",
    "ministral3_3b_mlx_4bit": "phase_04b_ministral3_mlx_4bit_16k.yaml",
}
SCREEN_SEEDS = [21, 22, 23]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-datasets-dir", type=Path, required=True,
                        help="Phase 4b datasets dir (pinned revision + exclusion source)")
    parser.add_argument("--exclude-bundle", action="append", type=Path, default=[],
                        help="Additional historical bundles whose problem IDs are excluded")
    parser.add_argument("--output-datasets-dir", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--sample-seed", type=int, default=20260820)
    parser.add_argument("--split-seed", type=int, default=20260821)
    parser.add_argument("--screen-max-new-tokens", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = read_json(args.source_datasets_dir / "dataset_manifest.json")
    revision = str(source_manifest["source_revision"])

    excluded: set[str] = {
        record.problem_id
        for record in read_problem_bundle(args.source_datasets_dir / "math_sample.jsonl")
    }
    for bundle in args.exclude_bundle:
        excluded |= {record.problem_id for record in read_problem_bundle(bundle)}

    records = load_problem_records("math", revision=revision)
    pool = [record for record in records if record.problem_id not in excluded]
    sample = build_problem_sample(
        pool, sample_size=args.sample_size, seed=args.sample_seed, levels=(1, 2, 3, 4, 5)
    )
    sample = assign_stratified_research_splits(sample, seed=args.split_seed)

    out = ensure_directory(args.output_datasets_dir)
    data_path, split_path = write_problem_bundle(sample, out, "math_sample")
    write_json_atomic(out / "dataset_manifest.json", {
        "bundle_version": "phase_04c_wave3_screen_v1",
        "amendment": "2026-08-20-phase-04c-cohort-expansion",
        "source_repository": source_manifest.get("source_repository"),
        "source_revision": revision,
        "sample_size": len(sample),
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "excluded_problem_ids": len(excluded),
        "outcomes_used_for_sampling_or_splitting": False,
        "math_sample_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "screen_seeds": SCREEN_SEEDS,
        "screen_max_new_tokens": args.screen_max_new_tokens,
    })

    configs_dir = ensure_directory(args.configs_dir)
    for model_key, source_name in MODEL_CONFIGS.items():
        source_path = args.project_root / "configs" / "experiments" / source_name
        config = yaml.safe_load(source_path.read_text())
        config["experiment_id"] = f"{config['experiment_id']}_wave3_screen"
        config["output_subdirectory"] = f"{config['output_subdirectory']}_wave3_screen"
        config["seeds"] = list(SCREEN_SEEDS)
        config["model"]["max_new_tokens"] = args.screen_max_new_tokens
        config["model"]["capture_hidden_states"] = False
        for dataset in config["datasets"]:
            dataset["sample_size"] = len(sample)
        target = configs_dir / source_name.replace("_16k.yaml", "_wave3_screen.yaml")
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"config: {target}")
    print(f"bundle: {data_path} ({len(sample)} problems, {len(excluded)} excluded)")


if __name__ == "__main__":
    main()
