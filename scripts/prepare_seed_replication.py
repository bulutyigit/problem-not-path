#!/usr/bin/env python
"""Wave 2 assets: bundle subset + seed-12 configs for the supplement problems.

Amendment: docs/protocol_amendments/2026-08-20-phase-04c-cohort-expansion.md
Research splits of the subset are preserved verbatim from the frozen Phase 4b
bundle; no outcome enters this step beyond the already-frozen A3 screen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reasonbench.datasets.splits import read_problem_bundle, write_problem_bundle
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic

MODEL_CONFIGS = {
    "gemma4_e4b_mlx_4bit": "phase_04b_gemma4_mlx_4bit_16k.yaml",
    "ministral3_3b_mlx_4bit": "phase_04b_ministral3_mlx_4bit_16k.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-datasets-dir", type=Path, required=True)
    parser.add_argument("--supplement-manifest", type=Path, required=True)
    parser.add_argument("--output-datasets-dir", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supplement = read_json(args.supplement_manifest)
    wanted = set(map(str, supplement["problem_ids"]))
    records = read_problem_bundle(args.source_datasets_dir / "math_sample.jsonl")
    subset = [record for record in records if record.problem_id in wanted]
    if {record.problem_id for record in subset} != wanted:
        missing = sorted(wanted - {record.problem_id for record in subset})
        raise RuntimeError(f"Supplement problems missing from source bundle: {missing}")

    out = ensure_directory(args.output_datasets_dir)
    data_path, split_path = write_problem_bundle(subset, out, "math_sample")
    write_json_atomic(out / "dataset_manifest.json", {
        "bundle_version": "phase_04c_wave2_seed_replication_v1",
        "amendment": "2026-08-20-phase-04c-cohort-expansion",
        "source_bundle_sha256": sha256_file(args.source_datasets_dir / "math_sample.jsonl"),
        "supplement_manifest_sha256": sha256_file(args.supplement_manifest),
        "sample_size": len(subset),
        "research_splits_preserved_from_source": True,
        "outcomes_used_for_sampling_or_splitting": False,
        "math_sample_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "problem_ids": sorted(wanted),
        "generation_seed": args.seed,
    })

    configs_dir = ensure_directory(args.configs_dir)
    written = []
    for model_key, source_name in MODEL_CONFIGS.items():
        source_path = args.project_root / "configs" / "experiments" / source_name
        config = yaml.safe_load(source_path.read_text())
        config["experiment_id"] = f"{config['experiment_id']}_wave2_seed{args.seed}"
        config["output_subdirectory"] = f"{config['output_subdirectory']}_wave2_seed{args.seed}"
        config["seeds"] = [args.seed]
        for dataset in config["datasets"]:
            dataset["sample_size"] = len(subset)
            # Screened cohorts are not level-balanced; an explicit levels list
            # would make build_problem_sample reject them. Empty levels selects
            # the whole bundle (sample_size == bundle size).
            dataset["levels"] = []
        target = configs_dir / source_name.replace("_16k.yaml", f"_wave2_seed{args.seed}.yaml")
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        written.append(str(target))
    print(f"bundle: {data_path} ({len(subset)} problems)")
    for path in written:
        print(f"config: {path}")


if __name__ == "__main__":
    main()
