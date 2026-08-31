#!/usr/bin/env python
"""Wave 3 step 2: freeze the expansion cohort from screening outcomes.

Amendment: docs/protocol_amendments/2026-08-20-phase-04c-cohort-expansion.md
Selection rule (frozen): total verified successes over the 6 screen samples
(2 models x 3 seeds) in [1, 5]. Emits the frozen cohort JSON (digest over the
selection), the filtered base-generation bundle (screen-bundle splits
preserved), and 16K base-generation configs at seed 11.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from reasonbench.datasets.splits import read_problem_bundle, write_problem_bundle
from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic

MODEL_CONFIGS = {
    "gemma4_e4b_mlx_4bit": "phase_04b_gemma4_mlx_4bit_16k.yaml",
    "ministral3_3b_mlx_4bit": "phase_04b_ministral3_mlx_4bit_16k.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screen-generation-dir", action="append", type=Path, required=True,
                        help="One per screened model")
    parser.add_argument("--screen-datasets-dir", type=Path, required=True)
    parser.add_argument("--output-datasets-dir", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--output-cohort", type=Path, required=True)
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--max-successes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    successes: Counter[str] = Counter()
    samples: Counter[str] = Counter()
    for directory in args.screen_generation_dir:
        for marker in sorted(directory.rglob("complete.json")):
            if not verify_trajectory_payload(marker.parent):
                raise RuntimeError(f"Corrupt screen trajectory: {marker.parent}")
            metadata = read_json(marker.parent / "metadata.json")
            problem_id = str(metadata["problem_id"])
            samples[problem_id] += 1
            successes[problem_id] += int(bool(metadata.get("verification", {}).get("correct")))

    expected = 2 * 3  # two models x three screen seeds
    incomplete = sorted(pid for pid, n in samples.items() if n != expected)
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} problems have != {expected} screen samples; finish the screen "
            f"first (e.g. {incomplete[:3]})"
        )
    selected = sorted(
        pid for pid in samples if args.min_successes <= successes[pid] <= args.max_successes
    )
    if not selected:
        raise RuntimeError("Screen selected zero problems")

    records = read_problem_bundle(args.screen_datasets_dir / "math_sample.jsonl")
    subset = [record for record in records if record.problem_id in set(selected)]
    out = ensure_directory(args.output_datasets_dir)
    data_path, split_path = write_problem_bundle(subset, out, "math_sample")
    write_json_atomic(out / "dataset_manifest.json", {
        "bundle_version": "phase_04c_wave3_expansion_v1",
        "amendment": "2026-08-20-phase-04c-cohort-expansion",
        "screen_datasets_sha256": sha256_file(args.screen_datasets_dir / "math_sample.jsonl"),
        "sample_size": len(subset),
        "research_splits_preserved_from_screen_bundle": True,
        "outcomes_used_for_sampling_or_splitting":
            "screen pass counts only, per frozen amendment rule",
        "math_sample_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
    })

    payload = {
        "schema_version": "phase04c_wave3_expansion_cohort_v1",
        "amendment": "2026-08-20-phase-04c-cohort-expansion",
        "selection_rule": {
            "total_successes_range": [args.min_successes, args.max_successes],
            "screen_samples_per_problem": expected,
        },
        "screen_success_distribution": dict(Counter(successes[pid] for pid in samples)),
        "selected_problem_count": len(selected),
        "selected_problem_ids": selected,
        "selected_success_counts": {pid: successes[pid] for pid in selected},
        "base_generation_seed": args.base_seed,
        "level_counts": dict(Counter(r.level for r in subset)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["selection_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output_cohort, payload)

    configs_dir = ensure_directory(args.configs_dir)
    for model_key, source_name in MODEL_CONFIGS.items():
        source_path = args.project_root / "configs" / "experiments" / source_name
        config = yaml.safe_load(source_path.read_text())
        config["experiment_id"] = f"{config['experiment_id']}_wave3_expansion"
        config["output_subdirectory"] = f"{config['output_subdirectory']}_wave3_expansion"
        config["seeds"] = [args.base_seed]
        for dataset in config["datasets"]:
            dataset["sample_size"] = len(subset)
            # Screened cohorts are not level-balanced; an explicit levels list
            # would make build_problem_sample reject them. Empty levels selects
            # the whole bundle (sample_size == bundle size).
            dataset["levels"] = []
        target = configs_dir / source_name.replace("_16k.yaml", "_wave3_expansion.yaml")
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"config: {target}")
    print(json.dumps({
        "selected": len(selected),
        "distribution": payload["screen_success_distribution"],
        "levels": payload["level_counts"],
        "cohort": str(args.output_cohort),
    }, indent=2))


if __name__ == "__main__":
    main()
