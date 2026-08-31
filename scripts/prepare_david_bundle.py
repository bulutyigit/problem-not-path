#!/usr/bin/env python
"""Bundle for the David (arXiv:2511.14773) re-analysis.

Amendment: docs/protocol_amendments/2026-08-28-david-reanalysis.md
Level-balanced draw from the pinned MATH revision: 375 per level over
levels {1, 2, 4, 5} = 750 easy + 750 hard, matching the paper's
construction (level 3 excluded, as in the paper). No exclusions: this is
an external reconstruction, independent of our frozen cohorts. Splits are
assigned for infrastructure compatibility only; the frozen analysis uses
repeated random splits per the amendment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.datasets.loader import build_problem_sample, load_problem_records
from reasonbench.datasets.splits import assign_stratified_research_splits, write_problem_bundle
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-datasets-dir", type=Path, required=True,
                        help="Datasets dir carrying the pinned source_revision")
    parser.add_argument("--output-datasets-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--sample-seed", type=int, default=20260829)
    parser.add_argument("--split-seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = read_json(args.source_datasets_dir / "dataset_manifest.json")
    revision = str(source_manifest["source_revision"])
    records = load_problem_records("math", revision=revision)
    sample = build_problem_sample(
        records, sample_size=args.sample_size, seed=args.sample_seed, levels=(1, 2, 4, 5)
    )
    sample = assign_stratified_research_splits(sample, seed=args.split_seed)
    out = ensure_directory(args.output_datasets_dir)
    data_path, split_path = write_problem_bundle(sample, out, "math_sample")
    write_json_atomic(out / "dataset_manifest.json", {
        "bundle_version": "phase_ext_david_v1",
        "amendment": "2026-08-28-david-reanalysis",
        "source_repository": source_manifest.get("source_repository"),
        "source_revision": revision,
        "sample_size": len(sample),
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "levels": [1, 2, 4, 5],
        "outcomes_used_for_sampling_or_splitting": False,
        "math_sample_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
    })
    levels = {}
    for record in sample:
        levels[record.level] = levels.get(record.level, 0) + 1
    print(f"bundle: {data_path} ({len(sample)} problems; per-level {levels})")


if __name__ == "__main__":
    main()
