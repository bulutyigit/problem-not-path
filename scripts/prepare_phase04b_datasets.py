#!/usr/bin/env python
"""Build the immutable level-stratified MATH bundle for fresh Phase 4b."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.datasets import (
    assign_stratified_research_splits,
    build_problem_sample,
    load_problem_records,
    write_problem_bundle,
)
from reasonbench.datasets.loader import DATASET_SOURCES
from reasonbench.datasets.splits import read_problem_bundle
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--math-revision")
    parser.add_argument(
        "--backend-profile",
        choices=("bf16_cuda", "mlx_4bit"),
        default="bf16_cuda",
        help="Freeze model identities appropriate to the selected generation backend.",
    )
    parser.add_argument(
        "--historical-bundle",
        action="append",
        type=Path,
        default=[],
        help=(
            "Previously used MATH bundle(s). Their problem IDs are excluded from both the "
            "fresh confirmation panel and the challenge panel."
        ),
    )
    return parser.parse_args()


def _resolved_revision(requested_revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to pin the Phase 4b dataset") from exc
    information = HfApi().dataset_info(
        repo_id=DATASET_SOURCES["math"]["repository"], revision=requested_revision
    )
    if not information.sha:
        raise RuntimeError("Could not resolve an immutable MATH dataset revision")
    return str(information.sha)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    revision = _resolved_revision(args.math_revision)
    source = load_problem_records("math", revision=revision)
    if not args.historical_bundle:
        raise ValueError(
            "Phase 4b requires at least one historical MATH bundle so its confirmation "
            "panel can be disjoint from prior exploratory problem IDs."
        )
    historical_ids: set[str] = set()
    for bundle in args.historical_bundle:
        if not bundle.exists():
            raise FileNotFoundError(f"Historical bundle requested for exclusion is missing: {bundle}")
        historical_ids.update(record.problem_id for record in read_problem_bundle(bundle))
    fresh_candidates = [record for record in source if record.problem_id not in historical_ids]
    sample = build_problem_sample(
        fresh_candidates,
        sample_size=100,
        seed=args.seed,
        levels=(1, 2, 3, 4, 5),
    )
    assigned = assign_stratified_research_splits(sample, seed=args.seed)
    data_path, split_path = write_problem_bundle(assigned, output_dir, "math_sample")
    main_ids = {record.problem_id for record in assigned}
    challenge_candidates = [
        record
        for record in source
        if record.level in {4, 5}
        and record.problem_id not in main_ids
        and record.problem_id not in historical_ids
    ]
    challenge_sample = build_problem_sample(
        challenge_candidates,
        sample_size=50,
        seed=args.seed + 1,
        levels=(4, 5),
    )
    challenge_records = [replace(record, research_split="challenge") for record in challenge_sample]
    challenge_path, _ = write_problem_bundle(
        challenge_records, output_dir, "high_difficulty_challenge_50"
    )
    per_level_split = {
        str(level): dict(
            Counter(record.research_split for record in assigned if record.level == level)
        )
        for level in range(1, 6)
    }
    project_root = Path(__file__).resolve().parents[1]
    accepted_model_configs = []
    filenames = (
        (
            "phase_04b_gemma4_mlx_4bit_16k.yaml",
            "phase_04b_qwen35_mlx_4bit_16k.yaml",
            "phase_04b_ministral3_mlx_4bit_16k.yaml",
        )
        if args.backend_profile == "mlx_4bit"
        else (
            "phase_04b_gemma4_16k.yaml",
            "phase_04b_qwen35_16k.yaml",
            "phase_04b_ministral3_16k.yaml",
        )
    )
    for filename in filenames:
        config_path = project_root / "configs" / "experiments" / filename
        config = load_experiment_config(config_path)
        accepted_model_configs.append(
            {
                "filename": filename,
                "file_sha256": sha256_file(config_path),
                "experiment_id": config.experiment_id,
                "model_key": config.model.key,
                "max_new_tokens": config.model.max_new_tokens,
                "seeds": list(config.seeds),
            }
        )
    configured_seed_sets = {
        tuple(model_config["seeds"]) for model_config in accepted_model_configs
    }
    if len(configured_seed_sets) != 1:
        raise ValueError(
            "All accepted Phase 4b experiment configs must freeze the same generation seeds"
        )
    frozen_generation_seeds = list(next(iter(configured_seed_sets)))
    manifest = {
        "bundle_version": f"phase_04b_math_stratified_v2_{args.backend_profile}",
        "backend_profile": args.backend_profile,
        "source_repository": DATASET_SOURCES["math"]["repository"],
        "source_revision": revision,
        "sample_size": 100,
        "level_counts": {str(level): sum(record.level == level for record in assigned) for level in range(1, 6)},
        "split_counts_by_level": per_level_split,
        "split_counts": dict(Counter(record.research_split for record in assigned)),
        "split_seed": args.seed,
        "outcomes_used_for_sampling_or_splitting": False,
        "historical_problem_ids_excluded_from_main": len(historical_ids),
        "historical_overlap_with_main": len(main_ids & historical_ids),
        "problem_ids_by_split": {
            split: [record.problem_id for record in assigned if record.research_split == split]
            for split in ("train", "validation", "test")
        },
        "math_sample_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "frozen_generation_seeds": frozen_generation_seeds,
        "accepted_model_configs": accepted_model_configs,
        "challenge_panel": {
            "path": str(challenge_path),
            "sha256": sha256_file(challenge_path),
            "sample_size": 50,
            "levels": {str(level): sum(record.level == level for record in challenge_records) for level in (4, 5)},
            "excluded_main_problem_ids": len(main_ids),
            "excluded_historical_problem_ids": len(historical_ids),
            "problem_ids": [record.problem_id for record in challenge_records],
        },
    }
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)
    print(output_dir / "dataset_manifest.json")


if __name__ == "__main__":
    main()
