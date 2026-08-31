#!/usr/bin/env python
"""Freeze an outcome-independent matched cohort for sparse breakthrough probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from reasonbench.constants import (
    PHASE4_BREAKTHROUGH_ANCHORS,
    PHASE4_CONTINUATIONS_PER_ANCHOR,
    PHASE4_SUCCESS_BASIN_THRESHOLD,
)
from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.storage import read_json, sha256_file, write_json_atomic

MODEL_KEYS = (
    "gemma4_e4b_mlx_4bit",
    "qwen35_4b_mlx_4bit",
    "ministral3_3b_mlx_4bit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem-count", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--selection-seed", type=int, default=20260816)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--anchor", action="append", type=int, default=[])
    parser.add_argument("--continuations", type=int, default=PHASE4_CONTINUATIONS_PER_ANCHOR)
    parser.add_argument("--threshold", type=float, default=PHASE4_SUCCESS_BASIN_THRESHOLD)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--reasoning-continuation-budget", type=int, default=1024)
    parser.add_argument("--final-answer-reserve", type=int, default=512)
    return parser.parse_args()


def _balanced_problem_ids(
    records: list[dict], *, problem_count: int, selection_seed: int
) -> list[str]:
    if problem_count < 5 or problem_count % 5:
        raise ValueError("problem-count must be at least 5 and divisible by five MATH levels")
    by_level: dict[int, list[str]] = {level: [] for level in range(1, 6)}
    for record in records:
        level = int(record["level"])
        if level in by_level:
            by_level[level].append(str(record["problem_id"]))
    per_level = problem_count // 5
    selected: list[str] = []
    for level in range(1, 6):
        candidates = sorted(set(by_level[level]))
        if len(candidates) < per_level:
            raise ValueError(
                f"Need {per_level} common level-{level} problems, found {len(candidates)}"
            )
        rng = random.Random(f"{selection_seed}:level:{level}")
        rng.shuffle(candidates)
        selected.extend(candidates[:per_level])
    return sorted(selected)


def _selection_candidates_by_level(records: list[dict]) -> dict[str, list[str]]:
    return {
        str(level): sorted(
            {
                str(record["problem_id"])
                for record in records
                if int(record["level"]) == level
            }
        )
        for level in range(1, 6)
    }


def main() -> None:
    args = parse_args()
    model_keys = tuple(args.model_key) or MODEL_KEYS
    anchors = tuple(sorted(set(args.anchor or PHASE4_BREAKTHROUGH_ANCHORS)))
    if not anchors or any(anchor <= 0 for anchor in anchors):
        raise ValueError("Breakthrough anchors must be positive")
    if args.continuations < 2:
        raise ValueError("At least two continuations per anchor are required")
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if args.refinement_rounds < 0:
        raise ValueError("refinement-rounds must be non-negative")
    if args.reasoning_continuation_budget <= 0 or args.final_answer_reserve <= 0:
        raise ValueError("Continuation and answer budgets must be positive")

    indexed: dict[tuple[str, str], dict] = {}
    for marker in sorted(args.generation_dir.rglob("complete.json")):
        if not verify_trajectory_payload(marker.parent):
            raise RuntimeError(f"Corrupt base trajectory: {marker.parent}")
        metadata = read_json(marker.parent / "metadata.json")
        if str(metadata.get("model_key")) not in model_keys:
            continue
        if int(metadata.get("seed", -1)) != args.base_seed:
            continue
        key = (str(metadata["model_key"]), str(metadata["problem_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate matched base trajectory for {key}")
        indexed[key] = {
            "run_id": str(metadata["run_id"]),
            "model_key": str(metadata["model_key"]),
            "problem_id": str(metadata["problem_id"]),
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

    common_ids = set.intersection(
        *[
            {problem_id for model, problem_id in indexed if model == model_key}
            for model_key in model_keys
        ]
    )
    common_records = [
        indexed[(model_keys[0], problem_id)] for problem_id in sorted(common_ids)
    ]
    selected_ids = _balanced_problem_ids(
        common_records,
        problem_count=args.problem_count,
        selection_seed=args.selection_seed,
    )
    for problem_id in selected_ids:
        matched = [indexed[(model_key, problem_id)] for model_key in model_keys]
        identities = {
            (
                record["dataset"],
                record["research_split"],
                record["level"],
                record["category"],
                record["dataset_bundle_sha256"],
            )
            for record in matched
        }
        if len(identities) != 1:
            raise ValueError(
                f"Matched models disagree on immutable problem metadata for {problem_id}"
            )
    trajectories = [
        indexed[(model_key, problem_id)]
        for problem_id in selected_ids
        for model_key in model_keys
    ]
    pilot_problem_ids = [
        next(
            problem_id
            for problem_id in selected_ids
            if indexed[(model_keys[0], problem_id)]["level"] == level
        )
        for level in range(1, 6)
    ]
    payload = {
        "schema_version": "phase04c_breakthrough_probe_manifest_v2",
        "source_generation_directory": str(args.generation_dir),
        "selection_policy": {
            "analyst_blind_to_phase04b_outcomes": False,
            "outcome_independent_metadata_only": True,
            "matched_across_models": True,
            "selection_fields": ["problem_id", "level", "model_key", "seed"],
            "forbidden_selection_fields": [
                "correct",
                "finish_reason",
                "generated_tokens",
                "entropy",
                "trajectory_length",
            ],
            "selection_seed": args.selection_seed,
            "base_seed": args.base_seed,
            "ordered_candidates_by_level": _selection_candidates_by_level(
                common_records
            ),
        },
        "probe_protocol": {
            "anchors": list(anchors),
            "continuations_per_anchor": args.continuations,
            "success_threshold": args.threshold,
            "require_next_anchor_stability": True,
            "refinement_rounds": args.refinement_rounds,
            "reasoning_continuation_budget": args.reasoning_continuation_budget,
            "final_answer_reserve": args.final_answer_reserve,
            "max_total_generated_tokens": 16384,
        },
        "models": list(model_keys),
        "problem_count": len(selected_ids),
        "trajectory_count": len(trajectories),
        "level_counts": dict(Counter(record["level"] for record in common_records if record["problem_id"] in selected_ids)),
        "problem_ids": selected_ids,
        "pilot_problem_ids": pilot_problem_ids,
        "pilot_run_ids": [
            indexed[(model_key, problem_id)]["run_id"]
            for problem_id in pilot_problem_ids
            for model_key in model_keys
        ],
        "trajectories": trajectories,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["selection_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
