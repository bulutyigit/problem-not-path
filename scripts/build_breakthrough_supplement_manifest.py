#!/usr/bin/env python
"""A3: freeze a supplement probe cohort screened on cross-model solvability.

Amendment: docs/protocol_amendments/2026-08-19-phase-04c-probe-sensitivity-and-supplement.md
Selection rule (deterministic, no sampling): every Phase 4b problem outside the
development cohort whose three-model terminal solved-count at 16K equals 1 or 2.
This intentionally departs from metadata-only selection; it conditions on base
generation terminal outcomes (never on probe outcomes, which do not exist for
these problems at freeze time). All supplement estimands are conditional on
this screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.storage import read_json, sha256_file, write_json_atomic

SCREEN_MODEL_KEYS = (
    "gemma4_e4b_mlx_4bit",
    "qwen35_4b_mlx_4bit",
    "ministral3_3b_mlx_4bit",
)
PROBE_MODEL_KEYS = ("gemma4_e4b_mlx_4bit", "ministral3_3b_mlx_4bit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--solved-min", type=int, default=1)
    parser.add_argument("--solved-max", type=int, default=2)
    parser.add_argument(
        "--preselected-manifest", type=Path,
        help=(
            "Skip the three-model outcome screen and take problem_ids (or "
            "selected_problem_ids) from this frozen manifest; used for expansion "
            "waves whose cohort was already screened and frozen elsewhere."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development = read_json(args.development_manifest)
    development_ids = set(map(str, development["problem_ids"]))
    protocol = development["probe_protocol"]
    preselected_ids: set[str] | None = None
    if args.preselected_manifest is not None:
        preselected = read_json(args.preselected_manifest)
        preselected_ids = set(map(str, preselected.get(
            "problem_ids", preselected.get("selected_problem_ids", [])
        )))
        if not preselected_ids:
            raise ValueError("Preselected manifest carries no problem ids")
    index_model_keys = PROBE_MODEL_KEYS if preselected_ids is not None else SCREEN_MODEL_KEYS

    indexed: dict[tuple[str, str], dict] = {}
    solved: dict[str, int] = {}
    for marker in sorted(args.generation_dir.rglob("complete.json")):
        if not verify_trajectory_payload(marker.parent):
            raise RuntimeError(f"Corrupt base trajectory: {marker.parent}")
        metadata = read_json(marker.parent / "metadata.json")
        model_key = str(metadata.get("model_key"))
        if model_key not in index_model_keys:
            continue
        if int(metadata.get("seed", -1)) != args.base_seed:
            continue
        problem_id = str(metadata["problem_id"])
        key = (model_key, problem_id)
        if key in indexed:
            raise ValueError(f"Duplicate matched base trajectory for {key}")
        correct = bool(metadata.get("verification", {}).get("correct"))
        solved[problem_id] = solved.get(problem_id, 0) + int(correct)
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
            "screen_terminal_correct": correct,
        }

    complete_ids = {
        problem_id
        for problem_id in {pid for _, pid in indexed}
        if all((mk, problem_id) in indexed for mk in index_model_keys)
    }
    if preselected_ids is not None:
        missing = sorted(preselected_ids - complete_ids)
        if missing:
            raise RuntimeError(f"Preselected problems lack complete base trajectories: {missing[:5]}")
        selected_ids = sorted(preselected_ids)
    else:
        pool = sorted(complete_ids - development_ids)
        selected_ids = [
            pid for pid in pool if args.solved_min <= solved[pid] <= args.solved_max
        ]
    if not selected_ids:
        raise RuntimeError("Screen selected zero problems; check inputs")
    for problem_id in selected_ids:
        identities = {
            (
                indexed[(mk, problem_id)]["dataset"],
                indexed[(mk, problem_id)]["research_split"],
                indexed[(mk, problem_id)]["level"],
                indexed[(mk, problem_id)]["category"],
                indexed[(mk, problem_id)]["dataset_bundle_sha256"],
            )
            for mk in index_model_keys
        }
        if len(identities) != 1:
            raise ValueError(f"Matched models disagree on problem metadata for {problem_id}")

    trajectories = [
        {k: v for k, v in indexed[(mk, pid)].items() if k != "screen_terminal_correct"}
        for pid in selected_ids
        for mk in PROBE_MODEL_KEYS
    ]
    payload = {
        "schema_version": "phase04c_breakthrough_supplement_manifest_v1",
        "amendment": "2026-08-19-phase-04c-probe-sensitivity-and-supplement",
        "source_generation_directory": str(args.generation_dir),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "selection_policy": (
            {
                "outcome_independent_metadata_only": False,
                "screen": "inherited_from_preselected_manifest",
                "preselected_manifest": str(args.preselected_manifest),
                "preselected_manifest_sha256": sha256_file(args.preselected_manifest),
                "screen_uses_probe_outcomes": False,
                "base_seed": args.base_seed,
                "estimands_conditional_on_screen": True,
            }
            if preselected_ids is not None
            else {
                "outcome_independent_metadata_only": False,
                "screen": "three_model_terminal_solved_count_16k",
                "screen_model_keys": list(SCREEN_MODEL_KEYS),
                "screen_uses_probe_outcomes": False,
                "solved_count_range": [args.solved_min, args.solved_max],
                "base_seed": args.base_seed,
                "excluded_development_problem_ids": sorted(development_ids),
                "estimands_conditional_on_screen": True,
                "pool_size": len(pool),
                "pool_solved_count_distribution": dict(
                    Counter(solved[pid] for pid in pool)
                ),
            }
        ),
        "probe_protocol": dict(protocol),
        "models": list(PROBE_MODEL_KEYS),
        "problem_count": len(selected_ids),
        "trajectory_count": len(trajectories),
        "level_counts": dict(
            Counter(indexed[(PROBE_MODEL_KEYS[0], pid)]["level"] for pid in selected_ids)
        ),
        "problem_ids": selected_ids,
        "problem_screen_solved_counts": {pid: solved[pid] for pid in selected_ids},
        "pilot_problem_ids": [],
        "pilot_run_ids": [],
        "trajectories": trajectories,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["selection_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "problems": len(selected_ids),
        "trajectories": len(trajectories),
        "level_counts": payload["level_counts"],
        "solved_counts": payload["selection_policy"].get("pool_solved_count_distribution"),
    }, indent=2))


if __name__ == "__main__":
    main()
