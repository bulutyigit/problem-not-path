#!/usr/bin/env python
"""Validate one phase's trajectory outputs without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from reasonbench.storage import read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-trajectories", type=int)
    parser.add_argument("--minimum-completion-rate", type=float, default=0.98)
    parser.add_argument(
        "--expected-panel-manifest",
        type=Path,
        help=(
            "Immutable dataset manifest for a strict panel audit. When supplied, every "
            "model/problem/seed combination and dataset-manifest hash must match exactly."
        ),
    )
    return parser.parse_args()


def _strict_panel_expectations(manifest_path: Path) -> dict[str, object]:
    manifest = read_json(manifest_path)
    split_ids = manifest.get("problem_ids_by_split", {})
    expected_problems = {
        str(problem_id)
        for split in ("train", "validation", "test")
        for problem_id in split_ids.get(split, [])
    }
    configurations = manifest.get("accepted_model_configs", [])
    expected_models = {
        str(record["model_key"]): str(record["experiment_id"])
        for record in configurations
        if isinstance(record, dict)
        and record.get("model_key")
        and record.get("experiment_id")
    }
    expected_seeds = {int(seed) for seed in manifest.get("frozen_generation_seeds", [])}
    if not expected_problems or not expected_models or not expected_seeds:
        raise ValueError(
            "Expected panel manifest lacks problem IDs, accepted model identities, or frozen seeds"
        )
    expected_keys = {
        (model_key, experiment_id, problem_id, seed)
        for model_key, experiment_id in expected_models.items()
        for problem_id in expected_problems
        for seed in expected_seeds
    }
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "expected_problems": expected_problems,
        "expected_models": expected_models,
        "expected_seeds": expected_seeds,
        "expected_keys": expected_keys,
    }


def main() -> None:
    args = parse_args()
    complete_markers = sorted(
        {
            marker
            for run_directory in args.run_dir
            for marker in run_directory.rglob("complete.json")
        }
    )
    run_ids: list[str] = []
    counts: Counter[str] = Counter()
    problems: set[str] = set()
    signal_rows = 0
    missing_or_mismatched_payloads = 0
    observed_panel_keys: list[tuple[str, str, str, int]] = []
    manifest_hash_mismatches = 0
    observed_metadata: list[dict] = []
    for marker in complete_markers:
        completion = read_json(marker)
        try:
            metadata = read_json(marker.parent / "metadata.json")
        except Exception:
            missing_or_mismatched_payloads += 1
            continue
        observed_metadata.append(metadata)
        declared_files = completion.get("files", {})
        if not declared_files:
            missing_or_mismatched_payloads += 1
        for filename, record in declared_files.items():
            payload = marker.parent / filename
            if (
                not payload.exists()
                or payload.stat().st_size != record.get("size_bytes")
                or sha256_file(payload) != record.get("sha256")
            ):
                missing_or_mismatched_payloads += 1
        run_ids.append(str(metadata["run_id"]))
        problems.add(str(metadata["problem_id"]))
        signal_rows += int(completion.get("token_metric_rows", 0))
        verification = metadata.get("verification", {})
        counts["correct" if verification.get("correct") else "incorrect"] += 1
        counts[f"extraction_{verification.get('extraction_status', 'unknown')}"] += 1
        counts[f"finish_{metadata.get('finish_reason', 'unknown')}"] += 1
        if metadata.get("assigned_reasoning_budget") is not None:
            boundary_kind = (
                "forced"
                if metadata.get("reasoning_boundary_forced")
                else "natural"
            )
            counts[f"reasoning_boundary_{boundary_kind}"] += 1
        counts[f"dataset_{metadata.get('dataset', 'unknown')}"] += 1
        counts[f"model_{metadata.get('model_key', 'unknown')}"] += 1
        observed_panel_keys.append(
            (
                str(metadata.get("model_key", "unknown")),
                str(metadata.get("experiment_id", "unknown")),
                str(metadata.get("problem_id", "unknown")),
                int(metadata.get("seed", -1)),
            )
        )
    strict_panel: dict[str, object] | None = None
    if args.expected_panel_manifest is not None:
        expectations = _strict_panel_expectations(args.expected_panel_manifest)
        expected_keys = expectations["expected_keys"]
        observed_keys = set(observed_panel_keys)
        manifest_hash_mismatches = sum(
            1
            for metadata in observed_metadata
            if metadata.get("dataset_bundle_sha256") != expectations["manifest_sha256"]
        )
        strict_panel = {
            "manifest_path": str(args.expected_panel_manifest),
            "manifest_sha256": expectations["manifest_sha256"],
            "expected_models": expectations["expected_models"],
            "expected_problem_count": len(expectations["expected_problems"]),
            "expected_seeds": sorted(expectations["expected_seeds"]),
            "expected_trajectory_keys": len(expected_keys),
            "observed_trajectory_keys": len(observed_keys),
            "missing_trajectory_keys": len(expected_keys - observed_keys),
            "unexpected_trajectory_keys": len(observed_keys - expected_keys),
            "duplicate_trajectory_keys": len(observed_panel_keys) - len(observed_keys),
            "dataset_manifest_hash_mismatches": manifest_hash_mismatches,
        }
    duplicate_run_ids = len(run_ids) - len(set(run_ids))
    expected = args.expected_trajectories or len(run_ids)
    completion_rate = len(run_ids) / expected if expected else 0.0
    validation = {
        "expected_trajectories": expected,
        "completed_trajectories": len(run_ids),
        "problem_count": len(problems),
        "completion_rate": completion_rate,
        "duplicate_run_ids": duplicate_run_ids,
        "signal_rows": signal_rows,
        "missing_or_mismatched_payloads": missing_or_mismatched_payloads,
        "strict_panel": strict_panel,
        "counts": dict(counts),
        "valid": (
            completion_rate >= args.minimum_completion_rate
            and duplicate_run_ids == 0
            and signal_rows > 0
            and missing_or_mismatched_payloads == 0
            and (
                strict_panel is None
                or (
                    strict_panel["missing_trajectory_keys"] == 0
                    and strict_panel["unexpected_trajectory_keys"] == 0
                    and strict_panel["duplicate_trajectory_keys"] == 0
                    and strict_panel["dataset_manifest_hash_mismatches"] == 0
                )
            )
        ),
    }
    output_dir = args.output_dir or args.run_dir[0]
    write_json_atomic(output_dir / "generation_validation.json", validation)
    print(validation)


if __name__ == "__main__":
    main()
