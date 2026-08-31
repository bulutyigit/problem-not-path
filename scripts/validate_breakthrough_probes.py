#!/usr/bin/env python
"""Validate and merge sharded Phase 4c breakthrough-probe artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reasonbench.evaluation.breakthrough import AnchorProbe, derive_breakthrough_label
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", action="append", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-completion-rate", type=float, default=1.0)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--model-key", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_directory(args.output_dir)
    manifest = read_json(args.probe_manifest)
    manifest_sha = sha256_file(args.probe_manifest)
    expected_run_ids = (
        {str(run_id) for run_id in manifest.get("pilot_run_ids", [])}
        if args.pilot_only
        else {str(record["run_id"]) for record in manifest["trajectories"]}
    )
    if args.model_key:
        allowed = set(args.model_key)
        expected_run_ids &= {
            str(record["run_id"])
            for record in manifest["trajectories"]
            if str(record["model_key"]) in allowed
        }
    if not expected_run_ids:
        raise ValueError("Probe manifest contains no expected run IDs for this stage")
    summaries: dict[str, dict] = {}
    duplicate_summaries = 0
    for root in args.probe_dir:
        for path in root.rglob("trajectory_probe_summary.json"):
            row = read_json(path)
            run_id = str(row["run_id"])
            if run_id not in expected_run_ids:
                continue
            if run_id in summaries:
                duplicate_summaries += 1
            summaries[run_id] = row

    corrupt_branches = 0
    observed_branch_keys: list[tuple[str, int, int]] = []
    for root in args.probe_dir:
        for marker in root.rglob("branch_complete.json"):
            result_path = marker.parent / "result.json"
            completion = read_json(marker)
            result = read_json(result_path) if result_path.exists() else {}
            if str(result.get("source_run_id", "missing")) not in expected_run_ids:
                continue
            observed_branch_keys.append(
                (
                    str(result.get("source_run_id", "missing")),
                    int(result.get("anchor", -1)),
                    int(result.get("branch_index", -1)),
                )
            )
            if (
                not result_path.exists()
                or completion.get("result_size_bytes") != result_path.stat().st_size
                or completion.get("result_sha256") != sha256_file(result_path)
                or result.get("probe_manifest_sha256") != manifest_sha
            ):
                corrupt_branches += 1
    expected_branch_keys = {
        (str(summary["run_id"]), int(probe["anchor"]), branch_index)
        for summary in summaries.values()
        for probe in summary.get("probes", [])
        for branch_index in range(int(probe["continuations"]))
    }
    observed_branch_key_set = set(observed_branch_keys)
    duplicate_branches = len(observed_branch_keys) - len(observed_branch_key_set)
    missing_branches = expected_branch_keys - observed_branch_key_set
    unexpected_branches = observed_branch_key_set - expected_branch_keys
    observed_run_ids = set(summaries)
    completion_rate = (
        len(observed_run_ids & expected_run_ids) / len(expected_run_ids)
        if expected_run_ids
        else 0.0
    )
    manifest_mismatches = sum(
        summary.get("probe_manifest_sha256") != manifest_sha for summary in summaries.values()
    )
    unexpected = observed_run_ids - expected_run_ids
    missing = expected_run_ids - observed_run_ids
    valid = (
        completion_rate >= args.minimum_completion_rate
        and not missing
        and not unexpected
        and duplicate_summaries == 0
        and corrupt_branches == 0
        and not missing_branches
        and not unexpected_branches
        and duplicate_branches == 0
        and manifest_mismatches == 0
    )
    label_rows = [{key: value for key, value in row.items() if key != "probes"} for row in summaries.values()]
    if label_rows:
        pd.DataFrame(label_rows).sort_values(
            ["problem_id", "model_key", "seed"]
        ).to_parquet(output / "breakthrough_labels.parquet", index=False)
        sensitivity_rows = []
        for summary in summaries.values():
            probes = [
                AnchorProbe(
                    anchor=int(probe["anchor"]),
                    successes=int(probe["successes"]),
                    continuations=int(probe["continuations"]),
                )
                for probe in summary.get("probes", [])
            ]
            for threshold in (0.5, 0.75, 1.0):
                label = derive_breakthrough_label(probes, threshold=threshold)
                sensitivity_rows.append(
                    {
                        "run_id": summary["run_id"],
                        "problem_id": summary["problem_id"],
                        "model_key": summary["model_key"],
                        "threshold": threshold,
                        **label.to_dict(),
                    }
                )
        pd.DataFrame(sensitivity_rows).sort_values(
            ["threshold", "problem_id", "model_key"]
        ).to_parquet(output / "breakthrough_label_sensitivity.parquet", index=False)
    validation = {
        "valid": valid,
        "stage": "pilot" if args.pilot_only else "labeling_cohort",
        "probe_manifest_sha256": manifest_sha,
        "expected_trajectories": len(expected_run_ids),
        "completed_trajectories": len(observed_run_ids & expected_run_ids),
        "completion_rate": completion_rate,
        "missing_trajectories": len(missing),
        "unexpected_trajectories": len(unexpected),
        "duplicate_summaries": duplicate_summaries,
        "expected_branches": len(expected_branch_keys),
        "observed_branches": len(observed_branch_keys),
        "missing_branches": len(missing_branches),
        "unexpected_branches": len(unexpected_branches),
        "duplicate_branches": duplicate_branches,
        "corrupt_branches": corrupt_branches,
        "manifest_mismatches": manifest_mismatches,
        "events_observed": sum(bool(row["event_observed"]) for row in summaries.values()),
        "right_censored": sum(not bool(row["event_observed"]) for row in summaries.values()),
    }
    write_json_atomic(output / "breakthrough_probe_validation.json", validation)
    print(validation)
    if not valid:
        raise RuntimeError("Breakthrough probe validation failed; preserve the Pod for audit")


if __name__ == "__main__":
    main()
