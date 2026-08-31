#!/usr/bin/env python
"""Freeze Phase 4G actions before opening any held-out continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from reasonbench.evaluation.adaptive_routing import choose_arm, verify_policy_digest
from reasonbench.storage import read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--breakthrough-scores", type=Path)
    parser.add_argument("--allow-underpowered-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = read_json(args.policy)
    manifest = read_json(args.extension_manifest)
    if not verify_policy_digest(policy):
        raise RuntimeError("Frozen policy digest is invalid")
    if not policy.get("deployment_ready") and not args.allow_underpowered_policy:
        raise RuntimeError(
            "Policy failed its historical validation gate; use --allow-underpowered-policy "
            "only for an explicitly exploratory held-out run"
        )
    breakthrough = None
    if policy["policy_mode"] == "continuous_u512_plus_breakthrough":
        if not args.breakthrough_scores:
            raise RuntimeError("Breakthrough-aware policy requires frozen held-out scores")
        breakthrough = pd.read_parquet(args.breakthrough_scores).set_index("source_run_id")
    assignments = []
    for record in manifest["records"]:
        if not record.get("eligible"):
            continue
        model_key = str(record["model_key"])
        thresholds = policy["models"][model_key]["thresholds"]
        probability = None
        if breakthrough is not None:
            if record["run_id"] not in breakthrough.index:
                raise RuntimeError(f"Missing breakthrough score for {record['run_id']}")
            probability = float(
                breakthrough.at[record["run_id"], "breakthrough_probability_within_512"]
            )
        arm = choose_arm(
            float(record["uncertainty_score"]),
            short_max=float(thresholds["short_max"]),
            medium_max=float(thresholds["medium_max"]),
            breakthrough_probability=probability,
            breakthrough_continue_min=thresholds.get("breakthrough_continue_min"),
        )
        assignments.append(
            {
                "run_id": record["run_id"],
                "problem_id": record["problem_id"],
                "model_key": model_key,
                "uncertainty_score": record["uncertainty_score"],
                "breakthrough_probability_within_512": probability,
                "selected_arm": arm,
            }
        )
    payload = {
        "schema_version": "phase04g_frozen_routing_assignments_v1",
        "policy": str(args.policy),
        "policy_sha256": sha256_file(args.policy),
        "policy_digest": policy["policy_digest"],
        "extension_manifest": str(args.extension_manifest),
        "extension_manifest_sha256": sha256_file(args.extension_manifest),
        "assignments": assignments,
        "outcome_fields_opened": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["routing_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
