#!/usr/bin/env python
"""Score an untouched cohort and freeze Phase 5 routing before outcomes open."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from reasonbench.evaluation.breakthrough_controller import (
    artifact_digest,
    verify_artifact_digest,
)
from reasonbench.storage import read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--forecasters", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--prefix-features", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = read_json(args.policy)
    if not verify_artifact_digest(policy):
        raise RuntimeError("Frozen Phase 5 policy digest is invalid")
    if policy.get("status") != "frozen_for_external_evaluation":
        raise RuntimeError("Phase 5 policy is not frozen for external evaluation")
    if sha256_file(args.forecasters) != policy["forecasters_sha256"]:
        raise RuntimeError("Forecaster artifact differs from the frozen policy")
    if sha256_file(args.controller) != policy["controller_sha256"]:
        raise RuntimeError("Controller artifact differs from the frozen policy")

    dataset_manifest = read_json(args.dataset_manifest)
    if dataset_manifest.get("dataset") != "harp":
        raise RuntimeError("Confirmatory Phase 5 routing requires the frozen HARP cohort")
    if not dataset_manifest.get("selection_outcome_blind"):
        raise RuntimeError("HARP cohort was not declared outcome-blind")
    if int(dataset_manifest.get("math_overlap_count", -1)) != 0:
        raise RuntimeError("HARP cohort overlaps the MATH development panel")

    extension = read_json(args.extension_manifest)
    extension_by_run = {
        str(record["run_id"]): record
        for record in extension["records"]
        if record.get("eligible")
    }
    features = pd.read_parquet(args.prefix_features)
    features = features[features["run_id"].astype(str).isin(extension_by_run)].copy()
    if features["run_id"].duplicated().any():
        raise RuntimeError("External prefix table contains duplicate run IDs")
    expected = set(extension_by_run)
    observed = set(features["run_id"].astype(str))
    if observed != expected:
        raise RuntimeError(
            f"External feature coverage mismatch: missing={len(expected-observed)}, "
            f"unexpected={len(observed-expected)}"
        )
    features["uncertainty_score"] = features["run_id"].astype(str).map(
        {run_id: float(record["uncertainty_score"]) for run_id, record in extension_by_run.items()}
    )
    forecasters = joblib.load(args.forecasters)
    controller = joblib.load(args.controller)
    features["eventual_success_probability"] = forecasters["eventual_success"].predict_proba(
        features
    )
    features["breakthrough_probability_within_512"] = forecasters[
        "breakthrough"
    ].predict_proba(features)
    decisions = controller.choose(features)
    u512 = policy["u512_only_ablation_thresholds"]
    decisions["u512_selected_arm"] = features["uncertainty_score"].map(
        lambda value: (
            "short"
            if float(value) <= float(u512["short_max"])
            else "medium"
            if float(value) <= float(u512["medium_max"])
            else "long"
        )
    )
    scored = pd.concat(
        [
            features[
                [
                    "run_id",
                    "problem_id",
                    "model_key",
                    "dataset",
                    "level",
                    "uncertainty_score",
                    "eventual_success_probability",
                    "breakthrough_probability_within_512",
                ]
            ].reset_index(drop=True),
            decisions.reset_index(drop=True),
        ],
        axis=1,
    )
    if set(scored["dataset"].astype(str)) != {"harp"}:
        raise RuntimeError("External feature table is not exclusively HARP")
    records = scored.sort_values(["problem_id", "model_key"]).to_dict("records")
    payload = {
        "schema_version": "phase05_frozen_external_routing_v1",
        "policy": str(args.policy),
        "policy_sha256": sha256_file(args.policy),
        "policy_artifact_digest": policy["artifact_digest"],
        "forecasters_sha256": sha256_file(args.forecasters),
        "controller_sha256": sha256_file(args.controller),
        "prefix_features_sha256": sha256_file(args.prefix_features),
        "extension_manifest_sha256": sha256_file(args.extension_manifest),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "dataset": "harp",
        "assignments": records,
        "outcome_fields_opened": False,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    write_json_atomic(args.output, payload)
    scored.to_parquet(args.output.with_suffix(".parquet"), index=False)
    print(json.dumps({"assignments": len(records), "routing": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
