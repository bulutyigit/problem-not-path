#!/usr/bin/env python
"""Freeze the Phase 4C-U nested short/medium/long continuation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from reasonbench.evaluation.compute_extension import (
    PHASE04C_U_ANCHOR,
    PHASE04C_U_CONTINUATIONS_PER_ARM,
    PHASE04C_U_FINAL_ANSWER_RESERVE,
    PHASE04C_U_MAX_TOTAL_GENERATED_TOKENS,
    PHASE04C_U_TOTAL_REASONING_TARGETS,
    UNCERTAINTY_BLOCKS,
    UNCERTAINTY_FEATURES,
    UNCERTAINTY_SCORE_VERSION,
    UNCERTAINTY_SIGNS,
    UNCERTAINTY_TRANSFORMS,
    assign_balanced_uncertainty_strata,
    fit_percentile_references,
    score_uncertainty_components,
    serialize_references,
    validate_compute_extension_protocol,
    validate_phase04c_u_protocol,
)
from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.storage import read_json, sha256_file, write_json_atomic
from reasonbench.verification import verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--prefix-features", type=Path, required=True)
    parser.add_argument(
        "--reference-prefix-features",
        type=Path,
        help=(
            "Optional immutable prefix table used only to fit the training-split ECDF "
            "references. Defaults to --prefix-features. This permits a genuinely held-out "
            "cohort to be scored without fitting its uncertainty scale on itself."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, default=PHASE04C_U_ANCHOR)
    parser.add_argument(
        "--short-total-reasoning-tokens",
        type=int,
        default=PHASE04C_U_TOTAL_REASONING_TARGETS["short"],
    )
    parser.add_argument(
        "--medium-total-reasoning-tokens",
        type=int,
        default=PHASE04C_U_TOTAL_REASONING_TARGETS["medium"],
    )
    parser.add_argument(
        "--long-total-reasoning-tokens",
        type=int,
        default=PHASE04C_U_TOTAL_REASONING_TARGETS["long"],
    )
    parser.add_argument(
        "--final-answer-reserve",
        type=int,
        default=PHASE04C_U_FINAL_ANSWER_RESERVE,
    )
    parser.add_argument(
        "--continuations",
        type=int,
        default=PHASE04C_U_CONTINUATIONS_PER_ARM,
    )
    parser.add_argument(
        "--maximum-total-generated-tokens",
        type=int,
        default=PHASE04C_U_MAX_TOTAL_GENERATED_TOKENS,
    )
    parser.add_argument("--training-split", default="train")
    parser.add_argument(
        "--protocol-schema-version",
        choices=("phase04c_u_v1", "phase05_breakthrough_controller_v1"),
        default="phase04c_u_v1",
    )
    return parser.parse_args()


def _verify_probe_manifest(manifest: dict) -> None:
    declared = str(manifest.get("selection_digest", ""))
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "selection_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not declared or hashlib.sha256(canonical).hexdigest() != declared:
        raise RuntimeError("Source probe manifest selection_digest is invalid")


def _source_index(generation_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for marker in sorted(generation_dir.rglob("complete.json")):
        metadata = read_json(marker.parent / "metadata.json")
        run_id = str(metadata["run_id"])
        if run_id in indexed:
            raise ValueError(f"Duplicate source run_id under {generation_dir}: {run_id}")
        indexed[run_id] = marker.parent
    return indexed


def _reasoning_prefix(trajectory: Path, anchor: int) -> tuple[list[int], int, str]:
    frame = pd.read_parquet(trajectory / "token_metrics.parquet").sort_values("token_index")
    reasoning = frame[frame["segment"] == "thinking"]
    if reasoning.empty:
        reasoning = frame[frame["segment"] == "solution"]
    if len(reasoning) < anchor:
        raise ValueError(f"Trajectory contains {len(reasoning)} reasoning tokens, not {anchor}")
    cutoff = int(reasoning.iloc[anchor - 1]["token_index"])
    prefix = frame[frame["token_index"] <= cutoff]
    if prefix["token_index"].astype(int).tolist() != list(range(cutoff + 1)):
        raise RuntimeError(f"Non-contiguous token prefix in {trajectory}")
    return (
        prefix["token_id"].astype(int).tolist(),
        cutoff,
        "".join(prefix["token_text"].astype(str).tolist()),
    )


def main() -> None:
    args = parse_args()
    if args.anchor <= 0:
        raise ValueError("anchor must be positive")
    total_reasoning_targets = {
        "short": args.short_total_reasoning_tokens,
        "medium": args.medium_total_reasoning_tokens,
        "long": args.long_total_reasoning_tokens,
    }
    if not (
        args.anchor
        < total_reasoning_targets["short"]
        < total_reasoning_targets["medium"]
        < total_reasoning_targets["long"]
    ):
        raise ValueError("Require anchor < short < medium < long total reasoning targets")
    if args.final_answer_reserve <= 0 or args.continuations < 2:
        raise ValueError("Answer reserve must be positive and continuations must be >= 2")
    if (
        total_reasoning_targets["long"] + args.final_answer_reserve
        > args.maximum_total_generated_tokens
    ):
        raise ValueError("The long arm plus answer reserve exceeds the total token cap")
    continuation_budgets = {
        arm: target - args.anchor for arm, target in total_reasoning_targets.items()
    }

    probe_manifest = read_json(args.probe_manifest)
    _verify_probe_manifest(probe_manifest)
    probe_manifest_sha = sha256_file(args.probe_manifest)
    features = pd.read_parquet(args.prefix_features)
    reference_features_path = args.reference_prefix_features or args.prefix_features
    reference_features = pd.read_parquet(reference_features_path)
    required = {
        "run_id",
        "problem_id",
        "model_key",
        "research_split",
        "level",
        "observed_token_count",
        *UNCERTAINTY_FEATURES,
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Prefix feature table is missing columns: {sorted(missing)}")
    reference_missing = required - set(reference_features.columns)
    if reference_missing:
        raise ValueError(
            "Reference prefix feature table is missing columns: "
            f"{sorted(reference_missing)}"
        )
    if features["run_id"].duplicated().any():
        raise ValueError("Prefix feature table must contain one row per run_id")
    # Fit the score only on the prefix risk set; short trajectories are not
    # silently padded to 512 tokens.
    scale_frame = reference_features[
        reference_features["observed_token_count"] >= args.anchor
    ].copy()
    references = fit_percentile_references(
        scale_frame,
        training_split=args.training_split,
    )

    selected_run_ids = {str(record["run_id"]) for record in probe_manifest["trajectories"]}
    selected = features[features["run_id"].astype(str).isin(selected_run_ids)].copy()
    if len(selected) != len(selected_run_ids):
        missing_ids = selected_run_ids - set(selected["run_id"].astype(str))
        raise ValueError(f"Frozen Phase 4C runs are missing prefix features: {sorted(missing_ids)}")
    score_components = score_uncertainty_components(selected, references)
    selected[score_components.columns] = score_components

    source_index = _source_index(args.generation_dir)
    probe_records = {str(record["run_id"]): record for record in probe_manifest["trajectories"]}
    audit_rows: list[dict] = []
    for _, row in selected.sort_values("run_id").iterrows():
        run_id = str(row["run_id"])
        source = source_index.get(run_id)
        if source is None:
            raise FileNotFoundError(f"Frozen base trajectory is missing: {run_id}")
        record = probe_records[run_id]
        if not verify_trajectory_payload(source):
            raise RuntimeError(f"Frozen base trajectory is corrupt: {run_id}")
        if sha256_file(source / "complete.json") != record["source_complete_sha256"]:
            raise RuntimeError(f"Frozen base trajectory changed: {run_id}")
        metadata = read_json(source / "metadata.json")
        observed = int(row["observed_token_count"])
        eligible = observed >= args.anchor
        exclusion_reason = None if eligible else "reasoning_shorter_than_anchor"
        prefix_sha = None
        prefix_cutoff = None
        prefix_token_count = None
        answer_audit = None
        if eligible:
            prefix_ids, prefix_cutoff, prefix_text = _reasoning_prefix(source, args.anchor)
            prefix_token_count = len(prefix_ids)
            prefix_sha = hashlib.sha256(
                json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            verification = verify_answer(
                prefix_text,
                metadata["reference_answer"],
                metadata["dataset"],
            )
            answer_audit = verification.to_dict()
            if verification.correct:
                eligible = False
                exclusion_reason = "correct_answer_already_extractable_at_anchor"
        audit_rows.append(
            {
                "run_id": run_id,
                "problem_id": str(row["problem_id"]),
                "model_key": str(row["model_key"]),
                "research_split": str(row["research_split"]),
                "level": int(row["level"]),
                "category": row.get("category"),
                "base_seed": int(metadata["seed"]),
                "source_config_hash": str(metadata["config_hash"]),
                "model_revision": str(metadata["model_revision"]),
                "source_complete_sha256": str(record["source_complete_sha256"]),
                "observed_token_count": observed,
                "uncertainty_score": float(row["uncertainty_score"]),
                "uncertainty_blocks": {
                    block: float(row[f"uncertainty_block__{block}"])
                    for block in UNCERTAINTY_BLOCKS
                },
                "uncertainty_features": {
                    feature: float(row[feature]) for feature in UNCERTAINTY_FEATURES
                },
                "generated_prefix_sha256": prefix_sha,
                "generated_prefix_cutoff_index": prefix_cutoff,
                "generated_prefix_token_count": prefix_token_count,
                "answer_leak_audit": answer_audit,
                "eligible": eligible,
                "exclusion_reason": exclusion_reason,
            }
        )

    audit = pd.DataFrame(audit_rows)
    eligible = audit[audit["eligible"]].copy()
    eligible["uncertainty_stratum"] = assign_balanced_uncertainty_strata(eligible)
    stratum_by_run = dict(zip(eligible["run_id"], eligible["uncertainty_stratum"], strict=True))
    for record in audit_rows:
        record["uncertainty_stratum"] = stratum_by_run.get(record["run_id"], "ineligible")
        if record["eligible"] and record["uncertainty_stratum"] == "unassigned":
            record["eligible"] = False
            record["exclusion_reason"] = "odd_cell_median_unassigned"

    pilot_run_ids = {str(value) for value in probe_manifest.get("pilot_run_ids", [])}
    final_eligible = [record for record in audit_rows if record["eligible"]]
    for record in final_eligible:
        required_without_boundary = (
            int(record["generated_prefix_token_count"])
            + continuation_budgets["long"]
            + args.final_answer_reserve
        )
        if required_without_boundary >= args.maximum_total_generated_tokens:
            raise ValueError(f"Long arm leaves no forced-boundary slack for {record['run_id']}")
    protocol = {
        "protocol_schema_version": args.protocol_schema_version,
        "primary_anchor": args.anchor,
        "continuations_per_arm": args.continuations,
        "max_total_generated_tokens": args.maximum_total_generated_tokens,
        "nominal_prefix_and_boundary_overhead_reserve": (
            args.maximum_total_generated_tokens
            - total_reasoning_targets["long"]
            - args.final_answer_reserve
        ),
        "final_answer_reserve": args.final_answer_reserve,
        "budget_semantics": "target_total_reasoning_tokens_including_anchor",
        "arms": {
            arm: {
                "target_total_reasoning_tokens": total_reasoning_targets[arm],
                "reasoning_continuation_budget": continuation_budgets[arm],
            }
            for arm in ("short", "medium", "long")
        },
        "paired_branch_seeds": True,
        "nested_token_paths_required": True,
        "answer_available_at_anchor_excluded": True,
        "primary_estimand": "accuracy_long_minus_medium",
        "primary_policy_interaction": ("(long_minus_medium)_high_minus_(long_minus_medium)_low"),
        "secondary_estimands": [
            "accuracy_medium_minus_short",
            "accuracy_long_minus_short",
            "three_point_dose_response",
        ],
    }
    if args.protocol_schema_version == "phase04c_u_v1":
        validate_phase04c_u_protocol(protocol)
    else:
        validate_compute_extension_protocol(protocol)
    payload = {
        "schema_version": (
            "phase05_compute_extension_manifest_v1"
            if args.protocol_schema_version.startswith("phase05")
            else "phase04c_uncertainty_extension_manifest_v3"
        ),
        "source_probe_manifest": str(args.probe_manifest),
        "source_probe_manifest_sha256": probe_manifest_sha,
        "source_generation_directory": str(args.generation_dir),
        "source_prefix_features": str(args.prefix_features),
        "source_prefix_features_sha256": sha256_file(args.prefix_features),
        "uncertainty_reference_prefix_features": str(reference_features_path),
        "uncertainty_reference_prefix_features_sha256": sha256_file(reference_features_path),
        "protocol": protocol,
        "uncertainty_score": {
            "version": UNCERTAINTY_SCORE_VERSION,
            "training_split": args.training_split,
            "features": list(UNCERTAINTY_FEATURES),
            "blocks": {
                block: list(features) for block, features in UNCERTAINTY_BLOCKS.items()
            },
            "feature_transforms": dict(UNCERTAINTY_TRANSFORMS),
            "feature_orientations": {
                feature: ("higher_is_more_uncertain" if sign > 0 else "lower_is_more_uncertain")
                for feature, sign in UNCERTAINTY_SIGNS.items()
            },
            "range": [0.0, 1.0],
            "interpretation": "relative_model_conditional_index_not_failure_probability",
            "formula": (
                "mean over four conceptual blocks of the within-block mean of "
                "ECDF_model_train(oriented_and_transformed_feature)"
            ),
            "references": serialize_references(references),
            "stratification": "deterministic_ranked_halves_within_model_and_level",
        },
        "records": audit_rows,
        "eligible_run_ids": [record["run_id"] for record in final_eligible],
        "pilot_eligible_run_ids": [
            record["run_id"] for record in final_eligible if record["run_id"] in pilot_run_ids
        ],
        "counts": {
            "source_records": len(audit_rows),
            "eligible_records": len(final_eligible),
            "excluded_records": len(audit_rows) - len(final_eligible),
            "eligible_by_model": dict(Counter(record["model_key"] for record in final_eligible)),
            "eligible_by_stratum": dict(
                Counter(record["uncertainty_stratum"] for record in final_eligible)
            ),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["extension_digest"] = hashlib.sha256(canonical).hexdigest()
    write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
