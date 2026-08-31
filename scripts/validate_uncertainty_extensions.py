#!/usr/bin/env python
"""Validate Phase 4C-U branches and estimate paired budget effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reasonbench.evaluation.compute_extension import (
    paired_budget_effects,
    validate_compute_extension_protocol,
)
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic
from scripts.generate_breakthrough_probes import _branch_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", action="append", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--require-answer-remediation", action="store_true")
    parser.add_argument("--source-extension-dir", action="append", type=Path, default=[])
    parser.add_argument("--final-answer-token-limit", type=int, default=4096)
    parser.add_argument("--model-key", action="append", default=[])
    return parser.parse_args()


def _verify_manifest(manifest: dict) -> None:
    declared = str(manifest.get("extension_digest", ""))
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "extension_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not declared or hashlib.sha256(canonical).hexdigest() != declared:
        raise RuntimeError("Extension manifest digest is missing or invalid")
    validate_compute_extension_protocol(manifest.get("protocol", {}))


def _nested_token_path_flags(by_arm: dict[str, dict | pd.Series]) -> tuple[bool, bool]:
    """Return short-in-medium and medium-in-long reasoning-path checks."""

    reasoning_paths: dict[str, list[int]] = {}
    for arm in ("short", "medium", "long"):
        row = by_arm[arm]
        count = int(row["reasoning_continuation_token_count"])
        reasoning_paths[arm] = [int(token) for token in row["continuation_token_ids"][:count]]
    short_path = reasoning_paths["short"]
    medium_path = reasoning_paths["medium"]
    long_path = reasoning_paths["long"]
    return (
        short_path == medium_path[: len(short_path)],
        medium_path == long_path[: len(medium_path)],
    )


def _result_index(roots: list[Path]) -> dict[tuple[str, str, int], tuple[Path, dict]]:
    indexed: dict[tuple[str, str, int], tuple[Path, dict]] = {}
    for root in roots:
        for path in sorted(root.rglob("result.json")):
            marker_path = path.parent / "branch_complete.json"
            if not marker_path.exists():
                raise RuntimeError(f"Source branch completion marker is missing: {path}")
            marker = read_json(marker_path)
            if (
                marker.get("result_sha256") != sha256_file(path)
                or marker.get("result_size_bytes") != path.stat().st_size
            ):
                raise RuntimeError(f"Source branch is corrupt: {path}")
            payload = read_json(path)
            key = (
                str(payload.get("source_run_id", "missing")),
                str(payload.get("budget_arm", "missing")),
                int(payload.get("branch_index", -1)),
            )
            if key in indexed:
                raise ValueError(f"Duplicate source branch for remediation validation: {key}")
            indexed[key] = (path, payload)
    return indexed


def _cluster_bootstrap_intervals(
    paired: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, list[float] | None]:
    contrast_names = (
        "medium_minus_short",
        "long_minus_medium",
        "long_minus_short",
    )
    output_names = (
        *(f"delta_{name}_high" for name in contrast_names),
        *(f"delta_{name}_low" for name in contrast_names),
        *(f"interaction_{name}" for name in contrast_names),
    )
    if repetitions < 1 or paired.empty:
        return {name: None for name in output_names}
    problem_ids = sorted(paired["problem_id"].astype(str).unique())
    if len(problem_ids) < 2:
        return {name: None for name in output_names}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in output_names}
    by_problem = {
        problem_id: paired[paired["problem_id"].astype(str) == problem_id]
        for problem_id in problem_ids
    }
    for _ in range(repetitions):
        sampled = rng.choice(problem_ids, size=len(problem_ids), replace=True)
        blocks = []
        for copy_index, problem_id in enumerate(sampled):
            block = by_problem[str(problem_id)].copy()
            block["bootstrap_problem_id"] = f"{problem_id}:{copy_index}"
            blocks.append(block)
        sample = pd.concat(blocks, ignore_index=True)
        effects = paired_budget_effects(sample)
        for contrast in contrast_names:
            for stratum in ("high", "low"):
                value = effects["strata"][stratum]["contrasts"].get(contrast)
                if value is not None:
                    draws[f"delta_{contrast}_{stratum}"].append(float(value))
            interaction = effects["interactions_high_minus_low"].get(contrast)
            if interaction is not None:
                draws[f"interaction_{contrast}"].append(float(interaction))
    return {
        key: (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else None
        )
        for key, values in draws.items()
    }


def main() -> None:
    args = parse_args()
    if args.require_answer_remediation and not args.source_extension_dir:
        raise ValueError("Remediation validation requires --source-extension-dir")
    manifest = read_json(args.extension_manifest)
    _verify_manifest(manifest)
    manifest_sha = sha256_file(args.extension_manifest)
    protocol = manifest["protocol"]
    eligible_ids = (
        {str(value) for value in manifest.get("pilot_eligible_run_ids", [])}
        if args.pilot_only
        else {str(value) for value in manifest["eligible_run_ids"]}
    )
    records = {
        str(record["run_id"]): record
        for record in manifest["records"]
        if record.get("eligible")
        and str(record["run_id"]) in eligible_ids
        and (not args.model_key or str(record["model_key"]) in set(args.model_key))
    }
    expected = {
        (run_id, arm, branch_index)
        for run_id in records
        for arm in protocol["arms"]
        for branch_index in range(int(protocol["continuations_per_arm"]))
    }
    observed_paths: dict[tuple[str, str, int], Path] = {}
    source_results = _result_index(args.source_extension_dir)
    duplicates = 0
    corrupt = 0
    identity_mismatches = 0
    remediation_mismatches = 0
    result_rows: list[dict] = []
    for root in args.extension_dir:
        for marker in sorted(root.rglob("branch_complete.json")):
            result_path = marker.parent / "result.json"
            completion = read_json(marker)
            result = read_json(result_path) if result_path.exists() else {}
            key = (
                str(result.get("source_run_id", "missing")),
                str(result.get("budget_arm", "missing")),
                int(result.get("branch_index", -1)),
            )
            if key in observed_paths:
                duplicates += 1
            observed_paths[key] = result_path
            if (
                not result_path.exists()
                or completion.get("result_size_bytes") != result_path.stat().st_size
                or completion.get("result_sha256") != sha256_file(result_path)
            ):
                corrupt += 1
                continue
            record = records.get(key[0])
            expected_identity = None
            if record is not None and key[1] in protocol["arms"]:
                arm_protocol = protocol["arms"][key[1]]
                expected_identity = {
                    "source_run_id": key[0],
                    "budget_arm": key[1],
                    "branch_index": key[2],
                    "branch_seed": _branch_seed(key[0], int(protocol["primary_anchor"]), key[2]),
                    "anchor": int(protocol["primary_anchor"]),
                    "generated_prefix_sha256": record["generated_prefix_sha256"],
                    "generated_prefix_token_count": int(record["generated_prefix_token_count"]),
                    "reasoning_continuation_budget": int(
                        arm_protocol["reasoning_continuation_budget"]
                    ),
                    "target_total_reasoning_tokens": int(
                        arm_protocol["target_total_reasoning_tokens"]
                    ),
                    "budget_semantics": protocol["budget_semantics"],
                    "final_answer_reserve": int(protocol["final_answer_reserve"]),
                    "max_total_generated_tokens": int(protocol["max_total_generated_tokens"]),
                    "extension_manifest_sha256": manifest_sha,
                    "uncertainty_score_version": manifest["uncertainty_score"]["version"],
                    "uncertainty_score": record["uncertainty_score"],
                    "uncertainty_stratum": record["uncertainty_stratum"],
                }
                if "protocol_schema_version" in protocol:
                    expected_identity["protocol_schema_version"] = protocol[
                        "protocol_schema_version"
                    ]
            if expected_identity is None or not all(
                result.get(name) == value for name, value in expected_identity.items()
            ):
                identity_mismatches += 1
            if args.require_answer_remediation:
                source_entry = source_results.get(key)
                if source_entry is None:
                    remediation_mismatches += 1
                else:
                    source_path, source = source_entry
                    source_tokens = [int(value) for value in source.get("continuation_token_ids", [])]
                    repaired_tokens = [int(value) for value in result.get("continuation_token_ids", [])]
                    reasoning_count = int(result.get("reasoning_continuation_token_count", -1))
                    inserted_count = int(result.get("inserted_boundary_token_count", -1))
                    derived_answer_count = len(repaired_tokens) - reasoning_count - inserted_count
                    added_count = len(repaired_tokens) - len(source_tokens)
                    source_was_censored = source.get("finish_reason") != "eos"
                    finish_reason = str(result.get("finish_reason", "missing"))
                    remediation_valid = all(
                        (
                            result.get("answer_remediation_version")
                            == "phase04g_answer_remediation_v1",
                            result.get("source_result_sha256") == sha256_file(source_path),
                            int(result.get("final_answer_token_limit", -1))
                            == args.final_answer_token_limit,
                            result.get("verification_before_remediation")
                            == source.get("verification"),
                            int(result.get("original_answer_token_count", -1))
                            == len(source_tokens)
                            - int(source.get("reasoning_continuation_token_count", -1))
                            - int(source.get("inserted_boundary_token_count", -1)),
                            int(result.get("final_answer_token_count", -1))
                            == derived_answer_count,
                            0 <= derived_answer_count <= args.final_answer_token_limit,
                            repaired_tokens[: len(source_tokens)] == source_tokens,
                            int(result.get("additional_answer_token_count", -1)) == added_count,
                            bool(result.get("remediation_applied")) == source_was_censored,
                            (added_count > 0) if source_was_censored else (added_count == 0),
                            finish_reason in {"eos", "answer_limit"},
                            (derived_answer_count == args.final_answer_token_limit)
                            if finish_reason == "answer_limit"
                            else (derived_answer_count <= args.final_answer_token_limit),
                            int(result.get("combined_generated_token_count", -1))
                            == int(result.get("generated_prefix_token_count", -2))
                            + len(repaired_tokens),
                            int(result.get("reasoning_continuation_token_count", -1))
                            == int(source.get("reasoning_continuation_token_count", -2)),
                        )
                    )
                    if not remediation_valid:
                        remediation_mismatches += 1
            result_rows.append(result)

    observed = set(observed_paths)
    missing = expected - observed
    unexpected = observed - expected
    valid = not any(
        (
            missing,
            unexpected,
            duplicates,
            corrupt,
            identity_mismatches,
            remediation_mismatches,
        )
    )
    output = ensure_directory(args.output_dir)
    paired_rows: list[dict] = []
    nested_token_path_mismatches = 0
    if valid:
        frame = pd.DataFrame(result_rows)
        for (run_id, branch_index), group in frame.groupby(
            ["source_run_id", "branch_index"], sort=True
        ):
            by_arm = {str(row["budget_arm"]): row for _, row in group.iterrows()}
            if set(by_arm) != {"short", "medium", "long"}:
                raise RuntimeError(f"Incomplete paired arms for {run_id}:{branch_index}")
            short = by_arm["short"]
            medium = by_arm["medium"]
            long = by_arm["long"]
            if (
                len(
                    {
                        int(short["branch_seed"]),
                        int(medium["branch_seed"]),
                        int(long["branch_seed"]),
                    }
                )
                != 1
            ):
                raise RuntimeError(f"Paired arms use different seeds for {run_id}:{branch_index}")
            short_nested, medium_nested = _nested_token_path_flags(by_arm)
            if not short_nested or not medium_nested:
                nested_token_path_mismatches += 1
            record = records[str(run_id)]
            paired_rows.append(
                {
                    "problem_id": record["problem_id"],
                    "source_run_id": run_id,
                    "model_key": record["model_key"],
                    "level": record["level"],
                    "research_split": record["research_split"],
                    "branch_index": int(branch_index),
                    "branch_seed": int(short["branch_seed"]),
                    "uncertainty_score": float(record["uncertainty_score"]),
                    "uncertainty_stratum": record["uncertainty_stratum"],
                    "short_correct": bool(short["verification"]["correct"]),
                    "medium_correct": bool(medium["verification"]["correct"]),
                    "long_correct": bool(long["verification"]["correct"]),
                    "short_continuation_tokens": int(short["continuation_token_count"]),
                    "medium_continuation_tokens": int(medium["continuation_token_count"]),
                    "long_continuation_tokens": int(long["continuation_token_count"]),
                    "short_reasoning_tokens": int(short["reasoning_continuation_token_count"]),
                    "medium_reasoning_tokens": int(medium["reasoning_continuation_token_count"]),
                    "long_reasoning_tokens": int(long["reasoning_continuation_token_count"]),
                    "short_answer_tokens": int(
                        short.get(
                            "final_answer_token_count",
                            int(short["continuation_token_count"])
                            - int(short["reasoning_continuation_token_count"])
                            - int(short.get("inserted_boundary_token_count", 0)),
                        )
                    ),
                    "medium_answer_tokens": int(
                        medium.get(
                            "final_answer_token_count",
                            int(medium["continuation_token_count"])
                            - int(medium["reasoning_continuation_token_count"])
                            - int(medium.get("inserted_boundary_token_count", 0)),
                        )
                    ),
                    "long_answer_tokens": int(
                        long.get(
                            "final_answer_token_count",
                            int(long["continuation_token_count"])
                            - int(long["reasoning_continuation_token_count"])
                            - int(long.get("inserted_boundary_token_count", 0)),
                        )
                    ),
                    "short_total_generated_tokens": int(
                        short.get(
                            "combined_generated_token_count",
                            int(short["generated_prefix_token_count"])
                            + int(short["continuation_token_count"]),
                        )
                    ),
                    "medium_total_generated_tokens": int(
                        medium.get(
                            "combined_generated_token_count",
                            int(medium["generated_prefix_token_count"])
                            + int(medium["continuation_token_count"]),
                        )
                    ),
                    "long_total_generated_tokens": int(
                        long.get(
                            "combined_generated_token_count",
                            int(long["generated_prefix_token_count"])
                            + int(long["continuation_token_count"]),
                        )
                    ),
                    "short_extraction_status": str(
                        short["verification"].get("extraction_status", "unknown")
                    ),
                    "medium_extraction_status": str(
                        medium["verification"].get("extraction_status", "unknown")
                    ),
                    "long_extraction_status": str(
                        long["verification"].get("extraction_status", "unknown")
                    ),
                    "short_finish_reason": short["finish_reason"],
                    "medium_finish_reason": medium["finish_reason"],
                    "long_finish_reason": long["finish_reason"],
                    "short_reused": bool(short.get("reused_short_branch")),
                    "short_path_nested_in_medium": short_nested,
                    "medium_path_nested_in_long": medium_nested,
                }
            )
    valid = bool(valid and nested_token_path_mismatches == 0)
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_parquet(output / "uncertainty_extension_pairs.parquet", index=False)
    if valid and not paired.empty:
        effects = paired_budget_effects(paired)
        effects["problem_cluster_bootstrap_95ci"] = _cluster_bootstrap_intervals(
            paired,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
    else:
        effects = {
            "paired_branches": 0,
            "strata": {},
            "interactions_high_minus_low": {},
            "primary_contrast": "long_minus_medium",
            "problem_cluster_bootstrap_95ci": {},
        }
    answer_completion = {}
    if args.require_answer_remediation and not paired.empty:
        for arm in ("short", "medium", "long"):
            finish = paired[f"{arm}_finish_reason"].astype(str)
            extraction = paired[f"{arm}_extraction_status"].astype(str)
            answer_completion[arm] = {
                "paired_rows": len(paired),
                "eos_count": int(finish.eq("eos").sum()),
                "answer_limit_count": int(finish.eq("answer_limit").sum()),
                "answer_limit_rate": float(finish.eq("answer_limit").mean()),
                "missing_extraction_count": int(extraction.eq("missing").sum()),
                "missing_extraction_rate": float(extraction.eq("missing").mean()),
            }
    residual_answer_censoring = any(
        record["answer_limit_count"] > 0 for record in answer_completion.values()
    )
    validation = {
        "valid": valid,
        "stage": "mechanical_pilot" if args.pilot_only else "full_extension_cohort",
        "inferential_claim_allowed": bool(valid and not args.pilot_only),
        "estimand": (
            f"correctness_within_reasoning_arm_and_{args.final_answer_token_limit}_answer_tokens"
            if args.require_answer_remediation
            else "correctness_within_configured_generation_budget"
        ),
        "eventual_answer_claim_allowed": bool(
            valid and not args.pilot_only and not residual_answer_censoring
        ),
        "residual_answer_censoring_detected": residual_answer_censoring,
        "answer_completion": answer_completion,
        "extension_manifest_sha256": manifest_sha,
        "eligible_trajectories": len(records),
        "model_keys": sorted({str(record["model_key"]) for record in records.values()}),
        "expected_branches": len(expected),
        "observed_branches": len(observed),
        "missing_branches": len(missing),
        "unexpected_branches": len(unexpected),
        "duplicate_branches": duplicates,
        "corrupt_branches": corrupt,
        "identity_mismatches": identity_mismatches,
        "answer_remediation_required": args.require_answer_remediation,
        "answer_remediation_mismatches": remediation_mismatches,
        "final_answer_token_limit": (
            args.final_answer_token_limit if args.require_answer_remediation else None
        ),
        "nested_token_path_mismatches": nested_token_path_mismatches,
        "effects": effects,
    }
    write_json_atomic(output / "uncertainty_extension_validation.json", validation)
    print(validation)
    if not valid:
        raise RuntimeError("Uncertainty-extension validation failed; preserve artifacts")


if __name__ == "__main__":
    main()
