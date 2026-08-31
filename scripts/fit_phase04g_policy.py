#!/usr/bin/env python
"""Fit and freeze a transparent three-action Phase 4G routing policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reasonbench.evaluation.adaptive_routing import (
    evaluate_actions,
    fit_three_action_thresholds,
    materialize_actions,
    policy_digest,
)
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", action="append", required=True)
    parser.add_argument("--breakthrough-scores", type=Path)
    parser.add_argument("--max-accuracy-gap", type=float, default=0.05)
    parser.add_argument("--minimum-validation-problems", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_parquet(args.development_pairs)
    pairs = pairs[pairs["model_key"].isin(args.model_key)].copy()
    if pairs.empty:
        raise ValueError("No requested models exist in development pairs")
    breakthrough_column = None
    breakthrough_manifest = None
    if args.breakthrough_scores:
        scores = pd.read_parquet(args.breakthrough_scores)
        required = {"source_run_id", "breakthrough_probability_within_512"}
        if missing := required - set(scores.columns):
            raise ValueError(f"Breakthrough score table is missing: {sorted(missing)}")
        if scores["source_run_id"].duplicated().any():
            raise ValueError("Breakthrough score table must have one row per source_run_id")
        pairs = pairs.merge(scores[list(required)], on="source_run_id", how="left", validate="many_to_one")
        if pairs["breakthrough_probability_within_512"].isna().any():
            raise RuntimeError("Frozen breakthrough scores do not cover all development trajectories")
        breakthrough_column = "breakthrough_probability_within_512"
        breakthrough_manifest = {
            "path": str(args.breakthrough_scores),
            "sha256": sha256_file(args.breakthrough_scores),
        }

    models: dict[str, dict] = {}
    candidate_frames = []
    for model_key, model_frame in pairs.groupby("model_key", sort=True):
        fit_frame = model_frame[model_frame["research_split"].eq("train")].copy()
        validation_frame = model_frame[~model_frame["research_split"].eq("train")].copy()
        if fit_frame.empty or validation_frame.empty:
            raise RuntimeError(f"{model_key} needs non-empty train and historical validation rows")
        winner, candidates = fit_three_action_thresholds(
            fit_frame,
            max_accuracy_gap=args.max_accuracy_gap,
            breakthrough_column=breakthrough_column,
        )
        thresholds = {
            "short_max": float(winner["short_max"]),
            "medium_max": float(winner["medium_max"]),
            "breakthrough_continue_min": (
                None
                if pd.isna(winner["breakthrough_continue_min"])
                else float(winner["breakthrough_continue_min"])
            ),
        }
        validation_actions = materialize_actions(
            validation_frame,
            thresholds,
            breakthrough_column=breakthrough_column,
        )
        validation_metrics = evaluate_actions(validation_frame, validation_actions)
        fixed_long_validation = evaluate_actions(
            validation_frame,
            pd.Series("long", index=validation_frame.index),
        )
        validation_problems = int(validation_frame["problem_id"].nunique())
        validation_gate = (
            validation_problems >= args.minimum_validation_problems
            and validation_metrics["accuracy"]
            >= fixed_long_validation["accuracy"] - args.max_accuracy_gap
        )
        models[model_key] = {
            "thresholds": thresholds,
            "fit": {
                "problems": int(fit_frame["problem_id"].nunique()),
                "branches": len(fit_frame),
                "selected_candidate": winner,
            },
            "historical_validation": {
                "problems": validation_problems,
                "branches": len(validation_frame),
                "policy": validation_metrics,
                "fixed_long": fixed_long_validation,
                "gate_passed": validation_gate,
            },
        }
        candidates.insert(0, "model_key", model_key)
        candidate_frames.append(candidates)

    deployment_ready = all(record["historical_validation"]["gate_passed"] for record in models.values())
    payload = {
        "schema_version": "phase04g_three_action_policy_v1",
        "policy_mode": (
            "continuous_u512_plus_breakthrough" if breakthrough_column else "continuous_u512_only"
        ),
        "development_pairs": str(args.development_pairs),
        "development_pairs_sha256": sha256_file(args.development_pairs),
        "breakthrough_scores": breakthrough_manifest,
        "max_accuracy_gap": args.max_accuracy_gap,
        "selection_objective": (
            "minimum_total_generated_tokens_within_fixed_long_accuracy_gap"
        ),
        "models": models,
        "deployment_ready": deployment_ready,
        "underpowered": not deployment_ready,
        "underpowered_reason": (
            None
            if deployment_ready
            else "historical_validation_has_too_few_problems_or_misses_accuracy_gate"
        ),
    }
    payload["policy_digest"] = policy_digest(payload)
    output = ensure_directory(args.output_dir)
    write_json_atomic(output / "phase04g_policy.json", payload)
    pd.concat(candidate_frames, ignore_index=True).to_csv(
        output / "phase04g_policy_candidate_grid.csv", index=False
    )
    print(output / "phase04g_policy.json")


if __name__ == "__main__":
    main()
