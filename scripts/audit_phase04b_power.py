#!/usr/bin/env python
"""Audit endpoint support without conditioning descriptive Phase 4b analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reasonbench.storage import ensure_directory, write_json_atomic, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="correct")
    parser.add_argument("--minimum-test-class-problems", type=int, default=5)
    return parser.parse_args()


def _problem_outcomes(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["model_key", "research_split", "level", "problem_id"], dropna=False)[
            target_column
        ]
        .max()
        .reset_index()
    )
    grouped[target_column] = grouped[target_column].astype(bool)
    return grouped


def main() -> None:
    args = parse_args()
    if args.minimum_test_class_problems < 1:
        raise ValueError("minimum-test-class-problems must be positive")
    frame = pd.read_parquet(args.features)
    required = {
        "model_key",
        "research_split",
        "level",
        "problem_id",
        "seed",
        args.target_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing endpoint audit columns: {sorted(missing)}")
    clusters = _problem_outcomes(frame, args.target_column)
    rows: list[dict[str, object]] = []
    for keys, group in clusters.groupby(["model_key", "research_split", "level"], dropna=False):
        model_key, split, level = keys
        positives = int(group[args.target_column].sum())
        rows.append(
            {
                "model_key": model_key,
                "research_split": split,
                "level": None if pd.isna(level) else int(level),
                "problem_clusters": int(len(group)),
                "positive_problem_clusters": positives,
                "negative_problem_clusters": int(len(group) - positives),
                "nondegenerate": positives > 0 and positives < len(group),
            }
        )
    outcome_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["model_key", "research_split", "level"], dropna=False):
        model_key, split, level = keys
        targets = group[args.target_column].astype(bool)
        row: dict[str, object] = {
            "model_key": model_key,
            "research_split": split,
            "level": None if pd.isna(level) else int(level),
            "trajectories": int(len(group)),
            "target_positive": int(targets.sum()),
            "target_negative": int((~targets).sum()),
        }
        if "finish_reason" in group:
            row["finish_reasons"] = {
                str(reason): int(count)
                for reason, count in group["finish_reason"].value_counts().sort_index().items()
            }
        if "parse_status" in group:
            row["parse_failures"] = int(
                (~group["parse_status"].astype(str).isin({"boxed", "final_answer"})).sum()
            )
        outcome_rows.append(row)
    split_nondegenerate = {
        split: bool(group[args.target_column].nunique() == 2)
        for split, group in clusters.groupby("research_split", sort=True)
    }
    required_splits_present = all(
        split in split_nondegenerate for split in ("train", "validation", "test")
    )
    train_validation_test_nondegenerate = required_splits_present and all(
        split_nondegenerate[split] for split in ("train", "validation", "test")
    )
    test_clusters = clusters[clusters["research_split"] == "test"]
    positive_test_clusters = int(
        test_clusters.loc[test_clusters[args.target_column], "problem_id"].nunique()
    )
    negative_test_clusters = int(
        test_clusters.loc[~test_clusters[args.target_column], "problem_id"].nunique()
    )
    test_nondegenerate = (
        test_clusters[args.target_column].nunique() == 2 if not test_clusters.empty else False
    )
    model_test_nondegenerate = {
        str(model): bool(group[args.target_column].nunique() == 2)
        for model, group in test_clusters.groupby("model_key", sort=True)
    }
    model_test_class_counts = {
        str(model): {
            "positive_problem_clusters": int(
                group.loc[group[args.target_column], "problem_id"].nunique()
            ),
            "negative_problem_clusters": int(
                group.loc[~group[args.target_column], "problem_id"].nunique()
            ),
        }
        for model, group in test_clusters.groupby("model_key", sort=True)
    }
    predictor_eligible = (
        min(positive_test_clusters, negative_test_clusters) >= args.minimum_test_class_problems
        and test_nondegenerate
        and train_validation_test_nondegenerate
    )
    seed_rows: list[dict[str, object]] = []
    for (model_key, problem_id), group in frame.groupby(["model_key", "problem_id"], sort=True):
        outcomes = group[args.target_column].astype(bool)
        if group["seed"].nunique() != 2:
            continue
        seed_rows.append(
            {
                "model_key": model_key,
                "problem_id": problem_id,
                "seed_count": int(group["seed"].nunique()),
                "concordant": bool(outcomes.nunique() == 1),
                "disagreed": bool(outcomes.nunique() > 1),
            }
        )
    output_dir = ensure_directory(args.output_dir)
    summary = {
        "technical_status": "passed",
        "target_column": args.target_column,
        "minimum_test_class_problem_clusters": args.minimum_test_class_problems,
        "pooled_test_positive_problem_clusters": positive_test_clusters,
        "pooled_test_negative_problem_clusters": negative_test_clusters,
        "pooled_test_nondegenerate": test_nondegenerate,
        "model_test_nondegenerate": model_test_nondegenerate,
        "model_test_class_counts": model_test_class_counts,
        "predictor_eligible": predictor_eligible,
        "split_nondegenerate": split_nondegenerate,
        "train_validation_test_nondegenerate": train_validation_test_nondegenerate,
        "scientific_outcome": "predictor_eligible" if predictor_eligible else "descriptive_only",
        "decision": (
            "Both terminal-correctness classes have enough held-out problem support for the "
            "simple eventual-success baseline, while Phase 4b remains descriptive."
            if predictor_eligible
            else "Do not fit the eventual-success baseline. Continue with the pre-registered "
            "descriptive trajectory analyses and report sparse class support."
        ),
        "problem_cluster_counts": rows,
        "trajectory_outcome_counts": outcome_rows,
        "two_seed_outcome_concordance": {
            "eligible_problem_clusters": len(seed_rows),
            "concordant_problem_clusters": int(sum(row["concordant"] for row in seed_rows)),
            "disagreed_problem_clusters": int(sum(row["disagreed"] for row in seed_rows)),
        },
        "outcomes_used_for_split": False,
    }
    write_json_atomic(output_dir / "phase04b_power_audit.json", summary)
    lines = [
        "# Phase 4b label-only power audit",
        "",
        f"- Primary endpoint: `{args.target_column}`",
        f"- Pooled positive / negative test problem clusters: **{positive_test_clusters} / {negative_test_clusters}**",
        f"- Required per class: **{args.minimum_test_class_problems}**",
        f"- Predictor eligibility: **{'PASS' if predictor_eligible else 'DESCRIPTIVE ONLY'}**",
        "",
        "This audit intentionally reports labels and protocol metadata only. Phase 4b always "
        "runs its pre-registered descriptive feature/outcome analyses; it does not fit a primary "
        "risk classifier under sparse failure support.",
    ]
    write_text_atomic(output_dir / "phase04b_power_audit.md", "\n".join(lines) + "\n")
    print(output_dir / "phase04b_power_audit.json")


if __name__ == "__main__":
    main()
