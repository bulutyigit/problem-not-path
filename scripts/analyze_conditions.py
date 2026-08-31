#!/usr/bin/env python
"""Analyze reasoning mode, assigned budget, or cross-model conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reasonbench.evaluation.conditions import (
    paired_condition_difference,
    summarize_conditions,
)
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic
from reasonbench.visualization import (
    plot_condition_accuracy,
    plot_condition_profile,
    plot_correctness_feature_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--condition-column", required=True)
    parser.add_argument("--paired-left")
    parser.add_argument("--paired-right")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def _coerce_condition(value: str | None, series: pd.Series):
    if value is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    frame = pd.read_parquet(args.features)
    write_json_atomic(
        output_dir / "condition_input_manifest.json",
        {
            "features_path": str(args.features),
            "features_sha256": sha256_file(args.features),
            "rows": len(frame),
        },
    )
    condition = args.condition_column
    if condition not in frame:
        raise ValueError(f"Condition column is missing: {condition}")
    summary = summarize_conditions(
        frame,
        group_columns=[condition, "dataset"],
        repetitions=args.bootstrap_repetitions,
    )
    contrasts = {}
    left = _coerce_condition(args.paired_left, frame[condition])
    right = _coerce_condition(args.paired_right, frame[condition])
    if left is not None and right is not None:
        for metric in [
            "correct",
            "trajectory_token_count",
            "normalized_entropy_mean",
            "geometry_mean_relative_velocity",
        ]:
            contrasts[metric] = paired_condition_difference(
                frame,
                condition_column=condition,
                left=left,
                right=right,
                value_column=metric,
                repetitions=args.bootstrap_repetitions,
            )
    marginal_budget_returns = {}
    if args.phase == "phase_02" and condition == "assigned_reasoning_budget":
        for lower, upper in ((512.0, 2048.0), (2048.0, 8192.0)):
            label = f"{int(lower)}_to_{int(upper)}"
            marginal_budget_returns[label] = {
                metric: paired_condition_difference(
                    frame,
                    condition_column=condition,
                    left=lower,
                    right=upper,
                    value_column=metric,
                    repetitions=args.bootstrap_repetitions,
                )
                for metric in (
                    "correct",
                    "trajectory_token_count",
                    "elapsed_seconds",
                    "reasoning_boundary_forced",
                    "normalized_entropy_mean",
                    "geometry_mean_relative_velocity",
                )
            }
    analysis = {
        "summary": summary,
        "paired_contrasts": contrasts,
        "marginal_budget_returns": marginal_budget_returns,
    }
    write_json_atomic(output_dir / "condition_analysis.json", analysis)
    plot_condition_accuracy(
        frame,
        condition_column=condition,
        output_path=output_dir / "condition_accuracy.png",
    )
    plot_condition_profile(
        frame,
        condition_column=condition,
        output_path=output_dir / "condition_profile.png",
    )
    plot_correctness_feature_profile(
        frame,
        output_path=output_dir / "correctness_feature_profile.png",
    )
    correctness = contrasts.get("correct")
    scientific_outcome = "inconclusive"
    if correctness:
        if correctness["ci_low"] > 0 or correctness["ci_high"] < 0:
            scientific_outcome = "positive"
        else:
            scientific_outcome = "limited"
    phase_decisions = {
        "phase_01": "continue",
        "phase_02": "continue",
        "phase_03": "continue",
    }
    write_json_atomic(
        output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": scientific_outcome,
            "next_decision": phase_decisions.get(args.phase, "review"),
            "summary": (
                f"Condition analysis for {condition} completed with problem-clustered "
                "confidence intervals."
            ),
            "metrics": {
                "trajectories": len(frame),
                "problems": int(frame["problem_id"].nunique()),
                "conditions": int(frame[condition].nunique(dropna=False)),
                "paired_correctness_difference": (
                    correctness["difference"] if correctness else None
                ),
                "marginal_budget_contrasts": len(marginal_budget_returns),
            },
            "warnings": [],
        },
    )


if __name__ == "__main__":
    main()
