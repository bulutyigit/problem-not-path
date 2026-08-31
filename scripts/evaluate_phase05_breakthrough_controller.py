#!/usr/bin/env python
"""Evaluate the frozen breakthrough-aware controller on untouched HARP arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.evaluation.breakthrough_controller import ARMS, verify_artifact_digest
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic

POLICIES = {
    "fixed_short": "Fixed short",
    "fixed_medium": "Fixed medium",
    "fixed_long": "Fixed long",
    "u512_only": "U512-only tertile routing",
    "phase05_controller": "Breakthrough-aware response controller",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _materialize(pairs: pd.DataFrame, routing: dict) -> pd.DataFrame:
    assignment = pd.DataFrame(routing["assignments"])
    required = {"run_id", "selected_arm", "u512_selected_arm"}
    if missing := required - set(assignment.columns):
        raise RuntimeError(f"Routing assignments are missing: {sorted(missing)}")
    if assignment["run_id"].duplicated().any():
        raise RuntimeError("Routing has duplicate run IDs")
    selected = assignment.set_index("run_id")
    rows: list[pd.DataFrame] = []
    for policy in POLICIES:
        frame = pairs[["problem_id", "source_run_id", "model_key", "branch_index"]].copy()
        frame["policy"] = policy
        if policy.startswith("fixed_"):
            frame["selected_arm"] = policy.removeprefix("fixed_")
        else:
            column = "u512_selected_arm" if policy == "u512_only" else "selected_arm"
            frame["selected_arm"] = frame["source_run_id"].astype(str).map(selected[column])
        if frame["selected_arm"].isna().any():
            raise RuntimeError("Frozen routing does not cover every HARP trajectory")
        if not set(frame["selected_arm"]).issubset(ARMS):
            raise RuntimeError("Frozen routing contains an unknown arm")
        frame["correct"] = [
            bool(pairs.at[index, f"{arm}_correct"])
            for index, arm in frame["selected_arm"].items()
        ]
        frame["reasoning_tokens"] = [
            512 + int(pairs.at[index, f"{arm}_reasoning_tokens"])
            for index, arm in frame["selected_arm"].items()
        ]
        frame["answer_tokens"] = [
            int(pairs.at[index, f"{arm}_answer_tokens"])
            for index, arm in frame["selected_arm"].items()
        ]
        frame["total_generated_tokens"] = [
            int(pairs.at[index, f"{arm}_total_generated_tokens"])
            for index, arm in frame["selected_arm"].items()
        ]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _summary(frame: pd.DataFrame, *, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for policy_index, (policy, group) in enumerate(frame.groupby("policy", sort=False)):
        clusters = group.groupby("problem_id", sort=True).agg(
            correct_sum=("correct", "sum"),
            token_sum=("total_generated_tokens", "sum"),
            count=("correct", "size"),
        )
        rng = np.random.default_rng(seed + policy_index)
        sampled = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
        denominators = clusters["count"].to_numpy(float)[sampled].sum(axis=1)
        accuracy = clusters["correct_sum"].to_numpy(float)[sampled].sum(axis=1) / denominators
        tokens = clusters["token_sum"].to_numpy(float)[sampled].sum(axis=1) / denominators
        rows.append(
            {
                "policy": policy,
                "label": POLICIES[policy],
                "accuracy": float(group["correct"].mean()),
                "accuracy_ci_low": float(np.quantile(accuracy, 0.025)),
                "accuracy_ci_high": float(np.quantile(accuracy, 0.975)),
                "mean_total_generated_tokens": float(group["total_generated_tokens"].mean()),
                "tokens_ci_low": float(np.quantile(tokens, 0.025)),
                "tokens_ci_high": float(np.quantile(tokens, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _paired_difference(
    frame: pd.DataFrame,
    *,
    reference: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    pivot = frame.pivot(
        index=["problem_id", "source_run_id", "model_key", "branch_index"],
        columns="policy",
        values=["correct", "total_generated_tokens"],
    )
    rows: list[dict] = []
    for policy in POLICIES:
        if policy == reference:
            continue
        delta = pd.DataFrame(
            {
                "problem_id": pivot.index.get_level_values("problem_id"),
                "accuracy": pivot["correct"][policy].to_numpy(float)
                - pivot["correct"][reference].to_numpy(float),
                "tokens": pivot["total_generated_tokens"][policy].to_numpy(float)
                - pivot["total_generated_tokens"][reference].to_numpy(float),
            }
        )
        clusters = delta.groupby("problem_id", sort=True).agg(
            accuracy_sum=("accuracy", "sum"), tokens_sum=("tokens", "sum"), count=("accuracy", "size")
        )
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
        denominator = clusters["count"].to_numpy(float)[sampled].sum(axis=1)
        accuracy_draws = clusters["accuracy_sum"].to_numpy(float)[sampled].sum(axis=1) / denominator
        token_draws = clusters["tokens_sum"].to_numpy(float)[sampled].sum(axis=1) / denominator
        rows.append(
            {
                "policy": policy,
                "reference": reference,
                "accuracy_difference": float(delta["accuracy"].mean()),
                "accuracy_ci_low": float(np.quantile(accuracy_draws, 0.025)),
                "accuracy_ci_high": float(np.quantile(accuracy_draws, 0.975)),
                "token_difference": float(delta["tokens"].mean()),
                "tokens_ci_low": float(np.quantile(token_draws, 0.025)),
                "tokens_ci_high": float(np.quantile(token_draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validation = read_json(args.validation)
    if not validation.get("valid"):
        raise RuntimeError("HARP branches must pass validation before outcomes open")
    routing = read_json(args.routing_manifest)
    if not verify_artifact_digest(routing):
        raise RuntimeError("Frozen HARP routing digest is invalid")
    if routing.get("outcome_fields_opened") is not False or routing.get("dataset") != "harp":
        raise RuntimeError("Routing was not frozen outcome-blind on HARP")
    pairs = pd.read_parquet(args.pairs)
    outcomes = _materialize(pairs, routing)
    summary = _summary(outcomes, repetitions=args.bootstrap_repetitions, seed=args.seed)
    versus_medium = _paired_difference(
        outcomes,
        reference="fixed_medium",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 10,
    )
    versus_long = _paired_difference(
        outcomes,
        reference="fixed_long",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 20,
    )
    output = ensure_directory(args.output_dir)
    outcomes.to_parquet(output / "phase05_policy_outcomes.parquet", index=False)
    summary.to_csv(output / "phase05_policy_frontier.csv", index=False)
    versus_medium.to_csv(output / "phase05_policy_vs_fixed_medium.csv", index=False)
    versus_long.to_csv(output / "phase05_policy_vs_fixed_long.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    for row in summary.itertuples():
        axis.errorbar(
            row.mean_total_generated_tokens,
            row.accuracy,
            xerr=[[row.mean_total_generated_tokens - row.tokens_ci_low], [row.tokens_ci_high - row.mean_total_generated_tokens]],
            yerr=[[row.accuracy - row.accuracy_ci_low], [row.accuracy_ci_high - row.accuracy]],
            marker="o",
            linestyle="none",
            capsize=3,
            label=row.label,
        )
    axis.set_xlabel("Mean total generated tokens")
    axis.set_ylabel("Final-answer accuracy")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    axis.set_title("Phase 5: one-shot HARP accuracy-compute frontier")
    fig.tight_layout()
    fig.savefig(output / "phase05_harp_policy_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    policy_row = summary.set_index("policy").loc["phase05_controller"]
    versus_medium_row = versus_medium.set_index("policy").loc["phase05_controller"]
    versus_u_row = _paired_difference(
        outcomes,
        reference="u512_only",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 30,
    ).set_index("policy").loc["phase05_controller"]
    report = "\n".join(
        [
            "# Phase 5 breakthrough-aware controller — one-shot HARP report",
            "",
            "The routing manifest was frozen before any HARP continuation outcome was opened.",
            "MATH was used only for model/controller development; HARP is the external test.",
            "",
            "| Policy | Accuracy | Mean total generated tokens |",
            "|---|---:|---:|",
            *[
                f"| {row.label} | {row.accuracy:.1%} [{row.accuracy_ci_low:.1%}, {row.accuracy_ci_high:.1%}] | {row.mean_total_generated_tokens:,.0f} |"
                for row in summary.itertuples()
            ],
            "",
            f"Controller vs fixed medium: {versus_medium_row.accuracy_difference:+.1%} accuracy "
            f"[{versus_medium_row.accuracy_ci_low:+.1%}, {versus_medium_row.accuracy_ci_high:+.1%}], "
            f"{versus_medium_row.token_difference:+,.0f} tokens.",
            "",
            f"Controller vs U512-only: {versus_u_row.accuracy_difference:+.1%} accuracy "
            f"[{versus_u_row.accuracy_ci_low:+.1%}, {versus_u_row.accuracy_ci_high:+.1%}], "
            f"{versus_u_row.token_difference:+,.0f} tokens.",
            "",
            "The primary claim is supported only if the breakthrough-aware controller improves the "
            "accuracy-compute frontier relative to both fixed medium and the frozen U512-only ablation.",
        ]
    )
    (output / "phase05_harp_report.md").write_text(report + "\n", encoding="utf-8")
    audit = {
        "schema_version": "phase05_external_evaluation_v1",
        "routing_sha256": sha256_file(args.routing_manifest),
        "pairs_sha256": sha256_file(args.pairs),
        "validation_sha256": sha256_file(args.validation),
        "rows": len(outcomes),
        "problems": int(outcomes["problem_id"].nunique()),
        "controller_accuracy": float(policy_row.accuracy),
        "outcomes_opened_after_freeze": True,
    }
    write_json_atomic(output / "phase05_evaluation_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
