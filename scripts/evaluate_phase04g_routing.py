#!/usr/bin/env python
"""Evaluate the frozen Phase 4G three-action policy on matched branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, read_json, sha256_file

POLICIES = {
    "fixed_short": "Fixed short",
    "fixed_medium": "Fixed medium",
    "fixed_long": "Fixed long",
    "development_model_only": "Development-selected model-only policy",
    "phase04g_policy": "Frozen three-action policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    return parser.parse_args()


def _verify_routing(payload: dict) -> None:
    declared = str(payload.get("routing_digest", ""))
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "routing_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not declared or declared != hashlib.sha256(canonical).hexdigest():
        raise RuntimeError("Routing manifest digest is invalid")


def _development_model_actions(policy: dict) -> dict[str, str]:
    """Freeze a model-only baseline from development data, never held-out outcomes.

    Phase 4G fits thresholds separately for each model.  A pooled frontier could
    therefore look adaptive even when it merely routes Gemma to short and
    Ministral to long.  This baseline exposes that possibility directly.
    """
    actions: dict[str, str] = {}
    arm_order = ("short", "medium", "long")
    for model_key, model_policy in policy["models"].items():
        candidate = model_policy["fit"]["selected_candidate"]
        rates = {arm: float(candidate[f"route_rate_{arm}"]) for arm in arm_order}
        actions[str(model_key)] = max(arm_order, key=lambda arm: (rates[arm], -arm_order.index(arm)))
    return actions


def _materialize(pairs: pd.DataFrame, routing: dict, policy: dict) -> pd.DataFrame:
    assigned = {str(row["run_id"]): str(row["selected_arm"]) for row in routing["assignments"]}
    model_actions = _development_model_actions(policy)
    rows = []
    for policy in POLICIES:
        frame = pairs[["problem_id", "source_run_id", "model_key", "branch_index"]].copy()
        frame["policy"] = policy
        if policy.startswith("fixed_"):
            frame["selected_arm"] = policy.removeprefix("fixed_")
        elif policy == "development_model_only":
            frame["selected_arm"] = frame["model_key"].astype(str).map(model_actions)
        else:
            frame["selected_arm"] = frame["source_run_id"].astype(str).map(assigned)
        if frame["selected_arm"].isna().any():
            raise RuntimeError("Routing manifest does not cover every held-out trajectory")
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
            if f"{arm}_answer_tokens" in pairs.columns
            else int(pairs.at[index, f"{arm}_continuation_tokens"])
            - int(pairs.at[index, f"{arm}_reasoning_tokens"])
            for index, arm in frame["selected_arm"].items()
        ]
        frame["total_generated_tokens"] = [
            int(pairs.at[index, f"{arm}_total_generated_tokens"])
            if f"{arm}_total_generated_tokens" in pairs.columns
            else int(frame.at[index, "reasoning_tokens"] + frame.at[index, "answer_tokens"])
            for index, arm in frame["selected_arm"].items()
        ]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _summary(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for policy, group in frame.groupby("policy", sort=False):
        stats = group.groupby("problem_id", sort=True).agg(
            correct_sum=("correct", "sum"),
            reasoning_sum=("reasoning_tokens", "sum"),
            answer_sum=("answer_tokens", "sum"),
            total_sum=("total_generated_tokens", "sum"),
            count=("correct", "size"),
        )
        indices = rng.integers(0, len(stats), size=(repetitions, len(stats)))
        counts = stats["count"].to_numpy(float)[indices].sum(axis=1)
        accuracy = stats["correct_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        reasoning = stats["reasoning_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        answer = stats["answer_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        total = stats["total_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        rows.append(
            {
                "policy": policy,
                "label": POLICIES[policy],
                "accuracy": float(group["correct"].mean()),
                "accuracy_ci_low": float(np.quantile(accuracy, 0.025)),
                "accuracy_ci_high": float(np.quantile(accuracy, 0.975)),
                "mean_reasoning_tokens": float(group["reasoning_tokens"].mean()),
                "reasoning_tokens_ci_low": float(np.quantile(reasoning, 0.025)),
                "reasoning_tokens_ci_high": float(np.quantile(reasoning, 0.975)),
                "mean_answer_tokens": float(group["answer_tokens"].mean()),
                "answer_tokens_ci_low": float(np.quantile(answer, 0.025)),
                "answer_tokens_ci_high": float(np.quantile(answer, 0.975)),
                "mean_total_generated_tokens": float(group["total_generated_tokens"].mean()),
                "tokens_ci_low": float(np.quantile(total, 0.025)),
                "tokens_ci_high": float(np.quantile(total, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _paired_differences(
    frame: pd.DataFrame,
    *,
    reference_policy: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Problem-cluster bootstrap contrasts against a common policy."""
    pivot = frame.pivot(
        index=["problem_id", "source_run_id", "model_key", "branch_index"],
        columns="policy",
        values=["correct", "total_generated_tokens"],
    )
    if reference_policy not in pivot["correct"]:
        raise RuntimeError(f"Missing reference policy: {reference_policy}")
    rows = []
    for policy in POLICIES:
        if policy == reference_policy:
            continue
        row_deltas = pd.DataFrame(
            {
                "correct_delta": (
                    pivot["correct"][policy].to_numpy(float)
                    - pivot["correct"][reference_policy].to_numpy(float)
                ),
                "token_delta": (
                    pivot["total_generated_tokens"][policy].to_numpy(float)
                    - pivot["total_generated_tokens"][reference_policy].to_numpy(float)
                ),
                "problem_id": pivot.index.get_level_values("problem_id"),
            }
        )
        clusters = row_deltas.groupby("problem_id", sort=True).agg(
            correct_sum=("correct_delta", "sum"),
            token_sum=("token_delta", "sum"),
            count=("correct_delta", "size"),
        )
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
        counts = clusters["count"].to_numpy(float)[indices].sum(axis=1)
        boot_accuracy = clusters["correct_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        boot_tokens = clusters["token_sum"].to_numpy(float)[indices].sum(axis=1) / counts
        rows.append(
            {
                "policy": policy,
                "reference_policy": reference_policy,
                "accuracy_difference": float(row_deltas["correct_delta"].mean()),
                "accuracy_difference_ci_low": float(np.quantile(boot_accuracy, 0.025)),
                "accuracy_difference_ci_high": float(np.quantile(boot_accuracy, 0.975)),
                "token_difference": float(row_deltas["token_delta"].mean()),
                "token_difference_ci_low": float(np.quantile(boot_tokens, 0.025)),
                "token_difference_ci_high": float(np.quantile(boot_tokens, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _plot_by_model(summary: pd.DataFrame, output: Path) -> None:
    models = list(summary["model_key"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    colors = ["#457b9d", "#6c757d", "#e76f51", "#7b2cbf", "#2a9d8f"]
    for axis, model_key in zip(axes, models, strict=True):
        group = summary[summary["model_key"] == model_key]
        for color, row in zip(colors, group.itertuples(), strict=True):
            axis.errorbar(
                row.mean_total_generated_tokens,
                row.accuracy,
                xerr=[[row.mean_total_generated_tokens - row.tokens_ci_low], [row.tokens_ci_high - row.mean_total_generated_tokens]],
                yerr=[[row.accuracy - row.accuracy_ci_low], [row.accuracy_ci_high - row.accuracy]],
                fmt="o", color=color, capsize=3, label=row.label,
            )
        axis.set_title(model_key)
        axis.set_xlabel("Mean total generated tokens")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Final-answer accuracy")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Phase 4G: held-out policy frontier by model", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validation = read_json(args.validation)
    if not validation.get("valid"):
        raise RuntimeError("Held-out branches must validate before policy analysis")
    routing = read_json(args.routing_manifest)
    _verify_routing(routing)
    policy = read_json(Path(routing["policy"]))
    if policy.get("policy_digest") != routing.get("policy_digest"):
        raise RuntimeError("Frozen routing manifest does not match the policy digest")
    pairs = pd.read_parquet(args.pairs)
    outcomes = _materialize(pairs, routing, policy)
    summary = _summary(outcomes, args.bootstrap_repetitions, args.bootstrap_seed)
    summary_by_model = pd.concat(
        [
            _summary(group, args.bootstrap_repetitions, args.bootstrap_seed + index + 1).assign(model_key=model_key)
            for index, (model_key, group) in enumerate(outcomes.groupby("model_key", sort=True))
        ],
        ignore_index=True,
    )
    against_long = _paired_differences(
        outcomes,
        reference_policy="fixed_long",
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed + 10,
    )
    against_model_only = _paired_differences(
        outcomes,
        reference_policy="development_model_only",
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed + 20,
    )
    output = ensure_directory(args.output_dir)
    outcomes.to_parquet(output / "phase04g_policy_outcomes.parquet", index=False)
    summary.to_csv(output / "phase04g_policy_summary.csv", index=False)
    summary_by_model.to_csv(output / "phase04g_policy_summary_by_model.csv", index=False)
    against_long.to_csv(output / "phase04g_policy_vs_fixed_long.csv", index=False)
    against_model_only.to_csv(output / "phase04g_policy_vs_model_only.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    colors = ["#457b9d", "#6c757d", "#e76f51", "#7b2cbf", "#2a9d8f"]
    for color, row in zip(colors, summary.itertuples(), strict=True):
        axis.errorbar(
            row.mean_total_generated_tokens,
            row.accuracy,
            xerr=[[row.mean_total_generated_tokens - row.tokens_ci_low], [row.tokens_ci_high - row.mean_total_generated_tokens]],
            yerr=[[row.accuracy - row.accuracy_ci_low], [row.accuracy_ci_high - row.accuracy]],
            fmt="o",
            color=color,
            capsize=3,
            label=row.label,
        )
    axis.set_xlabel("Mean total generated tokens")
    axis.set_ylabel("Final-answer accuracy")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.2)
    axis.legend()
    axis.set_title("Phase 4G: frozen continuous-score routing on fresh held-out MATH")
    fig.tight_layout()
    fig.savefig(output / "phase04g_policy_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _plot_by_model(summary_by_model, output / "phase04g_policy_frontier_by_model.png")

    indexed = summary.set_index("policy")
    adaptive = indexed.loc["phase04g_policy"]
    long = indexed.loc["fixed_long"]
    policy_actions = _development_model_actions(policy)
    adaptive_vs_long = against_long.set_index("policy").loc["phase04g_policy"]
    adaptive_vs_model_only = against_model_only.set_index("policy").loc["phase04g_policy"]
    report = "\n".join(
        [
            "# Phase 4G held-out three-action routing report",
            "",
            "| Policy | Accuracy | Mean reasoning tokens | Mean final-answer tokens | Mean total generated tokens |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row.label} | {row.accuracy:.1%} "
                f"[{row.accuracy_ci_low:.1%}, {row.accuracy_ci_high:.1%}] | "
                f"{row.mean_reasoning_tokens:,.0f} | {row.mean_answer_tokens:,.0f} | "
                f"{row.mean_total_generated_tokens:,.0f} |"
                for row in summary.itertuples()
            ],
            "",
            f"Adaptive minus fixed long: {adaptive.accuracy - long.accuracy:+.1%} accuracy and "
            f"{adaptive.mean_total_generated_tokens - long.mean_total_generated_tokens:+,.0f} total generated tokens.",
            "",
            "## Essential model-only control",
            "",
            "Thresholds were fitted separately by model. This control freezes one development-selected "
            "arm per model and tests whether the continuous U512 score adds value beyond knowing the model identity.",
            "",
            "| Model | Development-selected fixed arm |",
            "|---|---|",
            *[f"| {model_key} | {arm} |" for model_key, arm in sorted(policy_actions.items())],
            "",
            f"Frozen policy minus fixed long (problem-cluster bootstrap): "
            f"{adaptive_vs_long.accuracy_difference:+.1%} accuracy "
            f"[{adaptive_vs_long.accuracy_difference_ci_low:+.1%}, {adaptive_vs_long.accuracy_difference_ci_high:+.1%}], "
            f"{adaptive_vs_long.token_difference:+,.0f} tokens "
            f"[{adaptive_vs_long.token_difference_ci_low:+,.0f}, {adaptive_vs_long.token_difference_ci_high:+,.0f}].",
            f"Frozen policy minus development-selected model-only control: "
            f"{adaptive_vs_model_only.accuracy_difference:+.1%} accuracy "
            f"[{adaptive_vs_model_only.accuracy_difference_ci_low:+.1%}, {adaptive_vs_model_only.accuracy_difference_ci_high:+.1%}], "
            f"{adaptive_vs_model_only.token_difference:+,.0f} tokens "
            f"[{adaptive_vs_model_only.token_difference_ci_low:+,.0f}, {adaptive_vs_model_only.token_difference_ci_high:+,.0f}].",
            "",
            "Interpretation constraint: a favourable pooled frontier alone is not evidence that U512 "
            "routes individual problems effectively. The frozen policy must also improve on, or make a "
            "meaningful compute--accuracy trade-off relative to, the model-only control.",
            "",
            f"Routing manifest SHA-256: `{sha256_file(args.routing_manifest)}`",
            f"Paired table SHA-256: `{sha256_file(args.pairs)}`",
        ]
    ) + "\n"
    (output / "phase04g_policy_report.md").write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
