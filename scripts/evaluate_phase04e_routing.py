#!/usr/bin/env python
"""Evaluate frozen adaptive-compute policies from matched Phase 4E branches.

Every candidate has both short, medium, and long continuations.  A routing
policy therefore selects one already-generated matched outcome per trajectory;
the reported compute is the selected arm's realized reasoning consumption, not
the experimental cost of generating all counterfactual arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

ANCHOR = 512
POLICIES = {
    "fixed_short": "Short for every prefix",
    "fixed_medium": "Medium for every prefix",
    "fixed_long": "Long for every prefix",
    "u512_selective": "Long only for frozen high-U512 prefixes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args()


def _chosen_arm(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "fixed_short":
        return pd.Series("short", index=frame.index)
    if policy == "fixed_medium":
        return pd.Series("medium", index=frame.index)
    if policy == "fixed_long":
        return pd.Series("long", index=frame.index)
    if policy == "u512_selective":
        return pd.Series(
            np.where(frame["uncertainty_stratum"].eq("high"), "long", "short"),
            index=frame.index,
        )
    raise ValueError(f"Unknown policy: {policy}")


def _materialize_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    result = frame[["problem_id", "model_key", "branch_index", "uncertainty_stratum"]].copy()
    result["policy"] = policy
    result["policy_label"] = POLICIES[policy]
    result["selected_arm"] = _chosen_arm(frame, policy)
    result["correct"] = [
        bool(frame.at[index, f"{arm}_correct"])
        for index, arm in result["selected_arm"].items()
    ]
    result["realized_reasoning_tokens"] = [
        ANCHOR + int(frame.at[index, f"{arm}_reasoning_tokens"])
        for index, arm in result["selected_arm"].items()
    ]
    result["long_route"] = result["selected_arm"].eq("long")
    return result


def _bootstrap_summary(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    if frame["problem_id"].nunique() < 2:
        raise RuntimeError("Need at least two distinct problem clusters for policy analysis")
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for policy, group in frame.groupby("policy", sort=False):
        point_accuracy = float(group["correct"].mean())
        point_tokens = float(group["realized_reasoning_tokens"].mean())
        point_route = float(group["long_route"].mean())
        stats = group.groupby("problem_id", sort=True).agg(
            correct_sum=("correct", "sum"),
            token_sum=("realized_reasoning_tokens", "sum"),
            count=("correct", "size"),
        )
        sampled_indices = rng.integers(0, len(stats), size=(repetitions, len(stats)))
        counts = stats["count"].to_numpy(float)[sampled_indices].sum(axis=1)
        draws_accuracy = stats["correct_sum"].to_numpy(float)[sampled_indices].sum(axis=1) / counts
        draws_tokens = stats["token_sum"].to_numpy(float)[sampled_indices].sum(axis=1) / counts
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICIES[policy],
                "accuracy": point_accuracy,
                "accuracy_ci_low": float(np.quantile(draws_accuracy, 0.025)),
                "accuracy_ci_high": float(np.quantile(draws_accuracy, 0.975)),
                "mean_reasoning_tokens": point_tokens,
                "tokens_ci_low": float(np.quantile(draws_tokens, 0.025)),
                "tokens_ci_high": float(np.quantile(draws_tokens, 0.975)),
                "long_route_rate": point_route,
                "branches": len(group),
                "problems": int(group["problem_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_policy_comparisons(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate paired policy deltas with resampling clustered by problem.

    All policies select a continuation for the same model/problem/branch tuple.
    Resampling those matched tuples together preserves their counterfactual
    pairing while treating the underlying MATH problem as the independent unit.
    """
    value_columns = ["correct", "realized_reasoning_tokens"]
    identifiers = ["problem_id", "model_key", "branch_index"]
    wide = frame.pivot(index=identifiers, columns="policy", values=value_columns)
    if wide.isna().any().any():
        raise RuntimeError("Every matched trajectory must contain every policy")
    if wide.index.get_level_values("problem_id").nunique() < 2:
        raise RuntimeError("Need at least two distinct problem clusters for comparisons")
    comparisons = [
        ("u512_selective", "fixed_long"),
        ("u512_selective", "fixed_medium"),
        ("u512_selective", "fixed_short"),
        ("fixed_long", "fixed_medium"),
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for policy_a, policy_b in comparisons:
        point_accuracy = float((wide[("correct", policy_a)] - wide[("correct", policy_b)]).mean())
        point_tokens = float(
            (wide[("realized_reasoning_tokens", policy_a)] - wide[("realized_reasoning_tokens", policy_b)]).mean()
        )
        deltas = pd.DataFrame(
            {
                "problem_id": wide.index.get_level_values("problem_id").to_numpy(),
                "accuracy": (
                    wide[("correct", policy_a)].astype(float) - wide[("correct", policy_b)].astype(float)
                ).to_numpy(),
                "tokens": (
                    wide[("realized_reasoning_tokens", policy_a)].astype(float)
                    - wide[("realized_reasoning_tokens", policy_b)].astype(float)
                ).to_numpy(),
            }
        )
        stats = deltas.groupby("problem_id", sort=True).agg(
            accuracy_sum=("accuracy", "sum"),
            token_sum=("tokens", "sum"),
            count=("accuracy", "size"),
        )
        sampled_indices = rng.integers(0, len(stats), size=(repetitions, len(stats)))
        counts = stats["count"].to_numpy(float)[sampled_indices].sum(axis=1)
        accuracy_draws = stats["accuracy_sum"].to_numpy(float)[sampled_indices].sum(axis=1) / counts
        token_draws = stats["token_sum"].to_numpy(float)[sampled_indices].sum(axis=1) / counts
        rows.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "comparison": f"{POLICIES[policy_a]} minus {POLICIES[policy_b]}",
                "accuracy_delta_a_minus_b": point_accuracy,
                "accuracy_delta_ci_low": float(np.quantile(accuracy_draws, 0.025)),
                "accuracy_delta_ci_high": float(np.quantile(accuracy_draws, 0.975)),
                "token_delta_a_minus_b": point_tokens,
                "token_delta_ci_low": float(np.quantile(token_draws, 0.025)),
                "token_delta_ci_high": float(np.quantile(token_draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path) -> None:
    fig, (accuracy_axis, frontier_axis) = plt.subplots(1, 2, figsize=(13, 5.2))
    order = list(POLICIES)
    data = summary.set_index("policy").loc[order].reset_index()
    colors = ["#457b9d", "#6c757d", "#e76f51", "#2a9d8f"]
    x = np.arange(len(data))
    y = data["accuracy"].to_numpy(float)
    low = data["accuracy_ci_low"].to_numpy(float)
    high = data["accuracy_ci_high"].to_numpy(float)
    accuracy_axis.bar(x, y, color=colors)
    accuracy_axis.errorbar(x, y, yerr=np.vstack((y - low, high - y)), fmt="none", color="black", capsize=4)
    accuracy_axis.set_xticks(x, ["Short", "Medium", "Long", "U512\nselective"])
    accuracy_axis.set_ylim(0, 1.03)
    accuracy_axis.set_ylabel("Final-answer accuracy")
    accuracy_axis.set_title("Offline matched policy evaluation")
    accuracy_axis.grid(alpha=0.2, axis="y")
    for index, value in enumerate(y):
        accuracy_axis.text(index, value + 0.035, f"{value:.0%}", ha="center", fontsize=10)

    for index, row in data.iterrows():
        frontier_axis.errorbar(
            row["mean_reasoning_tokens"],
            row["accuracy"],
            xerr=[[row["mean_reasoning_tokens"] - row["tokens_ci_low"]], [row["tokens_ci_high"] - row["mean_reasoning_tokens"]]],
            yerr=[[row["accuracy"] - row["accuracy_ci_low"]], [row["accuracy_ci_high"] - row["accuracy"]]],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
        frontier_axis.annotate(
            ["Short", "Medium", "Long", "U512 selective"][index],
            (row["mean_reasoning_tokens"], row["accuracy"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    frontier_axis.set_xlabel("Mean realized reasoning tokens (512-token prefix included)")
    frontier_axis.set_ylabel("Final-answer accuracy")
    frontier_axis.set_title("Accuracy–compute frontier")
    frontier_axis.grid(alpha=0.2)
    fig.suptitle("Phase 4E: frozen U512 routing policy on held-out MATH", y=1.02)
    fig.text(
        0.5,
        -0.03,
        "Intervals: problem-clustered 95% bootstrap; selective policy routes only frozen high-U512 prefixes to the long arm.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_by_model(summary: pd.DataFrame, output: Path) -> None:
    models = list(summary["model_key"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(6.1 * len(models), 4.9), squeeze=False)
    colors = ["#457b9d", "#6c757d", "#e76f51", "#2a9d8f"]
    labels = ["Short", "Medium", "Long", "U512 selective"]
    for axis, model_key in zip(axes[0], models, strict=True):
        data = summary[summary["model_key"].eq(model_key)].set_index("policy").loc[list(POLICIES)].reset_index()
        for index, row in data.iterrows():
            axis.scatter(
                row["mean_reasoning_tokens"],
                row["accuracy"],
                color=colors[index],
                s=70,
                label=labels[index],
                zorder=3,
            )
            axis.annotate(
                labels[index],
                (row["mean_reasoning_tokens"], row["accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.set_title(model_key)
        axis.set_xlabel("Mean realized reasoning tokens")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("Final-answer accuracy")
    axes[0][-1].legend(loc="best", fontsize=9)
    fig.suptitle("Phase 4E held-out policy frontier by model", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    pairs = pd.read_parquet(args.pairs)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    manifest = json.loads(args.extension_manifest.read_text(encoding="utf-8"))
    if not validation.get("valid"):
        raise RuntimeError("Phase 4E branch validation must pass before policy evaluation")
    if validation.get("extension_manifest_sha256") != sha256_file(args.extension_manifest):
        raise RuntimeError("Validation and extension manifest hashes disagree")
    required = {"problem_id", "model_key", "uncertainty_stratum", "short_correct", "medium_correct", "long_correct"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Paired table is missing columns: {sorted(missing)}")
    if set(pairs["uncertainty_stratum"]) - {"low", "high"}:
        raise RuntimeError("Adaptive policy requires frozen binary U512 strata")

    policy_rows = pd.concat([_materialize_policy(pairs, policy) for policy in POLICIES], ignore_index=True)
    summary = _bootstrap_summary(policy_rows, args.bootstrap_repetitions, args.bootstrap_seed)
    comparisons = _bootstrap_policy_comparisons(
        policy_rows,
        args.bootstrap_repetitions,
        args.bootstrap_seed + 1,
    )
    by_model = pd.concat(
        [
            _bootstrap_summary(
                policy_rows[policy_rows["model_key"].eq(model_key)],
                args.bootstrap_repetitions,
                args.bootstrap_seed + index + 2,
            ).assign(model_key=model_key)
            for index, model_key in enumerate(sorted(policy_rows["model_key"].unique()))
        ],
        ignore_index=True,
    )
    output = ensure_directory(args.output_dir)
    policy_rows.to_parquet(output / "policy_outcomes.parquet", index=False)
    summary.to_csv(output / "policy_summary.csv", index=False)
    comparisons.to_csv(output / "policy_pairwise_effects.csv", index=False)
    by_model.to_csv(output / "policy_by_model_summary.csv", index=False)
    _plot(summary, output / "phase04e_policy_frontier.png")
    _plot_by_model(by_model, output / "phase04e_policy_frontier_by_model.png")

    indexed = summary.set_index("policy")
    selective = indexed.loc["u512_selective"]
    long = indexed.loc["fixed_long"]
    short = indexed.loc["fixed_short"]
    strata_effects = validation["effects"]["strata"]
    interaction_interval = validation["effects"]["problem_cluster_bootstrap_95ci"][
        "interaction_long_minus_medium"
    ]
    interaction = validation["effects"]["interactions_high_minus_low"]["long_minus_medium"]
    primary_comparison = comparisons[
        comparisons["policy_a"].eq("u512_selective") & comparisons["policy_b"].eq("fixed_long")
    ].iloc[0]
    report = "\n".join(
        [
            "# Phase 4E: held-out U512 adaptive-compute policy evaluation",
            "",
            "## Frozen policy",
            "",
            "After exactly 512 reasoning tokens, route a trajectory to the 24K long continuation "
            "only when its pre-frozen, model × difficulty balanced U512 stratum is high; otherwise "
            "route it to the 1K short continuation. U512 references come from the Phase 4B training split. "
            "No Phase 4E answer, continuation, or terminal feature enters this decision.",
            "",
            "## Policy summary",
            "",
            "| Policy | Accuracy | Mean realized reasoning tokens | Long-route rate |",
            "|---|---:|---:|---:|",
            *[
                f"| {row.policy_label} | {row.accuracy:.1%} "
                f"[{row.accuracy_ci_low:.1%}, {row.accuracy_ci_high:.1%}] | "
                f"{row.mean_reasoning_tokens:,.0f} | {row.long_route_rate:.1%} |"
                for row in summary.itertuples()
            ],
            "",
            "## Interpretation rule",
            "",
            "The selective policy is useful only if it reaches an accuracy close to the fixed-long policy while "
            "using materially fewer realized reasoning tokens. The result is an offline evaluation from paired "
            "counterfactual branches: deployment would generate only the chosen branch, whereas this experiment "
            "generated all matched branches to identify the policy effect.",
            "",
            f"- Selective versus fixed long: {selective.accuracy - long.accuracy:+.1%} accuracy and "
            f"{selective.mean_reasoning_tokens - long.mean_reasoning_tokens:+,.0f} mean reasoning tokens.",
            f"- Selective versus fixed short: {selective.accuracy - short.accuracy:+.1%} accuracy and "
            f"{selective.mean_reasoning_tokens - short.mean_reasoning_tokens:+,.0f} mean reasoning tokens.",
            "",
            "## Paired policy contrasts",
            "",
            "Differences below are paired within the same model/problem/continuation-seed tuple and use a "
            "problem-clustered 95% bootstrap interval. A confidence interval spanning zero is not evidence "
            "that the policies differ reliably at this cohort size.",
            "",
            "| Contrast (A − B) | Accuracy difference | Reasoning-token difference |",
            "|---|---:|---:|",
            *[
                f"| {row.comparison} | {row.accuracy_delta_a_minus_b:+.1%} "
                f"[{row.accuracy_delta_ci_low:+.1%}, {row.accuracy_delta_ci_high:+.1%}] | "
                f"{row.token_delta_a_minus_b:+,.0f} "
                f"[{row.token_delta_ci_low:+,.0f}, {row.token_delta_ci_high:+,.0f}] |"
                for row in comparisons.itertuples()
            ],
            "",
            "## Decision",
            "",
            f"The frozen high-U512 → long / low-U512 → short rule saves "
            f"{-primary_comparison.token_delta_a_minus_b / long.mean_reasoning_tokens:.1%} of the realized "
            f"reasoning tokens relative to fixed long, but its accuracy is "
            f"{primary_comparison.accuracy_delta_a_minus_b:+.1%} relative to fixed long "
            f"[{primary_comparison.accuracy_delta_ci_low:+.1%}, {primary_comparison.accuracy_delta_ci_high:+.1%}]. "
            "It is therefore not a demonstrated replacement for fixed-long reasoning. This is a useful negative "
            "result: a binary U512 threshold alone is too coarse for compute routing, even though extra compute "
            "improves outcomes overall.",
            "",
            "The correct next policy-development step is to freeze a validation-tuned three-action rule "
            "(short / medium / long), then test it on a new held-out cohort. It must not be tuned using the "
            "outcomes in this Phase 4E cohort.",
            "",
            "## Does high U512 identify who benefits more from extra compute?",
            "",
            "| Frozen U512 stratum | Short accuracy | Medium accuracy | Long accuracy | Long − medium |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {stratum.title()} | {effects['short_accuracy']:.1%} | "
                f"{effects['medium_accuracy']:.1%} | {effects['long_accuracy']:.1%} | "
                f"{effects['contrasts']['long_minus_medium']:+.1%} |"
                for stratum, effects in strata_effects.items()
            ],
            "",
            f"The raw long-minus-medium gain is {interaction:+.1%} larger in the high-U512 stratum, but the "
            f"problem-clustered interaction interval is [{interaction_interval[0]:+.1%}, "
            f"{interaction_interval[1]:+.1%}] and spans zero. Thus Phase 4E supports the generic claim that "
            "more reasoning compute improves accuracy, not the selective causal claim that U512 reliably identifies "
            "the examples with the greatest marginal return to that compute.",
            "",
            "## Provenance",
            "",
            f"- Extension manifest SHA-256: `{sha256_file(args.extension_manifest)}`",
            f"- Paired branch table SHA-256: `{sha256_file(args.pairs)}`",
            f"- Frozen manifest policy interaction: `{manifest['protocol']['primary_policy_interaction']}`",
        ]
    ) + "\n"
    (output / "phase04e_policy_report.md").write_text(report, encoding="utf-8")
    write_json_atomic(
        output / "analysis_manifest.json",
        {
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(args.pairs),
            "validation": str(args.validation),
            "extension_manifest": str(args.extension_manifest),
            "extension_manifest_sha256": sha256_file(args.extension_manifest),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "policy_version": "phase04e_u512_high_to_long_v1",
            "outputs": {
                "policy_summary": "policy_summary.csv",
                "pairwise_effects": "policy_pairwise_effects.csv",
                "policy_by_model_summary": "policy_by_model_summary.csv",
            },
        },
    )
    print(output)


if __name__ == "__main__":
    main()
