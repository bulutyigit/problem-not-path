#!/usr/bin/env python
"""Create explanatory Phase 4E plots from matched held-out continuations.

The plots make three distinct questions legible without fitting a new model:

1. Is each additional compute arm beneficial, by model and frozen U512 stratum?
2. Which matched trajectories are rescued or harmed by each additional arm?
3. How does the accuracy response to budget differ between low- and high-U512?

All uncertainty intervals resample MATH problems, not individual continuation
branches, because branches from the same problem are correlated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

MODEL_LABELS = {
    "gemma4_e4b_mlx_4bit": "Gemma 4 E4B",
    "ministral3_3b_mlx_4bit": "Ministral 3 3B",
}
ARM_LABELS = {"short": "Short (1K)", "medium": "Medium (4K)", "long": "Long (24K)"}
CONTRASTS = {
    "medium_minus_short": ("medium", "short", "Medium − short"),
    "long_minus_medium": ("long", "medium", "Long − medium"),
    "long_minus_short": ("long", "short", "Long − short"),
}
STRATUM_COLORS = {"low": "#457b9d", "high": "#e76f51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--artifact-prefix", default="phase04e")
    parser.add_argument("--phase-label", default="Phase 4E")
    return parser.parse_args()


def _bootstrap_mean(values: pd.Series, problem_ids: pd.Series, repetitions: int, seed: int) -> tuple[float, float, float]:
    """Return a problem-clustered mean and percentile interval."""
    table = pd.DataFrame({"problem_id": problem_ids.to_numpy(), "value": values.to_numpy(float)})
    grouped = table.groupby("problem_id", sort=True).agg(total=("value", "sum"), count=("value", "size"))
    if len(grouped) < 2:
        raise RuntimeError("At least two problem clusters are required for bootstrap intervals")
    point = float(table["value"].mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    counts = grouped["count"].to_numpy(float)[indices].sum(axis=1)
    draws = grouped["total"].to_numpy(float)[indices].sum(axis=1) / counts
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _effect_table(pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    cohorts: list[tuple[str, pd.DataFrame]] = [("Pooled", pairs)]
    cohorts.extend((MODEL_LABELS[key], group) for key, group in pairs.groupby("model_key", sort=True))
    for cohort_index, (cohort, frame) in enumerate(cohorts):
        for stratum_index, (stratum, group) in enumerate(frame.groupby("uncertainty_stratum", sort=True)):
            for contrast_index, (contrast, (after, before, label)) in enumerate(CONTRASTS.items()):
                delta = group[f"{after}_correct"].astype(float) - group[f"{before}_correct"].astype(float)
                effect, lower, upper = _bootstrap_mean(
                    delta,
                    group["problem_id"],
                    repetitions,
                    seed + cohort_index * 100 + stratum_index * 10 + contrast_index,
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "stratum": stratum,
                        "contrast": contrast,
                        "contrast_label": label,
                        "accuracy_effect": effect,
                        "ci_low": lower,
                        "ci_high": upper,
                        "branches": len(group),
                        "problems": group["problem_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def _budget_summary(pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for model_index, (model_key, model_frame) in enumerate(pairs.groupby("model_key", sort=True)):
        for stratum_index, (stratum, group) in enumerate(model_frame.groupby("uncertainty_stratum", sort=True)):
            for arm_index, arm in enumerate(("short", "medium", "long")):
                accuracy, lower, upper = _bootstrap_mean(
                    group[f"{arm}_correct"].astype(float),
                    group["problem_id"],
                    repetitions,
                    seed + model_index * 100 + stratum_index * 10 + arm_index,
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        "stratum": stratum,
                        "arm": arm,
                        "arm_label": ARM_LABELS[arm],
                        "accuracy": accuracy,
                        "ci_low": lower,
                        "ci_high": upper,
                        "mean_realized_reasoning_tokens": float(512 + group[f"{arm}_reasoning_tokens"].mean()),
                        "branches": len(group),
                        "problems": group["problem_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def _transition_table(pairs: pd.DataFrame) -> pd.DataFrame:
    labels = ("Wrong → wrong", "Wrong → correct (rescue)", "Correct → correct", "Correct → wrong (harm)")
    rows: list[dict] = []
    for transition, (after, before, _) in {
        "short_to_medium": ("medium", "short", "Short → medium"),
        "medium_to_long": ("long", "medium", "Medium → long"),
    }.items():
        for (model_key, stratum), group in pairs.groupby(["model_key", "uncertainty_stratum"], sort=True):
            before_values = group[f"{before}_correct"].astype(bool)
            after_values = group[f"{after}_correct"].astype(bool)
            categories = np.select(
                [
                    ~before_values & ~after_values,
                    ~before_values & after_values,
                    before_values & after_values,
                    before_values & ~after_values,
                ],
                labels,
                default=labels[0],
            )
            counts = pd.Series(categories).value_counts()
            for order, label in enumerate(labels):
                rows.append(
                    {
                        "transition": transition,
                        "transition_label": {"short_to_medium": "Short → medium", "medium_to_long": "Medium → long"}[transition],
                        "model_key": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        "stratum": stratum,
                        "outcome": label,
                        "outcome_order": order,
                        "count": int(counts.get(label, 0)),
                        "rate": float(counts.get(label, 0) / len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _censoring_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual final-answer censoring after answer remediation."""
    rows: list[dict] = []
    for model_key, group in pairs.groupby("model_key", sort=True):
        for arm in ("short", "medium", "long"):
            finish = group[f"{arm}_finish_reason"].astype(str)
            extraction = group[f"{arm}_extraction_status"].astype(str)
            correct = group[f"{arm}_correct"].astype(bool)
            eos = finish.eq("eos")
            capped = finish.eq("answer_limit")
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "branches": len(group),
                    "eos_rate": float(eos.mean()),
                    "answer_limit_rate": float(capped.mean()),
                    "missing_extraction_rate": float(extraction.eq("missing").mean()),
                    "accuracy_all": float(correct.mean()),
                    "accuracy_eos_only": float(correct[eos].mean()) if eos.any() else np.nan,
                    "accuracy_answer_limit_only": (
                        float(correct[capped].mean()) if capped.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_effects(effects: pd.DataFrame, output: Path) -> None:
    cohorts = ["Pooled", *[MODEL_LABELS[key] for key in sorted(MODEL_LABELS)]]
    strata = ("low", "high")
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(15, 6.2), sharey=True)
    labels = [f"{cohort} — {stratum.title()}" for cohort in cohorts for stratum in strata]
    y_positions = np.arange(len(labels))[::-1]
    for axis, (contrast, (_, _, title)) in zip(axes, CONTRASTS.items(), strict=True):
        subset = effects[effects["contrast"].eq(contrast)]
        for y, (cohort, stratum) in zip(y_positions, [(cohort, stratum) for cohort in cohorts for stratum in strata], strict=True):
            row = subset[subset["cohort"].eq(cohort) & subset["stratum"].eq(stratum)].iloc[0]
            axis.errorbar(
                row["accuracy_effect"],
                y,
                xerr=[[row["accuracy_effect"] - row["ci_low"]], [row["ci_high"] - row["accuracy_effect"]]],
                fmt="o",
                color=STRATUM_COLORS[stratum],
                capsize=3,
            )
        axis.axvline(0, color="#555555", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Accuracy effect")
        axis.grid(alpha=0.2, axis="x")
        axis.set_xlim(-0.3, 1.05)
    axes[0].set_yticks(y_positions, labels)
    fig.suptitle("Marginal accuracy benefit of additional reasoning compute", y=1.02)
    fig.text(
        0.5,
        -0.03,
        "Points: matched branch differences; intervals: problem-clustered 95% bootstrap. High/low U512 was frozen before continuations.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_budget_response(summary: pd.DataFrame, output: Path) -> None:
    models = list(summary["model_key"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.1), sharey=True, squeeze=False)
    positions = np.arange(3)
    for axis, model_key in zip(axes[0], models, strict=True):
        frame = summary[summary["model_key"].eq(model_key)]
        for stratum in ("low", "high"):
            subset = (
                frame[frame["stratum"].eq(stratum)]
                .set_index("arm")
                .loc[["short", "medium", "long"]]
                .reset_index()
            )
            values = subset["accuracy"].to_numpy(float)
            lower = subset["ci_low"].to_numpy(float)
            upper = subset["ci_high"].to_numpy(float)
            axis.errorbar(
                positions,
                values,
                yerr=np.vstack((values - lower, upper - values)),
                color=STRATUM_COLORS[stratum],
                marker="o",
                linewidth=2,
                capsize=3,
                label=f"{stratum.title()} U512",
            )
        axis.set_xticks(positions, [ARM_LABELS[arm] for arm in ("short", "medium", "long")])
        axis.set_title(MODEL_LABELS[model_key])
        axis.set_ylim(-0.04, 1.04)
        axis.set_xlabel("Continuation budget")
        axis.grid(alpha=0.2, axis="y")
    axes[0][0].set_ylabel("Final-answer accuracy")
    axes[0][-1].legend(loc="best")
    fig.suptitle("Accuracy response to reasoning budget, stratified by frozen U512", y=1.02)
    fig.text(
        0.5,
        -0.03,
        "Arm labels are caps; realized token use varies because trajectories may stop early. Intervals are problem-clustered 95% bootstrap.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_transitions(transitions: pd.DataFrame, output: Path) -> None:
    models = list(transitions["model_key"].drop_duplicates())
    transition_keys = ("short_to_medium", "medium_to_long")
    categories = list(transitions.sort_values("outcome_order")["outcome"].drop_duplicates())
    fig, axes = plt.subplots(2, len(models), figsize=(7 * len(models), 8.2), sharey=True, squeeze=False)
    x = np.arange(len(categories))
    width = 0.36
    for row_index, transition in enumerate(transition_keys):
        for column_index, model_key in enumerate(models):
            axis = axes[row_index][column_index]
            subset = transitions[
                transitions["transition"].eq(transition) & transitions["model_key"].eq(model_key)
            ]
            for offset, stratum in zip((-width / 2, width / 2), ("low", "high"), strict=True):
                values = (
                    subset[subset["stratum"].eq(stratum)]
                    .set_index("outcome")
                    .loc[categories, "count"]
                    .to_numpy(float)
                )
                axis.bar(x + offset, values, width=width, color=STRATUM_COLORS[stratum], label=f"{stratum.title()} U512")
            axis.set_xticks(x, categories, rotation=17, ha="right")
            axis.set_title(f"{MODEL_LABELS[model_key]} — {subset['transition_label'].iloc[0]}")
            axis.grid(alpha=0.2, axis="y")
            if row_index == 1:
                axis.set_xlabel("Matched outcome transition")
    axes[0][0].set_ylabel("Matched continuation branches")
    axes[1][0].set_ylabel("Matched continuation branches")
    axes[0][-1].legend(loc="upper right")
    fig.suptitle("Which trajectories are rescued or harmed by additional reasoning?", y=1.01)
    fig.text(
        0.5,
        -0.01,
        "Each panel contains matched branches from the same source prefix; exact cohort sizes are reported in the accompanying CSV.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_censoring(censoring: pd.DataFrame, output: Path) -> None:
    models = list(censoring["model_key"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    x = np.arange(3)
    width = 0.36
    for axis, model_key in zip(axes, models, strict=True):
        frame = censoring[censoring["model_key"].eq(model_key)].set_index("arm").loc[
            ["short", "medium", "long"]
        ]
        capped = frame["answer_limit_rate"].to_numpy(float)
        missing = frame["missing_extraction_rate"].to_numpy(float)
        axis.bar(x - width / 2, capped, width, label="Hit 4K answer limit", color="#e76f51")
        axis.bar(x + width / 2, missing, width, label="Answer extraction missing", color="#6c757d")
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in ("short", "medium", "long")])
        axis.set_title(MODEL_LABELS[model_key])
        axis.set_xlabel("Reasoning arm")
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Fraction of matched branches")
    axes[-1].legend(loc="best")
    fig.suptitle("Residual final-answer censoring after 4K remediation", y=1.02)
    fig.text(
        0.5,
        -0.03,
        "A high answer-limit rate means an arm's accuracy remains a lower-bound outcome rather than a clean completed-answer estimate.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    pairs = pd.read_parquet(args.pairs)
    required = {"problem_id", "model_key", "uncertainty_stratum", "short_correct", "medium_correct", "long_correct"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"Paired table is missing columns: {sorted(missing)}")
    unknown_models = set(pairs["model_key"]) - set(MODEL_LABELS)
    if unknown_models:
        raise ValueError(f"Unknown model keys: {sorted(unknown_models)}")
    output = ensure_directory(args.output_dir)
    effects = _effect_table(pairs, args.bootstrap_repetitions, args.bootstrap_seed)
    budget = _budget_summary(pairs, args.bootstrap_repetitions, args.bootstrap_seed + 10_000)
    transitions = _transition_table(pairs)
    censoring = _censoring_table(pairs)
    prefix = args.artifact_prefix
    effects.to_csv(output / f"{prefix}_marginal_compute_effects.csv", index=False)
    budget.to_csv(output / f"{prefix}_budget_response_by_stratum.csv", index=False)
    transitions.to_csv(output / f"{prefix}_matched_transition_counts.csv", index=False)
    censoring.to_csv(output / f"{prefix}_answer_censoring_by_arm.csv", index=False)
    _plot_effects(effects, output / f"{prefix}_marginal_compute_effects.png")
    _plot_budget_response(budget, output / f"{prefix}_budget_response_by_stratum.png")
    _plot_transitions(transitions, output / f"{prefix}_matched_rescue_harm.png")
    _plot_censoring(censoring, output / f"{prefix}_answer_censoring_by_arm.png")
    write_json_atomic(
        output / f"{prefix}_explanatory_analysis_manifest.json",
        {
            "phase_label": args.phase_label,
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(args.pairs),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "outputs": [
                f"{prefix}_marginal_compute_effects.png",
                f"{prefix}_budget_response_by_stratum.png",
                f"{prefix}_matched_rescue_harm.png",
                f"{prefix}_answer_censoring_by_arm.png",
            ],
        },
    )
    print(output)


if __name__ == "__main__":
    main()
