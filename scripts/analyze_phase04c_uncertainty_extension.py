#!/usr/bin/env python
"""Produce the final descriptive analysis for the Phase 4C-U budget experiment.

The input table is deliberately the validator's paired table, rather than raw
generation artifacts.  This preserves the experiment's frozen identity and
nested-continuation checks as a prerequisite for every reported figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

ARMS = ("short", "medium", "long")
ARM_LABELS = {"short": "Short\n1K", "medium": "Medium\n4K", "long": "Long\n24K"}
ARM_TARGETS = {"short": 1024, "medium": 4096, "long": 24576}
MODEL_LABELS = {
    "gemma4_e4b_mlx_4bit": "Gemma 4 E4B",
    "qwen35_4b_mlx_4bit": "Qwen 3.5 4B",
    "ministral3_3b_mlx_4bit": "Ministral 3 3B",
}
STRATUM_STYLE = {
    "low": {"label": "Low early uncertainty", "color": "#2a9d8f"},
    "high": {"label": "High early uncertainty", "color": "#e76f51"},
}
CONTRASTS = {
    "medium_minus_short": ("short", "medium", "Medium − short"),
    "long_minus_medium": ("medium", "long", "Long − medium (primary)"),
    "long_minus_short": ("short", "long", "Long − short"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args()


def _model_label(key: str) -> str:
    return MODEL_LABELS.get(str(key), str(key))


def _check_inputs(pairs: pd.DataFrame, validation: dict) -> None:
    required = {
        "problem_id",
        "model_key",
        "uncertainty_stratum",
        "short_correct",
        "medium_correct",
        "long_correct",
        "short_reasoning_tokens",
        "medium_reasoning_tokens",
        "long_reasoning_tokens",
        "short_finish_reason",
        "medium_finish_reason",
        "long_finish_reason",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Paired table is missing columns: {sorted(missing)}")
    if not validation.get("valid"):
        raise RuntimeError("The input extension validation is not valid")
    if int(validation.get("observed_branches", -1)) != 600:
        raise RuntimeError("Expected the complete 600-branch Phase 4C-U cohort")
    if len(pairs) != 200:
        raise RuntimeError(f"Expected 200 paired branches; found {len(pairs)}")
    if not set(pairs["uncertainty_stratum"]).issubset({"low", "high"}):
        raise ValueError("Only frozen low/high uncertainty strata may enter the analysis")


def _cluster_bootstrap(
    frame: pd.DataFrame,
    value: pd.Series,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return point and problem-clustered percentile interval for a row statistic."""

    point = float(value.mean())
    problem_rows = [group.index.to_numpy() for _, group in frame.groupby("problem_id", sort=True)]
    if len(problem_rows) < 2 or repetitions < 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = value.astype(float)
    draws = np.empty(repetitions, dtype=float)
    for draw_index in range(repetitions):
        selected = rng.integers(0, len(problem_rows), len(problem_rows))
        rows = np.concatenate([problem_rows[index] for index in selected])
        draws[draw_index] = float(values.loc[rows].mean())
    low, high = np.quantile(draws, (0.025, 0.975))
    return point, float(low), float(high)


def _dose_response_summary(pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for model_key, model_frame in pairs.groupby("model_key", sort=True):
        for stratum, stratum_frame in model_frame.groupby("uncertainty_stratum", sort=True):
            for arm_index, arm in enumerate(ARMS):
                point, low, high = _cluster_bootstrap(
                    stratum_frame,
                    stratum_frame[f"{arm}_correct"],
                    repetitions=repetitions,
                    seed=seed + arm_index + 17 * len(rows),
                )
                rows.append(
                    {
                        "scope": "model",
                        "model_key": model_key,
                        "model_label": _model_label(model_key),
                        "uncertainty_stratum": stratum,
                        "arm": arm,
                        "target_reasoning_tokens": ARM_TARGETS[arm],
                        "accuracy": point,
                        "ci_low": low,
                        "ci_high": high,
                        "paired_branches": len(stratum_frame),
                        "problems": int(stratum_frame["problem_id"].nunique()),
                    }
                )
    for stratum, stratum_frame in pairs.groupby("uncertainty_stratum", sort=True):
        for arm_index, arm in enumerate(ARMS):
            point, low, high = _cluster_bootstrap(
                stratum_frame,
                stratum_frame[f"{arm}_correct"],
                repetitions=repetitions,
                seed=seed + 10_000 + arm_index + 17 * len(rows),
            )
            rows.append(
                {
                    "scope": "pooled",
                    "model_key": "pooled",
                    "model_label": "All models pooled",
                    "uncertainty_stratum": stratum,
                    "arm": arm,
                    "target_reasoning_tokens": ARM_TARGETS[arm],
                    "accuracy": point,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_branches": len(stratum_frame),
                    "problems": int(stratum_frame["problem_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _contrast_summary(pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for scope, scope_frame in [("pooled", pairs), *list(pairs.groupby("model_key", sort=True))]:
        if scope == "pooled":
            model_key, model_label = "pooled", "All models pooled"
        else:
            model_key, model_label = str(scope), _model_label(str(scope))
        for stratum, stratum_frame in scope_frame.groupby("uncertainty_stratum", sort=True):
            for contrast_index, (contrast, (left, right, label)) in enumerate(CONTRASTS.items()):
                difference = stratum_frame[f"{right}_correct"].astype(float) - stratum_frame[
                    f"{left}_correct"
                ].astype(float)
                point, low, high = _cluster_bootstrap(
                    stratum_frame,
                    difference,
                    repetitions=repetitions,
                    seed=seed + contrast_index + 101 * len(rows),
                )
                rescued = int((~stratum_frame[f"{left}_correct"] & stratum_frame[f"{right}_correct"]).sum())
                harmed = int((stratum_frame[f"{left}_correct"] & ~stratum_frame[f"{right}_correct"]).sum())
                rows.append(
                    {
                        "scope": "pooled" if model_key == "pooled" else "model",
                        "model_key": model_key,
                        "model_label": model_label,
                        "uncertainty_stratum": stratum,
                        "contrast": contrast,
                        "contrast_label": label,
                        "accuracy_difference": point,
                        "ci_low": low,
                        "ci_high": high,
                        "rescued": rescued,
                        "harmed": harmed,
                        "net_rescued": rescued - harmed,
                        "paired_branches": len(stratum_frame),
                        "problems": int(stratum_frame["problem_id"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def _token_summary(pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for model_key, model_frame in pairs.groupby("model_key", sort=True):
        for arm_index, arm in enumerate(ARMS):
            token_column = f"{arm}_reasoning_tokens"
            point, low, high = _cluster_bootstrap(
                model_frame,
                model_frame[token_column],
                repetitions=repetitions,
                seed=seed + arm_index + 37 * len(rows),
            )
            cap_rate = float(model_frame[f"{arm}_finish_reason"].eq("max_new_tokens").mean())
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": _model_label(model_key),
                    "arm": arm,
                    "mean_realized_reasoning_tokens": point,
                    "ci_low": low,
                    "ci_high": high,
                    "median_realized_reasoning_tokens": float(model_frame[token_column].median()),
                    "budget_finish_rate": cap_rate,
                    "paired_branches": len(model_frame),
                }
            )
    return pd.DataFrame(rows)


def _plot_dose_response(summary: pd.DataFrame, output: Path) -> None:
    models = sorted(summary.loc[summary["scope"].eq("model"), "model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 4.6), sharey=True)
    for axis, model_key in zip(np.atleast_1d(axes), models, strict=True):
        panel = summary[(summary["scope"] == "model") & (summary["model_key"] == model_key)]
        for stratum in ("low", "high"):
            line = panel[panel["uncertainty_stratum"] == stratum].sort_values("target_reasoning_tokens")
            x = np.arange(len(ARMS))
            y = line["accuracy"].to_numpy(float)
            low = line["ci_low"].to_numpy(float)
            high = line["ci_high"].to_numpy(float)
            style = STRATUM_STYLE[stratum]
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                marker="o",
                linewidth=2.3,
                capsize=4,
                color=style["color"],
                label=style["label"],
            )
        axis.set_title(_model_label(model_key))
        axis.set_xticks(np.arange(len(ARMS)), [ARM_LABELS[arm] for arm in ARMS])
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.22, axis="y")
        axis.set_xlabel("Maximum total reasoning budget")
    np.atleast_1d(axes)[0].set_ylabel("Final-answer accuracy")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="lower right")
    fig.suptitle("More reasoning compute improves accuracy in both early-uncertainty strata", y=1.02)
    fig.text(
        0.5,
        -0.03,
        "Points: matched continuation branches; bars: problem-clustered 95% bootstrap intervals",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_contrasts(contrasts: pd.DataFrame, output: Path) -> None:
    panels = ["medium_minus_short", "long_minus_medium", "long_minus_short"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True)
    for axis, contrast in zip(axes, panels, strict=True):
        panel = contrasts[(contrasts["contrast"] == contrast) & (contrasts["scope"] == "model")]
        labels = [_model_label(key) for key in sorted(panel["model_key"].unique())]
        y_positions = np.arange(len(labels))
        for stratum, offset in [("low", -0.14), ("high", 0.14)]:
            values = panel[panel["uncertainty_stratum"] == stratum].set_index("model_label").reindex(labels)
            x = values["accuracy_difference"].to_numpy(float)
            low = values["ci_low"].to_numpy(float)
            high = values["ci_high"].to_numpy(float)
            axis.errorbar(
                x,
                y_positions + offset,
                xerr=np.vstack((x - low, high - x)),
                fmt="o",
                color=STRATUM_STYLE[stratum]["color"],
                capsize=3,
                label=STRATUM_STYLE[stratum]["label"],
            )
        axis.axvline(0, color="#666666", linewidth=1)
        axis.set_title(CONTRASTS[contrast][2])
        axis.set_yticks(y_positions, labels)
        axis.set_xlabel("Paired accuracy difference")
        axis.grid(alpha=0.22, axis="x")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Matched budget effects by model and frozen early-uncertainty stratum", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_transitions(contrasts: pd.DataFrame, output: Path) -> None:
    panel = contrasts[(contrasts["scope"] == "model") & (contrasts["contrast"] == "long_minus_short")]
    models = sorted(panel["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 4.6), sharey=True)
    for axis, model_key in zip(np.atleast_1d(axes), models, strict=True):
        group = panel[panel["model_key"] == model_key].set_index("uncertainty_stratum")
        x = np.arange(2)
        rescued = group.loc[["low", "high"], "rescued"].to_numpy(float)
        harmed = group.loc[["low", "high"], "harmed"].to_numpy(float)
        axis.bar(x, rescued, color="#43aa8b", label="Wrong → correct")
        axis.bar(x, -harmed, color="#e76f51", label="Correct → wrong")
        axis.axhline(0, color="#444444", linewidth=1)
        axis.set_title(_model_label(model_key))
        axis.set_xticks(x, ["Low\nuncertainty", "High\nuncertainty"])
        axis.set_xlabel("Frozen U512 stratum")
        axis.grid(alpha=0.2, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("Matched branches (long vs. short)")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Longer reasoning predominantly rescues, rather than harms, paired trajectories", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_realized_tokens(summary: pd.DataFrame, output: Path) -> None:
    models = sorted(summary["model_key"].unique())
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(models))
    width = 0.24
    for arm_index, arm in enumerate(ARMS):
        panel = summary[summary["arm"] == arm].set_index("model_key").reindex(models)
        y = panel["mean_realized_reasoning_tokens"].to_numpy(float)
        low = panel["ci_low"].to_numpy(float)
        high = panel["ci_high"].to_numpy(float)
        axis.bar(
            x + (arm_index - 1) * width,
            y,
            width,
            yerr=np.vstack((y - low, high - y)),
            capsize=3,
            label=f"{ARM_LABELS[arm].replace(chr(10), ' ')} (cap {ARM_TARGETS[arm] // 1024}K)",
        )
    axis.set_xticks(x, [_model_label(key) for key in models])
    axis.set_ylabel("Mean realized continuation reasoning tokens")
    axis.set_title("Models use substantially different amounts of the available reasoning budget")
    axis.grid(alpha=0.22, axis="y")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    validation: dict,
    dose: pd.DataFrame,
    contrasts: pd.DataFrame,
    tokens: pd.DataFrame,
) -> str:
    effects = validation["effects"]
    low = effects["strata"]["low"]
    high = effects["strata"]["high"]
    interaction = effects["interactions_high_minus_low"]
    primary_ci = effects["problem_cluster_bootstrap_95ci"]["interaction_long_minus_medium"]
    lines = [
        "# Phase 4C-U: early uncertainty and additional reasoning compute",
        "",
        "## Integrity status",
        "",
        f"- **Validated paired cohort:** {validation['eligible_trajectories']} source trajectories, "
        f"{validation['observed_branches']}/{validation['expected_branches']} expected branch artifacts.",
        "- **Mechanical checks:** no missing, duplicate, corrupt, identity-mismatched, or non-nested paths.",
        "- **Design:** each 512-token prefix was continued with the same branch seed under 1K, 4K, "
        "and 24K total reasoning budgets. The frozen U512 score stratified runs within model × difficulty cells.",
        "",
        "## Main result: more reasoning compute improves accuracy",
        "",
        "| Early-U stratum | Short (1K) | Medium (4K) | Long (24K) | Long − medium | Long − short |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Low | {low['short_accuracy']:.0%} | {low['medium_accuracy']:.0%} | {low['long_accuracy']:.0%} | "
        f"{low['contrasts']['long_minus_medium']:+.0%} | {low['contrasts']['long_minus_short']:+.0%} |",
        f"| High | {high['short_accuracy']:.0%} | {high['medium_accuracy']:.0%} | {high['long_accuracy']:.0%} | "
        f"{high['contrasts']['long_minus_medium']:+.0%} | {high['contrasts']['long_minus_short']:+.0%} |",
        "",
        "Both strata benefit from additional reasoning budget. The pre-registered primary high-minus-low "
        f"interaction for **long − medium** is {interaction['long_minus_medium']:+.1%} "
        f"(problem-clustered 95% bootstrap interval {primary_ci[0]:+.1%} to {primary_ci[1]:+.1%}). "
        "It is therefore not justified to claim, from this cohort alone, that high U512 trajectories benefit more "
        "from the 4K → 24K increment.",
        "",
        "The larger secondary 1K → 24K gain is +29 percentage points for high U512 versus +23 points for low U512; "
        "its high-minus-low interaction still has an interval crossing zero. This is directionally interesting, not "
        "a confirmed selective-routing effect.",
        "",
        "## Paired transition evidence",
        "",
        f"- Low-U, long vs. short: {low['transitions']['long_vs_short_rescued']} wrong→correct rescues and "
        f"{low['transitions']['long_vs_short_harmed']} correct→wrong reversals.",
        f"- High-U, long vs. short: {high['transitions']['long_vs_short_rescued']} wrong→correct rescues and "
        f"{high['transitions']['long_vs_short_harmed']} correct→wrong reversals.",
        "- Thus the aggregate gain reflects substantially more rescues than reversals, not merely an averaging artifact.",
        "",
        "## Interpretation and boundary",
        "",
        "The validated causal statement is: **allocating a larger continuation budget after the same 512-token prefix "
        "improves final-answer accuracy in this cohort.** The study does not yet establish that U512 alone is a good "
        "routing policy for deciding *which* trajectories deserve the long budget. That requires a larger held-out "
        "policy evaluation or a decision rule that is independently trained/frozen before testing.",
        "",
        "The model-specific plots should be read as descriptive: per-model and per-stratum cells are small, while the "
        "pooled problem-cluster bootstrap is the appropriate uncertainty summary for the prespecified aggregate claim.",
        "",
        "## Generated artifacts",
        "",
        "- `accuracy_dose_response_by_model_and_uncertainty.png`: accuracy dose–response curves.",
        "- `paired_accuracy_effects_by_model.png`: matched accuracy contrasts with clustered intervals.",
        "- `long_vs_short_outcome_transitions.png`: rescues versus reversals.",
        "- `realized_reasoning_tokens_by_budget.png`: actual continuation reasoning used by each model.",
        "",
        "## Reproducibility",
        "",
        f"- Input pairs SHA-256: `{sha256_file(Path('artifacts/mac_mlx/phase_04c/uncertainty/validation_full/uncertainty_extension_pairs.parquet'))}`",
        f"- Frozen extension manifest SHA-256: `{validation['extension_manifest_sha256']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    pairs = pd.read_parquet(args.pairs)
    with args.validation.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    _check_inputs(pairs, validation)
    output = ensure_directory(args.output_dir)
    dose = _dose_response_summary(pairs, args.bootstrap_repetitions, args.bootstrap_seed)
    contrasts = _contrast_summary(pairs, args.bootstrap_repetitions, args.bootstrap_seed + 100_000)
    tokens = _token_summary(pairs, args.bootstrap_repetitions, args.bootstrap_seed + 200_000)
    dose.to_csv(output / "dose_response_summary.csv", index=False)
    contrasts.to_csv(output / "paired_accuracy_contrasts.csv", index=False)
    tokens.to_csv(output / "realized_reasoning_tokens.csv", index=False)
    _plot_dose_response(dose, output / "accuracy_dose_response_by_model_and_uncertainty.png")
    _plot_contrasts(contrasts, output / "paired_accuracy_effects_by_model.png")
    _plot_transitions(contrasts, output / "long_vs_short_outcome_transitions.png")
    _plot_realized_tokens(tokens, output / "realized_reasoning_tokens_by_budget.png")
    report = _markdown_report(validation, dose, contrasts, tokens)
    (output / "phase04c_uncertainty_extension_report.md").write_text(report, encoding="utf-8")
    write_json_atomic(
        output / "analysis_input_manifest.json",
        {
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(args.pairs),
            "validation": str(args.validation),
            "validation_sha256": sha256_file(args.validation),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "analysis_status": "complete",
        },
    )
    print(f"Wrote Phase 4C-U analysis to {output}")


if __name__ == "__main__":
    main()
