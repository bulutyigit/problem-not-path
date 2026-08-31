#!/usr/bin/env python
"""Compare the matched 8K and 16K cross-model generation panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

KEYS = ["model_key", "dataset", "problem_id", "seed"]
MODEL_LABELS = {
    "gemma4_e4b": "Gemma 4 E4B",
    "qwen35_4b": "Qwen 3.5 4B",
    "ministral3_3b": "Ministral 3 3B",
}
LIMIT_FINISH_REASONS = {"max_new_tokens", "answer_reserve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--extended-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def _prepare(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    required = {
        *KEYS,
        "correct",
        "finish_reason",
        "trajectory_token_count",
        "generated_tokens",
        "elapsed_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature frame is missing columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Feature frame contains duplicate matched keys for {budget} tokens")
    result = frame.copy()
    result["budget"] = budget
    result["correct"] = result["correct"].astype(bool)
    result["capped"] = result["finish_reason"].isin(LIMIT_FINISH_REASONS)
    result["eos"] = result["finish_reason"].eq("eos")
    return result


def _clustered_mean_interval(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    point = float(frame[column].astype(float).mean())
    groups = [group for _, group in frame.groupby("problem_id", sort=False)]
    if not groups or repetitions <= 0:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.integers(0, len(groups), len(groups))
        draws[index] = (
            pd.concat([groups[group_index] for group_index in sampled], ignore_index=True)[column]
            .astype(float)
            .mean()
        )
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_interval(
    paired: pd.DataFrame,
    left: str,
    right: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    differences = paired[right].astype(float) - paired[left].astype(float)
    point = float(differences.mean())
    groups = [group.index.to_numpy() for _, group in paired.groupby("problem_id", sort=False)]
    if not groups or repetitions <= 0:
        low = high = np.nan
    else:
        rng = np.random.default_rng(seed)
        draws = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            sampled = rng.integers(0, len(groups), len(groups))
            rows = np.concatenate([groups[group_index] for group_index in sampled])
            draws[index] = float(differences.loc[rows].mean())
        low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "difference_16k_minus_8k": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "trajectories": len(paired),
        "problems": int(paired["problem_id"].nunique()),
    }


def _summarize(combined: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = {
        "accuracy": "correct",
        "eos_rate": "eos",
        "capped_rate": "capped",
        "mean_reasoning_tokens": "trajectory_token_count",
        "mean_generated_tokens": "generated_tokens",
        "mean_elapsed_seconds": "elapsed_seconds",
    }
    for (model_key, budget), group in combined.groupby(["model_key", "budget"], sort=True):
        row: dict = {
            "model_key": model_key,
            "model_label": MODEL_LABELS.get(str(model_key), str(model_key)),
            "budget": int(budget),
            "trajectories": len(group),
            "problems": int(group["problem_id"].nunique()),
        }
        for offset, (name, column) in enumerate(metrics.items()):
            value, low, high = _clustered_mean_interval(
                group,
                column,
                repetitions,
                seed=20260728 + offset + int(budget),
            )
            row[name] = value
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _make_paired(baseline: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *KEYS,
        "correct",
        "eos",
        "capped",
        "finish_reason",
        "trajectory_token_count",
        "generated_tokens",
        "elapsed_seconds",
    ]
    paired = baseline[columns].merge(
        extended[columns],
        on=KEYS,
        how="inner",
        suffixes=("_8k", "_16k"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(extended):
        raise ValueError(
            "The 8K and 16K panels are not exactly matched by model, dataset, problem, and seed"
        )
    paired = paired.reset_index(drop=True)
    paired["correct_transition"] = (
        paired["correct_8k"].map({False: "wrong", True: "correct"})
        + " → "
        + paired["correct_16k"].map({False: "wrong", True: "correct"})
    )
    paired["extended_outcome"] = np.select(
        [
            paired["eos_16k"] & paired["correct_16k"],
            paired["eos_16k"] & ~paired["correct_16k"],
            paired["capped_16k"] & paired["correct_16k"],
            paired["capped_16k"] & ~paired["correct_16k"],
        ],
        ["EOS + correct", "EOS + wrong", "still capped + correct", "still capped + wrong"],
        default="other finish",
    )
    return paired


def _paired_statistics(paired: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    metrics = {
        "accuracy": ("correct_8k", "correct_16k"),
        "eos_rate": ("eos_8k", "eos_16k"),
        "capped_rate": ("capped_8k", "capped_16k"),
        "reasoning_tokens": ("trajectory_token_count_8k", "trajectory_token_count_16k"),
        "generated_tokens": ("generated_tokens_8k", "generated_tokens_16k"),
        "elapsed_seconds": ("elapsed_seconds_8k", "elapsed_seconds_16k"),
    }
    rows: list[dict] = []
    for model_offset, (model_key, group) in enumerate(paired.groupby("model_key", sort=True)):
        for metric_offset, (metric, (left, right)) in enumerate(metrics.items()):
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS.get(str(model_key), str(model_key)),
                    "metric": metric,
                    **_paired_interval(
                        group,
                        left,
                        right,
                        repetitions,
                        seed=20260728 + model_offset * 20 + metric_offset,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _capped_fates(paired: pd.DataFrame) -> pd.DataFrame:
    baseline_capped = paired[paired["capped_8k"]].copy()
    counts = (
        baseline_capped.groupby(["model_key", "extended_outcome"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("model_key")["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    counts["model_label"] = counts["model_key"].map(MODEL_LABELS).fillna(counts["model_key"])
    return counts


def _plot_budget_response(summary: pd.DataFrame, output_path: Path) -> None:
    panels = [
        ("accuracy", "Accuracy"),
        ("eos_rate", "Natural EOS rate"),
        ("capped_rate", "Token-limit rate"),
        ("mean_reasoning_tokens", "Mean reasoning tokens"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for _, model_frame in summary.groupby("model_key", sort=True):
            model_frame = model_frame.sort_values("budget")
            y = model_frame[metric].to_numpy(float)
            low = model_frame[f"{metric}_ci_low"].to_numpy(float)
            high = model_frame[f"{metric}_ci_high"].to_numpy(float)
            axis.errorbar(
                model_frame["budget"] / 1024,
                y,
                yerr=np.vstack([y - low, high - y]),
                marker="o",
                capsize=4,
                label=model_frame["model_label"].iloc[0],
            )
        axis.set_title(title)
        axis.set_xlabel("Generation cap (K tokens)")
        axis.grid(alpha=0.25)
        if metric.endswith("rate") or metric == "accuracy":
            axis.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Matched 8K → 16K generation-cap response")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_capped_fates(fates: pd.DataFrame, output_path: Path) -> None:
    order = [
        "EOS + correct",
        "EOS + wrong",
        "still capped + correct",
        "still capped + wrong",
        "other finish",
    ]
    pivot = fates.pivot(
        index="model_label", columns="extended_outcome", values="proportion"
    ).fillna(0)
    pivot = pivot.reindex(columns=[column for column in order if column in pivot], fill_value=0)
    axis = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5.5),
        color=["#2ca02c", "#d62728", "#8fd175", "#ff9896", "#7f7f7f"][: len(pivot.columns)],
    )
    axis.set_title("What happened to trajectories capped at 8K?")
    axis.set_ylabel("Proportion of 8K-capped trajectories")
    axis.set_xlabel("")
    axis.set_ylim(0, 1)
    axis.legend(title="16K outcome", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=180)
    plt.close(axis.figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    baseline = _prepare(pd.read_parquet(args.baseline_features), 8192)
    extended = _prepare(pd.read_parquet(args.extended_features), 16384)
    paired = _make_paired(baseline, extended)
    combined = pd.concat([baseline, extended], ignore_index=True)
    summary = _summarize(combined, args.bootstrap_repetitions)
    paired_statistics = _paired_statistics(paired, args.bootstrap_repetitions)
    capped_fates = _capped_fates(paired)

    summary.to_parquet(output_dir / "cap_extension_summary.parquet", index=False)
    paired.to_parquet(output_dir / "cap_extension_matched_trajectories.parquet", index=False)
    paired_statistics.to_parquet(
        output_dir / "cap_extension_paired_differences.parquet", index=False
    )
    capped_fates.to_parquet(output_dir / "cap_extension_capped_fates.parquet", index=False)
    _plot_budget_response(summary, output_dir / "cap_extension_budget_response.png")
    _plot_capped_fates(capped_fates, output_dir / "cap_extension_capped_fates.png")

    baseline_capped = paired[paired["capped_8k"]]
    newly_correct = int((~baseline_capped["correct_8k"] & baseline_capped["correct_16k"]).sum())
    newly_wrong = int((baseline_capped["correct_8k"] & ~baseline_capped["correct_16k"]).sum())
    resolved_to_eos = int(baseline_capped["eos_16k"].sum())
    still_capped = int(baseline_capped["capped_16k"].sum())
    eos_baseline = paired[paired["eos_8k"]]
    eos_reproduction_mismatches = int(
        (
            (eos_baseline["correct_8k"] != eos_baseline["correct_16k"])
            | (eos_baseline["finish_reason_8k"] != eos_baseline["finish_reason_16k"])
            | (
                eos_baseline["trajectory_token_count_8k"]
                != eos_baseline["trajectory_token_count_16k"]
            )
        ).sum()
    )
    baseline_eos_count = len(eos_baseline)
    mismatch_rate = (
        eos_reproduction_mismatches / baseline_eos_count if baseline_eos_count else 0.0
    )
    warnings = [
        "Phase 4 still imposes a 16,384-token censoring boundary; trajectories that hit it remain right-censored.",
        "The matched panel uses one seed, so sampling variability is not estimated in this phase.",
        "Longer budgets change the observed trajectory and do not identify a causal effect of entropy on correctness.",
    ]
    if eos_reproduction_mismatches:
        warnings.append(
            f"{eos_reproduction_mismatches}/{baseline_eos_count} baseline-EOS trajectories "
            "did not reproduce exactly at 16K. Bitwise reproduction is not guaranteed across "
            "GPUs, drivers, or nondeterministic kernels; read the paired 8K-to-16K deltas as "
            "matched-problem/seed resamples, not token-identical continuations."
        )
    write_json_atomic(
        output_dir / "cap_extension_input_manifest.json",
        {
            "baseline_features": str(args.baseline_features),
            "baseline_sha256": sha256_file(args.baseline_features),
            "extended_features": str(args.extended_features),
            "extended_sha256": sha256_file(args.extended_features),
            "matched_trajectories": len(paired),
        },
    )
    cap_summary = {
        "technical_status": "passed",
        "scientific_outcome": "positive" if resolved_to_eos or newly_correct else "limited",
        # Non-reproduction of baseline-EOS trajectories caveats the paired
        # interpretation but does not gate Phase 5, which consumes only the
        # 16K panel. The mismatch rate is reported instead of a blocking
        # decision.
        "next_decision": "run_prediction",
        "summary": (
            "All 300 matched model/problem trajectories were rerun with a 16K cap and "
            "compared against the Phase 3 8K panel."
        ),
        "metrics": {
            "matched_trajectories": len(paired),
            "baseline_capped": len(baseline_capped),
            "resolved_to_eos_at_16k": resolved_to_eos,
            "still_capped_at_16k": still_capped,
            "wrong_to_correct_among_8k_capped": newly_correct,
            "correct_to_wrong_among_8k_capped": newly_wrong,
            "baseline_eos": baseline_eos_count,
            "baseline_eos_reproduction_mismatches": eos_reproduction_mismatches,
            "baseline_eos_reproduction_mismatch_rate": mismatch_rate,
        },
        "warnings": warnings,
    }
    write_json_atomic(output_dir / "cap_extension_summary.json", cap_summary)


if __name__ == "__main__":
    main()
