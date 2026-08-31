#!/usr/bin/env python
"""Confirmatory early-signal test on the held-out test split.

Amendment: docs/protocol_amendments/2026-08-27-phase-05-confirmatory-early-signal-test.md
Frozen plan, applied verbatim: prefix-512 features; endpoints eventual success
(primary) and scratch-solvability at 4,096 (secondary, non-instant cells);
feature sets early_baseline vs early_blocks via the frozen resolver;
logistic_regression via fit_predictor; 5-fold StratifiedGroupKFold OOF sanity
on train+validation; single-shot test AUROC with problem-clustered bootstrap;
success iff the paired dAUROC 95% CI excludes 0. Level-free sensitivity and
per-model descriptives are reported, never promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from reasonbench.evaluation.predictor import fit_predictor
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

MODEL_KEYS = ("gemma4_e4b_mlx_4bit", "ministral3_3b_mlx_4bit")
TAU = 0.75
LABEL_COLUMNS = ["cohort", "regime", "instant", "R_4096", "t_f_1024", "event_observed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forecast-token", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def assemble(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_parquet(args.labels)
    labels = labels[labels.model_key.isin(MODEL_KEYS)].copy()
    frames = []
    for path in args.features:
        frame = pd.read_parquet(path)
        frame = frame[frame.model_key.isin(MODEL_KEYS)].copy()
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    features = features[features.problem_id.isin(set(labels.problem_id))].copy()
    merged = features.merge(
        labels[["problem_id", "model_key", *LABEL_COLUMNS]],
        on=["problem_id", "model_key"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        missing = set(zip(labels.problem_id, labels.model_key)) - set(
            zip(merged.problem_id, merged.model_key)
        )
        raise RuntimeError(f"{len(missing)} labeled cells lack features: {sorted(missing)[:4]}")
    split_check = labels.merge(
        merged[["problem_id", "model_key", "research_split"]],
        on=["problem_id", "model_key"],
        suffixes=("_label", "_feature"),
    )
    disagree = split_check[
        split_check.research_split_label != split_check.research_split_feature
    ]
    if len(disagree):
        raise RuntimeError(f"research_split disagrees for {disagree.problem_id.tolist()[:4]}")
    included = merged[merged.observed_token_count >= args.forecast_token].copy()
    exclusions = (
        merged[merged.observed_token_count < args.forecast_token]
        .groupby(["research_split", "model_key"])
        .size()
        .rename("cells")
        .reset_index()
        .to_dict(orient="records")
    )
    audit = {
        "labeled_cells": int(len(merged)),
        "included_cells": int(len(included)),
        "excluded_short_runs": exclusions,
    }
    return included, audit


def power_gate(frame: pd.DataFrame, target: str, folds: int) -> dict:
    groups = frame.groupby(target)["problem_id"].nunique()
    verdict = "pass" if (len(groups) == 2 and int(groups.min()) >= folds) else "underpowered"
    return {"problem_groups_per_class": {str(k): int(v) for k, v in groups.items()},
            "required_per_class": folds, "verdict": verdict}


def oof_auroc(frame: pd.DataFrame, target: str, feature_set: str, folds: int, seed: int) -> float:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(frame, frame[target], groups=frame["problem_id"])
    ):
        predictor = fit_predictor(
            frame,
            train=frame.iloc[train_index],
            validation=frame.iloc[:0],
            feature_set=feature_set,
            model_name="logistic_regression",
            seed=seed + fold,
            target_column=target,
        )
        oof.iloc[heldout_index] = predictor.predict_proba(frame.iloc[heldout_index])
    return float(roc_auc_score(frame[target].astype(int), oof))


def clustered_ci(
    test: pd.DataFrame, target: str, draws: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    problems = np.array(sorted(test.problem_id.unique()))
    by_problem = {pid: group for pid, group in test.groupby("problem_id")}
    stats = {"baseline": [], "early_blocks": [], "delta": []}
    skipped = 0
    for _ in range(draws):
        drawn = pd.concat(
            [by_problem[pid] for pid in rng.choice(problems, size=len(problems))],
            ignore_index=True,
        )
        y = drawn[target].astype(int)
        if y.nunique() < 2:
            skipped += 1
            continue
        base = roc_auc_score(y, drawn["p_baseline"])
        early = roc_auc_score(y, drawn["p_early_blocks"])
        stats["baseline"].append(base)
        stats["early_blocks"].append(early)
        stats["delta"].append(early - base)
    out = {"draws": draws, "skipped_single_class_draws": skipped}
    for name, values in stats.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        out[name] = {"ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}
    return out


def run_endpoint(
    frame: pd.DataFrame, *, endpoint: str, target: str, args: argparse.Namespace,
    level_free: bool = False,
) -> dict:
    working = frame.drop(columns=["level"]) if level_free else frame
    trainval = working[working.research_split.isin(("train", "validation"))].copy()
    test = working[working.research_split.eq("test")].copy()
    result: dict = {
        "endpoint": endpoint,
        "target": target,
        "level_free": level_free,
        "rows": {"trainval": int(len(trainval)), "test": int(len(test))},
        "problems": {
            "trainval": int(trainval.problem_id.nunique()),
            "test": int(test.problem_id.nunique()),
        },
        "test_class_counts": {
            str(k): int(v) for k, v in test[target].astype(int).value_counts().items()
        },
        "power_gate": power_gate(trainval, target, args.folds),
    }
    if result["power_gate"]["verdict"] != "pass":
        return result
    result["oof_auroc_trainval"] = {}
    predictions = {}
    for feature_set in ("early_baseline", "early_blocks"):
        result["oof_auroc_trainval"][feature_set] = round(
            oof_auroc(trainval, target, feature_set, args.folds, args.seed), 4
        )
        final = fit_predictor(
            trainval,
            train=trainval,
            validation=trainval.iloc[:0],
            feature_set=feature_set,
            model_name="logistic_regression",
            seed=args.seed + 10_000,
            target_column=target,
        )
        predictions[feature_set] = final.predict_proba(test)
    test = test.assign(
        p_baseline=predictions["early_baseline"], p_early_blocks=predictions["early_blocks"]
    )
    y = test[target].astype(int)
    if y.nunique() < 2:
        result["test_auroc"] = "undefined_single_class"
        return result
    result["test_auroc"] = {
        "early_baseline": round(float(roc_auc_score(y, test.p_baseline)), 4),
        "early_blocks": round(float(roc_auc_score(y, test.p_early_blocks)), 4),
    }
    result["test_delta_auroc"] = round(
        result["test_auroc"]["early_blocks"] - result["test_auroc"]["early_baseline"], 4
    )
    result["clustered_bootstrap"] = clustered_ci(test, target, args.bootstrap, args.seed)
    delta_ci = result["clustered_bootstrap"]["delta"]
    result["success_criterion_met"] = bool(delta_ci["ci_low"] > 0)
    result["per_model_descriptive"] = {
        model: {
            "n": int(len(group)),
            "auroc_baseline": (
                round(float(roc_auc_score(group[target].astype(int), group.p_baseline)), 4)
                if group[target].nunique() == 2 else "undefined"
            ),
            "auroc_early_blocks": (
                round(float(roc_auc_score(group[target].astype(int), group.p_early_blocks)), 4)
                if group[target].nunique() == 2 else "undefined"
            ),
        }
        for model, group in test.groupby("model_key")
    }
    result["_test_predictions"] = test[
        ["problem_id", "model_key", "research_split", target, "p_baseline", "p_early_blocks"]
    ]
    return result


def main() -> None:
    args = parse_args()
    out = ensure_directory(args.output_dir)
    frame, assembly_audit = assemble(args)
    frame.to_parquet(out / "confirmatory_cell_table.parquet", index=False)

    frame["eventual_success"] = frame["correct"].astype(int)
    eventual = run_endpoint(
        frame, endpoint="primary_eventual_success", target="eventual_success", args=args
    )
    eventual_level_free = run_endpoint(
        frame, endpoint="primary_eventual_success", target="eventual_success",
        args=args, level_free=True,
    )

    secondary_frame = frame[~frame.instant & frame.R_4096.notna()].copy()
    secondary_frame["scratch_solvable_4096"] = (secondary_frame.R_4096 >= TAU).astype(int)
    secondary = run_endpoint(
        secondary_frame, endpoint="secondary_scratch_4096",
        target="scratch_solvable_4096", args=args,
    )
    secondary_level_free = run_endpoint(
        secondary_frame, endpoint="secondary_scratch_4096",
        target="scratch_solvable_4096", args=args, level_free=True,
    )

    interior = frame[frame.event_observed & ~frame.instant]
    horizon_counts = interior.groupby("research_split").size().to_dict()
    horizon = {
        "endpoint": "horizon_within_512",
        "interior_event_cells_by_split": {str(k): int(v) for k, v in horizon_counts.items()},
        "verdict": "underpowered_not_fit",
        "note": "far below the >=5 problem groups per class per fold requirement",
    }

    report = {
        "amendment": "2026-08-27-phase-05-confirmatory-early-signal-test",
        "labels_sha256": sha256_file(args.labels),
        "features_sha256": {str(p): sha256_file(p) for p in args.features},
        "forecast_token": args.forecast_token,
        "assembly": assembly_audit,
        "primary": {k: v for k, v in eventual.items() if not k.startswith("_")},
        "primary_level_free": {
            k: v for k, v in eventual_level_free.items() if not k.startswith("_")
        },
        "secondary": {k: v for k, v in secondary.items() if not k.startswith("_")},
        "secondary_level_free": {
            k: v for k, v in secondary_level_free.items() if not k.startswith("_")
        },
        "horizon": horizon,
    }
    for name, result in (("primary", eventual), ("secondary", secondary)):
        predictions = result.get("_test_predictions")
        if predictions is not None:
            predictions.to_parquet(out / f"test_predictions_{name}.parquet", index=False)
    write_json_atomic(out / "confirmatory_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
