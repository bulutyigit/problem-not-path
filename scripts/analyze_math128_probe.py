#!/usr/bin/env python
"""Probe battery for the math128 distill dump (frozen plan applied verbatim).

Amendment: docs/protocol_amendments/2026-08-31-math128-distill-probe.md
Endpoints per anchor t: pooled AUROC on held-out problems (literature's
metric, on the 8-per-problem pooled set) and within-problem AUROC on
held-out mid-band problems (per-problem AUROC over its samples,
failure-count-weighted), both from problem-disjoint 5-fold OOF predictions.
Reference: the LOO pass-rate baseline (within-problem 0.5 by construction).

Deviation recorded here: the stored streams support only the
predictive-ambiguity/dynamics summary subset (entropy, surprisal, top-1);
hidden-geometry and spectral blocks were not retained, so the summary-probe
arm is labeled stream_summary, not the full frozen 15-block set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reasonbench.storage import ensure_directory, write_json_atomic

ANCHORS = (4, 8, 16, 32, 64, 128, 192, 256, 384, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def robust_slope(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def stream_summary(streams: dict, t: int) -> np.ndarray:
    entropy = streams["entropy"][:t].astype(float)
    surprisal = streams["surprisal"][:t].astype(float)
    top1 = streams["top1"][:t].astype(float)
    rises = np.diff(surprisal) if len(surprisal) > 1 else np.zeros(1)
    return np.array([
        entropy.mean(), entropy.std(), robust_slope(entropy),
        surprisal.mean(), rises.max() if len(rises) else 0.0,
        top1.mean(), top1.std(), robust_slope(top1),
    ])


def probe_pipeline(n_rows: int, n_features: int, seed: int) -> Pipeline:
    components = min(128, n_features, max(2, n_rows - 1))
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed)),
    ])


def oof_predictions(x, y, groups, folds, seed):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups=groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = probe_pipeline(len(tr), x.shape[1], seed + fold)
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict_proba(x[te])[:, 1]
    return oof


def pooled_ci(frame, column, draws, seed):
    rng = np.random.default_rng(seed)
    problems = frame.problem.unique()
    values = []
    for _ in range(draws):
        chosen = rng.choice(problems, size=len(problems))
        parts = [frame[frame.problem.eq(p)] for p in chosen]
        d = pd.concat(parts)
        if d.correct.nunique() < 2:
            continue
        values.append(roc_auc_score(d.correct, d[column]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def within_problem_metric(frame, column, pair_weighted=False):
    per = []
    for problem, group in frame.groupby("problem"):
        y = group.correct.to_numpy()
        if y.min() == y.max():
            continue
        fails = int((~y.astype(bool)).sum())
        weight = fails * int(y.sum()) if pair_weighted else fails
        per.append((problem, roc_auc_score(y, group[column]), weight))
    return per


def within_ci(per, draws, seed):
    rng = np.random.default_rng(seed)
    per = np.array([(a, w) for _, a, w in per], dtype=float)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(per), size=len(per))
        chosen = per[idx]
        values.append(np.average(chosen[:, 0], weights=chosen[:, 1]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main() -> None:
    args = parse_args()
    out = ensure_directory(args.output_dir)
    index = pd.read_parquet(args.extraction_dir / "extraction_index.parquet")
    rates = index.groupby("problem").correct.mean()
    print(f"trajectories: {len(index)} | mean top1 fidelity: "
          f"{index.top1_agreement.mean():.4f}")

    # Load stored arrays eagerly and close each archive (6,480 open handles
    # would exceed the default macOS file-descriptor limit).
    anchors_by_row: dict[int, dict[int, np.ndarray]] = {}
    streams_by_row: dict[int, dict[str, np.ndarray]] = {}
    for i, row in index.iterrows():
        with np.load(args.extraction_dir / "states" /
                     f"p{row['problem']:03d}_s{row['sample']:03d}.npz") as npz:
            streams_by_row[i] = {k: npz[k] for k in ("entropy", "surprisal", "top1")}
            anchors_by_row[i] = {
                t: npz[f"anchor_{t}"] for t in ANCHORS if f"anchor_{t}" in npz
            }

    # LOO pass-rate baseline per row from the FULL verification table
    # (255 other samples for every problem), not the extracted subset.
    full = pd.read_parquet(args.extraction_dir / "sample_correctness.parquet")
    counts = full.groupby("problem").correct.agg(["sum", "size"])
    counts.index = counts.index.astype(int)
    index = index.join(counts, on="problem")
    index["loo_rate"] = (index["sum"] - index.correct) / (index["size"] - 1)

    report = {"anchors": {}, "n_trajectories": int(len(index)),
              "mean_top1_fidelity": round(float(index.top1_agreement.mean()), 4),
              "fidelity_p10": round(float(index.top1_agreement.quantile(0.1)), 4)}
    for t in ANCHORS:
        usable = index[index.max_anchor >= t].copy()
        rows = usable.index.to_numpy()
        x_state = np.stack([anchors_by_row[i][t] for i in rows]).astype(np.float32)
        x_stream = np.stack([stream_summary(streams_by_row[i], t) for i in rows])
        y = usable.correct.astype(int).to_numpy()
        groups = usable.problem.to_numpy()
        usable["p_state"] = oof_predictions(x_state, y, groups, args.folds, args.seed)
        usable["p_stream"] = oof_predictions(x_stream, y, groups, args.folds, args.seed)

        pooled = usable[usable.in_pooled_set & usable.p_state.notna()]
        entry = {"n_rows": int(len(usable)), "n_pooled_rows": int(len(pooled))}
        for name, column in (("state_probe", "p_state"), ("stream_summary", "p_stream"),
                             ("loo_pass_rate", "loo_rate")):
            sub = pooled[pooled[column].notna()]
            auroc = round(float(roc_auc_score(sub.correct, sub[column])), 4)
            lo, hi = pooled_ci(sub, column, args.bootstrap, args.seed)
            wframe = usable[usable.subset.eq("within") & usable[column].notna()]
            per = within_problem_metric(wframe, column)
            per_pw = within_problem_metric(wframe, column, pair_weighted=True)
            weights = sum(w for *_, w in per)
            within = round(float(np.average([a for _, a, _ in per],
                                            weights=[w for *_, w in per])), 4)
            within_pw = round(float(np.average([a for _, a, _ in per_pw],
                                               weights=[w for *_, w in per_pw])), 4)
            wlo, whi = within_ci(per, args.bootstrap, args.seed)
            pwlo, pwhi = within_ci(per_pw, args.bootstrap, args.seed + 1)
            entry[name] = {
                "pooled_auroc": auroc, "pooled_ci": [lo, hi],
                "within_problem_auroc": within, "within_ci": [wlo, whi],
                "within_pair_weighted": within_pw,
                "within_pair_weighted_ci": [pwlo, pwhi],
                "within_problems_used": len(per), "within_failure_weight": int(weights),
            }
        report["anchors"][str(t)] = entry
        print(f"t={t}: state pooled {entry['state_probe']['pooled_auroc']} "
              f"within {entry['state_probe']['within_problem_auroc']} | "
              f"LOO pooled {entry['loo_pass_rate']['pooled_auroc']}")
    write_json_atomic(out / "math128_probe_report.json", report)
    print("report written")


if __name__ == "__main__":
    main()
