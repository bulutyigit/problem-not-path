#!/usr/bin/env python
"""Difficulty-ceiling analysis on public DeepSeek-R1 generation dumps.

Purpose (descriptive external analysis, defined before execution): quantify
how well a QUESTION-ONLY signal predicts trajectory-level final correctness
in the regime where probe-based positives are reported (large
reasoning-tuned model, competition math), using no trace content at all.

Data: open-r1/OpenR1-Math-220k, "all" shards. One row per generation;
y = correctness_math_verify. Problems with k >= 2 generations enter.

Estimator (frozen here, before any evaluation):
  difficulty_loo(g) = mean correctness of the problem's OTHER generations
  (leave-one-out, so the predicted trajectory's own outcome never enters
  its predictor). Primary metric: pooled AUROC of difficulty_loo for y,
  with a problem-clustered bootstrap (resample problems with replacement,
  2,000 draws, 95% percentile CI). Secondary: AUROC by generation count k
  (k=2 gives a binary predictor - an underestimate of the ceiling; larger
  k approaches it), and the non-LOO oracle pass rate as a labeled upper
  reference. Known conservative bias, recorded in advance: dataset curation
  drops some all-fail problems, truncating the difficulty distribution and
  UNDERstating the ceiling.

Output: artifacts/external/difficulty_ceiling/{rows.parquet, report.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score

from reasonbench.storage import ensure_directory, write_json_atomic

COLUMNS = ["uuid", "source", "problem_type", "correctness_math_verify"]
SHARDS = [
    f"hf://datasets/open-r1/OpenR1-Math-220k/all/default-{i:05d}-of-00010.parquet"
    for i in range(10)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def load_rows() -> pd.DataFrame:
    frames = []
    for shard in SHARDS:
        table = ds.dataset(shard, format="parquet").to_table(columns=COLUMNS)
        frames.append(table.to_pandas())
        print(f"loaded {shard.rsplit('/', 1)[-1]}: {len(frames[-1])} problems")
    problems = pd.concat(frames, ignore_index=True)
    rows = []
    for uuid, source, ptype, flags in problems.itertuples(index=False):
        flags = [bool(f) for f in flags]
        if len(flags) < 2:
            continue
        total = sum(flags)
        k = len(flags)
        for y in flags:
            rows.append((uuid, source, ptype, k, int(y), (total - y) / (k - 1)))
    return pd.DataFrame(
        rows, columns=["uuid", "source", "problem_type", "k", "correct", "difficulty_loo"]
    )


def clustered_auroc_ci(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    uuids = frame.uuid.to_numpy()
    order = np.argsort(uuids, kind="stable")
    y = frame.correct.to_numpy()[order]
    x = frame.difficulty_loo.to_numpy()[order]
    unique, starts = np.unique(uuids[order], return_index=True)
    bounds = np.append(starts, len(y))
    values = []
    skipped = 0
    for _ in range(draws):
        chosen = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([np.arange(bounds[c], bounds[c + 1]) for c in chosen])
        yy = y[index]
        if yy.min() == yy.max():
            skipped += 1
            continue
        values.append(roc_auc_score(yy, x[index]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"draws": draws, "skipped": skipped,
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


def main() -> None:
    args = parse_args()
    out = ensure_directory(args.output_dir)
    cache = out / "rows.parquet"
    if cache.exists():
        frame = pd.read_parquet(cache)
        print(f"reusing {cache} ({len(frame)} rows)")
    else:
        frame = load_rows()
        frame.to_parquet(cache, index=False)
    null_uuid_rows = int(frame.uuid.isna().sum())
    frame = frame.dropna(subset=["uuid"]).reset_index(drop=True)
    y = frame.correct.to_numpy()
    report = {
        "dataset": "open-r1/OpenR1-Math-220k (all shards)",
        "dropped_null_uuid_rows": null_uuid_rows,
        "problems": int(frame.uuid.nunique()),
        "generations": int(len(frame)),
        "k_distribution": {str(k): int(v) for k, v in frame.k.value_counts().sort_index().items()},
        "base_correct_rate": round(float(y.mean()), 4),
        "primary_auroc_difficulty_loo": round(float(roc_auc_score(y, frame.difficulty_loo)), 4),
        "primary_ci": clustered_auroc_ci(frame, args.bootstrap, args.seed),
        "auroc_by_k": {
            str(k): round(float(roc_auc_score(g.correct, g.difficulty_loo)), 4)
            for k, g in frame.groupby("k") if g.correct.nunique() == 2
        },
        "oracle_upper_reference_non_loo": None,
        "curation_caveat": "all-fail problems partly dropped by dataset curation; "
                           "ceiling is understated",
    }
    oracle = frame.groupby("uuid").correct.transform("mean")
    report["oracle_upper_reference_non_loo"] = round(float(roc_auc_score(y, oracle)), 4)
    write_json_atomic(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
