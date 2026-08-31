#!/usr/bin/env python
"""Question-only difficulty ceiling on non-math public dumps (GPQA-diamond).

Descriptive external analysis; estimator identical to the frozen OpenR1/
math128 leave-one-out pass-rate predictor (each sample's score is the mean
correctness of the problem's OTHER samples), with a problem-clustered
bootstrap. GPQA is 4-way multiple choice, so the guessing floor is 0.25 and
correct-by-luck attempts attenuate, never inflate, the ceiling.
"""
from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from reasonbench.storage import ensure_directory, write_json_atomic

ANSWER = re.compile(r"final answer is\s*:?\s*\**\(?([A-D])\)?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_directory(args.output_dir)
    rows = []
    unparsed = 0
    with tarfile.open(args.tarball) as tar:
        names = sorted(n for n in tar.getnames()
                       if n.endswith(".yaml") and "/logs/" not in n)
        for name in names:
            record = yaml.safe_load(tar.extractfile(name).read())
            gt = str(record["gt_answer"]).strip().upper()
            pid = Path(name).stem
            for i, sample in enumerate(record["samples"]):
                match = ANSWER.search(sample[-500:])
                if match is None:
                    unparsed += 1
                    correct = False  # unparseable = incorrect, as graders score it
                else:
                    correct = match.group(1).upper() == gt
                rows.append({"problem": pid, "sample": i, "correct": correct})
    frame = pd.DataFrame(rows)
    counts = frame.groupby("problem").correct.agg(["sum", "size"])
    frame = frame.join(counts, on="problem")
    frame["loo"] = (frame["sum"] - frame.correct) / (frame["size"] - 1)
    y = frame.correct.to_numpy()
    rates = counts["sum"] / counts["size"]

    rng = np.random.default_rng(args.seed)
    problems = frame.problem.unique()
    values = []
    for _ in range(args.bootstrap):
        chosen = rng.choice(problems, size=len(problems))
        d = pd.concat([frame[frame.problem.eq(p)] for p in chosen])
        if d.correct.nunique() == 2:
            values.append(roc_auc_score(d.correct, d.loo))
    lo, hi = np.percentile(values, [2.5, 97.5])
    report = {
        "problems": int(len(counts)), "samples": int(len(frame)),
        "unparsed_answers": int(unparsed),
        "overall_pass_rate": round(float(y.mean()), 4),
        "loo_ceiling_auroc": round(float(roc_auc_score(y, frame.loo)), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "pass_rate_bands": {
            "le_0.25": int((rates <= 0.25).sum()),
            "0.25_0.75": int(((rates > 0.25) & (rates < 0.75)).sum()),
            "ge_0.75": int((rates >= 0.75).sum()),
        },
    }
    frame.to_parquet(out / "rows.parquet", index=False)
    write_json_atomic(out / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()
