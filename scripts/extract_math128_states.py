#!/usr/bin/env python
"""Teacher-forced state extraction for the math128 distill probe test.

Amendment: docs/protocol_amendments/2026-08-31-math128-distill-probe.md
No sampling anywhere: each stored trajectory is the dump's own text,
re-tokenized and forwarded once through the 4-bit model. Stores pooled
anchor vectors, scalar token streams, and the top-1 fidelity share.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MODEL_ID = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
ANCHORS = (4, 8, 16, 32, 64, 128, 192, 256, 384, 512)
WINDOW = 512
POOL = 4
POOLED_PER_PROBLEM = 8
SELECTION_SEED = 20260831
MID_BAND = (0.05, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Local snapshot of the math128 dataset (yaml files)")
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_rows(correctness: pd.DataFrame) -> pd.DataFrame:
    rates = correctness.groupby("problem").correct.mean()
    mid = set(rates[(rates >= MID_BAND[0]) & (rates < MID_BAND[1])].index)
    rng = np.random.default_rng(SELECTION_SEED)
    keep = []
    for problem, group in correctness.groupby("problem"):
        if problem in mid:
            keep.append(group.assign(subset="within"))
        else:
            chosen = rng.choice(group["sample"].to_numpy(), size=POOLED_PER_PROBLEM,
                                replace=False)
            keep.append(group[group["sample"].isin(chosen)].assign(subset="pooled_only"))
    frame = pd.concat(keep, ignore_index=True)
    frame["in_pooled_set"] = frame.subset.eq("pooled_only")
    for problem in mid:
        rows = frame.index[frame.problem.eq(problem)]
        chosen = rng.choice(rows.to_numpy(), size=POOLED_PER_PROBLEM, replace=False)
        frame.loc[chosen, "in_pooled_set"] = True
    return frame


def main() -> None:
    import mlx.core as mx
    from mlx_lm import load

    args = parse_args()
    out = args.output_dir
    (out / "states").mkdir(parents=True, exist_ok=True)
    correctness = pd.read_parquet(args.correctness)
    selection = select_rows(correctness)
    selection.to_parquet(out / "selection.parquet", index=False)
    print(f"selected {len(selection)} trajectories "
          f"({int(selection.subset.eq('within').sum())} within-set)")

    model, tokenizer = load(MODEL_ID)
    tie = bool(getattr(model.args, "tie_word_embeddings", False))
    problems = {int(p.stem): yaml.safe_load(p.read_text())
                for p in args.dump_dir.glob("*.yaml")}

    rows = []
    started = time.perf_counter()
    for position, row in enumerate(selection.itertuples(index=False), start=1):
        state_path = out / "states" / f"p{row.problem:03d}_s{row.sample:03d}.npz"
        if args.resume and state_path.exists():
            continue
        record = problems[row.problem]
        prompt_ids = tokenizer.encode(record["prompt"])
        sample_ids = tokenizer.encode(record["samples"][row.sample],
                                      add_special_tokens=False)[:WINDOW]
        ids = mx.array([prompt_ids + sample_ids])
        hidden = model.model(ids)
        logits = (model.model.embed_tokens.as_linear(hidden) if tie
                  else model.lm_head(hidden))
        hidden = np.asarray(hidden.astype(mx.float16))[0]
        gen_start = len(prompt_ids)
        gen_len = len(sample_ids)

        # Scalar streams over generated positions: the logit row at absolute
        # position (gen_start + j - 1) predicts generated token j.
        entropy = np.zeros(gen_len, dtype=np.float32)
        surprisal = np.zeros(gen_len, dtype=np.float32)
        top1 = np.zeros(gen_len, dtype=np.float32)
        agree = np.zeros(gen_len, dtype=bool)
        chunk = 128
        for lo in range(0, gen_len, chunk):
            hi = min(lo + chunk, gen_len)
            rows_idx = mx.array(list(range(gen_start + lo - 1, gen_start + hi - 1)))
            piece = logits[0, rows_idx, :].astype(mx.float32)
            logp = piece - mx.logsumexp(piece, axis=-1, keepdims=True)
            probs = mx.exp(logp)
            entropy[lo:hi] = np.asarray(-mx.sum(probs * logp, axis=-1))
            targets = mx.array(sample_ids[lo:hi])
            surprisal[lo:hi] = np.asarray(-mx.take_along_axis(
                logp, targets[:, None], axis=-1)[:, 0])
            top_ids = mx.argmax(piece, axis=-1)
            top1[lo:hi] = np.asarray(mx.max(probs, axis=-1))
            agree[lo:hi] = np.asarray(top_ids == targets)
        del logits

        anchor_vectors = {}
        for t in ANCHORS:
            if gen_len < t:
                break
            window = hidden[gen_start + t - POOL: gen_start + t, :]
            anchor_vectors[f"anchor_{t}"] = window.mean(axis=0).astype(np.float16)
        np.savez_compressed(
            state_path, entropy=entropy, surprisal=surprisal, top1=top1,
            agree=agree, **anchor_vectors,
        )
        rows.append({
            "problem": row.problem, "sample": row.sample, "correct": bool(row.correct),
            "subset": row.subset, "in_pooled_set": bool(row.in_pooled_set),
            "prompt_tokens": len(prompt_ids), "window_tokens": gen_len,
            "max_anchor": max((t for t in ANCHORS if gen_len >= t), default=0),
            "top1_agreement": float(agree.mean()),
        })
        if position % 200 == 0:
            speed = position / (time.perf_counter() - started)
            print(f"{position}/{len(selection)} ({speed:.1f} traj/s)", flush=True)
    pd.DataFrame(rows).to_parquet(out / "extraction_index.parquet", index=False)
    print(f"done: {len(rows)} extracted")


if __name__ == "__main__":
    main()
