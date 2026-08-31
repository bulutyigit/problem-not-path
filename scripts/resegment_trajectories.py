#!/usr/bin/env python
"""Repair stored segment labels of committed trajectories in place.

Earlier segmentation matched marker token-id patterns, which silently failed
for markers that tokenize differently in context (Ministral's ``[/THINK]``
close tag and in-context ``\\boxed{``). Every committed
``token_metrics.parquet`` retains the decoded text of each token, so segments
can be recomputed exactly with the current text-span rules and no tokenizer.

Only trajectories whose labels actually change are rewritten. A rewrite
replaces ``token_metrics.parquet`` atomically and refreshes that file's size
and SHA-256 entry in ``complete.json``. Metadata, hidden states, and all other
artifacts are untouched. Trajectories produced by two-stage assigned-budget
generation are skipped: their stored labels are correct by construction and
their token stream omits the injected boundary text.

After repairing a phase, downstream products of that phase (feature tables,
analyses, and the phase artifact manifest) must be rebuilt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reasonbench.generation.segments import segment_token_texts
from reasonbench.generation.storage import _write_parquet_atomic
from reasonbench.storage import read_json, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the trajectories that would change without rewriting them.",
    )
    return parser.parse_args()


def resegment_directory(directory: Path, dry_run: bool) -> str:
    """Recompute one trajectory's segments; return the action taken."""

    metadata = read_json(directory / "metadata.json")
    if metadata.get("inserted_boundary_tokens", 0) or metadata.get(
        "reasoning_boundary_forced", False
    ):
        return "skipped_budgeted"
    metrics_path = directory / "token_metrics.parquet"
    frame = pd.read_parquet(metrics_path)
    if not frame["token_index"].is_monotonic_increasing:
        raise ValueError(f"token_index is not ordered in {metrics_path}")
    segments = segment_token_texts(
        [str(text) for text in frame["token_text"]],
        str(metadata.get("model_mode", "reasoning")),
    )
    if list(frame["segment"]) == segments:
        return "unchanged"
    if dry_run:
        return "would_change"
    frame["segment"] = segments
    _write_parquet_atomic(frame, metrics_path)
    completion_path = directory / "complete.json"
    completion = read_json(completion_path)
    files = completion.get("files", {})
    if "token_metrics.parquet" in files:
        files["token_metrics.parquet"] = {
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        }
    write_json_atomic(completion_path, completion)
    return "changed"


def main() -> None:
    args = parse_args()
    directories = sorted(
        {
            marker.parent
            for run_directory in args.run_dir
            for marker in run_directory.rglob("complete.json")
        }
    )
    if not directories:
        raise ValueError(f"No committed trajectories were found under {args.run_dir}")
    counts: dict[str, int] = {}
    for directory in directories:
        action = resegment_directory(directory, args.dry_run)
        counts[action] = counts.get(action, 0) + 1
    print(f"Scanned {len(directories)} trajectories: {counts}")
    if counts.get("changed"):
        print(
            "Rewritten trajectories invalidate previously extracted features, "
            "analyses, and phase artifact manifests; rebuild them for the "
            "affected phases."
        )


if __name__ == "__main__":
    main()
