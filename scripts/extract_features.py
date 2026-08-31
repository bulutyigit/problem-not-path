#!/usr/bin/env python
"""Extract complete and fixed-prefix trajectory feature tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from reasonbench.features import extract_feature_tables
from reasonbench.storage import ensure_directory, write_json_atomic


def _table_audit(frame) -> dict:
    """Expose missing feature counts instead of hiding unavailable blocks in imputation."""

    identifiers = {
        "run_id",
        "experiment_id",
        "phase_id",
        "problem_id",
        "research_split",
        "correct",
        "normal_completion",
        "noncompletion",
        "wrong_completion",
        "needs_intervention",
        "parse_status",
        "finish_reason",
        "boundary_status",
    }
    feature_columns = [column for column in frame.columns if column not in identifiers]
    return {
        "columns": len(frame.columns),
        "missing_values_by_model": {
            str(model): {
                column: int(group[column].isna().sum())
                for column in feature_columns
                if int(group[column].isna().sum())
            }
            for model, group in frame.groupby("model_key", dropna=False)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--without-spectral", action="store_true")
    parser.add_argument("--prefix-length", action="append", type=int, default=[])
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="CPU workers for independent trajectory feature extraction (default: up to 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    manifest: dict = {"run_directories": [str(path) for path in args.run_dir], "tables": {}}
    tables = extract_feature_tables(
        args.run_dir,
        include_spectral=not args.without_spectral,
        prefix_lengths=args.prefix_length,
        workers=args.workers,
    )
    full = tables[None]
    full_path = output_dir / "features_full.parquet"
    full.to_parquet(full_path, index=False)
    manifest["tables"]["full"] = {"path": str(full_path), "rows": len(full), **_table_audit(full)}
    for prefix in sorted(set(args.prefix_length)):
        frame = tables[prefix]
        path = output_dir / f"features_prefix_{prefix}.parquet"
        frame.to_parquet(path, index=False)
        manifest["tables"][f"prefix_{prefix}"] = {
            "path": str(path),
            "rows": len(frame),
            **_table_audit(frame),
        }
    write_json_atomic(output_dir / "feature_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
