#!/usr/bin/env python
"""Build Phase 4d horizon and hazard tables from frozen local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reasonbench.constants import PHASE4_FORECAST_HORIZONS
from reasonbench.evaluation.breakthrough import build_longitudinal_tables
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", action="append", type=int, default=[])
    return parser.parse_args()


def _split_audit(frame: pd.DataFrame) -> dict:
    split_problem_ids = {
        str(split): set(group["problem_id"].astype(str))
        for split, group in frame.groupby("research_split", dropna=False)
    }
    overlap = {
        f"{left}__{right}": len(split_problem_ids[left] & split_problem_ids[right])
        for index, left in enumerate(split_problem_ids)
        for right in list(split_problem_ids)[index + 1 :]
    }
    return {
        "rows": len(frame),
        "trajectories": int(frame["run_id"].nunique()),
        "problems": int(frame["problem_id"].nunique()),
        "rows_by_split": {
            str(split): int(count)
            for split, count in frame["research_split"].value_counts(dropna=False).items()
        },
        "problem_overlap_between_splits": overlap,
    }


def _eventual_success_panel(features_directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(features_directory.glob("features_prefix_*.parquet")):
        prefix = int(path.stem.rsplit("_", maxsplit=1)[1])
        frame = pd.read_parquet(path)
        frame = frame[frame["observed_token_count"] >= prefix].copy()
        frame["forecast_token"] = prefix
        frame["forecast_time_bin"] = f"t_{prefix}"
        frame["time_log1p"] = np.log1p(prefix)
        rows.append(frame)
    if not rows:
        raise ValueError(f"No dense prefix features found in {features_directory}")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["problem_id", "run_id", "forecast_token"]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output = ensure_directory(args.output_dir)
    labels = pd.read_parquet(args.labels)
    horizons = tuple(args.horizon) or PHASE4_FORECAST_HORIZONS
    horizon, hazard = build_longitudinal_tables(
        args.features_dir,
        labels,
        horizons=horizons,
    )
    eventual_success = _eventual_success_panel(args.features_dir)
    eventual_path = output / "eventual_success_table.parquet"
    horizon_path = output / "breakthrough_horizon_table.parquet"
    hazard_path = output / "breakthrough_hazard_table.parquet"
    eventual_success.to_parquet(eventual_path, index=False)
    horizon.to_parquet(horizon_path, index=False)
    hazard.to_parquet(hazard_path, index=False)
    audit = {
        "schema_version": "phase04d_longitudinal_tables_v1",
        "labels_path": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "features_directory": str(args.features_dir),
        "horizons": list(horizons),
        "eventual_success_table": {
            **_split_audit(eventual_success),
            "sha256": sha256_file(eventual_path),
        },
        "horizon_table": {**_split_audit(horizon), "sha256": sha256_file(horizon_path)},
        "hazard_table": {**_split_audit(hazard), "sha256": sha256_file(hazard_path)},
        "event_counts": {
            str(value): int(count)
            for value, count in labels["event_observed"].value_counts(dropna=False).items()
        },
    }
    if any(audit["horizon_table"]["problem_overlap_between_splits"].values()):
        raise RuntimeError("Problem leakage detected in breakthrough horizon table")
    if any(audit["hazard_table"]["problem_overlap_between_splits"].values()):
        raise RuntimeError("Problem leakage detected in breakthrough hazard table")
    if any(audit["eventual_success_table"]["problem_overlap_between_splits"].values()):
        raise RuntimeError("Problem leakage detected in eventual-success table")
    write_json_atomic(output / "breakthrough_table_audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
