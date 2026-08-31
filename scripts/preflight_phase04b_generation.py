#!/usr/bin/env python
"""Exercise the real 16K Phase 4b path before the paid 600-trajectory panel.

The short instrumentation smoke proves the schema.  This preflight proves the
operational assumption that the selected number of independent generation
processes fits on the target GPU with the actual 16K configuration.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.storage import ensure_directory, read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--generation-workers", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--maximum-allocated-gib", type=float, default=70.0)
    return parser.parse_args()


def _run_model_preflight(args: argparse.Namespace, config_path: Path) -> dict[str, object]:
    config = load_experiment_config(args.project_root / config_path)
    problem_count = args.generation_workers
    model_root = ensure_directory(args.output_dir / config.output_subdirectory)
    processes: list[tuple[subprocess.Popen[bytes], object]] = []
    for shard_index in range(args.generation_workers):
        log_path = model_root / f"shard_{shard_index:02d}.log"
        log_handle = log_path.open("wb")
        command = [
            sys.executable,
            "-u",
            str(args.project_root / "scripts" / "generate.py"),
            "--project-root",
            str(args.project_root),
            "--config",
            str(config_path),
            "--datasets-dir",
            str(args.datasets_dir),
            "--readiness-manifest",
            str(args.readiness_manifest),
            "--output-dir",
            str(model_root / f"shard_{shard_index:02d}"),
            "--resume",
            "--shard-count",
            str(args.generation_workers),
            "--shard-index",
            str(shard_index),
            "--maximum-problems",
            str(problem_count),
            "--fail-fast",
        ]
        processes.append(
            (subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle)
        )
    failures = []
    for process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(return_code)
    if failures:
        raise RuntimeError(
            f"16K preflight failed for {config.experiment_id}; inspect {model_root}/*.log"
        )
    validation_command = [
        sys.executable,
        "-u",
        str(args.project_root / "scripts" / "validate_generation.py"),
        "--run-dir",
        str(model_root),
        "--output-dir",
        str(model_root),
        "--expected-trajectories",
        str(problem_count),
        "--minimum-completion-rate",
        "1.0",
    ]
    subprocess.run(validation_command, check=True)
    validation = read_json(model_root / "generation_validation.json")
    if not validation["valid"]:
        raise RuntimeError(f"16K preflight payload validation failed for {config.experiment_id}")
    peak_allocated = []
    peak_reserved = []
    for metadata_path in model_root.rglob("metadata.json"):
        metadata = read_json(metadata_path)
        if metadata.get("peak_allocated_gib") is not None:
            peak_allocated.append(float(metadata["peak_allocated_gib"]))
        if metadata.get("peak_reserved_gib") is not None:
            peak_reserved.append(float(metadata["peak_reserved_gib"]))
    peak = max(peak_allocated, default=0.0)
    if peak >= args.maximum_allocated_gib:
        raise RuntimeError(
            f"16K preflight allocated {peak:.2f} GiB for {config.experiment_id}, exceeding "
            f"the safe limit {args.maximum_allocated_gib:.2f} GiB"
        )
    return {
        "experiment_id": config.experiment_id,
        "model_key": config.model.key,
        "max_new_tokens": config.model.max_new_tokens,
        "generation_workers": args.generation_workers,
        "tested_trajectories": problem_count,
        "validation": validation,
        "maximum_peak_allocated_gib": peak,
        "maximum_peak_reserved_gib": max(peak_reserved, default=0.0),
    }


def main() -> None:
    args = parse_args()
    if args.maximum_allocated_gib <= 0:
        raise ValueError("maximum-allocated-gib must be positive")
    output_dir = ensure_directory(args.output_dir)
    if not read_json(args.readiness_manifest).get("all_ready"):
        raise RuntimeError("Phase 4b preflight requires passing instrumented smoke results")
    results = [_run_model_preflight(args, path) for path in args.config]
    summary = {
        "technical_status": "passed",
        "purpose": (f"{args.generation_workers}-process 16K generation safety preflight"),
        "models": results,
        "recommended_generation_workers": args.generation_workers,
    }
    write_json_atomic(output_dir / "phase04b_generation_preflight.json", summary)
    print(output_dir / "phase04b_generation_preflight.json")


if __name__ == "__main__":
    main()
