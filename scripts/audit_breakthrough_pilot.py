#!/usr/bin/env python
"""Summarize Phase 4C-P cost, memory, label diversity, and verifier behavior."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from reasonbench.storage import read_json, write_json_atomic, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-safe-allocated-gib", type=float, default=70.0)
    return parser.parse_args()


def _finite(values) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def _model_family(model_key: str) -> str:
    return model_key.removesuffix("_mlx_4bit")


def main() -> None:
    args = parse_args()
    validation = read_json(args.phase_dir / "breakthrough_probe_validation.json")
    if validation.get("stage") != "pilot":
        raise ValueError("Pilot audit requires a pilot-stage probe validation")
    result_paths = [
        marker.parent / "result.json"
        for marker in sorted(args.phase_dir.rglob("branch_complete.json"))
        if (marker.parent / "result.json").exists()
    ]
    results = [read_json(path) for path in result_paths]
    summaries = [
        read_json(path)
        for path in sorted(args.phase_dir.rglob("trajectory_probe_summary.json"))
    ]
    elapsed = _finite(result.get("elapsed_seconds") for result in results)
    allocated = _finite(result.get("peak_allocated_gib") for result in results)
    reserved = _finite(result.get("peak_reserved_gib") for result in results)
    continuation_tokens = _finite(result.get("continuation_token_count") for result in results)
    models = Counter(str(summary["model_key"]) for summary in summaries)
    levels = Counter(int(summary["level"]) for summary in summaries)
    methods = Counter(
        str(result.get("verification", {}).get("verification_method", "unknown"))
        for result in results
    )
    forced = sum(bool(result.get("reasoning_boundary_forced")) for result in results)
    events = int(validation.get("events_observed", 0))
    censored = int(validation.get("right_censored", 0))
    memory_safe = bool(allocated) and max(allocated) < args.maximum_safe_allocated_gib
    label_diverse = events > 0 and censored > 0
    coverage_valid = (
        {_model_family(model) for model in models}
        == {"gemma4_e4b", "qwen35_4b", "ministral3_3b"}
        and set(levels) == {1, 2, 3, 4, 5}
        and all(count == 3 for count in levels.values())
    )
    projected_multiplier = 4.0  # 20 frozen problems / 5 pilot problems.
    audit = {
        "technical_status": (
            "passed"
            if validation.get("valid") and memory_safe and coverage_valid
            else "failed"
        ),
        "scientific_outcome": "pilot_label_diversity_observed" if label_diverse else "pilot_label_degenerate",
        "next_decision": "manual_cost_and_label_review_required",
        "validation": validation,
        "model_trajectory_counts": dict(models),
        "level_trajectory_counts": {str(level): count for level, count in sorted(levels.items())},
        "events_observed": events,
        "right_censored": censored,
        "label_diverse": label_diverse,
        "branches": len(results),
        "forced_reasoning_boundary_rate": forced / len(results) if results else None,
        "verification_methods": dict(methods),
        "elapsed_seconds": {
            "total": sum(elapsed),
            "median": float(np.median(elapsed)) if elapsed else None,
            "maximum": max(elapsed) if elapsed else None,
            "projected_full_cohort_serial_total": sum(elapsed) * projected_multiplier,
        },
        "continuation_tokens": {
            "total": sum(continuation_tokens),
            "median": float(np.median(continuation_tokens)) if continuation_tokens else None,
            "projected_full_cohort_total": sum(continuation_tokens) * projected_multiplier,
        },
        "gpu_memory": {
            "maximum_peak_allocated_gib": max(allocated) if allocated else None,
            "maximum_peak_reserved_gib": max(reserved) if reserved else None,
            "safe_allocated_gib_threshold": args.maximum_safe_allocated_gib,
            "memory_safe": memory_safe,
        },
        "automatic_full_cohort_launch_allowed": False,
    }
    write_json_atomic(args.output_dir / "breakthrough_pilot_audit.json", audit)
    lines = [
        "# Phase 4C-P breakthrough pilot audit",
        "",
        f"- Technical status: **{audit['technical_status']}**",
        f"- Stable events / right-censored: **{events} / {censored}**",
        f"- Branches: **{len(results)}**",
        f"- Peak allocated GPU memory: **{audit['gpu_memory']['maximum_peak_allocated_gib']} GiB**",
        f"- Projected full-cohort continuation tokens: **{audit['continuation_tokens']['projected_full_cohort_total']:.0f}**",
        "",
        "The full 20-problem cohort is never launched automatically. Review label diversity,",
        "forced-boundary rate, verifier methods, runtime projection, and the sensitivity table",
        "before rerunning Phase 4C without `--probe-pilot-only`.",
    ]
    write_text_atomic(
        args.output_dir / "breakthrough_pilot_audit.md", "\n".join(lines) + "\n"
    )
    print(audit)


if __name__ == "__main__":
    main()
