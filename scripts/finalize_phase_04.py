#!/usr/bin/env python
"""Combine Phase 4 cap-sensitivity and held-out early-failure evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.storage import read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-summary", type=Path, required=True)
    parser.add_argument("--early-summary", type=Path, required=True)
    parser.add_argument("--length-summary", type=Path, required=True)
    parser.add_argument("--dynamics-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = read_json(args.cap_summary)
    early = read_json(args.early_summary)
    length = read_json(args.length_summary)
    dynamics = read_json(args.dynamics_summary)
    components = (cap, early, length, dynamics)
    if any(component.get("technical_status") != "passed" for component in components):
        raise RuntimeError("All Phase 4 component analyses must pass before finalization")
    cap_metrics = cap.get("metrics", {})
    determinism_mismatches = int(cap_metrics.get("baseline_eos_reproduction_mismatches", 0))
    baseline_eos = int(cap_metrics.get("baseline_eos", 0))
    mismatch_rate = cap_metrics.get(
        "baseline_eos_reproduction_mismatch_rate",
        determinism_mismatches / baseline_eos if baseline_eos else 0.0,
    )
    positive_early = bool(early.get("metrics", {}).get("confirmatory_positive", False))
    # Phase 5 trains only on the 16K panel, so EOS-reproduction mismatches
    # caveat the paired 8K-to-16K interpretation without gating prediction.
    write_json_atomic(
        args.output,
        {
            "technical_status": "passed",
            "scientific_outcome": "positive" if positive_early else "limited",
            "next_decision": "run_prediction",
            "summary": (
                "Phase 4 quantified 8K-to-16K censoring sensitivity; tested whether "
                "the first 16–512 tokens predict correctness and future reasoning duration "
                "on held-out problems; and mapped uncertainty, geometry, and spectral dynamics."
            ),
            "metrics": {
                "determinism": {
                    "status": "mismatched" if determinism_mismatches else "clean",
                    "baseline_eos": baseline_eos,
                    "mismatches": determinism_mismatches,
                    "mismatch_rate": mismatch_rate,
                },
                "cap_extension": cap_metrics,
                "early_failure_prediction": early.get("metrics", {}),
                "early_length_prediction": length.get("metrics", {}),
                "prefix_dynamics": dynamics.get("metrics", {}),
            },
            "warnings": [
                *cap.get("warnings", []),
                *early.get("warnings", []),
                *length.get("warnings", []),
                *dynamics.get("warnings", []),
            ],
        },
    )


if __name__ == "__main__":
    main()
