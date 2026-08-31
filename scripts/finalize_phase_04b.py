#!/usr/bin/env python
"""Finalize the diagnostic Phase 4b without overstating an underpowered gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.storage import read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-audit", type=Path, required=True)
    parser.add_argument("--dynamics-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    power = read_json(args.power_audit)
    dynamics = (
        read_json(args.dynamics_summary)
        if args.dynamics_summary.exists()
        else {"technical_status": "not_run"}
    )
    predictor_eligible = bool(power["predictor_eligible"])
    summary = {
        "technical_status": "passed",
        # Phase 4b is an instrumentation/base-panel milestone rather than a
        # directional confirmatory test. Keep the registered outcome ontology
        # strict; the substantive target is evaluated only after Phase 4c
        # supplies breakthrough labels.
        "scientific_outcome": "not_applicable",
        "next_decision": "run_breakthrough_pilot",
        "summary": (
            "Phase 4b completed the frozen three-model base panel and dense prefix feature "
            "extraction. Terminal correctness remains the simple eventual-success baseline; "
            "the central breakthrough target is labeled separately by sparse Phase 4c probes."
        ),
        "metrics": {
            "target_column": power.get("target_column", "correct"),
            "predictor_eligibility": predictor_eligible,
            "pooled_test_positive_problem_clusters": power["pooled_test_positive_problem_clusters"],
            "pooled_test_negative_problem_clusters": power["pooled_test_negative_problem_clusters"],
            "minimum_required_class_problem_clusters": power["minimum_test_class_problem_clusters"],
            "dynamics_analysis_status": dynamics.get("technical_status"),
        },
        "warnings": [
            "Phase 4b alone does not identify breakthrough time or establish that extra reasoning improves outcomes.",
            "Phase 4c must freeze an outcome-blind cohort and validate exact-prefix continuation branches before Phase 4d forecasting.",
        ],
    }
    write_json_atomic(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
