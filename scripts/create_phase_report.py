#!/usr/bin/env python
"""Render a phase report from a compact phase summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.phases import PhaseStatus
from reasonbench.reporting import render_phase_report
from reasonbench.storage import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--next-decision")
    return parser.parse_args()


def _default_summary(phase: str, run_dir: Path) -> dict:
    validation_path = run_dir / "generation_validation.json"
    if validation_path.exists():
        validation = read_json(validation_path)
        return {
            "technical_status": "passed" if validation["valid"] else "incomplete",
            "scientific_outcome": "inconclusive",
            "next_decision": "review",
            "summary": "Generation validation completed. Review the diagnostics before continuing.",
            "metrics": {
                "completion_rate": validation["completion_rate"],
                "completed_trajectories": validation["completed_trajectories"],
                "duplicate_run_ids": validation["duplicate_run_ids"],
                "problem_count": validation["problem_count"],
            },
            "warnings": [],
        }
    readiness_path = run_dir / "model_readiness.json"
    dataset_path = run_dir / "dataset_manifest.json"
    if readiness_path.exists():
        readiness = read_json(readiness_path)
        datasets_ready = dataset_path.exists()
        all_ready = bool(readiness.get("all_ready")) and datasets_ready
        return {
            "technical_status": "passed" if all_ready else "incomplete",
            "scientific_outcome": "not_applicable",
            "next_decision": "continue" if all_ready else "review_adapters",
            "summary": "Runtime, model, instrumentation, and dataset readiness were evaluated.",
            "metrics": {
                "models_ready": sum(
                    record.get("status") == "ready"
                    for record in readiness.get("models", {}).values()
                ),
                "models_total": len(readiness.get("models", {})),
                "datasets_ready": datasets_ready,
            },
            "warnings": [],
        }
    return {
        "technical_status": "incomplete",
        "scientific_outcome": "inconclusive",
        "next_decision": "review",
        "summary": f"No recognized summary artifacts were found for {phase}.",
        "metrics": {},
        "warnings": ["The phase report was generated without a validation summary."],
    }


def main() -> None:
    args = parse_args()
    summary_path = args.summary_json
    summary = (
        read_json(summary_path)
        if summary_path is not None and summary_path.exists()
        else _default_summary(args.phase, args.run_dir)
    )
    if args.next_decision:
        summary["next_decision"] = args.next_decision
    status = PhaseStatus(
        phase_id=args.phase,
        technical_status=summary["technical_status"],
        scientific_outcome=summary.get("scientific_outcome", "inconclusive"),
        next_decision=summary.get("next_decision", "review"),
        summary=summary.get("summary", ""),
        metrics=summary.get("metrics", {}),
        warnings=summary.get("warnings", []),
    )
    report = render_phase_report(args.run_dir, status)
    print(report)


if __name__ == "__main__":
    main()
