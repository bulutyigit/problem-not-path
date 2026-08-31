#!/usr/bin/env python
"""Build the evidence synthesis and reproducibility index for Phase 7."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reasonbench.phases import load_phase_status
from reasonbench.storage import read_json, sha256_file, write_json_atomic, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def _validate_manifest(phase_root: Path, verify_hashes: bool) -> dict[str, Any]:
    manifest_path = phase_root / "artifacts_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    missing: list[str] = []
    mismatched: list[str] = []
    for record in manifest.get("files", []):
        path = phase_root / record["path"]
        if not path.exists():
            missing.append(record["path"])
        elif verify_hashes and sha256_file(path) != record["sha256"]:
            mismatched.append(record["path"])
    nested_payloads_checked = 0
    if verify_hashes:
        for completion_path in phase_root.rglob("trajectories/*/complete.json"):
            completion = read_json(completion_path)
            for filename, record in completion.get("files", {}).items():
                payload_path = completion_path.parent / filename
                nested_payloads_checked += 1
                relative = str(payload_path.relative_to(phase_root))
                if not payload_path.exists():
                    missing.append(relative)
                elif sha256_file(payload_path) != record["sha256"]:
                    mismatched.append(relative)
    if missing or mismatched:
        raise RuntimeError(
            f"Manifest validation failed for {phase_root.name}: "
            f"missing={missing[:5]}, mismatched={mismatched[:5]}"
        )
    return {
        "phase_id": phase_root.name,
        "manifest_sha256": sha256_file(manifest_path),
        "declared_file_count": manifest.get("file_count", 0),
        "missing_files": len(missing),
        "mismatched_files": len(mismatched),
        "hashes_recomputed": verify_hashes,
        "nested_payloads_checked": nested_payloads_checked,
    }


def _result_class(statuses: dict[str, Any]) -> str:
    prediction = statuses["phase_05"].scientific_outcome
    early = statuses["phase_06"].scientific_outcome
    if prediction == "negative":
        return "Negative"
    if prediction == "positive" and early == "positive":
        return "Positive"
    if prediction in {"positive", "limited"}:
        return "Limited"
    return "Inconclusive"


def _metric_text(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_ids = [f"phase_{index:02d}" for index in range(7)]
    statuses = {
        phase_id: load_phase_status(args.artifacts_root / phase_id) for phase_id in phase_ids
    }
    incomplete = [
        phase_id for phase_id, status in statuses.items() if status.technical_status != "passed"
    ]
    if incomplete:
        raise RuntimeError(f"Cannot synthesize technically incomplete phases: {incomplete}")
    manifest_index = [
        _validate_manifest(
            args.artifacts_root / phase_id,
            verify_hashes=args.verify_hashes,
        )
        for phase_id in phase_ids
    ]
    result_class = _result_class(statuses)
    lines = [
        "# ReasonBench Final Evidence Synthesis",
        "",
        f"Generated: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Result classification",
        "",
        f"**{result_class}**",
        "",
        (
            "This classification follows the preregistered interpretation: a Positive "
            "result requires improvement beyond difficulty plus length and useful "
            "early-prefix evidence; a Limited result indicates useful but inconsistent "
            "evidence; a "
            "Negative result indicates no reliable primary improvement."
        ),
        "",
        "## Phase evidence",
        "",
        "| Phase | Technical status | Scientific outcome | Decision | Key metrics |",
        "|---|---|---|---|---|",
    ]
    for phase_id in phase_ids:
        status = statuses[phase_id]
        lines.append(
            f"| {phase_id} | {status.technical_status} | "
            f"{status.scientific_outcome} | {status.next_decision} | "
            f"`{_metric_text(status.metrics)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Generated reasoning traces are observable outputs, not guaranteed faithful accounts of internal reasoning.",
            "- The primary predictors exclude final-answer tokens; whole-output analyses are sensitivity checks only.",
            "- Cross-model claims use normalized scalar summaries, never raw hidden coordinates.",
            "- Phase 2 is exploratory; Phase 3 applies its frozen hypotheses to newly added Gemma problems and to the previously unseen Ministral model family, not to an independent dataset.",
            "- Phase 2 seed-instability and geometry/spectral associations control for level, category, and mean reasoning length.",
            "- Phase 4 extends the generation cap from 8,192 to 16,384 tokens and tests early failure on held-out problems; trajectories still hitting 16K remain right-censored.",
            "- The Phase 3 and Phase 4 matched panels use one seed; sampling variability is not estimated there.",
            "- The current matched panel is MATH-only; model and dataset transfer remain deferred.",
            "- Early-prefix results are conditional on trajectories that remain active at each fixed prefix; coverage is reported.",
            "- A candidate stopping policy is not deployed or validated by this project.",
            "",
            "## Reproducibility",
            "",
            (
                "Each completed evidence-producing phase has a machine-readable status, "
                "a Markdown report, and a content manifest. The accompanying "
                "reproducibility index records every included manifest."
            ),
            "",
        ]
    )
    write_text_atomic(args.output_dir / "final_report.md", "\n".join(lines))
    write_json_atomic(
        args.output_dir / "reproducibility_manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "result_class": result_class,
            "phase_manifests": manifest_index,
        },
    )
    write_json_atomic(
        args.output_dir / "phase_summary.json",
        {
            "technical_status": "passed",
            "scientific_outcome": result_class.lower(),
            "next_decision": "review_final_evidence",
            "summary": "All accepted phase evidence was synthesized and indexed.",
            "metrics": {
                "result_class": result_class,
                "phases_synthesized": len(phase_ids),
                "manifests_validated": len(manifest_index),
            },
            "warnings": [
                "Phase 2 is exploratory rather than confirmatory.",
                "GSM8K out-of-domain confirmation is not part of the current matched panel.",
            ],
        },
    )
    print(args.output_dir / "final_report.md")


if __name__ == "__main__":
    main()
