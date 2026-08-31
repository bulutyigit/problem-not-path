#!/usr/bin/env python
"""Validate the required Phase 7 synthesis artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.storage import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        "final_report.md",
        "reproducibility_manifest.json",
        "phase_summary.json",
    ]
    missing = [name for name in required if not (args.phase_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 7 artifacts: {missing}")
    report = (args.phase_dir / "final_report.md").read_text(encoding="utf-8")
    for heading in [
        "# ReasonBench Final Evidence Synthesis",
        "## Result classification",
        "## Phase evidence",
        "## Interpretation boundaries",
        "## Reproducibility",
    ]:
        if heading not in report:
            raise ValueError(f"Final report is missing required heading: {heading}")
    manifest = read_json(args.phase_dir / "reproducibility_manifest.json")
    if len(manifest.get("phase_manifests", [])) != 7:
        raise ValueError("The reproducibility manifest must index Phases 0 through 6")
    print("Phase 7 synthesis validation passed.")


if __name__ == "__main__":
    main()
