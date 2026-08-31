from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from reasonbench.storage import read_json, write_json_atomic


def test_phase_04_finalizer_combines_cap_and_early_evidence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cap_path = tmp_path / "cap.json"
    early_path = tmp_path / "early.json"
    length_path = tmp_path / "length.json"
    dynamics_path = tmp_path / "dynamics.json"
    output_path = tmp_path / "phase_summary.json"
    write_json_atomic(
        cap_path,
        {
            "technical_status": "passed",
            "metrics": {"baseline_eos_reproduction_mismatches": 0},
            "warnings": ["cap warning"],
        },
    )
    write_json_atomic(
        early_path,
        {
            "technical_status": "passed",
            "metrics": {"confirmatory_positive": True},
            "warnings": ["early warning"],
        },
    )
    write_json_atomic(
        length_path,
        {"technical_status": "passed", "metrics": {"evaluation_rows": 12}, "warnings": []},
    )
    write_json_atomic(
        dynamics_path,
        {"technical_status": "passed", "metrics": {"features": 11}, "warnings": []},
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "finalize_phase_04.py"),
            "--cap-summary",
            str(cap_path),
            "--early-summary",
            str(early_path),
            "--length-summary",
            str(length_path),
            "--dynamics-summary",
            str(dynamics_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
    )

    summary = read_json(output_path)
    assert summary["scientific_outcome"] == "positive"
    assert summary["next_decision"] == "run_prediction"
    assert summary["metrics"]["determinism"]["status"] == "clean"
    assert summary["warnings"] == ["cap warning", "early warning"]


def test_phase_04_finalizer_reports_determinism_without_blocking(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cap_path = tmp_path / "cap.json"
    early_path = tmp_path / "early.json"
    length_path = tmp_path / "length.json"
    dynamics_path = tmp_path / "dynamics.json"
    output_path = tmp_path / "phase_summary.json"
    write_json_atomic(
        cap_path,
        {
            "technical_status": "passed",
            "metrics": {
                "baseline_eos": 175,
                "baseline_eos_reproduction_mismatches": 36,
                "baseline_eos_reproduction_mismatch_rate": 36 / 175,
            },
            "warnings": ["reproduction warning"],
        },
    )
    write_json_atomic(
        early_path,
        {
            "technical_status": "passed",
            "metrics": {"confirmatory_positive": False},
            "warnings": [],
        },
    )
    write_json_atomic(
        length_path,
        {"technical_status": "passed", "metrics": {}, "warnings": []},
    )
    write_json_atomic(
        dynamics_path,
        {"technical_status": "passed", "metrics": {}, "warnings": []},
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "finalize_phase_04.py"),
            "--cap-summary",
            str(cap_path),
            "--early-summary",
            str(early_path),
            "--length-summary",
            str(length_path),
            "--dynamics-summary",
            str(dynamics_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
    )

    summary = read_json(output_path)
    assert summary["next_decision"] == "run_prediction"
    assert summary["scientific_outcome"] == "limited"
    determinism = summary["metrics"]["determinism"]
    assert determinism["status"] == "mismatched"
    assert determinism["mismatches"] == 36
    assert determinism["mismatch_rate"] == 36 / 175
    assert "reproduction warning" in summary["warnings"]
