from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from reasonbench.storage import read_json


def _features(budget: int) -> pd.DataFrame:
    rows = []
    for model_key in ("gemma4_e4b", "qwen35_4b", "ministral3_3b"):
        for problem_index in range(4):
            was_capped = problem_index >= 2
            if budget == 8192:
                finish_reason = "max_new_tokens" if was_capped else "eos"
                tokens = 8192 if was_capped else 400 + problem_index
                correct = problem_index in {0, 2}
            else:
                finish_reason = "max_new_tokens" if problem_index == 3 else "eos"
                tokens = (
                    16384
                    if problem_index == 3
                    else (9000 if problem_index == 2 else 400 + problem_index)
                )
                correct = problem_index in {0, 2}
            rows.append(
                {
                    "model_key": model_key,
                    "dataset": "math",
                    "problem_id": f"problem-{problem_index}",
                    "seed": 11,
                    "correct": correct,
                    "finish_reason": finish_reason,
                    "trajectory_token_count": tokens,
                    "generated_tokens": tokens,
                    "elapsed_seconds": tokens / 100,
                }
            )
    return pd.DataFrame(rows)


def test_cap_extension_analysis_writes_matched_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = tmp_path / "baseline.parquet"
    extended_path = tmp_path / "extended.parquet"
    output_dir = tmp_path / "analysis"
    _features(8192).to_parquet(baseline_path, index=False)
    _features(16384).to_parquet(extended_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "analyze_cap_extension.py"),
            "--baseline-features",
            str(baseline_path),
            "--extended-features",
            str(extended_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-repetitions",
            "20",
        ],
        cwd=project_root,
        check=True,
    )

    status = read_json(output_dir / "cap_extension_summary.json")
    assert status["technical_status"] == "passed"
    assert status["next_decision"] == "run_prediction"
    assert status["metrics"]["matched_trajectories"] == 12
    assert status["metrics"]["baseline_capped"] == 6
    assert status["metrics"]["resolved_to_eos_at_16k"] == 3
    assert status["metrics"]["baseline_eos"] == 6
    mismatches = status["metrics"]["baseline_eos_reproduction_mismatches"]
    assert status["metrics"]["baseline_eos_reproduction_mismatch_rate"] == mismatches / 6
    # The phase decision belongs to finalize_phase_04.py; the cap analysis
    # must not publish a competing phase_summary.json.
    assert not (output_dir / "phase_summary.json").exists()
    assert (output_dir / "cap_extension_budget_response.png").exists()
    assert (output_dir / "cap_extension_capped_fates.png").exists()
