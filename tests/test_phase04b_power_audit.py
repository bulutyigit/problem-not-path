from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from reasonbench.storage import read_json


def _features(positive_test_problems: int) -> pd.DataFrame:
    rows = []
    for index in range(40):
        split = "train" if index < 24 else "validation" if index < 30 else "test"
        positive = index % 2 == 0 if split != "test" else index - 30 < positive_test_problems
        for model in ("gemma4_e4b", "qwen35_4b", "ministral3_3b"):
            for seed in (11, 23):
                rows.append(
                    {
                        "run_id": f"{model}-{index}-{seed}",
                        "model_key": model,
                        "research_split": split,
                        "level": index % 5 + 1,
                        "problem_id": f"p{index}",
                        "seed": seed,
                        "correct": positive,
                        "finish_reason": "eos",
                        "parse_status": "boxed",
                    }
                )
    return pd.DataFrame(rows)


def _run(tmp_path: Path, positive_test_problems: int) -> dict:
    root = Path(__file__).resolve().parents[1]
    features = tmp_path / "features.parquet"
    analysis = tmp_path / "analysis"
    _features(positive_test_problems).to_parquet(features, index=False)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_phase04b_power.py"),
            "--features",
            str(features),
            "--output-dir",
            str(analysis),
        ],
        cwd=root,
        check=True,
    )
    return read_json(analysis / "phase04b_power_audit.json")


def test_phase04b_power_audit_passes_at_five_positive_test_problem_clusters(tmp_path: Path) -> None:
    audit = _run(tmp_path, positive_test_problems=5)
    assert audit["predictor_eligible"]
    assert audit["pooled_test_positive_problem_clusters"] == 5
    assert audit["pooled_test_negative_problem_clusters"] == 5


def test_phase04b_power_audit_marks_sparse_endpoints_descriptive_only(tmp_path: Path) -> None:
    audit = _run(tmp_path, positive_test_problems=4)
    assert not audit["predictor_eligible"]
    assert audit["scientific_outcome"] == "descriptive_only"
