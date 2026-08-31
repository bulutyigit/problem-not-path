from __future__ import annotations

import json
from pathlib import Path

EXPECTED_NOTEBOOKS = [
    "00_research_readiness.ipynb",
    "01_reasoning_mode_ablation.ipynb",
    "02_gemma_math_difficulty_dynamics.ipynb",
    "03_cross_model_comparison.ipynb",
    "04_16k_early_failure_study.ipynb",
    "05_correctness_prediction.ipynb",
    "06_early_prediction_and_spectral_analysis.ipynb",
    "07_final_synthesis.ipynb",
]


def test_notebook_sequence_and_code_cell_syntax() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    assert sorted(path.name for path in root.glob("*.ipynb")) == EXPECTED_NOTEBOOKS
    for notebook_name in EXPECTED_NOTEBOOKS:
        path = root / notebook_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
        for cell_index, cell in enumerate(payload["cells"]):
            assert cell["cell_type"] in {"markdown", "code"}
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell["source"]),
                    f"{notebook_name}:cell_{cell_index}",
                    "exec",
                )


def test_notebooks_use_drive_first_project_root() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for notebook_name in EXPECTED_NOTEBOOKS:
        text = (root / notebook_name).read_text(encoding="utf-8")
        assert "/content/drive/MyDrive/how_models_reason" in text
        assert "requirements-colab.lock" in text
        assert "--no-deps" in text
        assert "REPOSITORY_URL" not in text
