#!/usr/bin/env python
"""Build the eight checked-in Google Colab phase notebooks.

Shared notebook behavior is generated from this file so Drive bootstrapping,
phase gates, and reporting remain consistent. The generated notebooks contain
only English prose and code.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


def _source(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def bootstrap(phase_id: str) -> str:
    return f"""
    # Edit only this path if the uploaded Drive folder has a different name.
    from pathlib import Path
    import os
    import subprocess
    import sys

    PROJECT_ROOT = Path("/content/drive/MyDrive/how_models_reason")
    PHASE_ID = "{phase_id}"

    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        print("Not running inside Google Colab; Drive mounting was skipped.")

    if not (PROJECT_ROOT / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"Project not found at {{PROJECT_ROOT}}. Upload the complete folder or "
            "change PROJECT_ROOT in this cell."
        )
    REQUIREMENTS_LOCK = PROJECT_ROOT / "requirements-colab.lock"
    if not REQUIREMENTS_LOCK.exists():
        raise FileNotFoundError(f"Pinned dependency lock not found: {{REQUIREMENTS_LOCK}}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--requirement", str(REQUIREMENTS_LOCK),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--no-deps", "-e", str(PROJECT_ROOT),
        ],
        check=True,
    )
    os.chdir(PROJECT_ROOT)

    from reasonbench.colab import resolve_colab_paths
    paths = resolve_colab_paths(PHASE_ID, PROJECT_ROOT)
    print(f"Project root: {{paths.project_root}}")
    print(f"Phase output: {{paths.phase_root}}")
    """


HELPERS = """
from IPython.display import Image, Markdown, display
import json
import time

from tqdm.auto import tqdm
from reasonbench.storage import write_json_atomic


def run_script(script_name, *arguments):
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / script_name),
        *[str(argument) for argument in arguments],
    ]
    print("$", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


class PhaseProgress:
    # Notebook-level progress across a phase's durable processing steps.

    def __init__(self, phase_id, total_steps):
        self.phase_id = phase_id
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step = None
        self.failed_step = None
        self.started = time.perf_counter()
        self.bar = tqdm(
            total=total_steps,
            desc=f"{phase_id}: workflow",
            unit="step",
            dynamic_ncols=True,
        )
        self._write_status("running")

    def run(self, label, script_name, *arguments):
        self.current_step = label
        self.bar.set_postfix_str(label)
        self._write_status("running")
        try:
            run_script(script_name, *arguments)
        except Exception:
            self.failed_step = label
            self._write_status("failed")
            raise
        self.completed_steps += 1
        self.bar.update(1)
        self._write_status("running")

    def close(self):
        self.current_step = None
        self.bar.close()
        self._write_status("complete")

    def _write_status(self, status):
        write_json_atomic(
            paths.phase_root / "notebook_progress.json",
            {
                "status": status,
                "phase_id": self.phase_id,
                "current_step": self.current_step,
                "failed_step": self.failed_step,
                "completed_steps": self.completed_steps,
                "total_steps": self.total_steps,
                "elapsed_seconds": time.perf_counter() - self.started,
            },
        )


def start_phase_progress(total_steps):
    global phase_progress
    phase_progress = PhaseProgress(PHASE_ID, total_steps)
    return phase_progress


def run_phase_step(label, script_name, *arguments):
    return phase_progress.run(label, script_name, *arguments)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show_markdown(path):
    display(Markdown(Path(path).read_text(encoding="utf-8")))
"""


def gate(previous_phase: str, decisions: tuple[str, ...]) -> str:
    ordered = sorted(set(decisions))
    decisions_literal = "{" + ", ".join(repr(value) for value in ordered) + "}"
    return f"""
    from reasonbench.phases import require_phase_gate
    previous_status = require_phase_gate(
        paths.artifacts_root,
        "{previous_phase}",
        allowed_decisions={decisions_literal},
    )
    print(previous_status)
    """


def report_cell(summary_expression: str) -> str:
    return f"""
    report_arguments = [
        "--phase", PHASE_ID,
        "--run-dir", paths.phase_root,
        "--summary-json", {summary_expression},
    ]
    if "phase_progress" in globals():
        run_phase_step("write phase report", "create_phase_report.py", *report_arguments)
        phase_progress.close()
    else:
        run_script("create_phase_report.py", *report_arguments)
    show_markdown(paths.phase_root / "phase_report.md")
    """


PHASE_00 = notebook(
    [
        markdown(
            """
            # Phase 0 — Research Readiness

            **Question:** Are the runtime, models, datasets, verifiers, and
            instrumentation reliable enough to begin?

            Select an **A100 40 GB or H100** runtime. Add `HF_TOKEN` in Colab Secrets and
            grant this notebook access. The token is never printed or persisted.
            Phase 0 makes no scientific claim; it creates the technical gate.
            """
        ),
        code(bootstrap("phase_00")),
        code(
            """
            REBUILD_DATASETS = False
            DATASET_SAMPLE_SIZE = 200
            SMOKE_MAX_NEW_TOKENS = 256
            MAXIMUM_ALLOCATED_GIB = 35.0
            """
        ),
        code(HELPERS),
        markdown(
            """
            ## Runtime and secret check

            Accepted primary runs require BF16 on one A100 or H100 with at least 39 GiB.
            Weights remain in the Colab VM cache; durable results go to Drive.
            """
        ),
        code(
            """
            from reasonbench.colab import (
                assert_supported_gpu,
                load_huggingface_token_from_colab_secret,
            )
            from reasonbench.storage import write_json_atomic

            if not load_huggingface_token_from_colab_secret():
                raise RuntimeError("Add HF_TOKEN to Colab Secrets and rerun this cell.")
            os.environ.setdefault("HF_HOME", "/content/huggingface")
            gpu = assert_supported_gpu()
            write_json_atomic(paths.phase_root / "gpu_readiness.json", gpu)
            gpu
            """
        ),
        markdown(
            """
            ## Immutable datasets and answer-verifier audit

            Resolve immutable dataset commits; sample 200 GSM8K and 200 balanced MATH
            problems; create problem-level train/validation/test assignments; and
            self-verify every stored reference answer.
            """
        ),
        code(
            """
            import shutil

            datasets_dir = paths.shared_root / "datasets"
            dataset_manifest_path = datasets_dir / "dataset_manifest.json"
            if REBUILD_DATASETS or not dataset_manifest_path.exists():
                run_script(
                    "prepare_datasets.py",
                    "--output-dir", datasets_dir,
                    "--gsm8k-size", DATASET_SAMPLE_SIZE,
                    "--math-size", DATASET_SAMPLE_SIZE,
                )
            else:
                print(f"Reusing dataset bundle: {datasets_dir}")

            shutil.copy2(
                dataset_manifest_path,
                paths.phase_root / "dataset_manifest.json",
            )
            shutil.copy2(
                datasets_dir / "verifier_audit.parquet",
                paths.phase_root / "verifier_audit.parquet",
            )
            splits_dir = paths.shared_root / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            for split_file in datasets_dir.glob("*_splits.json"):
                shutil.copy2(split_file, splits_dir / split_file.name)
            display(read_json(dataset_manifest_path))
            """
        ),
        markdown(
            """
            ## Instrumented smoke tests for all three checkpoints

            Each BF16 test verifies official model loading, immutable revision and
            license metadata, same-seed output equivalence with ordinary generation,
            exact token/logit alignment, finite scalar signals, sparse hidden capture,
            and peak allocated/reserved VRAM. Failures are recorded as `needs_adapter`;
            precision is never changed silently.
            """
        ),
        code(
            """
            model_configs = [
                "configs/models/gemma4_e4b_reasoning.yaml",
                "configs/models/qwen35_4b_reasoning.yaml",
                "configs/models/ministral3_3b_reasoning.yaml",
            ]
            arguments = [
                "--project-root", PROJECT_ROOT,
                "--output-dir", paths.phase_root,
                "--max-new-tokens", SMOKE_MAX_NEW_TOKENS,
                "--maximum-allocated-gib", MAXIMUM_ALLOCATED_GIB,
            ]
            for model_config in model_configs:
                arguments.extend(["--model-config", model_config])
            run_script("smoke_test_models.py", *arguments)

            readiness = read_json(paths.phase_root / "model_readiness.json")
            display(readiness)
            if not readiness["all_ready"]:
                raise RuntimeError(
                    "At least one checkpoint needs adapter review. Inspect its "
                    "smoke-test JSON and traceback before continuing."
                )
            """
        ),
        markdown(
            """
            ## Freeze the readiness gate

            Later notebooks require a technically passed report whose next decision
            is `continue`.
            """
        ),
        code(
            """
            run_script(
                "create_phase_report.py",
                "--phase", PHASE_ID,
                "--run-dir", paths.phase_root,
            )
            show_markdown(paths.phase_root / "phase_report.md")
            """
        ),
    ]
)


def generation_notebook(
    *,
    phase_id: str,
    title: str,
    question: str,
    design: str,
    previous_phase: str,
    decisions: tuple[str, ...],
    experiments: list[tuple[str, str]],
    expected: int,
    condition: str,
    pair: tuple[str, str] | None,
) -> dict:
    pair_line = ""
    if pair:
        pair_line = (
            f'analysis_args.extend(["--paired-left", "{pair[0]}", "--paired-right", "{pair[1]}"])'
        )
    analysis_tail = "else:"
    if phase_id == "phase_04":
        analysis_tail = """elif PHASE_ID == "phase_04":
    run_phase_step(
        "analyze 8K to 16K cap extension",
        "analyze_cap_extension.py",
        "--baseline-features",
        paths.artifacts_root / "phase_03" / "features" / "features_full.parquet",
        "--extended-features", features_dir / "features_full.parquet",
        "--output-dir", analysis_dir,
        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
    )
    early_arguments = [
        "--features-dir", features_dir,
        "--output-dir", analysis_dir,
        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
    ]
    for prefix_length in (16, 32, 64, 128, 256, 512):
        early_arguments.extend(["--prefix-length", prefix_length])
    run_phase_step(
        "evaluate held-out early failure prediction",
        "evaluate_early_prediction.py",
        *early_arguments,
    )
    run_phase_step(
        "evaluate held-out reasoning duration prediction",
        "evaluate_early_length.py",
        "--features-dir", features_dir,
        "--output-dir", analysis_dir,
        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
    )
    run_phase_step(
        "analyze prefix and spectral dynamics",
        "analyze_phase04_dynamics.py",
        "--features-dir", features_dir,
        "--output-dir", analysis_dir,
        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
    )
    run_phase_step(
        "finalize combined Phase 4 evidence",
        "finalize_phase_04.py",
        "--cap-summary", analysis_dir / "cap_extension_summary.json",
        "--early-summary", analysis_dir / "early_summary.json",
        "--length-summary", analysis_dir / "length_prediction_summary.json",
        "--dynamics-summary", analysis_dir / "dynamics_summary.json",
        "--output", analysis_dir / "phase_summary.json",
    )
    for figure in sorted(analysis_dir.glob("cap_extension_*.png")):
        display(Image(filename=str(figure)))
    for figure in sorted(analysis_dir.glob("early_failure_*.png")):
        display(Image(filename=str(figure)))
    for figure in sorted(analysis_dir.glob("early_length_*.png")):
        display(Image(filename=str(figure)))
    for figure in sorted(analysis_dir.glob("phase04_*.png")):
        display(Image(filename=str(figure)))
else:"""
    analysis_tail = analysis_tail.replace("\n", "\n                ")
    prefix_phase_ids = (
        ("phase_02", "phase_03", "phase_04") if phase_id == "phase_04" else ("phase_02", "phase_03")
    )
    prefix_lengths = (
        (16, 32, 64, 128, 256, 512)
        if phase_id == "phase_04"
        else (16, 32, 64, 128, 256, 512, 1024, 2048)
    )
    progress_expression = 'len(EXPERIMENTS) + 4 + int(PHASE_ID == "phase_03")'
    if phase_id == "phase_04":
        progress_expression = 'len(EXPERIMENTS) + 8 + int(PHASE_ID == "phase_03")'
    return notebook(
        [
            markdown(
                f"""
                # {title}

                **Question:** {question}

                {design}

                Generation is deterministic by configured seed, resumable, and
                idempotent. Completed run IDs are skipped; partial writes are never
                accepted as complete.
                """
            ),
            code(bootstrap(phase_id)),
            code(
                f"""
                BOOTSTRAP_REPETITIONS = 2000
                MINIMUM_COMPLETION_RATE = 0.98
                EXPECTED_TRAJECTORIES = {expected}
                EXPERIMENTS = {experiments!r}
                """
            ),
            code(HELPERS),
            markdown(
                """
                ## Prior-phase gate and runtime

                Scientific outcomes and technical execution are separate. The gate
                checks technical completion and the recorded next decision.
                """
            ),
            code(gate(previous_phase, decisions)),
            code(
                """
                if PHASE_ID == "phase_03":
                    hypothesis_freeze = (
                        paths.artifacts_root
                        / "phase_02"
                        / "analysis"
                        / "hypothesis_freeze.json"
                    )
                    if not hypothesis_freeze.exists():
                        raise FileNotFoundError(
                            "Phase 3 requires the Phase 2 hypothesis_freeze.json artifact."
                        )
                    display(read_json(hypothesis_freeze))
                """
            ),
            code(
                """
                from reasonbench.colab import (
                    assert_supported_gpu,
                    load_huggingface_token_from_colab_secret,
                )
                if not load_huggingface_token_from_colab_secret():
                    raise RuntimeError("HF_TOKEN is unavailable in Colab Secrets.")
                os.environ.setdefault("HF_HOME", "/content/huggingface")
                assert_supported_gpu()

                datasets_dir = paths.shared_root / "datasets"
                readiness_manifest = (
                    paths.artifacts_root / "phase_00" / "model_readiness.json"
                )
                generation_root = paths.phase_root / "generation"
                generation_root.mkdir(parents=True, exist_ok=True)
                scoped_generation_root = (
                    generation_root / EXPERIMENTS[0][1]
                    if PHASE_ID == "phase_02"
                    else generation_root
                )
                """
            ),
            markdown(
                """
                ## Resumable instrumented generation

                Full-vocabulary uncertainty is summarized online without persisting
                logits. Scalar signals are recorded per token and final-head-input
                hidden states every eight tokens. The progress bar exposes completed,
                failed, and resumed trajectories.
                """
            ),
            code(
                f"""
                phase_progress = start_phase_progress({progress_expression})
                for config_path, output_name in EXPERIMENTS:
                    generation_arguments = [
                        "generate.py",
                        "--project-root", PROJECT_ROOT,
                        "--config", config_path,
                        "--datasets-dir", datasets_dir,
                        "--readiness-manifest", readiness_manifest,
                        "--output-dir", generation_root / output_name,
                        "--resume",
                    ]
                    if PHASE_ID == "phase_02" and output_name == "gemma4":
                        generation_arguments.extend(
                            [
                                "--materialize-reuse-run-dir",
                                paths.artifacts_root
                                / "phase_01"
                                / "generation"
                                / "reasoning",
                            ]
                        )
                    if PHASE_ID == "phase_03" and output_name == "gemma4":
                        generation_arguments.extend(
                            [
                                "--materialize-reuse-run-dir",
                                paths.artifacts_root
                                / "phase_02"
                                / "generation"
                                / "gemma4",
                            ]
                        )
                    run_phase_step(f"generate {{output_name}}", *generation_arguments)
                """
            ),
            markdown(
                """
                ## Completeness validation

                Continue only when unique committed runs meet the declared completion
                threshold and contain nonempty aligned token-signal tables.
                """
            ),
            code(
                """
                validation_arguments = [
                    "--run-dir", scoped_generation_root,
                    "--output-dir", paths.phase_root,
                    "--expected-trajectories", EXPECTED_TRAJECTORIES,
                    "--minimum-completion-rate", MINIMUM_COMPLETION_RATE,
                ]
                run_phase_step(
                    "validate generation",
                    "validate_generation.py",
                    *validation_arguments,
                )
                validation = read_json(paths.phase_root / "generation_validation.json")
                display(validation)
                if not validation["valid"]:
                    raise RuntimeError(
                        "Generation is incomplete or invalid. Rerun generation or "
                        "inspect error records before analysis."
                    )
                """
            ),
            markdown(
                """
                ## Leakage-safe condition analysis

                Primary features exclude final-answer tokens. Paired contrasts align
                problem, dataset, and seed. Confidence intervals resample complete
                problem clusters.
                """
            ),
            code(
                f"""
                features_dir = paths.phase_root / "features"
                analysis_dir = paths.phase_root / "analysis"
                feature_arguments = [
                    "--run-dir", scoped_generation_root,
                    "--output-dir", features_dir,
                ]
                if PHASE_ID in {prefix_phase_ids!r}:
                    for prefix_length in {prefix_lengths!r}:
                        feature_arguments.extend(["--prefix-length", prefix_length])
                run_phase_step(
                    "extract features",
                    "extract_features.py",
                    *feature_arguments,
                )
                analysis_args = [
                    "--features", features_dir / "features_full.parquet",
                    "--output-dir", analysis_dir,
                    "--phase", PHASE_ID,
                    "--condition-column", "{condition}",
                    "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
                ]
                {pair_line}
                if PHASE_ID == "phase_02":
                    run_phase_step(
                        "analyze partial-trajectory failure dynamics",
                        "analyze_difficulty_dynamics.py",
                        "--features", features_dir / "features_full.parquet",
                        "--features-dir", features_dir,
                        "--features-dir", features_dir,
                        "--run-dir", scoped_generation_root,
                        "--output-dir", analysis_dir,
                        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
                    )
                    for figure in sorted(analysis_dir.glob("*.png")):
                        display(Image(filename=str(figure)))
                elif PHASE_ID == "phase_03":
                    run_phase_step(
                        "analyze cross-model difficulty",
                        "analyze_cross_model_difficulty.py",
                        "--features", features_dir / "features_full.parquet",
                        "--run-dir", generation_root,
                        "--output-dir", analysis_dir,
                        "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
                    )
                    pca_dir = analysis_dir / "hidden_pca"
                    run_phase_step(
                        "analyze hidden PCA",
                        "analyze_hidden_pca.py",
                        "--run-dir", generation_root,
                        "--output-dir", pca_dir,
                    )
                    display(read_json(pca_dir / "hidden_pca_summary.json"))
                    for pca_figure in sorted(pca_dir.glob("*.png")):
                        display(Image(filename=str(pca_figure)))
                {analysis_tail}
                    run_phase_step(
                        "analyze conditions",
                        "analyze_conditions.py",
                        *analysis_args,
                    )
                    display(read_json(analysis_dir / "condition_analysis.json"))
                    display(Image(filename=str(analysis_dir / "condition_accuracy.png")))
                    display(Image(filename=str(analysis_dir / "condition_profile.png")))
                    display(
                        Image(
                            filename=str(
                                analysis_dir / "correctness_feature_profile.png"
                            )
                        )
                    )
                """
            ),
            markdown(
                """
                ## Freeze and review

                Review warnings and the scientific interpretation before opening the
                next notebook.
                """
            ),
            code(report_cell('analysis_dir / "phase_summary.json"')),
        ]
    )


PHASE_01 = generation_notebook(
    phase_id="phase_01",
    title="Phase 1 — Reasoning Mode Ablation",
    question=(
        "With identical Gemma 4 E4B weights, how does enabling thinking change "
        "correctness and observable trajectory dynamics?"
    ),
    design=(
        "The same 50 GSM8K and 50 MATH problems and four seeds are run with "
        "reasoning enabled and disabled: 800 paired trajectories."
    ),
    previous_phase="phase_00",
    decisions=("continue",),
    experiments=[
        ("configs/experiments/phase_01_gemma4_reasoning.yaml", "reasoning"),
        ("configs/experiments/phase_01_gemma4_non_reasoning.yaml", "non_reasoning"),
    ],
    expected=800,
    condition="model_mode",
    pair=("non_reasoning", "reasoning"),
)

PHASE_02 = generation_notebook(
    phase_id="phase_02",
    title="Phase 2 — Gemma Partial-Trajectory Failure Dynamics",
    question=(
        "Can partial Gemma reasoning trajectories predict terminal failure beyond "
        "difficulty, category, and the number of observed tokens; and how are seed "
        "instability, hidden geometry, and spectral dynamics related?"
    ),
    design=(
        "Gemma 4 E4B IT solves a nested deterministic set of 20 problems per MATH "
        "level with four seeds: 100 problems and 400 trajectories. Fixed-token "
        "prefixes test confidence, geometry, spectral, and combined feature blocks "
        "against a difficulty/category/observed-token baseline. This is exploratory; "
        "the first 16–128 thinking tokens are separately tested for failure versus "
        "difficulty information, and hypotheses are frozen before Phase 3."
    ),
    previous_phase="phase_01",
    decisions=("continue",),
    experiments=[
        ("configs/experiments/phase_02_gemma4_math_difficulty.yaml", "gemma4"),
    ],
    expected=400,
    condition="level",
    pair=None,
)

PHASE_03 = generation_notebook(
    phase_id="phase_03",
    title="Phase 3 — Cross-Model Comparison",
    question=(
        "Under the same problems and comparable compute limits, how do reasoning "
        "dynamics differ across model families?"
    ),
    design=(
        "Gemma 4 E4B IT, Qwen3.5-4B, and Ministral 3 3B Reasoning solve the exact "
        "same 100 level-balanced MATH problems with one fixed seed under an 8,192-token "
        "limit: 300 paired trajectories. All Phase 2 Gemma runs are reused and "
        "materialized; Qwen and Ministral are generated as held-out model families. "
        "Seed-stability claims remain scoped to Phase 2."
    ),
    previous_phase="phase_02",
    decisions=("continue", "freeze_hypotheses"),
    experiments=[
        ("configs/experiments/phase_03_gemma4.yaml", "gemma4"),
        ("configs/experiments/phase_03_qwen35.yaml", "qwen35"),
        ("configs/experiments/phase_03_ministral3.yaml", "ministral3"),
    ],
    expected=300,
    condition="model_key",
    pair=None,
)


PHASE_04 = generation_notebook(
    phase_id="phase_04",
    title="Phase 4 — 16K Early-Failure Study",
    question=(
        "Do the first 16–512 reasoning tokens predict terminal failure on held-out "
        "problems once 8K censoring is reduced with a matched 16K rerun?"
    ),
    design=(
        "All three Phase 3 models solve the same 100 level-balanced MATH problems with "
        "the same seed and sampling settings. Only max_new_tokens changes from 8,192 to "
        "16,384. Cap sensitivity is the validity check; the primary analysis compares "
        "difficulty/context baselines against uncertainty, hidden-geometry, spectral, "
        "and combined prefix features on disjoint train/validation/test problem splits."
    ),
    previous_phase="phase_03",
    decisions=("continue",),
    experiments=[
        ("configs/experiments/phase_04_gemma4_16k.yaml", "gemma4"),
        ("configs/experiments/phase_04_qwen35_16k.yaml", "qwen35"),
        ("configs/experiments/phase_04_ministral3_16k.yaml", "ministral3"),
    ],
    expected=300,
    condition="model_key",
    pair=None,
)


PHASE_05 = notebook(
    [
        markdown(
            """
            # Phase 5 — Correctness Prediction

            **Question:** Do 16K trajectory features predict final-answer correctness
            better than difficulty plus reasoning length?

            Models are fit separately per language model. Training fits preprocessing
            and estimators, validation fits probability calibration, and immutable
            problem-level test data is used once for evaluation.
            """
        ),
        code(bootstrap("phase_05")),
        code("BOOTSTRAP_REPETITIONS = 2000"),
        code(HELPERS),
        code(gate("phase_04", ("run_prediction",))),
        markdown(
            """
            Feature sets span constant, difficulty, length, confidence, dynamic
            uncertainty, hidden geometry, spectral, and full combinations. The primary
            paired statistic is test AUROC for full minus difficulty-plus-length.
            """
        ),
        code(
            """
            phase_progress = start_phase_progress(2)
            features_path = (
                paths.artifacts_root / "phase_04" / "features" / "features_full.parquet"
            )
            analysis_dir = paths.phase_root / "analysis"
            run_phase_step(
                "train correctness predictors",
                "train_predictors.py",
                "--features", features_path,
                "--output-dir", analysis_dir,
                "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
            )
            display(read_json(analysis_dir / "primary_improvements.json"))
            for model_dir in sorted(p for p in analysis_dir.iterdir() if p.is_dir()):
                figure = model_dir / "ablation_auroc.png"
                if figure.exists():
                    display(Markdown(f"### {model_dir.name}"))
                    display(Image(filename=str(figure)))
                    display(Image(filename=str(model_dir / "calibration.png")))
                    display(Image(filename=str(model_dir / "full_logistic_effects.png")))
            """
        ),
        markdown(
            """
            A confidence interval above zero is positive primary evidence. An interval
            spanning zero is a limited scientific result, not a technical failure.
            """
        ),
        code(report_cell('analysis_dir / "phase_summary.json"')),
    ]
)


PHASE_06 = notebook(
    [
        markdown(
            """
            # Phase 6 — Early Prediction and Spectral Analysis

            **Questions:** How early can final correctness be predicted, and do
            spectral features add information beyond simpler dynamics?

            Fixed prefixes are operationally meaningful. Trajectories shorter than a
            prefix are not treated as though future tokens were observed; every result
            reports coverage and remaining-token opportunity.
            """
        ),
        code(bootstrap("phase_06")),
        code(
            """
            PREFIX_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
            BOOTSTRAP_REPETITIONS = 2000
            """
        ),
        code(HELPERS),
        code(gate("phase_05", ("run_early_prediction",))),
        markdown(
            """
            ## Prefix-censored feature extraction

            Scalar and hidden-state features are truncated before calculation, which
            prevents later trajectory values from leaking into early prefixes.
            """
        ),
        code(
            """
            phase_progress = start_phase_progress(4)
            generation_root = paths.artifacts_root / "phase_04" / "generation"
            features_dir = paths.phase_root / "features"
            arguments = [
                "--run-dir", generation_root,
                "--output-dir", features_dir,
            ]
            for prefix in PREFIX_LENGTHS:
                arguments.extend(["--prefix-length", prefix])
            run_phase_step(
                "extract prefix features",
                "extract_features.py",
                *arguments,
            )
            """
        ),
        markdown(
            """
            ## Early prediction and spectral increment

            Early models exclude spectral features. Spectral value is then tested with
            a paired Phase 5 comparison of `full` against `full_without_spectral` on
            the same held-out trajectories.
            """
        ),
        code(
            """
            analysis_dir = paths.phase_root / "analysis"
            arguments = [
                "--features-dir", features_dir,
                "--output-dir", analysis_dir,
                "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
            ]
            for prefix in PREFIX_LENGTHS:
                arguments.extend(["--prefix-length", prefix])
            run_phase_step(
                "evaluate early prediction",
                "evaluate_early_prediction.py",
                *arguments,
            )
            display(read_json(analysis_dir / "early_prediction_results.json"))
            display(Image(filename=str(analysis_dir / "early_prediction_auroc.png")))
            display(
                Image(filename=str(analysis_dir / "early_compute_opportunity.png"))
            )

            run_phase_step(
                "evaluate spectral increment",
                "evaluate_spectral_increment.py",
                "--prediction-root", paths.artifacts_root / "phase_05" / "analysis",
                "--early-results-dir", analysis_dir,
                "--output-dir", analysis_dir,
                "--bootstrap-repetitions", BOOTSTRAP_REPETITIONS,
            )
            display(read_json(analysis_dir / "spectral_increment_results.json"))
            """
        ),
        code(report_cell('analysis_dir / "phase_summary.json"')),
    ]
)


PHASE_07 = notebook(
    [
        markdown(
            """
            # Phase 7 — Final Synthesis

            **Question:** What conclusion is justified by the complete evidence, and
            what should the next project test?

            This phase validates accepted phase states and manifests, freezes the
            result class, records interpretation boundaries, and creates a
            reproducibility index. It does not refit scientific models.
            """
        ),
        code(bootstrap("phase_07")),
        code(
            """
            # Recomputing every Drive hash can be slow but is the strict default.
            VERIFY_ALL_HASHES = True
            """
        ),
        code(HELPERS),
        code(
            gate(
                "phase_06",
                (
                    "candidate_for_stopping",
                    "signal_too_late",
                    "insufficient_calibration",
                    "no_incremental_spectral_value",
                ),
            )
        ),
        markdown(
            """
            A Positive result requires improvement beyond difficulty plus length and
            useful calibrated early-prefix evidence. Limited and Negative results are valid research outcomes
            and do not alter technical completion.
            """
        ),
        code(
            """
            phase_progress = start_phase_progress(3)
            arguments = [
                "--artifacts-root", paths.artifacts_root,
                "--output-dir", paths.phase_root,
            ]
            if VERIFY_ALL_HASHES:
                arguments.append("--verify-hashes")
            run_phase_step("build final report", "build_final_report.py", *arguments)
            run_phase_step(
                "validate final report",
                "validate_final_report.py",
                "--phase-dir",
                paths.phase_root,
            )
            show_markdown(paths.phase_root / "final_report.md")
            """
        ),
        markdown(
            """
            Freeze the synthesis itself as a phase artifact, then download
            `artifacts/phase_07` and any earlier reports for joint interpretation.
            """
        ),
        code(report_cell('paths.phase_root / "phase_summary.json"')),
    ]
)


NOTEBOOKS = {
    "00_research_readiness.ipynb": PHASE_00,
    "01_reasoning_mode_ablation.ipynb": PHASE_01,
    "02_gemma_math_difficulty_dynamics.ipynb": PHASE_02,
    "03_cross_model_comparison.ipynb": PHASE_03,
    "04_16k_early_failure_study.ipynb": PHASE_04,
    "05_correctness_prediction.ipynb": PHASE_05,
    "06_early_prediction_and_spectral_analysis.ipynb": PHASE_06,
    "07_final_synthesis.ipynb": PHASE_07,
}


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, payload in NOTEBOOKS.items():
        path = NOTEBOOK_ROOT / filename
        path.write_text(
            json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
