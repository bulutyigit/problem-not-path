#!/usr/bin/env python
"""Run one ReasonBench phase on a persistent RunPod workspace.

This is the non-notebook counterpart of the checked-in Colab notebooks. It
keeps the same phase gates and commands while writing durable progress and log
files outside the individual phase directories.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from reasonbench.colab import assert_supported_gpu
from reasonbench.constants import PHASE4_DENSE_PREFIXES, PHASE_IDS
from reasonbench.phases import require_phase_gate
from reasonbench.storage import ensure_directory, read_json, write_json_atomic


@dataclass(frozen=True)
class Experiment:
    config: str
    output_name: str


@dataclass(frozen=True)
class ParallelScript:
    label: str
    script_name: str
    arguments: tuple[object, ...]
    progress_path: Path | None = None


@dataclass(frozen=True)
class GenerationPhase:
    previous_phase: str
    allowed_decisions: frozenset[str]
    experiments: tuple[Experiment, ...]
    expected_trajectories: int
    condition_column: str
    paired_conditions: tuple[str, str] | None = None
    dataset_directory_name: str = "datasets"


GENERATION_PHASES = {
    "phase_01": GenerationPhase(
        previous_phase="phase_00",
        allowed_decisions=frozenset({"continue"}),
        experiments=(
            Experiment("configs/experiments/phase_01_gemma4_reasoning.yaml", "reasoning"),
            Experiment(
                "configs/experiments/phase_01_gemma4_non_reasoning.yaml",
                "non_reasoning",
            ),
        ),
        expected_trajectories=800,
        condition_column="model_mode",
        paired_conditions=("non_reasoning", "reasoning"),
    ),
    "phase_02": GenerationPhase(
        previous_phase="phase_01",
        allowed_decisions=frozenset({"continue"}),
        experiments=(
            Experiment("configs/experiments/phase_02_gemma4_math_difficulty.yaml", "gemma4"),
        ),
        expected_trajectories=400,
        condition_column="level",
    ),
    "phase_03": GenerationPhase(
        previous_phase="phase_02",
        allowed_decisions=frozenset({"continue", "freeze_hypotheses", "review"}),
        experiments=(
            Experiment("configs/experiments/phase_03_gemma4.yaml", "gemma4"),
            Experiment("configs/experiments/phase_03_qwen35.yaml", "qwen35"),
            Experiment("configs/experiments/phase_03_ministral3.yaml", "ministral3"),
        ),
        expected_trajectories=300,
        condition_column="model_key",
    ),
    "phase_04": GenerationPhase(
        previous_phase="phase_03",
        allowed_decisions=frozenset({"continue"}),
        experiments=(
            Experiment("configs/experiments/phase_04_gemma4_16k.yaml", "gemma4"),
            Experiment("configs/experiments/phase_04_qwen35_16k.yaml", "qwen35"),
            Experiment("configs/experiments/phase_04_ministral3_16k.yaml", "ministral3"),
        ),
        expected_trajectories=300,
        condition_column="model_key",
    ),
    "phase_04b": GenerationPhase(
        previous_phase="phase_04",
        allowed_decisions=frozenset({"run_prediction"}),
        experiments=(
            Experiment("configs/experiments/phase_04b_gemma4_16k.yaml", "gemma4"),
            Experiment("configs/experiments/phase_04b_qwen35_16k.yaml", "qwen35"),
            Experiment("configs/experiments/phase_04b_ministral3_16k.yaml", "ministral3"),
        ),
        expected_trajectories=600,
        condition_column="model_key",
        dataset_directory_name="datasets_v2",
    ),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run one resumable ReasonBench phase on RunPod.")
    parser.add_argument("phase", choices=PHASE_IDS)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--minimum-completion-rate", type=float, default=0.98)
    parser.add_argument("--dataset-sample-size", type=int, default=200)
    parser.add_argument("--smoke-max-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-allocated-gib", type=float, default=35.0)
    parser.add_argument(
        "--generation-workers",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Concurrent generation workers. Benchmark 3–4 workers on an 80 GB GPU before a full run.",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="Concurrent CPU analysis workers.",
    )
    parser.add_argument("--rebuild-datasets", action="store_true")
    parser.add_argument("--rerun-smoke", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        help="Frozen Phase 4c breakthrough-probe manifest; built outcome-blind when omitted.",
    )
    parser.add_argument(
        "--probe-problem-count",
        type=int,
        default=20,
        help=("Matched MATH problems in the frozen Phase 4c labeling cohort."),
    )
    parser.add_argument(
        "--probe-pilot-only",
        action="store_true",
        help=(
            "Run the manifest's five-problem level-balanced pilot only. Rerun without this "
            "flag after reviewing runtime, event rate, and censoring; compatible branches resume."
        ),
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Stop after generation validation; run CPU analysis locally after copying artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without requiring a GPU, token, or prior artifacts.",
    )
    return parser.parse_args()


class RunPodPipeline:
    """Run the existing phase scripts with persistent paths and progress."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = args.project_root.expanduser().resolve()
        default_artifacts = os.environ.get("REASONBENCH_ARTIFACTS_ROOT")
        self.artifacts_root = (
            (
                args.artifacts_root
                or (
                    Path(default_artifacts)
                    if default_artifacts
                    else self.project_root / "artifacts"
                )
            )
            .expanduser()
            .resolve()
        )
        self.phase_root = self.artifacts_root / args.phase
        self.shared_root = self.artifacts_root / "shared"
        self.logs_root = self.artifacts_root / "runpod_logs" / args.phase
        self.progress_path = self.artifacts_root / "runpod_pipeline_progress.json"
        self.phase_id = args.phase
        self.started_at = datetime.now(UTC).isoformat()
        self.started_clock = time.monotonic()
        self.completed_steps = 0
        self.total_steps = 0
        self.current_step: str | None = None
        self.failed_step: str | None = None

        self._validate_project()
        for directory in (
            self.artifacts_root,
            self.phase_root,
            self.shared_root,
            self.logs_root,
        ):
            ensure_directory(directory)
        self._configure_cache()

    def _validate_project(self) -> None:
        required = ("pyproject.toml", "PLAN.md", "scripts")
        missing = [name for name in required if not (self.project_root / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"ReasonBench project is incomplete at {self.project_root}; missing {missing}"
            )

    def _configure_cache(self) -> None:
        cache_root = self.project_root.parent / ".cache" / "huggingface"
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if not self.args.dry_run:
            ensure_directory(Path(os.environ["HF_HOME"]))

    def _progress_payload(self, status: str) -> dict:
        return {
            "status": status,
            "phase_id": self.phase_id,
            "current_step": self.current_step,
            "failed_step": self.failed_step,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(time.monotonic() - self.started_clock, 3),
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "project_root": str(self.project_root),
            "artifacts_root": str(self.artifacts_root),
            "hf_home": os.environ["HF_HOME"],
            "generation_workers": self.args.generation_workers,
            "analysis_workers": self.args.analysis_workers,
        }

    def _write_progress(self, status: str) -> None:
        write_json_atomic(self.progress_path, self._progress_payload(status))

    def _begin(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._write_progress("running")
        print(f"RunPod phase: {self.phase_id}")
        print(f"Project: {self.project_root}")
        print(f"Artifacts: {self.artifacts_root}")
        print(f"Hugging Face cache: {os.environ['HF_HOME']}")

    def _finish_step(self) -> None:
        self.completed_steps += 1
        self.current_step = None
        self._write_progress("running")

    def _fail_step(self) -> None:
        self.failed_step = self.current_step
        self._write_progress("failed")

    def _run_action(self, label: str, action: Callable[[], None]) -> None:
        self.current_step = label
        self._write_progress("running")
        print(f"\n[{self.completed_steps + 1}/{self.total_steps}] {label}", flush=True)
        if self.args.dry_run:
            print("(dry-run: local artifact operation)")
            self._finish_step()
            return
        try:
            action()
        except Exception:
            self._fail_step()
            raise
        self._finish_step()

    def _run_script(self, label: str, script_name: str, *arguments: object) -> None:
        command = [
            sys.executable,
            "-u",
            str(self.project_root / "scripts" / script_name),
            *[str(argument) for argument in arguments],
        ]
        self.current_step = label
        self._write_progress("running")
        printable = shlex.join(command)
        print(f"\n[{self.completed_steps + 1}/{self.total_steps}] {label}")
        print(f"$ {printable}", flush=True)
        if self.args.dry_run:
            self._finish_step()
            return

        log_path = self.logs_root / f"{self.completed_steps + 1:02d}_{_slug(label)}.log"
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{datetime.now(UTC).isoformat()}] $ {printable}\n")
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if process.stdout is None:
                    raise RuntimeError("Could not capture child-process output")
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_handle.write(line)
                    log_handle.flush()
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)
        except Exception:
            self._fail_step()
            raise
        self._finish_step()

    def _run_parallel_scripts(
        self,
        label: str,
        invocations: tuple[ParallelScript, ...],
    ) -> None:
        commands = [
            [
                sys.executable,
                "-u",
                str(self.project_root / "scripts" / invocation.script_name),
                *[str(argument) for argument in invocation.arguments],
            ]
            for invocation in invocations
        ]
        self.current_step = label
        self._write_progress("running")
        print(f"\n[{self.completed_steps + 1}/{self.total_steps}] {label}")
        for invocation, command in zip(invocations, commands, strict=True):
            print(f"$ [{invocation.label}] {shlex.join(command)}")
        if self.args.dry_run:
            self._finish_step()
            return

        running = []
        try:
            for invocation, command in zip(invocations, commands, strict=True):
                log_path = self.logs_root / (
                    f"{self.completed_steps + 1:02d}_{_slug(label)}_{_slug(invocation.label)}.log"
                )
                log_handle = log_path.open("a", encoding="utf-8")
                log_handle.write(f"\n[{datetime.now(UTC).isoformat()}] $ {shlex.join(command)}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=os.environ.copy(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append((invocation, command, process, log_handle, log_path))
                print(f"{invocation.label} log: {log_path}")

            next_report = 0.0
            while True:
                failed = [item for item in running if (item[2].poll() or 0) != 0]
                if failed:
                    _invocation, command, process, _handle, _path = failed[0]
                    raise subprocess.CalledProcessError(process.returncode, command)
                if all(process.poll() is not None for _, _, process, _, _ in running):
                    break
                now = time.monotonic()
                if now >= next_report:
                    progress = []
                    for invocation, _, process, _, _ in running:
                        if invocation.progress_path and invocation.progress_path.exists():
                            payload = read_json(invocation.progress_path)
                            progress.append(
                                f"{invocation.label}: "
                                f"{payload.get('completed_trajectories', 0)}/"
                                f"{payload.get('expected_trajectories', '?')} "
                                f"({payload.get('status', 'running')})"
                            )
                        else:
                            state = "finished" if process.poll() is not None else "starting"
                            progress.append(f"{invocation.label}: {state}")
                    print("Parallel progress — " + "; ".join(progress), flush=True)
                    next_report = now + 30.0
                time.sleep(2.0)
        except BaseException:
            for _, _, process, _, _ in running:
                if process.poll() is None:
                    process.terminate()
            for _, _, process, _, _ in running:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self._fail_step()
            raise
        finally:
            for _, _, _, log_handle, _ in running:
                log_handle.close()
        self._finish_step()

    def _require_token_and_gpu(self) -> None:
        if self.args.dry_run:
            print("dry-run: HF_TOKEN and A100/H100 checks skipped")
            return
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN is missing. Export it in the RunPod terminal before this phase."
            )
        gpu = assert_supported_gpu()
        write_json_atomic(self.phase_root / "gpu_readiness.json", gpu)
        print(f"GPU ready: {gpu['device_name']} ({gpu['total_memory_gib']:.1f} GiB)")

    def _require_gate(self, previous_phase: str, decisions: frozenset[str]) -> None:
        if self.args.dry_run:
            print(
                f"dry-run: gate {previous_phase} -> {self.phase_id}; "
                f"allowed decisions={sorted(decisions)}"
            )
            return
        status = require_phase_gate(
            self.artifacts_root,
            previous_phase,
            allowed_decisions=set(decisions),
        )
        print(
            f"Gate passed: {previous_phase} is {status.technical_status}, "
            f"decision={status.next_decision}"
        )

    def run(self) -> None:
        try:
            if self.phase_id == "phase_00":
                self._run_phase_00()
            elif self.phase_id in GENERATION_PHASES:
                self._run_generation_phase(GENERATION_PHASES[self.phase_id])
            elif self.phase_id == "phase_04c":
                self._run_phase_04c()
            elif self.phase_id == "phase_05":
                self._run_phase_05()
            elif self.phase_id == "phase_06":
                self._run_phase_06()
            elif self.phase_id == "phase_07":
                self._run_phase_07()
            elif self.phase_id == "phase_04d":
                raise RuntimeError(
                    "Phase 4d is CPU-only. Run scripts/run_local_phase_analysis.py phase_04d "
                    "after copying Phase 4b features and Phase 4c labels locally."
                )
            elif self.phase_id in {"phase_04e", "phase_04f"}:
                raise RuntimeError(
                    f"{self.phase_id} is protocol-gated and cannot run until the Phase 4d "
                    "forecast and controller thresholds are frozen."
                )
            else:
                raise RuntimeError(f"No pipeline implementation exists for {self.phase_id}")
        except Exception:
            if self.failed_step is None:
                self.failed_step = self.current_step or "phase setup"
                self._write_progress("failed")
            raise
        self.current_step = None
        self._write_progress("complete")
        if self.args.generation_only or self.phase_id == "phase_04c":
            print(f"\n{self.phase_id} generation complete; local analysis is still required.")
        else:
            print(f"\n{self.phase_id} complete. Review {self.phase_root / 'phase_report.md'}")

    def _run_phase_04c(self) -> None:
        """Run sparse, exact-prefix continuations for breakthrough labels."""

        experiments = (
            Experiment("configs/experiments/phase_04b_gemma4_16k.yaml", "gemma4"),
            Experiment("configs/experiments/phase_04b_qwen35_16k.yaml", "qwen35"),
            Experiment("configs/experiments/phase_04b_ministral3_16k.yaml", "ministral3"),
        )
        self._begin(total_steps=6 if self.args.probe_pilot_only else 5)
        self._require_token_and_gpu()
        base_generation = self.artifacts_root / "phase_04b" / "generation"
        base_validation = self.artifacts_root / "phase_04b" / "generation_validation.json"
        if not self.args.dry_run and (
            not base_validation.exists() or not read_json(base_validation).get("valid")
        ):
            raise RuntimeError("Phase 4c requires a passing Phase 4b generation_validation.json")
        manifest = (
            self.args.probe_manifest.expanduser().resolve()
            if self.args.probe_manifest
            else self.phase_root / "breakthrough_probe_manifest.json"
        )
        if self.args.probe_manifest and not self.args.dry_run:
            self._run_action(
                "reuse frozen outcome-blind breakthrough cohort",
                lambda: print(f"Using probe manifest: {manifest}"),
            )
        else:
            self._run_script(
                "freeze outcome-blind breakthrough cohort",
                "build_breakthrough_probe_manifest.py",
                "--generation-dir",
                base_generation,
                "--output",
                manifest,
                "--problem-count",
                self.args.probe_problem_count,
            )
        readiness = (
            self.artifacts_root / "phase_04b" / "preflight" / "smoke" / ("model_readiness.json")
        )
        if not self.args.dry_run and not readiness.exists():
            readiness = self.artifacts_root / "phase_00" / "model_readiness.json"
        probe_roots: list[Path] = []
        for experiment in experiments:
            model_root = ensure_directory(self.phase_root / "probes" / experiment.output_name)
            probe_roots.append(model_root)
            common: list[object] = [
                "--project-root",
                self.project_root,
                "--config",
                experiment.config,
                "--readiness-manifest",
                readiness,
                "--base-run-dir",
                base_generation,
                "--probe-manifest",
                manifest,
                "--resume",
            ]
            if self.args.probe_pilot_only:
                common.append("--pilot-only")
            if self.args.generation_workers == 1:
                self._run_script(
                    f"probe breakthrough basin for {experiment.output_name}",
                    "generate_breakthrough_probes.py",
                    *common,
                    "--output-dir",
                    model_root,
                )
                continue
            invocations = []
            for shard_index in range(self.args.generation_workers):
                shard_root = model_root / f"shard_{shard_index:02d}"
                invocations.append(
                    ParallelScript(
                        label=f"shard {shard_index + 1}/{self.args.generation_workers}",
                        script_name="generate_breakthrough_probes.py",
                        arguments=(
                            *common,
                            "--output-dir",
                            shard_root,
                            "--shard-count",
                            self.args.generation_workers,
                            "--shard-index",
                            shard_index,
                        ),
                        progress_path=shard_root / "probe_progress.json",
                    )
                )
            self._run_parallel_scripts(
                f"probe {experiment.output_name} with {self.args.generation_workers} workers",
                tuple(invocations),
            )
        validation_arguments: list[object] = [
            "--probe-manifest",
            manifest,
            "--output-dir",
            self.phase_root,
        ]
        for probe_root in probe_roots:
            validation_arguments.extend(("--probe-dir", probe_root))
        if self.args.probe_pilot_only:
            validation_arguments.append("--pilot-only")
        self._run_script(
            "validate and merge breakthrough probes",
            "validate_breakthrough_probes.py",
            *validation_arguments,
        )
        if self.args.probe_pilot_only:
            self._run_script(
                "audit pilot cost, memory, verifier, and label diversity",
                "audit_breakthrough_pilot.py",
                "--phase-dir",
                self.phase_root,
                "--output-dir",
                self.phase_root,
                "--maximum-safe-allocated-gib",
                self.args.maximum_allocated_gib,
            )
        print(
            "Phase 4c GPU work is complete. Copy phase_04b/features and phase_04c locally, "
            "then run scripts/run_local_phase_analysis.py phase_04d."
        )

    def _run_phase_00(self) -> None:
        self._begin(total_steps=4)
        self._require_token_and_gpu()
        datasets_dir = self.shared_root / "datasets"
        manifest = datasets_dir / "dataset_manifest.json"

        if self.args.rebuild_datasets or not manifest.exists() or self.args.dry_run:
            self._run_script(
                "prepare immutable datasets",
                "prepare_datasets.py",
                "--output-dir",
                datasets_dir,
                "--gsm8k-size",
                self.args.dataset_sample_size,
                "--math-size",
                self.args.dataset_sample_size,
            )
        else:
            self._run_action(
                "reuse immutable datasets",
                lambda: print(f"Reusing dataset bundle: {datasets_dir}"),
            )

        def copy_readiness_artifacts() -> None:
            shutil.copy2(manifest, self.phase_root / "dataset_manifest.json")
            shutil.copy2(
                datasets_dir / "verifier_audit.parquet",
                self.phase_root / "verifier_audit.parquet",
            )
            splits_dir = ensure_directory(self.shared_root / "splits")
            for split_file in datasets_dir.glob("*_splits.json"):
                shutil.copy2(split_file, splits_dir / split_file.name)

        self._run_action("copy dataset readiness artifacts", copy_readiness_artifacts)

        readiness_path = self.phase_root / "model_readiness.json"
        reuse_smoke = False
        if readiness_path.exists() and not self.args.rerun_smoke:
            reuse_smoke = bool(read_json(readiness_path).get("all_ready"))
        if reuse_smoke and not self.args.dry_run:
            self._run_action(
                "reuse passing model smoke tests",
                lambda: print(f"Reusing passing smoke tests: {readiness_path}"),
            )
        else:
            smoke_arguments: list[object] = [
                "--project-root",
                self.project_root,
                "--output-dir",
                self.phase_root,
                "--max-new-tokens",
                self.args.smoke_max_new_tokens,
                "--maximum-allocated-gib",
                self.args.maximum_allocated_gib,
            ]
            for config in (
                "configs/models/gemma4_e4b_reasoning.yaml",
                "configs/models/qwen35_4b_reasoning.yaml",
                "configs/models/ministral3_3b_reasoning.yaml",
            ):
                smoke_arguments.extend(("--model-config", config))
            self._run_script(
                "smoke test all checkpoints",
                "smoke_test_models.py",
                *smoke_arguments,
            )

        def verify_smoke() -> None:
            if not read_json(readiness_path).get("all_ready"):
                raise RuntimeError(
                    "At least one checkpoint needs adapter review; inspect model_readiness.json."
                )

        if not self.args.dry_run:
            verify_smoke()
        self._run_script(
            "write phase report",
            "create_phase_report.py",
            "--phase",
            self.phase_id,
            "--run-dir",
            self.phase_root,
        )

    def _run_generation_phase(self, spec: GenerationPhase) -> None:
        analysis_steps = (
            2 if self.phase_id == "phase_04" else 3 if self.phase_id == "phase_04b" else 1
        )
        dataset_setup_steps = 1 if self.phase_id == "phase_04b" else 0
        preflight_steps = 2 if self.phase_id == "phase_04b" else 0
        post_generation_steps = 3 + analysis_steps
        self._begin(
            total_steps=len(spec.experiments)
            + dataset_setup_steps
            + preflight_steps
            + (1 if self.args.generation_only else post_generation_steps)
        )
        self._require_gate(spec.previous_phase, spec.allowed_decisions)
        if self.phase_id == "phase_03":
            freeze_path = self.artifacts_root / "phase_02" / "analysis" / "hypothesis_freeze.json"
            candidate_path = (
                self.artifacts_root / "phase_02" / "analysis" / "hypothesis_freeze_candidates.json"
            )
            if self.args.dry_run:
                print(
                    "dry-run: require Phase 2 hypothesis freeze or candidate record at "
                    f"{freeze_path} / {candidate_path}"
                )
            elif not freeze_path.exists() and not candidate_path.exists():
                raise FileNotFoundError(
                    "Phase 3 requires a Phase 2 hypothesis freeze or candidate record"
                )
        self._require_token_and_gpu()
        datasets_dir = self.shared_root / spec.dataset_directory_name
        if self.phase_id == "phase_04b":
            manifest = datasets_dir / "dataset_manifest.json"
            if self.args.rebuild_datasets or not manifest.exists() or self.args.dry_run:
                self._run_script(
                    "prepare immutable Phase 4b dataset bundle",
                    "prepare_phase04b_datasets.py",
                    "--output-dir",
                    datasets_dir,
                    "--historical-bundle",
                    self.shared_root / "datasets" / "math_sample.jsonl",
                )
            else:
                self._run_action(
                    "reuse immutable Phase 4b dataset bundle",
                    lambda: print(f"Reusing dataset bundle: {datasets_dir}"),
                )
        readiness_manifest = self.artifacts_root / "phase_00" / "model_readiness.json"
        if self.phase_id == "phase_04b":
            preflight_root = self.phase_root / "preflight"
            smoke_root = preflight_root / "smoke"
            self._run_script(
                "run Phase 4b real-checkpoint instrumentation smoke",
                "smoke_test_models.py",
                "--project-root",
                self.project_root,
                "--output-dir",
                smoke_root,
                "--max-new-tokens",
                self.args.smoke_max_new_tokens,
                "--maximum-allocated-gib",
                self.args.maximum_allocated_gib,
                "--model-config",
                "configs/models/gemma4_e4b_reasoning.yaml",
                "--model-config",
                "configs/models/qwen35_4b_reasoning.yaml",
                "--model-config",
                "configs/models/ministral3_3b_reasoning.yaml",
            )
            self._run_script(
                "prove 16K multi-worker generation safety",
                "preflight_phase04b_generation.py",
                "--project-root",
                self.project_root,
                "--datasets-dir",
                datasets_dir,
                "--readiness-manifest",
                smoke_root / "model_readiness.json",
                "--output-dir",
                preflight_root / "generation",
                "--config",
                "configs/experiments/phase_04b_gemma4_16k.yaml",
                "--config",
                "configs/experiments/phase_04b_qwen35_16k.yaml",
                "--config",
                "configs/experiments/phase_04b_ministral3_16k.yaml",
                "--generation-workers",
                self.args.generation_workers,
                "--maximum-allocated-gib",
                self.args.maximum_allocated_gib,
            )
            readiness_manifest = smoke_root / "model_readiness.json"
        generation_root = ensure_directory(self.phase_root / "generation")
        scoped_generation_root = (
            generation_root / spec.experiments[0].output_name
            if self.phase_id == "phase_02"
            else generation_root
        )

        for experiment in spec.experiments:
            experiment_root = generation_root / experiment.output_name
            arguments: list[object] = [
                "--project-root",
                self.project_root,
                "--config",
                experiment.config,
                "--datasets-dir",
                datasets_dir,
                "--readiness-manifest",
                readiness_manifest,
            ]
            reusable_directories = [experiment_root]
            external_reuse_directories: list[Path] = []
            if self.phase_id == "phase_02" and experiment.output_name == "gemma4":
                external_reuse_directories.append(
                    self.artifacts_root / "phase_01" / "generation" / "reasoning"
                )
            if self.phase_id == "phase_03" and experiment.output_name == "gemma4":
                external_reuse_directories.append(
                    self.artifacts_root / "phase_02" / "generation" / "gemma4"
                )
            if self.args.generation_workers == 1:
                single_arguments = [
                    *arguments,
                    "--output-dir",
                    experiment_root,
                    "--resume",
                ]
                for reusable_directory in external_reuse_directories:
                    single_arguments.extend(("--materialize-reuse-run-dir", reusable_directory))
                self._run_script(
                    f"generate {experiment.output_name}",
                    "generate.py",
                    *single_arguments,
                )
                continue

            invocations = []
            for shard_index in range(self.args.generation_workers):
                shard_root = experiment_root / f"shard_{shard_index:02d}"
                shard_arguments: list[object] = [
                    *arguments,
                    "--output-dir",
                    shard_root,
                    "--resume",
                    "--shard-count",
                    self.args.generation_workers,
                    "--shard-index",
                    shard_index,
                ]
                for reusable_directory in reusable_directories:
                    shard_arguments.extend(("--reuse-run-dir", reusable_directory))
                for reusable_directory in external_reuse_directories:
                    shard_arguments.extend(("--materialize-reuse-run-dir", reusable_directory))
                invocations.append(
                    ParallelScript(
                        label=f"shard {shard_index + 1}/{self.args.generation_workers}",
                        script_name="generate.py",
                        arguments=tuple(shard_arguments),
                        progress_path=shard_root / "generation_progress.json",
                    )
                )
            self._run_parallel_scripts(
                f"generate {experiment.output_name} with {self.args.generation_workers} workers",
                tuple(invocations),
            )

        validation_arguments: list[object] = [
            "--run-dir",
            scoped_generation_root,
            "--output-dir",
            self.phase_root,
            "--expected-trajectories",
            spec.expected_trajectories,
            "--minimum-completion-rate",
            self.args.minimum_completion_rate,
        ]
        if self.phase_id == "phase_04b":
            validation_arguments.extend(
                ("--expected-panel-manifest", datasets_dir / "dataset_manifest.json")
            )
        self._run_script(
            "validate generation",
            "validate_generation.py",
            *validation_arguments,
        )
        if self.args.generation_only:
            print(
                "Generation validation complete. Copy the phase artifacts locally and run "
                "scripts/run_local_phase_analysis.py before terminating the Pod."
            )
            return

        features_dir = self.phase_root / "features"
        feature_arguments: list[object] = [
            "--run-dir",
            scoped_generation_root,
            "--output-dir",
            features_dir,
        ]
        if self.phase_id in {"phase_02", "phase_03", "phase_04", "phase_04b"}:
            prefix_lengths = (
                PHASE4_DENSE_PREFIXES
                if self.phase_id == "phase_04b"
                else (16, 32, 64, 128, 256, 512)
                if self.phase_id == "phase_04"
                else (16, 32, 64, 128, 256, 512, 1024, 2048)
            )
            for prefix_length in prefix_lengths:
                feature_arguments.extend(("--prefix-length", prefix_length))
        self._run_script("extract features", "extract_features.py", *feature_arguments)

        analysis_dir = self.phase_root / "analysis"
        analysis_arguments: list[object] = [
            "--features",
            features_dir / "features_full.parquet",
            "--output-dir",
            analysis_dir,
            "--phase",
            self.phase_id,
            "--condition-column",
            spec.condition_column,
            "--bootstrap-repetitions",
            self.args.bootstrap_repetitions,
        ]
        if spec.paired_conditions:
            analysis_arguments.extend(
                (
                    "--paired-left",
                    spec.paired_conditions[0],
                    "--paired-right",
                    spec.paired_conditions[1],
                )
            )
        if self.phase_id == "phase_02":
            self._run_script(
                "analyze partial-trajectory failure dynamics",
                "analyze_difficulty_dynamics.py",
                "--features",
                features_dir / "features_full.parquet",
                "--features-dir",
                features_dir,
                "--run-dir",
                scoped_generation_root,
                "--output-dir",
                analysis_dir,
                "--bootstrap-repetitions",
                self.args.bootstrap_repetitions,
            )
        elif self.phase_id == "phase_03":
            self._run_parallel_scripts(
                "analyze cross-model difficulty and hidden PCA",
                (
                    ParallelScript(
                        label="cross-model difficulty",
                        script_name="analyze_cross_model_difficulty.py",
                        arguments=(
                            "--features",
                            features_dir / "features_full.parquet",
                            "--features-dir",
                            features_dir,
                            "--run-dir",
                            generation_root,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="hidden PCA",
                        script_name="analyze_hidden_pca.py",
                        arguments=(
                            "--run-dir",
                            generation_root,
                            "--output-dir",
                            analysis_dir / "hidden_pca",
                        ),
                    ),
                ),
            )
        elif self.phase_id == "phase_04":
            early_arguments: list[object] = [
                "--features-dir",
                features_dir,
                "--output-dir",
                analysis_dir,
                "--bootstrap-repetitions",
                self.args.bootstrap_repetitions,
                "--workers",
                self.args.analysis_workers,
            ]
            for prefix_length in (16, 32, 64, 128, 256, 512):
                early_arguments.extend(("--prefix-length", prefix_length))
            self._run_parallel_scripts(
                "analyze cap, early outcomes, and prefix dynamics",
                (
                    ParallelScript(
                        label="8K to 16K cap extension",
                        script_name="analyze_cap_extension.py",
                        arguments=(
                            "--baseline-features",
                            self.artifacts_root / "phase_03" / "features" / "features_full.parquet",
                            "--extended-features",
                            features_dir / "features_full.parquet",
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="held-out early failure prediction",
                        script_name="evaluate_early_prediction.py",
                        arguments=tuple(early_arguments),
                    ),
                    ParallelScript(
                        label="held-out reasoning duration prediction",
                        script_name="evaluate_early_length.py",
                        arguments=(
                            "--features-dir",
                            features_dir,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="prefix and spectral dynamics",
                        script_name="analyze_phase04_dynamics.py",
                        arguments=(
                            "--features-dir",
                            features_dir,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="relative spectral evolution",
                        script_name="analyze_phase04_relative_spectral.py",
                        arguments=(
                            "--generation-dir",
                            generation_root,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                ),
            )
            self._run_script(
                "finalize combined Phase 4 evidence",
                "finalize_phase_04.py",
                "--cap-summary",
                analysis_dir / "cap_extension_summary.json",
                "--early-summary",
                analysis_dir / "early_summary.json",
                "--length-summary",
                analysis_dir / "length_prediction_summary.json",
                "--dynamics-summary",
                analysis_dir / "dynamics_summary.json",
                "--output",
                analysis_dir / "phase_summary.json",
            )
        elif self.phase_id == "phase_04b":
            power_audit = analysis_dir / "phase04b_power_audit.json"
            self._run_script(
                "run blinded Phase 4b label-only power audit",
                "audit_phase04b_power.py",
                "--features",
                features_dir / "features_full.parquet",
                "--output-dir",
                analysis_dir,
                "--target-column",
                "correct",
            )
            self._run_parallel_scripts(
                "run descriptive Phase 4b trajectory analyses",
                (
                    ParallelScript(
                        label="prefix and spectral dynamics",
                        script_name="analyze_phase04_dynamics.py",
                        arguments=(
                            "--features-dir",
                            features_dir,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="relative spectral evolution",
                        script_name="analyze_phase04_relative_spectral.py",
                        arguments=(
                            "--generation-dir",
                            generation_root,
                            "--output-dir",
                            analysis_dir,
                            "--bootstrap-repetitions",
                            self.args.bootstrap_repetitions,
                        ),
                    ),
                    ParallelScript(
                        label="hidden PCA temporal flow",
                        script_name="analyze_hidden_pca.py",
                        arguments=(
                            "--run-dir",
                            generation_root,
                            "--output-dir",
                            analysis_dir / "hidden_pca",
                        ),
                    ),
                ),
            )
            self._run_script(
                "finalize Phase 4b diagnostic evidence",
                "finalize_phase_04b.py",
                "--power-audit",
                power_audit,
                "--dynamics-summary",
                analysis_dir / "dynamics_summary.json",
                "--output",
                analysis_dir / "phase_summary.json",
            )
        else:
            self._run_script(
                "analyze conditions",
                "analyze_conditions.py",
                *analysis_arguments,
            )
        self._write_report(analysis_dir / "phase_summary.json")

    def _run_phase_05(self) -> None:
        self._begin(total_steps=2)
        self._require_gate("phase_04", frozenset({"run_prediction"}))
        analysis_dir = self.phase_root / "analysis"
        self._run_script(
            "train correctness predictors",
            "train_predictors.py",
            "--features",
            self.artifacts_root / "phase_04" / "features" / "features_full.parquet",
            "--output-dir",
            analysis_dir,
            "--bootstrap-repetitions",
            self.args.bootstrap_repetitions,
            "--workers",
            self.args.analysis_workers,
        )
        self._write_report(analysis_dir / "phase_summary.json")

    def _run_phase_06(self) -> None:
        self._begin(total_steps=4)
        self._require_gate("phase_05", frozenset({"run_early_prediction"}))
        prefix_lengths = (16, 32, 64, 128, 256, 512, 1024, 2048)
        features_dir = self.phase_root / "features"
        feature_arguments: list[object] = [
            "--run-dir",
            self.artifacts_root / "phase_04" / "generation",
            "--output-dir",
            features_dir,
        ]
        for prefix in prefix_lengths:
            feature_arguments.extend(("--prefix-length", prefix))
        self._run_script(
            "extract prefix features",
            "extract_features.py",
            *feature_arguments,
        )

        analysis_dir = self.phase_root / "analysis"
        early_arguments: list[object] = [
            "--features-dir",
            features_dir,
            "--output-dir",
            analysis_dir,
            "--bootstrap-repetitions",
            self.args.bootstrap_repetitions,
            "--workers",
            self.args.analysis_workers,
        ]
        for prefix in prefix_lengths:
            early_arguments.extend(("--prefix-length", prefix))
        self._run_script(
            "evaluate early prediction",
            "evaluate_early_prediction.py",
            *early_arguments,
        )
        self._run_script(
            "evaluate spectral increment",
            "evaluate_spectral_increment.py",
            "--prediction-root",
            self.artifacts_root / "phase_05" / "analysis",
            "--early-results-dir",
            analysis_dir,
            "--output-dir",
            analysis_dir,
            "--bootstrap-repetitions",
            self.args.bootstrap_repetitions,
            "--workers",
            self.args.analysis_workers,
        )
        self._write_report(analysis_dir / "phase_summary.json")

    def _run_phase_07(self) -> None:
        self._begin(total_steps=3)
        self._require_gate(
            "phase_06",
            frozenset(
                {
                    "candidate_for_stopping",
                    "signal_too_late",
                    "insufficient_calibration",
                    "no_incremental_spectral_value",
                }
            ),
        )
        arguments: list[object] = [
            "--artifacts-root",
            self.artifacts_root,
            "--output-dir",
            self.phase_root,
        ]
        if not self.args.skip_hash_verification:
            arguments.append("--verify-hashes")
        self._run_script("build final report", "build_final_report.py", *arguments)
        self._run_script(
            "validate final report",
            "validate_final_report.py",
            "--phase-dir",
            self.phase_root,
        )
        self._write_report(self.phase_root / "phase_summary.json")

    def _write_report(self, summary_json: Path) -> None:
        self._run_script(
            "write phase report",
            "create_phase_report.py",
            "--phase",
            self.phase_id,
            "--run-dir",
            self.phase_root,
            "--summary-json",
            summary_json,
        )


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value.lower())
    return "_".join(part for part in normalized.split("_") if part)


def main() -> None:
    RunPodPipeline(parse_args()).run()


if __name__ == "__main__":
    main()
