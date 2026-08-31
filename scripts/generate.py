#!/usr/bin/env python
"""Run a resumable instrumented generation experiment."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from tqdm.auto import tqdm

from reasonbench.config import load_experiment_config
from reasonbench.datasets.loader import build_problem_sample
from reasonbench.datasets.splits import read_problem_bundle
from reasonbench.generation import InstrumentedGenerator
from reasonbench.generation.mlx_engine import MLXInstrumentedGenerator
from reasonbench.generation.mlx_modeling import (
    load_mlx_model_bundle,
    unload_mlx_model_bundle,
)
from reasonbench.generation.modeling import load_model_bundle, unload_model_bundle
from reasonbench.generation.storage import (
    materialize_reused_trajectory,
    trajectory_is_complete,
    trajectory_matches_metadata,
    verify_trajectory_payload,
    write_trajectory,
)
from reasonbench.instrumentation.recorder import TOKEN_METRIC_SCHEMA_VERSION
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import (
    deterministic_run_id,
    ensure_directory,
    read_json,
    sha256_file,
    write_json_atomic,
)
from reasonbench.verification import verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-run-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--materialize-reuse-run-dir",
        action="append",
        type=Path,
        default=[],
        help="External accepted runs to hard-link/copy into this experiment tree.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--maximum-problems",
        type=int,
        help=(
            "Optional deterministic prefix of the immutable problem selection. Used only by "
            "the Phase 4b GPU preflight; it never changes the stored dataset bundle."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--progress-checkpoint-tokens",
        type=int,
        default=512,
        help=(
            "Atomically refresh generation_progress.json after this many generated tokens. "
            "Completed trajectories remain the actual resumable checkpoint boundary."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Request best-effort deterministic CUDA kernels. Reproducibility "
            "still holds only within one GPU model, driver, and library stack."
        ),
    )
    return parser.parse_args()


def _resolve_revision(config, readiness_manifest: Path):
    readiness = read_json(readiness_manifest)
    model_record = readiness.get("models", {}).get(config.model.key)
    if not model_record or model_record.get("status") != "ready":
        raise RuntimeError(
            f"Model {config.model.key} is not ready according to {readiness_manifest}"
        )
    revision = model_record.get("resolved_revision")
    if not revision:
        raise RuntimeError(
            f"Smoke test did not resolve an immutable revision for {config.model.key}"
        )
    return replace(config, model=replace(config.model, revision=revision))


def _selected_problems(config, datasets_dir: Path):
    selected = []
    for dataset_config in config.datasets:
        bundle_path = datasets_dir / f"{dataset_config.name}_sample.jsonl"
        records = read_problem_bundle(bundle_path)
        sample = build_problem_sample(
            records,
            sample_size=dataset_config.sample_size,
            seed=dataset_config.seed,
            levels=dataset_config.levels,
            nested_base_sample_size=dataset_config.nested_base_sample_size,
        )
        selected.extend(sample)
    return selected


def _reusable_trajectories(
    config,
    run_directories: list[Path],
    dataset_bundle_sha256: str | None = None,
) -> dict[tuple[str, str, int], Path]:
    reusable: dict[tuple[str, str, int], Path] = {}
    expected_sampling = asdict(config.model.sampling)
    config_hash = config.config_hash() if callable(getattr(config, "config_hash", None)) else None
    for run_directory in run_directories:
        for marker in run_directory.rglob("complete.json"):
            if not verify_trajectory_payload(marker.parent):
                continue
            metadata = read_json(marker.parent / "metadata.json")
            if (
                (config_hash is None or metadata.get("config_hash") == config_hash)
                and metadata.get("model_key") == config.model.key
                and metadata.get("model_revision") == config.model.revision
                and metadata.get("model_backend", "transformers_cuda")
                == config.model.backend
                and metadata.get("model_dtype", "bfloat16") == config.model.dtype
                and metadata.get("model_mode") == config.model.mode
                and metadata.get("assigned_reasoning_budget") == config.model.reasoning_budget
                and metadata.get("reasoning_budget_policy") == config.model.reasoning_budget_policy
                and metadata.get("final_answer_reserve") == config.model.final_answer_reserve
                and metadata.get("max_new_tokens") == config.model.max_new_tokens
                and metadata.get("prompt_version") == config.prompt_version
                and metadata.get("sampling") == expected_sampling
                and metadata.get("token_metric_schema_version") == TOKEN_METRIC_SCHEMA_VERSION
                and (
                    dataset_bundle_sha256 is None
                    or metadata.get("dataset_bundle_sha256") == dataset_bundle_sha256
                )
            ):
                key = (
                    str(metadata["dataset"]),
                    str(metadata["problem_id"]),
                    int(metadata["seed"]),
                )
                reusable.setdefault(key, marker.parent)
    return reusable


def _reusable_pairs(
    config,
    run_directories: list[Path],
    dataset_bundle_sha256: str | None = None,
) -> set[tuple[str, str, int]]:
    """Compatibility wrapper returning accepted problem/seed keys."""

    return set(_reusable_trajectories(config, run_directories, dataset_bundle_sha256))


def _select_shard(items: list, *, shard_count: int, shard_index: int) -> list:
    """Select a deterministic, balanced slice without changing item order."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be between 0 and {shard_count - 1}; got {shard_index}")
    return [item for position, item in enumerate(items) if position % shard_count == shard_index]


def _resume_identity(
    config,
    *,
    dataset_bundle_sha256: str,
    problem,
    seed: int,
    run_id: str,
) -> dict[str, object]:
    """Return the immutable fields a local resume must exactly preserve."""

    return {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "phase_id": config.phase_id,
        "config_hash": config.config_hash(),
        "dataset_bundle_sha256": dataset_bundle_sha256,
        "token_metric_schema_version": TOKEN_METRIC_SCHEMA_VERSION,
        "model_key": config.model.key,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "model_backend": config.model.backend,
        "model_dtype": config.model.dtype,
        "model_mode": config.model.mode,
        "assigned_reasoning_budget": config.model.reasoning_budget,
        "reasoning_budget_policy": config.model.reasoning_budget_policy,
        "final_answer_reserve": config.model.final_answer_reserve,
        "max_new_tokens": config.model.max_new_tokens,
        "dataset": problem.dataset,
        "problem_id": problem.problem_id,
        "research_split": problem.research_split,
        "seed": seed,
        "prompt_version": config.prompt_version,
        "sampling": asdict(config.model.sampling),
    }


def _completed_trajectory_is_compatible(
    directory: Path,
    config,
    *,
    dataset_bundle_sha256: str,
    problem,
    seed: int,
) -> bool:
    run_id = deterministic_run_id(
        config.experiment_id,
        config.model.key,
        problem.dataset,
        problem.problem_id,
        seed,
    )
    return trajectory_matches_metadata(
        directory,
        _resume_identity(
            config,
            dataset_bundle_sha256=dataset_bundle_sha256,
            problem=problem,
            seed=seed,
            run_id=run_id,
        ),
    )


def _write_generation_progress(
    output_dir: Path,
    config,
    expected: int,
    counts: Counter[str],
    *,
    status: str,
    current: dict | None = None,
    started_at: str,
    global_expected: int,
    shard_count: int,
    shard_index: int,
) -> None:
    """Persist coarse progress so notebook output is not the only status signal."""

    write_json_atomic(
        output_dir / "generation_progress.json",
        {
            "status": status,
            "experiment_id": config.experiment_id,
            "expected_trajectories": expected,
            "global_expected_trajectories": global_expected,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "completed_trajectories": sum(
                counts[name]
                for name in (
                    "completed",
                    "failed",
                    "skipped_complete",
                    "reused_external",
                    "reused_visible",
                )
            ),
            "counts": dict(counts),
            "current_trajectory": current,
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def main() -> None:
    args = parse_args()
    if args.progress_checkpoint_tokens < 1:
        raise ValueError("progress-checkpoint-tokens must be positive")
    if args.deterministic:
        # cuBLAS only honors this if it is set before the first CUDA handle.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output_dir = ensure_directory(args.output_dir)
    config = load_experiment_config(args.project_root / args.config)
    config = _resolve_revision(config, args.readiness_manifest)
    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "experiment_id": config.experiment_id,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "deterministic_kernels_requested": args.deterministic,
            "token_metric_schema_version": TOKEN_METRIC_SCHEMA_VERSION,
        },
    )
    write_json_atomic(output_dir / "resolved_config.json", config.canonical_dict())
    bundle_manifest = args.datasets_dir / "dataset_manifest.json"
    if not bundle_manifest.exists():
        raise FileNotFoundError(f"Immutable dataset manifest is missing: {bundle_manifest}")
    dataset_bundle_sha256 = sha256_file(bundle_manifest)
    problems = _selected_problems(config, args.datasets_dir)
    if args.maximum_problems is not None:
        if args.maximum_problems < 1:
            raise ValueError("maximum-problems must be positive when supplied")
        problems = problems[: args.maximum_problems]
    level_counts = Counter(problem.level for problem in problems if problem.level is not None)
    write_json_atomic(
        output_dir / "problem_selection_manifest.json",
        {
            "experiment_id": config.experiment_id,
            "config_hash": config.config_hash(),
            "problem_count": len(problems),
            "unique_problem_count": len({problem.problem_id for problem in problems}),
            "dataset_counts": dict(Counter(problem.dataset for problem in problems)),
            "level_counts": {str(level): count for level, count in sorted(level_counts.items())},
            "problem_ids": [problem.problem_id for problem in problems],
            "seeds": list(config.seeds),
            "expected_trajectories": len(problems) * len(config.seeds),
            "research_split_counts": dict(Counter(problem.research_split for problem in problems)),
            "dataset_bundle_manifest": str(bundle_manifest),
            "dataset_bundle_sha256": dataset_bundle_sha256,
            "token_metric_schema_version": TOKEN_METRIC_SCHEMA_VERSION,
        },
    )
    plan = [
        (problem_index, problem, seed_index, seed)
        for problem_index, problem in enumerate(problems, start=1)
        for seed_index, seed in enumerate(config.seeds, start=1)
    ]
    global_expected = len(plan)
    shard_plan = _select_shard(
        plan,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    expected = len(shard_plan)
    trajectories_root = ensure_directory(output_dir / "trajectories")
    errors_root = ensure_directory(output_dir / "errors")
    visible_reusable_pairs = _reusable_pairs(
        config, args.reuse_run_dir, dataset_bundle_sha256
    )
    external_reusable = _reusable_trajectories(
        config,
        args.materialize_reuse_run_dir,
        dataset_bundle_sha256,
    )
    counts: Counter[str] = Counter()
    started_at = datetime.now(UTC).isoformat()
    _write_generation_progress(
        output_dir,
        config,
        expected,
        counts,
        status="loading_model",
        started_at=started_at,
        global_expected=global_expected,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    incompatible_local_trajectories = []
    for _, problem, _, seed in shard_plan:
        trajectory_dir = trajectories_root / deterministic_run_id(
            config.experiment_id,
            config.model.key,
            problem.dataset,
            problem.problem_id,
            seed,
        )
        if trajectory_is_complete(trajectory_dir) and not _completed_trajectory_is_compatible(
            trajectory_dir,
            config,
            dataset_bundle_sha256=dataset_bundle_sha256,
            problem=problem,
            seed=seed,
        ):
            incompatible_local_trajectories.append(str(trajectory_dir))
    if incompatible_local_trajectories:
        preview = "\n".join(incompatible_local_trajectories[:5])
        raise RuntimeError(
            "Refusing to mix incompatible or corrupt completed trajectories into a resumed "
            "experiment. Preserve them separately, then explicitly choose a fresh output "
            f"directory. First affected paths:\n{preview}"
        )
    requires_generation = any(
        (problem.dataset, problem.problem_id, seed) not in visible_reusable_pairs
        and (problem.dataset, problem.problem_id, seed) not in external_reusable
        and not _completed_trajectory_is_compatible(
            trajectories_root
            / deterministic_run_id(
                config.experiment_id,
                config.model.key,
                problem.dataset,
                problem.problem_id,
                seed,
            ),
            config,
            dataset_bundle_sha256=dataset_bundle_sha256,
            problem=problem,
            seed=seed,
        )
        for _, problem, _, seed in shard_plan
    )
    bundle = None
    generator = None
    if requires_generation:
        print(
            f"Loading {config.model.key} for shard {args.shard_index + 1}/"
            f"{args.shard_count} ({expected}/{global_expected} trajectories)...",
            flush=True,
        )
        if config.model.backend == "mlx_vlm":
            bundle = load_mlx_model_bundle(config.model)
            generator = MLXInstrumentedGenerator(bundle)
        else:
            bundle = load_model_bundle(config.model)
            generator = InstrumentedGenerator(bundle)
    else:
        print("All shard trajectories are complete or externally reusable; skipping load.")
    progress = tqdm(
        total=expected,
        desc=config.experiment_id,
        unit="trajectory",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=1.0,
    )
    try:
        for plan_position, (problem_index, problem, seed_index, seed) in enumerate(
            shard_plan,
            start=1,
        ):
            run_id = deterministic_run_id(
                config.experiment_id,
                config.model.key,
                problem.dataset,
                problem.problem_id,
                seed,
            )
            trajectory_dir = trajectories_root / run_id
            reuse_key = (problem.dataset, problem.problem_id, seed)
            current = {
                "problem_position": problem_index,
                "problem_count": len(problems),
                "shard_position": plan_position,
                "shard_size": expected,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "seed_position": seed_index,
                "seed_count": len(config.seeds),
                "dataset": problem.dataset,
                "problem_id": problem.problem_id,
                "run_id": run_id,
            }
            if reuse_key in external_reusable:
                source = external_reusable[reuse_key]
                destination = ensure_directory(output_dir / "reused_trajectories") / source.name
                materialize_reused_trajectory(source, destination)
                counts["reused_external"] += 1
                progress.update(1)
                _write_generation_progress(
                    output_dir,
                    config,
                    expected,
                    counts,
                    status="generating",
                    current=None,
                    started_at=started_at,
                    global_expected=global_expected,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
                continue
            if reuse_key in visible_reusable_pairs:
                counts["reused_visible"] += 1
                progress.update(1)
                _write_generation_progress(
                    output_dir,
                    config,
                    expected,
                    counts,
                    status="generating",
                    current=None,
                    started_at=started_at,
                    global_expected=global_expected,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
                continue
            if args.resume and _completed_trajectory_is_compatible(
                trajectory_dir,
                config,
                dataset_bundle_sha256=dataset_bundle_sha256,
                problem=problem,
                seed=seed,
            ):
                counts["skipped_complete"] += 1
                progress.update(1)
                _write_generation_progress(
                    output_dir,
                    config,
                    expected,
                    counts,
                    status="generating",
                    current=None,
                    started_at=started_at,
                    global_expected=global_expected,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
                continue
            set_global_seed(seed, deterministic=args.deterministic)
            started = time.perf_counter()
            _write_generation_progress(
                output_dir,
                config,
                expected,
                counts,
                status="generating",
                current=current,
                started_at=started_at,
                global_expected=global_expected,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
            )
            token_progress = tqdm(
                total=config.model.max_new_tokens,
                desc=(
                    f"{config.experiment_id} "
                    f"problem {problem_index}/{len(problems)}, seed {seed_index}"
                ),
                unit="token",
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
                mininterval=1.0,
            )
            token_count = 0
            next_progress_checkpoint = args.progress_checkpoint_tokens

            def update_token_progress(
                completed_tokens: int,
                progress_bar=token_progress,
            ) -> None:
                nonlocal token_count, next_progress_checkpoint
                if completed_tokens > token_count:
                    progress_bar.update(completed_tokens - token_count)
                    token_count = completed_tokens
                if completed_tokens >= next_progress_checkpoint:
                    checkpoint_current = {
                        **current,
                        "generated_tokens": completed_tokens,
                        "max_new_tokens": config.model.max_new_tokens,
                    }
                    _write_generation_progress(
                        output_dir,
                        config,
                        expected,
                        counts,
                        status="generating",
                        current=checkpoint_current,
                        started_at=started_at,
                        global_expected=global_expected,
                        shard_count=args.shard_count,
                        shard_index=args.shard_index,
                    )
                    next_progress_checkpoint = (
                        completed_tokens // args.progress_checkpoint_tokens + 1
                    ) * args.progress_checkpoint_tokens

            interrupted = False
            try:
                if config.model.backend == "mlx_vlm":
                    import mlx.core as mx

                    mx.reset_peak_memory()
                else:
                    try:
                        import torch

                        torch.cuda.reset_peak_memory_stats()
                    except ImportError:
                        pass
                if generator is None:
                    raise RuntimeError("Model generator was not loaded for a pending trajectory")
                generated = generator.generate(
                    problem.problem,
                    on_token=update_token_progress,
                )
                verification = verify_answer(
                    generated.generated_text,
                    problem.reference_answer,
                    problem.dataset,
                )
                if config.model.backend == "mlx_vlm":
                    import mlx.core as mx

                    peak_allocated_gib = mx.get_peak_memory() / 1024**3
                    peak_reserved_gib = None
                else:
                    try:
                        import torch

                        peak_allocated_gib = torch.cuda.max_memory_allocated(0) / 1024**3
                        peak_reserved_gib = torch.cuda.max_memory_reserved(0) / 1024**3
                    except ImportError:
                        peak_allocated_gib = None
                        peak_reserved_gib = None
                metadata = {
                    "run_id": run_id,
                    "experiment_id": config.experiment_id,
                    "phase_id": config.phase_id,
                    "config_hash": config.config_hash(),
                    "dataset_bundle_sha256": dataset_bundle_sha256,
                    "token_metric_schema_version": TOKEN_METRIC_SCHEMA_VERSION,
                    "model_key": config.model.key,
                    "model_id": config.model.model_id,
                    "model_revision": config.model.revision,
                    "model_backend": config.model.backend,
                    "model_source_id": config.model.source_model_id,
                    "model_dtype": config.model.dtype,
                    "model_architecture": getattr(bundle, "architecture", None),
                    "quantization_bits": getattr(bundle, "quantization_bits", None),
                    "quantization_group_size": getattr(
                        bundle, "quantization_group_size", None
                    ),
                    "model_mode": config.model.mode,
                    "assigned_reasoning_budget": config.model.reasoning_budget,
                    "reasoning_budget_policy": config.model.reasoning_budget_policy,
                    "final_answer_reserve": config.model.final_answer_reserve,
                    "max_new_tokens": config.model.max_new_tokens,
                    "dataset": problem.dataset,
                    "problem_id": problem.problem_id,
                    "research_split": problem.research_split,
                    "problem": problem.problem,
                    "reference_answer": problem.reference_answer,
                    "level": problem.level,
                    "category": problem.category,
                    "seed": seed,
                    "prompt_version": config.prompt_version,
                    "sampling": asdict(config.model.sampling),
                    "generated_text": generated.generated_text,
                    "reasoning_text": generated.reasoning_text,
                    "final_response_text": generated.final_response_text,
                    "boundary_status": generated.boundary_status,
                    "finish_reason": generated.finish_reason,
                    "generated_tokens": len(generated.generated_token_ids),
                    "signal_tokens": len(generated.signals),
                    "inserted_boundary_tokens": generated.inserted_boundary_token_count,
                    "reasoning_boundary_forced": generated.reasoning_boundary_forced,
                    "reasoning_stage_tokens": generated.reasoning_stage_token_count,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_gib": peak_allocated_gib,
                    "peak_reserved_gib": peak_reserved_gib,
                    "verification": verification.to_dict(),
                    "created_at": datetime.now(UTC).isoformat(),
                }
                write_trajectory(trajectory_dir, metadata, generated)
                counts["completed"] += 1
                counts["correct" if verification.correct else "incorrect"] += 1
            except (KeyboardInterrupt, SystemExit):
                interrupted = True
                raise
            except Exception as exc:
                counts["failed"] += 1
                write_json_atomic(
                    errors_root / f"{run_id}.json",
                    {
                        "run_id": run_id,
                        "problem_id": problem.problem_id,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if args.fail_fast:
                    raise
            finally:
                token_progress.close()
                if not interrupted:
                    progress.update(1)
                    progress.set_postfix(
                        completed=counts["completed"],
                        failed=counts["failed"],
                        skipped=counts["skipped_complete"],
                    )
                checkpoint_current = (
                    {
                        **current,
                        "generated_tokens": token_count,
                        "max_new_tokens": config.model.max_new_tokens,
                    }
                    if interrupted
                    else None
                )
                _write_generation_progress(
                    output_dir,
                    config,
                    expected,
                    counts,
                    status="interrupted" if interrupted else "generating",
                    current=checkpoint_current,
                    started_at=started_at,
                    global_expected=global_expected,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
                completed = sum(
                    counts[name]
                    for name in (
                        "completed",
                        "failed",
                        "skipped_complete",
                        "reused_external",
                        "reused_visible",
                    )
                )
                if not interrupted and (completed % 5 == 0 or counts["failed"]):
                    print(
                        f"Progress: {completed}/{expected} trajectories "
                        f"(completed={counts['completed']}, "
                        f"failed={counts['failed']}, "
                        f"resumed={counts['skipped_complete'] + counts['reused_external'] + counts['reused_visible']})",
                        flush=True,
                    )
    finally:
        progress.close()
        if bundle is not None:
            if config.model.backend == "mlx_vlm":
                unload_mlx_model_bundle(bundle)
            else:
                unload_model_bundle(bundle)
    summary = {
        "experiment_id": config.experiment_id,
        "expected_trajectories": expected,
        "global_expected_trajectories": global_expected,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "counts": dict(counts),
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(output_dir / "generation_summary.json", summary)
    write_json_atomic(
        output_dir / "external_reuse_manifest.json",
        {
            "sources": [str(path) for path in args.materialize_reuse_run_dir],
            "reused_trajectories": counts["reused_external"],
        },
    )
    _write_generation_progress(
        output_dir,
        config,
        expected,
        counts,
        status="complete",
        started_at=started_at,
        global_expected=global_expected,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(summary)


if __name__ == "__main__":
    main()
