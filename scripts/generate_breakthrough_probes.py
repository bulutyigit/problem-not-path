#!/usr/bin/env python
"""Label stable-success breakthroughs with sparse continuation probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reasonbench.config import load_experiment_config
from reasonbench.evaluation.breakthrough import AnchorProbe, derive_breakthrough_label
from reasonbench.generation import InstrumentedGenerator
from reasonbench.generation.modeling import load_model_bundle, unload_model_bundle
from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic
from reasonbench.verification import verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--maximum-trajectories", type=int)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _resolve_revision(config, readiness_manifest: Path):
    readiness = read_json(readiness_manifest)
    record = readiness.get("models", {}).get(config.model.key)
    if not record or record.get("status") != "ready" or not record.get("resolved_revision"):
        raise RuntimeError(
            f"Model {config.model.key} is not ready with a frozen revision in {readiness_manifest}"
        )
    return replace(config, model=replace(config.model, revision=record["resolved_revision"]))


def _branch_seed(run_id: str, anchor: int, branch_index: int) -> int:
    digest = hashlib.sha256(
        f"phase04c:{run_id}:{anchor}:{branch_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _prefix_payload(trajectory: Path, anchor: int) -> tuple[list[int], int]:
    frame = pd.read_parquet(trajectory / "token_metrics.parquet").sort_values("token_index")
    analysis = frame[frame["segment"] == "thinking"]
    if analysis.empty:
        analysis = frame[frame["segment"] == "solution"]
    if len(analysis) < anchor:
        raise ValueError(f"Trajectory has only {len(analysis)} reasoning tokens; anchor={anchor}")
    cutoff = int(analysis.iloc[anchor - 1]["token_index"])
    prefix = frame[frame["token_index"] <= cutoff]
    indices = prefix["token_index"].astype(int).tolist()
    if indices != list(range(cutoff + 1)):
        raise RuntimeError("Stored token metrics do not form a complete generated prefix")
    return prefix["token_id"].astype(int).tolist(), cutoff


def _result_is_compatible(path: Path, expected: dict) -> bool:
    marker = path / "branch_complete.json"
    result_path = path / "result.json"
    if not marker.exists() or not result_path.exists():
        return False
    try:
        completion = read_json(marker)
        result = read_json(result_path)
    except Exception:
        return False
    return (
        completion.get("result_sha256") == sha256_file(result_path)
        and completion.get("result_size_bytes") == result_path.stat().st_size
        and all(result.get(key) == value for key, value in expected.items())
    )


def _write_branch(path: Path, payload: dict) -> None:
    ensure_directory(path)
    result_path = path / "result.json"
    write_json_atomic(result_path, payload)
    write_json_atomic(
        path / "branch_complete.json",
        {
            "schema_version": "phase04c_branch_v1",
            "result_sha256": sha256_file(result_path),
            "result_size_bytes": result_path.stat().st_size,
        },
    )


def _probe_anchor(
    *,
    generator: Any,
    trajectory: Path,
    metadata: dict,
    config,
    output_dir: Path,
    anchor: int,
    continuations: int,
    total_budget: int,
    reasoning_continuation_budget: int,
    final_answer_reserve: int,
    probe_manifest_sha256: str,
    resume: bool,
    deterministic: bool,
) -> AnchorProbe:
    prefix_ids, cutoff = _prefix_payload(trajectory, anchor)
    if len(prefix_ids) >= total_budget:
        raise ValueError(
            f"Exact generated prefix already consumes total budget at reasoning anchor {anchor}"
        )
    prefix_sha = hashlib.sha256(
        json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    successes = 0
    for branch_index in range(continuations):
        seed = _branch_seed(metadata["run_id"], anchor, branch_index)
        branch_dir = output_dir / "probes" / metadata["run_id"] / f"anchor_{anchor}" / (
            f"branch_{branch_index:02d}"
        )
        identity = {
            "source_run_id": metadata["run_id"],
            "source_config_hash": metadata["config_hash"],
            "probe_config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "anchor": anchor,
            "branch_index": branch_index,
            "branch_seed": seed,
            "generated_prefix_sha256": prefix_sha,
            "generated_prefix_cutoff_index": cutoff,
            "max_total_generated_tokens": total_budget,
            "reasoning_continuation_budget": reasoning_continuation_budget,
            "final_answer_reserve": final_answer_reserve,
            "probe_manifest_sha256": probe_manifest_sha256,
        }
        if resume and _result_is_compatible(branch_dir, identity):
            result_payload = read_json(branch_dir / "result.json")
            successes += int(bool(result_payload["verification"]["correct"]))
            continue
        set_global_seed(seed, deterministic=deterministic)
        started = time.perf_counter()
        backend = str(generator.bundle.model_config.backend)
        if backend == "mlx_vlm":
            try:
                import mlx.core as mx

                mx.reset_peak_memory()
            except (ImportError, RuntimeError):
                pass
        else:
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                pass
        result = generator.continue_from_prefix_with_reasoning_budget(
            metadata["problem"],
            prefix_ids,
            reasoning_continuation_budget=reasoning_continuation_budget,
            final_answer_reserve=final_answer_reserve,
            max_total_generated_tokens=total_budget,
        )
        verification = verify_answer(
            result.generated_text,
            metadata["reference_answer"],
            metadata["dataset"],
        )
        if backend == "mlx_vlm":
            try:
                import mlx.core as mx

                peak_allocated_gib = mx.get_peak_memory() / 1024**3
                peak_reserved_gib = None
            except (ImportError, RuntimeError):
                peak_allocated_gib = None
                peak_reserved_gib = None
        else:
            try:
                import torch

                peak_allocated_gib = torch.cuda.max_memory_allocated(0) / 1024**3
                peak_reserved_gib = torch.cuda.max_memory_reserved(0) / 1024**3
            except (ImportError, RuntimeError):
                peak_allocated_gib = None
                peak_reserved_gib = None
        successes += int(verification.correct)
        _write_branch(
            branch_dir,
            {
                **identity,
                "problem_id": metadata["problem_id"],
                "model_key": metadata["model_key"],
                "model_backend": backend,
                "base_seed": metadata["seed"],
                "continuation_token_count": len(result.continuation_token_ids),
                "continuation_token_ids": result.continuation_token_ids,
                "combined_generated_token_count": len(result.generated_token_ids),
                "finish_reason": result.finish_reason,
                "reasoning_continuation_token_count": result.reasoning_continuation_token_count,
                "reasoning_boundary_forced": result.reasoning_boundary_forced,
                "inserted_boundary_token_count": result.inserted_boundary_token_count,
                "generated_text": result.generated_text,
                "verification": verification.to_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    return AnchorProbe(anchor=anchor, successes=successes, continuations=continuations)


def _source_index(base_run_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for marker in base_run_dir.rglob("complete.json"):
        metadata = read_json(marker.parent / "metadata.json")
        run_id = str(metadata["run_id"])
        if run_id in indexed:
            raise ValueError(f"Duplicate source run_id under {base_run_dir}: {run_id}")
        indexed[run_id] = marker.parent
    return indexed


def _verify_manifest_digest(manifest: dict) -> None:
    declared = str(manifest.get("selection_digest", ""))
    payload = {key: value for key, value in manifest.items() if key != "selection_digest"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not declared or declared != actual:
        raise RuntimeError("Probe manifest selection_digest is missing or invalid")


def _load_generator(config):
    if config.model.backend == "mlx_vlm":
        from reasonbench.generation.mlx_engine import MLXInstrumentedGenerator
        from reasonbench.generation.mlx_modeling import (
            load_mlx_model_bundle,
            unload_mlx_model_bundle,
        )

        bundle = load_mlx_model_bundle(config.model)
        return bundle, MLXInstrumentedGenerator(bundle), unload_mlx_model_bundle
    bundle = load_model_bundle(config.model)
    return bundle, InstrumentedGenerator(bundle), unload_model_bundle


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard-count/shard-index")
    output_dir = ensure_directory(args.output_dir)
    manifest = read_json(args.probe_manifest)
    _verify_manifest_digest(manifest)
    probe_manifest_sha256 = sha256_file(args.probe_manifest)
    protocol = manifest["probe_protocol"]
    config = _resolve_revision(
        load_experiment_config(args.project_root / args.config),
        args.readiness_manifest,
    )
    records = [
        record
        for record in manifest["trajectories"]
        if record["model_key"] == config.model.key
    ]
    if args.pilot_only:
        pilot_run_ids = {str(run_id) for run_id in manifest.get("pilot_run_ids", [])}
        if not pilot_run_ids:
            raise ValueError("Probe manifest does not define a level-balanced pilot cohort")
        records = [record for record in records if record["run_id"] in pilot_run_ids]
    records = [
        record for index, record in enumerate(records) if index % args.shard_count == args.shard_index
    ]
    if args.maximum_trajectories is not None:
        records = records[: args.maximum_trajectories]
    source_index = _source_index(args.base_run_dir)
    for record in records:
        source = source_index.get(record["run_id"])
        if source is None:
            raise FileNotFoundError(f"Base trajectory is missing: {record['run_id']}")
        marker = source / "complete.json"
        if not verify_trajectory_payload(source):
            raise RuntimeError(f"Base trajectory payload is corrupt: {record['run_id']}")
        if sha256_file(marker) != record["source_complete_sha256"]:
            raise RuntimeError(f"Base trajectory changed after cohort freeze: {record['run_id']}")
        metadata = read_json(source / "metadata.json")
        if metadata["config_hash"] != record["config_hash"]:
            raise RuntimeError(f"Base config hash mismatch: {record['run_id']}")
        if metadata["model_revision"] != record["model_revision"]:
            raise RuntimeError(f"Base model revision mismatch: {record['run_id']}")
        if metadata["dataset_bundle_sha256"] != record["dataset_bundle_sha256"]:
            raise RuntimeError(f"Base dataset bundle mismatch: {record['run_id']}")
    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04c",
            "probe_manifest_sha256": probe_manifest_sha256,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    )
    write_json_atomic(output_dir / "resolved_config.json", config.canonical_dict())
    progress_path = output_dir / "probe_progress.json"
    write_json_atomic(
        progress_path,
        {"status": "loading_model", "completed_trajectories": 0, "expected_trajectories": len(records)},
    )
    bundle, generator, unload = _load_generator(config)
    labels: list[dict] = []
    try:
        for position, record in enumerate(records, start=1):
            source = source_index[record["run_id"]]
            metadata = read_json(source / "metadata.json")
            probes: dict[int, AnchorProbe] = {}
            for anchor in protocol["anchors"]:
                try:
                    probes[int(anchor)] = _probe_anchor(
                        generator=generator,
                        trajectory=source,
                        metadata=metadata,
                        config=config,
                        output_dir=output_dir,
                        anchor=int(anchor),
                        continuations=int(protocol["continuations_per_anchor"]),
                        total_budget=int(protocol["max_total_generated_tokens"]),
                        reasoning_continuation_budget=int(
                            protocol["reasoning_continuation_budget"]
                        ),
                        final_answer_reserve=int(protocol["final_answer_reserve"]),
                        probe_manifest_sha256=probe_manifest_sha256,
                        resume=args.resume,
                        deterministic=args.deterministic,
                    )
                except ValueError as exc:
                    if "only" in str(exc) or "consumes total budget" in str(exc):
                        break
                    raise
            if not probes:
                raise RuntimeError(f"No valid breakthrough anchor for {record['run_id']}")
            for _ in range(int(protocol.get("refinement_rounds", 0))):
                label = derive_breakthrough_label(
                    probes.values(), threshold=float(protocol["success_threshold"])
                )
                if not label.event_observed or label.interval_upper is None:
                    break
                midpoint = (label.interval_lower + label.interval_upper) // 2
                if midpoint <= 0 or midpoint in probes or label.interval_upper - label.interval_lower <= 16:
                    break
                probes[midpoint] = _probe_anchor(
                    generator=generator,
                    trajectory=source,
                    metadata=metadata,
                    config=config,
                    output_dir=output_dir,
                    anchor=midpoint,
                    continuations=int(protocol["continuations_per_anchor"]),
                    total_budget=int(protocol["max_total_generated_tokens"]),
                    reasoning_continuation_budget=int(
                        protocol["reasoning_continuation_budget"]
                    ),
                    final_answer_reserve=int(protocol["final_answer_reserve"]),
                    probe_manifest_sha256=probe_manifest_sha256,
                    resume=args.resume,
                    deterministic=args.deterministic,
                )
            label = derive_breakthrough_label(
                probes.values(), threshold=float(protocol["success_threshold"])
            )
            label_row = {
                "run_id": metadata["run_id"],
                "problem_id": metadata["problem_id"],
                "model_key": metadata["model_key"],
                "dataset": metadata["dataset"],
                "research_split": metadata["research_split"],
                "level": metadata.get("level"),
                "category": metadata.get("category"),
                "seed": metadata["seed"],
                **label.to_dict(),
                "probes": [probe.to_dict() for probe in sorted(probes.values(), key=lambda item: item.anchor)],
                "probe_manifest_sha256": probe_manifest_sha256,
            }
            labels.append(label_row)
            trajectory_summary = output_dir / "probes" / metadata["run_id"] / "trajectory_probe_summary.json"
            write_json_atomic(trajectory_summary, label_row)
            write_json_atomic(
                progress_path,
                {
                    "status": "probing",
                    "completed_trajectories": position,
                    "expected_trajectories": len(records),
                    "current_run_id": metadata["run_id"],
                },
            )
    finally:
        unload(bundle)
    pd.DataFrame(labels).drop(columns=["probes"]).to_parquet(
        output_dir / "breakthrough_labels.parquet", index=False
    )
    write_json_atomic(
        output_dir / "probe_summary.json",
        {
            "status": "complete",
            "model_key": config.model.key,
            "completed_trajectories": len(labels),
            "events_observed": sum(bool(row["event_observed"]) for row in labels),
            "right_censored": sum(not bool(row["event_observed"]) for row in labels),
            "probe_manifest_sha256": probe_manifest_sha256,
        },
    )
    write_json_atomic(
        progress_path,
        {"status": "complete", "completed_trajectories": len(labels), "expected_trajectories": len(records)},
    )


if __name__ == "__main__":
    main()
