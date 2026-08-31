#!/usr/bin/env python
"""Generate nested short/medium/long Phase 4C-U continuations from exact prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.evaluation.compute_extension import (
    UNCERTAINTY_SCORE_VERSION,
    validate_compute_extension_protocol,
)
from reasonbench.generation.storage import verify_trajectory_payload
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic
from reasonbench.verification import verify_answer
from scripts.generate_breakthrough_probes import (
    _branch_seed,
    _load_generator,
    _prefix_payload,
    _resolve_revision,
    _source_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-probe-dir", action="append", type=Path, default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--maximum-trajectories", type=int)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _verify_manifest(manifest: dict) -> None:
    declared = str(manifest.get("extension_digest", ""))
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "extension_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not declared or hashlib.sha256(canonical).hexdigest() != declared:
        raise RuntimeError("Extension manifest digest is missing or invalid")
    if manifest.get("uncertainty_score", {}).get("version") != UNCERTAINTY_SCORE_VERSION:
        raise RuntimeError("Extension manifest uses an unsupported uncertainty-score version")
    validate_compute_extension_protocol(manifest.get("protocol", {}))


def _write_result(path: Path, payload: dict) -> None:
    ensure_directory(path)
    result_path = path / "result.json"
    write_json_atomic(result_path, payload)
    write_json_atomic(
        path / "branch_complete.json",
        {
            "schema_version": "phase04c_extension_branch_v1",
            "result_sha256": sha256_file(result_path),
            "result_size_bytes": result_path.stat().st_size,
        },
    )


def _compatible(path: Path, identity: dict) -> bool:
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
        and all(result.get(key) == value for key, value in identity.items())
    )


def _index_short_results(roots: list[Path]) -> dict[tuple[str, int, int], Path]:
    indexed: dict[tuple[str, int, int], Path] = {}
    for root in roots:
        for path in sorted(root.rglob("result.json")):
            marker = path.parent / "branch_complete.json"
            if not marker.exists():
                continue
            result = read_json(path)
            key = (
                str(result.get("source_run_id", "")),
                int(result.get("anchor", -1)),
                int(result.get("branch_index", -1)),
            )
            if key in indexed:
                raise ValueError(f"Duplicate reusable short-branch result: {key}")
            completion = read_json(marker)
            if (
                completion.get("result_sha256") != sha256_file(path)
                or completion.get("result_size_bytes") != path.stat().st_size
            ):
                raise RuntimeError(f"Corrupt reusable short-branch result: {path}")
            indexed[key] = path
    return indexed


def _memory_reset(backend: str) -> None:
    try:
        if backend == "mlx_vlm":
            import mlx.core as mx

            mx.reset_peak_memory()
        else:
            import torch

            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def _memory_peak(backend: str) -> tuple[float | None, float | None]:
    try:
        if backend == "mlx_vlm":
            import mlx.core as mx

            return mx.get_peak_memory() / 1024**3, None
        import torch

        return (
            torch.cuda.max_memory_allocated(0) / 1024**3,
            torch.cuda.max_memory_reserved(0) / 1024**3,
        )
    except (ImportError, RuntimeError):
        return None, None


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard-count/shard-index")
    manifest = read_json(args.extension_manifest)
    _verify_manifest(manifest)
    manifest_sha = sha256_file(args.extension_manifest)
    protocol = manifest["protocol"]
    records = [record for record in manifest["records"] if record.get("eligible")]
    if args.pilot_only:
        pilot_ids = {str(value) for value in manifest.get("pilot_eligible_run_ids", [])}
        records = [record for record in records if record["run_id"] in pilot_ids]
    config = _resolve_revision(
        load_experiment_config(args.project_root / args.config),
        args.readiness_manifest,
    )
    records = [record for record in records if record["model_key"] == config.model.key]
    records = [
        record
        for index, record in enumerate(records)
        if index % args.shard_count == args.shard_index
    ]
    if args.maximum_trajectories is not None:
        records = records[: args.maximum_trajectories]
    if not records:
        raise ValueError(f"No eligible extension records for model {config.model.key}")

    output = ensure_directory(args.output_dir)
    protocol_schema_version = protocol.get("protocol_schema_version", "phase04c_u_v1")
    source_index = _source_index(args.base_run_dir)
    short_results = _index_short_results(args.short_probe_dir)
    for record in records:
        source = source_index.get(record["run_id"])
        if source is None:
            raise FileNotFoundError(f"Base trajectory is missing: {record['run_id']}")
        if not verify_trajectory_payload(source):
            raise RuntimeError(f"Base trajectory payload is corrupt: {record['run_id']}")
        if sha256_file(source / "complete.json") != record["source_complete_sha256"]:
            raise RuntimeError(f"Base trajectory changed: {record['run_id']}")

    write_runtime_manifest(
        output / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": (
                "phase_05_breakthrough"
                if protocol_schema_version.startswith("phase05")
                else "phase_04c_u"
            ),
            "protocol_schema_version": protocol_schema_version,
            "extension_manifest_sha256": manifest_sha,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "model_backend": config.model.backend,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    )
    write_json_atomic(output / "resolved_config.json", config.canonical_dict())
    progress_path = output / "extension_progress.json"
    expected_branches = (
        len(records) * len(protocol["arms"]) * int(protocol["continuations_per_arm"])
    )
    write_json_atomic(
        progress_path,
        {
            "status": "loading_model",
            "completed_branches": 0,
            "expected_branches": expected_branches,
        },
    )

    bundle, generator, unload = _load_generator(config)
    completed = 0
    reused = 0
    correct_by_arm = {arm: 0 for arm in protocol["arms"]}
    try:
        for record in records:
            source = source_index[record["run_id"]]
            metadata = read_json(source / "metadata.json")
            prefix_ids, cutoff = _prefix_payload(source, int(protocol["primary_anchor"]))
            prefix_sha = hashlib.sha256(
                json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if prefix_sha != record["generated_prefix_sha256"]:
                raise RuntimeError(f"Exact prefix hash changed: {record['run_id']}")
            if len(prefix_ids) != int(record["generated_prefix_token_count"]):
                raise RuntimeError(f"Exact prefix token count changed: {record['run_id']}")
            for branch_index in range(int(protocol["continuations_per_arm"])):
                seed = _branch_seed(record["run_id"], int(protocol["primary_anchor"]), branch_index)
                for arm, arm_protocol in protocol["arms"].items():
                    reasoning_budget = int(arm_protocol["reasoning_continuation_budget"])
                    target_total_reasoning = int(arm_protocol["target_total_reasoning_tokens"])
                    branch_dir = (
                        output
                        / "extensions"
                        / record["run_id"]
                        / f"anchor_{int(protocol['primary_anchor'])}"
                        / arm
                        / f"branch_{branch_index:02d}"
                    )
                    identity = {
                        "protocol_schema_version": protocol_schema_version,
                        "source_run_id": record["run_id"],
                        "source_config_hash": record["source_config_hash"],
                        "extension_config_hash": config.config_hash(),
                        "model_revision": config.model.revision,
                        "model_backend": config.model.backend,
                        "anchor": int(protocol["primary_anchor"]),
                        "budget_arm": arm,
                        "branch_index": branch_index,
                        "branch_seed": seed,
                        "generated_prefix_sha256": prefix_sha,
                        "generated_prefix_cutoff_index": cutoff,
                        "generated_prefix_token_count": len(prefix_ids),
                        "reasoning_continuation_budget": reasoning_budget,
                        "target_total_reasoning_tokens": target_total_reasoning,
                        "budget_semantics": protocol["budget_semantics"],
                        "final_answer_reserve": int(protocol["final_answer_reserve"]),
                        "max_total_generated_tokens": int(protocol["max_total_generated_tokens"]),
                        "extension_manifest_sha256": manifest_sha,
                        "source_probe_manifest_sha256": manifest["source_probe_manifest_sha256"],
                        "uncertainty_score_version": manifest["uncertainty_score"]["version"],
                        "uncertainty_score": record["uncertainty_score"],
                        "uncertainty_stratum": record["uncertainty_stratum"],
                    }
                    if args.resume and _compatible(branch_dir, identity):
                        result_payload = read_json(branch_dir / "result.json")
                        correct_by_arm[arm] += int(result_payload["verification"]["correct"])
                        completed += 1
                        reused += int(bool(result_payload.get("reused_short_branch")))
                        continue

                    reusable = short_results.get(
                        (record["run_id"], int(protocol["primary_anchor"]), branch_index)
                    )
                    source_payload = read_json(reusable) if arm == "short" and reusable else None
                    if source_payload is not None:
                        expected_short = {
                            "source_run_id": record["run_id"],
                            "anchor": int(protocol["primary_anchor"]),
                            "branch_index": branch_index,
                            "branch_seed": seed,
                            "generated_prefix_sha256": prefix_sha,
                            "reasoning_continuation_budget": reasoning_budget,
                            "final_answer_reserve": int(protocol["final_answer_reserve"]),
                            "probe_manifest_sha256": manifest["source_probe_manifest_sha256"],
                        }
                        if not all(source_payload.get(k) == v for k, v in expected_short.items()):
                            # Phase 4C-P currently uses a 1,024-token *additional*
                            # probe budget, whereas Phase 4C-U freezes total
                            # reasoning targets. An incompatible result is never
                            # mixed into this experiment; it is simply regenerated.
                            source_payload = None
                    if source_payload is not None:
                        payload = {
                            **source_payload,
                            **identity,
                            "reused_short_branch": True,
                            "reused_from_result_sha256": sha256_file(reusable),
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        reused += 1
                    else:
                        set_global_seed(seed, deterministic=args.deterministic)
                        _memory_reset(config.model.backend)
                        started = time.perf_counter()
                        result = generator.continue_from_prefix_with_reasoning_budget(
                            metadata["problem"],
                            prefix_ids,
                            reasoning_continuation_budget=reasoning_budget,
                            final_answer_reserve=int(protocol["final_answer_reserve"]),
                            max_total_generated_tokens=int(protocol["max_total_generated_tokens"]),
                        )
                        verification = verify_answer(
                            result.generated_text,
                            metadata["reference_answer"],
                            metadata["dataset"],
                        )
                        peak_allocated, peak_reserved = _memory_peak(config.model.backend)
                        payload = {
                            **identity,
                            "problem_id": metadata["problem_id"],
                            "model_key": metadata["model_key"],
                            "level": metadata.get("level"),
                            "category": metadata.get("category"),
                            "research_split": metadata.get("research_split"),
                            "base_seed": metadata["seed"],
                            "continuation_token_count": len(result.continuation_token_ids),
                            "continuation_token_ids": result.continuation_token_ids,
                            "combined_generated_token_count": len(result.generated_token_ids),
                            "finish_reason": result.finish_reason,
                            "reasoning_continuation_token_count": (
                                result.reasoning_continuation_token_count
                            ),
                            "reasoning_total_token_count": (
                                int(protocol["primary_anchor"])
                                + result.reasoning_continuation_token_count
                            ),
                            "reasoning_boundary_forced": result.reasoning_boundary_forced,
                            "inserted_boundary_token_count": (result.inserted_boundary_token_count),
                            "generated_text": result.generated_text,
                            "verification": verification.to_dict(),
                            "elapsed_seconds": time.perf_counter() - started,
                            "peak_allocated_gib": peak_allocated,
                            "peak_reserved_gib": peak_reserved,
                            "reused_short_branch": False,
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                    _write_result(branch_dir, payload)
                    correct_by_arm[arm] += int(payload["verification"]["correct"])
                    completed += 1
                    write_json_atomic(
                        progress_path,
                        {
                            "status": "generating",
                            "completed_branches": completed,
                            "expected_branches": expected_branches,
                            "current_run_id": record["run_id"],
                            "current_budget_arm": arm,
                            "reused_short_branches": reused,
                        },
                    )
    finally:
        unload(bundle)

    summary = {
        "status": "complete",
        "model_key": config.model.key,
        "completed_trajectories": len(records),
        "completed_branches": completed,
        "expected_branches": expected_branches,
        "reused_short_branches": reused,
        "correct_by_arm": correct_by_arm,
        "extension_manifest_sha256": manifest_sha,
    }
    write_json_atomic(output / "extension_summary.json", summary)
    write_json_atomic(progress_path, summary)


if __name__ == "__main__":
    main()
