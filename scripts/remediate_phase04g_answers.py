#!/usr/bin/env python
"""Extend censored Phase 4G final-answer channels without regenerating reasoning.

The original Phase 4G branches reserved only 512 tokens after the reasoning
boundary.  That censored verbose final responses, especially Gemma's, and
turned missing ``\\boxed{}`` answers into false mathematical failures.  This
stage replays each exact stored branch and extends only its final-answer tail.
Original artifacts are immutable and are linked by SHA-256 in every repaired
result.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.generation.mlx_engine import _sample_from_generated_prefix
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic
from reasonbench.verification import verify_answer
from scripts.generate_breakthrough_probes import (
    _load_generator,
    _prefix_payload,
    _resolve_revision,
    _source_index,
)

REMEDIATION_VERSION = "phase04g_answer_remediation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--source-extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-answer-token-limit", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _repair_seed(source_run_id: str, arm: str, branch_index: int) -> int:
    digest = hashlib.sha256(
        f"{REMEDIATION_VERSION}:{source_run_id}:{arm}:{branch_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _write_result(path: Path, payload: dict) -> None:
    ensure_directory(path)
    result_path = path / "result.json"
    write_json_atomic(result_path, payload)
    write_json_atomic(
        path / "branch_complete.json",
        {
            "schema_version": REMEDIATION_VERSION,
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
        completion.get("schema_version") == REMEDIATION_VERSION
        and completion.get("result_sha256") == sha256_file(result_path)
        and completion.get("result_size_bytes") == result_path.stat().st_size
        and all(result.get(key) == value for key, value in identity.items())
    )


def _source_results(root: Path, model_key: str) -> list[tuple[Path, dict]]:
    results: list[tuple[Path, dict]] = []
    for path in sorted(root.rglob("result.json")):
        marker = path.parent / "branch_complete.json"
        if not marker.exists():
            continue
        completion = read_json(marker)
        payload = read_json(path)
        if payload.get("model_key") != model_key:
            continue
        if (
            completion.get("result_sha256") != sha256_file(path)
            or completion.get("result_size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"Corrupt source branch: {path}")
        results.append((path, payload))
    if not results:
        raise ValueError(f"No source branches found for {model_key} under {root}")
    return results


def main() -> None:
    args = parse_args()
    if args.final_answer_token_limit <= 512:
        raise ValueError("The remediation answer limit must exceed the original 512-token reserve")
    config = _resolve_revision(
        load_experiment_config(args.project_root / args.config),
        args.readiness_manifest,
    )
    if config.model.backend != "mlx_vlm":
        raise ValueError("Phase 4G local remediation currently requires the MLX backend")
    sources = _source_results(args.source_extension_dir, config.model.key)
    trajectories = _source_index(args.base_run_dir)
    output = ensure_directory(args.output_dir)
    progress_path = output / "answer_remediation_progress.json"
    write_runtime_manifest(
        output / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04g_answer_remediation",
            "remediation_version": REMEDIATION_VERSION,
            "model_key": config.model.key,
            "model_revision": config.model.revision,
            "final_answer_token_limit": args.final_answer_token_limit,
        },
    )
    write_json_atomic(
        progress_path,
        {"status": "loading_model", "completed_branches": 0, "expected_branches": len(sources)},
    )

    bundle, _generator, unload = _load_generator(config)
    completed = 0
    extended = 0
    correct_before = 0
    correct_after = 0
    try:
        for source_path, source in sources:
            run_id = str(source["source_run_id"])
            trajectory = trajectories.get(run_id)
            if trajectory is None:
                raise FileNotFoundError(f"Missing exact source trajectory: {run_id}")
            metadata = read_json(trajectory / "metadata.json")
            prefix_ids, cutoff = _prefix_payload(trajectory, int(source["anchor"]))
            if cutoff != int(source["generated_prefix_cutoff_index"]):
                raise RuntimeError(f"Prefix cutoff changed for {run_id}")
            continuation_ids = [int(value) for value in source["continuation_token_ids"]]
            combined_ids = [*prefix_ids, *continuation_ids]
            if len(combined_ids) != int(source["combined_generated_token_count"]):
                raise RuntimeError(f"Stored combined-token count is inconsistent: {source_path}")
            reasoning_count = int(source["reasoning_continuation_token_count"])
            inserted_count = int(source["inserted_boundary_token_count"])
            answer_count_before = len(continuation_ids) - reasoning_count - inserted_count
            if answer_count_before < 0 or answer_count_before > args.final_answer_token_limit:
                raise RuntimeError(f"Invalid source answer-token count: {source_path}")
            source_sha = sha256_file(source_path)
            arm = str(source["budget_arm"])
            branch_index = int(source["branch_index"])
            repair_seed = _repair_seed(run_id, arm, branch_index)
            relative = source_path.parent.relative_to(args.source_extension_dir)
            branch_dir = output / relative
            identity = {
                "answer_remediation_version": REMEDIATION_VERSION,
                "source_result_sha256": source_sha,
                "source_run_id": run_id,
                "budget_arm": arm,
                "branch_index": branch_index,
                "final_answer_token_limit": args.final_answer_token_limit,
                "remediation_seed": repair_seed,
            }
            if args.resume and _compatible(branch_dir, identity):
                repaired = read_json(branch_dir / "result.json")
                completed += 1
                extended += int(bool(repaired["remediation_applied"]))
                correct_before += int(bool(repaired["verification_before_remediation"]["correct"]))
                correct_after += int(bool(repaired["verification"]["correct"]))
                continue

            additional_ids: list[int] = []
            continuation_finish = "eos" if source["finish_reason"] == "eos" else "max_new_tokens"
            started = time.perf_counter()
            if source["finish_reason"] != "eos":
                remaining = args.final_answer_token_limit - answer_count_before
                if remaining <= 0:
                    raise RuntimeError(f"Censored branch has no remediation capacity: {source_path}")
                set_global_seed(repair_seed, deterministic=args.deterministic)
                additional_ids, continuation_finish = _sample_from_generated_prefix(
                    bundle,
                    metadata["problem"],
                    combined_ids,
                    max_new_tokens=remaining,
                )
                extended += 1
            repaired_continuation = [*continuation_ids, *additional_ids]
            repaired_combined = [*prefix_ids, *repaired_continuation]
            generated_text = bundle.tokenizer.decode(
                repaired_combined,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            verification = verify_answer(
                generated_text,
                metadata["reference_answer"],
                metadata["dataset"],
            )
            answer_count_after = answer_count_before + len(additional_ids)
            finish_reason = "eos" if continuation_finish == "eos" else "answer_limit"
            repaired = {
                **source,
                **identity,
                "source_result": str(source_path),
                "source_finish_reason": source["finish_reason"],
                "source_final_answer_reserve": int(source["final_answer_reserve"]),
                "verification_before_remediation": source["verification"],
                "original_answer_token_count": answer_count_before,
                "additional_answer_token_count": len(additional_ids),
                "final_answer_token_count": answer_count_after,
                "remediation_applied": bool(additional_ids),
                "continuation_token_count": len(repaired_continuation),
                "continuation_token_ids": repaired_continuation,
                "combined_generated_token_count": len(repaired_combined),
                "generated_text": generated_text,
                "finish_reason": finish_reason,
                "verification": verification.to_dict(),
                "remediation_elapsed_seconds": time.perf_counter() - started,
                "remediated_at": datetime.now(UTC).isoformat(),
            }
            _write_result(branch_dir, repaired)
            completed += 1
            correct_before += int(bool(source["verification"]["correct"]))
            correct_after += int(bool(verification.correct))
            write_json_atomic(
                progress_path,
                {
                    "status": "remediating",
                    "completed_branches": completed,
                    "expected_branches": len(sources),
                    "extended_branches": extended,
                    "correct_before": correct_before,
                    "correct_after": correct_after,
                    "current_run_id": run_id,
                    "current_budget_arm": arm,
                },
            )
    finally:
        unload(bundle)

    summary = {
        "status": "complete",
        "remediation_version": REMEDIATION_VERSION,
        "model_key": config.model.key,
        "completed_branches": completed,
        "expected_branches": len(sources),
        "extended_branches": extended,
        "final_answer_token_limit": args.final_answer_token_limit,
        "correct_before": correct_before,
        "correct_after": correct_after,
    }
    write_json_atomic(output / "answer_remediation_summary.json", summary)
    write_json_atomic(progress_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
