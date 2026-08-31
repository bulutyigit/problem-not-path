#!/usr/bin/env python
"""A1 sensitivity labels: replicate the final anchor of high-rate censored trajectories.

Amendment: docs/protocol_amendments/2026-08-19-phase-04c-probe-sensitivity-and-supplement.md
Canonical probe directories are read-only inputs; every new branch lands under
--output-dir. The primary τ/next-anchor labels are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_breakthrough_probes as gbp  # noqa: E402

from reasonbench.config import load_experiment_config  # noqa: E402
from reasonbench.runtime import set_global_seed, write_runtime_manifest  # noqa: E402
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic  # noqa: E402
from reasonbench.verification import verify_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True,
                        help="Canonical probe output directory for this model (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-continuations", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _extra_branches(
    *,
    generator,
    trajectory: Path,
    metadata: dict,
    config,
    output_dir: Path,
    anchor: int,
    branch_indices: range,
    total_budget: int,
    reasoning_continuation_budget: int,
    final_answer_reserve: int,
    probe_manifest_sha256: str,
    resume: bool,
    deterministic: bool,
) -> int:
    prefix_ids, cutoff = gbp._prefix_payload(trajectory, anchor)
    prefix_sha = hashlib.sha256(
        json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    successes = 0
    for branch_index in branch_indices:
        seed = gbp._branch_seed(metadata["run_id"], anchor, branch_index)
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
            "sensitivity_variant": "terminal_replication",
        }
        if resume and gbp._result_is_compatible(branch_dir, identity):
            successes += int(bool(read_json(branch_dir / "result.json")["verification"]["correct"]))
            continue
        set_global_seed(seed, deterministic=deterministic)
        started = time.perf_counter()
        result = generator.continue_from_prefix_with_reasoning_budget(
            metadata["problem"],
            prefix_ids,
            reasoning_continuation_budget=reasoning_continuation_budget,
            final_answer_reserve=final_answer_reserve,
            max_total_generated_tokens=total_budget,
        )
        verification = verify_answer(
            result.generated_text, metadata["reference_answer"], metadata["dataset"]
        )
        successes += int(verification.correct)
        gbp._write_branch(
            branch_dir,
            {
                **identity,
                "problem_id": metadata["problem_id"],
                "model_key": metadata["model_key"],
                "base_seed": metadata["seed"],
                "continuation_token_count": len(result.continuation_token_ids),
                "finish_reason": result.finish_reason,
                "generated_text": result.generated_text,
                "verification": verification.to_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    return successes


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    manifest = read_json(args.probe_manifest)
    gbp._verify_manifest_digest(manifest)
    probe_manifest_sha256 = sha256_file(args.probe_manifest)
    protocol = manifest["probe_protocol"]
    threshold = float(protocol["success_threshold"])
    base_continuations = int(protocol["continuations_per_anchor"])
    config = gbp._resolve_revision(
        load_experiment_config(args.project_root / args.config), args.readiness_manifest
    )

    # Candidates: right-censored canonical trajectories whose final anchor is at threshold.
    candidates: list[dict] = []
    rows: list[dict] = []
    for summary_path in sorted((args.probe_dir / "probes").glob("*/trajectory_probe_summary.json")):
        payload = read_json(summary_path)
        if payload["model_key"] != config.model.key:
            continue
        probes = sorted(payload["probes"], key=lambda p: p["anchor"])
        final = probes[-1]
        row = {
            "run_id": payload["run_id"],
            "problem_id": payload["problem_id"],
            "model_key": payload["model_key"],
            "level": payload.get("level"),
            "event_observed": payload["event_observed"],
            "censoring_time": payload["censoring_time"],
            "final_anchor": final["anchor"],
            "final_rate": final["successes"] / final["continuations"],
            "previous_anchor": probes[-2]["anchor"] if len(probes) > 1 else 0,
            "final_successes_primary": final["successes"],
        }
        rows.append(row)
        if not payload["event_observed"] and row["final_rate"] >= threshold:
            candidates.append(row)

    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04c_sensitivity_terminal",
            "probe_manifest_sha256": probe_manifest_sha256,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "candidates": [row["run_id"] for row in candidates],
        },
    )
    if not candidates:
        write_json_atomic(
            output_dir / "terminal_stability_summary.json",
            {"status": "complete", "model_key": config.model.key, "candidates": 0,
             "events_added": 0, "probe_manifest_sha256": probe_manifest_sha256},
        )
        print(f"No terminal-replication candidates for {config.model.key}")
        return

    source_index = gbp._source_index(args.base_run_dir)
    bundle, generator, unload = gbp._load_generator(config)
    variants: list[dict] = []
    try:
        for row in candidates:
            source = source_index[row["run_id"]]
            metadata = read_json(source / "metadata.json")
            extra_successes = _extra_branches(
                generator=generator,
                trajectory=source,
                metadata=metadata,
                config=config,
                output_dir=output_dir,
                anchor=int(row["final_anchor"]),
                branch_indices=range(base_continuations, base_continuations + args.extra_continuations),
                total_budget=int(protocol["max_total_generated_tokens"]),
                reasoning_continuation_budget=int(protocol["reasoning_continuation_budget"]),
                final_answer_reserve=int(protocol["final_answer_reserve"]),
                probe_manifest_sha256=probe_manifest_sha256,
                resume=args.resume,
                deterministic=args.deterministic,
            )
            pooled_successes = int(row["final_successes_primary"]) + extra_successes
            pooled_continuations = base_continuations + args.extra_continuations
            pooled_rate = pooled_successes / pooled_continuations
            event = pooled_rate >= threshold
            variants.append({
                **{k: row[k] for k in ("run_id", "problem_id", "model_key", "level",
                                        "final_anchor", "previous_anchor", "final_rate")},
                "extra_successes": extra_successes,
                "pooled_successes": pooled_successes,
                "pooled_continuations": pooled_continuations,
                "pooled_rate": pooled_rate,
                "variant_event_observed": event,
                "variant_interval_lower": int(row["previous_anchor"]) if event else int(row["final_anchor"]),
                "variant_interval_upper": int(row["final_anchor"]) if event else None,
                "variant_event_time_proxy": int(row["final_anchor"]) if event else None,
                "stability_rule": "terminal_replication",
            })
    finally:
        unload(bundle)

    frame = pd.DataFrame(rows).merge(
        pd.DataFrame(variants).drop(columns=["problem_id", "model_key", "level",
                                              "final_anchor", "previous_anchor", "final_rate"]),
        on="run_id", how="left",
    )
    frame.to_parquet(output_dir / "terminal_stability_labels.parquet", index=False)
    write_json_atomic(
        output_dir / "terminal_stability_summary.json",
        {
            "status": "complete",
            "model_key": config.model.key,
            "threshold": threshold,
            "candidates": len(candidates),
            "events_added": int(sum(v["variant_event_observed"] for v in variants)),
            "variants": variants,
            "probe_manifest_sha256": probe_manifest_sha256,
        },
    )
    print(json.dumps({"candidates": len(candidates),
                      "events_added": int(sum(v["variant_event_observed"] for v in variants))}))


if __name__ == "__main__":
    main()
