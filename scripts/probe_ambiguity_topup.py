#!/usr/bin/env python
"""A5.1 ambiguity-triggered enlargement for probe anchors.

Amendment: docs/protocol_amendments/2026-08-20-phase-04c-a5-1-wave3-gate.md
Rule (frozen, forward-looking): any probe cell whose 4-attempt success count
lies in {1, 2, 3} is enlarged to 8 attempts before labels are derived;
thresholds apply to pooled rates and are unchanged. Extreme cells (0/4, 4/4)
stay at 4. Canonical probe artifacts are read-only; enlargement branches and
pooled summaries land under --output-dir.
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
from reasonbench.evaluation.breakthrough import AnchorProbe, derive_breakthrough_label  # noqa: E402
from reasonbench.runtime import set_global_seed, write_runtime_manifest  # noqa: E402
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic  # noqa: E402
from reasonbench.verification import verify_answer  # noqa: E402

VARIANT = "a51_ambiguity_enlargement"


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
    parser.add_argument("--plan-only", action="store_true",
                        help="Report ambiguous-cell counts and exit without loading the model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _enlarge_anchor(*, generator, trajectory, metadata, config, output_dir, anchor,
                    branch_indices, protocol, probe_manifest_sha256, resume, deterministic) -> int:
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
            "max_total_generated_tokens": int(protocol["max_total_generated_tokens"]),
            "reasoning_continuation_budget": int(protocol["reasoning_continuation_budget"]),
            "final_answer_reserve": int(protocol["final_answer_reserve"]),
            "probe_manifest_sha256": probe_manifest_sha256,
            "sensitivity_variant": VARIANT,
        }
        if resume and gbp._result_is_compatible(branch_dir, identity):
            successes += int(bool(read_json(branch_dir / "result.json")["verification"]["correct"]))
            continue
        set_global_seed(seed, deterministic=deterministic)
        started = time.perf_counter()
        result = generator.continue_from_prefix_with_reasoning_budget(
            metadata["problem"],
            prefix_ids,
            reasoning_continuation_budget=int(protocol["reasoning_continuation_budget"]),
            final_answer_reserve=int(protocol["final_answer_reserve"]),
            max_total_generated_tokens=int(protocol["max_total_generated_tokens"]),
        )
        verification = verify_answer(
            result.generated_text, metadata["reference_answer"], metadata["dataset"]
        )
        successes += int(verification.correct)
        gbp._write_branch(branch_dir, {
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
        })
    return successes


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    manifest = read_json(args.probe_manifest)
    gbp._verify_manifest_digest(manifest)
    probe_manifest_sha256 = sha256_file(args.probe_manifest)
    protocol = manifest["probe_protocol"]
    threshold = float(protocol["success_threshold"])
    base_m = int(protocol["continuations_per_anchor"])
    config = gbp._resolve_revision(
        load_experiment_config(args.project_root / args.config), args.readiness_manifest
    )

    trajectories: list[dict] = []
    ambiguous_cells = 0
    for summary_path in sorted((args.probe_dir / "probes").glob("*/trajectory_probe_summary.json")):
        payload = read_json(summary_path)
        if payload["model_key"] != config.model.key:
            continue
        ambiguous = [p["anchor"] for p in payload["probes"]
                     if 0 < p["successes"] < p["continuations"]]
        ambiguous_cells += len(ambiguous)
        trajectories.append({"payload": payload, "ambiguous": sorted(ambiguous)})
    est_tokens = ambiguous_cells * base_m * (
        int(protocol["reasoning_continuation_budget"]) + int(protocol["final_answer_reserve"])
    )
    print(json.dumps({
        "model_key": config.model.key,
        "trajectories": len(trajectories),
        "trajectories_with_ambiguity": sum(1 for t in trajectories if t["ambiguous"]),
        "ambiguous_cells": ambiguous_cells,
        "new_branches": ambiguous_cells * base_m,
        "worst_case_generated_tokens": est_tokens,
    }, indent=2))
    if args.plan_only:
        return

    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04c_a51_ambiguity_enlargement",
            "probe_manifest_sha256": probe_manifest_sha256,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "ambiguous_cells": ambiguous_cells,
        },
    )
    source_index = gbp._source_index(args.base_run_dir)
    progress = output_dir / "topup_progress.json"
    bundle, generator, unload = gbp._load_generator(config)
    rows: list[dict] = []
    changed = 0
    try:
        for position, item in enumerate(trajectories, start=1):
            payload = item["payload"]
            source = source_index[payload["run_id"]]
            metadata = read_json(source / "metadata.json")
            pooled: list[AnchorProbe] = []
            for probe in sorted(payload["probes"], key=lambda p: p["anchor"]):
                anchor = int(probe["anchor"])
                if anchor in item["ambiguous"]:
                    extra = _enlarge_anchor(
                        generator=generator, trajectory=source, metadata=metadata,
                        config=config, output_dir=output_dir, anchor=anchor,
                        branch_indices=range(base_m, 2 * base_m), protocol=protocol,
                        probe_manifest_sha256=probe_manifest_sha256,
                        resume=args.resume, deterministic=args.deterministic,
                    )
                    pooled.append(AnchorProbe(anchor, probe["successes"] + extra, 2 * base_m))
                else:
                    pooled.append(AnchorProbe(anchor, probe["successes"], probe["continuations"]))
            label = derive_breakthrough_label(pooled, threshold=threshold)
            pooled_payload = {
                **{k: payload[k] for k in ("run_id", "problem_id", "model_key", "dataset",
                                            "research_split", "level", "category", "seed")},
                **label.to_dict(),
                "probes": [p.to_dict() for p in pooled],
                "probe_manifest_sha256": probe_manifest_sha256,
                "protocol_variant": VARIANT,
                "original_event_observed": payload["event_observed"],
                "original_event_time_proxy": payload.get("event_time_proxy"),
            }
            write_json_atomic(
                output_dir / "probes" / payload["run_id"] / "trajectory_probe_summary.json",
                pooled_payload,
            )
            changed += int(
                (label.event_observed, label.event_time_proxy)
                != (payload["event_observed"], payload.get("event_time_proxy"))
            )
            rows.append({k: v for k, v in pooled_payload.items() if k != "probes"})
            write_json_atomic(progress, {"status": "enlarging", "completed_trajectories": position,
                                         "expected_trajectories": len(trajectories)})
    finally:
        unload(bundle)

    pd.DataFrame(rows).to_parquet(output_dir / "pooled_labels.parquet", index=False)
    write_json_atomic(output_dir / "topup_summary.json", {
        "status": "complete",
        "model_key": config.model.key,
        "trajectories": len(trajectories),
        "ambiguous_cells": ambiguous_cells,
        "labels_changed_vs_canonical": changed,
        "probe_manifest_sha256": probe_manifest_sha256,
    })
    write_json_atomic(progress, {"status": "complete", "completed_trajectories": len(trajectories),
                                 "expected_trajectories": len(trajectories)})
    print(f"labels changed vs canonical: {changed}/{len(trajectories)}")


if __name__ == "__main__":
    main()
