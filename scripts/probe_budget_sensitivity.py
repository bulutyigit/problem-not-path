#!/usr/bin/env python
"""A2 falsification probe: re-label interior-event anchors with a 4,096-token budget.

Amendment: docs/protocol_amendments/2026-08-19-phase-04c-probe-sensitivity-and-supplement.md
Re-probes every canonical anchor <= stability_anchor of interior-event
trajectories with a larger reasoning continuation budget and identical branch
seeds, so each large-budget branch extends its canonical counterpart (paired).
Readout: does the first τ-crossing shift earlier by ~the budget delta
(mechanical length confound) or stay put (genuine prefix breakthrough)?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_breakthrough_probes as gbp  # noqa: E402

from reasonbench.config import load_experiment_config  # noqa: E402
from reasonbench.evaluation.breakthrough import AnchorProbe, derive_breakthrough_label  # noqa: E402
from reasonbench.runtime import write_runtime_manifest  # noqa: E402
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic  # noqa: E402


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
    parser.add_argument("--reasoning-budget", type=int, default=4096)
    parser.add_argument("--extra-anchor", action="append", type=int, default=[],
                        help="Additional anchors to probe under the large budget (e.g. 16 32) "
                             "to resolve floor-censored crossings")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _first_crossing(probes: list[AnchorProbe], threshold: float) -> int | None:
    for probe in sorted(probes, key=lambda p: p.anchor):
        if probe.success_rate >= threshold:
            return probe.anchor
    return None


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    manifest = read_json(args.probe_manifest)
    gbp._verify_manifest_digest(manifest)
    probe_manifest_sha256 = sha256_file(args.probe_manifest)
    protocol = manifest["probe_protocol"]
    threshold = float(protocol["success_threshold"])
    canonical_budget = int(protocol["reasoning_continuation_budget"])
    if args.reasoning_budget <= canonical_budget:
        raise ValueError("reasoning-budget must exceed the canonical continuation budget")
    config = gbp._resolve_revision(
        load_experiment_config(args.project_root / args.config), args.readiness_manifest
    )

    # Interior-event trajectories: event observed with interval_upper > 16.
    targets: list[dict] = []
    for summary_path in sorted((args.probe_dir / "probes").glob("*/trajectory_probe_summary.json")):
        payload = read_json(summary_path)
        if payload["model_key"] != config.model.key:
            continue
        if not payload["event_observed"] or payload["interval_upper"] is None:
            continue
        if int(payload["interval_upper"]) <= 16:
            continue
        anchors = sorted(
            {p["anchor"] for p in payload["probes"]
             if p["anchor"] <= int(payload["stability_anchor"])}
            | {int(a) for a in args.extra_anchor}
        )
        targets.append({
            "payload": payload,
            "anchors": anchors,
            "canonical_probes": {int(p["anchor"]): p for p in payload["probes"]},
        })

    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04c_sensitivity_budget",
            "probe_manifest_sha256": probe_manifest_sha256,
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "reasoning_budget": args.reasoning_budget,
            "targets": [t["payload"]["run_id"] for t in targets],
        },
    )
    if not targets:
        write_json_atomic(
            output_dir / "budget_sensitivity_summary.json",
            {"status": "complete", "model_key": config.model.key, "targets": 0,
             "probe_manifest_sha256": probe_manifest_sha256},
        )
        print(f"No interior-event trajectories for {config.model.key}")
        return

    source_index = gbp._source_index(args.base_run_dir)
    bundle, generator, unload = gbp._load_generator(config)
    rows: list[dict] = []
    progress = output_dir / "probe_progress.json"
    try:
        for position, target in enumerate(targets, start=1):
            payload = target["payload"]
            source = source_index[payload["run_id"]]
            metadata = read_json(source / "metadata.json")
            variant_probes: list[AnchorProbe] = []
            for anchor in target["anchors"]:
                variant_probes.append(gbp._probe_anchor(
                    generator=generator,
                    trajectory=source,
                    metadata=metadata,
                    config=config,
                    output_dir=output_dir,
                    anchor=int(anchor),
                    continuations=int(protocol["continuations_per_anchor"]),
                    total_budget=int(protocol["max_total_generated_tokens"]),
                    reasoning_continuation_budget=args.reasoning_budget,
                    final_answer_reserve=int(protocol["final_answer_reserve"]),
                    probe_manifest_sha256=probe_manifest_sha256,
                    resume=args.resume,
                    deterministic=args.deterministic,
                ))
            canonical_probes = [
                AnchorProbe(anchor=a, successes=p["successes"], continuations=p["continuations"])
                for a, p in target["canonical_probes"].items() if a in set(target["anchors"])
            ]
            variant_label = derive_breakthrough_label(variant_probes, threshold=threshold)
            canonical_first = _first_crossing(canonical_probes, threshold)
            variant_first = _first_crossing(variant_probes, threshold)
            rows.append({
                "run_id": payload["run_id"],
                "problem_id": payload["problem_id"],
                "model_key": payload["model_key"],
                "level": payload.get("level"),
                "canonical_budget": canonical_budget,
                "variant_budget": args.reasoning_budget,
                "probed_anchors": list(target["anchors"]),
                "canonical_stable_anchor": int(payload["stable_anchor"]),
                "canonical_first_crossing": canonical_first,
                "variant_first_crossing": variant_first,
                "variant_event_observed": variant_label.event_observed,
                "variant_stable_anchor": variant_label.stable_anchor,
                "first_crossing_shift": (
                    None if variant_first is None or canonical_first is None
                    else int(canonical_first) - int(variant_first)
                ),
                "canonical_rates": {p.anchor: p.success_rate for p in sorted(canonical_probes, key=lambda x: x.anchor)},
                "variant_rates": {p.anchor: p.success_rate for p in sorted(variant_probes, key=lambda x: x.anchor)},
            })
            write_json_atomic(progress, {
                "status": "probing", "completed_trajectories": position,
                "expected_trajectories": len(targets), "current_run_id": payload["run_id"],
            })
    finally:
        unload(bundle)

    serializable = [
        {**row,
         "probed_anchors": json.dumps(row["probed_anchors"]),
         "canonical_rates": json.dumps(row["canonical_rates"]),
         "variant_rates": json.dumps(row["variant_rates"])}
        for row in rows
    ]
    pd.DataFrame(serializable).to_parquet(output_dir / "budget_sensitivity_labels.parquet", index=False)
    shifts = [row["first_crossing_shift"] for row in rows if row["first_crossing_shift"] is not None]
    budget_delta = args.reasoning_budget - canonical_budget
    summary = {
        "status": "complete",
        "model_key": config.model.key,
        "targets": len(targets),
        "budget_delta": budget_delta,
        "first_crossing_shifts": shifts,
        "shifted_ge_two_thirds_delta": int(sum(s >= (2 * budget_delta) // 3 for s in shifts)),
        "probe_manifest_sha256": probe_manifest_sha256,
        "rows": serializable,
    }
    write_json_atomic(output_dir / "budget_sensitivity_summary.json", summary)
    write_json_atomic(progress, {"status": "complete", "completed_trajectories": len(targets),
                                 "expected_trajectories": len(targets)})
    print(json.dumps({k: summary[k] for k in ("targets", "budget_delta", "first_crossing_shifts",
                                              "shifted_ge_two_thirds_delta")}))


if __name__ == "__main__":
    main()
