#!/usr/bin/env python
"""A5 restart control: per-problem fresh-attempt success curves R(C).

Amendment: docs/protocol_amendments/2026-08-20-phase-04c-a5-restart-controlled-breakthrough.md
For each in-scope problem x model, run m fresh attempts (empty prefix, the
frozen two-stage reasoning/answer machinery) at each reasoning budget C.
Deterministic seeds: sha256("phase04c-restart:{problem_id}:{model_key}:{C}:{branch}").
Canonical probe artifacts are never touched; outputs land under --output-dir.
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

DEFAULT_BUDGETS = (1024, 2048, 4096, 8192)
MAX_TOTAL = 16384  # frozen protocol ceiling; leaves room for forced-close tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True,
                        help="Base generation dir supplying problem metadata")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, default=[],
                        help="Probe manifest(s) whose problem_ids define the scope")
    parser.add_argument("--pilot-from-a2", type=Path,
                        help="budget_sensitivity_summary.json; scope = its tested problems")
    parser.add_argument("--problem-id", action="append", default=[],
                        help="Additional explicit problem ids")
    parser.add_argument("--canonical-probe-dir", action="append", type=Path, default=[],
                        help="Probe dirs used to detect instant solvers (reduced budget grid)")
    parser.add_argument("--budget", action="append", type=int, default=[],
                        help=f"Reasoning budgets (default {list(DEFAULT_BUDGETS)})")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--final-answer-reserve", type=int, default=512)
    parser.add_argument("--plan-only", action="store_true",
                        help="Print the work plan and exit without loading the model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _restart_seed(problem_id: str, model_key: str, budget: int, branch: int) -> int:
    digest = hashlib.sha256(
        f"phase04c-restart:{problem_id}:{model_key}:{budget}:{branch}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def main() -> None:
    args = parse_args()
    budgets = tuple(sorted(set(args.budget))) or DEFAULT_BUDGETS
    if args.branches < 1:
        raise ValueError("branches must be positive")
    output_dir = ensure_directory(args.output_dir)
    config = gbp._resolve_revision(
        load_experiment_config(args.project_root / args.config), args.readiness_manifest
    )
    model_key = config.model.key

    scope: set[str] = set(map(str, args.problem_id))
    manifest_shas = {}
    for manifest_path in args.manifest:
        manifest = read_json(manifest_path)
        gbp._verify_manifest_digest(manifest)
        manifest_shas[str(manifest_path)] = sha256_file(manifest_path)
        scope |= set(map(str, manifest["problem_ids"]))
    if args.pilot_from_a2 is not None:
        a2 = read_json(args.pilot_from_a2)
        scope |= {str(row["problem_id"]) for row in a2.get("rows", [])}
    if not scope:
        raise ValueError("Empty scope: pass --manifest, --pilot-from-a2, or --problem-id")

    # one metadata source per problem for this model (deterministic pick)
    candidates: dict[str, tuple[tuple[int, str], Path]] = {}
    for run_id, source in gbp._source_index(args.base_run_dir).items():
        metadata = read_json(source / "metadata.json")
        if str(metadata.get("model_key")) != model_key:
            continue
        problem_id = str(metadata["problem_id"])
        if problem_id not in scope:
            continue
        order = (int(metadata.get("seed", 0)), run_id)
        if problem_id not in candidates or order < candidates[problem_id][0]:
            candidates[problem_id] = (order, source)
    missing = sorted(scope - set(candidates))
    if missing:
        raise FileNotFoundError(f"No base trajectory for {model_key} on: {missing[:5]}")

    # instant solvers for this model -> reduced budget grid (R(1024) only)
    instant: set[str] = set()
    for probe_dir in args.canonical_probe_dir:
        for summary_path in probe_dir.glob("probes/*/trajectory_probe_summary.json"):
            payload = read_json(summary_path)
            if payload["model_key"] != model_key:
                continue
            if payload["event_observed"] and payload.get("interval_upper") is not None \
                    and payload["interval_upper"] <= 16:
                instant.add(str(payload["problem_id"]))

    plan = [
        (problem_id, budgets if problem_id not in instant else (budgets[0],))
        for problem_id in sorted(candidates)
    ]
    total_attempts = sum(len(b) for _, b in plan) * args.branches
    est_tokens = sum(c + args.final_answer_reserve for _, bs in plan for c in bs) * args.branches
    print(json.dumps({
        "model_key": model_key,
        "problems": len(plan),
        "instant_reduced": len([p for p, _ in plan if p in instant]),
        "budgets": list(budgets),
        "branches": args.branches,
        "total_attempts": total_attempts,
        "worst_case_generated_tokens": est_tokens,
    }, indent=2))
    if args.plan_only:
        return

    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={
            "phase_id": "phase_04c_a5_restart_baseline",
            "config_hash": config.config_hash(),
            "model_revision": config.model.revision,
            "budgets": list(budgets),
            "branches": args.branches,
            "scope_manifests": manifest_shas,
            "pilot_from_a2": str(args.pilot_from_a2) if args.pilot_from_a2 else None,
        },
    )
    progress = output_dir / "restart_progress.json"
    write_json_atomic(progress, {"status": "loading_model", "completed_problems": 0,
                                 "expected_problems": len(plan)})
    bundle, generator, unload = gbp._load_generator(config)
    rows: list[dict] = []
    try:
        for position, (problem_id, problem_budgets) in enumerate(plan, start=1):
            metadata = read_json(candidates[problem_id][1] / "metadata.json")
            for budget in problem_budgets:
                successes = 0
                for branch in range(args.branches):
                    seed = _restart_seed(problem_id, model_key, budget, branch)
                    branch_dir = output_dir / "restarts" / problem_id / f"budget_{budget}" / (
                        f"branch_{branch:02d}"
                    )
                    identity = {
                        "variant": "restart_baseline",
                        "problem_id": problem_id,
                        "model_key": model_key,
                        "probe_config_hash": config.config_hash(),
                        "model_revision": config.model.revision,
                        "reasoning_continuation_budget": budget,
                        "final_answer_reserve": args.final_answer_reserve,
                        "max_total_generated_tokens": MAX_TOTAL,
                        "branch_index": branch,
                        "branch_seed": seed,
                    }
                    if args.resume and gbp._result_is_compatible(branch_dir, identity):
                        payload = read_json(branch_dir / "result.json")
                        successes += int(bool(payload["verification"]["correct"]))
                        continue
                    set_global_seed(seed, deterministic=args.deterministic)
                    started = time.perf_counter()
                    result = generator.continue_from_prefix_with_reasoning_budget(
                        metadata["problem"],
                        [],
                        reasoning_continuation_budget=budget,
                        final_answer_reserve=args.final_answer_reserve,
                        max_total_generated_tokens=MAX_TOTAL,
                    )
                    verification = verify_answer(
                        result.generated_text, metadata["reference_answer"], metadata["dataset"]
                    )
                    successes += int(verification.correct)
                    gbp._write_branch(branch_dir, {
                        **identity,
                        "dataset": metadata["dataset"],
                        "generated_token_count": len(result.generated_token_ids),
                        "reasoning_continuation_token_count": result.reasoning_continuation_token_count,
                        "finish_reason": result.finish_reason,
                        "generated_text": result.generated_text,
                        "verification": verification.to_dict(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "created_at": datetime.now(UTC).isoformat(),
                    })
                rows.append({
                    "problem_id": problem_id,
                    "model_key": model_key,
                    "level": metadata.get("level"),
                    "budget": budget,
                    "successes": successes,
                    "attempts": args.branches,
                    "restart_rate": successes / args.branches,
                })
            write_json_atomic(progress, {"status": "probing", "completed_problems": position,
                                         "expected_problems": len(plan),
                                         "current_problem": problem_id})
    finally:
        unload(bundle)

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_dir / "restart_panel.parquet", index=False)
    write_json_atomic(output_dir / "restart_summary.json", {
        "status": "complete",
        "model_key": model_key,
        "problems": len(plan),
        "budgets": list(budgets),
        "branches": args.branches,
        "panel": rows,
    })
    write_json_atomic(progress, {"status": "complete", "completed_problems": len(plan),
                                 "expected_problems": len(plan)})
    print(f"panel -> {output_dir / 'restart_panel.parquet'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
