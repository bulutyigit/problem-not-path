#!/usr/bin/env python
"""Show durable per-model and in-flight progress for the local MLX Phase 4B run."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from reasonbench.storage import read_json

MODELS = (
    ("gemma4", "Gemma 4 E4B"),
    ("qwen35", "Qwen 3.5 4B"),
    ("ministral3", "Ministral 3 3B"),
)


def _render(phase_root: Path) -> str:
    lines = [f"Phase 4B MLX status — {datetime.now().astimezone().isoformat(timespec='seconds')}"]
    total_complete = 0
    total_expected = 0
    for model_key, label in MODELS:
        progress_path = (
            phase_root
            / "generation"
            / f"{model_key}_mlx_4bit"
            / "generation_progress.json"
        )
        if not progress_path.exists():
            lines.append(f"{label:18} not started")
            total_expected += 100
            continue
        try:
            progress = read_json(progress_path)
        except Exception as exc:
            lines.append(f"{label:18} checkpoint unreadable: {exc}")
            continue
        complete = int(progress.get("completed_trajectories", 0))
        expected = int(
            progress.get(
                "global_expected_trajectories",
                progress.get("expected_trajectories", 100),
            )
        )
        total_complete += complete
        total_expected += expected
        detail = ""
        current = progress.get("current_trajectory")
        if isinstance(current, dict):
            problem = current.get("problem_position", "?")
            problem_count = current.get("problem_count", expected)
            generated = current.get("generated_tokens")
            limit = current.get("max_new_tokens")
            detail = f" | problem {problem}/{problem_count}"
            if generated is not None:
                detail += f" | token {generated}/{limit}"
        lines.append(
            f"{label:18} {complete:3d}/{expected:<3d} "
            f"{progress.get('status', 'unknown')}{detail}"
        )
    lines.append(f"TOTAL              {total_complete:3d}/{total_expected:<3d}")
    lines.append(f"Checkpoint root: {phase_root}")
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=project_root / "artifacts" / "mac_mlx" / "phase_04b",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        print(_render(args.phase_root.resolve()), flush=True)
        if not args.watch:
            break
        print("-" * 72, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
