#!/usr/bin/env python
"""A5 full-cohort labeling: join restart panels onto frozen probe results.

Amendments: A5 (2026-08-20) and A5.1. Rules applied exactly as frozen:
  R̂: log-C interpolation on the measured grid, clamped at the ends
  advantage(t) = p̂(t; 1024) − R̂(t + 1024)
  T_V(δ): earliest anchor with p̂ ≥ τ and advantage ≥ δ, stable at the next
  regimes: instant (interval_upper ≤ 16) reported separately with its single
  R(1024) cell; else budget_limited if R̂(4096) ≥ τ; else prefix_limited if a
  δ=0.5 crossing exists; else no_crossing_at_primary_delta when the prefix
  curve crossed τ (T_F exists) and unsolved otherwise. Terminal A1 events
  (pooled final-anchor, no stability anchor) are annotated, never promoted.
  A5.1 sensitivity: a conservative advantage against the upper envelope of
  the bracketing grid values is reported alongside.

Cohorts: dev + supplement (pooled restart panel) and wave3 (dedicated panel;
probe rates come from the A5.1 pooled top-up summaries, which cover all 49
trajectories with enlarged cells already folded in).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reasonbench.storage import ensure_directory, read_json, write_json_atomic

TAU, B = 0.75, 1024
DELTAS = (0.25, 0.5, 0.75)
MODELS = {
    "gemma4": "gemma4_e4b_mlx_4bit",
    "ministral3": "ministral3_3b_mlx_4bit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase04c-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def r_hat(panel: dict[int, float], total: float, conservative: bool = False) -> float:
    grid = sorted(panel)
    if total <= grid[0]:
        return panel[grid[0]]
    if total >= grid[-1]:
        return panel[grid[-1]]
    for lo, hi in zip(grid, grid[1:]):
        if lo <= total <= hi:
            if conservative:
                return max(panel[lo], panel[hi])
            frac = (np.log2(total) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
            return panel[lo] + frac * (panel[hi] - panel[lo])
    raise ValueError(total)


def first_stable_crossing(adv, delta):
    for k in range(len(adv) - 1):
        t, p, a = adv[k]
        _, p2, a2 = adv[k + 1]
        if p >= TAU and a >= delta and p2 >= TAU and a2 >= delta:
            return t
    return None


def main() -> None:
    args = parse_args()
    root = args.phase04c_root
    out = ensure_directory(args.output_dir)
    rows = []
    for short, model_key in MODELS.items():
        panels: dict[str, dict[int, float]] = {}
        for panel_file in (
            root / "probes/sensitivity/restart_baseline" / short / "restart_panel_pooled.parquet",
            root / "probes/sensitivity/restart_baseline_wave3" / short / "restart_panel.parquet",
        ):
            panel_frame = pd.read_parquet(panel_file)
            fresh = {
                pid: dict(zip(g.budget.astype(int), g.restart_rate.astype(float)))
                for pid, g in panel_frame.groupby("problem_id")
            }
            overlap = set(fresh) & set(panels)
            if overlap:
                raise RuntimeError(f"Restart panels overlap on {sorted(overlap)[:3]}")
            panels.update(fresh)
        terminal: dict[str, dict] = {}
        for variant_dir in (
            "terminal_stability", "terminal_stability_supplement", "terminal_stability_wave3"
        ):
            path = root / "probes/sensitivity" / variant_dir / short / "terminal_stability_summary.json"
            if path.exists():
                for v in read_json(path).get("variants", []):
                    if v["variant_event_observed"]:
                        terminal[v["problem_id"]] = v
        for cohort, probe_dir in (
            ("dev", "probes/models"),
            ("supplement", "probes/supplement"),
            ("wave3", "probes/wave3_topup"),
        ):
            for summary_path in sorted((root / probe_dir / short / "probes").glob("*/trajectory_probe_summary.json")):
                s = read_json(summary_path)
                pid = s["problem_id"]
                panel = panels[pid]
                probes = sorted((p["anchor"], p["successes"] / p["continuations"]) for p in s["probes"])
                adv = [(t, p, p - r_hat(panel, t + B)) for t, p in probes]
                adv_cons = [(t, p, p - r_hat(panel, t + B, conservative=True)) for t, p in probes]
                t_v = {d: first_stable_crossing(adv, d) for d in DELTAS}
                t_v_cons = {d: first_stable_crossing(adv_cons, d) for d in DELTAS}
                instant = bool(s["event_observed"]) and s["interval_upper"] is not None and s["interval_upper"] <= 16
                has_4096 = 4096 in panel
                term = terminal.get(pid)
                if instant:
                    if panel.get(1024, 0.0) >= TAU:
                        regime = "instant_scratch_solvable"
                    elif t_v[0.5] is not None:
                        regime = "instant_prefix_advantaged"
                    else:
                        regime = "instant_ambiguous"
                elif has_4096 and r_hat(panel, 4096) >= TAU:
                    regime = "budget_limited"
                elif t_v[0.5] is not None:
                    regime = "prefix_limited"
                elif s["event_observed"]:
                    regime = "no_crossing_at_primary_delta"
                elif term is not None:
                    regime = "terminal_A1"
                elif has_4096 and r_hat(panel, 4096) >= TAU:
                    regime = "unsolved_scratch_solvable"
                else:
                    regime = "unsolved_hard"
                rows.append({
                    "model": short, "model_key": model_key, "cohort": cohort,
                    "problem_id": pid, "level": s.get("level"),
                    "research_split": s.get("research_split"),
                    "t_f_1024": s.get("stable_anchor"),
                    "event_observed": bool(s["event_observed"]),
                    "instant": instant,
                    "regime": regime,
                    "R_1024": panel.get(1024), "R_2048": panel.get(2048),
                    "R_4096": panel.get(4096), "R_8192": panel.get(8192),
                    "max_advantage": round(max(a for *_, a in adv), 3),
                    "t_v_d025": t_v[0.25], "t_v_d050": t_v[0.5], "t_v_d075": t_v[0.75],
                    "t_v_cons_d050": t_v_cons[0.5],
                    "conservative_agrees_d050": (t_v[0.5] is None) == (t_v_cons[0.5] is None),
                    "terminal_a1_anchor": int(term["final_anchor"]) if term else None,
                    "terminal_a1_advantage": (
                        round(term["pooled_rate"] - r_hat(panel, term["final_anchor"] + B), 3)
                        if term else None
                    ),
                })
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "a5_full_cohort_labels.parquet", index=False)

    summary = {"tau": TAU, "budget": B, "primary_delta": 0.5}
    summary["regime_counts"] = {
        m: g.regime.value_counts().to_dict() for m, g in frame.groupby("model")
    }
    summary["t_v_events_primary"] = frame[frame.t_v_d050.notna() & ~frame.instant][
        ["model", "problem_id", "cohort", "level", "t_f_1024", "t_v_d050", "max_advantage", "regime"]
    ].to_dict(orient="records")
    summary["delta_sensitivity_counts"] = {
        f"delta_{d}": int((frame[~frame.instant][f"t_v_d{str(d).replace('0.', '0')[:4].replace('.', '')}"]
                           if False else frame[~frame.instant][
                               {0.25: "t_v_d025", 0.5: "t_v_d050", 0.75: "t_v_d075"}[d]
                           ]).notna().sum())
        for d in DELTAS
    }
    summary["conservative_disagreements_d050"] = frame[~frame.conservative_agrees_d050][
        ["model", "problem_id", "t_v_d050", "t_v_cons_d050"]
    ].to_dict(orient="records")
    pivot = frame.pivot(index="problem_id", columns="model", values="regime")
    collapse = {
        "instant_scratch_solvable": "instant", "instant_prefix_advantaged": "instant",
        "instant_ambiguous": "instant", "budget_limited": "budget_limited",
        "prefix_limited": "prefix_limited", "no_crossing_at_primary_delta": "other",
        "terminal_A1": "other", "unsolved_scratch_solvable": "unsolved", "unsolved_hard": "unsolved",
    }
    both = pivot.dropna()
    agree = (both.gemma4.map(collapse) == both.ministral3.map(collapse)).sum()
    summary["cross_model_top_class_agreement"] = f"{int(agree)}/{len(both)}"
    write_json_atomic(out / "a5_full_cohort_summary.json", summary)
    pd.set_option("display.width", 240)
    print(frame.groupby(["model", "regime"]).size().unstack(fill_value=0).T.to_string())
    print("\nT_V (primary delta) events:")
    for r in summary["t_v_events_primary"]:
        print("  ", r)
    print("\ndelta sensitivity:", summary["delta_sensitivity_counts"])
    print("conservative disagreements:", len(summary["conservative_disagreements_d050"]))
    print("cross-model top-class agreement:", summary["cross_model_top_class_agreement"])


if __name__ == "__main__":
    main()
