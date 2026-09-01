#!/usr/bin/env python
"""Render the six paper figures that carry model-name labels.

Regenerates from frozen artifacts (labels parquet, pooled top-up
summaries, A1 layer, A2 budget-sensitivity summaries). Written when the
Gemma checkpoint was found mislabeled in earlier figure runs; model
display names live in MODELS below and nowhere else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MODELS = {"gemma4": "Gemma-4 E4B", "ministral3": "Ministral-3 3B"}
MODEL_KEYS = {"gemma4": "gemma4_e4b_mlx_4bit", "ministral3": "ministral3_3b_mlx_4bit"}
SURFACE, INK, SEC, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                        "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, GREEN, AMBER, PINK = "#3f7ad9", "#e0592a", "#1baf7a", "#eda100", "#e87ba4"
TAU = 0.75

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": SEC,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
})


def load_topup(root: Path, short: str) -> list[dict]:
    payloads = []
    for path in sorted((root / f"probes/wave3_topup/{short}/probes").glob(
            "*/trajectory_probe_summary.json")):
        payloads.append(json.loads(path.read_text()))
    return payloads


def fig_regime_map(labels: pd.DataFrame, out: Path) -> None:
    coll = {"instant_scratch_solvable": "instant", "instant_prefix_advantaged": "instant",
            "instant_ambiguous": "instant", "budget_limited": "budget-limited",
            "prefix_limited": "prefix-limited", "no_crossing_at_primary_delta": "no-crossing",
            "terminal_A1": "terminal-A1", "unsolved_scratch_solvable": "unsolved",
            "unsolved_hard": "unsolved"}
    frame = labels.assign(cls=labels.regime.map(coll))
    classes = ["instant", "budget-limited", "prefix-limited", "terminal-A1",
               "no-crossing", "unsolved"]
    colors = {"instant": "#9ec5f4", "budget-limited": AMBER, "prefix-limited": GREEN,
              "terminal-A1": PINK, "no-crossing": "#8a7ad9", "unsolved": "#d4d2ca"}
    piv = frame.pivot(index="problem_id", columns="model", values="cls")
    piv["cohort"] = frame.groupby("problem_id").cohort.first()
    ordered = []
    for cohort in ("dev", "supplement", "wave3"):
        sub = piv[piv.cohort.eq(cohort)].copy()
        sub["k1"] = sub["ministral3"].map(classes.index)
        sub["k2"] = sub["gemma4"].map(classes.index)
        ordered.append(sub.sort_values(["k1", "k2"]))
    piv = pd.concat(ordered)
    rows = [("gemma4", MODELS["gemma4"]), ("ministral3", MODELS["ministral3"])]
    matrix = np.array([[classes.index(piv.iloc[i][m]) for i in range(len(piv))]
                       for m, _ in rows])
    fig = plt.figure(figsize=(13.2, 4.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap([colors[c] for c in classes]),
              vmin=0, vmax=len(classes) - 1, interpolation="nearest")
    ax.set_yticks([0, 1]); ax.set_yticklabels([label for _, label in rows], fontsize=9)
    ax.set_xticks([]); ax.grid(False)
    bounds = np.cumsum([piv.cohort.eq(c).sum() for c in ("dev", "supplement", "wave3")])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color=SURFACE, lw=3)
    for name, lo, hi in zip(("dev (20)", "supplement (20)", "wave3 (49)"),
                            np.r_[0, bounds[:-1]], bounds):
        ax.text((lo + hi - 1) / 2, -0.62, name, ha="center", fontsize=8.6, color=SEC)
    ax.set_xlim(-0.5, len(piv) - 0.5)
    ax.set_title("Every problem × model cell in the project, classified under the "
                 "frozen A5 rules", loc="left", fontsize=9.4, color=SEC, pad=18)
    ax2 = fig.add_subplot(gs[1])
    counts = frame.groupby(["model", "cls"]).size().unstack(fill_value=0).reindex(
        columns=classes, fill_value=0)
    lefts = {m: 0 for m, _ in rows}
    for cls in classes:
        for row, (m, _) in enumerate(rows):
            value = counts.loc[m, cls]
            if value:
                ax2.barh(row, value, left=lefts[m], color=colors[cls], height=0.62)
                if value >= 3:
                    dark = cls in ("instant", "budget-limited", "unsolved")
                    ax2.text(lefts[m] + value / 2, row, str(value), ha="center",
                             va="center", fontsize=8.6, color=INK if dark else "white")
                lefts[m] += value
    ax2.set_yticks([0, 1]); ax2.set_yticklabels([label for _, label in rows], fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlim(0, 89)
    ax2.set_xlabel("Cells (of 89 problems per model)")
    ax2.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=colors[c], label=c) for c in classes],
               frameon=False, fontsize=8.6, ncols=6, loc="outside lower center")
    fig.suptitle("Final regime map — 89 problems × 2 models, 178 cells, three cohorts",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out / "a5_project_regime_map.png", dpi=180)
    plt.close(fig)


def fig_dose_response(labels: pd.DataFrame, out: Path) -> None:
    budgets = [1024, 2048, 4096, 8192]
    frame = labels[(labels.cohort == "wave3") & ~labels.instant]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
    rng = np.random.default_rng(7)
    for (short, label), color in zip(MODELS.items(), (BLUE, ORANGE)):
        group = frame[frame.model.eq(short)]
        rates = group[["R_1024", "R_2048", "R_4096", "R_8192"]].to_numpy(float)
        rates = rates[~np.isnan(rates).any(axis=1)]
        for row in rates:
            axes[0].plot(budgets, row + rng.uniform(-0.012, 0.012, 1),
                         color=color, alpha=0.16, lw=1.1)
        share = [(rates[:, i] >= TAU).mean() for i in range(4)]
        axes[1].plot(budgets, share, "-o", color=color, lw=2.4, ms=6,
                     label=f"{label} (n={len(rates)})")
        for x, s in zip(budgets, share):
            axes[1].annotate(f"{s:.0%}", (x, s), textcoords="offset points",
                             xytext=(0, 9 if short == "ministral3" else -15),
                             ha="center", fontsize=8.4, color=color)
    axes[0].set_title("Individual restart curves R(C), one line per problem",
                      loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("From-scratch success rate (4 attempts)")
    axes[1].set_title("Share of cells solved from scratch (R ≥ 0.75)",
                      loc="left", fontsize=10, color=INK)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks(budgets)
        ax.set_xticklabels(["1,024", "2,048", "4,096", "8,192"])
        ax.set_xlabel("Restart reasoning budget C (tokens)")
        ax.set_ylim(-0.04, 1.06)
    fig.suptitle("Wave 3 restart dose–response — non-instant cells: Ministral climbs "
                 "with budget, Gemma stays flat", fontsize=11.5, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "wave3_restart_dose_response.png", dpi=180)
    plt.close(fig)


def fig_topup_resolution(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.84))
    for ax, (short, label) in zip(axes, [("ministral3", MODELS["ministral3"]),
                                         ("gemma4", MODELS["gemma4"])]):
        pooled = []
        for payload in load_topup(root, short):
            pooled += [p["successes"] for p in payload["probes"]
                       if p["continuations"] == 8]
        counts = pd.Series(pooled).value_counts().reindex(range(9), fill_value=0)
        colors = [MUTED if k <= 2 else (GREEN if k >= 6 else AMBER) for k in range(9)]
        ax.bar(range(9), counts.values, color=colors, width=0.7)
        for k, v in counts.items():
            if v:
                ax.text(k, v + 0.4, str(v), ha="center", fontsize=8.5, color=INK)
        low = int((counts[counts.index <= 2]).sum())
        mid = int((counts[(counts.index >= 3) & (counts.index <= 5)]).sum())
        high = int((counts[counts.index >= 6]).sum())
        ax.set_title(f"{label} — {low + mid + high} cells", fontsize=10.5,
                     color=INK, loc="left")
        ax.text(0.98, 0.94, f"low {low} · mid {mid} · high {high}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.6, color=SEC)
        ax.set_xticks(range(9))
        ax.set_xlabel("Pooled successes (of 8) on cells that were 1–3 of 4")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cells")
    fig.legend(handles=[Patch(color=MUTED, label="resolved unsolvable (≤2/8)"),
                        Patch(color=AMBER, label="intermediate after 8 attempts (3–5/8)"),
                        Patch(color=GREEN, label="resolved solvable (≥6/8)")],
               frameon=False, fontsize=8.5, ncols=3, loc="upper right",
               bbox_to_anchor=(0.99, 0.92))
    fig.suptitle('A5.1 ambiguity enlargement — what 4-attempt "maybe" cells became '
                 "at 8 attempts", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.savefig(out / "wave3_topup_resolution.png", dpi=180)
    plt.close(fig)


def fig_label_hardening(out: Path) -> None:
    order = ["instant", "interior", "A1-event", "censored"]
    colors = {"instant": "#9ec5f4", "interior": GREEN, "A1-event": PINK,
              "censored": "#c9c7c1"}
    data = {
        "ministral3": {"canonical\n(4 attempts)": [7, 8, 13, 21],
                       "final\n(pooled + A1)": [7, 4, 12, 26]},
        "gemma4": {"canonical\n(4 attempts)": [10, 5, 0, 34],
                   "final\n(pooled + A1)": [9, 5, 0, 35]},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    for ax, short in zip(axes, ("ministral3", "gemma4")):
        for x, (column, values) in enumerate(data[short].items()):
            bottom = 0
            for cls, value in zip(order, values):
                ax.bar(x, value, bottom=bottom, color=colors[cls], width=0.55)
                if value >= 3:
                    ax.text(x, bottom + value / 2, str(value), ha="center",
                            va="center", fontsize=9, color=INK)
                bottom += value
        ax.set_xticks(range(len(data[short])))
        ax.set_xticklabels(list(data[short].keys()), fontsize=9)
        ax.set_title(MODELS[short], loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Trajectories (of 49)")
    fig.legend(handles=[Patch(color=colors[c], label=c) for c in order],
               frameon=False, fontsize=8.8, ncols=4, loc="outside upper right")
    fig.suptitle("Wave 3 labels before and after noise hardening",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.savefig(out / "wave3_label_hardening.png", dpi=180)
    plt.close(fig)


def fig_swimmer(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 12.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    for ax, (short, color) in zip(axes, [("gemma4", BLUE), ("ministral3", ORANGE)]):
        rows = []
        for payload in load_topup(root, short):
            final = max(payload["probes"], key=lambda p: p["anchor"])
            final_rate = final["successes"] / final["continuations"]
            rows.append({
                "level": payload.get("level"),
                "event": bool(payload["event_observed"]),
                "lower": payload.get("interval_lower"),
                "upper": payload.get("interval_upper"),
                "censor": payload.get("censoring_time"),
                "pink": (not payload["event_observed"]) and final_rate >= TAU,
            })
        frame = pd.DataFrame(rows)
        events = frame[frame.event].sort_values("upper")
        censored = frame[~frame.event].sort_values(["censor", "pink"])
        ordered = pd.concat([events, censored]).reset_index(drop=True)
        for i, row in ordered.iterrows():
            y = len(ordered) - 1 - i
            if row.event:
                lo = row.lower if pd.notna(row.lower) and row.lower > 0 else 12
                ax.plot([12, lo], [y, y], color=GRID, lw=2)
                ax.plot([lo, row.upper], [y, y], color=color, lw=3.2,
                        solid_capstyle="round")
                ax.scatter(row.upper, y, color=color, s=28, zorder=3)
            else:
                line = PINK if row.pink else "#d8d6cf"
                ax.plot([12, row.censor], [y, y], color=line, lw=2)
                ax.scatter(row.censor, y, marker=">", facecolor="none",
                           edgecolor=PINK if row.pink else MUTED, s=34, zorder=3)
            ax.text(10.5, y, f"L{int(row.level)}", ha="right", va="center",
                    fontsize=6.4, color=MUTED)
        instant = int((frame.event & (frame.upper <= 16)).sum())
        delayed = int((frame.event & (frame.upper > 16)).sum())
        pink = int(frame.pink.sum())
        ax.set_title(f"{MODELS[short]} (4-bit) — instant {instant} · delayed "
                     f"{delayed} · censored {int((~frame.event).sum())} "
                     f"(at threshold: {pink})", loc="left", fontsize=9.6, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 4096, 8192])
        ax.set_xticklabels(["16", "64", "256", "1024", "4096", "8192"])
        ax.set_xlim(10, 11000)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Prefix tokens (log scale)")
    fig.suptitle("Wave 3 probes — 49 problems × 2 models: breakthrough intervals and "
                 "censoring\npink rows: censored at threshold (terminal-replication "
                 "candidates)", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.savefig(out / "swimmer_wave3.png", dpi=180)
    plt.close(fig)


def fig_budget_falsification(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    for ax, (short, color) in zip(axes, [("gemma4", BLUE), ("ministral3", ORANGE)]):
        summary = json.loads((root / "probes/sensitivity/budget_4096" / short /
                              "budget_sensitivity_summary.json").read_text())
        rows = sorted(summary["rows"], key=lambda r: r["canonical_first_crossing"] or 0)
        for i, row in enumerate(rows):
            canonical = row["canonical_first_crossing"]
            variant = row["variant_first_crossing"]
            if canonical is None:
                continue
            variant_x = variant if variant is not None else canonical
            ax.plot([canonical, variant_x], [i, i], color=BASE, lw=1.6, zorder=1)
            ax.scatter(canonical, i, facecolor="none", edgecolor=MUTED, s=150,
                       linewidths=2.0, zorder=3,
                       label="B = 1,024" if i == 0 else None)
            ax.scatter(variant_x, i, color=color, s=56, zorder=4,
                       label="B = 4,096" if i == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['problem_id'].split('_')[1]} (L{r['level']})"
                            for r in rows], fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 4096])
        ax.set_xticklabels(["16", "64", "256", "1,024", "4,096"])
        ax.set_xlabel("Earliest crossing anchor (prefix tokens)")
        ax.set_title(MODELS[short], loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.suptitle("A2 budget falsification — the same trajectories probed at two"
                 " continuation budgets:\nMinistral's crossings collapse toward"
                 " 16-token prefixes, Gemma's stay pinned (rings = B = 1,024)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "budget_falsification.png", dpi=180)
    plt.close(fig)




def fig_matched_compute(root: Path, labels: pd.DataFrame, out: Path) -> None:
    from matplotlib.lines import Line2D
    term = json.loads((root / "probes/sensitivity/terminal_stability_wave3/"
                       "ministral3/terminal_stability_summary.json").read_text())
    pooled_rate = {v["problem_id"]: v["pooled_rate"] for v in term["variants"]
                   if v["variant_event_observed"]}
    lab = labels[(labels.cohort == "wave3") & (labels.model == "ministral3")
                 & labels.terminal_a1_anchor.notna()].copy()
    lab["prefix_rate"] = lab.problem_id.map(pooled_rate)
    lab["matched_R"] = lab.prefix_rate - lab.terminal_a1_advantage
    lab = lab.sort_values(["terminal_a1_anchor", "terminal_a1_advantage"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.8, 6.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.80))
    for i, row in lab.iterrows():
        ax.plot([row.matched_R, row.prefix_rate], [i, i], color=BASE, lw=1.6, zorder=1)
        ax.scatter(row.matched_R, i, color=MUTED, s=46, zorder=3)
        ax.scatter(row.prefix_rate, i, color=ORANGE, s=52, zorder=4)
        ax.scatter(row.R_8192, i, facecolor="none", edgecolor=INK, s=64,
                   marker="^", zorder=2, linewidths=1.2)
    ax.axvline(TAU, color=MUTED, lw=1, ls="--")
    ax.text(TAU + 0.005, len(lab) - 0.4, "τ = 0.75", fontsize=8.4, color=MUTED)
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(
        [("%s  (t = %s)" % (p.split("_")[1], format(int(a), ","))) +
         ("*" if a == 8192 else "")
         for p, a in zip(lab.problem_id, lab.terminal_a1_anchor)], fontsize=8.4)
    ax.set_xlabel("Success rate"); ax.set_xlim(-0.04, 1.06)
    ax.grid(axis="y", visible=False)
    ax.set_title("In-grid matched comparisons: prefix wins 9/9 (median +0.31); "
                 "starred boundary proxies: 2 wins, 2 ties.\nLargest measured "
                 "restart reaches τ in 11/13, the prefix rate in 9/13 "
                 "(* R(9,216) unmeasured; R(8,192) proxy, bias direction unknown)",
                 loc="left", fontsize=8.8, color=SEC, pad=10)
    handles = [Line2D([], [], marker="o", ls="", color=ORANGE,
                      label="continue own prefix at final anchor t (pooled, 8 branches)"),
               Line2D([], [], marker="o", ls="", color=MUTED,
                      label="restart at matched budget R\u0302(t + 1,024)"),
               Line2D([], [], marker="^", ls="", markerfacecolor="none",
                      markeredgecolor=INK, label="restart at 8,192 (largest budget)")]
    fig.legend(handles=handles, frameon=False, fontsize=8.6, loc="outside upper left",
               bbox_to_anchor=(0.035, 0.93), ncols=1)
    fig.suptitle("Matched-budget value of Ministral's long prefixes — 13 censored "
                 "Wave 3 cells", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.savefig(out / "wave3_matched_compute.png", dpi=180)
    plt.close(fig)


def fig_dissection(report_path: Path, out: Path) -> None:
    r = json.loads(report_path.read_text())
    ts = [4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
    pooled = [r["anchors"][str(t)]["state_probe"]["pooled_auroc"] for t in ts]
    p_lo = [r["anchors"][str(t)]["state_probe"]["pooled_ci"][0] for t in ts]
    p_hi = [r["anchors"][str(t)]["state_probe"]["pooled_ci"][1] for t in ts]
    within = [r["anchors"][str(t)]["state_probe"]["within_problem_auroc"] for t in ts]
    w_lo = [r["anchors"][str(t)]["state_probe"]["within_ci"][0] for t in ts]
    w_hi = [r["anchors"][str(t)]["state_probe"]["within_ci"][1] for t in ts]
    ceiling = r["anchors"]["4"]["loo_pass_rate"]["pooled_auroc"]
    fig, ax = plt.subplots(figsize=(9.8, 5.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    ax.axhline(ceiling, color=MUTED, lw=1.4, ls=":")
    ax.text(510, ceiling - 0.025,
            "difficulty ceiling: LOO pass rate of 255 other samples "
            f"({ceiling:.3f})", ha="right", fontsize=8.4, color=SEC)
    ax.axhline(0.5, color=INK, lw=1)
    ax.text(510, 0.508, "chance", ha="right", fontsize=8.2, color=MUTED)
    ax.fill_between(ts, p_lo, p_hi, color=ORANGE, alpha=0.15)
    ax.plot(ts, pooled, "-o", color=ORANGE, lw=2.2, ms=5.5,
            label="pooled across problems, 8-per-problem set (n = 1,024)")
    ax.fill_between(ts, w_lo, w_hi, color="#0f8a74", alpha=0.18)
    ax.plot(ts, within, "-o", color="#0f8a74", lw=2.2, ms=5.5,
            label="within problem, 22 mid-band problems (21 at t = 512; n = 5,632)")
    ax.scatter([4], [0.84], marker="D", s=52, facecolor="none",
               edgecolor="#8a7ad9", linewidths=1.6, zorder=5)
    ax.annotate("0.84 reported by the\nuncontrolled original", (4, 0.84),
                textcoords="offset points", xytext=(14, 8), fontsize=8.2,
                color="#8a7ad9")
    ax.set_xscale("log", base=2); ax.set_xticks(ts)
    ax.set_xticklabels([str(t) for t in ts])
    ax.set_xlabel("Probe anchor t (generated tokens)")
    ax.set_ylabel("AUROC for trajectory correctness")
    ax.set_ylim(0.42, 1.01)
    ax.legend(frameon=False, fontsize=8.6, loc="center right")
    fig.suptitle("Dissecting an early-window probe positive — R1-Distill-7B "
                 "public trajectories", fontsize=11.5, color=INK, x=0.01, ha="left")
    ax.set_title("Same probe, two evaluation sets: the pooled positive replicates "
                 "at t = 4 and is indistinguishable from chance within problem at every anchor",
                 loc="left", fontsize=8.8, color=SEC)
    fig.savefig(out / "math128_probe_dissection.png", dpi=180)
    plt.close(fig)




def fig_ceiling(report_path: Path, out: Path) -> None:
    r = json.loads(report_path.read_text())
    point = r["loo_ceiling_auroc"] if "loo_ceiling_auroc" in r else r["primary_auroc_difficulty_loo"]
    ci = r.get("ci") or [r["primary_ci"]["ci_low"], r["primary_ci"]["ci_high"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    ax.axhspan(0.79, 0.95, color="#f3d9c9", alpha=0.55, zorder=0)
    ax.text(2.42, 0.945, "range of published probe AUROCs\n(internal-state "
            "positives, no question-only baseline)", ha="right", va="top",
            fontsize=8.6, color="#b06a34")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(2.42, 0.507, "chance", fontsize=8, color=MUTED, ha="right")
    ax.scatter([0], [0.5], color=MUTED, s=40, zorder=3)
    ax.plot([1, 1], ci, color="#0f8a74", lw=3, solid_capstyle="round", zorder=3)
    ax.scatter([1], [point], color="#0f8a74", s=70, zorder=4)
    ax.annotate(f"{point:.3f}", (1, point), textcoords="offset points",
                xytext=(12, 0), fontsize=10, color="#0f8a74", va="center")
    oracle = r.get("oracle_upper_reference_non_loo")
    if oracle:
        ax.scatter([2], [oracle], facecolor="none", edgecolor=INK, s=70,
                   marker="^", zorder=3)
        ax.annotate(f"{oracle:.3f}", (2, oracle), textcoords="offset points",
                    xytext=(12, 0), fontsize=9, color=SEC, va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["chance",
                        "trace-blind difficulty proxy:\nLOO pass rate of other "
                        "attempts\n(192,315 R1 generations, 91,573 problems)",
                        "oracle pass rate\n(non-LOO reference)"], fontsize=8.8)
    ax.set_ylabel("AUROC for trajectory-level correctness")
    ax.set_ylim(0.45, 1.02); ax.set_xlim(-0.4, 2.45)
    ax.grid(axis="x", visible=False)
    fig.suptitle("A trace-blind difficulty proxy already sits inside the published "
                 "probe range — DeepSeek-R1 public dumps",
                 fontsize=11.2, color=INK, x=0.01, ha="left")
    ax.set_title("Estimator frozen before evaluation; likely conservative (k = 2 for 92% "
                 "of problems; curation drops some all-fail problems)",
                 loc="left", fontsize=8.4, color=SEC)
    fig.savefig(out / "difficulty_ceiling_openr1.png", dpi=180)
    plt.close(fig)


def fig_sweep(sweep_root: Path, confirmatory_report: Path, out: Path) -> None:
    """Post-hoc forecast-point sweep (t=512 is the pre-registered point)."""
    ts = (128, 256, 512, 1024, 2048)
    rows = {"primary": [], "secondary": []}
    for t in ts:
        path = (confirmatory_report if t == 512
                else sweep_root / f"t_{t}" / "confirmatory_report.json")
        report = json.loads(path.read_text())
        for endpoint in rows:
            block = report[endpoint]
            ci = block["clustered_bootstrap"]["delta"]
            rows[endpoint].append((block["test_delta_auroc"],
                                   ci["ci_low"], ci["ci_high"],
                                   block["rows"]["test"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    titles = {"primary": "Primary: eventual success at 16K",
              "secondary": "Secondary: scratch-solvable at 4,096"}
    for ax, endpoint in zip(axes, ("primary", "secondary")):
        ax.axhline(0.0, color=INK, lw=0.9)
        for x, (t, (delta, lo, hi, n)) in enumerate(zip(ts, rows[endpoint])):
            starred = t == 512
            color = GREEN if starred else MUTED
            ax.vlines(x, lo, hi, color=color, lw=2.2 if starred else 1.8)
            ax.plot(x, delta, "o", color=color, ms=6.5 if starred else 5.5,
                    zorder=5)
            ax.annotate(f"n={n}", (x, lo), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=7.4,
                        color=MUTED)
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"{t:,}*" if t == 512 else f"{t:,}" for t in ts],
                           fontsize=9)
        ax.set_xlabel("Forecast point t (prefix tokens)", fontsize=9)
        ax.set_title(titles[endpoint], loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ΔAUROC (early signals − baseline), 95% CI",
                       fontsize=9)
    fig.text(0.01, 0.895, "* = the pre-registered point; all others post-hoc",
             fontsize=8, color=SEC)
    fig.suptitle("Post-hoc forecast-point sweep — no tested window yields "
                 "a detectable gain", fontsize=11.5, color=INK, x=0.01,
                 ha="left")
    fig.savefig(out / "early_signal_prefix_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--phase04c-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dissection-report", type=Path, default=None)
    parser.add_argument("--ceiling-report", type=Path, default=None)
    parser.add_argument("--sweep-root", type=Path, default=None)
    parser.add_argument("--confirmatory-report", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.labels)
    fig_regime_map(labels, args.output_dir)
    fig_dose_response(labels, args.output_dir)
    fig_topup_resolution(args.phase04c_root, args.output_dir)
    fig_label_hardening(args.output_dir)
    fig_swimmer(args.phase04c_root, args.output_dir)
    fig_budget_falsification(args.phase04c_root, args.output_dir)
    fig_matched_compute(args.phase04c_root, labels, args.output_dir)
    if args.dissection_report and args.dissection_report.exists():
        fig_dissection(args.dissection_report, args.output_dir)
    if args.ceiling_report and args.ceiling_report.exists():
        fig_ceiling(args.ceiling_report, args.output_dir)
    if args.sweep_root and args.confirmatory_report:
        fig_sweep(args.sweep_root, args.confirmatory_report, args.output_dir)
    print("figures rendered to", args.output_dir)


if __name__ == "__main__":
    main()
