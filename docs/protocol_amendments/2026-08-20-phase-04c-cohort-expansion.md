# Protocol amendment: Phase 4c cohort expansion (waves, gates, and sizing)

Frozen: 2026-08-20, before any expansion outcome exists. Motivation: the
development cohort (40 trajectories, 8 primary interior events + 3 A1
sensitivity events) is too small to separate candidate signals — at 8v4
trajectories the standard error of a standardized mean difference is ≈ 0.6,
so screen effects up to |d| ≈ 1.2 are compatible with noise. The unit that
buys information is the **problem** (clusters), secondarily within-problem
seed replication.

## Gates (checked in order; compute is not spent past a failed gate)

- **G1 — construct validity.** The A2 continuation-budget falsification probe
  (both models) must come back "construct supported" per its pre-registered
  rule (median first-crossing shift ≤ one anchor step). If the mechanical
  length confound is flagged, expansion pauses until the probe protocol is
  redesigned; scaling a broken label is forbidden.
- **G2 — supplement yield.** Wave 1 must produce ≥ 6 new interior events
  across its 40 trajectories. Below that, the screen/anchor design is
  revisited before Wave 3 spend.
- **G3 — frozen selection.** All selection rules in this document are frozen
  now; outcomes never edit them retroactively.

## Wave 1 — probe the frozen supplement cohort (no new design)

The A3 supplement manifest (20 intermediate-difficulty problems × 2 models)
is probed with the unchanged protocol. Expected yield, based on the pool
statistics: 8–16 additional interior events. Mac, ≈ 3–4 h per model.

## Wave 2 — within-problem seed replication

New base trajectories at **seed 12** for the 20 supplement problems × 2
models (16,384 tokens, instrumented, unchanged configs except the seed), then
probes under the same manifest rules. Purpose: (a) more events on problems
already known to be informative; (b) a new estimand — **seed stability of
T_B** (do two runs of the same model on the same problem break through at
similar times?), which no current data can answer and which bounds how much
of T_B is problem-level versus trajectory-level.

Assets: `scripts/prepare_seed_replication.py` writes the 20-problem bundle
subset (original research splits preserved verbatim) and the two seed-12
configs. Manifest: `build_breakthrough_supplement_manifest.py
--preselected-manifest` (screen inherited from A3; no new outcome
conditioning). Mac cost: ≈ 13–20 h base generation + ≈ 6–8 h probes.

## Wave 3 — fresh problems (the real expansion)

1. **Bundle:** 150 new level-balanced MATH problems, pinned to the same
   dataset revision as the Phase 4b bundle, excluding every problem ID in the
   Phase 4b bundle and the challenge bundle (passed via `--exclude-bundle`);
   stratified research splits assigned at bundle creation (seed 20260820),
   before any outcome exists. The pre-pivot Phase 1–3 historical bundles are
   not present on this machine; overlap with them is tolerated (those phases
   never touched the breakthrough pipeline, so no outcome can leak into this
   selection) — if the historical bundle file is restored, pass it as an
   additional `--exclude-bundle`. Asset: `scripts/prepare_expansion_screen.py`.
2. **Screen:** 3 seeds (21, 22, 23) × 2 models × 3,072-token budget, no
   hidden-state capture, standard `generate.py`. Selection rule, frozen:
   total successes over the 6 screen samples ∈ **[1, 5]** (at least one
   success and one failure). Known bias, accepted deliberately: a 3,072-token
   screen favors problems with attainable short solutions — coherent with the
   probe's own bounded 1,024+512 continuation regime. Expected band size from
   pool statistics: ≈ 25–40 problems.
3. **Base generation:** 16,384-token instrumented trajectories, seed 11, both
   models, selected problems only. Asset: `scripts/select_expansion_cohort.py`
   freezes the cohort (digest over the selection), writes the filtered bundle
   and the base-generation configs.
4. **Probes:** unchanged protocol via a manifest built from the new
   generation directory.

**Backend recommendation:** Wave 3 is the natural point to run the BF16 CUDA
profile (RunPod) instead of MLX 4-bit — it removes the quantization confound
flagged in review while adding scale. If run on BF16, the readiness/smoke
gate and configs use the `bf16_cuda` profile and the manifest records the
backend; 4-bit Mac remains the fallback. Cost either way, rough: screen
≈ 2.7M generated tokens; base generation ≈ 0.5–1.0M tokens at 16K with
instrumentation; probes ≈ 2× the completed development probe run.

## Sizing target

Waves 1–3 together move the cohort from 20 problems / 8 interior events to
≈ 65–80 problems / **≥ 30 interior events**, the threshold below which the
statistician's review declared timing covariates unidentifiable. Splits:
new problems carry their own pre-outcome stratified splits; all expansion
estimands remain conditional on the intermediate-difficulty screens; the
test split stays untouched until the single confirmatory run.

## Wave 1 / Gate G2 resolution (recorded 2026-08-20, after outcomes)

Wave 1 completed cleanly (40/40 trajectories, 1,048/1,048 branches, 0
corrupt). Outcomes per model: Ministral 5 instant / 3 interior (T_B ≈ 80,
160, 320) / 12 censored; Gemma 7 instant / 1 interior (T_B ≈ 48) / 12
censored; cross-model class agreement 15/20. Two new Ministral
terminal-replication candidates (final-anchor rates 1.0 @1,024 and 0.75
@8,192) go through the A1 sensitivity rule.

**Gate G2 fails on primary labels: 4 new interior events < 6** (at most 6 if
both A1 candidates confirm — borderline either way against the 8–16
expectation). Diagnosis, recorded for the redesign: the screen selected on
**16K terminal solvability**, but probes test solvability under **1,024-token
continuations** — the same budget-indexing lesson as A2. A problem solvable
only via long solutions passes the screen yet censors under probes (60% of
the supplement censored). Future cohort screens must select on
probe-budget-scale solvability — exactly what the A5 restart panel R(C)
measures — rather than on terminal outcomes at the full generation budget.

Status: G1 and G2 now point at the same prerequisite. Wave 3 remains paused;
the A5 pilot is the critical path. Wave 2 (seed replication) remains valid
but is deprioritized behind the A5 pilot. Wave 1 data stands as planned for
eventual-success/cure estimands (+40 labeled trajectories, 24 censored).
