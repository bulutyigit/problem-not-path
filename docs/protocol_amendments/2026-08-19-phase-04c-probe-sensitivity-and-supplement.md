# Protocol amendment: Phase 4c probe sensitivity analyses and supplement cohort

Frozen: 2026-08-19, after the full development probe cohort completed labeling
(40 trajectories, 23 events / 17 right-censored) and **before any amendment
outcome exists**. The primary label definition (τ = 0.75, m = 4, next-anchor
stability, interval/right censoring) is unchanged; nothing here overwrites the
frozen labels in `phase_05_breakthrough/development_labels/`.

## Motivating observations (from the completed cohort)

1. 4 of Ministral's 9 right-censored trajectories reach the τ = 0.75 threshold
   at their final probed anchor (4,096) — two at 4/4 — but cannot satisfy
   next-anchor stability because no later anchor exists. Their censoring is
   plausibly a grid artifact, not model inability.
2. Only 8 of 40 trajectories have interior breakthrough times. All 8 were
   labeled with a fixed 1,024-token continuation budget, so `T_B` could be a
   mechanical reparameterization of solution length (`T_B ≈ length − budget`).
   No test in the frozen protocol can currently falsify this.
3. Level-balanced selection produced 15 instant solvers and 17 censored
   trajectories; cross-model terminal outcomes (16K, seed 11) identify an
   intermediate-difficulty band that level does not.

## A1 — Terminal-replication stability (sensitivity labels only)

For every right-censored trajectory whose **final probed anchor** has success
rate ≥ τ, generate `extra_continuations = 4` additional branches at that same
anchor (branch indices 4–7 under the existing deterministic seed scheme,
`sha256("phase04c:{run_id}:{anchor}:{branch_index}")`). The sensitivity rule:
the event is deemed stable if the pooled rate over all 8 branches is ≥ τ
(≥ 6/8), with `interval = (previous_anchor, final_anchor]` and
`stability_rule = "terminal_replication"`.

These labels are written to
`phase_04c/probes/sensitivity/terminal_stability/<model>/` and reported only as
a sensitivity analysis alongside the primary labels. Canonical branch
directories are never modified.

## A2 — Continuation-budget falsification probe

For every trajectory with an interior event (`event_observed` and
`interval_upper > 16`; currently 2 Gemma + 6 Ministral), re-probe every
canonical anchor `≤ stability_anchor` with
`reasoning_continuation_budget = 4096` (reserve 512, total cap 16,384
unchanged), m = 4, **identical branch seeds** to the canonical run so each
4,096-budget branch extends its 1,024-budget counterpart (paired comparison).

Pre-registered readout, per trajectory: the first τ-crossing anchor and the
stable anchor under the 4,096 budget versus the canonical 1,024 budget.
Decision rule, committed before outcomes exist:

- **Mechanical-confound flag** if, for the majority of re-probed trajectories,
  the first τ-crossing shifts earlier by ≥ 2,048 tokens (≥ ⅔ of the 3,072-token
  budget delta) — `T_B` then tracks remaining solution length and the
  construct's incremental validity must be demonstrated before further use.
- **Construct supported** if the median shift is at most one anchor step.
- Intermediate outcomes are reported as ambiguous; no post-hoc reinterpretation.

Outputs land in `phase_04c/probes/sensitivity/budget_4096/<model>/`.

## A3 — Supplement cohort screened on cross-model solvability

Selection rule (frozen here, applied deterministically): from the 80 Phase 4b
problems not in the development cohort, select **every** problem whose
three-model terminal solved-count at 16K (seed 11, base generation) is 1 or 2.
This yields 20 problems (levels 1/2/3/4/5 → 1/6/3/5/5; imbalance documented,
not corrected). Probed models: `gemma4_e4b_mlx_4bit` and
`ministral3_3b_mlx_4bit`, reusing the frozen Phase 4b base trajectories and the
unchanged probe protocol.

Departure from the metadata-only selection rule, stated explicitly: this screen
conditions on **base-generation terminal outcomes**, which the original policy
forbade. It does not condition on probe outcomes (none exist for these problems
at freeze time). All supplement-cohort estimands are conditional on the
intermediate-difficulty screen; no marginal-population claim will be made from
this cohort. The screen is enrichment on a design variable, standard in
event-time studies, adopted because the outcome-blind cohort demonstrably
concentrates mass at the instant/never extremes (observation 3).

Manifest: `phase_04c/manifests/breakthrough_supplement_manifest.json`, same
schema and digest scheme as the development manifest.

## Execution and cost

All three runs are user-launched local MLX generation:

- A1: 4 trajectories × 4 branches (Ministral only) — minutes.
- A2: ~45 anchor-probes × 4 branches at ≤ 4,608 generated tokens — hours.
- A3: 20 problems × 2 models under the standard protocol — comparable to the
  completed development run.

## A2 resolution (recorded 2026-08-20, after outcomes)

Both models completed, plus a floor-resolving extension probing anchors 16/32
under the 4,096 budget (`--extra-anchor`). Outcomes:

- The pre-registered numeric mechanical flag did not trigger (0/8 shifts
  ≥ 2,048 tokens) — but that criterion was unreachable for Ministral, whose
  canonical crossings were all ≤ 768; the flag's operationalization was too
  coarse, which we record as a design lesson.
- **Gemma: construct supported.** Both crossings pinned at 256 under the 4×
  budget; success below the crossing stays sub-threshold even at 4,096.
  Prefix-limited breakthrough.
- **Ministral: budget-limited, construct not supported for the six primary
  interior events.** With the 4,096 budget every trajectory succeeds from a
  16-token prefix (rates 0.75–1.0) — effectively no prefix is needed.
  `T_B(1024)` for these events measures when remaining work fits into 1,024
  tokens, not when the prefix acquires decisive content.
- Strict base-length arithmetic (`T_B = length − budget`) remains rejected for
  both models: continuations solve within ≤ 1,024 tokens while base
  trajectories ran 2–5× longer.

Consequence: breakthrough labels are budget-indexed, `T_B(B)`, and any timing
claim must carry a budget-sensitivity control. Expansion gate G1 is treated as
**failed for Ministral under the current protocol**; Wave 3 waits for a
redesigned label (restart-controlled value-of-prefix or an explicit p(t, B)
surface). Waves 1–2 proceed: eventual-success/cure estimands and Gemma timing
are unaffected.
