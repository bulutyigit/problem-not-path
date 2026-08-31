# Protocol amendment A5: restart-controlled breakthrough labels

Frozen: 2026-08-20, before any restart-control outcome exists. Motivated by
the A2 resolution: under the fixed 1,024-token continuation budget, the
breakthrough label measured prefix content for Gemma but remaining-budget fit
for Ministral. A5 separates those two quantities by adding the missing control
arm: what the model achieves from **no prefix at all**, at matched total
compute.

## Two estimands, both kept

The A2 result does not make the old label worthless; it shows there are two
distinct objects, and each keeps a name:

- **T_F(B) — budget-fit time** (the old `T_B`): earliest stable anchor whose
  bounded continuations succeed. Measures "remaining work now fits in B
  tokens." This is the **deployment-relevant** quantity for stop/extend
  controllers and remains the Phase 5 controller input, always reported with
  its budget index.
- **T_V(B) — prefix-value breakthrough** (new, the scientific object):
  earliest stable anchor where the prefix beats a compute-matched restart —
  i.e., where the trajectory's content demonstrably contributes beyond what
  the model does from scratch with the same total token budget.

## The restart control R

For each problem × model, estimate the restart curve

```text
R(C) = P(correct | fresh attempt from the problem statement, reasoning budget C)
```

with m = 4 fresh attempts at each C ∈ {1024, 2048, 4096, 8192} (final-answer
reserve 512 unchanged; deterministic seeds
`sha256("phase04c-restart:{problem_id}:{model_key}:{C}:{branch}")`).
R is per-problem, shared across that problem's trajectories — one restart
panel serves every anchor comparison. R̂(C) at intermediate C is interpolated
on log C between measured points (never extrapolated below 1,024 or above
8,192; anchors whose t+B exceeds 8,192 use R̂(8192), reported as conservative).

## Label definition

For anchor t probed with continuation budget B = 1,024 (canonical probes are
reused, no regeneration):

```text
advantage(t) = p(t, B) − R̂(t + B)
```

- **Prefix-value crossing** at t: `p(t, B) ≥ τ` (τ = 0.75 unchanged) AND
  `advantage(t) ≥ δ` with primary **δ = 0.5** (two of four branches);
  sensitivity reported at δ ∈ {0.25, 0.75}.
- **T_V** = earliest prefix-value crossing whose next probed anchor also
  satisfies both conditions (stability rule unchanged); interval/right
  censoring semantics carried over verbatim.
- **Regime classification** per problem × model, pre-registered:
  - `budget_limited`: R̂(4096) ≥ τ (solvable from scratch given room) — no
    T_V is claimed regardless of p(t, B);
  - `prefix_limited`: some prefix-value crossing exists;
  - `unsolved`: neither, within the probed range.

Under this rule the six Ministral A2 trajectories are expected to classify as
budget_limited and Gemma's two as prefix_limited — that expectation is
recorded here, before the restart data exists, as the pilot's success
criterion (see gate).

## Scope and cost

Restart panels are generated for every (dev ∪ supplement) problem × model
that is **not an instant solver** for that model (instant solvers'
classification is determined by their existing p(16, B) ≈ 1 and a single
R(1024) cell, which is still generated for completeness of the surface).
Estimated scope after Wave 1: ≈ 30–40 problems × 2 models × 4 budgets × 4
attempts ≈ 2–3M generated tokens (≈ 10–20 h Mac MLX, shardable; a BF16 RunPod
run satisfies the same manifest). Implementation: a standalone
`probe_restart_baseline.py` reusing the probe machinery with an empty prefix,
plus a labeling script that joins restart panels onto the frozen canonical
probe results. Canonical probe artifacts are never modified.

## Pilot and gate (replaces the failed G1 for Wave 3)

1. **Pilot:** restart panels for the 8 A2-tested problems only (+ their
   labels). Success criterion: the pre-registered regime expectations above
   reproduce (Ministral 6 → budget_limited; Gemma 2 → prefix_limited).
2. If the pilot passes, Wave 3 unblocks with T_V as the primary timing label
   and T_F(B) retained as the controller label; the Wave 3 probe manifest
   gains the restart panel as a required companion artifact.
3. If the pilot fails, the failure mode is reported and expansion stays
   paused; no post-hoc rule edits.

## Explicit non-goals

- No re-labeling of already-frozen T_F results; published summaries keep
  their budget index.
- δ, τ, budget grid, and the regime thresholds are not tuned after outcomes;
  sensitivity grids are reported as such.
- This is the final label redesign before the confirmatory loop; further
  construct changes would require a new phase, not an amendment.

## Pilot resolution (recorded 2026-08-20, after outcomes)

Both restart panels completed (6 + 2 problems × 4 budgets × 4 attempts).
Result against the pre-registered expectation: **7/8 exact matches**, and the
primary discriminator — R̂(4096) ≥ τ versus < τ — matched on **8/8**:

- **Ministral: 6/6 budget_limited** (R(4096) = 0.75–1.00 on every problem).
  One of them (math_03159) does show a compute-matched prefix advantage of
  0.55 at t = 768 under the 1,024 budget, but the frozen rule correctly
  blocks a T_V claim because the problem solves from scratch at 4,096.
- **Gemma math_03159: prefix_limited**, the pilot's cleanest genuine event —
  restart stays at 0.00 through C = 2,048 while the prefix curve reaches τ;
  T_V = 896 with advantage 0.75, stable, at every δ ∈ {0.25, 0.5, 0.75}.
- **Gemma math_03190: borderline miss.** Not budget_limited (R(4096) = 0.00),
  and a prefix-value crossing exists at δ = 0.25 (t = 256, advantage 0.42)
  but not at the primary δ = 0.5 — a gap of 0.08 against a one-attempt
  resolution of 0.25 on each side.

Also recorded: restart rates are non-monotone in C on several problems
(e.g., 1.00 → 0.50 from 4,096 to 8,192) — the m = 4 noise floor documented
for probes applies equally to restart panels.

**Gate decision, per the frozen wording:** the success criterion is not fully
met (7/8), so Wave 3 does not unblock yet. The recorded failure mode is
measurement resolution on a single cell, not a rule defect; the remedy stays
inside existing rules: enlarge the sample on the borderline cell (additional
restart attempts for gemma × math_03190 under the same deterministic seed
scheme, branches 4–7) and re-evaluate the same frozen criterion once. No
threshold is edited.

## Re-evaluation of the borderline cell (final; recorded 2026-08-20)

The permitted single enlargement ran: gemma × math_03190 restart attempts
doubled to 8 per budget (branches 4–7, same seed scheme). Pooled rates:
R(1024) = 3/8, R(2048) = 6/8, R(4096) = 2/8, R(8192) = 4/8. Outcome: the
prefix advantage at t = 256 **fell** from 0.42 to 0.25, and the δ = 0.25
crossing also disappeared (stability fails at t = 512, advantage 0.16). Final
classification: `no_crossing_at_primary_delta` — the problem is neither
budget-limited at 4,096 nor shows demonstrable prefix value; it is
substantially solvable from scratch at mid budgets (6/8 at 2,048).

**Final pilot tally: 7/8. The gate criterion as frozen is not met; Wave 3
remains paused under this amendment.** Reclassified failure mode: not
measurement resolution but a **wrong prior** — the pre-registered expectation
assumed both Gemma problems were prefix-limited, which the A2 data could not
actually establish for math_03190 (A2 only probed the prefix side). The
instrument itself performed as designed: 6/6 budget-limited detected, one
maximal-advantage prefix-value event validated (T_V = 896), and one
genuinely-ambiguous case correctly refused a label under every δ.

Additional observation, now at n = 8 per cell: from-scratch performance on
math_03190 is non-monotone in budget (6/8 at 2,048 vs 2/8 at 4,096) — a
concrete instance of longer thinking degrading from-scratch success.

Any Wave 3 unblocking now requires a new, forward-looking pre-registered
gate (a successor amendment defined on instrument performance rather than
prior correctness), or a documented protocol deviation; this amendment's own
gate is exhausted and its result stands.
