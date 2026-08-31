# Protocol amendment A5.1: successor gate for Wave 3

Adopted: 2026-08-20, **with full knowledge of the A5 pilot outcomes.** This
amendment is therefore a documented, reasoned protocol change — it does not
and cannot claim pre-registration status with respect to the pilot data. The
A5 gate's own record stands unedited: final tally 7/8, criterion not met,
Wave 3 paused under A5. What follows replaces that gate; the honest sequence
(gate frozen → gate failed → gate superseded with reasons) is part of the
project's audit trail and is reportable as such.

## Why the A5 gate is the wrong instrument for this decision

The A5 pilot criterion required a **prior expectation** (all six Ministral
problems budget-limited, both Gemma problems prefix-limited) to reproduce.
Six of eight expectations tested the instrument against regimes that A2 had
actually established. The eighth — gemma × math_03190 — encoded a guess that
A2 could never have supported: A2 probed only the prefix side of that
problem, so its from-scratch behaviour was unknown when the expectation was
frozen. The enlarged sample showed the guess was wrong (the problem is
half-solvable from scratch at 2,048 and shows no stable prefix advantage at
any δ). A criterion that fails when a *prior* is wrong, while the
*measurement* behaves exactly as designed, gates on the wrong quantity.

## Replacement criterion: instrument performance

Wave 3 unblocks iff all three hold on the recorded pilot data:

1. **Primary discrimination.** The budget-limited criterion
   (R̂(4096) ≥ τ vs < τ) agrees with the A2-established regime on every
   problem where A2 established one. Recorded: **8/8.**
2. **Positive control.** At least one prefix-value breakthrough is validated
   at the primary δ with the unchanged stability rule. Recorded:
   gemma × math_03159, **T_V = 896**, advantage 0.75, stable at every δ.
3. **Refusal behaviour.** On cases satisfying neither regime criterion, the
   instrument refuses a label rather than forcing one, and enlarging the
   sample moves the estimate rather than the rule. Recorded:
   gemma × math_03190, advantage 0.42 → 0.25 under 4 → 8 attempts, label
   correctly withheld.

All three conditions are met; **Wave 3 unblocks upon adoption of this
amendment**, with the supersession of A5's gate recorded as a deviation from
the original decision rule.

## Genuinely forward-looking Wave 3 commitments (outcomes unknown)

These bind future data and are pre-registered in the ordinary sense:

1. **Labels.** T_V (restart-controlled, δ = 0.5 primary, sensitivity
   {0.25, 0.75}) is the primary timing label for Wave 3; T_F(B) is retained,
   budget-indexed, as the controller/deployment label. Every Wave 3 problem
   ships with its restart panel as a required companion artifact (A5 scope
   rule), and regime classification (budget_limited / prefix_limited /
   no-crossing) is reported for the full cohort.
2. **Screen alignment.** The Wave 3 screen (3 seeds × 3,072-token fresh
   attempts, select total successes ∈ [1, 5] of 6 — frozen in the
   2026-08-20 expansion amendment) is itself a from-scratch measure at
   probe-scale budget, repairing the Wave 1 misalignment; it is kept
   unchanged.
3. **Ambiguity-triggered enlargement.** For Wave 3 measurements (probe
   anchors and restart cells alike), any cell whose 4-attempt success count
   lies in {1, 2, 3} is enlarged to 8 attempts before labels are derived;
   thresholds (τ, δ) apply to pooled rates and are not changed. Extreme cells
   (0/4, 4/4) stay at 4. This generalizes the A1/borderline lesson into the
   standard protocol for new data.
4. **Non-monotonicity sensitivity.** R̂ interpolation assumes local
   monotonicity that pilot data visibly violates. The primary advantage
   computation keeps log-C interpolation; a conservative sensitivity variant
   recomputes advantage against the **larger** of the two bracketing grid
   values (upper envelope). Both are reported; disagreements are flagged, not
   resolved post hoc.
5. **Power rule for timing models.** No covariate-bearing timing model is fit
   unless the pooled cohort holds at least **10 interior T_V events per model
   parameter**; below that, timing results are reported descriptively with
   the `underpowered` status, per the project's standing rule.

## Non-goals

- No re-scoring of A5 pilot data under new rules; its numbers stand as
  recorded.
- No change to τ, δ, the budget grid, the regime thresholds, or the Wave 3
  screen band.
- This amendment supersedes exactly one thing: the exhausted A5 pilot gate.
