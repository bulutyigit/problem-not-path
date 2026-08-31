# Protocol amendment: Phase 2 partial-trajectory failure dynamics

Initial amendment: 2026-08-03

Analysis refinement after Phase 1 review: 2026-08-04

Operational model substitution before Phase 2 analysis: 2026-08-04

## Change

The preregistered external reasoning-budget ablation is replaced by an
exploratory Gemma 4 E4B IT study on 100 MATH problems. The panel is nested: it
contains all 50 Phase 1 MATH problems and adds 10 deterministic problems per
level, yielding exactly 20 problems from each labeled difficulty level. Four
decoding seeds produce 400 trajectories under the same 8,192-token limit used
by the subsequent cross-model comparison.

After Phase 1 review, the primary exploratory question is refined to:

> Can a partial Gemma reasoning trajectory predict terminal failure beyond
> difficulty, category, and the number of observed tokens?

Fixed prefixes of 16, 32, 64, 128, 256, 512, 1,024, and 2,048 reasoning tokens
are evaluated.
Failure is the positive class. A difficulty/category/observed-token baseline is
compared with confidence-dynamic, hidden-geometry, spectral, and combined
feature blocks using immutable problem-level splits and problem-clustered
uncertainty intervals.

The first 16, 32, 64, and 128 thinking tokens form a prespecified onset analysis.
Mean normalized entropy, largest local entropy rise, mean sampled-token
surprisal, and mean top-1/top-2 probability margin are tested separately for:

- terminal-failure information after controlling for MATH level and category,
- difficulty information after controlling for terminal-failure rate and category.

This prevents difficulty-associated uncertainty from being presented as an
early failure signal, or vice versa.

The second prespecified analysis asks whether problem-level seed instability is
associated with geometry and spectral features, and whether geometry and
spectral blocks are redundant or complementary after controlling for level,
category, and mean reasoning length.

Phase 3 is correspondingly changed to a matched comparison of Gemma 4 E4B,
Qwen3.5-9B, and Ministral 3 8B on the exact same 100 MATH problems and four
seeds. The matching Phase 1 Gemma subset is reused inside Phase 2 only after
exact generation-setting checks. The complete Phase 2 Gemma panel is then
materialized inside the Phase 3 artifact tree; Qwen and Ministral are generated
as previously unseen model-family checks.

## Timing and status

The initial amendment was made after Phase 1 generation had begun. The refined
failure-prediction question was recorded after reviewing Phase 1 and before any
Phase 2 or Phase 3 analysis. A Qwen operational pilot was stopped because its
throughput was impractical; no pilot outcome or feature analysis was inspected,
and its trajectories are excluded from all Phase 2 artifacts and claims. The
model substitution therefore precedes hypothesis selection. Phase 2 is
exploratory: it may generate
hypotheses about failure-associated entropy, surprisal, logit margins,
hidden-state motion, spectral structure, seed instability, and local spikes,
but it is not confirmatory evidence for features selected using the same data.

## Interpretation boundaries

- Token predictive entropy is not identified as epistemic uncertainty.
- Four seeds are repeated stochastic observations; the effective independent
  sampling unit is the problem.
- Confidence intervals resample complete problem clusters.
- Raw hidden coordinates are never compared across model families.
- MATH category is retained as a control for difficulty-category imbalance.
- Every prefix result reports active-trajectory coverage; shorter trajectories
  are not treated as though unobserved future tokens were available.
- Geometry/spectral associations are computed from normalized scalar summaries,
  never raw cross-model hidden coordinates.
- In Phase 3, both Qwen and Ministral test the frozen feature blocks on
  previously unseen model families over the same MATH panel. Neither comparison
  is an independent-dataset confirmation.
- GSM8K is reserved for later out-of-domain confirmation after hypotheses and
  uncertainty models are frozen.
- Finish-reason instrumentation must pass audit before censoring or truncation
  claims are interpreted.

## Motivation

The amendment prioritizes a controlled, nested, level-labeled design and a
reusable matched cross-model panel. It reframes the project around calibrated
terminal-failure prediction from partially observed stochastic reasoning
trajectories, rather than descriptive token statistics alone. Geometry and
spectral analyses test whether trajectory shape and temporal frequency content
offer complementary early-warning information.
