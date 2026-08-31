# Protocol amendment: Phase 4 breakthrough forecasting

Base protocol frozen: 2026-08-16  
Phase 4C-U scoring/budget revision frozen: 2026-08-17

This amendment supersedes the older two-model terminal-failure and
`needs_intervention` Phase 4b/4c design. It does not delete the older document;
that file remains an audit trail.

## Scientific target

Primary target:

```text
P(T_B <= t + k | prefix features through t), k in {128, 256, 512}
```

`T_B` is the first interval-censored prefix anchor whose bounded continuations
reach at least 75% verifier success and whose next observed anchor also reaches
that threshold. No stable pair before observation ends is right-censored.

The retained baseline is `P(final correctness | prefix features)`. A composite
failure/intervention endpoint is not the primary label. A sequence encoder is
out of scope and the surface-text control is deferred.

## Frozen stages

1. **4B:** completed 100 level-balanced MATH problems × three models × seed 11 =
   300 base trajectories, 16,384 maximum generated tokens. The second seed was
   removed by the frozen compute-budget amendment and remains a replication.
2. **4C-P:** nested pilot containing one frozen problem per MATH level and all
   three models.
3. **4C-L:** the frozen 20-problem matched labeling cohort, launched only after
   manual pilot cost/label review.
4. **4C-U:** nested 512-token-prefix short/medium/long continuation arms testing
   whether high-uncertainty states benefit disproportionately from extra compute.
5. **4D:** local grouped eventual-success, breakthrough-horizon, and
   discrete-time logistic hazard models.
6. **4E:** prospective matched-budget compute control after the 4D policy is
   frozen.
7. **4F:** one-shot HARP short-answer OOD test after de-duplication and policy
   freeze.

## Continuation contract

- sparse anchors: 64, 128, 256, 512, 1024, 2048, 4096, 8192 reasoning tokens;
- four independent deterministic branch seeds per anchor;
- exact replay of every generated token through the selected reasoning-token
  cutoff;
- 1,024 additional reasoning tokens plus a 512-token answer reserve;
- adapter-specific reasoning-close insertion only when the model does not close
  naturally;
- never exceed 16,384 total generated tokens;
- refine only the first observed low-to-stable-high transition interval;
- sensitivity labels at thresholds 0.50, 0.75, and 1.00.

## Uncertainty-conditioned extension contract

- primary exact reasoning anchor: 512 tokens;
- primary score: a block-balanced `[0,1]` index. Within-model training-split
  empirical-CDF percentiles are first averaged inside predictive ambiguity,
  temporal instability, geometry instability, and spectral instability; the
  four block scores are then averaged equally. This prevents several highly
  correlated entropy-derived summaries from dominating the score. It is a
  relative index, not a calibrated failure probability;
- deterministic high/low ranked halves within model and MATH level;
- exclude prefixes shorter than 512 tokens or with a verifier-extractable
  correct answer already present;
- matched branch seeds for total reasoning targets 1,024, 4,096, and 24,576
  tokens, all including the frozen 512-token prefix;
- exact nested token paths across short, medium, and long arms;
- 512-token answer reserve for all arms and a 25,600 generated-token ceiling;
- primary estimand `accuracy(24K)-accuracy(4K)` within high uncertainty;
  primary policy interaction is its high-minus-low difference; 4K-minus-1K
  and 24K-minus-1K are secondary dose-response contrasts;
- the pilot validates mechanics only; inferential claims require the frozen full
  cohort and problem-cluster intervals.

## Dependence and censoring

Dense feature rows do not create new independent samples. Every model, seed,
prefix, and continuation derived from one `problem_id` remains in its immutable
train, validation, or test split. Confidence intervals resample problem
clusters. Ambiguous interval-censored horizons are omitted, never assigned a
negative label. The hazard risk set ends at the event proxy or censoring time.

## Fail-closed operational checks

- Phase 4B requires the completed exact 3 × 100 × 1 panel and verified payload
  hashes.
- Phase 4C resume identity includes source/config/model revision, probe manifest,
  exact prefix, anchor, branch seed, and fixed continuation budgets.
- Phase 4C validation requires exact branch keys and result hashes.
- Phase 4C-U additionally binds score version, stratum, budget arm, paired seed,
  and exact prefix hash; full-cohort execution requires an explicit approval
  flag after pilot review.
- The pilot reports projected full-cohort tokens/runtime, memory peaks,
  forced-boundary rate, verifier methods, event count, and censoring count; it
  never launches the full cohort automatically.
- Phase 4D reports underpowered instead of fitting a complex rescue model when
  train or test lacks both classes.
- Phase 4E and 4F remain closed until their frozen input artifacts exist.
