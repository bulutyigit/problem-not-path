# Phase 5 confirmatory early-signal test (2026-08-27, frozen before any test-split evaluation)

## Question

Do within-run early-window signals (first 512 generated tokens) predict a
trajectory's outcome **beyond what problem difficulty already predicts**?
The exploratory dev-cohort analysis answered "no" (early signals ≤ difficulty
baseline everywhere). This is the single confirmatory test of that question on
the held-out test split, which has never entered any analysis, figure, or
threshold choice.

## Data (all frozen before this amendment)

- 178 instrumented 16K base trajectories: 89 problems × 2 models
  (dev 20 + supplement 20 + wave3 49; phase_04b generation for dev+supplement,
  wave3_expansion generation for wave3).
- Research splits assigned before outcomes: train 54 / validation 18 /
  test 17 problems. Fitting uses train+validation only; test is evaluated once.
- Forecast point t = 512: features are the frozen prefix-512 tables
  (phase_04b/features for dev+supplement; wave3 extracted with the same
  `extract_features.py`, same settings, prefix 512).
- Inclusion rule (same as the frozen fit script): a cell enters an endpoint
  only if its base run reached ≥ 512 generated tokens
  (`observed_token_count ≥ 512`). Exclusion counts are reported per split.
  `observed_token_count` in prefix tables is capped at the prefix and is
  therefore knowable at forecast time.

## Endpoints

1. **Primary — eventual success:** did this 16K run verify correct?
   (`correct` of the base trajectory). Pooled over both models.
2. **Secondary — scratch-solvability at 4,096:** for non-instant cells,
   Y = 1{R̂(4096) ≥ 0.75} from the A5 restart panels (log-C interpolation,
   the budget_limited condition). Pooled over both models.
3. **Horizon endpoint (breakthrough within next 512 tokens):** reported only
   as power-gate arithmetic; no fit unless every gate passes (not expected).

## Frozen analysis plan

- Feature sets, via the frozen `feature_columns()` resolver:
  - baseline = `early_baseline` (difficulty columns + model_key + forecast
    constants; **no within-run signals**),
  - main = `early_blocks` (= baseline + the 15 frozen block summaries).
  - Sensitivity: both sets with `level` dropped (level-free variants),
    reported descriptively.
- Model: `fit_predictor(model_name="logistic_regression")`, unchanged
  preprocessing. No calibration layer (AUROC is invariant to it).
- Train-side sanity: 5-fold StratifiedGroupKFold OOF AUROC on
  train+validation, grouped by problem_id (both models' rows of a problem
  travel together). No hyperparameter search anywhere.
- Final models: refit on all train+validation rows, then frozen.
- Power gate (per endpoint, checked before unblinding test): each class must
  have ≥ 5 problem groups in train+validation. Test-side support is reported
  with the result; if the test split turns out single-class, AUROC is reported
  as undefined and the endpoint as uninformative — not retried.
- Test evaluation (single shot): AUROC of baseline and of early_blocks;
  primary statistic **ΔAUROC = AUROC(early_blocks) − AUROC(baseline)** on the
  same rows; problem-clustered bootstrap (resample test problems with
  replacement, 2,000 draws, both models' cells travel with their problem),
  95% percentile CI on each AUROC and on the paired ΔAUROC.
- **Success criterion:** the ΔAUROC 95% CI excludes 0 (positive side).
  Anything else is a confirmed negative: early-window signals add no
  information beyond difficulty at this scale. Per-model breakdowns are
  reported descriptively only.

## Openly recorded prior exposure of the test split

During labeling and cohort accounting the following test-split aggregates were
observed (no featurization, no outcomes joined to signals): split sizes; the
regime-count-by-split table (test: 9 budget_limited / 8 instant_scratch /
1 instant_prefix_advantaged / 16 unsolved_hard). No early-window feature of
any test problem has been inspected, and no threshold anywhere was chosen
after seeing test data.

## Outputs

`artifacts/mac_mlx/phase_05_breakthrough/confirmatory_early_signal/`:
assembled cell table, frozen-model coefficients, OOF audit,
`confirmatory_report.json`, figures. The result is committed as-is,
positive or negative.

## Post-hoc addendum (2026-08-27, recorded after the single confirmatory run)

The confirmatory run at t = 512 completed with the negative recorded above the
same day. At the researcher's request, the identical pipeline is additionally
run at forecast points t ∈ {128, 256, 1024, 2048} as an **openly post-hoc
sensitivity sweep** — five test-split evaluations in total once this sweep
completes. These runs are exploratory by construction: the pre-registered
result remains the t = 512 evaluation alone; a positive ΔAUROC at any other
prefix cannot be promoted to a confirmed finding from this data and would
require fresh pre-registered data to confirm. Larger prefixes also shrink and
select the sample (only runs that survived to t enter), which changes the
estimand; per-prefix inclusion counts are reported with the results under
`artifacts/mac_mlx/phase_05_breakthrough/early_signal_posthoc_sweep/`.
