# Phase 5 breakthrough-aware controller execution

Phase 5 is intentionally named `phase_05_breakthrough` on disk so it cannot be
confused with the repository's older `phase_05` correctness-predictor notebook.
Only Gemma 4 E4B and Ministral 3B are in scope.

## Scientific contract

1. Complete and validate the full frozen MATH breakthrough-probe cohort.
2. Build interval-censoring-aware horizon tables.
3. Produce problem-grouped out-of-fold MATH forecasts and arm-response
   predictions; select the compute penalty; refit and freeze.
4. Canonicalize and deduplicate the official HARP short-answer dataset.
5. Generate HARP prefixes and freeze every routing decision before opening
   continuation outcomes.
6. Generate matched short/medium/long counterfactual arms once and evaluate.

The expensive stages require `--approve-external-run`. Every generator uses
atomic branch completion files and `--resume`; rerunning a stage resumes only
hash-compatible outputs.

## Status

```bash
./.venv/bin/python scripts/run_phase05_breakthrough.py status
```

## A. Finish the MATH development labels

Run each shard in a separate terminal if desired. On one Mac, use one shard and
let the two models run sequentially:

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  complete-development-probes --approve-external-run

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  validate-development-probes

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  build-development-tables

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  fit-freeze-controller
```

The last command must reject pilot-only labels. Do not bypass that gate.

## B. Freeze HARP

Download `HARP.jsonl.zip` from the official HARP repository and pass its local
path explicitly:

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py prepare-harp \
  --harp-source /absolute/path/to/HARP.jsonl.zip
```

This writes the canonical cohort, duplicate audit, and immutable manifest under
`artifacts/mac_mlx/phase_05_breakthrough/datasets/`.

## C. Generate prefixes and freeze routing

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  generate-harp-prefixes --approve-external-run

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  extract-harp-prefix-features

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  build-harp-extension-manifest

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  freeze-harp-routing
```

Do not inspect or generate HARP continuation outcomes before
`harp_routing_manifest.json` exists and its digest validates.

## D. Generate matched counterfactual arms and evaluate once

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  generate-harp-arms --approve-external-run

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  validate-harp-arms

./.venv/bin/python -u scripts/run_phase05_breakthrough.py \
  evaluate-harp
```

The confirmatory report is written to
`artifacts/mac_mlx/phase_05_breakthrough/analysis/phase05_harp_report.md`.

## Interpretation gate

The main controller must be compared against fixed medium and the frozen
U512-only ablation. A positive Phase 5 result requires a better external HARP
accuracy-compute frontier, not merely a higher raw accuracy obtained by spending
more tokens. If the controller fails this comparison, report the negative result
without retuning on HARP.

## Fit hardening (2026-08-20, before any controller freeze)

Applied after an external code review, while no controller has ever been fit
or frozen (so no frozen artifact changes meaning):

1. **Intersection-level power gate.** `fit_phase05_breakthrough_controller.py`
   now writes `phase05_fit_status.json` and stops with
   `underpowered_after_intersection` based on the post-filter,
   post-response-pair-intersection target tables (per-class problem-group
   support vs folds), instead of relying on the labeling-cohort count. On the
   current data the gate reports: horizon target 17 rows / 10 problems /
   2 positives in 1 problem group — underpowered; eventual-success passes
   (30 rows / 18 problems, 4v16 problem groups).
2. **Low-dimensional forecast features.** New frozen `early_blocks` feature
   set (15 fixed block summaries + early baseline ≈ 21 predictors,
   fail-closed on missing columns) is the fit default; `early_full`
   (~335 predictors) remains available only as an explicit sensitivity flag.
3. **Fail-closed penalty selection.** `select_compute_penalty` now raises
   `no feasible compute penalty` instead of silently freezing the cheapest
   infeasible policy when no grid point meets the frozen accuracy gap.

Also recorded from the same review: the 120 development response rows
represent ~18 problem clusters (30 sources × 4 branches) and must be reported
as such; supplement-cohort labels cannot enter the controller fit without new
short/medium/long arm generations (their problems have zero overlap with the
response pairs); and controller targets are budget-indexed (`T_F`) — they are
not evidence of prefix-value breakthroughs (`T_V`, amendment A5) and must not
be named as such in reports.
