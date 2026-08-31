# External re-analysis: David (arXiv:2511.14773) with difficulty controls
(2026-08-28, frozen before any generation or probing)

## Purpose

Test whether the closest published early-window probe positive — Temporal
Predictors of Outcome in Reasoning Language Models (Joey David, 2025):
ROC-AUC ≈ 0.84 at t = 4 reasoning tokens — survives the two controls our
confirmatory test used: a question-only difficulty baseline and
problem-clustered evaluation. No code or data were released; the setup is
reconstructed from the paper. This is a **descriptive external
re-analysis**, not a confirmatory test of our own hypotheses; every outcome
is reported.

## Reconstruction (faithful where specified, frozen here where not)

- Model: Qwen3-8B (paper's reasoning-tuned model), 4-bit MLX
  (`mlx-community/Qwen3-8B-4bit`) — quantization is a deviation, recorded;
  the paper's Llama-3.1-8B arm is out of scope (reasoning regime is the one
  at issue).
- Data: MATH test pool, 750 easy (levels 1–2) + 750 hard (levels 4–5),
  level-balanced draw with frozen seed 20260829 (the paper's exact 1,500
  problems are unknown).
- Generation: greedy, max_new_tokens = 512, thinking mode, one trajectory
  per problem (config `phase_ext_david_qwen3_8b_mlx_4bit.yaml`, seed 11).
  Correctness = our standard verifier on the generated text (an unfinished
  answer counts as incorrect, as the 512 cap implies in the paper).
- Probe (replication target): final-layer hidden states, mean-pooled over
  the last 4 reasoning tokens at prefix length t, PCA to ≤128 components,
  L2-regularized logistic regression, t ∈ {4, 8, 16, 32, 64, 128, 192,
  256, 384, 512}; 80/20 stratified random split with class-weight
  balancing, averaged over 20 random splits (the paper does not state a
  repeat count; 20 is frozen here).

## Frozen additions (the point of the exercise)

1. **Question-only baseline**: logistic regression on the frozen difficulty
   features (MATH level + problem text statistics used in our confirmatory
   test), same splits.
2. **Combined model**: baseline features + probe score (the probe's
   out-of-fold prediction as a single feature), to measure incremental
   value.
3. **Problem-clustered evaluation**: with one greedy trajectory per problem
   a random split is already problem-disjoint; the control here is
   difficulty leakage, not problem overlap. We therefore also report
   **within-half AUROCs** (easy-only, hard-only) and **within-level**
   AUROCs for both probe and baseline, plus pooled AUROC with a
   problem-bootstrap 95% CI (2,000 draws).
4. Primary quantity: **ΔAUROC(probe + baseline vs baseline alone)** at each
   t, with bootstrap CI; and the decomposition "pooled minus within-half"
   as the measured difficulty-leakage share of the probe's AUROC.

## Stated expectations (recorded before outcomes)

From our confirmatory null and the OpenR1 ceiling: the pooled probe AUROC
will replicate well above chance, a large share of it will vanish
within-half/within-level, and ΔAUROC over the question-only baseline will
be small or zero. If instead the probe survives the baseline (as
pre-generation activation probes did in Lugoloobi et al. 2026), that is
reported as a genuine beyond-text signal in this regime and reconciled with
our feature-level null in the discussion.

## Outputs

`artifacts/mac_mlx/phase_ext_david/`: generation under `generation/`,
probe tables and `david_reanalysis_report.json` under `analysis/`, figure
`david_reanalysis.png`.
