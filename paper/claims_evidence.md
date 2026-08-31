# Claims–evidence backbone (single source of truth for the paper)

Every sentence in the paper that makes a claim must trace to a row here.
Numbers are final (all runs complete as of 2026-08-28).

| # | Claim (paper wording draft) | Evidence artifact | Key numbers | Figure |
|---|---|---|---|---|
| C1 | Fixed-budget truncation probes measure budget fit (T_F), not prefix value (T_V) | A2 budget-sensitivity run: probes/sensitivity/budget_4096/ | Ministral solves from 16-token prefixes at 4096 budget; dev crossings dissolve | budget_falsification.png |
| C2 | Genuine prefix-locked value is rare: 1/178 cells | a5_labels_project/a5_full_cohort_labels.parquet + summary | 1 prefix-limited (gemma math_03159, T_V=896, adv 0.75); δ=0.25→7, δ=0.75→2; conservative agrees 178/178 | a5_project_regime_map.png |
| C3 | Restart dose–response separates compute-starved vs capability-limited failure | restart_baseline_wave3/*/restart_panel.parquet | Ministral 0→19→45→79% (0 non-monotone); Gemma 10–12% flat (5 wobbles); agreement 42/89 | wave3_restart_dose_response.png |
| C4 | Accumulated reasoning = compute compression, not reachability | terminal_stability_wave3 + labels (13 censored cells) | matched-compute advantage +11/13 (median +0.31); restart@8192 catches up 11/13 | wave3_matched_compute.png |
| C5 | A genuinely stochastic solvability band exists | wave3_topup pooled labels (148 enlarged cells) | 55 low / 55 mid (3–5 of 8) / 38 high; labels trimmed not inflated (−6/+1 events) | wave3_topup_resolution.png |
| C6 | Early internal signals add nothing beyond difficulty (pre-registered null) | confirmatory_early_signal/confirmatory_report.json | primary Δ+0.026 [−0.054,+0.167]; secondary Δ−0.090 [−0.213,+0.033]; level-free agrees | confirmatory_auroc.png |
| C6b | The null is not a forecast-point artifact (post-hoc sweep) | early_signal_posthoc_sweep/t_*/ | 8/10 deltas negative across t∈{128..2048}; all CIs straddle 0 | early_signal_prefix_sweep.png |
| C7 | Difficulty is the operative signal; its strength is model-dependent | same report, per-model descriptives | baseline 0.88 (secondary); Gemma 1.0 both endpoints; Ministral 0.63–0.73 | (table) |
| C8 | Timing prediction is not testable at this scale — events are too rare | horizon gate outputs | 21 interior events project-wide; 4 in test; gate: underpowered, not fit | (text) |
| C9 | 4-attempt labels are systematically optimistic; replication cuts both ways | A1 outputs (dev+wave3) + A5.1 summaries | A1 wave3: 13/16 confirmed (Ministral), 0/1 (Gemma); pooling: −6/+1 | wave3_label_hardening.png (appendix) |
| C10 | A question-only difficulty ceiling sits inside the published probe range, in their own regime (R1, competition math) | artifacts/external/difficulty_ceiling/report.json (estimator frozen in script docstring pre-run) | AUROC 0.873 [0.870, 0.876], 91,573 problems / 192,315 R1 generations; conservative (k=2 dominant, curation truncation); scoped as vulnerability claim + falsifiable prediction, not refutation | difficulty_ceiling_openr1.png |

Discipline instruments to cite in Methods/Appendix: frozen amendments with
dates and commit order; failed gates recorded (G1, G2, A5 pilot 7/8);
confirmatory amendment committed before evaluation (6672ee8 → adb1a02);
post-hoc sweep labeled as such in the amendment addendum.

Paper figure budget (main text): regime map, dose-response, matched-compute,
topup-resolution, confirmatory forest, prefix-sweep, budget-falsification.
Appendix: swimmers, label-hardening, A1 details, level-free variants.
| C11 | An early-window probe positive replicates and is dissected: all problem-identity, no detectable within-attempt signal, difficulty content decays by t=16-32 | artifacts/external/math128_distill7b/analysis_v2/math128_probe_report.json (frozen amendment 2026-08-31 + audit re-analysis) | t=4 pooled 0.849 [0.735,0.919]; within 0.496 [0.466,0.527] failure-w. / 0.515 [0.481,0.562] pair-w., chance under both at all 10 anchors; full-dump LOO ceiling 0.981; fidelity 0.906; pooled set n=1,024, within set n=5,632 | math128_probe_dissection.png |
| C12 | The difficulty ceiling is not a math artifact | artifacts/external/gpqa_ceiling/report.json (same frozen LOO estimator) | GPQA-diamond, Llama-3.3-70B (Singhi et al. dump): AUROC 0.917 [0.875, 0.938], 64 questions x 256 samples, pass rate 0.457; 4-way guessing floor attenuates the ceiling | (text; fig optional) |
