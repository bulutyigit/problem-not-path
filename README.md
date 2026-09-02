# It's the Problem, Not the Path

**Budget and Difficulty Confounds in LLM Reasoning Trajectories**

This is the code, the frozen protocols, and the result artifacts behind the
paper. I'm Yigit Utku Bulut, a master's student at Johannes Kepler
University Linz; I did this work independently. A preprint link will
appear here as soon as it is live.

If you are short on time: the paper's every number traces to a file in
`results/`, produced by a script in `scripts/`, under a rule frozen in
`docs/protocol_amendments/` before the outcome was known. The table
further down maps them one to one.

## What the paper is about

Two things get said about the reasoning traces of language models. One is
that they contain *breakthrough moments* — points where the accumulated
reasoning suddenly unlocks the answer. The other is that a run's fate is
*legible early* — that a probe on the hidden states a few tokens in can
already tell you whether the attempt will succeed.

Both claims are usually measured without a control that, once you see it,
is hard to unsee. For breakthroughs: nobody checks whether the same
solution could have been reached by simply starting over with the same
total token budget. For early signals: nobody checks whether the probe is
just recognising *which problem* it is looking at, rather than how this
particular attempt is going. Problem difficulty predicts outcomes very
well on its own, and a probe evaluated across problems gets credit for
reading it.

I built both controls — a from-scratch restart baseline at matched token
budget, and a within-problem evaluation on top of a question-only
difficulty baseline — and ran them on two small open models, then on
public data at much larger scale.

## What I found, in plain words

**Most "breakthroughs" are budget artifacts.** Across 178 problem–model
cells (89 MATH problems, two models, 16K-token instrumented traces),
exactly one survives as a genuine prefix-limited event once the restart
control is applied. The rest mark the point where a solution starts to
*fit the continuation budget*, not where the prefix earned something.

**The two models fail differently.** For Ministral-3 3B, problems it
cannot solve from scratch at 1,024 tokens dissolve as the budget grows
(0% → 79% solved by 8,192 tokens): it is compute-starved. For Gemma-4
E4B the curve stays flat at 10–12%: within the budgets I could measure,
more tokens do not help.

**Long reasoning is mostly compression, not new reach.** Where the
comparison is exactly matched, continuing the model's own prefix beats
restarting in all nine cases. But an 8,192-token restart reaches the same
threshold in 11 of 13 cells anyway. Accumulated reasoning buys the same
successes cheaper; within the measured range it rarely buys successes a
fresh start could not.

**Early signals add nothing I could detect beyond difficulty.** In a
pre-registered, single-shot test, adding fifteen early-window dynamics
features to a question-only baseline moved AUROC by +0.026 (95% CI −0.054
to +0.167) — an absence of detected gain, not proof of equivalence. Then,
on public DeepSeek-R1 data, a trace-blind proxy (the pass rate of the
problem's *other* attempts) reaches AUROC 0.873, squarely inside the range
published probes report. And when I rebuilt the closest published
early-window probe on a public dump with 256 samples per problem, it
recovered the published number pooled across problems (0.849) and sat at
chance within problem (0.496 at t=4, indistinguishable from 0.5 at all
ten anchors).

**There is a little within-attempt signal — just not where the positives
look.** A post-hoc check I ran after all frozen endpoints were reported
found a small average residual that appears only from about 32 tokens in,
concentrated in three easy problems where failures are rare. In two of
them the failing attempts are four to six times shorter than the
successes: the model answering fast and wrong, visible from the first
tokens. The common failures on hard problems are not legible at all.

## Where each number comes from

| Result | Number | Produced by | Artifact |
|---|---|---|---|
| Final regime map, 178 cells | 1/178 prefix-limited | `scripts/label_a5_full_cohort.py` | `results/a5_full_cohort_labels.parquet` |
| Restart dose–response | 0→19→45→79% vs 10–12% flat | `scripts/probe_restart_baseline.py` | same labels file |
| Matched-budget prefix value | 9/9 exact-match wins (boundary proxies: 2 wins, 2 ties); restart@8192 reaches τ 11/13 | `scripts/probe_terminal_stability.py` + labels | same |
| Intermediate-solvability band | 55/148 cells at 3–5 of 8 | `scripts/probe_ambiguity_topup.py` | same |
| Pre-registered prediction null | ΔAUROC +0.026 / −0.090, CIs ∋ 0 | `scripts/run_confirmatory_early_signal_test.py` | `results/confirmatory_report.json` |
| Forecast-point sweep (post-hoc) | 8/10 deltas negative | same script, other t | `results/posthoc_sweep_t*.json` |
| Trace-blind difficulty ceiling, R1 dumps | AUROC 0.873 (192k gens) | `scripts/analyze_public_difficulty_ceiling.py` | `results/openr1_difficulty_ceiling.json` |
| Ceiling off-math, GPQA-diamond | AUROC 0.917 (16k samples) | `scripts/analyze_nonmath_ceiling.py` | `results/gpqa_difficulty_ceiling.json` |
| Probe dissection, R1-Distill-7B | pooled 0.849 vs within 0.496 | `scripts/verify_math128_dump.py` → `extract_math128_states.py` → `analyze_math128_probe.py` | `results/math128_probe_report.json` |
| Within-problem robustness (post-hoc) | centered probe 0.49 at t=4 → 0.56 at t=32; 3/22 problems separable | `scripts/analyze_math128_within_robustness.py`, `..._followup.py` | `results/within_robustness_posthoc.json`, `..._followup.json` |

The paper's figures are in `results/figures/`; `scripts/render_paper_figures.py`
regenerates all of them from the frozen artifacts.

## What is in here

```
docs/protocol_amendments/   the dated decision rules, frozen before outcomes
docs/claims_evidence.md     every claim in the paper, the artifact behind
                            it, and its numbers
docs/                       runbooks for the generation pipeline
reasonbench/                the library: instrumented generation (MLX),
                            probes, censoring-aware labels, features, predictors
scripts/                    every pipeline and analysis stage
configs/                    model and experiment configurations
tests/                      unit tests for labels, gates, and predictors
results/                    the small artifacts that back every number
```

## Running it yourself

You need Python 3.12; the generation stages run on Apple silicon through
MLX (`pip install -e .` from `pyproject.toml`, or `uv sync`). The
public-data analyses are the cheap part — no GPU, no generation:

```
python scripts/analyze_public_difficulty_ceiling.py --output-dir out/ceiling
python scripts/verify_math128_dump.py           # verifies all 32,768 samples on CPU
python scripts/extract_math128_states.py ...    # teacher-forced forwards (Apple silicon)
python scripts/analyze_math128_probe.py ...     # the pooled-vs-within dissection
python scripts/analyze_nonmath_ceiling.py ...   # the GPQA-diamond ceiling
python scripts/render_paper_figures.py ...      # every paper figure
pytest -q                                       # the test suite
```

The models are `mlx-community/gemma-4-e4b-it-4bit` (of `google/gemma-4-E4B-it`)
and `mlx-community/Ministral-3-3B-Reasoning-2512-4bit` (of
`mistralai/Ministral-3-3B-Reasoning-2512`); the public-dump dissection
re-scores through `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`. Exact
ids are in `configs/experiments/`. One thing I could not ship: the
teacher-forced hidden-state tensors run to several gigabytes.
`extract_math128_states.py` regenerates them deterministically from the
public dump; everything in `results/` is small and carries the numbers
the paper cites.

The full instrumented-cohort pipeline — generation, probes, restart
panels, labels, the confirmatory test — is written up step by step in
`docs/phase04c_expansion_runbook.md` and `docs/phase05_execution.md`.
Every stage resumes from where it stopped and writes an atomic
completion marker, because I ran all of it on one Mac over several weeks.

## On pre-registration

Every decision rule in this study was written down, dated, and committed
before the outcome it governs was observed. That includes the rules that
turned out badly: two cohort gates and one pilot gate failed, and each
failure is recorded as a failure and resolved by a new amendment, never by
relabeling. Anything I added after seeing results is labeled post-hoc in
the amendment that reports it and in the paper. The paper's Appendix A
condenses the timeline; the documents themselves are in
`docs/protocol_amendments/`. The private working repository, whose commit
history timestamps each freeze against each result, is available to
reviewers on request.

## Data

Everything external is public. The difficulty ceiling uses
[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k).
The dissection uses a
[math128 R1-Distill-7B dump](https://huggingface.co/datasets/nishadsinghi/math128_solutions_r1_distill_qwen_7b_32K_tokens)
released by the first author of Singhi et al. (COLM 2025) alongside that
paper's data — the paper itself does not describe it — and the
[GPQA-diamond Llama-3.3-70B dump](https://huggingface.co/datasets/sc-genrm-scaling/GPQA_diamond_Solutions_Llama-3.3-70B-Instruct)
from the same group. Problems come from [MATH](https://github.com/hendrycks/math)
and [GPQA](https://github.com/idavidrein/gpqa).

I owe the dump authors a real thank-you: keeping every attempt, failures
included, at 256 samples per problem is exactly what makes a within-problem
control possible. If you release rollouts, please release them unfiltered.

## A few honest caveats

Two small models in 4-bit quantization, one benchmark family, one base
seed per cell. Budget matching counts generated tokens, not FLOPs or
latency. The dissection re-scores full-precision generations through a
4-bit model (top-1 fidelity 0.906, reported) and rests on one public dump.
The confirmatory null is an absence of detected gain — its interval
leaves moderate effects open — and the within-problem dissection is the
sharper evidence. The natural next experiment is the same frozen
instrument on a large RL-trained reasoner; that is future work, and the
apparatus here is built to run it.

## License and citation

MIT (see `LICENSE`). If this is useful to you:

```bibtex
@article{bulut2026problemnotpath,
  title  = {It's the Problem, Not the Path: Budget and Difficulty
            Confounds in LLM Reasoning Trajectories},
  author = {Bulut, Yigit Utku},
  year   = {2026},
  note   = {Preprint; DOI to follow}
}
```

Questions, disagreements, or a dump I should run this on — open an issue.
