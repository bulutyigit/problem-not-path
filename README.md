# It's the Problem, Not the Path

**Budget and Difficulty Confounds in LLM Reasoning Trajectories**

Code, frozen protocols, and result artifacts for the paper (arXiv link to
follow). Yigit Utku Bulut, Johannes Kepler University Linz.

## TL;DR

Two widely believed trajectory-level phenomena largely dissolve under two
cheap controls:

- **"Breakthrough / aha moments"** measured by truncate-and-resample probes
  mostly mark the point where a solution starts to *fit the continuation
  budget*, not where the prefix accumulated value. The control is a
  from-scratch **restart baseline at matched total compute**: only 1 of 178
  problem×model cells survives it.
- **"A run's fate is legible in early hidden states"** (probe AUROCs of
  0.79–0.95 in the literature) is largely the probe reading *which problem
  it is*. The controls are a **question-only difficulty baseline** and a
  **within-problem evaluation**: a published early-window positive
  replicates exactly (pooled 0.849 at t=4) and sits at chance (0.496)
  within problem, at every anchor.

Positive findings survive the same instruments: restart dose–response
curves separate compute-starved from capability-limited failure
(0%→79% vs flat), a long prefix is worth roughly a budget multiple
(compression, not reachability), and about a third of ambiguous
intermediate states are genuinely stochastic (success 0.3–0.6 at n=8).

## Key results → where they live

| Result | Number | Produced by | Artifact |
|---|---|---|---|
| Final regime map, 178 cells | 1/178 prefix-limited | `scripts/label_a5_full_cohort.py` | `results/a5_full_cohort_labels.parquet` |
| Restart dose–response | 0→19→45→79% vs 10–12% flat | `scripts/probe_restart_baseline.py` | same labels file |
| Matched-compute prefix value | 11/13 wins, 11/13 caught up | `scripts/probe_terminal_stability.py` + labels | same |
| Stochastic solvability band | 55/148 cells at 3–5 of 8 | `scripts/probe_ambiguity_topup.py` | same |
| Pre-registered prediction null | ΔAUROC +0.026 / −0.090, CIs ∋ 0 | `scripts/run_confirmatory_early_signal_test.py` | `results/confirmatory_report.json` |
| Forecast-point sweep (post-hoc) | 8/10 deltas negative | same script, other t | `results/posthoc_sweep_t*.json` |
| Question-only ceiling, R1 dumps | AUROC 0.873 (192k gens) | `scripts/analyze_public_difficulty_ceiling.py` | `results/openr1_difficulty_ceiling.json` |
| Ceiling off-math, GPQA-diamond | AUROC 0.917 (16k samples) | `scripts/analyze_nonmath_ceiling.py` | `results/gpqa_difficulty_ceiling.json` |
| Probe dissection, R1-Distill-7B | pooled 0.849 vs within 0.496 | `scripts/verify_math128_dump.py` → `extract_math128_states.py` → `analyze_math128_probe.py` | `results/math128_probe_report.json` |

Paper figures are under `results/figures/`.

## Repository map

```
paper/                  LaTeX source of the paper
docs/protocol_amendments/   the frozen, dated decision rules (pre-registration)
docs/                   runbooks for the generation pipeline
reasonbench/            library: instrumented generation (MLX), probes,
                        censoring-aware labels, feature extraction, predictors
scripts/                every pipeline and analysis stage (see table above)
configs/                model and experiment configurations
tests/                  unit tests for labels, gates, and predictors
results/                small result artifacts backing every number in the paper
```

## Reproducing

Environment: Python 3.12, Apple silicon for the MLX generation stages
(`pip install -e .` from `pyproject.toml`; `uv sync` also works). The
public-data analyses need no GPU and no generation:

```
python scripts/analyze_public_difficulty_ceiling.py --output-dir out/ceiling
python scripts/verify_math128_dump.py           # 32,768-sample verification
python scripts/extract_math128_states.py ...    # teacher-forced forwards (Apple silicon)
python scripts/analyze_math128_probe.py ...     # the pooled-vs-within dissection
python scripts/analyze_nonmath_ceiling.py ...   # GPQA-diamond ceiling
python scripts/render_paper_figures.py ...      # regenerate every paper figure
pytest -q                                       # unit tests
```

Model checkpoints: `mlx-community/gemma-4-e4b-it-4bit` (of
`google/gemma-4-E4B-it`) and `mlx-community/Ministral-3-3B-Reasoning-2512-4bit`
(of `mistralai/Ministral-3-3B-Reasoning-2512`); the public-dump dissection
re-scores with `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`. Exact ids
also appear in `configs/experiments/`. Large artifacts: the teacher-forced
hidden-state tensors (several GB) are not shipped — `extract_math128_states.py`
regenerates them deterministically from the public dump; the shipped
`results/` parquet/JSON files are small and carry every number cited in the
paper.

The instrumented-cohort pipeline (generation → probes → restart panels →
labels → confirmatory test) is documented step by step in
`docs/phase04c_expansion_runbook.md` and `docs/phase05_execution.md`;
every stage is resumable and writes atomic completion markers.

## Pre-registration

Every decision rule was frozen in a dated amendment under
`docs/protocol_amendments/` before the outcomes it governs were observed —
including the gates that failed and how each was resolved. The paper's
Appendix A condenses the timeline. The internal working repository, whose
commit history timestamps each freeze against each result, is available to
reviewers on request.

## Data

All external data is public: [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
the [math128 R1-Distill-7B dump](https://huggingface.co/datasets/nishadsinghi/math128_solutions_r1_distill_qwen_7b_32K_tokens)
and [GPQA-diamond Llama-3.3-70B dump](https://huggingface.co/datasets/sc-genrm-scaling/GPQA_diamond_Solutions_Llama-3.3-70B-Instruct)
released by Singhi et al. (COLM 2025), the
[MATH](https://github.com/hendrycks/math) benchmark, and
[GPQA](https://github.com/idavidrein/gpqa). We thank the dump authors for
releasing complete, unfiltered samples — that practice is what made the
within-problem control possible.

## License and citation

MIT (see `LICENSE`). If you use this work:

```bibtex
@article{bulut2026problemnotpath,
  title  = {It's the Problem, Not the Path: Budget and Difficulty
            Confounds in LLM Reasoning Trajectories},
  author = {Bulut, Yigit Utku},
  year   = {2026},
  note   = {arXiv, to appear}
}
```
