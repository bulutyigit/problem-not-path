# Phase 4c cohort expansion — runbook

Amendment: `protocol_amendments/2026-08-20-phase-04c-cohort-expansion.md`
Run everything from the project root, one command at a time (each run loads a
model; never run two generations in parallel on the Mac). Every step is
resume-safe: rerun the same command after an interruption. Every command below
is complete and copy-pasteable on its own.

> Convention (fixed 2026-08-20): `generate.py` does not read
> `output_subdirectory`; every generation command passes a per-model
> `--output-dir` explicitly, matching the paths the selection step expects.

## Gate G1 — budget falsification (A2), required before Wave 3 spend

```bash
./.venv/bin/python -u scripts/probe_budget_sensitivity.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04b/generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_probe_manifest.json --probe-dir artifacts/mac_mlx/phase_04c/probes/models/ministral3 --output-dir artifacts/mac_mlx/phase_04c/probes/sensitivity/budget_4096/ministral3 --resume
```

```bash
./.venv/bin/python -u scripts/probe_budget_sensitivity.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04b/generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_probe_manifest.json --probe-dir artifacts/mac_mlx/phase_04c/probes/models/gemma4 --output-dir artifacts/mac_mlx/phase_04c/probes/sensitivity/budget_4096/gemma4 --resume
```

Readout: `budget_sensitivity_summary.json` → `first_crossing_shifts`; decision
rule frozen in the 2026-08-19 amendment.

## Wave 1 — probe the frozen supplement cohort

```bash
./.venv/bin/python -u scripts/generate_breakthrough_probes.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04b/generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_supplement_manifest.json --output-dir artifacts/mac_mlx/phase_04c/probes/supplement/ministral3 --resume
```

```bash
./.venv/bin/python -u scripts/generate_breakthrough_probes.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04b/generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_supplement_manifest.json --output-dir artifacts/mac_mlx/phase_04c/probes/supplement/gemma4 --resume
```

Gate G2: ≥ 6 new interior events across the 40 supplement trajectories.

## Wave 2 — seed-12 replication (assets built 2026-08-20)

Bundle: `artifacts/mac_mlx/phase_04c/expansion/wave2_datasets/` · configs:
`configs/experiments/phase_04b_{gemma4,ministral3}_mlx_4bit_wave2_seed12.yaml`

Step 1 — base generation (16K, instrumented), one model at a time:

```bash
./.venv/bin/python -u scripts/generate.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04b_ministral3_mlx_4bit_wave2_seed12.yaml --datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave2_datasets --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --output-dir artifacts/mac_mlx/phase_04c/expansion/wave2_generation/ministral3_mlx_4bit_wave2_seed12 --resume
```

```bash
./.venv/bin/python -u scripts/generate.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04b_gemma4_mlx_4bit_wave2_seed12.yaml --datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave2_datasets --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --output-dir artifacts/mac_mlx/phase_04c/expansion/wave2_generation/gemma4_mlx_4bit_wave2_seed12 --resume
```

Step 2 — probe manifest for the new trajectories (screen inherited from A3):

```bash
./.venv/bin/python scripts/build_breakthrough_supplement_manifest.py --generation-dir artifacts/mac_mlx/phase_04c/expansion/wave2_generation --development-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_probe_manifest.json --preselected-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_supplement_manifest.json --base-seed 12 --output artifacts/mac_mlx/phase_04c/manifests/breakthrough_wave2_manifest.json
```

Step 3 — probes:

```bash
./.venv/bin/python -u scripts/generate_breakthrough_probes.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04c/expansion/wave2_generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_wave2_manifest.json --output-dir artifacts/mac_mlx/phase_04c/probes/wave2/ministral3 --resume
```

```bash
./.venv/bin/python -u scripts/generate_breakthrough_probes.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04c/expansion/wave2_generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_wave2_manifest.json --output-dir artifacts/mac_mlx/phase_04c/probes/wave2/gemma4 --resume
```

## Wave 3 — fresh problems (run after G1 and G2 pass)

Step 1 — screening bundle + configs (downloads the pinned MATH revision):

```bash
./.venv/bin/python scripts/prepare_expansion_screen.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --source-datasets-dir artifacts/mac_mlx/shared/datasets_v2 --exclude-bundle artifacts/mac_mlx/shared/datasets_v2/high_difficulty_challenge_50.jsonl --output-datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_datasets --configs-dir configs/experiments
```

Step 2 — screen generation (3 seeds × 3,072 tokens, no hidden states; hours —
consider the RunPod BF16 profile here, see amendment):

```bash
./.venv/bin/python -u scripts/generate.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04b_ministral3_mlx_4bit_wave3_screen.yaml --datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_datasets --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --output-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_generation/ministral3_mlx_4bit_wave3_screen --resume
```

```bash
./.venv/bin/python -u scripts/generate.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04b_gemma4_mlx_4bit_wave3_screen.yaml --datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_datasets --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --output-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_generation/gemma4_mlx_4bit_wave3_screen --resume
```

Step 3 — freeze the cohort (rule: 1–5 successes of 6) and emit base-gen
configs (adjust the two `--screen-generation-dir` paths if the generation
subdirectories differ; they are the `output_subdirectory` values of the two
screen configs):

```bash
./.venv/bin/python scripts/select_expansion_cohort.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --screen-generation-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_generation/ministral3_mlx_4bit_wave3_screen --screen-generation-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_generation/gemma4_mlx_4bit_wave3_screen --screen-datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave3_screen_datasets --output-datasets-dir artifacts/mac_mlx/phase_04c/expansion/wave3_datasets --configs-dir configs/experiments --output-cohort artifacts/mac_mlx/phase_04c/manifests/wave3_expansion_cohort.json
```

Step 4 — base generation with the emitted `*_wave3_expansion.yaml` configs
against `wave3_datasets` (same `generate.py` pattern as Wave 2, output dir
`artifacts/mac_mlx/phase_04c/expansion/wave3_generation`), then the probe
manifest via `--preselected-manifest artifacts/mac_mlx/phase_04c/manifests/wave3_expansion_cohort.json`
against that generation dir, then probes into `probes/wave3/{model}` — the
same three-command pattern as Wave 2, with wave3 paths.

## After each wave

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py validate-development-probes
```

```bash
./.venv/bin/python -u scripts/run_phase05_breakthrough.py build-development-tables
```

(Validation currently reads the canonical dev dirs; merging supplement/wave
labels into the development tables is the next pipeline change once Wave 1
lands.)

## Wave 3 — step 5: A5.1 ambiguity enlargement (after probes, before labeling)

Any probe cell with 1–3 successes of 4 is enlarged to 8 attempts and labels
are re-derived from pooled counts (`scripts/probe_ambiguity_topup.py`;
`--plan-only` previews cost without loading the model). Canonical probe
dirs stay untouched; pooled summaries land under `probes/wave3_topup/<model>`.

```bash
./.venv/bin/python -u scripts/probe_ambiguity_topup.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_ministral3_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04c/expansion/wave3_generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_wave3_manifest.json --probe-dir artifacts/mac_mlx/phase_04c/probes/wave3/ministral3 --output-dir artifacts/mac_mlx/phase_04c/probes/wave3_topup/ministral3 --resume
```

```bash
./.venv/bin/python -u scripts/probe_ambiguity_topup.py --project-root /Users/bulutyigit/Documents/pycharm_projects/how_models_reason --config configs/experiments/phase_04c_gemma4_mlx_4bit_25k.yaml --readiness-manifest artifacts/mac_mlx/phase_04b/preflight/smoke/model_readiness.json --base-run-dir artifacts/mac_mlx/phase_04c/expansion/wave3_generation --probe-manifest artifacts/mac_mlx/phase_04c/manifests/breakthrough_wave3_manifest.json --probe-dir artifacts/mac_mlx/phase_04c/probes/wave3/gemma4 --output-dir artifacts/mac_mlx/phase_04c/probes/wave3_topup/gemma4 --resume
```
