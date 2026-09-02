#!/usr/bin/env python
"""POST-HOC robustness check (2026-09-02, after all frozen endpoints were reported)."""
A: problem-centered features, problem-disjoint folds (transferable within-attempt direction?)
B: per-problem oracle, sample-disjoint folds inside each problem (any linear within signal at all?)
C: uncentered re-run (must reproduce the frozen battery's within values)."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

root = Path("artifacts/external/math128_distill7b")
index = pd.read_parquet(root / "extraction_index.parquet")
full = pd.read_parquet(root / "sample_correctness.parquet")
S = full.groupby("problem").correct.sum()
band = S[(S >= 14) & (S <= 242)].index.astype(int).tolist()
assert len(band) == 22, len(band)
idx = index[index.problem.isin(band)].reset_index(drop=True)
print("within rows:", len(idx), "| problems:", idx.problem.nunique(), file=sys.stderr)
ANCHORS = (4, 8, 16, 32, 64, 128, 192, 256, 384, 512)
states = {}
for i, r in idx.iterrows():
    with np.load(root / "states" / f"p{r.problem:03d}_s{r['sample']:03d}.npz") as z:
        states[i] = {t: z[f"anchor_{t}"].astype(np.float32) for t in ANCHORS if f"anchor_{t}" in z}
rng = np.random.default_rng(20260901)

def pipe(n, d, k=128, seed=20260831):
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=min(k, d, n - 1), random_state=seed)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

def per_problem_auroc(frame, col):
    rows = []
    for p, g in frame.groupby("problem"):
        y = g.correct.astype(int).to_numpy(); s = g[col].to_numpy()
        if y.min() == y.max(): continue
        rows.append((p, roc_auc_score(y, s), int((y == 0).sum()), int((y == 1).sum())))
    return pd.DataFrame(rows, columns=["problem", "auroc", "fails", "succ"])

def summarize(per, draws=1000):
    def agg(d):
        fw = np.average(d.auroc, weights=d.fails); pw = np.average(d.auroc, weights=d.fails * d.succ)
        return fw, pw
    fw, pw = agg(per)
    boots = np.array([agg(per.sample(len(per), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(draws)])
    return {"n_problems": int(len(per)), "failure_weighted": round(float(fw), 4),
            "fw_ci": [round(float(x), 4) for x in np.percentile(boots[:, 0], [2.5, 97.5])],
            "pair_weighted": round(float(pw), 4),
            "pw_ci": [round(float(x), 4) for x in np.percentile(boots[:, 1], [2.5, 97.5])]}

report = {}
for t in ANCHORS:
    use = idx[idx.max_anchor >= t].copy()
    rows = use.index.to_numpy()
    X = np.stack([states[i][t] for i in rows]); y = use.correct.astype(int).to_numpy(); g = use.problem.to_numpy()
    # per-problem centering (label-free)
    Xc = X.copy()
    for p in np.unique(g):
        m = g == p; Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
    out = {}
    for name, feats in (("C_uncentered", X), ("A_centered", Xc)):
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=20260831).split(feats, y, g):
            mdl = pipe(len(tr), feats.shape[1]).fit(feats[tr], y[tr])
            oof[te] = mdl.predict_proba(feats[te])[:, 1]
        use[name] = oof
        out[name] = summarize(per_problem_auroc(use, name))
    # B: per-problem oracle, sample-disjoint folds inside each problem
    recs = []
    for p in np.unique(g):
        m = g == p; Xp, yp = X[m], y[m]
        if yp.min() == yp.max() or min((yp == 0).sum(), (yp == 1).sum()) < 5: continue
        oof = np.full(len(yp), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260831).split(Xp, yp):
            mdl = pipe(len(tr), Xp.shape[1], k=32).fit(Xp[tr], yp[tr])
            oof[te] = mdl.predict_proba(Xp[te])[:, 1]
        recs.append((p, roc_auc_score(yp, oof), int((yp == 0).sum()), int((yp == 1).sum())))
    per_b = pd.DataFrame(recs, columns=["problem", "auroc", "fails", "succ"])
    out["B_per_problem_oracle"] = summarize(per_b)
    out["B_per_problem_oracle"]["max_single_problem_auroc"] = round(float(per_b.auroc.max()), 3)
    out["B_per_problem_oracle"]["problems_above_0.6"] = int((per_b.auroc > 0.6).sum())
    report[t] = out
    print(f"t={t:>3}: C {out['C_uncentered']['failure_weighted']:.3f} {out['C_uncentered']['fw_ci']} | "
          f"A {out['A_centered']['failure_weighted']:.3f} {out['A_centered']['fw_ci']} pw {out['A_centered']['pair_weighted']:.3f} | "
          f"B {out['B_per_problem_oracle']['failure_weighted']:.3f} {out['B_per_problem_oracle']['fw_ci']} "
          f"max {out['B_per_problem_oracle']['max_single_problem_auroc']} >0.6: {out['B_per_problem_oracle']['problems_above_0.6']}", flush=True)
Path("artifacts/external/math128_distill7b/analysis_v2/within_robustness_posthoc.json").write_text(json.dumps(report, indent=1))
