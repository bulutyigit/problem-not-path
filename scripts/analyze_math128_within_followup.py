#!/usr/bin/env python
"""POST-HOC follow-up (2026-09-02): per-problem detail, cross-anchor consistency, within-problem permutation null."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
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
S = full.groupby("problem").correct.sum(); band = S[(S >= 14) & (S <= 242)].index.astype(int).tolist()
idx = index[index.problem.isin(band)].reset_index(drop=True)
CHECK = (4, 32, 64, 128)
states = {}
for i, r in idx.iterrows():
    with np.load(root / "states" / f"p{r.problem:03d}_s{r['sample']:03d}.npz") as z:
        states[i] = {t: z[f"anchor_{t}"].astype(np.float32) for t in CHECK if f"anchor_{t}" in z}

def pipe(n, d, k, seed=20260831):
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(k, d, n-1), random_state=seed)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

def oracle(X, y, g, seed=20260831):
    recs = []
    for p in np.unique(g):
        m = g == p; Xp, yp = X[m], y[m]
        if min((yp == 0).sum(), (yp == 1).sum()) < 5: continue
        oof = np.full(len(yp), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xp, yp):
            oof[te] = pipe(len(tr), Xp.shape[1], 32).fit(Xp[tr], yp[tr]).predict_proba(Xp[te])[:, 1]
        recs.append((int(p), roc_auc_score(yp, oof), int((yp == 0).sum()), int((yp == 1).sum())))
    return pd.DataFrame(recs, columns=["problem", "auroc", "fails", "succ"])

def centered_transfer(X, y, g, seed=20260831):
    Xc = X.copy()
    for p in np.unique(g):
        m = g == p; Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xc, y, g):
        oof[te] = pipe(len(tr), Xc.shape[1], 128).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
    recs = []
    for p in np.unique(g):
        m = g == p
        recs.append((int(p), roc_auc_score(y[m], oof[m]), int((y[m] == 0).sum()), int((y[m] == 1).sum())))
    return pd.DataFrame(recs, columns=["problem", "auroc", "fails", "succ"])

def fw(d): return float(np.average(d.auroc, weights=d.fails))
rng = np.random.default_rng(7)
out = {}
per_B = {}; per_A = {}
for t in CHECK:
    use = idx[idx.max_anchor >= t]; rows = use.index.to_numpy()
    X = np.stack([states[i][t] for i in rows]); y = use.correct.astype(int).to_numpy(); g = use.problem.to_numpy()
    per_A[t] = centered_transfer(X, y, g); per_B[t] = oracle(X, y, g)
    # permutation null: shuffle labels WITHIN each problem (keeps per-problem rates), 12 perms
    nullB, nullBmax, nullA = [], [], []
    for k in range(12):
        yp = y.copy()
        for p in np.unique(g):
            m = np.where(g == p)[0]; yp[m] = rng.permutation(yp[m])
        b = oracle(X, yp, g, seed=100 + k); nullB.append(fw(b)); nullBmax.append(float(b.auroc.max()))
        nullA.append(fw(centered_transfer(X, yp, g, seed=100 + k)))
    out[t] = {"A_fw": round(fw(per_A[t]), 4), "A_null_fw": [round(min(nullA), 3), round(max(nullA), 3)],
              "B_fw": round(fw(per_B[t]), 4), "B_null_fw": [round(min(nullB), 3), round(max(nullB), 3)],
              "B_max": round(float(per_B[t].auroc.max()), 3), "B_null_max": [round(min(nullBmax), 3), round(max(nullBmax), 3)],
              "B_above_0.6": int((per_B[t].auroc > 0.6).sum()),
              "B_null_above_0.6_max": None}
    print(f"t={t:>3} | A fw {out[t]['A_fw']:.3f} (null range {out[t]['A_null_fw']}) | "
          f"B fw {out[t]['B_fw']:.3f} (null {out[t]['B_null_fw']}) | B max {out[t]['B_max']} (null max range {out[t]['B_null_max']})", flush=True)

print("\n=== per-problem oracle (B) AUROC by anchor, with fails/succ ===")
tab = per_B[4][["problem", "fails", "succ"]].copy()
for t in CHECK: tab = tab.merge(per_B[t][["problem", "auroc"]].rename(columns={"auroc": f"B{t}"}), on="problem", how="left")
for t in CHECK: tab = tab.merge(per_A[t][["problem", "auroc"]].rename(columns={"auroc": f"A{t}"}), on="problem", how="left")
print(tab.round(3).sort_values("B32", ascending=False).to_string(index=False))
print("\ncross-anchor consistency of per-problem oracle AUROC (Spearman):")
for a, b in ((32, 64), (64, 128), (32, 128), (4, 32)):
    m = per_B[a].merge(per_B[b], on="problem"); print(f"  B{a} vs B{b}: rho={spearmanr(m.auroc_x, m.auroc_y).correlation:.2f}")
m = per_A[32].merge(per_B[32], on="problem"); print(f"  A32 vs B32: rho={spearmanr(m.auroc_x, m.auroc_y).correlation:.2f}")
Path("artifacts/external/math128_distill7b/analysis_v2/within_robustness_followup.json").write_text(json.dumps(out, indent=1))
