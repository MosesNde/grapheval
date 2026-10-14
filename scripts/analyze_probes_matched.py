#!/usr/bin/env python3
"""
analyze_probes_matched.py — the matched-probe comparison the residual test missed.

The nonlinear probe on the residual returned 0.5655, not chance. That breaks the
reading of analyze_residual.py, and for a reason worth stating plainly:
subtracting a ridge fit removes LINEAR PREDICTABILITY from z_0, not INFORMATION.
Label information that z_0 carries nonlinearly survives into r, so AUC(r) > 0.5
does not show the graph contributes anything new. The residual is the wrong
instrument for the nonlinear case.

The quantity that actually answers the question is conditional and needs no
residual at all: under a probe class held fixed, how much better does the graph
representation predict the label than the edge-free one?

    delta(probe) = AUC(probe; z_G) - AUC(probe; z_0)

Computed for a linear probe and an MLP, which is the cell the earlier run left
empty -- it reported MLP on z_G and on r but never on z_0, so there was nothing
to compare against. Reads cached representations; no GPU, no retraining.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "repo"
d = np.load(ROOT / "runs" / f"pooled_reps_{MODE}.npz")
zg_tr, z0_tr, y_tr = d["zg_tr"], d["z0_tr"], d["y_tr"]
zg_te, z0_te, y_te = d["zg_te"], d["z0_te"], d["y_te"]

def run(name, Xtr, Xte, kind, seed=0):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if kind == "linear":
        clf = LogisticRegression(max_iter=3000, C=1.0)
    elif kind == "gbm":
        # low-variance nonlinear probe: the MLP never beat logistic regression
        # on either input, so the null must not rest on it alone
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=seed, n_jobs=-1, verbose=-1)
        clf.fit(A, y_tr)
        return float(roc_auc_score(y_te, clf.predict_proba(B)[:, 1]))
    else:
        # generous budget: the earlier MLP scored BELOW the linear probe on the
        # same input, which means it was underfitting, not measuring capacity
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                            random_state=seed, early_stopping=True,
                            n_iter_no_change=25, learning_rate_init=1e-3)
    clf.fit(A, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(B)[:, 1]))

out = {}
print(f"{'probe':>10s}{'z_0 (no edges)':>18s}{'z_G (graph)':>14s}{'delta':>10s}")
for kind in ("linear", "mlp", "gbm"):
    if kind == "linear":
        a0, ag = run("z0", z0_tr, z0_te, kind), run("zG", zg_tr, zg_te, kind)
        s0 = sg = 0.0
    elif kind == "gbm":
        v0 = [run("z0", z0_tr, z0_te, kind, s) for s in (0, 1, 2)]
        vg = [run("zG", zg_tr, zg_te, kind, s) for s in (0, 1, 2)]
        a0, s0 = float(np.mean(v0)), float(np.std(v0, ddof=1))
        ag, sg = float(np.mean(vg)), float(np.std(vg, ddof=1))
    else:
        v0 = [run("z0", z0_tr, z0_te, kind, s) for s in (0, 1, 2)]
        vg = [run("zG", zg_tr, zg_te, kind, s) for s in (0, 1, 2)]
        a0, s0 = float(np.mean(v0)), float(np.std(v0, ddof=1))
        ag, sg = float(np.mean(vg)), float(np.std(vg, ddof=1))
    out[kind] = {"z0": round(a0, 4), "zG": round(ag, 4),
                 "delta": round(ag - a0, 4),
                 "z0_sd": round(s0, 4), "zG_sd": round(sg, 4)}
    print(f"{kind:>10s}{a0:>18.4f}{ag:>14.4f}{ag-a0:>+10.4f}")

p = ROOT / "runs" / f"probes_matched_{MODE}.json"
p.write_text(json.dumps(out, indent=2))
print(f"\n-> {p}")
