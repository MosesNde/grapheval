#!/usr/bin/env python3
"""
analyze_residual_robust.py — hardening the residual result against two objections.

analyze_residual.py found the graph's unique component r large (45% of the
representation's norm) and uninformative (linear probe AUC 0.494). Two things a
reviewer will push on:

 1. The ridge solve warned of ill-conditioning at alpha=1. Sweep alpha. Note the
    direction of the risk is benign: under-regularising can only move MORE of
    z_G into the fitted part, and over-regularising leaves MORE signal in r. A
    residual that stays at chance across the sweep cannot be an artifact of the
    solve.

 2. The probe on r is linear. If the graph's contribution were encoded
    nonlinearly, a linear probe would miss it. So probe r with an MLP as well.

Reuses the saved checkpoint; no retraining. Also writes the pooled
representations to .npz so later analyses need no GPU at all.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode
from analyze_residual import pooled, probe
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys as _sys
MODE = _sys.argv[1] if len(_sys.argv) > 1 else "repo"
cfg = dict(T.CONFIG); cfg.update(batch_size=256, num_workers=6, pin_memory=True)
dev = T.get_device(); T.set_seed(cfg["seed"])
corpus, fams, splits = load_split(MODE)
G, _ = encode(corpus, splits, fams, cfg, dev, Path("/home/moses/graph_cache_cwe"))
vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
for gs in G.values():
    for g in gs:
        g.label_id = g.label_id.clamp_(0, vocab - 1)
        g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
        g.y = torch.tensor(int(g.y.item() > 0))

npz = ROOT / "runs" / f"pooled_reps_{MODE}.npz"
if npz.exists():
    d = np.load(npz)
    zg_tr, z0_tr, y_tr = d["zg_tr"], d["z0_tr"], d["y_tr"]
    zg_te, z0_te, y_te = d["zg_te"], d["z0_te"], d["y_te"]
    print(f"[cache] {npz}")
else:
    m = T.build_model("gat", G["train"][0].x.shape[1], vocab, cfg,
                      T.MAX_RELATIONS, 2).to(dev)
    m.load_state_dict(torch.load(ROOT / "runs" / f"mech_gat_{MODE}.pt"))
    m.eval()
    zg_tr, z0_tr, y_tr = pooled(m, G["train"], dev)
    zg_te, z0_te, y_te = pooled(m, G["test"], dev)
    np.savez_compressed(npz, zg_tr=zg_tr, z0_tr=z0_tr, y_tr=y_tr,
                        zg_te=zg_te, z0_te=z0_te, y_te=y_te)
    print(f"[saved] {npz}")

print(f"\nRIDGE ALPHA SWEEP  (residual probe should stay at chance)")
print(f"{'alpha':>10s}{'R^2':>10s}{'||r||/||z_G||':>16s}{'AUC(r) linear':>16s}")
rows = {}
for al in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    W = Ridge(alpha=al).fit(z0_tr, zg_tr)
    r_tr, r_te = zg_tr - W.predict(z0_tr), zg_te - W.predict(z0_te)
    r2 = float(1 - ((zg_te - W.predict(z0_te)) ** 2).sum() /
               ((zg_te - zg_tr.mean(0)) ** 2).sum())
    fn = float(np.linalg.norm(r_te) / np.linalg.norm(zg_te))
    au = probe(r_tr, y_tr, r_te, y_te)
    rows[str(al)] = {"r2": round(r2, 4), "resid_frac": round(fn, 4),
                     "auc_linear": round(au, 4)}
    print(f"{al:10.1f}{r2:10.4f}{fn:16.4f}{au:16.4f}")

# nonlinear probe on the residual at the default alpha
W = Ridge(alpha=1.0).fit(z0_tr, zg_tr)
r_tr, r_te = zg_tr - W.predict(z0_tr), zg_te - W.predict(z0_te)
sc = StandardScaler().fit(r_tr)
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=120, random_state=0,
                    early_stopping=True)
mlp.fit(sc.transform(r_tr), y_tr)
auc_mlp = float(roc_auc_score(y_te, mlp.predict_proba(sc.transform(r_te))[:, 1]))

sc2 = StandardScaler().fit(zg_tr)
mlp2 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=120, random_state=0,
                     early_stopping=True)
mlp2.fit(sc2.transform(zg_tr), y_tr)
auc_mlp_zg = float(roc_auc_score(y_te, mlp2.predict_proba(sc2.transform(zg_te))[:, 1]))

print(f"\nNONLINEAR PROBE (MLP, one hidden layer of 128)")
print(f"  z_G  full graph representation      {auc_mlp_zg:.4f}")
print(f"  r    what the graph uniquely adds   {auc_mlp:.4f}")

out = {"mode": MODE, "alpha_sweep": rows,
       "mlp_probe": {"zG": round(auc_mlp_zg, 4), "residual": round(auc_mlp, 4)}}
p = ROOT / "runs" / f"residual_robust_{MODE}.json"
p.write_text(json.dumps(out, indent=2))
print(f"\n-> {p}")
