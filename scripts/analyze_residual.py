#!/usr/bin/env python3
"""
analyze_residual.py — the graph changes the representation. Does the change help?

analyze_mechanism.py refuted over-smoothing: Dirichlet energy falls only 1.30 ->
0.61 across three layers, node representations do not collapse, and deleting the
edges from a trained model moves the pooled representation substantially
(cos = 0.798). So the network is not ignoring the graph. It is using it, and the
use does not pay.

This asks where the graph's contribution goes. Let

    z_G = pool(GNN(X, E))     the representation the classifier actually sees
    z_0 = pool(GNN(X, empty)) the same weights with the edges deleted

Fit a ridge map W on the TRAIN partition predicting z_G from z_0, and split the
test representation into a redundant part and a residual:

    z_G = W z_0 + r ,    r perpendicular to the span of the edge-free features.

r is precisely what message passing adds that could not have been recovered from
the pooled node features alone. Three logistic probes then locate the signal:

    AUC(z_0)  what the tokens alone support
    AUC(z_G)  what the full graph representation supports
    AUC(r)    what the graph's genuinely new component supports

If AUC(r) ~= 0.5 while cos(z_G, z_0) is well below 1, the conclusion is sharp:
the edges move the representation a long way, into directions that carry no
information about the label.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def pooled(model, graphs, dev, bs=128):
    """Return (z_G, z_0, y): pooled reps with and without edges, same weights."""
    ld = DataLoader(graphs, batch_size=bs, shuffle=False)
    ZG, Z0, Y = [], [], []
    with torch.no_grad():
        for b in ld:
            b = b.to(dev)
            x0 = model.input_proj(torch.cat(
                [b.x.float(), model.label_emb(b.label_id)], dim=1))
            e = model.edge_emb(b.edge_type); ei = b.edge_index
            x1 = model.norm1(F.elu(model.gat1(x0, ei, edge_attr=e)))
            x2 = model.norm2(F.elu(model.gat2(x1, ei, edge_attr=e)) + x1)
            x3 = model.norm3(F.elu(model.gat3(x2, ei, edge_attr=e)) + x2)
            emp = torch.zeros((2, 0), dtype=torch.long, device=dev)
            e0 = model.edge_emb(torch.zeros((0,), dtype=torch.long, device=dev))
            y1 = model.norm1(F.elu(model.gat1(x0, emp, edge_attr=e0)))
            y2 = model.norm2(F.elu(model.gat2(y1, emp, edge_attr=e0)) + y1)
            y3 = model.norm3(F.elu(model.gat3(y2, emp, edge_attr=e0)) + y2)
            pl = model.pool if model.pool is not None else T.global_mean_pool
            ZG.append(pl(x3, b.batch).cpu().numpy())
            Z0.append(pl(y3, b.batch).cpu().numpy())
            Y.append(b.y.cpu().numpy())
    zg, z0, y = np.concatenate(ZG), np.concatenate(Z0), np.concatenate(Y)
    for nm, arr in (("z_G", zg), ("z_0", z0)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{nm} contains non-finite values -- the "
                               f"edge-free forward pass is degenerate, not a "
                               f"valid comparison")
    return zg, z0, y


def probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    a = ap.parse_args()

    cfg = dict(T.CONFIG); cfg.update(batch_size=256, num_workers=6,
                                     pin_memory=True, epochs=a.epochs,
                                     early_stop_patience=15)
    dev = T.get_device(); T.set_seed(cfg["seed"])
    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache))
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
    base_dim = G["train"][0].x.shape[1]

    ckpt = ROOT / "runs" / f"mech_gat_{a.mode}.pt"
    model = T.train_expert(G["train"], base_dim, vocab, cfg, dev,
                           T.MAX_RELATIONS, tag="mech-gat", num_classes=2)
    torch.save(model.state_dict(), ckpt)
    print(f"[ckpt] {ckpt}", flush=True)
    model.eval()

    print("[dump] pooled representations, train then test", flush=True)
    zg_tr, z0_tr, y_tr = pooled(model, G["train"], dev)
    zg_te, z0_te, y_te = pooled(model, G["test"], dev)

    # how much of the graph representation is recoverable from the edge-free one
    W = Ridge(alpha=1.0).fit(z0_tr, zg_tr)
    pred_te = W.predict(z0_te)
    ss_res = ((zg_te - pred_te) ** 2).sum()
    ss_tot = ((zg_te - zg_tr.mean(0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    r_tr = zg_tr - W.predict(z0_tr)
    r_te = zg_te - pred_te
    frac_norm = float(np.linalg.norm(r_te) / np.linalg.norm(zg_te))

    auc_z0 = probe(z0_tr, y_tr, z0_te, y_te)
    auc_zg = probe(zg_tr, y_tr, zg_te, y_te)
    auc_r  = probe(r_tr,  y_tr, r_te,  y_te)
    cos = float(F.cosine_similarity(torch.tensor(zg_te),
                                    torch.tensor(z0_te), dim=1).mean())

    print(f"\nWHERE THE GRAPH'S CONTRIBUTION GOES  (mode={a.mode}, "
          f"n_train={len(y_tr)}, n_test={len(y_te)})")
    print(f"  cos(z_G, z_0)                         {cos:.4f}")
    print(f"  R^2 of z_0 -> z_G (ridge)             {r2:.4f}")
    print(f"  ||r|| / ||z_G||  (new component)      {frac_norm:.4f}")
    print(f"\n  linear probe ROC-AUC")
    print(f"    z_0  edge-free representation       {auc_z0:.4f}")
    print(f"    z_G  full graph representation      {auc_zg:.4f}")
    print(f"    r    what the graph uniquely adds   {auc_r:.4f}")

    out = {"mode": a.mode, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "cos_zG_z0": round(cos, 4), "ridge_r2_z0_to_zG": round(r2, 4),
           "residual_norm_fraction": round(frac_norm, 4),
           "probe_auc": {"z0": round(auc_z0, 4), "zG": round(auc_zg, 4),
                         "residual": round(auc_r, 4)}}
    p = ROOT / "runs" / f"residual_{a.mode}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
