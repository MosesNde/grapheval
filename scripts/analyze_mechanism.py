#!/usr/bin/env python3
"""
analyze_mechanism.py — WHY does the code property graph contribute so little?

The paper measures the effect but does not explain it. This tests the leading
mechanical hypothesis: over-smoothing.

THE ARGUMENT
------------
Message passing in GAT/GCN/GIN is a low-pass filter on the graph. Writing H^(l)
for node representations at layer l, repeated aggregation contracts differences
between adjacent nodes. Quantify that with the Dirichlet energy over L2-
normalised rows,

    E(H) = (1/|E|) * sum_{(i,j) in E} || h_i/||h_i|| - h_j/||h_j|| ||^2 ,

which is 0 when every node points the same way and 4 when adjacent nodes are
antipodal. If E(H^(l)) decays sharply with depth, all h_i converge toward a
common direction h*, and any permutation-invariant pooling gives

    pool(H^(L)) ~= h* ,

so the classifier sees essentially one vector. That is exactly what the
edge-free model computes directly from pooled input features -- which would
explain why deleting the edges costs so little.

THE DECISIVE TEST
-----------------
Beyond per-layer energy, we take one TRAINED graph model and run each batch
twice with identical weights: once with the real edge set and once with the
edges removed. If

    cos( pool(GNN(X, E)),  pool(GNN(X, empty)) )  ~= 1

then message passing is barely moving the representation the classifier
consumes, and the ablation result follows mechanically rather than
coincidentally.

Run:  python3 analyze_mechanism.py --mode repo --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
import training as T                      # noqa: E402
from train_cwe import load_split, encode   # noqa: E402


def dirichlet(h: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Mean squared distance between L2-normalised endpoints of each edge."""
    if edge_index.numel() == 0:
        return float("nan")
    hn = F.normalize(h, dim=1)
    d = hn[edge_index[0]] - hn[edge_index[1]]
    return float((d * d).sum(1).mean())


def mean_pairwise_cos(h: torch.Tensor, cap: int = 512) -> float:
    if h.shape[0] < 2:
        return float("nan")
    idx = torch.randperm(h.shape[0])[:cap]
    hn = F.normalize(h[idx], dim=1)
    c = hn @ hn.T
    iu = torch.triu_indices(len(idx), len(idx), offset=1)
    return float(c[iu[0], iu[1]].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-batches", type=int, default=40)
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    cfg.update(batch_size=256, num_workers=6, pin_memory=True,
               epochs=a.epochs, early_stop_patience=15)
    dev = T.get_device()
    T.set_seed(cfg["seed"])

    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache), limit=a.limit)
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    nrel = T.MAX_RELATIONS
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, nrel - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
    base_dim = G["train"][0].x.shape[1]

    print("[train] one GAT, binary, for the mechanism analysis", flush=True)
    model = T.train_expert(G["train"], base_dim, vocab, cfg, dev, nrel,
                           tag="mech-gat", num_classes=2)
    model.eval()

    from torch_geometric.loader import DataLoader
    loader = DataLoader(G["test"], batch_size=64, shuffle=False)

    rows, cos_pool = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= a.n_batches:
                break
            batch = batch.to(dev)
            x0 = model.input_proj(torch.cat(
                [batch.x.float(), model.label_emb(batch.label_id)], dim=1))
            e = model.edge_emb(batch.edge_type)
            ei = batch.edge_index

            # replicate GATWithLabelEmb.forward layer by layer
            x1 = model.norm1(F.elu(model.gat1(x0, ei, edge_attr=e)))
            x2 = model.norm2(F.elu(model.gat2(x1, ei, edge_attr=e)) + x1)
            x3 = model.norm3(F.elu(model.gat3(x2, ei, edge_attr=e)) + x2)
            for l, h in enumerate([x0, x1, x2, x3]):
                rows.append({"layer": l,
                             "dirichlet": dirichlet(h, ei),
                             "pairwise_cos": mean_pairwise_cos(h)})

            # same weights, edges deleted: what does the classifier actually see?
            empty = torch.zeros((2, 0), dtype=torch.long, device=dev)
            e0 = model.edge_emb(torch.zeros((0,), dtype=torch.long, device=dev))
            y1 = model.norm1(F.elu(model.gat1(x0, empty, edge_attr=e0)))
            y2 = model.norm2(F.elu(model.gat2(y1, empty, edge_attr=e0)) + y1)
            y3 = model.norm3(F.elu(model.gat3(y2, empty, edge_attr=e0)) + y2)

            pg = (model.pool(x3, batch.batch) if model.pool is not None
                  else T.global_mean_pool(x3, batch.batch))
            pn = (model.pool(y3, batch.batch) if model.pool is not None
                  else T.global_mean_pool(y3, batch.batch))
            cos_pool.append(F.cosine_similarity(pg, pn, dim=1).cpu().numpy())

    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["layer"]].append((r["dirichlet"], r["pairwise_cos"]))
    print("\nOVER-SMOOTHING ACROSS LAYERS (test partition)")
    print(f"{'layer':7s}{'Dirichlet energy':>18s}{'mean pairwise cos':>20s}")
    out_layers = {}
    for l in sorted(agg):
        d = np.nanmean([v[0] for v in agg[l]])
        c = np.nanmean([v[1] for v in agg[l]])
        nm = "input" if l == 0 else f"GAT {l}"
        out_layers[nm] = {"dirichlet": round(float(d), 4),
                          "pairwise_cos": round(float(c), 4)}
        print(f"{nm:7s}{d:18.4f}{c:20.4f}")

    cp = np.concatenate(cos_pool)
    print(f"\nSAME TRAINED WEIGHTS, EDGES DELETED AT INFERENCE")
    print(f"  cos(pooled with graph, pooled without) = "
          f"{cp.mean():.4f} +/- {cp.std():.4f}   (n={len(cp)} graphs)")
    print(f"  fraction of graphs above 0.99: {float((cp > 0.99).mean()):.3f}")

    res = {"mode": a.mode, "layers": out_layers,
           "pooled_cos_with_vs_without_edges": {
               "mean": round(float(cp.mean()), 4),
               "std": round(float(cp.std()), 4),
               "frac_above_0.99": round(float((cp > 0.99).mean()), 4),
               "n_graphs": int(len(cp))}}
    p = ROOT / "runs" / f"mechanism_{a.mode}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
