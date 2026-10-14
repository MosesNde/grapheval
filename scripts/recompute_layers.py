#!/usr/bin/env python3
"""
recompute_layers.py — put every number in the mechanism section on one model.

analyze_mechanism.py and analyze_residual.py each trained their own GAT. Same
seed, but dropout and cuDNN nondeterminism made them diverge (repo loss 8.85 vs
10.06; random 11.50 vs 10.87), so the layer-wise energies were measured on a
different network than the cosines, the ridge decomposition and the probes.

This recomputes the layer diagnostics from the SAVED checkpoint, the same model
the probes were run on, and over the full test partition rather than the 2,560
graph subsample the original used.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode
from analyze_mechanism import dirichlet, mean_pairwise_cos
from torch_geometric.loader import DataLoader

MODE = sys.argv[1] if len(sys.argv) > 1 else "repo"
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

n_nodes = np.array([int(g.x.shape[0]) for g in G["test"]])
n_edges = np.array([int(g.edge_index.shape[1]) for g in G["test"]])

m = T.build_model("gat", G["train"][0].x.shape[1], vocab, cfg, T.MAX_RELATIONS, 2).to(dev)
m.load_state_dict(torch.load(ROOT / "runs" / f"mech_gat_{MODE}.pt"))
m.eval()

agg = collections.defaultdict(list); cosl = []
with torch.no_grad():
    for b in DataLoader(G["test"], batch_size=64, shuffle=False):
        b = b.to(dev)
        x0 = m.input_proj(torch.cat([b.x.float(), m.label_emb(b.label_id)], 1))
        e, ei = m.edge_emb(b.edge_type), b.edge_index
        x1 = m.norm1(F.elu(m.gat1(x0, ei, edge_attr=e)))
        x2 = m.norm2(F.elu(m.gat2(x1, ei, edge_attr=e)) + x1)
        x3 = m.norm3(F.elu(m.gat3(x2, ei, edge_attr=e)) + x2)
        for l, h in enumerate([x0, x1, x2, x3]):
            agg[l].append((dirichlet(h, ei), mean_pairwise_cos(h)))
        emp = torch.zeros((2, 0), dtype=torch.long, device=dev)
        e0 = m.edge_emb(torch.zeros((0,), dtype=torch.long, device=dev))
        y1 = m.norm1(F.elu(m.gat1(x0, emp, edge_attr=e0)))
        y2 = m.norm2(F.elu(m.gat2(y1, emp, edge_attr=e0)) + y1)
        y3 = m.norm3(F.elu(m.gat3(y2, emp, edge_attr=e0)) + y2)
        pl = m.pool if m.pool is not None else T.global_mean_pool
        cosl.append(F.cosine_similarity(pl(x3, b.batch), pl(y3, b.batch), dim=1).cpu().numpy())

layers = {}
print(f"MODE={MODE}  (checkpoint model, full test partition n={len(G['test'])})")
print(f"{'layer':7s}{'Dirichlet':>12s}{'pairwise cos':>15s}")
for l in sorted(agg):
    d = float(np.nanmean([v[0] for v in agg[l]]))
    c = float(np.nanmean([v[1] for v in agg[l]]))
    nm = "input" if l == 0 else f"GAT {l}"
    layers[nm] = {"dirichlet": round(d, 4), "pairwise_cos": round(c, 4)}
    print(f"{nm:7s}{d:12.4f}{c:15.4f}")

cp = np.concatenate(cosl)
decay = layers["input"]["dirichlet"] / layers["GAT 3"]["dirichlet"]
print(f"\n  energy decay factor            {decay:.3f}")
print(f"  cos(pooled with, without)      {cp.mean():.4f} +/- {cp.std():.4f}")
print(f"  frac above 0.99                {float((cp > 0.99).mean()):.4f}")
print(f"\n  test graphs: median nodes {int(np.median(n_nodes))}, "
      f"p90 {int(np.percentile(n_nodes, 90))}, "
      f"mean degree {n_edges.sum() / max(n_nodes.sum(), 1):.2f}")

out = {"mode": MODE, "source": "checkpoint, full test partition",
       "n_test": int(len(G["test"])), "layers": layers,
       "energy_decay_factor": round(decay, 3),
       "pooled_cos": {"mean": round(float(cp.mean()), 4),
                      "std": round(float(cp.std()), 4),
                      "frac_above_0.99": round(float((cp > 0.99).mean()), 4)},
       "graph_stats": {"median_nodes": int(np.median(n_nodes)),
                       "p90_nodes": int(np.percentile(n_nodes, 90)),
                       "mean_degree": round(float(n_edges.sum() / n_nodes.sum()), 2)}}
p = ROOT / "runs" / f"layers_ckpt_{MODE}.json"
p.write_text(json.dumps(out, indent=2)); print(f"\n-> {p}")
