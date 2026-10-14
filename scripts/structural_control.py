#!/usr/bin/env python3
"""
structural_control.py — a positive control for the pipeline's ability to read
graph structure at all.

Why this exists. The `topology` ablation (node text zeroed, node types collapsed)
scores at chance, and the paper reads that as "structure alone carries no
vulnerability signal". A reviewer, or an advisor, can read the identical number
as "this pipeline cannot use structure", and nothing in the paper separates the
two. They are the competing hypotheses, so the ambiguity is fatal.

This resolves it. The graphs are real CPGs, but:

  * node features are replaced with random noise AND every node type is
    collapsed to a single id, so nothing about a node carries information
    about the label, and
  * the label is a property of the edge set alone.

Under those conditions a model that can read structure must beat chance, and a
model with the edges deleted must sit AT chance because it has been given
nothing else. Two tasks, so the result does not hinge on one arbitrary target:

  cfg-cycle    does the control-flow subgraph contain a directed cycle?
               (a loop; decidable from CFG edges alone)
  ddg-density  do data-dependence edges outnumber control-flow edges?

Reading the outcome:
  graph >> chance and nograph == chance  -> the pipeline reads structure; the
      `topology` null is a fact about the corpus, not about the code.
  graph == chance                        -> the pipeline cannot read structure,
      and every edge-related result in the paper is suspect.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode

AST, CFG, DDG, CDG = 0, 1, 2, 3          # ids verified against the cache


def has_cfg_cycle(g) -> int:
    """Directed cycle among CFG edges: iteratively strip nodes with no
    outgoing CFG edge; anything left is on a cycle."""
    ei, et = g.edge_index, g.edge_type
    m = et == CFG
    if int(m.sum()) == 0:
        return 0
    src = ei[0][m].tolist(); dst = ei[1][m].tolist()
    out = {}
    for s, d in zip(src, dst):
        out.setdefault(s, set()).add(d)
    nodes = set(src) | set(dst)
    changed = True
    while changed:
        changed = False
        for n in list(nodes):
            if not (out.get(n, set()) & nodes):
                nodes.discard(n); changed = True
    return int(len(nodes) > 0)


def ddg_heavy(g) -> int:
    et = g.edge_type
    return int(int((et == DDG).sum()) > int((et == CFG).sum()))


TASKS = {"cfg-cycle": has_cfg_cycle, "ddg-density": ddg_heavy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cfg-cycle", choices=list(TASKS))
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    cfg.update(batch_size=256, num_workers=6, pin_memory=True,
               epochs=a.epochs, early_stop_patience=15)
    if a.seed is not None:
        cfg["seed"] = a.seed
    dev = T.get_device(); T.set_seed(cfg["seed"])

    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache))
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2

    fn = TASKS[a.task]
    rng = np.random.default_rng(cfg["seed"])
    counts = {}
    for part, gs in G.items():
        pos = 0
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            y = fn(g)
            pos += y
            g.y = torch.tensor(y)
            # Neither node text NOR node type may carry information about the
            # label, or the edge-free arm can solve the task from the node
            # multiset alone: a function with a loop contains CONTROL_STRUCTURE
            # nodes, which is most of the answer to cfg-cycle.
            g.x = torch.tensor(rng.standard_normal(g.x.shape), dtype=torch.float32)
            g.label_id = torch.zeros_like(g.label_id)
        counts[part] = (pos, len(gs))
        print(f"  [{part}] positives {pos}/{len(gs)} = {pos/max(len(gs),1):.3f}", flush=True)

    base_dim = G["train"][0].x.shape[1]
    out = {"task": a.task, "mode": a.mode, "seed": cfg["seed"],
           "prevalence": {k: round(v[0]/max(v[1],1), 4) for k, v in counts.items()}}

    for arm in ("graph", "nograph"):
        import copy
        H = {k: [g.clone() for g in v] for k, v in G.items()}
        if arm == "nograph":
            for gs in H.values():
                for g in gs:
                    g.edge_index = torch.zeros((2, 0), dtype=torch.long)
                    g.edge_type = torch.zeros((0,), dtype=torch.long)
        print(f"\n[train] {arm}", flush=True)
        m = T.train_expert(H["train"], base_dim, vocab, cfg, dev,
                           T.MAX_RELATIONS, tag=f"struct-{a.task}-{arm}",
                           num_classes=2)
        p = T.probs_from_model(m, H["test"], dev, 2)[:, 1]
        y = np.array([int(g.y.item()) for g in H["test"]])
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
        out[arm] = round(auc, 4)
        print(f"  {arm} ROC-AUC {auc:.4f}", flush=True)
        del m; torch.cuda.empty_cache()

    out["delta"] = round(out["graph"] - out["nograph"], 4)
    print(f"\n{a.task}: graph {out['graph']:.4f}   no graph {out['nograph']:.4f}"
          f"   delta {out['delta']:+.4f}")
    sfx = "" if a.seed is None else f"-s{a.seed}"
    p_ = ROOT / "runs" / f"structural_control_{a.task}_{a.mode}{sfx}.json"
    p_.write_text(json.dumps(out, indent=2))
    print(f"-> {p_}")


if __name__ == "__main__":
    main()
