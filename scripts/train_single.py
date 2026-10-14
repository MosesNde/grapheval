#!/usr/bin/env python3
"""
train_single.py — one model on the whole training partition, instead of a
five-fold ensemble.

train_cwe.py reports a 5-fold ensemble in which each member trains on 80% of the
training partition and picks its epoch on a slice carved from that same 80%.
Under a repository-disjoint protocol that slice shares repositories with the
data being fitted, so early stopping rewards memorisation. Training one model on
the whole partition and selecting on the held-out validation repositories scores
higher: 0.652 against 0.618 on `repo` with a bidirectional graph.

Everything else is held to train_cwe's settings so the two are comparable: same
cached node features, same ablations, same architecture, same corpus and splits.
What differs is the training procedure, and that is the point of the file.

Schedule follows the runs that produced 0.652: AdamW, linear warmup over 10% of
steps then cosine decay, gradient accumulation to an effective batch of 128,
bf16 autocast, 15 epochs, model chosen by validation ROC-AUC.

    python3 train_single.py --mode repo --reverse-edges --seed 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode, apply_feature_ablation, binary_metrics


def run_epoch(model, loader, dev, opt=None, sched=None, accum=1):
    train = opt is not None
    model.train(train)
    ps, ys, tot, n = [], [], 0.0, 0
    for step, b in enumerate(loader):
        b = b.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b)
            loss = F.cross_entropy(logits, b.y)
        if train:
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched: sched.step()
        tot += float(loss.detach()) * b.num_graphs; n += b.num_graphs
        ps.append(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().numpy())
        ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    return tot / max(n, 1), p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--backbone", default="gat")
    ap.add_argument("--edges", default="all", choices=["all", "untyped", "none"])
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--features", default="full",
                    choices=["full", "type-only", "random", "topology"])
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    ap.add_argument("--task", default="family", choices=["family", "binary"])
    ap.add_argument("--split-suffix", default="")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--node-cap", type=int, default=0,
                    help="truncate graphs to this many nodes (0 = no cap). The "
                         "runs that reached 0.652 used 400, which truncates "
                         "8.4%% of repo training graphs, so it is part of that "
                         "result rather than a neutral memory guard.")
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    cfg.update(num_workers=6, pin_memory=True, backbone=a.backbone)
    if a.seed is not None:
        cfg["seed"] = a.seed
    dev = T.get_device(); T.set_seed(cfg["seed"])

    corpus, fams, splits = load_split(a.mode, task=a.task,
                                      suffix=a.split_suffix)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache),
                  limit=a.limit, task=a.task)
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    apply_feature_ablation(G, a.features, seed=cfg["seed"])
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
            if a.edges == "none":
                g.edge_index = torch.zeros((2, 0), dtype=torch.long)
                g.edge_type = torch.zeros((0,), dtype=torch.long)
            elif a.edges == "untyped":
                g.edge_type = torch.zeros_like(g.edge_type)
            elif a.reverse_edges and g.edge_index.numel():
                g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], 1)
                g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
    if a.node_cap:
        cut = 0
        for gs in G.values():
            for g in gs:
                if g.num_nodes > a.node_cap:
                    cut += 1
                    m = (g.edge_index[0] < a.node_cap) & (g.edge_index[1] < a.node_cap)
                    g.edge_index = g.edge_index[:, m]; g.edge_type = g.edge_type[m]
                    g.x = g.x[:a.node_cap]; g.label_id = g.label_id[:a.node_cap]
        print(f"[node-cap] {a.node_cap}: truncated {cut} graphs", flush=True)

    base_dim = G["train"][0].x.shape[1]

    ld = lambda part, sh: DataLoader(G[part], batch_size=a.batch_size, shuffle=sh,
                                     num_workers=4)
    tr_ld, va_ld, te_ld = ld("train", True), ld("val", False), ld("test", False)
    model = T.build_model(a.backbone, base_dim, vocab, cfg,
                          T.MAX_RELATIONS, 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(tr_ld) // a.accum) * a.epochs
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_frac * steps), steps)
    print(f"[single] {len(G['train'])} train / {len(G['val'])} val / "
          f"{len(G['test'])} test, effective batch {a.batch_size*a.accum}, "
          f"{steps} steps", flush=True)

    best, best_state, hist = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        loss, _, _ = run_epoch(model, tr_ld, dev, opt, sched, a.accum)
        with torch.no_grad():
            _, vp, vy = run_epoch(model, va_ld, dev)
        vauc = roc_auc_score(vy, vp) if 0 < vy.mean() < 1 else float("nan")
        hist.append({"epoch": ep, "loss": round(loss, 4), "val_auc": round(vauc, 4)})
        print(f"  ep {ep}/{a.epochs} loss {loss:.4f} val {vauc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best = vauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)

    with torch.no_grad():
        _, tp, ty = run_epoch(model, te_ld, dev)
    res = binary_metrics(ty, tp, ty.mean())
    print(f"\nsingle-model test ROC-AUC {res['roc_auc']:.4f}")

    tag = (f"{a.mode}-{a.backbone}"
           + ("" if a.features == "full" else f"-{a.features}")
           + ("" if a.edges == "all" else f"-e{a.edges}")
           + ("-rev" if a.reverse_edges else "")
           + ("" if a.seed is None else f"-s{a.seed}")
           + (f"-nc{a.node_cap}" if a.node_cap else "")
           + (f"-{a.tag}" if a.tag else ""))
    out = {"single_model": res, "best_val_auc": round(float(best), 4),
           "history": hist, "mode": a.mode, "backbone": a.backbone,
           "edges": a.edges, "reverse_edges": a.reverse_edges,
           "features": a.features, "seed": cfg["seed"]}
    p = ROOT / "runs" / f"{tag}_single.json"
    p.write_text(json.dumps(out, indent=2)); print(f"-> {p}")


if __name__ == "__main__":
    main()
