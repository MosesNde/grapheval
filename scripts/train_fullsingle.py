#!/usr/bin/env python3
"""
train_fullsingle.py — the controlled arm for the ensemble-vs-single comparison.

run_stage1 trains five models, each on four fifths of the training partition,
and averages them. train_single.py trains one model on the whole partition but
with a DIFFERENT recipe (15 epochs, effective batch 128, cosine+warmup, bf16),
so comparing the two confounds ensembling, data quantity and the recipe.

This file removes that confound: it calls the same train_expert with the same
cfg that run_stage1 uses, on the whole training partition, selecting the epoch
on the same held-out validation partition. The ONLY difference from the
ensemble is that one model sees 5/5 and nothing is averaged.

Together with the per-fold scores already stored in *_stage1.json, this gives:
    individual fold model (4/5, this recipe)   <- per_fold_roc_auc
    ensemble of those five                      <- ensemble_mean_only
    one model on 5/5, this recipe               <- produced here
so ensembling and data quantity separate cleanly at a fixed recipe.

Writes runs/{tag}_fullsingle.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
import training as T                                  # noqa: E402
from train_cwe import load_split, encode, binary_metrics   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--backbone", default="gat")
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    # defaults mirror train_cwe.py exactly
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    cfg.update(batch_size=a.batch_size, num_workers=a.workers, pin_memory=True,
               epochs=a.epochs, early_stop_patience=a.patience)
    cfg["backbone"] = a.backbone
    if a.seed is not None:
        cfg["seed"] = a.seed
    T.set_seed(cfg["seed"])
    dev = T.get_device()
    print(f"[cfg] backbone={cfg['backbone']} batch={cfg['batch_size']} "
          f"epochs={cfg['epochs']} patience={cfg['early_stop_patience']}", flush=True)

    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache))
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    nrel = T.MAX_RELATIONS
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            if hasattr(g, "edge_type") and g.edge_type.numel():
                g.edge_type = g.edge_type.clamp_(0, nrel - 1)

    if a.reverse_edges:
        n0 = n1 = 0
        for gs in G.values():
            for g in gs:
                if not (hasattr(g, "edge_index") and g.edge_index.numel()):
                    continue
                n0 += g.edge_index.shape[1]
                g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], dim=1)
                g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
                n1 += g.edge_index.shape[1]
        print(f"[reverse-edges] {n0} -> {n1} edges", flush=True)

    def binarize(gs):
        out = []
        for g in gs:
            g2 = g.clone(); g2.y = torch.tensor(int(g.y.item() > 0)); out.append(g2)
        return out
    tr, va, te = binarize(G["train"]), binarize(G["val"]), binarize(G["test"])
    base_dim = tr[0].x.shape[1]
    print(f"[data] {len(tr)} train (5/5) / {len(va)} val / {len(te)} test", flush=True)

    tag = (f"{a.mode}-{cfg['backbone']}" + ("-rev" if a.reverse_edges else "")
           + ("" if a.seed is None else f"-s{a.seed}"))

    # identical call to the one run_stage1 makes per fold, but on the whole
    # training partition rather than four fifths of it
    m = T.train_expert(tr, base_dim, vocab, cfg, dev, nrel,
                       tag=f"{tag}-fullsingle", num_classes=2, val_graphs=va)
    p = T.probs_from_model(m, te, dev, 2)[:, 1]
    y = np.array([int(g.y.item()) for g in te])
    res = binary_metrics(y, p, y.mean())

    out = ROOT / "runs" / f"{tag}_fullsingle.json"
    out.write_text(json.dumps({"single_full_partition": res,
                               "n_train": len(tr), "n_test": len(te),
                               "cfg": {k: cfg[k] for k in
                                       ("backbone","batch_size","epochs",
                                        "early_stop_patience","lr","seed")}}, indent=2))
    print(f"[done] roc_auc={res['roc_auc']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
