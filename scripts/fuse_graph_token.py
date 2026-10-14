#!/usr/bin/env python3
"""
fuse_graph_token.py — does the graph add anything a token model does not have?

The graph pipeline reaches ROC-AUC 0.613 on the repository-disjoint split and
LineVul reaches 0.792 on the identical partition. A second stage is only worth
building if the two models are wrong about different samples; if the graph's
predictions are a noisy subset of the token model's, fusion cannot help.

This trains the bidirectional graph model once, writes a score for every test
function, aligns those scores with LineVul's released per-sample logits, and
measures three things:

  1. agreement       rank correlation between the two score vectors, and the
                     ROC-AUC each achieves on the samples the other gets wrong
  2. fusion          an unweighted average of rank-normalised scores. No
                     parameter is fitted, so nothing is learned from the test
                     labels and the number needs no held-out correction
  3. fitted fusion   a logistic combiner, cross-fitted in two folds within the
                     test partition, as an upper bound on what a tuned second
                     stage could reach

Reported alongside each model alone. A fusion that does not beat the better
component is evidence that the graph carries no complementary signal.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode


def rank01(v):
    """Rank-normalise to [0,1] so two differently-calibrated scores can be averaged."""
    order = np.argsort(np.argsort(v))
    return order / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--reverse-edges", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    a = ap.parse_args()

    # These must match the settings every reported run uses. train_cwe.py sets
    # them from its own defaults, not from CONFIG, so taking CONFIG unaltered
    # trains at batch 16 with a learning rate tuned for 256.
    cfg = dict(T.CONFIG); cfg.update(batch_size=256, num_workers=6,
                                     pin_memory=True, epochs=120,
                                     early_stop_patience=20)
    if a.seed is not None:
        cfg["seed"] = a.seed
    dev = T.get_device(); T.set_seed(cfg["seed"])

    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path(a.cache))
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
            if a.reverse_edges and g.edge_index.numel():
                g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], 1)
                g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
    base_dim = G["train"][0].x.shape[1]

    # five-fold ensemble over the training partition, exactly as run_stage1 does
    from sklearn.model_selection import StratifiedKFold as SKF
    tr, te = G["train"], G["test"]
    y_tr = np.array([int(g.y.item()) for g in tr])
    acc = []
    for k, (i_a, _) in enumerate(SKF(5, shuffle=True, random_state=cfg["seed"])
                                 .split(np.arange(len(tr)), y_tr)):
        m = T.train_expert([tr[i] for i in i_a], base_dim, vocab, cfg, dev,
                           T.MAX_RELATIONS, tag=f"fuse-f{k+1}", num_classes=2)
        acc.append(T.probs_from_model(m, te, dev, 2)[:, 1])
        del m; torch.cuda.empty_cache()
        print(f"  fold {k+1}/5 done", flush=True)
    g_score = np.mean(np.stack(acc), axis=0)
    g_ids = [g.sample_id for g in te]
    y_g = np.array([int(g.y.item()) for g in te])
    print(f"\ngraph model alone: {roc_auc_score(y_g, g_score):.4f}")

    # LineVul rows follow the split's id order, minus ids with no cached source
    srcset = {json.loads(l)["sample_id"] for l in open(ROOT/"data"/"source_cache.jsonl") if l.strip()}
    lv_ids = [s for s in splits["test"] if s in srcset]
    lg = np.load(ROOT/"runs"/"linevul_raw"/f"{a.mode}_logits.npy")
    assert len(lv_ids) == len(lg), (len(lv_ids), len(lg))
    lv = dict(zip(lv_ids, 1/(1+np.exp(-(lg[:,1]-lg[:,0])))))

    keep = [i for i, s in enumerate(g_ids) if s in lv]
    gs = g_score[keep]; ys = y_g[keep]
    ls = np.array([lv[g_ids[i]] for i in keep])
    print(f"samples scored by both models: {len(keep)}")
    print(f"  graph   {roc_auc_score(ys, gs):.4f}")
    print(f"  LineVul {roc_auc_score(ys, ls):.4f}")

    rho = spearmanr(gs, ls).statistic
    print(f"\nrank correlation between the two score vectors: {rho:.3f}")
    lv_wrong = (ls > np.median(ls)) != ys.astype(bool)
    if lv_wrong.sum() > 50 and 0 < ys[lv_wrong].mean() < 1:
        print(f"  graph ROC-AUC on the {int(lv_wrong.sum())} samples LineVul gets wrong: "
              f"{roc_auc_score(ys[lv_wrong], gs[lv_wrong]):.4f}")

    gr, lr = rank01(gs), rank01(ls)
    out = {"n": len(keep), "graph": round(float(roc_auc_score(ys, gs)), 4),
           "linevul": round(float(roc_auc_score(ys, ls)), 4),
           "spearman": round(float(rho), 4)}
    print("\nfusion, no parameters fitted:")
    for w in (0.25, 0.5, 0.75):
        s = w*gr + (1-w)*lr
        auc = roc_auc_score(ys, s)
        out[f"avg_w{w}"] = round(float(auc), 4)
        print(f"  {w:.2f}*graph + {1-w:.2f}*LineVul : {auc:.4f}")

    # cross-fitted logistic combiner: an upper bound, since it sees test labels
    X = np.c_[gr, lr]; pred = np.zeros(len(ys))
    for i_tr, i_te in StratifiedKFold(2, shuffle=True, random_state=0).split(X, ys):
        lr_m = LogisticRegression(max_iter=1000).fit(X[i_tr], ys[i_tr])
        pred[i_te] = lr_m.predict_proba(X[i_te])[:, 1]
    out["fitted_stack"] = round(float(roc_auc_score(ys, pred)), 4)
    print(f"\nfitted combiner, 2-fold cross-fit within test (upper bound): "
          f"{out['fitted_stack']:.4f}")

    p = ROOT/"runs"/f"fusion_{a.mode}.json"
    p.write_text(json.dumps(out, indent=2)); print(f"\n-> {p}")


if __name__ == "__main__":
    main()
