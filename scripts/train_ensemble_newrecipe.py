#!/usr/bin/env python3
"""
train_ensemble_newrecipe.py — the second controlled arm.

train_fullsingle.py showed that at a FIXED recipe (train_expert: batch 256, 120
epochs, ReduceLROnPlateau, fp32) a five-fold ensemble BEATS one model on the
whole partition, and the extra fifth of data is worth nothing. The large gap we
had been attributing to "single vs ensemble" is therefore the recipe:
train_single.py uses 15 epochs, cosine+warmup, effective batch 128, bf16.

This file completes the 2x2 by applying train_single.py's recipe to a five-fold
ensemble: five members, each on four fifths, each trained with that loop and
selected on the same held-out validation partition, then averaged.

                      old recipe            new recipe
  single 5/5          train_fullsingle      train_single.py
  ensemble 5x4/5      run_stage1            THIS FILE

Writes runs/{tag}_ensnew.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
import training as T                                        # noqa: E402
from train_cwe import load_split, encode, binary_metrics    # noqa: E402
from train_single import run_epoch                          # noqa: E402  same loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--backbone", default="gat")
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    # identical to train_single.py's defaults
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    cfg.update(num_workers=6, pin_memory=True, backbone=a.backbone)
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
    tr = G["train"]
    y_tr = np.array([int(g.y.item()) for g in tr])
    print(f"[ens-new] {len(tr)} train / {len(G['val'])} val / {len(G['test'])} test",
          flush=True)

    va_ld = DataLoader(G["val"],  batch_size=a.batch_size, shuffle=False, num_workers=4)
    te_ld = DataLoader(G["test"], batch_size=a.batch_size, shuffle=False, num_workers=4)
    from transformers import get_cosine_schedule_with_warmup

    skf = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
    te_preds, per_fold = [], []
    for k, (idx, _) in enumerate(skf.split(np.arange(len(tr)), y_tr), 1):
        sub = [tr[i] for i in idx]
        tr_ld = DataLoader(sub, batch_size=a.batch_size, shuffle=True, num_workers=4)
        model = T.build_model(a.backbone, base_dim, vocab, cfg,
                              T.MAX_RELATIONS, 2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = (len(tr_ld) // a.accum) * a.epochs
        sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_frac*steps), steps)
        best, best_state = -1.0, None
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            loss, _, _ = run_epoch(model, tr_ld, dev, opt, sched, a.accum)
            with torch.no_grad():
                _, vp, vy = run_epoch(model, va_ld, dev)
            vauc = roc_auc_score(vy, vp) if 0 < vy.mean() < 1 else float("nan")
            print(f"  f{k} ep {ep}/{a.epochs} loss {loss:.4f} val {vauc:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if vauc > best:
                best = vauc
                best_state = {kk: v.detach().cpu().clone()
                              for kk, v in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
        with torch.no_grad():
            _, tp, ty = run_epoch(model, te_ld, dev)
        te_preds.append(tp)
        per_fold.append(round(float(roc_auc_score(ty, tp)), 4))
        print(f"  fold {k} test ROC-AUC {per_fold[-1]:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    mean_p = np.mean(np.stack(te_preds), axis=0)
    res = binary_metrics(ty, mean_p, ty.mean())
    tag = (f"{a.mode}-{a.backbone}" + ("-rev" if a.reverse_edges else "")
           + ("" if a.seed is None else f"-s{a.seed}"))
    import platform, subprocess
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git = None
    # Provenance: the hyperparameters are recorded explicitly because an earlier
    # sweep ran this file at its defaults while the single-model arm used the
    # validation-selected recipe, producing an unmatched 2x2 that the artifacts
    # could not be used to detect.
    out = ROOT / "runs" / f"{tag}_ensnew{'-' + a.tag if a.tag else ''}.json"
    out.write_text(json.dumps({
        "ensemble_new_recipe": res,
        "per_fold_roc_auc": per_fold,
        "fold_mean": round(float(np.mean(per_fold)), 4),
        "seed": cfg["seed"],
        "config": {"mode": a.mode, "backbone": a.backbone,
                   "reverse_edges": a.reverse_edges,
                   "epochs": a.epochs, "batch_size": a.batch_size,
                   "accum": a.accum, "effective_batch": a.batch_size * a.accum,
                   "lr": a.lr, "warmup_frac": a.warmup_frac,
                   "n_folds": 5, "aggregation": "mean of member probabilities",
                   "checkpoint_selection": "best val ROC-AUC on the held-out "
                                           "validation partition",
                   "optimizer": "AdamW", "weight_decay": 0.01,
                   "schedule": "linear warmup then cosine", "precision": "bf16"},
        "env": {"python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "git_commit": git},
    }, indent=2))
    print(f"[done] ensemble roc_auc={res['roc_auc']:.4f} "
          f"(fold mean {np.mean(per_fold):.4f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
