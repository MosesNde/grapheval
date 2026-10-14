"""
gat_deep_binary.py
===================
Controlled depth comparison: trains a 6-layer GAT (matching Devign's 6-round
GGRN propagation depth) on the SAME binary task, SAME train/test split, as
both Stage 1 (3-layer GAT, ~0.65 macro F1) and the Devign baseline
(0.7101 macro F1).

Purpose: isolate whether Devign's binary-task advantage comes from
propagation DEPTH (6 rounds vs 3 layers) or from its architecture/readout
mechanism (gated GRU propagation + dual-branch Conv Module vs attention +
mean/attn-pool). If this 6-layer GAT closes most of the gap to Devign,
depth was the main factor. If it doesn't, the gap is more likely
architectural (GGRN propagation style and/or Conv Module readout).

THIS SCRIPT DOES NOT MODIFY training.py, devign.py, OR ANY EXISTING OUTPUT.
It imports utilities from training.py read-only and writes to NEW filenames:
    <DATA_DIR>/gat6layer_binary_results.json
    <DATA_DIR>/gat6layer_binary_checkpoint.pt

Usage:
    python gat_deep_binary.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import classification_report, f1_score
from torch_geometric.nn import GATConv, global_mean_pool

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv  # noqa: E402

CONFIG    = tv.CONFIG
DATA_DIR  = Path(CONFIG["DATA_DIR"])
CACHE_DIR = Path(CONFIG.get("CACHE_DIR", DATA_DIR / "graph_cache"))

GAT6_CONFIG = {
    "hidden_dim":          CONFIG.get("hidden_dim", 256),
    "label_emb_dim":       CONFIG.get("label_emb_dim", 64),
    "gat_heads":           CONFIG.get("gat_heads", 8),
    "gat_edge_emb_dim":    CONFIG.get("gat_edge_emb_dim", 32),
    "dropout":             CONFIG.get("dropout", 0.3),
    "attn_pool":           CONFIG.get("attn_pool", True),
    "n_layers":            6,   # ← matches Devign's 6 GGRN propagation rounds
    "lr":                  CONFIG.get("lr", 1e-4),
    "weight_decay":        CONFIG.get("weight_decay", 1e-5),
    "batch_size":          CONFIG.get("batch_size", 32),
    "epochs":              CONFIG.get("epochs", 200),
    "val_frac":            CONFIG.get("val_frac", 0.1),
    "early_stop_patience": CONFIG.get("early_stop_patience", 50),
    "sched_plateau":       CONFIG.get("sched_plateau", True),
    "focal_gamma":         CONFIG.get("focal_gamma", 2.0),
    "seed":                CONFIG.get("seed", 42),
}


# ─────────────────────────────────────────────────────────────────────────────
# 6-LAYER GAT (variable depth, same building blocks as training.py's
# GATWithLabelEmb — attention conv, edge-type embedding, LayerNorm,
# residual connections, optional attention pooling — just generalized to
# an arbitrary number of layers instead of hardcoded to 3).
# ─────────────────────────────────────────────────────────────────────────────
class GATDeepWithLabelEmb(nn.Module):
    def __init__(self, base_dim, vocab, label_emb_dim, hidden_dim,
                 heads, dropout, attn_pool, num_relations, num_classes,
                 n_layers=6, edge_emb_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(max(vocab, 1), label_emb_dim)
        self.edge_emb  = nn.Embedding(max(num_relations, 1), edge_emb_dim)
        raw_dim = base_dim + label_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        out_ch1 = max(1, hidden_dim // heads)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                # First layer uses multi-head attention with concat, as in
                # training.py's GATWithLabelEmb, to match its capacity at
                # the input boundary.
                self.convs.append(GATConv(hidden_dim, out_ch1, heads=heads,
                                          concat=True, dropout=dropout,
                                          edge_dim=edge_emb_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1,
                                          concat=False, dropout=dropout,
                                          edge_dim=edge_emb_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.pool    = tv.make_attn_pool(hidden_dim) if attn_pool else None
        self.cls     = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.input_proj(
            torch.cat([data.x.float(), self.label_emb(data.label_id)], dim=1)
        )
        e_attr = self.edge_emb(data.edge_type) if hasattr(data, "edge_type") else None

        prev = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.elu(conv(prev, data.edge_index, edge_attr=e_attr))
            h = norm(h + prev) if h.shape == prev.shape else norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            prev = h

        pooled = self.pool(prev, data.batch) if self.pool else global_mean_pool(prev, data.batch)
        return self.cls(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (mirrors training.py's train_expert exactly, parameterized over
# this standalone model instead of build_model's backbone dispatch)
# ─────────────────────────────────────────────────────────────────────────────
def train_gat_deep(graphs_subset, base_dim, vocab, cfg, device,
                   num_relations, tag="gat6"):
    import random
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.loader import DataLoader

    if not graphs_subset:
        return None

    model = GATDeepWithLabelEmb(
        base_dim, vocab, cfg["label_emb_dim"], cfg["hidden_dim"],
        cfg["gat_heads"], cfg["dropout"], cfg["attn_pool"],
        num_relations, num_classes=2, n_layers=cfg["n_layers"],
        edge_emb_dim=cfg["gat_edge_emb_dim"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    graphs_subset = list(graphs_subset)
    random.shuffle(graphs_subset)
    y_sub  = np.array([int(g.y.item()) for g in graphs_subset])
    counts = np.bincount(y_sub, minlength=2).astype(np.float32)
    w      = counts.sum() / (2 * np.maximum(counts, 1.0))
    w_ten  = torch.tensor(w, dtype=torch.float32).to(device)
    crit   = tv.FocalLoss(weight=w_ten, gamma=cfg["focal_gamma"])

    n_val      = max(1, int(cfg["val_frac"] * len(graphs_subset)))
    val_graphs = graphs_subset[:n_val]
    tr_graphs  = graphs_subset[n_val:] or graphs_subset

    samp_w  = torch.tensor(
        1.0 / np.maximum(counts[y_sub[n_val:]], 1.0), dtype=torch.float
    )
    sampler = WeightedRandomSampler(samp_w, num_samples=len(tr_graphs), replacement=True)
    tr_ldr  = DataLoader(tr_graphs, batch_size=cfg["batch_size"], sampler=sampler)
    val_ldr = DataLoader(val_graphs, batch_size=cfg["batch_size"], shuffle=False)

    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
             if cfg["sched_plateau"] else None)

    best_metric, best_state, patience = -1.0, None, 0
    PATIENCE = cfg["early_stop_patience"]

    for epoch in range(cfg["epochs"]):
        model.train()
        ep_loss = 0.0
        for batch in tr_ldr:
            batch          = batch.to(device)
            batch.label_id = batch.label_id.to(device)
            if hasattr(batch, "edge_type"):
                batch.edge_type = batch.edge_type.to(device)
            loss = crit(model(batch), batch.y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item()

        model.eval()
        la, ll = [], []
        with torch.no_grad():
            for vb in val_ldr:
                vb          = vb.to(device)
                vb.label_id = vb.label_id.to(device)
                if hasattr(vb, "edge_type"):
                    vb.edge_type = vb.edge_type.to(device)
                la.append(model(vb).detach().cpu())
                ll.append(vb.y.detach().cpu())

        metric = 0.0
        if la:
            preds = torch.cat(la).argmax(dim=1).numpy()
            lbls  = torch.cat(ll).numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = f1_score(lbls, preds, average="macro", zero_division=0)

        if sched:
            sched.step(metric)

        if metric > best_metric:
            best_metric, best_state, patience = metric, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  [{tag}] early stop ep {epoch+1} | best_macro_f1={best_metric:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  [{tag}] ep {epoch+1:4d} | loss={ep_loss:.4f} | macro_f1={metric:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def probs_from_model(model, graphs, device):
    from torch_geometric.loader import DataLoader
    if model is None or not graphs:
        return np.full((len(graphs), 2), 0.5, dtype=np.float32)
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=64, shuffle=False):
        batch          = batch.to(device)
        batch.label_id = batch.label_id.to(device)
        if hasattr(batch, "edge_type"):
            batch.edge_type = batch.edge_type.to(device)
        out.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tv.set_seed(GAT6_CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] 6-layer GAT — controlled depth comparison vs Devign (6 GGRN rounds)")

    out_results_path = DATA_DIR / "gat6layer_binary_results.json"
    out_ckpt_path    = DATA_DIR / "gat6layer_binary_checkpoint.pt"
    if out_results_path.exists() or out_ckpt_path.exists():
        print(f"\n[error] Output file(s) already exist:")
        if out_results_path.exists():
            print(f"        {out_results_path}")
        if out_ckpt_path.exists():
            print(f"        {out_ckpt_path}")
        print("[error] Refusing to overwrite. Rename/move the existing file(s) first.")
        raise SystemExit(1)

    print("\n[step] Loading dataset (uses cache if available)...")
    featurizer = tv.NodeFeaturizer(CONFIG, device)
    graphs, y_all = tv.load_dataset(DATA_DIR, featurizer, verbose=True,
                                     cache_dir=CACHE_DIR)

    base_dim = graphs[0].x.shape[1]
    vocab = max(len(tv.NodeFeaturizer._label_vocab), 512)
    num_relations = tv.MAX_RELATIONS
    print(f"[info] base_dim={base_dim} | vocab={vocab}")

    # Identical split logic/seed to training.py and devign.py — same test
    # set across all three comparisons (Stage 1 GAT-3, Devign, GAT-6).
    from sklearn.model_selection import train_test_split
    strat_keys = np.array([
        f"{g.provenance}-{int(g.owasp_class.item())}-{int(g.is_vulnerable.item())}"
        for g in graphs
    ])
    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2,
            random_state=CONFIG["seed"], stratify=strat_keys,
        )
    except ValueError:
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2, random_state=CONFIG["seed"]
        )

    tr_graphs = [graphs[i] for i in tr_idx]
    te_graphs = [graphs[i] for i in te_idx]
    print(f"[info] train={len(tr_graphs)} | test={len(te_graphs)}")

    def to_binary(gs):
        out = []
        for g in gs:
            g2 = g.clone()
            g2.y = torch.tensor(1 if int(g.y.item()) > 0 else 0, dtype=torch.long)
            out.append(g2)
        return out

    bin_tr = to_binary(tr_graphs)
    bin_te = to_binary(te_graphs)

    print("\n" + "="*70)
    print("  6-LAYER GAT — binary-only, depth-matched to Devign")
    print("  Same train/test split, same featurization as Stage 1 and Devign")
    print("="*70)

    model = train_gat_deep(bin_tr, base_dim, vocab, GAT6_CONFIG, device,
                           num_relations, tag="gat6")

    if model is None:
        print("[error] Training failed — no graphs?")
        raise SystemExit(1)

    probs  = probs_from_model(model, bin_te, device)
    y_pred = probs.argmax(axis=1)
    y_true = np.array([int(g.y.item()) for g in bin_te])

    print("\n--- 6-Layer GAT — Binary Classification Report ---")
    report = classification_report(
        y_true, y_pred, target_names=["NON_VULNERABLE", "VULNERABLE"],
        digits=4, zero_division=0,
    )
    print(report)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  6-Layer GAT Macro F1: {macro_f1:.4f}")
    print(f"  (compare: Stage 1 3-layer GAT ≈ 0.65, Devign 6-layer GGRN = 0.7101)")

    with open(out_results_path, "w") as f:
        json.dump({
            "macro_f1": float(macro_f1),
            "n_test": len(bin_te),
            "n_layers": GAT6_CONFIG["n_layers"],
            "note": ("Controlled depth comparison: 6-layer attention-based GAT "
                     "vs 3-layer Stage 1 GAT vs 6-round Devign GGRN, all on "
                     "the identical binary train/test split."),
        }, f, indent=2)
    print(f"  Results → {out_results_path}")

    torch.save({"model_state_dict": model.state_dict(), "base_dim": base_dim,
               "vocab": vocab}, out_ckpt_path)
    print(f"  Checkpoint → {out_ckpt_path}")
    print("="*70)
    print("[done]")
