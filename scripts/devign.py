"""
devign.py
=========
Standalone Devign (Zhou et al., NeurIPS 2019) binary baseline.

"Devign: Effective Vulnerability Identification by Learning Comprehensive
Program Semantics via Graph Neural Networks"

Implements the two components from the original paper:
  1. GGRN  — Gated Graph Recurrent Network. Iteratively propagates node
             information along graph edges using a GRU update rule
             (torch_geometric's GatedGraphConv implements exactly this).
  2. Conv Module — concatenates the ORIGINAL node features with the GGRN
             output, applies two parallel 1D-CNN branches (one over the
             concatenated features, one over the GGRN output alone),
             multiplies their outputs element-wise (a gating mechanism),
             and average-pools over nodes to produce the graph-level
             representation.

Devign was designed and validated ONLY for binary vulnerable/not-vulnerable
classification — there is no published multiclass variant. This script
trains and evaluates it strictly as a binary classifier, on the SAME
train/test split used for Stage 1 in training.py, collapsed to binary.
This addresses reviewer R1.3's request for a Devign baseline comparison
within a feasible timeframe (a full multiclass Devign retraining attempt
would take days and would not match how Devign was actually designed).

THIS SCRIPT DOES NOT MODIFY training.py OR ANY EXISTING OUTPUT FILE.
It imports data-loading / featurization utilities from training.py
read-only, and writes its own results to NEW filenames only:
    <DATA_DIR>/devign_baseline_results.json
    <DATA_DIR>/devign_baseline_checkpoint.pt

Usage:
    python devign.py
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
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_dense_batch

# ─────────────────────────────────────────────────────────────────────────────
# Import shared utilities from training.py WITHOUT modifying it.
# Assumes training.py lives in the same directory (e.g. /home/moses/scripts/).
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv  # noqa: E402  (must come after sys.path manipulation)

CONFIG   = tv.CONFIG
DATA_DIR = Path(CONFIG["DATA_DIR"])
CACHE_DIR = Path(CONFIG.get("CACHE_DIR", DATA_DIR / "graph_cache"))

# Devign-specific hyperparameters (kept separate from training.py's CONFIG
# so nothing there needs to change).
DEVIGN_CONFIG = {
    "devign_hidden_dim":  200,
    "devign_ggrn_layers": 6,
    "devign_conv_kernel": 3,
    "devign_dropout":     0.2,
    "lr":                 CONFIG.get("lr", 1e-4),
    "weight_decay":       CONFIG.get("weight_decay", 1e-5),
    "batch_size":         CONFIG.get("batch_size", 32),
    "epochs":             CONFIG.get("epochs", 200),
    "val_frac":           CONFIG.get("val_frac", 0.1),
    "early_stop_patience": CONFIG.get("early_stop_patience", 50),
    "sched_plateau":      CONFIG.get("sched_plateau", True),
    "focal_gamma":        CONFIG.get("focal_gamma", 2.0),
    "seed":               CONFIG.get("seed", 42),
}


# ─────────────────────────────────────────────────────────────────────────────
# DEVIGN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class DevignBackbone(nn.Module):
    """
    Faithful re-implementation of the Devign architecture (Zhou et al. 2019).

    Differences from training.py's GAT/GCN/GIN backbones, by design:
      - Message passing: gated GRU propagation (GGRN) instead of attention
        or convolution. This matches Devign's actual mechanism rather than
        approximating it with a layer already used elsewhere.
      - Readout: gated dual-Conv1d pooling instead of attention-pool or
        mean-pool, matching the original Conv Module exactly.
      - No label-embedding fusion before the GNN layers — Devign's original
        formulation feeds node features directly into the GGRN. To stay
        close to the published architecture we do the same here
        (GraphCodeBERT embedding only); the label-id signal used in
        training.py's backbones is intentionally omitted for fidelity.
    """
    def __init__(self, base_dim, hidden_dim, num_layers, num_classes,
                 conv_kernel=3, dropout=0.2):
        super().__init__()
        if num_classes != 2:
            print(f"[warn] DevignBackbone is binary-only per the original "
                  f"paper; got num_classes={num_classes}. Proceeding, but "
                  f"this is NOT how Devign was designed or validated.")

        self.input_proj = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.ReLU(),
        )

        # GGRN: gated graph propagation. torch_geometric's GatedGraphConv
        # implements exactly this — `num_layers` rounds of GRU-gated message
        # passing along edges, sharing weights across rounds.
        self.ggrn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # Conv Module (readout), per Devign Figure 2 / Section 3.3:
        #   Z = concat([X, GGRN(X)])
        #   Y_1 = ReLU(Conv1d(Z))             "branch with full context"
        #   Y_2 = ReLU(Conv1d(GGRN(X)))       "branch with propagated features"
        #   gated = Y_1 * Y_2
        #   graph_repr = AvgPool_nodes(gated)
        self.x_match_proj = (nn.Identity() if base_dim == hidden_dim
                              else nn.Linear(base_dim, hidden_dim))
        concat_dim = hidden_dim * 2

        self.conv_full = nn.Sequential(
            nn.Conv1d(concat_dim, hidden_dim, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.ReLU(),
        )
        self.conv_ggrn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.cls     = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x_raw    = data.x.float()                              # [N, base_dim]
        x_proj   = self.input_proj(x_raw)                       # [N, H]
        ggrn_out = self.ggrn(x_proj, data.edge_index)           # [N, H]

        x_for_concat = self.x_match_proj(x_raw)                 # [N, H]
        z = torch.cat([x_for_concat, ggrn_out], dim=1)          # [N, 2H]

        # NODE CAP. to_dense_batch pads every graph in the batch out to the
        # largest node count present. This corpus has a median of 57 nodes but
        # a maximum of 9,466, so one outlier turns a 96-graph batch into a
        # 96 x 9466 x 1024 activation and the readout OOMs on a 95GB card --
        # it did, twice, at batch 512 and again at 96. Capping is also what the
        # original Devign implementation does; graphs beyond the cap are
        # truncated, which affects roughly 1% of this corpus (p99 = 1,630).
        _CAP = 2000
        z_dense, mask = to_dense_batch(z, data.batch, max_num_nodes=_CAP)
        ggrn_dense, _ = to_dense_batch(ggrn_out, data.batch, max_num_nodes=_CAP)

        z_dense    = z_dense.transpose(1, 2)     # [B, 2H, N_max]
        ggrn_dense = ggrn_dense.transpose(1, 2)  # [B,  H, N_max]

        y1 = self.conv_full(z_dense)             # [B, H, N_max]
        y2 = self.conv_ggrn(ggrn_dense)          # [B, H, N_max]
        gated = y1 * y2                          # [B, H, N_max]

        mask_f = mask.unsqueeze(1).float()                   # [B, 1, N_max]
        gated  = gated * mask_f
        node_counts = mask_f.sum(dim=2).clamp(min=1)         # [B, 1]
        pooled = gated.sum(dim=2) / node_counts               # [B, H]

        pooled = self.dropout(pooled)
        return self.cls(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (mirrors training.py's train_expert, but model-agnostic to
# the label_id / edge_type fields that DevignBackbone doesn't use)
# ─────────────────────────────────────────────────────────────────────────────
def train_devign(graphs_subset, base_dim, cfg, device, tag="devign"):
    import random
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.loader import DataLoader

    if not graphs_subset:
        return None

    model = DevignBackbone(
        base_dim    = base_dim,
        hidden_dim  = cfg["devign_hidden_dim"],
        num_layers  = cfg["devign_ggrn_layers"],
        num_classes = 2,
        conv_kernel = cfg["devign_conv_kernel"],
        dropout     = cfg["devign_dropout"],
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
            batch = batch.to(device)
            loss  = crit(model(batch), batch.y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item()

        model.eval()
        la, ll = [], []
        with torch.no_grad():
            for vb in val_ldr:
                vb = vb.to(device)
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
def probs_from_devign(model, graphs, device):
    from torch_geometric.loader import DataLoader
    if model is None or not graphs:
        return np.full((len(graphs), 2), 0.5, dtype=np.float32)
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=64, shuffle=False):
        batch = batch.to(device)
        out.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tv.set_seed(DEVIGN_CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] Devign baseline — binary-only (Zhou et al. 2019)")
    print(f"[info] Output files (new, will NOT overwrite existing results):")
    print(f"       {DATA_DIR / 'devign_baseline_results.json'}")
    print(f"       {DATA_DIR / 'devign_baseline_checkpoint.pt'}")

    out_results_path = DATA_DIR / "devign_baseline_results.json"
    out_ckpt_path    = DATA_DIR / "devign_baseline_checkpoint.pt"
    if out_results_path.exists() or out_ckpt_path.exists():
        print(f"\n[error] Output file(s) already exist:")
        if out_results_path.exists():
            print(f"        {out_results_path}")
        if out_ckpt_path.exists():
            print(f"        {out_ckpt_path}")
        print("[error] Refusing to overwrite. Rename/move the existing file(s),")
        print("        or change DEVIGN_RESULTS_SUFFIX below, then re-run.")
        raise SystemExit(1)

    print("\n[step] Loading dataset (uses cache if available)...")
    featurizer = tv.NodeFeaturizer(CONFIG, device)
    graphs, y_all = tv.load_dataset(DATA_DIR, featurizer, verbose=True,
                                     cache_dir=CACHE_DIR)

    base_dim = graphs[0].x.shape[1]
    print(f"[info] base_dim={base_dim}")

    # Same stratified train/test split logic as training.py, same seed —
    # ensures Devign sees the EXACT same split as Stage 1 for a fair
    # apples-to-apples binary comparison.
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
    print("  DEVIGN BASELINE (Zhou et al. 2019) — binary-only")
    print("  Reviewer R1.3: architectural comparison against Stage 1")
    print("  NOTE: Devign has no published multiclass variant; reported")
    print("        here strictly as vulnerable-vs-safe, matching its")
    print("        original design and validation scope.")
    print("="*70)

    model = train_devign(bin_tr, base_dim, DEVIGN_CONFIG, device, tag="devign")

    if model is None:
        print("[error] Training failed — no graphs?")
        raise SystemExit(1)

    probs  = probs_from_devign(model, bin_te, device)
    y_pred = probs.argmax(axis=1)
    y_true = np.array([int(g.y.item()) for g in bin_te])

    print("\n--- Devign Baseline — Binary Classification Report ---")
    report = classification_report(
        y_true, y_pred, target_names=["NON_VULNERABLE", "VULNERABLE"],
        digits=4, zero_division=0,
    )
    print(report)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Devign Macro F1: {macro_f1:.4f}")

    with open(out_results_path, "w") as f:
        json.dump({
            "macro_f1": float(macro_f1),
            "n_test": len(bin_te),
            "note": ("Binary-only baseline. Devign (Zhou et al. 2019) has no "
                     "published multiclass variant; this is the architecture's "
                     "designed and validated scope."),
        }, f, indent=2)
    print(f"  Results → {out_results_path}")

    torch.save({"model_state_dict": model.state_dict(), "base_dim": base_dim},
               out_ckpt_path)
    print(f"  Checkpoint → {out_ckpt_path}")
    print("="*70)
    print("[done]")
