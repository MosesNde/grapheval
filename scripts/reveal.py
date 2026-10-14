"""
reveal.py
=========
Standalone ReVeal (Chakraborty et al., IEEE TSE 2021) binary baseline.

"Deep Learning based Vulnerability Detection: Are We There Yet?"

ReVeal represents code as a Code Property Graph, uses a Gated Graph Neural
Network (GGNN) to obtain node-level representations, then performs global
graph pooling followed by a Multi-Layer Perceptron (MLP) classifier head
for vulnerable/not-vulnerable prediction.

This is architecturally a simpler relative of Devign: both use GGNN-style
gated message passing, but ReVeal's readout is a plain pooled-MLP rather
than Devign's dual-branch gated Conv Module. Comparing the two isolates
the effect of the READOUT mechanism while holding message passing fixed.

ReVeal, like Devign, was designed and validated ONLY for binary
vulnerable/not-vulnerable classification — no published multiclass variant.
This script trains and evaluates it strictly as a binary classifier, on
the SAME train/test split used for Stage 1 and the Devign baseline.
Framed as an external related-work baseline (not a proposed contribution
of this paper), in the same spirit as the Devign comparison.

THIS SCRIPT DOES NOT MODIFY training.py, devign.py, OR ANY EXISTING OUTPUT.
It imports utilities from training.py read-only, and writes its own
results to NEW filenames only:
    <DATA_DIR>/reveal_baseline_results.json
    <DATA_DIR>/reveal_baseline_checkpoint.pt

Usage:
    python reveal.py
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
from torch_geometric.nn import GatedGraphConv, global_mean_pool

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv  # noqa: E402

CONFIG    = tv.CONFIG
DATA_DIR  = Path(CONFIG["DATA_DIR"])
CACHE_DIR = Path(CONFIG.get("CACHE_DIR", DATA_DIR / "graph_cache"))

REVEAL_CONFIG = {
    "reveal_hidden_dim":   200,   # same as Devign's GGRN hidden dim, for a fair comparison
    "reveal_ggnn_layers":  6,     # matches Devign's 6-round propagation depth
    "reveal_mlp_hidden":   256,   # ReVeal's classifier-head MLP hidden size
    "reveal_dropout":      0.2,
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
# REVEAL ARCHITECTURE
#
# Per Chakraborty et al. 2021 (Section IV-B, "Re-purposing a GGNN-based
# model"): CPG nodes are embedded, passed through a GGNN for several rounds
# of message passing, the resulting node representations are pooled into a
# single graph-level vector, and a standard MLP performs the final
# vulnerable/not-vulnerable classification. No gating, no dual-branch
# convolution, no FiLM conditioning — deliberately the simplest GGNN+readout
# combination among the architectures compared here.
# ─────────────────────────────────────────────────────────────────────────────
class RevealBackbone(nn.Module):
    """
    GGNN message passing + global mean pooling + MLP classifier head,
    matching ReVeal's published architecture.

    Differences from DevignBackbone, by design (this IS the comparison):
      - Readout: plain global mean pooling instead of Devign's gated
        dual-branch Conv Module.
      - Classifier: a standard 2-layer MLP with BatchNorm/ReLU/Dropout,
        rather than a single linear layer over a gated representation.
      - No raw-feature concatenation before pooling — ReVeal pools only the
        GGNN's propagated representations, not a concat of [raw, GGNN(raw)].

    Same as Devign, by design (held constant for fair comparison):
      - GGNN message passing (GatedGraphConv) with matching hidden_dim and
        num_layers, so any performance difference reflects the READOUT
        mechanism, not message-passing capacity.
      - No label-embedding fusion before the GNN layers — GraphCodeBERT
        embedding only, for fidelity to the original paper's formulation
        (ReVeal's original node features are program-analysis-derived
        embeddings, not GraphCodeBERT, but this codebase standardizes on
        GraphCodeBERT across all baselines for a fair, consistent input).
    """
    def __init__(self, base_dim, hidden_dim, num_layers, mlp_hidden,
                 num_classes, dropout=0.2):
        super().__init__()
        if num_classes != 2:
            print(f"[warn] RevealBackbone is binary-only per the original "
                  f"paper; got num_classes={num_classes}. Proceeding, but "
                  f"this is NOT how ReVeal was designed or validated.")

        self.input_proj = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.ReLU(),
        )

        # GGNN: identical mechanism to Devign's GGRN component (gated,
        # GRU-style propagation along edges), held constant so the
        # comparison isolates the readout/classifier difference.
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # ReVeal's classifier head: pooled graph representation → MLP → logits.
        # No gating, no concatenation with raw features — just the GGNN's
        # final propagated state, globally pooled.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def forward(self, data):
        x_proj   = self.input_proj(data.x.float())          # [N, H]
        ggnn_out = self.ggnn(x_proj, data.edge_index)        # [N, H]
        pooled   = global_mean_pool(ggnn_out, data.batch)    # [B, H]
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (mirrors devign.py's train_devign exactly, parameterized over
# RevealBackbone instead of DevignBackbone)
# ─────────────────────────────────────────────────────────────────────────────
def train_reveal(graphs_subset, base_dim, cfg, device, tag="reveal"):
    import random
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.loader import DataLoader

    if not graphs_subset:
        return None

    model = RevealBackbone(
        base_dim    = base_dim,
        hidden_dim  = cfg["reveal_hidden_dim"],
        num_layers  = cfg["reveal_ggnn_layers"],
        mlp_hidden  = cfg["reveal_mlp_hidden"],
        num_classes = 2,
        dropout     = cfg["reveal_dropout"],
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
def probs_from_reveal(model, graphs, device):
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
    tv.set_seed(REVEAL_CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] ReVeal baseline — binary-only (Chakraborty et al. 2021)")
    print(f"[info] Output files (new, will NOT overwrite existing results):")
    print(f"       {DATA_DIR / 'reveal_baseline_results.json'}")
    print(f"       {DATA_DIR / 'reveal_baseline_checkpoint.pt'}")

    out_results_path = DATA_DIR / "reveal_baseline_results.json"
    out_ckpt_path    = DATA_DIR / "reveal_baseline_checkpoint.pt"
    if out_results_path.exists() or out_ckpt_path.exists():
        print(f"\n[error] Output file(s) already exist:")
        if out_results_path.exists():
            print(f"        {out_results_path}")
        if out_ckpt_path.exists():
            print(f"        {out_ckpt_path}")
        print("[error] Refusing to overwrite. Rename/move the existing file(s),")
        print("        then re-run.")
        raise SystemExit(1)

    print("\n[step] Loading dataset (uses cache if available)...")
    featurizer = tv.NodeFeaturizer(CONFIG, device)
    graphs, y_all = tv.load_dataset(DATA_DIR, featurizer, verbose=True,
                                     cache_dir=CACHE_DIR)

    base_dim = graphs[0].x.shape[1]
    print(f"[info] base_dim={base_dim}")

    # Identical split logic/seed to training.py and devign.py — same test
    # set across all baseline comparisons (Stage 1 GAT-3, Devign, GAT-6,
    # ReVeal).
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
    print("  REVEAL BASELINE (Chakraborty et al. 2021) — binary-only")
    print("  Related-work baseline (not a proposed contribution of this paper)")
    print("  NOTE: ReVeal has no published multiclass variant; reported")
    print("        here strictly as vulnerable-vs-safe, matching its")
    print("        original design and validation scope.")
    print("="*70)

    model = train_reveal(bin_tr, base_dim, REVEAL_CONFIG, device, tag="reveal")

    if model is None:
        print("[error] Training failed — no graphs?")
        raise SystemExit(1)

    probs  = probs_from_reveal(model, bin_te, device)
    y_pred = probs.argmax(axis=1)
    y_true = np.array([int(g.y.item()) for g in bin_te])

    print("\n--- ReVeal Baseline — Binary Classification Report ---")
    report = classification_report(
        y_true, y_pred, target_names=["NON_VULNERABLE", "VULNERABLE"],
        digits=4, zero_division=0,
    )
    print(report)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  ReVeal Macro F1: {macro_f1:.4f}")
    print(f"  (compare: Stage 1 3-layer GAT ≈ 0.65, Devign GGRN+Conv = 0.7101,")
    print(f"            6-layer GAT = 0.6769)")

    with open(out_results_path, "w") as f:
        json.dump({
            "macro_f1": float(macro_f1),
            "n_test": len(bin_te),
            "note": ("Binary-only related-work baseline. ReVeal (Chakraborty "
                     "et al. 2021) has no published multiclass variant; this "
                     "is the architecture's designed and validated scope. "
                     "Uses identical GGNN message passing to the Devign "
                     "baseline (same hidden_dim, same num_layers) so any "
                     "performance difference isolates the readout mechanism "
                     "(plain pooled-MLP here vs Devign's gated dual-branch "
                     "Conv Module)."),
        }, f, indent=2)
    print(f"  Results → {out_results_path}")

    torch.save({"model_state_dict": model.state_dict(), "base_dim": base_dim},
               out_ckpt_path)
    print(f"  Checkpoint → {out_ckpt_path}")
    print("="*70)
    print("[done]")
