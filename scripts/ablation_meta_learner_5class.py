"""
ablation_meta_learner.py
========================
Ablation study: LightGBM vs MLP vs LogisticRegression as Stage 1
and Stage 2 meta-learners on frozen GNN OOF probability features.

Addresses reviewer comment:
  "The use of LightGBM as a meta-classifier is unconventional.
   Provide an ablation study comparing this to a standard MLP readout."

No GNN retraining — loads pre-computed OOF arrays from checkpoints/.
All three meta-learners see identical features: a fair comparison.

Prerequisites:
  Run training_5class_v2.py at least once to generate:
    rebuttal/checkpoints/oof_X_tr.npy   -- Stage 1 train OOF features
    rebuttal/checkpoints/oof_X_te.npy   -- Stage 1 test  OOF features
    rebuttal/checkpoints/s2_X_tr.npy    -- Stage 2 train specialist features
    rebuttal/checkpoints/s2_X_te.npy    -- Stage 2 test  specialist features
    rebuttal/checkpoints/y_tr.npy       -- full train labels
    rebuttal/checkpoints/y_te.npy       -- full test  labels
    rebuttal/checkpoints/y_tr_s2.npy    -- vulnerable-only train labels (0-indexed)
    rebuttal/checkpoints/y_te_s2.npy    -- vulnerable-only test  labels (0-indexed)

Usage:
    python ablation_meta_learner.py

Output:
    rebuttal/ablation_meta_learner.json
"""

import json, pickle, warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("[warn] lightgbm not installed — skipping LightGBM variant")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/moses/rebuttal")
CKPT_DIR = DATA_DIR / "checkpoints" / "5class"
SEED     = 42

OWASP_NAMES_S1 = {
    0: "NON_VULNERABLE",
    1: "A01_Access_Control",
    2: "A02_A07_Crypto_Auth",
    3: "A03_Injection",
    4: "OTHER_VULNERABLE",
}
OWASP_NAMES_S2 = {
    0: "A01_Access_Control",
    1: "A02_A07_Crypto_Auth",
    2: "A03_Injection",
    3: "OTHER_VULNERABLE",
}
NUM_CLASSES_S1 = 5
NUM_CLASSES_S2 = 4

# ── MLP definition ────────────────────────────────────────────────────────────

class MLPMeta(nn.Module):
    """
    2-hidden-layer MLP on GNN OOF probability vectors.
    Mirrors what a standard GNN readout MLP would look like:
    two hidden layers with BatchNorm + ReLU + Dropout.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray,
              num_classes: int,
              hidden_dim: int = 128,
              epochs: int = 300,
              lr: float = 1e-3,
              seed: int = 42) -> np.ndarray:
    """Train MLP with early stopping; return test-set predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPMeta(X_tr.shape[1], hidden_dim, num_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5)

    # Class weights
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1.0)),
                     dtype=torch.float32).to(device)

    # 10% internal validation for early stopping
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_i, vl_i = next(sss.split(X_tr, y_tr))
    Xtr = torch.tensor(X_tr[tr_i], dtype=torch.float32).to(device)
    ytr = torch.tensor(y_tr[tr_i], dtype=torch.long).to(device)
    Xvl = torch.tensor(X_tr[vl_i], dtype=torch.float32).to(device)
    yvl = torch.tensor(y_tr[vl_i], dtype=torch.long).to(device)

    best_f1, best_state, patience, PATIENCE = -1.0, None, 0, 25

    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(Xtr), ytr, weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl_pred = model(Xvl).argmax(dim=1).cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vl_f1 = f1_score(yvl.cpu().numpy(), vl_pred,
                             average="macro", zero_division=0)
        sched.step(vl_f1)

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(Xte).argmax(dim=1).cpu().numpy()
    return preds


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             class_names: dict) -> dict:
    """Print classification report and return metrics dict."""
    num_classes   = len(class_names)
    present       = sorted(set(y_true.tolist()))
    present_names = [class_names[i] for i in present]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        acc      = float(accuracy_score(y_true, y_pred))
        report   = classification_report(
            y_true, y_pred,
            labels=present, target_names=present_names,
            digits=4, zero_division=0)

    print(f"\n{'─'*62}")
    print(f"  {name}")
    print(f"{'─'*62}")
    print(report)
    print(f"  Macro F1 = {macro_f1:.4f}  |  Accuracy = {acc:.4f}")

    per_class = {}
    for i, cn in class_names.items():
        mask = y_true == i
        if mask.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, r, f, _ = precision_recall_fscore_support(
                (y_true == i).astype(int),
                (y_pred == i).astype(int),
                average="binary", zero_division=0)
        per_class[cn] = {
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f), 4),
            "support":   int(mask.sum()),
        }

    return {"macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "per_class": per_class}


# ── Run one stage ablation ────────────────────────────────────────────────────

def run_ablation(stage_label: str,
                 X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, y_te: np.ndarray,
                 num_classes: int, class_names: dict,
                 hidden_dim: int = 128,
                 lgb_num_class: int = None,
                 use_cv: bool = False) -> dict:
    """
    use_cv=True: evaluate via 5-fold CV on X_tr (avoids in-sample leakage).
    use_cv=False: train on X_tr, evaluate on X_te (standard).
    Stage 2 uses use_cv=True because X_tr_s2 contains in-sample specialist probs.
    """
    lgb_num_class = lgb_num_class or num_classes
    results = {}

    print(f"\n{'='*62}")
    print(f"  {stage_label} ABLATION")
    print(f"  Features: {X_tr.shape[1]}-dim GNN OOF probability vectors")
    if use_cv:
        print(f"  Evaluation: 5-fold cross-validation on train set (avoids leakage)")
    else:
        print(f"  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")
    print(f"  Classes: {num_classes}")
    print(f"{'='*62}")

    def cv_predict(fit_fn):
        """5-fold CV predictions on X_tr to avoid in-sample leakage."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred_cv = np.zeros(len(y_tr), dtype=np.int64)
        for tri, vli in skf.split(X_tr, y_tr):
            m = fit_fn(X_tr[tri], y_tr[tri])
            y_pred_cv[vli] = m(X_tr[vli])
        return y_pred_cv

    # ── LightGBM ──────────────────────────────────────────────────────────
    if _HAS_LGB:
        def lgb_fit_predict(Xtr, ytr):
            m = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multiclass", num_class=lgb_num_class,
                n_jobs=-1, class_weight="balanced",
                min_child_samples=max(1, len(ytr) // 100),
                num_leaves=min(31, len(ytr) // 20),
                random_state=SEED, verbose=-1,
            )
            m.fit(Xtr, ytr)
            return lambda X: m.predict(X)

        if use_cv:
            y_pred = cv_predict(lgb_fit_predict)
            eval_y_true = y_tr
        else:
            gbm = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multiclass", num_class=lgb_num_class,
                n_jobs=-1, class_weight="balanced",
                min_child_samples=max(1, len(y_tr) // 100),
                num_leaves=min(31, len(y_tr) // 20),
                random_state=SEED, verbose=-1,
            )
            gbm.fit(X_tr, y_tr)
            y_pred = gbm.predict(X_te)
            eval_y_true = y_te

        results["LightGBM"] = evaluate(
            f"{stage_label} — LightGBM (proposed)",
            eval_y_true, y_pred, class_names)

    # ── MLP ───────────────────────────────────────────────────────────────
    if use_cv:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred_mlp = np.zeros(len(y_tr), dtype=np.int64)
        for tri, vli in skf.split(X_tr, y_tr):
            preds = train_mlp(X_tr[tri], y_tr[tri], X_tr[vli],
                              num_classes=num_classes,
                              hidden_dim=hidden_dim, seed=SEED)
            y_pred_mlp[vli] = preds
        eval_y_true_mlp = y_tr
    else:
        y_pred_mlp = train_mlp(X_tr, y_tr, X_te,
                               num_classes=num_classes,
                               hidden_dim=hidden_dim, seed=SEED)
        eval_y_true_mlp = y_te

    results["MLP"] = evaluate(
        f"{stage_label} — MLP [2×{hidden_dim}, BN+ReLU+Dropout=0.3] (ablation)",
        eval_y_true_mlp, y_pred_mlp, class_names)

    # ── Logistic Regression ───────────────────────────────────────────────
    if use_cv:
        from sklearn.model_selection import cross_val_predict
        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                solver="lbfgs", random_state=SEED)
        y_pred_lr = cross_val_predict(lr, X_tr, y_tr, cv=5)
        eval_y_true_lr = y_tr
    else:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                solver="lbfgs", random_state=SEED)
        lr.fit(X_tr, y_tr)
        y_pred_lr = lr.predict(X_te)
        eval_y_true_lr = y_te

    results["LogReg"] = evaluate(
        f"{stage_label} — Logistic Regression (baseline)",
        eval_y_true_lr, y_pred_lr, class_names)

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(s1: dict, s2: dict) -> None:
    print("\n" + "=" * 62)
    print("  ABLATION SUMMARY — Macro F1 (higher is better)")
    print("=" * 62)
    print(f"  {'Meta-Learner':<30} {'Stage 1':>10} {'Stage 2':>10}  Notes")
    print("  " + "─" * 58)
    labels = [("LightGBM", "proposed"), ("MLP", "ablation"),
               ("LogReg",  "baseline")]
    for name, note in labels:
        s1f = s1.get(name, {}).get("macro_f1")
        s2f = s2.get(name, {}).get("macro_f1")
        s1s = f"{s1f:.4f}" if s1f is not None else "  N/A  "
        s2s = f"{s2f:.4f}" if s2f is not None else "  N/A  "
        print(f"  {name:<30} {s1s:>10} {s2s:>10}  ← {note}")
    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ablation study: meta-learner comparison on frozen GNN OOF features")
    print(f"Checkpoint dir : {CKPT_DIR}\n")

    # Verify required files
    required = ["oof_X_tr.npy", "oof_X_te.npy",
                "s2_X_tr_probs.npy",  "s2_X_te_probs.npy",
                "y_tr.npy",     "y_te.npy",
                "y_tr_s2.npy",  "y_te_s2.npy"]
    missing = [f for f in required if not (CKPT_DIR / f).exists()]
    if missing:
        print(f"[error] Missing checkpoint files: {missing}")
        print("Run training_5class_v2.py first to generate them.")
        raise SystemExit(1)

    # Load everything
    X_tr    = np.load(str(CKPT_DIR / "oof_X_tr.npy"))
    X_te    = np.load(str(CKPT_DIR / "oof_X_te.npy"))
    y_tr    = np.load(str(CKPT_DIR / "y_tr.npy"))
    y_te    = np.load(str(CKPT_DIR / "y_te.npy"))
    X_tr_s2 = np.load(str(CKPT_DIR / "s2_X_tr_probs.npy"))
    X_te_s2 = np.load(str(CKPT_DIR / "s2_X_te_probs.npy"))
    y_tr_s2 = np.load(str(CKPT_DIR / "y_tr_s2.npy"))
    y_te_s2 = np.load(str(CKPT_DIR / "y_te_s2.npy"))

    print(f"Stage 1: X_tr={X_tr.shape}  X_te={X_te.shape}")
    print(f"         y_tr dist: {dict(sorted((k,int(v)) for k,v in zip(*np.unique(y_tr,return_counts=True))))}")
    print(f"Stage 2: X_tr={X_tr_s2.shape}  X_te={X_te_s2.shape}")
    print(f"         y_tr dist: {dict(sorted((k,int(v)) for k,v in zip(*np.unique(y_tr_s2,return_counts=True))))}")

    # Stage 1 ablation
    s1_results = run_ablation(
        "STAGE 1",
        X_tr, y_tr, X_te, y_te,
        num_classes=NUM_CLASSES_S1,
        class_names=OWASP_NAMES_S1,
        hidden_dim=128,
    )

    # Stage 2 ablation — use CV on train for fair comparison,
    # plus end-to-end test evaluation for paper reporting
    s2_results = run_ablation(
        "STAGE 2",
        X_tr_s2, y_tr_s2, X_te_s2, y_te_s2,
        num_classes=NUM_CLASSES_S2,
        class_names=OWASP_NAMES_S2,
        hidden_dim=64,
        lgb_num_class=NUM_CLASSES_S2,
        use_cv=True,
    )

    # Also run end-to-end test evaluation for Stage 2 (train on full X_tr_s2)
    print("\n" + "="*62)
    print("  STAGE 2 — End-to-End Test Set Evaluation")
    print("  (Train on full X_tr_s2, evaluate on held-out X_te_s2)")
    print("="*62)
    s2_test_results = run_ablation(
        "STAGE 2 (test)",
        X_tr_s2, y_tr_s2, X_te_s2, y_te_s2,
        num_classes=NUM_CLASSES_S2,
        class_names=OWASP_NAMES_S2,
        hidden_dim=64,
        lgb_num_class=NUM_CLASSES_S2,
        use_cv=False,
    )
    s2_results["_test"] = s2_test_results

    print("\n" + "=" * 62)
    print("  ABLATION SUMMARY — Macro F1 (higher is better)")
    print("=" * 62)
    print(f"  {'Meta-Learner':<30} {'S1 Test':>8} {'S2 CV':>8} {'S2 Test':>8}  Notes")
    print("  " + "─" * 60)
    labels = [("LightGBM", "proposed"), ("MLP", "ablation"), ("LogReg", "baseline")]
    for name, note in labels:
        s1f  = s1_results.get(name, {}).get("macro_f1")
        s2cv = s2_results.get(name, {}).get("macro_f1")
        s2t  = s2_results.get("_test", {}).get(name, {}).get("macro_f1")
        s1s  = f"{s1f:.4f}" if s1f  is not None else "  N/A"
        s2cs = f"{s2cv:.4f}" if s2cv is not None else "  N/A"
        s2ts = f"{s2t:.4f}"  if s2t  is not None else "  N/A"
        print(f"  {name:<30} {s1s:>8} {s2cs:>8} {s2ts:>8}  ← {note}")
    print("=" * 62)
    print("  S1 Test : Stage 1 evaluated on held-out test set")
    print("  S2 CV   : Stage 2 evaluated via 5-fold CV on train")
    print("  S2 Test : Stage 2 evaluated on held-out test set")

    # Save
    out = {
        "description": (
            "Ablation: LightGBM vs MLP vs LogReg on frozen GNN OOF "
            "probability vectors. No GNN retraining — identical features "
            "for all variants. MLP: 2 hidden layers, BatchNorm, ReLU, "
            "Dropout=0.3, early stopping on 10% internal val."
        ),
        "stage1": s1_results,
        "stage2": s2_results,
    }
    out_path = DATA_DIR / "outputs" / "5class" / "ablation_meta_learner.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved → {out_path}")
    print("[done]")
