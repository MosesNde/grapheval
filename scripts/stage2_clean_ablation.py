# stage2_clean_ablation.py
#
# Clean, leakage-free meta-learner comparison for Stage 2 (LightGBM vs MLP
# vs LogisticRegression), evaluated ENTIRELY within the legitimate held-out
# test-set features.
#
# WHY THIS SCRIPT EXISTS
# -----------------------
# ablation_meta_learner.py's Stage 2 "S2 Test" numbers (train meta-learner on
# s2_X_tr_probs.npy, evaluate on s2_X_te_probs.npy) are NOT comparable to
# Table IV's 0.9938 macro F1. s2_X_tr_probs.npy was built by
# export_stage2_meta_features() running each specialist's FINAL, fully-
# trained model on its OWN training graphs — in-sample, overconfident scores,
# not the out-of-fold probabilities the production meta-learner (lgbm_stage2.pkl)
# was actually trained on. The resulting train/test feature-distribution
# mismatch tanks every meta-learner's test performance regardless of which
# one is used, which is why LightGBM scores ~0.49 there despite being the
# literal same architecture that scores 0.9938 in the main results.
#
# s2_X_te_probs.npy itself has no such problem: every specialist genuinely
# never saw those 1,062 graphs during its own training. So instead of
# train-on-leaked / test-on-clean, this script runs 5-fold cross-validation
# ENTIRELY within the 1,062 clean test samples, for all three meta-learners.
# Every meta-learner is trained and evaluated on features with the same
# (honest) distribution this time, so both the relative ranking AND the
# absolute macro F1 numbers are meaningful and reportable.
#
# Loads only:
#   rebuttal/checkpoints/11class/s2_X_te_probs.npy
#   rebuttal/checkpoints/11class/y_te_s2.npy
# (s2_X_tr_probs.npy / y_tr_s2.npy are intentionally NOT used — that's the
# whole point.)
#
# Meta-learner hyperparameters are kept identical to ablation_meta_learner.py
# so this is a direct, defensible companion to that study, not a different
# comparison.
#
# Usage:
#   python3 stage2_clean_ablation.py
#
# Output:
#   rebuttal/outputs/11class/stage2_clean_ablation.json

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("[warn] lightgbm not installed — skipping LightGBM variant")

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/moses/rebuttal")
CKPT_DIR = DATA_DIR / "checkpoints" / "11class"
SEED     = 42
N_FOLDS  = 5
HIDDEN_DIM = 64   # matches Stage 2's hidden_dim in ablation_meta_learner.py

OWASP_NAMES_S2 = {
    0: "A01_Broken_Access_Control",
    1: "A02_Cryptographic_Failures",
    2: "A03_Injection",
    3: "A04_Insecure_Design",
    4: "A05_Security_Misconfiguration",
    5: "A06_Vulnerable_and_Outdated_Components",
    6: "A07_Identification_and_Authentication_Failures",
    7: "A08_Software_and_Data_Integrity_Failures",
    8: "A09_Security_Logging_and_Monitoring_Failures",
    9: "A10_Server-Side_Request_Forgery",
}
NUM_CLASSES_S2 = 10


# ── MLP definition — identical to ablation_meta_learner.py ────────────────────

class MLPMeta(nn.Module):
    """2-hidden-layer MLP on GNN OOF probability vectors. Mirrors a standard
    GNN readout MLP: two hidden layers with BatchNorm + ReLU + Dropout."""
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


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
              num_classes: int, hidden_dim: int = HIDDEN_DIM,
              epochs: int = 300, lr: float = 1e-3, seed: int = SEED) -> np.ndarray:
    """Train MLP with early stopping (10% internal val split carved out of
    X_tr only — never touches X_val); return predictions on X_val."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPMeta(X_tr.shape[1], hidden_dim, num_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1.0)),
                     dtype=torch.float32).to(device)

    # Guard: with only 5 folds on 1,062 samples, some inner-fold training
    # partitions could be too small for a stratified internal split if a
    # class is very rare. Fall back to a non-stratified split if needed.
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_i, vl_i = next(sss.split(X_tr, y_tr))
    except ValueError:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_tr))
        cut = max(1, int(0.1 * len(X_tr)))
        vl_i, tr_i = idx[:cut], idx[cut:]

    Xtr = torch.tensor(X_tr[tr_i], dtype=torch.float32).to(device)
    ytr = torch.tensor(y_tr[tr_i], dtype=torch.long).to(device)
    Xvl = torch.tensor(X_tr[vl_i], dtype=torch.float32).to(device)
    yvl = torch.tensor(y_tr[vl_i], dtype=torch.long).to(device)

    best_f1, best_state, patience, PATIENCE = -1.0, None, 0, 25

    for _ in range(epochs):
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
            vl_f1 = f1_score(yvl.cpu().numpy(), vl_pred, average="macro", zero_division=0)
        sched.step(vl_f1)

        if vl_f1 > best_f1:
            best_f1, best_state, patience = vl_f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(Xv).argmax(dim=1).cpu().numpy()
    return preds


# ── LightGBM / LogReg fit-predict helpers, identical hyperparameters ──────────

def lgb_fit_predict(Xtr, ytr, Xval):
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="multiclass", num_class=NUM_CLASSES_S2,
        n_jobs=-1, class_weight="balanced",
        min_child_samples=max(1, len(ytr) // 100),
        num_leaves=min(31, len(ytr) // 20),
        random_state=SEED, verbose=-1,
    )
    m.fit(Xtr, ytr)
    return m.predict(Xval)


def logreg_fit_predict(Xtr, ytr, Xval):
    m = LogisticRegression(max_iter=2000, class_weight="balanced",
                           solver="lbfgs", random_state=SEED)
    m.fit(Xtr, ytr)
    return m.predict(Xval)


# ── Clean within-test-set CV ──────────────────────────────────────────────────

def cv_within_test(X: np.ndarray, y: np.ndarray, fit_predict_fn, name: str) -> dict:
    """5-fold stratified CV entirely within X/y (the clean test-set features).
    Returns out-of-fold predictions covering every sample exactly once, plus
    the resulting metrics — every meta-learner sees the identical, honest
    feature distribution for both its training folds and its held-out fold."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_pred = np.zeros(len(y), dtype=np.int64)

    for fold, (tri, vli) in enumerate(skf.split(X, y)):
        y_pred[vli] = fit_predict_fn(X[tri], y[tri], X[vli])

    present = sorted(set(y.tolist()))
    names   = [OWASP_NAMES_S2[i] for i in present]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
        report   = classification_report(y, y_pred, labels=present, target_names=names,
                                          digits=4, zero_division=0, output_dict=True)
        report_str = classification_report(y, y_pred, labels=present, target_names=names,
                                            digits=4, zero_division=0)

    acc = float(accuracy_score(y, y_pred))

    print(f"\n{'─'*62}")
    print(f"  STAGE 2 (clean, within-test CV) — {name}")
    print(f"{'─'*62}")
    print(report_str)
    print(f"  Macro F1 = {macro_f1:.4f}  |  Accuracy = {acc:.4f}")

    return {"macro_f1": macro_f1, "accuracy": acc, "report": report}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    x_path = CKPT_DIR / "s2_X_te_probs.npy"
    y_path = CKPT_DIR / "y_te_s2.npy"
    if not x_path.exists() or not y_path.exists():
        raise SystemExit(f"[fatal] Missing {x_path} or {y_path}. "
                          f"Run training.py's export_stage2_meta_features step first.")

    X = np.load(str(x_path))
    y = np.load(str(y_path))
    print(f"Stage 2 — clean ablation: X={X.shape}  y={y.shape}")
    print(f"y dist: {dict(sorted((int(k), int(v)) for k, v in zip(*np.unique(y, return_counts=True))))}")
    print(f"NOTE: s2_X_tr_probs.npy / y_tr_s2.npy are intentionally NOT loaded.")
    print(f"      Only the {len(y)} legitimate held-out test samples are used,")
    print(f"      via {N_FOLDS}-fold CV entirely within them.")

    results = {}

    if _HAS_LGB:
        results["LightGBM"] = cv_within_test(X, y, lgb_fit_predict, "LightGBM (proposed)")
    else:
        print("[warn] Skipping LightGBM — not installed")

    def mlp_fit_predict(Xtr, ytr, Xval):
        return train_mlp(Xtr, ytr, Xval, num_classes=NUM_CLASSES_S2, hidden_dim=HIDDEN_DIM, seed=SEED)

    results["MLP"] = cv_within_test(X, y, mlp_fit_predict,
                                     f"MLP [2x{HIDDEN_DIM}, BN+ReLU+Dropout=0.3] (ablation)")
    results["LogReg"] = cv_within_test(X, y, logreg_fit_predict, "Logistic Regression (baseline)")

    print(f"\n{'='*62}")
    print("  STAGE 2 — CLEAN ABLATION SUMMARY (Macro F1, higher is better)")
    print(f"  {N_FOLDS}-fold CV entirely within the {len(y)} legitimate held-out")
    print("  test samples — no train/test feature-distribution mismatch.")
    print(f"{'='*62}")
    labels = [("LightGBM", "proposed"), ("MLP", "ablation"), ("LogReg", "baseline")]
    for name, note in labels:
        if name in results:
            print(f"  {name:<12} macro_f1={results[name]['macro_f1']:.4f}  "
                  f"accuracy={results[name]['accuracy']:.4f}  ← {note}")
    print(f"{'='*62}")

    out = {
        "description": (
            "Leakage-free Stage 2 meta-learner ablation: 5-fold CV entirely "
            "within the legitimate held-out test-set specialist features "
            "(s2_X_te_probs.npy), avoiding the train/test feature-distribution "
            "mismatch present in s2_X_tr_probs.npy (built from each "
            "specialist's in-sample final model, not OOF). Hyperparameters "
            "identical to ablation_meta_learner.py."
        ),
        "n_samples": int(len(y)),
        "n_folds": N_FOLDS,
        "results": results,
    }
    out_path = DATA_DIR / "outputs" / "11class" / "stage2_clean_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved -> {out_path}")
    print("[done]")


if __name__ == "__main__":
    main()
