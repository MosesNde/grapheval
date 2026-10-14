"""
megavul_restricted_eval.py
==========================
Restricted-class external evaluation of the Stage 2 pipeline on MegaVul,
reporting macro F1 over only the four well-populated OWASP categories:

    A01 Broken Access Control              (1,095 MegaVul samples)
    A02 Cryptographic Failures             (  232 MegaVul samples)
    A03 Injection                          (  894 MegaVul samples)
    A07 Identification & Auth. Failures    (  163 MegaVul samples)

Excluded from the macro average (too few samples to be meaningful):
    A04 Insecure Design                    (   23 samples)
    A05 Security Misconfiguration          (   35 samples)
    A08 Software/Data Integrity Failures   (   48 samples)
    A09 Logging & Monitoring Failures      (    6 samples)
    A10 Server-Side Request Forgery        (    4 samples)

Evaluations reported:
  1. Zero-shot: original meta-learner, no adaptation
  2. Recalibrated: refitted meta-learner (lgbm_stage2_recalibrated.pkl)
     evaluated on held-out slice

Macro F1 is computed only over {A01, A02, A03, A07} in both cases.
Per-class precision/recall/F1 are also reported for full transparency.

OUTPUT (new, never overwrites existing):
    <DATA_DIR>/megavul_restricted_eval.json

Usage:
    python megavul_restricted_eval.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv

CONFIG          = tv.CONFIG
DATA_DIR        = Path(CONFIG["DATA_DIR"])
STAGE2_DIR      = Path(CONFIG["STAGE2_DIR"])
MEGAVUL_DIR     = Path(CONFIG.get("MEGAVUL_DIR", "/home/moses/Megavul_testset"))
CACHE_DIR       = Path(CONFIG.get("CACHE_DIR", DATA_DIR / "graph_cache"))
SPECIALISTS_DIR = STAGE2_DIR / "stage2_specialists"

# The four well-populated classes to include in the restricted evaluation
RESTRICTED_CLASSES = {1, 2, 3, 7}   # A01, A02, A03, A07

# Map class index -> short OWASP name for reporting
CLASS_NAMES = {
    1: "A01_Broken_Access_Control",
    2: "A02_Cryptographic_Failures",
    3: "A03_Injection",
    7: "A07_Identification_and_Authentication_Failures",
}


def run_stage2_inference(graphs, labels, meta_learner, specialists, device):
    """
    Run the full Stage 2 pipeline: specialists -> meta-learner -> prediction.
    Returns predicted class indices (1-indexed OWASP) for each graph.
    """
    n = len(graphs)
    meta_features = np.zeros((n, 10), dtype=np.float32)

    for cls in range(1, 10 + 1):
        spec_path = SPECIALISTS_DIR / f"specialist_{cls:02d}.pt"
        if not spec_path.exists():
            continue
        ckpt  = torch.load(spec_path, map_location=device, weights_only=False)
        model = tv.build_model(
            ckpt["backbone"], ckpt["base_dim"], ckpt["vocab"],
            CONFIG, tv.MAX_RELATIONS, num_classes=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Clamp label_ids defensively
        vocab = max(len(tv.NodeFeaturizer._label_vocab), 512)
        graphs_clamped = []
        for g in graphs:
            g2 = g.clone()
            if hasattr(g2, "label_id") and g2.label_id.numel() > 0:
                g2.label_id = g2.label_id.clamp(0, vocab - 1)
            graphs_clamped.append(g2)

        probs = tv.probs_from_model(model, graphs_clamped, device, num_classes=2)
        meta_features[:, cls - 1] = probs[:, 1]
        del model
        torch.cuda.empty_cache()

    raw_preds = meta_learner.predict(meta_features)   # 0-indexed classes
    return raw_preds + 1                               # back to 1-indexed


def restricted_report(y_true, y_pred, restricted_classes, class_names, tag):
    """
    Compute and print classification metrics restricted to the specified
    classes. Samples whose true label is not in restricted_classes are
    excluded entirely from the evaluation.
    """
    mask     = np.isin(y_true, list(restricted_classes))
    y_t      = y_true[mask]
    y_p      = y_pred[mask]

    # Remap to contiguous 0-indexed labels for sklearn
    sorted_cls = sorted(restricted_classes)
    remap = {c: i for i, c in enumerate(sorted_cls)}
    y_t_r = np.array([remap[c] for c in y_t])
    y_p_r = np.array([remap.get(c, -1) for c in y_p])

    target_names = [class_names[c] for c in sorted_cls]

    print(f"\n{'='*72}")
    print(f"  {tag}")
    print(f"  Restricted to: {', '.join(class_names[c] for c in sorted_cls)}")
    print(f"  N samples evaluated: {mask.sum()} "
          f"(excluded {(~mask).sum()} samples from other classes)")
    print(f"{'='*72}")
    print(classification_report(
        y_t_r, y_p_r,
        labels=list(range(len(sorted_cls))),
        target_names=target_names,
        digits=4, zero_division=0,
    ))
    macro_f1 = f1_score(y_t_r, y_p_r, average="macro",
                        labels=list(range(len(sorted_cls))), zero_division=0)
    print(f"  Restricted Macro F1 ({', '.join(f'A{c:02d}' for c in sorted_cls)}): "
          f"{macro_f1:.4f}")
    print(f"{'='*72}")

    per_class = {}
    for i, cls in enumerate(sorted_cls):
        cls_mask = y_t_r == i
        if cls_mask.sum() == 0:
            continue
        per_class[class_names[cls]] = {
            "n": int(cls_mask.sum()),
            "f1": float(f1_score(y_t_r == i, y_p_r == i,
                                  average="binary", zero_division=0)),
        }
    return float(macro_f1), per_class


if __name__ == "__main__":
    tv.set_seed(CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] Restricted MegaVul evaluation — classes: "
          f"{', '.join(f'A{c:02d}' for c in sorted(RESTRICTED_CLASSES))}")

    out_path = DATA_DIR / "megavul_restricted_eval.json"
    if out_path.exists():
        print(f"\n[error] Output already exists — refusing to overwrite: {out_path}")
        raise SystemExit(1)

    # ── Load MegaVul ──────────────────────────────────────────────────────
    print("\n[step] Loading MegaVul...")
    featurizer = tv.NodeFeaturizer(CONFIG, device)
    graphs, labels = tv.load_megavul_test_set(
        MEGAVUL_DIR, featurizer, verbose=True, cache_dir=CACHE_DIR
    )
    labels = np.array(labels, dtype=np.int64)
    print(f"[info] Loaded {len(graphs)} MegaVul graphs")

    # ── Load specialists list (reused for both evaluations) ───────────────
    print("\n[step] Loading specialists...")
    specialists = {}
    for cls in range(1, 10 + 1):
        p = SPECIALISTS_DIR / f"specialist_{cls:02d}.pt"
        if p.exists():
            specialists[cls] = p
            print(f"  specialist_{cls:02d}.pt found")
        else:
            print(f"  specialist_{cls:02d}.pt MISSING")

    results = {}

    # ── Evaluation 1: Zero-shot ───────────────────────────────────────────
    print("\n[step] Zero-shot evaluation (original meta-learner)...")
    ml_path = STAGE2_DIR / "lgbm_stage2.pkl"
    if not ml_path.exists():
        print(f"[error] {ml_path} not found")
        raise SystemExit(1)
    with open(ml_path, "rb") as f:
        ml_original = pickle.load(f)

    preds_zeroshot = run_stage2_inference(graphs, labels, ml_original,
                                          specialists, device)
    macro_f1_zs, per_class_zs = restricted_report(
        labels, preds_zeroshot, RESTRICTED_CLASSES, CLASS_NAMES,
        "ZERO-SHOT (original meta-learner, no adaptation)"
    )
    results["zero_shot"] = {"macro_f1_restricted": macro_f1_zs,
                            "per_class": per_class_zs}

    # ── Evaluation 2: Recalibrated ────────────────────────────────────────
    print("\n[step] Recalibrated evaluation (refitted meta-learner)...")
    ml_recal_path = STAGE2_DIR / "lgbm_stage2_recalibrated.pkl"
    if not ml_recal_path.exists():
        print(f"[warn] {ml_recal_path} not found — skipping recalibrated eval")
    else:
        with open(ml_recal_path, "rb") as f:
            ml_recal = pickle.load(f)

        # Use the same held-out 1750-sample slice as the main recalibration
        # experiment (same seed, same split) so numbers are comparable
        rng      = np.random.default_rng(CONFIG["seed"])
        idx_all  = np.arange(len(graphs))
        calib_n  = 750
        perm     = rng.permutation(len(graphs))
        held_idx = perm[calib_n:]   # the 1,750-sample held-out slice

        graphs_held = [graphs[i] for i in held_idx]
        labels_held = labels[held_idx]

        preds_recal = run_stage2_inference(graphs_held, labels_held, ml_recal,
                                           specialists, device)
        macro_f1_rc, per_class_rc = restricted_report(
            labels_held, preds_recal, RESTRICTED_CLASSES, CLASS_NAMES,
            "RECALIBRATED (refitted meta-learner, 1,750-sample held-out)"
        )
        results["recalibrated"] = {
            "macro_f1_restricted": macro_f1_rc,
            "n_held_out": len(graphs_held),
            "per_class": per_class_rc,
        }

    # ── Save ──────────────────────────────────────────────────────────────
    results["restricted_classes"] = sorted(RESTRICTED_CLASSES)
    results["class_names"] = CLASS_NAMES
    results["note"] = (
        "Macro F1 computed only over A01, A02, A03, A07 — the four "
        "MegaVul categories with >=163 samples. A04/A05/A08/A09/A10 "
        "excluded due to insufficient samples (<=48)."
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out_path}")
    print("[done]")
