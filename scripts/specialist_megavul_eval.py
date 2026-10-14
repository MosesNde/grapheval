"""
specialist_megavul_eval.py
==========================
Evaluates each Stage 2 one-vs-rest OWASP specialist INDIVIDUALLY on
MegaVul, computing precision, recall, F1, and AUC for each specialist
on its own positive class.

Motivation:
  The paper's original claim that "specialists generalize to real-world
  code with 99-100% recall on MegaVul" used recall alone on a one-vs-rest
  model, which is weak evidence — a specialist biased toward always
  predicting "yes" trivially achieves 100% recall. This script computes
  the full set of binary classification metrics for each specialist:
    - Precision (how many of its "yes" predictions are correct)
    - Recall    (how many of its true positives it catches)
    - F1        (harmonic mean of precision and recall)
    - AUC-ROC   (area under the ROC curve, threshold-independent)

  Each specialist k is evaluated in binary mode:
    positive class  = MegaVul samples whose OWASP label is k
    negative class  = all OTHER MegaVul samples (other OWASP categories)

  This is the natural one-vs-rest evaluation and directly addresses
  whether each specialist's confident-score distribution on real-world
  code reflects genuine learned structure (high precision + AUC) rather
  than a trivial "predict yes" bias (high recall, low precision/AUC).

  Note: only classes with >=30 MegaVul samples are included in the
  detailed report, matching the threshold used elsewhere in the pipeline.

OUTPUT FILES (new, never overwrites existing):
    <DATA_DIR>/specialist_megavul_eval.json
    (human-readable per-specialist metrics table printed to stdout)

Usage:
    python specialist_megavul_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv

CONFIG          = tv.CONFIG
DATA_DIR        = Path(CONFIG["DATA_DIR"])
STAGE2_DIR      = Path(CONFIG["STAGE2_DIR"])
MEGAVUL_DIR     = Path(CONFIG.get("MEGAVUL_DIR", "/home/moses/Megavul_testset"))
CACHE_DIR       = Path(CONFIG.get("CACHE_DIR", DATA_DIR / "graph_cache"))
SPECIALISTS_DIR = STAGE2_DIR / "stage2_specialists"
MIN_SAMPLES     = 30   # minimum MegaVul samples per class to report

if __name__ == "__main__":
    tv.set_seed(CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] Specialist MegaVul evaluation — per-specialist precision/recall/F1/AUC")
    print(f"[info] Specialists: {SPECIALISTS_DIR}")
    print(f"[info] MegaVul:     {MEGAVUL_DIR}")

    out_path = DATA_DIR / "specialist_megavul_eval.json"
    if out_path.exists():
        print(f"\n[error] Output already exists — refusing to overwrite: {out_path}")
        raise SystemExit(1)

    # ── Load MegaVul ──────────────────────────────────────────────────────
    print("\n[step] Loading MegaVul test set...")
    featurizer = tv.NodeFeaturizer(CONFIG, device)
    graphs, labels = tv.load_megavul_test_set(
        MEGAVUL_DIR, featurizer, verbose=True, cache_dir=CACHE_DIR
    )
    labels = np.array(labels, dtype=np.int64)
    print(f"[info] Loaded {len(graphs)} MegaVul graphs")

    if len(graphs) == 0:
        print("[error] No MegaVul graphs loaded — check MEGAVUL_DIR path.")
        raise SystemExit(1)

    # Clamp label_ids defensively
    vocab = max(len(tv.NodeFeaturizer._label_vocab), 512)
    for g in graphs:
        if hasattr(g, "label_id") and g.label_id.numel() > 0:
            g.label_id = g.label_id.clamp(0, vocab - 1)

    base_dim     = graphs[0].x.shape[1]
    num_relations = tv.MAX_RELATIONS

    # ── Load and evaluate each specialist individually ────────────────────
    print("\n[step] Evaluating each specialist individually on MegaVul...")

    N_OWASP = 10
    results = {}

    for cls in range(1, N_OWASP + 1):
        spec_path = SPECIALISTS_DIR / f"specialist_{cls:02d}.pt"
        owasp_name = tv.OWASP_NAMES.get(cls, f"A{cls:02d}")

        if not spec_path.exists():
            print(f"  [skip] specialist_{cls:02d}.pt not found")
            continue

        # Samples whose true label is cls = positive class for this specialist
        pos_mask = (labels == cls)
        n_pos = int(pos_mask.sum())

        if n_pos < MIN_SAMPLES:
            print(f"  [skip] {owasp_name}: only {n_pos} samples (<{MIN_SAMPLES}) — excluded")
            continue

        # Load specialist
        ckpt = torch.load(spec_path, map_location=device, weights_only=False)
        model = tv.build_model(
            ckpt["backbone"], ckpt["base_dim"], ckpt["vocab"], CONFIG,
            num_relations, num_classes=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Run inference on ALL MegaVul graphs
        probs = tv.probs_from_model(model, graphs, device, num_classes=2)
        scores   = probs[:, 1]           # P(vulnerable for class k)
        preds    = (scores >= 0.5).astype(int)
        y_binary = pos_mask.astype(int)  # 1 = this specialist's positive class

        del model
        torch.cuda.empty_cache()

        # Compute metrics
        prec = precision_score(y_binary, preds, zero_division=0)
        rec  = recall_score(y_binary, preds, zero_division=0)
        f1   = f1_score(y_binary, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_binary, scores)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_binary, scores)
        except ValueError:
            ap = float("nan")

        n_neg    = int((~pos_mask).sum())
        n_pred_pos = int(preds.sum())

        results[owasp_name] = {
            "owasp_class":   cls,
            "n_positive":    n_pos,
            "n_negative":    n_neg,
            "n_predicted_positive": n_pred_pos,
            "precision":     float(prec),
            "recall":        float(rec),
            "f1":            float(f1),
            "auc_roc":       float(auc),
            "avg_precision": float(ap),
        }

        print(f"  {owasp_name:<50}  "
              f"n+={n_pos:4d}  n-={n_neg:4d}  "
              f"prec={prec:.4f}  rec={rec:.4f}  "
              f"F1={f1:.4f}  AUC={auc:.4f}  AP={ap:.4f}")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SPECIALIST-LEVEL MEGAVUL EVALUATION (each specialist evaluated individually)")
    print(f"  Positive class = samples belonging to that specialist's OWASP category")
    print(f"  Negative class = all other MegaVul samples")
    print(f"  Threshold = 0.5 for precision/recall/F1; AUC/AP are threshold-free")
    print(f"{'='*90}")
    print(f"  {'Category':<50} {'n+':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'AP':>7}")
    print(f"  {'-'*88}")
    for name, r in results.items():
        print(f"  {name:<50} {r['n_positive']:>5} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {r['auc_roc']:>7.4f} "
              f"{r['avg_precision']:>7.4f}")
    print(f"{'='*90}")

    # Summary averages (macro across reported classes)
    if results:
        precs = [r["precision"] for r in results.values()]
        recs  = [r["recall"]    for r in results.values()]
        f1s   = [r["f1"]        for r in results.values()]
        aucs  = [r["auc_roc"]   for r in results.values() if not np.isnan(r["auc_roc"])]
        aps   = [r["avg_precision"] for r in results.values() if not np.isnan(r["avg_precision"])]
        print(f"  {'Macro average':<50} {'':>5} "
              f"{np.mean(precs):>7.4f} {np.mean(recs):>7.4f} "
              f"{np.mean(f1s):>7.4f} {np.mean(aucs):>7.4f} "
              f"{np.mean(aps):>7.4f}")
        print(f"{'='*90}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "description": (
            "Per-specialist binary evaluation on MegaVul. Each specialist k "
            "is evaluated with its own OWASP category as the positive class "
            "and all other MegaVul samples as negatives. Metrics at threshold "
            "0.5; AUC-ROC and AP are threshold-free. Classes with <30 MegaVul "
            "samples are excluded."
        ),
        "min_samples_threshold": MIN_SAMPLES,
        "n_megavul_total": len(graphs),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results → {out_path}")
    print("[done]")
