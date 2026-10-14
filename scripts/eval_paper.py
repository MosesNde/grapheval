# eval_generalist_indistribution.py
#
# Purpose: the external binary number (MegaVul+Devign, via evaluate_external_testset)
# was computed using the `generalist` checkpoint. The in-distribution binary number
# you've been citing (~0.65 macro F1) came from a DIFFERENT model — the 5-fold OOF
# ensemble + meta-learner — whose fold weights were never saved to disk, so it can't
# be reproduced. This script closes that gap WITHOUT retraining anything: it loads
# the already-saved `gnn_checkpoint.pt` (the generalist, with all the metadata needed
# to reconstruct the model + featurizer exactly), reloads the exact 1,997 held-out
# in-distribution test graphs recorded in test_split.json (using the same embedding
# cache, so this is fast — no re-embedding), and reports the SAME binary-collapse
# metric `evaluate_external_testset` already produced for MegaVul/Devign.
#
# Result: a true same-model, in-distribution-vs-external comparison for the binary
# vulnerable/non-vulnerable view.
#
# Usage:
#   Place this file next to training.py (same directory), then:
#   python3 eval_generalist_indistribution.py
#
# Output:
#   Prints a classification_report identical in format to the external eval,
#   and saves rebuttal/generalist_indistribution_eval.json

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Reuse everything from the existing training script — no duplicated logic,
# no risk of the featurizer/model construction silently drifting out of sync.
import training as T


def main():
    device = T.get_device()
    DATA_DIR = Path(T.CONFIG["DATA_DIR"])
    CACHE_DIR = Path(T.CONFIG["CACHE_DIR"]) if T.CONFIG.get("CACHE_DIR") else None

    ckpt_path = DATA_DIR / "gnn_checkpoint.pt"
    if not ckpt_path.exists():
        raise SystemExit(
            f"[fatal] {ckpt_path} not found. This script needs the full "
            f"checkpoint (with vocab/label_vocab/edge_type_map metadata), "
            f"not the bare generalist_checkpoint.pt."
        )

    print(f"[info] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Restore exactly the vocab/edge-type state the model was trained with ──
    # Without this, label_id() would assign DIFFERENT integer ids to node
    # labels than the ones the model's embedding table actually learned,
    # silently corrupting every prediction.
    T.NodeFeaturizer._label_vocab = dict(ckpt["label_vocab"])
    T._EDGE_TYPE_MAP.clear()
    T._EDGE_TYPE_MAP.update(ckpt["edge_type_map"])

    backbone      = ckpt["backbone"]
    base_dim      = ckpt["base_dim"]
    vocab         = ckpt["vocab"]
    num_relations = ckpt["num_relations"]
    num_classes   = ckpt["num_classes"]
    owasp_names   = ckpt["owasp_names"]
    cfg           = ckpt["config"]

    print(f"[info] backbone={backbone} base_dim={base_dim} vocab={vocab} "
          f"num_classes={num_classes}")

    # ── Rebuild the generalist model and load its trained weights ─────────
    model = T.build_model(backbone, base_dim, vocab, cfg, num_relations, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[info] Generalist model rebuilt and weights loaded.")

    # ── Reconstruct the exact in-distribution held-out test set ───────────
    split_path = DATA_DIR / "test_split.json"
    if not split_path.exists():
        raise SystemExit(f"[fatal] {split_path} not found.")

    with open(split_path) as f:
        split_meta = json.load(f)
    print(f"[info] test_split.json: {len(split_meta)} graphs recorded")

    featurizer = T.NodeFeaturizer(cfg, device)

    graphs = []
    missing = 0
    for entry in split_meta:
        dot_path = Path(entry["dot_path"])
        if not dot_path.exists():
            missing += 1
            continue
        d = T.dot_to_pyg(
            dot_path,
            owasp_class=entry["owasp_class"],
            is_vulnerable=entry["is_vulnerable"],
            featurizer=featurizer,
            provenance=entry["provenance"],
            cache_dir=CACHE_DIR,
        )
        if d is not None:
            graphs.append(d)

    print(f"[info] Rebuilt {len(graphs)}/{len(split_meta)} graphs "
          f"({missing} dot files no longer found on disk)")

    if not graphs:
        raise SystemExit("[fatal] No graphs reconstructed — check DATA_DIR/CACHE_DIR paths.")

    # Defensive clamp, same as the external eval path, in case anything's
    # at the edge of the vocab range.
    for g in graphs:
        if hasattr(g, "label_id") and g.label_id.numel() > 0:
            g.label_id = g.label_id.clamp(0, vocab - 1)

    # ── Run inference ───────────────────────────────────────────────────────
    probs = T.probs_from_model(model, graphs, device, num_classes=num_classes)
    y_true_multi = np.array([int(g.y.item()) for g in graphs])
    y_pred_multi = probs.argmax(axis=1)

    # Full 11-class report (generalist's OWN multiclass performance,
    # directly comparable to its own behavior on MegaVul/Devign).
    target_names = [owasp_names.get(i, str(i)) for i in range(num_classes)]
    print(f"\n{'='*70}")
    print("  Generalist — IN-DISTRIBUTION Multiclass Report")
    print(f"  ({len(graphs)} held-out test graphs, same model as external eval)")
    print(f"{'='*70}")
    print(classification_report(y_true_multi, y_pred_multi,
                                 target_names=target_names,
                                 digits=4, zero_division=0))

    # Binary collapse — IDENTICAL method to evaluate_external_testset's
    # `y_pred = (probs.argmax(axis=1) > 0).astype(int)`.
    y_true_bin = (y_true_multi > 0).astype(int)
    y_pred_bin = (y_pred_multi > 0).astype(int)

    print(f"{'='*70}")
    print("  Generalist — IN-DISTRIBUTION Binary Collapse")
    print(f"  (same binary-collapse method as the external MegaVul+Devign eval)")
    print(f"{'='*70}")
    print(classification_report(y_true_bin, y_pred_bin,
                                 target_names=["NON_VULNERABLE", "VULNERABLE"],
                                 digits=4, zero_division=0))

    summary = {
        "n_test": len(graphs),
        "multiclass_macro_f1": float(f1_score(y_true_multi, y_pred_multi,
                                               average="macro", zero_division=0)),
        "binary_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "binary_macro_f1": float(f1_score(y_true_bin, y_pred_bin,
                                           average="macro", zero_division=0)),
    }
    out_path = DATA_DIR / "generalist_indistribution_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved -> {out_path}")
    print(f"{'='*70}\n")
    print("Compare this binary_macro_f1 directly against external_eval.json's "
          "binary_f1 — both numbers now come from the SAME generalist model.")


if __name__ == "__main__":
    main()
