# train_parallel.py
#
# Runs Stage 2 specialist training across 4 GPUs in parallel.
# Each GPU handles a subset of the 10 OWASP specialists.
#
# Usage:
#   python3 train_parallel.py
#
# This script spawns 4 worker processes, one per GPU.
# The shared stage2_checkpoint.npz is updated atomically after each
# specialist so you can monitor progress from the main process.
#
# GPU assignment (10 specialists across 4 GPUs):
#   GPU 0 → A01, A02, A03   (3 specialists, A01 has most samples)
#   GPU 1 → A04, A05, A06
#   GPU 2 → A07, A08
#   GPU 3 → A09, A10
# A01 is the heaviest (796 positives) so GPU 0 gets only 3 to balance load.

import os
import sys
import time
import fcntl
import pickle
import random
import warnings
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Import everything from your training script ───────────────────────────
# Add the scripts directory to path so we can import training
SCRIPT_DIR = Path("/home/moses/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

# We import selectively to avoid re-running __main__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "training", SCRIPT_DIR / "training.py"
)
# We'll use exec-based import to avoid the if __name__ == "__main__" block
_mod_src = (SCRIPT_DIR / "training.py").read_text()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match training.py
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/home/moses/rebuttal")
CACHE_DIR      = Path("/home/moses/graph_cache")
CKPT_PATH      = DATA_DIR / "output" / "allclass" / "stage2_checkpoint.npz"
LOCK_PATH      = DATA_DIR / "output" / "allclass" / "stage2_checkpoint.lock"

# Which specialists each GPU handles (0-indexed, 0=A01 ... 9=A10)
GPU_SPECIALIST_MAP = {
    0: [0, 1, 2],   # A01, A02, A03
    1: [3, 4, 5],   # A04, A05, A06
    2: [6, 7],      # A07, A08
    3: [8, 9],      # A09, A10
}

OWASP_NAMES = {
    0:  "NON_VULNERABLE",
    1:  "A01_Broken_Access_Control",
    2:  "A02_Cryptographic_Failures",
    3:  "A03_Injection",
    4:  "A04_Insecure_Design",
    5:  "A05_Security_Misconfiguration",
    6:  "A06_Vulnerable_and_Outdated_Components",
    7:  "A07_Identification_and_Authentication_Failures",
    8:  "A08_Software_and_Data_Integrity_Failures",
    9:  "A09_Security_Logging_and_Monitoring_Failures",
    10: "A10_Server-Side_Request_Forgery",
}


# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC CHECKPOINT UPDATE
# Uses a file lock so two GPU workers writing simultaneously don't corrupt
# the shared checkpoint.
# ─────────────────────────────────────────────────────────────────────────────
def atomic_checkpoint_update(
    cls_idx: int,
    X_tr_col: np.ndarray,   # shape (n_vuln_tr,)
    X_te_col: np.ndarray,   # shape (n_vuln_te,)
    n_vuln_tr: int,
    n_vuln_te: int,
    n_owasp: int = 10,
) -> None:
    """
    Update one column of the shared checkpoint file atomically.
    Uses an exclusive file lock so concurrent GPU workers don't race.
    """
    lock_path = str(LOCK_PATH)
    ckpt_path = str(CKPT_PATH)

    # Acquire exclusive file lock
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            # Load current state
            if CKPT_PATH.exists():
                ckpt       = np.load(ckpt_path)
                X_tr_s2    = ckpt["X_tr_s2"].astype(np.float32)
                X_te_s2    = ckpt["X_te_s2"].astype(np.float32)
                y_tr_s2    = ckpt["y_tr_s2"]
                y_te_s2    = ckpt["y_te_s2"]
                completed  = set(int(x) for x in ckpt["completed"])
            else:
                X_tr_s2   = np.full((n_vuln_tr, n_owasp), 0.5, dtype=np.float32)
                X_te_s2   = np.full((n_vuln_te, n_owasp), 0.5, dtype=np.float32)
                y_tr_s2   = None
                y_te_s2   = None
                completed = set()

            # Write this specialist's column
            X_tr_s2[:, cls_idx] = X_tr_col
            X_te_s2[:, cls_idx] = X_te_col
            completed.add(cls_idx)

            np.savez(
                ckpt_path,
                X_tr_s2   = X_tr_s2,
                X_te_s2   = X_te_s2,
                y_tr_s2   = y_tr_s2 if y_tr_s2 is not None else np.zeros(n_vuln_tr),
                y_te_s2   = y_te_s2 if y_te_s2 is not None else np.zeros(n_vuln_te),
                completed = np.array(sorted(completed), dtype=np.int64),
            )
            name = OWASP_NAMES.get(cls_idx + 1, f"A{cls_idx+1:02d}")
            print(f"  [ckpt] {name} saved ({len(completed)}/10 complete)", flush=True)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER: trains specialists assigned to one GPU
# ─────────────────────────────────────────────────────────────────────────────
def gpu_worker(
    gpu_id: int,
    specialist_indices: List[int],
    graphs_pkl: str,          # path to pickled (tr_graphs, te_graphs, y_tr, y_te)
    base_dim: int,
    vocab: int,
    cfg: dict,
    num_relations: int,
    log_prefix: str,
) -> None:
    """
    Runs in a separate process. Trains all specialists assigned to gpu_id.
    Writes results atomically to the shared checkpoint.
    """
    # Pin this process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU {gpu_id}] Starting — specialists {specialist_indices} on {device}",
          flush=True)

    # Load data and restore vocab from main process
    with open(graphs_pkl, "rb") as f:
        tr_graphs, te_graphs, y_tr_s2_full, y_te_s2_full, label_vocab = pickle.load(f)

    # Re-import on this process and restore vocab before any model is built
    sys.path.insert(0, str(SCRIPT_DIR))
    import importlib
    import training as tv2
    
    # Restore label vocab from main process
    tv2.NodeFeaturizer._label_vocab = label_vocab
    print(f"[GPU {gpu_id}] Vocab tracking restored: {len(label_vocab)} mapped keys. Allocated vocab dimension: {vocab}", flush=True)

    nonvuln_tr = [g for g in tr_graphs if int(g.y.item()) == 0]
    nonvuln_te = [g for g in te_graphs if int(g.y.item()) == 0]
    vuln_tr    = [g for g in tr_graphs if int(g.y.item()) > 0]
    vuln_te    = [g for g in te_graphs if int(g.y.item()) > 0]

    cls_tr_idx = {c: [i for i, g in enumerate(vuln_tr) if int(g.y.item()) == c]
                  for c in range(1, 11)}
    cls_te_idx = {c: [i for i, g in enumerate(vuln_te) if int(g.y.item()) == c]
                  for c in range(1, 11)}

    n_folds  = min(cfg.get("oof_max_folds", 5), 5)
    spec_cfg = {**cfg, "early_stop_patience": cfg.get("early_stop_patience", 80)}

    def relabel(graphs, new_y):
        out = []
        for g in graphs:
            g2 = g.clone()
            g2.y = torch.tensor(new_y, dtype=torch.long)
            out.append(g2)
        return out

    # Check which are already done
    completed_already = set()
    if CKPT_PATH.exists():
        ckpt = np.load(str(CKPT_PATH))
        completed_already = set(int(x) for x in ckpt["completed"])

    for cls_idx in specialist_indices:
        owasp_class = cls_idx + 1
        name        = OWASP_NAMES.get(owasp_class, f"A{owasp_class:02d}")

        if cls_idx in completed_already:
            print(f"[GPU {gpu_id}] [{name}] already done — skipping", flush=True)
            continue

        pos_tr       = [vuln_tr[i] for i in cls_tr_idx[owasp_class]]
        pos_te       = [vuln_te[i] for i in cls_te_idx[owasp_class]]
        te_positions = cls_te_idx[owasp_class]
        tr_positions = cls_tr_idx[owasp_class]

        print(f"[GPU {gpu_id}] [{name}] pos={len(pos_tr)} neg={len(nonvuln_tr)}",
              flush=True)

        spec_tr_all = relabel(pos_tr, 1) + relabel(nonvuln_tr, 0)
        spec_te_all = relabel(pos_te, 1) + relabel(nonvuln_te, 0)
        spec_tr_y   = np.array([int(g.y.item()) for g in spec_tr_all])
        can_fold    = (np.bincount(spec_tr_y).min() >= 2) and (n_folds >= 2)

        # ── Test-set probabilities ────────────────────────────────────────
        if can_fold:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=cfg["seed"])
            fold_te = []
            for fold, (tri, _) in enumerate(skf.split(
                    np.arange(len(spec_tr_all)), spec_tr_y)):
                m = tv2.train_expert(
                    [spec_tr_all[i] for i in tri],
                    base_dim, vocab, spec_cfg, device, num_relations,
                    tag=f"G{gpu_id}-{name[:5]}-f{fold+1}", num_classes=2,
                )
                fold_te.append(tv2.probs_from_model(m, spec_te_all, device, 2)[:, 1])
                del m; torch.cuda.empty_cache()
            te_probs_full = np.mean(np.stack(fold_te), axis=0)
        else:
            m = tv2.train_expert(spec_tr_all, base_dim, vocab, spec_cfg,
                                 device, num_relations,
                                 tag=f"G{gpu_id}-{name[:5]}", num_classes=2)
            te_probs_full = tv2.probs_from_model(m, spec_te_all, device, 2)[:, 1]
            del m; torch.cuda.empty_cache()

        # Map test probabilities back to vuln_te positions
        X_te_col = np.full(len(vuln_te), 0.5, dtype=np.float32)
        for j, k in enumerate(te_positions):
            X_te_col[k] = te_probs_full[j]

        # ── OOF training probabilities ────────────────────────────────────
        X_tr_col = np.full(len(vuln_tr), 0.5, dtype=np.float32)
        if can_fold:
            oof_pos_probs = np.full(len(pos_tr), 0.5, dtype=np.float32)
            skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=cfg["seed"] + 7)
            for fold, (tri2, vali2) in enumerate(skf2.split(
                    np.arange(len(spec_tr_all)), spec_tr_y)):
                pos_vali_local = [i for i in vali2 if i < len(pos_tr)]
                if not pos_vali_local:
                    continue
                m2 = tv2.train_expert(
                    [spec_tr_all[i] for i in tri2],
                    base_dim, vocab, spec_cfg, device, num_relations,
                    tag=f"G{gpu_id}-{name[:5]}-oof{fold+1}", num_classes=2,
                )
                p2 = tv2.probs_from_model(
                    m2, [spec_tr_all[i] for i in pos_vali_local], device, 2
                )[:, 1]
                for local_i, spec_i in enumerate(pos_vali_local):
                    oof_pos_probs[spec_i] = p2[local_i]
                del m2; torch.cuda.empty_cache()
            for j, k in enumerate(tr_positions):
                X_tr_col[k] = oof_pos_probs[j]

        # ── Write atomically to shared checkpoint ─────────────────────────
        atomic_checkpoint_update(
            cls_idx, X_tr_col, X_te_col,
            len(vuln_tr), len(vuln_te),
        )
        print(f"[GPU {gpu_id}] [{name}] complete", flush=True)

    print(f"[GPU {gpu_id}] All specialists done.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: prepare data, spawn workers, wait, run meta-learner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # ── Import training_v2 for data loading and meta-learner ─────────────
    sys.path.insert(0, str(SCRIPT_DIR))
    import training as tv2

    tv2.set_seed(tv2.CONFIG["seed"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "output" / "allclass").mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)

    print("[main] Loading GraphCodeBERT...", flush=True)
    device     = "cuda:0"   # main process uses GPU 0 for data loading
    featurizer = tv2.NodeFeaturizer(tv2.CONFIG, device)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_cached = sum(1 for _ in CACHE_DIR.glob("*.pkl"))
    print(f"[main] Embedding cache: {n_cached} graphs cached", flush=True)

    print("[main] Loading dataset...", flush=True)
    graphs, y_all = tv2.load_dataset(
        DATA_DIR, featurizer, verbose=True, cache_dir=CACHE_DIR
    )

    base_dim      = graphs[0].x.shape[1]
    num_relations = tv2.MAX_RELATIONS

    # ── FIXED: RECONSTRUCT VOCAB SIZE FROM CACHED GRAPH OBJECTS ──────────
    # Since load_dataset pulls from cache, tv2.NodeFeaturizer._label_vocab 
    # remains empty ({}). We calculate the correct boundary explicitly here.
    all_max_ids = []
    for g in graphs:
        if hasattr(g, 'label_id') and g.label_id.numel() > 0:
            all_max_ids.append(int(g.label_id.max().item()))
            
    # Set vocab size to maximum index found + safety padding headroom
    vocab = max(all_max_ids) + 50 if all_max_ids else 1000
    print(f"[main] Calculated true vocabulary size limits across cache: {vocab}", flush=True)
    # ─────────────────────────────────────────────────────────────────────

    # Train/test split (same seed = same split as single-GPU run)
    strat_keys = np.array([
        f"{g.provenance}-{int(g.owasp_class.item())}-{int(g.is_vulnerable.item())}"
        for g in graphs
    ])
    from sklearn.model_selection import train_test_split
    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2,
            random_state=tv2.CONFIG["seed"], stratify=strat_keys,
        )
    except ValueError:
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2,
            random_state=tv2.CONFIG["seed"]
        )

    tr_graphs = [graphs[i] for i in tr_idx]
    te_graphs = [graphs[i] for i in te_idx]
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    vuln_tr = [g for g in tr_graphs if int(g.y.item()) > 0]
    vuln_te = [g for g in te_graphs if int(g.y.item()) > 0]
    y_tr_s2 = np.array([int(g.y.item()) - 1 for g in vuln_tr], dtype=np.int64)
    y_te_s2 = np.array([int(g.y.item()) - 1 for g in vuln_te], dtype=np.int64)

    print(f"[main] train={len(tr_graphs)} | test={len(te_graphs)}", flush=True)

    # ── Pickle graphs so worker processes can load them ───────────────────
    graphs_pkl = str(DATA_DIR / "output" / "allclass" / "graphs_worker.pkl")
    print(f"[main] Pickling graphs for workers → {graphs_pkl}", flush=True)
    
    label_vocab = dict(tv2.NodeFeaturizer._label_vocab)
    with open(graphs_pkl, "wb") as f:
        pickle.dump((tr_graphs, te_graphs, y_tr_s2, y_te_s2, label_vocab), f, protocol=4)
    print(f"[main] Graphs pickled ({Path(graphs_pkl).stat().st_size / 1e9:.2f} GB)",
          flush=True)

    # ── Initialise checkpoint with y labels (workers only write X columns) ─
    if not CKPT_PATH.exists():
        np.savez(
            str(CKPT_PATH),
            X_tr_s2   = np.full((len(vuln_tr), 10), 0.5, dtype=np.float32),
            X_te_s2   = np.full((len(vuln_te), 10), 0.5, dtype=np.float32),
            y_tr_s2   = y_tr_s2,
            y_te_s2   = y_te_s2,
            completed = np.array([], dtype=np.int64),
        )
        print("[main] Fresh checkpoint initialised", flush=True)
    else:
        # Update y labels in existing checkpoint (in case of restart)
        ckpt = np.load(str(CKPT_PATH))
        np.savez(str(CKPT_PATH), **{**dict(ckpt), "y_tr_s2": y_tr_s2, "y_te_s2": y_te_s2})
        completed = set(int(x) for x in ckpt["completed"])
        print(f"[main] Existing checkpoint: {sorted(completed)} complete", flush=True)

    # ── Launch one worker process per GPU ─────────────────────────────────
    print("[main] Spawning GPU workers...", flush=True)
    workers = []
    for gpu_id, spec_indices in GPU_SPECIALIST_MAP.items():
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, spec_indices, graphs_pkl,
                  base_dim, vocab, tv2.CONFIG, num_relations, f"GPU{gpu_id}"),
            name=f"gpu_worker_{gpu_id}",
        )
        p.start()
        workers.append(p)
        print(f"[main] GPU {gpu_id} worker started (PID {p.pid})", flush=True)

    # ── Wait for all workers ──────────────────────────────────────────────
    for p in workers:
        p.join()
        if p.exitcode != 0:
            print(f"[warn] Worker {p.name} exited with code {p.exitcode}", flush=True)

    print("\n[main] All GPU workers finished.", flush=True)

    # ── Meta-learner: combine specialist outputs ──────────────────────────
    ckpt = np.load(str(CKPT_PATH))
    X_tr_s2 = tv2.sanitize_meta(ckpt["X_tr_s2"])
    X_te_s2 = tv2.sanitize_meta(ckpt["X_te_s2"])
    y_tr_s2 = ckpt["y_tr_s2"]
    y_te_s2 = ckpt["y_te_s2"]
    completed = set(int(x) for x in ckpt["completed"])

    if len(completed) < 10:
        print(f"[warn] Only {len(completed)}/10 specialists completed — "
              f"meta-learner results will be partial", flush=True)

    import lightgbm as lgb
    from sklearn.metrics import classification_report, f1_score

    gbm_s2 = lgb.LGBMClassifier(
        **tv2.CONFIG["gbm_params"],
        objective        = "multiclass",
        num_class        = 10,
        n_jobs           = -1,
        min_child_samples= max(1, len(y_tr_s2) // 100),
        class_weight     = "balanced",
        verbose          = -1,
    )
    gbm_s2.fit(X_tr_s2, y_tr_s2)
    import pickle as pkl
    with open(DATA_DIR / "output" / "allclass" / "lgbm_stage2.pkl", "wb") as f:
        pkl.dump(gbm_s2, f)

    s2_pred = gbm_s2.predict(X_te_s2)
    target_names = [tv2.OWASP_NAMES.get(i + 1, f"A{i+1:02d}") for i in range(10)]
    print(f"\n{'='*70}")
    print(f"  Stage 2 — 10-class (vulnerable samples only) — {len(vuln_te)} test samples")
    print(f"{'='*70}")
    print(classification_report(y_te_s2, s2_pred,
                                 target_names=target_names, digits=4, zero_division=0))
    macro = f1_score(y_te_s2, s2_pred, average="macro", zero_division=0)
    print(f"  Macro F1: {macro:.4f}")

    import json
    with open(DATA_DIR / "output" / "allclass" / "stage2_results.json", "w") as f:
        json.dump({"macro_f1": float(macro), "n_test": len(vuln_te)}, f, indent=2)

    # Clean up worker pickle file
    Path(graphs_pkl).unlink(missing_ok=True)
    print("[main] Done.", flush=True)