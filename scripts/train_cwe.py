#!/usr/bin/env python3
"""
train_cwe.py — Phase 2 driver: train on the real-CVE, CWE-family corpus using
the audited splits.

WHAT IS DIFFERENT FROM training.py
----------------------------------
training.py discovers data by walking OWASP-named folders and makes its own
stratified split. Both are now wrong:

  * the taxonomy is CWE families, not OWASP categories (real C/C++ CVE data
    cannot populate OWASP -- 55% of it maps to no category at all);
  * the split must be the audited, pre-computed one. Re-splitting here would
    discard the repository/commit/pair/clone disjointness that make_splits.py
    established and printed evidence for.

So this driver reads data/cwe_corpus.jsonl plus data/splits_<mode>_family.json
and reuses training.py's graph encoding and model code unchanged.

METRICS
-------
Macro F1 alone is what the previous submission reported and it is not enough
for a detector evaluated at a realistic base rate. Reviewer 1 asked for
"PR-AUC, MCC, false-positive rate, recall at a fixed false-positive budget,
and calibrated precision under realistic base rates"; Stage 1 reports all of
them, each quoted against the partition's own prevalence.

Run:
    python3 train_cwe.py --stage featurize          # encode graphs, fill cache
    python3 train_cwe.py --stage 1                  # binary detector
    python3 train_cwe.py --stage 2                  # CWE-family specialists
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "runs"

import training as T  # noqa: E402  (reuses dot_to_pyg, train_expert, models)


# ─────────────────────────────────────────────────────────────────────────────
def load_split(mode: str = "repo", task: str = "family",
               suffix: str = "") -> tuple[dict, list[str], dict]:
    """Load the corpus and one pre-computed split.

    task="binary" exists for corpora with no CWE labels at all. Devign is the
    case that matters -- it is the independent second corpus and is
    vulnerable/not-vulnerable only, so family filtering would silently drop
    every one of its vulnerable samples."""
    corpus = {}
    for line in (DATA / "cwe_corpus.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            corpus[r["sample_id"]] = r
    spec = json.loads((DATA / f"splits_{mode}_{task}{suffix}.json").read_text())
    if task == "binary":
        return corpus, [], spec["splits"]
    fams = sorted({corpus[s]["family"] for part in spec["splits"].values()
                   for s in part
                   if corpus[s]["label"] == 1 and corpus[s]["family"]})
    return corpus, fams, spec["splits"]


def encode(corpus, splits, fams, cfg, device, cache_dir: Path, limit=None,
           task="family"):
    """Graph -> PyG Data, class 0 = safe, 1..N = CWE family index."""
    fam_idx = {f: i + 1 for i, f in enumerate(fams)}
    feat = T.NodeFeaturizer(cfg, device=device)
    out = {}
    t0 = time.time()
    for part, ids in splits.items():
        if limit:
            ids = ids[:limit]
        graphs, n_skip = [], 0
        for k, sid in enumerate(ids):
            r = corpus[sid]
            if task == "binary":
                cls = int(r["label"])
            else:
                cls = fam_idx.get(r["family"], 0) if r["label"] == 1 else 0
            if task != "binary" and r["label"] == 1 and cls == 0:
                n_skip += 1
                continue           # vulnerable but family not in the kept set
            g = T.dot_to_pyg(Path(r["dot_path"]), cls, bool(r["label"]),
                             feat, r["provenance"], cache_dir=cache_dir)
            if g is None:
                n_skip += 1
                continue
            # Carry the audit fields so results can be sliced by them later.
            g.stub = bool(r.get("stub"))
            g.repo = r.get("repo", "")
            g.sample_id = sid
            graphs.append(g)
            if k and k % 2000 == 0:
                print(f"  [{part}] {k}/{len(ids)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  [{part}] {len(graphs)} graphs, {n_skip} skipped")
        out[part] = graphs
    return out, fam_idx



def apply_feature_ablation(G, mode: str, seed: int = 42) -> None:
    """Strip node information in place, to separate what the model READS from
    what it INFERS from structure.

    Every result so far -- GNN and edge-free alike -- lets the model see the
    code, because node features are GraphCodeBERT embeddings of each node's
    snippet. So the edge-free control isolated STRUCTURE, not TEXT. These modes
    invert that.

      full       GraphCodeBERT embedding + node-type embedding  (unchanged)
      type-only  text zeroed; the model sees the graph and each node's
                 syntactic KIND (METHOD, CALL, IDENTIFIER, BLOCK) but never an
                 identifier or literal
      random     text replaced with fixed seeded noise of the same width; the
                 node-type embedding still applies, so this is type-only plus
                 a distractor -- it costs dimensionality without adding signal
      topology   text zeroed AND every node type forced to a single id, so
                 nothing survives but the graph's shape and its edge types

    NOTE ON THE FLOOR: `random` is NOT the floor, which is easy to get wrong.
    The model concatenates x with an embedding of label_id, so noise in x still
    leaves the node types visible -- it is type-only with a distractor bolted
    on. `topology` is the true floor, and it is the one that makes a non-zero
    type-only score interpretable: graph size alone correlates with the label
    (longer functions are likelier to be vulnerable), so any claim that syntax
    types carry signal has to clear what pure shape already achieves.
    """
    if mode == "full":
        return
    gen = torch.Generator().manual_seed(seed)
    for part in G.values():
        for g in part:
            if mode in ("type-only", "topology"):
                g.x = torch.zeros_like(g.x)
            elif mode == "random":
                g.x = torch.randn(g.x.shape, generator=gen)
            if mode == "topology":
                g.label_id = torch.zeros_like(g.label_id)
    print(f"[features] ablation '{mode}' applied to all partitions")


# ─────────────────────────────────────────────────────────────────────────────
def binary_metrics(y, p, prevalence: float, fpr_budget=(0.01, 0.05)) -> dict:
    """y in {0,1}, p = P(vulnerable). Everything a base-rate-sensitive claim
    needs, so no number in the paper is quotable without its prevalence."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 matthews_corrcoef, roc_auc_score, roc_curve)
    yhat = (p >= 0.5).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tp = int(((yhat == 1) & (y == 1)).sum())
    m = {
        "prevalence": round(float(prevalence), 4),
        "macro_f1": round(float(f1_score(y, yhat, average="macro")), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "mcc": round(float(matthews_corrcoef(y, yhat)), 4),
        "fpr": round(fp / max(1, fp + tn), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    fpr, tpr, _ = roc_curve(y, p)
    for b in fpr_budget:
        i = int(np.searchsorted(fpr, b, side="right") - 1)
        m[f"recall_at_fpr_{b}"] = round(float(tpr[max(0, i)]), 4)
    return m


def run_stage1(G, cfg, device, base_dim, vocab, nrel, tag) -> dict:
    """5-fold OOF-stacked binary GAT ensemble + LightGBM."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    def binarize(gs):
        out = []
        for g in gs:
            g2 = g.clone(); g2.y = torch.tensor(int(g.y.item() > 0))
            out.append(g2)
        return out

    tr, va, te = binarize(G["train"]), binarize(G["val"]), binarize(G["test"])
    y_tr = np.array([int(g.y.item()) for g in tr])
    y_te = np.array([int(g.y.item()) for g in te])

    skf = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
    oof = np.zeros((len(tr), 2), dtype=np.float32)
    te_acc = []
    for k, (a, b) in enumerate(skf.split(np.arange(len(tr)), y_tr)):
        # Early stopping selects on the held-out validation partition, whose
        # repositories appear in neither training nor test. Carving a validation
        # slice out of the fold instead lets it share repositories with training,
        # which costs 0.058 ROC-AUC under this protocol.
        m = T.train_expert([tr[i] for i in a], base_dim, vocab, cfg, device,
                           nrel, tag=f"{tag}-s1-f{k+1}", num_classes=2,
                           val_graphs=va)
        oof[b] = T.probs_from_model(m, [tr[i] for i in b], device, 2)
        te_acc.append(T.probs_from_model(m, te, device, 2))
        del m; torch.cuda.empty_cache()
    te_p = np.mean(np.stack(te_acc), axis=0)

    meta = lgb.LGBMClassifier(**cfg["gbm_params"], objective="binary",
                              class_weight="balanced", verbose=-1, n_jobs=-1)
    meta.fit(T.sanitize_meta(oof), y_tr)
    p_te = meta.predict_proba(T.sanitize_meta(te_p))[:, 1]

    res = {"ensemble_then_meta": binary_metrics(y_te, p_te, y_te.mean()),
           "ensemble_mean_only": binary_metrics(y_te, te_p[:, 1], y_te.mean())}

    # Per-fold test scores. The central claim rests on gaps of ~0.03, and a
    # single averaged number gives a reviewer no way to judge whether that is
    # signal or noise. Each fold is an independently trained model evaluated on
    # the same held-out set, so the spread is a lower bound on run-to-run
    # variance -- it shares the split, so it understates seed variance.
    per_fold = [binary_metrics(y_te, tp[:, 1], y_te.mean())["roc_auc"]
                for tp in te_acc]
    res["per_fold_roc_auc"] = per_fold
    res["fold_mean"] = round(float(np.mean(per_fold)), 4)
    res["fold_std"] = round(float(np.std(per_fold)), 4)
    # Stubs carry no program structure; if the detector scores them like real
    # graphs that is evidence it is reading structure, not an artefact.
    stub = np.array([bool(getattr(g, "stub", False)) for g in te])
    for name, mask in (("non_stub", ~stub), ("stub", stub)):
        if mask.sum() > 50 and 0 < y_te[mask].mean() < 1:
            res[f"test_{name}"] = binary_metrics(y_te[mask], p_te[mask],
                                                 y_te[mask].mean())
    return res


def run_stage2(G, cfg, device, base_dim, vocab, nrel, fams, tag,
               neg_cap: int = 20000) -> dict:
    """Ten one-vs-rest CWE-family specialists + LightGBM, with the corrected
    DENSE meta-feature construction and the leak guard enabled."""
    import lightgbm as lgb
    from sklearn.metrics import classification_report, f1_score

    N = len(fams)
    tr_v = [g for g in G["train"] if int(g.y.item()) > 0]
    te_v = [g for g in G["test"] if int(g.y.item()) > 0]
    tr_s = [g for g in G["train"] if int(g.y.item()) == 0]
    te_s = [g for g in G["test"] if int(g.y.item()) == 0]
    y_tr = np.array([int(g.y.item()) - 1 for g in tr_v])
    y_te = np.array([int(g.y.item()) - 1 for g in te_v])
    print(f"[stage2] vulnerable train={len(tr_v)} test={len(te_v)} | "
          f"safe pool train={len(tr_s)} test={len(te_s)}")

    X_tr = np.full((len(tr_v), N), np.nan, dtype=np.float32)
    X_te = np.full((len(te_v), N), np.nan, dtype=np.float32)

    def relabel(gs, y):
        out = []
        for g in gs:
            g2 = g.clone(); g2.y = torch.tensor(y, dtype=torch.long)
            out.append(g2)
        return out

    import random as _rnd
    for ci in range(N):
        pos = [g for g in tr_v if int(g.y.item()) == ci + 1]
        if len(pos) < 10:
            print(f"  [{fams[ci]}] only {len(pos)} positives -- skipped")
            continue

        # Negative sampling. The design point of one-vs-rest here is that a
        # specialist's negatives come from the WHOLE safe pool rather than the
        # safe code paired with its own category -- that is what removes the
        # partition-identity shortcut. Using every one of the ~62k safe graphs
        # is not required for that: a large random draw from the same pool
        # preserves the property, while training all 10 specialists x 5 folds
        # against the full pool projects to ~62 hours. The draw is seeded per
        # specialist so the run reproduces.
        negs = tr_s
        if neg_cap and len(tr_s) > neg_cap:
            negs = _rnd.Random(cfg["seed"] + ci).sample(tr_s, neg_cap)
        spec_tr = relabel(pos, 1) + relabel(negs, 0)
        yb = np.array([int(g.y.item()) for g in spec_tr])
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(min(5, cfg["oof_max_folds"]), shuffle=True,
                              random_state=cfg["seed"] + 7)
        own = {id(g): i for i, g in enumerate(tr_v) if int(g.y.item()) == ci + 1}
        other_idx = [i for i, g in enumerate(tr_v) if int(g.y.item()) != ci + 1]
        other = [tr_v[i] for i in other_idx]
        acc_other = np.zeros(len(other)); n_models = 0
        te_folds = []
        for k, (a, b) in enumerate(skf.split(np.arange(len(spec_tr)), yb)):
            m = T.train_expert([spec_tr[i] for i in a], base_dim, vocab, cfg,
                               device, nrel, tag=f"{tag}-s2-{fams[ci][:6]}-f{k+1}",
                               num_classes=2)
            # own positives: only the fold that held them out may score them
            held = [i for i in b if i < len(pos)]
            if held:
                p = T.probs_from_model(m, [spec_tr[i] for i in held], device, 2)[:, 1]
                for j, si in enumerate(held):
                    X_tr[own[id(pos[si])], ci] = p[j]
            # other-category samples were never in this specialist's pool
            if other:
                acc_other += T.probs_from_model(m, other, device, 2)[:, 1]
                n_models += 1
            te_folds.append(T.probs_from_model(m, te_v, device, 2)[:, 1])
            del m; torch.cuda.empty_cache()
        if n_models:
            mo = acc_other / n_models
            for j, i in enumerate(other_idx):
                X_tr[i, ci] = mo[j]
        X_te[:, ci] = np.mean(np.stack(te_folds), axis=0)
        print(f"  [{fams[ci]}] done ({len(pos)} pos)", flush=True)

    X_tr, X_te = T.sanitize_meta(X_tr), T.sanitize_meta(X_te)
    T.assert_meta_features_dense(X_tr, y_tr, "Stage2/train")
    T.assert_meta_features_dense(X_te, y_te, "Stage2/test")

    gbm = lgb.LGBMClassifier(**cfg["gbm_params"], objective="multiclass",
                             num_class=N, class_weight="balanced",
                             verbose=-1, n_jobs=-1)
    gbm.fit(X_tr, y_tr)
    pred = gbm.predict(X_te)
    rep = classification_report(y_te, pred, target_names=fams,
                                output_dict=True, zero_division=0)
    np.save(OUT / f"{tag}_s2_X_te.npy", X_te)
    np.save(OUT / f"{tag}_s2_y_te.npy", y_te)
    pickle.dump(gbm, open(OUT / f"{tag}_s2_lgbm.pkl", "wb"))
    return {"macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
            "n_test": len(y_te), "per_class": rep}


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["featurize", "1", "2"])
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap graphs per partition (smoke tests)")
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--task", default="family", choices=["family", "binary"],
                    help="'binary' for corpora with no CWE labels (Devign)")
    ap.add_argument("--split-suffix", default="",
                    help="e.g. '_devign' -> splits_<mode>_binary_devign.json")
    ap.add_argument("--tag", default="",
                    help="suffix appended to the run name; REQUIRED when "
                         "changing --cache, or a run with different node "
                         "features silently overwrites an existing result")
    ap.add_argument("--seed", type=int, default=None,
                    help="override CONFIG seed; changes init, folds and "
                         "sampling but NOT the split, which is fixed on disk")
    ap.add_argument("--edges", default="all",
                    choices=["all", "untyped", "none"],
                    help="'untyped' collapses AST/CFG/DDG/CDG to one relation "
                         "(Reviewer 2's typed-vs-untyped ablation); 'none' "
                         "removes edges entirely")
    ap.add_argument("--layers", type=int, default=3,
                    help="message-passing depth. At 3 bidirectional layers a "
                         "node reaches 74%% of its function; 6 reaches 99.8%%")
    ap.add_argument("--reverse-edges", action="store_true",
                    help="add each edge in the opposite direction as a distinct "
                         "relation type (AST^-1 etc). Joern exports directed "
                         "edges, so without this a node only ever aggregates "
                         "from its predecessors and sees ~19%% of its function "
                         "within three layers")
    ap.add_argument("--edge-ann", default="none",
                    choices=["none", "hashed", "structural", "shuffled"],
                    help="how DDG variable annotations reach the attention: "
                         "'hashed' is lexical and can carry project-specific "
                         "naming, 'structural' describes the flowing value "
                         "without naming it")
    ap.add_argument("--features", default="full",
                    choices=["full", "type-only", "random", "topology"],
                    help="ablate node features (see apply_feature_ablation)")
    ap.add_argument("--backbone", default=None,
                    choices=["gat","gcn","gin","nograph","devign","reveal","livable","gtrans","sage"],
                    help="override the GNN backbone (default: CONFIG's 'gat')")
    ap.add_argument("--neg-cap", type=int, default=20000,
                    help="max safe graphs per Stage-2 specialist "
                         "(0 = use the entire pool)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(T.CONFIG)
    # The stock CONFIG was tuned for ~8k graphs. This corpus has 70k training
    # graphs, and at batch_size 16 with single-threaded collation the first run
    # sat at 3% GPU / 9800% CPU -- under 50 epochs in 111 minutes, putting
    # Stage 1 alone near 55 hours. Larger batches plus parallel collation move
    # the work onto the GPU where it belongs.
    cfg.update(batch_size=int(a.batch_size), num_workers=int(a.workers),
               pin_memory=True, epochs=int(a.epochs),
               early_stop_patience=int(a.patience))
    if a.backbone:
        cfg["backbone"] = a.backbone
    if a.seed is not None:
        cfg["seed"] = a.seed
    T.set_seed(cfg["seed"])
    print(f"[cfg] backbone={cfg['backbone']} batch={cfg['batch_size']} "
          f"workers={cfg['num_workers']} epochs={cfg['epochs']} "
          f"patience={cfg['early_stop_patience']}")
    device = T.get_device()
    T.set_seed(cfg["seed"])

    corpus, fams, splits = load_split(a.mode, a.task, a.split_suffix)
    print(f"[data] mode={a.mode} | {len(fams)} families: {fams}")
    for k, v in splits.items():
        print(f"       {k}: {len(v)}")

    G, fam_idx = encode(corpus, splits, fams, cfg, device,
                        Path(a.cache), limit=a.limit, task=a.task)
    if a.stage == "featurize":
        print("[done] cache warm; rerun with --stage 1")
        return

    cfg["gat_layers"] = a.layers
    cfg["edge_ann"] = a.edge_ann
    if a.edge_ann == "shuffled":
        # Capacity control for the annotation arms. Identical architecture and
        # parameter count to `hashed`, but each edge's bucket id is resampled
        # from the corpus-wide empirical distribution, so the marginal statistics
        # are preserved and only the edge-to-annotation correspondence is lost.
        # A drop here as large as `hashed` means the wider edge channel is the
        # cause, not the annotations.
        import numpy as _np
        pool = _np.concatenate([g.edge_hash.numpy() for g in G["train"]])
        rng = _np.random.default_rng(cfg["seed"])
        for part in G.values():
            for g in part:
                g.edge_hash = torch.as_tensor(
                    rng.choice(pool, size=g.edge_hash.numel()), dtype=torch.long)
        print(f"[edge-ann] shuffled: bucket ids resampled from {len(pool)} "
              f"training edges", flush=True)
        cfg["edge_ann"] = "hashed"          # same module, permuted input
    if a.edge_ann != "none":
        miss = [p for p, gs in G.items()
                if gs and not hasattr(gs[0], "edge_hash")]
        if miss:
            raise SystemExit(f"--edge-ann needs migrated cache; {miss} lack "
                             f"edge_hash. Run migrate_edge_ann.py first.")
    apply_feature_ablation(G, a.features, seed=cfg["seed"])
    if a.edges != "all":
        n = 0
        for part in G.values():
            for g in part:
                if a.edges == "untyped":
                    g.edge_type = torch.zeros_like(g.edge_type)
                else:                       # none
                    g.edge_index = torch.zeros((2, 0), dtype=torch.long)
                    g.edge_type = torch.zeros((0,), dtype=torch.long)
                n += 1
        print(f"[edges] '{a.edges}' applied to {n} graphs")

    base_dim = G["train"][0].x.shape[1]

    # Vocabulary must span EVERY partition, not just train. Sizing the label
    # embedding from train alone lets a val/test graph carry a label_id past
    # the end of the table, which surfaces as an opaque CUDA device-side
    # assert inside GATConv rather than an index error. Clamp as well, so an
    # id from outside the corpus can never do it either.
    vocab = max(int(g.label_id.max().item())
                for gs in G.values() for g in gs) + 2
    nrel = T.MAX_RELATIONS
    for gs in G.values():
        for g in gs:
            if g.label_id.numel():
                g.label_id = g.label_id.clamp_(0, vocab - 1)
            if hasattr(g, "edge_type") and g.edge_type.numel():
                g.edge_type = g.edge_type.clamp_(0, nrel - 1)
    if a.reverse_edges:
        n_before = n_after = 0
        for gs in G.values():
            for g in gs:
                if not (hasattr(g, "edge_index") and g.edge_index.numel()):
                    continue
                n_before += g.edge_index.shape[1]
                rev = g.edge_index.flip(0)
                g.edge_index = torch.cat([g.edge_index, rev], dim=1)
                # reverse relations occupy ids 4..7; forward types are 0..3
                g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
                for extra in ("edge_hash", "edge_struct"):
                    if hasattr(g, extra):
                        t = getattr(g, extra)
                        setattr(g, extra, torch.cat([t, t], dim=0))
                n_after += g.edge_index.shape[1]
        print(f"[reverse-edges] {n_before} -> {n_after} edges; relation ids 4-7 "
              f"carry the reversed directions", flush=True)

    print(f"[model] base_dim={base_dim} vocab={vocab} relations={nrel}")
    tag = (f"{a.mode}-{cfg['backbone']}"
           + ("" if a.features == "full" else f"-{a.features}")
           + ("" if a.edges == "all" else f"-e{a.edges}")
           + ("" if a.edge_ann == "none" else f"-ann{a.edge_ann}")
           + ("-rev" if a.reverse_edges else "")
           + ("" if a.layers == 3 else f"-L{a.layers}")
           + ("" if a.seed is None else f"-s{a.seed}")
           + (f"-{a.tag}" if a.tag else "")
           + ('-smoke' if a.limit else ''))

    if a.stage == "1":
        res = run_stage1(G, cfg, device, base_dim, vocab, nrel, tag)
    else:
        res = run_stage2(G, cfg, device, base_dim, vocab, nrel, fams, tag,
                         neg_cap=a.neg_cap)

    path = OUT / f"{tag}_stage{a.stage}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:3000])
    print(f"\n[done] -> {path}")


if __name__ == "__main__":
    main()
