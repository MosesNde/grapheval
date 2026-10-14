#!/usr/bin/env python3
"""
featurize_from_linevul.py — re-featurize node snippets using LineVul's
FINE-TUNED encoder, to fill the missing cell in the graph/encoder decomposition.

THE CONFOUND THIS RESOLVES
--------------------------
Official LineVul reaches 0.790 repository-disjoint against 0.594 for our best
graph model. But LineVul fine-tunes CodeBERT end-to-end, while our graph models
consume FROZEN GraphCodeBERT embeddings -- training.py puts the encoder in
eval() under no_grad, so it never updates. The comparison therefore confounds
"graph vs no graph" with "frozen vs fine-tuned encoder", and the honest reading
of the gap is ambiguous between them.

Filling the 2x2:

                        no graph      with graph
    frozen encoder        0.575          0.594
    fine-tuned encoder    0.790            ?     <- this script

Fine-tuning end-to-end through a 500-node graph is computationally out of reach
(every node would need a transformer forward and backward pass). Instead we take
LineVul's already-fine-tuned encoder and use it as a frozen feature extractor
for node snippets. That is not identical to end-to-end training, and it is a
lower bound on what a jointly trained model could reach -- but it isolates the
question actually at issue: given a fine-tuned encoder, does the graph add
anything?

Run:
    python3 featurize_from_linevul.py --ckpt <path>.bin --out /home/moses/graph_cache_lv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/home/moses/graph_cache_lv")
    ap.add_argument("--src-cache", default="/home/moses/graph_cache_cwe")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=128,
                    help="must match training.py's codebert_max_len (128), or "
                         "the comparison is confounded by truncation")
    ap.add_argument("--pool", default="mean", choices=["mean", "cls"],
                    help="mean-pool token states, or take CLS. CLS COLLAPSES: "
                         "fine-tuning for binary sequence classification "
                         "optimises CLS for linear separability into two "
                         "classes and nothing else, so on out-of-distribution "
                         "node snippets it degenerates to a near-constant "
                         "(measured: per-dim std 0.0001, mean pairwise cosine "
                         "1.000). Mean pooling retains token-level detail.")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    import training as T

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    enc = AutoModel.from_pretrained("microsoft/codebert-base").to(dev)

    # WHICH WEIGHTS. LineVul's Model subclasses RobertaForSequenceClassification
    # and ALSO holds its own `self.encoder`. Its forward() calls
    # self.encoder.roberta(...), so `encoder.roberta.*` is the fine-tuned
    # encoder. The inherited `roberta.*` attribute is never used in forward,
    # is randomly initialised by super().__init__(config), and never receives
    # gradient -- both key sets are present in the checkpoint and they are NOT
    # identical. Matching loosely would silently load the untrained copy, so
    # the prefix is pinned and the result is verified below.
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    PREF = "encoder.roberta."
    remap = {k[len(PREF):]: v for k, v in sd.items() if k.startswith(PREF)}
    if not remap:
        raise SystemExit(f"no keys under {PREF!r}; checkpoint layout differs")
    missing, unexpected = enc.load_state_dict(remap, strict=False)
    print(f"[enc] loaded {len(remap)} tensors from {PREF}* "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    # Verify we actually got fine-tuned weights and not a pristine or random
    # copy: the last encoder layer must differ from stock CodeBERT.
    _base = AutoModel.from_pretrained("microsoft/codebert-base")
    k = "encoder.layer.11.output.dense.weight"
    d = float((enc.state_dict()[k].cpu() - _base.state_dict()[k]).abs().mean())
    print(f"[enc] mean abs diff vs stock CodeBERT at {k}: {d:.5f}")
    if d < 1e-6:
        raise SystemExit("loaded encoder is identical to stock CodeBERT -- "
                         "fine-tuned weights were not applied")
    del _base
    enc.eval()

    corpus = [json.loads(l) for l in
              (DATA / "cwe_corpus.jsonl").read_text().splitlines() if l]
    src_cache = Path(a.src_cache)
    done = skipped = 0

    @torch.no_grad()
    def embed(snips):
        vecs = []
        for i in range(0, len(snips), a.batch):
            b = tok(snips[i:i + a.batch], truncation=True, max_length=a.max_len,
                    padding="max_length", return_tensors="pt").to(dev)
            hs = enc(**b).last_hidden_state
            if a.pool == "cls":
                h = hs[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                h = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            vecs.append(h.float().cpu().numpy())
        return np.concatenate(vecs, 0) if vecs else np.zeros((0, 768), np.float32)

    for r in corpus:
        key = T._cache_key(Path(r["dot_path"]))
        dst = out / f"{key}.pkl"
        if dst.exists():
            done += 1; continue
        srcp = src_cache / f"{key}.pkl"
        if not srcp.exists():
            skipped += 1; continue
        try:
            with open(srcp, "rb") as f:
                g = pickle.load(f)
        except Exception:
            skipped += 1; continue
        nodes, _ = T.parse_dot(Path(r["dot_path"]))
        # parse_dot already returns {'label': <node type>, 'code': <snippet>}.
        # An earlier version called _parse_html_label on v['label'] -- but that
        # field is the TYPE ('METHOD'), not raw HTML, so it yielded an empty
        # string for every node. The encoder then produced one identical vector
        # throughout, which looked exactly like representation collapse and was
        # in fact empty input.
        snips = [v.get("code", "") if isinstance(v, dict) else ""
                 for v in nodes.values()]
        snips = (snips + [""] * g.x.shape[0])[:g.x.shape[0]]
        if done == 0 and sum(1 for t in snips if t.strip()) == 0:
            raise SystemExit("first graph yielded no non-empty node snippets -- "
                             "extraction is broken; fix before featurising")
        g.x = torch.from_numpy(embed(snips).astype(np.float32))
        with open(dst, "wb") as f:
            pickle.dump(g, f, protocol=4)
        done += 1
        if done % 5000 == 0:
            print(f"  {done}/{len(corpus)}", flush=True)

    # DEGENERACY CHECK. The first version of this script silently produced
    # collapsed features -- every node the same vector -- and the downstream
    # runs completed with zero errors and plausible-looking scores. Loading the
    # right weights and finishing without a traceback is not evidence the
    # output carries signal, so measure it.
    import random as _r
    _fs = _r.Random(0).sample(sorted(str(p) for p in out.glob("*.pkl")),
                              min(300, len(list(out.glob("*.pkl")))))
    _X = []
    for _p in _fs:
        try:
            with open(_p, "rb") as f:
                _X.append(pickle.load(f).x.numpy())
        except Exception:
            pass
    if _X:
        _A = np.concatenate(_X, 0)
        _sub = _A[np.random.RandomState(0).choice(
            len(_A), min(2000, len(_A)), replace=False)]
        _n = _sub / (np.linalg.norm(_sub, axis=1, keepdims=True) + 1e-9)
        _cos = float((_n @ _n.T)[np.triu_indices(len(_n), 1)].mean())
        _std = float(_A.std(0).mean())
        print(f"[check] per-dim std {_std:.4f} | mean pairwise cosine {_cos:.3f}")
        if _std < 0.01 or _cos > 0.99:
            raise SystemExit(
                "FEATURES ARE DEGENERATE -- essentially one vector for every "
                "node. Any model trained on these is reading a constant. "
                "Do not use this cache.")
        print("[check] features carry variance; safe to train on")

    print(f"[done] {done} written, {skipped} skipped -> {out}")


if __name__ == "__main__":
    main()
