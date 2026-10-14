#!/usr/bin/env python3
"""
build_bow_features.py — hashed bag-of-words node features, as a weak-encoder
control for the "the graph contributes nothing" claim.

WHY THIS EXISTS
---------------
The strongest objection to that claim is saturation: GraphCodeBERT already
encodes each node's local context so richly that message passing has nothing
left to add, and with weaker features the graph would matter. If that were
true, the claim would be about GraphCodeBERT rather than about code property
graphs.

This builds deliberately impoverished features -- token counts over each node's
own code snippet, hashed into a fixed width, no pretraining, no context, no
semantics. If the graph-versus-no-graph gap stays small with THESE features too,
saturation is ruled out and the claim survives its best counter-argument.

Width is set to 768 to match GraphCodeBERT exactly, so capacity, parameter
count and the input projection are all held constant. The only thing that
changes is what the vector means.

Output mirrors the embedding cache: one pickle per graph, keyed identically, so
train_cwe.py --cache <this dir> is the only change needed at training time.

Run:
    python3 build_bow_features.py --workers 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

DIM = 768                     # match GraphCodeBERT width exactly
OUT_DIR = Path("/home/moses/graph_cache_bow")
SRC_CACHE = Path("/home/moses/graph_cache_cwe")

_LABEL_RE = re.compile(r"\[\s*label\s*=\s*<(.*)>\s*\]\s*$", re.I | re.M)
_BR_RE = re.compile(r"<BR\s*/?>", re.I)
_TOK_RE = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")
_ENTITY = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'"}


def node_snippets(dot_path: str) -> list[str]:
    """The code text carried by each node, in file order -- the same order
    parse_dot() yields, so the rows line up with the cached graph's nodes."""
    try:
        text = Path(dot_path).read_text(errors="replace")
    except OSError:
        return []
    out = []
    for m in _LABEL_RE.finditer(text):
        parts = _BR_RE.split(m.group(1), maxsplit=1)
        code = parts[1] if len(parts) > 1 else ""
        for ent, ch in _ENTITY.items():
            code = code.replace(ent, ch)
        out.append(code)
    return out


def bow_vector(snippet: str) -> np.ndarray:
    """Hashed token counts, L2-normalised. Deliberately shallow: a token's
    identity is all that survives -- no order, no context, no pretraining."""
    v = np.zeros(DIM, dtype=np.float32)
    for tok in _TOK_RE.findall(snippet):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(),
                           "big")
        v[h % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def convert(args) -> str:
    dot_path, key = args
    dst = OUT_DIR / f"{key}.pkl"
    if dst.exists():
        return "cached"
    src = SRC_CACHE / f"{key}.pkl"
    if not src.exists():
        return "missing_src"
    try:
        with open(src, "rb") as f:
            g = pickle.load(f)
    except Exception:
        return "unreadable"

    snips = node_snippets(dot_path)
    n = g.x.shape[0]
    # parse_dot may drop nodes the featurizer kept, or vice versa; align on the
    # shorter run and zero-pad rather than guess, so a mismatch degrades
    # gracefully instead of silently shifting every row by one.
    X = np.zeros((n, DIM), dtype=np.float32)
    for i in range(min(n, len(snips))):
        X[i] = bow_vector(snips[i])
    g.x = torch.from_numpy(X)
    with open(dst, "wb") as f:
        pickle.dump(g, f, protocol=4)
    return "ok" if len(snips) >= n else "short"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import training as T

    corpus = [json.loads(l) for l in
              (ROOT / "data" / "cwe_corpus.jsonl").read_text().splitlines() if l]
    work = [(r["dot_path"], T._cache_key(Path(r["dot_path"]))) for r in corpus]
    print(f"[bow] {len(work)} graphs -> {OUT_DIR} (dim={DIM})")

    counts: dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(convert, work, chunksize=64):
            counts[res] = counts.get(res, 0) + 1
            done += 1
            if done % 20000 == 0:
                print(f"  {done}/{len(work)}  {counts}", flush=True)
    print(f"[bow] done: {counts}")


if __name__ == "__main__":
    main()
