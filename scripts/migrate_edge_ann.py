#!/usr/bin/env python3
"""
migrate_edge_ann.py — attach DDG annotation features to the cached graphs.

The cache stores GraphCodeBERT node embeddings, which cost 8-15 hours to build,
so this does not rebuild it. It re-parses each graph's .dot for the edge labels
only (no GPU, no encoder) and appends two tensors to the pickle.

Alignment is the whole risk: the new tensors must be row-for-row identical to
the stored edge_index. The parse is replayed with the same node ordering
(`sorted(nodes)`, as NodeFeaturizer.build uses) and the same src/dst filter as
dot_to_pyg, and then the recomputed edge_type is checked against the stored one.
A mismatch aborts that graph rather than writing a silently misaligned tensor.

    python3 migrate_edge_ann.py --workers 24
"""
from __future__ import annotations
import argparse, pickle, sys, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import training as T
from edge_ann import hashed, structural, STRUCT_DIM

# _map_etype assigns ids in first-encounter order, which differs per process and
# so cannot be replayed here. The mapping actually baked into the cache was
# recovered by aligning 600 cached graphs against a fresh parse; it is pinned so
# the alignment check below is meaningful rather than self-fulfilling.
CANONICAL = {"AST": 0, "CFG": 1, "DDG": 2, "CDG": 3}


def one(path_str: str) -> str:
    p = Path(path_str)
    try:
        d = pickle.loads(p.read_bytes())
    except Exception:
        return "unreadable"
    if hasattr(d, "edge_hash"):
        return "already"
    dot = Path(d.dot_path)
    if not dot.exists():
        return "dot_missing"
    try:
        nodes, edges = T.parse_dot_full(dot)
    except Exception:
        return "parse_failed"
    if not nodes or not edges:
        return "empty"

    node_ids = sorted(nodes.keys())
    row = {nid: i for i, nid in enumerate(node_ids)}
    code = [nodes[n]["code"] or "" for n in node_ids]

    et, eh, es = [], [], []
    for (s, etype, ann, t) in edges:
        if s in row and t in row:
            et.append(CANONICAL.get(etype, T.MAX_RELATIONS - 1))
            eh.append(hashed(ann))
            es.append(structural(ann, code[row[s]], code[row[t]]))

    stored = d.edge_type.tolist()
    if et != stored:
        return "misaligned"

    d.edge_hash = torch.tensor(eh, dtype=torch.long)
    d.edge_struct = torch.tensor(es, dtype=torch.float32) if es else \
        torch.zeros((0, STRUCT_DIM), dtype=torch.float32)
    p.write_bytes(pickle.dumps(d))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/moses/graph_cache_cwe")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = [str(f) for f in Path(a.cache).rglob("*.pkl")]
    if a.limit:
        files = files[:a.limit]
    print(f"[migrate] {len(files)} cached graphs", flush=True)
    import collections
    tally = collections.Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=64), 1):
            tally[r] += 1
            if i % 20000 == 0:
                print(f"  {i}/{len(files)}  {dict(tally)}", flush=True)
    print("[migrate] done:", dict(tally))
    bad = sum(v for k, v in tally.items() if k not in ("ok", "already"))
    if bad:
        print(f"[migrate] WARNING {bad} graphs not migrated")


if __name__ == "__main__":
    main()
