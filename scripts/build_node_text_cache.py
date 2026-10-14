#!/usr/bin/env python3
"""
build_node_text_cache.py — per-node source text, aligned with the graph cache.

Joint fine-tuning needs the encoder to see tokens, not the frozen 768-dim
vectors the graph cache stores. This writes one file per graph holding that
graph's node texts in exactly the row order the cached Data object uses, so a
text at position i corresponds to feature row i and to the endpoints recorded in
edge_index.

Alignment is checked per graph rather than assumed: the number of texts must
equal the number of feature rows, and NodeFeaturizer.build sorts node ids, so
the same ordering is reproduced here. A graph whose counts disagree is skipped
and reported rather than written.

    python3 build_node_text_cache.py --workers 24
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T

OUT = Path("/home/moses/node_text_cache")


def one(args):
    sid, dot_path, cache_pkl = args
    out = OUT / f"{sid}.json"
    if out.exists():
        return "already"
    try:
        d = pickle.loads(Path(cache_pkl).read_bytes())
    except Exception:
        return "no_cache"
    try:
        nodes, _ = T.parse_dot(Path(dot_path))
    except Exception:
        return "parse_failed"
    if not nodes:
        return "empty"
    # NodeFeaturizer.build sorts ids and falls back to the type label when the
    # code attribute is blank; both are reproduced exactly.
    node_ids = sorted(nodes.keys())
    texts = [nodes[n]["code"].strip() if nodes[n]["code"].strip()
             else nodes[n]["label"] for n in node_ids]
    if len(texts) != int(d.x.shape[0]):
        return "misaligned"
    out.write_text(json.dumps(texts))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import hashlib
    corpus = [json.loads(l) for l in open(ROOT / "data" / "cwe_corpus.jsonl")]
    if a.limit:
        corpus = corpus[:a.limit]
    jobs = []
    for r in corpus:
        dp = Path(r["dot_path"])
        try:
            st = dp.stat()
        except OSError:
            continue
        key = hashlib.sha256(f"{dp.resolve()}:{st.st_size}:{st.st_mtime_ns}"
                             .encode()).hexdigest()
        pkl = Path("/home/moses/graph_cache_cwe") / f"{key}.pkl"
        if pkl.exists():
            jobs.append((r["sample_id"], str(dp), str(pkl)))
    print(f"[cache] {len(jobs)} graphs with a cache entry", flush=True)

    import collections
    tally = collections.Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=64), 1):
            tally[r] += 1
            if i % 20000 == 0:
                print(f"  {i}/{len(jobs)} {dict(tally)}", flush=True)
    print("[cache] done:", dict(tally))


if __name__ == "__main__":
    main()
