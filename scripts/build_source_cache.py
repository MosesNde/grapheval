#!/usr/bin/env python3
"""
build_source_cache.py — extract raw function source for every corpus sample.

WHY
---
Every model tested so far reads code through the graph: node features are
GraphCodeBERT embeddings of each node's snippet, pooled. The `nograph` control
removes the edges but still reads code node-by-node.

A token model reads the whole function as ONE sequence with full self-attention
across it. That is strictly more powerful than pooling per-node embeddings, and
it is the published alternative to graph methods (LineVul, and the token
baseline DeepDFA compares against). Reviewer 2 asked for it by name.

Answering "is the graph necessary?" against that baseline is a comparison
against what the field actually uses, not an ablation of our own pipeline.

OUTPUT
    data/source_cache.jsonl   {sample_id, code}

Run:  python3 build_source_cache.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CPGX = Path("/home/moses/CPGExplainer/data")
OUT = DATA / "source_cache.jsonl"


def main() -> None:
    corpus = [json.loads(l) for l in
              (DATA / "cwe_corpus.jsonl").read_text().splitlines() if l]
    want = {r["sample_id"]: r["provenance"] for r in corpus}
    print(f"[src] need source for {len(want)} samples")
    found: dict[str, str] = {}

    # ── BigVul: sample ids are bigvul-<ordinal> and bigvul-safe-<ordinal>,
    # where the ordinal is the csv.DictReader record index. Replay that loop.
    csv.field_size_limit(2**31 - 1)
    need_ord = {}
    for sid in want:
        if sid.startswith("bigvul-safe-"):
            need_ord[int(sid.rsplit("-", 1)[1])] = (sid, "safe")
        elif sid.startswith("bigvul-"):
            try:
                need_ord[int(sid.rsplit("-", 1)[1])] = (sid, "vuln")
            except ValueError:
                pass
    print(f"[src] BigVul ordinals wanted: {len(need_ord)}")
    with open(CPGX / "MSR_data_cleaned.csv", newline="", encoding="utf-8",
              errors="replace") as f:
        rd = csv.DictReader(f)
        i = -1
        while True:
            i += 1
            try:
                row = next(rd)
            except StopIteration:
                break
            except csv.Error:
                continue
            hit = need_ord.get(i)
            if not hit:
                continue
            sid, kind = hit
            # A vulnerable sample is the pre-patch function; a safe one is the
            # post-patch function, matching how the CPGs were built.
            code = ((row.get("func_before") or "") if kind == "vuln"
                    else (row.get("func_after") or "")).strip()
            if not code:
                code = (row.get("func_before") or "").strip()
            if code:
                found[sid] = code
            if i % 50000 == 0:
                print(f"  scanned {i}", flush=True)
    print(f"[src] BigVul resolved {sum(1 for k in found)}")

    # ── CVEfixes: ids are cvefixes-<vuln_id>-<safe_id>::vuln|safe
    ids = {}
    for sid in want:
        if not sid.startswith("cvefixes-"):
            continue
        base, _, side = sid.partition("::")
        parts = base.split("-")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            ids[parts[1] if side == "vuln" else parts[2]] = sid
    if ids:
        conn = sqlite3.connect(f"file:{CPGX/'CVEfixes.db'}?mode=ro", uri=True)
        q = ",".join("?" * min(len(ids), 900))
        keys = list(ids)
        for start in range(0, len(keys), 900):
            chunk = keys[start:start + 900]
            rows = conn.execute(
                f"SELECT method_change_id, code FROM method_change "
                f"WHERE method_change_id IN ({','.join('?'*len(chunk))})",
                chunk).fetchall()
            for mcid, code in rows:
                if code:
                    found[ids[str(mcid)]] = code
        conn.close()
    print(f"[src] after CVEfixes: {len(found)}")

    with OUT.open("w") as f:
        for sid, code in found.items():
            f.write(json.dumps({"sample_id": sid, "code": code}) + "\n")
    miss = len(want) - len(found)
    print(f"[src] wrote {len(found)} to {OUT}  ({miss} without source)")
    by_prov: dict[str, int] = {}
    for sid in want:
        if sid not in found:
            by_prov[want[sid]] = by_prov.get(want[sid], 0) + 1
    if by_prov:
        print(f"[src] missing by corpus: {by_prov}")


if __name__ == "__main__":
    main()
