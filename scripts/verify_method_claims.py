#!/usr/bin/env python3
"""
verify_method_claims.py — reproduce two Method claims that only ever lived in
a code comment or a prior session's notes.

  1. The MinHash estimator's accuracy: "validated against true Jaccard on 240
     sampled pairs (mean signed error -0.0015)", and "64 permutations missed 4
     of 77 genuine duplicates at the threshold".

  2. The share of BigVul's vulnerable functions sitting in repositories that the
     truncated safe pool left with no non-vulnerable function ("70%").

Writes runs/method_claims.json so both become checkable artifacts.
"""
from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ROOT = HERE.parent
from make_splits import graph_tokens, SHINGLE, JACCARD_T

def shingle_set(tokens):
    if len(tokens) < SHINGLE: return None
    return {" ".join(tokens[i:i+SHINGLE]) for i in range(len(tokens)-SHINGLE+1)} or None

def sig(shingles, n_perm):
    hashed = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
              for s in shingles]
    return tuple(min((h ^ (k * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF for h in hashed)
                 for k in range(n_perm))

def est(a, b): return sum(x == y for x, y in zip(a, b)) / len(a)

def main():
    rng = random.Random(42)
    corpus = [json.loads(l) for l in open(ROOT/"data"/"cwe_corpus.jsonl")]
    out = {}

    # ---- 1. MinHash accuracy -------------------------------------------------
    # Pairs are drawn to include genuine near-duplicates, otherwise almost every
    # random pair has Jaccard ~0 and the threshold is never exercised.
    by_pair = {}
    for r in corpus:
        by_pair.setdefault(r.get("pair_id"), []).append(r)
    dup_pairs = [v for k, v in by_pair.items() if k is not None and len(v) >= 2]
    rng.shuffle(dup_pairs)
    pairs = [(g[0], g[1]) for g in dup_pairs[:400]]
    rng.shuffle(corpus)
    pairs += [(corpus[i], corpus[i+1]) for i in range(0, 400, 2)]

    rows = []
    for a, b in pairs:
        sa, sb = shingle_set(graph_tokens(a["dot_path"])), shingle_set(graph_tokens(b["dot_path"]))
        if not sa or not sb: continue
        true = len(sa & sb) / len(sa | sb)
        rows.append((true, est(sig(sa,64), sig(sb,64)), est(sig(sa,128), sig(sb,128))))
        if len(rows) >= 240: break

    import statistics as st
    e64 = [e - t for t, e, _ in rows]; e128 = [e - t for t, _, e in rows]
    genuine = [r for r in rows if r[0] >= JACCARD_T]
    missed64 = sum(1 for t, e, _ in genuine if e < JACCARD_T)
    missed128 = sum(1 for t, _, e in genuine if e < JACCARD_T)
    out["minhash"] = {
        "n_pairs": len(rows), "n_genuine_duplicates": len(genuine),
        "mean_signed_error_64": round(st.mean(e64), 4),
        "mean_abs_error_64": round(st.mean(abs(x) for x in e64), 4),
        "mean_signed_error_128": round(st.mean(e128), 4),
        "mean_abs_error_128": round(st.mean(abs(x) for x in e128), 4),
        "missed_at_64": missed64, "missed_at_128": missed128}
    print("MinHash validation")
    for k, v in out["minhash"].items(): print(f"  {k:24s} {v}")

    # ---- 2. share of vulnerable functions in safe-less repositories ----------
    from build_dataset_bigvul import load_bigvul, SAFE_RATIO
    v, s = load_bigvul(Path("/home/moses/CPGExplainer/data/MSR_data_cleaned.csv"))
    prefix = s[:int(len(v)*SAFE_RATIO)]
    safe_projects = {r["project"] for r in prefix}
    vuln_projects = {r["project"] for r in v}
    safeless = vuln_projects - safe_projects
    in_safeless = sum(1 for r in v if r["project"] in safeless)
    out["prefix_truncation"] = {
        "vulnerability_bearing_repos": len(vuln_projects),
        "repos_with_no_safe_after_truncation": len(safeless),
        "share_of_vulnerable_functions_in_them_pct": round(in_safeless/len(v)*100, 1)}
    print("\nPrefix truncation")
    for k, val in out["prefix_truncation"].items(): print(f"  {k:44s} {val}")

    p = ROOT/"runs"/"method_claims.json"; p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")

if __name__ == "__main__":
    main()
