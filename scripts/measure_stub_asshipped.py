#!/usr/bin/env python3
"""
measure_stub_asshipped.py — verify the pre-fix stub rates by reproducing them.

The paper reports BigVul stub rates of 43.7% (vulnerable) and 9.8% (safe) under
a C-only frontend. Those were measured on CPGs that no longer exist: every graph
was regenerated with language detection, and the truncated safe pool was
replaced by per-repository sampling. Rather than cite a number nobody can check,
this reproduces the original condition directly.

Method: replay load_bigvul in CSV order, take the vulnerable set and the same
first 21,526 safe records the original build used, draw a random sample from
each, hand every function to Joern as `.c` with NO fallback, and count how many
come back as a `<global>` stub. Sampling error at n=1500 is under 1.3 points,
far below the ~34-point gap at issue.
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from build_dataset_bigvul import load_bigvul, SAFE_RATIO
from regen_cpgs import JOERN_PARSE, JOERN_EXPORT, JOERN_TIMEOUT

def stub_as_c(src: str):
    """True if Joern returns a <global> stub when the function is parsed as .c."""
    with tempfile.TemporaryDirectory(prefix="stubchk_") as td:
        td = Path(td); f, cpg, out = td/"in.c", td/"cpg.bin", td/"o"
        f.write_text(src, encoding="utf-8", errors="replace")
        try:
            r = subprocess.run([JOERN_PARSE, str(f), "-o", str(cpg)],
                               capture_output=True, text=True, timeout=JOERN_TIMEOUT)
            if r.returncode != 0 or not cpg.exists(): return None
            r = subprocess.run([JOERN_EXPORT, str(cpg), "--repr", "cpg14",
                                "--format", "dot", "--out", str(out)],
                               capture_output=True, text=True, timeout=JOERN_TIMEOUT)
            if r.returncode != 0: return None
        except (subprocess.TimeoutExpired, OSError):
            return None
        c = sorted(((p.stat().st_size, p) for p in out.rglob("*.dot")), reverse=True)
        if not c: return None
        t = c[0][1].read_text(errors="replace")
        if "<BR/>" not in t[:4000]: return None
        return "global" in t[:60]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    v, s = load_bigvul(Path("/home/moses/CPGExplainer/data/MSR_data_cleaned.csv"))
    prefix = s[:int(len(v)*SAFE_RATIO)]          # the pool the original build used
    rng = random.Random(42)
    sets = {"bigvul vulnerable": rng.sample(v, min(a.n, len(v))),
            "bigvul safe (original prefix)": rng.sample(prefix, min(a.n, len(prefix)))}
    out = {}
    for name, recs in sets.items():
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(stub_as_c, [r["func_src"] for r in recs], chunksize=4))
        ok = [x for x in res if x is not None]
        rate = sum(ok)/len(ok)*100 if ok else float("nan")
        out[name] = {"n_sampled": len(recs), "n_parsed": len(ok),
                     "stub_rate_pct": round(rate, 1)}
        print(f"  {name:32s} {rate:5.1f}%  (n={len(ok)} of {len(recs)})", flush=True)
    g = out["bigvul vulnerable"]["stub_rate_pct"] - out["bigvul safe (original prefix)"]["stub_rate_pct"]
    out["gap_points"] = round(g, 1)
    print(f"\n  gap: {g:.1f} points")
    p = Path("/home/moses/TSE_Journal/runs/stub_asshipped.json")
    p.write_text(json.dumps(out, indent=2)); print(f"-> {p}")

if __name__ == "__main__":
    main()
