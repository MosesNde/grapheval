#!/usr/bin/env python3
"""
regen_cpgs.py — regenerate the corpus CPGs with per-function language detection.

THE PROBLEM
-----------
Every existing builder hands Joern a file named `in.c`, whatever the function
actually is. C++ then fails to parse and Joern emits a 9-node
`digraph "<global>"` stub carrying no program structure. Measured over the
current corpus (600-sample per cell):

    bigvul   vulnerable  43.7% stubs      cvefixes vulnerable  23.5%
    bigvul   safe         9.8% stubs      cvefixes safe        20.3%
    devign   vulnerable   9.5% stubs      devign   safe         7.5%

BigVul's 34-point gap between labels is the dangerous part: "is this graph
degenerate" predicts the label without reference to the code. The cause is that
its safe pool was the CSV prefix (linux, qemu, php -- C, parses cleanly) while
its vulnerable set spans Chrome and Android (C++, stubs). BigVul's `lang`
column reads "C" for all of them, so it cannot be trusted.

THE FIX, AND WHY IT IS NOT "ALWAYS USE .cpp"
--------------------------------------------
Measured on 40 stubbing functions: 29 parse correctly when written as `.cpp`.
But measured on 40 plain-C functions, 12 that parse fine as `.c` STUB as
`.cpp` -- C is not a subset of what Joern's C++ frontend accepts here. Neither
extension dominates.

So: guess from the source text, try that, and fall back to the other extension
if the result is a stub. Where both succeed the node counts match (median
ratio 1.00), so the choice is not a source of bias.

Output goes to dots_v2/ alongside the originals -- nothing is overwritten until
the result has been checked.

Run:
    python3 regen_cpgs.py --source safe_ext --workers 24
    python3 regen_cpgs.py --source bigvul   --workers 24
    python3 regen_cpgs.py --source cvefixes --workers 24
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CPGX = Path("/home/moses/CPGExplainer")
DATA = CPGX / "data"
OUT_ROOT = CPGX / "dots_v2"
JOERN_PARSE = "/home/moses/bin/joern/joern-cli/joern-parse"
JOERN_EXPORT = "/home/moses/bin/joern/joern-cli/joern-export"
JOERN_TIMEOUT = 60
SEED = 42

# Tokens that only appear in C++ (or are overwhelmingly C++ in this corpus).
_CPP = re.compile(
    r"::|\btemplate\s*<|\bclass\s+\w|\bnamespace\b|\bpublic:|\bprivate:|"
    r"\bprotected:|\bnullptr\b|\bstd::|\bvirtual\b|\bnew\s+\w|\bdelete\s+\w|"
    r"\bthrow\b|\bcatch\s*\(|\bdynamic_cast|\bstatic_cast|\breinterpret_cast|"
    r"\bconst_cast|\bbool\s+\w+\s*=\s*(true|false)\b|\boperator\b")


def guess_ext(src: str) -> str:
    return ".cpp" if _CPP.search(src) else ".c"


def _export(src: str, ext: str, td: Path):
    """Returns (is_real_cpg, node_count, dot_text) or None on Joern failure."""
    cfile, cpg, out = td / f"in{ext}", td / "cpg.bin", td / f"o{ext[1:]}"
    cfile.write_text(src, encoding="utf-8", errors="replace")
    try:
        r = subprocess.run([JOERN_PARSE, str(cfile), "-o", str(cpg)],
                           capture_output=True, text=True, timeout=JOERN_TIMEOUT)
        if r.returncode != 0 or not cpg.exists():
            return None
        r = subprocess.run([JOERN_EXPORT, str(cpg), "--repr", "cpg14",
                            "--format", "dot", "--out", str(out)],
                           capture_output=True, text=True, timeout=JOERN_TIMEOUT)
        if r.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    cands = sorted(((p.stat().st_size, p) for p in out.rglob("*.dot")),
                   reverse=True)
    if not cands:
        return None
    text = cands[0][1].read_text(errors="replace")
    # The cpg14 export always uses HTML-style labels; anything else means the
    # wrong --repr was used and the artefact is not comparable.
    if "<BR/>" not in text[:4000]:
        return None
    is_real = "global" not in text[:60]
    return is_real, text.count("label"), text


def convert(args) -> dict:
    sample_id, src, out_path, meta = args
    out = Path(out_path)
    base = {**meta, "sample_id": sample_id, "dot_path": str(out)}

    if out.exists() and out.stat().st_size > 100:
        head = out.read_text(errors="replace")[:200]
        if "<BR/>" in head:
            return {**base, "status": "ok", "ext": "cached",
                    "stub": "global" in head[:60]}
        out.unlink()

    first = guess_ext(src)
    second = ".c" if first == ".cpp" else ".cpp"
    with tempfile.TemporaryDirectory(prefix="joern_regen_") as td:
        td = Path(td)
        best = None
        used = None
        for ext in (first, second):
            res = _export(src, ext, td)
            if res is None:
                continue
            if best is None or (res[0] and not best[0]) or \
               (res[0] == best[0] and res[1] > best[1]):
                best, used = res, ext
            if res[0]:                      # a real CPG: stop, do not pay twice
                break
        if best is None:
            return {**base, "status": "joern_failed", "ext": None, "stub": None}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(best[2], encoding="utf-8", errors="replace")
        return {**base, "status": "ok", "ext": used, "stub": not best[0],
                "nodes": best[1]}


# ─────────────────────────────────────────────────────────────────────────────
def jobs_bigvul() -> list:
    csv.field_size_limit(2**31 - 1)
    out = OUT_ROOT / "bigvul"
    work = []
    with open(DATA / "MSR_data_cleaned.csv", newline="", encoding="utf-8",
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
            if row.get("lang", "").strip() not in ("C", "C++"):
                continue
            # VULNERABLE ONLY. BigVul's non-vulnerable functions are supplied by
            # --source safe_ext, which samples them per repository. The original
            # safe pool was the CSV prefix and is dropped entirely rather than
            # regenerated: it is biased by construction, and keeping both would
            # convert ~156k functions twice.
            if str(row.get("vul", "-1")).strip() != "1":
                continue
            src = (row.get("func_before") or "").strip()
            if len(src) < 50:
                continue
            sid = f"bigvul-{i}"
            work.append((sid, src, str(out / "vulnerable" / f"{sid}.dot"),
                         {"label": 1, "source": "bigvul"}))
    return work


def jobs_cvefixes() -> list:
    out = OUT_ROOT / "cvefixes"
    conn = sqlite3.connect(f"file:{DATA/'CVEfixes.db'}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT mc_v.method_change_id, mc_s.method_change_id,
               mc_v.code, mc_s.code
        FROM method_change mc_v
        JOIN method_change mc_s
          ON mc_v.file_change_id = mc_s.file_change_id AND mc_v.name = mc_s.name
        JOIN file_change fc ON mc_v.file_change_id = fc.file_change_id
        WHERE mc_v.before_change='True' AND mc_s.before_change='False'
          AND length(mc_v.code)>50 AND length(mc_s.code)>50
          AND fc.programming_language IN ('C','C++') AND mc_v.code != mc_s.code
    """).fetchall()
    conn.close()
    # Same dedup rule as build_cwe_corpus: one pair per vulnerable method,
    # numerically lowest safe id.
    best = {}
    for v, s, cv, cs in rows:
        if v not in best or int(s) < int(best[v][0]):
            best[v] = (s, cv, cs)
    work = []
    for v, (s, cv, cs) in best.items():
        sid = f"cvefixes-{v}-{s}"
        work.append((f"{sid}::vuln", cv,
                     str(out / "vulnerable" / f"{sid}.dot"),
                     {"label": 1, "source": "cvefixes", "pair": sid}))
        work.append((f"{sid}::safe", cs,
                     str(out / "non_vulnerable" / f"{sid}.dot"),
                     {"label": 0, "source": "cvefixes", "pair": sid}))
    return work


def jobs_safe_ext(ratio: float) -> list:
    import build_bigvul_safe as B
    selected, _ = B.scan(ratio)
    out = OUT_ROOT / "bigvul_safe_ext"
    return [(f"bigvul-safe-{r['idx']}", r["func_src"],
             str(out / f"bigvul-safe-{r['idx']}.dot"),
             {"label": 0, "source": "bigvul", "repo": r["repo"],
              "commit": r["commit"], "cve": r["cve"]})
            for r in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["bigvul", "cvefixes", "safe_ext"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ratio", type=float, default=8.0)
    a = ap.parse_args()

    print(f"[plan] collecting {a.source} work list ...", flush=True)
    work = {"bigvul": jobs_bigvul, "cvefixes": jobs_cvefixes,
            "safe_ext": lambda: jobs_safe_ext(a.ratio)}[a.source]()
    random.Random(SEED).shuffle(work)
    print(f"[plan] {len(work)} functions, {a.workers} workers")

    meta_path = DATA / f"{a.source}_v2_meta.json"
    samples, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(convert, work, chunksize=4):
            samples.append(res)
            done += 1
            if done % 2000 == 0:
                ok = sum(1 for s in samples if s["status"] == "ok")
                stub = sum(1 for s in samples if s.get("stub"))
                print(f"  {done}/{len(work)}  ok={ok}  stub={100*stub/max(1,ok):.1f}%",
                      flush=True)
                meta_path.write_text(json.dumps({"samples": samples}, indent=2))

    ok = sum(1 for s in samples if s["status"] == "ok")
    stub = sum(1 for s in samples if s.get("stub"))
    byext = {}
    for s in samples:
        byext[s.get("ext")] = byext.get(s.get("ext"), 0) + 1
    meta_path.write_text(json.dumps(
        {"samples": samples, "n_ok": ok, "n_fail": len(samples) - ok,
         "stub_rate": stub / max(1, ok), "by_extension": byext}, indent=2))
    print(f"\n[done] {ok}/{len(samples)} converted | stub rate "
          f"{100*stub/max(1,ok):.1f}% | extensions {byext}")
    print(f"       meta -> {meta_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
