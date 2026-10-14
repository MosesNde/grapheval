#!/usr/bin/env python3
"""
build_bigvul_safe.py — convert the non-vulnerable BigVul functions that the
original build skipped, sampled PER REPOSITORY.

WHY
---
build_dataset_bigvul.py capped the safe pool with

    safe_records = safe_records[:max_safe]        # SAFE_RATIO = 2.0

which takes the first 21,526 rows in CSV order rather than a sample. The result
is not merely a smaller safe pool, it is a structurally skewed one:

    221 of 287 BigVul repositories have vulnerable functions and ZERO safe ones,
    and those repositories hold 70% of BigVul's vulnerable functions.

Chrome, Android, ImageMagick, php-src, tcpdump and jasper are therefore 100%
vulnerable in the corpus, while qemu sits at 0.042 and php at 0.050. Two
consequences, and the second is the serious one:

  1. Repository-disjoint splits cannot equalise base rate, because base rate is
     a property of the repository rather than of the sample.

  2. Within training, "which repository is this" becomes a usable proxy for the
     label. That is the same Clever Hans failure that sank the OWASP corpus --
     a different mechanism (sampling, not generation) producing the same
     disease. Fixing it is a validity requirement, not a nicety.

WHAT THIS DOES
--------------
Streams the CSV once, groups non-vulnerable C/C++ functions by project, and
selects up to `--ratio` safe functions per vulnerable function IN THAT SAME
project, so every repository contributes both classes at a comparable rate.
Selection within a project is a seeded random sample, not a prefix.

Then runs Joern over whatever is not already on disk, in parallel, resumable.

    python3 build_bigvul_safe.py --scan-only          # report, convert nothing
    python3 build_bigvul_safe.py --ratio 4 --workers 24
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CSV_PATH = Path("/home/moses/CPGExplainer/data/MSR_data_cleaned.csv")
OUT_DOTS = Path("/home/moses/CPGExplainer/dots/bigvul_safe_ext")
OUT_META = Path("/home/moses/CPGExplainer/data/bigvul_safe_ext_meta.json")
EXISTING_META = Path("/home/moses/CPGExplainer/data/bigvul_meta.json")

JOERN_PARSE = "/home/moses/bin/joern/joern-cli/joern-parse"
JOERN_EXPORT = "/home/moses/bin/joern/joern-cli/joern-export"
JOERN_TIMEOUT = 60
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
def scan(ratio: float) -> tuple[list[dict], dict]:
    """One CSV pass. Returns the selected safe records and a per-repo report.

    The reader loop mirrors build_dataset_bigvul.py exactly -- same
    csv.DictReader, same increment order -- so `idx` stays comparable with the
    bigvul-<idx> sample ids already on disk.
    """
    csv.field_size_limit(2**31 - 1)

    vuln_by_repo: collections.Counter = collections.Counter()
    safe_by_repo: dict[str, list[dict]] = collections.defaultdict(list)

    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        i = -1
        while True:
            i += 1
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                continue
            if row.get("lang", "").strip() not in ("C", "C++"):
                continue

            label = str(row.get("vul", "-1")).strip()
            repo = (row.get("project") or "").strip()
            if label == "1":
                if len((row.get("func_before") or "").strip()) >= 50:
                    vuln_by_repo[repo] += 1
            elif label == "0":
                src = (row.get("func_after") or "").strip()
                if len(src) < 50:
                    src = (row.get("func_before") or "").strip()
                if len(src) < 50:
                    continue
                safe_by_repo[repo].append({
                    "idx": i, "func_src": src, "repo": repo,
                    "commit": (row.get("commit_id") or "").strip(),
                    "cve": (row.get("CVE ID") or "").strip(),
                })
            if i % 50000 == 0:
                print(f"  scanned {i} rows", flush=True)

    rng = random.Random(SEED)
    selected: list[dict] = []
    report = {"per_repo": {}, "ratio": ratio}
    for repo, nv in vuln_by_repo.items():
        pool = safe_by_repo.get(repo, [])
        want = int(round(nv * ratio))
        take = pool if len(pool) <= want else rng.sample(pool, want)
        selected.extend(take)
        report["per_repo"][repo] = {
            "vulnerable": nv, "safe_available": len(pool),
            "safe_selected": len(take), "shortfall": max(0, want - len(pool)),
        }

    # Repositories with safe functions but no vulnerable ones contribute
    # nothing to the fine-grained task and are skipped, but they matter for the
    # binary stage's negative pool, so record what was left on the table.
    report["safe_in_repos_with_no_vulns"] = sum(
        len(v) for k, v in safe_by_repo.items() if k not in vuln_by_repo)
    report["n_selected"] = len(selected)
    report["n_repos_with_vulns"] = len(vuln_by_repo)
    report["n_repos_now_covered"] = sum(
        1 for r in report["per_repo"].values() if r["safe_selected"] > 0)
    report["n_repos_still_empty"] = sum(
        1 for r in report["per_repo"].values() if r["safe_selected"] == 0)
    return selected, report


# ─────────────────────────────────────────────────────────────────────────────
def joern_one(args) -> dict:
    idx, src, repo, commit, cve = args
    out_dot = OUT_DOTS / f"bigvul-safe-{idx}.dot"
    base = {"sample_id": f"bigvul-safe-{idx}", "label": 0, "source": "bigvul",
            "repo": repo, "commit": commit, "cve": cve,
            "dot_path": str(out_dot)}
    if out_dot.exists() and out_dot.stat().st_size > 100:
        # Resume must also re-check format, or a bad earlier run is inherited.
        if "<BR/>" in out_dot.read_text(errors="replace")[:4000]:
            return {**base, "status": "ok", "msg": "already_exists"}
        out_dot.unlink()

    with tempfile.TemporaryDirectory(prefix="joern_bvsafe_") as td:
        td = Path(td)
        cfile, cpg, expdir = td / "in.c", td / "cpg.bin", td / "out"
        cfile.write_text(src, encoding="utf-8", errors="replace")
        try:
            p = subprocess.run([JOERN_PARSE, str(cfile), "-o", str(cpg)],
                               capture_output=True, text=True,
                               timeout=JOERN_TIMEOUT)
            if p.returncode != 0 or not cpg.exists():
                return {**base, "status": "joern_failed", "msg": "parse"}
            # MUST match build_dataset_bigvul.py exactly: --repr cpg14 --out.
            # Using "--repr all" instead produces a structurally DIFFERENT
            # artefact -- a whole-file export (~60 nodes, CODE="<empty>",
            # attribute-style labels) rather than a per-method CPG (~569 nodes,
            # HTML-style <TYPE, LINE<BR/>CODE> labels). Mixing the two would
            # correlate graph encoding with the label, since every new graph is
            # non-vulnerable: a far worse shortcut than the one this script
            # exists to remove.
            p = subprocess.run([JOERN_EXPORT, str(cpg),
                                "--repr", "cpg14", "--format", "dot",
                                "--out", str(expdir)],
                               capture_output=True, text=True,
                               timeout=JOERN_TIMEOUT)
            if p.returncode != 0:
                return {**base, "status": "joern_failed", "msg": "export"}
        except subprocess.TimeoutExpired:
            return {**base, "status": "joern_failed", "msg": "timeout"}
        except OSError as e:
            return {**base, "status": "joern_failed", "msg": f"os:{e}"}

        cands = sorted(((p.stat().st_size, p) for p in expdir.rglob("*.dot")),
                       reverse=True)
        if not cands:
            return {**base, "status": "joern_failed", "msg": "no_dots"}

        # Verify the artefact is the SAME KIND as the existing corpus before
        # keeping it. A silently-different export is what made the first run of
        # this script worthless, and nothing downstream would have caught it --
        # the graphs loaded, counted and split fine, they just encoded the
        # label in their node format.
        head = cands[0][1].read_text(errors="replace")[:4000]
        if "<BR/>" not in head:
            return {**base, "status": "wrong_format",
                    "msg": "not a cpg14 per-method export"}

        out_dot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cands[0][1], out_dot)
        return {**base, "status": "ok", "msg": "ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="safe functions to select per vulnerable function, "
                         "per repository")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scan-only", action="store_true")
    a = ap.parse_args()

    print(f"[scan] streaming {CSV_PATH.name} (ratio={a.ratio}) ...")
    selected, report = scan(a.ratio)

    print("\n" + "=" * 68)
    print(f"  {report['n_selected']} safe functions selected across "
          f"{report['n_repos_now_covered']}/{report['n_repos_with_vulns']} "
          f"repositories that have vulnerabilities")
    print(f"  repositories still with no safe function: "
          f"{report['n_repos_still_empty']}")
    print(f"  safe functions in repos with no vulnerabilities (unused here): "
          f"{report['safe_in_repos_with_no_vulns']}")
    print("=" * 68)
    print(f"\n  {'repo':22s} {'vuln':>7s} {'avail':>8s} {'take':>7s} {'short':>7s}")
    for repo, d in sorted(report["per_repo"].items(),
                          key=lambda kv: -kv[1]["vulnerable"])[:15]:
        print(f"  {repo[:22]:22s} {d['vulnerable']:7d} {d['safe_available']:8d} "
              f"{d['safe_selected']:7d} {d['shortfall']:7d}")

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    (OUT_META.parent / "bigvul_safe_ext_scan.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n  scan report -> {OUT_META.parent/'bigvul_safe_ext_scan.json'}")

    if a.scan_only:
        print("  --scan-only: stopping before conversion.")
        return

    OUT_DOTS.mkdir(parents=True, exist_ok=True)
    work = [(r["idx"], r["func_src"], r["repo"], r["commit"], r["cve"])
            for r in selected]
    # `selected` is built repo by repo, so an interrupted run would leave the
    # early repositories complete and the later ones untouched -- reintroducing
    # exactly the per-repository skew this script exists to remove. Shuffling
    # makes any partial result an unbiased subset of the intended sample.
    random.Random(SEED).shuffle(work)
    print(f"\n[joern] converting {len(work)} functions with {a.workers} workers")
    print("        (order shuffled: a partial run is still an unbiased sample)")

    samples, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(joern_one, work, chunksize=4):
            samples.append(res)
            done += 1
            if done % 2000 == 0:
                ok = sum(1 for s in samples if s["status"] == "ok")
                print(f"  {done}/{len(work)}  ok={ok}", flush=True)
                OUT_META.write_text(json.dumps(
                    {"samples": samples,
                     "n_ok": ok, "n_fail": done - ok}, indent=2))

    ok = sum(1 for s in samples if s["status"] == "ok")
    OUT_META.write_text(json.dumps(
        {"samples": samples, "n_ok": ok, "n_fail": len(samples) - ok},
        indent=2))
    print(f"\n[done] {ok}/{len(samples)} converted -> {OUT_DOTS}")
    print(f"       meta -> {OUT_META}")
    print("       next: add a loader for this meta to build_cwe_corpus.py")


if __name__ == "__main__":
    main()
