#!/usr/bin/env python3
# build_dataset_cvefixes.py
#
# Dataset 1: CVEfixes — paired before/after C/C++ functions
#
#   before_change='True'  → VULNERABLE    → dots/cvefixes/vulnerable/
#   before_change='False' → NON_VULNERABLE → dots/cvefixes/non_vulnerable/
#
# Both come from the same CVE patch record — the strongest possible
# ground truth. The detector learns what distinguishes the vulnerable
# version from the fixed version of the same function.
#
# Output:
#   /home/moses/CPGExplainer/dots/cvefixes/vulnerable/<id>.dot
#   /home/moses/CPGExplainer/dots/cvefixes/non_vulnerable/<id>.dot
#   /home/moses/CPGExplainer/data/cvefixes_meta.json
#
# Resumable. Run in screen: ~6-10 hours with 24 workers.
#
# Run:
#   cd /home/moses/CPGExplainer
#   screen -S cvefixes
#   python scripts/build_dataset_cvefixes.py 2>&1 | tee data/build_cvefixes.log

import json
import multiprocessing as mp
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE         = Path("/home/moses/CPGExplainer")
DB_PATH      = BASE / "data" / "CVEfixes.db"
OUTPUT_ROOT  = BASE / "dots" / "cvefixes"
META_PATH    = BASE / "data" / "cvefixes_meta.json"

VULN_DIR     = OUTPUT_ROOT / "vulnerable"
SAFE_DIR     = OUTPUT_ROOT / "non_vulnerable"

JOERN_PARSE  = "/home/moses/bin/joern/joern-cli/joern-parse"
JOERN_EXPORT = "/home/moses/bin/joern/joern-cli/joern-export"

JOERN_TIMEOUT = 60
N_WORKERS     = 24


# ─────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────
def fetch_paired_records(db_path: Path) -> List[dict]:
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            mc_v.method_change_id   AS vuln_id,
            mc_s.method_change_id   AS safe_id,
            mc_v.name               AS func_name,
            mc_v.code               AS vuln_code,
            mc_s.code               AS safe_code,
            fc.filename             AS filename,
            fc.programming_language AS lang
        FROM method_change mc_v
        JOIN method_change mc_s
          ON mc_v.file_change_id = mc_s.file_change_id
         AND mc_v.name           = mc_s.name
        JOIN file_change fc
          ON mc_v.file_change_id = fc.file_change_id
        WHERE mc_v.before_change = 'True'
          AND mc_s.before_change = 'False'
          AND mc_v.code IS NOT NULL AND length(mc_v.code) > 50
          AND mc_s.code IS NOT NULL AND length(mc_s.code) > 50
          AND fc.programming_language IN ('C', 'C++')
          AND mc_v.code != mc_s.code
    """)
    cols    = [d[0] for d in cur.description]
    records = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    print(f"[info] CVEfixes: {len(records)} paired C/C++ records "
          f"→ {len(records)*2} binary samples")
    return records


# ─────────────────────────────────────────────────────────────
# JOERN
# ─────────────────────────────────────────────────────────────
def joern_one(c_text: str, out_dot: Path) -> Tuple[bool, str]:
    if out_dot.exists() and out_dot.stat().st_size > 100:
        return True, "already_exists"

    with tempfile.TemporaryDirectory(prefix="joern_cvefixes_") as tmpdir:
        tmpdir  = Path(tmpdir)
        c_file  = tmpdir / "input.c"
        cpg_bin = tmpdir / "cpg.bin"
        c_file.write_text(c_text, encoding="utf-8", errors="replace")

        try:
            r = subprocess.run(
                [JOERN_PARSE, str(c_file), "-o", str(cpg_bin)],
                capture_output=True, text=True, timeout=JOERN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, "parse_timeout"
        except FileNotFoundError as e:
            return False, f"joern_not_found:{e}"
        if r.returncode != 0:
            return False, f"parse_exit_{r.returncode}"

        export_dir = tmpdir / "export"
        try:
            r = subprocess.run(
                [JOERN_EXPORT, str(cpg_bin),
                 "--repr", "cpg14", "--format", "dot",
                 "--out", str(export_dir)],
                capture_output=True, text=True, timeout=JOERN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, "export_timeout"
        if r.returncode != 0:
            return False, f"export_exit_{r.returncode}"

        candidates = []
        for f in export_dir.rglob("*.dot"):
            try:
                size = f.stat().st_size
                if size < 100:
                    continue
                head = f.read_bytes()[:2000].decode("utf-8", errors="ignore")
                if "METHOD" in head:
                    candidates.append((size, f))
            except Exception:
                continue

        if not candidates:
            return False, "no_method_dots"

        candidates.sort(reverse=True)
        out_dot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(candidates[0][1], out_dot)
        return True, "ok"


# ─────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────
def _process_pair(args: tuple) -> dict:
    record, vuln_dot_str, safe_dot_str = args
    vuln_dot = Path(vuln_dot_str)
    safe_dot = Path(safe_dot_str)

    ok_v, msg_v = joern_one(record["vuln_code"], vuln_dot)
    ok_s, msg_s = joern_one(record["safe_code"], safe_dot)

    sid  = f"cvefixes-{record['vuln_id']}-{record['safe_id']}"
    base = {"sample_id": sid, "func_name": record["func_name"],
            "filename": record["filename"]}

    if ok_v and ok_s:
        return {**base, "vuln_dot": str(vuln_dot),
                "safe_dot": str(safe_dot), "status": "ok"}
    else:
        for dot in [vuln_dot, safe_dot]:
            if dot.exists():
                try: dot.unlink()
                except: pass
        return {**base, "vuln_msg": msg_v, "safe_msg": msg_s,
                "status": "joern_failed"}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CPGExplainer — Dataset 1: CVEfixes")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"[fatal] CVEfixes.db not found at {DB_PATH}")
        print(f'  wget -O {DB_PATH} "https://zenodo.org/record/7029359/files/CVEfixes.db?download=1"')
        sys.exit(1)
    if not Path(JOERN_PARSE).exists():
        print(f"[fatal] joern-parse not found at {JOERN_PARSE}")
        sys.exit(1)

    VULN_DIR.mkdir(parents=True, exist_ok=True)
    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    # Resume
    completed_ids  = set()
    existing       = []
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                meta = json.load(f)
            existing      = meta.get("samples", [])
            completed_ids = {s["sample_id"] for s in existing
                             if s.get("status") == "ok"}
            print(f"[info] Resuming — {len(completed_ids)} pairs done")
        except Exception as e:
            print(f"[warn] Could not load resume state: {e}")

    samples = list(existing)
    records = fetch_paired_records(DB_PATH)

    work = []
    for r in records:
        sid = f"cvefixes-{r['vuln_id']}-{r['safe_id']}"
        if sid in completed_ids:
            continue
        work.append((r,
                     str(VULN_DIR / f"{sid}.dot"),
                     str(SAFE_DIR / f"{sid}.dot")))

    print(f"[info] {len(work)} pairs to process ({len(completed_ids)} already done)")

    n_ok = n_fail = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_pair, work, chunksize=4), 1
        ):
            samples.append(result)
            if result["status"] == "ok": n_ok += 1
            else: n_fail += 1
            if i % 100 == 0:
                _save(samples, META_PATH)
                print(f"  [{i}/{len(work)}] ok={n_ok} fail={n_fail}")

    _save(samples, META_PATH)
    ok_total = len(completed_ids) + n_ok

    print("\n" + "=" * 60)
    print("SUMMARY — CVEfixes")
    print("=" * 60)
    print(f"  Pairs OK (total):       {ok_total}")
    print(f"  Binary samples:         {ok_total * 2}")
    print(f"  Failed this run:        {n_fail}")
    print(f"  Vulnerable dots:        {VULN_DIR}")
    print(f"  Non-vulnerable dots:    {SAFE_DIR}")
    print(f"\nNext: run dot_preprocess.py")
    print(f"  INPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  OUTPUT_DIR = {BASE}/training_data/cvefixes/")


def _save(samples, path):
    ok   = sum(1 for s in samples if s.get("status") == "ok")
    fail = len(samples) - ok
    with open(path, "w") as f:
        json.dump({"samples": samples, "n_ok": ok, "n_fail": fail}, f)


if __name__ == "__main__":
    main()
