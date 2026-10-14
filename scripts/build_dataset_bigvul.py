#!/usr/bin/env python3
# build_dataset_bigvul.py
#
# Dataset 2: BigVul (vulnerable + non-vulnerable)
#
# Source:
#   BigVul CSV (Fan et al., MSR 2020)
#   Download: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X
#   Place at: /home/moses/CPGExplainer/data/bigvul_raw.csv
#
# Labels:
#   vul == 1  → VULNERABLE  → dots/bigvul/vulnerable/
#   vul == 0  → NON_VULNERABLE → dots/bigvul/non_vulnerable/
#
# We balance the dataset: cap non-vulnerable at 2× vulnerable count.
#
# Output:
#   /home/moses/CPGExplainer/dots/bigvul/vulnerable/<id>.dot
#   /home/moses/CPGExplainer/dots/bigvul/non_vulnerable/<id>.dot
#   /home/moses/CPGExplainer/data/bigvul_meta.json
#
# Resumable — skips dots already on disk.
# Run in screen: ~12-20 hours with 24 workers.
#
# Run:
#   cd /home/moses/CPGExplainer
#   screen -S bigvul
#   python scripts/build_dataset_bigvul.py 2>&1 | tee data/build_bigvul.log

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE          = Path("/home/moses/CPGExplainer")
CSV_PATH      = BASE / "data" / "MSR_data_cleaned.csv"
OUTPUT_ROOT   = BASE / "dots" / "bigvul"
META_PATH     = BASE / "data" / "bigvul_meta.json"

VULN_DIR      = OUTPUT_ROOT / "vulnerable"
SAFE_DIR      = OUTPUT_ROOT / "non_vulnerable"

JOERN_PARSE   = "/home/moses/bin/joern/joern-cli/joern-parse"
JOERN_EXPORT  = "/home/moses/bin/joern/joern-cli/joern-export"

JOERN_TIMEOUT = 60
N_WORKERS     = 24

# Cap non-vulnerable at this multiple of vulnerable count
# to keep dataset balanced without throwing away too much
SAFE_RATIO    = 2.0


# ─────────────────────────────────────────────────────────────
# BIGVUL CSV LOADING
# ─────────────────────────────────────────────────────────────
def load_bigvul(csv_path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load BigVul CSV (MSR_data_cleaned.csv). Returns (vulnerable, non_vulnerable).

    Confirmed column names:
      func_before  — vulnerable function source
      func_after   — patched function (used as non-vulnerable)
      vul          — 0 or 1
      lang         — 'C', 'C++', etc.
      CVE ID       — CVE identifier
      CWE ID       — CWE identifier
    """
    import csv
    csv.field_size_limit(2**31 - 1)  # max — some BigVul rows exceed 10MB

    vuln = []
    safe = []
    n_skipped = 0

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        i = -1
        while True:
            i += 1
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                n_skipped += 1
                continue
            lang = row.get("lang", "").strip()
            if lang not in ("C", "C++"):
                n_skipped += 1
                continue

            label = str(row.get("vul", "-1")).strip()

            if label == "1":
                func_src = row.get("func_before", "").strip()
                if len(func_src) < 50:
                    n_skipped += 1
                    continue
                vuln.append({
                    "idx":      i,
                    "func_src": func_src,
                    "cve_id":   row.get("CVE ID", ""),
                    "cwe_id":   row.get("CWE ID", ""),
                    "project":  row.get("project", ""),
                    "lang":     lang,
                })
            elif label == "0":
                func_src = row.get("func_after", "").strip()
                if not func_src or len(func_src) < 50:
                    func_src = row.get("func_before", "").strip()
                if len(func_src) < 50:
                    n_skipped += 1
                    continue
                safe.append({
                    "idx":      i,
                    "func_src": func_src,
                    "project":  row.get("project", ""),
                    "lang":     lang,
                })
            else:
                n_skipped += 1

            if (i + 1) % 10000 == 0:
                print(f"  [load] {i+1} rows | vuln={len(vuln)} safe={len(safe)} skip={n_skipped}")

    print(f"[info] BigVul: {len(vuln)} vulnerable, {len(safe)} non-vulnerable "
          f"(C/C++ only, {n_skipped} skipped)")
    return vuln, safe


# ─────────────────────────────────────────────────────────────
# JOERN
# ─────────────────────────────────────────────────────────────
def joern_one(c_text: str, out_dot: Path) -> Tuple[bool, str]:
    if out_dot.exists() and out_dot.stat().st_size > 100:
        return True, "already_exists"

    with tempfile.TemporaryDirectory(prefix="joern_bigvul_") as tmpdir:
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
def _process_one(args: tuple) -> dict:
    record, out_dot_str, label = args
    out_dot = Path(out_dot_str)
    ok, msg = joern_one(record["func_src"], out_dot)

    sample_id = f"bigvul-{record['idx']}"
    base = {
        "sample_id": sample_id,
        "func_name": record.get("func_name", ""),
        "cve_id":    record.get("cve_id", ""),
        "label":     label,
        "source":    "bigvul",
    }

    if ok:
        return {**base, "dot_path": str(out_dot), "status": "ok"}
    else:
        if out_dot.exists():
            try: out_dot.unlink()
            except: pass
        return {**base, "msg": msg, "status": "joern_failed"}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CPGExplainer — Dataset 2: BigVul")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"[fatal] BigVul CSV not found at {CSV_PATH}")
        print(f"[fatal] Download from:")
        print(f"  https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X")
        print(f"  or: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset")
        sys.exit(1)

    if not Path(JOERN_PARSE).exists():
        print(f"[fatal] joern-parse not found at {JOERN_PARSE}")
        sys.exit(1)

    VULN_DIR.mkdir(parents=True, exist_ok=True)
    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    # Resume state
    completed_ids = set()
    existing_samples = []
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                meta = json.load(f)
            existing_samples = meta.get("samples", [])
            completed_ids = {s["sample_id"] for s in existing_samples
                             if s.get("status") == "ok"}
            print(f"[info] Resuming — {len(completed_ids)} done")
        except Exception as e:
            print(f"[warn] Could not load resume state: {e}")

    samples = list(existing_samples)

    # Load BigVul
    vuln_records, safe_records = load_bigvul(CSV_PATH)

    # Balance: cap safe at SAFE_RATIO * vulnerable
    max_safe = int(len(vuln_records) * SAFE_RATIO)
    safe_records = safe_records[:max_safe]
    print(f"[info] Using {len(vuln_records)} vulnerable + {len(safe_records)} non-vulnerable "
          f"(cap={max_safe})")

    # Build work list
    work = []
    for r in vuln_records:
        sid = f"bigvul-{r['idx']}"
        if sid in completed_ids:
            continue
        out_dot = VULN_DIR / f"{sid}.dot"
        work.append((r, str(out_dot), 1))

    for r in safe_records:
        sid = f"bigvul-{r['idx']}"
        if sid in completed_ids:
            continue
        out_dot = SAFE_DIR / f"{sid}.dot"
        work.append((r, str(out_dot), 0))

    print(f"[info] {len(work)} functions to process ({len(completed_ids)} already done)")

    n_ok = n_fail = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_one, work, chunksize=4), 1
        ):
            samples.append(result)
            if result["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            if i % 200 == 0:
                _save_meta(samples, META_PATH)
                print(f"  [{i}/{len(work)}] ok={n_ok} fail={n_fail}")

    _save_meta(samples, META_PATH)

    ok_vuln = sum(1 for s in samples
                  if s.get("status") == "ok" and s.get("label") == 1)
    ok_safe = sum(1 for s in samples
                  if s.get("status") == "ok" and s.get("label") == 0)

    print("\n" + "=" * 60)
    print("DATASET BUILD SUMMARY — BigVul")
    print("=" * 60)
    print(f"  Vulnerable OK:          {ok_vuln}")
    print(f"  Non-vulnerable OK:      {ok_safe}")
    print(f"  Total OK:               {ok_vuln + ok_safe}")
    print(f"  Failed:                 {n_fail}")
    print(f"  Vulnerable dot dir:     {VULN_DIR}")
    print(f"  Non-vulnerable dot dir: {SAFE_DIR}")
    print(f"  Metadata:               {META_PATH}")
    print(f"\nNext step: run dot_preprocess.py with:")
    print(f"  INPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  OUTPUT_DIR = {BASE}/training_data/bigvul/")


def _save_meta(samples, path):
    ok   = sum(1 for s in samples if s.get("status") == "ok")
    fail = sum(1 for s in samples if s.get("status") != "ok")
    with open(path, "w") as f:
        json.dump({"samples": samples, "n_ok": ok, "n_fail": fail}, f)


if __name__ == "__main__":
    main()
