#!/usr/bin/env python3
# build_dataset_devign.py
#
# Dataset 3: Devign — C functions from QEMU and FFmpeg
#
#   target=1 → VULNERABLE    → dots/devign/vulnerable/
#   target=0 → NON_VULNERABLE → dots/devign/non_vulnerable/
#
# Source:
#   Zhou et al., "Devign: Effective Vulnerability Identification by
#   Learning Comprehensive Program Semantics via Graph Neural Networks"
#   NeurIPS 2019
#
#   JSON download (function.json):
#   https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
#
#   Place at: /home/moses/CPGExplainer/data/devign.json
#
# Output:
#   /home/moses/CPGExplainer/dots/devign/vulnerable/<id>.dot
#   /home/moses/CPGExplainer/dots/devign/non_vulnerable/<id>.dot
#   /home/moses/CPGExplainer/data/devign_meta.json
#
# Resumable. Run in screen: ~4-6 hours with 24 workers.
#
# Run:
#   cd /home/moses/CPGExplainer
#   screen -S devign
#   python scripts/build_dataset_devign.py 2>&1 | tee data/build_devign.log

import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE         = Path("/home/moses/CPGExplainer")
JSON_PATH    = BASE / "data" / "devign.json"
OUTPUT_ROOT  = BASE / "dots" / "devign"
META_PATH    = BASE / "data" / "devign_meta.json"

VULN_DIR     = OUTPUT_ROOT / "vulnerable"
SAFE_DIR     = OUTPUT_ROOT / "non_vulnerable"

JOERN_PARSE  = "/home/moses/bin/joern/joern-cli/joern-parse"
JOERN_EXPORT = "/home/moses/bin/joern/joern-cli/joern-export"

JOERN_TIMEOUT = 60
N_WORKERS     = 24


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
def load_devign(path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Devign JSON format:
      [{"func": "...", "target": 0|1, "project": "...", "commit_id": "..."}, ...]
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    vuln = []
    safe = []
    for i, r in enumerate(data):
        func = r.get("func", "").strip()
        if len(func) < 50:
            continue
        label = r.get("target", -1)
        record = {
            "idx":       i,
            "func":      func,
            "project":   r.get("project", ""),
            "commit_id": r.get("commit_id", ""),
            "label":     label,
        }
        if label == 1:
            vuln.append(record)
        elif label == 0:
            safe.append(record)

    print(f"[info] Devign: {len(vuln)} vulnerable, {len(safe)} non-vulnerable")
    return vuln, safe


# ─────────────────────────────────────────────────────────────
# JOERN
# ─────────────────────────────────────────────────────────────
def joern_one(c_text: str, out_dot: Path) -> Tuple[bool, str]:
    if out_dot.exists() and out_dot.stat().st_size > 100:
        return True, "already_exists"

    with tempfile.TemporaryDirectory(prefix="joern_devign_") as tmpdir:
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
    record, out_dot_str = args
    out_dot = Path(out_dot_str)
    ok, msg = joern_one(record["func"], out_dot)

    sid  = f"devign-{record['idx']}"
    base = {
        "sample_id": sid,
        "project":   record["project"],
        "commit_id": record["commit_id"],
        "label":     record["label"],
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
    print("CPGExplainer — Dataset 3: Devign")
    print("=" * 60)

    if not JSON_PATH.exists():
        print(f"[fatal] devign.json not found at {JSON_PATH}")
        print(f"[fatal] Download from:")
        print(f"  https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF")
        print(f"  Place at: {JSON_PATH}")
        sys.exit(1)
    if not Path(JOERN_PARSE).exists():
        print(f"[fatal] joern-parse not found at {JOERN_PARSE}")
        sys.exit(1)

    VULN_DIR.mkdir(parents=True, exist_ok=True)
    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    # Resume
    completed_ids = set()
    existing      = []
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                meta = json.load(f)
            existing      = meta.get("samples", [])
            completed_ids = {s["sample_id"] for s in existing
                             if s.get("status") == "ok"}
            print(f"[info] Resuming — {len(completed_ids)} done")
        except Exception as e:
            print(f"[warn] Could not load resume state: {e}")

    samples = list(existing)
    vuln_records, safe_records = load_devign(JSON_PATH)

    # Build work list
    work = []
    for r in vuln_records:
        sid = f"devign-{r['idx']}"
        if sid in completed_ids:
            continue
        work.append((r, str(VULN_DIR / f"{sid}.dot")))

    for r in safe_records:
        sid = f"devign-{r['idx']}"
        if sid in completed_ids:
            continue
        work.append((r, str(SAFE_DIR / f"{sid}.dot")))

    print(f"[info] {len(work)} functions to process ({len(completed_ids)} already done)")

    n_ok = n_fail = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_one, work, chunksize=4), 1
        ):
            samples.append(result)
            if result["status"] == "ok": n_ok += 1
            else: n_fail += 1
            if i % 100 == 0:
                _save(samples, META_PATH)
                print(f"  [{i}/{len(work)}] ok={n_ok} fail={n_fail}")

    _save(samples, META_PATH)
    ok_total = len(completed_ids) + n_ok

    ok_vuln = sum(1 for s in samples
                  if s.get("status") == "ok" and s.get("label") == 1)
    ok_safe = sum(1 for s in samples
                  if s.get("status") == "ok" and s.get("label") == 0)

    print("\n" + "=" * 60)
    print("SUMMARY — Devign")
    print("=" * 60)
    print(f"  Total OK:               {ok_total}")
    print(f"  Vulnerable OK:          {ok_vuln}")
    print(f"  Non-vulnerable OK:      {ok_safe}")
    print(f"  Failed this run:        {n_fail}")
    print(f"  Vulnerable dots:        {VULN_DIR}")
    print(f"  Non-vulnerable dots:    {SAFE_DIR}")
    print(f"\nNext: run dot_preprocess.py")
    print(f"  INPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  OUTPUT_DIR = {BASE}/training_data/devign/")


def _save(samples, path):
    ok   = sum(1 for s in samples if s.get("status") == "ok")
    fail = len(samples) - ok
    with open(path, "w") as f:
        json.dump({"samples": samples, "n_ok": ok, "n_fail": fail}, f)


if __name__ == "__main__":
    main()
