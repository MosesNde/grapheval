#!/usr/bin/env python3
"""
export_for_baselines.py — write our audited splits in the input format each
published baseline expects, so the AUTHORS' OWN CODE can be run unmodified.

WHY EXPORT RATHER THAN REIMPLEMENT
----------------------------------
Reporting a number under a method's name requires that method's implementation,
not an approximation of its architecture. So instead of porting models into our
pipeline, we push our data into theirs: their model definition, their training
loop, their hyperparameters, our splits.

Holding the split fixed is the whole point of the comparison, so the split is
the one thing that must come from us. Everything else stays theirs.

LineVul expects a CSV with columns `processed_func` (raw function text) and
`target` (0/1), passed as --train_data_file / --eval_data_file /
--test_data_file.

Run:
    python3 export_for_baselines.py --mode repo
    python3 export_for_baselines.py --mode random --out-dir data/baseline_csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--task", default="family")
    ap.add_argument("--split-suffix", default="")
    ap.add_argument("--out-dir", default=str(DATA / "baseline_csv"))
    a = ap.parse_args()

    import pandas as pd

    src = {}
    for line in (DATA / "source_cache.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            src[r["sample_id"]] = r["code"]
    corpus = {}
    for line in (DATA / "cwe_corpus.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            corpus[r["sample_id"]] = r

    spec = json.loads(
        (DATA / f"splits_{a.mode}_{a.task}{a.split_suffix}.json").read_text())
    out = Path(a.out_dir) / f"{a.mode}{a.split_suffix}"
    out.mkdir(parents=True, exist_ok=True)

    # LineVul's loader reads train/val/test by filename, so the partition names
    # are fixed by its interface rather than ours.
    name_map = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
    summary = {}
    for part, ids in spec["splits"].items():
        rows = [{"processed_func": src[s], "target": int(corpus[s]["label"])}
                for s in ids if s in src]
        df = pd.DataFrame(rows)
        path = out / name_map[part]
        df.to_csv(path, index=False)
        summary[part] = {
            "rows": len(df),
            "dropped_no_source": len(ids) - len(df),
            "prevalence": round(float(df["target"].mean()), 4) if len(df) else 0.0,
        }
        print(f"  [{part}] {len(df)} rows -> {path.name} "
              f"(prevalence {summary[part]['prevalence']}, "
              f"{summary[part]['dropped_no_source']} without source)")

    (out / "export_summary.json").write_text(json.dumps(
        {"mode": a.mode, "task": a.task, "suffix": a.split_suffix,
         "partitions": summary}, indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
