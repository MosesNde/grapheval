#!/usr/bin/env python3
"""
export_deepdfa_splits.py — write our audited splits in DeepDFA's named-split
format, so DeepDFA's own code can be run on them unmodified.

DeepDFA (ISU-PAAL/DeepDFA, ICSE 2024) supports named splits: ds_partition()
calls get_named_splits_map(), which reads
storage/external/splits/<name>.csv with columns `example_index` and `split`.
That is the hook -- no modification to their code is required, only a file.

ID MAPPING, VERIFIED
    DeepDFA's `id` is BigVul's own index column ("Unnamed: 0"). Our BigVul
    sample ids are bigvul-<ordinal>, where the ordinal is the csv.DictReader
    record index. These agree: checked by md5 of func_before against the CSV
    row for sampled ids, all matching. Do not assume this for other corpora.

SCOPE, AND WHY IT MATTERS
    DeepDFA runs on BigVul only. Our corpus is BigVul + CVEfixes + Devign, so
    a DeepDFA run covers a SUBSET of our splits. Comparing it against our
    full-corpus numbers would confound method with dataset, so this also emits
    a BigVul-only id list per split, and our own backbones should be re-run on
    that restriction before the two are placed in the same table.

Run:
    python3 export_deepdfa_splits.py --mode repo
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
    ap.add_argument("--out-dir", default=str(DATA / "deepdfa_splits"))
    a = ap.parse_args()

    import pandas as pd

    corpus = {}
    for line in (DATA / "cwe_corpus.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            corpus[r["sample_id"]] = r
    spec = json.loads((DATA / f"splits_{a.mode}_{a.task}.json").read_text())

    rows, kept, dropped = [], {}, 0
    for part, ids in spec["splits"].items():
        n = 0
        for sid in ids:
            # BigVul vulnerable rows only carry an id DeepDFA knows; our
            # re-sampled safe pool (bigvul-safe-*) uses the same ordinals but
            # DeepDFA derives its own labels from the CSV, so both map cleanly.
            if sid.startswith("bigvul-safe-"):
                idx = int(sid.rsplit("-", 1)[1])
            elif sid.startswith("bigvul-"):
                idx = int(sid.rsplit("-", 1)[1])
            else:
                dropped += 1
                continue
            rows.append({"example_index": idx, "split": part})
            n += 1
        kept[part] = n

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates("example_index")
    name = f"ours_{a.mode}"
    df.to_csv(out / f"{name}.csv", index=False)

    summary = {"mode": a.mode, "name": name,
               "kept_by_partition": kept,
               "dropped_non_bigvul": dropped,
               "total_rows": int(len(df))}
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  kept {kept}  (dropped {dropped} non-BigVul samples)")
    print(f"  -> {out/f'{name}.csv'}  ({len(df)} unique ids)")
    print(f"  copy into DeepDFA at storage/external/splits/{name}.csv")

    # BigVul-only id list, so our own models can be re-run on the identical
    # restriction and the comparison stays about the method.
    restrict = {p: [s for s in spec["splits"][p]
                    if s.startswith("bigvul-")] for p in spec["splits"]}
    (out / f"{name}_bigvul_only_sampleids.json").write_text(
        json.dumps({"mode": a.mode, "splits": restrict}, indent=2))
    print(f"  -> BigVul-only sample ids for a matched re-run of our backbones")


if __name__ == "__main__":
    main()
