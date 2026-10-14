#!/usr/bin/env python3
"""
audit_clone_random.py — the clone-overlap number Table 2 is missing.

The `random` split is built with clone_check disabled, because it is the control
showing what a plain stratified shuffle admits. A side effect is that
clone_families() then unions pair groups only, so the audit's
clone_family_overlap for that split equals its pair overlap by construction
rather than by measurement.

This computes real clone families over the whole corpus (MinHash, cached) and
measures how many of them straddle the random split's train/test boundary.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from make_splits import clone_families, CORPUS

records = [json.loads(l) for l in open(CORPUS)]
fam = clone_families(records, enable=True)
sp = json.load(open(Path(__file__).resolve().parent.parent/"data"/"splits_random_family.json"))["splits"]
tr, te = set(sp["train"]), set(sp["test"])
ftr = {fam[s] for s in tr if s in fam}
fte = {fam[s] for s in te if s in fam}
overlap = len(ftr & fte)
shared = overlap
n_leaked = sum(1 for s in te if s in fam and fam[s] in ftr)
print(f"\nrandom split, real clone families:")
print(f"  families spanning train and test : {shared}")
print(f"  test samples in a family that also appears in train : {n_leaked}")
print(f"  (table currently reports 1,590, which is the pair count)")
out = Path(__file__).resolve().parent.parent/"runs"/"clone_overlap_random.json"
out.write_text(json.dumps({"clone_families_spanning_train_test": shared,
                           "test_samples_leaked_by_clone_family": n_leaked}, indent=2))
print(f"-> {out}")
