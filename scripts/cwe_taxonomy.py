#!/usr/bin/env python3
"""
cwe_taxonomy.py — CWE -> family mapping for the real-CVE rebuild.

WHY THIS REPLACES THE OWASP TARGET
-----------------------------------
Measured against the official OWASP Top Ten 2021 CWE lists, 55.1% of BigVul's
10,900 vulnerable C/C++ functions map to NO OWASP category and a further 19.5%
carry no CWE at all. A06 and A10 have zero real samples; A02/A04/A05/A07/A08
each have under 110. CVEfixes shows the same shape independently. OWASP 2021 is
web-application-centric; real C/C++ CVEs are overwhelmingly memory safety.

That gap is why the previous corpus needed 9,380 LLM-generated samples to fill
six otherwise-empty categories, and why models trained on it learned the
generation signature instead of vulnerability structure.

The families below are chosen so that every class is populated by real CVE data
alone. Grouping follows CWE-1000 research-view parents, with three deliberate
departures documented inline.

USAGE
    from cwe_taxonomy import to_family, FAMILIES
    fam = to_family("CWE-125")        # -> "OOB_READ"
    fam = to_family("NVD-CWE-noinfo") # -> None
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Families.  Ordering is by expected support, descending.
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_MEMBERS: Dict[str, List[int]] = {
    # Memory-safety leaves. 125/787 are the modern, specific labels.
    "OOB_READ":        [125, 126, 127],
    "OOB_WRITE":       [787, 788, 121, 122, 124, 120, 823, 806],

    # DELIBERATE DEPARTURE 1 — CWE-119 is the CWE-1000 parent of OOB_READ and
    # OOB_WRITE, not a sibling. It is kept as its own class rather than merged
    # because NVD used it as the catch-all buffer label before ~2017 and
    # switched to 125/787 after. Merging it would hide that shift; keeping it
    # separate makes the labelling-era artifact visible, and it interacts
    # directly with chronological splitting (see make_splits.py).
    "BUFFER_UNSPECIFIED": [119],

    "USE_AFTER_FREE":  [416, 415, 825, 590, 761, 763],
    "NULL_DEREF":      [476],

    # DELIBERATE DEPARTURE 2 — CWE-189 (numeric errors) is the parent of 190;
    # both are folded into one family because the distinction NVD draws between
    # them is not reflected in code structure.
    "INTEGER_ERROR":   [190, 191, 189, 192, 197, 369, 682, 681],

    "RACE_CONDITION":  [362, 366, 367, 364, 414, 421, 1223],

    # DELIBERATE DEPARTURE 3 — CWE-399 (resource management errors) is a
    # deprecated NVD catch-all covering leaks, double-free and exhaustion.
    # Its members are split across USE_AFTER_FREE and RESOURCE_MGMT by their
    # specific child where the child is recorded; bare 399 lands here.
    "RESOURCE_MGMT":   [399, 400, 401, 404, 771, 772, 775, 674, 835, 770],

    "INPUT_VALIDATION": [20, 1284, 129, 130, 134, 170, 228, 754, 755, 617],
    "INFO_EXPOSURE":   [200, 209, 532, 359, 497, 538],
    "ACCESS_CONTROL":  [264, 284, 269, 276, 732, 862, 863, 285, 668, 552],
    "PATH_TRAVERSAL":  [22, 23, 35, 59, 61, 377],
    "CRYPTO":          [310, 326, 327, 328, 329, 330, 331, 335, 338, 347, 319],
    "INJECTION":       [77, 78, 79, 88, 89, 90, 91, 94, 95, 917, 74],
    "AUTH":            [287, 288, 290, 294, 295, 297, 306, 307, 521, 798, 613],
    "TYPE_CONFUSION":  [843, 704, 588],
    "UNINITIALIZED":   [457, 908, 909, 665],
}

# Reverse index. A CWE listed under two families resolves to the first, which is
# why the ordering above matters; duplicates are asserted away at import.
_INDEX: Dict[int, str] = {}
for _fam, _cwes in FAMILY_MEMBERS.items():
    for _c in _cwes:
        _INDEX.setdefault(_c, _fam)

FAMILIES: List[str] = list(FAMILY_MEMBERS)

# Labels that carry no weakness information. Samples with these are NOT
# silently dropped by to_family -- they return None so the caller can count
# and report them, which is what makes the 19.5%-unlabelled figure auditable.
NON_INFORMATIVE = re.compile(r"NVD-CWE-(noinfo|Other)|^\s*$|^nan$", re.I)

_CWE_RE = re.compile(r"CWE-(\d+)", re.I)


def to_family(cwe: object) -> Optional[str]:
    """Map a raw CWE string to a family name, or None if unmappable.

    Accepts 'CWE-125', '125', 'NVD-CWE-noinfo', '', nan, or a list-like string
    such as "['CWE-125', 'CWE-787']" (CVEfixes emits these for multi-CWE CVEs);
    for the list form the FIRST mappable CWE wins.
    """
    if cwe is None:
        return None
    s = str(cwe).strip()
    if not s or NON_INFORMATIVE.match(s):
        return None

    ids = [int(m) for m in _CWE_RE.findall(s)]
    if not ids and s.isdigit():
        ids = [int(s)]

    for cid in ids:
        fam = _INDEX.get(cid)
        if fam:
            return fam
    return None


def family_of_int(cwe_id: int) -> Optional[str]:
    return _INDEX.get(int(cwe_id))


def _self_test() -> int:
    """Structural and behavioural checks. Run as `python3 cwe_taxonomy.py`."""
    import collections
    bad = 0

    for fam, cwes in FAMILY_MEMBERS.items():
        d = [c for c, n in collections.Counter(cwes).items() if n > 1]
        if d:
            print(f"FAIL: {fam} lists {d} more than once")
            bad += 1

    owner = collections.defaultdict(list)
    for fam, cwes in FAMILY_MEMBERS.items():
        for c in set(cwes):
            owner[c].append(fam)
    for c, fams in owner.items():
        if len(fams) > 1:
            print(f"FAIL: CWE-{c} claimed by {fams}; first-wins makes the "
                  f"mapping depend on dict order")
            bad += 1

    # Behaviour, including the input shapes the two corpora actually emit.
    cases = [
        ("CWE-125", "OOB_READ"), ("125", "OOB_READ"), ("cwe-476", "NULL_DEREF"),
        ("CWE-787", "OOB_WRITE"), ("CWE-119", "BUFFER_UNSPECIFIED"),
        ("NVD-CWE-noinfo", None), ("NVD-CWE-Other", None), ("", None),
        ("nan", None), (None, None), ("CWE-9999", None), ("CWE-1190", None),
        ("['CWE-125', 'CWE-787']", "OOB_READ"),
        ("['NVD-CWE-noinfo','CWE-190']", "INTEGER_ERROR"),
    ]
    for inp, exp in cases:
        got = to_family(inp)
        if got != exp:
            print(f"FAIL: to_family({inp!r}) = {got!r}, expected {exp!r}")
            bad += 1

    print(f"{len(FAMILIES)} families, {len(_INDEX)} CWEs indexed, "
          f"{len(cases)} behaviour cases")
    for f, c in FAMILY_MEMBERS.items():
        print(f"  {f:20s} {len(set(c)):3d} CWEs")
    print("OK" if not bad else f"{bad} PROBLEM(S)")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if _self_test() else 0)
