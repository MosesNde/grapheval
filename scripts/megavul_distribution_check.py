"""
megavul_distribution_check.py
==============================
Counts how many MegaVul vulnerable functions map to each OWASP category
using the CWE->OWASP mapping from regenerate_megavul.py.
No CPG generation — just JSON/parquet parsing.
Runs in a few minutes.

Usage:
    python3 megavul_distribution_check.py
"""
from datasets import load_dataset
from collections import Counter

CWE_TO_OWASP = {
    **{c: "A01_Broken_Access_Control" for c in [
        "CWE-22","CWE-23","CWE-35","CWE-59","CWE-200","CWE-201","CWE-219",
        "CWE-264","CWE-275","CWE-276","CWE-284","CWE-285","CWE-352","CWE-359",
        "CWE-377","CWE-402","CWE-425","CWE-441","CWE-497","CWE-538","CWE-540",
        "CWE-548","CWE-552","CWE-566","CWE-601","CWE-639","CWE-651","CWE-668",
        "CWE-706","CWE-862","CWE-863","CWE-913","CWE-922","CWE-1275",
    ]},
    **{c: "A02_Cryptographic_Failures" for c in [
        "CWE-261","CWE-296","CWE-310","CWE-319","CWE-321","CWE-322","CWE-323",
        "CWE-324","CWE-325","CWE-326","CWE-327","CWE-328","CWE-329","CWE-330",
        "CWE-331","CWE-335","CWE-336","CWE-337","CWE-338","CWE-340","CWE-347",
        "CWE-523","CWE-720","CWE-757","CWE-759","CWE-760","CWE-780","CWE-818",
        "CWE-916",
    ]},
    **{c: "A03_Injection" for c in [
        "CWE-20","CWE-74","CWE-75","CWE-77","CWE-78","CWE-79","CWE-80",
        "CWE-83","CWE-87","CWE-88","CWE-89","CWE-90","CWE-91","CWE-93",
        "CWE-94","CWE-95","CWE-96","CWE-97","CWE-98","CWE-99","CWE-100",
        "CWE-113","CWE-116","CWE-138","CWE-184","CWE-470","CWE-471",
        "CWE-564","CWE-610","CWE-643","CWE-644","CWE-652","CWE-917",
        "CWE-119","CWE-120","CWE-121","CWE-122","CWE-123","CWE-124",
        "CWE-125","CWE-126","CWE-127","CWE-128","CWE-129","CWE-130",
        "CWE-131","CWE-134","CWE-135","CWE-170","CWE-193","CWE-194",
        "CWE-415","CWE-416","CWE-476","CWE-590","CWE-665","CWE-680",
        "CWE-762","CWE-763","CWE-787","CWE-824","CWE-825",
    ]},
    **{c: "A04_Insecure_Design" for c in [
        "CWE-73","CWE-183","CWE-189","CWE-190","CWE-191","CWE-192",
        "CWE-209","CWE-213","CWE-235","CWE-256","CWE-257","CWE-266",
        "CWE-269","CWE-280","CWE-311","CWE-312","CWE-313","CWE-316",
        "CWE-369","CWE-400","CWE-401","CWE-404","CWE-405","CWE-406",
        "CWE-419","CWE-430","CWE-434","CWE-444","CWE-451","CWE-472",
        "CWE-501","CWE-522","CWE-525","CWE-539","CWE-561","CWE-570",
        "CWE-571","CWE-579","CWE-598","CWE-602","CWE-617","CWE-636",
        "CWE-637","CWE-638","CWE-642","CWE-646","CWE-650","CWE-653",
        "CWE-656","CWE-657","CWE-670","CWE-672","CWE-675","CWE-676",
        "CWE-681","CWE-682","CWE-683","CWE-684","CWE-691","CWE-696",
        "CWE-698","CWE-754","CWE-755","CWE-758","CWE-770","CWE-772",
        "CWE-779","CWE-783","CWE-799","CWE-820","CWE-835",
    ]},
    **{c: "A05_Security_Misconfiguration" for c in [
        "CWE-2","CWE-11","CWE-13","CWE-15","CWE-16","CWE-260","CWE-315",
        "CWE-520","CWE-526","CWE-537","CWE-541","CWE-547","CWE-611","CWE-614",
        "CWE-756","CWE-776","CWE-942","CWE-1004","CWE-1032","CWE-1174",
    ]},
    **{c: "A06_Vulnerable_and_Outdated_Components" for c in [
        "CWE-1035","CWE-1104",
    ]},
    **{c: "A07_Identification_and_Authentication_Failures" for c in [
        "CWE-255","CWE-259","CWE-287","CWE-288","CWE-290","CWE-294","CWE-295",
        "CWE-297","CWE-300","CWE-301","CWE-302","CWE-303","CWE-304","CWE-305",
        "CWE-306","CWE-307","CWE-346","CWE-384","CWE-521","CWE-613","CWE-620",
        "CWE-640","CWE-798","CWE-940","CWE-1216",
    ]},
    **{c: "A08_Software_and_Data_Integrity_Failures" for c in [
        "CWE-345","CWE-353","CWE-426","CWE-494","CWE-502","CWE-565",
        "CWE-784","CWE-829","CWE-830",
    ]},
    **{c: "A09_Security_Logging_and_Monitoring_Failures" for c in [
        "CWE-117","CWE-223","CWE-532","CWE-778",
    ]},
    "CWE-918": "A10_Server-Side_Request_Forgery",
}

print("Loading MegaVul from HuggingFace...")
dataset = load_dataset("hitoshura25/megavul", split="train")
print(f"Total samples (including non-vulnerable): {len(dataset):,}")

# Show actual field values from first few samples to understand structure
print(f"\nFields: {list(dataset[0].keys())}")
print(f"\nSample field values (first 3 rows):")
for i in range(min(3, len(dataset))):
    item = dataset[i]
    print(f"  [{i}] language={item.get('language')!r}  "
          f"cwe_id={item.get('cwe_id')!r}  "
          f"vulnerable_code_len={len(item.get('vulnerable_code') or '')}  "
          f"is_vul={item.get('is_vul')!r}")
print()

class_counts = Counter()
unmapped     = Counter()
n_vulnerable = 0
skipped_lang  = 0
skipped_short = 0
skipped_notvuln = 0

for item in dataset:
    # In hitoshura25/megavul, ALL rows are vulnerable functions
    # (non-vulnerable functions are not included in this HuggingFace version)
    # The vulnerable_code field contains the pre-patch function
    is_vul = True

    # Filter to C/C++
    lang = (item.get("language") or "").strip().lower()
    if lang and lang not in ("c", "c++", "cpp"):
        skipped_lang += 1
        continue

    # No length filter — include all functions regardless of size
    func = (item.get("vulnerable_code") or item.get("func_before")
            or item.get("func") or "")
    if not func:
        skipped_short += 1
        continue

    n_vulnerable += 1

    # Map CWE -> OWASP
    raw  = item.get("cwe_id") or item.get("cwe") or ""
    cwes = raw if isinstance(raw, list) else [raw]
    owasp = next(
        (CWE_TO_OWASP[c.strip()] for c in cwes
         if c and isinstance(c, str) and CWE_TO_OWASP.get(c.strip())),
        None
    )
    if owasp:
        class_counts[owasp] += 1
    else:
        for c in cwes:
            if c and isinstance(c, str):
                unmapped[c.strip()] += 1

print(f"Vulnerable C/C++ functions (no length filter): {n_vulnerable:,}")
print(f"  Skipped wrong language:  {skipped_lang:,}")
print(f"  Skipped empty function:  {skipped_short:,}")
print(f"\nMapped to OWASP:   {sum(class_counts.values()):,}")
print(f"Unmapped (no CWE match): {sum(unmapped.values()):,}")

print(f"\n{'='*60}")
print(f"  Per-class distribution (usable for training):")
print(f"{'='*60}")
total_mapped = sum(class_counts.values())
for cls, n in sorted(class_counts.items()):
    pct = 100 * n / total_mapped if total_mapped else 0
    feasible = "✓" if n >= 200 else "✗ too few"
    print(f"  {cls:<45} {n:>5,}  ({pct:4.1f}%)  {feasible}")
print(f"{'='*60}")
print(f"  TOTAL                                         {total_mapped:>5,}")
print(f"{'='*60}")

print(f"\nTop 20 unmapped CWEs (largest groups missing from map):")
for cwe, n in unmapped.most_common(20):
    print(f"  {cwe:<15} {n:>5,}")
