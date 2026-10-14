#!/usr/bin/env python3
"""
make_splits.py — leakage-resistant splits for the real-CVE CWE-family corpus.

WHAT THE REVIEWERS ASKED FOR
    "The manuscript reports a single split stratified by category and label,
     but does not state that it is grouped by vulnerable/safe pair, original
     file, commit, repository, prompt family, or clone family. ... The main
     experiment should be repeated with pair-grouped, clone-aware, and
     repository-disjoint splits."                            (Reviewer 2, #2)

    "...chronological splitting, deduplication, and realistic low-base-rate
     evaluation."                                            (Reviewer 1, #3)

This produces splits that satisfy all of it, and -- just as important --
an audit that proves they do, so the claim can be made in the paper with a
number attached rather than an assurance.

SPLIT MODES
    repo    repository-disjoint. The default and the one to headline: no
            project appears in more than one partition, so a model cannot
            score by recognising a codebase's house style.
    time    chronological by CVE year. Train on the past, test on the future,
            which is the only split that matches how a detector is deployed.
            Interacts with the CWE-119 labelling shift (see cwe_taxonomy.py):
            NVD used 119 as the buffer catch-all before ~2017 and 125/787
            after, so a chronological split moves label mass between families.
            That is a real deployment effect, not a bug, but it must be
            reported alongside the numbers.
    group   pair/commit-grouped but repository-mixed. Isolates exactly what
            repository overlap is worth: everything else is still controlled.
    random  NO grouping of any kind -- a plain stratified shuffle over
            individual functions. This is what most published GNN
            vulnerability-detection results use, so it is the control that
            makes the comparison against `repo` interpretable. It deliberately
            leaves duplicates, patched twins and same-project code on both
            sides; that is the point.

NEAR-DUPLICATE HANDLING
    Exact and near-duplicate functions are collapsed into clone families
    BEFORE splitting, and a family is an atomic unit. Detection is a MinHash
    over token 5-shingles of each CPG's node CODE attributes -- cheap, needs
    no source checkout, and catches the vendored-copy and backported-patch
    cases that dominate C/C++ corpora.

Run:
    python3 make_splits.py --mode repo   --min-support 500
    python3 make_splits.py --mode time   --test-from 2019
    python3 make_splits.py --audit-only  data/splits_repo.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "cwe_corpus.jsonl"
OUT_DIR = ROOT / "data"

SEED = 42
# Validated against true Jaccard on 240 sampled pairs: the estimator is
# essentially unbiased (mean signed error -0.0015, mean abs error 0.017), but
# at N_PERM=64 its standard error near the threshold is ~0.045, and 4 of 77
# genuine duplicates fell below the cut. For a leakage control a false negative
# is the costly direction, so N_PERM is doubled to halve the variance.
N_PERM = 128
BAND_ROWS = 4
N_BANDS = N_PERM // BAND_ROWS
SHINGLE = 5
JACCARD_T = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Near-duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
# Joern's dot export writes one node per line as:
#     "<id>" [label = <TYPE, LINE<BR/>SOURCE TEXT> ]
# The source text is everything after the FIRST <BR/>. Anchoring the terminator
# on '> ]' at end of line matters: a naive [^>]* stops at the '>' inside <BR/>
# and captures only "TYPE, LINE", which is identical scaffolding for every C
# function. That collapses every graph into one clone family.
_LABEL_RE = re.compile(r"\[\s*label\s*=\s*<(.*)>\s*\]\s*$", re.I | re.M)
_BR_RE = re.compile(r"<BR\s*/?>", re.I)
_TOK_RE = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")
_ENTITY = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'"}


def graph_tokens(dot_path: str, cap: int = 6000) -> list[str]:
    """Token stream from the SOURCE TEXT carried by a .dot CPG's nodes.

    Identifier names are kept: renaming is exactly the kind of trivial edit a
    clone check must see through, but wholesale identifier normalisation would
    merge genuinely distinct functions that share a skeleton.

    Note Joern truncates long node code with an ellipsis in the dot export, so
    this is a partial view of each function -- still far more discriminative
    than node types, but it means near-duplicate recall is a lower bound.
    """
    try:
        text = Path(dot_path).read_text(errors="replace")
    except OSError:
        return []

    toks: list[str] = []
    for m in _LABEL_RE.finditer(text):
        label = m.group(1)
        parts = _BR_RE.split(label, maxsplit=1)
        if len(parts) < 2:
            continue                       # type/line only, carries no code
        code = parts[1]
        for ent, ch in _ENTITY.items():
            code = code.replace(ent, ch)
        toks.extend(_TOK_RE.findall(code))
        if len(toks) >= cap:
            break
    return toks[:cap]


def minhash(tokens: list[str]) -> tuple[int, ...] | None:
    if len(tokens) < SHINGLE:
        return None
    shingles = {" ".join(tokens[i:i + SHINGLE])
                for i in range(len(tokens) - SHINGLE + 1)}
    if not shingles:
        return None
    hashed = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                             "big") for s in shingles]
    sig = []
    for k in range(N_PERM):
        salt = k * 0x9E3779B97F4A7C15
        sig.append(min((h ^ salt) & 0xFFFFFFFFFFFFFFFF for h in hashed))
    return tuple(sig)


class Union:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def clone_families(records: list[dict], enable: bool) -> dict[str, str]:
    """sample_id -> clone-family id, based on CODE IDENTITY only.

    Pair groups are unioned in unconditionally (a function and its patched twin
    are the same code); exact-duplicate and near-duplicate signatures are added
    on top. Commit grouping deliberately lives in split_by_group() instead --
    see the note there.
    """
    uf = Union()
    for r in records:
        uf.union(f"pair::{r['pair_id']}", r["sample_id"])

    if enable:
        # Signatures depend only on the file and the hashing parameters, and
        # every split mode hashes the same corpus. Caching turns a ~20 minute
        # pass into seconds for every run after the first, which matters
        # because the evaluation protocol needs several splits.
        cache_path = OUT_DIR / f"minhash_cache_p{N_PERM}_s{SHINGLE}.json"
        cache: dict[str, list] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
                print(f"  [clone] loaded {len(cache)} cached signatures")
            except (ValueError, OSError):
                cache = {}

        print("  [clone] hashing CPGs ...", flush=True)
        sigs: dict[str, tuple[int, ...]] = {}
        bands = collections.defaultdict(list)
        n = 0
        n_new = n_cached = 0
        for r in records:
            hit = cache.get(r["dot_path"])
            if hit is not None:
                sig = tuple(hit) if hit else None
                n_cached += 1
            else:
                sig = minhash(graph_tokens(r["dot_path"]))
                cache[r["dot_path"]] = list(sig) if sig else []
                n_new += 1
            if sig is None:
                continue
            sigs[r["sample_id"]] = sig
            n += 1
            # LSH banding. Candidate generation only -- every pair is verified
            # against the full signature below, so this is tuned for recall.
            for b in range(N_BANDS):
                bands[(b, sig[b * BAND_ROWS:(b + 1) * BAND_ROWS])].append(
                    r["sample_id"])
            if n % 5000 == 0:
                print(f"  [clone] hashed {n}", flush=True)
        # Count reads, not signatures: n excludes graphs too small to hash, so
        # "n - n_new" went negative and printed "-17 from cache".
        print(f"  [clone] {n} graphs signed from {n_cached + n_new} read "
              f"({n_new} newly hashed, {n_cached} cached)")
        if n_new:
            try:
                cache_path.write_text(json.dumps(cache))
            except OSError as e:
                print(f"  [clone] could not write signature cache: {e}")

        # Banding produces CANDIDATES, not matches. Unioning every candidate
        # lets single-linkage chain A~B~C~... into one component that swallows
        # the corpus -- an earlier run produced a single family covering all
        # 42,585 graphs, leaving 8 samples for test. Each candidate pair is
        # therefore verified against the full signature before being unioned,
        # and degenerate buckets are skipped outright.
        # Exact duplicates first, and unconditionally. They are precisely the
        # samples that land in ENORMOUS bands -- 300 copies of one trivial
        # function collide in every band -- so the bucket-size cap below would
        # skip them and silently miss the most blatant duplication in the
        # corpus. Identical signatures need no verification.
        by_sig = collections.defaultdict(list)
        for sid, sig in sigs.items():
            by_sig[sig].append(sid)
        n_exact = 0
        for members in by_sig.values():
            for m in members[1:]:
                uf.union(members[0], m)
                n_exact += 1
        print(f"  [clone] {n_exact} exact-duplicate graphs merged "
              f"({len(by_sig)} distinct signatures)")

        MAX_BUCKET = 40
        checked: set[tuple[str, str]] = set()
        n_cand = n_union = n_skipped = 0
        for members in bands.values():
            if len(members) > MAX_BUCKET:
                n_skipped += 1
                continue
            for i, a in enumerate(members):
                for b_ in members[i + 1:]:
                    key = (a, b_) if a < b_ else (b_, a)
                    if key in checked:
                        continue
                    checked.add(key)
                    n_cand += 1
                    sa, sb = sigs[a], sigs[b_]
                    est = sum(x == y for x, y in zip(sa, sb)) / len(sa)
                    if est >= JACCARD_T:
                        uf.union(a, b_)
                        n_union += 1
        print(f"  [clone] {n_cand} candidate pairs, {n_union} verified "
              f"(Jaccard>={JACCARD_T}), {n_skipped} degenerate buckets skipped")

        fam_sizes = collections.Counter(uf.find(s) for s in sigs)
        biggest = fam_sizes.most_common(1)[0][1] if fam_sizes else 0
        multi = sum(1 for v in fam_sizes.values() if v > 1)
        print(f"  [clone] {len(fam_sizes)} families | {multi} with >1 member "
              f"| largest {biggest} ({biggest/max(1,n):.1%} of corpus)")
        if biggest > 0.10 * n:
            print("  [clone] WARNING: largest family exceeds 10% of the corpus; "
                  "single-linkage collapse likely -- inspect before trusting "
                  "the split")

    return {r["sample_id"]: uf.find(r["sample_id"]) for r in records}


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────
def split_by_repo(records, fam_of, ratios):
    """Assign whole repositories, largest first, to whichever partition is
    furthest below its quota. Greedy rather than random because repository
    sizes are heavy-tailed: Chrome alone is 3,939 BigVul functions, and a
    random assignment routinely lands it wholly in test."""
    by_repo = collections.defaultdict(list)
    for r in records:
        by_repo[r["repo"] or f"__norepo__{r['sample_id']}"].append(r)

    def rkey(r):
        return r["repo"] or f"__norepo__{r['sample_id']}"

    uf = Union()

    # A clone family that straddles repositories forces those repositories to
    # travel together, otherwise the clone check is defeated by the repo split.
    fam_repo = collections.defaultdict(set)
    for r in records:
        fam_repo[fam_of[r["sample_id"]]].add(rkey(r))
    for repos in fam_repo.values():
        rs = sorted(repos)
        for other in rs[1:]:
            uf.union(rs[0], other)

    # Repository NAMES are not repository IDENTITIES. BigVul records a bare
    # project string ("Chrome") while CVEfixes records a clone URL
    # ("https://github.com/chromium/chromium"), so one upstream project can
    # enter the corpus under two labels and a name-disjoint split still puts
    # the same codebase on both sides. A shared commit hash is proof of shared
    # identity regardless of naming, so repos sharing any commit are unioned.
    # Without this the audit reports 122 overlapping commits at train&test.
    commit_repo = collections.defaultdict(set)
    for r in records:
        if r.get("commit"):
            commit_repo[r["commit"]].add(rkey(r))
    n_merged = 0
    for repos in commit_repo.values():
        rs = sorted(repos)
        for other in rs[1:]:
            if uf.find(rs[0]) != uf.find(other):
                n_merged += 1
            uf.union(rs[0], other)
    if n_merged:
        print(f"  [repo] merged {n_merged} repository labels that share "
              f"commits across corpora")

    blocks = collections.defaultdict(list)
    for repo, rs in by_repo.items():
        blocks[uf.find(repo)].extend(rs)

    order = sorted(blocks.values(), key=len, reverse=True)

    # How much freedom does the assignment actually have? If a class keeps most
    # of its samples inside one indivisible block, no policy can hit a 70/10/20
    # share for it, and chasing that with tie-break tuning would be fitting the
    # heuristic to noise. Reporting the ceiling makes the limitation a stated
    # property of repository-disjoint evaluation rather than a silent artefact.
    fam_tot = collections.Counter(r["family"] for r in records
                                  if r["label"] == 1 and r["family"])
    conc = {}
    for f, tot_f in fam_tot.items():
        conc[f] = max(sum(1 for r in b if r["family"] == f)
                      for b in order) / max(1, tot_f)
    top = sorted(conc.items(), key=lambda kv: -kv[1])[:4]
    head_share = sum(len(b) for b in order[:12]) / max(1, len(records))
    print(f"  [repo] {len(order)} blocks; the 12 largest hold "
          f"{head_share:.0%} of the corpus")
    print("  [repo] most concentrated families (share in a single block): "
          + ", ".join(f"{f} {s:.0%}" for f, s in top))

    names = ["train", "val", "test"]
    return _greedy_assign(blocks_order=order, records=records,
                          names=names, ratios=ratios)


def _greedy_assign(blocks_order, records, names, ratios):
    """Place whole blocks, holding partition SIZE first and using prevalence
    and per-family shape only to break ties.

    Weighting matters and the obvious weighting is wrong. Prevalence in this
    corpus is a per-repository property, not a sample-level one: the Linux
    block sits at 0.268, qemu at 0.042, php at 0.038 and Chrome at 1.000 --
    every Chrome record BigVul contributes is vulnerable, because its
    non-vulnerable functions were capped and drawn from elsewhere. Weighting
    prevalence equally with size therefore tears the partitions apart; a first
    attempt produced 13,752 train against 24,019 test.

    Repository-disjointness and balanced base rates are in genuine tension
    here, and no assignment policy resolves it -- the fix is upstream, in
    generating the missing non-vulnerable CPGs so each repository contributes
    both classes. Until then size leads, the residual prevalence gap is
    reported by the audit, and every base-rate-sensitive metric must be quoted
    against its partition's stated prevalence.
    """
    blocks_order = list(blocks_order)
    total = len(records)
    n_vuln = sum(1 for r in records if r["label"] == 1)
    fam_tot = collections.Counter(r["family"] for r in records
                                  if r["label"] == 1 and r["family"])

    # One greedy pass over one block order is an arbitrary point in a large
    # search space, and sweeping the tie-break tolerance showed why: the worst
    # class-share deviation moved 0.271 -> 0.150 -> 0.323 -> 0.165 across
    # tolerances 0.15/0.25/0.35/0.50, i.e. noise, not signal. Tuning against
    # that would be fitting the heuristic to chance. Instead the objective is
    # written down and optimised directly: run several deterministic restarts
    # that keep the biggest blocks placed first (they have no real freedom) and
    # shuffle the rest, then keep the assignment that best matches the target
    # sizes and per-family shares. Seeded, so the result is reproducible.
    HEAD = 12                      # blocks large enough that order is forced
    RESTARTS = 24

    def score(out) -> tuple[float, float]:
        size_err = max(abs(len(out[n]) / max(1, total) - ratios[i])
                       for i, n in enumerate(names))
        fam_dev = 0.0
        for f, tot_f in fam_tot.items():
            for i, n in enumerate(names):
                got = sum(1 for r in out[n]
                          if r["label"] == 1 and r["family"] == f)
                fam_dev = max(fam_dev, abs(got / max(1, tot_f) - ratios[i]))
        return (size_err, fam_dev)

    tgt_size = {n: ratios[i] * total for i, n in enumerate(names)}
    tgt_vuln = {n: ratios[i] * n_vuln for i, n in enumerate(names)}
    tgt_fam = {n: {f: ratios[i] * c for f, c in fam_tot.items()}
               for i, n in enumerate(names)}

    def one_pass(order):
        have_size = {n: 0 for n in names}
        have_vuln = {n: 0 for n in names}
        have_fam = {n: collections.Counter() for n in names}
        out = {n: [] for n in names}

        for blk in order:
            b_v = sum(1 for r in blk if r["label"] == 1)
            b_f = collections.Counter(r["family"] for r in blk
                                      if r["label"] == 1 and r["family"])

            # Primary rule: fill whichever partition has the largest REMAINING
            # capacity. Minimising post-placement distance instead looks
            # reasonable and is not -- it sends a block wherever it "fits"
            # tightest, so the 9,692-record Linux block lands in the 8,517-slot
            # test partition and every later block compensates.
            deficit = {n: tgt_size[n] - have_size[n] for n in names}
            best = max(deficit.values())
            span = max(1.0, abs(best))
            # Near-ties on capacity are settled on distribution shape.
            contenders = [n for n in names
                          if best - deficit[n] <= 0.15 * span] or names

            def tie(n):
                c = abs(have_vuln[n] + b_v - tgt_vuln[n]) / max(1.0, tgt_vuln[n])
                for f, t in tgt_fam[n].items():
                    c += 0.2 * abs(have_fam[n][f] + b_f.get(f, 0) - t) / max(1.0, t)
                return c

            pick = min(contenders, key=tie)
            out[pick].extend(blk)
            have_size[pick] += len(blk)
            have_vuln[pick] += b_v
            have_fam[pick].update(b_f)
        return out

    head, tail = blocks_order[:HEAD], blocks_order[HEAD:]
    best_out, best_score = None, None
    for k in range(RESTARTS):
        order = head + tail if k == 0 else head + random.Random(SEED + k).sample(
            tail, len(tail))
        cand = one_pass(order)
        sc = score(cand)
        if best_score is None or sc < best_score:
            best_out, best_score = cand, sc
    print(f"  [assign] {RESTARTS} restarts | best size error {best_score[0]:.4f}, "
          f"worst family-share deviation {best_score[1]:.3f}")
    return best_out


def split_random(records, ratios):
    """Plain stratified shuffle over individual functions -- no grouping.

    This is the permissive protocol most published results use. It is included
    as an experimental CONTROL, never as a way to report a better number: it
    leaves near-duplicates, a function and its own patched twin, and code from
    the same repository on both sides of the split. Comparing it against `repo`
    on identical data with an identical model measures what those shortcuts are
    worth.
    """
    from sklearn.model_selection import train_test_split
    y = [f"{r['label']}|{r['family'] or ''}" for r in records]
    idx = list(range(len(records)))
    tr, rest = train_test_split(idx, test_size=ratios[1] + ratios[2],
                                random_state=SEED, stratify=y)
    y_rest = [y[i] for i in rest]
    va, te = train_test_split(rest,
                              test_size=ratios[2] / (ratios[1] + ratios[2]),
                              random_state=SEED, stratify=y_rest)
    print("  [random] NO grouping applied -- duplicates, patched twins and "
          "same-repo code\n           may appear on both sides. Permissive "
          "control, not a result.")
    return {"train": [records[i] for i in tr],
            "val": [records[i] for i in va],
            "test": [records[i] for i in te]}


def split_by_time(records, test_from: int, val_from: int, fam_of=None,
                  keep_undated: bool = False):
    out = {"train": [], "val": [], "test": []}
    undated = []
    for r in records:
        y = r.get("year")
        if y is None:
            undated.append(r)
        elif y >= test_from:
            out["test"].append(r)
        elif y >= val_from:
            out["val"].append(r)
        else:
            out["train"].append(r)

    # Undated samples have no position on the timeline, so a chronological
    # protocol cannot place them. Dumping them all into train (the previous
    # behaviour) is not neutral: it silently shifts train's corpus mix and base
    # rate relative to test. Dropping them is the honest default; --keep-undated
    # restores the old behaviour for anyone who wants the extra training data
    # and is willing to report the distortion.
    if undated:
        if keep_undated:
            out["train"].extend(undated)
            print(f"  [time] {len(undated)} undated samples placed in train "
                  f"(distorts train's corpus mix and base rate)")
        else:
            print(f"  [time] dropped {len(undated)} undated samples "
                  f"(no CVE year; cannot be placed chronologically)")

    # A chronological split is defined by the date, so a repository legitimately
    # appears on both sides -- that is the protocol, not a defect. Near-duplicate
    # CODE spanning the cutoff is a different matter: a function vendored,
    # backported or barely edited across the boundary lets the model recall
    # rather than generalise. An unpurged run leaves 46 such families straddling
    # train and test. Later partitions yield, since train must stay the past.
    if fam_of is not None:
        seen = {fam_of[r["sample_id"]] for r in out["train"]}
        for later in ("val", "test"):
            kept, dropped = [], 0
            for r in out[later]:
                if fam_of[r["sample_id"]] in seen:
                    dropped += 1
                else:
                    kept.append(r)
            out[later] = kept
            seen |= {fam_of[r["sample_id"]] for r in kept}
            if dropped:
                print(f"  [time] dropped {dropped} {later} samples whose clone "
                      f"family already appears in an earlier partition")
    return out


def split_by_group(records, fam_of, ratios):
    """Pair/commit/clone-grouped but repository-MIXED, so the paper can price
    what repository overlap was worth.

    Commit grouping belongs here rather than in clone_families(). BigVul gives
    every row its own pair_id, yet 64.9% of its records sit in commits holding
    BOTH a vulnerable and a non-vulnerable function -- the touched and untouched
    code of one patch -- so without it this mode would split a patch across
    partitions. `repo` mode already unions repositories that share a commit and
    needs no help; folding commits into clone_families() instead would coarsen
    ITS blocks for nothing, and measurably did: ACCESS_CONTROL's test cell fell
    from 384 to 62 and INFO_EXPOSURE's from 207 to 46.
    """
    uf = Union()
    for r in records:
        uf.union(f"fam::{fam_of[r['sample_id']]}", r["sample_id"])
        if r.get("commit"):
            uf.union(f"commit::{r['commit']}", r["sample_id"])

    fams = collections.defaultdict(list)
    for r in records:
        fams[uf.find(r["sample_id"])].append(r)
    keys = sorted(fams)
    random.Random(SEED).shuffle(keys)
    return _greedy_assign(blocks_order=[fams[k] for k in keys], records=records,
                          names=["train", "val", "test"], ratios=ratios)


# ─────────────────────────────────────────────────────────────────────────────
def audit(splits, fam_of) -> dict:
    """Prove the split is what it claims. Every number here is reportable."""
    part = {n: {r["sample_id"] for r in rs} for n, rs in splits.items()}
    idx = {r["sample_id"]: r for rs in splits.values() for r in rs}

    def keyset(name, key):
        return {idx[s].get(key) for s in part[name] if idx[s].get(key)}

    res: dict = {"sizes": {n: len(v) for n, v in part.items()}}
    for key in ("repo", "commit", "pair_id"):
        res[f"{key}_overlap"] = {
            f"{a}&{b}": len(keyset(a, key) & keyset(b, key))
            for a, b in (("train", "test"), ("train", "val"), ("val", "test"))}

    famset = {n: {fam_of[s] for s in part[n]} for n in part}
    res["clone_family_overlap"] = {
        f"{a}&{b}": len(famset[a] & famset[b])
        for a, b in (("train", "test"), ("train", "val"), ("val", "test"))}

    for n, rs in splits.items():
        v = [r for r in rs if r["label"] == 1]
        res.setdefault("prevalence", {})[n] = round(len(v) / max(1, len(rs)), 4)
        res.setdefault("family_dist", {})[n] = dict(
            collections.Counter(r["family"] for r in v if r["family"]).most_common())
        res.setdefault("corpus_dist", {})[n] = dict(
            collections.Counter(r["provenance"] for r in rs))
        yrs = [r["year"] for r in rs if r.get("year")]
        if yrs:
            res.setdefault("year_range", {})[n] = [min(yrs), max(yrs)]
    return res


def print_audit(a: dict, mode: str = "repo") -> None:
    print("\n" + "=" * 68)
    print("  SPLIT AUDIT")
    print("=" * 68)
    print(f"  sizes: {a['sizes']}")
    print(f"  prevalence: {a.get('prevalence')}")
    if "year_range" in a:
        print(f"  year range: {a['year_range']}")
    print(f"  corpus mix: {a.get('corpus_dist')}")
    # A partition whose corpus mix diverges from the whole confounds whatever
    # the split is meant to isolate. It bites hardest chronologically: BigVul
    # ends at 2019 (median 2015) while CVEfixes runs to 2024 (median 2019), so
    # a 2019 cutoff separates the two CORPORA almost as cleanly as it separates
    # years, and a "temporal" result is then partly a corpus-shift result.
    cd = a.get("corpus_dist", {})
    if cd:
        overall = collections.Counter()
        for d in cd.values():
            overall.update(d)
        tot = sum(overall.values()) or 1
        worst = 0.0
        for part, d in cd.items():
            n = sum(d.values()) or 1
            for corp, gtot in overall.items():
                worst = max(worst, abs(d.get(corp, 0) / n - gtot / tot))
        if worst > 0.20:
            print(f"\n  WARNING: corpus mix diverges by up to {worst:.0%} between a "
                  f"partition and the corpus\n           as a whole. Results will "
                  f"partly reflect corpus shift, not the factor this\n           "
                  f"split is meant to isolate. Consider --corpus to hold it fixed.")

    pv = a.get("prevalence", {})
    if pv:
        gap = max(pv.values()) - min(pv.values())
        if gap > 0.05:
            print(f"\n  NOTE: prevalence spans {gap:.3f} across partitions "
                  f"(min {min(pv.values()):.3f}, max {max(pv.values()):.3f}).")
            print("        Base rate is a per-repository property in this corpus, so a "
                  "repository-\n        disjoint split cannot equalise it. Quote "
                  "precision, PR-AUC and any tuned\n        threshold against the "
                  "partition prevalence stated above.")
    # In chronological mode a repository SHOULD span the cutoff -- the split is
    # defined by date, not by project -- so flagging that as a leak would be
    # wrong. Every other key must still be clean.
    expected = ({"repo_overlap"} if mode in ("time", "group") else set())
    if mode == "random":
        expected = {"repo_overlap", "commit_overlap", "pair_id_overlap",
                    "clone_family_overlap"}
    print("\n  leakage checks (all must be 0 for train&test):")
    for k in ("repo_overlap", "commit_overlap", "pair_id_overlap",
              "clone_family_overlap"):
        tt = a[k]["train&test"]
        if k in expected:
            verdict = ("EXPECTED -- permissive control"
                       if mode == "random"
                       else f"by design ({mode} split does not separate repos)")
        else:
            verdict = "OK" if tt == 0 else "<-- LEAK"
        print(f"    {k:24s} train&test = {tt:<6d} {verdict}")
    print("\n  family distribution per partition:")
    fams = sorted({f for d in a["family_dist"].values() for f in d})
    print(f"    {'family':24s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for f in fams:
        print(f"    {f:24s} "
              f"{a['family_dist']['train'].get(f,0):7d} "
              f"{a['family_dist']['val'].get(f,0):7d} "
              f"{a['family_dist']['test'].get(f,0):7d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repo", "time", "group", "random"],
                    default="repo")
    ap.add_argument("--min-support", type=int, default=500,
                    help="drop families with fewer real samples than this")
    ap.add_argument("--test-from", type=int, default=2019)
    ap.add_argument("--val-from", type=int, default=2018)
    ap.add_argument("--ratios", default="0.7,0.1,0.2")
    ap.add_argument("--corpus", default=None,
                    help="restrict to one provenance (bigvul|cvefixes|devign); "
                         "use for chronological runs to hold corpus fixed")
    ap.add_argument("--keep-undated", action="store_true",
                    help="time mode: put undated samples in train instead of "
                         "dropping them (distorts train's mix)")
    ap.add_argument("--no-clone-check", action="store_true",
                    help="skip MinHash (fast, but the audit cannot prove "
                         "clone-disjointness)")
    ap.add_argument("--task", choices=["binary", "family"], default="family")
    ap.add_argument("--audit-only")
    a = ap.parse_args()

    if a.audit_only:
        d = json.loads(Path(a.audit_only).read_text())
        print_audit(d["audit"], d.get("mode", "repo"))
        return

    records = [json.loads(l) for l in CORPUS.read_text().splitlines() if l]
    print(f"loaded {len(records)} records")
    if a.corpus:
        records = [r for r in records if r["provenance"] == a.corpus]
        print(f"restricted to corpus={a.corpus}: {len(records)} records")

    if a.task == "family":
        # Fine-grained task: only CWE-carrying corpora, only families with
        # enough real support to be worth a class.
        support = collections.Counter(r["family"] for r in records
                                      if r["label"] == 1 and r["family"])
        keep = {f for f, n in support.items() if n >= a.min_support}
        print(f"keeping {len(keep)} families with >={a.min_support} samples: "
              f"{sorted(keep)}")
        records = [r for r in records
                   if (r["label"] == 1 and r["family"] in keep)
                   or (r["label"] == 0 and r["provenance"] != "devign")]
        print(f"  -> {len(records)} records for the fine-grained task")

    fam_of = clone_families(records, enable=not a.no_clone_check)
    ratios = [float(x) for x in a.ratios.split(",")]

    if a.mode == "random":
        splits = split_random(records, ratios)
    elif a.mode == "repo":
        splits = split_by_repo(records, fam_of, ratios)
    elif a.mode == "time":
        splits = split_by_time(records, a.test_from, a.val_from, fam_of,
                               keep_undated=a.keep_undated)
    else:
        splits = split_by_group(records, fam_of, ratios)

    aud = audit(splits, fam_of)
    print_audit(aud, a.mode)

    suffix = f"_{a.corpus}" if a.corpus else ""
    out = OUT_DIR / f"splits_{a.mode}_{a.task}{suffix}.json"
    out.write_text(json.dumps({
        "mode": a.mode, "task": a.task, "seed": SEED,
        "corpus": a.corpus, "keep_undated": a.keep_undated,
        "min_support": a.min_support,
        "clone_check": not a.no_clone_check,
        "splits": {n: [r["sample_id"] for r in rs] for n, rs in splits.items()},
        "audit": aud,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
