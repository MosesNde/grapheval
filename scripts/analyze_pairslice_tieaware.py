#!/usr/bin/env python3
"""
analyze_pairslice_tieaware.py -- aggregate the paired vulnerable/patched runs
into the numbers the paper reports.

A strict win rate charges every tie as a failure. Where a patch leaves no trace
in the graph the two members can featurise identically, the model returns the
same score, and the pair is undecidable rather than wrong. This script reports
the disposition of every pair as win/loss/tie, both the strict and the tie-aware
accuracy, and two intervals that answer different questions:

  run interval   -- t interval over the runs. Asks how much the figure moves
                    when training is repeated. It is labelled "run" and not
                    "seed" deliberately: repeating a run at a fixed seed does
                    not reproduce it, because set_seed does not make the CUDA
                    scatter reductions used by message passing deterministic
                    and the WeightedRandomSampler is built without an explicit
                    generator. Two sets of five runs at identical configuration
                    differ by up to 0.13 on a single cell.
  pair interval  -- bootstrap over pairs of each pair's mean outcome across the
                    five seeds. Asks how much it moves with a different sample
                    of pairs. Pairs are aligned across seeds by sample id and
                    the alignment is asserted, not assumed.

Point estimates come from the win/loss/tie counts the run wrote at full
precision. The pair interval is rebuilt from the stored per-pair scores, which
in the -tieaware runs were rounded to six decimals; that resolution turns a pair
separated by ~1e-7 into a tie and moves the interval centre by about 0.002, well
inside the interval width. pair_slice_test.py now stores full precision and an
explicit per-pair outcome code, so later runs need no such allowance.
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("/home/moses/TSE_Journal")
SLICES = ["structure-only", "text-differs", "near-identical"]
ARMS = ["graph", "nograph"]


def load(tag):
    fs = sorted(glob.glob(str(ROOT / "runs" / f"pair_slice_repo-s*-{tag}.json")))
    if not fs:
        sys.exit(f"no artifacts matching -{tag}")
    return [(Path(f).name, json.load(open(f))) for f in fs]


def exact_per_pair(d, sl, arm):
    """Per-pair outcome, 1.0 win / 0.5 tie / 0.0 loss.

    The -exact runs write the outcome directly, at full precision. The older
    -tieaware runs do not: their scores are rounded to six decimals, which on
    the near-identical slice turns a difference of ~1e-7 into an apparent tie.
    Reconstructing outcomes from those scores inflates the tie count (3-5 real
    ties become 25-47 on structure-only/nograph) and moves the rate by up to
    0.03, so a bootstrap built from them is approximate and is labelled as such.
    """
    r = d[sl]
    if f"{arm}_outcome" in r:
        return np.asarray(r[f"{arm}_outcome"], float), True
    v = np.asarray(r[f"{arm}_score_vuln"], float)
    q = np.asarray(r[f"{arm}_score_patched"], float)
    x = v - q
    return np.where(x == 0, 0.5, (x > 0).astype(float)), False


def outcome(sv, sp, tol=0.0):
    """1.0 win, 0.5 tie, 0.0 loss -- per pair."""
    d = np.asarray(sv) - np.asarray(sp)
    return np.where(np.abs(d) <= tol, 0.5, (d > 0).astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exact",
                    help="artifact suffix to aggregate; 'exact' (full precision, "
                         "the default) or 'tieaware' (the superseded rounded runs)")
    ap.add_argument("--pool", default="",
                    help="comma-separated tags to pool as independent runs, e.g. "
                         "'exact,tieaware'. Repeating a seed does not reproduce "
                         "its result (set_seed does not make the CUDA scatter "
                         "reductions deterministic), so runs at the same seed in "
                         "different sets are independent replications and pooling "
                         "them is legitimate. Point estimates and run-level tests "
                         "use every pooled run; the pair-level bootstrap uses only "
                         "the runs that carry exact per-pair outcomes.")
    a = ap.parse_args()
    if a.pool:
        tags = [t.strip() for t in a.pool.split(",") if t.strip()]
        runs = [r for t in tags for r in load(t)]
        print(f"pooling {len(runs)} runs over tags {tags}")
    else:
        runs = load(a.tag)
    print(f"{len(runs)} runs: " + ", ".join(n for n, _ in runs))

    disp = {d["_disposition"][s] for _, d in runs for s in SLICES}
    print(f"\ndisposition identical across runs: "
          f"{ {s: runs[0][1]['_disposition'][s] for s in SLICES} }")
    for _, d in runs:
        assert d["_disposition"] == runs[0][1]["_disposition"], \
            "pair disposition differs between runs"
        assert d["_eligible_pairs"] == runs[0][1]["_eligible_pairs"]

    rows = []
    for sl in SLICES:
        n = runs[0][1][sl]["n_pairs"]
        # alignment check
        ids0 = (runs[0][1][sl]["pair_vuln_sid"], runs[0][1][sl]["pair_patched_sid"])
        for nm, d in runs:
            assert (d[sl]["pair_vuln_sid"], d[sl]["pair_patched_sid"]) == ids0, \
                f"pair order differs in {nm} slice {sl}"
        for arm in ARMS:
            W = np.array([d[sl][f"{arm}_wins"] for _, d in runs], float)
            L = np.array([d[sl][f"{arm}_losses"] for _, d in runs], float)
            T = np.array([d[sl][f"{arm}_ties"] for _, d in runs], float)
            assert np.all(W + L + T == n), f"W+L+T != n for {sl}/{arm}"
            strict = W / n
            tieaw = (W + 0.5 * T) / n
            # per-pair mean outcome across seeds, then bootstrap over pairs
            got = [exact_per_pair(d, sl, arm) for _, d in runs]
            # Restrict the pair-level bootstrap to runs with exact per-pair
            # outcomes; rounded runs would inject artificial ties into it.
            ex = [g[0] for g in got if g[1]]
            is_exact = len(ex) > 0
            per_seed = np.array(ex) if is_exact else np.array([g[0] for g in got])
            n_pairboot = len(per_seed)
            # Cross-check: the per-pair outcomes must reproduce the counts the
            # run wrote at full precision. Exact artifacts must match exactly;
            # rounded ones are expected to disagree, and by how much is reported.
            recon = per_seed.sum(axis=1)
            booked_all = np.array([(d[sl][f"{arm}_wins"] + 0.5 * d[sl][f"{arm}_ties"])
                                   for _, d in runs], float)
            booked = booked_all[[i for i, g in enumerate(got) if g[1]]] \
                if is_exact else booked_all
            if is_exact:
                assert np.allclose(recon, booked), \
                    f"per-pair outcomes disagree with stored counts for {sl}/{arm}"
            per_pair = per_seed.mean(axis=0)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n, size=(10000, n))
            bs = per_pair[idx].mean(axis=1)
            pci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
            # seed interval on the tie-aware accuracy
            se = tieaw.std(ddof=1) / np.sqrt(len(tieaw))
            h = stats.t.ppf(0.975, len(tieaw) - 1) * se
            rows.append(dict(slice=sl, arm=arm, n=n,
                             W=W.mean(), L=L.mean(), T=T.mean(),
                             Tfrac=T.mean() / n,
                             strict=strict.mean(), strict_sd=strict.std(ddof=1),
                             tieaw=tieaw.mean(), tieaw_sd=tieaw.std(ddof=1),
                             seed_lo=tieaw.mean() - h, seed_hi=tieaw.mean() + h,
                             pair_lo=pci[0], pair_hi=pci[1],
                             pair_exact=is_exact, n_pairboot=n_pairboot,
                             recon_gap=float(np.mean(recon - booked) / n),
                             per_pair=per_pair, tieaw_seeds=tieaw))

    print(f"\n{'slice':16s}{'arm':>9s}{'n':>6s}{'W':>7s}{'L':>7s}{'T':>7s}"
          f"{'tie%':>7s}{'strict':>9s}{'tieaware':>10s}"
          f"{'run CI':>18s}{'pair CI':>18s}")
    for r in rows:
        note = "" if r["pair_exact"] else f"  approx, rate off by {r['recon_gap']:+.3f}"
        print(f"{r['slice']:16s}{r['arm']:>9s}{r['n']:6d}"
              f"{r['W']:7.1f}{r['L']:7.1f}{r['T']:7.1f}{100*r['Tfrac']:6.1f}%"
              f"{r['strict']:9.4f}{r['tieaw']:10.4f}"
              f"   [{r['seed_lo']:.3f},{r['seed_hi']:.3f}]"
              f"   [{r['pair_lo']:.3f},{r['pair_hi']:.3f}]{note}")

    print(f"\n  pair bootstrap uses {rows[0]['n_pairboot']} run(s) with exact "
          f"per-pair outcomes; point estimates use all {len(runs)}")
    print("\n=== is the tie-aware accuracy distinguishable from chance? ===")
    for r in rows:
        t, p = stats.ttest_1samp(r["tieaw_seeds"], 0.5)
        v = "below" if r["tieaw"] < 0.5 else "above"
        print(f"  {r['slice']:16s}{r['arm']:>9s}  {r['tieaw']:.4f}  "
              f"t={t:+.2f} p={p:.4f}  ({v} chance)"
              + ("  SIGNIFICANT" if p < 0.05 else ""))

    print("\n=== graph vs no-graph, tie-aware, paired over seeds ===")
    for sl in SLICES:
        g = [r for r in rows if r["slice"] == sl and r["arm"] == "graph"][0]
        ng = [r for r in rows if r["slice"] == sl and r["arm"] == "nograph"][0]
        d = g["tieaw_seeds"] - ng["tieaw_seeds"]
        t, p = stats.ttest_rel(g["tieaw_seeds"], ng["tieaw_seeds"])
        print(f"  {sl:16s} delta {d.mean():+.4f}  p={p:.4f}")

    print("\n=== LaTeX ===")
    for sl in SLICES:
        g = [r for r in rows if r["slice"] == sl and r["arm"] == "graph"][0]
        ng = [r for r in rows if r["slice"] == sl and r["arm"] == "nograph"][0]
        nm = {"structure-only": "structure only", "text-differs": "text differs",
              "near-identical": "near-identical"}[sl]
        print(f"{nm:16s} & {g['n']:,} & {g['W']:.0f} & {g['L']:.0f} & {g['T']:.0f} "
              f"& ${g['strict']:.3f}$ & ${g['tieaw']:.3f}$ "
              f"& [{g['seed_lo']:.3f}, {g['seed_hi']:.3f}] "
              f"& [{g['pair_lo']:.3f}, {g['pair_hi']:.3f}] "
              f"& ${ng['tieaw']:.3f}$ \\\\")


if __name__ == "__main__":
    main()
