#!/usr/bin/env python3
"""
pair_slice_test.py — does the graph help where structure is the only difference?

The aggregate edge-deletion gap is small, which reads as "edges do not matter".
A reviewer's objection, correctly, is that swapping two lines can change CFG/DFG
while leaving the node set almost intact, so such a case turns on structure.
Measuring in aggregate hides those cases, because they are a small minority of
the corpus.

This evaluates on the pairs directly. For each vulnerable/patched pair held in
the test partition the model scores both members, and the pair is recorded as a
win, a loss or a tie. That is a within-pair comparison, so project identity,
length and style cancel. Pairs are compared by Jaccard similarity of node text
and of edges at a threshold of 0.95, and every eligible pair falls into exactly
one of three slices:

  structure-only   node text >= 0.95, edges <  0.95    165 pairs,  8.0%
  text-differs     node text <  0.95                 1,341 pairs, 64.9%
  near-identical   node text >= 0.95, edges >= 0.95    561 pairs, 27.1%

The near-identical slice is not a leftover. There the patch alters neither the
node text nor the edges appreciably, so the representation does not encode the
difference the model is asked to find, and the two members can featurise
identically. Ties concentrate there and have to be reported as ties rather than
charged as failures.

What each slice can and cannot settle. The structure-only slice is the one where
edges are the only thing the patch moves, so comparing the graph arm against the
edge-free arm there measures whether edges carry signal this pipeline can use. A
null result on that slice bounds what this pipeline extracts from edges on these
pairs; it does not by itself identify where the limit sits. Architecture,
encoding, the frozen node features, label noise in the mined pairs, and the
absence of the signal from the corpus are all consistent with the same null, and
this script separates none of them. Attributing the result to any one of them
needs the controls in the surrounding sections, not this measurement alone.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split, encode
from torch_geometric.loader import DataLoader


def jac(a, b):
    inter = sum((a & b).values()); union = sum((a | b).values())
    return inter / union if union else 1.0


def summarise(dot):
    nodes, edges = T.parse_dot(Path(dot))
    if not nodes or not edges:
        return None
    code = collections.Counter((nodes[n]["code"] or "").strip() for n in nodes)
    ekey = collections.Counter(
        (et, (nodes[s]["code"] or "").strip(), (nodes[d]["code"] or "").strip())
        for (s, et, d) in edges if s in nodes and d in nodes)
    return code, ekey


@torch.no_grad()
def scores(model, graphs, dev, bs=128):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=bs, shuffle=False):
        b = b.to(dev)
        out.append(torch.softmax(model(b), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


TOL = 1e-6

def disposition(sv, sp, tol=0.0):
    w = l = t = 0
    for a, b in zip(sv, sp):
        if abs(a - b) <= tol:
            t += 1
        elif a > b:
            w += 1
        else:
            l += 1
    return w, l, t

def boot_ci(sv, sp, tol, n_boot=10000, seed=0):
    """Percentile interval over PAIRS for the tie-aware accuracy. Pairs are
    the resampling unit; seed variance is reported separately across runs."""
    rng = np.random.default_rng(seed)
    v = np.asarray(sv); q = np.asarray(sp)
    d = v - q
    per_pair = np.where(np.abs(d) <= tol, 0.5, (d > 0).astype(float))
    n = len(per_pair)
    if n == 0:
        return None, None
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = per_pair[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--layers", type=int, default=3)
    # The leading dash is added here rather than passed on the command line:
    # argparse reads a value beginning with "-" as the next option and exits 2.
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = dict(T.CONFIG); cfg.update(batch_size=256, num_workers=6,
                                     pin_memory=True, epochs=a.epochs,
                                     early_stop_patience=15)
    if a.seed is not None:
        cfg["seed"] = a.seed
    dev = T.get_device(); T.set_seed(cfg["seed"])
    corpus, fams, splits = load_split(a.mode)
    G, _ = encode(corpus, splits, fams, cfg, dev, Path("/home/moses/graph_cache_cwe"))
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    for gs in G.values():
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
    if a.reverse_edges:
        for gs in G.values():
            for g in gs:
                if g.edge_index.numel():
                    g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], 1)
                    g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
        print("[reverse-edges] applied", flush=True)
    cfg["gat_layers"] = a.layers
    base_dim = G["train"][0].x.shape[1]

    # pairs whose two members both sit in the test partition
    test = G["test"]
    sid_to_row = {g.sample_id: i for i, g in enumerate(test)}
    by_pair = collections.defaultdict(list)
    for g in test:
        r = corpus[g.sample_id]
        if r.get("pair_id") is not None:
            by_pair[r["pair_id"]].append(g.sample_id)
    pairs = []
    for pid, sids in by_pair.items():
        if len(sids) != 2:
            continue
        labs = [int(corpus[s]["label"]) for s in sids]
        if set(labs) != {0, 1}:
            continue
        vul = sids[labs.index(1)]; pat = sids[labs.index(0)]
        pairs.append((vul, pat))
    print(f"[pairs] {len(pairs)} vulnerable/patched pairs inside the test partition",
          flush=True)

    # Every eligible pair lands in exactly one bucket. An earlier version used
    # an if/elif with no else, which silently dropped the pairs whose members
    # are near-identical in BOTH node text and edges -- 27% of the total. Those
    # are the cases where the patch leaves no trace in the graph at all, which
    # is a result rather than an exclusion.
    sliced = {"structure-only": [], "text-differs": [], "near-identical": []}
    unparseable = 0
    for vul, pat in pairs:
        A = summarise(corpus[vul]["dot_path"]); B = summarise(corpus[pat]["dot_path"])
        if not A or not B:
            unparseable += 1
            continue
        nj, ej = jac(A[0], B[0]), jac(A[1], B[1])
        if nj < 0.95:
            sliced["text-differs"].append((vul, pat))
        elif ej < 0.95:
            sliced["structure-only"].append((vul, pat))
        else:
            sliced["near-identical"].append((vul, pat))
    assert sum(len(v) for v in sliced.values()) + unparseable == len(pairs), \
        "pair disposition does not account for every eligible pair"
    for k, v in sliced.items():
        print(f"  {k:16s} {len(v)} pairs")
    print(f"  {'unparseable':16s} {unparseable} pairs")

    results = {}
    for arm in ("graph", "nograph"):
        Gm = {k: [g.clone() for g in v] for k, v in G.items()}
        if arm == "nograph":
            for part in Gm.values():
                for g in part:
                    g.edge_index = torch.zeros((2, 0), dtype=torch.long)
                    g.edge_type = torch.zeros((0,), dtype=torch.long)
        print(f"\n[train] {arm}", flush=True)
        m = T.train_expert(Gm["train"], base_dim, vocab, cfg, dev,
                           T.MAX_RELATIONS, tag=f"pairslice-{arm}", num_classes=2)
        s = scores(m, Gm["test"], dev)
        results[arm] = s
        del m; torch.cuda.empty_cache()

    # A strict win rate (score_vuln > score_patched) charges every tie as a
    # failure. Where a patch leaves no trace in the graph the two members can
    # featurise identically, in which case the model returns the same score and
    # the pair is genuinely undecidable rather than wrong. So the disposition of
    # every pair is reported as win / loss / tie, and two accuracies alongside
    # it: the strict rate, and a tie-aware rate that credits a tie as half a
    # win, which is the value a coin flip on ties would earn in expectation.
    #
    # EXACT_TIE is float equality on the model's score. TOL_TIE repeats the
    # count with a 1e-6 band so the paper can state the tie mass is a property
    # of the pairs rather than of the tolerance.
    print(f"\n{'slice':18s}{'n':>6s}{'arm':>9s}{'W':>6s}{'L':>6s}{'T':>5s}"
          f"{'strict':>9s}{'tieaware':>10s}{'95% CI':>18s}")
    out = {"_disposition": {k: len(v) for k, v in sliced.items()},
           "_unparseable": unparseable, "_eligible_pairs": len(pairs),
           "_metric": {"strict": "score_vuln > score_patched; ties count as losses",
                       "tie_aware": "(wins + 0.5*ties)/n_pairs",
                       "tie_definition": "exact float equality of the model score",
                       "tie_tolerance_variant": TOL,
                       "ci": "2.5/97.5 percentile bootstrap over pairs, 10,000 "
                             "resamples, resampling unit = pair"}}
    for name, ps in sliced.items():
        if not ps:
            continue
        # Pair identities, in the same order as every per-pair array below.
        # The cross-seed interval averages each pair's outcome over seeds, so
        # alignment between runs must be checkable rather than assumed.
        rec = {"n_pairs": len(ps),
               "pair_vuln_sid": [str(v) for v, _ in ps],
               "pair_patched_sid": [str(p_) for _, p_ in ps]}
        acc = {}
        for arm in ("graph", "nograph"):
            sc = results[arm]
            sv = [float(sc[sid_to_row[v]]) for v, _ in ps]
            sp = [float(sc[sid_to_row[p_]]) for _, p_ in ps]
            w, l, t = disposition(sv, sp)
            wt, lt, tt = disposition(sv, sp, TOL)
            strict = w / len(ps)
            tie_aware = (w + 0.5 * t) / len(ps)
            lo, hi = boot_ci(sv, sp, 0.0, seed=cfg["seed"] or 0)
            acc[arm] = strict
            rec[arm] = strict                      # back-compatible key
            rec[f"{arm}_wins"] = w
            rec[f"{arm}_losses"] = l
            rec[f"{arm}_ties"] = t
            rec[f"{arm}_ties_tol{TOL:g}"] = tt
            rec[f"{arm}_strict_acc"] = round(strict, 4)
            rec[f"{arm}_tie_aware_acc"] = round(tie_aware, 4)
            rec[f"{arm}_tie_aware_ci95"] = [round(lo, 4), round(hi, 4)]
            rec[f"{arm}_mean_score_vuln"] = round(float(np.mean(sv)), 4)
            rec[f"{arm}_mean_score_patched"] = round(float(np.mean(sp)), 4)
            # Raw per-pair scores, so wins, losses, ties, any tie tolerance and
            # any interval can be recomputed later without retraining. The
            # previous version stored only win booleans, which had already
            # folded ties into losses and could not be undone.
            # Full precision, not rounded: rounding to 6 decimals turns a pair
            # separated by ~1e-7 into an exact tie on recomputation, which
            # shifted a recomputed accuracy by 0.002 against the counts above.
            rec[f"{arm}_score_vuln"] = sv
            rec[f"{arm}_score_patched"] = sp
            rec[f"{arm}_outcome"] = [1.0 if a > b else (0.0 if b > a else 0.5)
                                     for a, b in zip(sv, sp)]
            rec[f"wins_{arm}"] = [bool(a > b) for a, b in zip(sv, sp)]
            print(f"{name:18s}{len(ps):6d}{arm:>9s}{w:6d}{l:6d}{t:5d}"
                  f"{strict:9.4f}{tie_aware:10.4f}   [{lo:.3f},{hi:.3f}]")
        rec["delta"] = round(acc["graph"] - acc["nograph"], 4)
        rec["delta_tie_aware"] = round(rec["graph_tie_aware_acc"]
                                       - rec["nograph_tie_aware_acc"], 4)
        out[name] = rec

    sfx = ("" if a.seed is None else f"-s{a.seed}") \
          + (f"-{a.tag}" if a.tag else "") \
          + ("-rev" if a.reverse_edges else "") \
          + ("" if a.layers == 3 else f"-L{a.layers}")
    import platform, subprocess
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git = None
    # Full test-partition scores at full precision, with the sample-id order, so
    # a change of slice definition or tie rule can be re-analysed without
    # retraining. The models themselves are not persisted by train_expert.
    out["_test_sample_ids"] = [str(g.sample_id) for g in test]
    out["_test_scores"] = {arm: [float(x) for x in results[arm]]
                           for arm in ("graph", "nograph")}
    out["config"] = {"mode": a.mode, "epochs": a.epochs, "seed": cfg["seed"],
                     "reverse_edges": a.reverse_edges, "layers": a.layers,
                     "batch_size": cfg["batch_size"],
                     "early_stop_patience": cfg["early_stop_patience"],
                     "arms": ["graph", "nograph"],
                     "nograph_arm": "same trained pipeline with edge_index and "
                                    "edge_type emptied, so only the graph is removed",
                     "checkpoint_selection": "train_expert early stopping, "
                                             "patience 15",
                     "slice_threshold": "Jaccard 0.95 on node text and on edges"}
    out["env"] = {"python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda,
                  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "git_commit": git}
    p = ROOT / "runs" / f"pair_slice_{a.mode}{sfx}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
