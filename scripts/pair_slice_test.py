#!/usr/bin/env python3
"""
pair_slice_test.py — does the graph help where structure is the only difference?

The aggregate edge-deletion gap is small, which reads as "edges do not matter".
A reviewer's objection, correctly, is that swapping two lines can change CFG/DFG
while leaving the node set almost intact, and that such a case must depend on
structure. Measuring that in aggregate hides it: only 9.3% of vulnerable/patched
pairs in this corpus have near-identical node text with different edges, while
61.5% change both, where token content alone separates the two versions.

This evaluates on the pairs directly. For each vulnerable/patched pair held in
the test partition, a model is correct if it scores the vulnerable member above
the patched one. That is a within-pair comparison, so project identity, length
and style cancel. Two slices are reported:

  structure-only   node text >= 0.95 Jaccard, edges < 0.95   (the objection)
  text-differs     node text < 0.95                          (the rest)

If edges carry vulnerability signal that this pipeline can use, the graph model
should beat the edge-free model on the structure-only slice. If it does not, the
limitation is the architecture or the encoding, not the corpus.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--layers", type=int, default=3)
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

    print(f"\n{'slice':18s}{'n':>6s}{'graph':>10s}{'no graph':>11s}{'delta':>9s}")
    out = {"_disposition": {k: len(v) for k, v in sliced.items()},
           "_unparseable": unparseable, "_eligible_pairs": len(pairs)}
    for name, ps in sliced.items():
        if not ps:
            continue
        acc = {}
        for arm in ("graph", "nograph"):
            s = results[arm]
            wins = [s[sid_to_row[v]] > s[sid_to_row[p]] for v, p in ps]
            acc[arm] = float(np.mean(wins))
        d = acc["graph"] - acc["nograph"]
        gv = [float(results["graph"][sid_to_row[v]]) for v, _ in ps]
        gp = [float(results["graph"][sid_to_row[p_]]) for _, p_ in ps]
        nv = [float(results["nograph"][sid_to_row[v]]) for v, _ in ps]
        np_ = [float(results["nograph"][sid_to_row[p_]]) for _, p_ in ps]
        out[name] = {"n_pairs": len(ps), **acc, "delta": round(d, 4),
                     "graph_mean_score_vuln": round(float(np.mean(gv)), 4),
                     "graph_mean_score_patched": round(float(np.mean(gp)), 4),
                     "nograph_mean_score_vuln": round(float(np.mean(nv)), 4),
                     "nograph_mean_score_patched": round(float(np.mean(np_)), 4),
                     # per-pair win indicators, so the interval over pairs can be
                     # bootstrapped afterwards without retraining
                     "wins_graph": [bool(results["graph"][sid_to_row[v]] >
                                         results["graph"][sid_to_row[p_]])
                                    for v, p_ in ps],
                     "wins_nograph": [bool(results["nograph"][sid_to_row[v]] >
                                           results["nograph"][sid_to_row[p_]])
                                      for v, p_ in ps]}
        print(f"{name:18s}{len(ps):6d}{acc['graph']:10.4f}{acc['nograph']:11.4f}{d:+9.4f}")

    sfx = ("" if a.seed is None else f"-s{a.seed}") \
          + ("-rev" if a.reverse_edges else "") \
          + ("" if a.layers == 3 else f"-L{a.layers}")
    p = ROOT / "runs" / f"pair_slice_{a.mode}{sfx}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
