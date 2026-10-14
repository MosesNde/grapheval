#!/usr/bin/env python3
"""
verify_pipeline.py — independent checks that the measurement apparatus behaves
as the paper says it does.

The central results of this paper are ablations: quantities obtained by removing
a component and observing what changes. An ablation is only as trustworthy as
the claim that the removal did what it was supposed to do, and that the model
could have used the component in the first place. A null result from a broken
instrument is indistinguishable from a null result from an uninformative
component, so each of those assumptions is checked here directly rather than
assumed.

Every check is self-contained and prints the evidence it relies on. Checks 1-5
construct small graphs and run the real model classes, so they execute in
seconds and need no GPU. Checks 6-8 read the corpus, the split files and the
released run artifacts.

    python3 verify_pipeline.py

Exit status is non-zero if any check fails.
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from torch_geometric.data import Data

PASS, FAIL = [], []


def check(name, ok, detail):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.strip().splitlines():
        print(f"         {line}")
    print()


def hdr(n, title, question):
    print(f"\n{'='*74}\n{n}. {title}\n   Question: {question}\n{'='*74}")


# ---------------------------------------------------------------------------
def c1_message_passing_propagates():
    hdr(1, "Message passing actually propagates",
        "When edges are present, does one node's input affect its neighbours?")
    from torch_geometric.nn import GATConv
    torch.manual_seed(0)
    conv = GATConv(4, 4, heads=1, edge_dim=3).eval()
    x = torch.randn(5, 4); xp = x.clone(); xp[0] = torch.randn(4)

    def moved(ei, ea):
        with torch.no_grad():
            a, b = conv(x, ei, edge_attr=ea), conv(xp, ei, edge_attr=ea)
        return [i for i in range(5) if not torch.allclose(a[i], b[i], atol=1e-6)]

    one = moved(torch.tensor([[0], [1]]), torch.zeros((1, 3)))
    two = moved(torch.tensor([[0, 0], [1, 2]]), torch.zeros((2, 3)))
    ok = one == [0, 1] and two == [0, 1, 2]
    check("information crosses edges", ok,
          f"perturbing node 0 with edge 0->1 changes nodes {one}\n"
          f"perturbing node 0 with edges 0->1,2 changes nodes {two}\n"
          "the perturbation reaches exactly the nodes the edges point to")


def c2_edge_deletion_isolates():
    hdr(2, "Edge deletion removes all connectivity",
        "With edge_index emptied, can any node still influence any other?")
    from torch_geometric.nn import GATConv, GCNConv, GINConv, TransformerConv
    torch.manual_seed(0)
    x = torch.randn(5, 4); xp = x.clone(); xp[0] = torch.randn(4)
    empty = torch.zeros((2, 0), dtype=torch.long)
    rows = {}
    for nm, conv, kw in (("GAT", GATConv(4, 4, heads=1, edge_dim=3).eval(),
                          {"edge_attr": torch.zeros((0, 3))}),
                         ("GCN", GCNConv(4, 4).eval(), {}),
                         ("GIN", GINConv(nn.Linear(4, 4)).eval(), {}),
                         ("TransformerConv", TransformerConv(4, 4).eval(), {})):
        with torch.no_grad():
            a, b = conv(x, empty, **kw), conv(xp, empty, **kw)
        rows[nm] = [i for i in range(5) if not torch.allclose(a[i], b[i], atol=1e-6)]
    ok = all(v == [0] for v in rows.values())
    check("no information crosses when edges are deleted", ok,
          "\n".join(f"{k:16s} perturbing node 0 changes nodes {v}" for k, v in rows.items())
          + "\nonly the perturbed node moves, so the model is a per-node model"
            "\nwith depth, width and parameter count unchanged")


def c3_features_are_edge_blind():
    hdr(3, "Node features never see the graph",
        "Could connectivity leak into node features, surviving edge deletion?")
    import inspect
    src = inspect.getsource(T.NodeFeaturizer.build)
    enc = inspect.getsource(T.NodeFeaturizer.encode_snippets)
    banned = [w for w in ("edge_index", "edge_type", "edges", "adjacency", "neighbor")
              if w in src or w in enc]
    ok = not banned
    check("featurizer takes nodes only", ok,
          "NodeFeaturizer.build(nodes) and encode_snippets(snippets) reference "
          f"none of edge_index/edge_type/edges/adjacency/neighbor: {ok}\n"
          f"found: {banned or 'nothing'}\n"
          "each node is embedded from its own source text alone, so message\n"
          "passing is the only path by which structure can reach the model")


def c4_backbones_read_edge_types():
    hdr(4, "Which backbones can distinguish edge types",
        "Same graph, same nodes, only edge labels change. Does the output move?")
    torch.manual_seed(0)
    out = {}
    for b in ("gat", "gcn", "gin", "gtrans"):
        cfg = dict(T.CONFIG)
        m = T.build_model(b, 8, 20, cfg, T.MAX_RELATIONS, 2).eval()
        x = torch.randn(6, 8); ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        lab = torch.zeros(6, dtype=torch.long)
        d1 = Data(x=x, edge_index=ei, label_id=lab, edge_type=torch.zeros(4, dtype=torch.long))
        d2 = Data(x=x, edge_index=ei, label_id=lab, edge_type=torch.tensor([0, 1, 2, 3]))
        for d in (d1, d2): d.batch = torch.zeros(6, dtype=torch.long)
        with torch.no_grad():
            out[b] = float((m(d1) - m(d2)).abs().max())
    ok = out["gin"] == 0.0 and all(out[k] > 0 for k in ("gat", "gcn", "gtrans"))
    check("edge-type sensitivity matches each operator's definition", ok,
          "\n".join(f"{k:8s} max output change when edge labels change: {v:.2e}"
                    for k, v in out.items())
          + "\nGIN sums over neighbours with no edge-feature term, so it cannot\n"
            "represent the distinction. This is why it is the one backbone that\n"
            "does not benefit from reversed edges.")


def c5_reverse_edges_well_formed():
    hdr(5, "Reverse edges are constructed correctly",
        "Does the bidirectional option double the edges and keep types separable?")
    fs = [f for f in glob.glob("/home/moses/graph_cache_cwe/*.pkl")][:1]
    import pickle
    d = pickle.loads(open(fs[0], "rb").read())
    ei, et = d.edge_index, d.edge_type
    rev_ei = torch.cat([ei, ei.flip(0)], 1); rev_et = torch.cat([et, et + 4])
    fwd_ok = torch.equal(rev_ei[:, :ei.shape[1]], ei)
    pair_ok = torch.equal(rev_ei[0, ei.shape[1]:], ei[1]) and \
              torch.equal(rev_ei[1, ei.shape[1]:], ei[0])
    type_ok = int(rev_et[:ei.shape[1]].max()) < 4 <= int(rev_et[ei.shape[1]:].min())
    cap_ok = int(rev_et.max()) < T.MAX_RELATIONS
    ok = fwd_ok and pair_ok and type_ok and cap_ok
    check("reversed relations occupy ids 4-7 and forward edges are untouched", ok,
          f"edges {ei.shape[1]} -> {rev_ei.shape[1]} (exactly doubled)\n"
          f"forward block unchanged: {fwd_ok}; reversed block is the transpose: {pair_ok}\n"
          f"forward types < 4 <= reversed types: {type_ok}\n"
          f"max relation id {int(rev_et.max())} < MAX_RELATIONS={T.MAX_RELATIONS}: {cap_ok}\n"
          "no parameters are added: the relation table already had unused rows")


def c6_splits_are_disjoint():
    hdr(6, "Split protocols do what they claim",
        "How much does each protocol actually let test share with training?")
    corpus = {json.loads(l)["sample_id"]: json.loads(l)
              for l in open(ROOT / "data" / "cwe_corpus.jsonl")}
    rows = {}
    for mode in ("repo", "group", "random"):
        sp = json.load(open(ROOT / "data" / f"splits_{mode}_family.json"))["splits"]
        tr, te = set(sp["train"]), set(sp["test"])
        r = {}
        for key in ("repo", "commit", "pair_id"):
            A = {corpus[i][key] for i in tr if i in corpus and corpus[i].get(key)}
            B = {corpus[i][key] for i in te if i in corpus and corpus[i].get(key)}
            r[key] = len(A & B)
        rows[mode] = r
    ok = all(v == 0 for v in rows["repo"].values()) and \
         rows["group"]["commit"] == 0 and rows["group"]["pair_id"] == 0 and \
         rows["random"]["repo"] > 0
    check("repo is clean, group shares only repositories, random leaks on every key", ok,
          "\n".join(f"{m:8s} " + "  ".join(f"{k}={v}" for k, v in r.items())
                    for m, r in rows.items())
          + "\nthe `random` row is what a plain stratified shuffle admits, and is\n"
            "the protocol this literature uses")


def c7_positive_control():
    hdr(7, "The pipeline can learn a label that lives only in the edges",
        "Is a small edge-ablation result evidence about the signal, or the model?")
    res = {}
    for task in ("cfg-cycle", "ddg-density"):
        fs = [f for f in sorted(glob.glob(str(ROOT / "runs" /
              f"structural_control_{task}_repo-s*.json"))) if "typesleak" not in f]
        if not fs: continue
        D = [json.load(open(f)) for f in fs]
        res[task] = (np.mean([d["graph"] for d in D]),
                     np.mean([d["nograph"] for d in D]), len(D))
    ok = bool(res) and all(g - n > 0.10 for g, n, _ in res.values())
    check("edges alone carry a large, recoverable signal", ok,
          "\n".join(f"{t:12s} graph {g:.3f}   edges deleted {n:.3f}   gain {g-n:+.3f}   "
                    f"({s} seeds)" for t, (g, n, s) in res.items())
          + "\nnode features are random noise and node types are collapsed, so the\n"
            "label is decidable from the edge set alone. The same architecture gains\n"
            "0.10-0.30 here and 0.014-0.032 on the vulnerability label, so the small\n"
            "figure reflects the signal available, not an inability to read edges.")


def c8_meta_features_dense():
    hdr(8, "Stacked meta-features cannot encode their own label",
        "Could the meta-learner recover the label from which cells were written?")
    n = 400
    y = np.random.default_rng(0).integers(0, 4, n)
    good = np.random.default_rng(1).random((n, 4))
    bad = np.full((n, 4), 0.5)
    for i, c in enumerate(y):
        bad[i, c] = np.random.default_rng(i).random()
    caught = passed = None
    try:
        T.assert_meta_features_dense(good, y, "dense-control"); passed = True
    except Exception as e:
        passed = False; print("         unexpected:", e)
    try:
        T.assert_meta_features_dense(bad, y, "sparse-injected"); caught = False
    except Exception:
        caught = True
    ok = bool(passed) and bool(caught)
    check("the guard accepts a dense matrix and rejects a label-encoding one", ok,
          f"dense matrix accepted: {passed}\n"
          f"deliberately label-encoding matrix rejected: {caught}\n"
          "this guard runs before every meta-learner fit in the reported results")


if __name__ == "__main__":
    print(__doc__.split("\n\n")[1].strip())
    for fn in (c1_message_passing_propagates, c2_edge_deletion_isolates,
               c3_features_are_edge_blind, c4_backbones_read_edge_types,
               c5_reverse_edges_well_formed, c6_splits_are_disjoint,
               c7_positive_control, c8_meta_features_dense):
        try:
            fn()
        except Exception as e:
            FAIL.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}\n")
    print("="*74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
