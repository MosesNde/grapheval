#!/usr/bin/env python3
"""
make_pipeline_extract.py — assemble the code path that produces the paper's
numbers into one readable document.

training.py is 2,752 lines and carries paths that the reported results do not
use: a second classification stage, several backbones that appear only in the
architecture table, and code retained from an earlier version of this project.
Reading it end to end is a poor way to audit what was actually run.

This extracts, verbatim and in execution order, the functions a .dot file passes
through on its way to a reported number, annotating each with what it does and
what an auditor should check. Every excerpt carries its source file and line
range so it can be checked against the original.

    python3 make_pipeline_extract.py > PIPELINE_EXTRACT.txt
"""
from __future__ import annotations
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("training.py", 316, 431, "1. Parsing: .dot to nodes and edges",
     """Joern exports one .dot per function. This reads it into a node table and
an edge list of (source, type, destination) triples.

What to check:
  * the edge type is the prefix before the colon, so `DDG: stream_info`
    becomes relation `DDG`. The variable annotation is discarded here.
  * edges keep the direction Joern emitted. Nothing adds the reverse
    direction at this point; that is an explicit option applied later.
  * four relation types occur in practice: AST, CFG, DDG, CDG."""),

    ("training.py", 435, 501, "2. Node features: text to vectors",
     """Each node's source text is embedded with GraphCodeBERT, independently of
every other node.

What to check:
  * build() receives `nodes` only. There is no edge argument, no adjacency,
    no neighbour information. This is what makes the edge-deletion ablation
    meaningful: connectivity cannot leak into the features and survive it.
  * node_ids are sorted, fixing the row order that the edge list indexes into.
  * empty code falls back to the node's type label so that all empty nodes do
    not collapse to one identical vector."""),

    ("training.py", 553, 606, "3. Assembly: building the graph object",
     """Nodes and edges become a PyTorch Geometric Data object.

What to check:
  * edges are kept only when both endpoints survived featurisation.
  * edge_index is [source; destination]; PyG propagates source to destination,
    so a node aggregates from its predecessors.
  * the result is cached keyed on the .dot file's path, size and mtime, so a
    changed input invalidates its cache entry."""),

    ("training.py", 957, 1043, "4. The model",
     """Three GAT layers over the node features, with a learned embedding for
node type and another for edge relation.

What to check:
  * input is the 768-dim GraphCodeBERT vector concatenated with a 64-dim node
    type embedding, projected to width 512.
  * `e_attr` is the 32-dim relation embedding, passed to every layer as
    edge_attr, so attention between two nodes depends on which relation joins
    them. This is the mechanism the `untyped` ablation removes.
  * residual connections and LayerNorm after each layer.
  * `edge_ann` and `num_layers` default to the reported configuration; they
    exist for the annotation and depth experiments and are inert otherwise."""),

    ("train_cwe.py", 78, 116, "5. Loading a split",
     """Applies a split file to the corpus and encodes each partition.

What to check:
  * the split is read from disk, not recomputed, so every arm of an ablation
    sees exactly the same partition.
  * the label is binary here: 1 if the function is vulnerable."""),

    ("train_cwe.py", 185, 239, "6. Training and evaluation",
     """Five-fold cross-validation on the training partition; each fold's model
scores the held-out fold and the test partition; the test scores are averaged.

What to check:
  * folds are drawn from the training partition only. The test partition is
    never trained on and never used for early stopping.
  * `ensemble_mean_only` is the number reported in the paper: the mean of the
    five folds' test predictions.
  * a meta-learner is fitted over out-of-fold predictions, and
    assert_meta_features_dense() runs first to reject a feature matrix that
    could encode its own label."""),

    ("training.py", 1361, 1462, "7. The training loop",
     """One fold's model.

What to check:
  * the validation split for early stopping is carved from the training
    subset, never from the test partition.
  * the model restored at the end is the best-validation checkpoint, not the
    last epoch.
  * loss, optimiser, scheduler and patience are read from CONFIG and are
    identical across every arm of every ablation."""),
]

ABLATIONS = """
=============================================================================
HOW EACH ABLATION IS APPLIED
=============================================================================
All ablations modify the data after loading and before the model is built, so
depth, width, parameter count, optimiser, folds and metrics are untouched. The
relevant code is train_cwe.py, in main():

  --edges none        g.edge_index = zeros((2,0)); g.edge_type = zeros((0,))
                      every edge removed; nodes, features and model unchanged

  --edges untyped     g.edge_type = zeros_like(g.edge_type)
                      connectivity kept, all four relations collapsed to one

  --features type-only   node text zeroed, node type embedding kept
  --features topology    node text zeroed and node types collapsed
  --features random      node features replaced with noise

  --reverse-edges     edge_index = cat([ei, ei.flip(0)])
                      edge_type  = cat([et, et + 4])
                      each edge also traversable backwards, as a distinct
                      relation. MAX_RELATIONS is already 8 with 4 in use, so
                      no parameters are added.

  --seed N            changes initialisation, fold assignment and sampling.
                      It does NOT change the split, which is fixed on disk.
"""

def main():
    print(__doc__.split("\n\n")[1].strip())
    print()
    for fname, a, b, title, note in STAGES:
        print("=" * 77)
        print(f"{title}")
        print(f"source: scripts/{fname} lines {a}-{b}")
        print("=" * 77)
        for line in note.strip().splitlines():
            print(f"  {line}")
        print()
        src = (HERE / fname).read_text().split("\n")[a - 1 : b]
        w = len(str(b))
        for i, line in enumerate(src, start=a):
            print(f"  {str(i).rjust(w)} | {line}")
        print()
    print(ABLATIONS)

if __name__ == "__main__":
    main()
