#!/usr/bin/env python3
"""
edge_ann.py — features for the DDG variable annotations the pipeline discards.

Joern labels data-dependence edges with the value that flows along them
(`DDG: stream_info`, `DDG: &s->gb`, `DDG: <RET>`). `parse_dot` keeps only the
prefix before the colon, so every DDG edge reaches the model as an anonymous
relation. Across 1,500 CPGs there are 33,920 distinct annotations, with 8,295
needed to cover 80% of non-empty DDG edges: far too heavy a tail for a lookup
table, and hopeless out-of-vocabulary under a repository-disjoint split.

Two encodings, deliberately different in kind:

  hashed      the annotation string hashed into a fixed bucket space. Lexical,
              so it can carry project-specific naming. Variable names are the
              canonical spurious feature in this literature, which makes this
              the arm that should help under `random` and not under `repo`.

  structural  what the annotation *is*, never what it is called: return value,
              constant, pointer dereference, plain identifier, and whether the
              token actually occurs in the endpoint nodes' code. Immune to
              naming, so it is the honest test of whether data-flow detail was
              being thrown away.
"""
from __future__ import annotations
import hashlib, re

HASH_BUCKETS = 4096
STRUCT_DIM = 8

_ENT = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'"}
_IDENT = re.compile(r"^[A-Za-z_]\w*$")
_NUM = re.compile(r"^[-+]?(0[xX][0-9a-fA-F]+|\d+)$")


def decode(s: str) -> str:
    for k, v in _ENT.items():
        s = s.replace(k, v)
    return s.strip()


def hashed(ann: str) -> int:
    """Bucket id. Empty annotation gets its own reserved bucket 0."""
    a = decode(ann)
    if not a:
        return 0
    h = hashlib.blake2b(a.encode(), digest_size=8).digest()
    return 1 + (int.from_bytes(h, "big") % (HASH_BUCKETS - 1))


def structural(ann: str, src_code: str, dst_code: str) -> list[float]:
    """Naming-independent description of the flowing value."""
    a = decode(ann)
    core = a.split()[-1].lstrip("&*") if a else ""
    return [
        1.0 if not a else 0.0,                       # bare edge, no annotation
        1.0 if "<RET>" in a else 0.0,                # return value
        1.0 if "<unknown>" in a else 0.0,            # unresolved by Joern
        1.0 if ("->" in a or "&" in a or "*" in a or "[" in a) else 0.0,
        1.0 if _NUM.match(core) else 0.0,            # literal constant
        1.0 if _IDENT.match(core) else 0.0,          # plain identifier
        1.0 if core and core in src_code else 0.0,   # token present at source
        1.0 if core and core in dst_code else 0.0,   # token present at sink
    ]
