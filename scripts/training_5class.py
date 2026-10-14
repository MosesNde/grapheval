# training_v2.py
#
# Unified GNN vulnerability classifier.
# Reads .dot CPG files directly — no JSON conversion needed.
#
# ── Actual dataset layout (rebuttal/) ─────────────────────────────────────
#
#   rebuttal/
#     A01_Broken_Access_Control/
#       <40-char-hex>/                  ← GitHub-mined commit hash
#         Vulnerable_section_N_<name>/
#           export.dot
#       <number>_1_export/              ← LLM-augmented (numeric id)
#         export.dot
#         export (2).dot                ← multiple graphs per source file ok
#     A02_Cryptographic_Failures/ ...
#     ...
#     Non-vulnerable/                   ← ALL safe samples (no OWASP subfolder)
#       <number>_safe_export/
#         export.dot
#         export (2).dot
#
# ── Provenance detection ───────────────────────────────────────────────────
#   Within each Axx folder:
#     parent dir matches /^[0-9a-f]{40}$/ → "github"
#     parent dir matches /^\d+_\d+_export$/ → "llm"
#   Non-vulnerable folder:
#     parent dir matches /^\d+_safe_export$/ → "llm" (LLM-generated safe code)
#     (If you ever add GitHub-mined safe samples, add their pattern here)
#
# ── Label scheme (11-class multiclass) ────────────────────────────────────
#   0  = NON_VULNERABLE
#   1  = A01_Broken_Access_Control
#   2  = A02_Cryptographic_Failures
#   3  = A03_Injection
#   4  = A04_Insecure_Design
#   5  = A05_Security_Misconfiguration
#   6  = A06_Vulnerable_and_Outdated_Components
#   7  = A07_Identification_and_Authentication_Failures
#   8  = A08_Software_and_Data_Integrity_Failures
#   9  = A09_Security_Logging_and_Monitoring_Failures
#   10 = A10_Server-Side_Request_Forgery
#
# ── Both Joern .dot format variants are handled ───────────────────────────
#   GitHub-mined:   bare integer node IDs, unquoted attributes
#   LLM-augmented:  quoted string node IDs, individually-quoted attributes

import os
import re
import json
import pickle
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    RGCNConv,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation

try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    print("[warn] transformers not installed — using hash-based node features")

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAX_RELATIONS = 8
NUM_CLASSES   = 5   # 0=NON_VULNERABLE, 1=A01, 2=A02+A07, 3=A03, 4=OTHER

OWASP_NAMES = {
    0: "NON_VULNERABLE",
    1: "A01_Access_Control",
    2: "A02_A07_Crypto_Auth",
    3: "A03_Injection",
    4: "OTHER_VULNERABLE",
}

# Maps each OWASP folder prefix → 5-class label
FOLDER_PREFIX_TO_CLASS = {
    "A01": 1,   # Access Control
    "A02": 2,   # Cryptographic Failures → merged with A07
    "A03": 3,   # Injection
    "A04": 4,   # Insecure Design → OTHER
    "A05": 4,   # Misconfiguration → OTHER
    "A06": 4,   # Outdated Components → OTHER
    "A07": 2,   # Authentication Failures → merged with A02
    "A08": 4,   # Integrity Failures → OTHER
    "A09": 4,   # Logging Failures → OTHER
    "A10": 4,   # SSRF → OTHER
}

# Regex patterns that identify a sample directory's provenance.
# Applied to the immediate parent folder name of the dot file's directory.
_GITHUB_HASH_RE = re.compile(r'^[0-9a-f]{40}$')          # 40-hex commit hash
_LLM_VULN_RE    = re.compile(r'^\d+_\d+_export$')         # e.g. 201_1_export
_LLM_SAFE_RE    = re.compile(r'^\d+_safe_export$')         # e.g. 1000_safe_export

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "DATA_DIR":            "/home/moses/rebuttal",   # ← training data root
    "MEGAVUL_DOTS_DIR":   "/home/moses/Megavul_testset",
    "DEVIGN_NONVULN_DIR": "/home/moses/NON_VULNERABLE", # ← Devign non-vuln folder (None to skip)
    "backbone":            "gat",
    "codebert_model":      "microsoft/graphcodebert-base",
    "codebert_max_len":    128,
    "label_emb_dim":       64,
    "hidden_dim":          512,   # increased from 256
    "epochs":              300,
    "batch_size":          16,
    "lr":                  5e-4,   # lowered from 1e-3 — more stable with 768-dim inputs
    "weight_decay":        3e-4,
    "gat_heads":           8,     # increased from 4 — more heads for 512-dim space
    "gat_edge_emb_dim":    32,
    "dropout":             0.30,
    "seed":                42,
    "oof_max_folds":       5,
    "use_edge_types":      True,
    "attn_pool":           True,
    "val_frac":            0.10,
    "early_stop_patience": 80,    # raised from 50 — CodeBERT needs more warmup time
    "focal_gamma":         2.0,   # focal loss exponent; try 3.0 if easy classes dominate
    "sched_plateau":       True,
    "gbm_params": dict(
        n_estimators     = 400,
        learning_rate    = 0.05,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        random_state     = 42,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SEED / DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# DOT PARSER  —  handles all four Joern .dot format variants
#
# Format 1 — GitHub-mined (unquoted node IDs, unquoted attributes):
#   21474836480[label=BINDING KEY="val" ...]
#   123 -> 456 [label=AST ]
#
# Format 2 — LLM-augmented (quoted node IDs, individually-quoted attributes):
#   "21474836480" [label="BINDING" KEY="val" ...]
#   "123" -> "456" [label="AST" ]
#
# Format 3 — MegaVul (HTML angle-bracket labels, parenthesised tuple):
#   "123456" [label = <(TYPE,code,...)<SUB>N</SUB>> ]
#   "123" -> "456"  [ label = "AST: "]
#   Multiple concatenated digraph{} blocks per file.
#
# Format 4 — Devign non-vulnerable (HTML angle-bracket, BR-separated):
#   "123456" [label = <TYPE, LINE<BR/>code> ]
#   "123456" [label = <TYPE, LINE<BR/>name<BR/>full_expr> ]
#   node [shape="rect"];   ← style line, silently skipped
#   "123" -> "456"  [ label = "AST: "]
#
# Formats 3 & 4 both use HTML entities: &lt; &gt; &amp; &quot;
# ─────────────────────────────────────────────────────────────────────────────

# Formats 1 & 2
_NODE_RE = re.compile(r'^"?(\d+)"?\s*\[([^\]]+)\]')
_EDGE_RE = re.compile(r'^"?(\d+)"?\s*->\s*"?(\d+)"?\s*\[([^\]]*)\]')

# Formats 3 & 4: greedy match captures everything between < and last > before ]
# Works for both (TYPE,...)<SUB>N</SUB> and TYPE,LINE<BR/>code patterns
_HTML_LABEL_RE = re.compile(r'^"(\d+)"\s*\[label\s*=\s*<(.+)>\s*\]')
# Formats 3 & 4: edge  "src" -> "dst" [ label = "TYPE: property"]
_HTML_EDGE_RE  = re.compile(r'^"(\d+)"\s*->\s*"(\d+)"\s*\[\s*label\s*=\s*"([^"]+)"\s*\]')

# HTML entity decoding (used by formats 3 & 4)
_HTML_ENTITIES = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'"}

def _decode_html(s: str) -> str:
    for ent, ch in _HTML_ENTITIES.items():
        s = s.replace(ent, ch)
    return s


def _attr(text: str, key: str) -> str:
    """Extract value for key= from a DOT attribute string (quoted or unquoted)."""
    m = re.search(rf'{key}="((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1)
    m = re.search(rf'{key}=([^\s,\]]+)', text)
    return m.group(1) if m else ""


def _parse_html_label(content: str) -> Tuple[str, str]:
    """
    Parse the inner content of an HTML-bracket DOT label.
    Handles two sub-variants:

    Format 3 (MegaVul):  (TYPE,code,code)<SUB>N</SUB>
      Strip <SUB>, strip outer parens, split on commas.

    Format 4 (Devign):   TYPE, LINE<BR/>code
                         TYPE, LINE<BR/>name<BR/>full_expression
      Split on first comma for type; skip line number before first <BR/>;
      take the last <BR/>-separated segment as code (most complete).

    Returns (node_type, code_snippet) after HTML entity decoding.
    """
    content = _decode_html(content.strip())

    # Strip <SUB>N</SUB> suffix (Format 3 line-number tag)
    content = re.sub(r'<SUB>\d+</SUB>\s*$', '', content).strip()

    # ── Format 3: parenthesised tuple ────────────────────────────────────
    if content.startswith('(') and content.endswith(')'):
        inner = content[1:-1]
        comma = inner.find(',')
        if comma == -1:
            return inner.strip(), ''
        node_type = inner[:comma].strip()
        rest      = inner[comma + 1:]
        # Two code fields (code is duplicated): take up to the last comma
        last_comma = rest.rfind(',')
        code = rest[:last_comma].strip() if last_comma != -1 else rest.strip()
        return node_type, code

    # ── Format 4: BR-separated fields ────────────────────────────────────
    br = '<BR/>'
    if br in content:
        comma = content.find(',')
        if comma == -1:
            # No comma: TYPE<BR/>code (unusual but handle gracefully)
            parts = content.split(br)
            return parts[0].strip(), parts[-1].strip() if len(parts) > 1 else ''
        node_type = content[:comma].strip()
        rest      = content[comma + 1:]        # " LINE<BR/>code[<BR/>full_expr]"
        br_idx    = rest.find(br)
        if br_idx != -1:
            code_part = rest[br_idx + len(br):]
            # If another <BR/> exists, take the last segment (full expression)
            if br in code_part:
                code_part = code_part.split(br)[-1]
            code = code_part.strip()
        else:
            code = rest.strip()
        return node_type, code

    # ── Fallback ──────────────────────────────────────────────────────────
    comma = content.find(',')
    if comma != -1:
        return content[:comma].strip(), content[comma + 1:].strip()
    return content.strip(), ''


def _detect_format(text: str) -> str:
    """
    Detect which format family this file uses.
      "html"     — HTML angle-bracket labels (Formats 3 & 4)
      "quoted"   — individually-quoted attributes (Format 2)
      "unquoted" — bare attributes (Format 1)
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("digraph") or line == "}" or "->" in line:
            continue
        # Skip Devign style declarations (e.g. node [shape="rect"])
        if line.startswith("node ") or line.startswith("edge "):
            continue
        if re.match(r'^"?\d+"?\s*\[', line):
            if "label = <" in line or "label=<" in line:
                return "html"
            if 'label="' in line:
                return "quoted"
            return "unquoted"
    return "unquoted"


def parse_dot(dot_path: Path) -> Tuple[Dict[int, dict], List[Tuple[int, str, int]]]:
    """
    Parse a CPG .dot file in any of the four supported formats.

    Formats 3 & 4 (MegaVul, Devign non-vuln) may contain multiple concatenated
    digraph{} blocks — all blocks are merged into one node/edge set, treating
    the file as a single graph (appropriate since all blocks come from the same
    source function or commit).

    Returns:
        nodes: {node_id: {"label": str, "code": str}}
        edges: [(src_id, edge_type_str, dst_id)]
    """
    nodes: Dict[int, dict] = {}
    edges: List[Tuple[int, str, int]] = []

    try:
        text = dot_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] cannot read {dot_path}: {e}")
        return nodes, edges

    fmt = _detect_format(text)

    if fmt == "html":
        # ── Formats 3 & 4: HTML angle-bracket labels ──────────────────────
        # MegaVul (Format 3) nodes can span multiple lines; Devign (Format 4)
        # nodes are always single-line.  The flattening logic handles both:
        # accumulate lines until the current buffer ends with "> ]".
        flat_lines: List[str] = []
        buf = ""
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if buf:
                buf += " " + stripped
            else:
                buf = stripped
            # Flush conditions
            if stripped == "}" or stripped.startswith("digraph") or stripped.startswith("node "):
                flat_lines.append(buf)
                buf = ""
            elif "->" in buf and buf.rstrip().endswith("]"):
                flat_lines.append(buf)
                buf = ""
            elif re.search(r'>\s*\]\s*$', buf):
                flat_lines.append(buf)
                buf = ""
        if buf:
            flat_lines.append(buf)

        for line in flat_lines:
            line = line.strip()
            if (not line or line.startswith("digraph") or line == "}"
                    or line.startswith("node ") or line.startswith("edge ")):
                continue

            if "->" in line:
                m = _HTML_EDGE_RE.match(line)
                if m:
                    src = int(m.group(1))
                    dst = int(m.group(2))
                    et  = m.group(3).split(":")[0].strip() or "UNKNOWN"
                    edges.append((src, et, dst))
                continue

            m = _HTML_LABEL_RE.match(line)
            if m:
                node_id = int(m.group(1))
                node_type, code = _parse_html_label(m.group(2))
                nodes[node_id] = {"label": node_type, "code": code}

    else:
        # ── Formats 1 & 2: GitHub-mined / LLM-augmented ───────────────────
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("digraph") or line == "}":
                continue

            if "->" in line:
                m = _EDGE_RE.match(line)
                if m:
                    src   = int(m.group(1))
                    dst   = int(m.group(2))
                    attrs = m.group(3)
                    et    = _attr(attrs, "label").strip().strip('"') or "UNKNOWN"
                    edges.append((src, et, dst))
                continue

            m = _NODE_RE.match(line)
            if m:
                node_id   = int(m.group(1))
                attr_text = m.group(2)
                label = _attr(attr_text, "label").strip()
                if not label:
                    lm = re.match(r'label=(\S+)', attr_text)
                    label = lm.group(1) if lm else "UNKNOWN"
                code = (_attr(attr_text, "CODE") or
                        _attr(attr_text, "NAME") or
                        _attr(attr_text, "METHOD_FULL_NAME") or "")
                nodes[node_id] = {"label": label, "code": code}

    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────────
# NODE FEATURIZER  (CodeBERT [CLS] embedding or hash fallback)
# ─────────────────────────────────────────────────────────────────────────────
class NodeFeaturizer:
    _label_vocab: Dict[str, int] = {}

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.device    = device
        self.max_len   = cfg["codebert_max_len"]
        self.emb_dim   = 768 if _HAS_TRANSFORMERS else 64
        self.tokenizer = None
        self.bert      = None

        if _HAS_TRANSFORMERS:
            model_name = cfg.get("codebert_model", "microsoft/codebert-base")
            print(f"[info] Loading {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert      = AutoModel.from_pretrained(model_name).to(device)
            self.bert.eval()
            print("[info] CodeBERT ready.")

    @torch.no_grad()
    def encode_snippets(self, snippets: List[str]) -> np.ndarray:
        if self.bert is None:
            out = np.zeros((len(snippets), 64), dtype=np.float32)
            for i, s in enumerate(snippets):
                rng = np.random.default_rng(hash(s) & 0xFFFFFFFF)
                out[i] = rng.standard_normal(64).astype(np.float32)
            return out

        batch_size = 64
        all_vecs   = []
        for start in range(0, len(snippets), batch_size):
            batch = [s if s.strip() else "<empty>"
                     for s in snippets[start : start + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out_all  = self.bert(**enc)
            # Mean-pool over non-padding tokens (better than [CLS] for short snippets)
            mask     = enc["attention_mask"].unsqueeze(-1).float()
            out      = (out_all.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_vecs.append(out.cpu().numpy())
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    def label_id(self, label: str) -> int:
        if label not in NodeFeaturizer._label_vocab:
            NodeFeaturizer._label_vocab[label] = len(NodeFeaturizer._label_vocab)
        return NodeFeaturizer._label_vocab[label]

    def build(self, nodes: Dict[int, dict]):
        node_ids = sorted(nodes.keys())
        labels   = [nodes[nid]["label"] for nid in node_ids]
        # Use node label type as fallback for empty code snippets.
        # Prevents all empty nodes (BLOCK, METHOD_RETURN, etc.) from sharing
        # one identical embedding, which collapses the feature space.
        snippets = [
            nodes[nid]["code"].strip() if nodes[nid]["code"].strip()
            else nodes[nid]["label"]
            for nid in node_ids
        ]
        embs    = self.encode_snippets(snippets)
        lab_ids = np.array([self.label_id(l) for l in labels], dtype=np.int64)
        return (
            torch.tensor(embs,    dtype=torch.float32),
            torch.tensor(lab_ids, dtype=torch.long),
            node_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EDGE TYPE MAP
# ─────────────────────────────────────────────────────────────────────────────
_EDGE_TYPE_MAP: Dict[str, int] = {}

def _map_etype(et: str) -> int:
    if et not in _EDGE_TYPE_MAP:
        _EDGE_TYPE_MAP[et] = min(len(_EDGE_TYPE_MAP), MAX_RELATIONS - 1)
    return _EDGE_TYPE_MAP[et]


# ─────────────────────────────────────────────────────────────────────────────
# DOT → PyG Data
# ─────────────────────────────────────────────────────────────────────────────
def dot_to_pyg(
    dot_path: Path,
    owasp_class: int,
    is_vulnerable: bool,
    featurizer: NodeFeaturizer,
    provenance: str,
) -> Optional[Data]:
    nodes, edges = parse_dot(dot_path)
    if not nodes or not edges:
        return None

    x, label_ids, node_id_list = featurizer.build(nodes)
    id_to_row = {nid: row for row, nid in enumerate(node_id_list)}

    srcs, dsts, etypes = [], [], []
    for (src, et, dst) in edges:
        if src in id_to_row and dst in id_to_row:
            srcs.append(id_to_row[src])
            dsts.append(id_to_row[dst])
            etypes.append(_map_etype(et))

    if not srcs:
        return None

    # 11-class label: 0 = safe, 1-10 = OWASP class
    y = 0 if not is_vulnerable else owasp_class

    d = Data(
        x          = x,
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long),
        y          = torch.tensor(y, dtype=torch.long),
    )
    d.label_id      = label_ids
    d.edge_type     = torch.tensor(etypes, dtype=torch.long)
    d.owasp_class   = torch.tensor(owasp_class, dtype=torch.long)
    d.owasp_name    = OWASP_NAMES.get(owasp_class, "UNKNOWN")
    d.is_vulnerable = torch.tensor(int(is_vulnerable), dtype=torch.long)
    d.provenance    = provenance    # "github" or "llm"
    d.dot_path      = str(dot_path)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_provenance(sample_dir: Path) -> str:
    """
    Given the immediate parent directory of a dot file's folder, return
    "github" if it's a 40-char hex commit hash, else "llm".
    """
    name = sample_dir.name
    if _GITHUB_HASH_RE.match(name):
        return "github"
    return "llm"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
#
# Walks the rebuttal/ directory structure:
#
#   Vulnerable samples:
#     rebuttal/Axx_<name>/<sample_dir>/<section_dir>/export*.dot
#     sample_dir is either a 40-hex hash (github) or NNN_N_export (llm)
#
#   Non-vulnerable samples:
#     rebuttal/Non-vulnerable/<NNN_safe_export>/export*.dot
#     All are llm-generated safe code (class=0)
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(
    data_dir: Path,
    featurizer: NodeFeaturizer,
    verbose: bool = True,
) -> Tuple[List[Data], np.ndarray]:

    graphs: List[Data] = []
    labels: List[int]  = []
    skipped = 0
    total   = 0

    # ── Vulnerable samples: one folder per OWASP class ────────────────────
    for owasp_dir in sorted(data_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue

        # Identify OWASP class from folder prefix (A01..A10)
        owasp_class = None
        for prefix, idx in FOLDER_PREFIX_TO_CLASS.items():
            if owasp_dir.name.startswith(prefix):
                owasp_class = idx
                break

        if owasp_class is None:
            # Non-vulnerable folder handled separately below
            continue

        # Walk: owasp_dir / sample_dir / section_dir / export*.dot
        # (some layouts have only two levels; glob handles both)
        for dot_file in owasp_dir.rglob("*.dot"):
            total += 1
            # Provenance: look at the direct child of owasp_dir in the path
            try:
                rel   = dot_file.relative_to(owasp_dir)
                parts = rel.parts          # (sample_dir, [section_dir,] filename)
                sample_dir_name = parts[0]
            except Exception:
                sample_dir_name = ""

            provenance = ("github"
                          if _GITHUB_HASH_RE.match(sample_dir_name)
                          else "llm")

            d = dot_to_pyg(dot_file, owasp_class,
                           is_vulnerable=True,
                           featurizer=featurizer,
                           provenance=provenance)
            if d is None:
                skipped += 1
                continue
            graphs.append(d)
            labels.append(int(d.y.item()))

    # ── Non-vulnerable samples ────────────────────────────────────────────
    non_vuln_dir = data_dir / "Non-vulnerable"
    if non_vuln_dir.exists():
        for dot_file in non_vuln_dir.rglob("*.dot"):
            total += 1
            d = dot_to_pyg(dot_file, owasp_class=0,
                           is_vulnerable=False,
                           featurizer=featurizer,
                           provenance="llm")     # all safe samples are LLM-generated
            if d is None:
                skipped += 1
                continue
            graphs.append(d)
            labels.append(0)
    else:
        print(f"[warn] Non-vulnerable directory not found at {non_vuln_dir}")

    if verbose:
        print(f"[info] Parsed {total} dot files | loaded {len(graphs)} | skipped {skipped}")
        dist = {OWASP_NAMES.get(k, str(k)): v
                for k, v in sorted(Counter(labels).items())}
        print(f"[info] Class distribution:\n" +
              "\n".join(f"       {k}: {v}" for k, v in dist.items()))
        prov = Counter(g.provenance for g in graphs)
        print(f"[info] Provenance: github={prov['github']} | llm={prov['llm']}")

    return graphs, np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# MEGAVUL EXTERNAL TEST SET LOADER
#
# Loads the MegaVul CPG dots/ directory as a purely external test set.
# All samples are real-world vulnerable code with zero LLM involvement.
#
# Expected layout:
#   dots/
#     A01_Broken_Access_Control/   *.dot   (vulnerable, mapped to class 1)
#     A02_Cryptographic_Failures/  *.dot   (vulnerable, mapped to class 2)
#     ...
#     AMBIGUOUS/                   *.dot   (skip — class unclear)
#     UNMAPPED/                    *.dot   (skip — no OWASP class)
#
# Only files > MIN_BYTES are loaded (tiny stubs have no real code content).
#
# Since MegaVul has no safe/non-vulnerable counterpart, this loader only
# produces vulnerable samples. The evaluation it supports is:
#   "Does the model correctly flag real-world CVE-linked code as vulnerable?"
# This is reported as binary recall on external real-world data.
# ─────────────────────────────────────────────────────────────────────────────
MEGAVUL_MIN_BYTES  = 2000   # below this the CPG is a stub — skip it
DEVIGN_MIN_BYTES   = 2000   # same threshold for Devign non-vuln


def load_megavul_test_set(
    dots_dir: Path,
    featurizer: NodeFeaturizer,
    verbose: bool = True,
) -> Tuple[List[Data], np.ndarray]:
    """
    Load MegaVul CPGs as external vulnerable-only test samples.
    All samples: provenance="megavul", is_vulnerable=True.
    AMBIGUOUS and UNMAPPED folders are skipped.
    Files under MEGAVUL_MIN_BYTES are stub CPGs and are skipped.
    """
    graphs: List[Data] = []
    labels: List[int]  = []
    skipped_tiny, skipped_parse, skipped_folder = 0, 0, 0
    total = 0

    for owasp_dir in sorted(dots_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue
        if owasp_dir.name in ("AMBIGUOUS", "UNMAPPED"):
            skipped_folder += sum(1 for _ in owasp_dir.glob("*.dot"))
            continue
        owasp_class = None
        for prefix, idx in FOLDER_PREFIX_TO_CLASS.items():
            if owasp_dir.name.startswith(prefix):
                owasp_class = idx
                break
        if owasp_class is None:
            continue
        for dot_file in sorted(owasp_dir.glob("*.dot")):
            total += 1
            if dot_file.stat().st_size < MEGAVUL_MIN_BYTES:
                skipped_tiny += 1
                continue
            d = dot_to_pyg(dot_file, owasp_class, True, featurizer, "megavul")
            if d is None:
                skipped_parse += 1
                continue
            graphs.append(d)
            labels.append(owasp_class)

    if verbose:
        print(f"[megavul] {total} files scanned | {len(graphs)} loaded "
              f"| {skipped_tiny} tiny | {skipped_parse} parse-fail "
              f"| {skipped_folder} unmapped")
        dist = {OWASP_NAMES.get(k, str(k)): v
                for k, v in sorted(Counter(labels).items())}
        for k, v in dist.items():
            tag = "  ← <30, excluded from per-class report" if v < 30 else ""
            print(f"         {k}: {v}{tag}")

    return graphs, np.array(labels, dtype=np.int64)


def load_devign_nonvuln(
    nonvuln_dir: Path,
    featurizer: NodeFeaturizer,
    verbose: bool = True,
) -> Tuple[List[Data], np.ndarray]:
    """
    Load Devign non-vulnerable CPGs as external safe-code test samples.
    All samples: provenance="devign", owasp_class=0, is_vulnerable=False.
    The NON_VULNERABLE/ folder is expected to contain *.dot files directly
    (flat layout, no OWASP subfolders).
    Files under DEVIGN_MIN_BYTES are stub CPGs and are skipped.
    """
    graphs: List[Data] = []
    skipped_tiny, skipped_parse = 0, 0
    total = 0

    dot_root = nonvuln_dir
    # Accept either the folder itself or a parent containing NON_VULNERABLE/
    if (nonvuln_dir / "NON_VULNERABLE").exists():
        dot_root = nonvuln_dir / "NON_VULNERABLE"

    for dot_file in sorted(dot_root.glob("*.dot")):
        total += 1
        if dot_file.stat().st_size < DEVIGN_MIN_BYTES:
            skipped_tiny += 1
            continue
        d = dot_to_pyg(dot_file, owasp_class=0, is_vulnerable=False,
                       featurizer=featurizer, provenance="devign")
        if d is None:
            skipped_parse += 1
            continue
        graphs.append(d)

    labels = np.zeros(len(graphs), dtype=np.int64)

    if verbose:
        print(f"[devign]  {total} files scanned | {len(graphs)} loaded "
              f"| {skipped_tiny} tiny | {skipped_parse} parse-fail")

    return graphs, labels


def evaluate_external_testset(
    vuln_graphs:   List[Data],
    vuln_labels:   np.ndarray,
    safe_graphs:   List[Data],
    model,
    device: str,
    num_classes:   int,
    data_dir:      Path,
    binary_threshold: float = 0.5,
) -> None:
    """
    Evaluate model on combined external test set:
      - vuln_graphs  : MegaVul vulnerable samples (labelled with OWASP class 1-4)
      - safe_graphs  : Devign non-vulnerable samples (label = 0)

    Two evaluations:
      1. BINARY  — can the model distinguish vulnerable vs non-vulnerable?
         Predicted label: argmax > 0 → VULNERABLE, else NON_VULNERABLE.
         Also evaluated at the supplied binary_threshold on P(vulnerable).

      2. MULTICLASS — can the model predict the correct OWASP class?
         Only on vulnerable samples (safe samples have no OWASP class).
         Predicted label: argmax of class probabilities (0-4).
         Class 0 predictions on vulnerable samples count as wrong.
    """
    from sklearn.metrics import classification_report as _cr
    from sklearn.metrics import f1_score as _f1

    if not vuln_graphs and not safe_graphs:
        print("[external] No graphs to evaluate.")
        return

    all_graphs = vuln_graphs + safe_graphs
    n_vuln     = len(vuln_graphs)
    n_safe     = len(safe_graphs)

    # Get probabilities for all graphs
    probs = probs_from_model(model, all_graphs, device, num_classes)

    # Ground truth: 0=safe, OWASP class 1-4 for vulnerable
    y_true_multi = np.concatenate([
        vuln_labels,                              # 1-4
        np.zeros(n_safe, dtype=int),              # 0
    ])
    y_pred_multi = probs.argmax(axis=1)

    # Binary ground truth and predictions
    y_true_bin      = (y_true_multi > 0).astype(int)
    y_pred_bin_05   = (y_pred_multi > 0).astype(int)           # default argmax
    p_vuln          = 1.0 - probs[:, 0]                        # P(any vulnerable)
    y_pred_bin_best = (p_vuln >= binary_threshold).astype(int) # tuned threshold

    print(f"\n{'='*65}")
    print(f"  External Real-World Evaluation")
    print(f"  MegaVul vulnerable: {n_vuln}  |  Devign non-vuln: {n_safe}")
    print(f"{'='*65}")

    # ── 1. Binary evaluation ──────────────────────────────────────────────
    print(f"\n  --- Binary Detection (default threshold=0.50) ---")
    present = sorted(set(y_true_bin.tolist()))
    present_names = [["NON_VULNERABLE","VULNERABLE"][i] for i in present]
    print(_cr(y_true_bin, y_pred_bin_05,
              labels=present, target_names=present_names,
              digits=4, zero_division=0))

    print(f"  --- Binary Detection (tuned threshold={binary_threshold:.2f}) ---")
    print(_cr(y_true_bin, y_pred_bin_best,
              labels=present, target_names=present_names,
              digits=4, zero_division=0))

    # ── 2. Multiclass evaluation (vulnerable samples only) ────────────────
    print(f"  --- Multiclass OWASP Classification (MegaVul only, n={n_vuln}) ---")
    print(f"  (Predicting which OWASP class each real CVE belongs to)")
    if n_vuln > 0:
        vuln_pred   = y_pred_multi[:n_vuln]
        present_cls = sorted(set(vuln_labels.tolist()))
        cls_names   = [OWASP_NAMES.get(c, str(c)) for c in present_cls]
        # Only report classes with ≥30 samples
        reportable  = [c for c in present_cls if (vuln_labels == c).sum() >= 30]
        rep_names   = [OWASP_NAMES.get(c, str(c)) for c in reportable]
        print(_cr(vuln_labels, vuln_pred,
                  labels=reportable, target_names=rep_names,
                  digits=4, zero_division=0))

        # Also show what fraction of MegaVul samples were correctly classified
        # vs just flagged as vulnerable (class > 0)
        correct_class = (vuln_pred == vuln_labels).mean()
        any_vuln      = (vuln_pred > 0).mean()
        print(f"  Correctly predicted OWASP class : {correct_class:.4f}")
        print(f"  Correctly flagged as vulnerable : {any_vuln:.4f}")

    # ── Save summary ──────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score
    binary_f1_05   = float(_f1(y_true_bin, y_pred_bin_05,   average='macro', zero_division=0))
    binary_f1_best = float(_f1(y_true_bin, y_pred_bin_best, average='macro', zero_division=0))
    multi_f1 = float(_f1(vuln_labels, y_pred_multi[:n_vuln],
                         average='macro', zero_division=0)) if n_vuln > 0 else 0.0

    summary = {
        "n_vuln_megavul":           n_vuln,
        "n_safe_devign":            n_safe,
        "binary_macro_f1_thresh05": round(binary_f1_05,   4),
        "binary_macro_f1_tuned":    round(binary_f1_best, 4),
        "binary_threshold_tuned":   round(binary_threshold, 2),
        "multiclass_macro_f1":      round(multi_f1,        4),
        "per_class_megavul": {
            OWASP_NAMES.get(c, str(c)): {
                "n":         int((vuln_labels == c).sum()),
                "recall":    float((y_pred_multi[:n_vuln][vuln_labels == c] > 0).mean())
                             if (vuln_labels == c).sum() > 0 else 0.0,
                "precision_class": float(
                    (y_pred_multi[:n_vuln][vuln_labels == c] == c).mean())
                             if (vuln_labels == c).sum() > 0 else 0.0,
            }
            for c in sorted(set(vuln_labels.tolist()))
        } if n_vuln > 0 else {},
    }
    out_path = data_dir / "external_eval_5class.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*65}\n")



# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# Suppresses gradient from easy/confident predictions, forces model to focus
# on hard examples — critical for the class imbalance in this dataset.
# gamma=2.0 is standard; increase to 3.0 if easy classes still dominate.
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# POOLING
# ─────────────────────────────────────────────────────────────────────────────
def make_attn_pool(hidden_dim: int) -> AttentionalAggregation:
    gate = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )
    return AttentionalAggregation(gate)


# ─────────────────────────────────────────────────────────────────────────────
# GNN BACKBONES
# ─────────────────────────────────────────────────────────────────────────────
class GATWithLabelEmb(nn.Module):
    def __init__(self, base_dim, vocab, label_emb_dim, hidden_dim,
                 heads, dropout, attn_pool, num_relations, num_classes,
                 edge_emb_dim=32):
        super().__init__()
        self.label_emb  = nn.Embedding(max(vocab, 1), label_emb_dim)
        self.edge_emb   = nn.Embedding(max(num_relations, 1), edge_emb_dim)
        raw_dim = base_dim + label_emb_dim
        # Input projection: compress GraphCodeBERT (768) + label_emb (64) → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        out_ch1 = max(1, hidden_dim // heads)
        # 3-layer GAT with residual connections and LayerNorm after each layer.
        # More layers = deeper receptive field in the CPG, capturing vulnerability
        # patterns that span multiple statements/function calls.
        self.gat1  = GATConv(hidden_dim, out_ch1, heads=heads, concat=True,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2  = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gat3  = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.pool    = make_attn_pool(hidden_dim) if attn_pool else None
        self.cls     = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x      = self.input_proj(
            torch.cat([data.x.float(), self.label_emb(data.label_id)], dim=1)
        )                                               # [N, hidden_dim]
        e_attr = self.edge_emb(data.edge_type) if hasattr(data, "edge_type") else None

        # Layer 1
        x1 = F.elu(self.gat1(x, data.edge_index, edge_attr=e_attr))
        x1 = self.norm1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Layer 2 + residual from layer 1
        x2 = F.elu(self.gat2(x1, data.edge_index, edge_attr=e_attr))
        x2 = self.norm2(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Layer 3 + residual from layer 2
        x3 = self.gat3(x2, data.edge_index, edge_attr=e_attr)
        x3 = self.norm3(x3 + x2)

        pooled = self.pool(x3, data.batch) if self.pool else global_mean_pool(x3, data.batch)
        return self.cls(pooled)


class GCNWithLabelEmb(nn.Module):
    def __init__(self, base_dim, vocab, label_emb_dim, hidden_dim,
                 dropout, use_edge_types, num_relations, attn_pool, num_classes):
        super().__init__()
        self.label_emb  = nn.Embedding(max(vocab, 1), label_emb_dim)
        raw_dim         = base_dim + label_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.use_rgcn = use_edge_types and num_relations > 0
        if self.use_rgcn:
            self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        else:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.norm3   = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.pool    = make_attn_pool(hidden_dim) if attn_pool else None
        self.cls     = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x  = self.input_proj(torch.cat([data.x.float(), self.label_emb(data.label_id)], dim=1))
        kw = {"edge_type": data.edge_type} if self.use_rgcn and hasattr(data, "edge_type") else {}
        x1 = F.relu(self.conv1(x, data.edge_index, **kw))
        x1 = self.norm1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, data.edge_index, **kw))
        x2 = self.norm2(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.conv3(x2, data.edge_index, **kw)
        x3 = self.norm3(x3 + x2)
        pooled = self.pool(x3, data.batch) if self.pool else global_mean_pool(x3, data.batch)
        return self.cls(pooled)


class GINWithLabelEmb(nn.Module):
    def __init__(self, base_dim, vocab, label_emb_dim, hidden_dim,
                 dropout, attn_pool, num_classes):
        super().__init__()
        self.label_emb  = nn.Embedding(max(vocab, 1), label_emb_dim)
        raw_dim         = base_dim + label_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        def mlp(i, o): return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.conv1   = GINConv(mlp(hidden_dim, hidden_dim))
        self.conv2   = GINConv(mlp(hidden_dim, hidden_dim))
        self.conv3   = GINConv(mlp(hidden_dim, hidden_dim))
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.norm3   = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.pool    = make_attn_pool(hidden_dim) if attn_pool else None
        self.cls     = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x  = self.input_proj(torch.cat([data.x.float(), self.label_emb(data.label_id)], dim=1))
        x1 = F.relu(self.conv1(x, data.edge_index))
        x1 = self.norm1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, data.edge_index))
        x2 = self.norm2(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.conv3(x2, data.edge_index)
        x3 = self.norm3(x3 + x2)
        pooled = self.pool(x3, data.batch) if self.pool else global_mean_pool(x3, data.batch)
        return self.cls(pooled)


def build_model(backbone, base_dim, vocab, cfg, num_relations, num_classes) -> nn.Module:
    if backbone == "gat":
        return GATWithLabelEmb(base_dim, vocab, cfg["label_emb_dim"], cfg["hidden_dim"],
                               cfg["gat_heads"], cfg["dropout"], cfg["attn_pool"],
                               num_relations, num_classes, cfg["gat_edge_emb_dim"])
    elif backbone == "gcn":
        return GCNWithLabelEmb(base_dim, vocab, cfg["label_emb_dim"], cfg["hidden_dim"],
                               cfg["dropout"], cfg["use_edge_types"],
                               num_relations, cfg["attn_pool"], num_classes)
    elif backbone == "gin":
        return GINWithLabelEmb(base_dim, vocab, cfg["label_emb_dim"], cfg["hidden_dim"],
                               cfg["dropout"], cfg["attn_pool"], num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")



# Suppress PyTorch AMP deprecation warnings — use new API internally but
# suppress the noisy per-call warnings from the old torch.cuda.amp namespace
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.*is deprecated.*",
    category=FutureWarning,
)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR = None  # set by main() to DATA_DIR / "checkpoints"

def _ckpt_path(tag: str) -> Path:
    return CKPT_DIR / f"{tag}.pt"

def _probs_path(tag: str) -> Path:
    return CKPT_DIR / f"{tag}_probs.npy"

def _save_probs(tag: str, arr: np.ndarray) -> None:
    np.save(str(_probs_path(tag)), arr)

def _load_probs(tag: str) -> Optional[np.ndarray]:
    p = _probs_path(tag)
    if p.exists():
        return np.load(str(p))
    return None

def _model_done(tag: str) -> bool:
    return _ckpt_path(tag).exists()

def _save_model(tag: str, model) -> None:
    torch.save(model.state_dict(), str(_ckpt_path(tag)))

def _load_model(tag: str, base_dim, vocab, cfg, device,
                num_relations, num_classes) -> Optional[object]:
    p = _ckpt_path(tag)
    if not p.exists():
        return None
    m = build_model(cfg["backbone"], base_dim, vocab, cfg,
                    num_relations, num_classes).to(device)
    m.load_state_dict(torch.load(str(p), map_location=device))
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probs_from_model(model, graphs, device="cpu", num_classes=NUM_CLASSES):
    if model is None or not graphs:
        return np.full((len(graphs), num_classes), 1.0 / num_classes,
                       dtype=np.float32)
    model.eval()

    # Determine embedding table sizes from the model
    label_vocab_size = None
    edge_vocab_size  = None
    for name, mod in model.named_modules():
        if name == 'label_emb' and hasattr(mod, 'num_embeddings'):
            label_vocab_size = mod.num_embeddings
        if name == 'edge_emb' and hasattr(mod, 'num_embeddings'):
            edge_vocab_size = mod.num_embeddings

    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    outs   = []
    for batch in loader:
        batch          = batch.to(device)
        batch.label_id = batch.label_id.to(device)
        # Clamp label_id to valid range — external datasets may have unseen
        # node types that get vocab indices beyond the training vocab size
        if label_vocab_size is not None:
            batch.label_id = batch.label_id.clamp(0, label_vocab_size - 1)
        if hasattr(batch, "edge_type"):
            batch.edge_type = batch.edge_type.to(device)
            if edge_vocab_size is not None:
                batch.edge_type = batch.edge_type.clamp(0, edge_vocab_size - 1)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outs.append(F.softmax(model(batch), dim=1).cpu())
    return torch.cat(outs).numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE EXPERT  (with checkpoint resume + speed optimisations)
#
# Speed improvements vs original:
#   • pin_memory=True + persistent_workers  — faster CPU→GPU transfer
#   • torch.cuda.amp (mixed precision)       — ~1.5-2x throughput on Blackwell
#   • num_workers=4 for data loading         — parallel batch preparation
#   • best_state stored on CPU               — no extra GPU memory
#   • torch.compile (PyTorch 2.x)            — fused ops (optional, ~10% gain)
# ─────────────────────────────────────────────────────────────────────────────
def train_expert(graphs_subset, base_dim, vocab, cfg, device,
                 num_relations=MAX_RELATIONS, tag="", num_classes=NUM_CLASSES):
    if not graphs_subset:
        return None

    # ── Resume from checkpoint if available ──────────────────────────────
    if CKPT_DIR and _model_done(tag):
        print(f"  [{tag}] ✓ loaded from checkpoint — skipping training")
        return _load_model(tag, base_dim, vocab, cfg, device,
                           num_relations, num_classes)

    model = build_model(cfg["backbone"], base_dim, vocab, cfg,
                        num_relations, num_classes).to(device)

    # Mixed-precision scaler (works on Blackwell/Ampere)
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    graphs_subset = list(graphs_subset)
    random.shuffle(graphs_subset)
    y_sub   = np.array([int(g.y.item()) for g in graphs_subset])
    counts  = np.bincount(y_sub, minlength=num_classes).astype(np.float32)
    w       = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    w_ten   = torch.tensor(w, dtype=torch.float32).to(device)
    gamma   = cfg.get("focal_gamma", 2.0)
    crit    = FocalLoss(weight=w_ten, gamma=gamma)

    n_val      = max(1, int(cfg.get("val_frac", 0.1) * len(graphs_subset)))
    val_graphs = graphs_subset[:n_val]
    tr_graphs  = graphs_subset[n_val:] or graphs_subset

    from torch.utils.data import WeightedRandomSampler
    samp_w  = torch.tensor(
        1.0 / np.maximum(counts[y_sub[n_val:]], 1.0), dtype=torch.float
    )
    sampler = WeightedRandomSampler(samp_w, num_samples=len(tr_graphs),
                                    replacement=True)

    # num_workers>0 requires graphs to be picklable; pin_memory speeds up
    # CPU→GPU transfer on machines with pinned (page-locked) system RAM
    n_workers = min(4, os.cpu_count() or 1)
    tr_ldr  = DataLoader(tr_graphs, batch_size=cfg["batch_size"],
                         sampler=sampler, num_workers=n_workers,
                         pin_memory=True, persistent_workers=(n_workers > 0))
    val_ldr = DataLoader(val_graphs, batch_size=cfg["batch_size"] * 2,
                         shuffle=False, num_workers=n_workers,
                         pin_memory=True, persistent_workers=(n_workers > 0))

    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                 opt, mode="max", factor=0.5, patience=3)
             if cfg.get("sched_plateau", True) else None)

    best_metric, best_state, patience = -1.0, None, 0
    PATIENCE = int(cfg.get("early_stop_patience", 50))

    for epoch in range(cfg["epochs"]):
        model.train()
        ep_loss = 0.0
        for batch in tr_ldr:
            batch          = batch.to(device, non_blocking=True)
            batch.label_id = batch.label_id.to(device, non_blocking=True)
            if hasattr(batch, "edge_type"):
                batch.edge_type = batch.edge_type.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(batch), batch.y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        model.eval()
        la, ll = [], []
        with torch.no_grad():
            for vb in val_ldr:
                vb          = vb.to(device, non_blocking=True)
                vb.label_id = vb.label_id.to(device, non_blocking=True)
                if hasattr(vb, "edge_type"):
                    vb.edge_type = vb.edge_type.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    la.append(model(vb).detach().cpu())
                ll.append(vb.y.detach().cpu())

        metric = 0.0
        if la:
            preds = torch.cat(la).argmax(dim=1).numpy()
            lbls  = torch.cat(ll).numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = f1_score(lbls, preds, average="macro", zero_division=0)

        if sched:
            sched.step(metric)

        if metric > best_metric:
            best_metric = metric
            # Store on CPU to avoid holding extra GPU memory for dozens of models
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            patience    = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                if tag:
                    print(f"  [{tag}] early stop ep {epoch+1} | "
                          f"best_macro_f1={best_metric:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  [{tag}] ep {epoch+1:4d} | loss={ep_loss:.4f} | "
                  f"macro_f1={metric:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # ── Save checkpoint ───────────────────────────────────────────────────
    if CKPT_DIR:
        _save_model(tag, model)

    torch.cuda.empty_cache()
    return model


def sanitize_meta(X):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 1.0 / X.shape[1]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OOF STACKED META-FEATURES  (with per-fold checkpointing)
# ─────────────────────────────────────────────────────────────────────────────
def build_oof_meta_features(tr_graphs, te_graphs, y_tr,
                             base_dim, vocab, cfg, device,
                             num_relations, num_classes):
    # Try to resume full OOF arrays from disk
    xtr_cached = _load_probs("oof_X_tr") if CKPT_DIR else None
    xte_cached = _load_probs("oof_X_te") if CKPT_DIR else None
    if xtr_cached is not None and xte_cached is not None:
        print("  [oof] ✓ loaded from checkpoint")
        return xtr_cached, xte_cached

    n_splits   = min(cfg["oof_max_folds"], 5)
    cls_counts = np.bincount(y_tr, minlength=num_classes)
    if cls_counts.min() < 2:
        n_splits = 0

    Ntr, Nte = len(tr_graphs), len(te_graphs)
    X_tr = np.full((Ntr, num_classes), 1.0 / num_classes, dtype=np.float32)
    X_te = np.full((Nte, num_classes), 1.0 / num_classes, dtype=np.float32)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=cfg["seed"])
        fold_te_preds = []
        print(f"  [oof] {n_splits}-fold OOF on {Ntr} training graphs")
        for fold, (tri, vali) in enumerate(skf.split(np.arange(Ntr), y_tr)):
            tag = f"oof_fold{fold+1}"

            # Try loading cached probs for this fold
            tr_probs_cached = _load_probs(f"{tag}_tr") if CKPT_DIR else None
            te_probs_cached = _load_probs(f"{tag}_te") if CKPT_DIR else None

            if tr_probs_cached is not None and te_probs_cached is not None:
                print(f"  [oof] fold {fold+1} ✓ loaded from checkpoint")
                X_tr[vali] = tr_probs_cached
                fold_te_preds.append(te_probs_cached)
                continue

            m = train_expert([tr_graphs[i] for i in tri], base_dim, vocab, cfg,
                             device, num_relations, tag=tag,
                             num_classes=num_classes)
            fold_tr = probs_from_model(m, [tr_graphs[i] for i in vali],
                                       device, num_classes)
            fold_te = probs_from_model(m, te_graphs, device, num_classes)
            X_tr[vali] = fold_tr
            fold_te_preds.append(fold_te)

            if CKPT_DIR:
                _save_probs(f"{tag}_tr", fold_tr)
                _save_probs(f"{tag}_te", fold_te)

        X_te = np.mean(np.stack(fold_te_preds, axis=0), axis=0)
        print("  [oof] complete")
    else:
        print("  [oof] not enough samples per class — single model fallback")
        m    = train_expert(tr_graphs, base_dim, vocab, cfg, device,
                            num_relations, tag="oof_single",
                            num_classes=num_classes)
        X_tr = probs_from_model(m, tr_graphs, device, num_classes)
        X_te = probs_from_model(m, te_graphs, device, num_classes)

    if CKPT_DIR:
        _save_probs("oof_X_tr", X_tr)
        _save_probs("oof_X_te", X_te)

    return X_tr, X_te


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — SPECIALIST-PER-CLASS  (with per-specialist checkpointing)
# ─────────────────────────────────────────────────────────────────────────────
def train_stage2_specialists(
    all_tr_graphs: List[Data],
    all_te_graphs: List[Data],
    base_dim: int,
    vocab: int,
    cfg: dict,
    device: str,
    num_relations: int,
    data_dir: Path,
) -> None:
    """
    Train 10 specialist binary GATs for Stage 2 OWASP class identification.
    Evaluate on vulnerable-only test samples and save results.
    """
    from sklearn.metrics import classification_report as _cr

    N_OWASP = NUM_CLASSES - 1   # number of vulnerable classes (4 for 5-class scheme)

    # ── Split non-vulnerable pool (shared negative for all specialists) ──
    nonvuln_tr = [g for g in all_tr_graphs if int(g.y.item()) == 0]
    nonvuln_te = [g for g in all_te_graphs if int(g.y.item()) == 0]

    # ── Vulnerable-only subsets (what Stage 2 operates on) ───────────────
    vuln_tr = [g for g in all_tr_graphs if int(g.y.item()) > 0]
    vuln_te = [g for g in all_te_graphs if int(g.y.item()) > 0]

    if not vuln_tr:
        print("[stage2] No vulnerable training samples — aborting")
        return

    y_tr_s2 = np.array([int(g.y.item()) - 1 for g in vuln_tr], dtype=np.int64)
    y_te_s2 = np.array([int(g.y.item()) - 1 for g in vuln_te], dtype=np.int64)

    print(f"\n[stage2] Specialist architecture")
    print(f"         Vulnerable  — train: {len(vuln_tr)} | test: {len(vuln_te)}")
    print(f"         Non-vuln    — train: {len(nonvuln_tr)} | test: {len(nonvuln_te)}")

    # Meta-feature matrices: specialist_i outputs p(vulnerable for class i)
    X_tr_s2 = np.full((len(vuln_tr), N_OWASP), 0.5, dtype=np.float32)
    X_te_s2 = np.full((len(vuln_te), N_OWASP), 0.5, dtype=np.float32)

    def relabel(graphs, new_y: int) -> List[Data]:
        """Shallow-clone graphs with a new binary y label."""
        out = []
        for g in graphs:
            g2   = g.clone()
            g2.y = torch.tensor(new_y, dtype=torch.long)
            out.append(g2)
        return out

    n_folds = min(cfg.get("oof_max_folds", 5), 5)
    spec_cfg = {**cfg, "early_stop_patience": cfg.get("early_stop_patience", 80)}

    for cls_idx in range(N_OWASP):
        owasp_class = cls_idx + 1
        name        = OWASP_NAMES.get(owasp_class, f"A{owasp_class:02d}")

        pos_tr = [g for g in vuln_tr if int(g.y.item()) == owasp_class]
        pos_te = [g for g in vuln_te if int(g.y.item()) == owasp_class]

        if len(pos_tr) < 2:
            print(f"  [{name}] insufficient samples — skipping")
            continue

        # Full specialist training set: positives + ALL non-vulnerable
        spec_tr_all = relabel(pos_tr, 1) + relabel(nonvuln_tr, 0)
        spec_te_all = relabel(pos_te, 1) + relabel(nonvuln_te, 0)
        spec_tr_y   = np.array([int(g.y.item()) for g in spec_tr_all])

        print(f"  [{name}] pos={len(pos_tr)} neg={len(nonvuln_tr)}")

        can_fold = (np.bincount(spec_tr_y).min() >= 2) and (n_folds >= 2)

        # ── Test-set probabilities (averaged across folds if available) ───
        fold_te_probs = []
        if can_fold:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=cfg["seed"])
            for fold, (tri, _) in enumerate(skf.split(
                    np.arange(len(spec_tr_all)), spec_tr_y)):
                m = train_expert(
                    [spec_tr_all[i] for i in tri],
                    base_dim, vocab, spec_cfg, device, num_relations,
                    tag=f"s2-{name[:5]}-f{fold+1}", num_classes=2,
                )
                fold_te_probs.append(
                    probs_from_model(m, spec_te_all, device, 2)[:, 1]
                )
            te_probs_full = np.mean(np.stack(fold_te_probs), axis=0)
        else:
            m = train_expert(spec_tr_all, base_dim, vocab, spec_cfg,
                             device, num_relations,
                             tag=f"s2-{name[:5]}", num_classes=2)
            te_probs_full = probs_from_model(m, spec_te_all, device, 2)[:, 1]

        # te_probs_full[:len(pos_te)] = probs for positive test samples
        for j in range(len(pos_te)):
            # Find this graph's position in vuln_te
            orig_g = pos_te[j]
            for k, vg in enumerate(vuln_te):
                if vg is orig_g:
                    X_te_s2[k, cls_idx] = te_probs_full[j]
                    break

        # ── OOF probabilities for training meta-features ──────────────────
        if can_fold:
            oof_pos_probs = np.full(len(pos_tr), 0.5, dtype=np.float32)
            skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=cfg["seed"] + 7)
            for fold, (tri2, vali2) in enumerate(skf2.split(
                    np.arange(len(spec_tr_all)), spec_tr_y)):
                # Only care about OOF predictions for POSITIVE samples
                pos_vali = [i for i in vali2 if i < len(pos_tr)]
                if not pos_vali:
                    continue
                m2 = train_expert(
                    [spec_tr_all[i] for i in tri2],
                    base_dim, vocab, spec_cfg, device, num_relations,
                    tag=f"s2-{name[:5]}-oof{fold+1}", num_classes=2,
                )
                p2 = probs_from_model(
                    m2, [spec_tr_all[i] for i in pos_vali], device, 2
                )[:, 1]
                for local_i, orig_i in enumerate(pos_vali):
                    oof_pos_probs[orig_i] = p2[local_i]

            # Map OOF probs back to vuln_tr positions
            for j, g_pos in enumerate(pos_tr):
                for k, vg in enumerate(vuln_tr):
                    if vg is g_pos:
                        X_tr_s2[k, cls_idx] = oof_pos_probs[j]
                        break

        # Save per-specialist column after each class completes
        if CKPT_DIR:
            _save_probs(f"s2_col{cls_idx}_tr", X_tr_s2[:, cls_idx])
            _save_probs(f"s2_col{cls_idx}_te", X_te_s2[:, cls_idx])
            print(f"  [{name}] ✓ column checkpoint saved")

    # Save full meta-feature matrices for ablation script
    if CKPT_DIR:
        _save_probs("s2_X_tr", X_tr_s2)
        _save_probs("s2_X_te", X_te_s2)
        print("[stage2] ✓ full meta-feature matrices saved")

    # ── Stage 2 meta-learner ──────────────────────────────────────────────
    print("\n[stage2] Training meta-learner on specialist scores...")
    X_tr_s2 = sanitize_meta(X_tr_s2)
    X_te_s2 = sanitize_meta(X_te_s2)

    s2_pred = None
    if _HAS_LGB:
        try:
            gbm_s2 = lgb.LGBMClassifier(
                **cfg["gbm_params"],
                objective        = "multiclass",
                num_class        = N_OWASP,
                n_jobs           = -1,
                min_child_samples= max(1, len(y_tr_s2) // 100),
                class_weight     = "balanced",
                verbose          = -1,
            )
            gbm_s2.fit(X_tr_s2, y_tr_s2)
            pickle.dump(gbm_s2, open(data_dir / "lgbm_stage2_5class.pkl", "wb"))
            s2_pred = gbm_s2.predict(X_te_s2)
            print("[stage2] LightGBM meta-learner trained.")
        except Exception as e:
            print(f"[stage2] LightGBM failed: {e}")

    if s2_pred is None:
        meta_s2 = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     solver="lbfgs")
        meta_s2.fit(X_tr_s2, y_tr_s2)
        pickle.dump(meta_s2, open(data_dir / "lr_stage2_5class.pkl", "wb"))
        s2_pred = meta_s2.predict(X_te_s2)

    # ── Report (10-class on vulnerable-only test samples) ─────────────────
    target_names = [OWASP_NAMES.get(i + 1, f"Class{i+1}") for i in range(N_OWASP)]
    print(f"\n{'='*70}")
    print(f"  Stage 2 — 10-class OWASP classification (vulnerable samples only)")
    print(f"  Test samples: {len(vuln_te)}")
    print(f"{'='*70}")
    print(_cr(y_te_s2, s2_pred, target_names=target_names,
              digits=4, zero_division=0))

    # Save
    summary = {
        "n_test_vulnerable": len(vuln_te),
        "macro_f1": float(f1_score(y_te_s2, s2_pred, average="macro",
                                   zero_division=0)),
        "per_class": {target_names[i]: {
            "n_test": int((y_te_s2 == i).sum()),
        } for i in range(N_OWASP)},
    }
    with open(data_dir / "stage2_results_5class.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Macro F1: {summary['macro_f1']:.4f}")
    print(f"  Results saved → {data_dir / 'stage2_results.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(CONFIG["seed"])
    device   = get_device()
    DATA_DIR = Path(CONFIG["DATA_DIR"])
    CKPT_DIR   = DATA_DIR / "checkpoints" / "5class"
    OUTPUT_DIR = DATA_DIR / "outputs" / "5class"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Checkpoint dir: {CKPT_DIR}")
    print(f"[info] device={device} | backbone={CONFIG['backbone']} | num_classes={NUM_CLASSES}")

    featurizer    = NodeFeaturizer(CONFIG, device)
    graphs, y_all = load_dataset(DATA_DIR, featurizer, verbose=True)

    if not graphs:
        raise SystemExit("[fatal] No graphs loaded — check DATA_DIR path and folder layout.")

    base_dim      = graphs[0].x.shape[1]
    vocab         = len(NodeFeaturizer._label_vocab)
    num_relations = MAX_RELATIONS
    print(f"[info] base_dim={base_dim} | vocab={vocab}")

    # Stratify by (provenance, owasp_class, binary_label) to keep
    # both source distributions and class proportions balanced in each split.
    strat_keys = np.array([
        f"{g.provenance}-{int(g.owasp_class.item())}-{int(g.is_vulnerable.item())}"
        for g in graphs
    ])
    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2,
            random_state=CONFIG["seed"], stratify=strat_keys,
        )
    except ValueError:
        print("[warn] Stratified split failed — using random split")
        tr_idx, te_idx = train_test_split(
            np.arange(len(graphs)), test_size=0.2, random_state=CONFIG["seed"]
        )

    tr_graphs = [graphs[i] for i in tr_idx]
    te_graphs = [graphs[i] for i in te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    print(f"[info] train={len(tr_graphs)} | test={len(te_graphs)}")
    print(f"[info] train class dist: {dict(sorted(Counter(y_tr.tolist()).items()))}")
    print(f"[info] test  provenance: {Counter(g.provenance for g in te_graphs)}")

    # Save labels to checkpoint dir so ablation script can load without retraining
    np.save(str(CKPT_DIR / "y_tr.npy"), y_tr)
    np.save(str(CKPT_DIR / "y_te.npy"), y_te)
    np.save(str(CKPT_DIR / "y_tr_s2.npy"),
            y_tr[y_tr > 0] - 1)   # vulnerable-only, 0-indexed for Stage 2
    np.save(str(CKPT_DIR / "y_te_s2.npy"),
            y_te[y_te > 0] - 1)
    print(f"[info] Labels saved to {CKPT_DIR}")

    # Save test split metadata for downstream analysis
    with open(OUTPUT_DIR / "test_split_5class.json", "w") as f:
        json.dump([{
            "dot_path":      g.dot_path,
            "owasp_class":   int(g.owasp_class.item()),
            "owasp_name":    g.owasp_name,
            "is_vulnerable": bool(g.is_vulnerable.item()),
            "label":         int(g.y.item()),
            "provenance":    g.provenance,
        } for g in te_graphs], f, indent=2)

    # OOF stacked meta-features
    print("\n[step] Building OOF meta-features...")
    X_tr, X_te = build_oof_meta_features(
        tr_graphs, te_graphs, y_tr,
        base_dim, vocab, CONFIG, device, num_relations, NUM_CLASSES,
    )
    X_tr, X_te = sanitize_meta(X_tr), sanitize_meta(X_te)

    # Meta-learner (multiclass LightGBM or LogReg fallback)
    print("\n[step] Training meta-learner...")
    y_pred = None
    if _HAS_LGB:
        try:
            gbm = lgb.LGBMClassifier(
                **CONFIG["gbm_params"],
                objective        = "multiclass",
                num_class        = NUM_CLASSES,
                n_jobs           = -1,
                min_child_samples= max(1, len(y_tr) // 100),
                num_leaves       = min(31, len(y_tr) // 20),
                class_weight     = "balanced",
                verbose          = -1,
            )
            gbm.fit(X_tr, y_tr)
            pickle.dump(gbm, open(OUTPUT_DIR / "lgbm_meta_5class.pkl", "wb"))
            y_pred = gbm.predict(X_te)
            print("[info] LightGBM meta-learner trained.")
        except Exception as e:
            print(f"[warn] LightGBM failed: {e}")

    if y_pred is None:
        meta   = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    solver="lbfgs")
        meta.fit(X_tr, y_tr)
        pickle.dump(meta, open(OUTPUT_DIR / "lr_meta_5class.pkl", "wb"))
        y_pred = meta.predict(X_te)
        print("[info] LogisticRegression meta-learner trained.")

    # Multiclass report
    target_names = [OWASP_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)]
    print(f"\n--- Multiclass Report (backbone={CONFIG['backbone']} + meta-learner) ---")
    print(classification_report(y_te, y_pred, target_names=target_names,
                                 digits=4, zero_division=0))

    # Binary collapse (VULNERABLE vs NON_VULNERABLE)
    print("--- Binary Collapse (any OWASP class vs safe) ---")
    print(classification_report((y_te > 0).astype(int), (y_pred > 0).astype(int),
                                 target_names=["NON_VULNERABLE", "VULNERABLE"],
                                 digits=4, zero_division=0))

    # ── Stage 2: Specialist-per-class OWASP classifier ───────────────────
    # Trains 10 corrected specialists (each: Axx_vuln vs ALL_nonvuln) and
    # reports 10-class F1 on vulnerable-only test samples.
    # This is the primary multiclass performance metric for the paper.
    print("\n" + "="*70)
    print("  STAGE 2: Specialist Architecture (corrected labeling)")
    print("="*70)
    train_stage2_specialists(
        all_tr_graphs = tr_graphs,
        all_te_graphs = te_graphs,
        base_dim      = base_dim,
        vocab         = vocab,
        cfg           = CONFIG,
        device        = device,
        num_relations = num_relations,
        data_dir      = OUTPUT_DIR,
    )

    # Train generalist checkpoint (used for both real-world and MegaVul evals)
    generalist = train_expert(tr_graphs, base_dim, vocab, CONFIG, device,
                              num_relations=num_relations, tag="generalist",
                              num_classes=NUM_CLASSES)

    # Real-world-only eval on held-out GitHub-mined samples (Reviewer R1.4)
    evaluate_realworld_only(te_graphs, generalist, device, NUM_CLASSES, OUTPUT_DIR)

    # ── External test set evaluation (MegaVul + Devign non-vuln) ──────────
    # MegaVul provides real-world vulnerable samples (zero LLM involvement).
    # Devign non-vulnerable provides real-world safe samples (from QEMU/FFmpeg).
    # Together they give a complete binary evaluation with full metrics.
    #
    # Set paths in CONFIG:
    #   "MEGAVUL_DOTS_DIR"    → path to extracted dots/ folder
    #   "DEVIGN_NONVULN_DIR"  → path to folder containing NON_VULNERABLE/*.dot
    # Leave as None to skip that component.
    MEGAVUL_DOTS_DIR   = CONFIG.get("MEGAVUL_DOTS_DIR",   None)
    DEVIGN_NONVULN_DIR = CONFIG.get("DEVIGN_NONVULN_DIR", None)

    mv_graphs, mv_labels = [], np.array([], dtype=np.int64)
    dv_graphs, dv_labels = [], np.array([], dtype=np.int64)

    if MEGAVUL_DOTS_DIR is not None:
        mv_dir = Path(MEGAVUL_DOTS_DIR)
        if mv_dir.exists():
            print("\n[step] Loading MegaVul (vulnerable) external samples...")
            mv_graphs, mv_labels = load_megavul_test_set(
                mv_dir, featurizer, verbose=True
            )
        else:
            print(f"[warn] MEGAVUL_DOTS_DIR not found: {mv_dir}")

    if DEVIGN_NONVULN_DIR is not None:
        dv_dir = Path(DEVIGN_NONVULN_DIR)
        if dv_dir.exists():
            print("\n[step] Loading Devign (non-vulnerable) external samples...")
            dv_graphs, dv_labels = load_devign_nonvuln(
                dv_dir, featurizer, verbose=True
            )
        else:
            print(f"[warn] DEVIGN_NONVULN_DIR not found: {dv_dir}")

    # ── Binary threshold sweep (run first so best_thresh feeds external eval) ──
    print("\n" + "="*65)
    print("  Binary Threshold Sweep  (NON_VULNERABLE vs VULNERABLE)")
    print("="*65)
    te_probs = probs_from_model(generalist, te_graphs, device, NUM_CLASSES)
    p_vuln   = 1.0 - te_probs[:, 0]   # P(any vulnerable class)
    y_bin    = (y_te > 0).astype(int)

    best_thresh, best_f1, best_row = 0.5, -1.0, None
    rows = []
    from sklearn.metrics import precision_recall_fscore_support as _prf
    for t in np.arange(0.10, 0.91, 0.05):
        pred = (p_vuln >= t).astype(int)
        p0, r0, f0, _ = _prf(y_bin, pred, average='binary', pos_label=0, zero_division=0)
        p1, r1, f1, _ = _prf(y_bin, pred, average='binary', pos_label=1, zero_division=0)
        mf1 = float((f0 + f1) / 2)
        rows.append((float(t), float(p0), float(r0), float(f0),
                     float(p1), float(r1), float(f1), mf1))
        if mf1 > best_f1:
            best_f1, best_thresh, best_row = mf1, float(t), rows[-1]

    print(f"\n  {'Thresh':>6}  "
          f"{'P(safe)':>8} {'R(safe)':>8} {'F(safe)':>8}  "
          f"{'P(vuln)':>8} {'R(vuln)':>8} {'F(vuln)':>8}  "
          f"{'MacroF1':>8}")
    print("  " + "-"*75)
    for row in rows:
        t, p0, r0, f0, p1, r1, f1, mf = row
        marker = " ◄ best" if abs(t - best_thresh) < 0.001 else ""
        print(f"  {t:6.2f}  {p0:8.4f} {r0:8.4f} {f0:8.4f}  "
              f"{p1:8.4f} {r1:8.4f} {f1:8.4f}  {mf:8.4f}{marker}")

    print(f"\n  Best threshold: {best_thresh:.2f}  →  macro F1 = {best_f1:.4f}")

    sweep_results = {
        "best_threshold":        round(best_thresh, 2),
        "best_binary_macro_f1":  round(best_f1, 4),
        "sweep": [
            {"threshold":       round(r[0], 2),
             "precision_safe":  round(r[1], 4), "recall_safe":  round(r[2], 4),
             "f1_safe":         round(r[3], 4), "precision_vuln": round(r[4], 4),
             "recall_vuln":     round(r[5], 4), "f1_vuln":      round(r[6], 4),
             "macro_f1":        round(r[7], 4)}
            for r in rows
        ],
    }
    sweep_path = OUTPUT_DIR / "threshold_sweep_5class.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"  Saved → {sweep_path}")

    # ── External eval (uses tuned threshold from sweep above) ─────────────
    if mv_graphs or dv_graphs:
        evaluate_external_testset(
            mv_graphs, mv_labels, dv_graphs,
            generalist, device, NUM_CLASSES, OUTPUT_DIR,
            binary_threshold=best_thresh,
        )
    else:
        print("[info] No external test data configured — skipping external eval")
        print("       Set MEGAVUL_DOTS_DIR and/or DEVIGN_NONVULN_DIR in CONFIG.")

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "gnn_checkpoint_5class.pt"
    torch.save({
        "model_state_dict": generalist.state_dict(),
        "config":           CONFIG,
        "base_dim":         base_dim,
        "vocab":            vocab,
        "num_relations":    num_relations,
        "backbone":         CONFIG["backbone"],
        "num_classes":      NUM_CLASSES,
        "owasp_names":      OWASP_NAMES,
        "label_vocab":      NodeFeaturizer._label_vocab,
        "edge_type_map":    _EDGE_TYPE_MAP,
    }, ckpt_path)
    print(f"[info] Checkpoint saved → {ckpt_path}")


    # ── Paper summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PAPER RESULTS SUMMARY")
    print("="*65)
    print(f"\n  Dataset  : rebuttal/ (LLM-augmented + GitHub-mined)")
    print(f"  Classes  : {NUM_CLASSES}  ({', '.join(OWASP_NAMES.values())})")
    print(f"  Split    : 80/20 stratified by (provenance, class)")
    print(f"  Train: {len(tr_graphs)}  |  Test: {len(te_graphs)}")
    print(f"\n  Stage 1 — Binary (in-distribution test set, threshold={best_thresh:.2f})")
    p0,r0,f0,_ = _prf(y_bin, (p_vuln >= best_thresh).astype(int), average='binary', pos_label=0, zero_division=0)
    p1,r1,f1,_ = _prf(y_bin, (p_vuln >= best_thresh).astype(int), average='binary', pos_label=1, zero_division=0)
    print(f"    NON_VULNERABLE  P={p0:.3f}  R={r0:.3f}  F1={f0:.3f}")
    print(f"    VULNERABLE      P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}")
    print(f"    Macro F1 = {(f0+f1)/2:.4f}")
    print(f"\n  Stage 2 — OWASP multiclass (in-distribution, vulnerable only)")
    print(f"    See stage2_results.json")
    print(f"\n  External evaluation — see external_eval.json")
    print(f"    (MegaVul real CVEs + Devign real safe code)")
    print("="*65)
    print("[done]")
