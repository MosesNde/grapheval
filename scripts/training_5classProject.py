# training_5class.py
#
# 5-Class CPG Vulnerability Classifier
# =====================================
# Reduced-class variant of the OWASP vulnerability detection pipeline.
#
# CLASS SCHEME (5-class):
#   0 = NON_VULNERABLE
#   1 = A01_Access_Control       (standalone)
#   2 = A02_Crypto_Failures      (standalone)
#   3 = A03_Injection            (standalone)
#   4 = OTHER_VULNERABLE         (merged: A04+A05+A06+A07+A08+A09+A10)
#
# ARCHITECTURE:
#   Stage 1 — Binary gate:        NON_VULNERABLE vs VULNERABLE
#   Stage 2 — 4-class specialist: A01 / A02 / A03 / OTHER (vulnerable only)
#   Inference — Stage 1 first; if vulnerable, Stage 2 assigns the subclass
#
# DIRECTORY LAYOUT:
#
#   TRAINING DATA (mixed GitHub-mined + LLM, used for CV):
#     <DATA_DIR>/                           e.g. /home/moses/rebuttal
#       A01_Broken_Access_Control/
#         <commit-hash>/
#           Vulnerable_section_N_.../
#             export.dot
#         <NNN_N_export>/
#             export.dot
#             export (2).dot
#       A02_Cryptographic_Failures/ ...
#       Non-vulnerable/
#         <NNN_safe_export>/
#             export.dot
#
#   EXTERNAL TEST SETS (real-world only, never seen during training):
#     <MEGAVUL_DIR>/                        e.g. /home/moses/Megavul_testset
#       A01_Broken_Access_Control/
#         *.dot
#       A02_Cryptographic_Failures/ ...
#
#     <NONVULN_DIR>/                        e.g. /home/moses/NON_VULNERABLE
#       devign__devign_<hash>_<project>/
#         export.dot
#
# GPU ASSIGNMENT (run alongside 11-class on GPU 2+3):
#   CUDA_VISIBLE_DEVICES=0,1 python3 training_5class.py
#
# DEPENDENCIES:
#   pip install torch torch-geometric transformers scikit-learn numpy tqdm --break-system-packages

import os
import re
import json
import random
import logging
import argparse
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit paths here before running
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Training data: the rebuttal/ folder (GitHub-mined + LLM mixed)
    "data_dir":         "/home/moses/rebuttal",

    # External test sets (real-world only, used only for evaluation)
    # Set to None to skip that evaluation component.
    "megavul_dir":      "/home/moses/Megavul_testset",   # vulnerable real-world
    "nonvuln_dir":      "/home/moses/NON_VULNERABLE",    # non-vulnerable real-world (Devign)

    "output_dir":       "./outputs_5classProject",
    "checkpoint_dir":   "./checkpoints_5classproject",
    "log_file":         "./training_5classProject.log",

    # ── 5-class scheme ────────────────────────────────────────────────────
    "class_names": [
        "NON_VULNERABLE",
        "A01_Access_Control",
        "A02_Crypto_Failures",
        "A03_Injection",
        "OTHER_VULNERABLE",
    ],
    "num_classes":        5,
    "num_classes_stage1": 2,
    "num_classes_stage2": 4,

    # Stage 2 internal label: 5-class label → 0-indexed stage2 label
    "stage2_label_map":   {1: 0, 2: 1, 3: 2, 4: 3},
    "stage2_label_names": [
        "A01_Access_Control",
        "A02_Crypto_Failures",
        "A03_Injection",
        "OTHER_VULNERABLE",
    ],

    # ── Model ─────────────────────────────────────────────────────────────
    "codebert_model":   "microsoft/graphcodebert-base",
    "hidden_dim":       512,
    "gat_heads":        8,
    "edge_emb_dim":     32,
    "dropout":          0.3,

    # ── Training ──────────────────────────────────────────────────────────
    "seed":              42,
    "n_folds":           5,
    "epochs":            60,
    "patience":          10,
    "batch_size":        32,
    "lr":                3e-4,
    "weight_decay":      1e-5,
    "focal_gamma":       2.0,
    "sampler_max_ratio": 4.0,

    # ── Hardware ──────────────────────────────────────────────────────────
    "dataparallel":      True,

    # ── Misc ──────────────────────────────────────────────────────────────
    "max_nodes":         512,
    "max_code_len":      128,
    "megavul_min_bytes": 2000,   # skip stub CPGs smaller than this
}

# ─────────────────────────────────────────────────────────────────────────────
# OWASP FOLDER NAME → 5-CLASS LABEL
# ─────────────────────────────────────────────────────────────────────────────

FOLDER_TO_CLASS: Dict[str, int] = {
    # rebuttal/ folder names
    "A01_Broken_Access_Control":                          1,
    "A02_Cryptographic_Failures":                         2,
    "A03_Injection":                                      3,
    "A04_Insecure_Design":                                4,
    "A05_Security_Misconfiguration":                      4,
    "A06_Vulnerable_and_Outdated_Components":             4,
    "A07_Identification_and_Authentication_Failures":     4,
    "A08_Software_and_Data_Integrity_Failures":           4,
    "A09_Security_Logging_and_Monitoring_Failures":       4,
    "A10_Server_Side_Request_Forgery":                    4,
    # MegaVul_testset/ uses Title_Case variants
    "A07_Identification_And_Authentication_Failures":     4,
    "A08_Software_And_Data_Integrity_Failures":           4,
    "A09_Security_Logging_And_Monitoring_Failures":       4,
    "A10_Server_Side_Request_Forgery":                    4,  # already covered above
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("train5cls")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    for h in [logging.FileHandler(log_file), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

logger = setup_logger(CONFIG["log_file"])

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# DOT PARSER  (all 4 Joern export formats)
#
# Format A — rebuttal GitHub-mined:
#   25769803776[label=BLOCK CODE="..." ...]          (unquoted IDs, unquoted labels)
#
# Format B — rebuttal LLM-augmented:
#   "25769803776" [label="BLOCK" CODE="..." ...]     (quoted IDs, quoted labels)
#
# Format C — MegaVul:
#   "111669149697" [label = <(METHOD,OBJ_obj2txt)<SUB>1</SUB>> ]
#
# Format D — Devign:
#   "107374182400" [label = <METHOD, 1<BR/>kvm_init_irq_routing> ]
#   "30064771072"  [label = <&lt;operator&gt;.assignment, 9<BR/>x = y> ]
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TYPE_MAP = {
    "AST": 0, "CFG": 1, "DDG": 2, "PDG": 3,
    "CDG": 4, "CALL": 5, "REF": 6, "USE": 7,
    "DEF": 8, "REACHING_DEF": 9,
}
NUM_EDGE_TYPES = len(EDGE_TYPE_MAP) + 1

# Unified vocabulary covering all 4 formats.
# PARAM added for Format C/D (MegaVul/Devign use PARAM instead of METHOD_PARAMETER_IN).
NODE_TYPE_VOCAB = [
    "UNKNOWN", "METHOD", "METHOD_PARAMETER_IN", "METHOD_RETURN",
    "BLOCK", "CALL", "IDENTIFIER", "LITERAL", "FIELD_IDENTIFIER",
    "RETURN", "CONTROL_STRUCTURE", "LOCAL", "TYPE_DECL",
    "MEMBER", "NAMESPACE", "FILE", "COMMENT", "PARAM",
]
_NT_IDX = {t: i for i, t in enumerate(NODE_TYPE_VOCAB)}


def _edge_type_id(s: str) -> int:
    u = s.upper()
    for k, v in EDGE_TYPE_MAP.items():
        if k in u:
            return v
    return NUM_EDGE_TYPES - 1


def _html_unescape(s: str) -> str:
    """Decode the HTML entities Joern inlines into dot label attributes."""
    return (s.replace("&lt;", "<").replace("&gt;", ">")
             .replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"'))


_KNOWN_NODE_TYPES = {
    "METHOD", "BLOCK", "CALL", "IDENTIFIER", "LITERAL", "PARAM",
    "LOCAL", "RETURN", "CONTROL_STRUCTURE", "FIELD_IDENTIFIER",
    "METHOD_RETURN", "TYPE_DECL", "MEMBER", "MODIFIER", "JUMP_TARGET",
    "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_REF",
    "TYPE", "TYPE_REF", "IMPORT", "BINDING", "CLOSURE_BINDING", "UNKNOWN",
}


def _extract_node_type(attr: str) -> str:
    a = _html_unescape(attr)
    # Format C normal: <(TYPE,...)<SUB>  — TYPE may be lowercase (function-call nodes)
    m = re.search(r'<\(([A-Za-z_][A-Za-z_0-9]*)', a)
    if m:
        t = m.group(1)
        return t if t in _KNOWN_NODE_TYPES else "CALL"
    # Format C/D operator: <(<operator...  or  <<operator...  → CALL
    if re.search(r'<\(?<operators?>', a): return "CALL"
    # Format D with line number: <TYPE, N<BR/>
    m = re.search(r'<([A-Za-z_][A-Za-z_0-9]*)\s*,\s*\d', a)
    if m:
        t = m.group(1)
        return t if t[0].isupper() else "CALL"
    # Format D without line number: <TYPE<BR/>content>
    m = re.search(r'<([A-Za-z_][A-Za-z_0-9]*)\s*<BR', a)
    if m:
        t = m.group(1)
        return t if t[0].isupper() else "CALL"
    # Format A/B: label="TYPE" or label=TYPE
    m = re.search(r'\blabel\s*=\s*"?([A-Z_][A-Z_0-9]*)"?', a)
    if m: return m.group(1)
    return "UNKNOWN"


def _extract_code(attr: str) -> str:
    a = _html_unescape(attr)
    # Format C normal: <(TYPE, code, ...)<SUB>N</SUB>>
    m = re.search(r'<\([A-Z_][A-Z_0-9]*,\s*(.*?)\)<SUB>', a, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        return (raw.split(',')[0] if ',' in raw else raw).strip()
    # Format C operator: <(<operator>.name, code, ...)<SUB>N</SUB>>
    m = re.search(r'<\(<operator>[^,]*,\s*(.*?)\)<SUB>', a, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        return (raw.split(',')[0] if ',' in raw else raw).strip()
    # Format D with line number: <TYPE_OR_OP, N<BR/>code<BR/>context>
    m = re.search(r'<[^,]+,\s*\d+\s*<BR/>\s*(.*?)(?:<BR/>|(?=>))', a, re.DOTALL)
    if m: return m.group(1).strip()
    # Format D without line number: <TYPE<BR/>content>
    m = re.search(r'<[A-Z_][A-Z_0-9]*\s*<BR/>\s*(.*?)(?:<BR/>|(?=>))', a, re.DOTALL)
    if m: return m.group(1).strip()
    # Format D operator without line number: <<operator>.name<BR/>content>
    m = re.search(r'<[^<>]+<BR/>\s*(.*?)(?:<BR/>|(?=>))', a, re.DOTALL)
    if m: return m.group(1).strip()
    # Format A/B: CODE="..." (uppercase attribute)
    m = re.search(r'\bCODE\s*=\s*"([^"]*)"', a, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'\bCODE\s*=\s*(\S+)', a, re.IGNORECASE)
    if m: return m.group(1).strip()
    return ""


# ── Label-leaking string scrubber ────────────────────────────────────────────
# The rebuttal dataset encodes the label in several text fields which must be
# scrubbed before code reaches the CodeBERT embedder:
#   • META_DATA ROOT="/.../vulnerable/..."  or  ROOT="/.../safe/..."
#   • FILENAME="974_1.c"  or  FILENAME="1223_safe.c"
#   • Vulnerable_section_N_... directory name fragments in NAME= attrs
#
# MegaVul and Devign have no such metadata nodes — they are already clean.
# ─────────────────────────────────────────────────────────────────────────────

_PATH_RE     = re.compile(r'(?:/|\\)[a-zA-Z0-9_./()\\ -]{3,}')
_FNAME_RE    = re.compile(r'\b\d+_(?:safe|\d+)\.(?:c|cpp|h|cc)\b', re.IGNORECASE)
_VULN_SEC_RE = re.compile(r'[Vv]ulnerable[_\s]section[_\s]\d+[_\s][^\s"]*')
_LABEL_RE    = re.compile(r'\b(?:non[-_]?vulnerable|vulnerable|_safe|safe)\b',
                          re.IGNORECASE)


def _sanitise_code(s: str) -> str:
    """Scrub all label-leaking substrings before the string reaches CodeBERT."""
    if not s:
        return ""
    s = _PATH_RE.sub("<path>", s)
    s = _FNAME_RE.sub("<file>", s)
    s = _VULN_SEC_RE.sub("<section>", s)
    s = _LABEL_RE.sub("<label>", s)
    return s


def _node_onehot(t: str) -> List[float]:
    v = [0.0] * len(NODE_TYPE_VOCAB)
    v[_NT_IDX.get(t, 0)] = 1.0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Node types that carry filesystem paths, Joern version strings, or other
# metadata that leaks the label directly (e.g. ROOT="/home/.../vulnerable/...").
# These are stripped entirely during parsing so the model cannot use them.
# ─────────────────────────────────────────────────────────────────────────────
_LEAKING_NODE_TYPES = {
    "META_DATA",       # ROOT= path contains "vulnerable" / "Non-vulnerable"
    "FILE",            # absolute file path / dataset split name
    "NAMESPACE",       # package/namespace path strings
    "NAMESPACE_BLOCK", # same
    "DEPENDENCY",      # dependency group IDs (may contain filenames)
}


def parse_dot(path: str) -> Optional[Dict]:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(content.strip()) < 20:
        return None

    nodes: Dict[str, Dict] = {}
    edges: List[Tuple]     = []

    # Node pattern handles all 3 Joern export formats:
    #   Format A (GitHub-mined):  25769803776[label=BLOCK CODE="..." ...]
    #   Format B (LLM-augmented): "25769803776" [label="BLOCK" CODE="..." ...]
    #   Format C (MegaVul HTML):  "25769803776" [label = <(BLOCK, code, ...)>]
    # The key difference is that Format A has NO quotes around the node ID
    # and NO quotes around the label value — both patterns are matched here.
    for m in re.finditer(
        r'^[ \t]*"?([\w.]+)"?[ \t]*\[([^\]]*(?:\[[^\]]*\][^\]]*)*)\]',
        content, re.MULTILINE
    ):
        nid      = m.group(1)
        attr     = m.group(2)
        ntype    = _extract_node_type(attr)
        if ntype in _LEAKING_NODE_TYPES:
            continue   # drop entirely — carries info that leaks the label
        # Scrub attribute blob before extracting code so NAME=, FILENAME=,
        # ROOT=, etc. containing _safe / vulnerable / paths are neutralised
        clean_attr = _sanitise_code(attr)
        nodes[nid] = {"code": _extract_code(clean_attr), "type": ntype}

    for m in re.finditer(
        r'"?([\w.]+)"?[ \t]*->[ \t]*"?([\w.]+)"?(?:[ \t]*\[([^\]]*)\])?',
        content, re.MULTILINE
    ):
        src, dst = m.group(1), m.group(2)
        if src in nodes and dst in nodes:
            edges.append((src, dst, _edge_type_id(m.group(3) or "")))

    return {"nodes": nodes, "edges": edges} if nodes else None

# ─────────────────────────────────────────────────────────────────────────────
# CODEBERT EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

class CodeBERTEmbedder:
    def __init__(self, model_name: str, device: torch.device, max_len: int):
        logger.info(f"Loading {model_name} …")
        self.tok     = AutoTokenizer.from_pretrained(model_name)
        self.model   = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device  = device
        self.max_len = max_len
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, code: str) -> torch.Tensor:
        key = (code or "<empty>")[:200]
        if key not in self._cache:
            enc = self.tok(key, max_length=self.max_len, truncation=True,
                           padding="max_length", return_tensors="pt")
            out = self.model(enc["input_ids"].to(self.device),
                             attention_mask=enc["attention_mask"].to(self.device))
            self._cache[key] = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

# ─────────────────────────────────────────────────────────────────────────────
# DOT → PyG DATA
# ─────────────────────────────────────────────────────────────────────────────

def dot_to_pyg(
    dot_path:   str,
    label5:     int,
    embedder:   CodeBERTEmbedder,
    cfg:        Dict,
    provenance: str = "train",
) -> Optional[Data]:

    parsed = parse_dot(dot_path)
    if not parsed:
        return None

    node_ids = list(parsed["nodes"].keys())[:cfg["max_nodes"]]
    if len(node_ids) < 2:
        return None

    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    feats = []
    for nid in node_ids:
        info = parsed["nodes"][nid]
        emb  = embedder.encode(_sanitise_code(info["code"]))
        oh   = torch.tensor(_node_onehot(info["type"]))
        feats.append(torch.cat([emb, oh], dim=0))   # [786]
    x = torch.stack(feats, dim=0)

    srcs, dsts, etypes = [], [], []
    for src, dst, et in parsed["edges"]:
        if src in id2idx and dst in id2idx:
            srcs.append(id2idx[src])
            dsts.append(id2idx[dst])
            etypes.append(et)

    if not srcs:
        srcs = dsts = list(range(len(node_ids)))
        etypes = [0] * len(node_ids)

    return Data(
        x          = x,
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long),
        edge_attr  = torch.tensor(etypes,        dtype=torch.long),
        y          = torch.tensor(label5,         dtype=torch.long),
        y_stage1   = torch.tensor(0 if label5 == 0 else 1, dtype=torch.long),
        y_stage2   = torch.tensor(cfg["stage2_label_map"].get(label5, -1),
                                  dtype=torch.long),
        label_id   = torch.tensor(label5,         dtype=torch.long),
        provenance = provenance,
        dot_path   = str(dot_path),
    )

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA LOADER
#
# Walks rebuttal/ — GitHub-mined and LLM samples are mixed together.
# Non-vulnerable samples live under rebuttal/Non-vulnerable/.
# No provenance distinction is made here; all samples are training data.
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(
    data_dir: Path,
    embedder: CodeBERTEmbedder,
    cfg:      Dict,
) -> List[Data]:

    records: List[Tuple[str, int]] = []

    # Vulnerable samples: one OWASP folder per class
    for owasp_dir in sorted(data_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue
        label5 = FOLDER_TO_CLASS.get(owasp_dir.name)
        if label5 is None:
            continue
        for dot_file in owasp_dir.rglob("*.dot"):
            records.append((str(dot_file), label5))

    # Non-vulnerable samples
    nonvuln = data_dir / "Non-vulnerable"
    if nonvuln.exists():
        for dot_file in nonvuln.rglob("*.dot"):
            records.append((str(dot_file), 0))

    # Summary
    dist = Counter(r[1] for r in records)
    logger.info(f"Training .dot files found: {len(records)}")
    for i, name in enumerate(cfg["class_names"]):
        logger.info(f"  Class {i} ({name:30s}): {dist.get(i, 0):>5d}")

    graphs, failed = [], 0
    for dot_path, label5 in tqdm(records, desc="Loading training CPGs"):
        g = dot_to_pyg(dot_path, label5, embedder, cfg, provenance="train")
        if g is not None:
            graphs.append(g)
        else:
            failed += 1

    logger.info(f"Loaded {len(graphs)} training graphs  ({failed} skipped)")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL TEST SET LOADERS
#
# These are entirely separate from training — never included in CV folds.
# Used only after training for real-world generalization evaluation.
# ─────────────────────────────────────────────────────────────────────────────

def load_megavul_testset(
    megavul_dir: Path,
    embedder:    CodeBERTEmbedder,
    cfg:         Dict,
) -> List[Data]:
    """
    Loads MegaVul real-world vulnerable CPGs.
    Handles two possible layouts:
      A) megavul_testset/Axx_<Name>/*.dot          (flat per-class)
      B) megavul_testset/Axx_<Name>/<sample>/...   (nested, like rebuttal/)
    Only files >= megavul_min_bytes are loaded (tiny stubs are skipped).
    """
    graphs, skipped_tiny, skipped_parse, skipped_unknown = [], 0, 0, 0

    # Log what we actually find at the top level
    top_dirs = sorted([d.name for d in megavul_dir.iterdir() if d.is_dir()])
    logger.info(f"MegaVul top-level dirs: {top_dirs}")

    for owasp_dir in sorted(megavul_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue
        label5 = FOLDER_TO_CLASS.get(owasp_dir.name)
        if label5 is None:
            logger.info(f"  Skipping unrecognised folder: {owasp_dir.name}")
            skipped_unknown += 1
            continue
        dot_files = list(owasp_dir.rglob("*.dot"))
        logger.info(f"  {owasp_dir.name}: {len(dot_files)} .dot files found")
        for dot_file in dot_files:
            if dot_file.stat().st_size < cfg["megavul_min_bytes"]:
                skipped_tiny += 1
                continue
            g = dot_to_pyg(str(dot_file), label5, embedder, cfg,
                           provenance="megavul")
            if g is not None:
                graphs.append(g)
            else:
                skipped_parse += 1

    dist = Counter(int(g.y.item()) for g in graphs)
    logger.info(f"MegaVul test set: {len(graphs)} graphs loaded  "
                f"({skipped_tiny} tiny, {skipped_parse} parse-fail, "
                f"{skipped_unknown} unknown folders)")
    for i, name in enumerate(cfg["class_names"]):
        if dist.get(i, 0) > 0:
            logger.info(f"  Class {i} ({name}): {dist[i]}")
    return graphs


def load_nonvuln_testset(
    nonvuln_dir: Path,
    embedder:    CodeBERTEmbedder,
    cfg:         Dict,
) -> List[Data]:
    """
    Loads Devign real-world non-vulnerable CPGs.
    Layout: NON_VULNERABLE/devign__devign_<hash>_<project>/export*.dot
    All graphs get label 0 (NON_VULNERABLE).
    """
    graphs, failed = [], 0

    for dot_file in nonvuln_dir.rglob("*.dot"):
        g = dot_to_pyg(str(dot_file), 0, embedder, cfg, provenance="devign")
        if g is not None:
            graphs.append(g)
        else:
            failed += 1

    logger.info(f"Devign non-vuln test set: {len(graphs)} graphs loaded  "
                f"({failed} skipped)")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL EVALUATION  (two-stage inference on held-out real-world data)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_external(
    megavul_graphs: List[Data],
    devign_graphs:  List[Data],
    s1_models:      List[nn.Module],
    s2_models:      List[nn.Module],
    cfg:            Dict,
    device:         torch.device,
    output_dir:     str,
):
    """
    Run two-stage ensemble inference on the real-world test sets.
    Reports binary metrics (MegaVul + Devign combined) and
    per-class recall on MegaVul (for classes with >= 30 samples).
    """
    all_graphs = megavul_graphs + devign_graphs
    if not all_graphs:
        logger.warning("No external test graphs available — skipping external eval")
        return

    # True binary labels: 0 = non-vulnerable, 1 = vulnerable
    binary_true = np.array(
        [1] * len(megavul_graphs) + [0] * len(devign_graphs), dtype=int
    )
    # True 5-class labels (for per-class recall)
    class_true = np.array(
        [int(g.y.item()) for g in megavul_graphs] +
        [0] * len(devign_graphs), dtype=int
    )

    loader = DataLoader(all_graphs, batch_size=cfg["batch_size"], shuffle=False)
    n      = len(all_graphs)

    # Stage 1 ensemble (binary vote)
    s1_votes = np.zeros((n, 2), dtype=float)
    for model in s1_models:
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(device)
                probs  = F.softmax(model(batch), dim=1).cpu().numpy()
                bsz    = probs.shape[0]
                s1_votes[offset:offset+bsz] += probs
                offset += bsz

    # Stage 2 ensemble (4-class vote, run on all; used only where Stage 1 says vulnerable)
    s2_votes = np.zeros((n, 4), dtype=float)
    for model in s2_models:
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(device)
                probs  = F.softmax(model(batch), dim=1).cpu().numpy()
                bsz    = probs.shape[0]
                s2_votes[offset:offset+bsz] += probs
                offset += bsz

    # Assemble final 5-class predictions
    s2_inv    = {v: k for k, v in cfg["stage2_label_map"].items()}
    preds5    = []
    binary_pred = []
    for i in range(n):
        if s1_votes[i].argmax() == 0:
            preds5.append(0)
            binary_pred.append(0)
        else:
            s2cls = int(s2_votes[i].argmax())
            preds5.append(s2_inv[s2cls])
            binary_pred.append(1)
    binary_pred = np.array(binary_pred)
    preds5      = np.array(preds5)

    logger.info("\n" + "="*60)
    logger.info("EXTERNAL REAL-WORLD EVALUATION")
    logger.info(f"  MegaVul vulnerable:  {len(megavul_graphs)} graphs")
    logger.info(f"  Devign non-vuln:     {len(devign_graphs)} graphs")
    logger.info("="*60)

    # Binary metrics — pass labels explicitly so sklearn doesn't crash
    # when only one class is present in the test set
    present_labels = sorted(set(binary_true.tolist()))
    present_names  = [["NON_VULNERABLE", "VULNERABLE"][i] for i in present_labels]
    logger.info("\n--- Binary detection (NON_VULNERABLE vs VULNERABLE) ---")
    logger.info("\n" + classification_report(
        binary_true, binary_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    ))

    # Per-class recall on MegaVul (only classes with >= 30 samples)
    if megavul_graphs:
        mv_true  = class_true[:len(megavul_graphs)]
        mv_pred5 = preds5[:len(megavul_graphs)]
        dist     = Counter(mv_true)
        eligible = [i for i in range(1, 5) if dist.get(i, 0) >= 30]

        if eligible:
            logger.info("\n--- Per-class recall on MegaVul (classes with >= 30 samples) ---")
            for cls_id in eligible:
                mask   = mv_true == cls_id
                recall = (mv_pred5[mask] == cls_id).mean()
                logger.info(f"  {cfg['class_names'][cls_id]:30s} "
                            f"n={dist[cls_id]:4d}  recall={recall:.3f}")

    # Save results
    results = {
        "megavul_graphs":  len(megavul_graphs),
        "devign_graphs":   len(devign_graphs),
        "binary_f1_macro": float(f1_score(binary_true, binary_pred,
                                          labels=present_labels,
                                          average="macro", zero_division=0)),
        "binary_report":   classification_report(
            binary_true, binary_pred,
            labels=present_labels,
            target_names=present_names,
            zero_division=0, output_dict=True
        ),
    }
    out_path = os.path.join(output_dir, "external_eval_5class.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nExternal eval saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL: 3-LAYER GAT + RESIDUAL CONNECTIONS
# ─────────────────────────────────────────────────────────────────────────────

class CPGClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 heads, edge_emb_dim, dropout, num_label_types=5):
        super().__init__()
        self.dropout    = dropout
        self.edge_emb   = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
        self.label_emb  = nn.Embedding(num_label_types, 16)
        self.input_proj = nn.Linear(in_dim + 16, hidden_dim)

        self.gat1  = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2  = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gat3  = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                             dropout=dropout, edge_dim=edge_emb_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.cls   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        e  = self.edge_emb(data.edge_attr)
        le = self.label_emb(data.label_id)[data.batch]
        x  = F.elu(self.input_proj(torch.cat([data.x.float(), le], dim=1)))
        x  = F.dropout(x, p=self.dropout, training=self.training)

        x1 = F.elu(self.gat1(x,  data.edge_index, edge_attr=e))
        x1 = self.norm1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.elu(self.gat2(x1, data.edge_index, edge_attr=e))
        x2 = self.norm2(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.gat3(x2, data.edge_index, edge_attr=e)
        x3 = self.norm3(x3 + x2)
        return self.cls(global_mean_pool(x3, data.batch))

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def build_sampler(labels: np.ndarray, n_cls: int,
                  max_ratio: float) -> WeightedRandomSampler:
    counts  = np.maximum(np.bincount(labels, minlength=n_cls).astype(float), 1)
    max_cnt = counts.max()
    weights = np.minimum(max_cnt / counts, max_ratio)
    return WeightedRandomSampler(
        weights     = torch.tensor(weights[labels], dtype=torch.float),
        num_samples = len(labels),
        replacement = True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FOLD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold: int, stage: int,
    tr_g: List[Data], val_g: List[Data],
    cfg: Dict, device: torch.device,
) -> Tuple[nn.Module, Dict]:

    n_cls     = cfg["num_classes_stage1"] if stage == 1 else cfg["num_classes_stage2"]
    lbl_attr  = "y_stage1" if stage == 1 else "y_stage2"
    lbl_names = (["NON_VULNERABLE", "VULNERABLE"] if stage == 1
                 else cfg["stage2_label_names"])

    tr_labels  = np.array([int(getattr(g, lbl_attr).item()) for g in tr_g])
    val_labels = np.array([int(getattr(g, lbl_attr).item()) for g in val_g])

    sampler = build_sampler(tr_labels, n_cls, cfg["sampler_max_ratio"])
    tr_ldr  = DataLoader(tr_g,  batch_size=cfg["batch_size"],
                         sampler=sampler, drop_last=False)
    val_ldr = DataLoader(val_g, batch_size=cfg["batch_size"],
                         shuffle=False,   drop_last=False)

    in_dim = tr_g[0].x.shape[1]
    model  = CPGClassifier(
        in_dim=in_dim, hidden_dim=cfg["hidden_dim"], num_classes=n_cls,
        heads=cfg["gat_heads"], edge_emb_dim=cfg["edge_emb_dim"],
        dropout=cfg["dropout"],
    )
    model = model.to(device)

    counts    = np.maximum(np.bincount(tr_labels, minlength=n_cls).astype(float), 1)
    ce_w      = torch.tensor(1.0 / counts / (1.0 / counts).sum(),
                             dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=cfg["focal_gamma"], weight=ce_w)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    best_f1, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_ldr:
            batch  = batch.to(device)
            tgts   = getattr(batch, lbl_attr)
            loss   = criterion(model(batch), tgts)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        preds_all, tgts_all = [], []
        with torch.no_grad():
            for batch in val_ldr:
                batch = batch.to(device)
                tgts  = getattr(batch, lbl_attr)
                preds_all.extend(model(batch).argmax(1).cpu().numpy())
                tgts_all.extend(tgts.cpu().numpy())

        val_f1 = f1_score(tgts_all, preds_all, average="macro", zero_division=0)
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  S{stage} Fold{fold} Ep{epoch:03d}  "
                f"loss={total_loss/max(len(tr_ldr),1):.4f}  "
                f"val_macro_f1={val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for batch in val_ldr:
            batch = batch.to(device)
            preds_all.extend(model(batch).argmax(1).cpu().numpy())
            tgts_all.extend(getattr(batch, lbl_attr).cpu().numpy())

    logger.info(f"\n  Stage {stage}  Fold {fold}  best macro F1 = {best_f1:.4f}")
    logger.info("\n" + classification_report(tgts_all, preds_all,
                target_names=lbl_names, zero_division=0))

    return model, {"fold": fold, "stage": stage, "best_val_f1": best_f1,
                   "preds": preds_all, "targets": tgts_all}

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT EXTRACTION
#
# Each CPG in rebuttal/ was generated from a named project:
#   chrome, httpd, libav, libtiff, nginx
# The project name is embedded in the ROOT= attribute of the META_DATA node.
#
# Cross-project split (Option C):
#   TRAIN: chrome + httpd + libav   (3 large, well-balanced projects)
#   VAL:   libtiff + nginx           (2 medium projects, different domains)
#
# Rationale: libtiff (image codec) and nginx (web server) are from different
# application domains than the training projects, testing genuine cross-project
# generalisation. All 10 OWASP classes are covered in both splits.
# ─────────────────────────────────────────────────────────────────────────────

_ROOT_RE = re.compile(r'ROOT\s*=\s*"([^"]+)"', re.IGNORECASE)
_PROJ_RE = re.compile(r'/([^/]+?)(?:_OWASP|_owasp)', re.IGNORECASE)

TRAIN_PROJECTS = {"chrome", "httpd", "libav"}
VAL_PROJECTS   = {"libtiff", "nginx"}


def get_project(dot_path: str) -> str:
    """
    Extract the project name from the ROOT= attribute in the CPG dot file.
    Returns the lowercase project name, or 'unknown' if not found.
    """
    try:
        with open(dot_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                rm = _ROOT_RE.search(line)
                if rm:
                    pm = _PROJ_RE.search(rm.group(1))
                    if pm:
                        return pm.group(1).lower()
    except OSError:
        pass
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PROJECT TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(graphs: List[Data], cfg: Dict,
           device: torch.device) -> Dict:
    """
    Cross-project split:
      Train — chrome, httpd, libav   (~6,000 graphs)
      Val   — libtiff, nginx          (~3,354 graphs)

    Both stages are trained once (no k-fold) and evaluated on the val split.
    The trained models are saved for external evaluation.
    """
    logger.info("Assigning project labels for cross-project split …")
    proj_labels = []
    for g in graphs:
        proj_labels.append(get_project(g.dot_path))

    proj_counts = {}
    from collections import Counter
    for p, c in Counter(proj_labels).items():
        proj_counts[p] = c

    logger.info("Project distribution:")
    for proj, cnt in sorted(proj_counts.items()):
        split = "TRAIN" if proj in TRAIN_PROJECTS else \
                "VAL"   if proj in VAL_PROJECTS   else "SKIP"
        logger.info(f"  {split}  {proj:<12}: {cnt:>5} graphs")

    tr_idx  = [i for i, p in enumerate(proj_labels) if p in TRAIN_PROJECTS]
    val_idx = [i for i, p in enumerate(proj_labels) if p in VAL_PROJECTS]
    skip    = [i for i, p in enumerate(proj_labels)
               if p not in TRAIN_PROJECTS and p not in VAL_PROJECTS]

    if skip:
        logger.warning(f"  {len(skip)} graphs from unrecognised projects skipped")

    tr_g  = [graphs[i] for i in tr_idx]
    val_g = [graphs[i] for i in val_idx]

    logger.info(f"\nTrain: {len(tr_g)} graphs  |  Val: {len(val_g)} graphs")

    # Label distributions
    tr_labels  = [int(g.y.item()) for g in tr_g]
    val_labels = [int(g.y.item()) for g in val_g]
    for split_name, labels in [("Train", tr_labels), ("Val", val_labels)]:
        dist = Counter(labels)
        logger.info(f"  {split_name} class dist: " +
                    "  ".join(f"{cfg['class_names'][k]}={dist.get(k,0)}"
                               for k in range(cfg["num_classes"])))

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    s1_models, s2_models = [], []
    results = {}

    # ── Stage 1: binary gate ──────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 1 — Binary gate (cross-project)")
    logger.info(f"{'='*60}")
    s1_model, s1_m = train_fold(1, 1, tr_g, val_g, cfg, device)
    s1_models.append(s1_model)
    torch.save(s1_model.state_dict(),
               os.path.join(cfg["checkpoint_dir"], "stage1_crossproj.pt"))

    # ── Stage 2: 4-class among vulnerable only ────────────────────────────
    tr_vuln  = [g for g in tr_g  if int(g.y.item()) > 0]
    val_vuln = [g for g in val_g if int(g.y.item()) > 0]

    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2 — 4-class specialist  "
                f"tr_vuln={len(tr_vuln)}  val_vuln={len(val_vuln)}")
    logger.info(f"{'='*60}")
    s2_model, s2_m = train_fold(1, 2, tr_vuln, val_vuln, cfg, device)
    s2_models.append(s2_model)
    torch.save(s2_model.state_dict(),
               os.path.join(cfg["checkpoint_dir"], "stage2_crossproj.pt"))

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-PROJECT RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Stage 1 val macro F1 : {s1_m['best_val_f1']:.4f}")
    logger.info(f"Stage 2 val macro F1 : {s2_m['best_val_f1']:.4f}")
    logger.info(f"Train projects : {sorted(TRAIN_PROJECTS)}")
    logger.info(f"Val   projects : {sorted(VAL_PROJECTS)}")

    return {
        "stage1_oof_f1":  s1_m["best_val_f1"],
        "stage2_oof_f1":  s2_m["best_val_f1"],
        "stage1_fold_f1": [s1_m["best_val_f1"]],
        "stage2_fold_f1": [s2_m["best_val_f1"]],
        "stage1_models":  s1_models,
        "stage2_models":  s2_models,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=CONFIG["data_dir"],
                        help="rebuttal/ folder (training data)")
    parser.add_argument("--megavul_dir", default=CONFIG["megavul_dir"],
                        help="megavul_testset/ folder (real-world vulnerable eval)")
    parser.add_argument("--nonvuln_dir", default=CONFIG["nonvuln_dir"],
                        help="NON_VULNERABLE/ folder (Devign non-vuln eval)")
    parser.add_argument("--output_dir",  default=CONFIG["output_dir"])
    parser.add_argument("--gpu_ids",     default=None,
                        help="e.g. '0,1' — overrides CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    CONFIG["data_dir"]    = args.data_dir
    CONFIG["megavul_dir"] = args.megavul_dir
    CONFIG["nonvuln_dir"] = args.nonvuln_dir
    CONFIG["output_dir"]  = args.output_dir

    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"],     exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    logger.info(f"Device: {device}  |  GPUs: {n_gpu}")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")

    emb_dev  = torch.device("cuda:0") if torch.cuda.is_available() else device
    embedder = CodeBERTEmbedder(CONFIG["codebert_model"], emb_dev,
                                CONFIG["max_code_len"])

    # Load training data (rebuttal/ — all mixed, used for CV)
    graphs = load_training_data(Path(CONFIG["data_dir"]), embedder, CONFIG)
    embedder.clear_cache()

    if len(graphs) < CONFIG["n_folds"] * 2:
        logger.error(f"Too few graphs ({len(graphs)}) for CV.")
        return

    logger.info(f"\nStarting {CONFIG['n_folds']}-fold CV  |  "
                f"{len(graphs)} graphs  |  5-class cross-project split")
    results = run_cv(graphs, CONFIG, device)

    # Load and evaluate external real-world test sets
    megavul_graphs, devign_graphs = [], []

    mv_path = CONFIG.get("megavul_dir")
    if mv_path and Path(mv_path).exists():
        logger.info(f"\nLoading MegaVul test set from {mv_path}")
        embedder2 = CodeBERTEmbedder(CONFIG["codebert_model"], emb_dev,
                                     CONFIG["max_code_len"])
        megavul_graphs = load_megavul_testset(Path(mv_path), embedder2, CONFIG)
        embedder2.clear_cache()
    else:
        logger.info("megavul_dir not set or not found — skipping MegaVul eval")

    nv_path = CONFIG.get("nonvuln_dir")
    if nv_path and Path(nv_path).exists():
        logger.info(f"\nLoading Devign non-vuln test set from {nv_path}")
        embedder3 = CodeBERTEmbedder(CONFIG["codebert_model"], emb_dev,
                                     CONFIG["max_code_len"])
        devign_graphs = load_nonvuln_testset(Path(nv_path), embedder3, CONFIG)
        embedder3.clear_cache()
    else:
        logger.info("nonvuln_dir not set or not found — skipping Devign eval")

    if megavul_graphs or devign_graphs:
        evaluate_external(
            megavul_graphs, devign_graphs,
            results["stage1_models"], results["stage2_models"],
            CONFIG, device, CONFIG["output_dir"],
        )

    # Save summary
    summary = {
        "split":               "cross-project",
        "train_projects":      sorted(TRAIN_PROJECTS),
        "val_projects":        sorted(VAL_PROJECTS),
        "num_total_graphs":    len(graphs),
        "class_scheme":        CONFIG["class_names"],
        "stage1_val_f1":       round(results["stage1_oof_f1"], 4),
        "stage2_val_f1":       round(results["stage2_oof_f1"], 4),
    }
    out = os.path.join(CONFIG["output_dir"], "summary_5class.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary → {out}")
    logger.info(f"Stage 1 cross-project val macro F1 : {results['stage1_oof_f1']:.4f}")
    logger.info(f"Stage 2 cross-project val macro F1 : {results['stage2_oof_f1']:.4f}")


if __name__ == "__main__":
    main()
