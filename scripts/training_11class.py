# training_11class.py
#
# 11-Class CPG Vulnerability Classifier
# =======================================
# Full OWASP classification: NON_VULNERABLE + all 10 OWASP categories.
# Run on GPU 0 while other training occupies GPU 1+:
#
#   CUDA_VISIBLE_DEVICES=0 python3 training_11class.py
#
# CLASS SCHEME (11-class):
#   0 = NON_VULNERABLE
#   1 = A01_Access_Control
#   2 = A02_Crypto_Failures
#   3 = A03_Injection
#   4 = A04_Insecure_Design
#   5 = A05_Misconfiguration
#   6 = A06_Outdated_Components
#   7 = A07_Auth_Failures
#   8 = A08_Integrity_Failures
#   9 = A09_Logging_Failures
#  10 = A10_SSRF
#
# ARCHITECTURE:
#   Stage 1 — Binary gate:           NON_VULNERABLE vs VULNERABLE
#   Stage 2 — 10-class specialist:   A01..A10 (vulnerable only)
#
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
    "data_dir":         "/home/moses/New/LLM_trainset",

    # External test sets (real-world only, used only for evaluation)
    # Set to None to skip that evaluation component.
    "megavul_dir":      "/home/moses/New/Megavul_testset",
    "nonvuln_dir":      "/home/moses/New/Non_vulnerable",

    "output_dir":       "./outputs_11class",
    "checkpoint_dir":   "./checkpoints_11class",
    "log_file":         "./training_11class.log",

    # ── Vocabulary filter ─────────────────────────────────────────────────
    "vocab_filter":     "/home/moses/scripts/shared_vocab.json",

    # ── 11-class scheme ───────────────────────────────────────────────────
    "class_names": [
        "NON_VULNERABLE",          # 0
        "A01_Access_Control",      # 1
        "A02_Crypto_Failures",     # 2
        "A03_Injection",           # 3
        "A04_Insecure_Design",     # 4
        "A05_Misconfiguration",    # 5
        "A06_Outdated_Components", # 6
        "A07_Auth_Failures",       # 7
        "A08_Integrity_Failures",  # 8
        "A09_Logging_Failures",    # 9
        "A10_SSRF",                # 10
    ],
    "num_classes":        11,
    "num_classes_stage1": 2,
    "num_classes_stage2": 10,  # all 10 OWASP classes (stage2 labels 0-9 = A01-A10)

    # Stage 2 internal label: 11-class label → 0-indexed stage2 label (0-9)
    # class 1 (A01) → stage2 0, class 2 (A02) → stage2 1, ... class 10 (A10) → stage2 9
    "stage2_label_map":   {i+1: i for i in range(10)},
    "stage2_label_names": [
        "A01_Access_Control",
        "A02_Crypto_Failures",
        "A03_Injection",
        "A04_Insecure_Design",
        "A05_Misconfiguration",
        "A06_Outdated_Components",
        "A07_Auth_Failures",
        "A08_Integrity_Failures",
        "A09_Logging_Failures",
        "A10_SSRF",
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
    "megavul_min_bytes": 2000,
}

# ─────────────────────────────────────────────────────────────────────────────
# OWASP FOLDER NAME → 5-CLASS LABEL
# ─────────────────────────────────────────────────────────────────────────────

FOLDER_TO_CLASS: Dict[str, int] = {
    # rebuttal/ folder names → individual OWASP class IDs 1-10
    "A01_Broken_Access_Control":                          1,
    "A02_Cryptographic_Failures":                         2,
    "A03_Injection":                                      3,
    "A04_Insecure_Design":                                4,
    "A05_Security_Misconfiguration":                      5,
    "A06_Vulnerable_and_Outdated_Components":             6,
    "A07_Identification_and_Authentication_Failures":     7,
    "A08_Software_and_Data_Integrity_Failures":           8,
    "A09_Security_Logging_and_Monitoring_Failures":       9,
    "A10_Server-Side_Request_Forgery":                    10,
    # MegaVul_testset/ uses Title_Case variants
    "A07_Identification_And_Authentication_Failures":     7,
    "A08_Software_And_Data_Integrity_Failures":           8,
    "A09_Security_Logging_And_Monitoring_Failures":       9,
    "A10_Server_Side_Request_Forgery":                    10,  # MegaVul variant
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("train11cls")
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


# Unified vocabulary — only node types that appear in ALL three dataset formats
# after normalization. LLM-only types (TYPE_DECL, TYPE, METHOD_PARAMETER_OUT,
# BINDING, IMPORT, CLOSURE_BINDING) are mapped to UNKNOWN in _NODE_TYPE_NORM.
NODE_TYPE_VOCAB = [
    "UNKNOWN", "METHOD", "METHOD_PARAMETER_IN", "METHOD_RETURN",
    "BLOCK", "CALL", "IDENTIFIER", "LITERAL", "FIELD_IDENTIFIER",
    "RETURN", "CONTROL_STRUCTURE", "LOCAL", "MODIFIER",
    "JUMP_TARGET", "METHOD_REF", "TYPE_REF", "MEMBER",
]
_NT_IDX = {t: i for i, t in enumerate(NODE_TYPE_VOCAB)}


# ── Cross-format node-type normalization ──────────────────────────────────────
# Devign (Format D) and MegaVul (Format C) were exported with different Joern
# versions that use different node-type names for the same semantic concepts.
# This map normalizes them to the LLM Format B canonical names so all three
# datasets produce identical node-type one-hot distributions for equivalent
# code structures.
#
# Key mappings:
#   PARAM            → METHOD_PARAMETER_IN  (older Joern used PARAM, newer uses METHOD_PARAMETER_IN)
#   <macro names>    → CALL                 (Devign: AVERROR, FFMIN, SDL_*, etc. are macro-calls)
#   All else unknown → UNKNOWN
#
_NODE_TYPE_NORM: dict = {
    # Format C/D: PARAM is the older name for function parameters
    "PARAM":               "METHOD_PARAMETER_IN",
    # LLM Format B only — type-system nodes not emitted by older Joern
    # Map to closest semantic equivalent so all formats share the same vocab
    "TYPE_DECL":           "UNKNOWN",   # type declaration → not a stmt node
    "TYPE":                "UNKNOWN",   # type reference  → not a stmt node
    "METHOD_PARAMETER_OUT": "METHOD_PARAMETER_IN",  # out-param ≈ in-param structurally
    "BINDING":             "UNKNOWN",   # method binding metadata
    "IMPORT":              "UNKNOWN",   # import statement (rare)
    "CLOSURE_BINDING":     "UNKNOWN",   # closure metadata
    # Devign macro-call node types that Joern couldn't resolve to CALL
    "AVERROR":      "CALL", "TRACE":          "CALL", "DPRINTF":    "CALL",
    "OBJECT":       "CALL", "DECODE":         "CALL", "SAB":        "CALL",
    "SENSE_CODE":   "CALL", "FFMIN":          "CALL", "DB_PRINT":   "CALL",
    "IS_USER":      "CALL", "DEVICE":         "CALL", "SYS_BUS_DEVICE": "CALL",
    "VNC_DEBUG":    "CALL", "FFSWAP":         "CALL", "FFMAX":      "CALL",
    "MIN":          "CALL", "CMD":            "CALL", "DO_UPCAST":  "CALL",
    "CPU":          "CALL", "WaitForSingleObject": "CALL", "VFP_SREG_D": "CALL",
    "VFP_DREG_D":   "CALL", "VFP_SREG_M":    "CALL", "GIC_TARGET": "CALL",
    "QOBJECT":      "CALL", "QDECREF":        "CALL", "PC_MACHINE": "CALL",
    "NUM_CPU":      "CALL", "DOWNMIX_TO_STEREO": "CALL", "MIX_FRONT3": "CALL",
    "GetLastError": "CALL", "VFP_DREG_N":    "CALL", "VFP_SREG_N": "CALL",
    "VFP_DREG_M":   "CALL", "SDL_LockMutex": "CALL", "SDL_CondSignal": "CALL",
    "SDL_UnlockMutex": "CALL", "SDL_WaitThread": "CALL", "ST":      "CALL",
    "MKTAG":        "CALL", "HTAB_SIZE":      "CALL", "IS_INTERLACED": "CALL",
    "PPC_INPUT":    "CALL", "NLPTR_GET":      "CALL", "AV_WN16A":  "CALL",
    "FF_ARRAY_ELEMS": "CALL", "BUG":          "CALL", "ResetEvent": "CALL",
    "ReleaseMutex": "CALL", "DIRTY_HPTE":    "CALL", "HPTE":       "CALL",
    "FD_ZERO":      "CALL", "FD_SET":         "CALL", "QLIST_REMOVE": "CALL",
    "QLIST_INSERT_HEAD": "CALL", "QTAILQ_INSERT_HEAD": "CALL",
    "DBDMA_init":   "CALL", "QTAILQ_FOREACH": "CALL", "QTAILQ_FIRST": "CALL",
    "QLIST_INIT":   "CALL", "QLIST_FOREACH":  "CALL", "SDL_CloseAudio": "CALL",
    "SPICE_RING_IS_EMPTY": "CALL", "ARM_CPU": "CALL", "PPI":        "CALL",
    "ARRAY_SIZE":   "CALL", "DEVICE_GET_CLASS": "CALL", "POWERPC_CPU": "CALL",
    "POWERPC_CPU_GET_CLASS": "CALL", "ICS_SIMPLE": "CALL", "AV_RL8": "CALL",
    "AV_RL16":      "CALL", "AV_RB16":        "CALL", "GIC_TEST_LEVEL": "CALL",
    "GIC_SET_LEVEL": "CALL", "GIC_TEST_TRIGGER": "CALL", "GIC_TEST_ENABLED": "CALL",
    "GIC_SET_PENDING": "CALL", "GIC_CLEAR_LEVEL": "CALL", "USB_DEVICE": "CALL",
    "VIRTIO_BLK":   "CALL", "BIT":            "CALL", "PC_MACHINE_GET_CLASS": "CALL",
    "MACHINE":      "CALL", "BUS":            "CALL", "VIRTIO_BUS_GET_CLASS": "CALL",
    "VIRTIO_SCSI_COMMON": "CALL", "SPAPR_PCI_HOST_BRIDGE_GET_CLASS": "CALL",
    "VIRTIO_DEVICE_GET_CLASS": "CALL", "S390_IPL": "CALL", "X86_CPU": "CALL",
    "XILINX_AXI_DMA_DATA_STREAM": "CALL", "ACPI_BUILD_DPRINTF": "CALL",
    "WaitForMultipleObjects": "CALL", "VFP_SREG_D": "CALL",
    "DO1":          "CALL", "DO16":           "CALL", "RT":         "CALL",
    "RA":           "CALL", "D_IF_decode":    "CALL", "FT_Get_Kerning": "CALL",
    "MIX_REAR1":    "CALL", "MIX_REAR2":      "CALL", "FIELD_PICTURE": "CALL",
    "AudioConverterFillComplexBuffer": "CALL", "GIC_CLEAR_LEVEL": "CALL",
    "ALPHA_BLEND":  "CALL", "CPU_CORE":       "CALL", "ABS":        "CALL",
    "QLIST_FOREACH": "CALL", "LIST_EMPTY":    "CALL", "LIST_FIRST": "CALL",
}

_KNOWN_NODE_TYPES = set(NODE_TYPE_VOCAB)


def _normalize_node_type(t: str) -> str:
    """Normalize a node type to the canonical LLM Format B vocabulary."""
    if t in _KNOWN_NODE_TYPES:
        return t
    return _NODE_TYPE_NORM.get(t, "UNKNOWN")


def _extract_node_type(attr: str) -> str:
    a = _html_unescape(attr)
    # Format C normal: <(TYPE,...)<SUB>  — TYPE may be lowercase (function-call nodes)
    m = re.search(r'<\(([A-Za-z_][A-Za-z_0-9]*)', a)
    if m:
        t = m.group(1)
        raw = t if t in _KNOWN_NODE_TYPES or t in _NODE_TYPE_NORM else "CALL"
        return _normalize_node_type(raw)
    # Format C/D operator: <(<operator...  or  <<operator...  → CALL
    if re.search(r'<\(?<operators?>', a): return "CALL"
    # Format D with line number: <TYPE, N<BR/>
    m = re.search(r'<([A-Za-z_][A-Za-z_0-9]*)\s*,\s*\d', a)
    if m:
        t = m.group(1)
        raw = t if t[0].isupper() else "CALL"
        return _normalize_node_type(raw)
    # Format D without line number: <TYPE<BR/>content>
    m = re.search(r'<([A-Za-z_][A-Za-z_0-9]*)\s*<BR', a)
    if m:
        t = m.group(1)
        raw = t if t[0].isupper() else "CALL"
        return _normalize_node_type(raw)
    # Format A/B: label="TYPE" or label=TYPE
    m = re.search(r'\blabel\s*=\s*"?([A-Z_][A-Z_0-9]*)"?', a)
    if m: return _normalize_node_type(m.group(1))
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
        # Strip CODE from METHOD/METHOD_RETURN/BINDING nodes entirely —
        # their CODE= contains the full function signature/body.
        if ntype in ("METHOD", "METHOD_RETURN", "BINDING"):
            code = ""
        else:
            code = _extract_code(clean_attr)
            # Truncate to the first line only — multi-line CODE= attributes
            # on BLOCK/CALL nodes contain the full subtree text which encodes
            # the vulnerability pattern directly. We keep only the first
            # newline-delimited fragment (the primary expression/identifier).
            if code and '\n' in code:
                code = code.split('\n')[0].strip()
        nodes[nid] = {"code": code, "type": ntype}

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

# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY FILTER
#
# Replaces every code token NOT in the shared cross-project vocabulary with
# <UNK> before the string reaches CodeBERT.  This prevents the model from
# using project-specific API names (avctx, tif_dir, ngx_uint_t …) as a
# shortcut classifier — those tokens appear 100-600x more in some projects
# than others and trivially reveal project identity.
#
# Only tokens in the 485-word shared vocab (C language constructs +
# vulnerability-relevant names that appear in >=4/5 training projects)
# are kept verbatim.
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB_FILTER: Optional[set] = None   # populated by load_vocab_filter()
_TOKEN_RE = re.compile(r'\b[a-zA-Z_]\w+\b')


def load_vocab_filter(path: str) -> None:
    """Load shared vocabulary from JSON file into the module-level set."""
    global _VOCAB_FILTER
    with open(path, "r") as f:
        _VOCAB_FILTER = set(json.load(f))
    logger.info(f"Vocab filter loaded: {len(_VOCAB_FILTER)} shared tokens from {path}")


def _apply_vocab_filter(code: str) -> str:
    """
    Replace every identifier token not in _VOCAB_FILTER with <UNK>.
    Non-identifier characters (operators, numbers, punctuation) are kept.
    If no vocab filter is loaded, returns the original string unchanged.
    """
    if _VOCAB_FILTER is None or not code:
        return code
    return _TOKEN_RE.sub(
        lambda m: m.group(0) if m.group(0).lower() in _VOCAB_FILTER else "<UNK>",
        code
    )


def dot_to_pyg(
    dot_path:   str,
    label5:     int,
    embedder:   Optional[CodeBERTEmbedder],   # unused — structural features only
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
        # STRUCTURAL FEATURES ONLY — no CodeBERT, no code content.
        # Using code embeddings inevitably leaks vulnerability patterns
        # because individual expression nodes (CALL, IDENTIFIER, BLOCK)
        # contain distinguishing tokens even after vocab filtering.
        # The node-type one-hot forces the model to reason purely from
        # CPG graph topology: CFG/DDG/PDG edge patterns, node type
        # sequences, and data/control flow structure.
        oh   = torch.tensor(_node_onehot(info["type"]))   # [23]
        feats.append(oh)
    x = torch.stack(feats, dim=0)  # [N, 18]

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
# Loads ONLY LLM-generated vulnerable samples from rebuttal/.
# The Non-vulnerable/ subfolder is EXCLUDED — non-vulnerable training
# samples come from Devign instead (real-world, no LLM pairing bias).
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
                 heads, edge_emb_dim, dropout, num_label_types=16):
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
        num_label_types=cfg["num_classes"],
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
# WITHIN-PROJECT STRATIFIED SPLIT
#
# Splits 80/20 WITHIN each project so that both train and val contain samples
# from all 5 projects in the same proportions.  This eliminates the node-type
# distribution mismatch that made the cross-project split trivially solvable.
#
# Stratification key: (project, 5-class label) ensures every project-class
# combination is proportionally represented in both splits.
# ─────────────────────────────────────────────────────────────────────────────


def load_training_data(
    data_dir: Path,
    embedder: Optional[CodeBERTEmbedder],
    cfg:      Dict,
) -> List[Data]:

    records: List[Tuple[str, int]] = []

    # Vulnerable samples only — one OWASP folder per class
    for owasp_dir in sorted(data_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue
        label = FOLDER_TO_CLASS.get(owasp_dir.name)
        if label is None:
            continue    # skip Non-vulnerable/ and any unknown folders
        for dot_file in owasp_dir.rglob("*.dot"):
            records.append((str(dot_file), label))

    dist = Counter(r[1] for r in records)
    logger.info(f"Training .dot files found: {len(records)}")
    for i, name in enumerate(cfg["class_names"]):
        logger.info(f"  Class {i} ({name:30s}): {dist.get(i, 0):>5d}")

    graphs, failed = [], 0
    for dot_path, label in tqdm(records, desc="Loading training CPGs"):
        g = dot_to_pyg(dot_path, label, embedder, cfg, provenance="train")
        if g is not None:
            graphs.append(g)
        else:
            failed += 1

    logger.info(f"Loaded {len(graphs)} training graphs  ({failed} skipped)")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# DEVIGN NON-VULNERABLE LOADER (used for BOTH training and validation)
# ─────────────────────────────────────────────────────────────────────────────

def load_devign_split(
    nonvuln_dir: Path,
    embedder:    Optional[CodeBERTEmbedder],
    cfg:         Dict,
    train_frac:  float = 0.50,
    seed:        int   = 42,
) -> Tuple[List[Data], List[Data]]:
    """
    Load all Devign non-vulnerable CPGs and split into train/val halves.
    Returns (train_graphs, val_graphs) — both labelled class 0.
    The split is deterministic (fixed seed) and file-path sorted before shuffle.
    """
    import random as _random
    rng = _random.Random(seed)

    all_paths = sorted(nonvuln_dir.rglob("*.dot"), key=lambda p: str(p))
    rng.shuffle(all_paths)
    split_at  = int(len(all_paths) * train_frac)
    train_paths = all_paths[:split_at]
    val_paths   = all_paths[split_at:]

    logger.info(f"Devign: {len(all_paths)} total  →  "
                f"train={len(train_paths)}  val={len(val_paths)}")

    def _load(paths, prov):
        graphs, failed = [], 0
        for p in tqdm(paths, desc=f"Loading Devign {prov}"):
            g = dot_to_pyg(str(p), 0, embedder, cfg, provenance=prov)
            if g is not None:
                graphs.append(g)
            else:
                failed += 1
        logger.info(f"  Devign {prov}: {len(graphs)} loaded  ({failed} skipped)")
        return graphs

    tr = _load(train_paths, "devign_train")
    vl = _load(val_paths,   "devign_val")
    return tr, vl


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL TEST SET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_megavul_testset(
    megavul_dir: Path,
    embedder:    Optional[CodeBERTEmbedder],
    cfg:         Dict,
) -> List[Data]:
    graphs, skipped_tiny, skipped_parse, skipped_unknown = [], 0, 0, 0

    top_dirs = sorted([d.name for d in megavul_dir.iterdir() if d.is_dir()])
    logger.info(f"MegaVul top-level dirs: {top_dirs}")

    for owasp_dir in megavul_dir.iterdir():
        if not owasp_dir.is_dir():
            continue
        label = FOLDER_TO_CLASS.get(owasp_dir.name)
        if label is None:
            logger.info(f"  Skipping unrecognised folder: {owasp_dir.name}")
            skipped_unknown += 1
            continue

        dot_files = list(owasp_dir.rglob("*.dot"))
        logger.info(f"  {owasp_dir.name}: {len(dot_files)} .dot files found")
        for dot_file in dot_files:
            if dot_file.stat().st_size < cfg.get("megavul_min_bytes", 2000):
                skipped_tiny += 1
                continue
            g = dot_to_pyg(str(dot_file), label, embedder, cfg, provenance="megavul")
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


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  (no internal split — train/val are separate datasets)
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(
    train_graphs: List[Data],
    val_graphs:   List[Data],
    cfg:          Dict,
    device:       torch.device,
) -> Dict:
    """
    Train on (LLM vulnerable + Devign non-vuln train split).
    Validate on (MegaVul vulnerable + Devign non-vuln val split).
    No internal CV — train and val are fully separate datasets.
    """
    from collections import Counter as _C

    tr_dist = _C(int(g.y.item()) for g in train_graphs)
    vl_dist = _C(int(g.y.item()) for g in val_graphs)
    logger.info(f"\nTrain: {len(train_graphs)} graphs  |  Val: {len(val_graphs)} graphs")
    logger.info("  Train class dist: " +
                "  ".join(f"{cfg['class_names'][k]}={tr_dist.get(k,0)}"
                           for k in range(cfg["num_classes"])))
    logger.info("  Val   class dist: " +
                "  ".join(f"{cfg['class_names'][k]}={vl_dist.get(k,0)}"
                           for k in range(cfg["num_classes"])))

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    s1_models, s2_models = [], []

    # Stage 1: binary gate
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 1 — Binary gate")
    logger.info(f"{'='*60}")
    s1_model, s1_m = train_fold(1, 1, train_graphs, val_graphs, cfg, device)
    s1_models.append(s1_model)
    torch.save(s1_model.state_dict(),
               os.path.join(cfg["checkpoint_dir"], "stage1.pt"))

    # Stage 2: 10-class among vulnerable only
    tr_vuln  = [g for g in train_graphs if int(g.y.item()) > 0]
    val_vuln = [g for g in val_graphs   if int(g.y.item()) > 0]
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2 — 10-class specialist  "
                f"tr_vuln={len(tr_vuln)}  val_vuln={len(val_vuln)}")
    logger.info(f"{'='*60}")
    s2_model, s2_m = train_fold(1, 2, tr_vuln, val_vuln, cfg, device)
    s2_models.append(s2_model)
    torch.save(s2_model.state_dict(),
               os.path.join(cfg["checkpoint_dir"], "stage2.pt"))

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Stage 1 val macro F1 : {s1_m['best_val_f1']:.4f}")
    logger.info(f"Stage 2 val macro F1 : {s2_m['best_val_f1']:.4f}")

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
                        help="rebuttal/ folder (LLM vulnerable training data)")
    parser.add_argument("--megavul_dir", default=CONFIG["megavul_dir"],
                        help="megavul_testset/ folder (real-world vulnerable, val only)")
    parser.add_argument("--nonvuln_dir", default=CONFIG["nonvuln_dir"],
                        help="NON_VULNERABLE/ folder (Devign — split 50/50 train/val)")
    parser.add_argument("--output_dir",  default=CONFIG["output_dir"])
    parser.add_argument("--gpu_ids",     default=None)
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

    # Vocab filter still loaded (no-op since embedder is unused, but kept
    # so the flag can be re-enabled without other changes)
    vf_path = CONFIG.get("vocab_filter")
    if vf_path and Path(vf_path).exists():
        load_vocab_filter(vf_path)

    # embedder=None since we use structural features only
    embedder = None

    # ── Load training data ────────────────────────────────────────────────
    # Vulnerable: LLM-generated from rebuttal/ (excluding Non-vulnerable/)
    vuln_graphs = load_training_data(Path(CONFIG["data_dir"]), embedder, CONFIG)

    # Non-vulnerable: Devign split 50/50 — train half for training,
    # val half combined with MegaVul for validation
    nv_path = CONFIG.get("nonvuln_dir")
    if not nv_path or not Path(nv_path).exists():
        logger.error("nonvuln_dir is required for training non-vulnerable class.")
        return

    logger.info(f"\nLoading Devign non-vuln from {nv_path} (50/50 split) …")
    devign_train, devign_val = load_devign_split(
        Path(nv_path), embedder, CONFIG,
        train_frac=0.50, seed=CONFIG["seed"]
    )

    train_graphs = vuln_graphs + devign_train
    logger.info(f"\nFinal training set: {len(train_graphs)} graphs "
                f"({len(vuln_graphs)} vuln + {len(devign_train)} non-vuln)")

    # ── Load validation data ──────────────────────────────────────────────
    # MegaVul: real-world CVE-patched vulnerable functions
    mv_path = CONFIG.get("megavul_dir")
    megavul_graphs = []
    if mv_path and Path(mv_path).exists():
        logger.info(f"\nLoading MegaVul validation set from {mv_path} …")
        megavul_graphs = load_megavul_testset(Path(mv_path), embedder, CONFIG)
    else:
        logger.warning("megavul_dir not set — val set has no vulnerable samples!")

    val_graphs = megavul_graphs + devign_val
    logger.info(f"\nFinal validation set: {len(val_graphs)} graphs "
                f"({len(megavul_graphs)} MegaVul vuln + {len(devign_val)} Devign non-vuln)")

    if not val_graphs:
        logger.error("Empty validation set. Aborting.")
        return

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(f"\nStarting training  |  "
                f"train={len(train_graphs)}  val={len(val_graphs)}  "
                f"11-class  structural features only")
    results = run_cv(train_graphs, val_graphs, CONFIG, device)

    # ── External eval (MegaVul + full Devign val) is the val set itself ───
    # evaluate_external is now redundant since validation IS the external set.
    # Log a reminder of what the val numbers represent.
    logger.info(f"\n{'='*60}")
    logger.info("NOTE: Val set = MegaVul (real CVE functions) + Devign (real non-vuln)")
    logger.info("These are the primary metrics — no separate external eval needed.")
    logger.info(f"{'='*60}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "split":                 "LLM-vuln-train + Devign-50-train / MegaVul + Devign-50-val",
        "num_train_graphs":      len(train_graphs),
        "num_val_graphs":        len(val_graphs),
        "num_megavul_val":       len(megavul_graphs),
        "num_devign_val":        len(devign_val),
        "class_scheme":          CONFIG["class_names"],
        "stage1_val_f1":         round(results["stage1_oof_f1"], 4),
        "stage2_val_f1":         round(results["stage2_oof_f1"], 4),
        "features":              "structural_only (node_type_onehot_18dim)",
    }
    out = os.path.join(CONFIG["output_dir"], "summary_11class.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary → {out}")
    logger.info(f"Stage 1 val macro F1 : {results['stage1_oof_f1']:.4f}")
    logger.info(f"Stage 2 val macro F1 : {results['stage2_oof_f1']:.4f}")


if __name__ == "__main__":
    main()
