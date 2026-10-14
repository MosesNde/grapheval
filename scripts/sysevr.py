"""
sysevr.py
=========
Standalone SySeVR (Li et al., IEEE TDSC 2021) binary baseline.

"SySeVR: A Framework for Using Deep Learning to Detect Software
Vulnerabilities"

SySeVR's original pipeline:
  1. Extract Syntax-based Vulnerability Candidates (SyVCs) — specific
     syntax patterns: library/API function calls, array usage, pointer
     usage, arithmetic expressions.
  2. For each SyVC, compute a PROGRAM SLICE — the set of statements
     connected to the candidate via data and control dependence (forward
     and backward).
  3. Each slice is tokenized into a sequence and vectorized.
  4. A BiLSTM (or BiGRU) classifies each slice as vulnerable or not.

THIS IMPLEMENTATION — CPG-based slice approximation, built ENTIRELY from
this codebase's existing CPGs in DATA_DIR (the same files training.py
already loads — e.g. /home/moses/rebuttal/A0X_.../**/export.dot):

  - REACHING_DEF edges -> data dependency (Joern's equivalent of SySeVR's
                          "Data Dependency" relation)
  - CDG edges          -> control dependency (Joern's Control Dependence
                          Graph, equivalent to SySeVR's "Control
                          Dependency" relation)

  For each syntax candidate (CALL node matching one of the four SyVC
  patterns), we collect all CPG nodes reachable via REACHING_DEF/CDG edges
  (both directions), then reconstruct ordered, deduplicated slice TEXT
  directly from each node's own CODE attribute (no external source-file
  matching required -- every CPG node already carries the source text it
  represents). Nodes are ordered by (LINE_NUMBER, COLUMN_NUMBER, node_id)
  to approximate natural reading order. Large container nodes (BLOCK/
  METHOD nodes whose CODE attribute contains an entire nested sub-tree as
  one string) are excluded in favor of their atomic child nodes, to avoid
  massive duplicated text.

  NOTE: this codebase's CPGs may have multi-line CODE attribute values
  (e.g. a BLOCK node's CODE spans an entire compound statement across many
  lines). The dot parser below tracks quote-parity across lines to handle
  this correctly -- a known limitation in training.py's own line-by-line
  Format 1/2 parser, which truncates such nodes. This script reimplements
  parsing locally with that fix; training.py itself is NOT modified.

  This is a faithful reproduction of SySeVR's slicing CONCEPT using this
  codebase's existing dependency-graph infrastructure, not a verbatim
  re-implementation of SySeVR's original Checkmarx-based tool. This
  caveat should be stated in the paper alongside the result.

Like Devign and ReVeal, SySeVR was designed and validated ONLY for binary
vulnerable/not-vulnerable classification at the FUNCTION level (a function
is labeled vulnerable if ANY of its slices is flagged) -- no published
multiclass variant. Framed as a related-work baseline, not a proposed
contribution of this paper.

THIS SCRIPT DOES NOT MODIFY training.py, devign.py, reveal.py, OR ANY
EXISTING OUTPUT. It reads CONFIG from training.py (read-only) and writes
its own results to NEW filenames only:
    <DATA_DIR>/sysevr_baseline_results.json
    <DATA_DIR>/sysevr_baseline_checkpoint.pt
    <DATA_DIR>/sysevr_vocab.json
    <DATA_DIR>/sysevr_slice_cache/   (cached extracted slices, new directory)

Usage:
    python sysevr.py
"""

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import training as tv  # noqa: E402

CONFIG      = tv.CONFIG
DATA_DIR    = Path(CONFIG["DATA_DIR"])
SLICE_CACHE = DATA_DIR / "sysevr_slice_cache"

SYSEVR_CONFIG = {
    "max_seq_len":         500,
    "embed_dim":           128,
    "lstm_hidden":         128,
    "lstm_layers":         2,
    "mlp_hidden":          128,
    "dropout":             0.3,
    "lr":                  1e-3,
    "weight_decay":        1e-5,
    "batch_size":          64,
    "epochs":              100,
    "val_frac":            0.1,
    "early_stop_patience": 15,
    "focal_gamma":         2.0,
    "seed":                CONFIG.get("seed", 42),
    "min_vocab_freq":      2,
    "max_slices_per_file": 8,
}


# -----------------------------------------------------------------------------
# MULTI-LINE-AWARE DOT PARSER
#
# training.py's parse_dot() processes Format 1/2 line-by-line, which
# truncates nodes whose CODE attribute spans multiple lines (common for
# BLOCK/METHOD nodes containing compound statements). This parser tracks
# quote-parity across accumulated lines to correctly handle multi-line
# quoted attribute values before attempting to match a complete node/edge
# record. Verified against a real sample (export.dot) to recover ALL nodes
# (no edges referencing undefined node IDs), vs. under-recovery with the
# naive line-by-line approach.
# -----------------------------------------------------------------------------
_NODE_RE = re.compile(r'^"?(\d+)"?\s*\[(.*)\]\s*;?\s*$', re.DOTALL)
_EDGE_RE = re.compile(r'^"?(\d+)"?\s*->\s*"?(\d+)"?\s*\[(.*)\]\s*;?\s*$', re.DOTALL)


def _attr_val(attrs: str, key: str) -> str:
    m = re.search(rf'{key}\s*=\s*"((?:[^"\\]|\\.)*)"', attrs, re.DOTALL)
    if not m:
        return ""
    return (m.group(1).replace('\\"', '"')
                       .replace("\\n", "\n")
                       .replace("\\t", "\t"))


def _count_unescaped_quotes(s: str) -> int:
    count = 0
    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i - 1] != '\\'):
            count += 1
    return count


def parse_dot_for_slicing(dot_path: Path) -> Tuple[Dict[int, dict], List[Tuple[int, str, int]]]:
    """
    Returns nodes: {id: {label, code, method_full_name, line, col}}
            edges: [(src, edge_type, dst)]
    Handles multi-line quoted attribute values correctly (see module
    docstring). Only supports Format 1/2 (unquoted/quoted GitHub-style)
    dot files, which is what this codebase's training CPGs use.
    """
    try:
        text = dot_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}, []

    nodes: Dict[int, dict] = {}
    edges: List[Tuple[int, str, int]] = []
    buf = ""

    for raw_line in text.splitlines():
        buf = buf + "\n" + raw_line if buf else raw_line

        if _count_unescaped_quotes(buf) % 2 == 1:
            continue

        stripped = buf.strip()
        if not stripped or stripped.startswith("digraph") or stripped == "}":
            buf = ""
            continue

        if re.search(r'\]\s*;?\s*$', stripped):
            line = stripped
            bracket_pos = line.find('[')
            prefix = line[:bracket_pos] if bracket_pos >= 0 else line

            if "->" in prefix:
                m = _EDGE_RE.match(line)
                if m:
                    src, dst, attrs = int(m.group(1)), int(m.group(2)), m.group(3)
                    et = _attr_val(attrs, "label") or "UNKNOWN"
                    edges.append((src, et, dst))
            else:
                m = _NODE_RE.match(line)
                if m:
                    node_id, attrs = int(m.group(1)), m.group(2)
                    line_raw = _attr_val(attrs, "LINE_NUMBER")
                    col_raw  = _attr_val(attrs, "COLUMN_NUMBER")
                    nodes[node_id] = {
                        "label":             _attr_val(attrs, "label"),
                        "code":              _attr_val(attrs, "CODE"),
                        "method_full_name":  _attr_val(attrs, "METHOD_FULL_NAME"),
                        "line": int(line_raw) if line_raw.isdigit() else -1,
                        "col":  int(col_raw)  if col_raw.isdigit()  else -1,
                    }
            buf = ""

    return nodes, edges


# -----------------------------------------------------------------------------
# Syntax-based Vulnerability Candidate (SyVC) detection
# -----------------------------------------------------------------------------
_ARRAY_OPS   = {"<operator>.indirectIndexAccess"}
_POINTER_OPS = {"<operator>.indirection", "<operator>.addressOf",
                "<operator>.indirectFieldAccess", "<operator>.indirectMemberAccess"}
_ARITH_OPS   = {"<operator>.assignment", "<operator>.addition", "<operator>.subtraction",
                "<operator>.multiplication", "<operator>.division", "<operator>.modulo",
                "<operator>.preIncrement", "<operator>.preDecrement",
                "<operator>.postIncrement", "<operator>.postDecrement",
                "<operator>.assignmentPlus", "<operator>.assignmentMinus",
                "<operator>.assignmentMultiplication", "<operator>.assignmentDivision"}

def is_syntax_candidate(node: dict) -> bool:
    if node.get("label") != "CALL":
        return False
    full_name = node.get("method_full_name", "")
    if full_name in _ARRAY_OPS or full_name in _POINTER_OPS or full_name in _ARITH_OPS:
        return True
    if full_name and not full_name.startswith("<operator>"):
        return True
    return False


# -----------------------------------------------------------------------------
# Slice extraction (CODE-attribute based -- no external source file needed)
# -----------------------------------------------------------------------------
_CONTAINER_LABELS  = {"BLOCK", "METHOD"}
_CONTAINER_MAX_LEN = 200

def extract_slices(dot_path: Path, max_slices_per_file: int = 8) -> List[str]:
    nodes, edges = parse_dot_for_slicing(dot_path)
    if not nodes:
        return []

    adj: Dict[int, List[int]] = {}
    for src, et, dst in edges:
        if et in ("REACHING_DEF", "CDG"):
            if src in nodes and dst in nodes:
                adj.setdefault(src, []).append(dst)
                adj.setdefault(dst, []).append(src)

    candidates = [nid for nid, n in nodes.items() if is_syntax_candidate(n)]

    def nodes_to_slice_text(node_ids) -> str:
        slice_nodes = []
        for nid in node_ids:
            n = nodes.get(nid)
            if not n or not n["code"] or n["code"] == "<empty>" or n["line"] <= 0:
                continue
            if n["label"] in _CONTAINER_LABELS and len(n["code"]) > _CONTAINER_MAX_LEN:
                continue
            slice_nodes.append((nid, n))
        slice_nodes.sort(key=lambda x: (x[1]["line"], x[1]["col"], x[0]))
        seen, lines = set(), []
        for nid, n in slice_nodes:
            key = (n["line"], n["code"])
            if key not in seen:
                seen.add(key)
                lines.append(n["code"])
        return "\n".join(lines).strip()

    if not candidates:
        whole = nodes_to_slice_text(nodes.keys())
        return [whole] if whole else []

    slices = []
    seen_slice_texts = set()
    for cand_id in candidates[:max_slices_per_file]:
        visited = {cand_id}
        queue = [cand_id]
        while queue:
            cur = queue.pop()
            for nbr in adj.get(cur, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        text = nodes_to_slice_text(visited)
        # Dedup: tightly-coupled small functions often produce near-identical
        # dependency closures for multiple distinct candidates (e.g. several
        # candidates all pulling in nearly the whole function body). Keeping
        # only unique slice texts avoids wasting training signal on exact
        # duplicates within the same file.
        if text and text not in seen_slice_texts:
            slices.append(text)
            seen_slice_texts.add(text)

    if not slices:
        whole = nodes_to_slice_text(nodes.keys())
        return [whole] if whole else []

    return slices


# -----------------------------------------------------------------------------
# File enumeration -- reuses training.py's exact folder-walking convention
# -----------------------------------------------------------------------------
def enumerate_dot_files(data_dir: Path) -> List[Tuple[Path, int]]:
    """Returns [(dot_path, binary_label)] -- label 1=vulnerable, 0=safe."""
    out = []
    for owasp_dir in sorted(data_dir.iterdir()):
        if not owasp_dir.is_dir():
            continue
        owasp_class = None
        for prefix, idx in tv.FOLDER_PREFIX_TO_CLASS.items():
            if owasp_dir.name.startswith(prefix):
                owasp_class = idx
                break
        if owasp_class is None:
            continue
        for dot_file in owasp_dir.rglob("*.dot"):
            out.append((dot_file, 1))

    non_vuln_dir = data_dir / "Non-vulnerable"
    if non_vuln_dir.exists():
        for dot_file in non_vuln_dir.rglob("*.dot"):
            out.append((dot_file, 0))
    else:
        print(f"[warn] Non-vulnerable directory not found at {non_vuln_dir}")

    return out


# -----------------------------------------------------------------------------
# Tokenization + vocabulary
# -----------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+\.?[0-9]*|[^\sA-Za-z0-9_]")

def tokenize(code: str) -> List[str]:
    return _TOKEN_RE.findall(code)


def build_vocab(token_lists: List[List[str]], min_freq: int) -> Dict[str, int]:
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counter.most_common():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------------------------
# BiLSTM model
# -----------------------------------------------------------------------------
class SySeVRBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 mlp_hidden, num_classes, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, encoded, lengths, labels):
        self.encoded = encoded
        self.lengths = lengths
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.encoded[idx], dtype=torch.long),
                torch.tensor(self.lengths[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_sysevr(X_tr, len_tr, y_tr, X_val, len_val, y_val,
                 vocab_size, cfg, device, tag="sysevr"):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    model = SySeVRBiLSTM(
        vocab_size, cfg["embed_dim"], cfg["lstm_hidden"], cfg["lstm_layers"],
        cfg["mlp_hidden"], num_classes=2, dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)

    counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    w      = counts.sum() / (2 * np.maximum(counts, 1.0))
    w_ten  = torch.tensor(w, dtype=torch.float32).to(device)
    crit   = tv.FocalLoss(weight=w_ten, gamma=cfg["focal_gamma"])

    tr_ds  = SliceDataset(X_tr, len_tr, y_tr)
    val_ds = SliceDataset(X_val, len_val, y_val)

    samp_w  = 1.0 / np.maximum(counts[y_tr], 1.0)
    sampler = WeightedRandomSampler(torch.tensor(samp_w, dtype=torch.float),
                                    num_samples=len(tr_ds), replacement=True)
    tr_ldr  = DataLoader(tr_ds, batch_size=cfg["batch_size"], sampler=sampler)
    val_ldr = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    best_metric, best_state, patience = -1.0, None, 0
    PATIENCE = cfg["early_stop_patience"]

    for epoch in range(cfg["epochs"]):
        model.train()
        ep_loss = 0.0
        for xb, lb, yb in tr_ldr:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            loss = crit(model(xb, lb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item()

        model.eval()
        la, ll = [], []
        with torch.no_grad():
            for xb, lb, yb in val_ldr:
                xb, lb = xb.to(device), lb.to(device)
                la.append(model(xb, lb).detach().cpu())
                ll.append(yb)

        metric = 0.0
        if la:
            preds = torch.cat(la).argmax(dim=1).numpy()
            lbls  = torch.cat(ll).numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = f1_score(lbls, preds, average="macro", zero_division=0)

        sched.step(metric)

        if metric > best_metric:
            best_metric, best_state, patience = metric, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  [{tag}] early stop ep {epoch+1} | best_macro_f1={best_metric:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] ep {epoch+1:4d} | loss={ep_loss:.4f} | macro_f1={metric:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_sysevr(model, X, lengths, device, batch_size=128):
    model.eval()
    ds  = SliceDataset(X, lengths, np.zeros(len(X), dtype=np.int64))
    ldr = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for xb, lb, _ in ldr:
        xb, lb = xb.to(device), lb.to(device)
        out.append(F.softmax(model(xb, lb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    tv.set_seed(SYSEVR_CONFIG["seed"])
    device = tv.get_device()
    print(f"[info] device={device}")
    print(f"[info] SySeVR baseline -- binary-only, CPG-based slice approximation")
    print(f"[info] DATA_DIR={DATA_DIR}")
    print(f"[info] Built entirely from existing CPGs (no raw .c file matching)")

    out_results_path = DATA_DIR / "sysevr_baseline_results.json"
    out_ckpt_path    = DATA_DIR / "sysevr_baseline_checkpoint.pt"
    out_vocab_path   = DATA_DIR / "sysevr_vocab.json"
    if out_results_path.exists() or out_ckpt_path.exists():
        print(f"\n[error] Output file(s) already exist -- refusing to overwrite:")
        if out_results_path.exists(): print(f"        {out_results_path}")
        if out_ckpt_path.exists():    print(f"        {out_ckpt_path}")
        raise SystemExit(1)

    SLICE_CACHE.mkdir(parents=True, exist_ok=True)

    print("\n[step] Enumerating .dot files...")
    dot_files = enumerate_dot_files(DATA_DIR)
    print(f"[info] Found {len(dot_files)} .dot files "
          f"({sum(1 for _, y in dot_files if y == 1)} vulnerable, "
          f"{sum(1 for _, y in dot_files if y == 0)} safe)")
    if not dot_files:
        print("[error] No .dot files found under DATA_DIR -- check CONFIG['DATA_DIR'].")
        raise SystemExit(1)

    print(f"\n[step] Extracting slices from {len(dot_files)} CPGs "
          f"(cached in {SLICE_CACHE})...")
    import hashlib
    all_slice_texts: List[str] = []
    all_slice_labels: List[int] = []
    n_cache_hit, n_cache_miss, n_empty = 0, 0, 0

    for i, (dot_path, y) in enumerate(dot_files):
        cache_key  = hashlib.sha256(str(dot_path).encode()).hexdigest()[:24]
        cache_file = SLICE_CACHE / f"{cache_key}.json"
        if cache_file.exists():
            try:
                slices = json.loads(cache_file.read_text())
                n_cache_hit += 1
            except Exception:
                slices = extract_slices(dot_path, SYSEVR_CONFIG["max_slices_per_file"])
                cache_file.write_text(json.dumps(slices))
                n_cache_miss += 1
        else:
            slices = extract_slices(dot_path, SYSEVR_CONFIG["max_slices_per_file"])
            cache_file.write_text(json.dumps(slices))
            n_cache_miss += 1

        if not slices:
            n_empty += 1
        for s in slices:
            all_slice_texts.append(s)
            all_slice_labels.append(y)

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(dot_files)}] processed | "
                  f"{len(all_slice_texts)} slices so far | "
                  f"cache hit={n_cache_hit} miss={n_cache_miss} | empty={n_empty}")

    print(f"[info] Total slices extracted: {len(all_slice_texts)} "
          f"(from {len(dot_files)} files, {n_empty} produced zero slices)")
    print(f"[info] Slice label distribution: {dict(Counter(all_slice_labels))}")

    if len(all_slice_texts) < 50:
        print("[error] Too few slices extracted to train meaningfully. "
              "Check that CDG/REACHING_DEF edges exist in your .dot files.")
        raise SystemExit(1)

    print("\n[step] Tokenizing slices and building vocabulary...")
    all_tokens = [tokenize(s) for s in all_slice_texts]
    vocab = build_vocab(all_tokens, SYSEVR_CONFIG["min_vocab_freq"])
    print(f"[info] Vocabulary size: {len(vocab)}")
    with open(out_vocab_path, "w") as f:
        json.dump(vocab, f)

    max_len = SYSEVR_CONFIG["max_seq_len"]
    encoded = [encode(toks, vocab, max_len) for toks in all_tokens]
    lengths = [min(len(toks), max_len) if len(toks) > 0 else 1 for toks in all_tokens]
    labels  = np.array(all_slice_labels, dtype=np.int64)

    X = np.array(encoded, dtype=np.int64)
    L = np.array(lengths, dtype=np.int64)

    print("\n[step] Splitting train/val/test...")
    idx_all = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=0.2, random_state=SYSEVR_CONFIG["seed"], stratify=labels,
    )
    y_tr_full = labels[tr_idx]
    tr_idx2, val_idx = train_test_split(
        tr_idx, test_size=SYSEVR_CONFIG["val_frac"],
        random_state=SYSEVR_CONFIG["seed"], stratify=y_tr_full,
    )

    X_tr, L_tr, y_tr     = X[tr_idx2], L[tr_idx2], labels[tr_idx2]
    X_val, L_val, y_val  = X[val_idx], L[val_idx], labels[val_idx]
    X_te, L_te, y_te     = X[te_idx], L[te_idx], labels[te_idx]
    print(f"[info] train={len(X_tr)} | val={len(X_val)} | test={len(X_te)}")

    print("\n" + "="*70)
    print("  SYSEVR BASELINE (Li et al. 2021) -- binary-only")
    print("  CPG-based slice approximation (REACHING_DEF + CDG dependency")
    print("  closure from syntax candidates), BiLSTM classifier")
    print("  Related-work baseline (not a proposed contribution of this paper)")
    print("="*70)

    model = train_sysevr(
        X_tr, L_tr, y_tr, X_val, L_val, y_val,
        vocab_size=len(vocab), cfg=SYSEVR_CONFIG, device=device, tag="sysevr",
    )

    probs  = predict_sysevr(model, X_te, L_te, device)
    y_pred = probs.argmax(axis=1)

    print("\n--- SySeVR Baseline -- Slice-Level Binary Classification Report ---")
    print(classification_report(y_te, y_pred,
                                target_names=["NON_VULNERABLE", "VULNERABLE"],
                                digits=4, zero_division=0))
    macro_f1_slice = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"  SySeVR Macro F1 (slice-level): {macro_f1_slice:.4f}")

    print("\n[step] Computing function-level aggregation...")
    full_probs = predict_sysevr(model, X, L, device)
    full_preds = full_probs.argmax(axis=1)

    offset = 0
    func_y_true, func_y_pred = [], []
    for dot_path, y_file in dot_files:
        cache_key  = hashlib.sha256(str(dot_path).encode()).hexdigest()[:24]
        cache_file = SLICE_CACHE / f"{cache_key}.json"
        n_slices = len(json.loads(cache_file.read_text())) if cache_file.exists() else 0
        if n_slices > 0:
            file_preds = full_preds[offset:offset + n_slices]
            func_y_true.append(y_file)
            func_y_pred.append(int(file_preds.max()))
        offset += n_slices

    func_y_true = np.array(func_y_true)
    func_y_pred = np.array(func_y_pred)
    print(f"[info] Function-level samples: {len(func_y_true)}")
    print("\n--- SySeVR Baseline -- Function-Level Binary Classification Report ---")
    print(classification_report(func_y_true, func_y_pred,
                                target_names=["NON_VULNERABLE", "VULNERABLE"],
                                digits=4, zero_division=0))
    macro_f1_func = f1_score(func_y_true, func_y_pred, average="macro", zero_division=0)
    print(f"  SySeVR Macro F1 (function-level, ANY-slice aggregation): {macro_f1_func:.4f}")
    print(f"  (compare: Stage 1 3-layer GAT ~ 0.65, Devign GGRN+Conv = 0.7101,")
    print(f"            6-layer GAT = 0.6769, ReVeal GGNN+MLP = 0.6991)")

    with open(out_results_path, "w") as f:
        json.dump({
            "macro_f1_slice_level": float(macro_f1_slice),
            "macro_f1_function_level": float(macro_f1_func),
            "n_slices_total": len(all_slice_texts),
            "n_slices_test": len(X_te),
            "n_functions_total": len(dot_files),
            "n_functions_with_slices": len(func_y_true),
            "vocab_size": len(vocab),
            "note": ("Binary-only related-work baseline. Slices approximated "
                     "via REACHING_DEF (data dependency) + CDG (control "
                     "dependency) closure from this codebase's existing Joern "
                     "CPGs around detected syntax candidates (library/API "
                     "calls, array usage, pointer usage, arithmetic "
                     "expressions). Slice text reconstructed directly from "
                     "each CPG node's own CODE attribute, ordered by "
                     "(LINE_NUMBER, COLUMN_NUMBER) -- NOT a verbatim "
                     "reproduction of SySeVR's original Checkmarx-based "
                     "slicer. Function-level label = ANY constituent slice "
                     "predicted vulnerable, matching SySeVR's original "
                     "aggregation scheme."),
        }, f, indent=2)
    print(f"\n  Results -> {out_results_path}")

    torch.save({"model_state_dict": model.state_dict(),
               "vocab_size": len(vocab), "config": SYSEVR_CONFIG}, out_ckpt_path)
    print(f"  Checkpoint -> {out_ckpt_path}")
    print("="*70)
    print("[done]")
