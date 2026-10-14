#!/usr/bin/env python3
"""
train_token.py — LineVul-style token baseline: a transformer over raw function
text, with no graph at all.

WHY THIS IS THE COMPARISON THAT MATTERS
---------------------------------------
The `nograph` control in train_cwe.py removes edges but still reads code
node-by-node and pools the per-node embeddings. A token model reads the whole
function as ONE sequence with full self-attention, which is strictly more
powerful, and it is the published alternative to graph-based detection rather
than an ablation of our own pipeline.

So this answers a different question. `nograph` asks "do the edges help?".
This asks "is the graph pipeline worth anything at all, against what the field
would otherwise use?"

PROTOCOL
    Identical splits, identical metrics, identical seeds. One model per run
    rather than a 5-fold ensemble -- that is how LineVul is actually trained,
    and it hands the ADVANTAGE to the graph models, which are ensembled. If the
    token model wins anyway, the conclusion is only stronger.

Run:
    python3 train_token.py --mode repo
    python3 train_token.py --mode random --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "runs"

from train_cwe import binary_metrics  # noqa: E402  same metric block


class CodeDS(Dataset):
    def __init__(self, items, tok, max_len):
        self.items, self.tok, self.max_len = items, tok, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        code, y = self.items[i]
        enc = self.tok(code, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                torch.tensor(y, dtype=torch.long))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--task", default="family", choices=["family", "binary"])
    ap.add_argument("--split-suffix", default="")
    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per partition (smoke tests)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    src = {}
    for line in (DATA / "source_cache.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            src[r["sample_id"]] = r["code"]
    corpus = {}
    for line in (DATA / "cwe_corpus.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            corpus[r["sample_id"]] = r
    spec = json.loads(
        (DATA / f"splits_{a.mode}_{a.task}{a.split_suffix}.json").read_text())

    parts = {}
    for name, ids in spec["splits"].items():
        if a.limit:
            ids = ids[:a.limit]
        items = [(src[s], int(corpus[s]["label"])) for s in ids if s in src]
        parts[name] = items
        print(f"  [{name}] {len(items)} with source "
              f"({len(ids)-len(items)} dropped)")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    tr = DataLoader(CodeDS(parts["train"], tok, a.max_len),
                    batch_size=a.batch_size, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)
    te = DataLoader(CodeDS(parts["test"], tok, a.max_len),
                    batch_size=a.batch_size * 2, shuffle=False, num_workers=8)

    # Class weighting, matching the focal/weighted setup the graph models use,
    # so the comparison is not confounded by how imbalance is handled.
    ys = np.array([y for _, y in parts["train"]])
    w = torch.tensor(len(ys) / (2 * np.maximum(np.bincount(ys, minlength=2), 1)),
                     dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)

    scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for k, (ids, mask, y) in enumerate(tr):
            ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(input_ids=ids, attention_mask=mask).logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item()
            if k and k % 500 == 0:
                print(f"  ep{ep+1} step {k}/{len(tr)} loss={tot/k:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[ep {ep+1}] mean loss {tot/max(1,len(tr)):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    model.eval(); probs, ys_te = [], []
    with torch.no_grad():
        for ids, mask, y in te:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids.to(dev),
                           attention_mask=mask.to(dev)).logits
            probs.append(torch.softmax(lg.float(), 1)[:, 1].cpu().numpy())
            ys_te.append(y.numpy())
    p = np.concatenate(probs); yv = np.concatenate(ys_te)

    res = {"token_model": binary_metrics(yv, p, yv.mean()),
           "model": a.model, "epochs": a.epochs, "seed": a.seed,
           "n_test": int(len(yv)),
           "note": "single model, not a 5-fold ensemble; graph baselines are "
                   "ensembled, so this understates the token model"}
    tag = f"{a.mode}-token{a.split_suffix}" + \
          ("" if a.seed == 42 else f"-s{a.seed}") + (f"-{a.tag}" if a.tag else "")
    (OUT / f"{tag}_stage1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["token_model"], indent=2))
    print(f"[done] -> {OUT / f'{tag}_stage1.json'}")


if __name__ == "__main__":
    main()
