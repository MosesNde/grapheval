#!/usr/bin/env python3
"""
joint_finetune.py — train the encoder and the graph network together.

Every graph result in this paper consumes frozen GraphCodeBERT features, while
the token baseline it is compared against fine-tunes its encoder end to end.
The +0.150 gap therefore mixes representation with training regime, and the
paper reports that confound as unresolved. This resolves it by training both
parts of the graph pipeline jointly.

The obstacle is cost. A function is one sequence to a token model but 165 node
sequences on average here, so a naive implementation runs roughly 165x the
encoder work per epoch. Two properties of the data remove most of that:

  * node text is short. The median node is 11 tokens and the 99th percentile is
    28, so a 32-token window covers essentially all of it, against the 128 used
    for the frozen features.
  * node text repeats. Only 37.5% of node texts in a sample of 46,554 nodes are
    distinct, the rest being `<empty>`, `case`, `break;` and similar. Encoding
    the unique texts in a batch and scattering the results back cuts the work
    by about 62%.

Together these bring an epoch to a few times the cost of the token baseline
rather than 165 times.

    python3 joint_finetune.py --mode repo --reverse-edges --epochs 6
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader   # PyG's loader re-collates; we supply our own

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
import training as T
from train_cwe import load_split

TEXT_CACHE = Path("/home/moses/node_text_cache")


# ---------------------------------------------------------------------------
class JointModel(nn.Module):
    """GraphCodeBERT producing node features, then the same GAT stack.

    The graph half is structurally identical to the frozen-feature model so the
    comparison isolates the training regime: same depth, width, heads, relation
    embedding, pooling and classifier.
    """

    def __init__(self, vocab, cfg, num_relations, encoder_name, n_unfrozen,
                 freeze_all=False):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(encoder_name)
        h = self.enc.config.hidden_size

        # Freezing the lower layers keeps the backward pass affordable and is
        # the usual compromise when fine-tuning a large encoder on a small task.
        # n_unfrozen=0 leaves the encoder entirely trainable.
        if freeze_all:
            for p_ in self.enc.parameters():
                p_.requires_grad = False
        elif n_unfrozen > 0:
            for p in self.enc.parameters():
                p.requires_grad = False
            for layer in self.enc.encoder.layer[-n_unfrozen:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.gnn = T.build_model("gat", h, vocab, cfg, num_relations, 2)

    def encode_nodes(self, ids, mask, inverse):
        """Encode the unique texts in a batch, then expand back to one row per node."""
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        # mean over real tokens, which matches how the frozen features were built
        m = mask.unsqueeze(-1).float()
        uniq = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return uniq[inverse]

    def forward(self, batch, ids, mask, inverse):
        batch.x = self.encode_nodes(ids, mask, inverse)
        return self.gnn(batch)


# ---------------------------------------------------------------------------
def build_dataset(graphs, sample_ids):
    """Attach each graph's node texts, dropping graphs whose cache is missing."""
    out = []
    for g, sid in zip(graphs, sample_ids):
        f = TEXT_CACHE / f"{sid}.json"
        if not f.exists():
            continue
        texts = json.loads(f.read_text())
        if len(texts) != g.num_nodes:
            continue
        g.texts = texts
        out.append(g)
    return out


def collate(items, tokenizer, max_len, node_cap):
    """Batch graphs and tokenise only the distinct node texts they contain."""
    kept = []
    for g in items:
        if g.num_nodes > node_cap:      # bound memory on the long tail
            keep = torch.arange(node_cap)
            m = (g.edge_index[0] < node_cap) & (g.edge_index[1] < node_cap)
            g = Data(x=torch.zeros(node_cap, 1), edge_index=g.edge_index[:, m],
                     edge_type=g.edge_type[m], label_id=g.label_id[:node_cap],
                     y=g.y, texts=g.texts[:node_cap])
        kept.append(g)
    texts = [t for g in kept for t in g.texts]
    uniq, inverse = np.unique(np.array(texts, dtype=object), return_inverse=True)
    enc = tokenizer(list(uniq), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    b = Batch.from_data_list([Data(x=torch.zeros(g.num_nodes, 1),
                                   edge_index=g.edge_index, edge_type=g.edge_type,
                                   label_id=g.label_id, y=g.y) for g in kept])
    return b, enc["input_ids"], enc["attention_mask"], torch.as_tensor(inverse)


def run_epoch(model, loader, dev, opt=None, sched=None, accum=1):
    train = opt is not None
    model.train(train)
    tot, n, ps, ys = 0.0, 0, [], []
    for step, (b, ids, mask, inv) in enumerate(loader):
        b, ids, mask, inv = b.to(dev), ids.to(dev), mask.to(dev), inv.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b, ids, mask, inv)
            loss = F.cross_entropy(logits, b.y)
        if train:
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched is not None: sched.step()
        tot += float(loss.detach()) * b.num_graphs; n += b.num_graphs
        ps.append(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().numpy())
        ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    return tot / max(n, 1), roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repo")
    ap.add_argument("--reverse-edges", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8,
                    help="gradient accumulation; effective batch = batch-size * accum")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--node-cap", type=int, default=400)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--unfrozen-layers", type=int, default=4,
                    help="top encoder layers left trainable; 0 = all trainable")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--val-source", default="partition",
                    choices=["partition", "internal"],
                    help="'partition' selects the epoch on the held-out val "
                         "repositories; 'internal' carves 10%% off the training "
                         "set as train_expert does, which shares repositories "
                         "with training")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of the training partition to use; 0.8 "
                         "matches what each 5-fold model sees")
    ap.add_argument("--edges", default="all", choices=["all", "none"],
                    help="'none' deletes every edge, leaving a per-node model "
                         "with identical depth, width and parameter count")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="matched control: identical data, loop, schedule and "
                         "model, but the encoder receives no gradients. The "
                         "5-fold runs each train on 80%% of the partition, so "
                         "they are not a like-for-like baseline for a single "
                         "model trained on all of it.")
    a = ap.parse_args()

    cfg = dict(T.CONFIG)
    if a.seed is not None:
        cfg["seed"] = a.seed
    dev = T.get_device(); T.set_seed(cfg["seed"])

    from transformers import AutoTokenizer
    name = cfg["codebert_model"]
    tokenizer = AutoTokenizer.from_pretrained(name)

    # graphs come from the existing cache; only x is replaced at run time
    from train_cwe import encode as encode_graphs
    corpus, fams, splits = load_split(a.mode)
    G, _ = encode_graphs(corpus, splits, fams, cfg, dev,
                         Path("/home/moses/graph_cache_cwe"), limit=a.limit)
    vocab = max(int(g.label_id.max().item()) for gs in G.values() for g in gs) + 2
    ids_by_part = {k: (v[:a.limit] if a.limit else v) for k, v in splits.items()}

    parts = {}
    for part in ("train", "val", "test"):
        gs = G.get(part, [])
        for g in gs:
            g.label_id = g.label_id.clamp_(0, vocab - 1)
            g.edge_type = g.edge_type.clamp_(0, T.MAX_RELATIONS - 1)
            g.y = torch.tensor(int(g.y.item() > 0))
            if a.edges == "none":
                g.edge_index = torch.zeros((2, 0), dtype=torch.long)
                g.edge_type = torch.zeros((0,), dtype=torch.long)
            elif a.reverse_edges and g.edge_index.numel():
                g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], 1)
                g.edge_type = torch.cat([g.edge_type, g.edge_type + 4])
        parts[part] = build_dataset(gs, [g.sample_id for g in gs])
        print(f"  [{part}] {len(parts[part])} graphs with node text", flush=True)

    # isolate the two procedural differences against the 5-fold pipeline
    rng = np.random.default_rng(cfg["seed"])
    if a.val_source == "internal":
        tr = parts["train"]; idx = rng.permutation(len(tr))
        k = int(0.1 * len(tr))
        parts["val"] = [tr[i] for i in idx[:k]]
        parts["train"] = [tr[i] for i in idx[k:]]
        print(f"  [val-source] internal: {len(parts['val'])} held out from "
              f"training, sharing repositories with it", flush=True)
    if a.train_frac < 1.0:
        tr = parts["train"]; idx = rng.permutation(len(tr))
        keep = int(a.train_frac * len(tr))
        parts["train"] = [tr[i] for i in idx[:keep]]
        print(f"  [train-frac] {a.train_frac}: {len(parts['train'])} graphs", flush=True)

    coll = lambda items: collate(items, tokenizer, a.max_len, a.node_cap)
    loaders = {k: DataLoader(v, batch_size=a.batch_size, shuffle=(k == "train"),
                             collate_fn=coll, num_workers=4)
               for k, v in parts.items() if v}

    model = JointModel(vocab, cfg, T.MAX_RELATIONS, name, a.unfrozen_layers,
                       freeze_all=a.freeze_encoder).to(dev)
    enc_p = [p for p in model.enc.parameters() if p.requires_grad]
    if not enc_p:            # frozen control: AdamW rejects an empty group
        enc_p = [torch.zeros(1, requires_grad=True, device=dev)]
    head_p = list(model.gnn.parameters())
    print(f"[model] encoder trainable {sum(p.numel() for p in enc_p)/1e6:.1f}M, "
          f"graph {sum(p.numel() for p in head_p)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.encoder_lr},
                             {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
    steps = (len(loaders["train"]) // a.accum) * a.epochs
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_frac * steps), steps)
    print(f"[sched] effective batch {a.batch_size * a.accum}, "
          f"{steps} optimiser steps, {int(a.warmup_frac*steps)} warmup", flush=True)

    best, best_state, hist = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, loaders["train"], dev, opt, sched, a.accum)
        with torch.no_grad():
            va_loss, va_auc = run_epoch(model, loaders.get("val") or loaders["test"], dev)
        hist.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                     "train_auc": round(tr_auc, 4), "val_auc": round(va_auc, 4)})
        print(f"  ep {ep}/{a.epochs}  loss {tr_loss:.4f}  train {tr_auc:.4f}  "
              f"val {va_auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if va_auc > best:
            best = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, te_auc = run_epoch(model, loaders["test"], dev)
    print(f"\njoint fine-tuning, {a.mode}: test ROC-AUC {te_auc:.4f}")

    out = {"mode": a.mode, "reverse_edges": a.reverse_edges, "seed": cfg["seed"],
           "unfrozen_layers": a.unfrozen_layers, "max_len": a.max_len,
           "encoder_lr": a.encoder_lr, "epochs": a.epochs,
           "frozen_encoder": a.freeze_encoder, "edges": a.edges,
           "val_source": a.val_source, "train_frac": a.train_frac,
           "test_roc_auc": round(float(te_auc), 4), "best_val": round(float(best), 4),
           "history": hist}
    sfx = ("-noedge" if a.edges == "none" else ("-rev" if a.reverse_edges else "")) + ("" if a.seed is None else f"-s{a.seed}") \
          + (f"-{a.tag}" if a.tag else "")
    p = ROOT / "runs" / f"joint_{a.mode}{sfx}.json"
    p.write_text(json.dumps(out, indent=2)); print(f"-> {p}")


if __name__ == "__main__":
    main()
