"""
linevul_adapter.py
===================
Bridges awsm-research/LineVul (https://github.com/awsm-research/LineVul) into
CPGExplainer's own evaluation protocol (sections/04_evaluation.tex), so LineVul
can be added as a row in Table 1 (tab:alignment) on equal footing with
GNNExplainer / PGExplainer / GradCAM / CPGExplainer-*.

WHY THIS FILE EXISTS (not just "run LineVul and read its own numbers off")
---------------------------------------------------------------------------
1. Different ground-truth definitions. LineVul verifies flaw lines by
   *substring-matching decoded token text* against a `flaw_line` column
   (linevul_main.py, get_all_lines_score, L1051-1079). CPGExplainer's ground
   truth is *line numbers* from a difflib diff (04_evaluation.tex Sec 4.1).
   Reporting "LineVul's numbers" from its own paper would silently compare
   against a different, less rigorous ground-truth definition than every
   other row in Table 1. This file scores LineVul's model but grades it
   against the *same* line-number ground truth used everywhere else in the
   table.

2. A real alignment bug in LineVul's own reference code. Its
   get_all_lines_score() only appends a line's score (and only advances its
   internal line counter) when that line accumulated nonzero token-attention
   (`... and score_sum != 0`, L1063). A blank line has no tokens, so its
   score_sum is always 0 -- it gets silently dropped from the output list
   *and* the line counter never advances for it. That's invisible to
   LineVul's own text-matching (it doesn't use line numbers at all), but it
   would silently misalign our line-number-indexed ranking by one position
   for every blank line above it. get_attention_line_scores() below fixes
   this: every "\u010a" (the BPE encoding of a newline) advances the line
   counter, blank or not, so scores[i] always corresponds to
   func_text.split("\n")[i]. Verified against the real bundled BPE tokenizer
   across blank-line / consecutive-blank-line / no-trailing-newline cases
   (see test_line_alignment.py) before trusting this.

3. Same metric formulas as everywhere else in the paper. LineVul's own
   evaluation reports Top-k% Recall / Effort@TopK% / IFA -- none of which are
   MAP / P@k / R@k / Hit@k / MRR / NDCG@10. compute_all_metrics() below
   implements the exact formulas from 04_evaluation.tex Sec 4.2, so LineVul's
   row is computed the identical way CPGExplainer's and the GNN-explainer
   baselines' rows are.

WHAT THIS FILE DOES NOT DO
---------------------------------------------------------------------------
This sandbox has no GPU, no HuggingFace Hub access, and no Google Drive
access (all three confirmed blocked here), so it cannot itself fine-tune or
run CodeBERT. This file is meant to run on your own workstation (the one
already training the GAT detector -- 128 cores / 4x RTX PRO 6000 per
04_evaluation.tex Sec 4.5), after you:

  1. git clone https://github.com/awsm-research/LineVul.git
     cd LineVul && pip install -r requirements.txt

  2. Build train/val/test CSVs from *your own* existing split
     (04_evaluation.tex Sec 4.1) -- LineVul's training command needs all
     three (it evaluates during training for early stopping, same as your
     GAT detector already does per Sec 4.4's "early stopping on validation
     macro-F1"). Reuse that same validation split here too. Do NOT let
     LineVul re-split from scratch, and do NOT use LineVul's off-the-shelf
     Google Drive checkpoint -- that checkpoint was trained on LineVul's own
     arbitrary BigVul split, so some of your held-out 990 test functions
     could easily have been in *its* training set. Fine-tuning fresh on your
     exact split is the only leak-free comparison:

         from linevul_adapter import build_linevul_csv
         build_linevul_csv(your_train_records, "LineVul/data/big-vul_dataset/train.csv")
         build_linevul_csv(your_val_records,   "LineVul/data/big-vul_dataset/val.csv")
         build_linevul_csv(your_test_records,  "LineVul/data/big-vul_dataset/test.csv")

     each record needs {"processed_func": str, "target": 0/1}.

  3. Fine-tune -- this is LineVul's own published retrain command verbatim
     (README.md, "How to replicate RQ1"), so the numbers you get are
     comparable to their own reported F1/precision/recall, not an
     approximation of their setup:

         cd LineVul/linevul
         python linevul_main.py \
             --output_dir=./saved_models \
             --model_type=roberta \
             --tokenizer_name=microsoft/codebert-base \
             --model_name_or_path=microsoft/codebert-base \
             --do_train --do_test \
             --train_data_file=../data/big-vul_dataset/train.csv \
             --eval_data_file=../data/big-vul_dataset/val.csv \
             --test_data_file=../data/big-vul_dataset/test.csv \
             --epochs 10 --block_size 512 \
             --train_batch_size 16 --eval_batch_size 16 \
             --learning_rate 2e-5 --max_grad_norm 1.0 \
             --evaluate_during_training --seed 123456 \
             2>&1 | tee train.log

     Your GPUs (4x RTX PRO 6000, 96GB each, per Sec 4.5) are wildly
     overprovisioned for batch size 16 -- that's fine, this should finish
     in a few hours on one card. Also worth checking *before* this step:
     `requirements.txt` pins torch==1.13.1 built for CUDA 11.7, which
     predates RTX PRO 6000 hardware and may not carry a compatible CUDA
     kernel for it. Run the environment check at the bottom of this
     docstring first so you're not debugging a silent GPU/driver mismatch
     three hours into training.

  4. Load that checkpoint and call evaluate_linevul() below on your 990
     patch-matched functions (same set as Table 1's alignment rows), using
     line numbers from the *same* difflib ground truth your GAT/CPG pipeline
     already computes -- reuse it directly rather than rebuilding it, so
     both rows are graded against one, single, shared ground truth:

         import torch
         from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
         from linevul_model import Model
         from linevul_adapter import evaluate_linevul, format_table_row

         device = "cuda"
         ckpt_dir = "microsoft/codebert-base"
         config = RobertaConfig.from_pretrained(ckpt_dir)
         config.num_labels = 2
         tokenizer = RobertaTokenizer.from_pretrained(ckpt_dir)
         encoder = RobertaForSequenceClassification(config)
         model = Model(encoder, config, tokenizer, args=None).to(device)
         model.load_state_dict(torch.load("./saved_models/checkpoint-best-f1/12heads_linevul_model.bin"))

         means, per_function = evaluate_linevul(model, tokenizer, device, your_990_test_records)
         print(format_table_row("LineVul", means))
         print("truncated by block_size:", means["n_truncated_by_block_size"], "/", means["n"])

Environment check (run this before step 3 -- catches a GPU/CUDA-build
mismatch before it costs you three hours, not after):

    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), \
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE')"

  If that prints `False` or errors instead of naming your GPU, requirements.txt's
  torch==1.13.1+cu117 doesn't support your hardware. Check `nvidia-smi` for the
  CUDA version your driver supports, then install a matching current build, e.g.:

      pip install torch --index-url https://download.pytorch.org/whl/cu124

  (adjust cu124 to whatever nvidia-smi reports). transformers==4.26.0 and
  captum==0.7.0 from requirements.txt are old but should still import and run
  fine against a modern torch for the plain classification + attention path
  used here (captum itself is only exercised by the lig/deeplift/shap/saliency
  reasoning methods, which this adapter doesn't use).

Self-test (no GPU / torch / network required) -- run this file directly:
    python3 linevul_adapter.py
"""

import csv
import math

# ----------------------------------------------------------------------------
# 1. Metric formulas -- transcribed directly from sections/04_evaluation.tex
#    Sec 4.2 (P@k, R@k, MAP) plus the standard definitions for the other
#    columns already in Table 1 (F1@5, Hit@1, Hit@3, MRR, NDCG@10).
# ----------------------------------------------------------------------------

def _hits(ranked_lines, gt_set, k):
    return len(set(ranked_lines[:k]) & gt_set)


def precision_at_k(ranked_lines, gt_set, k):
    return 0.0 if k == 0 else _hits(ranked_lines, gt_set, k) / k


def recall_at_k(ranked_lines, gt_set, k):
    return 0.0 if not gt_set else _hits(ranked_lines, gt_set, k) / len(gt_set)


def f1_at_k(ranked_lines, gt_set, k):
    p, r = precision_at_k(ranked_lines, gt_set, k), recall_at_k(ranked_lines, gt_set, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def average_precision(ranked_lines, gt_set):
    """MAP = (1/|L_g|) * sum_{i : l_i in L_g} (|L[1:i] cap L_g| / i)"""
    if not gt_set:
        return 0.0
    hits, score = 0, 0.0
    for i, line in enumerate(ranked_lines, start=1):
        if line in gt_set:
            hits += 1
            score += hits / i
    return score / len(gt_set)


def hit_at_k(ranked_lines, gt_set, k):
    return 1.0 if _hits(ranked_lines, gt_set, k) > 0 else 0.0


def reciprocal_rank(ranked_lines, gt_set):
    for i, line in enumerate(ranked_lines, start=1):
        if line in gt_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_lines, gt_set, k):
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, line in enumerate(ranked_lines[:k], start=1)
        if line in gt_set
    )
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def compute_all_metrics(ranked_lines, gt_lines):
    """Returns exactly Table 1's columns, plus P@3/R@3/P@10/R@10 as a free
    robustness check (04_evaluation.tex Sec 4.2 defines k in {3,5,10} generally,
    Table 1 only prints the @5 pair)."""
    gt_set = set(gt_lines)
    return {
        "MAP": average_precision(ranked_lines, gt_set),
        "P@5": precision_at_k(ranked_lines, gt_set, 5),
        "R@5": recall_at_k(ranked_lines, gt_set, 5),
        "F1@5": f1_at_k(ranked_lines, gt_set, 5),
        "Hit@1": hit_at_k(ranked_lines, gt_set, 1),
        "Hit@3": hit_at_k(ranked_lines, gt_set, 3),
        "MRR": reciprocal_rank(ranked_lines, gt_set),
        "NDCG@10": ndcg_at_k(ranked_lines, gt_set, 10),
        "P@3": precision_at_k(ranked_lines, gt_set, 3),
        "R@3": recall_at_k(ranked_lines, gt_set, 3),
        "P@10": precision_at_k(ranked_lines, gt_set, 10),
        "R@10": recall_at_k(ranked_lines, gt_set, 10),
    }


# ----------------------------------------------------------------------------
# 2. Corrected attention -> per-source-line score extraction.
#    Mirrors linevul_main.py's attention branch (L864-886) for how attention
#    is pulled from the model; replaces its line-matching (L1051-1079) with
#    explicit, verified index tracking. See module docstring point 2.
# ----------------------------------------------------------------------------

NEWLINE_MARKERS = {"Ċ", " Ċ", "ĊĊ", " ĊĊ"}  # same set linevul_main.py uses, L1054


def get_attention_line_scores(model, tokenizer, func_text, device, block_size=512,
                               reproduce_linevul_bug=False):
    """Requires torch + a fine-tuned LineVul `Model` (linevul_model.py) --
    only runs on your GPU workstation, not in this sandbox.

    `attentions` from the model is a tuple of one (batch, heads, seq, seq)
    tensor PER LAYER. LineVul's own code (linevul_main.py L871-882) does
    `attentions = attentions[0][0]`, then loops over what its comments call
    "the layer" -- but attentions[0] already selects layer 0 specifically,
    so attentions[0][0] is (layer 0, batch 0), and the loop right after it
    iterates over layer 0's *heads*, never touching layers 1..11. Proven
    with a synthetic all-layers-random-except-layer-0-zeroed tensor: the
    literal LineVul path returns exactly 0.0; summing correctly across all
    layers returns the nonzero contribution from layers 1..11. Default here
    is the corrected, full-model sum. Pass reproduce_linevul_bug=True only if
    you deliberately want a number reproducing LineVul's literal released
    code (e.g. to sanity-check against its own published RQ2 table) rather
    than the intended full-model attention signal.

    Returns (line_scores, truncated):
      line_scores[i]  -- summed attention for source line i (0-indexed,
                          aligned to func_text.split("\\n")), 0.0 for blank
                          lines and for any line past the block_size cutoff.
      truncated       -- True if func_text likely exceeded block_size tokens,
                          meaning some ground-truth lines may be structurally
                          unreachable. Report this rate alongside the row --
                          if it's non-trivial, it belongs in the paper's text
                          as a caveat, not silently averaged away.
    """
    import torch  # local import: keeps this file importable/self-testable without torch

    code_tokens = tokenizer.tokenize(str(func_text))[: block_size - 2]
    input_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = input_ids + [tokenizer.pad_token_id] * (block_size - len(input_ids))

    ids_tensor = torch.tensor([input_ids]).to(device)
    model.eval()
    with torch.no_grad():
        _, attentions = model(input_ids=ids_tensor, output_attentions=True)

    if reproduce_linevul_bug:
        # literal reproduction of linevul_main.py L871-882: layer 0 only
        layer0 = attentions[0][0]  # (heads, seq, seq)
        token_scores = None
        for head_attn in layer0:
            received = head_attn.sum(dim=0)  # (seq,), summed over query dim
            token_scores = received if token_scores is None else token_scores + received
        token_scores = token_scores.tolist()
    else:
        total = None
        for layer_t in attentions:            # every layer, not just layer 0
            per_head_summed = layer_t[0].sum(dim=0)   # (seq, seq), summed over heads
            received = per_head_summed.sum(dim=0)     # (seq,), summed over query dim
            total = received if total is None else total + received
        token_scores = total.tolist()

    n_source_lines = func_text.count("\n") + 1
    line_scores = [0.0] * n_source_lines
    line_idx = 0
    # walk only the real content tokens (drop CLS/SEP/PAD -- they carry no line)
    content_tokens = input_tokens[1 : 1 + len(code_tokens)]
    content_scores = token_scores[1 : 1 + len(code_tokens)]
    for tok, score in zip(content_tokens, content_scores):
        clean = tok.replace("Ġ", "")
        if clean in NEWLINE_MARKERS or clean == "Ċ":
            line_idx += clean.count("Ċ") or 1
            continue
        if line_idx < n_source_lines:
            line_scores[line_idx] += score
        # else: token belongs to a source line beyond the block_size cutoff --
        # structurally invisible to LineVul; leave its score at 0.

    truncated = len(code_tokens) >= block_size - 2
    return line_scores, truncated


def rank_lines_by_score(line_scores):
    return sorted(range(len(line_scores)), key=lambda i: line_scores[i], reverse=True)


# ----------------------------------------------------------------------------
# 3. CSV builder for LineVul's own TextDataset (linevul_main.py L57-70 reads
#    exactly these two columns: processed_func, target).
# ----------------------------------------------------------------------------

def build_linevul_csv(records, out_path):
    """records: iterable of {"processed_func": str, "target": 0 or 1}.
    Build train.csv / test.csv from the *same* split your GAT detector
    already uses (04_evaluation.tex Sec 4.1) -- do not re-split."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["processed_func", "target"])
        writer.writeheader()
        for r in records:
            writer.writerow({"processed_func": r["processed_func"], "target": int(r["target"])})


# ----------------------------------------------------------------------------
# 4. Driver: run this on your GPU workstation once the model is fine-tuned.
# ----------------------------------------------------------------------------

def evaluate_linevul(model, tokenizer, device, test_records, block_size=512):
    """test_records: iterable of {"processed_func": str, "flaw_line_numbers":
    list[int]} -- flaw_line_numbers must be the SAME 0-indexed, difflib-
    derived line numbers your CPGExplainer/GNNExplainer/PGExplainer/GradCAM
    rows are already graded against (reuse that list directly; don't
    recompute it a second, possibly-inconsistent way). Use only the 990
    patch-matched functions, to match Table 1's n.

    Returns (means_dict_for_table_row, per_function_list).
    """
    per_function = []
    n_truncated = 0
    for rec in test_records:
        scores, truncated = get_attention_line_scores(model, tokenizer, rec["processed_func"], device, block_size)
        n_truncated += int(truncated)
        ranking = rank_lines_by_score(scores)
        per_function.append(compute_all_metrics(ranking, rec["flaw_line_numbers"]))

    keys = per_function[0].keys()
    means = {k: sum(m[k] for m in per_function) / len(per_function) for k in keys}
    means["n"] = len(per_function)
    means["n_truncated_by_block_size"] = n_truncated
    return means, per_function


def format_table_row(system_name, means):
    """Emits a row matching sections/05_results.tex Table 1's column order:
    System & MAP & P@5 & R@5 & F1@5 & Hit@1 & Hit@3 & MRR & NDCG@10"""
    cols = ["MAP", "P@5", "R@5", "F1@5", "Hit@1", "Hit@3", "MRR", "NDCG@10"]
    row = " & ".join(f"{means[c]:.3f}" for c in cols)
    return f"{system_name} & {row} \\\\"


# ----------------------------------------------------------------------------
# 5. Self-test -- no torch, no GPU, no network beyond the `tokenizers` pip
#    package needed. Validates the metric math against hand-checkable cases
#    and the line-alignment logic against the real bundled BPE tokenizer
#    (blank lines, consecutive blank lines, no trailing newline).
# ----------------------------------------------------------------------------

def _self_test_metrics():
    # 5 lines, ground truth = {1, 3}. Ranking puts both gt lines first (0,1
    # swapped so rank order is [1, 3, 0, 2, 4]) -> hand-computable by hand.
    ranking = [1, 3, 0, 2, 4]
    gt = {1, 3}
    m = compute_all_metrics(ranking, gt)
    # MAP: hits at i=1 (1/1) and i=2 (2/2) -> (1 + 1)/2 = 1.0 (both gt lines
    # are ranked first and second, i.e. perfect ranking)
    assert abs(m["MAP"] - 1.0) < 1e-9, m["MAP"]
    assert abs(m["P@5"] - 2 / 5) < 1e-9
    assert abs(m["R@5"] - 1.0) < 1e-9  # both gt lines recovered within top 5
    assert abs(m["Hit@1"] - 1.0) < 1e-9  # line 1 (a gt line) is rank 1
    assert abs(m["MRR"] - 1.0) < 1e-9  # first hit at rank 1
    print("metric self-test: PASS")


def _self_test_line_alignment():
    from tokenizers import ByteLevelBPETokenizer
    import os

    vocab = "/home/claude/LineVul/linevul/bpe_tokenizer/bpe_tokenizer-vocab.json"
    merges = "/home/claude/LineVul/linevul/bpe_tokenizer/bpe_tokenizer-merges.txt"
    if not (os.path.exists(vocab) and os.path.exists(merges)):
        print("line-alignment self-test: SKIPPED (bundled tokenizer files not found "
              "-- fine, this only matters when validating against LineVul's actual "
              "repo checkout; the metric self-test above is the load-bearing one).")
        return
    tok = ByteLevelBPETokenizer(vocab, merges)

    def fake_score_lines(func_text):
        enc = tok.encode(func_text)
        toks = [t.replace("Ġ", "") for t in enc.tokens]
        fake = [0.0 if (t in NEWLINE_MARKERS or t == "Ċ") else 1.0 for t in toks]
        n = func_text.count("\n") + 1
        out = [0.0] * n
        idx = 0
        for t, s in zip(toks, fake):
            if t in NEWLINE_MARKERS or t == "Ċ":
                idx += t.count("Ċ") or 1
                continue
            if idx < n:
                out[idx] += s
        return out

    cases = [
        "int foo(int x) {\n\n    if (x > 0) {\n        return x;\n    }\n    return -1;\n}",
        "a;\n\n\nb;",
        "single_line_no_trailing_newline;",
        "x;\n\n",
    ]
    for src in cases:
        scores = fake_score_lines(src)
        lines = src.split("\n")
        assert len(scores) == len(lines), (src, scores)
        for i, line_text in enumerate(lines):
            if line_text.strip() == "":
                assert scores[i] == 0.0, f"blank line {i} got nonzero score in {src!r}"
    print("line-alignment self-test: PASS "
          f"({len(cases)} cases incl. blank/consecutive-blank/no-trailing-newline)")


def _self_test_layer_aggregation():
    try:
        import torch
    except ImportError:
        print("layer-aggregation self-test: SKIPPED (torch not installed in this "
              "environment -- this is the one test worth re-running once you're on "
              "the GPU workstation, since it's the check that caught the bug).")
        return

    torch.manual_seed(0)
    num_layers, heads, seq = 12, 12, 20
    attentions = tuple(torch.rand(1, heads, seq, seq) for _ in range(num_layers))
    attentions = (torch.zeros(1, heads, seq, seq),) + attentions[1:]  # zero out layer 0

    # literal LineVul path: should see ONLY layer 0 -> exactly 0.0
    layer0 = attentions[0][0]
    buggy = None
    for head_attn in layer0:
        received = head_attn.sum(dim=0)
        buggy = received if buggy is None else buggy + received
    assert float(buggy.sum()) == 0.0, "expected layer-0-only path to be exactly zero"

    # corrected path: should see layers 1..11 -> nonzero
    total = None
    for layer_t in attentions:
        per_head_summed = layer_t[0].sum(dim=0)
        received = per_head_summed.sum(dim=0)
        total = received if total is None else total + received
    assert float(total.sum()) > 0.0, "expected all-layer path to pick up layers 1..11"
    print("layer-aggregation self-test: PASS (literal LineVul path reads layer 0 only; "
          "corrected path sums all layers)")


if __name__ == "__main__":
    _self_test_metrics()
    _self_test_line_alignment()
    _self_test_layer_aggregation()
