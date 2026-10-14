#!/usr/bin/env python3
"""Generate the findings deck as a .pptx."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches as In, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN

OUT = Path("/home/moses/TSE_Journal/CPG_findings.pptx")

INK   = C(0x17, 0x1B, 0x21)
INK2  = C(0x3D, 0x46, 0x53)
MUTE  = C(0x6B, 0x76, 0x86)
BAD   = C(0x8C, 0x2F, 0x39)
GOOD  = C(0x1F, 0x5E, 0x63)
FLAG  = C(0x8A, 0x62, 0x12)
BG    = C(0xF7, 0xF8, 0xFA)
WHITE = C(0xFF, 0xFF, 0xFF)
RULE  = C(0xDB, 0xE0, 0xE8)

prs = Presentation()
prs.slide_width, prs.slide_height = In(13.333), In(7.5)
BLANK = prs.slide_layouts[6]


def slide(bg=BG):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def tb(s, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
       font="Calibri", space_after=6, line=1.15):
    box = s.shapes.add_textbox(In(x), In(y), In(w), In(h))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line
        for r in p.runs:
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
            r.font.color.rgb = color
    return box


def bullets(s, x, y, w, h, items, size=17):
    box = s.shapes.add_textbox(In(x), In(y), In(w), In(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        txt, col, bold = (it if isinstance(it, tuple) else (it, INK2, False))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + txt
        p.space_after = Pt(11)
        p.line_spacing = 1.18
        for r in p.runs:
            r.font.size, r.font.name = Pt(size), "Calibri"
            r.font.color.rgb, r.font.bold = col, bold


def header(s, kicker, title, tcolor=INK):
    tb(s, 0.7, 0.45, 12, 0.35, kicker, size=12, bold=True, color=MUTE)
    tb(s, 0.7, 0.85, 12, 0.9, title, size=32, bold=True, color=tcolor)
    ln = s.shapes.add_shape(1, In(0.7), In(1.72), In(11.9), Pt(1.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = RULE; ln.line.fill.background()
    ln.shadow.inherit = False


def table(s, x, y, w, rows, colw=None, size=14, head_bg=C(0xEF, 0xF1, 0xF5),
          hl=None):
    nr, nc = len(rows), len(rows[0])
    h = 0.42 * nr
    shp = s.shapes.add_table(nr, nc, In(x), In(y), In(w), In(h))
    t = shp.table
    if colw:
        for i, cw in enumerate(colw):
            t.columns[i].width = In(cw)
    for ri, row in enumerate(rows):
        t.rows[ri].height = In(0.42)
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = head_bg if ri == 0 else WHITE
            cell.margin_left, cell.margin_right = In(0.12), In(0.12)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.name = "Consolas" if ci and ri else "Calibri"
                    r.font.bold = (ri == 0)
                    r.font.color.rgb = INK if ri == 0 else INK2
                    if hl and (ri, ci) in hl:
                        r.font.color.rgb = hl[(ri, ci)]
                        r.font.bold = True
    return shp


def big(s, x, y, w, value, label, color=BAD):
    tb(s, x, y, w, 0.9, value, size=46, bold=True, color=color)
    tb(s, x, y + 0.85, w, 0.7, label, size=13, color=MUTE)


# ── 1 title ──────────────────────────────────────────────────────────────
s = slide(C(0x17, 0x1B, 0x21))
tb(s, 0.9, 2.0, 11.5, 0.4, "TSE SUBMISSION · POST-MORTEM AND FINDINGS",
   size=13, bold=True, color=C(0x7E, 0x8B, 0x9C))
tb(s, 0.9, 2.5, 11.5, 1.9,
   "What CPG-based vulnerability\ndetection actually measures",
   size=44, bold=True, color=WHITE, line=1.05)
tb(s, 0.9, 4.7, 11.0, 1.2,
   "A reported 99.4% was a defect. The corrected pipeline performs at chance.\n"
   "Two thirds of apparent performance is leakage; the graph adds +0.003.",
   size=17, color=C(0xB4, 0xBE, 0xCC), line=1.35)
tb(s, 0.9, 6.4, 11.0, 0.4,
   "Moses Ndebugre · North Carolina A&T State University",
   size=13, color=C(0x7E, 0x8B, 0x9C))

# ── 2 situation ──────────────────────────────────────────────────────────
s = slide()
header(s, "WHERE WE STARTED", "Second reject, three reviewers, one shared doubt")
bullets(s, 0.7, 2.1, 6.0, 4.5, [
    ("Reported 99.38% macro-F1 for OWASP categorisation", INK, True),
    "Reported 0.723 for binary detection",
    "External validation on MegaVul: 4.4% zero-shot",
    ("All three reviewers called the result untrustworthy", BAD, True),
    "None could identify why",
])
tb(s, 7.2, 2.1, 5.4, 0.35, "WHAT THEY SAID", size=12, bold=True, color=MUTE)
tb(s, 7.2, 2.55, 5.4, 3.6,
   "“The principal result is not supported by sufficiently independent "
   "label validation or a leakage-resistant evaluation.”\n\n"
   "“CPG-OWASP looks more like a training corpus engineered for high "
   "in-distribution scores than a trustworthy benchmark.”",
   size=15, color=INK2, line=1.4)
tb(s, 7.2, 5.9, 5.4, 0.4, "Reviewers 2 and 3", size=12, color=MUTE)

# ── 3 finding 1 ──────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 1", "The headline number was a software defect", BAD)
tb(s, 0.7, 2.05, 11.9, 1.0,
   "Stage 2 pre-filled its meta-feature matrix with 0.5, then wrote each "
   "specialist's score back only into the rows of its own category. Every "
   "sample was 0.5 in all columns except its true label.", size=17, color=INK2)
tb(s, 0.7, 3.15, 11.9, 0.45,
   "The classifier only had to find the column that was not 0.5.",
   size=17, bold=True, color=BAD)
table(s, 0.7, 3.85, 11.9, [
    ["Meta-feature construction", "Macro-F1", "Accuracy"],
    ["Sparse fill — what the paper ran", "0.9885", "0.9896"],
    ["Dense — every specialist scores every sample", "0.4903", "0.4962"],
], colw=[7.9, 2.0, 2.0], hl={(1, 1): BAD, (2, 1): GOOD})
tb(s, 0.7, 5.6, 11.9, 0.9,
   "Reported 0.9938. Reproduced 0.9885 from the saved artifacts. "
   "The honest number is 0.49.", size=16, bold=True, color=INK)
tb(s, 0.7, 6.35, 11.9, 0.5,
   "Nothing published is affected — the SVCC 2025 paper used different methods.",
   size=14, color=MUTE)

# ── 4 the proof ──────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 1 · PROOF", "The bug reports perfection from noise")
tb(s, 0.7, 2.05, 11.9, 0.8,
   "The construction was replicated against synthetic specialists of "
   "controlled skill, to test the only property that matters: does the score "
   "measure anything at all?", size=17, color=INK2)
table(s, 0.7, 3.0, 11.9, [
    ["True specialist skill", "Sparse (buggy)", "Dense (fixed)"],
    ["0.00 — specialists are pure noise", "1.0000", "0.0951"],
    ["0.35", "0.9900", "0.3956"],
    ["0.85", "0.9892", "0.8555"],
], colw=[7.9, 2.0, 2.0],
   hl={(1, 1): BAD, (2, 1): BAD, (3, 1): BAD,
       (1, 2): GOOD, (2, 2): GOOD, (3, 2): GOOD})
tb(s, 0.7, 5.1, 11.9, 1.2,
   "The old construction reports a perfect 1.0000 from specialists that know "
   "nothing. It was never measuring the specialists. The corrected one starts "
   "at chance (0.100) and rises with skill.", size=17, bold=True, color=INK)

# ── 5 why hidden ─────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 2", "Why nobody caught it: the corpus hid the bug", BAD)
big(s, 0.7, 2.2, 3.0, "88%", "of held-out vulnerable samples\nwere LLM-generated")
big(s, 4.2, 2.2, 3.0, "7 of 10", "categories contained\nzero real samples")
big(s, 7.7, 2.2, 3.0, "0.89", "macro-F1 detecting LLM vs real\nfrom specialist scores alone")
tb(s, 0.7, 4.5, 11.9, 1.6,
   "The models had learned to recognise GPT-4.1-mini's writing style, not "
   "vulnerabilities. Train on the synthetic half and test on the real half — "
   "same dataset, same categories — and performance collapses to 0.063 macro-F1.",
   size=17, color=INK2, line=1.35)
tb(s, 0.7, 5.9, 11.9, 0.8,
   "On real code the binary detector answered “vulnerable” to "
   "everything: accuracy 0.22688, which is the base rate to five decimals.",
   size=16, bold=True, color=BAD)

# ── 6 root cause ─────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 3 · ROOT CAUSE", "The taxonomy cannot be populated by real C/C++ data")
table(s, 0.7, 2.1, 6.6, [
    ["OWASP category", "Real functions", "Share"],
    ["A01 Broken Access Control", "1,305", "12.0%"],
    ["A03 Injection", "1,235", "11.4%"],
    ["A02 Cryptographic Failures", "101", "0.9%"],
    ["A04 / A05 / A07 / A08", "< 60 each", "< 0.5%"],
    ["A06 Components · A10 SSRF", "0", "0.0%"],
    ["No OWASP category exists", "5,975", "55.1%"],
], colw=[3.4, 1.7, 1.5], hl={(5, 1): BAD, (5, 2): BAD, (6, 1): BAD, (6, 2): BAD})
bullets(s, 7.6, 2.15, 5.0, 4.3, [
    ("OWASP 2021 is a web-application taxonomy", INK, True),
    "Real C/C++ CVEs are memory safety: CWE-119, 125, 787, 416, 190",
    ("So the 9,380 generated samples were load-bearing", BAD, True),
    "Without them, six categories are empty and there is no task",
    "Every reviewer objection follows from this one mismatch",
])

# ── 7 rebuild ────────────────────────────────────────────────────────────
s = slide()
header(s, "THE REBUILD", "Real CVE data, audited splits, zero synthetic samples", GOOD)
for i, (v, l) in enumerate([("131,903", "CPGs, all CVE-anchored"),
                            ("30,881", "vulnerable functions"),
                            ("949", "distinct repositories"),
                            ("10", "CWE families over 500"),
                            ("0", "LLM-generated samples")]):
    big(s, 0.7 + i * 2.45, 2.2, 2.3, v, l, color=GOOD if i == 4 else INK)
tb(s, 0.7, 4.4, 11.9, 0.4, "SPLITS, WITH THE AUDIT PRINTED", size=12,
   bold=True, color=MUTE)
table(s, 0.7, 4.85, 11.9, [
    ["Protocol", "repo", "commit", "pair", "clone"],
    ["repository-disjoint (headline)", "0", "0", "0", "0"],
    ["chronological, corpus held fixed", "by design", "0", "0", "0"],
    ["random (permissive control)", "626", "3,717", "1,590", "1,590"],
], colw=[5.9, 1.5, 1.5, 1.5, 1.5],
   hl={(3, 1): BAD, (3, 2): BAD, (3, 3): BAD, (3, 4): BAD})
tb(s, 0.7, 6.5, 11.9, 0.5,
   "Under a plain split, 1,590 functions sit in training with their own "
   "patched twin in test.", size=15, bold=True, color=BAD)

# ── 8 stub ───────────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 4", "A label-correlated artifact in a widely-used benchmark", BAD)
tb(s, 0.7, 2.05, 11.9, 0.9,
   "Joern is handed every function as .c, so C++ fails to parse and emits a "
   "9-node empty stub. BigVul's vulnerable set is C++-heavy; its safe set was "
   "the CSV prefix, which is C-heavy.", size=17, color=INK2)
table(s, 0.7, 3.1, 11.9, [
    ["Corpus / label", "Stub rate before", "After language detection"],
    ["BigVul vulnerable", "43.7%", "8.4%"],
    ["BigVul safe", "9.8%", "11.5%"],
    ["CVEfixes vulnerable", "23.5%", "8.5%"],
    ["CVEfixes safe", "20.3%", "8.6%"],
], colw=[6.9, 2.5, 2.5], hl={(1, 1): BAD, (2, 1): BAD, (1, 2): GOOD, (3, 2): GOOD})
tb(s, 0.7, 5.5, 11.9, 1.2,
   "A 34-point gap between labels: “is this graph degenerate” "
   "predicted the label with no reference to the code. 4,708 of BigVul's "
   "10,763 vulnerable functions were C++ parsed as C.", size=17, bold=True,
   color=INK)

# ── 9 honest numbers ─────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 5", "Evaluated honestly, the architecture does not work", BAD)
table(s, 0.7, 2.1, 7.4, [
    ["Metric (repository-disjoint)", "Value", "Chance"],
    ["ROC-AUC", "0.607", "0.500"],
    ["PR-AUC", "0.242", "0.189"],
    ["MCC", "0.088", "0.000"],
    ["Recall at 1% false-positive rate", "0.016", "0.010"],
    ["Stage 2 — CWE family, 10-way", "0.093", "0.100"],
], colw=[4.4, 1.5, 1.5],
   hl={(4, 1): BAD, (5, 1): BAD})
bullets(s, 8.4, 2.15, 4.2, 4.2, [
    ("Catches 1.6% of vulnerabilities at a tolerable alert budget", BAD, True),
    "MCC 0.088 is almost no correlation with truth",
    ("Stage 2 is below random guessing", BAD, True),
    "The old paper claimed 0.9938 for that same task",
])
tb(s, 0.7, 6.2, 11.9, 0.6,
   "Validation macro-F1 was 0.65; test was 0.40. The model learns the training "
   "repositories and does not transfer.", size=16, bold=True, color=INK)

# ── 10 the ladder ─────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 6 · THE CENTRAL RESULT",
       "Two thirds of apparent performance is evaluation leakage", GOOD)
tb(s, 0.7, 2.0, 11.9, 0.5,
   "Same corpus. Same architecture. Only the split protocol changes.",
   size=17, color=INK2)
table(s, 0.7, 2.6, 11.9, [
    ["Split protocol", "ROC-AUC", "seeds"],
    ["random - no grouping (what the field uses)", "0.806 +/- 0.002", "5"],
    ["grouped - no duplicates, repos still shared", "0.666", "1"],
    ["repository-disjoint - nothing shared", "0.594 +/- 0.011", "5"],
], colw=[7.9, 2.4, 1.6], hl={(1, 1): BAD, (3, 1): GOOD})
tb(s, 0.7, 4.35, 11.9, 0.45, "DECOMPOSITION OF SIGNAL ABOVE CHANCE",
   size=12, bold=True, color=MUTE)
for i, (v, l, c) in enumerate([
        ("46%", "duplicates, patched twins,\nshared commits", BAD),
        ("23%", "repository overlap", BAD),
        ("31%", "genuine - survives\nevery control", GOOD)]):
    big(s, 0.7 + i * 4.0, 4.8, 3.6, v, l, color=c)
tb(s, 0.7, 6.6, 11.9, 0.5,
   "The grouped rung is a single seed; the two endpoints are seed-averaged.",
   size=13, color=MUTE)

# ── 11 replication ───────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 7", "The pattern replicates across architectures")
table(s, 0.7, 2.1, 11.9, [
    ["Backbone", "random", "grouped", "repo-disjoint", "collapse"],
    ["GAT (attention)", "0.807", "0.666", "0.607", "0.200"],
    ["GCN (spectral)", "0.820", "0.694", "0.642", "0.178"],
    ["GIN (most expressive)", "0.805", "0.676", "0.593", "0.213"],
], colw=[4.7, 1.8, 1.8, 1.8, 1.8])
tb(s, 0.7, 4.0, 11.9, 1.3,
   "All three collapse the same way, so the finding is about the protocol "
   "rather than any one model. They also land within 0.015 of each other under "
   "a random split — including GIN, which is provably the most expressive of "
   "the three at distinguishing graph structure.", size=17, color=INK2,
   line=1.35)
tb(s, 0.7, 5.4, 11.9, 0.9,
   "If message passing were doing the work, GIN should separate from GCN. "
   "It does not.", size=18, bold=True, color=INK)

# ── 12 the graph ─────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 8", "What the graph is actually worth", BAD)
tb(s, 0.7, 1.95, 11.9, 0.45,
   "Same model with every edge deleted. Five seeds per cell, three on the "
   "random rows.", size=16, color=INK2)
table(s, 0.7, 2.5, 11.9, [
    ["Protocol / node features", "full graph", "no graph", "gap", "p"],
    ["random  ·  GraphCodeBERT", "0.806", "0.803", "+0.003", "0.301"],
    ["random  ·  bag-of-words", "0.768", "0.755", "+0.013", "0.009"],
    ["repo  ·  GraphCodeBERT", "0.594", "0.581", "+0.014", "0.067"],
    ["repo  ·  bag-of-words", "0.557", "0.538", "+0.019", "0.006"],
], colw=[5.1, 2.0, 2.0, 1.6, 1.2],
   hl={(1, 3): GOOD, (1, 4): GOOD})
tb(s, 0.7, 4.75, 11.9, 1.1,
   "In the configuration most published work occupies - random split, strong "
   "pretrained encoder - the graph contributes +0.003 and does not reach "
   "significance.", size=18, bold=True, color=BAD, line=1.3)
tb(s, 0.7, 5.95, 11.9, 1.0,
   "It becomes detectable only by weakening the encoder or strengthening the "
   "evaluation. The encoder itself is worth +0.038, and edge types are worth "
   "nothing at all.", size=16, color=INK2, line=1.3)

# ── 13 floor ─────────────────────────────────────────────────────────────
s = slide()
header(s, "FINDING 9 · THE FULL ABLATION",
       "Remove the code text and only memorisation is left", BAD)
table(s, 0.7, 2.05, 11.9, [
    ["What the model may see", "random", "grouped", "repo-disjoint"],
    ["text + node types + graph", "0.807", "0.666", "0.607"],
    ["text + node types, edges deleted", "0.806", "0.667", "0.575"],
    ["NO text - node types + graph only", "0.714", "0.598", "0.476"],
    ["graph shape only", "0.452", "0.461", "0.494"],
    ["chance", "0.500", "0.500", "0.500"],
], colw=[5.9, 2.0, 2.0, 2.0],
   hl={(3, 1): BAD, (3, 3): GOOD, (1, 1): BAD, (2, 1): BAD})
tb(s, 0.7, 4.75, 11.9, 1.4,
   "With no code text at all - only node types and graph structure - the model "
   "still scores 0.714 under a random split. That is higher than many published "
   "results. Under a repository-disjoint split the same model scores 0.476: "
   "chance.", size=17, bold=True, color=INK, line=1.3)
tb(s, 0.7, 6.15, 11.9, 0.9,
   "Node types cannot indicate a vulnerability - \"three CALL nodes and two "
   "IDENTIFIERs\" says nothing about a buffer overflow. But they are a "
   "fingerprint. The 0.714 is recognition, not detection.",
   size=16, color=INK2, line=1.3)

# ── 14 what it means ─────────────────────────────────────────────────────
s = slide()
header(s, "WHAT THIS MEANS", "The contribution is a measurement, not a detector", GOOD)
bullets(s, 0.7, 2.1, 5.9, 4.6, [
    ("The detector paper is not viable", BAD, True),
    "Not from a bug — the pipeline is clean, the splits audited, the guard passes",
    "The honest number is simply at chance",
    ("The measurement paper is strong", GOOD, True),
    "Every reviewer objection becomes a finding you own",
    "Corroborates ReVeal, PrimeVul and Li et al. with a mechanism",
])
tb(s, 6.9, 2.15, 5.7, 0.4, "WHAT IS NEW HERE", size=12, bold=True, color=MUTE)
bullets(s, 6.9, 2.6, 5.7, 4.2, [
    "First decomposition of the drop BY LEAKAGE TYPE on one corpus, model held fixed",
    "First demonstration that edge removal costs nothing under permissive evaluation",
    "A reproducible ~0.99 to 0.49 stacking-leak demonstration",
    "A label-correlated artifact in BigVul affecting anyone who builds CPGs from it",
])

# ── 15 caveat + next ─────────────────────────────────────────────────────
s = slide()
header(s, "A METHODS NOTE WORTH REPORTING",
       "Single-run ablations are unreliable at these effect sizes", FLAG)
table(s, 0.7, 2.1, 11.9, [
    ["Quantity", "at n=1", "at n=3", "at n=5"],
    ["graph contribution, GraphCodeBERT + repo", "+0.032", "+0.016", "+0.014"],
    ["graph contribution, bag-of-words + repo", "-0.002", "-", "+0.019"],
], colw=[6.5, 1.8, 1.8, 1.8], hl={(1, 1): BAD, (2, 1): BAD})
tb(s, 0.7, 3.75, 11.9, 1.3,
   "Both single-seed values were wrong, in opposite directions, on gaps the "
   "conclusions depended on. Run-to-run standard deviation is ~0.010 - the "
   "same size as the effects being measured.", size=17, color=INK2, line=1.3)
bullets(s, 0.7, 5.15, 5.9, 2.0, [
    ("Complete", GOOD, True),
    "Corpus, four split protocols, three backbones, feature and edge ablations",
    "Seed study: 4 cells, n=5 / n=3",
])
bullets(s, 6.9, 5.15, 5.7, 2.0, [
    ("Open", INK, True),
    "Backbone and ladder rungs are still single-seed",
    "word2vec features; LIVABLE / Snopy / CausalVul; a second corpus",
])

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
