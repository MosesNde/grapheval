#!/usr/bin/env python3
"""
make_figures.py — the paper's two figures, generated from run artifacts.

Nothing is hard-coded: every value is read from runs/*.json, so a figure cannot
drift from the numbers in the tables. Seed-averaged where seeds exist, with the
count annotated, because several of the gaps shown are close to the ~0.010
run-to-run standard deviation and a bare point would overstate them.

Output: Journal/v2/figs/{ladder,graph_worth}.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "Journal" / "v2" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTE, BAD, GOOD = "#171B21", "#6B7686", "#8C2F39", "#1F5E63"

# Categorical hues, keyed by BACKBONE so a series keeps its colour no matter
# what order it is plotted in or which subset a figure shows. Validated as a
# 4-slot set on the all-pairs list (every line is visible at once and three of
# them converge at the repo rung, so adjacent-only separation is not enough):
# worst CVD dE 9.2, worst normal-vision dE 16.3, both above their floors.
# Aqua sits below 3:1 on white, which the direct end-labels relieve.
# GAT, GIN and the graph transformer converge to within 0.013 at the repo rung,
# so they take the three best-separated hues (worst normal-vision dE 24.0).
# GCN, which sits clear of the others at every rung, takes violet: its closest
# pair is violet-blue at dE 16.3, and those two lines never come near touching.
# Marker shape repeats the encoding so the figure survives greyscale printing.
SERIES = {
    "GAT":           ("#2a78d6", "o"),   # blue
    "GIN":           ("#eb6834", "s"),   # orange
    "Graph transf.": ("#1baf7a", "^"),   # aqua
    "GCN":           ("#4a3aa7", "D"),   # violet
}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})


def g(tag: str):
    p = RUNS / f"{tag}_stage1.json"
    if not p.exists():
        return None
    return json.load(open(p))["ensemble_mean_only"]["roc_auc"]


def avg(tags):
    v = [g(t) for t in tags]
    v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) if v else None


# ── Figure 1: the ladder, per architecture ──────────────────────────────────
# LineVul is deliberately absent from this figure. It belongs to RQ4, and the
# figure's claim is that ONLY the split protocol changes between the lines
# drawn -- which is false of LineVul, whose encoder and training regime also
# differ from the graph models. Set SHOW_LINEVUL = True to restore it.
SHOW_LINEVUL = False


def fig_ladder():
    modes = ["random", "group", "repo"]
    labels = ["random\n(no grouping)", "grouped\n(repos shared)",
              "repository-\ndisjoint"]
    series = {
        "GAT": (avg(["random-gat", "random-gat-s1", "random-gat-s2", "random-gat-s3", "random-gat-s4"]), avg(["group-gat", "group-gat-s1", "group-gat-s3", "group-gat-s4"]),
                avg(["repo", "repo-gat-s1", "repo-gat-s2", "repo-gat-s3",
                     "repo-gat-s4"])),
        "GCN": (g("random-gcn"), g("group-gcn"), g("repo-gcn")),
        "GIN": (g("random-gin"), g("group-gin"), g("repo-gin")),
    }
    gt = [g(f"{m}-gtrans") for m in modes]
    if all(x is not None for x in gt):
        series["Graph transf."] = tuple(gt)
    lv = RUNS / "linevul_official_stage1.json"

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(3)
    ends = []
    for name, vals in series.items():
        y, err = [], []
        for v in vals:
            if isinstance(v, tuple):
                y.append(v[0]); err.append(v[1])
            else:
                y.append(v); err.append(0.0)
        c, mk = SERIES.get(name, (MUTE, "o"))
        ax.errorbar(x, y, yerr=err, marker=mk, ms=5, lw=2.0, capsize=2.5,
                    color=c, alpha=.95, zorder=3,
                    markeredgecolor="white", markeredgewidth=.7)
        ends.append((y[-1], name))

    # stagger the end labels: several backbones finish within 0.002 of each
    # other on the repo rung and would otherwise overprint
    ends.sort()
    last = None
    for yv, name in ends:
        off = yv if last is None or yv - last > 0.020 else last + 0.020
        c, mk = SERIES.get(name, (MUTE, "o"))
        # Text stays ink; a coloured marker beside it carries identity, so the
        # label never relies on the text colour. Always draw the leader, since
        # the end points overlap even when the labels do not.
        ax.annotate(name, (x[-1], yv), xytext=(x[-1] + 0.18, off),
                    fontsize=8, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", lw=.6, color=c, alpha=.75))
        ax.plot([x[-1] + 0.13], [off], marker=mk, ms=4.5, color=c,
                markeredgecolor="white", markeredgewidth=.6, clip_on=False)
        last = off

    if SHOW_LINEVUL and lv.exists():
        d = json.load(open(lv))["results"]
        yv = [d[m]["roc_auc"] for m in modes]
        ax.plot(x, yv, marker="s", ms=5, lw=2.2, color=BAD)
        ax.annotate("LineVul\n(no graph)", (x[-1], yv[-1]),
                    textcoords="offset points", xytext=(6, -6),
                    fontsize=8, color=BAD, weight="bold")

    ax.axhline(0.5, ls=":", lw=1, color=INK, alpha=.5)
    ax.annotate("chance", (0.02, 0.505), fontsize=7.5, color=INK, alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 0.97) if SHOW_LINEVUL else ax.set_ylim(0.47, 0.86)
    ax.set_xlim(-0.25, 3.05)
    ax.set_title("Only the split protocol changes", fontsize=10, loc="left",
                 weight="bold", pad=8)
    ax.grid(axis="y", alpha=.25, lw=.6)
    fig.tight_layout()
    fig.savefig(OUT / "ladder.pdf", bbox_inches="tight")
    print(f"  wrote {OUT/'ladder.pdf'}")


# ── Figure 2: what each component is worth ──────────────────────────────────
def fig_graph_worth():
    cells = [
        ("random\nGraphCodeBERT",
         avg(["random-gat", "random-gat-s1", "random-gat-s2", "random-gat-s3", "random-gat-s4"]),
         avg(["random-gat-enone", "random-gat-enone-s1", "random-gat-enone-s2",
              "random-gat-enone-s3", "random-gat-enone-s4"])),
        ("random\nbag-of-words",
         avg(["random-gat-bow", "random-gat-s1-bow", "random-gat-s2-bow",
              "random-gat-s3-bow", "random-gat-s4-bow"]),
         avg(["random-gat-enone-bow", "random-gat-enone-s1-bow",
              "random-gat-enone-s2-bow", "random-gat-enone-s3-bow",
              "random-gat-enone-s4-bow"])),
        ("repo\nGraphCodeBERT",
         avg(["repo-gat", "repo-gat-s1", "repo-gat-s2", "repo-gat-s3", "repo-gat-s4"]),
         avg(["repo-gat-enone", "repo-gat-enone-s1", "repo-gat-enone-s2",
              "repo-gat-enone-s3", "repo-gat-enone-s4"])),
        ("repo\nbag-of-words",
         avg(["repo-gat-bow", "repo-gat-s1-bow", "repo-gat-s2-bow",
              "repo-gat-s3-bow", "repo-gat-s4-bow"]),
         avg(["repo-gat-enone-bow", "repo-gat-enone-s1-bow",
              "repo-gat-enone-s2-bow", "repo-gat-enone-s3-bow",
              "repo-gat-enone-s4-bow"])),
    ]
    cells = [c for c in cells if c[1] and c[2]]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    y = np.arange(len(cells))
    gaps = [c[1][0] - c[2][0] for c in cells]
    errs = [np.hypot(c[1][1], c[2][1]) for c in cells]
    ax.barh(y, gaps, xerr=errs, height=.55, color=[GOOD if v > 0 else BAD for v in gaps],
            alpha=.85, error_kw=dict(lw=1, capsize=3, ecolor=INK))
    for i, (v, c, e) in enumerate(zip(gaps, cells, errs)):
        tip = v + e if v > 0 else v - e
        ax.annotate(f"{v:+.3f} (n={c[1][2]})", (tip, i), textcoords="offset points",
                    xytext=(10 if v > 0 else -10, 0), fontsize=8, va="center",
                    ha="left" if v > 0 else "right", color=INK)
    ax.axvline(0, lw=1, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cells], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC gained by adding the entire code property graph")
    ax.set_xlim(-0.012, 0.062)
    ax.set_title("What the graph is worth", fontsize=10, loc="left",
                 weight="bold", pad=8)
    ax.grid(axis="x", alpha=.25, lw=.6)
    # the encoder, for scale
    enc = None
    a = avg(["repo-gat", "repo-gat-s1", "repo-gat-s2", "repo-gat-s3", "repo-gat-s4"])
    b = avg(["repo-gat-bow", "repo-gat-s1-bow", "repo-gat-s2-bow",
             "repo-gat-s3-bow", "repo-gat-s4-bow"])
    if a and b:
        enc = a[0] - b[0]
        ax.axvline(enc, ls="--", lw=1.2, color=BAD)
        # vertical, inside the axes: the space right of the bars is empty and
        # anything above the plot collides with the title
        # no on-figure label: the value labels run out past this line and any
        # text here collides with one of them. The caption names the value.
    fig.tight_layout()
    fig.savefig(OUT / "graph_worth.pdf", bbox_inches="tight")
    print(f"  wrote {OUT/'graph_worth.pdf'}")


if __name__ == "__main__":
    fig_ladder()
    fig_graph_worth()
