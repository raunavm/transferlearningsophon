#!/usr/bin/env python3
"""The rewrite's main and appendix figures (PI 2026-11-26, ledger 34.3, after 2026-11-24 RW-3, RW-5, RW-8, GO-6, GO-9),
drawn from the gallery's parts with the rewrite's names (RW-5) and no flavor-boundary partition in any of them:

    python3 experiments/FIGS/figstyle/v2_main.py [--outdir paper/journal/v2/figures] [--check]

  fig1_design         Fig. 1: the label sets, the nesting of one example group (GO-6); one flow of pretraining and use
  fig2_accuracy       Fig. 2: accuracy against the label set, linear probe and fine-tuned, trained from scratch dashed
  fig3_classes        Fig. 3: per-class recall of the 43- and 17-class models against the 188-class one; 1 - AUC of the
                      pairs of classes, one row a pair
  fig5_resonance      Fig. 4: X->bb and X->bs, signal efficiency against background efficiency, the ratio to 188 classes
  fig4_training_size  Fig. 5: fine-tuned JetClass-II and JetClass accuracy against the number of training jets, with
                      the difference to 188 classes
  figA1  recall of each JetClass-II class under a linear probe, 162, 43 and 17 against 188 classes (GO-9: probes only)
  figA2  the pair curves of the resonance pairs; figA6 those of the QCD pairs (ledger 34.3: G2 split in two)
  figA3  linear probes against the number of training jets, Table 2's pretrained rows
  figA4  the mass output's response and resolution against jet pT and generator-level mass
  figA5  validation accuracy of the pretraining runs against epoch
  figA7  fine-tuned top tagging and quark/gluon against the number of training jets (moved from Fig. 5)

Every number is the build's, through the gallery's loaders (gallery_common: the probe fits, the complete fine-tuning
read-out budget/fixed_metrics.json, the curve files, stats/ledger33_exports.json); a bar or band is the standard
deviation over runs, as in the tables. Where curves overlap, a lower panel gives them relative to 188 classes.
--check: figcheck's measured findings per figure (<stem>_check.json) and its print-size rasters.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import re

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import gallery_common as C  # noqa: E402
import gallery_labelset as G  # noqa: E402
import gallery_perclass as PC  # noqa: E402
import gallery_physics as PH  # noqa: E402
import gallery_roc as R  # noqa: E402
import lines_v6 as L  # noqa: E402
import style_v5 as S  # noqa: E402

M = C.M
OUT = C.P.REPO / "paper/journal/v2/figures"
# RW-5: one name each; the random groupings of a class pair by whether they merge it (LEDGER PS-5)
NAMES = {"RAND": "17 classes, random grouping", "RAND_MERGED": "17 classes, random grouping,\npair merged",
         "RAND_APART": "17 classes, random grouping,\npair apart", "L162_MASS": "162 classes + mass regression",
         "R16_Q1_MASS": "17 classes + mass regression", "SCRATCH": "trained from scratch", "SOPHON": "Sophon (released)",
         "INIT_BN": "untrained network"}
OUTPUT_SCORES = "(output scores)"        # Sophon's X->bb discriminant, summed from the network's output scores
REJ30 = r"$\mathrm{Rej}_{30\%}$"         # ParT's notation
PERCENT = 100.0                          # PI 2026-11-25 (ledger 32.3): accuracy in percent on every accuracy axis
SUMMARY_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "L162_MASS", "R16_Q1_MASS")     # Table 2's pretrained rows
PAIR_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND_MERGED", "RAND_APART")
SIZE_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "SCRATCH")
SIZE_PANELS = tuple((key, ds, met, {"qg": "Quark/gluon"}.get(key, title)) for key, ds, met, title in G.SIZE5
                    if key != "qg_herwig")
SIZE_Y = {"accuracy": "Accuracy (%)", "r30": f"Rejection ({REJ30})"}
XBX_NETS = ("L188", "L162", "R42_Q1", "R16_Q1")
X_EPS_B = r"Background efficiency ($\varepsilon_B$)"
RECALL_ROWS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))
VAL_LEGEND_W = 5.6                        # cm: "162 classes + mass regression" and its key beside Fig. A5's panel
FADED = 0.3            # a class within three standard deviations of equal recall: faded (editor 8f085fe, item 48)
WITHIN_FADED = "faded: within three s.d."
LOWER = 0.4                               # a lower panel's height over its panel's (relative to 188 classes)
DOT, BAR = 4.0, 1.0                       # pt: a mean's point and its standard-deviation bar in the point plots


# ledger 35, section 4: figure text at the caption's size or one step smaller; the captions are set \small (10.95 pt
# in the 12 pt class), the figures at \footnotesize, 10 pt, every label, tick, title and legend
FONT = 10.0
FONT_KEYS = ("LABEL_PT5", "TICK_PT5", "TITLE_PT5", "LEGEND_PT5")


@contextlib.contextmanager
def names():
    """The rewrite's names in every legend the gallery's parts draw, and its font size, for the length of a drawing."""
    legend, tag = dict(S.LEGEND), PH.DIRECT_TAG
    fonts = {k: getattr(S, k) for k in FONT_KEYS}
    S.LEGEND.update(NAMES)
    PH.DIRECT_TAG = OUTPUT_SCORES
    for k in FONT_KEYS:
        setattr(S, k, FONT)
    try:
        yield
    finally:
        S.LEGEND.clear()
        S.LEGEND.update(legend)
        PH.DIRECT_TAG = tag
        for k, v in fonts.items():
            setattr(S, k, v)


def v2_style() -> None:
    """style_v5's paper style at FONT (call inside names(), which sets the size constants it reads)."""
    S.use_style5(simplify=True)


@contextlib.contextmanager
def before_save(hook):
    """hook(fig) runs on a figure the gallery's parts draw, before it is saved."""
    save = S.save

    def with_hook(fig, out):
        hook(fig)
        return save(fig, out)
    S.save = with_hook
    try:
        yield
    finally:
        S.save = save


def _pct(value):
    """A value function of gallery_labelset (_hv_values' value(m)) in percent."""
    return lambda m: None if (x := value(m)) is None else PERCENT * x


def _pct_series(maker):
    """A training-size series maker (gallery_labelset._tuned, _probe) with its accuracy panels in percent."""
    def make(key, ds, met):
        series, line, mark, label = maker(key, ds, met)
        if met == "accuracy":
            series = {n: [PERCENT * v for v in vals] for n, vals in series.items()}
        return series, line, mark, label
    return make


def _msd(v) -> tuple:
    v = [float(x) for x in v]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _point(ax, x, y, v, colour, filled=True, marker="o", horizontal=False, single_open=True, half=False) -> None:
    """A mean over runs with a bar of their standard deviation; filled False (or, with single_open, a single run) an open
    marker; half a marker filled on its left half only."""
    m, sd = _msd(v)
    if sd:
        kw = {"xerr" if horizontal else "yerr": sd}
        ax.errorbar(x if not horizontal else m, y if horizontal else m, **kw, fmt="none", ecolor=colour,
                    elinewidth=BAR, capsize=0, zorder=2.5)
    open_ = (single_open and len(v) == 1) or not filled
    size = DOT * (S.STAR_SCALE if marker == "*" else 1.0)
    style = ({"fillstyle": "left", "markerfacecolor": colour, "markerfacecoloralt": "white"} if half and not open_
             else {"markerfacecolor": "white" if open_ else colour})
    ax.plot([m if horizontal else x], [y if horizontal else m], linestyle="none", marker=marker, markersize=size,
            markeredgecolor=colour, markeredgewidth=0.9, zorder=2.7, **style)


# ------------------------------------------------------------------ Fig. 1: the label sets and the pipeline
FIG1_H = 7.9                # cm: about a third of the text height (ledger 35, section 4: "about 40% of a page")
GREY_LINE, BOX_EDGE = "#9AA3B5", "#4A4A4A"
FIG1_GROUP_NAME = {"2P_HAD_2PARTON|B": r"$b$ quark", "2P_HAD_2PARTON|C": r"$c$ quark", "2P_HAD_2PARTON|LG": "light"}
FIG1_TASKS = ("JetClass-II", "JetClass", "Top tagging", "Quark/gluon", r"$X\to b\bar b$, $X\to bs$")


def fig1_classes() -> list:
    """The ten classes of the first 17-class group (a resonance decaying to two quarks or gluons) in the label maps'
    order, each with its group under the 162, 43 and 17 classes: [(class name, {label set: group})]."""
    import csv
    rows = list(csv.DictReader((C.P.REPO / "configs/labelmaps/rung_label_maps.v1.csv").open()))
    group = [r for r in rows if r["R16_Q1_name"] == "2P_HAD_2PARTON"]
    group.sort(key=lambda r: (int(r["R42_Q1"]), int(r["L188"])))
    if len(group) != 10 or {r["R42_Q1_name"] for r in group} != set(FIG1_GROUP_NAME):
        raise SystemExit(f"FATAL: the example group has {len(group)} classes, not the ten of Fig. 1")
    return [(r["class_name"], {k: r[k] for k in ("L188", "L162", "R42_Q1", "R16_Q1", "R42_Q1_name")}) for r in group]


def _class_tex(name: str) -> str:
    f = name.removeprefix("label_X_")
    return rf"${f[0]}\bar{{{f[1]}}}$" if f[0] == f[1] and f[0] != "g" else rf"${f}$"


def draw_fig1(path: pathlib.Path) -> None:
    """(a) A four-level tree for one group of decays: the ten classes as text along the top, a node per class of each
    label set (in its colour) joined to its parent by thin lines, the three 43-class groups named; (b) one row,
    jets -> network -> features -> frozen-feature classifier | fine-tuning -> the five tasks. Under 60 words."""
    fig = S.figure_cm(S.PRINT_W, FIG1_H)
    ax = S.axes_cm(fig, 0, 0, S.PRINT_W, FIG1_H)
    ax.set_xlim(0, S.PRINT_W)
    ax.set_ylim(FIG1_H, 0)                         # y in cm from the top
    ax.axis("off")
    txt = {"fontsize": FONT, "va": "center"}
    classes = fig1_classes()
    x0, x1 = 3.4, 14.4
    xs = list(np.linspace(x0, x1, len(classes)))
    y_cls, rows = 0.95, {"L188": 1.55, "L162": 2.15, "R42_Q1": 2.95, "R16_Q1": 4.2}
    name = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes"}
    ax.text(0.1, 0.3, "(a) Label sets", fontsize=FONT, va="center")
    ax.text(x0 - 0.55, y_cls, r"$X\to$", ha="right", **txt)
    for x, (c, _) in zip(xs, classes):
        ax.text(x, y_cls, _class_tex(c), ha="center", **txt)
    sq = 0.17
    pos = {}
    for key, y in rows.items():
        ax.text(0.1, y, name[key], ha="left", **txt)
        groups = {}
        for x, (_, g) in zip(xs, classes):
            groups.setdefault(g[key], []).append(x)
        pos[key] = {gid: float(np.mean(v)) for gid, v in groups.items()}
        for gid, x in pos[key].items():
            ax.add_patch(matplotlib.patches.Rectangle((x - sq / 2, y - sq / 2), sq, sq, fc=S.SERIES[key], ec="none",
                                                      zorder=3))
    order = list(rows)
    for upper, lower in zip(order, order[1:]):    # each node joined to its parent once
        ya, yb = rows[upper] + sq / 2, rows[lower] - sq / 2
        ymid = (ya + yb) / 2 + (0.15 if lower == "R16_Q1" else 0)
        for gid, xa in pos[upper].items():
            g = next(g for _, g in classes if g[upper] == gid)
            xb = pos[lower][g[lower]]
            ax.plot([xa, xa, xb, xb], [ya, ymid, ymid, yb], color=GREY_LINE, linewidth=0.7, zorder=1,
                    solid_joinstyle="miter")
    for gid, x in pos["R42_Q1"].items():          # the 43-class groups named beside their nodes
        nm = next(g["R42_Q1_name"] for _, g in classes if g["R42_Q1"] == gid)
        ax.text(x + 0.2, rows["R42_Q1"] + 0.36, FIG1_GROUP_NAME[nm], ha="left", **txt)
    # (b) the pipeline, one row
    yb_, hb = 6.45, 0.75
    ax.text(0.1, 5.15, "(b) Pretraining and use", fontsize=FONT, va="center")

    def box(x, w, y, h, text, face="white"):
        ax.add_patch(matplotlib.patches.Rectangle((x, y - h / 2), w, h, fc=face, ec=BOX_EDGE, lw=0.7, zorder=2))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=FONT, linespacing=1.15, zorder=3)

    def arrow(p, q):
        ax.add_patch(matplotlib.patches.FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                                        color=BOX_EDGE, shrinkA=0, shrinkB=0, zorder=2))
    box(0.1, 2.2, yb_, hb, "JetClass-II jets")
    arrow((2.3, yb_), (2.75, yb_))
    box(2.8, 2.65, yb_, 1.5, "Particle\nTransformer\n(one label set)", face="#F2F2F2")
    arrow((5.45, yb_ + 0.45), (5.9, yb_ + 0.45))
    box(5.95, 1.5, yb_ + 0.45, hb, "Features")
    up, dn = yb_ + 0.45, yb_ - 0.6                  # y grows downward: the classifier below fine-tuning
    box(8.0, 2.55, up, 0.95, "Frozen-feature\nclassifier")
    box(8.0, 2.55, dn, hb, "Fine-tuning")
    arrow((7.45, up), (7.95, up))
    arrow((5.45, dn), (7.95, dn))
    xt = 11.15
    box(xt, S.PRINT_W - xt - 0.1, yb_ - 0.08, 2.15, "\n".join(FIG1_TASKS))
    arrow((10.55, up), (xt - 0.05, up))
    arrow((10.55, dn), (xt - 0.05, dn))
    fig.exact_size = True
    S.save(fig, path)


def fig1_words() -> int:
    """The words Fig. 1 prints (ledger 35, section 4: under 60)."""
    words = ["(a)", "Label", "sets", "X"] + [""] * 10 + "188 classes 162 classes 43 classes 17 classes".split()
    words += ["b", "quark", "c", "quark", "light", "(b)", "Pretraining", "and", "use", "JetClass-II", "jets",
              "Particle", "Transformer", "(one", "label", "set)", "Features", "Frozen-feature", "classifier",
              "Fine-tuning"] + " ".join(FIG1_TASKS).split()
    return len(words)


# ------------------------------------------------------------------ Fig. 2: accuracy for every label set, a dot plot
# ledger 35, section 4: the label sets are categories (they can be built many ways), so no line joins them; families
# apart (nested; the random groupings, each draw a point beside their mean; the flavor-boundary groupings)
ACC_COLS = (("L188", 0.0), ("L162", 1.0), ("R42_Q1", 2.0), ("R16_Q1", 3.0), ("RAND", 4.5), ("FLAV_F0", 6.0),
            ("FLAV_F1", 7.0), ("FLAV_F1R", 8.0))
ACC_TICKS = ("188", "162", "43", "17", "random", "no\nflavor", "$b$\nquark", "random\nflavor")
ACC_FAMILIES = (("nested", 0.0, 3.0), ("flavor boundary", 6.0, 8.0))
ACC_SEP = (3.75, 5.25)
ACC_PANELS = (("jc2", 1_000_000, "jc2_188", "JetClass-II"), ("jc1", 1_000_000, "jc1", "JetClass"))
ACC_MODES = (("probe", "Linear probe (%)", False), ("ft", "Fine-tuned (%)", True))
ACC_W, ACC_LEFT, ACC_WGAP, ACC_SUB, ACC_GAP = 6.1, 1.4, 1.45, 2.85, 0.25     # cm; ACC_GAP between frozen and fine-tuned
RUN_DOT, RUN_ALPHA, MEAN_DOT = 2.4, 0.45, 5.0


def accuracy_values(key: str, n: int, ds: str) -> dict:
    """{"probe": {label set: runs}, "ft": {label set: runs}, "scratch": runs}, in percent: Table 2's linear probes and the
    fine-tuned cells at n jets (Table 3, the appendix's full table for the flavor-boundary groupings), and the network
    trained from scratch, through the tables' value functions."""
    cell = (key, n, ds, "accuracy")
    out = {mode: {r: G._hv_values(r, _pct(fn(*cell)), cell) for r, _ in ACC_COLS}
           for mode, fn in (("probe", G._probe_value), ("ft", G._ft_value))}
    out["scratch"] = [PERCENT * v for v in G._scratch_value(*cell)]
    if not out["scratch"] or not all(v for d in (out["probe"], out["ft"]) for v in d.values()):
        raise C.Absent(f"accuracy for every label set, {key}: a cell without runs")
    return out


def _acc_column(ax, x: float, v: list, colour: str, filled: bool) -> list:
    """One label set: each run (or random draw) a faint dot, the mean a larger marker with its SD bar."""
    k = len(v)
    offs = np.linspace(-0.16, 0.16, k) if k > 1 else [0.0]
    ax.plot(x + np.asarray(offs) + 0.22, v, linestyle="none", marker="o", markersize=RUN_DOT, markeredgewidth=0,
            color=colour, alpha=RUN_ALPHA, zorder=2.2)
    m, sd = _msd(v)
    if sd:
        ax.errorbar([x], [m], yerr=sd, fmt="none", ecolor=colour, elinewidth=BAR, capsize=0, zorder=2.5)
    ax.plot([x], [m], linestyle="none", marker="o", markersize=MEAN_DOT, markerfacecolor=colour if filled else "white",
            markeredgecolor=colour, markeredgewidth=1.0, zorder=2.7)
    return [min(v), max(v), m - sd, m + sd]


def draw_fig2(path: pathlib.Path) -> None:
    """Accuracy (%) for every label set on JetClass-II (a) and JetClass (b): the linear probe on the frozen features
    (open, upper part) and the models fine-tuned on 10^6 jets (filled, lower part), each part its own range; the
    network trained from scratch a dashed line with its band in the fine-tuned part."""
    top = S.LEGEND_TOP + 0.5 * 2 + 0.55
    H = top + 2 * ACC_SUB + ACC_GAP + 1.75
    fig = S.figure_cm(S.PRINT_W, H)
    for k, (key, n, ds, title) in enumerate(ACC_PANELS):
        vals = accuracy_values(key, n, ds)
        x = ACC_LEFT + k * (ACC_W + ACC_WGAP)
        parts = []
        for i, (mode, ylabel, filled) in enumerate(ACC_MODES):
            ax = S.axes_cm(fig, x, top + i * (ACC_SUB + ACC_GAP), ACC_W, ACC_SUB, **({"sharex": parts[0]} if parts else {}))
            if parts:
                ax.set_label("<break>")             # figcheck: one panel in two parts
            parts.append(ax)
            ends = []
            for key_, xc in ACC_COLS:
                ends += _acc_column(ax, xc, vals[mode][key_], S.SERIES[key_], filled)
            if mode == "ft":
                m, sd = _msd(vals["scratch"])
                ax.axhspan(m - sd, m + sd, color=S.SERIES["SCRATCH"], alpha=L.BAND, linewidth=0, zorder=0.9)
                ax.axhline(m, color=S.SERIES["SCRATCH"], linestyle="--", linewidth=S.LW_CURVE, zorder=1.0)
                ends += [m - sd, m + sd]
            lo, hi = min(ends), max(ends)
            d = 0.1 * (hi - lo)
            ax.set_ylim(lo - d, hi + d)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.set_xlim(-0.55, ACC_COLS[-1][1] + 0.6)
            ax.xaxis.set_major_locator(ticker.FixedLocator([c for _, c in ACC_COLS]))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            for xsep in ACC_SEP:
                ax.axvline(xsep, color=S.GREY_MID, linewidth=S.LW_THIN, zorder=0.5)
            S.fine_grid(ax)
            S.labels(ax, ylabel=ylabel)
        parts[0].tick_params(axis="x", which="both", labelbottom=False)
        parts[1].xaxis.set_major_formatter(ticker.FixedFormatter(ACC_TICKS))
        S.clear_edge_labels(parts[0], low=True)
        S.clear_edge_labels(parts[1], high=True)
        S.title(parts[0], f"({'ab'[k]}) {title}")
        yfam = top + 2 * ACC_SUB + ACC_GAP + 1.2
        for name, a, b in ACC_FAMILIES:
            xc = x + ACC_W * ((a + b) / 2 + 0.55) / (ACC_COLS[-1][1] + 1.15)
            fig.text(xc / S.PRINT_W, 1 - yfam / H, name, ha="center", va="center", fontsize=FONT)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=MEAN_DOT, markerfacecolor="white",
                      markeredgecolor="#000000", markeredgewidth=1.0),
               Line2D([], [], linestyle="none", marker="o", markersize=MEAN_DOT, markerfacecolor="#000000",
                      markeredgecolor="#000000"),
               Line2D([], [], linestyle="none", marker="o", markersize=RUN_DOT, markeredgewidth=0, color="#000000",
                      alpha=RUN_ALPHA),
               (Patch(facecolor=S.SERIES["SCRATCH"], alpha=L.BAND, linewidth=0),
                Line2D([], [], color=S.SERIES["SCRATCH"], linestyle="--", linewidth=S.LW_CURVE))]
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=2,
                 labels=["linear probe, mean ± s.d.", "fine-tuned, mean ± s.d.", "single run or draw",
                         S.LEGEND["SCRATCH"]])
    S.save(fig, path)


# ------------------------------------------------------------------ the separation figure: accuracy against S(L)
SEP_CSV = C.P.DATA / "stats" / "a1_separation.csv"
SEP_KEY = {"RAND2_p1": "RAND", "RAND2_p2": "RAND", "RAND2_p3": "RAND", "RAND2_p4": "RAND", "RAND2_p5": "RAND"}
SEP_ORDER = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "FLAV_F0", "FLAV_F1", "FLAV_F1R")


def separation_rows() -> list:
    """[(label set, its colour key, S, accuracy %, SD %)] of the build's file stats/a1_separation.csv (twelve label
    sets: nested, five random groupings, three flavor-boundary groupings)."""
    import csv
    rows = list(csv.DictReader(SEP_CSV.open()))
    if len(rows) != 12:
        raise SystemExit(f"FATAL: {SEP_CSV.name} has {len(rows)} label sets, not 12")
    return [(r["label_set"], SEP_KEY.get(r["label_set"], r["label_set"]), float(r["S"]), PERCENT * float(r["accuracy"]),
             PERCENT * float(r["accuracy_sd"]) if r["accuracy_sd"] else 0.0) for r in rows]


def draw_fig_separation(path: pathlib.Path) -> None:
    """Frozen 188-class linear-probe accuracy on JetClass-II against the confusion-weighted separation S of each label
    set, every pretrained label set a point in its colour and marker with its SD bar; the legend beside the panel."""
    w, h, left, top = 7.4, 0.72 * 7.4, 1.5, 0.35
    fig = S.figure_cm(S.PRINT_W, top + h + 1.15)
    ax = S.axes_cm(fig, left, top, w, h)
    S.fine_grid(ax)
    xs, ys = [], []
    for _, key, s, acc, sd in separation_rows():
        if sd:
            ax.errorbar([s], [acc], yerr=sd, fmt="none", ecolor=S.SERIES[key], elinewidth=BAR, capsize=0, zorder=2.5)
        ax.plot([s], [acc], linestyle="none", marker=S.MARKERS[key], markersize=MEAN_DOT * (S.STAR_SCALE if S.MARKERS[key] == "*" else 1),
                markerfacecolor=S.SERIES[key], markeredgecolor=S.SERIES[key], zorder=2.7)
        xs += [s]
        ys += [acc - sd, acc + sd]
    ax.set_xlim(min(xs) - 0.04, max(xs) + 0.04)
    d = 0.08 * (max(ys) - min(ys))
    ax.set_ylim(min(ys) - d, max(ys) + d)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    S.labels(ax, xlabel=r"Confusion-weighted separation $S$", ylabel="Linear probe, JetClass-II (%)")
    handles = [Line2D([], [], linestyle="none", marker=S.MARKERS[k], markersize=MEAN_DOT, color=S.SERIES[k],
                      label=S.LEGEND[k]) for k in SEP_ORDER]
    x = left + w + 0.5
    S.slot_legend(fig, handles, x, top, S.PRINT_W - x - 0.1, h)
    S.save(fig, path)


# ------------------------------------------------------------------ Fig. 3 and Fig. A1: classes and pairs of classes
def _recall(fig, slots: list, rows: tuple) -> list:
    """gallery_perclass._draw_grid's panel for linear probes only (GO-9), one per (compared, reference) of rows at
    slots [(x, top, w)] cm: each class's recall (%) against its recall under the reference, means over runs with bars of
    the run spread and the bootstrap over test jets, coloured by what the panel's labels merge the class with; a class
    within three standard deviations of the reference is open. One range for every panel. Returns the legend's handles."""
    get = PC.SOURCES["jc2"][0]
    if not PC._mean(get, "L188"):
        raise C.Absent(PC.ABSENT)

    def side(row):
        m, sd, bt = PC._mean(get, row), PC._sd(get, row), PC._boot_mean("jc2", 0, row)
        return {c: (PERCENT * m[c], PERCENT * float(np.hypot(sd[c], bt.get(c, 0.0)))) for c in m}
    ref = {b: side(b) for b in {b for _, b in rows}}
    pts = [[(ref[b][c][0], v[0], c, ref[b][c][1], v[1]) for c, v in side(a).items() if c in ref[b]] for a, b in rows]
    vals = [v for panel in pts for x, y, _, ex, ey in panel for v in (x - ex, y - ey, x + ex, y + ey)]
    pad = 0.04 * (max(vals) - min(vals))
    lim = (min(vals) - pad, max(vals) + pad)
    looks = [PC._look("jc2", a) for a, _ in rows]
    for k, ((a, b), panel, (look, _), (x0, y0, w)) in enumerate(zip(rows, pts, looks, slots)):
        ax = S.axes_cm(fig, x0, y0, w, w)
        S.fine_grid(ax)
        ax.plot(lim, lim, **S.REFERENCE)
        marked = PC._significant("jc2", 0, (a, b))
        for x, y, c, ex, ey in panel:
            colour, marker, _, z = look(c)
            ax.errorbar(x, y, xerr=ex or None, yerr=ey or None, fmt="none", ecolor=colour, elinewidth=0.4, capsize=0,
                        alpha=0.6, zorder=z - 0.1)
            within = marked is not None and c not in marked
            ax.plot(x, y, color=colour, marker=marker, markersize=PC.DOT[2], linestyle="none", zorder=z,
                    markerfacecolor=colour, markeredgewidth=0, alpha=FADED if within else 1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        S.title(ax, f"({'abcdefg'[k]}) {S.LEGEND[a]}")
        S.labels(ax, xlabel=f"Recall (%), {S.LEGEND[b]}", ylabel=f"Recall (%), {S.LEGEND[a]}")
    handles = PC._handles("jc2", set().union(*(kinds for _, kinds in looks)))
    return handles + [Line2D([], [], color="#777777", marker="o", markersize=3, markeredgewidth=0, alpha=FADED,
                             linestyle="none", label=WITHIN_FADED),
                      Line2D([], [], **S.REFERENCE, label="equal recall")]


# PI 2026-11-25 (ledger 33.2(b)): X->gg against X->qq, a binary linear probe on the class-balanced test jets; the build
# exports its AUC per model and no curve (stats/ledger33_exports.json)
GG_QQ = C.P.DATA / "stats" / "ledger33_exports.json"
GG_QQ_TITLE = r"$X\to gg$ vs $q\bar{q}$"
GG_QQ_WHERE = {"L188": ("arms", "L188"), "L162": ("arms", "L162"), "R42_Q1": ("arms", "R42_Q1"),
               "R16_Q1": ("arms", "R16_Q1"), "RAND_MERGED": ("random_merged",), "RAND_APART": ("random_apart",)}
# Fig. 3(c): the pairs, one row each, in the family's order with X->gg vs X->qq after the resonance pairs
PAIRS_C = ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq", "gg_vs_qq", "qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light")


def gg_qq_values() -> dict:
    """{network: AUCs of its runs} of X->gg vs X->qq from the export, each list checked against its per-model values."""
    doc = json.loads(GG_QQ.read_text())["gg_vs_qq"]
    per = set(doc["per_model"].values())
    out = {}
    for net, where in GG_QQ_WHERE.items():
        node = doc
        for k in where:
            node = node[k]
        runs = [float(v) for v in node["runs"]]
        if not runs or not set(runs) <= per:
            raise SystemExit(f"FATAL: {GG_QQ.name}: gg_vs_qq {'/'.join(where)} runs not among its per-model AUCs")
        out[net] = runs
    return out


def pair_errors() -> dict:
    """{pair: {network: 1 - AUC of each run}}: the binary linear probe's AUC of the pair curves (pairs family, the
    curves of Figs. A2 and A6) and of X->gg vs X->qq (gg_qq_values); a network with no run of a kind (no random grouping
    keeps a QCD pair apart) is left out."""
    docs, out = R._family("pairs"), {}
    for k in PAIRS_C:
        if k == "gg_vs_qq":
            out[k] = {net: [1 - v for v in runs] for net, runs in gg_qq_values().items()}
            continue
        out[k] = {net: v for net in PAIR_NETS
                  if (v := [1 - float(docs[m]["curves"][k]["auc"]) for m in R._models(docs, net, k)
                            if m in docs and k in docs[m]["curves"]])}
    return out


def pair_status(k: str, net: str) -> str:
    """"merged", "partly" or "apart": whether a network's labels put every class of the pair's two sides in one class,
    some of them, or none (the label maps, mlst_results.merge_status; the export's status for X->gg vs X->qq; the random
    groupings by their curve, a grouping that merges part of the pair being in neither)."""
    if net in ("RAND_MERGED", "RAND_APART"):
        return "merged" if net == "RAND_MERGED" else "apart"
    if k == "gg_vs_qq":
        return json.loads(GG_QQ.read_text())["gg_vs_qq"]["status_of_label_sets"][net]
    return R.PAIR_STATUS[net][k]


def pair_merged(k: str, net: str) -> bool:
    return pair_status(k, net) == "merged"


def _pair_title(k: str) -> str:
    return GG_QQ_TITLE if k == "gg_vs_qq" else R.PAIR_TITLE[k]


def pair_relative(errs: dict) -> dict:
    """{pair: {network: each run's 1 - AUC over the 188-class model's mean}}."""
    return {k: {net: [x / _msd(e["L188"])[0] for x in v] for net, v in e.items()} for k, e in errs.items()}


def _pairs_panel(ax) -> list:
    """Fig. 3(c): one row a pair, 1 - AUC relative to the 188-class model on a log axis (editor 8f085fe, item 44: a
    change of a seventh, as on X->gg vs X->qq, reads where the absolute values span two decades), the 188-class value in
    the row name; one marker a network (its shape and colour), dodged within the row, the mean over runs with its
    standard deviation; open where the labels merge the pair, half-filled where they merge part of it. Returns the
    legend's handles."""
    errs = pair_errors()
    rel = pair_relative(errs)
    dodge = 0.13
    xs = []
    for i, k in enumerate(PAIRS_C):
        for j, net in enumerate(PAIR_NETS):
            if net not in rel[k]:
                continue
            key, status = R._style(net), pair_status(k, net)
            _point(ax, None, i + (j - (len(PAIR_NETS) - 1) / 2) * dodge, rel[k][net], S.SERIES[key],
                   marker=S.MARKERS.get(net, S.MARKERS[key]), horizontal=True, filled=status != "merged",
                   single_open=False, half=status == "partly")
            m, sd = _msd(rel[k][net])
            xs += [m - sd, m + sd]
    ax.axvline(1.0, **S.REFERENCE, zorder=1.5)
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.15, max(xs) * 1.15)
    S.log_axis(ax, "x", force=True, fine=True)
    ax.set_ylim(len(PAIRS_C) - 0.5, -0.5)
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(PAIRS_C))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(
        [f"{_pair_title(k)} ({_msd(errs[k]['L188'])[0]:.2g})" for k in PAIRS_C]))
    ax.tick_params(axis="y", which="both", length=0, labelsize=S.TICK_PT5)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    for y in np.arange(0.5, len(PAIRS_C) - 0.5):
        ax.axhline(y, color=S.GREY_LIGHT, linewidth=S.LW_THIN, zorder=0.5)
    ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.35, color="#b0b0b0", alpha=0.3)
    ax.set_axisbelow(True)
    S.labels(ax, xlabel=r"$1 - \mathrm{AUC}$, relative to 188 classes")
    mark = {"marker": "o", "markersize": DOT, "markeredgecolor": "#000000", "markeredgewidth": 0.9, "linestyle": "none"}
    return [Line2D([], [], linestyle="none", marker=S.MARKERS.get(net, S.MARKERS[R._style(net)]), markersize=DOT,
                   markerfacecolor=S.SERIES[R._style(net)], markeredgecolor=S.SERIES[R._style(net)],
                   label=R._name(net)) for net in PAIR_NETS] + [
        Line2D([], [], markerfacecolor="white", label="open: the labels merge the pair", **mark),
        Line2D([], [], fillstyle="left", markerfacecolor="#000000", markerfacecoloralt="white",
               label="half: they merge part of it", **mark)]


# ------------------------------------------------------------------ Fig. 3: recall by merge kind, and pairs of classes
MERGE_KINDS = (("Qcd", "QCD"), ("Quark", "quark\nflavor"), ("Lepton", "lepton\nflavor"), ("Both", "both"),
               ("Neither", "neither"))
MERGE_ARMS = (("R42_Q1", -0.12), ("R16_Q1", 0.12))


def recall_by_merge() -> dict:
    """{arm: {kind: (mean recall change in points, its standard deviation over runs, classes)}}, frozen JetClass-II,
    the 43- and 17-class models against the 188-class one, by what the label set merges a class with: the read-out's
    classification (experiments/FIGS/v2_rev35.recall_by_merge, the build's VRecall macros) with, per run, the mean over
    the kind's classes of the run's recall minus the 188-class mean; the SD over runs combines that spread with the
    188-class runs' spread of the same mean."""
    import csv
    R, root = M, C.P.REPO
    q, lep = R._orbits(root)
    groups = R._groups(root)
    names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    docs = {"lp": json.loads((C.P.DATA / "perclass_recall/probe_jc2.json").read_text())}
    ref = R.per_class_recall(docs, C.RUNS, "L188", "lp")
    out = {}
    for arm, _ in MERGE_ARMS:
        g = groups[arm]
        rec = R.per_class_recall(docs, C.RUNS, arm, "lp")

        def merged(c, key_of):
            n = names[c]
            return n in key_of and any(b != n and key_of[b] == key_of[n] and g[b] == g[n] for b in key_of)
        kinds = {k: [] for k, _ in MERGE_KINDS}
        for c in rec:
            if names[c].startswith("label_QCD"):
                kinds["Qcd"].append(c)
                continue
            mq, ml = merged(c, q), merged(c, lep)
            kinds["Both" if mq and ml else "Quark" if mq else "Lepton" if ml else "Neither"].append(c)
        out[arm] = {}
        for k, cs in kinds.items():
            ref_mean = {c: float(np.mean(ref[c])) for c in cs}
            per_run = [np.mean([100 * (rec[c][i] - ref_mean[c]) for c in cs]) for i in range(len(rec[cs[0]]))]
            ref_run = [np.mean([100 * ref[c][i] for c in cs]) for i in range(len(ref[cs[0]]))]
            sd = float(np.hypot(np.std(per_run, ddof=1), np.std(ref_run, ddof=1)))
            out[arm][k] = (float(np.mean(per_run)), sd, len(cs))
    return out


def _merge_panel(ax) -> None:
    """Fig. 3(a): the mean recall change by merge kind, the 43- and 17-class models side by side in their colours and
    markers, a bar their standard deviation over runs; a dotted line at no change."""
    data = recall_by_merge()
    for arm, dx in MERGE_ARMS:
        for i, (k, _) in enumerate(MERGE_KINDS):
            m, sd, _ = data[arm][k]
            ax.errorbar([i + dx], [m], yerr=sd, fmt="none", ecolor=S.SERIES[arm], elinewidth=BAR, capsize=0, zorder=2.5)
            ax.plot([i + dx], [m], linestyle="none", marker=S.MARKERS[arm], markersize=MEAN_DOT,
                    color=S.SERIES[arm], zorder=2.7)
    ax.axhline(0.0, **S.REFERENCE, zorder=1.5)
    ax.set_xlim(-0.5, len(MERGE_KINDS) - 0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(MERGE_KINDS))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([t for _, t in MERGE_KINDS]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    S.data_ylim(ax, pad=0.12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    S.fine_grid(ax)
    S.labels(ax, xlabel="Merged with", ylabel="Recall change (points)")


def draw_fig3(path: pathlib.Path) -> None:
    """(a) the mean change of per-class recall by merge kind (_merge_panel); (b) the pairs of classes (_pairs_panel),
    the row names left of it. One legend beside (a): the label sets of both panels and the marker fills of (b)."""
    wa, ha = 6.1, 0.72 * 6.1
    top_a, left = 0.6, 1.5
    y_b = top_a + ha + 1.75
    xb, wb = 6.4, S.PRINT_W - 6.4 - 0.15
    hb = 0.72 * wb
    H = y_b + hb + 1.0
    fig = S.figure_cm(S.PRINT_W, H)
    a = S.axes_cm(fig, left, top_a, wa, ha)
    _merge_panel(a)
    S.title(a, "(a) Per-class recall")
    b = S.axes_cm(fig, xb, y_b, wb, hb)
    handles = _pairs_panel(b)
    S.title(b, "(b) Pairs of classes")
    import textwrap
    x = left + wa + 0.5
    S.slot_legend(fig, handles, x, top_a, S.PRINT_W - x - 0.1, ha,
                  labels=[textwrap.fill(h.get_label(), 26) for h in handles])
    S.save(fig, path)


def draw_figA1(path: pathlib.Path) -> None:
    """The per-class recall scatters (162, 43, 17 against 188 classes; moved from Fig. 3, ledger 35), two a row, the
    legend in the fourth slot."""
    w, left, wgap, top, hgap = 6.15, 1.25, 1.3, 0.6, 1.5
    fig = S.figure_cm(S.PRINT_W, top + 2 * w + hgap + 0.95)
    slots = [(left + j * (w + wgap), top + i * (w + hgap), w) for i, j in ((0, 0), (0, 1), (1, 0))]
    handles = _recall(fig, slots, RECALL_ROWS)
    S.slot_legend(fig, handles, left + w + wgap, top + w + hgap, w, w)
    S.save(fig, path)


# ------------------------------------------------------------------ Figs. A2 and A6: the pair curves
A2_RATIO = 0.5          # ledger 34.3: taller ratio panels than the main figures' (0.35)


def _start_at_188(fig) -> None:
    """Each pair panel's x range starts where its 188-class curve does (ledger 34.3), every curve cut there."""
    ref = S.SERIES["L188"].lower()
    for ax in fig.axes:
        if not re.match(r"\([a-z]\) ", ax.get_title()):
            continue
        xs = [float(np.nanmin(ln.get_xdata())) for ln in ax.lines
              if ln.get_color().lower() == ref and ln.get_transform() == ax.transData and len(ln.get_xdata())]
        if not xs:
            continue
        start = min(xs)                          # where fewer than 20 background jets pass the 188-class model
        lo = math.floor(start * 20 + 1e-9) / 20      # the first tick at or below it
        pad = 0.02 * (1 - lo)
        group = [o for o in fig.axes if o is ax or ax.get_shared_x_axes().joined(ax, o)]
        for o in group:
            for ln in o.lines:
                if ln.get_transform() == o.transData and len(ln.get_xdata()) > 1:
                    x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                    keep = x >= start - 1e-9             # every curve from the same efficiency
                    ln.set_data(x[keep], y[keep])
        ax.set_xlim(lo - pad, 1 + pad)
        step = 0.1 if 1 - lo > 0.35 else 0.05
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(step / 2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda t, _, d=1 if step == 0.1 else 2: f"{t:.{d}f}" if t <= 1 + 1e-9 else ""))


def _pair_figure(path: pathlib.Path, keys: tuple) -> None:
    """The class-pair form (gallery_roc._roc_grid: rejection against signal efficiency, the ratio to 188 classes under
    each panel with every curve's band) for three pairs, two a row, the legend in the fourth slot."""
    docs = R._family("pairs")
    grid = [[(docs, k, PAIR_NETS, R.PAIR_TITLE[k]) for k in keys[:2]], [(docs, keys[2], PAIR_NETS, R.PAIR_TITLE[keys[2]]),
                                                                        None]]
    with before_save(_start_at_188):
        R._roc_grid(path, grid, w=R.PAIR_W[2], aspect=0.72, ratio_scale="log", own_y=True, wgap=R.PAIR_WGAP,
                    left=R.PAIR_LEFT - 0.35, legend_slot=(1, 1), bands=PAIR_NETS, ratio=A2_RATIO)


def draw_figA2(path: pathlib.Path) -> None:
    _pair_figure(path, ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq"))


def draw_figA6(path: pathlib.Path) -> None:
    _pair_figure(path, ("qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light"))


# ------------------------------------------------------------------ Fig. 4: the resonance benchmark
def draw_fig5(path: pathlib.Path) -> None:
    """Background rejection 1/eps_B (log) against signal efficiency for X->bb and X->bs (ParT's and Sophon's axes),
    from the curves of gallery_physics._xbx_series: a classifier trained on the frozen features of each label set,
    Sophon's network trained from scratch, and for X->bb the output scores of the label sets that keep it as a class
    (dash-dot); every curve the mean over runs with its standard-deviation band (the from-scratch band as is: the
    read-out found no run a rule removes, ledger 35 V2). Under each panel the ratio to the 188-class model; a dotted
    line at 1/eps_B = 100, the 1% background efficiency of the text."""
    XR = C.P._xr()
    panels = [PH._xbx_series(XR, sig, XBX_NETS, PH.CLASSIFIER, True, True) for sig, _ in XR.SIGNALS]
    fig, grid, rgrid = S.grid_cm(1, 2, w=6.1, aspect=0.85, ratio=0.35, top=S.LEGEND_TOP + 0.5 * 2 + 0.5,
                                 left=1.35, wgap=1.45, bottom=1.2)
    axes, raxes, low, rlo = list(grid[0]), list(rgrid[0]), [], []
    eps = np.asarray(C.P.EPS_S_GRID, float)
    for ax, rx, series in zip(axes, raxes, panels):
        curves = {}
        for key, label, style, rs in series:
            _, _, rej = C.P.xbx_curves(M, rs)
            ok = np.isfinite(rej).all(axis=0)
            mean = rej.mean(axis=0)
            sd = rej.std(axis=0, ddof=1) if len(rej) > 1 else np.zeros_like(mean)
            curves[key] = (eps[ok], mean[ok], sd[ok], style)
        ref = dict(zip(np.round(curves["L188"][0], 9), curves["L188"][1]))
        for key, label, style, _ in reversed(series):
            x, mean, sd, st = curves[key]
            ax.fill_between(x, mean - sd, mean + sd, color=st["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            ax.plot(x, mean, **{**st, "linewidth": S.LW_CURVE})
            low += [v for v in mean - sd if v > 0]
            keep = [i for i, xx in enumerate(np.round(x, 9)) if xx in ref]
            r = np.array([mean[i] / ref[np.round(x[i], 9)] for i in keep])
            rs_ = np.array([sd[i] / ref[np.round(x[i], 9)] for i in keep])
            xr = x[keep]
            rx.fill_between(xr, r - rs_, r + rs_, color=st["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            rx.plot(xr, r, **{**st, "linewidth": S.LW_CURVE})
            rlo += list(r - rs_) + list(r + rs_)
        ax.axhline(100.0, **S.REFERENCE, zorder=1.5)
        rx.axhline(1.0, color="#000000", linewidth=S.LW_THIN, zorder=1.4)
    for ax, rx, what in zip(axes, raxes, (r"$X\rightarrow b\bar{b}$", r"$X\rightarrow bs$")):
        S.fine_grid(ax)
        S.fine_grid(rx)
        ax.set_xlim(eps.min() - 0.02, 1.02)
        ax.set_yscale("log")
        ax.set_ylim(min(low) / 1.3, max(v for a_ in axes for ln in a_.lines for v in ln.get_ydata()) * 1.3)
        S.log_axis(ax, "y", force=True)
        S.clear_edge_labels(ax, low=True)
        S.title(ax, what)
        S.labels(ax, ylabel=r"Rejection ($1/\varepsilon_B$)")
        S.labels(rx, xlabel=S.X_EPS, ylabel="Ratio")
        rx.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    v = np.array([x for x in rlo if np.isfinite(x)] + [1.0])
    d = 0.08 * (v.max() - v.min())
    for rx in raxes:
        rx.set_ylim(max(v.min() - d, 0.0), v.max() + d)
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        S.clear_edge_labels(rx, high=True)
    S.letter_panels(axes)
    nets = [k for k in ("L188", "L162", "R42_Q1", "R16_Q1", "SCRATCH")]
    handles = [Line2D([], [], **{**S.look(k), "linewidth": S.LW_CURVE}) for k in nets]
    handles.append(Line2D([], [], color="#000000", linestyle=S.MODE_LINE["direct"], linewidth=S.LW_CURVE))
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=3, labels=[S.LEGEND[k] for k in nets] + ["output scores"])
    S.save(fig, path)


# ------------------------------------------------------------------ Fig. 5, Figs. A3 and A7: against training size
def _relative(series: dict, ref: dict, met: str) -> tuple:
    """(n, value, its uncertainty) of a series relative to 188 classes at the sizes both hold: the difference in
    percentage points for accuracy, the ratio for rejection; the two run spreads combined."""
    ns = sorted(set(series) & set(ref))
    a = np.array([_msd(series[n]) for n in ns])
    b = np.array([_msd(ref[n]) for n in ns])
    if met == "accuracy":
        return ns, a[:, 0] - b[:, 0], np.hypot(a[:, 1], b[:, 1])
    r = a[:, 0] / b[:, 0]
    return ns, r, r * np.hypot(a[:, 1] / a[:, 0], b[:, 1] / b[:, 0])


def _size_figure(path: pathlib.Path, makers: list, nets: tuple, panels: tuple, ncol: int) -> None:
    """gallery_labelset._draw_size's panels (the mean over runs against the number of training jets, a band its
    standard deviation) for the series of makers in panels, two a row, the legend of nets above in ncol columns; under
    each panel the label sets relative to 188 classes (_relative), trained from scratch left out of it."""
    rows = -(-len(nets) // ncol)
    w, h = G.SIZE_W, G.SIZE_ASPECT * G.SIZE_W
    lh, nr = LOWER * h, -(-len(panels) // 2)
    top, hgap = S.LEGEND_TOP + 0.45 * rows + 0.6, G.SIZE_HGAP + G.SIZE_XLABEL
    fig = S.figure_cm(S.PRINT_W, top + nr * (h + 0.1 + lh) + (nr - 1) * hgap + 1.0)
    axes, handles = [], {}
    for k, (key, ds, met, title) in enumerate(panels):
        i, j = divmod(k, 2)
        x, y = G.SIZE_LEFT + j * (w + G.SIZE_WGAP), top + i * (h + 0.1 + lh + hgap)
        ax = S.axes_cm(fig, x, y, w, h)
        rx = S.axes_cm(fig, x, y + h + 0.1, w, lh, sharex=ax)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
        panel = [m(key, ds, met) for m in makers]
        ns = [n for series, *_ in panel for n in series]
        ax.set_xscale("log")
        ax.set_xlim(min(ns) / G.SIZE_PAD, max(ns) * G.SIZE_PAD)
        rx.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        rx.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, title)
        S.labels(ax, ylabel=SIZE_Y[met])
        S.labels(rx, G.X_JETS, r"$\Delta$ (points)" if met == "accuracy" else "Ratio")
        for a in (ax, rx):
            S.fine_grid(a)
        ref = panel[nets.index("L188")][0]
        for n_, ((series, line, mark, label), net) in reversed(list(enumerate(zip(panel, nets)))):
            G._curve(ax, series, line, mark, label, True, L.SIZE, n_, len(panel))
            if net not in ("SCRATCH", "L188"):
                rel_n, rel, rel_sd = _relative(series, ref, met)
                L.scan(rx, rel_n, rel, rel_sd, rel_sd, color=line["color"], linestyle=line.get("linestyle", "-"),
                       probe=mark.get("markerfacecolor") == "white", label="_nolegend_", style="C")
        base = 0.0 if met == "accuracy" else 1.0
        rx.axhline(base, color=S.SERIES["L188"], linewidth=S.LW_CURVE, zorder=2.0)
        if met == "r30":
            G._log_y(ax)
        else:
            S.data_ylim(ax)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        S.data_ylim(rx, pad=0.12)
        lo_, hi_ = rx.get_ylim()
        span = hi_ - lo_
        rx.set_ylim(min(lo_, base - 0.1 * span), max(hi_, base + 0.1 * span))      # the 188-class line inside
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        S.clear_edge_labels(rx, high=True)
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    labels = [S.LEGEND[r] for r in nets]
    S.fig_legend(fig, [handles[lab] for lab in labels], top_cm=S.LEGEND_TOP, ncol=ncol, labels=labels)
    S.letter_panels(axes)
    S.save(fig, path)


def draw_fig4(path: pathlib.Path) -> None:
    """Fine-tuned JetClass-II and JetClass (every point the fixed-schedule cell at its row's recorded rate,
    gallery_labelset._settled), beside training from scratch."""
    _size_figure(path, [_pct_series(G._tuned(r)) for r in SIZE_NETS], SIZE_NETS, SIZE_PANELS[:2], 3)


def draw_figA7(path: pathlib.Path) -> None:
    """Fig. 5's form for fine-tuned top tagging and quark/gluon (ledger 34.3: moved from the main figure)."""
    _size_figure(path, [G._tuned(r) for r in SIZE_NETS], SIZE_NETS, SIZE_PANELS[2:], 3)


def draw_figA3(path: pathlib.Path) -> None:
    """Fig. 5's form for linear probes on the frozen features of Table 2's pretrained rows, the four tasks."""
    _size_figure(path, [_pct_series(G._probe(r)) for r in SUMMARY_ROWS], SUMMARY_ROWS, SIZE_PANELS, 3)


# ------------------------------------------------------------------ Figs. A4 and A5
def draw_figA4(path: pathlib.Path) -> None:
    """The old F9b (gallery_physics._binned_2x2): the mass output of the two models with mass regression beside the
    soft-drop mass, median response and resolution against jet pT and generator-level mass."""
    series = PH._mass_series()
    PH._binned_2x2(path, "all", series, 2, cols=[[PH.GROOMED], [label for *_, label in series]])


LAST = 10                  # the checkpoint is the best of the last ten epochs (Methods)
CKPT = C.P.DATA / "ckpt_identity" / "identity.json"


def chosen_epochs() -> dict:
    """{run: the 0-based epoch of its checkpoint}, from the build's record of the checkpoint each run is evaluated at,
    each the epoch of the highest validation accuracy among the last LAST (the build stops otherwise)."""
    runs = json.loads(CKPT.read_text())["runs"]
    docs = M._load(C.P.DATA / "compute")
    out = {}
    for m, r in runs.items():
        e = int(re.search(r"epoch-(\d+)", r["best70"]).group(1))
        if m in docs:
            va = [x["val_acc"] for x in docs[m]["epochs"]]
            if e != max(range(len(va) - LAST, len(va)), key=lambda k: va[k]):
                raise SystemExit(f"FATAL: {m}: checkpoint epoch {e} is not the best of the last {LAST}")
        out[m] = e
    return out


def draw_figA5(path: pathlib.Path) -> None:
    """The old VAL (gallery_labelset._draw_val's form and data, compute/) for Table 2's pretrained rows: each model's
    validation accuracy (%) on its own label set against the pretraining epoch, the mean over runs and a band of their
    standard deviation; the panel at a one-panel figure's size, the legend beside it."""
    docs, runs = M._load(C.P.DATA / "compute"), M._pooled(C.RUNS)
    top, w = 0.5, S.SINGLE_W - 1.5 - 0.25
    x0 = (S.PRINT_W - (1.5 + w + 0.4 + VAL_LEGEND_W)) / 2 + 1.5
    fig = S.figure_cm(S.PRINT_W, top + 0.85 * w + 1.0)
    ax = S.axes_cm(fig, x0, top, w, 0.85 * w)
    S.fine_grid(ax)
    for row in SUMMARY_ROWS:
        have = PERCENT * np.array([[e["val_acc"] for e in docs[m]["epochs"]] for m in runs[row]
                                   if m in docs and "val_acc" in docs[m]["epochs"][0]])
        if not len(have):
            raise C.Absent(f"no validation accuracy per epoch for {row} (compute/)")
        x, mean = np.arange(1, have.shape[1] + 1), have.mean(axis=0)
        if len(have) > 1:
            sd = have.std(axis=0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=S.SERIES[row], alpha=L.BAND, linewidth=0, zorder=1.0)
        ax.plot(x, mean, **S.look(row), linewidth=S.LW_CURVE, label=S.LEGEND[row])
    # the checkpoint each run is evaluated at: the build's choice (ckpt_identity, "best70": the best validation
    # accuracy of the last ten epochs), checked against these curves; epochs counted from one
    chosen = chosen_epochs()
    n_ep = len(next(iter(docs.values()))["epochs"])
    ax.axvspan(n_ep - LAST + 0.5, n_ep + 0.5, color=S.GREY_LIGHT, alpha=0.5, linewidth=0, zorder=0.6)
    for row in SUMMARY_ROWS:
        for m in runs[row]:
            e = chosen[m]
            ax.plot([e + 1], [PERCENT * docs[m]["epochs"][e]["val_acc"]], linestyle="none", marker="|", markersize=6,
                    markeredgewidth=1.0, color=S.SERIES[row], zorder=2.8)
    ax.set_xlim(0, n_ep + 1)
    S.labels(ax, "Pretraining epoch", "Validation accuracy (%)")
    handles = ax.get_legend_handles_labels()[0][:len(SUMMARY_ROWS)] + [
        Patch(facecolor=S.GREY_LIGHT, alpha=0.5, linewidth=0, label=f"last {LAST} epochs"),
        Line2D([], [], linestyle="none", marker="|", markersize=6, markeredgewidth=1.0, color="#000000",
               label="epoch used, each run")]
    S.slot_legend(fig, handles, x0 + w + 0.4, top, VAL_LEGEND_W, 0.85 * w)
    S.save(fig, path)


# the tallest figure the paper's page holds at the text width with a caption of up to 80 words: A4 less 2.5 cm top and
# bottom margins (24.7 cm), less five 12 pt caption lines and the caption skips (about 3.1 cm); figcheck sets a taller
# figure at this height (as it does at the results book's 18.5 cm), so its text would print smaller
PAPER_MAX_H_CM = 21.6
# (file stem, its drawing, printed width in cm, figcheck's least panel width in cm), in the paper's order
FIGURES = (("fig1_design", draw_fig1, 15.0, 6.1), ("fig2_accuracy", draw_fig2, 15.0, 6.1),
           ("fig_separation", draw_fig_separation, 15.0, 6.1), ("fig3_classes", draw_fig3, 15.0, 6.1),
           ("fig5_resonance", draw_fig5, 15.0, 6.1), ("fig4_training_size", draw_fig4, 15.0, 6.1),
           ("figA1", draw_figA1, 15.0, 4.5), ("figA2", draw_figA2, 15.0, 6.1), ("figA6", draw_figA6, 15.0, 6.1),
           ("figA3", draw_figA3, 15.0, 6.1), ("figA7", draw_figA7, 15.0, 6.1), ("figA4", draw_figA4, 15.0, 6.05),
           ("figA5", draw_figA5, 15.0, 6.1))
# the figure names for the text: \vfig{<stem>} with these labels (the writer's \label), in the paper's order
FIGURE_NAMES = {"fig1_design": "fig:design", "fig2_accuracy": "fig:accuracy", "fig_separation": "fig:separation",
                "fig3_classes": "fig:classes", "fig5_resonance": "fig:resonance", "fig4_training_size": "fig:size",
                "figA1": "fig:recall", "figA2": "fig:otherpairs", "figA6": "fig:qcdpairs", "figA3": "fig:probesize",
                "figA7": "fig:ftbinary", "figA4": "fig:masspt", "figA5": "fig:validation"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="figcheck's findings and print-size rasters beside each figure")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.check:
        import figcheck
        import style_v3
        import style_v4
        figcheck.BOOK_MAX_H_CM = PAPER_MAX_H_CM      # the paper's page, not the results book's
        spec = {stem: (cm, least) for stem, _, cm, least in FIGURES}
        original = style_v3.save

        def save(fig, path):
            stem = pathlib.Path(path).stem
            rep = figcheck.check(fig, stem, *spec[stem])
            out = original(fig, path)
            figcheck.rasters(pathlib.Path(path).with_suffix(".pdf"), rep["print_cm"])
            (a.outdir / f"{stem}_check.json").write_text(json.dumps(rep, indent=1))
            print(f"{stem}: {rep['n_findings']} measured findings {({k: v for k, v in rep['findings'].items() if v})}")
            return out

        for mod in (style_v3, style_v4, S):
            mod.save = save
    with names():
        for stem, draw, _, _ in FIGURES:
            v2_style()
            draw(a.outdir / stem)
            print(f"{stem}: drawn")
    print("V2 FIGURES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
