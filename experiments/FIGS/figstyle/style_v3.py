#!/usr/bin/env python3
"""The figure style of the paper, written again from the figures of the papers we follow (PI,
2026-11-04). Draft: it becomes experiments/FIGS/mlst_style.py figure by figure once the PI has
approved the gallery (docs/figstyle/gallery.pdf); nothing in the paper uses it before then.

The settings are Sophon's (arXiv:2405.12972), measured from the native PDFs of its arXiv source
(docs/figstyle/sophon_native.json; no plotting script exists): matplotlib's defaults with STIX text
and these non-default settings. Sophon's figures are 5 in wide at font size 10, printed at 0.74.
MLST has one 15 cm column and asks for at least 8 pt, so ours print at 1.0: same proportions.

Not Sophon's. X_EPS and Y_REJ are the words of the landscape paper (arXiv:1902.09914), in Sophon's form and place.
Our own choices: C4-C6 for the controlled partitions and black for from scratch (Sophon has neither model), and the
marker sizes, MARKER in gallery_labelset.py and markersize in gallery_physics.py (Sophon Fig. 5 has 3, Golling's native
size is 6; we copied neither).

A kind of figure Sophon has no instance of follows the closest paper in docs/figstyle/CATALOGUE.md
(section 1.2); each is named in docs/figstyle/PLAN.md.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDTH, FULL = 5.0, 15 * CM        # Sophon's figure width; the MLST column, for two panels side by side
HEIGHT = {1: 3.0, 2: 4.0, 3: 5.0}  # Sophon Figs. 5, 2, 3: panels stacked on a shared x axis
LABEL_PT, LEGEND_PT, TITLE_PT, TITLE_PAD = 10, 8, 10, 5      # Sophon Fig. 2
LW, LW_THIN = 1.5, 1.0            # matplotlib default; Sophon Fig. 5's reference and its error bars
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
         "#17becf"]

# one colour per model across the paper: the label sets in tab10's order, the controlled partitions next
SERIES = {"L188": TAB10[0], "L162": TAB10[1], "R42_Q1": TAB10[2], "R16_Q1": TAB10[3],
          "FLAV_F0": TAB10[4], "FLAV_F1": TAB10[5], "FLAV_F1R": TAB10[6], "SCRATCH": "#000000"}
NAME = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
        "FLAV_F0": "No flavor boundary", "FLAV_F1": "b-quark boundary", "FLAV_F1R": "Random boundary",
        "SCRATCH": "From scratch", "INIT_BN": "Random network", "KIN": "188 classes, kinematic inputs"}
REFERENCE = {"color": "#000000", "linestyle": "--", "linewidth": LW_THIN}     # Sophon Fig. 5's dashed line
LEVELS = ("L188", "L162", "R42_Q1", "R16_Q1")
CONTROLS = ("FLAV_F0", "FLAV_F1", "FLAV_F1R")

# the ten JetClass classes, one colour each, fixed everywhere (tab10 in order, the poster's order of classes)
JC1_CLASSES = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql", "label_Zqq",
               "label_Wqq", "label_Tbqq", "label_Tbl")
JC1_TEX = {"label_QCD": "QCD", "label_Hbb": r"$H\to b\bar{b}$", "label_Hcc": r"$H\to c\bar{c}$",
           "label_Hgg": r"$H\to gg$", "label_H4q": r"$H\to 4q$", "label_Hqql": r"$H\to \ell\nu qq'$",
           "label_Zqq": r"$Z\to q\bar{q}$", "label_Wqq": r"$W\to qq'$", "label_Tbqq": r"$t\to bqq'$",
           "label_Tbl": r"$t\to b\ell\nu$"}
JC1_COLOUR = dict(zip(JC1_CLASSES, TAB10))

X_EPS = r"Signal efficiency ($\varepsilon_S$)"
Y_REJ = r"Background rejection ($1/\varepsilon_B$)"
X_EPS_B = r"SM background efficiency ($\varepsilon_B$)"     # Sophon Fig. 2
Y_Z = r"Significance ($Z$)"


def use_style(simplify: bool = False) -> None:
    """Sophon's rcParams; everything else stays matplotlib's default. pdf.fonttype 42 is the one change
    of file encoding, for embedded TrueType. simplify=False is not Sophon's: it draws every recorded vertex,
    for the redraw of Sophon Fig. 2 and its pixel comparison; True is matplotlib's default, for any other figure."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": LABEL_PT,
                         "figure.subplot.bottom": 0.125, "pdf.fonttype": 42, "path.simplify": simplify})


def grid(ax) -> None:
    """Sophon Figs. 2, 5: dashed under the curves, the default colour #b0b0b0, at major and minor ticks on x and at
    major ticks on y (Sophon has no y minor ticks; on a log y axis minor grid lines would come and go with the range)."""
    ax.grid(which="both", linestyle="--", linewidth=0.5)
    ax.yaxis.grid(False, which="minor")
    ax.set_axisbelow(True)


def labels(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Word and symbol, at the end of the axis (all of Sophon's plots)."""
    if xlabel:
        ax.set_xlabel(xlabel, ha="right", x=1.0)
    if ylabel:
        ax.set_ylabel(ylabel, ha="right", y=1.0)


def title(ax, text: str) -> None:
    ax.set_title(text, fontsize=TITLE_PT, pad=TITLE_PAD)


def legend(ax, **kw):
    """Inside the axes, framed, with Sophon's reduced padding; loc upper right unless the data need the corner."""
    kw.setdefault("loc", "upper right")
    return ax.legend(fontsize=kw.pop("fontsize", LEGEND_PT), borderpad=0.3, labelspacing=0.3, **kw)


def save(fig, path: pathlib.Path) -> list:
    """bbox_inches='tight', as Sophon; vector PDF and a PNG for viewing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kw)
        out.append(path.with_suffix(suffix))
    plt.close(fig)
    return out
