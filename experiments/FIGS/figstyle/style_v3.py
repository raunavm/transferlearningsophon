#!/usr/bin/env python3
"""The figure style of the paper, written again from the figures of the papers we follow (PI,
2026-11-04). Draft: it becomes experiments/FIGS/mlst_style.py figure by figure once the PI has
approved the gallery (docs/figstyle/gallery.pdf); nothing in the paper uses it before then.

The settings are Sophon's (arXiv:2405.12972), measured from the native PDFs of its arXiv source
(docs/figstyle/sophon_native.json; no plotting script exists): matplotlib's defaults with STIX text
and these non-default settings. Sophon's figures are 5 in wide at font size 10, printed at 0.72 to 0.75, so
7.2 to 7.5 pt text. MLST allows figures 8.5 cm or 15 cm wide with text of at least 8 pt; ours use the 15 cm width and print at 1.0: same proportions.

The theme on top of them (PI, 2026-11-05), for every figure but the redraw of Sophon Fig. 2: landscape Fig. 5's grid
(fine_grid), thinner lines (LW_CURVE), the panel sizes below, and a heading saying what the figure shows.

Not Sophon's. X_EPS and Y_REJ are the words of the landscape paper (arXiv:1902.09914), in Sophon's form and place.
Our own choices: C4-C6 for the controlled partitions and black for from scratch (Sophon has neither model), and the
marker sizes, MARKER in gallery_labelset.py and markersize in gallery_physics.py (Sophon Fig. 5 has 3, Golling's native
size is 6; we copied neither).

A kind of figure Sophon has no instance of follows the precedent named for it in docs/figstyle/PLAN.md;
docs/figstyle/CATALOGUE.md section 1.2 lists the kinds and candidates.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CM = 1 / 2.54
WIDTH, FULL = 5.0, 15 * CM        # Sophon's figure width; the MLST column, for two panels side by side
HEIGHT = {1: 3.0, 2: 4.0, 3: 5.0}  # Sophon Figs. 5, 2, 3: panels stacked on a shared x axis
LABEL_PT, LEGEND_PT, TITLE_PT, TITLE_PAD = 10, 8, 10, 5      # Sophon Fig. 2
LW, LW_THIN = 1.5, 1.0            # matplotlib default; Sophon Fig. 5's reference and its error bars
LW_CURVE = 1.1
ONE_PANEL = (14 * CM, 11.5 * CM)
FOUR_PANELS = (FULL, 18 * CM)
STACKED = (14 * CM, 17 * CM)      # three panels on one x axis
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
         "#17becf"]

# one colour per model across the paper, with the exceptions in PLAN.md's colours row: the label sets in tab10's order, the controlled partitions next
SERIES = {"L188": TAB10[0], "L162": TAB10[1], "R42_Q1": TAB10[2], "R16_Q1": TAB10[3],
          "FLAV_F0": TAB10[4], "FLAV_F1": TAB10[5], "FLAV_F1R": TAB10[6], "SCRATCH": "#000000"}
NAME = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
        "FLAV_F0": "No flavor boundary", "FLAV_F1": "b-quark boundary", "FLAV_F1R": "Random flavor boundary",
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

# marker shapes, for greyscale print on line plots (MLST; PI, 2026-11-05); ROC figures stay colour-only
MARKERS = {"L188": "o", "L162": "s", "R42_Q1": "^", "R16_Q1": "D", "L162_MASS": "v", "R16_Q1_MASS": "P", "SCRATCH": "X",
           "FLAV_F0": "s", "FLAV_F1": "^", "FLAV_F1R": "D", "RAND": "v"}

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


def fine_grid(ax) -> None:
    """Landscape Fig. 5: dashed under the curves at major and minor ticks on both axes."""
    ax.minorticks_on()
    ax.grid(which="both", linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)


def heading(fig, text: str) -> None:
    """What the whole figure shows, above the panel titles."""
    fig.suptitle(text, fontsize=TITLE_PT, y=0.975)


def letter_panels(axes) -> None:
    """(a), (b), ... in front of each panel's title (MLST)."""
    for i, ax in enumerate(axes):
        ax.set_title(f"({chr(97 + i)}) {ax.get_title()}".rstrip(), fontsize=TITLE_PT, pad=TITLE_PAD)


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


# PI 2026-11-14: a figure that holds one mode names it in a heading above the panels ({page id: heading}, filled by
# gallery_figures); the figure grows HEADING_CM at the top and everything already drawn keeps its place below
HEADINGS: dict = {}
HEADING_CM = 0.5


def _add_heading(fig, text: str) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    cm = 1 / 2.54
    if not getattr(fig, "exact_size", False):          # a tight box takes the heading in as it is
        fig.suptitle(text, fontsize=9.0, y=1.0, va="bottom")
        return
    W, H = fig.get_size_inches()
    new_h = H + HEADING_CM * cm
    k = H / new_h
    anchors = [(leg, leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())) for leg in fig.legends]
    for art in fig.get_children():
        if art is fig.patch or art in fig.axes or art in fig.legends:
            continue
        if isinstance(art, Text) and art.get_transform() == fig.transFigure:
            x, y = art.get_position()
            art.set_position((x, y * k))
        elif isinstance(art, Line2D) and art.get_transform() == fig.transFigure:
            art.set_ydata([y * k for y in art.get_ydata()])
        else:
            raise ValueError(f"heading: no rule to move a figure-level {type(art).__name__}")
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position(original=True).bounds
        ax.set_position([x0, y0 * k, w, h * k], which="both")
    fig.set_size_inches(W, new_h)
    for leg, b in anchors:
        leg.set_bbox_to_anchor(Bbox.from_extents(b.x0, b.y0 * k, b.x1, b.y1 * k), transform=fig.transFigure)
    fig.text(0.5, 1 - 0.15 * cm / new_h, text, ha="center", va="top", fontsize=9.0)


def save(fig, path: pathlib.Path) -> list:
    """bbox_inches='tight', as Sophon; vector PDF and a PNG for viewing. Prints how many line vertices lie under each legend."""
    if (text := HEADINGS.get(pathlib.Path(path).stem)) is not None:
        _add_heading(fig, text)
    fig.canvas.draw()
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None:
            box = leg.get_window_extent(fig.canvas.get_renderer())
            n = sum(box.count_contains(ax.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])) for ln in ax.lines)
            print(f"{path.name}, {ax.get_title()!r}: {n} curve points under the legend")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), bbox_inches=None if getattr(fig, "exact_size", False) else "tight", **kw)
        out.append(path.with_suffix(suffix))
    plt.close(fig)
    return out
