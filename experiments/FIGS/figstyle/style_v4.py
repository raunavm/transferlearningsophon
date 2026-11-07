#!/usr/bin/env python3
"""The figure style of the paper, round 3 (PI, 2026-11-06): style_v3 with a lighter background and our own palette.

Background, spines, ticks, legend frame and error bars follow arXiv:2603.04084 (Garcia, Herrmann, Krzmanc, Selvaggi),
Figs. 4, 6 and 7 as measured in docs/figstyle/STYLE_REF_2603.md: a faint dashed grid at major y ticks only, all four
spines at 0.9 pt, ticks pointing in (5.3 and 2.7 pt) with minor ticks on all four sides, no legend frame, no error-bar
caps. Everything that paper does not override stays Sophon's, as in style_v3: STIX text and its sizes, legends inside
the axes, figure widths, line widths, axis words at the axis end.

The palette is ours, one mapping for every figure (docs/figstyle/PLAN.md, palette table):
  label sets 188, 162, 43, 17   viridis at 0, 0.30, 0.60, 0.85, darkest for 188, so coarser reads as one direction
  controlled partitions         tab10 orange, purple, red (none, b-quark, random flavour boundary)
  plain random partitions       mid grey;  random network: light grey;  from scratch: dark grey, dashed
  kinematic inputs only         the 188 colour, dotted
  mass regression               the 162 and 17 colours, with their own markers
  JetClass classes, JetClass-II families: their own fixed qualitative set (tab10), unchanged.
Checked by simulating deutan, protan and tritan vision (Machado 2009) in OKLab: the closest pair of the eight series
colours is 7.5 apart (x100) under colour blindness, inside the 6-8 band that needs a second cue, which the partitions'
line styles and markers give; 16 apart in normal vision. tab10 purple, brown and pink (the PI's example) put brown
5.8 from 162 and purple 4.3 from the grey, so orange and red replace them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_v3 import (CM, FULL, FOUR_PANELS, HEIGHT, JC1_CLASSES, JC1_COLOUR, JC1_TEX, LABEL_PT, LEGEND_PT,  # noqa: F401
                      LEVELS, CONTROLS, LW, LW_CURVE, LW_THIN, ONE_PANEL, STACKED, TAB10, TITLE_PAD, TITLE_PT,
                      WIDTH, X_EPS, X_EPS_B, Y_REJ, Y_Z, heading, labels, letter_panels, save, title)

VIRIDIS4 = ("#440154", "#355f8d", "#22a884", "#9bd93c")       # viridis at 0, 0.30, 0.60, 0.85
GREY_DARK, GREY_MID, GREY_LIGHT = "#404040", "#b0b0b0", "#d0d0d0"
SERIES = {"L188": VIRIDIS4[0], "L162": VIRIDIS4[1], "R42_Q1": VIRIDIS4[2], "R16_Q1": VIRIDIS4[3],
          "FLAV_F0": TAB10[1], "FLAV_F1": TAB10[4], "FLAV_F1R": TAB10[3],
          "RAND": GREY_MID, "INIT_BN": GREY_LIGHT, "SCRATCH": GREY_DARK,
          "KIN": VIRIDIS4[0], "L162_MASS": VIRIDIS4[1], "R16_Q1_MASS": VIRIDIS4[3]}
from style_v3 import NAME as _NAME
NAME = {**_NAME, "RAND": "Random partitions", "L162_MASS": "162 classes, mass regression",
        "R16_Q1_MASS": "17 classes, mass regression"}
LINESTYLE = {"SCRATCH": "--", "KIN": ":"}                      # every other series solid
MARKERS = {"L188": "o", "L162": "s", "R42_Q1": "^", "R16_Q1": "D", "L162_MASS": "v", "R16_Q1_MASS": "P", "SCRATCH": "X",
           "FLAV_F0": "s", "FLAV_F1": "^", "FLAV_F1R": "D", "RAND": "v", "INIT_BN": "x", "KIN": "o"}
REFERENCE = {"color": GREY_DARK, "linestyle": "--", "linewidth": LW_THIN}


def look(key: str, **kw) -> dict:
    """color and linestyle of a configuration, for ax.plot(**look(...))."""
    return {"color": SERIES[key], "linestyle": LINESTYLE.get(key, "-"), **kw}


def use_style(simplify: bool = False) -> None:
    """style_v3's Sophon rcParams with 2603.04084's spines, ticks, legend frame and error bars."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": LABEL_PT,
        "figure.subplot.bottom": 0.125, "pdf.fonttype": 42, "path.simplify": simplify,
        "axes.linewidth": 0.9, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 5.3, "ytick.major.size": 5.3, "xtick.minor.size": 2.7, "ytick.minor.size": 2.7,
        "xtick.major.width": 0.35, "ytick.major.width": 0.35, "xtick.minor.width": 0.27, "ytick.minor.width": 0.27,
        "legend.frameon": False, "errorbar.capsize": 0,
        "figure.subplot.wspace": 0.20, "figure.subplot.hspace": 0.25})


def grid(ax) -> None:
    """2603.04084 Figs. 4, 6: dashed #b0b0b0 at major y ticks only, alpha 0.25, under the data."""
    ax.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.35, color="#b0b0b0", alpha=0.25)
    ax.set_axisbelow(True)


fine_grid = grid       # the landscape grid of style_v3 is replaced everywhere


def legend(ax, **kw):
    """Inside the axes, no frame (2603.04084), Sophon's reduced padding; loc upper right unless the data need the corner."""
    kw.setdefault("loc", "upper right")
    kw.setdefault("frameon", False)
    return ax.legend(fontsize=kw.pop("fontsize", LEGEND_PT), borderpad=0.3, labelspacing=0.3, **kw)
