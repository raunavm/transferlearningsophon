#!/usr/bin/env python3
"""Figure style of the MLST paper: mplhep's CMS style, with the paper's font.

The style is scikit-hep's mplhep (pinned at 0.4.1), the one the Duarte group's papers use:
the MPGAN figure code (rkansal47/MPGAN plotting.py, rkansal47/mnist_graph_gan
jets/final_plots.py) calls plt.style.use(hep.style.CMS), as do the ucsd-hep-ex
repositories. It sets TeX Gyre Heros, ticks inside on all four sides with minor ticks,
axis labels at the ends of the axes, and Petroff's six-colour sequence (arXiv:2107.02270),
chosen for colour-vision deficiencies and recommended by CMS. No experiment label is
drawn: these are not CMS results.

One change to the style: the text is set in Computer Modern, the font of the paper (LaTeX's
default), instead of the style's TeX Gyre Heros, so that a figure reads like the text around
it. Where a LaTeX installation exists (the figure job installs TeX Live in the pod), every
label is typeset by LaTeX itself (text.usetex); otherwise matplotlib's own Computer Modern
(cmr10, mathtext 'cm') stands in, for previews. Mathematical symbols are italic, as in LaTeX.
Fonts are embedded as TrueType (pdf.fonttype 42), not Type 3.

The encoding is fixed here: the label sets of the hierarchy take the first four colours of
the sequence, finest first, and the randomly initialized network its grey. A panel is 6 in
(PANEL); the paper scales a figure to the text width.

experiments/FIGS/style.py stays as it is: the earlier figure scripts and their tests use it.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

REPO = pathlib.Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"

PETROFF6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
LEVEL_COLOURS = {188: PETROFF6[0], 162: PETROFF6[1], 43: PETROFF6[2], 17: PETROFF6[3]}
LEVEL_MARKERS = {188: "o", 162: "s", 43: "^", 17: "v"}
INIT_STYLE = {"color": PETROFF6[4], "marker": "X", "linestyle": "--"}
# Random partitions: Paul Tol's teal (vibrant scheme), not Petroff's violet, which sits too
# near the 17-class purple; from scratch: black, open markers, dotted
RAND_COLOUR = "#009988"
SCRATCH_STYLE = {"color": "black", "marker": "o", "markerfacecolor": "white", "linestyle": ":"}
# The other configurations (2026-10-31), one colour each, the same in every figure and in the
# design schematic (paper/journal/fig_design.tex defines the same colours for TikZ): the
# flavor-controlled partitions Tol's cyan with a marker each, the mass-regression models
# Petroff's violet, the kinematic-input models Tol's orange
FLAV_COLOUR = "#33bbee"
FLAV_MARKERS = {"F0": "D", "F1": "P", "F1R": "*"}
MASS_COLOUR = PETROFF6[5]
KIN_COLOUR = "#ee7733"
TEX_COLOURS = {"c188": LEVEL_COLOURS[188], "c162": LEVEL_COLOURS[162], "c43": LEVEL_COLOURS[43],
               "c17": LEVEL_COLOURS[17], "crand": RAND_COLOUR, "cflav": FLAV_COLOUR, "cmass": MASS_COLOUR,
               "ckin": KIN_COLOUR, "cinit": PETROFF6[4]}


def tex_colour_definitions() -> str:
    """\\definecolor lines for the manuscript preamble, from the mapping above."""
    return "".join(f"\\definecolor{{{k}}}{{HTML}}{{{v.lstrip('#').upper()}}}\n" for k, v in TEX_COLOURS.items())
# The style's text is 26 pt and its tick labels and legend 21.7 pt ('small'). A figure is
# printed at the text width (360 pt in the draft's 11-pt article class); with two panels of
# 6 in that prints the axis labels at about 11 pt, the size of the text, and the tick labels
# and legend at about 9 pt. The style's own 10 in panels would print them at 6-7 pt.
PANEL = 6.0
# Line width of the MPGAN paper's final figures (mnist_graph_gan jets/final_plots.py,
# line_opts linewidth 3); the style leaves it at matplotlib's default.
LINEWIDTH = 3


def use_style() -> None:
    """mplhep's CMS style in Computer Modern. Called by the figure scripts, never on import."""
    import shutil
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    if shutil.which("latex") and shutil.which("dvipng"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Roman"],
                             "text.latex.preamble": r"\usepackage{amsmath}"})
        return
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         # symbols in italic as in LaTeX (the style sets them upright)
                         "mathtext.default": "it",
                         # cmr10 has no minus sign: tick labels go through mathtext
                         "axes.formatter.use_mathtext": True})


def save(fig, name: str, outdir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Write <name>.pdf and <name>.png and return both paths."""
    out = pathlib.Path(FIGURES if outdir is None else outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths
