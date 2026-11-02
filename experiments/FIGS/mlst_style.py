#!/usr/bin/env python3
"""Figure style of the MLST paper: mplhep's CMS style, in Helvetica.

The style is scikit-hep's mplhep (pinned at 0.4.1), the one the Duarte group's papers use:
the MPGAN figure code (rkansal47/MPGAN plotting.py, rkansal47/mnist_graph_gan
jets/final_plots.py) calls plt.style.use(hep.style.CMS), as do the ucsd-hep-ex
repositories. It sets ticks inside on all four sides with minor ticks, axis labels at the
ends of the axes, and Petroff's six-colour sequence (arXiv:2107.02270), chosen for
colour-vision deficiencies and recommended by CMS. No experiment label is drawn: these are
not CMS results.

Font (PI, 2026-11-02, decision P12): Helvetica, one of the four fonts MLST lists for figures
(Times, Helvetica, Courier, Symbol) and mplhep's own family. Where LaTeX, dvipng and the helvet
and sansmath packages exist (the render job installs TeX Live: helvet.sty with texlive-latex-base,
the URW Helvetica it maps to with texlive-fonts-recommended, sansmath.sty with
texlive-latex-extra), every label is typeset by LaTeX in Helvetica with sans-serif mathematics,
as mplhep's own CMSTex style does (\\usepackage{helvet}, \\usepackage{sansmath}\\sansmath).
Otherwise matplotlib sets text and mathematics in the first of Helvetica, Nimbus Sans (URW's
clone with Helvetica's metrics), Arial and DejaVu Sans that it finds, for previews.
Mathematical symbols are italic, as in LaTeX. Fonts are embedded as TrueType (pdf.fonttype 42)
or, under LaTeX, as Type 1: never as Type 3.

Size. use_style(final=True) is the paper's (experiments/FIGS/paper_figures.py): the figure is
drawn at its printed size, ONE_COLUMN (8.5 cm) or FULL_WIDTH (15 cm) wide, the two widths MLST
gives, with the text between 8 and 12 pt there as MLST asks: axis labels and panel titles 10 pt,
tick labels and legends 8.3 pt ('small'); 'x-small' and smaller are never used and 'large'
(12 pt) is the largest. Lines, markers and ticks shrink with it. Such a figure is saved at its
exact size (save(..., tight=False)) and included at 100 %. use_style() alone keeps the earlier
layout drawn large and scaled down by the document (26 pt text, panels of PANEL = 6 in), which
the mock-ups and the deck use.

The encoding is fixed here (docs/FIGURE_PLAN.md section 1): the label sets of the hierarchy take
the first four colours of the sequence, finest first, each with its own marker; a variant of a
model keeps that model's colour and differs by marker and line style (the mass-regression
models: pentagon, dashed; the kinematic-only model: hexagon, dash-dot; the paper's figures set
these in their own _look); the randomly initialized network its grey; the random partitions
Paul Tol's teal and the partitions with controlled flavor boundaries Tol's cyan, a marker each;
the network trained from scratch black, open squares, dashed; the groomed mass black, solid;
the closest-output-node control grey, dotted. Colour is never the only cue.

experiments/FIGS/style.py stays as it is: the earlier figure scripts and their tests use it.
"""
from __future__ import annotations

import functools
import pathlib
import shutil
import subprocess

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
# near the 17-class purple; from scratch: black, open squares, dashed (the open square keeps it
# apart from the open-circle MLP probe in greyscale)
RAND_COLOUR = "#009988"
SCRATCH_STYLE = {"color": "black", "marker": "s", "markerfacecolor": "white", "linestyle": "--"}
# The random partitions with controlled flavor boundaries: Tol's cyan with a marker each
FLAV_COLOUR = "#33bbee"
FLAV_MARKERS = {"F0": "D", "F1": "P", "F1R": "*"}
# the groomed (soft-drop) mass, the reference of the mass regression; the closest-output-node
# control of the resonance tagging (Sophon, arXiv:2405.12972, Fig. 2: dotted grey)
GROOMED_STYLE = {"color": "black", "linestyle": "-", "linewidth": 1.5}
NODE_CONTROL_STYLE = {"color": PETROFF6[4], "linestyle": ":"}
# TikZ colours of the design figure. The mass-regression and kinematic-only variants have none of
# their own: they take their label set's colour with a dashed or dash-dot outline.
TEX_COLOURS = {"c188": LEVEL_COLOURS[188], "c162": LEVEL_COLOURS[162], "c43": LEVEL_COLOURS[43],
               "c17": LEVEL_COLOURS[17], "crand": RAND_COLOUR, "cflav": FLAV_COLOUR, "cinit": PETROFF6[4]}


def tex_colour_definitions() -> str:
    """\\definecolor lines for the manuscript preamble, from the mapping above."""
    return "".join(f"\\definecolor{{{k}}}{{HTML}}{{{v.lstrip('#').upper()}}}\n" for k, v in TEX_COLOURS.items())
# The scaled layout (use_style() without final): the style's text is 26 pt and its tick labels
# and legend 21.7 pt ('small'); two panels of 6 in printed at a 360 pt text width give axis
# labels of about 11 pt and tick labels of about 9 pt.
PANEL = 6.0
# Line width of the MPGAN paper's final figures (mnist_graph_gan jets/final_plots.py,
# line_opts linewidth 3); the style leaves it at matplotlib's default.
LINEWIDTH = 3

# The printed sizes (MLST: 8.5 cm one column, 15 cm full width) and the style at that size
CM = 1 / 2.54
ONE_COLUMN = 8.5 * CM
FULL_WIDTH = 15.0 * CM
FINAL_RC = {"font.size": 10.0, "axes.titlesize": "medium", "axes.labelsize": "medium",
            "xtick.labelsize": "small", "ytick.labelsize": "small", "legend.fontsize": "small",
            "axes.linewidth": 0.8, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
            "xtick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.major.size": 3.5, "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.major.width": 0.8, "ytick.minor.width": 0.6,
            "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
            "lines.linewidth": 1.2, "lines.markersize": 4.5, "lines.markeredgewidth": 0.8,
            "errorbar.capsize": 0, "patch.linewidth": 0.8, "hatch.linewidth": 0.5,
            "legend.borderaxespad": 0.3, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.3, "legend.handlelength": 2.0, "legend.borderpad": 0.3,
            "savefig.dpi": 300}
# Helvetica, then fonts with its metrics, then the one matplotlib always has (previews without LaTeX)
SANS = ("Helvetica", "Nimbus Sans", "Nimbus Sans L", "Arial", "DejaVu Sans")


@functools.lru_cache(maxsize=None)
def latex_helvetica() -> bool:
    """Whether LaTeX can set the labels in Helvetica with sans-serif mathematics: latex, dvipng
    and kpsewhich on the path, and helvet.sty and sansmath.sty found by kpsewhich."""
    if not all(shutil.which(x) for x in ("latex", "dvipng", "kpsewhich")):
        return False
    try:
        out = subprocess.run(["kpsewhich", "helvet.sty", "sansmath.sty"], capture_output=True, text=True,
                             timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return False
    return len(out) == 2


@functools.lru_cache(maxsize=None)
def sans_font() -> str:
    """The first font of SANS that matplotlib finds."""
    from matplotlib import font_manager
    for name in SANS:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


def use_style(final: bool = False) -> None:
    """mplhep's CMS style in Helvetica; final=True at the printed size (FINAL_RC). Called by the
    figure scripts, never on import. Starts from matplotlib's defaults, so a final figure drawn
    earlier in the process leaves nothing behind (sizes, LaTeX)."""
    plt.rcdefaults()
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    if final:
        plt.rcParams.update(FINAL_RC)
    if latex_helvetica():
        plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
                             "text.latex.preamble": r"\usepackage{amsmath}\usepackage{helvet}"
                                                    r"\usepackage{sansmath}\sansmath"})
        return
    font = sans_font()
    plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                         "font.sans-serif": [font] + [f for f in SANS if f != font],
                         "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.sf": font,
                         "mathtext.tt": font, "mathtext.cal": font, "mathtext.it": f"{font}:italic",
                         "mathtext.bf": f"{font}:bold",
                         # symbols in italic as in LaTeX (the style sets them upright)
                         "mathtext.default": "it"})


def save(fig, name: str, outdir: pathlib.Path | None = None, tight: bool = True,
         dpi: int = 150) -> list[pathlib.Path]:
    """Write <name>.pdf and <name>.png and return both paths. tight=False keeps the figure's own
    size (a figure drawn at its printed size); the PNG at dpi."""
    out = pathlib.Path(FIGURES if outdir is None else outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight" if tight else None, dpi=dpi if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths
