#!/usr/bin/env python3
"""Figure style of the MLST paper: Sophon's (arXiv:2405.12972), in plain matplotlib.

PI, 2026-11-02 (ROC brief section 10, overriding decision P12 "Helvetica/mplhep"): every figure
follows the Sophon paper's figures as measured from its PDF (Figs. 2-5): STIX serif text with
STIX mathematics (no LaTeX), titles centred above each panel, the x label at the right end of the
axis, a framed legend inside the panel (upper right) or, for figures of three or more panels, one
framed legend above them, a light-grey dashed grid on major and minor ticks (minor ticks, and so a
minor grid, only on log axes), matplotlib's outward ticks on the bottom and left with all four
spines, and matplotlib's default colour cycle tab10 in order. Every configuration has its own line
style (CONFIG below), a limit line is black dashed; error bars are round markers with capless bars.
Heatmaps and histograms switch the grid off per axes. Fonts are embedded as TrueType (pdf.fonttype
42), never as Type 3.

Size. use_style(final=True) is the paper's (experiments/FIGS/paper_figures.py): the figure is
drawn at its printed size, ONE_COLUMN (8.5 cm) or FULL_WIDTH (15 cm) wide, the two widths MLST
gives. Sophon prints axis labels at about 7.5 pt and ticks and legends at about 6.3 pt at 8.4 cm;
MLST asks for 8 to 12 pt, so the sizes keep Sophon's proportions floored at 8 pt: labels and
titles 9 pt, tick labels and legends 8 pt (ROC interface decision 9.1). Such a figure is saved at
its exact size (save(..., tight=False)) and included at 100 %. use_style() alone is the same style
with every size multiplied by SCALED, for the earlier layout drawn large and scaled down by the
document (panels of PANEL = 6 in), which the mock-ups, the deck and xbx_report use.

The encoding is fixed here, one table (CONFIG, keyed by the generator's configuration names) that
paper_figures.look reads: the four label sets take the first four tab10 colours, finest first,
solid, dashed, dash-dot and dotted, each with its own marker; a variant of a model (mass output,
kinematic inputs only) keeps that model's colour with a long-dash pattern (LONG_DASH, LONG_DASH_DOT);
the partitions with controlled flavor boundaries C4, C5, C6 with LONG_DASH, LONG_DASH_DOT and
DASH_DOT_DOT; the random partitions C9, markers only; the network trained from scratch black,
dash-dot-dot, open squares; the randomly initialized network C7, long dashes. The closest-output-node
control of F9 is C7 with a sparse dash-dot-dot and crosses, a pattern no other curve of that figure
has. Colour is never the only cue: line styles on curves, markers in point plots.

experiments/FIGS/style.py stays as it is: the earlier figure scripts and their tests use it.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

REPO = pathlib.Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"

# matplotlib's default colour cycle (tab10), in order: C0 ... C9
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
         "#17becf"]
LEVEL_COLOURS = {188: TAB10[0], 162: TAB10[1], 43: TAB10[2], 17: TAB10[3]}
LEVEL_MARKERS = {188: "o", 162: "s", 43: "^", 17: "v"}
# Line styles: every configuration a figure can show beside another has its own, so the curves are
# told apart without colour (colour-vision deficiency, greyscale print; PI 2026-11-02). The four label
# sets solid / dashed / dash-dot / dotted, fine to coarse; the rest long-dash patterns.
LONG_DASH = (0, (8, 2))
LONG_DASH_DOT = (0, (8, 2, 1.5, 2))
DASH_DOT_DOT = (0, (4, 1.5, 1.5, 1.5, 1.5, 1.5))
LEVEL_LINESTYLES = {188: "-", 162: (0, (5, 2)), 43: (0, (5, 1.5, 1.5, 1.5)), 17: (0, (1.5, 1.2))}
# the randomly initialized network: grey, long dashes
INIT_STYLE = {"color": TAB10[7], "marker": "X", "linestyle": LONG_DASH}
RAND_COLOUR = TAB10[9]
# trained from scratch: black, dash-dot-dot, open squares (the open square keeps it apart from the
# open-circle markers of fine-tuning in greyscale)
SCRATCH_STYLE = {"color": "black", "marker": "s", "markerfacecolor": "white", "linestyle": DASH_DOT_DOT}
# the partitions with controlled flavor boundaries: one colour and marker each (mlst_figures._look
# keeps FLAV_COLOUR for all three, as the deck draws them)
FLAV_COLOUR = TAB10[4]
FLAV_COLOURS = {"F0": TAB10[4], "F1": TAB10[5], "F1R": TAB10[6]}
FLAV_MARKERS = {"F0": "D", "F1": "P", "F1R": "d"}
# the groomed (soft-drop) mass, the reference of the mass regression; the closest-output-node
# control of the resonance tagging (Sophon Fig. 2 draws it grey dotted, but the 17-class label set is
# dotted beside it and from scratch dash-dot-dot): grey, a sparse dash-dot-dot with crosses
GROOMED_STYLE = {"color": "black", "linestyle": "-", "linewidth": 1.5}
NODE_CONTROL_STYLE = {"color": TAB10[7], "linestyle": (0, (3, 1.5, 1, 1.5, 1, 4)), "marker": "x", "markersize": 3,
                      "markevery": 0.12}
# the limit line of every ROC panel (the rejection at which fewer than the minimum number of
# background jets pass): black dashed, as Sophon's limit
LIMIT_STYLE = {"color": "black", "linestyle": "--", "linewidth": 0.8}
# the face of a reference histogram (Sophon Figs. 3-4: pastel fills, thin black edges)
PASTEL = "#d4eef0"

# Colour, ROC line style and point-plot marker of every configuration (keys: mlst_results' names;
# "scratch-v2" is mlst_results.SCRATCH)
CONFIG = {
    "L188": {"color": TAB10[0], "linestyle": LEVEL_LINESTYLES[188], "marker": "o"},
    "L162": {"color": TAB10[1], "linestyle": LEVEL_LINESTYLES[162], "marker": "s"},
    "R42_Q1": {"color": TAB10[2], "linestyle": LEVEL_LINESTYLES[43], "marker": "^"},
    "R16_Q1": {"color": TAB10[3], "linestyle": LEVEL_LINESTYLES[17], "marker": "v"},
    "L162_MASS": {"color": TAB10[1], "linestyle": LONG_DASH, "marker": "p"},
    "R16_Q1_MASS": {"color": TAB10[3], "linestyle": LONG_DASH_DOT, "marker": "8"},
    "KIN": {"color": TAB10[0], "linestyle": LONG_DASH, "marker": "h"},
    "FLAV_F0": {"color": FLAV_COLOURS["F0"], "linestyle": LONG_DASH, "marker": FLAV_MARKERS["F0"]},
    "FLAV_F1": {"color": FLAV_COLOURS["F1"], "linestyle": LONG_DASH_DOT, "marker": FLAV_MARKERS["F1"]},
    "FLAV_F1R": {"color": FLAV_COLOURS["F1R"], "linestyle": DASH_DOT_DOT, "marker": FLAV_MARKERS["F1R"]},
    "RAND": {"color": RAND_COLOUR, "linestyle": "none", "marker": "o"},
    "scratch-v2": dict(SCRATCH_STYLE),
    "INIT": dict(INIT_STYLE),
    "INIT_BN": dict(INIT_STYLE),
}
# TikZ colours of the design figure and the text (main.tex, fig_f1.tex)
TEX_COLOURS = {"c188": LEVEL_COLOURS[188], "c162": LEVEL_COLOURS[162], "c43": LEVEL_COLOURS[43],
               "c17": LEVEL_COLOURS[17], "crand": RAND_COLOUR, "cflav": FLAV_COLOUR, "cinit": TAB10[7]}


def tex_colour_definitions() -> str:
    """\\definecolor lines for the manuscript preamble, from the mapping above."""
    return "".join(f"\\definecolor{{{k}}}{{HTML}}{{{v.lstrip('#').upper()}}}\n" for k, v in TEX_COLOURS.items())


# The scaled layout (use_style() without final): panels of 6 in, every size of SOPHON_RC times SCALED
PANEL = 6.0
SCALED = 2.6
# Line width of the mock-ups' and the deck's series (they set it explicitly at the scaled size)
LINEWIDTH = 3

# The printed sizes (MLST: 8.5 cm one column, 15 cm full width)
CM = 1 / 2.54
ONE_COLUMN = 8.5 * CM
FULL_WIDTH = 15.0 * CM
SOPHON_RC = {
    "text.usetex": False, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.0, "axes.titlesize": 9.0, "axes.labelsize": 9.0, "figure.titlesize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
    "axes.titlelocation": "center", "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    "xaxis.labellocation": "right", "yaxis.labellocation": "center",
    "axes.prop_cycle": cycler(color=TAB10), "axes.linewidth": 0.6, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.which": "both", "grid.color": "0.8", "grid.linestyle": "--", "grid.linewidth": 0.4,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.top": False, "ytick.right": False,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.7, "ytick.minor.size": 1.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.5, "hatch.linewidth": 0.5, "errorbar.capsize": 0,
    "legend.frameon": True, "legend.fancybox": True, "legend.framealpha": 0.8, "legend.edgecolor": "0.8",
    "legend.loc": "upper right", "legend.borderpad": 0.4, "legend.labelspacing": 0.3, "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
}
# the sizes (points) the scaled layout multiplies by SCALED
SIZE_KEYS = ("font.size", "axes.titlesize", "axes.labelsize", "figure.titlesize", "xtick.labelsize",
             "ytick.labelsize", "legend.fontsize", "axes.titlepad", "axes.labelpad", "axes.linewidth",
             "grid.linewidth", "xtick.major.size", "ytick.major.size", "xtick.minor.size", "ytick.minor.size",
             "xtick.major.width", "ytick.major.width", "xtick.minor.width", "ytick.minor.width",
             "xtick.major.pad", "ytick.major.pad", "lines.linewidth", "lines.markersize",
             "lines.markeredgewidth", "patch.linewidth", "hatch.linewidth")


def use_style(final: bool = False) -> None:
    """Sophon's style; final=True at the printed size, else every size times SCALED. Called by the
    figure scripts, never on import. Starts from matplotlib's defaults, so a figure drawn earlier
    in the process leaves nothing behind."""
    plt.rcdefaults()
    plt.rcParams.update(SOPHON_RC)
    if not final:
        plt.rcParams.update({k: SOPHON_RC[k] * SCALED for k in SIZE_KEYS})


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
