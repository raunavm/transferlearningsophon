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
it. Matplotlib ships Computer Modern (cmr10) and draws mathematics in it (mathtext 'cm'), so
no LaTeX installation is needed on the laptop or in the cluster image.

The encoding is fixed here: the label sets of the hierarchy take the first four colours of
the sequence, finest first, and the randomly initialized network its grey. A panel is the
style's 10 x 10 in; the paper scales a figure to the text width.

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
PANEL = 10.0
# Line width of the MPGAN paper's final figures (mnist_graph_gan jets/final_plots.py,
# line_opts linewidth 3); the style leaves it at matplotlib's default.
LINEWIDTH = 3


def use_style() -> None:
    """mplhep's CMS style in Computer Modern. Called by the figure scripts, never on import."""
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
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
