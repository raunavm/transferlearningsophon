#!/usr/bin/env python3
"""One visual vocabulary for every figure in the journal paper.

THE ENCODING IS FIXED HERE, NOT IN THE FIGURE SCRIPTS. Colour carries the
pretraining label-set size; linestyle carries the probe kind (linear solid,
nonlinear dashed). A reader who has learnt the encoding on one figure has learnt
it on all of them, and a figure script that wants a different encoding has to
say so where everyone can see it.

WHY THESE COLOURS. They are four of the Okabe--Ito colour-blind-safe set, chosen
so that their WCAG relative luminances are also well separated (0.00, 0.17,
0.28, 0.46): the figure survives deuteranopia AND a greyscale printer, which a
hue-only palette does not. `relative_luminance` is exported so the test can hold
that property rather than trusting the comment. Luminance rises as the
vocabulary coarsens, so "lighter" reads as "less label information".

A marker per level rides beside the colour for the same reason -- a scatter of
single points has no linestyle to carry the probe kind and no area to carry the
hue.

Figure widths are the two-column journal page: 3.4 in fits one column, 7.0 in
spans both. `save` writes .pdf (the submission) and .png (the drafts and the
slide deck) side by side at 300 dpi.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"

# Okabe--Ito, ordered fine -> coarse with rising luminance. Keyed by the number
# of classes in the pretraining vocabulary, which is what the axis ticks say.
LEVEL_COLOURS = {188: "#000000", 162: "#0072B2", 43: "#009E73", 17: "#E69F00"}
LEVEL_MARKERS = {188: "o", 162: "s", 43: "^", 17: "D"}

# The probe kind is a linestyle, never a colour: a task's two probes must read as
# the same task. Open markers for the nonlinear probe so overlapping points at
# the same x stay separable in print.
PROBE_LINESTYLES = {"linear": "-", "mlp": "--"}
PROBE_FILLSTYLES = {"linear": "full", "mlp": "none"}
PROBE_LABELS = {"linear": "linear probe", "mlp": "nonlinear (MLP) probe"}

# Spare Okabe--Ito entries for anything that is not a granularity -- probe tasks,
# fine-tuning initialisations. `colour_cycle` hands them out deterministically.
SERIES_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                  "#E69F00", "#56B4E9", "#000000", "#F0E442"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

FIG_W_ONE_COLUMN = 3.4
FIG_W_TWO_COLUMN = 7.0
DPI = 300


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of an #rrggbb string, 0 (black) to 1 (white).

    This is the greyscale value a black-and-white printer renders, so it is the
    quantity the palette has to separate -- not the perceptual hue distance.
    """
    srgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def colour_cycle(keys) -> dict:
    """Stable colour (and marker) per key, in the order the keys are given."""
    keys = list(keys)
    if len(keys) > len(SERIES_COLOURS):
        raise SystemExit(f"FATAL: {len(keys)} series but only {len(SERIES_COLOURS)} "
                         f"colour-blind-safe colours; do not recycle one silently")
    return {k: SERIES_COLOURS[i] for i, k in enumerate(keys)}


def marker_cycle(keys) -> dict:
    return {k: SERIES_MARKERS[i] for i, k in enumerate(list(keys))}


def use_style() -> None:
    """Journal two-column rcParams. Called by the figure scripts, never on import.

    Not applied at import time on purpose: importing this module for its palette
    must not silently restyle a figure script that did not ask for it.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def level_label(n_classes: int) -> str:
    """How a granularity is named in prose and in a legend: by its class count."""
    return f"{n_classes}-class"


def save(fig, name: str, outdir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Write <name>.pdf and <name>.png at 300 dpi and return both paths."""
    out = pathlib.Path(FIGURES if outdir is None else outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI)
        paths.append(p)
    plt.close(fig)
    return paths
