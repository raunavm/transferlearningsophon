"""Line graphs against a scanned quantity: the styles of the PI's comparison (2026-11-11). The PI chose (2026-11-11):
scans (training-set size, a cut's efficiency) in C, no error bars and no offsets; binned measurements (jet pT, generator
mass) as points with +-1 SD bars and no lines, with a horizontal bar across the bin and no offset (PI 2026-11-12: as the
precedents); markers filled everywhere,
open circles only where linear probes and fine-tuned networks share a panel. B stays for the comparison page.

A, the current style: each model's own marker shape, filled or hollow, 4-5 pt with an outline; SD bands at 20%.
B, error bars: 1.3 pt lines; one marker shape (a 3 pt circle, no outline; open for a linear probe); +-1 SD as vertical
   bars, 1 pt, no caps, in the series colour; no bands; in a size scan a small horizontal offset between series.
C, light bands: 1.3 pt lines; 3 pt circles at the measured points; +-1 SD bands at 15% opacity, no edge, under every line.
Binned measurements (jet pT) are points with error bars and no lines in B and C. Colour is the model and line style the mode, as before (style_v5).
PI 2026-11-13 (L21, overlapping bands "make some of them very confusing and look bad"): where x takes a few values
(training-set size, jet-pT and generator-mass bins) no bands: points with thin vertical +-1 SD bars and no caps,
dodged a little in x per series (a fixed step in log x for training size, SIZE; within the bin for binned
measurements, no bar across the bin), joined by thin lines in the series colour. Where x is continuous (ROC ratios,
sculpting, figure of merit) a panel of more than BAND_ALL series keeps the band of 188 classes and of the mode's
reference only (keeps_band). PI 2026-11-13, later: every point of a discrete-x figure keeps its thin bar (XBAR), with
the per-series offset, whatever the number of series. PI 2026-11-14 (step 8): binned measurements are points with error
bars and a horizontal bar across the bin, no connecting lines (binned)."""
from __future__ import annotations

import numpy as np

import style_v5 as S

STYLES = ("A", "B", "C")
DOT = 3.0               # pt, the one marker of B and C
OPEN_EDGE = 0.7         # pt, the outline of an open (linear-probe) circle
OPEN = 2.6              # pt, an open circle without its outline: with it, as large as a filled one
BAR = 0.8               # pt, error bars (L21: thin)
XBAR = 0.5              # pt, the bars of scan B and binned, on every point of a discrete-x figure (PI 2026-11-13)
BAND = 0.15             # opacity of C's bands
SCAN = "C"              # the chosen style of every continuous scan (a cut's efficiency)
SIZE = "B"              # L21: the training-set size scan, error bars dodged in log x
THIN = 0.75            # pt, L21: the line joining a series' points
BAND_ALL = 3           # L21: a panel of at most this many series keeps every band
BAND_REFS = ("L188", "SCRATCH", "INIT", "INIT_BN")     # L21: 188 classes and the modes' references keep theirs in any panel
BIN_SPREAD = 0.5       # L21: the dodged points of a bin span this fraction of the narrowest bin's width
OFFSET_DEX = 0.04       # B: the largest log10 offset of a series from the true size in a size scan (about 10%); the
                        # series share +-OFFSET_DEX evenly, centred on the true size (audit 3, 2026-11-14)
OFFSET_TEXT = "Series are offset horizontally by at most 10% so that their error bars do not overlap."


def keeps_band(net: str, n: int) -> bool:
    """L21: whether a series of a continuous-x panel of n series draws its SD band."""
    return n <= BAND_ALL or net in BAND_REFS


# PI 2026-11-14 (audit 3): each network its own marker shape (style_v5.MARKERS), so the series part in greyscale too;
# a colour that is no network's (the soft-drop mass) keeps the circle
MARKER_OF = {c: S.MARKERS[k] for k, c in S.SERIES.items() if k in S.MARKERS}


def dot(color: str, probe: bool, shape: str = "o") -> dict:
    """B and C's marker: a small filled marker of the network's shape (MARKER_OF) with no outline, open for a linear
    probe; shape "x" for a series that shares a panel with a black reference of near its colour (the random network
    beside the soft-drop mass)."""
    shape = MARKER_OF.get(color.upper(), shape) if shape == "o" else shape
    if shape == "x":
        return {"marker": "x", "markersize": DOT + 0.5, "markeredgecolor": color, "markeredgewidth": OPEN_EDGE + 0.3}
    size = (OPEN if probe else DOT) * (S.STAR_SCALE if shape == "*" else 1.0)
    return {"marker": shape, "markersize": size, "markerfacecolor": "white" if probe else color,
            "markeredgecolor": color, "markeredgewidth": OPEN_EDGE if probe else 0.0}


def stats(by_x: dict, spread) -> tuple:
    """(x, mean, lower, upper) arrays from {x: [value per run]}; spread(values) -> (mean, minus, plus), None for one run."""
    xs = np.array(sorted(by_x), float)
    st = [spread(list(by_x[x])) for x in sorted(by_x)]
    mean = np.array([s[0] for s in st])
    lo = np.array([s[1] or 0.0 for s in st])
    hi = np.array([s[2] or 0.0 for s in st])
    return xs, mean, lo, hi


def scan(ax, xs, mean, lo, hi, *, color: str, linestyle: str, probe: bool, label: str, style: str,
         k: int = 0, n: int = 1, log_x: bool = True) -> None:
    """One series against a scanned quantity (training-set size, a cut) in style B or C; k of n series, for B's offset."""
    xs, mean, lo, hi = map(np.asarray, (xs, mean, lo, hi))
    line = {"color": color, "linestyle": linestyle, "linewidth": THIN if style == "B" else S.LW_CURVE}
    if style == "B":
        shift = OFFSET_DEX * (2 * k / (n - 1) - 1) if n > 1 else 0.0
        xo = xs * 10 ** shift if log_x else xs + shift * (xs.max() - xs.min() if len(xs) > 1 else 1.0)
        ax.errorbar(xo, mean, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=XBAR, capsize=0, zorder=2.4)
        ax.plot(xo, mean, **line, **dot(color, probe), label=label, zorder=2.6)
    elif style == "C":
        if np.any(lo) or np.any(hi):
            ax.fill_between(xs, mean - lo, mean + hi, color=color, alpha=BAND, linewidth=0, zorder=1.0)
        ax.plot(xs, mean, **line, **dot(color, probe), label=label, zorder=2.6)
    else:
        raise ValueError(f"scan: style {style!r} is not B or C (A is each figure's own code)")


def steps(ax, edges, mean, sd, *, color: str, linestyle: str, band: bool, label: str) -> None:
    """Binned measurements as a step line, flat across each bin, with a +-1 SD band under it (band) at BAND opacity
    (PI 2026-11-20: the binned mass figures as step lines with bands, not points; keeps_band decides band)."""
    e, m, sd = (np.asarray(a, float) for a in (edges, mean, sd))
    if band and np.any(sd):
        ax.stairs(m + sd, e, baseline=m - sd, fill=True, color=color, alpha=BAND, linewidth=0, zorder=1.0)
    ax.stairs(m, e, baseline=None, color=color, linestyle=linestyle, linewidth=S.LW_CURVE, label=label, zorder=2.6)


def binned(ax, edges, mean, lo, hi, *, color: str, probe: bool, label: str, shape: str = "o", k: int = 0,
           n: int = 1) -> None:
    """Binned measurements (PI 2026-11-14, step 8): the mean of the k-th of n series as a point near the bin centre,
    dodged within the bin (BIN_SPREAD of the narrowest bin's width shared by the n series, the same step in every bin)
    with a thin vertical bar (lo, hi: the uncertainty the claims use) and a thin horizontal bar across the whole bin;
    no line joins the points."""
    e = np.asarray(edges, float)
    step = BIN_SPREAD / (n - 1) if n > 1 else 0.0
    x = 0.5 * (e[1:] + e[:-1]) + (k - (n - 1) / 2) * step * np.diff(e).min()      # one step in every bin
    ax.errorbar(x, mean, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=XBAR, capsize=0, zorder=2.4)
    ok = np.isfinite(np.asarray(mean, float))
    ax.hlines(np.asarray(mean, float)[ok], e[:-1][ok], e[1:][ok], colors=color, linewidth=XBAR, zorder=2.3)
    ax.plot(x, mean, linestyle="none", **dot(color, probe, shape), label=label, zorder=2.6)
