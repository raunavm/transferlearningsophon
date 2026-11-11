"""Line graphs against a scanned quantity: the styles of the PI's comparison (2026-11-11).

A, the current style: each model's own marker shape, filled or hollow, 4-5 pt with an outline; SD bands at 20%.
B, error bars: 1.3 pt lines; one marker shape (a 3 pt circle, no outline; open for a linear probe); +-1 SD as vertical
   bars, 1 pt, no caps, in the series colour; no bands; in a size scan a small horizontal offset between series.
C, light bands: 1.3 pt lines; 3 pt circles at the measured points; +-1 SD bands at 15% opacity, no edge, under every line.
Binned measurements (jet pT) are points with error bars and no lines in B and C; categorical axes (the pretraining
label set) have no lines at all. Colour is the model and line style the mode, as before (style_v5)."""
from __future__ import annotations

import numpy as np

import style_v5 as S

STYLES = ("A", "B", "C")
DOT = 3.0               # pt, the one marker of B and C
OPEN_EDGE = 0.8         # pt, the outline of an open (linear-probe) circle
BAR = 1.0               # pt, error bars
BAND = 0.15             # opacity of C's bands
OFFSET_DEX = 0.022      # B: log10 offset between neighbouring series in a size scan (about 5%)
OFFSET_TEXT = "Series are offset horizontally by a few per cent so that their error bars do not overlap."


def dot(color: str, probe: bool) -> dict:
    """B and C's marker: a small filled circle with no outline, open for a linear probe."""
    return {"marker": "o", "markersize": DOT, "markerfacecolor": "white" if probe else color,
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
    line = {"color": color, "linestyle": linestyle, "linewidth": S.LW_CURVE}
    if style == "B":
        shift = OFFSET_DEX * (k - (n - 1) / 2)
        xo = xs * 10 ** shift if log_x else xs + shift * (xs.max() - xs.min() if len(xs) > 1 else 1.0)
        ax.errorbar(xo, mean, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=BAR, capsize=0, zorder=2.4)
        ax.plot(xo, mean, **line, **dot(color, probe), label=label, zorder=2.6)
    elif style == "C":
        if np.any(lo) or np.any(hi):
            ax.fill_between(xs, mean - lo, mean + hi, color=color, alpha=BAND, linewidth=0, zorder=1.0)
        ax.plot(xs, mean, **line, **dot(color, probe), label=label, zorder=2.6)
    else:
        raise ValueError(f"scan: style {style!r} is not B or C (A is each figure's own code)")


def binned(ax, centres, mean, lo, hi, *, color: str, probe: bool, label: str, k: int = 0, n: int = 1,
           step: float = 0.0) -> None:
    """Binned measurements (B and C alike): the mean at the bin centre with +-1 SD bars, no line; k of n series sit
    step apart within the bin so their bars do not overlap."""
    x = np.asarray(centres, float) + step * (k - (n - 1) / 2)
    ax.errorbar(x, mean, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=BAR, capsize=0, zorder=2.4)
    ax.plot(x, mean, linestyle="none", **dot(color, probe), label=label, zorder=2.6)


def category(ax, x: float, mean: float, lo, hi, *, color: str, probe: bool, label: str | None = None) -> list:
    """One slot of a categorical axis: the mean marker with its +-1 SD bar (none for one run); the bar's ends."""
    ax.errorbar([x], [mean], yerr=None if lo is None else [[lo], [hi]], fmt="none", ecolor=color, elinewidth=BAR,
                capsize=0, zorder=2.4)
    ax.plot([x], [mean], linestyle="none", **dot(color, probe), label=label, zorder=2.6)
    return [mean - (lo or 0.0), mean + (hi or 0.0)]
