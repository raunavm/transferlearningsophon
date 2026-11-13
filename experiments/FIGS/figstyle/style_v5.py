#!/usr/bin/env python3
"""The figure style of the paper, round 4 (PI, 2026-11-07; docs/figstyle/ROUND4_BRIEF.md): style_v4 with a new palette
and the mode encoding. Background, spines, ticks, fonts and widths are round 3's (style_v4), unchanged.

The line and colour rule (PI, 2026-11-13, binding decision L6): colour identifies the network, line style the use.

  network                             colour
  188 classes                         #332288 Tol muted indigo   L* 22
  162 classes                         #AA4499 Tol muted purple   L* 46
  162 classes + mass                  #CC8FC2 the 162 colour 40% towards white (tint)
  43 classes                          #999933 Tol muted olive    L* 62
  17 classes                          #44AA99 Tol muted teal     L* 64
  17 classes + mass                   #8FCCC2 the 17 colour 40% towards white
  no flavor boundary                  #CC6677 Tol muted rose
  b-quark boundary                    #117733 Tol muted green
  random flavor boundary              #88CCEE Tol muted cyan     (sand #DDCC77 swapped out: 16 CIEDE2000 from olive)
  random partitions                   #BBBBBB Tol grey (mid)
  188 classes, kinematic inputs only  #DDCC77 Tol muted sand
  released Sophon checkpoint          #CC3311 Tol vibrant red
  random network                      #333333 dark grey
  from scratch                        #555555 grey

  use                                                    line       marker
  fine-tuned, or the figure's only mode                  solid      filled
  linear probe beside fine-tuned networks                dashed     hollow
  direct discriminant (the network's own output scores)  dash-dot
  from scratch                                           solid, grey
  reference lines (y = x, random selection, unity)       dotted, black (REFERENCE); never a network
A mass network is a network of its own, in the lighter shade of its label set, solid and filled like any other.

Every pair of networks that can share a panel (palette_check.networks, its GROUPS: CIEDE2000 in normal vision and
under deuteranopia and protanopia, Machado 2009, and the L* difference in greyscale) is at least 10 apart in every view
but one pair the PI chose on 2026-11-07, 17 classes and the random partitions (8.6 under protanopia, 12 L* apart in
greyscale). The closest new pairs: 17 + mass and 17 (12.3 normal, 10.4 simulated, 14 L* lighter) and 162 + mass and 162
(21.0, 18.6), close by design; kinematic inputs only and 43 (16.1, 15.5); the Sophon checkpoint and 43 (10.6 under
deuteranopia); the random network and black (13.4) and from scratch (11.3, never in one figure). The headline dot
plots show every network but name each row: there 17 + mass nears the random partitions under colour-vision deficiency
(3.1). CLASS10 (the JetClass classes, the JetClass-II decay families, the UMAP groups) reuses no network colour: each of
its colours is at least 10.9 CIEDE2000 from every network colour, its first six (the families) at least 15.1 apart.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from style_v4 import (CM, FULL, FOUR_PANELS, HEIGHT, JC1_CLASSES, JC1_COLOUR, JC1_TEX, LABEL_PT, LEGEND_PT,  # noqa: F401
                      LEVELS, CONTROLS, LW, LW_CURVE, LW_THIN, ONE_PANEL, STACKED, TAB10, TITLE_PAD, TITLE_PT,
                      WIDTH, X_EPS, X_EPS_B, Y_REJ, Y_Z, labels, letter_panels, save, title,
                      use_style, grid, fine_grid, legend, NAME, REFERENCE)

GREY_DARK, GREY_MID, GREY_LIGHT = "#555555", "#BBBBBB", "#DDDDDD"
LW_CURVE = 1.3                                   # a curve and its legend handle, wide enough to tell the colours apart
HANDLE = 2.6                                     # legend handle length (em): a dashed handle shows two dashes
# One qualitative set for everything that is not a network: the ten JetClass classes, the JetClass-II decay families
# and the UMAP groups (PI 2026-11-11; was matplotlib's tab10). PI 2026-11-13 (L6): no colour that means a network, so
# not Tol muted any more: black and nine colours each at least 10.9 CIEDE2000 from every network colour (palette_check),
# the first six (the families) at least 15.1 apart (deuteranopia and protanopia counted at 2.5 times), the three after
# black (the UMAP groups) light or saturated enough to stand apart from black in small translucent dots: pear,
# vermillion, plum, dark teal, light orange; then maroon, chocolate, wine and orange.
CLASS10 = ["#000000", "#BBCC33", "#D55E00", "#762A83", "#00555A", "#EE8866", "#67001F", "#7F3B08", "#882255", "#E69F00"]
TAB10 = CLASS10                                  # the name the pages use
JC1_COLOUR = dict(zip(JC1_CLASSES, CLASS10))
# reference and guide lines (the diagonal, the random-selection floor, a ratio's unity line) are black dotted, the one
# use of dotted lines (PI 2026-11-13, L6)
REFERENCE = {**REFERENCE, "linestyle": ":"}


def grid(ax) -> None:
    """The faint grid at major y ticks (2603.04084 Figs. 4, 6), solid, so it reads as no line style of a network."""
    ax.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.35, color="#b0b0b0", alpha=0.3)
    ax.set_axisbelow(True)


fine_grid = grid
NAME = {**NAME, "FLAV_F1": "$b$-quark boundary"}          # quark flavours in italic, as everywhere in the paper
# a prime as a glyph beside the q (mathtext's q' raises it like a second superscript)
JC1_TEX = {k: v.replace("qq'", r"qq^{\prime}") for k, v in JC1_TEX.items()}     # Computer Modern: a raised prime


def tint(colour: str, share: float = 0.4) -> str:
    """colour moved share of the way towards white: the lighter shade of a label set that its mass network takes."""
    return "#" + "".join(f"{round(int(colour[i:i + 2], 16) + share * (255 - int(colour[i:i + 2], 16))):02X}" for i in (1, 3, 5))


# colour identifies the network (L6; the table in the docstring)
SERIES = {"L188": "#332288", "L162": "#AA4499", "R42_Q1": "#999933", "R16_Q1": "#44AA99",
          "FLAV_F0": "#CC6677", "FLAV_F1": "#117733", "FLAV_F1R": "#88CCEE", "RAND": GREY_MID,
          "KIN": "#DDCC77", "SOPHON": "#CC3311", "INIT_BN": "#333333", "SCRATCH": GREY_DARK}
SERIES.update({"L162_MASS": tint(SERIES["L162"]), "R16_Q1_MASS": tint(SERIES["R16_Q1"])})
MARKERS = {"L188": "o", "L162": "s", "R42_Q1": "^", "R16_Q1": "D", "L162_MASS": "v", "R16_Q1_MASS": "P", "SCRATCH": "X",
           "FLAV_F0": "p", "FLAV_F1": "*", "FLAV_F1R": "h", "RAND": "v", "INIT_BN": "x", "KIN": "o", "SOPHON": "d"}
# the controls' shapes differ from every label set's, so no control shares a label set's shape in greyscale
STAR_SCALE = 1.4                                 # "*" draws smaller than the other shapes at one markersize
# the line of a use: "probe" only where linear probes share a panel with fine-tuned networks (L6), "direct" the direct
# discriminant; from scratch, the mass networks and a figure's only mode are "ft", solid
MODE_LINE = {"ft": "-", "probe": "--", "direct": "-."}
MASS = ("L162_MASS", "R16_Q1_MASS")
# legend names (PI, 2026-11-07, final; the row names of every table): the paper's shortest names for the partitions; a
# tag after the name gives the use in words, never abbreviated (PI 2026-11-13, L7), and is dropped where the panel title
# or the caption gives one use for every curve. A figure that mixes uses tags every entry (F8, A17-probe).
LEGEND = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
          "RAND": "random partitions", "FLAV_F0": "no flavor boundary", "FLAV_F1": "$b$-quark boundary",
          "FLAV_F1R": "random flavor boundary", "L162_MASS": "162 classes + mass", "R16_Q1_MASS": "17 classes + mass",
          "KIN": "188 classes, kinematic inputs only", "SCRATCH": "from scratch", "INIT_BN": "random network",
          "SOPHON": "released Sophon checkpoint"}
PROBE, MLP_PROBE, FINE_TUNED = "(linear probe)", "(MLP probe)", "(fine-tuned)"
WORKING_POINT = {"inclusive": {"color": "#000000", "marker": "o", "fillstyle": "full"},
                 "10%": {"color": GREY_DARK, "marker": "s", "fillstyle": "none"},
                 "1%": {"color": "#999999", "marker": "x", "fillstyle": "full"}}


def pow10(v: float, _=None) -> str:
    """A log-axis tick label, 10^n at a power of ten and m x 10^n between (PI 2026-11-11: never "100000")."""
    if v <= 0:
        return ""
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10.0 ** e
    return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else rf"${m:g}\times10^{{{e}}}$"


def log_axis(ax, which: str = "y", force: bool = False, fine: bool = False) -> None:
    """A positive axis whose limits are set: logarithmic with 10^n labels when it spans a decade or more (with 2 and 5
    times 10^n labelled too while fewer than two powers of ten lie inside, then as plain numbers), linear with round ticks below a decade,
    where a log axis has no power of ten to label (force: logarithmic whatever it spans; fine: 1, 2 and 5 times 10^n labelled whatever it spans). Minor ticks are never labelled."""
    from matplotlib import ticker
    lo, hi = getattr(ax, f"get_{which}lim")()
    axis = getattr(ax, f"{which}axis")
    if hi / lo < 10 and not force:
        getattr(ax, f"set_{which}scale")("linear")
        getattr(ax, f"set_{which}lim")(lo, hi)
        axis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        return
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"set_{which}lim")(lo, hi)
    decades = [10.0 ** e for e in range(int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9)) + 1)]
    ticks = decades if len(decades) >= 2 and not fine else [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
                                               for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    axis.set_major_locator(ticker.FixedLocator(ticks))
    plain = ticks is not decades and 1e-3 <= lo and hi < 1e4      # 1, 2, 5: "200", "0.05", never "2 x 10^2"
    axis.set_major_formatter(ticker.FuncFormatter((lambda v, _: f"{v:g}") if plain else pow10))
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(ticker.NullFormatter())


def clear_edge_labels(ax, low: bool = False, high: bool = False, share: float = 0.07) -> None:
    """No y tick label within share of the axis's lower (low) or upper (high) end, where it would touch the panel
    stacked against that end; the ticks stay. The other labels: as they were on a log axis, otherwise the decimals the tick
    step needs, the same on every tick. Call once the limits and the locator are final."""
    from matplotlib import ticker
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    pos = (lambda v: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))) if log else (lambda v: (v - lo) / (hi - lo))

    was = ax.yaxis.get_major_formatter()     # a log axis keeps log_axis's labels (10^n, or plain numbers)

    def fmt(v, _):
        if (log and v <= 0) or (low and pos(v) < share) or (high and pos(v) > 1 - share):
            return ""
        if log:
            return was(v, None)
        locs = np.asarray(ax.yaxis.get_majorticklocs(), float)
        step = float(np.min(np.diff(locs))) if len(locs) > 1 else 1.0
        return f"{v:.{max(0, int(np.ceil(-np.log10(step) - 1e-9)))}f}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def data_ylim(ax, pad: float = 0.08) -> None:
    """y limits from everything drawn in data coordinates (lines, bands and error bars), padded by pad of the span, in
    log space on a log axis, where values at or below zero do not count."""
    from matplotlib.collections import LineCollection, PolyCollection
    ys = [np.asarray(ln.get_ydata(), float) for ln in ax.lines if ln.get_transform() == ax.transData and len(ln.get_ydata())]
    ys += [p.vertices[:, 1] for c in ax.collections if isinstance(c, PolyCollection) for p in c.get_paths()]
    ys += [np.asarray(seg, float)[:, 1] for c in ax.collections if isinstance(c, LineCollection)
           and c.get_transform() == ax.transData for seg in c.get_segments() if len(seg)]
    v = np.concatenate([np.ravel(y) for y in ys]) if ys else np.array([])
    v = v[np.isfinite(v)]
    if ax.get_yscale() == "log":
        v = v[v > 0]
    if not v.size:
        return
    if ax.get_yscale() == "log":
        a, b = np.log10(v.min()), np.log10(v.max())
        d = max(b - a, 0.05) * pad
        ax.set_ylim(10 ** (a - d), 10 ** (b + d))
    else:
        d = max(v.max() - v.min(), 1e-9) * pad
        ax.set_ylim(v.min() - d, v.max() + d)


def mode_of(key: str, mode: str) -> str:
    """From scratch is always solid ("ft"); every other network takes the use the figure gives it. A figure passes
    "probe" only where linear probes share a panel with fine-tuned networks (L6): otherwise solid."""
    return "ft" if key == "SCRATCH" else mode


def look(key: str, mode: str = "ft", **kw) -> dict:
    """color and linestyle of a network in a use (ft, probe, direct), for ax.plot(**look(...))."""
    return {"color": SERIES[key], "linestyle": MODE_LINE[mode_of(key, mode)], **kw}


def marker(key: str, mode: str = "ft", size: float | None = None) -> dict:
    """marker, fill and edge of a network in a use: hollow (white face) for a linear probe beside fine-tuned networks,
    filled otherwise."""
    out = {"marker": MARKERS[key], "markeredgecolor": SERIES[key],
           "markerfacecolor": "white" if mode_of(key, mode) == "probe" else SERIES[key]}
    if size is not None:
        out["markersize"] = size * STAR_SCALE if MARKERS[key] == "*" else size
    return out


# ------------------------------------------------------------------ round 4: exact print size (ROUND4_BRIEF Part 3)
# A figure is drawn at its printed width, 15 cm, with every panel placed in centimetres, and saved without a tight box,
# so 1 pt drawn is 1 pt printed. Text: axis labels and titles 9 pt, ticks and legends 8.5 pt (rule: at least 8 pt), so two
# panels of at least 6.5 cm fit beside each other (ticks at 8 pt) with their tick labels; the y-axis label is written once per row when
# the row shows one quantity.
PRINT_W = 15.0                                  # cm, \linewidth of the paper
SINGLE_W = 8.5                                  # cm, a one-panel figure (PI 2026-11-12): its panel the size of a 2x2 figure's
PANEL_W = 6.5                                   # cm, a panel of a two-column figure
PANEL_3 = 4.2                                   # cm, a panel of three in one row
LABEL_PT5, TICK_PT5, TITLE_PT5, LEGEND_PT5 = 9.0, 8.0, 9.0, 8.5


FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"     # lmroman10 (GUST Font License), from TeX
BODY_FONT = "Latin Modern Roman"        # the paper's body font (main.tex: article, Latin Modern; maths Computer Modern)


def use_style5(simplify: bool = False) -> None:
    use_style(simplify)
    from matplotlib import font_manager
    for f in sorted(FONT_DIR.glob("lmroman10-*.otf")):
        font_manager.fontManager.addfont(str(f))
    # PI 2026-11-11: every figure in the paper's body font, not STIX
    plt.rcParams.update({"font.family": BODY_FONT, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
    plt.rcParams.update({"font.size": LABEL_PT5, "axes.labelsize": LABEL_PT5, "axes.titlesize": TITLE_PT5,
                         "xtick.labelsize": TICK_PT5, "ytick.labelsize": TICK_PT5, "legend.fontsize": LEGEND_PT5,
                         "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 2.5, "axes.titlepad": 4.0})


def figure_cm(width: float, height: float):
    """A figure of exactly width x height cm, saved at that size (style_v3.save reads exact_size)."""
    fig = plt.figure(figsize=(width * CM, height * CM))
    fig.exact_size = True
    return fig


def axes_cm(fig, x: float, top: float, w: float, h: float, **kw):
    """Axes w x h cm whose upper left corner is x cm from the left and top cm from the top of fig."""
    W, H = fig.get_figwidth() / CM, fig.get_figheight() / CM
    return fig.add_axes((x / W, 1 - (top + h) / H, w / W, h / H), **kw)


def grid_cm(rows: int, cols: int = 2, *, w: float = PANEL_W, aspect: float = 0.85, ratio: float = 0.0,
            left: float = 1.05, wgap: float = 0.75, top: float = 0.6, hgap: float = 1.15, bottom: float = 0.95,
            ratio_gap: float = 0.1, width: float = PRINT_W):
    """(fig, axes[rows][cols], ratio axes[rows][cols] or None): panels w cm wide and aspect*w tall, each with a ratio
    sub-panel ratio*height tall under it when ratio > 0, sharing its x axis; top leaves room for the
    figure legend, hgap for the x tick labels and the next row's titles."""
    h = aspect * w
    rh = ratio * h
    cell = h + (rh + ratio_gap if ratio else 0)
    if left + cols * w + (cols - 1) * wgap > width + 1e-9:
        raise ValueError(f"{cols} panels of {w} cm do not fit in {width} cm")
    H = top + rows * cell + (rows - 1) * hgap + bottom
    fig = figure_cm(width, H)
    axes = np.empty((rows, cols), object)
    raxes = np.empty((rows, cols), object) if ratio else None
    for i in range(rows):
        for j in range(cols):
            x, y = left + j * (w + wgap), top + i * (cell + hgap)
            axes[i, j] = axes_cm(fig, x, y, w, h)
            if ratio:
                raxes[i, j] = axes_cm(fig, x, y + h + ratio_gap, w, rh, sharex=axes[i, j])
                axes[i, j].tick_params(labelbottom=False)
    return fig, axes, raxes


def title(ax, text: str) -> None:
    ax.set_title(text, fontsize=TITLE_PT5, pad=4.0)


def letter_panels(axes) -> None:
    for i, ax in enumerate(axes):
        ax.set_title(f"({chr(97 + i)}) {ax.get_title()}".rstrip(), fontsize=TITLE_PT5, pad=4.0)


# PI 2026-11-13 (L7): no figure title above the legend (the caption does that job), so the figure legend's top edge
# sits LEGEND_TOP cm below the figure's top (0.6 cm under a heading before, every top margin 0.45 cm smaller now)
LEGEND_TOP = 0.15


def legend(ax, **kw):
    kw.setdefault("loc", "upper right")
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", HANDLE)
    return ax.legend(fontsize=kw.pop("fontsize", LEGEND_PT5), borderpad=0.3, labelspacing=0.3, **kw)


def fig_legend(fig, handles, top_cm: float, ncol: int, **kw):
    """A figure-level legend centred above the panels, its top edge top_cm below the figure's top."""
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - top_cm / (fig.get_figheight() / CM)),
                      ncol=ncol, frameon=False, fontsize=LEGEND_PT5, borderpad=0.2, labelspacing=0.3,
                      columnspacing=1.2, handlelength=HANDLE, markerscale=1.0, **kw)     # PI 2026-11-12: keys the size of the plotted marks
