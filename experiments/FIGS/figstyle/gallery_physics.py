#!/usr/bin/env python3
"""Gallery mock-ups of the design schematic, the mass regression, the mass sculpting, the resonance
tagging, performance against jet pT and the JetClass confusion figures, in style_v4 only, with Sophon's Fig. 2 redrawn first as the check of that style. Layouts:
docs/figstyle/PLAN.md. Data come through paper_figures' selection code (gallery_common); nothing is drawn at
import."""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from gallery_bands import extend_to, spread_band
from gallery_common import HERE, RUNS, Absent, I, M, P, Page, S
from gallery_labelset import _log_y

GROOMED = "Soft-drop mass"
MASS_ROWS = (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))     # a mass model and its label set's name
NAMES = {**S.NAME, **{row: f"{S.NAME[base]}, mass regression" for row, base in MASS_ROWS}}
RATIO_ATLAS, RATIO_CMS = 0.44, 0.33     # lower axes over main axes: ATLAS 2024-015 Figs. 7-8, CMS Fig. 8 (catalogue)
STACK_GAP, COLUMN_GAP = 0.08, 0.32      # a main and its lower axes; two columns with their own y axes
SHOWN_FROM = 0.05       # a response axis covers the bins with this share of the largest bin of any series
HIST_ROOM = 1.45        # top of a histogram's y axis over its peak: room for the legend above the curves
SELECTION_COLOUR = {1: S.TAB10[0], 2: S.TAB10[6]}     # F10a, as CMS Fig. 8's: black inclusive; then CMS's blue, and pink:
# no colour of the palette (style_v4.SERIES), so no working point reads as a model
SCULPT_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS")
F10A_TOP = 260.0        # GeV, F10a: the bin edge that covers CMS Fig. 8's 30 to 250 GeV
F10B_FROM = 5e-3        # F10b: the first point is the 5.2e-3 grid point; CMS Fig. 9 left stops at 0.5%
DIRECT_LOOK = S.look("L188", linestyle="-.")     # F11: the 188-class model read out another way, as F8's linear probe
FITS = pathlib.Path("/data/results/eval/xbx/fits")      # the dry-run fits on the stand-in jets: cluster only
STANDIN = "xbb_jc2"


def _fit_y(ax, values: list, room: float = 0.0, pad: float = 0.08) -> None:
    """y limits from the finite values given. By hand since an ax.stairs line does not autoscale alike across
    matplotlib versions; room is empty space on top for a legend, as a share of the data's span."""
    v = np.concatenate([np.ravel(x) for x in values])
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    span = hi - lo or abs(hi) or 1.0
    ax.set_ylim(lo - pad * span, hi + (pad + room) * span)


# ------------------------------------------------------------------ the check of the style
# Sophon's eight curves in our palette (round 3): Sophon, trained on 188 classes, takes the 188 colour in both panels
# (its X->bq discriminant too, dotted as Sophon draws it); Sophon* (42 classes) the 43 colour; ParT and ParticleNet,
# taggers trained from scratch, the from-scratch dark grey and the mid grey. Line styles stay Sophon's.
SOPHON_COLOUR = {"Sophon": S.SERIES["L188"], "Sophon* (42 class)": S.SERIES["R42_Q1"],
                 "Sophon (transfer learning)": S.SERIES["L188"], "Sophon_{discr. = X->bq vs. QCD}": S.SERIES["L188"],
                 "ParT_{X->bb vs. QCD}": S.GREY_DARK, "ParT_{X->bs vs. QCD}": S.GREY_DARK,
                 "ParticleNet_{X->bb vs. QCD}": S.GREY_MID, "ParticleNet_{X->bs vs. QCD}": S.GREY_MID}


def draw_sophon_fig2(path: pathlib.Path) -> None:
    """Sophon Fig. 2 from the points read off its native PDF, through style_v4 and our palette; Sophon's figure size,
    panel gap (hspace 0.2, where style_v4 sets 0.25), limits, ticks, line widths and line styles."""
    d = json.loads((HERE / "sophon_fig2_points.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH, S.HEIGHT[2]), sharex=True, gridspec_kw={"hspace": 0.2})
    for k, (ax, p) in enumerate(zip(axes, d["panels"])):
        S.grid(ax)
        for c in p["curves"]:
            xy = np.array(c["points_xy"])
            ax.plot(xy[:, 0], xy[:, 1], color=SOPHON_COLOUR[c["name_printed"]], linewidth=S.LW,
                    linestyle=":" if c["dash_pdf"] else "-", label=c["name_mathtext"])
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 5.4)
        ax.set_yticks(range(6))
        S.title(ax, p["title_mathtext"])
        S.labels(ax, ylabel=p["y"]["label"])
        S.legend(ax, ncol=2 if k == 0 else 1)   # upper panel: two columns keep the frameless legend above the curves
    S.labels(axes[1], xlabel=d["panels"][1]["x"]["label"])
    S.save(fig, path)


# ------------------------------------------------------------------ F1
# Schematic rows of the four label sets, top to bottom, not to scale: each level's boxes are unions of the boxes
# to its left, the last row group is the QCD jets
BOXES = {"L188": (1,) * 14, "L162": (1,) * 11 + (3,), "R42_Q1": (2, 2, 2, 2, 2, 1, 3), "R16_Q1": (4, 4, 3, 3)}
MERGES = ("QCD quark\ncontent", "flavor beyond the\nheaviest quark, $s$ vs $u,d$,\nquark counts, gluon\nvs quark, $e$ vs $\\mu$",
          "$b$, $c$, light")
TRAINED, FROZEN = "#f2ead0", "#e6e6e6"     # round 4: no label-set hue outside the label-set columns (a warm neutral)
FAN = "#cccccc"
NEW_LAYER = "#8c8c8c"                       # the new head's weights: grey, not a model colour


def _network(ax, x: float, y: float, w: float, h: float, fill: str, words: str, size: float) -> list:
    """Sophon's network box with its feature column on the right edge; the column's node centres."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec="black", lw=0.8,
                                zorder=2))
    ax.text(x + (w - 0.12) / 2, y + h / 2, words, ha="center", va="center", fontsize=size, zorder=3)
    ax.add_patch(FancyBboxPatch((x + w - 0.14, y + 0.02), 0.14, h - 0.04, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec="black", lw=0.8, zorder=3))
    nodes = [(x + w - 0.07, v) for v in np.linspace(y + 0.12, y + h - 0.12, max(int(h / 0.14), 3))]
    for xy in nodes:
        ax.add_patch(Circle(xy, 0.04, fc="white", ec="black", lw=0.4, zorder=4))
    return nodes


def _hidden(ax, x: float, y0: float, y1: float, fill: str) -> list:
    """The head's hidden layer of 512 units (fc_params [(512, 0.1)], ReLU), drawn as a column of nodes."""
    nodes = [(x, v) for v in np.linspace(y0, y1, 6)]
    for xy in nodes:
        ax.add_patch(Circle(xy, 0.04, fc=fill, ec="black", lw=0.4, zorder=4))
    return nodes


def _fan(ax, a: list, b: list, colour: str, width: float) -> None:
    for p in a:
        for q in b:
            ax.plot(*zip(p, q), color=colour, linewidth=width, zorder=1.5)


def _use(ax, x: float, fill: str, word: str, action: str, name: str, hidden: bool) -> None:
    """One use of the features in (b): the network, a new head of two outputs (with a new 512-unit hidden layer
    when fine-tuning, as the fine-tuned network has one; a linear probe is one layer), what is trained, the name."""
    nodes = _network(ax, x, 0.72, 0.92, 0.84, fill, word, S.LEGEND_PT)
    outs = [(x + 1.8, 0.92), (x + 1.8, 1.36)]
    if hidden:
        mid = _hidden(ax, x + 1.36, 0.84, 1.44, "white")
        _fan(ax, nodes, mid, NEW_LAYER, 0.3)
        _fan(ax, mid, outs, NEW_LAYER, 0.3)
        ax.text(x + 1.36, 1.56, "512", ha="center", va="bottom", fontsize=S.LEGEND_PT, zorder=5)
    else:
        _fan(ax, nodes, outs, NEW_LAYER, 0.4)
    for xy in outs:
        ax.add_patch(Circle(xy, 0.05, fc="white", ec="black", lw=0.5, zorder=4))
    ax.text(x + 1.36, 0.6, action, ha="center", va="center", fontsize=S.LEGEND_PT, zorder=5,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#808080", lw=0.5))
    ax.text(x + 1.0, 0.3, name, ha="center", va="center", fontsize=S.LABEL_PT, zorder=5)


def draw_f1(path: pathlib.Path) -> None:
    w, h = S.FULL - 0.2, 4.9     # inches: one data unit is one inch on the page; save adds 0.1 in a side (pad_inches)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    bold = dict(fontsize=S.LABEL_PT, fontweight="bold", va="center")
    ax.text(0, 4.72, "(a) Pretraining", **bold)
    ax.text(3.9, 4.68, "Four nested label sets, one pretraining each", ha="center", va="center", fontsize=S.LABEL_PT)
    # the jets go into the one network
    ax.add_patch(Polygon([(0.05, 3.82), (0.05, 3.2), (0.5, 3.5)], fc="#d0d0d0", ec="none", zorder=2))
    ax.text(0.28, 3.05, "same jets", ha="center", va="center", fontsize=S.LEGEND_PT)
    ax.annotate("", xy=(0.72, 3.5), xytext=(0.5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0))
    nodes = _network(ax, 0.72, 2.95, 1.14, 1.1, TRAINED, "One network", S.LABEL_PT)
    ax.text(0.72 + 0.51, 3.2, "same initial\nweights", ha="center", va="center", fontsize=S.LEGEND_PT, zorder=3)
    ax.text(1.79, 4.13, "features", ha="center", va="center", fontsize=S.LEGEND_PT)
    # the label sets: one column each, a merged class spans the rows it merges
    # the head's hidden layer, between the features and every label set's outputs (fc_params [(512, 0.1)])
    hid = _hidden(ax, 2.2, 3.06, 3.94, TRAINED)
    ax.text(2.2, 4.13, "512", ha="center", va="center", fontsize=S.LEGEND_PT)     # as in (b); "512 hidden" ran into "features"
    _fan(ax, nodes, hid, FAN, 0.3)
    top, bottom, cw, pitch, x0 = 4.35, 2.75, 0.58, 0.80, 2.60     # the 17-class column ends at 5.58 in, inside the 5.71 in canvas
    rows = (top - bottom) / sum(BOXES["L188"])
    for k, (row, spans) in enumerate(BOXES.items()):
        x = x0 + k * pitch
        ax.text(x + cw / 2, top + 0.11, S.NAME[row], ha="center", va="center", fontsize=S.LABEL_PT)
        y = top
        for n in spans:
            ax.add_patch(Rectangle((x, y - n * rows), cw, n * rows, fc=to_rgba(S.SERIES[row], 0.25), ec="black", lw=0.5,
                                   zorder=2))
            y -= n * rows
        if spans[-1] > 1:
            ax.text(x + cw / 2, bottom + spans[-1] * rows / 2, "QCD", ha="center", va="center",
                    fontsize=S.LEGEND_PT, zorder=3)
    for a in hid:
        for r in range(sum(BOXES["L188"])):
            ax.plot(*zip(a, (x0, top - (r + 0.5) * rows)), color=FAN, linewidth=0.3, zorder=1)
    # what each merge removes
    ax.text(x0 + cw / 2 - 0.06, 2.62, "Each merge\nremoves:", ha="right", va="center", fontsize=S.LEGEND_PT)   # left of the first arrow, clear of the callout
    for k, words in enumerate(MERGES):
        x = x0 + k * pitch + cw / 2
        ax.annotate("", xy=(x + pitch, 2.69), xytext=(x, 2.69),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(x + pitch / 2, 2.63, words, ha="center", va="top", fontsize=S.LEGEND_PT)
    # the uses, on one card (Sophon's shadowed card and callout), since both start from the features
    ax.text(0, 2.02, "(b) Use", **bold)
    for dx, dy in ((0.05, 0.05), (0, 0)):
        ax.add_patch(Rectangle((0.25 + dx, 0.15 + dy), 5.4, 1.7, fc="#f0f0f0", ec="#808080", lw=0.6, zorder=0.5))
    tip, a, b = (1.79, 2.95), (1.3, 1.855), (2.2, 1.855)
    ax.add_patch(Polygon([tip, a, b], fc="#f0f0f0", ec="none", zorder=0.6))
    for corner in (a, b):
        ax.plot(*zip(tip, corner), color="#808080", linewidth=0.6, zorder=0.7)
    _use(ax, 0.45, FROZEN, "frozen", "Train a linear layer", "Frozen features, linear probe", hidden=False)
    _use(ax, 3.2, TRAINED, "trained", "Train all weights", "Fine-tuning", hidden=True)
    S.save(fig, path)


# ------------------------------------------------------------------ F9a, F9b, A17
ATLAS_FINE = M._load(P.DATA / "head" / "atlas_res_fine")     # ten pT bins (atlas_rebin.py); the same jets and inclusive values


def _atlas() -> dict:
    if not (ATLAS_FINE or I["atlas"]):
        raise Absent("no mass resolution files (head/atlas_res)")
    return ATLAS_FINE or I["atlas"]


GROOMED_LOOK = {"color": "#000000", "linestyle": "-", "linewidth": S.LW_THIN}    # solid black, thin: the reference


def _response(ax, atlas: dict, group: str, title: str) -> list:
    """The response histograms of the soft-drop mass and the two mass outputs (solid: nothing else shares the panel); returns the
    bins drawn and the series' peak."""
    g = P._groomed(atlas, group)["hist"]
    e = np.asarray(g["edges"], float)
    gr = P._frac(g)
    ax.stairs(gr, e, baseline=None, label=GROOMED, **GROOMED_LOOK)
    shown = [gr]
    for row, _ in MASS_ROWS:
        hs = [P._frac(b["hist"]) for m in RUNS[row] if (b := P._block(atlas, m, "regressed", group))]
        if hs:
            mean = np.mean(hs, axis=0)
            ax.stairs(mean, e, baseline=None, **S.look(row), linewidth=S.LW_CURVE, label=S.LEGEND[row])
            shown.append(mean)
    on = np.flatnonzero(np.any([s >= SHOWN_FROM * s.max() for s in shown], axis=0))
    ax.set_xlim(e[on[0]], e[on[-1] + 1])
    S.title(ax, title)
    S.labels(ax, xlabel=r"Response $r = m/m_{\mathrm{gen}}$")
    S.fine_grid(ax)
    return [max(s.max() for s in shown)]


def draw_f9a(path: pathlib.Path) -> None:
    """Round 4: two panels side by side, no ratio sub-panels (a 2x1 stack with them exceeds the page); one y range."""
    atlas = _atlas()
    fig, axes, _ = S.grid_cm(1, 2, w=6.65, aspect=0.85, top=1.6, left=1.2, wgap=0.3)
    peak = _response(axes[0, 0], atlas, "resonant", "Resonant jets") + _response(axes[0, 1], atlas, "qcd", "QCD jets")
    axes[0, 1].sharey(axes[0, 0])
    x0 = min(ax.get_xlim()[0] for ax in axes[0])
    x1 = max(ax.get_xlim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_xlim(x0, x1)                 # one response range: the two widths read against each other
    axes[0, 0].set_ylim(0, 1.08 * max(peak))
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))     # no 0 in the corner
    axes[0, 1].tick_params(labelleft=False)
    S.labels(axes[0, 0], ylabel="Fraction of jets")
    S.letter_panels(axes[0])
    S.heading(fig, "Mass regression: response")
    S.fig_legend(fig, axes[0, 0].get_legend_handles_labels()[0], top_cm=0.6, ncol=3)
    S.save(fig, path)


def _mass_series(mode: str = "ft") -> list:
    """(runs, estimator, line, mark, label) of the two mass-regression models' outputs: solid, or dashed (mode "mass")
    where they share a panel with the same label set's other output (A17-probe); one hollow marker per model in every
    figure (PI, 2026-11-07)."""
    return [(RUNS[row], "regressed", S.look(row, mode), S.marker(row, "mass"),
             S.LEGEND[row] + (" (mass output)" if mode == "mass" else "")) for row, _ in MASS_ROWS]


def _quantity(ax, rx, atlas, group, axis, series, which: str) -> tuple:
    """One column of one row: the median (which="median") or the resolution against axis, and below it the ratio
    to the soft-drop mass or the root square difference. The upper y range is set by the regression series, widened
    to the soft-drop mass where the series' differences stay readable (gallery_bands.extend_to, PI 2026-11-07).
    Returns (soft-drop values, bin edges) for _reference, drawn once the row's ranges are final."""
    g = P._groomed(atlas, group)[axis]
    e = np.asarray(g["edges"], float)
    c = 0.5 * (e[1:] + e[:-1])
    ref = P._arr(g[which])
    ax.plot([], [], **GROOMED_LOOK, label=GROOMED)                          # the legend key; bins drawn below
    rx.axhline(1.0 if which == "median" else 0.0, **GROOMED_LOOK)
    vals, lows = [], [[1.0 if which == "median" else 0.0]]
    for runs, est, line, mark, label in series:
        bl = [b[axis] for m in runs if (b := P._block(atlas, m, est, group))]
        if not bl:
            continue
        v = np.mean([P._arr(b[which]) for b in bl], axis=0)
        low = v / ref if which == "median" else np.sign(v - ref) * np.sqrt(np.abs(v ** 2 - ref ** 2))
        ax.stairs(v, e, baseline=None, **line, linewidth=S.LW_CURVE)
        rx.stairs(low, e, baseline=None, **line, linewidth=S.LW_CURVE)
        for a_, y_ in ((ax, v), (rx, low)):
            a_.plot(c, y_, linestyle="none", **mark, markersize=3)
        ax.plot([], [], **line, linewidth=S.LW_CURVE, **mark, markersize=3, label=label)     # the legend key
        vals.append(v)
        lows.append(low)
    _fit_y(ax, vals)
    _fit_y(rx, lows)
    if extend_to(*ax.get_ylim(), ref) is not None:
        _fit_y(ax, [*vals, ref])
    ax.set_xlim(e[0], e[-1])
    for a_ in (ax, rx):
        S.fine_grid(a_)
    return ref, e


def _reference(ax, ref, e) -> None:
    """The soft-drop mass in the bins where it lies in the y range; in the others an arrow at the top or bottom edge
    of the bin, and its values there in the panel title (inside the panel they would cross the series)."""
    lo, hi = ax.get_ylim()
    inside = (ref >= lo) & (ref <= hi)
    ax.hlines(ref[inside], e[:-1][inside], e[1:][inside], **GROOMED_LOOK)
    c = 0.5 * (e[1:] + e[:-1])
    words = []
    for out, y0, y1, sign in ((ref < lo, 0.1, 0.01, "\u2193"), (ref > hi, 0.9, 0.99, "\u2191")):
        if not out.any():
            continue
        for x in c[out]:
            ax.annotate("", xy=(x, y1), xytext=(x, y0), xycoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=7))
        v = ref[out]
        words.append(sign + (f"{v.min():.2f}" if np.ptp(v) < 0.005 else f"{v.min():.2f}\u2013{v.max():.2f}"))
    if words:
        S.title(ax, f"{GROOMED} " + " ".join(words))


X_PT, X_MGEN = r"Jet $p_{\mathrm{T}}$ [GeV]", r"Generator mass ($m_{\mathrm{gen}}$) [GeV]"


def _share_row(row) -> None:
    """One y range for the two panels of a row when each panel's own range covers at least half of their union (the
    tick labels then once, on the left); otherwise each keeps its own range and labels, so neither is flattened."""
    (a0, a1), (b0, b1) = row[0].get_ylim(), row[1].get_ylim()
    lo, hi = min(a0, b0), max(a1, b1)
    if min(a1 - a0, b1 - b0) >= 0.5 * (hi - lo):
        for ax in row:
            ax.set_ylim(lo, hi)
        row[1].tick_params(labelleft=False)


def _binned_2x2(path: pathlib.Path, group: str, series: list, heading: str, ncol: int) -> None:
    """Round 4 (ROUND4_BRIEF Part 3): median response (upper row) and resolution (lower row) against jet pT (left)
    and generator mass (right), 6.5 cm panels, each with its ratio (RSD) sub-panel; one y range per row; the legend
    above the panels."""
    atlas = _atlas()
    rows = math.ceil((len(series) + 1) / ncol)
    refs = np.empty((2, 2), object)
    fig, axes, raxes = S.grid_cm(2, 2, aspect=0.72, ratio=RATIO_ATLAS, top=0.95 + 0.45 * rows + 0.5, left=1.25,
                                 wgap=0.75, hgap=1.0)
    for i, which in enumerate(("median", "resolution")):
        for j, (axis, xlabel) in enumerate((("vs_pt", X_PT), ("vs_mgen", X_MGEN))):
            refs[i, j] = _quantity(axes[i, j], raxes[i, j], atlas, group, axis, series, which)
            raxes[i, j].tick_params(labelbottom=True)
            if i == 1:
                S.labels(raxes[i, j], xlabel=xlabel)
        for row_axes in (axes[i], raxes[i]):
            _share_row(row_axes)
        for j in range(2):
            _reference(axes[i, j], *refs[i, j])
        S.labels(axes[i, 0], ylabel=r"Median response ($r$)" if which == "median" else r"Resolution ($\sigma_r$)")
        S.labels(raxes[i, 0], ylabel="Ratio" if which == "median" else "RSD")
        for j in range(2):
            raxes[i, j].yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], prune="upper"))
            axes[i, j].yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], prune="lower"))
    fig.align_ylabels([axes[0, 0], raxes[0, 0], axes[1, 0], raxes[1, 0]])
    for rx in raxes[1]:         # RSD in one decimal in every figure
        rx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\u2212")))
    for rx in raxes[:, 1]:      # the right column's label at its last tick would run off the page
        hi = rx.get_xlim()[1]
        rx.xaxis.set_major_formatter(FuncFormatter(lambda v, _, hi=hi: "" if v >= hi - 1e-9 else f"{v:g}"))
    S.letter_panels(list(axes.flat))
    S.heading(fig, heading)
    S.fig_legend(fig, axes[0, 0].get_legend_handles_labels()[0], top_cm=0.6, ncol=ncol)
    S.save(fig, path)


def draw_f9b(path: pathlib.Path) -> None:
    _binned_2x2(path, "all", _mass_series(), "Mass regression against jet $p_{\\mathrm{T}}$ and generator mass", 3)


def _a17(path: pathlib.Path, group: str) -> None:
    _binned_2x2(path, group, _mass_series(), f"Mass regression on {'resonant' if group == 'resonant' else 'QCD'} jets", 3)


def draw_a17_probe(path: pathlib.Path) -> None:
    """Six series: the MLP on the frozen features of the four label sets (solid, filled) and the two mass outputs
    (dashed, hollow); the legend above the panels."""
    series = [(RUNS[a], "mlp", S.look(a, "ft"), S.marker(a, "ft"), f"{S.LEGEND[a]} {S.MLP_PROBE}") for a in S.LEVELS]
    series += _mass_series("mass")
    _binned_2x2(path, "all", series, "Mass from frozen features and from the mass output", 2)


# ------------------------------------------------------------------ F10a, F10b
def _sculpt() -> dict:
    if not I["sculpt"]:
        raise Absent("no mass sculpting curves (head/curves_v2/*_sculpting_v2.json)")
    return I["sculpt"]


def _points(ax, x, y, half, marker: str, colour: str, hollow: bool, width: float, label: str | None = None,
            zorder: float = 2) -> None:
    """CMS's points: a marker on the bin's centre and a horizontal bar across the bin, nothing joined."""
    ax.errorbar(x, y, xerr=half, fmt=marker, color=colour, markerfacecolor="none" if hollow else colour,
                markersize=2.5, elinewidth=width, capsize=0, label=label, zorder=zorder)


def _sculpt_panels(path: pathlib.Path, rows: tuple, heading: str) -> None:
    """Round 4: one 6.5 cm panel per model, two per row, each with its ratio to the inclusive distribution (CMS's
    height); working points in black and greys (style_v5.WORKING_POINT), no model colour, since the title names the
    model; one y range per row."""
    sc = _sculpt()
    shown = [(a, ms) for a in rows if (ms := [m for m in RUNS[a] if m in sc])]
    if not shown:
        raise Absent("no mass sculpting curve of the models shown")
    nrows = math.ceil(len(shown) / 2)
    fig, axes, raxes = S.grid_cm(nrows, 2, aspect=0.75, ratio=RATIO_CMS, top=1.6, left=1.3, wgap=0.35, hgap=1.0)
    panels = list(zip(axes.flat, raxes.flat))
    peak, ratios = 0.0, [[1.0]]
    for k_panel, ((ax, rx), (a, ms)) in enumerate(zip(panels, shown)):
        e = np.asarray(sc[ms[0]]["bins_jet_sdmass"], float)
        keep = e[1:] <= F10A_TOP
        mid, half = ((e[:-1] + e[1:]) / 2)[keep], (np.diff(e) / 2)[keep]
        before = np.asarray(sc[ms[0]]["hist_before"], float)
        before = before / before.sum()
        wp_in = S.WORKING_POINT["inclusive"]
        _points(ax, mid, before[keep], half, wp_in["marker"], wp_in["color"], False, S.LW_THIN, "Inclusive", zorder=2.1)
        peak = max(peak, before[keep].max())
        rx.axhline(1.0, **S.REFERENCE, zorder=1)      # under the points
        for wp, _ in P.WPS:
            look = S.WORKING_POINT[f"{100 * float(wp):g}%"]
            after = np.mean([np.asarray(sc[m]["working_points"][wp]["hist_after"], float)
                             / np.sum(sc[m]["working_points"][wp]["hist_after"]) for m in ms], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            label = f"$\\varepsilon_B = {100 * float(wp):g}\\%$"
            hollow = look["fillstyle"] == "none"
            _points(ax, mid, after[keep], half, look["marker"], look["color"], hollow, S.LW_THIN, label)
            _points(rx, mid, r[keep], half, look["marker"], look["color"], hollow, S.LW_THIN)
            peak = max(peak, after[keep].max())
            ratios.append(r[keep])
        ax.set_xlim(e[0], F10A_TOP)
        ax.tick_params(labelbottom=False)
        S.title(ax, NAMES[a])
        if k_panel >= len(shown) - 2:
            S.labels(rx, xlabel=r"Jet mass ($m_{\mathrm{SD}}$) [GeV]")
        S.fine_grid(ax)
        S.fine_grid(rx)
    for ax, rx in panels[len(shown):]:
        ax.axis("off")
        rx.axis("off")
    for i in range(nrows):
        axes[i, 0].set_ylim(0, 1.1 * peak)
        _fit_y(raxes[i, 0], ratios, pad=0.15)        # markers at the extremes clear of the frame
        for j in (1,):
            axes[i, j].sharey(axes[i, 0])
            raxes[i, j].sharey(raxes[i, 0])
            axes[i, j].tick_params(labelleft=False)
            raxes[i, j].tick_params(labelleft=False)
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        raxes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
        S.labels(axes[i, 0], ylabel="Fraction of jets")
        S.labels(raxes[i, 0], ylabel="Ratio to\ninclusive")
    S.letter_panels([ax for ax, _ in panels[:len(shown)]])
    S.heading(fig, heading)
    S.fig_legend(fig, axes[0, 0].get_legend_handles_labels()[0], top_cm=0.6, ncol=3)
    S.save(fig, path)


def draw_f10a(path: pathlib.Path) -> None:
    _sculpt_panels(path, tuple(S.LEVELS), "QCD jet mass before and after a cut on the pretraining head")


def draw_f10a_mass(path: pathlib.Path) -> None:
    _sculpt_panels(path, ("L162_MASS", "R16_Q1_MASS"),
                   "QCD jet mass before and after a cut on the pretraining head, with mass regression")


def draw_f10a_flav(path: pathlib.Path) -> None:
    _sculpt_panels(path, tuple(S.CONTROLS),
                   "QCD jet mass before and after a cut on the pretraining head, flavor-boundary partitions")


def _floor(sc: dict, m: str) -> dict:
    """{efficiency: the distance a random selection of as many QCD jets shows} on run m's curve (mlst_results._curve,
    the tables' floor): the same at every point for every run, which pass the same number of jets."""
    return {e: f for e, (_, f) in M._curve(sc, m).items()}


def _sculpt_curves(path: pathlib.Path, sc: dict, rows: tuple, heading: str, runs: dict | None = None) -> None:
    """Round 4: exact size, one 10 cm panel, the legend above it; each row's mean over runs with a band of the spread
    over runs (gallery_bands.spread_band), the label sets solid and filled, the mass models dashed and hollow (mode "mass"); the
    floor black dashed; both axes logarithmic, labelled in plain numbers. runs: {row: models}, RUNS by default."""
    runs = RUNS if runs is None else runs
    fig = S.figure_cm(S.PRINT_W, 10.6)
    ax = S.axes_cm(fig, 3.2, 2.4, 9.6, 7.1)
    S.fine_grid(ax)
    drawn = []
    for row in rows:
        have = [sc[m]["curve"] for m in runs[row] if m in sc]
        if not have:
            continue
        eff = np.array([c["eff"] for c in have[0]], float)
        per = np.array([[c["js_distance"] for c in h] for h in have], float)
        keep = eff >= F10B_FROM
        mode = "mass" if row in S.MASS else "ft"        # beside the 162- and 17-class curves of the same colour
        ax.plot(eff[keep], per.mean(axis=0)[keep], **S.look(row, mode), linewidth=S.LW_CURVE, **S.marker(row, mode, 3.5),
                label=S.LEGEND[row])
        if (band := spread_band(per)) is not None:
            ax.fill_between(eff[keep], band[0][keep], band[1][keep], color=S.SERIES[row], alpha=0.2, linewidth=0)
        drawn.append(row)
    if not drawn:
        raise Absent("no mass sculpting curve of the models shown")
    ref = _floor(sc, next(m for m in runs[drawn[0]] if m in sc))
    eff = np.array(sorted(ref))
    keep = eff >= F10B_FROM
    ax.plot(eff[keep], [ref[e] for e in eff[keep]], **S.REFERENCE, zorder=1.5, label="random selection of as many QCD jets")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis, lim, at in ((ax.xaxis, ax.get_xlim(), (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)),
                          (ax.yaxis, ax.get_ylim(), (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5))):
        axis.set_major_locator(FixedLocator([v for v in at if lim[0] <= v <= lim[1]]))
        axis.set_major_formatter(FuncFormatter((lambda v, _: f"{100 * v:g}%") if axis is ax.xaxis else (lambda v, _: f"{v:g}")))
        axis.set_minor_formatter(NullFormatter())       # efficiency in percent, as F10a's working points
    S.heading(fig, heading)
    S.labels(ax, r"QCD efficiency ($\varepsilon_B$)", "Jensen-Shannon distance")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=0.6, ncol=3)
    S.save(fig, path)


def draw_f10b(path: pathlib.Path) -> None:
    """The label sets and the two mass models (PI, 2026-11-08); the flavor boundaries on F10b-flav."""
    _sculpt_curves(path, _sculpt(), SCULPT_ROWS, "Mass sculpting of QCD jets by cuts on the pretraining head")


def draw_f10b_flav(path: pathlib.Path) -> None:
    sc = _sculpt()
    if not any(m in sc for a in S.CONTROLS for m in RUNS[a]):
        raise Absent("no mass sculpting curve of the flavor-boundary partitions")
    _sculpt_curves(path, sc, ("L188",) + tuple(S.CONTROLS),
                   "Mass sculpting of QCD jets by cuts on the pretraining head, flavor boundaries")


SCULPT_FT = "head/sculpting_ft"     # the fine-tuned networks' sculpting files, <model>.json (EVAL/sculpting_ft.py)


def draw_f10_ft(path: pathlib.Path) -> None:
    """F10b's form after fine-tuning on JetClass-II, scored on the natural-mix test jets the pretraining heads are
    (unweighted), so the floor is F10b's (_floor)."""
    sc = {p.stem.removesuffix("_sculpting"): json.loads(p.read_text()) for p in sorted((P.DATA / SCULPT_FT).glob("*.json"))}
    if not sc:
        raise Absent(f"no fine-tuned sculpting files ({SCULPT_FT}/<model>.json)")
    scratch = [f"scratch-s{k}" for k in (1, 2, 3)]
    if not all(m in sc for m in scratch):
        raise Absent(f"the three from-scratch networks' sculpting files ({SCULPT_FT}/scratch-s<k>.json)")
    _sculpt_curves(path, sc, SCULPT_ROWS + ("SCRATCH",), "Mass sculpting of QCD jets after fine-tuning on JetClass-II",
                   {**RUNS, "SCRATCH": scratch})


# ------------------------------------------------------------------ F10c (DisCo Fig. 3's plane)
F10C_KEY = "sig_eff_50"     # per run (head/sculpting_v3): {"rejection": 1/eps_B, "jsd": Jensen-Shannon divergence} at 50% eps_S
F10C_ROWS = tuple(S.LEVELS) + S.MASS + tuple(S.CONTROLS)


def draw_f10c(path: pathlib.Path) -> None:
    """One point per model: the mean over runs of R50 (x) and 1/JSD50 (y), with the spread over pretraining runs on
    both axes (mlst_figures._spread: SD for three runs, range for two); DisCo Fig. 3's axes, both logarithmic."""
    sc = I["sculpt"]
    fig = S.figure_cm(S.PRINT_W, 10.6)
    ax = S.axes_cm(fig, 3.2, 2.4, 9.6, 7.1)
    S.fine_grid(ax)
    drawn = 0
    for row in F10C_ROWS:
        pts = [sc[m][F10C_KEY] for m in RUNS[row] if F10C_KEY in sc.get(m, {})]
        if not pts:
            continue
        (x, xl, xh), (y, yl, yh) = (P.F._spread(v) for v in ([p["rejection"] for p in pts], [1 / p["jsd"] for p in pts]))
        mode = "mass" if row in S.MASS else "ft"
        ax.errorbar(x, y, xerr=None if xl is None else [[xl], [xh]], yerr=None if yl is None else [[yl], [yh]],
                    linestyle="none", color=S.SERIES[row], elinewidth=S.LW_THIN, capsize=0,
                    **S.marker(row, mode, 5), label=S.LEGEND[row])
        drawn += 1
    if not drawn:
        raise Absent(f"no {F10C_KEY} in the sculpting files (head/sculpting_v3/*_sculpting.json)")
    _log_y(ax, "x")
    _log_y(ax)
    S.heading(fig, "Cuts on the pretraining head: mass sculpting against rejection at 50% signal efficiency")
    S.labels(ax, r"QCD rejection at 50% resonant efficiency ($R_{50}$)", r"$1/\mathrm{JSD}_{50}$")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=0.6, ncol=3)      # three rows, clear of the panel
    S.save(fig, path)


# ------------------------------------------------------------------ F11
def _standin_series(XR, fits: dict) -> list:
    """(label, look, readouts per run) on the stand-in jets: the 188-class output scores (Sophon's direct
    discriminant, Eq. 4), then each label set's classifier on the frozen features (the fits' "head"; the linear
    probe where a fit has none)."""
    of = lambda row, key: [r for m in RUNS[row] if (r := XR.readout(fits.get(m, {}).get(STANDIN, {}), key))]
    out = []
    if direct := of("L188", "zero_shot_xbb"):
        out.append((f"{S.LEGEND['L188']} (direct discriminant)", DIRECT_LOOK, direct))
    for row in S.LEVELS:
        for key, word in (("head", "(transfer)"), ("linear", S.PROBE)):
            if rs := of(row, key):
                out.append((f"{S.LEGEND[row]} {word}", S.look(row), rs))
                break
    return out


def draw_f11(path: pathlib.Path) -> None:
    if not FITS.is_dir():
        raise Absent(f"the stand-in fits ({FITS}) are on the cluster only")
    XR = P._xr()
    series = _standin_series(XR, XR.load(FITS))
    if not series:
        raise Absent(f"no {STANDIN} readout of the label sets under {FITS}")
    fig, axes = plt.subplots(2, 1, figsize=S.ONE_PANEL, sharex=True)
    top = 0.0
    for label, style, rs in series:
        x, y, _ = P.xbx_curves(M, rs)
        mean = y.mean(axis=0)
        axes[0].plot(x, mean, **style, linewidth=S.LW_CURVE, label=label)
        top = max(top, float(np.nanmax(mean)))
    S.legend(axes[0])
    for ax, what in zip(axes, (r"$X\rightarrow bb$ tagging performance, stand-in jets",
                               r"$X\rightarrow bs$ tagging performance")):
        S.fine_grid(ax)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 1.5 * top)
        S.title(ax, what)
        S.labels(ax, ylabel=r"Figure of merit ($\varepsilon_S/\sqrt{\varepsilon_B}$)")
    S.labels(axes[1], xlabel=r"QCD efficiency ($\varepsilon_B$)")
    S.letter_panels(axes)
    S.save(fig, path)


# ------------------------------------------------------------------ PT, CONF (round 3, docs/figstyle/TAGGER_GAP.md)
# experiments/EVAL/perf_vs_pt.py's files, pulled by pull_mlst_data.py
PERF, CONF = P.DATA / "perf_vs_pt", P.DATA / "confusion_jc1"
PT_MARKER = 3.5


def _seed(model: str) -> str:
    return model.rsplit("-s", 1)[1]


def _by_seed(docs: dict, row: str) -> dict:
    """{seed: doc} of the runs of row that have a file."""
    return {_seed(m): docs[m] for m in RUNS[row] if m in docs}


def _edges(docs: dict) -> np.ndarray:
    edges = {tuple(d["pt_edges"]) for d in docs.values()}
    if len(edges) != 1:
        raise SystemExit(f"FATAL: the perf_vs_pt files hold {len(edges)} different pT binnings")
    return np.asarray(edges.pop(), float)


def _accuracy(doc: dict) -> np.ndarray:
    return np.array([np.nan if b["accuracy"] is None else b["accuracy"] for b in doc["bins"]])


def _bb_cc(doc: dict) -> np.ndarray:
    return np.array([np.nan if b["bb_vs_cc"] is None else b["bb_vs_cc"]["auc"] for b in doc["bins"]])


def _pt_series(ax, c, values: list, key: str, label: str, band: bool = True) -> None:
    """Mean over runs at the bin centres, joined, the series' marker; band the SD (range for two runs), as F8."""
    st = [P.F._spread(list(v)) for v in np.asarray(values).T]
    mean = np.array([s[0] for s in st])
    style = S.look(key, linewidth=S.LW_CURVE)
    ax.plot(c, mean, **style, **S.marker(key, size=PT_MARKER), label=label)
    if band and len(values) > 1:
        ax.fill_between(c, mean - np.array([s[1] for s in st]), mean + np.array([s[2] for s in st]),
                        color=style["color"], alpha=0.2, linewidth=0)


def _pt_panel(ax, docs: dict, rows: tuple, value, ylabel: str) -> None:
    """value(doc) per pT bin for each row's runs, absolute (round 4: accuracy as F2, AUC for X->bb vs X->cc)."""
    e = _edges(docs)
    c = 0.5 * (e[1:] + e[:-1])
    for row in rows:
        r = [value(d) for d in _by_seed(docs, row).values()]
        if r:
            _pt_series(ax, c, r, row, S.LEGEND[row])
    ax.set_xlim(e[0], e[-1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    if ylabel:
        S.labels(ax, ylabel=ylabel)


def draw_pt(path: pathlib.Path) -> None:
    """Round 4 (ROUND4_BRIEF Part 4): absolute values per pT bin, no ratio. Upper row JetClass-II accuracy of the
    linear probes, (a) per label set and (b) for the controlled partitions with the 188- and 17-class models for
    reference, on one y axis; lower row (c) JetClass accuracy after fine-tuning and (d) the probes' AUC of X->bb
    against X->cc, its quantity in its title (two y labels do not fit beside 6.5 cm panels). No from scratch."""
    jc2, jc1 = M._load(PERF / "jc2"), M._load(PERF / "jc1")
    if not (jc2 and jc1):
        raise Absent("no performance against jet pT (perf_vs_pt/jc2, perf_vs_pt/jc1)")
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=2.35, left=1.25, wgap=0.75, hgap=1.6)
    (a, b), (cx, d) = axes
    panels = ((a, jc2, S.LEVELS, _accuracy, "Accuracy", "JetClass-II, linear probes"),
              (b, jc2, ("L188", "R16_Q1") + tuple(S.CONTROLS), _accuracy, None, "JetClass-II, linear probes, controls"),
              (cx, jc1, S.LEVELS, _accuracy, "Accuracy", "JetClass, fine-tuned"),
              (d, jc2, S.LEVELS, _bb_cc, None, r"Linear probes, AUC $X\to b\bar{b}$ vs $c\bar{c}$"))
    for k, (ax, docs, rows, value, ylabel, text) in enumerate(panels):
        S.title(ax, text)
        S.fine_grid(ax)
        _pt_panel(ax, docs, rows, value, ylabel)
        S.labels(ax, xlabel=X_PT)          # (c) is JetClass, 500-1000 GeV: every panel its own
    lo, hi = min(a.get_ylim()[0], b.get_ylim()[0]), max(a.get_ylim()[1], b.get_ylim()[1])
    for ax in (a, b):
        ax.set_ylim(lo, hi)
    b.tick_params(labelleft=False)
    d.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))     # two decimals: room beside (c)
    d.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    cx.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))    # three decimals, as (a)
    fig.align_ylabels([a, cx])
    for ax in (a, b, d):        # the last label at 2500 would run off the page or into the next panel
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, hi=ax.get_xlim()[1]: "" if v >= hi - 1e-9 else f"{v:g}"))
    handles = {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    S.fig_legend(fig, list(handles.values()), top_cm=0.6, ncol=4)
    S.letter_panels(axes.flat)
    S.heading(fig, "Performance against jet $p_{\\mathrm{T}}$")
    S.save(fig, path)


CONF_CELL_PT = 8.0      # numbers in the cells: integers and one decimal fit 0.65 cm cells at 8 pt (round 4)


def _confusion(doc: dict) -> np.ndarray:
    """Row-normalised [true][predicted] in S.JC1_CLASSES order."""
    k = [doc["classes"].index(c) for c in S.JC1_CLASSES]
    rows = doc["row_normalised"]
    return np.array([[np.nan] * len(k) if rows[i] is None else [rows[i][j] for j in k] for i in k], float)


def _cells(ax, values: np.ndarray, text: list, dark) -> None:
    for (i, j), t in np.ndenumerate(np.asarray(text, dtype=object)):
        if t:
            ax.text(j, i, t, ha="center", va="center", fontsize=CONF_CELL_PT, color="white" if dark(values[i, j]) else "black")


def draw_conf(path: pathlib.Path) -> None:
    docs = M._load(CONF)
    ref = {s: _confusion(d) for s, d in _by_seed(docs, "L188").items()}
    if not ref:
        raise Absent("no JetClass confusion matrices of the 188-class model (confusion_jc1)")
    base = np.mean(list(ref.values()), axis=0)
    diffs = {row: 100 * np.mean([_confusion(d) - ref[s] for s, d in _by_seed(docs, row).items() if s in ref], axis=0)
             for row in S.LEVELS[1:] if set(_by_seed(docs, row)) & set(ref)}
    limit = max([float(np.nanmax(np.abs(v))) for v in diffs.values()] or [1.0])
    n = len(S.JC1_CLASSES)
    names = [S.JC1_TEX[c] for c in S.JC1_CLASSES]

    # round 4: four 6.5 cm squares; the colour bars run horizontally above their columns at the panels' full width
    # (two squares and the class names fill the 15 cm, so a vertical bar has no room); (a) in integer percent as CMS
    # prints it, (b)-(d) a difference only where it reaches one percentage point
    w, left, gap, cb_top, top, vgap = 6.5, 1.75, 0.15, 1.85, 2.75, 0.55
    fig = S.figure_cm(S.PRINT_W, top + 2 * w + vgap + 2.25)
    xs = (left, left + w + gap)
    panels = [S.axes_cm(fig, x, top + i * (w + vgap), w, w) for i in range(2) for x in xs]
    base_pc = 100 * base
    im = panels[0].imshow(base_pc, cmap="Blues", vmin=0, vmax=100, interpolation="none")
    _cells(panels[0], base_pc, [[f"{v:.0f}" for v in r] for r in base_pc], lambda v: v > 60)
    S.title(panels[0], S.NAME["L188"])
    images = [im]
    for ax, row in zip(panels[1:], S.LEVELS[1:]):
        S.title(ax, f"{S.NAME[row]} minus 188")
        if row in diffs:
            v = diffs[row]
            images.append(ax.imshow(v, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none"))
            _cells(ax, v, [[f"{x:.1f}".replace("-", "\u2212") if abs(x) >= 1.0 else "" for x in r] for r in v],
                   lambda x: abs(x) > 0.6 * limit)
    for k, ax in enumerate(panels):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names if k >= 2 else [], rotation=90)
        ax.set_yticklabels(names if k % 2 == 0 else [])
        ax.tick_params(which="both", length=0, top=False, right=False)
        ax.minorticks_off()
        S.labels(ax, "Predicted class" if k >= 2 else None, "True class" if k % 2 == 0 else None)
    for x, image, word in ((xs[0], images[0], "(a): fraction of the true class\n(percent)"),
                           (xs[1], images[-1], "(b)-(d): difference from 188 classes\n(percentage points)")):
        cax = S.axes_cm(fig, x, cb_top, w, 0.28)
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.ax.minorticks_off()
        if image is images[0]:
            cb.set_ticks([0, 20, 40, 60, 80])       # no "100" at the end, where it would meet the other bar
        cb.set_label(word, fontsize=S.LABEL_PT5, labelpad=4)
    S.letter_panels(panels)
    S.heading(fig, "JetClass, fine-tuned on $10^6$ jets: confusion matrices")
    S.save(fig, path)


# ------------------------------------------------------------------ the pages
SOPHON_FIG2_CAPTION = (
    "FIG. 2. Benchmark of Sophon model’s performance in the X→bb and X→bs jet tagging tasks, with the signal "
    "originated from a BSM resonance X0 with mX0 = 200 GeV that decays to bb and bs, and backgrounds corresponding "
    "to the 40 fb−1 of the full SM processes. The performance of various DNN models is compared in terms of the "
    "search’s discovery significance versus SM background efficiency calculated within the mass window "
    "150 < mSD < 230 GeV. A major conclusion is that the Sophon model’s direct tagging discriminant (for X→bb "
    "tagging) and its transfer learning version (for X→bs tagging) both outperform the current best results "
    "achievable in the LHC experiment using a ParT or ParticleNet tagger. It also confirms that the model improves "
    "performance when trained by large-scale classification. [Validation page, not one of our figures: Sophon's "
    "own caption, verbatim.]")
F1_CAPTION = (
    "Design of the study. (a) One network is pretrained on the same JetClass-II jets four times, once for each of "
    "four nested label sets with 188, 162, 43 and 17 classes, through a hidden layer of 512 units; below each "
    "merge is the distinction it removes. (b) The pretrained network is used downstream in two ways: a linear "
    "probe on its frozen features, or a new head with its own 512-unit hidden layer and all weights fine-tuned.")
F9A_CAPTION = (
    "The mass output of networks pretrained on JetClass-II with 162 and 17 classes and an auxiliary mass "
    "regression, used without further training: distribution of the response r = m/m_gen, a jet's mass over its "
    "generator-level mass, for (a) resonant and (b) QCD test jets, with the soft-drop mass (solid black). Each "
    "curve is the mean of two pretraining runs; the two networks nearly coincide. Both panels share their axes.")
F9B_CAPTION = (
    "The mass output of networks pretrained on JetClass-II with 162 and 17 classes and an auxiliary mass "
    "regression (two runs): median response r = m/m_gen (a, b) and resolution sigma_r, half the 15.9-84.1% "
    "quantile range of r over its median (c, d), of test jets against pT (a, c) and generator mass (b, d; 20-500 "
    "GeV). Lower panels: ratio to the soft-drop mass and signed root square difference (RSD). Arrows: soft-drop "
    "mass off the axis, its values in the title.")
A17_CAPTION = (
    "The mass output of networks pretrained on JetClass-II with 162 and 17 classes and an auxiliary mass "
    "regression (two runs), on {group} test jets: median response r (a, b) and resolution sigma_r (c, d), as in "
    "the main-text figure, against pT (a, c) and generator mass (b, d; 20-500 GeV). Lower panels: ratio to the "
    "soft-drop mass and signed root square difference (RSD). Arrows: soft-drop mass off the axis, its values in "
    "the panel title.")
F10A_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, used "
    "without further training: soft-drop mass of QCD test jets before any selection (inclusive) and after cuts on "
    "1 - g_QCD, one minus the summed QCD score, at QCD efficiencies eps_B of 10% and 1% (mean of three runs). "
    "Bars show the bin width; the axis stops at 260 GeV. Below each panel, the ratio to the inclusive "
    "distribution.")
F10A_MASS_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 162 and 17 classes and an auxiliary "
    "mass regression, used without further training: soft-drop mass of QCD test jets before any selection and "
    "after cuts on 1 - g_QCD at QCD efficiencies eps_B of 10% and 1% (mean of two runs), with the ratio to the "
    "inclusive distribution below. The axis stops at 260 GeV.")
F10A_FLAV_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 17 classes and no flavor boundary, a "
    "b-quark boundary or a random flavor boundary, used without further training: soft-drop mass of QCD test jets "
    "before any selection and after cuts on 1 - g_QCD at QCD efficiencies eps_B of 10% and 1% (mean of two runs), "
    "with the ratio to the inclusive distribution below. The axis stops at 260 GeV.")
F10B_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (solid, "
    "three runs) and with 162 and 17 classes plus mass regression (dashed, two runs), used without further "
    "training: Jensen-Shannon distance between the soft-drop mass of all QCD test jets and of those passing a "
    "cut on 1 - g_QCD, against its QCD efficiency eps_B. Bands: SD over runs, range over two. Black dashed: a "
    "random selection of as many QCD jets.")
F10B_FLAV_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 17-class partitions with no, a "
    "b-quark and a random flavor boundary (two runs each), and with 188 classes for reference (three runs), used "
    "without further training: Jensen-Shannon distance between the soft-drop mass of all QCD test jets and of "
    "those passing a cut on 1 - g_QCD, against its QCD efficiency eps_B. Bands: range over runs, SD over three. "
    "Black dashed: a random selection of as many QCD jets.")
F10_FT_CAPTION = (
    "Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (three runs) and 162 and 17 classes plus "
    "mass regression (two runs), and the same network without pretraining (three seeds), fine-tuned on one million "
    "class-balanced JetClass-II jets and scored on 2 million natural-mix test jets: Jensen-Shannon "
    "distance between QCD soft-drop mass before and after a cut on 1 - g_QCD, against QCD efficiency eps_B. Bands: "
    "SD, range over two. Black dashed: random selection of as many QCD jets.")
F10C_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II, used without further training, in the "
    "plane of the DisCo paper's Fig. 3: the inverse Jensen-Shannon divergence between the soft-drop mass of all "
    "QCD test jets and of those passing a cut on 1 - g_QCD (y), against the QCD rejection (x), both at the cut "
    "that keeps 50% of resonant test jets. Points are means, bars the SD over pretraining runs (range over two).")
F11_CAPTION = (
    "Networks pretrained on JetClass-II: signal efficiency over the square root of the background efficiency "
    "against QCD efficiency for X→bb on stand-in JetClass-II jets (upper) and X→bs (lower, no samples yet), from "
    "the direct discriminant of the 188-class pretraining head and from a classifier trained on the frozen "
    "features of each network (LP: linear probe). Curves are means over pretraining runs and end where fewer than "
    "20 background jets remain.")
PT_CAPTION = (
    "Networks pretrained on JetClass-II, in bins of jet transverse momentum: (a) accuracy of linear probes on the "
    "frozen features, trained and scored on class-balanced JetClass-II, with 188, 162, 43 and 17 pretraining "
    "classes; (b) the same for 17-class partitions with controlled flavor boundaries, with 188 and 17 classes for "
    "reference; (c) accuracy on JetClass after fine-tuning on one million jets; (d) the probes' AUC of X→bb̄ "
    "against X→cc̄. Points are means, bands the SD over pretraining runs.")
CONF_CAPTION = (
    "Networks pretrained on JetClass-II, fine-tuned on one million JetClass jets: (a) the percentage of each true "
    "class assigned to each predicted class with 188 pretraining classes; (b)-(d) the change in that percentage "
    "with 162, 43 and 17 classes, each against the 188-class network of the same seed, printed where it reaches "
    "one percentage point, which no change in (c) does. Values are means over three pretraining runs.")

ATLAS_LOOK = ("Sophon's font and sizes in place of ATLAS's; inward ticks on four sides and no legend frame "
              "(2603.04084, round 3); no experiment label, and ATLAS's text block (sqrt(s), jet definition, pT and mass selection) is dropped, so our selection appears on no figure")
A17_DIFFERS = (
    "As F9b, on one jet group and with a second row against generator mass, which ATLAS does not draw (its Fig. 9 has "
    "pT response instead); the legend appears once, in the first panel, to keep four panels light. Eight axes in all, a lower axes under each "
    "of four panels: within the limit of four panels only if lower axes do not count, for the PI to decide. 5.6 in high, not "
    "the 4.8 in of a plain 2x2, because each panel carries a lower axes. The rows hold different jets: the lower covers "
    "generator mass 20 to 500 GeV, so 24.5% of QCD and 4.7% of resonant jets are in the upper row only. No window on the "
    "regressed mass, where ATLAS keeps 40 < m^Reg < 300 GeV. " + ATLAS_LOOK + ".")
MASS_RUNS_NOTE = ("Mean of two runs for each mass regression model (the third runs were cut on 2026-10-22); "
                  "the soft-drop mass is the same in every file. Nothing else is absent.")

PAGES = [
    Page("SOPHON-FIG2", "Sophon Fig. 2 redrawn through the paper's style (validation)", ["2405.12972/fig2.png"],
         "Sophon Fig. 2", SOPHON_FIG2_CAPTION,
         "Round 4: only the palette (style_v5) and the background (grid, ticks, legend frame: 2603.04084's, round 3) "
         "differ from Sophon's figure; the data, axes, sizes, line styles and text are Sophon's, so this page is exempt "
         "from the round-4 layout checks. In detail, against Sophon's native settings (sophon_native.json): grid a faint dashed "
         "#b0b0b0 line at major y ticks only (0.35 pt, alpha 0.25), where Sophon's is 0.5 pt at full strength at major "
         "and minor x ticks and major y ticks; spines 0.9 pt, not 0.8; ticks pointing in, 5.3 and 2.7 pt, mirrored on all "
         "four sides with minor ticks on y too, where Sophon's point out (3.5 and 2 pt) on the left and bottom only, with "
         "no y minor ticks; no legend frame, where Sophon's is framed (#cccccc, alpha 0.8, rounded); palette ours: Sophon "
         "(and its X→bq discriminant, dotted) in the 188 colour, Sophon* (42 classes) in the 43 colour, ParT dark grey and "
         "ParticleNet mid grey, where Sophon has tab10 blue, orange and green, pure blue and grey. Unchanged and checked "
         "against the native PDF by compare_fig2.py: the data points, axes box (279 x 98.84 pt each), scales, limits, "
         "y ticks, eight curves at 1.5 pt with Sophon's solid and dotted dashes, every vertex inside the axes at the "
         "native position (0.0000 pt off); also STIX text, its sizes, the figure size and the panel gap (hspace 0.2, kept "
         "against style_v4's 0.25). The file encoding differs (pdf.fonttype 42). Without its frame the upper legend's "
         "ParticleNet key lies on the curves near 3e-3, where Sophon's alpha-0.8 frame covered them. It breaks our own "
         "rules on purpose (dotted series, long caption), because this page checks the style against Sophon.",
         draw_sophon_fig2,
         "The points are those of the native PDF after its path simplification; a curve starts where the PDF's does."),
    Page("F1", "Design of the study", ["2405.12972/fig1.png"], "Sophon Fig. 1", F1_CAPTION,
         "Our content in Sophon's shape: (a) above (b), jets into a network box with its feature column, grey fan, "
         "shadowed card and callout. Panels named Pretraining and Use, not Pre-training and Usage. Four output columns "
         "instead of Sophon's one, with a merged class spanning the rows it merges, to show the nesting; rows are "
         "schematic, not to scale. A heading and merge arrows added. The 512-unit hidden layer is drawn between the features and the "
         "outputs, as the real network has it (PI), and in the fine-tuning head; the linear probe has none. The fan from "
         "the hidden layer reaches the 188-class column only. Pill labels (resonant 2 prongs, resonant 3-4 prongs, QCD) removed; "
         "QCD is marked in the columns. One card with two uses and one callout "
         "instead of two cards, because both uses start from the features (Sophon's discriminants start from the output "
         "layer). Drawn with matplotlib patches in STIX instead of a vector drawing with jet icons, and "
         "no latent or output size formulae, which belong to the methods.",
         draw_f1),
    Page("F9a", "Mass regression: response", ["ATL-PHYS-PUB-2024-015/fig7.png"], "ATLAS 2024-015 Fig. 7", F9A_CAPTION,
         "Two jet groups instead of mass and pT responses, as we regress mass only, with panel titles ATLAS lacks. Three "
         "series instead of two: a thin solid black reference and the 162- and 17-class colours, solid, instead of purple dashed and green. "
         "Fraction of jets on the y axis instead of a.u., Response (m/m_gen) on x instead of m_Reco/m_Truth. x range from "
         "the data, per panel, instead of a fixed 0.7 to 1.3, because the soft-drop mass here is far wider. No vertical "
         "dotted line at 1, a second reference the PI limit forbids. No bands: ATLAS's are thinner than its lines. Same "
         "ratio height (0.44), but ratio ranges of about 0.1 to 2.6 and 0.2 to 3.1 against ATLAS's 0.35 to 1.55. "
         + ATLAS_LOOK + ".",
         draw_f9a, MASS_RUNS_NOTE),
    Page("F9b", "Mass regression: across jet pT", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8", F9B_CAPTION,
         "All jets in ten pT bins to 2500 GeV in reconstructed pT (PI: at least eight, ATLAS's density), instead of "
         "ATLAS's bins to 1500 GeV in truth pT. Connected steps instead of one disconnected segment per bin, and no bands (PI limit), though "
         "ATLAS draws them. Three series instead of two, colours as F9a, each model's marker (v, P) at the bin centres. No "
         "dotted line at 1 in the median panel. Faint dashed grid at major y ticks (round 3), where ATLAS has none. Median response (r) and Resolution "
         "(sigma_r) instead of ATLAS's longer axis words. The lower panels keep ATLAS's ratio "
         "and its RSD, sgn(s'-s) sqrt|s'^2-s^2|, at its height (0.44). " + ATLAS_LOOK + ".",
         draw_f9b, MASS_RUNS_NOTE),
    Page("A17-res", "Mass regression on resonant jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         A17_CAPTION.format(group="resonant"), A17_DIFFERS, lambda path: _a17(path, "resonant"), MASS_RUNS_NOTE),
    Page("A17-qcd", "Mass regression on QCD jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         A17_CAPTION.format(group="QCD"), A17_DIFFERS, lambda path: _a17(path, "qcd"), MASS_RUNS_NOTE),
    Page("A17-probe", "Mass from frozen features and from the mass output (appendix)", ["ATL-PHYS-PUB-2024-015/fig8.png"],
         "ATLAS 2024-015 Fig. 8",
         "The jet mass from networks pretrained on JetClass-II: an MLP probe (MLP) regressing it from the frozen "
         "features with 188, 162, 43 and 17 classes (solid, three runs), and the mass output with 162 and 17 "
         "classes and an auxiliary mass regression (dashed, two runs). Median response r (a, b) and resolution "
         "sigma_r (c, d) of test jets against pT (a, c) and generator mass (b, d; 20-500 GeV); sub-panels and "
         "arrows as in the main-text mass figure.",
         "As F9b, with six series and the soft-drop mass (PI): the four probes in the label sets' colours, the two mass "
         "outputs in the 162- and 17-class colours as F10b, one marker shape per series at the bin centres and in the legend, "
         "since each mass output shares its label set's colour. Round 4: the 2x2 of F9b, legend above the panels. "
         + ATLAS_LOOK + ".",
         draw_a17_probe, MASS_RUNS_NOTE),
    Page("F10a", "Mass sculpting: QCD mass", ["2604.09809/fig8.png"], "CMS JME-25-001 Fig. 8", F10A_CAPTION,
         "Six panels, one per model (PI), all jet pT, instead of CMS's six (two models, three pT ranges), with "
         "one y axis across them. Two working points instead of three, because the files hold 10% and "
         "1%. Mass from 20 to 260 GeV in 20 GeV bins, the bins that cover CMS's 30 to 250 GeV; the files run to 500 GeV, "
         "and the twelve bins cut hold 5.9% of the inclusive jets and 6.4 to 7.6% of the selected ones (fractions stay "
         "those of all jets to 500 GeV). Colours are black, blue and pink instead of black, blue, red, orange: no colour "
         "of our palette, where red and orange are random partitions with controlled flavor boundaries. "
         "Markers as CMS's first three, at 2.5 pt against 1.59 to 2.12 pt so that the bars show, inclusive on top. No "
         "vertical error bars: the PI's limit, where CMS's are statistical. Bars at 1.0 pt instead of 0.4. Dashed reference "
         "at 1 instead of solid, and \"Ratio\" instead of a stacked fraction. Same ratio height (0.33). Jet mass (m_SD) "
         "instead of m_SD, fraction of jets instead of A.U., titles instead of a label block, no legend frame, as CMS; "
         "STIX at 10 and 8 pt instead of Helvetica at 6.9 and 5.8 pt; inward ticks, as CMS.",
         draw_f10a,
         "Mean of three runs. Round 4: the four label sets in a 2x2, working points in black and greys; the mass "
         "models move to F10a-mass."),
    Page("F10a-mass", "Mass sculpting: the mass-regression models (appendix)", ["2604.09809/fig8.png"],
         "CMS JME-25-001 Fig. 8", F10A_MASS_CAPTION, "F10a's form for the two mass-regression models (PI, round 4).",
         draw_f10a_mass, "Mean of two runs."),
    Page("F10a-flav", "Mass sculpting: the flavor-boundary partitions (appendix)", ["2604.09809/fig8.png"],
         "CMS JME-25-001 Fig. 8", F10A_FLAV_CAPTION,
         "F10a's form for the three 17-class flavor-boundary partitions (PI, 2026-11-07, M4).", draw_f10a_flav,
         "Mean of two runs."),
    Page("F10b", "Mass sculpting: how much", ["2604.09809/fig9.png"], "CMS JME-25-001 Fig. 9 left", F10B_CAPTION,
         "One panel: CMS's right panel (against pT) is not drawn, since our curves hold all jet pT. Twelve efficiencies "
         "on a log axis rising to the right, the files' grid and the direction of the efficiency axis in Sophon Fig. 2 "
         "and F11, instead of five categories falling to the right. Starts at the 5.2e-3 grid point (0.5%), where CMS's "
         "panel stops; the four tighter points are cut, because the distance there is mostly finite statistics (at 0.1%, "
         "a random selection of as many jets gives 0.108 against the curves' 0.119 to 0.125). Distance over 20 to 500 GeV "
         "and all pT instead of CMS's 30 to 250 GeV and 600 to 1000 GeV. Six series (PI) instead of two, one marker shape each. "
         "Label sets in their colours; the mass regression models in their label sets' colours with their own markers (v, P). "
         "Markers joined by a 1.1 pt line, not step markers: each point is a cumulative selection, so a segment would span "
         "a bin that does not exist (for the PI to decide). Faint dashed grid at major y ticks (round 3), where "
         "CMS has none; no major y tick falls in this log range, so no grid line shows. Jensen-Shannon distance (d_JS) and QCD efficiency instead of JSD and eps_B [%]; y limits from the "
         "data instead of 0.007 to 1. The random-selection floor drawn as a black dashed line, where CMS has none (PI, 2026-11-07). No legend frame and no label "
         "block; STIX at 10 and 8 pt with inward ticks. Bands of the spread over runs (PI, 2026-11-08), where CMS draws "
         "none; the flavor-boundary partitions moved to F10b-flav.",
         draw_f10b,
         "Mean of two runs for the two mass regression models, three for the others. Appendix; main text only if the "
         "sculpting bootstrap (v2_head/sculpting_boot.json, on the map) gives at 1% 17 - 188 > 0 and 17 + mass - 17 < 0, "
         "each |diff| > 3 sigma (gallery_placement.F10B_RULE)."),
    Page("F10c", "Mass sculpting against rejection", ["2001.05310/fig3.png"], "DisCo Fig. 3", F10C_CAPTION,
         "DisCo's plane (1/JSD at 50% signal efficiency against R50), both axes logarithmic; one point per model in its "
         "color and marker with bars of the spread over pretraining runs on both axes, where DisCo draws envelopes over "
         "six trainings per method. Legend above the panel; no ideal-tagger arrow.",
         draw_f10c,
         "New page. Reads sig_eff_50 {rejection, jsd} from each run's sculpting file (head/sculpting_v3/<model>_sculpting.json, "
         "the 22 models of the sculpting rerun, the flavor-boundary partitions among them)."),
    Page("F10b-flav", "Mass sculpting: the flavor-boundary partitions, how much (appendix)", ["2604.09809/fig9.png"],
         "CMS JME-25-001 Fig. 9 left", F10B_FLAV_CAPTION,
         "F10b's form for the three flavor-boundary partitions, with 188 classes for reference (PI, 2026-11-08: out of "
         "the main-text F10b). A separate page rather than a panel on F10a-flav, whose 2x2 of mass distributions has no "
         "room for a distance panel.",
         draw_f10b_flav, "Two runs per partition, three for 188 classes."),
    Page("F10-ft", "Mass sculpting after fine-tuning (appendix)", ["2604.09809/fig9.png"],
         "CMS JME-25-001 Fig. 9 left", F10_FT_CAPTION,
         "F10b's form after fine-tuning (PI, 2026-11-08), on the same natural-mix test jets as F10b, unweighted, so "
         "the floor is F10b's: a random selection of as many QCD jets.",
         draw_f10_ft,
         "Reads head/sculpting_ft/<model>.json (experiments/EVAL/sculpting_ft.py): the curve's js_distance and n_pass; "
         "absent until pulled. From scratch: the three full-schedule networks (scratch-s1..3), dark grey dashed."),
    Page("F11", "Resonance tagging", ["2405.12972/fig2.png"], "Sophon Fig. 2", F11_CAPTION,
         "Same layout, limits on x, fonts and legend place; round 3's faint grid and thinner lines. y axis eps_S/sqrt(eps_B), in words figure of merit, instead of the "
         "significance Z, and x axis QCD efficiency instead of SM background efficiency, because Z needs the SM backgrounds, which "
         "we do not have. Upper panel on stand-in jets from JetClass-II, not Sophon's samples. Lower panel empty, "
         "not invented curves. Roles: the 188-class direct discriminant is Sophon's direct curve, in the 188 colour dash-dot, "
         "a second read-out of the 188-class model as F8's linear probe (Sophon: grey, dotted); the four classifiers on frozen features take Sophon's transfer role, moved "
         "to X→bb on the stand-in because X→bs has no samples; Sophon* (42 classes, separately trained) has no counterpart; "
         "from-scratch curves take the ParT and ParticleNet role and are not drawn yet. Five series against Sophon's four. "
         "Solid lines but the dash-dot discriminant, where Sophon's are dotted for two series. y limits from the data, with the top at "
         "1.5 times the highest curve to clear the legend, not 0 to 5.4. Curves have 26 points or fewer (the fits' "
         "efficiency grid) and end at 20 background jets, so they are coarser and shorter than Sophon's.",
         draw_f11,
         "Stand-in only: X→bb from JetClass-II jets, /data/results/eval/xbx/fits (cluster), so the page is absent off "
         "the cluster. X→bs: no samples exist, the panel is empty. The significance Z is not computed: it needs the SM "
         "backgrounds. From scratch is not drawn: only a two-epoch dry run exists."),
    Page("PT", "Performance against jet pT", ["2004.08262/fig17.png", "1808.07858/fig2.png"],
         "CMS DeepAK8 Fig. 17, ATLAS top/W Fig. 2", PT_CAPTION,
         "Absolute values per pT bin (round 4): accuracy in (a)-(c) and the X→bb̄ against X→cc̄ AUC in (d), not the "
         "precedents' mistag rate or rejection at a working point, which would be a bound in every bin here (fewer "
         "than 20 X→cc̄ jets pass at 50% X→bb̄ efficiency in all bins of the 188-class runs). Markers at the bin centres "
         "joined by lines with SD bands (range for two runs), as F9b, A17 and F8, instead of unjoined bin bars without "
         "uncertainty. Reconstructed jet pT in ten (JetClass-II) or five (JetClass) bins. Panel titles and a figure "
         "legend instead of the precedents' text block; no experiment label; from scratch not drawn.",
         draw_pt,
         "Four label sets and from scratch have three runs, the controlled partitions two."),
    Page("CONF", "JetClass confusion matrices after fine-tuning (appendix)", ["2604.09809/fig7.png"],
         "CMS JME-25-001 Fig. 7", CONF_CAPTION,
         "Ten classes instead of 34. Four panels instead of one: the 188-class matrix and three differences to it, on "
         "a diverging RdBu_r scale symmetric about 0 with one colour bar, which CMS does not have. The 188-class "
         "matrix keeps CMS's Blues, as fractions to two decimals instead of integer percentages; differences in "
         "percentage points to one decimal, signed by a minus only, printed where at least 0.5 in size. Cell numbers at 6.5 pt, below the 8 pt of "
         "the other text, as CMS's are (about 5 pt): two decimals do not fit at 8 pt in a 15 cm wide figure. Class "
         "names only on the outer axes (the four panels share them); x names at 90 degrees as CMS. Panel titles and a "
         "heading in place of CMS's experiment label; Sophon's font; no tick marks.",
         draw_conf,
         "Means over three runs, each label set's run minus the 188-class run of the same seed."),
]
