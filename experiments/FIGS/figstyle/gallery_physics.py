#!/usr/bin/env python3
"""Gallery mock-ups of the design schematic, the mass regression, the mass sculpting, the resonance
tagging, performance against jet pT and the JetClass confusion figures, in style_v4 only, with Sophon's Fig. 2 redrawn first as the check of that style. Layouts:
docs/figstyle/PLAN.md. Data come through paper_figures' selection code (gallery_common); nothing is drawn at
import."""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

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
TRAINED, FROZEN = "#cfe6f7", "#e6e6e6"
FAN = "#cccccc"
NEW_LAYER = "#9aa7e0"


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


def _response(ax, rx, atlas: dict, group: str, title: str) -> None:
    g = P._groomed(atlas, group)["hist"]
    e = np.asarray(g["edges"], float)
    gr = P._frac(g)
    enough = np.asarray(g["counts"], float) >= M.MASS_MIN_JETS
    ax.stairs(gr, e, baseline=None, label=GROOMED, **S.REFERENCE)
    rx.axhline(1.0, **S.REFERENCE)
    shown, ratios = [gr], []
    for row, _ in MASS_ROWS:
        hs = [P._frac(b["hist"]) for m in RUNS[row] if (b := P._block(atlas, m, "regressed", group))]
        if not hs:
            continue
        mean = np.mean(hs, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(enough, mean / gr, np.nan)
        ax.stairs(mean, e, baseline=None, color=S.SERIES[row], linewidth=S.LW_CURVE, label=NAMES[row])
        rx.stairs(r, e, baseline=None, color=S.SERIES[row], linewidth=S.LW_CURVE)
        shown.append(mean)
        ratios.append(r)
    on = np.flatnonzero(np.any([s >= SHOWN_FROM * s.max() for s in shown], axis=0))
    ax.set_xlim(e[on[0]], e[on[-1] + 1])
    ax.set_ylim(0, HIST_ROOM * max(s.max() for s in shown))
    _fit_y(rx, [[1.0]] + [r[on] for r in ratios])
    S.title(ax, title)
    S.labels(ax, ylabel="Fraction of jets")
    S.labels(rx, xlabel=r"Response ($m/m_{\mathrm{gen}}$)", ylabel="Ratio")
    S.legend(ax)
    S.fine_grid(ax)
    S.fine_grid(rx)


def draw_f9a(path: pathlib.Path) -> None:
    atlas = _atlas()
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL, S.ONE_PANEL[1]), sharex="col",
                             gridspec_kw=dict(height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP))
    _response(axes[0, 0], axes[1, 0], atlas, "resonant", "Resonant jets")
    _response(axes[0, 1], axes[1, 1], atlas, "qcd", "QCD jets")
    S.letter_panels(axes[0])
    S.heading(fig, "Mass regression: response")
    S.save(fig, path)


def _mass_series() -> list:
    """(runs, estimator, colour, label, marker) of the two mass-regression models' outputs."""
    return [(RUNS[row], "regressed", S.SERIES[row], NAMES[row], S.MARKERS[row]) for row, _ in MASS_ROWS]


def _binned(axes, atlas: dict, group: str, axis: str, xlabel: str, legends: tuple, room: float = 0.0,
            series: list | None = None) -> None:
    """ATLAS Fig. 8's four axes: [[median, resolution], [its ratio, its root square difference]] to the groomed
    mass, in the bins of axis. legends: the columns with a legend, room above the curves for it. series:
    (runs, estimator, colour, label, marker), the mass models' outputs by default."""
    (a0, a1), (r0, r1) = axes
    g = P._groomed(atlas, group)[axis]
    e = np.asarray(g["edges"], float)
    med0, res0 = P._arr(g["median"]), P._arr(g["resolution"])
    a0.stairs(med0, e, baseline=None, label=GROOMED, **S.REFERENCE)
    a1.stairs(res0, e, baseline=None, label=GROOMED, **S.REFERENCE)
    r0.axhline(1.0, **S.REFERENCE)
    r1.axhline(0.0, **S.REFERENCE)
    meds, ress, ratios, rsds = [med0], [res0], [[1.0]], [[0.0]]
    for runs, est, colour, label, marker in (series or _mass_series()):
        bl = [b[axis] for m in runs if (b := P._block(atlas, m, est, group))]
        if not bl:
            continue
        med = np.mean([P._arr(b["median"]) for b in bl], axis=0)
        res = np.mean([P._arr(b["resolution"]) for b in bl], axis=0)
        ratio = med / med0
        rsd = np.sign(res - res0) * np.sqrt(np.abs(res ** 2 - res0 ** 2))   # ATLAS 2024-015's root square difference
        kw = dict(baseline=None, color=colour, linewidth=S.LW_CURVE)
        a0.stairs(med, e, label=None if marker else label, **kw)
        a1.stairs(res, e, label=None if marker else label, **kw)
        if marker:      # one marker per bin centre: greyscale print (PI), the model's marker (style_v4.MARKERS)
            c = 0.5 * (e[1:] + e[:-1])
            for ax, v in ((a0, med), (a1, res), (r0, ratio), (r1, rsd)):
                ax.plot(c, v, color=colour, marker=marker, markersize=3, linestyle="none")
            for ax in (a0, a1):     # the legend key: line and marker, as two series may share a colour
                ax.plot([], [], color=colour, linewidth=S.LW_CURVE, marker=marker, markersize=3, label=label)
        r0.stairs(ratio, e, **kw)
        r1.stairs(rsd, e, **kw)
        meds.append(med)
        ress.append(res)
        ratios.append(ratio)
        rsds.append(rsd)
    for col, ax, values in ((0, a0, meds), (1, a1, ress)):
        _fit_y(ax, values, room=room if col in legends else 0.0)
        if col in legends:
            S.legend(ax)
    _fit_y(r0, ratios)
    _fit_y(r1, rsds)
    for ax in (a0, a1, r0, r1):
        S.fine_grid(ax)
    a0.set_xlim(e[0], e[-1])
    a1.set_xlim(e[0], e[-1])
    S.labels(a0, ylabel=r"Median response ($r$)")
    S.labels(a1, ylabel=r"Resolution ($\sigma_r$)")
    S.labels(r0, xlabel=xlabel, ylabel="Ratio")
    S.labels(r1, xlabel=xlabel, ylabel="RSD")


X_PT, X_MGEN = r"Jet $p_{\mathrm{T}}$ [GeV]", r"Generator mass ($m_{\mathrm{gen}}$) [GeV]"


def draw_f9b(path: pathlib.Path) -> None:
    atlas = _atlas()
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL, S.ONE_PANEL[1]), sharex="col",
                             gridspec_kw=dict(height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP))
    _binned(axes, atlas, "all", "vs_pt", X_PT, legends=(0, 1), room=0.55)
    S.letter_panels(axes[0])
    S.heading(fig, "Mass regression against jet $p_{\\mathrm{T}}$")
    S.save(fig, path)


def _a17(path: pathlib.Path, group: str) -> None:
    """Median and resolution (columns) against jet pT and generator mass (rows); the legend once."""
    atlas = _atlas()
    fig = plt.figure(figsize=S.FOUR_PANELS)
    outer = fig.add_gridspec(2, 1, hspace=0.25, top=0.94, bottom=0.07)
    main = []
    for k, (axis, xlabel) in enumerate((("vs_pt", X_PT), ("vs_mgen", X_MGEN))):
        inner = outer[k].subgridspec(2, 2, height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP)
        axes = inner.subplots(sharex="col")
        _binned(axes, atlas, group, axis, xlabel, legends=(0,) if k == 0 else (), room=0.8)
        main += list(axes[0])
    S.letter_panels(main)
    S.heading(fig, f"Mass regression on {'resonant' if group == 'resonant' else 'QCD'} jets")
    S.save(fig, path)


def draw_a17_probe(path: pathlib.Path) -> None:
    """F9b's form, six series: the MLP on the frozen features of the four label sets and the two mass outputs."""
    atlas = _atlas()
    series = [(RUNS[a], "mlp", S.SERIES[a], f"{S.NAME[a]}, MLP on frozen features", S.MARKERS[a]) for a in S.LEVELS]
    series += _mass_series()
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL, S.ONE_PANEL[1] + 2 * S.CM), sharex="col",
                             gridspec_kw=dict(height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP))
    _binned(axes, atlas, "all", "vs_pt", X_PT, legends=(1,), room=0.9, series=series)
    S.letter_panels(axes[0])
    S.heading(fig, "Mass from frozen features and from the mass output, against jet $p_{\\mathrm{T}}$")
    S.save(fig, path)


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


def draw_f10a(path: pathlib.Path) -> None:
    sc = _sculpt()
    shown = [(a, ms) for a in SCULPT_ROWS if (ms := [m for m in RUNS[a] if m in sc])]
    if not shown:
        raise Absent("no mass sculpting curve of the models shown")
    fig = plt.figure(figsize=S.FOUR_PANELS)
    outer = fig.add_gridspec(2, 1, hspace=0.2, top=0.92, bottom=0.07)
    grids = [outer[k].subgridspec(2, 3, height_ratios=(1, RATIO_CMS), hspace=STACK_GAP, wspace=0.08).subplots(sharex=True, sharey="row")
             for k in range(2)]
    panels = [p for g in grids for p in zip(g[0], g[1])]
    # the markers of CMS Fig. 8: open circle before the selection, filled circle, open square
    looks = (("o", True), ("o", False), ("s", True))
    peak, ratios = 0.0, [[1.0]]
    for k_panel, ((ax, rx), (a, ms)) in enumerate(zip(panels, shown)):
        e = np.asarray(sc[ms[0]]["bins_jet_sdmass"], float)
        keep = e[1:] <= F10A_TOP
        mid, half = ((e[:-1] + e[1:]) / 2)[keep], (np.diff(e) / 2)[keep]
        before = np.asarray(sc[ms[0]]["hist_before"], float)
        before = before / before.sum()
        _points(ax, mid, before[keep], half, looks[0][0], "#000000", looks[0][1], S.LW_THIN, "Inclusive", zorder=2.1)
        peak = max(peak, before[keep].max())
        rx.axhline(1.0, **S.REFERENCE)
        for k, (wp, _) in enumerate(P.WPS, start=1):
            after = np.mean([np.asarray(sc[m]["working_points"][wp]["hist_after"], float)
                             / np.sum(sc[m]["working_points"][wp]["hist_after"]) for m in ms], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            label = f"$\\varepsilon_B = {100 * float(wp):g}\\%$"
            _points(ax, mid, after[keep], half, looks[k][0], SELECTION_COLOUR[k], looks[k][1], S.LW_THIN, label)
            _points(rx, mid, r[keep], half, looks[k][0], SELECTION_COLOUR[k], looks[k][1], S.LW_THIN)
            peak = max(peak, after[keep].max())
            ratios.append(r[keep])
        ax.set_xlim(e[0], F10A_TOP)
        S.title(ax, NAMES[a].replace(", mass regression", " + mass"))     # the full words collide at this width
        if k_panel >= 3:
            S.labels(rx, xlabel=r"Jet mass ($m_{\mathrm{SD}}$) [GeV]")
        S.fine_grid(ax)
        S.fine_grid(rx)
    S.legend(panels[0][0])
    for g in grids:
        g[0, 0].set_ylim(0, 1.1 * peak)
        _fit_y(g[1, 0], ratios)
        S.labels(g[0, 0], ylabel="Fraction of jets")
        S.labels(g[1, 0], ylabel="Ratio")
    S.letter_panels([ax for ax, _ in panels])
    S.heading(fig, "QCD jet mass before and after a selection on the tagger")
    S.save(fig, path)


def draw_f10b(path: pathlib.Path) -> None:
    sc = _sculpt()
    fig, ax = plt.subplots(figsize=S.ONE_PANEL)
    S.fine_grid(ax)
    drawn = 0
    for row in SCULPT_ROWS:
        have = [sc[m]["curve"] for m in RUNS[row] if m in sc]
        if not have:
            continue
        eff = np.array([c["eff"] for c in have[0]], float)
        mean = np.mean([[c["js_distance"] for c in h] for h in have], axis=0)
        keep = eff >= F10B_FROM
        ax.plot(eff[keep], mean[keep], color=S.SERIES[row], linewidth=S.LW_CURVE, marker=S.MARKERS[row], markersize=3.5, label=NAMES[row])
        drawn += 1
    if not drawn:
        raise Absent("no mass sculpting curve of the models shown")
    ax.set_xscale("log")
    ax.set_yscale("log")
    S.title(ax, "Mass sculpting of QCD jets")
    S.labels(ax, r"QCD efficiency ($\varepsilon_B$)", r"Jensen-Shannon distance ($d_{\mathrm{JS}}$)")
    S.legend(ax, loc="best")
    S.save(fig, path)


# ------------------------------------------------------------------ F11
def _standin_series(XR, fits: dict) -> list:
    """(label, look, readouts per run) on the stand-in jets: the 188-class output scores (Sophon's direct
    discriminant, Eq. 4), then each label set's classifier on the frozen features (the fits' "head"; the linear
    probe where a fit has none)."""
    of = lambda row, key: [r for m in RUNS[row] if (r := XR.readout(fits.get(m, {}).get(STANDIN, {}), key))]
    out = []
    if direct := of("L188", "zero_shot_xbb"):
        out.append((f"{S.NAME['L188']}, direct discriminant", DIRECT_LOOK, direct))
    for row in S.LEVELS:
        for key, word in (("head", "transfer"), ("linear", "linear probe")):
            if rs := of(row, key):
                out.append((f"{S.NAME[row]}, {word}", S.look(row), rs))
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


def _one_minus_auc(doc: dict) -> np.ndarray:
    return np.array([np.nan if b["macro_auc_ovr"] is None else 1.0 - b["macro_auc_ovr"] for b in doc["bins"]])


def _pt_series(ax, c, values: list, key: str, label: str, band: bool = True) -> None:
    """Mean over runs at the bin centres, joined, the series' marker; band the SD (range for two runs), as F8."""
    st = [P.F._spread(list(v)) for v in np.asarray(values).T]
    mean = np.array([s[0] for s in st])
    style = S.look(key, linewidth=S.LW_CURVE)
    ax.plot(c, mean, **style, marker=S.MARKERS[key], markersize=PT_MARKER, label=label)
    if band and len(values) > 1:
        ax.fill_between(c, mean - np.array([s[1] for s in st]), mean + np.array([s[2] for s in st]),
                        color=style["color"], alpha=0.2, linewidth=0)


def _ratio_panel(ax, docs: dict, rows: tuple) -> None:
    """Macro (1 - AUC) of each run over that of the 188-class run of the same seed. From scratch is left out: its
    ratio (about 1.3 on JetClass) would squeeze the label sets' spread, which is what the figure asks about."""
    e = _edges(docs)
    c = 0.5 * (e[1:] + e[:-1])
    ref = {s: _one_minus_auc(d) for s, d in _by_seed(docs, "L188").items()}
    ax.axhline(1.0, **S.look("L188", linewidth=S.LW_CURVE), label=S.NAME["L188"])
    for row in rows:
        r = [_one_minus_auc(d) / ref[s] for s, d in _by_seed(docs, row).items() if s in ref]
        if r:
            _pt_series(ax, c, r, row, S.NAME[row])
    ax.set_xlim(e[0], e[-1])
    S.labels(ax, ylabel=r"Macro $(1-\mathrm{AUC})$ / 188 classes")


def _bb_cc_panel(ax, docs: dict) -> None:
    """X->bb against X->cc: (1 - AUC) of each run over that of the 188-class run of the same seed, log axis. The
    precedents' rejection at a working point cannot be drawn here: at 50% X->bb efficiency fewer than 20 X->cc jets
    pass in every bin of every 188-class run (250-930 jets per class and bin), so each value would be a bound."""
    e = _edges(docs)
    c = 0.5 * (e[1:] + e[:-1])
    pair = lambda d: np.array([np.nan if b["bb_vs_cc"] is None else 1.0 - b["bb_vs_cc"]["auc"] for b in d["bins"]])
    ref = {s: pair(d) for s, d in _by_seed(docs, "L188").items()}
    ax.axhline(1.0, **S.look("L188", linewidth=S.LW_CURVE), label=S.NAME["L188"])
    for row in S.LEVELS[1:]:
        r = [pair(d) / ref[s] for s, d in _by_seed(docs, row).items() if s in ref]
        if r:
            _pt_series(ax, c, r, row, S.NAME[row])
    ax.set_yscale("log")
    _log_y(ax)
    ax.set_xlim(e[0], e[-1])
    S.labels(ax, ylabel=r"$(1-\mathrm{AUC})$ / 188 classes")


def draw_pt(path: pathlib.Path) -> None:
    jc2, jc1 = M._load(PERF / "jc2"), M._load(PERF / "jc1")
    if not (jc2 or jc1):
        raise Absent("no performance against jet pT (perf_vs_pt/jc2, perf_vs_pt/jc1)")
    fig = plt.figure(figsize=S.FOUR_PANELS)
    axes = fig.subplots(2, 2, gridspec_kw={"left": 0.1, "right": 0.96, "top": 0.92, "bottom": 0.07, "wspace": 0.3,
                                           "hspace": 0.25})
    (a, b), (cx, d) = axes
    # the legend corner: (d)'s 1.0 line fills the bottom, so its legend goes above the curves
    panels = ((a, jc2, lambda ax: _ratio_panel(ax, jc2, S.LEVELS[1:]), "JetClass-II, linear probe", "best"),
              (b, jc2, lambda ax: _bb_cc_panel(ax, jc2), r"JetClass-II: $X\to b\bar{b}$ vs $X\to c\bar{c}$", "best"),
              (cx, jc1, lambda ax: _ratio_panel(ax, jc1, S.LEVELS[1:]), "JetClass, fine-tuned on $10^6$ jets", "best"),
              (d, jc2, lambda ax: _ratio_panel(ax, jc2, S.CONTROLS), "JetClass-II: controlled partitions", "upper left"))
    for ax, docs, draw, text, corner in panels:
        S.title(ax, text)
        S.labels(ax, xlabel=X_PT)
        if not docs:
            ax.set_yticks([])
            continue
        S.fine_grid(ax)
        draw(ax)
        if corner == "upper left":      # room above the curves for the four legend rows
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.4 * (hi - lo))
        S.legend(ax, loc=corner)
    S.letter_panels(axes.flat)
    S.heading(fig, "Performance against jet $p_{\\mathrm{T}}$")
    S.save(fig, path)


CONF_CELL_PT = 6.5      # numbers in the cells; 8 pt does not fit two decimals in a 10-column panel at 15 cm


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

    fig = plt.figure(figsize=(S.FULL, 13.5 * S.CM))
    gs = fig.add_gridspec(2, 5, width_ratios=(1, 0.05, 0.3, 1, 0.05), left=0.17, right=0.9, top=0.895, bottom=0.18,
                          wspace=0.05, hspace=0.2)
    panels = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 3])]
    bars = [(fig.add_subplot(gs[0, 1]), panels[0], panels[0], "Fraction of the true class")]
    im = panels[0].imshow(base, cmap="Blues", vmin=0, vmax=1, interpolation="none")
    _cells(panels[0], base, [[f"{v:.2f}" for v in r] for r in base], lambda v: v > 0.6)
    S.title(panels[0], S.NAME["L188"])
    images = [im]
    for ax, row in zip(panels[1:], S.LEVELS[1:]):
        S.title(ax, f"{S.NAME[row]} minus 188")
        if row in diffs:
            v = diffs[row]
            images.append(ax.imshow(v, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none"))
            _cells(ax, v, [[f"{x:.1f}".replace("-", "\u2212") if abs(x) >= 0.5 else "" for x in r] for r in v],
                   lambda x: abs(x) > 0.6 * limit)
    if diffs:
        bars.append((fig.add_subplot(gs[:, 4]), panels[1], panels[3], "Difference to 188 classes (percentage points)"))
    for k, ax in enumerate(panels):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names if k >= 2 else [], rotation=90)
        ax.set_yticklabels(names if k % 2 == 0 else [])
        ax.tick_params(which="both", length=0, top=False, right=False)
        ax.minorticks_off()
        S.labels(ax, "Predicted class" if k >= 2 else None, "True class" if k % 2 == 0 else None)
        ax.apply_aspect()
    for (cax, top, bottom, word), image in zip(bars, (images[0], images[-1])):
        x = cax.get_position()      # as tall as the square matrices beside it, not the grid cell
        cax.set_position([x.x0, bottom.get_position().y0, x.width, top.get_position().y1 - bottom.get_position().y0])
        cb = fig.colorbar(image, cax=cax)
        S.labels(cb.ax, ylabel=word)
        cb.ax.minorticks_off()
    S.letter_panels(panels)
    S.heading(fig, "JetClass after fine-tuning on $10^6$ jets: confusion matrices")
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
    "Design of the study. (a) One network is pretrained on the same jets four times, once for each of four nested "
    "label sets with 188, 162, 43 and 17 classes, through a hidden layer of 512 units; below each merge is the "
    "distinction it removes. (b) The pretrained network is then used in two ways: its frozen features feed a linear "
    "probe, or a new head with its own 512-unit hidden layer is added and all weights are fine-tuned.")
F9A_CAPTION = (
    "Distribution of the jet mass response, the ratio of a jet's mass to its generator-level mass, for resonant jets "
    "(left) and QCD jets (right): the soft-drop mass and the mass regression output of the models pretrained on 162 and "
    "17 classes, each the mean over runs; the axes hold 76 percent of the QCD soft-drop-mass jets. The lower panels show "
    "the ratio of each distribution to the soft-drop mass.")
F9B_CAPTION = (
    "Median jet mass response (left) and relative mass resolution (right) against the jet transverse momentum, for all "
    "jets: the soft-drop mass and the mass regression output of the models pretrained on 162 and 17 classes, each the "
    "mean over runs. Below are the ratio of each median to the soft-drop mass and the root square difference (RSD) of "
    "each resolution to the soft-drop mass.")
A17_CAPTION = (
    "Median jet mass response (left) and relative mass resolution (right) for {group} jets, against the jet transverse "
    "momentum (upper) and the generator-level jet mass (lower): the soft-drop mass and the mass regression output of the "
    "models pretrained on 162 and 17 classes, each the mean over runs{last_bin}. Below each panel are the ratio of the median "
    "and the root square difference (RSD) of the resolution to the soft-drop mass.")
F10A_CAPTION = (
    "Soft-drop mass distribution of QCD jets before any selection (inclusive) and after selections at QCD efficiencies "
    "of 10 and 1 percent on the score of the models pretrained on 188, 162, 43 and 17 classes and on 162 and 17 "
    "classes with mass regression (+ mass), each the mean over runs. Horizontal bars show the bin width. The lower "
    "panels show the ratio to the inclusive distribution.")
F10B_CAPTION = (
    "Jensen-Shannon distance between the soft-drop mass distributions of QCD jets before and after a selection on the "
    "model's score, against the QCD efficiency of the selection, for the models pretrained on 188, 162, 43 and 17 "
    "classes and on 162 and 17 classes with mass regression. Each point is the mean over runs.")
F11_CAPTION = (
    "Signal efficiency over the square root of the background efficiency, against the QCD efficiency, for X→bb on "
    "stand-in JetClass-II jets (upper) and for X→bs (lower), which has no samples yet. Curves show the direct "
    "discriminant of the 188-class model and a classifier on the frozen features of each label set, each the mean "
    "over runs; a curve ends where fewer than 20 background jets remain.")
PT_CAPTION = (
    "Performance against the jet transverse momentum. (a) JetClass-II linear probes: macro-averaged 1 - AUC of the "
    "models pretrained on 162, 43 and 17 classes, each divided by that of the 188-class model of the same seed. (b) "
    "As (a), for X→bb̄ against X→cc̄ alone, on a log scale. (c) As (a), for JetClass fine-tuned on one million jets. "
    "(d) As (a), for the controlled partitions. Points are means over runs, bands the standard deviation. "
    "What to look for: the cost of coarser labels grows with pT, from 44% more errors than the 188-class model at "
    "200-300 GeV to 74% more at 2-2.5 TeV for 17 classes (a), far outside the spread over runs, and the same way for "
    "the controlled partitions (d), while X→bb̄ against X→cc̄ (b) costs 17 classes a factor 3 to 12 at every pT.")
CONF_CAPTION = (
    "Confusion matrices on JetClass after fine-tuning on one million jets: (a) the fraction of each true class assigned "
    "to each predicted class by the model pretrained on 188 classes; (b)-(d) the change in that fraction, in percentage "
    "points, for the models pretrained on 162, 43 and 17 classes, each against the 188-class model of the same seed. "
    "Values are means over runs; differences under 0.5 points are not printed. "
    "LOOK: to be written after the data are seen.")

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
    Page("SOPHON-FIG2", "Sophon Fig. 2 redrawn through style_v4 (validation)", ["2405.12972/fig2.png"], "Sophon Fig. 2",
         SOPHON_FIG2_CAPTION,
         "Round 3 (style_v4) on purpose, against Sophon's native settings (sophon_native.json): grid a faint dashed "
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
         "series instead of two: a dark grey dashed reference and the 162- and 17-class colours, solid, instead of purple dashed and green. "
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
         A17_CAPTION.format(group="resonant", last_bin=""), A17_DIFFERS, lambda path: _a17(path, "resonant"), MASS_RUNS_NOTE),
    Page("A17-qcd", "Mass regression on QCD jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         A17_CAPTION.format(group="QCD", last_bin="; the soft-drop mass's last pT bin sets the upper right scale"),
         A17_DIFFERS + " The soft-drop mass's last pT bin (resolution 3.6, regression 0.28) sets the scale of the upper right "
         "panel; the axis is not clipped.", lambda path: _a17(path, "qcd"), MASS_RUNS_NOTE),
    Page("A17-probe", "Mass from frozen features and from the mass output (appendix)", ["ATL-PHYS-PUB-2024-015/fig8.png"],
         "ATLAS 2024-015 Fig. 8",
         "Median jet mass response (left) and relative mass resolution (right) against the jet transverse momentum, for all "
         "jets: the soft-drop mass, a nonlinear probe (MLP) regressing the mass from the frozen features of the models "
         "pretrained on 188, 162, 43 and 17 classes, and the mass regression output of the 162- and 17-class models, each "
         "the mean over runs. Below are the ratio of each median and the root square difference (RSD) of each resolution "
         "to the soft-drop mass.",
         "As F9b, with six series and the soft-drop mass (PI): the four probes in the label sets' colours, the two mass "
         "outputs in the 162- and 17-class colours as F10b, one marker shape per series at the bin centres and in the legend, "
         "since each mass output shares its label set's colour. The legend is in the resolution "
         "panel, with room above the curves. " + ATLAS_LOOK + ".",
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
         "Mean of two runs for the two mass regression models, three for the others. Nothing else is absent."),
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
         "data instead of 0.007 to 1. No random-selection band or floor: CMS has none (plan). No legend frame and no label "
         "block; STIX at 10 and 8 pt with inward ticks.",
         draw_f10b,
         "Mean of two runs for the two mass regression models, three for the others. Nothing else is absent."),
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
         "Ratio panels have no counterpart: the precedents plot one tagger's mistag rate or rejection per series, "
         "where every panel here divides each run's (1 - AUC) by the 188-class run of the same seed, so the label-set "
         "gap is read directly; from scratch is not drawn. Panel (b) is that ratio for X→bb against X→cc on a log "
         "axis, not the precedents' rejection at a working point, which is a bound in every bin (fewer than 20 X→cc "
         "jets pass at 50% X→bb efficiency in all 30 bins of the 188-class runs). Markers at the bin centres joined "
         "by lines, with SD bands (range for two runs), as F9b, A17 and F8, instead of the precedents' unjoined "
         "horizontal bin bars with no uncertainty. Reconstructed jet pT in our ten (JetClass-II) or five (JetClass) bins, instead of generator "
         "pT 300-2000 GeV (CMS) or true pT 200-2500 GeV (ATLAS). Panel titles and legends in place of the "
         "precedents' text block (sqrt(s), jet definition, selection); no experiment label. Our palette, Sophon's font, "
         "round 3's faint grid, frameless legends, inward ticks.",
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
