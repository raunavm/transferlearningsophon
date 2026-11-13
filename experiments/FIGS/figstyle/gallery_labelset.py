#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2, F2-ft, F2a, F2b, the training-size family (A8, A10, F8, their AUC versions, A8-part, A10-part), UMAP-jc1, UMAP-jc2-had, UMAP-jc2-lep, UMAP-top, UMAP-qg, VAL (docs/figstyle/PLAN.md), in
style_v4 only. The numbers come through the selection code of paper_figures and mlst_results; the drawing
is written here. All work happens inside the draw functions."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

import gallery_common as C
import gallery_placement as G
import lines_v6 as L
import style_v5 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I

GRID_2X2 = {"left": 0.1, "right": 0.96, "top": 0.92, "bottom": 0.06, "wspace": 0.3, "hspace": 0.2}     # room for repeated y labels, tick labels and titles
HEIGHT_HEAT = 8.0                                 # 188 rows; a 5-row group label at 10 pt still clears its neighbours
LEFT_HEAT = 0.26                                  # the longest group label at 10 pt, so the saved width stays 15 cm
MARKER = 4
X_JETS = "Number of training jets"
LEGEND_PANEL = 1        # JetClass: its curves stay high on the right, so its lower right is empty in F2, F8 and A10
PUBLISHED = {"jc1": (2_000_000, 0.836), "top": (1_200_000, 1602.0), "qg": (1_600_000, 101.2)}     # ParT's published values, as in the paper's benchmark tables
# ParT's published AUCs, as in the same tables (JetClass: averaged over class pairs, equal to one against the rest on
# its balanced test set, which mlst_v4 checks)
PUBLISHED_AUC = {"jc1": (2_000_000, 0.9834), "top": (1_200_000, 0.9858), "qg": (1_600_000, 0.9121)}
Y_OF = {"accuracy": "Accuracy", "r30": r"Rejection ($R_{30}$)", "macro_auc_ovr": "Mean AUC over classes", "auc": "AUC"}


def _chosen() -> dict:
    return M.chosen_substitutions(I["fill"], RUNS)[0]


def _panels_2x2():
    fig = plt.figure(figsize=S.FOUR_PANELS)
    return fig, fig.subplots(2, 2, gridspec_kw=GRID_2X2)


def _log_y(ax, which: str = "y") -> None:
    """R30 on a log axis labelled 10^n, or linear where its range is under a decade (style_v5.log_axis), the same rule
    in every panel. which="x": the same on x (F10c). The limits are taken on a log scale, where a band reaching zero or
    below does not set them."""
    if which == "y":
        ax.set_yscale("log")
        S.data_ylim(ax)             # lines and bands, on a log scale
    S.log_axis(ax, which)
    if which == "y" and ax.get_yscale() == "linear":
        S.data_ylim(ax)             # under a decade: linear, and the bands inside as well


def _empty_panel(ax) -> None:
    """A panel with no data keeps its frame, title and x axis, so the layout is the final one."""
    ax.set_yticks([])


# ------------------------------------------------------------------ F2, F2-ft, F2a, F2b: the headline value
# PI 2026-11-12/13 (L1, L5): the label set is a category, so a horizontal dot plot, one row a network, on the metric
# itself, one panel a task (HV_PANELS, Herwig included). L20 (2026-11-13) splits the family one mode a figure: F2 (main
# text) the linear probes and F2-ft (main text) the fine-tuned networks, headline metric only; F2a and F2b (appendix)
# the same in the task's AUC. Network rows in the one order of every figure and table (L5): the label sets, each mass network under its own;
# the 17-class partitions under an italic subheading; the mode's references (HV_REFS). Each row: the mean over its runs
# a filled circle in the network's colour, a horizontal bar its SD (the five random partitions: mean and SD over the
# five). The 188-class mean is a vertical line with a light band of its SD down the panel. A reference far from every
# other row (the random network) sits beyond an axis break, on the scale of the rest. Nothing printed, nothing clipped.
HV_PANELS = (("jc2", 1_000_000, "jc2_188", "accuracy", "JetClass-II"),
             ("jc1", 1_000_000, "jc1", "accuracy", "JetClass"), ("top", 1_200_000, "top", "r30", "Top tagging"),
             ("qg", 1_600_000, "qg", "r30", "Quark/gluon, Pythia"),
             ("qg_herwig", 1_600_000, "qg_herwig", "r30", "Quark/gluon, Herwig"))
# L20 (PI 2026-11-13: points-only panels at least 4.5 cm): two panels a row, one mode a figure (F2 and F2-ft the
# headline metric, F2a and F2b the AUC), the training-size family's slots, the legend in the sixth
HV_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
HV_LEGEND_SLOT = (2, 1)
# beside each task's headline metric its AUC (averaged over the classes, each against the rest, on JetClass-II and
# JetClass; of the binary task on top and quark/gluon); F2_GRID is every cell of the family. Table 1 and its AUC
# table read the same value functions (mlst_results.summary_probe, summary_ft, summary_scratch) over SUMMARY_GRID
# (the headline metric) and SUMMARY_AUC_GRID (the AUC)
F2_AUC = {"jc2": "macro_auc_ovr", "jc1": "macro_auc_ovr", "top": "auc", "qg": "auc", "qg_herwig": "auc"}
F2_GRID = [(key, n, ds, m) for key, n, ds, met, _ in HV_PANELS for m in (met, F2_AUC[key])]
HV_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
HV_PARTITIONS = ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")
HV_HEAD = "17-class partitions"
# the references of a mode (L5): the random network beside linear probes, the released Sophon checkpoint where the text
# cites it (F2a, beside tab:probes), from scratch beside fine-tuned networks
HV_REFS = {"probe": (M.INIT,), "probe_all": (M.INIT, M.SOPHON), "ft": ("SCRATCH",)}
HV_KEY = {M.INIT: "INIT_BN", M.SOPHON: "SOPHON"}            # their colour and name in style_v5 (INIT: "random network")
X_F2 = {"accuracy": "Accuracy", "r30": "$R_{30}$", "macro_auc_ovr": "AUC", "auc": "AUC"}
MEAN_DOT = 4.6                   # pt: the mean over runs
HV_UNIT = 0.31                   # cm per network row: a name with $b$ (0.36 cm) clears its neighbours (0.25 cm)
HV_ASPECT = 0.72                 # a panel's least height over its width
HV_WGAP, HV_RIGHT = 0.45, 0.12   # cm between panels (beyond figcheck's 3 mm to a break piece); right of the last
HV_STUB, HV_STUB_GAP = 0.55, 0.1  # cm: the part of a panel beyond an axis break, and the break
HV_BREAK = 1.5                   # a reference further from the other rows than this many times their span: a break
HV_INDENT = "   "                # the partitions under their subheading, as in the tables
HV_TITLE_PT = S.TITLE_PT5
HV_TICK_GAP, HV_TICK_PANEL = 0.15, 0.12      # cm at least between two x tick labels; a label and another panel
HV_WIDEN = (0.0, 0.15, 0.3, 0.45, 0.6)        # the widenings of a main part's range tried for its second label


def _hv_rows(refs: tuple) -> tuple[dict, float, list]:
    """({row: y}, the subheading's y, the dividers' y): the label sets from 0, the partitions under their subheading,
    the references after a divider."""
    rows = {k: float(i) for i, k in enumerate(HV_ROWS)}
    head = len(HV_ROWS) + 0.5
    rows.update({k: head + 1 + i for i, k in enumerate(HV_PARTITIONS)})
    last = max(rows.values())
    rows.update({k: last + 1.5 + i for i, k in enumerate(refs)})
    return rows, head, [len(HV_ROWS) - 0.25, last + 0.75]


def _hv_name(row: str) -> str:
    name = S.LEGEND[HV_KEY.get(row, row)]
    return HV_INDENT + name if row in HV_PARTITIONS else name


def _hv_colour(row: str) -> str:
    return S.SERIES[HV_KEY.get(row, row)]


_FITS = {}


def _fits() -> dict:
    """The v4 probe fits with the released Sophon checkpoint's beside them (as mlst_results' probe tables read it)."""
    if not _FITS:
        sophon = M._load(P.DATA / M.FITS_SOPHON).get(M.SOPHON_MODEL)
        _FITS.update({**I["fits"], **({M.SOPHON_MODEL: M.V4.remap(sophon)} if sophon else {})})
    return _FITS


def _probe_value(key, n, ds, met):
    """A run's linear-probe value in a cell (Table 1's source, mlst_results.summary_probe)."""
    return M.summary_probe(_fits(), I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    """A run's fine-tuned value in a cell (Table 1's source, mlst_results.summary_ft)."""
    return M.summary_ft(I["per"], key, n, met)


def _scratch_value(key, n, ds, met) -> list:
    """Each from-scratch seed's value in a cell (Table 1's source, mlst_results.summary_scratch)."""
    return M.summary_scratch(I["per"], key, n, met)


def _hv_values(row: str, value, cell) -> list:
    """The values a row draws in a cell: its runs (the five random partitions pooled), from scratch its seeds; the
    released Sophon checkpoint only on JetClass-II and JetClass, the top and quark/gluon probes having its missing
    inputs at zero where ours are substituted (mlst_results._bench_rec)."""
    if row == "SCRATCH":
        return _scratch_value(*cell)
    if row == M.SOPHON and cell[2] in ("top", "qg", "qg_herwig"):
        return []
    models = [M.SOPHON_MODEL] if row == M.SOPHON else M._pooled(RUNS)[M.RAND_POOLED if row == "RAND" else row]
    return [x for m in models if (x := value(m)) is not None] if value else []


def _f2_row(ax, y: float, v: list, color: str) -> list:
    """One network's row: the mean a filled circle with its SD bar (two or more runs). Returns the values that set
    the axis range: the runs (not drawn) and the bar's ends."""
    mean = float(np.mean(v))
    out = list(v) + [mean]
    if len(v) > 1:
        sd = float(np.std(v, ddof=1))
        ax.errorbar([mean], [y], xerr=[[sd], [sd]], fmt="none", ecolor=color, elinewidth=L.BAR, capsize=0, zorder=2.5)
        out += [mean - sd, mean + sd]
    ax.plot([mean], [y], linestyle="none", marker="o", markersize=MEAN_DOT, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.5, zorder=2.6)
    return out


def _hv_limits(xs: list, w: float) -> tuple:
    """x limits over xs on an axis w cm wide, padded by 8% of the span and at least so far that a mean's circle at
    either end stays inside (L7: no marker on or past an axis edge)."""
    lo, hi = min(xs), max(xs)
    span = hi - lo or abs(hi) * 1e-3 or 1e-3
    f = (MEAN_DOT / 2 + 2.0) / 72 * 2.54 / w        # a circle's radius and two points, as a share of the axis
    p = max(0.08 * span, f * span / (1 - 2 * f))
    return lo - p, hi + p


def _stub_ticks(lo: float, hi: float, step: float) -> list:
    """One round tick inside (lo, hi), the roundest: a multiple of the main axis's step or of the next smaller 5, 2
    or 1 times a power of ten, nearest the middle."""
    e = np.floor(np.log10(step))
    for s in sorted({m * 10.0 ** k for k in np.arange(e - 6, e + 1) for m in (1, 2, 5)} | {step}, reverse=True):
        if s <= step and (t := [k * s for k in range(int(np.ceil(lo / s)), int(np.floor(hi / s)) + 1)]):
            return [min(t, key=lambda v: abs(v - (lo + hi) / 2))]
    return []


def _break_marks(ax, side: str) -> None:
    """The two short slashes of an axis break at one side of a panel part, on its top and bottom edges."""
    x = 0.0 if side == "left" else 1.0
    ax.plot([x, x], [0, 1], transform=ax.transAxes, marker=[(-1, -1.6), (1, 1.6)], markersize=3.5, linestyle="none",
            color="#000000", markeredgewidth=S.LW_THIN, clip_on=False, zorder=5)


def _hv_panel(fig, x, top, w, h, rows, dividers, series, ref188, met):
    """One task's panel: series [(row, values, colour, reference?)], the dividers at their y; a reference far beyond the
    other rows' span goes to a stub beyond an axis break. Returns (its axes from left to right, the main axes, bars,
    bars hidden under the circle)."""
    main_xs = [v for r, vals, _, ref in series if vals and not ref for v in vals + _sd_ends(vals)]
    lo, hi = min(main_xs), max(main_xs)
    far = {r for r, vals, _, ref in series if ref and vals and (min(vals) > hi + HV_BREAK * (hi - lo)
                                                                 or max(vals) < lo - HV_BREAK * (hi - lo))}
    sides = {"left" if max(vals) < lo else "right" for r, vals, _, _ in series if r in far}
    if len(sides) > 1:
        raise ValueError("references beyond an axis break on both sides of a panel")
    side = sides.pop() if sides else None
    mw = w - (HV_STUB + HV_STUB_GAP if side else 0)
    mx = x + (HV_STUB + HV_STUB_GAP if side == "left" else 0)
    main = S.axes_cm(fig, mx, top, mw, h)
    stub = (S.axes_cm(fig, x if side == "left" else mx + mw + HV_STUB_GAP, top, HV_STUB, h) if side else None)
    if stub is not None:
        stub.set_label("<break>")                   # figcheck: a part of its panel, not a panel
    ymax = max(rows.values())
    hidden = bars = 0
    xs = {main: [], stub: []}
    if ref188:
        m, sd = float(np.mean(ref188)), float(np.std(ref188, ddof=1)) if len(ref188) > 1 else 0.0
        main.axvspan(m - sd, m + sd, color=S.SERIES["L188"], alpha=L.BAND, linewidth=0, zorder=0.8)
        main.axvline(m, color=S.SERIES["L188"], linewidth=S.LW_THIN, zorder=0.9)
    for r, vals, colour, _ in series:
        if vals:
            ax = stub if r in far else main
            xs[ax] += _f2_row(ax, rows[r], vals, colour)
    main.set_xlim(*_hv_limits(xs[main], mw))
    main.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3 if mw > 1.6 else 2, steps=[1, 2, 5, 10]))
    if stub is not None:
        scale = np.diff(main.get_xlim())[0] / mw                    # data units per cm, as the main part
        c = (min(xs[stub]) + max(xs[stub])) / 2
        half = max(scale * HV_STUB / 2, (max(xs[stub]) - min(xs[stub])) / 2 * 1.3)
        stub.set_xlim(c - half, c + half)
        step = float(np.diff(main.get_xticks()[:2])[0])
        stub.xaxis.set_major_locator(ticker.FixedLocator(_stub_ticks(c - 0.7 * half, c + 0.7 * half, step)))
        stub.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        stub.spines["right" if side == "left" else "left"].set_visible(False)
        main.spines[side].set_visible(False)
        _break_marks(stub, "right" if side == "left" else "left")
        _break_marks(main, side)
    parts = [a for a in ((stub, main) if side == "left" else (main, stub)) if a is not None]
    for ax in parts:
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.yaxis.set_major_locator(ticker.FixedLocator(list(rows.values())))
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        S.fine_grid(ax)
        for y in dividers:
            ax.axhline(y, color=S.GREY_MID, linewidth=S.LW_THIN, zorder=0.5)
        per_pt = (mw if ax is main else HV_STUB) / 2.54 * 72 / abs(np.diff(ax.get_xlim())[0])     # points per data unit
        segs = [c.get_segments()[0] for c in ax.collections if hasattr(c, "get_segments")]
        hidden += sum(s[1][0] - s[0][0] <= (MEAN_DOT + 0.5) / per_pt for s in segs)
        bars += len(segs)
    S.labels(main, X_F2[met], None)
    return parts, main, bars, hidden


def _sd_ends(vals: list) -> list:
    if len(vals) < 2:
        return []
    m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    return [m - sd, m + sd]


def _draw_hv(path, value, refs: tuple, cells: list, absent: str) -> None:
    """One mode a figure (L20: points-only panels at least 4.5 cm, so two a row): cells, the five tasks' (task key, n,
    data set, metric), in HV_SLOTS, the row names left of each row's first panel, the legend in the sixth slot.
    value(*cell): the values of a network (_hv_values); refs: the mode's references (HV_REFS)."""
    from matplotlib.patches import Patch
    gets = [value(*cell) for cell in cells]
    if not any(gets):
        raise C.Absent(absent)
    # the row names: measured at the tick size, the partitions indented, the subheading in italic
    probe = plt.figure()
    rend = probe.canvas.get_renderer()
    names = [_hv_name(r) for r in HV_ROWS + HV_PARTITIONS + refs]
    width = max([probe.text(0, 0, t, fontsize=S.TICK_PT5).get_window_extent(rend).width for t in names]
                + [probe.text(0, 0, HV_HEAD, fontsize=S.TICK_PT5, fontstyle="italic").get_window_extent(rend).width])
    plt.close(probe)
    name_x = 0.1                                                    # cm: the row names' left edge
    left = name_x + width / probe.dpi * 2.54 + 0.2
    w = (S.PRINT_W - left - HV_RIGHT - HV_WGAP) / 2
    rows, head, dividers = _hv_rows(refs)
    # at least HV_UNIT a row, and the panel at least HV_ASPECT of its width high (figcheck's aspect rule)
    h = max((max(rows.values()) + 1.2) * HV_UNIT, HV_ASPECT * w)
    top, hgap = 0.6, 0.95 + 0.5                       # a title; under a row its x tick labels and label, then a title
    nrows = max(i for i, _ in HV_SLOTS + (HV_LEGEND_SLOT,)) + 1
    fig = S.figure_cm(S.PRINT_W, top + nrows * h + (nrows - 1) * hgap + 0.95)
    hidden = bars = 0
    panels = []
    for k, ((i, j), cell, get, (key, n, ds, met, title)) in enumerate(zip(HV_SLOTS, cells, gets, HV_PANELS)):
        y0, x = top + i * (h + hgap), left + j * (w + HV_WGAP)
        series = [(r, _hv_values(r, get, cell), _hv_colour(r), r in refs) for r in HV_ROWS + HV_PARTITIONS + refs]
        if not any(vals for _, vals, _, _ in series):
            ax = S.axes_cm(fig, x, y0, w, h)
            _empty_panel(ax)
            parts, main = [ax], ax
        else:
            parts, main, b, hd = _hv_panel(fig, x, y0, w, h, rows, dividers, series, _hv_values("L188", get, cell),
                                           cell[3])
            bars, hidden = bars + b, hidden + hd
        # the title centred over the whole panel, both parts of a broken axis
        fx0, fx1 = main.get_position().x0, main.get_position().x1
        xc = ((x + w / 2) / S.PRINT_W - fx0) / (fx1 - fx0)
        main.set_title(f"({'abcdefghij'[k]}) {title}", fontsize=HV_TITLE_PT, pad=4.0, x=xc)
        if j == 0:                                   # the row names on the panel's leftmost part, left-aligned
            ax = parts[0]
            ax.yaxis.set_major_formatter(ticker.FixedFormatter([_hv_name(r) for r in rows]))
            ax.yaxis.set_minor_locator(ticker.FixedLocator([head]))
            ax.yaxis.set_minor_formatter(ticker.FixedFormatter([HV_HEAD]))
            pad = (ax.get_position().x0 * S.PRINT_W - name_x) / 2.54 * 72
            ax.tick_params(axis="y", which="both", pad=pad, labelsize=S.TICK_PT5)
            for lab in ax.get_yticklabels() + ax.get_yticklabels(minor=True):
                lab.set_horizontalalignment("left")
            for lab in ax.get_yticklabels(minor=True):
                lab.set_fontstyle("italic")
        panels.append((parts, x, w))
    handles = [Line2D([], [], color="#000000", linewidth=L.BAR, marker="o", markersize=MEAN_DOT,
                      markerfacecolor="#000000", markeredgecolor="white", markeredgewidth=0.5),
               (Patch(facecolor=S.SERIES["L188"], alpha=L.BAND, linewidth=0),
                Line2D([], [], color=S.SERIES["L188"], linewidth=S.LW_THIN))]
    i, j = HV_LEGEND_SLOT
    S.slot_legend(fig, handles, left + j * (w + HV_WGAP), top + i * (h + hgap), w, h,
                  labels=["mean and standard deviation\nover pretraining runs", "188 classes, mean and\nstandard deviation"])
    _hv_ticks(fig, panels)
    print(f"{path.name}: {hidden} of {bars} SD bars are shorter than the mean's circle")
    S.save(fig, path)


def _hv_ticks(fig, panels: list) -> None:
    """The x ticks of every panel, labels at least HV_TICK_GAP from each other and HV_TICK_PANEL from another panel
    (figcheck: within 1 mm) and on the page, row by row and panel by panel from left to right: the one tick beyond an
    axis break first, then on the panel's main part the multiples of the coarsest round step (5, 2, 1 times a power of
    ten) that leaves two labels, a label dropped where it would come too close to one already placed. The main part
    holds every pretrained network, so its scale must be readable from two labels: where it gets one, the label beyond
    the break moves away from it by a quarter, then half its width, then the main part's range widens about its
    centre (HV_WIDEN) and, last, the label beyond the break is left out. panels: [(parts, x cm, width cm)]."""
    from matplotlib.text import Text
    from matplotlib.transforms import ScaledTranslation
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 2.54

    def width(label: str) -> float:
        t = Text(0, 0, label, fontsize=S.TICK_PT5)
        t.set_figure(fig)
        return t.get_window_extent(r).width

    rows = {}
    for parts, x, w in panels:
        rows.setdefault(round(parts[0].get_position().y0, 4), []).append((parts, x, w))
    for row in rows.values():
        taken = []
        extent = {a: (x * px, (x + w) * px) for parts, x, w in row for a in parts}
        for parts, x, w in sorted(row, key=lambda p: p[1]):
            # another panel: not a part of this one (a label of the main part may reach toward its own break)
            others = [(x_ * px, (x_ + w_) * px) for parts_, x_, w_ in row if parts_ is not parts]

            def place(ax, locs, fmt, before, shift=0.0) -> tuple:
                out, boxes = [], []
                for v in locs:
                    c, half = ax.transData.transform((v, 0))[0] + shift, width(fmt(v)) / 2
                    if c - half >= 0 and c + half <= S.PRINT_W * px and not any(
                            c - half < b1 + g * px and b0 < c + half + g * px
                            for (b0, b1), g in [(b, HV_TICK_GAP) for b in before + boxes]
                            + [(b, HV_TICK_PANEL) for b in others]):
                        out.append(v)
                        boxes.append((c - half, c + half))
                return out, boxes

            def main_ticks(ax, before, halves) -> tuple:
                lo, hi = ax.get_xlim()
                e = int(np.floor(np.log10(hi - lo)))
                keep, boxes, fmt = [], [], None
                steps = [m * 10.0 ** k for k in range(e, e - 4, -1) for m in (5, 2, 1)]
                for off_share, step in [(0.0, s_) for s_ in steps] + ([(0.5, s_) for s_ in steps] if halves else []):
                    off = off_share * step          # the multiples of a step, then (halves) those halfway between them
                    locs = [off + k * step for k in range(int(np.ceil((lo - off) / step)),
                                                           int(np.floor((hi - off) / step)) + 1)]
                    if not 2 <= len(locs) <= 12:
                        continue
                    unit = step / 2 if off else step
                    f = (lambda d: lambda v: f"{v:.{d}f}")(max(0, -int(np.floor(np.log10(unit) + 1e-9))))
                    got, bx = place(ax, locs, f, before)
                    if len(got) >= 2 and not _nice_spacing(got, unit):      # labels a round interval apart only
                        got, bx = got[:1], bx[:1]
                    if len(got) > len(keep):
                        keep, boxes, fmt = got, bx, f
                    if len(keep) >= 2:
                        return keep, boxes, fmt
                return keep, boxes, fmt

            stub = next((a for a in parts if a.get_label() == "<break>"), None)
            main = next(a for a in parts if a.get_label() != "<break>")
            sfmt = lambda v: f"{v:g}"
            best, xlim = None, main.get_xlim()
            shares = (0.0, 0.25, 0.5) if stub is not None else (0.0,)
            configs = [(s_, e_) for s_ in shares for e_ in HV_WIDEN] + (
                [(None, e_) for e_ in HV_WIDEN] if stub is not None else [])
            for share, widen, halves in [(s_, e_, False) for s_, e_ in configs] + [(s_, e_, True) for s_, e_ in configs]:
                main.set_xlim(xlim[0] - widen * (xlim[1] - xlim[0]) / 2, xlim[1] + widen * (xlim[1] - xlim[0]) / 2)
                sboxes, shift = [], 0.0
                if stub is not None and share is not None:
                    away = -1.0 if stub.get_position().x0 < main.get_position().x0 else 1.0
                    v = list(stub.get_xticks())
                    shift = away * share * (width(sfmt(v[0])) if v else 0.0)
                    if share:                      # moved: it must clear what is placed, as any other label
                        got, sboxes = place(stub, v, sfmt, taken, shift)
                        if len(got) < len(v):
                            continue
                    else:
                        sboxes = [(stub.transData.transform((t, 0))[0] - width(sfmt(t)) / 2,
                                   stub.transData.transform((t, 0))[0] + width(sfmt(t)) / 2) for t in v]
                keep, boxes, fmt = main_ticks(main, taken + sboxes, halves)
                if best is None or len(keep) > len(best[2]):
                    best = (shift, sboxes, keep, boxes, fmt, share, main.get_xlim())
                if len(keep) >= 2:
                    break
            shift, sboxes, keep, boxes, fmt, share, lim = best
            main.set_xlim(*lim)
            if stub is not None:
                if share is None:
                    stub.xaxis.set_major_formatter(ticker.NullFormatter())
                else:
                    stub.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: sfmt(v)))
                if shift:
                    for lab in stub.get_xticklabels():
                        lab.set_transform(lab.get_transform() + ScaledTranslation(shift / fig.dpi, 0,
                                                                                  fig.dpi_scale_trans))
            main.xaxis.set_major_locator(ticker.FixedLocator(keep))
            main.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _, fmt=fmt: fmt(v)))
            taken += sboxes + boxes


def _nice_spacing(vals: list, step: float) -> bool:
    """True where vals, multiples of step, are evenly spaced by 1, 2, 4 or 5 steps (not 0.961 and 0.968)."""
    d = np.diff(vals)
    return bool(np.allclose(d, d[0], rtol=1e-6)) and any(abs(d[0] / step - n) < 1e-6 for n in (1, 2, 4, 5))


def _head(auc: bool = False) -> list:
    return [(key, n, ds, F2_AUC[key] if auc else met) for key, n, ds, met, _ in HV_PANELS]


def draw_f2(path) -> None:
    """Fig. 2: the linear probes, each task's headline metric (L20: the fine-tuned networks are F2-ft)."""
    _draw_hv(path, _probe_value, HV_REFS["probe"], _head(), "no linear probes (linprobe/fits_v4)")


def draw_f2_ft(path) -> None:
    """The fine-tuned networks, each task's headline metric."""
    _draw_hv(path, _ft_value, HV_REFS["ft"], _head(), "no fine-tuning read-outs (ft_mlst, budget)")


def draw_f2a(path) -> None:
    """F2's linear probes in the AUC, with the released Sophon checkpoint."""
    _draw_hv(path, _probe_value, HV_REFS["probe_all"], _head(auc=True), "no linear probes (linprobe/fits_v4)")


def draw_f2b(path) -> None:
    """F2-ft's fine-tuned networks in the AUC."""
    _draw_hv(path, _ft_value, HV_REFS["ft"], _head(auc=True), "no fine-tuning read-outs (ft_mlst, budget)")


# ------------------------------------------------------------------ the training-size family: metric against the number of jets
# Plan "Training size" (PI 2026-11-12/13): one figure per mode, the five tasks of the headline figure (Herwig included),
# x the training jets (log), y the task's headline metric (or its AUC), the mean over runs with an SD band (lines_v6 C).
# Curves in the common order (L5), the mass networks in their lighter shade, all solid (one mode a figure, L6); the
# mode's reference: from scratch (grey) beside fine-tuned networks, the random network (dark grey) beside linear probes.
# Five panels, two a row at 6.3 cm (L20: no curve panel under 6.1 cm, PI 2026-11-13 "Panels can't be so small, e.g.
# figure 5"): the two multi-class tasks (accuracy), top and quark/gluon on Pythia, quark/gluon on Herwig (R30), the
# legend in the sixth slot beside the last.
SIZE5 = (("jc2", "jc2_188", "accuracy", "JetClass-II"), ("jc1", "jc1", "accuracy", "JetClass"),
         ("top", "top", "r30", "Top tagging"), ("qg", "qg", "r30", "Quark/gluon, Pythia"),
         ("qg_herwig", "qg_herwig", "r30", "Quark/gluon, Herwig"))
SIZE5_AUC = tuple((key, ds, "macro_auc_ovr" if key in ("jc2", "jc1") else "auc", title) for key, ds, _, title in SIZE5)
SIZE_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))        # the legend in (2, 1)
SIZE_LEGEND_SLOT = (2, 1)
SIZE_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
SIZE_PARTITIONS = ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")     # 188 and 17 classes for reference
SIZE_LEFT, SIZE_WGAP, SIZE_HGAP, SIZE_RIGHT = 1.35, 0.85, 1.0, 0.15      # cm
SIZE_W = (S.PRINT_W - SIZE_LEFT - SIZE_WGAP - SIZE_RIGHT) / 2              # 6.3 cm
SIZE_ASPECT = 0.72          # three rows and the caption on one page (make_results_book.MAX_H_CM)
SIZE_XLABEL = 0.45          # cm: an x label under a panel with another below it
POOLED = M._pooled(RUNS)


def _curve(ax, series: dict, line: dict, mark: dict, label: str, band: bool, style: str = L.SIZE, k: int = 0,
           n: int = 1) -> None:
    """Mean over runs; line the colour and line style, mark the marker and its fill (style_v5.marker); with band,
    Golling's: the SD, same colour, no edge. style B or C: lines_v6.scan, the k-th of n series."""
    xs = sorted(series)
    if not xs:
        return
    if style != "A":
        L.scan(ax, *L.stats(series, P.F._spread), color=line["color"], linestyle=line.get("linestyle", "-"),
               probe=mark.get("markerfacecolor") == "white", label=label, style=style, k=k, n=n)
        return
    st = [P.F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    ax.plot(xs, mean, **line, linewidth=S.LW_CURVE, **mark, markersize=MARKER, label=label)
    if band:
        ax.fill_between(xs, mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st]),
                        color=line["color"], alpha=0.2, linewidth=0)


def _model(series: dict, key: str, mode: str, label: str) -> tuple:
    return series, S.look(key, mode), S.marker(key, mode), label


STAR = 8.0                  # pt, ParT's published value: larger than a point, small enough to leave the curves' ends
PUBLISHED_LABEL = "published Particle\nTransformer"      # L7: no abbreviation in a legend
SIZE_PAD = 1.7              # a panel's x axis runs from its smallest training set over SIZE_PAD to its largest times it


def _draw_size(path, series_of, absent: str, auc: bool = False, style: str = L.SIZE) -> None:
    """series_of(key, ds, met): [(series, line, mark, label)] of a panel, in the legend's order; drawn in reverse, so
    the first (188 classes) lies on top. auc: the same five tasks in AUC (SIZE5_AUC) with ParT's published AUCs."""
    spec, published = (SIZE5_AUC, PUBLISHED_AUC) if auc else (SIZE5, PUBLISHED)
    panels = [series_of(key, ds, met) for key, ds, met, _ in spec]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    w, h, top = SIZE_W, SIZE_ASPECT * SIZE_W, 0.6
    nrows = max(i for i, _ in SIZE_SLOTS + (SIZE_LEGEND_SLOT,)) + 1
    # the lowest panel of each column carries the x label; one with a panel below it needs the room for it
    lowest = {j: max(i for i, j_ in SIZE_SLOTS if j_ == j) for _, j in SIZE_SLOTS}
    gaps = [SIZE_HGAP + (SIZE_XLABEL if any(lowest[j] == i for j in lowest) else 0) for i in range(nrows - 1)]
    y_of = [top + i * h + sum(gaps[:i]) for i in range(nrows)]
    fig = S.figure_cm(S.PRINT_W, y_of[-1] + h + 1.0)
    axes, handles = [], {}
    for k, ((i, j), panel, (key, _, met, title)) in enumerate(zip(SIZE_SLOTS, panels, spec)):
        ax = S.axes_cm(fig, SIZE_LEFT + j * (w + SIZE_WGAP), y_of[i], w, h)
        axes.append(ax)
        ax.set_xscale("log")
        ns = [n for s_, *_ in panel for n in s_] + ([published[key][0]] if key in published else [])
        if ns:                        # each panel's range from its own points: no empty decade (PI 2026-11-11)
            ax.set_xlim(min(ns) / SIZE_PAD, max(ns) * SIZE_PAD)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, title)
        S.labels(ax, X_JETS if i == lowest[j] else None, Y_OF[met] if j == 0 else None)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        for n_, (series, line, mark, label) in reversed(list(enumerate(panel))):
            _curve(ax, series, line, mark, label, True, style, n_, len(panel))
        if key in published:      # open, over the error bars (2.4) and under the points (2.6): a network's point where
            # the curves end at the published size stays visible inside the star instead of under its outline
            ax.plot(*published[key], color="#000000", marker="*", markersize=STAR, markerfacecolor="none",
                    markeredgecolor="#000000", markeredgewidth=0.8, linestyle="none", label=PUBLISHED_LABEL, zorder=2.5)
        if met == "r30":
            _log_y(ax)
        else:
            S.data_ylim(ax)         # the bands and the points clear of the frame (L7)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))     # labels of two decimals
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    labels = [lab for *_, lab in max(panels, key=len)] + [PUBLISHED_LABEL]
    if PUBLISHED_LABEL in handles:        # the star at its plotted size (markerscale 1 below)
        handles[PUBLISHED_LABEL] = Line2D([], [], color="#000000", marker="*", markersize=STAR, linestyle="none",
                                          markerfacecolor="none", markeredgewidth=0.8)
    # the legend in the free slot, one entry a line, centred in the slot
    i, j = SIZE_LEGEND_SLOT
    S.slot_legend(fig, [handles[lab] for lab in labels if lab in handles], SIZE_LEFT + j * (w + SIZE_WGAP), y_of[i], w, h,
                  labels=[lab for lab in labels if lab in handles])
    S.letter_panels(axes)
    S.save(fig, path)


def _probe(row: str, mode: str = "ft"):
    """A network's linear-probe series of a panel, solid in a figure of probes only (mode "ft"), dashed beside
    fine-tuned networks (mode "probe"); the random network (as initialized) in its own colour."""
    key = "INIT_BN" if row == M.INIT else row
    chosen = _chosen()
    return lambda k, ds, met: _model(P.probe_series(M, I, POOLED, ds, met, row, chosen), key, mode, S.LEGEND[key]
                                     + (f" {S.PROBE}" if mode == "probe" else ""))


def _tuned(row: str, tag: bool = False):
    if row == "SCRATCH":
        return lambda k, ds, met: _model(P.scratch_series(M, I["per"], k, met), "SCRATCH", "ft", S.LEGEND["SCRATCH"])
    return lambda k, ds, met: _model(P.ft_series(M, I["per"], POOLED, k, met, row), row, "ft",
                                     S.LEGEND[row] + (f" {S.FINE_TUNED}" if tag else ""))


def _size(path, makers: list, absent: str, auc: bool, style: str = L.SIZE) -> None:
    _draw_size(path, lambda key, ds, met: [m(key, ds, met) for m in makers], absent, auc=auc, style=style)


def draw_f8(path, auc: bool = False) -> None:
    """The 188-class network in two uses, both in the 188 colour: fine-tuned solid, the linear probe beside it dashed
    with open circles (L6); from scratch grey."""
    _size(path, [_tuned("L188", tag=True), _probe("L188", "probe"), _tuned("SCRATCH")],
          "no fine-tuning read-outs or linear probes", auc)


def draw_a10(path, auc: bool = False) -> None:
    """Linear probes of the label sets and the mass networks, and the random network."""
    _size(path, [_probe(r) for r in SIZE_ROWS] + [_probe(M.INIT)], "no linear probes (linprobe/fits_v4)", auc)


def draw_a8(path, auc: bool = False, style: str = L.SIZE) -> None:
    """A10's form after fine-tuning: the label sets, the mass networks and from scratch."""
    _size(path, [_tuned(r) for r in SIZE_ROWS] + [_tuned("SCRATCH")], "no fine-tuning read-outs", auc, style)


def draw_a10_part(path) -> None:
    _size(path, [_probe(r) for r in SIZE_PARTITIONS], "no linear probes (linprobe/fits_v4)", False)


def draw_a8_part(path) -> None:
    _size(path, [_tuned(r) for r in SIZE_PARTITIONS], "no fine-tuning read-outs", False)


# ------------------------------------------------------------------ the UMAP and the pretraining curve
EMBED_NAMES = {"bb": r"$X\to b\bar{b}$", "cc": r"$X\to c\bar{c}$", "qq": r"$X\to q\bar{q}$", "QCD": "QCD",
               "ee": r"$X\to ee$", "mm": r"$X\to \mu\mu$", "top": "top", "gluon": "gluon", "quark": "quark",
               "bs": r"$X\to bs$",
               **{c.removeprefix("label_"): S.JC1_TEX[c] for c in S.JC1_CLASSES if c != "label_QCD"}}
# L12: rows 188, 162, 43, 17 classes, pretraining run 1 (umap_v3 holds all four; the same jets as umap_v2, rows_sha256)
EMBED_MODELS = (("mtx-l188-s1", "L188"), ("mtx-l162-s1", "L162"), ("mtx-r42q1-s1", "R42_Q1"), ("mtx-r16q1-s1", "R16_Q1"))
# Brief B / L12: the mass networks' grids, each directly under its own label set (L5), run 1 as above
EMBED_MASS = (("mtx-l162-s1", "L162"), ("mtx-l162mass-s1", "L162_MASS"), ("mtx-r16q1-s1", "R16_Q1"),
              ("mtx-r16q1mass-s1", "R16_Q1_MASS"))
# X->bb and X->bs against QCD on Sophon's samples: frozen only (L11, L12), one column a signal
EMBED_XBX = (("xbb", r"$X\to b\bar{b}$"), ("xbs", r"$X\to bs$"))
# X->bb its colour of the hadronic JetClass-II grid, QCD black, X->bs a third CLASS10 colour (brown, not an X decay there)
XBX_COLOURS = [S.TAB10[0], S.TAB10[1], S.TAB10[7]]      # _draw_embeddings' group order: QCD, X->bb, X->bs


UMAP_DIRS = ("umap_v3", "umap_v2", "umap", "umap_preview")     # four label sets (2026-11-13); round 3; the long-budget jc1 run; the preview
UMAP_SETTINGS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 1, "standardized": True}
UMAP_READ = {      # one sentence per figure on what its panels show, written after looking at them (round 4); for the text
    "UMAP-jc1": "Both models' frozen features set the leptonic decays apart, in two islands for 188 classes; "
                "fine-tuning separates most hadronic classes, Z→qq̄ partly mixed with H→bb̄ and W→qq′.",
    "UMAP-jc2-had": "The 188-class frozen features separate the three decays; the 17-class ones only order them, X→bb̄ "
                    "at one edge and X→qq̄ near QCD at the other; fine-tuning separates them.",
    "UMAP-jc2-lep": "Each class falls in two islands in every panel; the 188-class frozen features keep electrons and "
                    "muons apart, the 17-class ones mix them, and fine-tuning separates them.",
    "UMAP-top": "Top and QCD jets lie mostly apart in both models' frozen features; fine-tuned, the 17-class "
                "model gives two islands, some QCD jets in the top one; the 188-class model joins them by mixed jets.",
    "UMAP-qg": "Quark and gluon jets overlap in the frozen features of both models, with only a gradient between them, "
               "and separate partly after fine-tuning.",
}
# page -> (set, heading, the jets, what the network was fine-tuned on)
UMAP_PAGES = {
    "UMAP-jc1": ("jc1", "JetClass", "2,000 test jets of each JetClass class", "one million JetClass jets"),
    "UMAP-jc2-had": ("two_prong", "JetClass-II, hadronic decays and QCD",
                     "2,000 test jets each of X→bb̄, X→cc̄, X→qq̄ and QCD",
                     "one million class-balanced JetClass-II jets"),
    "UMAP-jc2-lep": ("leptons", r"JetClass-II, $X\to ee$ and $X\to\mu\mu$",
                     "2,000 test jets each of X→ee and X→μμ", "one million class-balanced JetClass-II jets"),
    "UMAP-top": ("top", "top tagging", "2,000 top and 2,000 QCD test jets", "the full top-tagging training set"),
    "UMAP-qg": ("qg", "quark/gluon", "2,000 quark and 2,000 gluon test jets (Pythia)", "the full quark/gluon training set"),
}


def _umap_root(stems: list):
    """The first of UMAP_DIRS holding every file of a figure: umap_v2/ where pulled, else (JetClass only) umap/ or
    the preview."""
    for name in UMAP_DIRS:
        if all((P.DATA / name / f"{s}.json").exists() for s in stems):
            return P.DATA / name
    raise C.Absent(f"no UMAP of the class token for {stems[0]} ({', '.join(d + '/' for d in UMAP_DIRS)})")


def _embedding(ax, d: dict, colours: list) -> None:
    x, y, g = np.array(d["x"]), np.array(d["y"]), np.array(d["group"])
    for k, name in enumerate(d["groups"]):
        ax.plot(x[g == k], y[g == k], color=colours[k], marker="o", markersize=1.2, linestyle="none", alpha=0.5,
                markeredgewidth=0, label=EMBED_NAMES[name])
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_embeddings(path, grid: list, colours=None, wgap: float = 0.3) -> None:
    """grid: rows of (file stem, title). Round 4: 6.5 cm square panels, the class legend above them at 8.5 pt. wgap:
    the gap between columns (the mass networks' longer titles need more)."""
    from matplotlib.lines import Line2D
    root = _umap_root([stem for row in grid for stem, _ in row])
    docs = [[json.loads((root / f"{stem}.json").read_text()) for stem, _ in row] for row in grid]
    if any(d.get("umap", UMAP_SETTINGS) != UMAP_SETTINGS for row in docs for d in row):
        raise SystemExit(f"FATAL: {root}: UMAP settings other than the caption's {UMAP_SETTINGS}")
    # the groups of every panel in order of appearance (the X->bb and X->bs columns share QCD), one colour a group
    groups = list(dict.fromkeys(name for row in docs for d in row for name in d["groups"]))
    if len(groups) > len(docs[0][0]["groups"]) and "QCD" in groups:        # the shared background first
        groups = ["QCD"] + [g for g in groups if g != "QCD"]
    ncol = 5 if len(groups) > 5 else len(groups)
    legend_rows_n = math.ceil(len(groups) / ncol)
    # four label-set rows: the grid and its caption on one page, so the panels flatter than square (the UMAP plane has no
    # scale to keep) and at least 4.5 cm wide (L20, points only)
    w, aspect = (7.0, 1.0) if len(grid) <= 2 else (4.8, 0.72)
    left = 0.35 if len(grid) <= 2 else (S.PRINT_W - len(grid[0]) * w - (len(grid[0]) - 1) * wgap) / 2     # centred
    fig, axes, _ = S.grid_cm(len(grid), len(grid[0]), aspect=aspect, w=w, left=left, wgap=wgap, hgap=0.6, bottom=0.3,
                             top=0.95 + 0.5 * legend_rows_n)
    if colours is None:         # QCD black, as in every class figure; the other classes take the colours after it
        rest = iter(S.TAB10[1:])
        colours = [S.TAB10[0] if name == "QCD" else next(rest) for name in groups]
        if len(groups) <= 3:    # black and vermillion (and purple), which stand apart in small translucent dots
            first = "QCD" if "QCD" in groups else groups[0]
            rest = iter((S.TAB10[2], S.TAB10[3]))
            colours = [S.TAB10[0] if name == first else next(rest) for name in groups]
    colour = dict(zip(groups, colours))
    for r, (row, drow) in enumerate(zip(grid, docs)):
        for ax, (_, title), d in zip(axes[r], row, drow):
            _embedding(ax, d, [colour[name] for name in d["groups"]])
            S.title(ax, title)
    handles = [Line2D([], [], color=colours[k], marker="o", markersize=5, linestyle="none",
                      label=EMBED_NAMES[name]) for k, name in enumerate(groups)]
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=ncol)
    S.letter_panels(axes.flat)
    S.save(fig, path)


def _draw_umap(path, page: str) -> None:
    """Rows: 188 and 17 classes; columns: frozen, fine-tuned on the task; the same jets in all four panels."""
    name = UMAP_PAGES[page][0]
    grid = [[(f"{name}_{m}_{kind}", f"{S.NAME[level]}, {word}") for kind, word in (("frozen", "frozen"), ("ft", "fine-tuned"))]
            for m, level in EMBED_MODELS]
    colours = [S.JC1_COLOUR[c] for c in S.JC1_CLASSES] if name == "jc1" else None
    _draw_embeddings(path, grid, colours=colours)


# the supplement's further grids (L12, L19): page -> (rows, columns as (file stem prefix, kind, panel word), heading)
UMAP_MORE = {
    "UMAP-xbx": (EMBED_MODELS, [(name, "frozen", word) for name, word in EMBED_XBX],
                 r"$X\to b\bar b$ and $X\to bs$ against QCD, frozen"),
    **{f"UMAP-mass-{page.removeprefix('UMAP-')}": (EMBED_MASS, [(UMAP_PAGES[page][0], kind, word) for kind, word in
                                                               (("frozen", "frozen"), ("ft", "fine-tuned"))],
                                                  f"mass networks, {UMAP_PAGES[page][1]}")
       for page in ("UMAP-jc2-had", "UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg")},
    "UMAP-mass-xbx": (EMBED_MASS, [(name, "frozen", word) for name, word in EMBED_XBX],
                      r"mass networks, $X\to b\bar b$ and $X\to bs$ against QCD, frozen"),
}
XBX_JETS = ("2,000 X→bb̄ and 2,000 QCD test jets (left) and 2,000 X→bs and 2,000 QCD test jets (right), all with "
            "soft-drop mass 150 to 230 GeV, from the samples of table {tab:xbx}")
QUALITATIVE = ("Every panel uses pretraining run 1, the same UMAP settings (15 neighbours, minimum distance 0.1) and seed, "
               "and the same jets as {}. The plots are qualitative: each panel is fitted separately, so positions "
               "cannot be compared between panels.")


def _draw_umap_more(path, page: str) -> None:
    rows, cols, _ = UMAP_MORE[page]
    grid = [[(f"{name}_{m}_{kind}", f"{S.LEGEND[level]}, {word}") for name, kind, word in cols] for m, level in rows]
    colours = {"jc1": [S.JC1_COLOUR[c] for c in S.JC1_CLASSES], "xbb": XBX_COLOURS}.get(cols[0][0])
    _draw_embeddings(path, grid, colours=colours, wgap=1.4 if rows == EMBED_MASS else 0.3)


def _umap_more_caption(page: str) -> str:
    """The supplement caption of a UMAP_MORE grid, with {fig:...} and {tab:...} for gallery_captions.fill."""
    rows, cols, _ = UMAP_MORE[page]
    xbx, mass = cols[0][0] == "xbb", rows == EMBED_MASS
    task = next((p for p in UMAP_PAGES if page == f"UMAP-mass-{p.removeprefix('UMAP-')}"), None)
    nets = ("of the networks pretrained with 162 classes, 162 classes + mass, 17 classes and 17 classes + mass (rows), "
            "each mass network under its own label set" if mass else
            "of the networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (rows)")
    what = (f"frozen, for {XBX_JETS}. The pretrained networks are not fine-tuned on these samples."
            if xbx else f"frozen (left) and fine-tuned on {UMAP_PAGES[task][3]} (right), for {UMAP_PAGES[task][2]}.")
    see = ("Figure {fig:UMAP-xbx} shows the label sets on the same jets." if xbx and mass else
           f"Figure {{fig:{task}}} shows the label sets on the same jets." if mass else "")
    return ("Two-dimensional embeddings, computed with uniform manifold approximation and projection (UMAP), of the "
            f"class token {nets}, {what} "
            + QUALITATIVE.format("the other panels of its column" if xbx else "every other panel") + f" {see}").rstrip()


def _umap_more_page(page: str) -> C.Page:
    rows, cols, heading = UMAP_MORE[page]
    xbx = cols[0][0] == "xbb"
    return C.Page(
        id=page,
        title=f"UMAP of the class token: {heading} (supplement)",
        precedent=["map2026_poster/fig5.png", "2610.36121/fig3.png"],
        precedent_label="MAP 2026 poster (frozen against fine-tuned), SuperJEPA Fig. 3",
        caption=_umap_more_caption(page),
        differs="The UMAP family's 4x2 grid (L12): " + ("the mass networks, each under its own label set (Brief B); "
                                                        if rows == EMBED_MASS else "")
                + ("X→bb̄ and X→bs against QCD as the two columns, frozen only (L11)." if xbx
                   else "columns frozen and fine-tuned."),
        draw=lambda path: _draw_umap_more(path, page),
        data_note="From umap_v3 (L19).")


def _umap_page(page: str) -> C.Page:
    _, heading, jets, ft = UMAP_PAGES[page]
    u = UMAP_SETTINGS
    return C.Page(
        id=page,
        title=f"UMAP of the class token, frozen and fine-tuned: {heading} (appendix)",
        precedent=["map2026_poster/fig5.png", "2610.36121/fig3.png"],
        precedent_label="MAP 2026 poster (frozen against fine-tuned), SuperJEPA Fig. 3",
        caption=f"Two-dimensional embeddings, computed with uniform manifold approximation and projection (UMAP), "
                f"of the class token of the networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (rows), "
                f"frozen (left) and fine-tuned on {ft} (right), for {jets} in every panel. Every "
                f"panel uses pretraining run 1, the same jets and the same UMAP settings ({u['n_neighbors']} "
                f"neighbours, minimum distance {u['min_dist']}) and random seed. The plots are qualitative, and "
                "positions cannot be compared between panels.",      # PI 2026-11-12: no result sentences
        differs="One 2x2 layout for every task (PI, round 3): rows label sets, columns frozen and fine-tuned, the "
                "poster's pairing. " + ("Ten class colours, JetClass's fixed tab10. " if page == "UMAP-jc1" else "tab10 in group order. ")
                + "One legend, first panel. " + (f"UMAP with {u['n_neighbors']} neighbours, minimum distance "
                                                   f"{u['min_dist']}, seed {u['random_state']}.") + (" Absent inputs zero-filled, as in every frozen probe and fine-tuning."
                                                 if page in ("UMAP-top", "UMAP-qg") else ""),
        draw=lambda path: _draw_umap(path, page),
        data_note=f"From umap_v2 (scripts/build_umap_jobs.py --out-name umap_v2). Reading, for the text: {UMAP_READ[page]}")




VAL_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")       # L5
VAL_PARTITIONS = ("R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")      # 17 classes, the one comparable height


def _draw_val(path, rows: tuple) -> None:
    """Exact size, one panel SINGLE_W wide (PI 2026-11-12), the legend above it in two columns read downwards (L5
    order); each network's mean over runs with a +-1 SD band (the five random partitions pooled)."""
    docs = M._load(P.DATA / "compute")
    runs = M._pooled(RUNS)
    legend_rows = math.ceil(len(rows) / 2)
    top, w = 0.5 + 0.45 * legend_rows, S.SINGLE_W - 1.5 - 0.25
    fig = S.figure_cm(S.SINGLE_W, top + 0.85 * w + 1.0)
    ax = S.axes_cm(fig, 1.5, top, w, 0.85 * w)
    S.fine_grid(ax)
    for row in rows:
        have = np.array([[e["val_acc"] for e in docs[m]["epochs"]] for m in runs[row]
                         if m in docs and "val_acc" in docs[m]["epochs"][0]])
        if not len(have):
            raise C.Absent(f"no validation accuracy per epoch for {row} (compute/)")
        x, mean = np.arange(1, have.shape[1] + 1), have.mean(axis=0)
        if len(have) > 1:
            sd = have.std(axis=0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=S.SERIES[row], alpha=L.BAND, linewidth=0, zorder=1.0)
        ax.plot(x, mean, **S.look(row), linewidth=S.LW_CURVE, label=S.LEGEND[row])
    S.labels(ax, "Pretraining epoch", "Validation accuracy")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=S.LEGEND_TOP, ncol=2)
    S.save(fig, path)


def draw_val(path) -> None:
    _draw_val(path, VAL_ROWS)


def draw_val_part(path) -> None:
    _draw_val(path, VAL_PARTITIONS)


# ------------------------------------------------------------------ the pages
HONG = ["2410.23129/fig2.png"]
SUBST_NOTE = ("Top-tagging linear probes fill the missing inputs with random draws and quark/gluon probes with the "
              "training mean: the substitution chosen for each task on validation, over all 30 models.")
TWO_RUNS = "The three controlled partitions have two runs each, the four label sets three."

F2_KEY = ("Circles: mean; bars: ±1 SD, hidden where shorter than the circle; three pretraining runs per label set, "
          "two per mass network and flavor-boundary partition, five partitions for random partitions, three "
          "from-scratch seeds. Line and band: the 188-class mean and SD. ")
HV_NOTE = (f"{TWO_RUNS} The two mass networks have two runs each, each plain random partition one. The random network "
           "lies beyond an axis break where it is far below every pretrained network (drawn on the scale of the rest). ")
SIZE_DIFFERS = ("The family's form (PI 2026-11-13): the five tasks of the headline figure, Herwig included, two multi-class "
                "panels over three binary ones and the legend in the free slot; the label sets with the mass networks "
                "in their lighter shades, all solid (one mode a figure); bands the SD over runs (Golling).")
SIZE_NOTE = ("Means over runs: three per label set, two per mass network, three from-scratch seeds or random-network "
             "initializations.")
# the supplement members' captions, filled like the paper's (gallery_captions holds the main-text and appendix ones)
SUPPLEMENT_CAPTIONS = {
    "VAL-part": (
        "The curves show the validation accuracy on JetClass-II against pretraining epoch for the 17-class partitions, "
        "with the network pretrained with 17 classes for reference, every network scored on its own 17 classes. Each "
        "curve is the mean, and its band ±1 standard deviation, over two pretraining runs per partition with a named "
        "flavor boundary, three for 17 classes, or the five random partitions. The learning rate decays from epoch 57, "
        "where every curve rises slightly. The values at the last epoch are in table {tab:compute}."
    ),
    "F8": (
        'Fine-tuning the network pretrained with 188 classes is more accurate than training from scratch on JetClass-II'
        ' and JetClass at every training-set size. The panels show accuracy or R30, the inverse background efficiency '
        'at 30% signal efficiency, against the number of training jets, for that network fine-tuned (solid) or as a '
        'linear probe on its frozen features (dashed) and for training from scratch (grey). Points, offset slightly in '
        'size, are means and error bars ±1 standard deviation over three pretraining runs or three trainings from '
        'scratch, and open stars mark the published Particle Transformer, trained from scratch, at its training-set '
        'size. Fine-tuning and training from scratch on JetClass-II and JetClass start at 10,000 jets, and the linear '
        'probes fill the missing top inputs with random draws and the quark/gluon inputs with the training mean, as in '
        'table {tab:probesnfull}. The values are in tables {tab:ftfull} and {tab:probesnfull}.'
    ),
    "F8-auc": (
        'The panels of figure {fig:F8} in the area under the receiver operating characteristic curve (AUC), averaged '
        'over the classes, each against the rest, on class-balanced JetClass-II and JetClass, and of the binary task on'
        ' top and quark/gluon tagging, scored on Pythia and on Herwig jets, against the number of training jets. The '
        'network pretrained with 188 classes is fine-tuned (solid) or a linear probe on its frozen features (dashed), '
        'and training from scratch is grey. Points, offset slightly in size, are means and error bars ±1 standard '
        'deviation over three pretraining runs or three trainings from scratch, and open stars mark the published '
        'Particle Transformer. Fine-tuning and training from scratch on JetClass-II and JetClass start at 10,000 jets, '
        'and the linear probes fill the missing top inputs with random draws and the quark/gluon inputs with the '
        'training mean, as in table {tab:probesnfull}. Values are in tables {tab:ftfull} and {tab:probesnfull}.'
    ),
    "A8-auc": (
        'The panels of figure {fig:A8} in the area under the receiver operating characteristic curve (AUC), averaged '
        'over the classes, each against the rest, on class-balanced JetClass-II and JetClass, and of the binary task on'
        ' top and quark/gluon tagging, scored on Pythia and on Herwig jets, for the fine-tuned networks of each label '
        'set, with and without the mass regression (+ mass), and training from scratch (grey). Points, offset slightly '
        'in size, are means and error bars ±1 standard deviation over three pretraining runs, two for + mass, or three '
        'trainings from scratch, and open stars mark the published Particle Transformer. JetClass-II and JetClass start'
        ' at 10,000 jets, and the values are in table {tab:ftfull}.'
    ),
    "A10-auc": (
        'The panels of figure {fig:A10} in the area under the receiver operating characteristic curve (AUC), averaged '
        'over the classes, each against the rest, on class-balanced JetClass-II and JetClass, and of the binary task on'
        ' top and quark/gluon tagging, scored on Pythia and on Herwig jets, for linear probes on the frozen features of'
        ' each label set, with and without the mass regression (+ mass), and of the random network (dark grey). Points,'
        ' offset slightly in size, are means and error bars ±1 standard deviation over three pretraining runs, two for '
        '+ mass, or three random initializations, and open stars mark the published Particle Transformer. The linear '
        'probes fill the missing top inputs with random draws and the quark/gluon inputs with the training mean, as in '
        'table {tab:probesnfull}, where the values are.'
    ),
    "A8-part": (
        'The networks pretrained with the four 17-class partitions in the form of figure {fig:A8}, fine-tuned and '
        'scored on class-balanced JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging, Pythia and '
        'Herwig, in R30, the inverse background efficiency at 30% signal efficiency, against the number of training '
        'jets, with 188 and 17 classes for reference. The partitions were fine-tuned on JetClass-II at one million jets'
        ' only, so that panel shows one point for each, fine-tuning on JetClass starts at 10,000 jets, and the particle'
        ' inputs that the top and quark/gluon data lack are set to zero. Points, offset slightly in size, are means and'
        ' error bars ±1 standard deviation over two pretraining runs for each flavor-boundary partition, the five '
        'random partitions with one run each, and three runs for 188 and 17 classes. Open stars mark the published '
        'Particle Transformer, and the values are in tables {tab:ft} and {tab:summary}.'
    ),
    "A10-part": (
        'The networks pretrained with the four 17-class partitions in the form of figure {fig:A10}, as linear probes on'
        ' their frozen features, trained and scored on class-balanced JetClass-II and JetClass (accuracy) and on top '
        'and quark/gluon tagging, Pythia and Herwig, in R30, the background rejection at 30% signal efficiency, against'
        ' the number of training jets, with 188 and 17 classes for reference. Points, offset slightly in size, are '
        'means and error bars ±1 standard deviation over two pretraining runs for each flavor-boundary partition, the '
        'five random partitions with one run each, and three runs for 188 and 17 classes. Open stars mark the published'
        ' Particle Transformer, and the values are in tables {tab:probesn} and {tab:summary}.'
    ),
}
SUPPLEMENT_CAPTIONS.update({page: _umap_more_caption(page) for page in UMAP_MORE})
PAGES = [
    C.Page(
        id="F2",
        title="Label granularity, linear probes",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with the mass regression, and "
                "with 17-class partitions, as linear probes on frozen features (upper row, beside the random network) "
                "and fine-tuned (lower row, beside training from scratch), at each task's largest training set: "
                "accuracy on class-balanced JetClass-II and on JetClass, R30 on top and quark/gluon tagging, Pythia "
                "and Herwig. " + F2_KEY + "Values: " + G.number("tab:summary") + ".",
        differs="Hong plots accuracy against the number of classes; here the label set is a category (PI 2026-11-11), "
                "one row a network, on the metric itself (PI 2026-11-12: actual values only), mean and SD; the two "
                "modes as two rows of five tasks, headline metric only (PI 2026-11-13, L1); L20: two panels a row, so "
                "one mode a figure, the fine-tuned networks in F2-ft.",
        draw=draw_f2,
        data_note=f"{HV_NOTE}{SUBST_NOTE} Herwig: the quark/gluon probes and networks scored on Herwig jets."),
    C.Page(
        id="F2-ft",
        title="Label granularity, fine-tuned",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="",
        differs="F2's form for the fine-tuned networks, beside training from scratch (L20: the old F2's lower row).",
        draw=draw_f2_ft,
        data_note="JetClass-II at 1,000,000 class-balanced jets. JetClass at 1,000,000 jets, top at 1,200,000, "
                  f"quark/gluon (Pythia and Herwig) at 1,600,000. {HV_NOTE}"),
    C.Page(
        id="F2a",
        title="Label granularity, linear probes, with AUC",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Linear probes on the frozen features of the networks of Fig. 2, the random network and the released "
                "Sophon checkpoint, at each task's largest training set: upper row the headline metric (accuracy, "
                "R30), lower row the AUC. " + F2_KEY + "Values: " + G.number("tab:summary") + " and "
                + G.number("tab:summaryauc") + ".",
        differs="F2's form in the AUC (L20: the old F2a's lower row; its upper row is F2).",
        draw=draw_f2a,
        data_note=f"{HV_NOTE}{SUBST_NOTE} The released Sophon checkpoint (one network) is drawn on JetClass-II and "
                  "JetClass only: its top and quark/gluon probes have the missing inputs set to zero (no substitution "
                  "probes exist for it), so they are not like-for-like with our rows; their values (R30 97, 54.3 and "
                  f"29.9 on Herwig) are in {G.number('tab:probesfull')}."),
    C.Page(
        id="F2b",
        title="Label granularity, fine-tuned, with AUC",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="The networks of Fig. 2 fine-tuned, and a network trained from scratch, on a million class-balanced "
                "JetClass-II and a million JetClass jets and the full top and quark/gluon sets (also scored on Herwig "
                "jets): upper row the headline metric (accuracy, R30), lower row the AUC. " + F2_KEY + "Values: "
                + G.number("tab:summary") + " and " + G.number("tab:summaryauc") + ".",
        differs="F2-ft's form in the AUC (L20: the old F2b's lower row; its upper row is F2-ft).",
        draw=draw_f2b,
        data_note="JetClass-II at 1,000,000 class-balanced jets. JetClass at 1,000,000 jets, top at 1,200,000, "
                  f"quark/gluon (Pythia and Herwig) at 1,600,000. {HV_NOTE}"),
    C.Page(
        id="A8",
        title="Training size, fine-tuned",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs=SIZE_DIFFERS + " Fine-tuned: from scratch (grey) is the reference.",
        draw=draw_a8,
        data_note=SIZE_NOTE + " On JetClass-II and JetClass the plan's sizes drop 1,000 jets for fine-tuning and from "
                  "scratch, so those curves start at 10,000 by design."),
    C.Page(
        id="A10",
        title="Training size, linear probes",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs=SIZE_DIFFERS + " Linear probes: the random network (dark grey) is the reference; the from-scratch curve "
                               "left the figure (plan).",
        draw=draw_a10,
        data_note=f"{SIZE_NOTE} {SUBST_NOTE}"),
    C.Page(
        id="F8",
        title="Training size, 188 classes against from scratch",
        precedent=["2401.13537/fig3.png", "2401.13536/fig1.png"],
        precedent_label="Golling Fig. 3, Vigl Fig. 1 top",
        caption="",
        differs="A named slice of the training-size family (L9): the 188-class network fine-tuned (solid) and as a linear "
                "probe (dashed, open circles; was a lighter tint, which now means + mass), from scratch grey; the "
                "family's five panels.",
        draw=draw_f8,
        data_note=f"Sizes below the full set use the fixed step budget with early stopping; the full JetClass, top and "
                  f"quark/gluon points use the full-set recipe. {SUBST_NOTE}"),
    C.Page(
        id="F8-auc",
        title="Training size, 188 classes against from scratch, AUC",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs="F8's form in AUC (PI 2026-11-11: the scaling figures need AUC as well as accuracy and rejection).",
        draw=lambda path: draw_f8(path, auc=True)),
    C.Page(
        id="A8-auc",
        title="Training size, fine-tuned, AUC",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs="A8's form in AUC.",
        draw=lambda path: draw_a8(path, auc=True)),
    C.Page(
        id="A10-auc",
        title="Training size, linear probes, AUC",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs="A10's form in AUC.",
        draw=lambda path: draw_a10(path, auc=True)),
    C.Page(
        id="A8-part",
        title="Training size, fine-tuned, 17-class partitions",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs="A8's form for the four 17-class partitions, with 188 and 17 classes for reference.",
        draw=draw_a8_part,
        data_note="The partitions were fine-tuned on JetClass-II at 1,000,000 jets only."),
    C.Page(
        id="A10-part",
        title="Training size, linear probes, 17-class partitions",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="",
        differs="A10's form for the four 17-class partitions, with 188 and 17 classes for reference.",
        draw=draw_a10_part,
        data_note=SUBST_NOTE),
    *[_umap_page(page) for page in UMAP_PAGES],
    *[_umap_more_page(page) for page in UMAP_MORE],
    C.Page(
        id="VAL",
        title="Pretraining validation accuracy against epoch (appendix)",
        precedent=["2404.16091/fig1.png"],
        precedent_label="OmniLearn Fig. 1 (validation loss against epochs)",
        caption="",
        differs="Accuracy, not loss; one curve per network, the mass networks in their lighter shades with SD bands "
                "(PI 2026-11-13); no marker for 80 points; theme grid.",
        draw=draw_val),
    C.Page(
        id="VAL-part",
        title="Pretraining validation accuracy, 17-class partitions (supplement)",
        precedent=["2404.16091/fig1.png"],
        precedent_label="OmniLearn Fig. 1 (validation loss against epochs)",
        caption="",
        differs="VAL's form for the 17-class partitions, the five random partitions as one curve with the band their "
                "SD, and 17 classes for reference: the one label set whose accuracy is on as many classes.",
        draw=draw_val_part,
        data_note="Two runs per flavor-boundary partition, one per random partition, three for 17 classes."),
]

import dataclasses  # noqa: E402
import gallery_captions  # noqa: E402  the paper's captions replace those above (PI 2026-11-12)
PAGES = [dataclasses.replace(p, caption=gallery_captions.fill(SUPPLEMENT_CAPTIONS[p.id])) if p.id in SUPPLEMENT_CAPTIONS
         else p for p in gallery_captions.apply(PAGES)]
