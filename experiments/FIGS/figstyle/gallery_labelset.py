#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2, F2a, F2b, the training-size family (A8, A10, F8, their AUC versions, A8-part, A10-part), UMAP-jc1, UMAP-jc2-had, UMAP-jc2-lep, UMAP-top, UMAP-qg, VAL (docs/figstyle/PLAN.md), in
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


# ------------------------------------------------------------------ F2, F2a, F2b: the headline value
# PI 2026-11-12/13 (L1, L5): the label set is a category, so a horizontal dot plot, one row a network, on the metric
# itself, one column a task (HV_PANELS, Herwig included). F2 (main text): the linear probes over the fine-tuned
# networks, headline metric only. F2a and F2b (appendix, the full versions): one mode each, the headline metric over the task's
# AUC. Network rows in the one order of every figure and table (L5): the label sets, each mass network under its own;
# the 17-class partitions under an italic subheading; the mode's references (HV_REFS). Each row: the mean over its runs
# a filled circle in the network's colour, a horizontal bar its SD (the five random partitions: mean and SD over the
# five). The 188-class mean is a vertical line with a light band of its SD down the panel. A reference far from every
# other row (the random network) sits beyond an axis break, on the scale of the rest. Nothing printed, nothing clipped.
HV_PANELS = (("jc2", 1_000_000, "jc2_188", "accuracy", "JetClass-II"),
             ("jc1", 1_000_000, "jc1", "accuracy", "JetClass"), ("top", 1_200_000, "top", "r30", "Top tagging"), ("qg", 1_600_000, "qg", "r30", "Quark/gluon\nPythia"),
             ("qg_herwig", 1_600_000, "qg_herwig", "r30", "Quark/gluon\nHerwig"))
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
HV_WGAP, HV_RIGHT = 0.3, 0.12    # cm between panels; right of the last
HV_STUB, HV_STUB_GAP = 0.55, 0.1  # cm: the part of a panel beyond an axis break, and the break
HV_BREAK = 1.5                   # a reference further from the other rows than this many times their span: a break
HV_INDENT = "   "                # the partitions under their subheading, as in the tables
# the panel titles half a point under the style's 9 pt: "(d) Quark/gluon" (2.33 cm at 9 pt) is wider than a panel
# and its gap
HV_TITLE_PT = 8.5
HV_TICK_GAP, HV_TICK_PANEL = 0.06, 0.12      # cm at least between two x tick labels; a label and another panel


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


def _draw_hv(path, grid: list, absent: str, mode_names: tuple = ()) -> None:
    """grid: two rows of five cells, each row (value function, its references, [(task key, n, data set, metric)]).
    mode_names: the use of each row of panels, written turned on the left (F2), or none (F2a, F2b: one use)."""
    from matplotlib.patches import Patch
    cells = [[(value, refs, cell, value(*cell)) for cell in row] for value, refs, row in grid]
    if not any(v for row in cells for *_, v in row):
        raise C.Absent(absent)
    refs_all = tuple(dict.fromkeys(r for _, refs, _ in grid for r in refs))
    # the row names: measured at the tick size, the partitions indented, the subheading in italic
    probe = plt.figure()
    rend = probe.canvas.get_renderer()
    names = [_hv_name(r) for r in HV_ROWS + HV_PARTITIONS + refs_all]
    width = max([probe.text(0, 0, t, fontsize=S.TICK_PT5).get_window_extent(rend).width for t in names]
                + [probe.text(0, 0, HV_HEAD, fontsize=S.TICK_PT5, fontstyle="italic").get_window_extent(rend).width])
    plt.close(probe)
    name_x = 0.45 if mode_names else 0.1                           # cm: the turned mode names before the row names
    left = name_x + width / probe.dpi * 2.54 + 0.2
    w = (S.PRINT_W - left - HV_RIGHT - 4 * HV_WGAP) / 5
    heights, dividers = [], []
    rows_of = []
    for _, refs, _ in grid:
        rows, head, div = _hv_rows(refs)
        rows_of.append((rows, head))
        heights.append((max(rows.values()) + 1.2) * HV_UNIT)
        dividers.append(div)
    two_line = any("\n" in t for *_, t in HV_PANELS)
    top = S.LEGEND_TOP + 0.45 + (0.85 if two_line else 0.5)
    hgap = 0.95 + (0.85 if two_line else 0.5)
    fig = S.figure_cm(S.PRINT_W, top + sum(heights) + hgap + 0.95)
    hidden = bars = 0
    panels, k = [], 0
    for i, (row, (rows, head), h, div) in enumerate(zip(cells, rows_of, heights, dividers)):
        y0 = top + sum(heights[:i]) + i * hgap
        for j, ((value, refs, cell, get), (key, n, ds, met, title)) in enumerate(zip(row, HV_PANELS)):
            series = [(r, _hv_values(r, get, cell), _hv_colour(r), r in refs) for r in HV_ROWS + HV_PARTITIONS + refs]
            x = left + j * (w + HV_WGAP)
            if not any(vals for _, vals, _, _ in series):
                ax = S.axes_cm(fig, x, y0, w, h)
                _empty_panel(ax)
                parts, main = [ax], ax
            else:
                parts, main, b, hd = _hv_panel(fig, x, y0, w, h, rows, div, series, _hv_values("L188", get, cell),
                                               cell[3])
                bars, hidden = bars + b, hidden + hd
            # the title centred over the whole panel, both parts of a broken axis
            fx0, fx1 = main.get_position().x0, main.get_position().x1
            xc = ((x + w / 2) / S.PRINT_W - fx0) / (fx1 - fx0)
            main.set_title(f"({'abcdefghij'[k]}) {title}", fontsize=HV_TITLE_PT, pad=4.0, x=xc)
            k += 1
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
        if mode_names:
            fig.text(0.18 / S.PRINT_W, 1 - (y0 + h / 2) / (fig.get_figheight() / S.CM), mode_names[i], rotation=90,
                     ha="center", va="center", fontsize=S.LABEL_PT5)
    handles = [Line2D([], [], color="#000000", linewidth=L.BAR, marker="o", markersize=MEAN_DOT,
                      markerfacecolor="#000000", markeredgecolor="white", markeredgewidth=0.5),
               (Patch(facecolor=S.SERIES["L188"], alpha=L.BAND, linewidth=0),
                Line2D([], [], color=S.SERIES["L188"], linewidth=S.LW_THIN))]
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=2,
                 labels=["mean and standard deviation over pretraining runs", "188 classes, mean and standard deviation"])
    _hv_ticks(fig, panels)
    print(f"{path.name}: {hidden} of {bars} SD bars are shorter than the mean's circle")
    S.save(fig, path)


def _hv_ticks(fig, panels: list) -> None:
    """The x ticks of every panel, labels at least HV_TICK_GAP from each other and HV_TICK_PANEL from another panel
    (figcheck: within 1 mm) and on the page, row by row from left to right: the one tick beyond an axis break first,
    then on each panel's main part the multiples of the coarsest round step (5, 2, 1 times a power of ten) that leaves
    two labels, a label dropped where it would come too close to one already placed. panels: [(parts, x cm, width cm)]."""
    from matplotlib.text import Text
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
        for ax in sorted(extent, key=lambda a: a.get_label() != "<break>"):
            others = [e for e in set(extent.values()) if e != extent[ax]]

            def place(locs, fmt) -> list:
                out, boxes = [], []
                for v in locs:
                    c, half = ax.transData.transform((v, 0))[0], width(fmt(v)) / 2
                    if c - half >= 0 and c + half <= S.PRINT_W * px and not any(
                            c - half < b1 + g * px and b0 < c + half + g * px
                            for (b0, b1), g in [(b, HV_TICK_GAP) for b in taken + boxes]
                            + [(b, HV_TICK_PANEL) for b in others]):
                        out.append(v)
                        boxes.append((c - half, c + half))
                return out, boxes

            if ax.get_label() == "<break>":
                fmt = lambda v: f"{v:g}"
                keep = list(ax.get_xticks())
                boxes = [(ax.transData.transform((v, 0))[0] - width(fmt(v)) / 2,
                          ax.transData.transform((v, 0))[0] + width(fmt(v)) / 2) for v in keep]
            else:
                lo, hi = ax.get_xlim()
                e = int(np.floor(np.log10(hi - lo)))
                keep, boxes, fmt = [], [], None
                for step in (m * 10.0 ** k for k in range(e, e - 4, -1) for m in (5, 2, 1)):
                    locs = [k * step for k in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1)]
                    if not 2 <= len(locs) <= 12:
                        continue
                    f = (lambda d: lambda v: f"{v:.{d}f}")(max(0, -int(np.floor(np.log10(step) + 1e-9))))
                    got, bx = place(locs, f)
                    if len(got) > len(keep):
                        keep, boxes, fmt = got, bx, f
                    if len(keep) >= 2:
                        break
            ax.xaxis.set_major_locator(ticker.FixedLocator(keep))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _, fmt=fmt: fmt(v)))
            taken += boxes


def draw_f2(path) -> None:
    """Fig. 2: the linear probes over the fine-tuned networks, each task's headline metric."""
    head = [(key, n, ds, met) for key, n, ds, met, _ in HV_PANELS]
    _draw_hv(path, [(_probe_value, HV_REFS["probe"], head), (_ft_value, HV_REFS["ft"], head)],
             "no linear probes or fine-tuning read-outs (linprobe/fits_v4, ft_mlst)",
             mode_names=("Linear probe", "Fine-tuned"))


def _full(value, refs, path, absent: str) -> None:
    grid = [[(key, n, ds, met) for key, n, ds, met, _ in HV_PANELS],
            [(key, n, ds, F2_AUC[key]) for key, n, ds, *_ in HV_PANELS]]
    _draw_hv(path, [(value, refs, row) for row in grid], absent)


def draw_f2a(path) -> None:
    """Fig. 2's linear probes in full: headline metric over AUC, with the released Sophon checkpoint."""
    _full(_probe_value, HV_REFS["probe_all"], path, "no linear probes (linprobe/fits_v4)")


def draw_f2b(path) -> None:
    """Fig. 2's fine-tuned networks in full: headline metric over AUC."""
    _full(_ft_value, HV_REFS["ft"], path, "no fine-tuning read-outs (ft_mlst, budget)")


# ------------------------------------------------------------------ the training-size family: metric against the number of jets
# Plan "Training size" (PI 2026-11-12/13): one figure per mode, the five tasks of the headline figure (Herwig included),
# x the training jets (log), y the task's headline metric (or its AUC), the mean over runs with an SD band (lines_v6 C).
# Curves in the common order (L5), the mass networks in their lighter shade, all solid (one mode a figure, L6); the
# mode's reference: from scratch (grey) beside fine-tuned networks, the random network (dark grey) beside linear probes.
# Five panels: the two multi-class tasks (accuracy) over the three binary ones (R30), three 3.9 cm panels a row, the
# legend in the slot beside the upper two (five panels in one row of 15 cm leave 2 cm each, too narrow for seven curves).
SIZE5 = (("jc2", "jc2_188", "accuracy", "JetClass-II"), ("jc1", "jc1", "accuracy", "JetClass"),
         ("top", "top", "r30", "Top tagging"), ("qg", "qg", "r30", "Quark/gluon, Pythia"),
         ("qg_herwig", "qg_herwig", "r30", "Quark/gluon, Herwig"))
SIZE5_AUC = tuple((key, ds, "macro_auc_ovr" if key in ("jc2", "jc1") else "auc", title) for key, ds, _, title in SIZE5)
SIZE_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))        # the legend in (0, 2)
SIZE_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
SIZE_PARTITIONS = ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")     # 188 and 17 classes for reference
SIZE_W, SIZE_LEFT, SIZE_WGAP, SIZE_HGAP = 3.9, 1.35, 0.85, 1.0      # cm
POOLED = M._pooled(RUNS)


def _curve(ax, series: dict, line: dict, mark: dict, label: str, band: bool, style: str = L.SCAN, k: int = 0,
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
SIZE_PAD = 1.7              # a panel's x axis runs from its smallest training set over SIZE_PAD to its largest times it


def _draw_size(path, series_of, absent: str, auc: bool = False, style: str = L.SCAN) -> None:
    """series_of(key, ds, met): [(series, line, mark, label)] of a panel, in the legend's order; drawn in reverse, so
    the first (188 classes) lies on top. auc: the same five tasks in AUC (SIZE5_AUC) with ParT's published AUCs."""
    spec, published = (SIZE5_AUC, PUBLISHED_AUC) if auc else (SIZE5, PUBLISHED)
    panels = [series_of(key, ds, met) for key, ds, met, _ in spec]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    w, h, top = SIZE_W, 0.9 * SIZE_W, 0.6
    fig = S.figure_cm(S.PRINT_W, top + 2 * h + SIZE_HGAP + 1.0)
    axes, handles = [], {}
    for k, ((i, j), panel, (key, _, met, title)) in enumerate(zip(SIZE_SLOTS, panels, spec)):
        ax = S.axes_cm(fig, SIZE_LEFT + j * (w + SIZE_WGAP), top + i * (h + SIZE_HGAP), w, h)
        axes.append(ax)
        ax.set_xscale("log")
        ns = [n for s_, *_ in panel for n in s_] + ([published[key][0]] if key in published else [])
        if ns:                        # each panel's range from its own points: no empty decade (PI 2026-11-11)
            ax.set_xlim(min(ns) / SIZE_PAD, max(ns) * SIZE_PAD)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, title)
        S.labels(ax, X_JETS if i == 1 else None, Y_OF[met] if j == 0 else None)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        for n_, (series, line, mark, label) in reversed(list(enumerate(panel))):
            _curve(ax, series, line, mark, label, True, style, n_, len(panel))
        if key in published:
            ax.plot(*published[key], color="#000000", marker="*", markersize=STAR, markeredgecolor="white", markeredgewidth=0.5,
                    linestyle="none", label="ParT, published", zorder=10)    # on top of every marker
        if met == "r30":
            _log_y(ax)
        else:
            S.data_ylim(ax)         # the bands and the points clear of the frame (L7)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))     # labels of two decimals
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    labels = [lab for *_, lab in max(panels, key=len)] + ["ParT, published"]
    if "ParT, published" in handles:        # the legend enlarges markers (fig_legend): the star at its plotted size
        handles["ParT, published"] = Line2D([], [], color="#000000", marker="*", markersize=STAR / 1.6, linestyle="none")
    # the legend in the slot right of the upper two panels, one entry a line, centred in the slot
    W, H = S.PRINT_W, fig.get_figheight() / S.CM
    x0 = SIZE_LEFT + 2 * w + SIZE_WGAP + 0.2            # 0.2 cm right of the second panel
    fig.legend(handles=[handles[lab] for lab in labels if lab in handles], labels=[lab for lab in labels if lab in handles],
               loc="center left", bbox_to_anchor=(x0 / W, 1 - (top + h / 2) / H), ncol=1, frameon=False,
               fontsize=S.LEGEND_PT5, borderpad=0.2, labelspacing=0.3, handlelength=S.HANDLE, markerscale=1.0)
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


def _size(path, makers: list, absent: str, auc: bool, style: str = L.SCAN) -> None:
    _draw_size(path, lambda key, ds, met: [m(key, ds, met) for m in makers], absent, auc=auc, style=style)


def draw_f8(path, auc: bool = False) -> None:
    """The 188-class network in two uses, both in the 188 colour: fine-tuned solid, the linear probe beside it dashed
    with open circles (L6); from scratch grey."""
    _size(path, [_tuned("L188", tag=True), _probe("L188", "probe"), _tuned("SCRATCH")],
          "no fine-tuning read-outs or linear probes", auc)


def draw_a10(path, auc: bool = False) -> None:
    """Linear probes of the label sets and the mass networks, and the random network."""
    _size(path, [_probe(r) for r in SIZE_ROWS] + [_probe(M.INIT)], "no linear probes (linprobe/fits_v4)", auc)


def draw_a8(path, auc: bool = False, style: str = L.SCAN) -> None:
    """A10's form after fine-tuning: the label sets, the mass networks and from scratch."""
    _size(path, [_tuned(r) for r in SIZE_ROWS] + [_tuned("SCRATCH")], "no fine-tuning read-outs", auc, style)


def draw_a10_part(path) -> None:
    _size(path, [_probe(r) for r in SIZE_PARTITIONS], "no linear probes (linprobe/fits_v4)", False)


def draw_a8_part(path) -> None:
    _size(path, [_tuned(r) for r in SIZE_PARTITIONS], "no fine-tuning read-outs", False)


# ------------------------------------------------------------------ the UMAP and the pretraining curve
EMBED_NAMES = {"bb": r"$X\to b\bar{b}$", "cc": r"$X\to c\bar{c}$", "qq": r"$X\to q\bar{q}$", "QCD": "QCD",
               "ee": r"$X\to ee$", "mm": r"$X\to \mu\mu$", "top": "top", "gluon": "gluon", "quark": "quark",
               **{c.removeprefix("label_"): S.JC1_TEX[c] for c in S.JC1_CLASSES if c != "label_QCD"}}
EMBED_MODELS = (("mtx-l188-s1", "L188"), ("mtx-r16q1-s1", "R16_Q1"))


UMAP_DIRS = ("umap_v2", "umap", "umap_preview")     # round 3 (all four tasks); the long-budget jc1 run; the preview
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


def _draw_embeddings(path, grid: list, colours=None) -> None:
    """grid: rows of (file stem, title). Round 4: 6.5 cm square panels, the class legend above them at 8.5 pt."""
    from matplotlib.lines import Line2D
    root = _umap_root([stem for row in grid for stem, _ in row])
    docs = [[json.loads((root / f"{stem}.json").read_text()) for stem, _ in row] for row in grid]
    if any(d.get("umap", UMAP_SETTINGS) != UMAP_SETTINGS for row in docs for d in row):
        raise SystemExit(f"FATAL: {root}: UMAP settings other than the caption's {UMAP_SETTINGS}")
    groups = docs[0][0]["groups"]
    ncol = 5 if len(groups) > 5 else len(groups)
    legend_rows_n = math.ceil(len(groups) / ncol)
    fig, axes, _ = S.grid_cm(len(grid), len(grid[0]), aspect=1.0, w=7.0, left=0.35, wgap=0.3, hgap=0.6, bottom=0.3,
                             top=0.95 + 0.5 * legend_rows_n)
    if colours is None:         # QCD black, as in every class figure; the other classes take the colours after it
        rest = iter(S.TAB10[1:])
        colours = [S.TAB10[0] if name == "QCD" else next(rest) for name in groups]
        if len(groups) == 2:    # black and vermillion, which stand apart in small translucent dots
            first = "QCD" if "QCD" in groups else groups[0]
            colours = [S.TAB10[0] if name == first else S.TAB10[2] for name in groups]
    for r, (row, drow) in enumerate(zip(grid, docs)):
        for ax, (_, title), d in zip(axes[r], row, drow):
            _embedding(ax, d, colours)
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


def _umap_page(page: str) -> C.Page:
    _, heading, jets, ft = UMAP_PAGES[page]
    u = UMAP_SETTINGS
    return C.Page(
        id=page,
        title=f"UMAP of the class token, frozen and fine-tuned: {heading} (appendix)",
        precedent=["map2026_poster/fig5.png", "2610.36121/fig3.png"],
        precedent_label="MAP 2026 poster (frozen against fine-tuned), SuperJEPA Fig. 3",
        caption=f"UMAP of the class token of networks pretrained on JetClass-II with 188 classes (upper) and 17 (lower), "
                f"frozen (left) and fine-tuned on {ft} (right), for {jets} in every panel; the plots are qualitative, "
                "without axis scales.",      # PI 2026-11-12: no result sentences; the reading is for the text
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
        "curve is the mean, and its band ±1 standard deviation, over two pretraining runs per flavor-boundary "
        "partition, three for 17 classes, or the five random partitions. The learning rate decays from epoch 57, "
        "where every curve rises slightly."
    ),
    "F8": (
        "Fine-tuning the network pretrained with 188 classes is more accurate than training from scratch on JetClass-II "
        "and JetClass at every training-set size. On top tagging its mean rejection is higher at 1,000 and 100,000 jets "
        "(R30 127 ± 21 against 49 ± 12, and 850 ± 90 against 560 ± 90), differences within four standard deviations of "
        "the spread over three runs. The panels show accuracy or R30, the inverse background efficiency at 30% signal "
        "efficiency, against the number of training jets, for that network fine-tuned (solid) or as a linear probe on "
        "its frozen features (dashed) and for training from scratch (grey). Points are means and bands ±1 standard "
        "deviation over three pretraining runs or three trainings from scratch, stars mark the published Particle "
        "Transformer (ParT), and the values are in tables {tab:ftfull} and {tab:probesnfull}."
    ),
    "F8-auc": (
        "The panels of figure {fig:F8} in the area under the ROC curve (AUC), averaged over the classes, each against "
        "the rest, on class-balanced JetClass-II and JetClass, and of the binary task on top and quark/gluon tagging, "
        "scored on Pythia and on Herwig jets, against the number of training jets. The network pretrained with 188 "
        "classes is fine-tuned (solid) or a linear probe on its frozen features (dashed), and training from scratch is "
        "grey. Points are means and bands ±1 standard deviation over three pretraining runs or three trainings from "
        "scratch, and stars mark the published Particle Transformer (ParT). Values are in tables {tab:ftfull} and "
        "{tab:probesnfull}."
    ),
    "A8-auc": (
        "The panels of figure {fig:A8} in the area under the ROC curve (AUC), averaged over the classes, each against "
        "the rest, on class-balanced JetClass-II and JetClass, and of the binary task on top and quark/gluon tagging, "
        "scored on Pythia and on Herwig jets, for the fine-tuned networks of each label set, with and without the mass "
        "regression (+ mass), and training from scratch (grey). Points are means and bands ±1 standard deviation over "
        "three pretraining runs, two for + mass, or three trainings from scratch, and stars mark the published "
        "Particle Transformer (ParT). JetClass-II and JetClass start at 10,000 jets, and the values are in table "
        "{tab:ftfull}."
    ),
    "A10-auc": (
        "The panels of figure {fig:A10} in the area under the ROC curve (AUC), averaged over the classes, each against "
        "the rest, on class-balanced JetClass-II and JetClass, and of the binary task on top and quark/gluon tagging, "
        "scored on Pythia and on Herwig jets, for linear probes on the frozen features of each label set, with and "
        "without the mass regression (+ mass), and of the random network (dark grey). Points are means and bands ±1 "
        "standard deviation over three pretraining runs, two for + mass, or three random initializations, and stars "
        "mark the published Particle Transformer (ParT). Values are in table {tab:probesnfull}."
    ),
    "A8-part": (
        "The networks pretrained with the four 17-class partitions in the form of figure {fig:A8}, fine-tuned and "
        "scored on class-balanced JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging, Pythia and "
        "Herwig (R30), against the number of training jets, with 188 and 17 classes for reference. The partitions were "
        "fine-tuned on JetClass-II at one million jets only, so that panel shows one point for each. Points are means "
        "and bands ±1 standard deviation over two pretraining runs for each flavor-boundary partition, the five random "
        "partitions with one run each, and three runs for 188 and 17 classes. Stars mark the published Particle "
        "Transformer (ParT), and the values are in tables {tab:ft} and {tab:summary}."
    ),
    "A10-part": (
        "The networks pretrained with the four 17-class partitions in the form of figure {fig:A10}, as linear probes on "
        "their frozen features, trained and scored on class-balanced JetClass-II and JetClass (accuracy) and on top and "
        "quark/gluon tagging, Pythia and Herwig (R30), against the number of training jets, with 188 and 17 classes for "
        "reference. Points are means and bands ±1 standard deviation over two pretraining runs for each "
        "flavor-boundary partition, the five random partitions with one run each, and three runs for 188 and 17 "
        "classes. Stars mark the published Particle Transformer (ParT), and the values are in tables {tab:probesn} "
        "and {tab:summary}."
    ),
}
PAGES = [
    C.Page(
        id="F2",
        title="Label granularity, linear probes and fine-tuned",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with the mass regression, and "
                "with 17-class partitions, as linear probes on frozen features (upper row, beside the random network) "
                "and fine-tuned (lower row, beside training from scratch), at each task's largest training set: "
                "accuracy on class-balanced JetClass-II and on JetClass, R30 on top and quark/gluon tagging, Pythia "
                "and Herwig. " + F2_KEY + "Values: " + G.number("tab:summary") + ".",
        differs="Hong plots accuracy against the number of classes; here the label set is a category (PI 2026-11-11), "
                "one row a network, on the metric itself (PI 2026-11-12: actual values only), mean and SD; the two "
                "modes as two rows of five tasks, headline metric only (PI 2026-11-13, L1).",
        draw=draw_f2,
        data_note=f"{HV_NOTE}{SUBST_NOTE} Herwig: the quark/gluon probes and networks scored on Herwig jets."),
    C.Page(
        id="F2a",
        title="Label granularity, linear probes, with AUC",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Linear probes on the frozen features of the networks of Fig. 2, the random network and the released "
                "Sophon checkpoint, at each task's largest training set: upper row the headline metric (accuracy, "
                "R30), lower row the AUC. " + F2_KEY + "Values: " + G.number("tab:summary") + " and "
                + G.number("tab:summaryauc") + ".",
        differs="As F2; one mode, its headline metric over its AUC.",
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
        differs="As F2; one mode, its headline metric over its AUC.",
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
