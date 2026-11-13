#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2a, F2b, F3-main, F3-app, F3-ft, F8, A8, A10, UMAP-jc1, UMAP-jc2-had, UMAP-jc2-lep, UMAP-top, UMAP-qg, VAL (docs/figstyle/PLAN.md), in
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
from gallery_roc import FAMILIES, family_label, family_of
import lines_v6 as L
import style_v5 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I

GRID_2X2 = {"left": 0.1, "right": 0.96, "top": 0.92, "bottom": 0.06, "wspace": 0.3, "hspace": 0.2}     # room for repeated y labels, tick labels and titles
HEIGHT_HEAT = 8.0                                 # 188 rows; a 5-row group label at 10 pt still clears its neighbours
LEFT_HEAT = 0.26                                  # the longest group label at 10 pt, so the saved width stays 15 cm
MARKER = 4
X_JETS = "Number of training jets"
LEGEND_PANEL = 1        # JetClass: its curves stay high on the right, so its lower right is empty in F2, F8 and A10
TITLES = {key: title for key, _, _, _, title, _ in P.F2_PANELS}
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


# ------------------------------------------------------------------ F2: metric against the number of classes
# Figs. 2 and 3 (PI 2026-11-12: actual values only): the label set is a category, so a horizontal dot plot, one row a
# network, on the metric itself. Each row: the mean over its runs a filled circle in the network's colour, a horizontal
# bar its SD (PI 2026-11-12: no dots for single runs; the five random partitions: mean and SD over the five). The 188-class mean is a vertical line with a light band of its SD
# down the panel, and the first row. From scratch is the last row of Fig. 3 only. Nothing printed, nothing clipped.
F2_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1")
F2_CONTROLS = ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")      # rows below the first divider, those the figure shows
F2_NAMES = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
            "RAND": "random partitions", "FLAV_F0": "no flavor boundary", "FLAV_F1": "$b$-quark boundary",
            "FLAV_F1R": "random flavor boundary", "SCRATCH": S.LEGEND["SCRATCH"]}
F2_LEFT, F2_W, F2_H, F2_WGAP, F2_HGAP = 4.2, 5.0, 5.0, 0.45, 1.35
X_F2 = {"accuracy": "Accuracy", "r30": "Rejection ($R_{30}$)"}
MEAN_DOT = 4.6                   # pt: the mean over runs


def _range(ax, ys: list, log: bool, pad: float = 0.12) -> tuple:
    """y limits from the plotted series only, padded by pad of the span (in log space on a log axis)."""
    lo, hi = min(ys), max(ys)
    if log:
        a, b = math.log10(lo), math.log10(hi)
        d = max(b - a, 0.05) * pad
        return 10 ** (a - d), 10 ** (b + d)
    d = max(hi - lo, 1e-3) * pad
    return lo - d, hi + d


def _f2_rows(groups, scratch: bool) -> dict:
    """{row key: y}: the label sets from 0, the controls the figure shows after a gap, from scratch after another."""
    shown = [k for k in F2_CONTROLS if any(key == k for _, key, _ in groups)]
    rows = {**{k: float(i) for i, k in enumerate(F2_ROWS)}, **{k: 4.5 + i for i, k in enumerate(shown)}}
    return {**rows, "SCRATCH": max(rows.values()) + 1.5} if scratch else rows


def _f2_bracket(fig, ax, ys: list) -> None:
    """ "17 classes" on the left of the control rows' labels: a bracket spanning them, the words turned along it."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    x = 0.6 / S.PRINT_W                   # 0.6 cm from the figure's left edge, left of the longest row label
    a, b = min(ys) - 0.4, max(ys) + 0.4
    tick = 0.12 / S.PRINT_W
    fig.add_artist(plt.Line2D([x + tick, x, x, x + tick], [a, a, b, b], transform=tr, color="#000000",
                              linewidth=S.LW_THIN))
    fig.text(x - 0.05 / S.PRINT_W, (a + b) / 2, "17 classes", transform=tr, rotation=90, ha="right", va="center",
             fontsize=S.TICK_PT5)


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


def _draw_f2(path, value_of, groups, absent: str, heading: str, scratch: bool) -> None:
    """value_of(key, n, ds, met): a run's value or None for the task; groups: (legend label, style key, rows) of the
    17-class controls. A 2x2 grid of near-square panels, the row labels on the left panels."""
    from matplotlib.patches import Patch
    values = [value_of(key, n, ds, met) for key, n, ds, met, *_ in P.F2_PANELS]
    if not any(values):
        raise C.Absent(absent)
    rows = _f2_rows(groups, scratch)
    controls = [y for k, y in rows.items() if k in F2_CONTROLS]
    top, bottom = 1.75, 1.0
    fig = S.figure_cm(S.PRINT_W, top + 2 * F2_H + F2_HGAP + bottom)
    axes = [S.axes_cm(fig, F2_LEFT + j * (F2_W + F2_WGAP), top + i * (F2_H + F2_HGAP), F2_W, F2_H)
            for i in range(2) for j in range(2)]
    hidden = bars = 0
    for k, (ax, value, (key, _, ds, met, *_)) in enumerate(zip(axes, values, P.F2_PANELS)):
        S.title(ax, TITLES[key])
        ax.set_ylim(max(rows.values()) + 0.6, -0.6)
        ax.yaxis.set_major_locator(ticker.FixedLocator(list(rows.values())))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter([F2_NAMES[r] if k % 2 == 0 else "" for r in rows]))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="y", length=0)
        for y in [3.75] + ([rows["SCRATCH"] - 0.75] if scratch else []):
            ax.axhline(y, color=S.GREY_MID, linewidth=S.LW_THIN, zorder=0.5)
        S.labels(ax, X_F2[met], None)
        if value is None:
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        xs = []
        ref = P._vals(RUNS, "L188", value)
        if ref:
            m, sd = float(np.mean(ref)), float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
            ax.axvspan(m - sd, m + sd, color=S.SERIES["L188"], alpha=L.BAND, linewidth=0, zorder=0.8)
            ax.axvline(m, color=S.SERIES["L188"], linewidth=S.LW_THIN, zorder=0.9)
        for r in F2_ROWS:
            if v := P._vals(RUNS, r, value):
                xs += _f2_row(ax, rows[r], v, S.SERIES[r])
        for _, key_, rws in groups:
            if v := [x for r in rws for x in P._vals(RUNS, r, value)]:
                xs += _f2_row(ax, rows[key_], v, S.SERIES[key_])
        if scratch and (sc := P.scratch_series(M, I["per"], key, met)):
            xs += _f2_row(ax, rows["SCRATCH"], sc[max(sc)], S.look("SCRATCH")["color"])
        ax.set_xlim(*_range(ax, xs, False, pad=0.08))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        per_pt = F2_W / 2.54 * 72 / abs(np.diff(ax.get_xlim())[0])          # points per data unit
        hidden += sum(c.get_segments()[0][1][0] - c.get_segments()[0][0][0] <= (MEAN_DOT + 0.5) / per_pt
                      for c in ax.collections if hasattr(c, "get_segments"))
        bars += sum(hasattr(c, "get_segments") for c in ax.collections)
    for i in (0, 2):
        _f2_bracket(fig, axes[i], controls)
    S.letter_panels(axes)
    S.heading(fig, heading)
    grey = "#000000"
    handles = [Line2D([], [], color=grey, linewidth=L.BAR, marker="o", markersize=MEAN_DOT, markerfacecolor=grey,
                      markeredgecolor="white", markeredgewidth=0.5),
               (Patch(facecolor=S.SERIES["L188"], alpha=L.BAND, linewidth=0),
                Line2D([], [], color=S.SERIES["L188"], linewidth=S.LW_THIN))]
    labels = ["mean and SD over pretraining runs", "188 classes, mean and SD"]
    S.fig_legend(fig, handles, top_cm=0.6, ncol=2, labels=labels)
    print(f"{heading}: {hidden} of {bars} SD bars are shorter than the mean's circle")
    S.save(fig, path)


def draw_f2a(path) -> None:
    _draw_f2(path, _probe_value, _controlled() + [(S.LEGEND["RAND"], "RAND", M.RAND)],
             "no linear probes (linprobe/fits_v4)", "Linear probes on frozen features", scratch=False)


def _probe_value(key, n, ds, met):
    return P._f2_probe(M, I["fits"], I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    return P._f2_ft(M, I["per"], key, n, met)


def _controlled():
    return [(S.LEGEND[a], a, [a]) for a in S.CONTROLS]


def draw_f2b(path) -> None:
    _draw_f2(path, _ft_value, _controlled() + [(S.LEGEND["RAND"], "RAND", M.RAND)],
             "no fine-tuning read-outs (ft_mlst, budget)", "After fine-tuning",
             scratch=True)


# ------------------------------------------------------------------ F3-main, F3-app: every class against the 188-class model
def _class_auc(models: list) -> dict:
    """{class: AUC, the mean over the models that hold it}."""
    per = [p for m in models if (p := M._per_class(I["fits"], m))]
    return {c: float(np.mean([p[c] for p in per if c in p])) for c in set().union(*per)} if per else {}


def _ft_class_auc(models: list) -> dict:
    """{class: AUC} after fine-tuning on 1e6 balanced JetClass-II jets, the mean over the models that hold it
    (curves_v2/jc2_ft_ovr)."""
    docs = C.ROC.get("jc2_ft_ovr") or {}
    per = [{int(c): v["auc"] for c, v in docs[m]["curves"].items()} for m in models if m in docs]
    return {c: float(np.mean([p[c] for p in per if c in p])) for c in set().union(*per)} if per else {}


def _draw_scatter(path, panels: list, auc_of=_class_auc, word: str = "Linear probes", letters: str = "abcd", scope=None,
                  absent: str = "no class-balanced JetClass-II linear probes (linprobe/fits_v4)",
                  ylabel: str = "AUC, the panel's network", step: float | None = None) -> None:
    """panels: (runs, title) or (runs, title, auc_of), two to four; a panel's own auc_of (F3-main: probe | fine-tuned)
    reads both its model and the 188-class one. One point per class; below the dotted line the class does worse than
    with 188 classes. Square panels, two per row at 6.5 cm, three in one row at 4.2 cm (PI 2026-11-11: no empty slot);
    the legend above the panels; letters names the panels (a second figure continues the first's letters)."""
    from matplotlib.lines import Line2D
    of = lambda panel: panel[2] if len(panel) > 2 else auc_of
    if not all(of(p)(RUNS["L188"]) for p in panels):
        raise C.Absent(absent)
    family = {int(r["jet_label"]): family_of(r["class_name"]) for r in M._class_rows(P.REPO)}
    ncols = 3 if len(panels) == 3 else 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes, _ = S.grid_cm(nrows, ncols, w=S.PANEL_3 if ncols == 3 else S.PANEL_W, aspect=1.0, top=2.65, left=1.3, wgap=0.4)
    cells = list(axes.flat)
    for ax in cells[1:]:
        ax.sharex(cells[0])
        ax.sharey(cells[0])
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)      # one y range: its tick labels once, on the left
    def scored(panel) -> list:
        full, other = of(panel)(RUNS["L188"]), of(panel)(panel[0])
        return [(full[c], other[c], c) for c in full if c in other]
    pairs = [scored(p) for p in panels]
    # one range for the two figures of a split (scope: every panel of both), so their axes match
    ranged = pairs if scope is None else [scored(p) for p in scope]
    lo = min(v for pts in ranged for x, y, *_ in pts for v in (x, y))
    lim = (lo - 0.04 * (1 - lo), 1.0 + 0.04 * (1 - lo))     # the lowest point and the points at AUC 1 clear of the frame
    for k, (ax, pts, (_, title, *_)) in enumerate(zip(cells, pairs, panels)):
        S.fine_grid(ax)
        if not pts:
            ax.text(0.5, 0.5, "not yet fine-tuned", transform=ax.transAxes, ha="center", va="center", fontsize=S.LEGEND_PT5)
        ax.plot(lim, lim, **S.REFERENCE)
        for i, name in enumerate(FAMILIES):
            xy = [(x, y) for x, y, c in pts if family[c] == name]
            if xy:
                ax.plot(*zip(*xy), color=S.TAB10[i], marker="o", markersize=3, linestyle="none", clip_on=False,
                        zorder=3.1 if len(xy) < 10 else 3)    # over the frame; a small family over the large ones
        S.title(ax, f"({letters[k]}) {title}")
        if k % ncols == 0:
            S.labels(ax, ylabel=ylabel)
        if k + ncols >= len(panels):
            S.labels(ax, xlabel="AUC, 188 classes")
    cells[0].set_xlim(lim)
    cells[0].set_ylim(lim)
    for axis in (cells[0].xaxis, cells[0].yaxis):    # one tick step on both axes of every panel
        axis.set_major_locator(ticker.MultipleLocator(step) if step else ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    handles = [Line2D([], [], color=S.TAB10[i], marker="o", markersize=3, linestyle="none", label=family_label(f))
               for i, f in enumerate(FAMILIES)] + [Line2D([], [], **S.REFERENCE, label="same as 188 classes")]
    S.fig_legend(fig, handles, top_cm=0.6, ncol=3)
    S.heading(fig, f"{word} on $10^6$ JetClass-II jets, class by class")
    S.save(fig, path)


def _app_panels() -> list:
    """The probe pages in F3-ft's order (PI, 2026-11-07): 162, 43 and 17 classes and the random partitions, then the
    three flavor boundaries as F3-ft2; the two mass-regression models, which have no fine-tuned page, on F3-app3."""
    return _ft_panels()


def _mass_panels() -> list:
    return [(RUNS[a], f"{S.NAME[b]} + mass") for a, b in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))]


def _ft_panels() -> list:
    return ([(RUNS[a], S.NAME[a]) for a in ("L162", "R42_Q1", "R16_Q1")]
            + [([m for r in M.RAND for m in RUNS[r]], "Five random partitions")] + [(RUNS[a], S.NAME[a]) for a in S.CONTROLS])


FT_ABSENT = "no per-class curves after fine-tuning (curves_v2/jc2_ft_ovr)"


def draw_f3_main(path) -> None:
    """17 against 188 classes, linear probe | fine-tuned (PI, 2026-11-08), on one axis range, so the shrink after
    fine-tuning shows."""
    _draw_scatter(path, [(RUNS["R16_Q1"], f"{S.LEGEND['R16_Q1']} {S.PROBE}", _class_auc),
                         (RUNS["R16_Q1"], f"{S.LEGEND['R16_Q1']} {S.FINE_TUNED}", _ft_class_auc)],
                  word="Linear probes (LP) and fine-tuned (f.t.)", ylabel=f"AUC, {S.LEGEND['R16_Q1']}",
                  step=0.02)     # PI 2026-11-12: a label every 0.02 on both axes


def draw_f3_app(path) -> None:
    _draw_scatter(path, _app_panels()[:4], scope=_app_panels())


def draw_f3_app2(path) -> None:
    _draw_scatter(path, _app_panels()[4:], letters="efg", scope=_app_panels())


def draw_f3_app3(path) -> None:
    _draw_scatter(path, _mass_panels())


def draw_f3_ft(path) -> None:
    """F3-main's form after fine-tuning (curves_v2/jc2_ft_ovr): every model fine-tuned on JetClass-II."""
    _draw_scatter(path, _ft_panels()[:4], auc_of=_ft_class_auc, word="Fine-tuned", absent=FT_ABSENT, scope=_ft_panels())


def draw_f3_ft2(path) -> None:
    _draw_scatter(path, _ft_panels()[4:], auc_of=_ft_class_auc, word="Fine-tuned", absent=FT_ABSENT, letters="efg",
                  scope=_ft_panels())


# ------------------------------------------------------------------ F8, A10: metric against the number of jets
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


def _draw_size(path, series_of, band: bool, absent: str, heading: str, ncol: int = 3, auc: bool = False,
               style: str = L.SCAN) -> None:
    """series_of(key, ds, met): [(series, line, mark, label)] of a panel. Round 4: four 6.5 cm panels, the legend above
    in ncol columns. auc: the same four tasks in AUC (P.SIZE_PANELS_AUC) with ParT's published AUCs."""
    spec, published = (P.SIZE_PANELS_AUC, PUBLISHED_AUC) if auc else (P.SIZE_PANELS, PUBLISHED)
    panels = [series_of(key, ds, met) for key, ds, met, *_ in spec]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    rows = math.ceil((max(len(p) for p in panels) + 1) / ncol)       # the series and the published star
    fig, axes, _ = S.grid_cm(2, 2, w=6.35, aspect=0.85, top=1.4 + 0.42 * rows, left=1.4, wgap=0.8)
    for k, (ax, panel, (key, _, met, *_)) in enumerate(zip(axes.flat, panels, spec)):
        ax.set_xscale("log")
        ns = [n for s_, *_ in panel for n in s_] + ([published[key][0]] if key in published else [])
        if ns:                        # each panel's range from its own points: no empty decade (PI 2026-11-11)
            ax.set_xlim(min(ns) / SIZE_PAD, max(ns) * SIZE_PAD)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, TITLES[key])
        S.labels(ax, X_JETS if k >= 2 else None, Y_OF[met] if k % 2 == 0 else None)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        for i, (series, line, mark, label) in enumerate(panel):
            _curve(ax, series, line, mark, label, band, style, i, len(panel))
        if key in published:
            ax.plot(*published[key], color="#000000", marker="*", markersize=STAR, markeredgecolor="white", markeredgewidth=0.5,
                    linestyle="none", label="ParT, published", zorder=10)    # on top of every marker
        if met == "r30":
            _log_y(ax)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    handles, labels = {}, {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    if "ParT, published" in handles:        # the legend enlarges markers (fig_legend): the star at its plotted size
        handles["ParT, published"] = Line2D([], [], color="#000000", marker="*", markersize=STAR / 1.6, linestyle="none",
                                            label="ParT, published")
    S.fig_legend(fig, list(handles.values()), top_cm=0.6, ncol=ncol)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.save(fig, path)


def draw_f8(path, auc: bool = False) -> None:
    """One model in three modes: fine-tuned in the 188 colour, the linear probe in its 40% tint (style_v5.TINT_188,
    used only here), from scratch dark grey."""
    chosen = _chosen()
    probe_line = {"color": S.TINT_188, "linestyle": S.MODE_LINE["probe"]}
    probe_mark = {"marker": "s", "markeredgecolor": S.TINT_188, "markerfacecolor": "white"}
    series = lambda key, ds, met: [_model(P.ft_series(M, I["per"], RUNS, key, met, "L188"), "L188", "ft",
                                          f"{S.LEGEND['L188']} {S.FINE_TUNED}"),
                                   (P.probe_series(M, I, RUNS, ds, met, "L188", chosen), probe_line, probe_mark,
                                    f"{S.LEGEND['L188']} {S.PROBE}"),
                                   _model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                          S.LEGEND['SCRATCH'])]
    _draw_size(path, series, True, "no fine-tuning read-outs or linear probes", "188-class pretraining against training from scratch"
               + (", AUC" if auc else ""), ncol=2, auc=auc)


def draw_a10(path, auc: bool = False) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [_model(P.probe_series(M, I, RUNS, ds, met, row, chosen), row, "ft", S.LEGEND[row])   # probes only: solid
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  S.LEGEND['SCRATCH'])]
    _draw_size(path, series, True, "no linear probes (linprobe/fits_v4)", "Linear probes on frozen features"
               + (", AUC" if auc else ""), auc=auc)


def draw_a8(path, auc: bool = False, style: str = L.SCAN) -> None:
    """A10's form after fine-tuning: the four label sets and from scratch."""
    series = lambda key, ds, met: [_model(P.ft_series(M, I["per"], RUNS, key, met, row), row, "ft", S.LEGEND[row])
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  S.LEGEND["SCRATCH"])]
    _draw_size(path, series, True, "no fine-tuning read-outs", "Fine-tuned" + (", AUC" if auc else ""), auc=auc,
               style=style)


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


def _draw_embeddings(path, grid: list, heading: str, colours=None, legend_rows=None) -> None:
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
                             top=1.4 + 0.5 * legend_rows_n)
    if colours is None:         # QCD black, as in every class figure; the other classes take the colours after it
        rest = iter(S.TAB10[1:])
        colours = [S.TAB10[0] if name == "QCD" else next(rest) for name in groups]
        if len(groups) == 2:    # black and rose, which stand apart in small translucent dots where indigo does not
            first = "QCD" if "QCD" in groups else groups[0]
            colours = [S.TAB10[0] if name == first else S.TAB10[5] for name in groups]
    for r, (row, drow) in enumerate(zip(grid, docs)):
        for ax, (_, title), d in zip(axes[r], row, drow):
            _embedding(ax, d, colours)
            S.title(ax, title)
    handles = [Line2D([], [], color=colours[k], marker="o", markersize=5, linestyle="none",
                      label=EMBED_NAMES[name]) for k, name in enumerate(groups)]
    S.fig_legend(fig, handles, top_cm=0.6, ncol=ncol)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.save(fig, path)


def _draw_umap(path, page: str) -> None:
    """Rows: 188 and 17 classes; columns: frozen, fine-tuned on the task; the same jets in all four panels."""
    name, heading = UMAP_PAGES[page][:2]
    heading = f"Class token: {heading}"
    grid = [[(f"{name}_{m}_{kind}", f"{S.NAME[level]}, {word}") for kind, word in (("frozen", "frozen"), ("ft", "fine-tuned"))]
            for m, level in EMBED_MODELS]
    colours = [S.JC1_COLOUR[c] for c in S.JC1_CLASSES] if name == "jc1" else None
    _draw_embeddings(path, grid, heading, colours=colours, legend_rows=(0,))


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




def draw_val(path) -> None:
    """Exact size, one panel SINGLE_W wide (PI 2026-11-12), the legend above it."""
    docs = M._load(P.DATA / "compute")
    w = S.SINGLE_W - 1.5 - 0.25
    fig = S.figure_cm(S.SINGLE_W, 1.75 + 0.85 * w + 1.0)
    ax = S.axes_cm(fig, 1.5, 1.75, w, 0.85 * w)
    S.fine_grid(ax)
    for level in S.LEVELS:
        have = [[e["val_acc"] for e in docs[m]["epochs"]] for m in RUNS[level] if m in docs and "val_acc" in docs[m]["epochs"][0]]
        if not have:
            raise C.Absent("no validation accuracy per epoch (compute/)")
        ax.plot(range(1, len(have[0]) + 1), np.mean(have, axis=0), **S.look(level), linewidth=S.LW_CURVE, label=S.LEGEND[level])
    S.heading(fig, "Pretraining on JetClass-II")
    S.labels(ax, "Pretraining epoch", "Validation accuracy")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=0.6, ncol=2)     # above the panel, 188 first, as every figure
    S.save(fig, path)


# ------------------------------------------------------------------ the pages
HONG = ["2410.23129/fig2.png"]
HEATMAPS = ["2604.09809/fig7.png", "2606.14813/fig12.png"]
SUBST_NOTE = ("Top-tagging linear probes fill the missing inputs with random draws and quark/gluon probes with the "
              "training mean: the substitution chosen for each task on validation, over all 30 models.")
TWO_RUNS = "The three controlled partitions have two runs each, the four label sets three."

F2_KEY = ("Circles: mean; bars: ±1 SD, hidden where shorter than the circle; three pretraining runs per network; "
          "two for the flavor-boundary partitions; five partitions for random partitions. "
          "Line and band: the 188-class mean and SD. ")
PAGES = [
    C.Page(
        id="F2a",
        title="Label granularity, linear probes",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 "
                "classes and with 17-class partitions, at each task's largest training set: accuracy on class-balanced "
                "JetClass-II and on JetClass, R30 on top and quark/gluon tagging. " + F2_KEY
                + "Values: " + G.number("tab:summary") + ".",
        differs="Hong plots accuracy against the number of classes; here the label set is a category (PI 2026-11-11), "
                "one row a network, on the metric itself (PI 2026-11-12: actual values only), mean and SD.",
        draw=draw_f2a,
        data_note=f"{TWO_RUNS} Each plain random partition has one run. {SUBST_NOTE}"),
    C.Page(
        id="F2b",
        title="Label granularity, after fine-tuning",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and with 17-class "
                "partitions, fine-tuned, and a network trained from scratch, on a million class-balanced "
                "JetClass-II and a million JetClass jets (accuracy) and the full top and quark/gluon sets "
                "(R30). " + F2_KEY + "Values: "
                + G.number("tab:summary") + ".",
        differs="As F2a.",
        draw=draw_f2b,
        data_note="JetClass-II at 1,000,000 class-balanced jets. JetClass at 1,000,000 jets, top at 1,200,000, quark/gluon at 1,600,000. "
                  f"{TWO_RUNS} Top: every difference between label sets lies within the run-to-run SD (56 to 151); the "
                  "random-boundary marker lies 133 to 188 below the label sets. Quark/gluon: differences between label "
                  "sets are comparable to the run spread (SD 0.8 to 2.2, up to 1.7 between 43 and 17 classes); the "
                  "no-flavor-boundary marker lies 2.4 and 2.7 above the b-quark and random-boundary markers, more than "
                  "any label-set SD."),
    C.Page(
        id="F3-main",
        title="Cost class by class, frozen and fine-tuned",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the dropped heatmap's",
        caption="Networks pretrained on JetClass-II with 17 and with 188 classes, then (a) a linear probe (LP) on "
                "their frozen features or (b) the whole network fine-tuned (f.t.), on one million class-balanced "
                "JetClass-II jets: each class's area under the curve with 17 classes against the same with 188. "
                "Both panels share one range; dotted: equal AUC. Colors are decay families; means over three "
                "pretraining runs.",
        differs="F3-scatter's (b) and F3-ft's (c) side by side (PI, 2026-11-08), on one axis range set by both panels, "
                "so the shrink after fine-tuning shows; replaces the heatmap F3 and F3-scatter in the main text.",
        draw=draw_f3_main),
    C.Page(
        id="F3-app",
        title="Cost class by class, other models, first part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II, trained on one "
                "million class-balanced JetClass-II jets: each class's area under the curve with (a) 162, (b) 43 and "
                "(c) 17 pretraining classes and (d) the five 17-class random partitions averaged, against the same "
                "with 188 classes. Below the dotted line the class does worse than with 188. Colors are decay "
                "families; means over three pretraining runs, in (d) over the five partitions, one run each.",
        differs="As F3-main. Round 4: seven panels split 4+3 over two figures, so every panel is a 6.5 cm square.",
        draw=draw_f3_app),
    C.Page(
        id="F3-app2",
        title="Cost class by class, other models, second part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Continued from the previous figure: linear probes on the frozen features of networks pretrained "
                "on JetClass-II with 17-class partitions with (e) no flavor boundary, (f) a b-quark boundary and "
                "(g) a random flavor boundary, trained on one million class-balanced JetClass-II jets; each "
                "class's area under the curve against the same with 188 classes. Colors are decay families; "
                "values are means over two pretraining runs.",
        differs="The second part of F3-app; the legend takes the empty fourth cell.",
        draw=draw_f3_app2),
    C.Page(
        id="F3-app3",
        title="Cost class by class, mass-regression models (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with (a) 162 and (b) "
                "17 classes plus an auxiliary mass regression, trained on one million class-balanced JetClass-II "
                "jets: each class's area under the curve against the same with 188 classes and no mass "
                "regression. Below the dotted line the class does worse than with 188. Colors are decay families; "
                "values are means over two pretraining runs.",
        differs="F3-main's form for the two mass-regression models, moved out of F3-app so that the probe and "
                "fine-tuning pages share one panel order (PI, 2026-11-07).",
        draw=draw_f3_app3),
    C.Page(
        id="F3-ft",
        title="Cost class by class after fine-tuning, first part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Networks pretrained on JetClass-II, fine-tuned on one million class-balanced JetClass-II jets: "
                "each class's area under the curve with (a) 162, (b) 43 and (c) 17 pretraining classes and (d) "
                "the five 17-class random partitions averaged, against the same with 188 classes. Below the "
                "dotted line the class does worse than with 188. Colors are decay families; means over three "
                "pretraining runs, in (d) over the five partitions, one run each.",
        differs="F3-main's form after fine-tuning (PI). Round 4: seven panels split 4+3 over two figures.",
        draw=draw_f3_ft),
    C.Page(
        id="F3-ft2",
        title="Cost class by class after fine-tuning, second part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Continued from the previous figure: networks pretrained on JetClass-II with 17-class partitions "
                "with (e) no flavor boundary, (f) a b-quark boundary and (g) a random flavor boundary, fine-tuned "
                "on one million class-balanced JetClass-II jets; each class's area under the curve against the "
                "same with 188 classes. Colors are decay families; values are means over two pretraining runs.",
        differs="The second part of F3-ft; the legend takes the empty fourth cell.",
        draw=draw_f3_ft2),
    C.Page(
        id="F8",
        title="Pretraining against the number of training jets",
        precedent=["2401.13537/fig3.png", "2401.13536/fig1.png"],
        precedent_label="Golling Fig. 3, Vigl Fig. 1 top",
        caption="The network pretrained on JetClass-II with 188 classes, fine-tuned (f.t.) and as a linear probe "
                "on its frozen features (LP), and a network trained from scratch, "
                "scored on class-balanced JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging "
                "(R30), against the number of training jets. Points are means, bands the SD over three "
                "pretraining runs (from scratch: three trainings). Stars: published ParT (JetClass: same inputs; "
                "top: kinematic; quark/gluon: kinematic and particle type).",
        differs="Four panels, one per task, against Golling's one. Fine-tuned solid and the linear probe dash-dot, both in the "
                "188-class colour (one model, two uses), and from scratch dark grey dashed, against Golling's green, "
                "red and blue. Sophon's font; round 3's faint grid at major y ticks, where Golling has none; no legend "
                "frame, as Golling. Lines 1.1 pt against about 0.75. Log y on R30. Band as Golling: "
                "same colour, alpha 0.2, no edge. Against Vigl Fig. 1: no ratio panel, no data-efficiency side panel, no dashed variant, no "
                "second legend.",
        draw=draw_f8,
        data_note="Quark/gluon fine-tuned: the pretrained runs have every size; from scratch has the full set only, its "
                  "cells at 1,000 to 100,000 jets are finishing in the current job and join when read out. On JetClass-II and JetClass the plan's sizes drop 1,000 jets for "
                  "fine-tuning and from scratch, so those curves start at 10,000 by design. Sizes below the full set use the fixed step budget with early "
                  "stopping; the full JetClass, top and quark/gluon points use the full-set recipe. JetClass-II at "
                  "1,000,000: all three from-scratch runs and 3 of 16 pretrained runs reached the 40-epoch limit, so "
                  f"from scratch is probably still improving there. {SUBST_NOTE}"),
    C.Page(
        id="F8-auc",
        title="Pretraining against the number of training jets, AUC (appendix)",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="The network pretrained on JetClass-II with 188 classes, fine-tuned (f.t.) and as a linear probe "
                "on its frozen features (LP), and a network trained from scratch: "
                "the AUC averaged over the classes on class-balanced JetClass-II and JetClass "
                "and the AUC of top and quark/gluon tagging, "
                "against the number of training jets. Points are means, bands the SD over three pretraining runs (from "
                "scratch: three trainings). Stars: published ParT (JetClass: same inputs; top: kinematic; "
                "quark/gluon: kinematic and particle type).",
        differs="F8's form in AUC (PI 2026-11-11: the scaling figures need AUC as well as accuracy and rejection).",
        draw=lambda path: draw_f8(path, auc=True)),
    C.Page(
        id="A8",
        title="Fine-tuned against the number of training jets (appendix)",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Fine-tuned networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, trained and "
                "scored on class-balanced JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging "
                "(R30), against the number of training jets; dashed, trained from scratch. Points "
                "are means, bands the SD over three pretraining runs (from scratch: three trainings). Stars: "
                "published ParT (JetClass: same inputs; top: kinematic; quark/gluon: kinematic and particle "
                "type).",
        differs="A10's form, fine-tuned (PI: the task x model x mode grid).",
        draw=draw_a8,
        data_note="From the fine-tuning read-outs as they stand: the long-budget cells at 10^6 on JetClass and "
                  "JetClass-II, and the small-size benchmark cells still running, replace their points when read out."),
    C.Page(
        id="A8-auc",
        title="Fine-tuned against the number of training jets, AUC (appendix)",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Fine-tuned networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, and (dashed) a "
                "network trained from scratch: "
                "the AUC averaged over the classes on class-balanced JetClass-II and JetClass "
                "and the AUC of top and quark/gluon tagging, "
                "against the number of training jets. Points are means, bands the SD over three pretraining runs (from "
                "scratch: three trainings). Stars: published ParT (JetClass: same inputs; top: kinematic; "
                "quark/gluon: kinematic and particle type).",
        differs="A8's form in AUC (PI 2026-11-11: the scaling figures need AUC as well as accuracy and rejection).",
        draw=lambda path: draw_a8(path, auc=True)),
    C.Page(
        id="A10",
        title="Linear probes against the number of training jets",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and "
                "17 classes, trained and scored on class-balanced JetClass-II and JetClass (accuracy) and on top "
                "and quark/gluon tagging (R30), against the number of training jets; dashed, a network "
                "trained from scratch. Points are means, bands the SD over three pretraining runs (from "
                "scratch: three trainings). Stars: published ParT (JetClass: same inputs; top: kinematic; "
                "quark/gluon: kinematic and particle type).",
        differs="Four panels against Golling's one; four label sets against three training modes. SD bands as "
                "Golling's (PI, 2026-11-07, M1). Label sets in Tol muted, darkest for 188, from scratch dark grey "
                "dashed. Sophon's font; faint grid at major y ticks, where Golling has none; no legend "
                "frame, as Golling. Log y on R30.",
        draw=draw_a10,
        data_note=SUBST_NOTE),
    C.Page(
        id="A10-auc",
        title="Linear probes against the number of training jets, AUC",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and "
                "17 classes, and (dashed) a network trained from scratch: "
                "the AUC averaged over the classes on class-balanced JetClass-II and JetClass "
                "and the AUC of top and quark/gluon tagging, "
                "against the number of training jets. Points are means, bands the SD over three pretraining runs (from "
                "scratch: three trainings). Stars: published ParT (JetClass: same inputs; top: kinematic; "
                "quark/gluon: kinematic and particle type).",
        differs="A10's form in AUC (PI 2026-11-11: the scaling figures need AUC as well as accuracy and rejection).",
        draw=lambda path: draw_a10(path, auc=True)),
    *[_umap_page(page) for page in UMAP_PAGES],
    C.Page(
        id="VAL",
        title="Pretraining validation accuracy against epoch (appendix)",
        precedent=["2404.16091/fig1.png"],
        precedent_label="OmniLearn Fig. 1 (validation loss against epochs)",
        caption="Pretraining on JetClass-II with 188, 162, 43 and 17 classes: validation accuracy against epoch, "
                "each network scored on its own classes (fewer classes for a coarser label set), the mean over three "
                "pretraining runs. The learning rate is constant to epoch 56 and decays after it.",
        differs="Accuracy, not loss; one curve per label set; no marker for 80 points; theme grid.",
        draw=draw_val),
]
