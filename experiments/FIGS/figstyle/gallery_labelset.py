#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2a, F2b, F3, F3-scatter, F3-app, F3-ft, F8, A8, A10, UMAP-jc1, UMAP-jc2-had, UMAP-jc2-lep, UMAP-top, UMAP-qg, VAL (docs/figstyle/PLAN.md), in
style_v4 only. The numbers come through the selection code of paper_figures and mlst_results; the drawing
is written here. All work happens inside the draw functions."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import Normalize

import gallery_common as C
from gallery_roc import FAMILIES, family_of
import style_v5 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I

GRID_2X2 = {"left": 0.1, "right": 0.96, "top": 0.92, "bottom": 0.06, "wspace": 0.3, "hspace": 0.2}     # room for repeated y labels, tick labels and titles
HEIGHT_HEAT = 8.0                                 # 188 rows; a 5-row group label at 10 pt still clears its neighbours
LEFT_HEAT = 0.26                                  # the longest group label at 10 pt, so the saved width stays 15 cm
MARKER = 4
Y_ACC, Y_R30 = "Accuracy", r"Rejection ($R_{30}$)"
X_CLASSES, X_JETS = "Number of pretraining classes", "Number of training jets"
LEGEND_PANEL = 1        # JetClass: its curves stay high on the right, so its lower right is empty in F2, F8 and A10
TITLES = {key: title for key, _, _, _, title, _ in P.F2_PANELS}
HEAT_F3 = ("L162", "R42_Q1", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R")
PUBLISHED = {"jc1": (2_000_000, 0.836), "top": (1_200_000, 1602.0), "qg": (1_600_000, 101.2)}     # ParT's published values, as in the paper's benchmark tables


def _chosen() -> dict:
    return M.chosen_substitutions(I["fill"], RUNS)[0]


def _panels_2x2():
    fig = plt.figure(figsize=S.FOUR_PANELS)
    return fig, fig.subplots(2, 2, gridspec_kw=GRID_2X2)


def _log_y(ax) -> None:
    """R30 on a log axis, labelled in plain numbers (1, 2, 5 times a power of ten; powers of ten only when that gives
    more than seven), the same in every panel; minor ticks are never labelled."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    vals = [m * 10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))) for m in (1, 2, 5)]
    if sum(lo <= v <= hi for v in vals) > 7:       # several decades: the powers of ten only
        vals = [v for v in vals if f"{v:g}".startswith("1")]
    if sum(lo <= v <= hi for v in vals) < 3:
        vals = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ax.yaxis.set_major_locator(ticker.FixedLocator([v for v in vals if lo <= v <= hi]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def _empty_panel(ax) -> None:
    """A panel with no data keeps its frame, title and x axis, so the layout is the final one."""
    ax.set_yticks([])


# ------------------------------------------------------------------ F2: metric against the number of classes
def _classes_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(9.6, 260)            # the leftmost random partition clear of the spine
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left")):   # 162 and 188 lean apart
        lab.set_horizontalalignment(ha)


def _mean(row: str, value):
    v = P._vals(RUNS, row, value)
    return float(np.mean(v)) if v else None


CONNECTOR = {"color": "#888888", "linewidth": S.LW_THIN, "zorder": 1}      # joins the hierarchy's points: no model colour
CONTROL_X = {"FLAV_F0": 15.0, "FLAV_F1": 13.7, "FLAV_F1R": 12.5, "RAND": 11.4}  # the 17-class controls, offset left of 17


def _range(ax, ys: list, log: bool, pad: float = 0.12) -> tuple:
    """y limits from the plotted series only, padded by pad of the span (in log space on a log axis)."""
    lo, hi = min(ys), max(ys)
    if log:
        a, b = math.log10(lo), math.log10(hi)
        d = max(b - a, 0.05) * pad
        return 10 ** (a - d), 10 ** (b + d)
    d = max(hi - lo, 1e-3) * pad
    return lo - d, hi + d


def _f2_handles(groups: list) -> list:
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], linestyle="none", **S.marker(r, size=MARKER + 0.5), label=S.NAME[r]) for r in S.LEVELS]
    hs += [Line2D([], [], linestyle="none", **S.marker(r, size=MARKER + 0.5), label=lab) for lab, r, _ in groups]
    hs += [Line2D([], [], linestyle="none", label=" ")] * (4 - len(groups))     # F2b: the legend laid out as F2a's
    hs += [Line2D([], [], **CONNECTOR, label="Hierarchy"),
           Line2D([], [], **S.look("SCRATCH", linewidth=S.LW_CURVE), label="From scratch, no pretraining")]
    return hs


def _draw_f2(path, value_of, groups, absent: str, heading: str) -> None:
    """value_of(key, n, ds, met): a run's value or None for the task; groups: (legend label, style key, rows) of the
    markers at 17 classes, drawn left of 17 at CONTROL_X so they do not hide the 17-class point. Filled markers in both
    F2a and F2b: each holds one mode (PI, 2026-11-07)."""
    values = [value_of(key, n, ds, met) for key, n, ds, met, *_ in P.F2_PANELS]
    if not any(values):
        raise C.Absent(absent)
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=2.6, left=1.15, wgap=0.8)     # 0.05 cm spare: the right frame whole   # room for "0.850"-wide ticks
    for k, (ax, value, (key, _, _, met, *_)) in enumerate(zip(axes.flat, values, P.F2_PANELS)):
        S.title(ax, TITLES[key])
        S.labels(ax, X_CLASSES if k >= 2 else None, (Y_R30 if met == "r30" else Y_ACC) if k % 2 == 0 else None)
        if value is None:
            _empty_panel(ax)
            _classes_axis(ax)
            continue
        S.fine_grid(ax)
        line = [(row, x, y) for row, x in P.X_OF.items() if (y := _mean(row, value)) is not None]
        ax.plot([x for _, x, _ in line], [y for *_, y in line], **CONNECTOR)
        ys = [y for *_, y in line]
        for _, key_, rows in groups:
            for row in rows:
                if (y := _mean(row, value)) is not None:
                    spread = 1 + 0.035 * (rows.index(row) - (len(rows) - 1) / 2)    # the five random partitions apart
                    ax.plot(CONTROL_X[key_] * spread, y, linestyle="none", **S.marker(key_, size=MARKER), zorder=2)
                    ys.append(y)
        for row, x, y in line:      # each label set's point in its own colour and marker, over the controls
            ax.plot(x, y, linestyle="none", **S.marker(row, size=MARKER + 0.5), zorder=3)
        log = met == "r30"
        scratch = P.scratch_series(M, I["per"], key, met)
        if scratch:     # always drawn as its line; the y range is widened to hold it (PI, 2026-11-07)
            value = float(np.mean(scratch[max(scratch)]))
            ax.axhline(value, **S.look("SCRATCH", linewidth=S.LW_CURVE), zorder=1.5)
            ys.append(value)
        ax.set_ylim(*_range(ax, ys, log))
        if log:
            _log_y(ax)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))     # round steps, few decimals
        _classes_axis(ax)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.fig_legend(fig, _f2_handles(groups), top_cm=0.6, ncol=4)
    S.save(fig, path)


def _probe_value(key, n, ds, met):
    return P._f2_probe(M, I["fits"], I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    return P._f2_ft(M, I["per"], key, n, met)


def _controlled():
    return [(S.NAME[a], a, [a]) for a in S.CONTROLS]


def draw_f2a(path) -> None:
    _draw_f2(path, _probe_value, _controlled() + [(S.NAME["RAND"], "RAND", M.RAND)],
             "no linear probes (linprobe/fits_v4)", "Linear probes on frozen features")


def draw_f2b(path) -> None:
    _draw_f2(path, _ft_value, _controlled(), "no fine-tuning read-outs (ft_mlst, budget)", "After fine-tuning")


# ------------------------------------------------------------------ F3: per-class heatmap
def _draw_heat(path, arms) -> None:
    fits = I["fits"]
    if not P._has(fits, "jc2_188"):
        raise C.Absent("no class-balanced JetClass-II linear probes (linprobe/fits_v4)")
    rows, R = P.heat(M, fits, RUNS, P.REPO)
    keys = [a for a, _ in P.MK.HEAT]
    D = R[:, [keys.index(a) for a in arms]]
    if np.isnan(D).all():
        raise C.Absent("no per-class linear probes for these columns (linprobe/fits_v4)")
    limit = float(np.nanmax(np.abs(R)))      # nothing clips
    n_res = sum(not r["class_name"].startswith("label_QCD") for r in rows)
    row = lambda i: i + (i >= n_res)         # a blank row between the resonant and the QCD classes

    # round 4: exact size; a thin gap between the hierarchy's columns and the controls'; colour bar the panel's height
    gap = 0.3                                   # in columns
    n_hier = sum(a in S.LEVELS for a in arms)
    data = np.ma.masked_invalid(np.insert(D, n_res, np.nan, axis=0))
    ye = np.arange(data.shape[0] + 1)
    left, top, w, h = 4.3, 1.0, 8.4, 19.5
    fig = S.figure_cm(S.PRINT_W, top + h + 1.9)
    ax = S.axes_cm(fig, left, top, w, h)
    cmap = colormaps["RdBu"].copy()
    cmap.set_bad("white")
    norm = Normalize(-limit, limit)
    im = ax.pcolormesh(np.arange(n_hier + 1), ye, data[:, :n_hier], cmap=cmap, norm=norm)
    ax.pcolormesh(np.arange(n_hier, len(arms) + 1) + gap, ye, data[:, n_hier:], cmap=cmap, norm=norm)
    xe = np.array([0.0, len(arms) + gap])
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[-1], 0)
    centres = [i + 0.5 + (gap if i >= n_hier else 0.0) for i in range(len(arms))]
    ax.set_xticks(centres)
    short = {"FLAV_F0": "No\nflavor\nboundary", "FLAV_F1": "$b$-quark\nboundary", "FLAV_F1R": "Random\nflavor\nboundary"}
    ax.set_xticklabels([short.get(a, _column(a).replace(" ", "\n")) for a in arms], fontsize=S.TICK_PT5, va="top")
    ax.tick_params(axis="both", which="both", length=0, top=False, right=False)
    ax.minorticks_off()

    labs = [P.MK.GROUP_TEX.get(r["R16_Q1_name"], r["R16_Q1_name"].replace("_", " ")) for r in rows]
    starts = [i for i in range(len(labs)) if i == 0 or labs[i] != labs[i - 1]]
    ends = starts[1:] + [len(labs)]
    for a in starts[1:]:
        if a != n_res:
            ax.axhline(row(a), color="black", linewidth=0.4)
    ax.set_yticks([(row(a) + row(b - 1)) / 2 + 0.5 for a, b in zip(starts, ends)])
    ax.set_yticklabels([labs[a] for a in starts])
    S.labels(ax, "Pretraining labels", "JetClass-II class")
    ax.xaxis.labelpad = 6

    cax = S.axes_cm(fig, left + w + 0.3, top, 0.35, h)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)    # the full range, no hairline at its end
    cax.set_facecolor(cmap(0.0))       # the renderer leaves a hairline under the lowest cell: fill it with that colour
    cb.ax.minorticks_off()
    S.labels(cb.ax, ylabel="AUC difference to 188 classes (percentage points)")
    S.heading(fig, "Linear probes on JetClass-II, class by class")
    S.save(fig, path)


def _column(arm: str) -> str:
    return S.NAME.get(arm) or dict(P.MK.HEAT)[arm]


def draw_f3(path) -> None:
    _draw_heat(path, HEAT_F3)


# ------------------------------------------------------------------ F3-scatter, F3-app: every class against the 188-class model
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
                  absent: str = "no class-balanced JetClass-II linear probes (linprobe/fits_v4)") -> None:
    """panels: (runs, title), two to four. One point per class; below the dashed line the class does worse than with
    188 classes. Round 4: square 6.5 cm panels, two per row; the legend above the panels, or in the empty fourth cell
    of a three-panel figure; letters names the panels (a second figure continues the first's letters)."""
    from matplotlib.lines import Line2D
    full = auc_of(RUNS["L188"])
    if not full:
        raise C.Absent(absent)
    family = {int(r["jet_label"]): family_of(r["class_name"]) for r in M._class_rows(P.REPO)}
    nrows = math.ceil(len(panels) / 2)
    in_cell = len(panels) == 3
    fig, axes, _ = S.grid_cm(nrows, 2, aspect=1.0, top=1.15 if in_cell else 2.2, left=1.3, wgap=0.4)
    cells = list(axes.flat)
    for ax in cells[1:]:
        ax.sharex(cells[0])
        ax.sharey(cells[0])
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)      # one y range: its tick labels once, on the left
    pairs = [[(full[c], other[c], c) for c in full if c in other] for other in (auc_of(models) for models, _ in panels)]
    # one range for the two figures of a split (scope: every panel of both), so their axes match
    ranged = pairs if scope is None else [[(full[c], o[c]) for c in full if c in o] for o in (auc_of(m) for m, _ in scope)]
    lo = min(v for pts in ranged for x, y, *_ in pts for v in (x, y))
    lim = (lo - 0.04 * (1 - lo), 1.0 + 0.01 * (1 - lo))     # the lowest point and the points at AUC 1 clear of the frame
    for k, (ax, pts, (_, title)) in enumerate(zip(cells, pairs, panels)):
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
        if k % 2 == 0:
            S.labels(ax, ylabel="AUC, this model")
        if k + 2 >= len(panels):
            S.labels(ax, xlabel="AUC, 188 classes")
    cells[0].set_xlim(lim)
    cells[0].set_ylim(lim)
    for axis in (cells[0].xaxis, cells[0].yaxis):    # one tick step on both axes of every panel
        axis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    handles = [Line2D([], [], color=S.TAB10[i], marker="o", markersize=4, linestyle="none", label=f)
               for i, f in enumerate(FAMILIES)] + [Line2D([], [], **S.REFERENCE, label="Same as 188 classes")]
    if in_cell:
        cells[-1].axis("off")
        cells[-1].legend(handles=handles, loc="center", fontsize=S.LEGEND_PT5, frameon=False, borderpad=0.3,
                         labelspacing=0.4)
    else:
        S.fig_legend(fig, handles, top_cm=0.6, ncol=4)
    S.heading(fig, f"{word} on $10^6$ JetClass-II jets, class by class")
    S.save(fig, path)


def _app_panels() -> list:
    return ([(RUNS["L162"], S.NAME["L162"]), ([m for r in M.RAND for m in RUNS[r]], "Five random partitions")]
            + [(RUNS[a], S.NAME[a]) for a in S.CONTROLS]
            + [(RUNS[a], f"{S.NAME[b]}, mass regression") for a, b in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))])


def _ft_panels() -> list:
    return ([(RUNS[a], S.NAME[a]) for a in ("L162", "R42_Q1", "R16_Q1")]
            + [([m for r in M.RAND for m in RUNS[r]], "Five random partitions")] + [(RUNS[a], S.NAME[a]) for a in S.CONTROLS])


FT_ABSENT = "no per-class curves after fine-tuning (curves_v2/jc2_ft_ovr)"


def draw_f3_scatter(path) -> None:
    _draw_scatter(path, [(RUNS[a], S.NAME[a]) for a in ("R42_Q1", "R16_Q1")])


def draw_f3_app(path) -> None:
    _draw_scatter(path, _app_panels()[:4], scope=_app_panels())


def draw_f3_app2(path) -> None:
    _draw_scatter(path, _app_panels()[4:], letters="efg", scope=_app_panels())


def draw_f3_ft(path) -> None:
    """F3-scatter's form after fine-tuning (curves_v2/jc2_ft_ovr): every model fine-tuned on JetClass-II."""
    _draw_scatter(path, _ft_panels()[:4], auc_of=_ft_class_auc, word="Fine-tuned", absent=FT_ABSENT, scope=_ft_panels())


def draw_f3_ft2(path) -> None:
    _draw_scatter(path, _ft_panels()[4:], auc_of=_ft_class_auc, word="Fine-tuned", absent=FT_ABSENT, letters="efg",
                  scope=_ft_panels())


# ------------------------------------------------------------------ F8, A10: metric against the number of jets
def _curve(ax, series: dict, line: dict, mark: dict, label: str, band: bool) -> None:
    """Mean over runs; line the colour and line style, mark the marker and its fill (style_v5.marker); with band,
    Golling's: the SD (range for two runs), same colour, no edge."""
    xs = sorted(series)
    if not xs:
        return
    st = [P.F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    ax.plot(xs, mean, **line, linewidth=S.LW_CURVE, **mark, markersize=MARKER, label=label)
    if band:
        ax.fill_between(xs, mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st]),
                        color=line["color"], alpha=0.2, linewidth=0)


def _model(series: dict, key: str, mode: str, label: str) -> tuple:
    return series, S.look(key, mode), S.marker(key, mode), label


SIZE_X = (6e2, 2.6e6)       # 10^3 to the largest training set (1.6e6 quark/gluon jets), padded


def _draw_size(path, series_of, band: bool, absent: str, heading: str) -> None:
    """series_of(key, ds, met): [(series, line, mark, label)] of a panel. Round 4: four 6.5 cm panels, the legend above."""
    panels = [series_of(key, ds, met) for key, ds, met, *_ in P.SIZE_PANELS]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=2.25, left=1.2, wgap=0.8)
    for k, (ax, panel, (key, _, met, *_)) in enumerate(zip(axes.flat, panels, P.SIZE_PANELS)):
        ax.set_xscale("log")
        ax.set_xlim(*SIZE_X)          # one x range in every panel, so the decades line up
        S.title(ax, TITLES[key])
        S.labels(ax, X_JETS if k >= 2 else None, (Y_R30 if met == "r30" else Y_ACC) if k % 2 == 0 else None)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        for series, line, mark, label in panel:
            _curve(ax, series, line, mark, label, band)
        if key in PUBLISHED:
            ax.plot(*PUBLISHED[key], color="#000000", marker="*", markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                    linestyle="none", label="ParT, published", zorder=4)     # on top, its white edge clear of the points
        if met == "r30":
            _log_y(ax)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    handles, labels = {}, {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    S.fig_legend(fig, list(handles.values()), top_cm=0.6, ncol=len(handles) if len(handles) <= 4 else 3)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.save(fig, path)


def draw_f8(path) -> None:
    """One model in three modes: fine-tuned in the 188 colour, the linear probe in its 40% tint (style_v5.TINT_188,
    used only here), from scratch dark grey."""
    chosen = _chosen()
    probe_line = {"color": S.TINT_188, "linestyle": S.MODE_LINE["probe"]}
    probe_mark = {"marker": "s", "markeredgecolor": S.TINT_188, "markerfacecolor": "white"}
    series = lambda key, ds, met: [_model(P.ft_series(M, I["per"], RUNS, key, met, "L188"), "L188", "ft", "Fine-tuned"),
                                   (P.probe_series(M, I, RUNS, ds, met, "L188", chosen), probe_line, probe_mark,
                                    "Linear probe"),
                                   _model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft", S.NAME["SCRATCH"])]
    _draw_size(path, series, True, "no fine-tuning read-outs or linear probes", "188-class pretraining against training from scratch")


def draw_a10(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [_model(P.probe_series(M, I, RUNS, ds, met, row, chosen), row, "ft", S.NAME[row])   # probes only: solid
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  S.NAME["SCRATCH"])]
    _draw_size(path, series, False, "no linear probes (linprobe/fits_v4)", "Linear probes, per pretraining label set")


def draw_a8(path) -> None:
    """A10's form after fine-tuning: the four label sets and from scratch."""
    series = lambda key, ds, met: [_model(P.ft_series(M, I["per"], RUNS, key, met, row), row, "ft", S.NAME[row])
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  S.NAME["SCRATCH"])]
    _draw_size(path, series, False, "no fine-tuning read-outs", "Fine-tuned, per pretraining label set")


# ------------------------------------------------------------------ the UMAP and the pretraining curve
EMBED_NAMES = {"bb": r"$X\to b\bar{b}$", "cc": r"$X\to c\bar{c}$", "qq": r"$X\to q\bar{q}$", "QCD": "QCD",
               "ee": r"$X\to ee$", "mm": r"$X\to \mu\mu$", "top": "Top", "gluon": "Gluon", "quark": "Quark",
               **{c.removeprefix("label_"): S.JC1_TEX[c] for c in S.JC1_CLASSES if c != "label_QCD"}}
EMBED_MODELS = (("mtx-l188-s1", "L188"), ("mtx-r16q1-s1", "R16_Q1"))


UMAP_DIRS = ("umap_v2", "umap", "umap_preview")     # round 3 (all four tasks); the long-budget jc1 run; the preview
UMAP_SETTINGS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 1, "standardized": True}
UMAP_READ = {      # one sentence per figure on what its panels show, written after looking at them (round 4)
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
    fig, axes, _ = S.grid_cm(len(grid), len(grid[0]), aspect=1.0, w=6.5, left=0.8, wgap=0.3, hgap=0.6, bottom=0.3,
                             top=1.15 + 0.5 * legend_rows_n)
    for r, (row, drow) in enumerate(zip(grid, docs)):
        for ax, (_, title), d in zip(axes[r], row, drow):
            _embedding(ax, d, colours or S.TAB10)
            S.title(ax, title)
    handles = [Line2D([], [], color=(colours or S.TAB10)[k], marker="o", markersize=5, linestyle="none",
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
        caption=f"UMAP of the class token of a network pretrained on 188 classes (upper) and one on 17 (lower), frozen "
                f"(left) and fine-tuned on {ft} (right), for {jets} in every panel; the axes carry no scale. "
                f"{UMAP_READ[page]}",
        differs="One 2x2 layout for every task (PI, round 3): rows label sets, columns frozen and fine-tuned, the "
                "poster's pairing. " + ("Ten class colours, JetClass's fixed tab10. " if page == "UMAP-jc1" else "tab10 in group order. ")
                + "One legend, first panel. " + (f"UMAP with {u['n_neighbors']} neighbours, minimum distance "
                                                   f"{u['min_dist']}, seed {u['random_state']}.") + (" Absent inputs zero-filled, as in every frozen probe and fine-tuning."
                                                 if page in ("UMAP-top", "UMAP-qg") else ""),
        draw=lambda path: _draw_umap(path, page),
        data_note="From umap_v2 (scripts/build_umap_jobs.py --out-name umap_v2).")


VAL_W = 10.5         # cm: one panel needs no more; the paper includes it at this width


def draw_val(path) -> None:
    """Round 4: exact size, 10.5 cm wide (\\paperfig{10.5cm}), the panel 8.6 cm wide with the legend inside."""
    docs = M._load(P.DATA / "compute")
    fig = S.figure_cm(VAL_W, 8.6)
    ax = S.axes_cm(fig, 1.5, 0.7, 8.6, 6.9)
    S.fine_grid(ax)
    for level in S.LEVELS:
        have = [[e["val_acc"] for e in docs[m]["epochs"]] for m in RUNS[level] if m in docs and "val_acc" in docs[m]["epochs"][0]]
        if not have:
            raise C.Absent("no validation accuracy per epoch (compute/)")
        ax.plot(range(1, len(have[0]) + 1), np.mean(have, axis=0), **S.look(level), linewidth=S.LW_CURVE, label=S.NAME[level])
    S.title(ax, "Pretraining on JetClass-II")
    S.labels(ax, "Pretraining epoch", "Validation accuracy")
    h_, l_ = ax.get_legend_handles_labels()
    S.legend(ax, handles=h_[::-1], labels=l_[::-1], loc="center right")   # top to bottom as the curves lie
    S.save(fig, path)


# ------------------------------------------------------------------ the pages
HONG = ["2410.23129/fig2.png"]
HEATMAPS = ["2604.09809/fig7.png", "2606.14813/fig12.png"]
SUBST_NOTE = ("Top-tagging linear probes fill the missing inputs with random draws and quark/gluon probes with the "
              "training mean: the substitution chosen for each task on validation, over all 30 models.")
TWO_RUNS = "The three controlled partitions have two runs each, the four label sets three."

PAGES = [
    C.Page(
        id="F2a",
        title="Label granularity, linear probes",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Linear probes on the frozen features against the number of pretraining classes: accuracy on "
                "class-balanced JetClass-II and on JetClass, and background rejection R30 at 30% signal efficiency on top "
                "and quark/gluon (Pythia) tagging, log axis. Markers are means over runs and the grey line (hierarchy) joins the "
                "four label sets. "
                "The 17-class random partitions, three with controlled flavor boundaries and five plain, are drawn just "
                "left of 17. The dashed line is the network trained from scratch.",
        differs="Four panels against Hong's one, for the four tasks; Hong has no controls. Round 4: 6.5 cm panels in a "
                "2x2, a figure legend above them that names every colour, filled markers (probes only in the figure) and a grey line "
                "for the hierarchy, controls offset left of 17 with their own shapes, and each y range holding the "
                "from-scratch line.",
        draw=draw_f2a,
        data_note=f"{TWO_RUNS} Each plain random partition has one run. {SUBST_NOTE}"),
    C.Page(
        id="F2b",
        title="Label granularity, after fine-tuning",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Fine-tuned models against the number of pretraining classes: accuracy on class-balanced JetClass-II "
                "and on JetClass, and background rejection R30 at 30% signal efficiency on top and quark/gluon tagging. "
                "Log axes on (c, d). Markers are means over runs and the grey line (hierarchy) joins the four label sets. The three 17-class "
                "partitions with controlled flavor boundaries are drawn just left of 17. The dashed line is the network "
                "trained from scratch.",
        differs="As F2a. Hong's own figure is fine-tuned too, so this page is the closer "
                "copy.",
        draw=draw_f2b,
        data_note="JetClass-II at 1,000,000 class-balanced jets. JetClass at 1,000,000 jets, top at 1,200,000, quark/gluon at 1,600,000. "
                  f"{TWO_RUNS} Top: every difference between label sets lies within the run-to-run SD (56 to 151); the "
                  "random-boundary marker lies 133 to 188 below the label sets. Quark/gluon: differences between label "
                  "sets are comparable to the run spread (SD 0.8 to 2.2, up to 1.7 between 43 and 17 classes); the "
                  "no-flavor-boundary marker lies 2.4 and 2.7 above the b-quark and random-boundary markers, more than "
                  "any label-set SD."),
    C.Page(
        id="F3",
        title="Cost class by class",
        precedent=HEATMAPS,
        precedent_label="CMS Fig. 7, JetParticle-JEPA Fig. 12",
        caption="Difference in per-class AUC between each model and the 188-class model, in percentage points, for "
                "the 188 classes of class-balanced JetClass-II, from linear probes. Columns are the 162-, 43- and 17-class label sets "
                "and the three random partitions with controlled flavor boundaries. Rows are grouped by decay topology, "
                "with the QCD classes at the bottom. Values are means over runs; red marks an AUC below that of "
                "the 188-class model. Every class's values are in the data release.",
        differs="Signed values on a diverging RdBu scale symmetric about 0: Blues would hide the sign. Not a confusion "
                "matrix: rows are classes, columns are models. No numbers in cells: 188 rows cannot hold them. Rows "
                "named by 15 group labels, since a single class name cannot reach 8 pt at 188 rows; thin lines between "
                "groups and a blank row before QCD in place of per-row labels; no row tick marks. One panel as CMS, against "
                "JetParticle-JEPA's three, and no titles where it has three and a heading. Taller than either "
                "precedent, with wide cells, not square. Colour bar about 62% of the axes height, against the full "
                "height in both. Column labels at 45 degrees as JetParticle-JEPA, not CMS's 90. Axis words at the "
                "axis end as CMS, not JetParticle-JEPA. Sophon's font. No experiment label.",
        draw=draw_f3,
        data_note="The three controlled partitions are means over two runs, the label sets over three."),
    C.Page(
        id="F3-scatter",
        title="Cost class by class, per-class AUC (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Each of the 188 classes of class-balanced JetClass-II, from linear probes trained on one million jets: the class's area under the "
                "curve for the 43- and 17-class label sets, against the same for the 188-class model. On the dashed line "
                "the two agree; below it the class does worse than with 188 classes. Colors are decay families. Values "
                "are means over runs.",
        differs="One point per class on linear axes, the same range on both, from the data minimum to 1 (PI). No 1 - AUC. "
                "Round 4: two 6.5 cm squares, the legend above them.",
        draw=draw_f3_scatter),
    C.Page(
        id="F3-app",
        title="Cost class by class, other models, first part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Each class of class-balanced JetClass-II, from linear probes trained on one million jets: its area under the curve for (a) the "
                "162-class label set, (b) the five plain random partitions averaged, (c) the no-flavor-boundary and (d) "
                "the b-quark-boundary partitions, against the same for the 188-class model. Below the dashed line the "
                "class does worse than with 188 classes. Colors are decay families; values are means over runs.",
        differs="As F3-scatter. Round 4: seven panels split 4+3 over two figures, so every panel is a 6.5 cm square.",
        draw=draw_f3_app),
    C.Page(
        id="F3-app2",
        title="Cost class by class, other models, second part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Continued from the previous figure: each class's area under the curve for (e) the random-flavor-boundary "
                "partition and the (f) 162- and (g) 17-class label sets pretrained with an auxiliary regression of the jet "
                "mass, against the same for the 188-class model, from linear probes trained on one million jets on class-balanced JetClass-II. "
                "Colors are decay families; values are means over runs.",
        differs="The second part of F3-app; the legend takes the empty fourth cell.",
        draw=draw_f3_app2),
    C.Page(
        id="F3-ft",
        title="Cost class by class after fine-tuning, first part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Each class of class-balanced JetClass-II after fine-tuning on one million jets: its area under the curve "
                "for the (a) 162-, (b) 43- and (c) 17-class label sets and (d) the five plain random partitions averaged, "
                "against the same for the 188-class model. Below the dashed line the class does worse than with 188 "
                "classes. Colors are decay families; values are means over three runs, in (d) over the five partitions, one run each.",
        differs="F3-scatter's form after fine-tuning (PI). Round 4: seven panels split 4+3 over two figures.",
        draw=draw_f3_ft),
    C.Page(
        id="F3-ft2",
        title="Cost class by class after fine-tuning, second part (appendix)",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the heatmap's",
        caption="Continued from the previous figure: each class's area under the curve after fine-tuning on one million "
                "class-balanced JetClass-II jets, for the partitions with (e) no flavor boundary, (f) a b-quark boundary "
                "and (g) a random flavor boundary, against the same for the 188-class model. Colors are decay families; "
                "values are means over two runs.",
        differs="The second part of F3-ft; the legend takes the empty fourth cell.",
        draw=draw_f3_ft2),
    C.Page(
        id="F8",
        title="Pretraining against the number of training jets",
        precedent=["2401.13537/fig3.png", "2401.13536/fig1.png"],
        precedent_label="Golling Fig. 3, Vigl Fig. 1 top",
        caption="Accuracy on class-balanced JetClass-II and on JetClass, and rejection R30 at 30% signal efficiency on "
                "top and quark/gluon tagging, against the number of training jets: the 188-class model fine-tuned and "
                "with a linear probe, and a network trained from scratch. Points are means; bands the standard deviation "
                "over three runs. Fine-tuning on JetClass-II and JetClass starts at 10^4 jets. Stars are published ParT: same "
                "inputs on JetClass, kinematic only on top, kinematic and particle type on quark/gluon.",
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
        id="A8",
        title="Fine-tuned against the number of training jets (appendix)",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Fine-tuned models of the four label sets against the number of training jets: accuracy on class-balanced "
                "JetClass-II and on JetClass, and rejection R30 at 30% signal efficiency on top and quark/gluon tagging, "
                "with a network trained from scratch dashed. Points are means over three runs; overlapping "
                "markers agree within their size. Fine-tuning on JetClass-II and JetClass starts at 10^4 jets. Stars are published ParT: same "
                "inputs on JetClass, kinematic only on top, kinematic and particle type on quark/gluon.",
        differs="A10's form, fine-tuned (PI: the task x model x mode grid).",
        draw=draw_a8,
        data_note="From the fine-tuning read-outs as they stand: the long-budget cells at 10^6 on JetClass and "
                  "JetClass-II, and the small-size benchmark cells still running, replace their points when read out."),
    C.Page(
        id="A10",
        title="Linear probes against the number of training jets",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes of the four label sets against the number of training jets: accuracy on class-balanced "
                "JetClass-II and on JetClass, and rejection R30 at 30% signal efficiency on top and quark/gluon tagging, "
                "with a network trained from scratch dashed. Points are means over three runs; overlapping markers "
                "agree within their size. Stars are published ParT: same "
                "inputs on JetClass, kinematic only on top, kinematic and particle type on quark/gluon.",
        differs="Four panels against Golling's one; four label sets against three training modes. No bands, where "
                "Golling has them: four would overlap (plan). Label sets in Tol muted, darkest for 188, from scratch dark grey "
                "dashed. Sophon's font; faint grid at major y ticks, where Golling has none; no legend "
                "frame, as Golling. Log y on R30.",
        draw=draw_a10,
        data_note=SUBST_NOTE),
    *[_umap_page(page) for page in UMAP_PAGES],
    C.Page(
        id="VAL",
        title="Pretraining validation accuracy against epoch (appendix)",
        precedent=["2404.16091/fig1.png"],
        precedent_label="OmniLearn Fig. 1 (validation loss against epochs)",
        caption="Validation accuracy of the pretraining against epoch for the 188-, 162-, 43- and 17-class label sets, each "
                "on its own classes, as the mean over three runs. A coarser label set has fewer classes to tell apart, so "
                "its accuracy is not comparable with a finer set's. The step near epoch 56 is where the learning rate "
                "starts to decay.",
        differs="Accuracy, not loss; one curve per label set; no marker for 80 points; theme grid.",
        draw=draw_val),
]
