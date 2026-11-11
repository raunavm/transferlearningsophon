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
PUBLISHED = {"jc1": (2_000_000, 0.836), "top": (1_200_000, 1602.0), "qg": (1_600_000, 101.2)}     # ParT's published values, as in the paper's benchmark tables


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
CLASS_COUNTS = (17, 43, 162, 188)
CLASS_X = {"R16_Q1": 0, "R42_Q1": 1, "L162": 2, "L188": 3}     # one evenly spaced slot per label set, labelled by its count


def _classes_axis(ax) -> None:
    """The four label sets as evenly spaced slots labelled with their class counts (PI 2026-11-11: on a log axis the
    162 and 188 labels touched and the 17-class controls, offset to the left, read as fewer classes)."""
    ax.set_xlim(-0.5, 3.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(4)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(n) for n in CLASS_COUNTS]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def _mean(row: str, value):
    v = P._vals(RUNS, row, value)
    return float(np.mean(v)) if v else None


CONNECTOR = {"color": "#888888", "linewidth": S.LW_THIN, "zorder": 1}      # joins the hierarchy's points: no model colour
# the 17-class controls sit at 17 classes, small symmetric offsets either side of the 17-class point
CONTROL_X = {"FLAV_F0": -0.26, "FLAV_F1": -0.13, "FLAV_F1R": 0.13, "RAND": 0.33}


def _spread_point(ax, x: float, vals: list, key: str, size: float, zorder: float) -> list:
    """The mean over runs as the model's marker with a bar of the spread over pretraining runs (mlst_figures._spread:
    the SD for three runs, the range for two, none for one); returns the bar's ends (M1, PI 2026-11-07)."""
    mean, lo, hi = P.F._spread(vals)
    ax.errorbar(x, mean, yerr=None if lo is None else [[lo], [hi]], linestyle="none", color=S.SERIES[key],
                elinewidth=S.LW_THIN, capsize=0, **S.marker(key, size=size), zorder=zorder)
    return [mean - (lo or 0), mean + (hi or 0)]


def _range(ax, ys: list, log: bool, pad: float = 0.12) -> tuple:
    """y limits from the plotted series only, padded by pad of the span (in log space on a log axis)."""
    lo, hi = min(ys), max(ys)
    if log:
        a, b = math.log10(lo), math.log10(hi)
        d = max(b - a, 0.05) * pad
        return 10 ** (a - d), 10 ** (b + d)
    d = max(hi - lo, 1e-3) * pad
    return lo - d, hi + d


def _f2_handles(groups: list, scratch: str) -> list:
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], linestyle="none", **S.marker(r, size=MARKER + 0.5), label=S.LEGEND[r]) for r in S.LEVELS]
    hs += [Line2D([], [], linestyle="none", **S.marker(r, size=MARKER + 0.5), label=lab) for lab, r, _ in groups]
    hs += [Line2D([], [], **CONNECTOR, label="nested label sets"),
           Line2D([], [], **S.look("SCRATCH", linewidth=S.LW_CURVE), label=scratch)]
    return hs


def _draw_f2(path, value_of, groups, absent: str, heading: str, scratch_label: str = S.LEGEND["SCRATCH"]) -> None:
    """value_of(key, n, ds, met): a run's value or None for the task; groups: (legend label, style key, rows) of the
    markers at 17 classes, drawn at 17 with the offsets of CONTROL_X so they do not hide the 17-class point. Filled markers in both
    F2a and F2b: each holds one mode (PI, 2026-11-07)."""
    values = [value_of(key, n, ds, met) for key, n, ds, met, *_ in P.F2_PANELS]
    if not any(values):
        raise C.Absent(absent)
    fig, axes, _ = S.grid_cm(2, 2, w=6.25, aspect=0.85, top=2.6, left=1.65, wgap=0.75)     # room for the widest tick labels
    for k, (ax, value, (key, _, _, met, *_)) in enumerate(zip(axes.flat, values, P.F2_PANELS)):
        S.title(ax, TITLES[key])
        S.labels(ax, X_CLASSES if k >= 2 else None, (Y_R30 if met == "r30" else Y_ACC) if k % 2 == 0 else None)
        if value is None:
            _empty_panel(ax)
            _classes_axis(ax)
            continue
        S.fine_grid(ax)
        line = sorted(((row, x, y) for row, x in CLASS_X.items() if (y := _mean(row, value)) is not None), key=lambda t: t[1])
        ax.plot([x for _, x, _ in line], [y for *_, y in line], **CONNECTOR)
        ys = []
        for _, key_, rows in groups:
            for row in rows:
                if v := P._vals(RUNS, row, value):
                    spread = 0.05 * (rows.index(row) - (len(rows) - 1) / 2)    # the five random partitions apart, each countable
                    ys += _spread_point(ax, CONTROL_X[key_] + spread, v, key_, MARKER, 2)
        for row, x, _ in line:      # each label set's point in its own colour and marker, over the controls
            ys += _spread_point(ax, x, P._vals(RUNS, row, value), row, MARKER + 0.5, 3)
        log = met == "r30"
        scratch = P.scratch_series(M, I["per"], key, met)
        if scratch:     # always drawn as its line; the y range is widened to hold it (PI, 2026-11-07)
            value = float(np.mean(scratch[max(scratch)]))
            ax.axhline(value, **S.look("SCRATCH", linewidth=S.LW_CURVE), zorder=1.5)
            ys.append(value)
        ax.set_ylim(*_range(ax, ys, log))
        if log:
            S.log_axis(ax)          # the limits above hold the bars and the from-scratch line
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))     # round steps, few decimals
        _classes_axis(ax)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.fig_legend(fig, _f2_handles(groups, scratch_label), top_cm=0.6, ncol=4)
    S.save(fig, path)


def _probe_value(key, n, ds, met):
    return P._f2_probe(M, I["fits"], I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    return P._f2_ft(M, I["per"], key, n, met)


def _controlled():
    return [(S.LEGEND[a], a, [a]) for a in S.CONTROLS]


def draw_f2a(path) -> None:
    _draw_f2(path, _probe_value, _controlled() + [(S.LEGEND["RAND"], "RAND", M.RAND)],
             "no linear probes (linprobe/fits_v4)", "Linear probes on frozen features")


def draw_f2b(path) -> None:
    _draw_f2(path, _ft_value, _controlled(), "no fine-tuning read-outs (ft_mlst, budget)", "After fine-tuning")


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
                  ylabel: str = "AUC, the panel's network") -> None:
    """panels: (runs, title) or (runs, title, auc_of), two to four; a panel's own auc_of (F3-main: probe | fine-tuned)
    reads both its model and the 188-class one. One point per class; below the dashed line the class does worse than
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
        axis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    handles = [Line2D([], [], color=S.TAB10[i], marker="o", markersize=4, linestyle="none", label=f)
               for i, f in enumerate(FAMILIES)] + [Line2D([], [], **S.REFERENCE, label="Same as 188 classes")]
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
                  word="Linear probes (LP) and fine-tuned (f.t.)", ylabel=f"AUC, {S.LEGEND['R16_Q1']}")


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


SIZE_PAD = 1.7              # a panel's x axis runs from its smallest training set over SIZE_PAD to its largest times it


def _draw_size(path, series_of, band: bool, absent: str, heading: str, ncol: int = 3) -> None:
    """series_of(key, ds, met): [(series, line, mark, label)] of a panel. Round 4: four 6.5 cm panels, the legend above
    in ncol columns."""
    panels = [series_of(key, ds, met) for key, ds, met, *_ in P.SIZE_PANELS]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    rows = math.ceil((max(len(p) for p in panels) + 1) / ncol)       # the series and the published star
    fig, axes, _ = S.grid_cm(2, 2, w=6.35, aspect=0.85, top=1.4 + 0.42 * rows, left=1.4, wgap=0.8)
    for k, (ax, panel, (key, _, met, *_)) in enumerate(zip(axes.flat, panels, P.SIZE_PANELS)):
        ax.set_xscale("log")
        ns = [n for s_, *_ in panel for n in s_] + ([PUBLISHED[key][0]] if key in PUBLISHED else [])
        if ns:                        # each panel's range from its own points: no empty decade (PI 2026-11-11)
            ax.set_xlim(min(ns) / SIZE_PAD, max(ns) * SIZE_PAD)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, TITLES[key])
        S.labels(ax, X_JETS if k >= 2 else None, (Y_R30 if met == "r30" else Y_ACC) if k % 2 == 0 else None)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.fine_grid(ax)
        for series, line, mark, label in panel:
            _curve(ax, series, line, mark, label, band)
        if key in PUBLISHED:
            ax.plot(*PUBLISHED[key], color="#000000", marker="*", markersize=11, markeredgecolor="white", markeredgewidth=0.7,
                    linestyle="none", label="ParT, published", zorder=10)    # on top of every marker, large enough to see
        if met == "r30":
            _log_y(ax)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    handles, labels = {}, {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    S.fig_legend(fig, list(handles.values()), top_cm=0.6, ncol=ncol)
    S.letter_panels(axes.flat)
    S.heading(fig, heading)
    S.save(fig, path)


def draw_f8(path) -> None:
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
                                          f"{S.LEGEND['SCRATCH']} {S.FINE_TUNED}")]
    _draw_size(path, series, True, "no fine-tuning read-outs or linear probes", "188-class pretraining against training from scratch",
               ncol=2)


def draw_a10(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [_model(P.probe_series(M, I, RUNS, ds, met, row, chosen), row, "ft", S.LEGEND[row])   # probes only: solid
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  f"{S.LEGEND['SCRATCH']} {S.FINE_TUNED}")]
    _draw_size(path, series, True, "no linear probes (linprobe/fits_v4)", "Linear probes on frozen features")


def draw_a8(path) -> None:
    """A10's form after fine-tuning: the four label sets and from scratch."""
    series = lambda key, ds, met: [_model(P.ft_series(M, I["per"], RUNS, key, met, row), row, "ft", S.LEGEND[row])
                                   for row in S.LEVELS] + [_model(P.scratch_series(M, I["per"], key, met), "SCRATCH", "ft",
                                                                  S.LEGEND["SCRATCH"])]
    _draw_size(path, series, True, "no fine-tuning read-outs", "Fine-tuned")


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
    fig, axes, _ = S.grid_cm(len(grid), len(grid[0]), aspect=1.0, w=7.0, left=0.35, wgap=0.3, hgap=0.6, bottom=0.3,
                             top=1.4 + 0.5 * legend_rows_n)
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
        caption=f"UMAP of the class token of networks pretrained on JetClass-II with 188 classes (upper) and 17 (lower), "
                f"frozen (left) and fine-tuned on {ft} (right), for {jets} in every panel; the plots are qualitative, "
                f"without axis scales. {UMAP_READ[page]}",
        differs="One 2x2 layout for every task (PI, round 3): rows label sets, columns frozen and fine-tuned, the "
                "poster's pairing. " + ("Ten class colours, JetClass's fixed tab10. " if page == "UMAP-jc1" else "tab10 in group order. ")
                + "One legend, first panel. " + (f"UMAP with {u['n_neighbors']} neighbours, minimum distance "
                                                   f"{u['min_dist']}, seed {u['random_state']}.") + (" Absent inputs zero-filled, as in every frozen probe and fine-tuning."
                                                 if page in ("UMAP-top", "UMAP-qg") else ""),
        draw=lambda path: _draw_umap(path, page),
        data_note="From umap_v2 (scripts/build_umap_jobs.py --out-name umap_v2).")




def draw_val(path) -> None:
    """Exact size, one panel across the text width (PI 2026-11-11), the legend above it."""
    docs = M._load(P.DATA / "compute")
    w = S.PRINT_W - 1.5 - 0.2
    fig = S.figure_cm(S.PRINT_W, 1.6 + 0.7 * w + 1.0)
    ax = S.axes_cm(fig, 1.5, 1.6, w, 0.7 * w)
    S.fine_grid(ax)
    for level in S.LEVELS:
        have = [[e["val_acc"] for e in docs[m]["epochs"]] for m in RUNS[level] if m in docs and "val_acc" in docs[m]["epochs"][0]]
        if not have:
            raise C.Absent("no validation accuracy per epoch (compute/)")
        ax.plot(range(1, len(have[0]) + 1), np.mean(have, axis=0), **S.look(level), linewidth=S.LW_CURVE, label=S.LEGEND[level])
    S.heading(fig, "Pretraining on JetClass-II")
    S.labels(ax, "Pretraining epoch", "Validation accuracy")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=0.6, ncol=4)     # above the panel, 188 first, as every figure
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
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and "
                "17 classes and with 17-class random partitions, scored on class-balanced JetClass-II and "
                "JetClass (accuracy) and on top and quark/gluon tagging (rejection R30 at 30% signal efficiency), "
                "against the number of pretraining classes. Markers are means; bars the SD over three pretraining "
                "runs, the range over two for the flavor-boundary partitions; each random partition has one run. "
                "Dashed: no pretraining, trained from scratch.",
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
        caption="Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and with three 17-class "
                "partitions with controlled flavor boundaries, fine-tuned and scored on class-balanced "
                "JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging (rejection R30 at 30% "
                "signal efficiency), against the number of pretraining classes. Markers are means; bars the SD "
                "over three pretraining runs, the range over two for the partitions. Dashed: no pretraining, "
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
        id="F3-main",
        title="Cost class by class, frozen and fine-tuned",
        precedent=HEATMAPS,
        precedent_label="None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the dropped heatmap's",
        caption="Networks pretrained on JetClass-II with 17 and with 188 classes, then (a) a linear probe (LP) on "
                "their frozen features or (b) the whole network fine-tuned (f.t.), on one million class-balanced "
                "JetClass-II jets: each class's area under the curve with 17 classes against the same with 188. "
                "Both panels share one range. Below the dashed line the class does worse with 17 classes. Colors "
                "are decay families; means over three pretraining runs.",
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
                "with 188 classes. Below the dashed line the class does worse than with 188. Colors are decay "
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
                "regression. Below the dashed line the class does worse than with 188. Colors are decay families; "
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
                "dashed line the class does worse than with 188. Colors are decay families; means over three "
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
                "on its frozen features (LP), and a network without pretraining trained from scratch (f.t.), "
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
        id="A8",
        title="Fine-tuned against the number of training jets (appendix)",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Fine-tuned networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, trained and "
                "scored on class-balanced JetClass-II and JetClass (accuracy) and on top and quark/gluon tagging "
                "(R30), against the number of training jets; dashed, no pretraining, trained from scratch. Points "
                "are means, bands the SD over three pretraining runs (from scratch: three trainings). Stars: "
                "published ParT (JetClass: same inputs; top: kinematic; quark/gluon: kinematic and particle "
                "type).",
        differs="A10's form, fine-tuned (PI: the task x model x mode grid).",
        draw=draw_a8,
        data_note="From the fine-tuning read-outs as they stand: the long-budget cells at 10^6 on JetClass and "
                  "JetClass-II, and the small-size benchmark cells still running, replace their points when read out."),
    C.Page(
        id="A10",
        title="Linear probes against the number of training jets",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and "
                "17 classes, trained and scored on class-balanced JetClass-II and JetClass (accuracy) and on top "
                "and quark/gluon tagging (R30), against the number of training jets; dashed, no pretraining, "
                "fine-tuned from scratch (f.t.). Points are means, bands the SD over three pretraining runs (from "
                "scratch: three trainings). Stars: published ParT (JetClass: same inputs; top: kinematic; "
                "quark/gluon: kinematic and particle type).",
        differs="Four panels against Golling's one; four label sets against three training modes. SD bands as "
                "Golling's (PI, 2026-11-07, M1). Label sets in Tol muted, darkest for 188, from scratch dark grey "
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
        caption="Pretraining on JetClass-II with 188, 162, 43 and 17 classes: validation accuracy against epoch, "
                "each network on its own classes, the mean over three runs. A coarser set of labels has fewer "
                "classes to tell apart, so its accuracy is not comparable with a finer set's. The step near epoch "
                "56 is where the learning rate starts to decay.",
        differs="Accuracy, not loss; one curve per label set; no marker for 80 points; theme grid.",
        draw=draw_val),
]
