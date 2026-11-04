#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2a, F2b, F2-app, F3, F3-app, F7, F8, A10 (docs/figstyle/PLAN.md), in
style_v3 only. The numbers come through the selection code of paper_figures and mlst_results; the drawing
is written here. All work happens inside the draw functions."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import Normalize

import gallery_common as C
import style_v3 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I

HEIGHT_2X2 = 4.8
SPACING_2X2 = {"wspace": 0.35, "hspace": 0.3}     # defaults leave no room for repeated y labels, or tick labels and titles
HEIGHT_HEAT = 8.0                                 # 188 rows; a 5-row group label at 10 pt still clears its neighbours
LEFT_HEAT = 0.26                                  # the longest group label at 10 pt, so the saved width stays 15 cm
MARKER = 5
Y_ACC, Y_R30 = "Accuracy", r"Rejection ($R_{30}$)"
X_CLASSES, X_JETS = "Number of pretraining classes", "Number of training jets"
LEGEND_PANEL = 1        # JetClass: its curves stay high on the right, so its lower right is empty in F2, F8 and A10
TITLES = {key: title for key, _, _, _, title, _ in P.F2_PANELS}
RANDOM_COLOUR = S.TAB10[7]
SUBST_COLOUR = dict(zip(P.SUBST, S.TAB10[7:10]))
HEAT_F3 = ("L162", "R42_Q1", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R")
HEAT_APP = M.RAND
PROBE_COLOUR = S.TAB10[8]   # F8: C1 means 162 classes in every other figure


def _chosen() -> dict:
    return M.chosen_substitutions(I["fill"], RUNS)[0]


def _panels_2x2():
    fig = plt.figure(figsize=(S.FULL, HEIGHT_2X2))
    return fig, fig.subplots(2, 2, gridspec_kw=SPACING_2X2)


def _log_y(ax) -> None:
    """R30 on a log axis: 10^k labels when two powers of ten lie in the range, otherwise a few round plain
    numbers (10^k would leave one label or none); minor ticks are never labelled."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) >= 1:
        return
    vals = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ax.yaxis.set_major_locator(ticker.FixedLocator([v for v in vals if lo <= v <= hi]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def _empty_panel(ax) -> None:
    """A panel with no data keeps its frame, title and x axis, so the layout is the final one."""
    ax.set_yticks([])


# ------------------------------------------------------------------ F2: metric against the number of classes
def _classes_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(12, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left")):   # 162 and 188 lean apart
        lab.set_horizontalalignment(ha)


def _mean(row: str, value):
    v = P._vals(RUNS, row, value)
    return float(np.mean(v)) if v else None


def _draw_f2(path, value_of, groups, absent: str) -> None:
    """value_of(key, n, ds, met): a run's value or None for the task; groups: (legend label, colour, rows) of
    the markers at 17 classes, one marker per row."""
    values = [value_of(key, n, ds, met) for key, n, ds, met, *_ in P.F2_PANELS]
    if not any(values):
        raise C.Absent(absent)
    fig, axes = _panels_2x2()
    for k, (ax, value, (key, _, _, met, *_)) in enumerate(zip(axes.flat, values, P.F2_PANELS)):
        S.title(ax, TITLES[key])
        S.labels(ax, X_CLASSES if k >= 2 else None, Y_R30 if met == "r30" else Y_ACC)
        if value is None:
            _empty_panel(ax)
        else:
            S.grid(ax)
            line = [(x, y) for row, x in P.X_OF.items() if (y := _mean(row, value)) is not None]
            ax.plot(*zip(*line), color=S.TAB10[0], linewidth=S.LW, marker="o", markersize=MARKER, label="Hierarchy")
            for label, colour, rows in groups:
                ys = [y for row in rows if (y := _mean(row, value)) is not None]
                if ys:
                    ax.plot([P.X_OF["R16_Q1"]] * len(ys), ys, color=colour, linestyle="none", marker="o",
                            markersize=MARKER, label=label)
            if met == "r30":
                _log_y(ax)
        _classes_axis(ax)
    S.legend(axes.flat[LEGEND_PANEL], loc="lower right")
    S.save(fig, path)


def _probe_value(key, n, ds, met):
    return P._f2_probe(M, I["fits"], I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    return P._f2_ft(M, I["per"], key, n, met)


def _controlled():
    return [(S.NAME[a], S.SERIES[a], [a]) for a in S.CONTROLS]


def draw_f2a(path) -> None:
    _draw_f2(path, _probe_value, _controlled(), "no linear probes (linprobe/fits_v4)")


def draw_f2b(path) -> None:
    _draw_f2(path, _ft_value, _controlled(), "no fine-tuning read-outs (ft_mlst, budget)")


def draw_f2_app(path) -> None:
    _draw_f2(path, _probe_value, [("Random partitions", RANDOM_COLOUR, M.RAND)], "no linear probes (linprobe/fits_v4)")


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
    limit = float(np.nanmax(np.abs(R)))      # the same scale for F3 and F3-app; nothing clips
    n_res = sum(not r["class_name"].startswith("label_QCD") for r in rows)
    row = lambda i: i + (i >= n_res)         # a blank row between the resonant and the QCD classes

    fig = plt.figure(figsize=(S.FULL, HEIGHT_HEAT))
    ax = fig.subplots()
    fig.subplots_adjust(left=LEFT_HEAT)
    cmap = colormaps["RdBu"].copy()
    cmap.set_bad("white")
    im = ax.imshow(np.insert(D, n_res, np.nan, axis=0), aspect="auto", cmap=cmap, norm=Normalize(-limit, limit),
                   interpolation="none")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_column(a) for a in arms], rotation=45, ha="right", rotation_mode="anchor")

    labs = [P.MK.GROUP_TEX.get(r["R16_Q1_name"], r["R16_Q1_name"].replace("_", " ")) for r in rows]
    starts = [i for i in range(len(labs)) if i == 0 or labs[i] != labs[i - 1]]
    ends = starts[1:] + [len(labs)]
    for a in starts[1:]:
        if a != n_res:
            ax.axhline(row(a) - 0.5, color="black", linewidth=0.4)
    ax.set_yticks([(row(a) + row(b - 1)) / 2 for a, b in zip(starts, ends)])
    ax.set_yticklabels([labs[a] for a in starts])
    ax.tick_params(axis="y", length=0)
    S.labels(ax, "Pretraining labels", "JetClass-II class")

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    S.labels(cb.ax, ylabel="AUC difference to 188 classes (percentage points)")
    S.save(fig, path)


def _column(arm: str) -> str:
    return S.NAME.get(arm) or dict(P.MK.HEAT)[arm]


def draw_f3(path) -> None:
    _draw_heat(path, HEAT_F3)


def draw_f3_app(path) -> None:
    _draw_heat(path, HEAT_APP)


# ------------------------------------------------------------------ F7: missing inputs
def draw_f7(path) -> None:
    fits, raw, fill = I["fits"], I["raw"], I["fill"]
    if not raw:
        raise C.Absent("no JetClass-II linear probes with the inputs set to zero (linprobe/fits_v4)")
    panels = (("JetClass-II, top inputs", "Accuracy drop (%)", lambda m, s: M._jc2_drop(raw, fill, m, "top", s)),
              (TITLES["top"], Y_R30, lambda m, s: P._bench_r30(M, fits, fill, m, "top", s)),
              (TITLES["qg"], Y_R30, lambda m, s: P._bench_r30(M, fits, fill, m, "qg", s)))
    fig = plt.figure(figsize=(S.WIDTH, S.HEIGHT[3]))
    axes = fig.subplots(3, 1, sharex=True)
    x = np.arange(len(S.LEVELS))
    width = 0.27
    for ax, (title, ylabel, value) in zip(axes, panels):
        for k, s in enumerate(P.SUBST):
            means = [_mean(row, lambda m: value(m, s)) for row in S.LEVELS]
            ax.bar(x + (k - 1) * width, [np.nan if v is None else v for v in means], width, color=SUBST_COLOUR[s],
                   linewidth=0, label=P.SUBST_NAME[s])
        S.title(ax, title)
        S.labels(ax, ylabel=ylabel)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(top=1.4 * axes[0].get_ylim()[1])       # room for the one-row legend above the bars
    S.legend(axes[0], ncol=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([S.NAME[row] for row in S.LEVELS])
    S.labels(axes[-1], "Pretraining label set")
    S.save(fig, path)


# ------------------------------------------------------------------ F8, A10: metric against the number of jets
def _curve(ax, series: dict, colour: str, label: str, band: bool) -> None:
    """Mean over runs with filled circles; with band, Golling's: the SD (range for two runs), same colour, no edge."""
    xs = sorted(series)
    if not xs:
        return
    st = [P.F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    ax.plot(xs, mean, color=colour, linewidth=S.LW, marker="o", markersize=MARKER, label=label)
    if band:
        ax.fill_between(xs, mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st]),
                        color=colour, alpha=0.2, linewidth=0)


def _draw_size(path, series_of, band: bool, absent: str) -> None:
    """series_of(key, ds, met): [(series, colour, label)] of a panel."""
    panels = [series_of(key, ds, met) for key, ds, met, *_ in P.SIZE_PANELS]
    if not any(s for p in panels for s, *_ in p):
        raise C.Absent(absent)
    fig, axes = _panels_2x2()
    for k, (ax, panel, (key, _, met, *_)) in enumerate(zip(axes.flat, panels, P.SIZE_PANELS)):
        ax.set_xscale("log")
        S.title(ax, TITLES[key])
        S.labels(ax, X_JETS if k >= 2 else None, Y_R30 if met == "r30" else Y_ACC)
        if not any(s for s, *_ in panel):
            _empty_panel(ax)
            continue
        S.grid(ax)
        for series, colour, label in panel:
            _curve(ax, series, colour, label, band)
        if met == "r30":
            _log_y(ax)
    S.legend(axes.flat[LEGEND_PANEL], loc="lower right")
    S.save(fig, path)


def draw_f8(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [(P.ft_series(M, I["per"], RUNS, key, met, "L188"), S.TAB10[0], "Fine-tuned"),
                                   (P.probe_series(M, I, RUNS, ds, met, "L188", chosen), PROBE_COLOUR, "Linear probe"),
                                   (P.scratch_series(M, I["per"], key, met), S.SERIES["SCRATCH"], S.NAME["SCRATCH"])]
    _draw_size(path, series, True, "no fine-tuning read-outs or linear probes")


def draw_a10(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [(P.probe_series(M, I, RUNS, ds, met, row, chosen), S.SERIES[row], S.NAME[row])
                                   for row in S.LEVELS]
    _draw_size(path, series, False, "no linear probes (linprobe/fits_v4)")


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
        caption="Linear probes against the number of pretraining classes: accuracy on class-balanced JetClass-II "
                "and on JetClass, and rejection R30, the background rejection at 30% signal efficiency, on top "
                "tagging and quark/gluon tagging. The line joins the four nested label sets; the markers at 17 "
                "classes are partitions with controlled flavor boundaries. All values are means over runs.",
        differs="Four panels against Hong's one, for the four tasks. Three partitions as markers at 17 classes: Hong "
                "has no controls. No baseline line: we have no single baseline. Circles, not crosses, and a line "
                "through the points, not a smoothed curve. x ticks only at the four class counts (they are the data), "
                "no x minor ticks. Log y on the R30 panels, as the PI decided. Sophon's font, sizes, 1.5 pt line, "
                "dashed grid and framed 8 pt legend in place of Hong's bold sans-serif and heavy lines. "
                "Legend in one panel only (Sophon Fig. 3). Our axis words.",
        draw=draw_f2a,
        data_note=f"{TWO_RUNS} {SUBST_NOTE}"),
    C.Page(
        id="F2b",
        title="Label granularity, after fine-tuning",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Fine-tuned models against the number of pretraining classes: accuracy on class-balanced JetClass-II "
                "and on JetClass, and rejection R30, the background rejection at 30% signal efficiency, on top "
                "tagging and quark/gluon tagging. The line joins the four nested label sets; the markers at 17 "
                "classes are partitions with controlled flavor boundaries. All values are means over runs; on the two "
                "benchmarks the differences lie within the spread between runs.",
        differs="As F2a: four panels, three partitions as markers, no baseline line, circles, ticks at the class "
                "counts, log y on R30, Sophon's font and legend, legend in one panel. Hong's own figure is fine-tuned "
                "too, so this page is the closer copy.",
        draw=draw_f2b,
        data_note="JetClass-II at 1,000,000 class-balanced jets, with no markers: the controlled partitions were not "
                  "fine-tuned on this task. JetClass at 1,000,000 jets, top at 1,200,000, quark/gluon at 1,600,000. "
                  f"{TWO_RUNS} Top and quark/gluon: every difference between label sets lies within the run-to-run SD "
                  "(top 56 to 151, quark/gluon 0.8 to 2.2 on the label sets), except the random boundary on top."),
    C.Page(
        id="F8",
        title="Pretraining against the number of training jets",
        precedent=["2401.13537/fig3.png", "2401.13536/fig1.png"],
        precedent_label="Golling Fig. 3, Vigl Fig. 1 top",
        caption="Accuracy on JetClass-II and JetClass, and rejection R30, the background rejection at 30% signal "
                "efficiency, on top tagging and quark/gluon tagging, against the number of training jets. Series are "
                "the 188-class model fine-tuned, the same model with a linear probe, and a network trained from "
                "scratch. Points are means over runs; bands show the standard deviation over runs.",
        differs="Four panels, one per task, against Golling's one. Fine-tuned in the 188-class colour, the linear probe "
                "in tab10 C8 (C1 means 162 classes across our figures) and from scratch black, against Golling's green, "
                "red and blue. Sophon's font, dashed grid and framed legend, "
                "where Golling has no grid and no frame. Lines 1.5 pt against about 0.75. Log y on R30. Band as Golling: "
                "same colour, alpha 0.2, no edge. Against Vigl Fig. 1: no ratio panel, no data-efficiency side panel, no dashed variant, no "
                "second legend.",
        draw=draw_f8,
        data_note="Quark/gluon fine-tuned and from scratch are one point each until the running cells finish (the "
                  "smaller sets have barely started). Sizes below the full set use the fixed step budget with early "
                  "stopping; the full JetClass, top and quark/gluon points use the full-set recipe. JetClass-II at "
                  "1,000,000: all three from-scratch runs and 3 of 16 pretrained runs reached the 40-epoch limit, so "
                  f"from scratch is still improving there. {SUBST_NOTE}"),
    C.Page(
        id="A10",
        title="Linear probes against the number of training jets",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes of the four label sets against the number of training jets: accuracy on JetClass-II "
                "and JetClass, and rejection R30, the background rejection at 30% signal efficiency, on top tagging "
                "and quark/gluon tagging. Points are means over runs. The colours are those of the label sets in the "
                "other figures, from 188 classes down to 17 classes, and the spread over runs is given in the tables.",
        differs="Four panels against Golling's one; four label sets against three training modes. No bands, where "
                "Golling has them: four would overlap (plan). tab10 in the order of the label sets. Sophon's font, "
                "dashed grid and framed legend, where Golling has no grid and no frame. Log y on R30.",
        draw=draw_a10,
        data_note=SUBST_NOTE),
]
