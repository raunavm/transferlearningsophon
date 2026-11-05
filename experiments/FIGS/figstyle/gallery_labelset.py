#!/usr/bin/env python3
"""The gallery's label-set mock-ups: F2a, F2b, F2-app, F3, F3-app, F8, A10 (docs/figstyle/PLAN.md), in
style_v3 only. The numbers come through the selection code of paper_figures and mlst_results; the drawing
is written here. All work happens inside the draw functions."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import Normalize

import gallery_common as C
from gallery_roc import FAMILIES, family_of
import style_v3 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I

GRID_2X2 = {"left": 0.1, "right": 0.96, "top": 0.92, "bottom": 0.06, "wspace": 0.3, "hspace": 0.2}     # room for repeated y labels, tick labels and titles
HEIGHT_HEAT = 8.0                                 # 188 rows; a 5-row group label at 10 pt still clears its neighbours
LEFT_HEAT = 0.26                                  # the longest group label at 10 pt, so the saved width stays 15 cm
MARKER = 4
Y_ACC, Y_R30 = "Accuracy", r"Rejection ($R_{30}$)"
X_CLASSES, X_JETS = "Number of pretraining classes", "Number of training jets"
X_SCRATCH = 8      # F2: where the network with no pretraining sits; it has the task's classes, not a pretraining count, so the tick names it
LEGEND_PANEL = 1        # JetClass: its curves stay high on the right, so its lower right is empty in F2, F8 and A10
TITLES = {key: title for key, _, _, _, title, _ in P.F2_PANELS}
RANDOM_COLOUR = S.TAB10[7]
HEAT_F3 = ("L162", "R42_Q1", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R")
HEAT_APP = M.RAND
PUBLISHED = {"jc1": (2_000_000, 0.836), "top": (1_200_000, 1602.0), "qg": (1_600_000, 101.2)}     # ParT's published values, as in the paper's benchmark tables
PROBE_COLOUR = S.TAB10[8]   # F8: C1 means 162 classes in every other figure


def _chosen() -> dict:
    return M.chosen_substitutions(I["fill"], RUNS)[0]


def _panels_2x2():
    fig = plt.figure(figsize=S.FOUR_PANELS)
    return fig, fig.subplots(2, 2, gridspec_kw=GRID_2X2)


def _log_y(ax) -> None:
    """R30 on a log axis: 10^k labels when two powers of ten lie in the range, otherwise a few round plain
    numbers (10^k would leave one label or none); minor ticks are never labelled."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) >= 1:
        return
    vals = [m * 10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))) for m in (1, 2, 5)]
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
    ax.set_xlim(5.5, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([X_SCRATCH, 17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: "From\nscratch" if v == X_SCRATCH else f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "center", "right", "left")):   # 162 and 188 lean apart
        lab.set_horizontalalignment(ha)


def _mean(row: str, value):
    v = P._vals(RUNS, row, value)
    return float(np.mean(v)) if v else None


def _draw_f2(path, value_of, groups, absent: str, heading: str, legend_panel: int = LEGEND_PANEL) -> None:
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
            S.fine_grid(ax)
            line = [(x, y) for row, x in P.X_OF.items() if (y := _mean(row, value)) is not None]
            ax.plot(*zip(*line), color=S.TAB10[0], linewidth=S.LW_CURVE, marker="o", markersize=MARKER, label="Hierarchy")
            for label, colour, rows in groups:
                ys = [y for row in rows if (y := _mean(row, value)) is not None]
                if ys:
                    ax.plot([P.X_OF["R16_Q1"]] * len(ys), ys, color=colour, linestyle="none", marker="o",
                            markersize=MARKER, label=label)
            scratch = P.scratch_series(M, I["per"], key, met)
            if scratch:
                ax.plot(X_SCRATCH, float(np.mean(scratch[max(scratch)])), color=S.SERIES["SCRATCH"], marker="o",
                        markersize=MARKER, linestyle="none")
            if met == "r30":
                _log_y(ax)
        _classes_axis(ax)
    S.legend(axes.flat[legend_panel], loc="best")
    S.heading(fig, heading)
    S.save(fig, path)


def _probe_value(key, n, ds, met):
    return P._f2_probe(M, I["fits"], I["fill"], _chosen(), ds, met)


def _ft_value(key, n, ds, met):
    return P._f2_ft(M, I["per"], key, n, met)


def _controlled():
    return [(S.NAME[a], S.SERIES[a], [a]) for a in S.CONTROLS]


def draw_f2a(path) -> None:
    _draw_f2(path, _probe_value, _controlled(), "no linear probes (linprobe/fits_v4)", "Linear probes")


def draw_f2b(path) -> None:
    _draw_f2(path, _ft_value, _controlled(), "no fine-tuning read-outs (ft_mlst, budget)", "After fine-tuning", legend_panel=2)


def draw_f2_app(path) -> None:
    _draw_f2(path, _probe_value, [("Random partitions", RANDOM_COLOUR, M.RAND)], "no linear probes (linprobe/fits_v4)",
             "Linear probes")


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
    fig.subplots_adjust(left=LEFT_HEAT, top=0.95)
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
    S.heading(fig, "Linear probes on JetClass-II, class by class")
    S.save(fig, path)


def _column(arm: str) -> str:
    return S.NAME.get(arm) or dict(P.MK.HEAT)[arm]


def draw_f3(path) -> None:
    _draw_heat(path, HEAT_F3)


def draw_f3_app(path) -> None:
    _draw_heat(path, HEAT_APP)


# ------------------------------------------------------------------ F3-alt: every class against the 188-class model
def _one_minus_auc(models: list) -> dict:
    """{class: 1 - AUC, the mean over the models that hold it}."""
    per = [p for m in models if (p := M._per_class(I["fits"], m))]
    return {c: 1 - float(np.mean([p[c] for p in per if c in p])) for c in set().union(*per)} if per else {}


def _draw_scatter(path, panels: list) -> None:
    """panels: (runs, title). One point per class; above the dashed line the class does worse than with 188 classes."""
    full = _one_minus_auc(RUNS["L188"])
    if not full:
        raise C.Absent("no class-balanced JetClass-II linear probes (linprobe/fits_v4)")
    family = {int(r["jet_label"]): family_of(r["class_name"]) for r in M._class_rows(P.REPO)}
    fig = plt.figure(figsize=S.FOUR_PANELS)
    axes = fig.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={**GRID_2X2, "wspace": 0.1, "hspace": 0.1})
    pairs = [[(full[c], other[c], c) for c in full if c in other and full[c] > 0 and other[c] > 0]
             for other in (_one_minus_auc(models) for models, _ in panels)]
    values = [v for pts in pairs for x, y, _ in pts for v in (x, y)]
    lim = (0.8 * min(values), 1.25 * max(values))
    for k, (ax, pts, (_, title)) in enumerate(zip(axes.flat, pairs, panels)):
        S.fine_grid(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(lim, lim, **S.REFERENCE, label="Same as 188 classes")
        for i, name in enumerate(FAMILIES):
            xy = [(x, y) for x, y, c in pts if family[c] == name]
            if xy:
                ax.plot(*zip(*xy), color=S.TAB10[i], marker="o", markersize=3, linestyle="none", label=name)
        S.title(ax, title)
        S.labels(ax, "1 - AUC, 188 classes" if k >= 2 else None, "1 - AUC, this model" if k % 2 == 0 else None)
    axes[0, 0].set_xlim(lim)
    axes[0, 0].set_ylim(lim)
    S.legend(axes[0, 0], loc="lower right", fontsize=7)
    S.heading(fig, "Linear probes on JetClass-II, class by class")
    S.save(fig, path)


def draw_f3_alt(path) -> None:
    _draw_scatter(path, [(RUNS[a], S.NAME[a]) for a in ("R42_Q1", "R16_Q1", "FLAV_F0", "FLAV_F1")])


def draw_f3_app_alt(path) -> None:
    _draw_scatter(path, [(RUNS["L162"], S.NAME["L162"]), (RUNS["FLAV_F1R"], S.NAME["FLAV_F1R"]),
                         ([m for r in M.RAND for m in RUNS[r]], "Five random partitions"),
                         (RUNS["R16_Q1_MASS"], "17 classes, mass regression")])


# ------------------------------------------------------------------ F8, A10: metric against the number of jets
def _curve(ax, series: dict, colour: str, label: str, band: bool) -> None:
    """Mean over runs with filled circles; with band, Golling's: the SD (range for two runs), same colour, no edge."""
    xs = sorted(series)
    if not xs:
        return
    st = [P.F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    ax.plot(xs, mean, color=colour, linewidth=S.LW_CURVE, marker="o", markersize=MARKER, label=label)
    if band:
        ax.fill_between(xs, mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st]),
                        color=colour, alpha=0.2, linewidth=0)


def _draw_size(path, series_of, band: bool, absent: str, heading: str) -> None:
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
        S.fine_grid(ax)
        for series, colour, label in panel:
            _curve(ax, series, colour, label, band)
        if key in PUBLISHED:
            ax.plot(*PUBLISHED[key], color="#000000", marker="*", markersize=8, linestyle="none", label="ParT, published")
        if met == "r30":
            _log_y(ax)
    S.legend(axes.flat[LEGEND_PANEL], loc="lower right")
    S.heading(fig, heading)
    S.save(fig, path)


def draw_f8(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [(P.ft_series(M, I["per"], RUNS, key, met, "L188"), S.TAB10[0], "Fine-tuned"),
                                   (P.probe_series(M, I, RUNS, ds, met, "L188", chosen), PROBE_COLOUR, "Linear probe"),
                                   (P.scratch_series(M, I["per"], key, met), S.SERIES["SCRATCH"], S.NAME["SCRATCH"])]
    _draw_size(path, series, True, "no fine-tuning read-outs or linear probes", "Pretrained on 188 classes, against training from scratch")


def draw_a10(path) -> None:
    chosen = _chosen()
    series = lambda key, ds, met: [(P.probe_series(M, I, RUNS, ds, met, row, chosen), S.SERIES[row], S.NAME[row])
                                   for row in S.LEVELS] + [(P.scratch_series(M, I["per"], key, met), S.SERIES["SCRATCH"], S.NAME["SCRATCH"])]
    _draw_size(path, series, False, "no linear probes (linprobe/fits_v4)", "Linear probes, per pretraining label set")


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
                "classes are random partitions with controlled flavor boundaries. The leftmost point is a network trained from "
                "scratch. All values are means over runs.",
        differs="Four panels against Hong's one, for the four tasks. Three partitions as markers at 17 classes: Hong "
                "has no controls. From scratch is a named point at the left of the x axis (PI, 2026-11-05), not Hong's horizontal line, which reads as a value at every class count. Circles, not crosses, and a line "
                "through the points, not a smoothed curve. x ticks only at the four class counts (they are the data), "
                "no x minor ticks; x limits beyond the data, where Hong's end on the first and last points. Log y on "
                "the R30 panels, our choice, with y minor ticks Hong lacks. Ticks outward on left and bottom, "
                "not inward on four sides. Panel titles, where Hong has none. Sophon's font and sizes, 1.1 pt line, "
                "landscape's grid and framed 8 pt legend in place of Hong's bold sans-serif and heavy lines. "
                "Legend in one panel only (Sophon Fig. 3). Our axis words, at the axis end, not centred.",
        draw=draw_f2a,
        data_note=f"{TWO_RUNS} {SUBST_NOTE}"),
    C.Page(
        id="F2b",
        title="Label granularity, after fine-tuning",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Fine-tuned models against the number of pretraining classes: accuracy on class-balanced JetClass-II "
                "and on JetClass, and rejection R30 at 30% signal efficiency, on top "
                "tagging and quark/gluon tagging. The line joins the label sets; the markers at 17 "
                "classes (none on JetClass-II) are random partitions with controlled flavor boundaries. The leftmost point is a "
                "network trained from scratch. Values are means over runs; on top tagging the label sets differ by less than the "
                "spread between runs.",
        differs="As F2a, except: markers on three panels, none on JetClass-II. From scratch is a named point at the left "
                "(PI, 2026-11-05). Hong's own figure is fine-tuned too, so this page is the closer copy.",
        draw=draw_f2b,
        data_note="JetClass-II at 1,000,000 class-balanced jets, with no markers: the controlled partitions were not "
                  "fine-tuned on this task. JetClass at 1,000,000 jets, top at 1,200,000, quark/gluon at 1,600,000. "
                  f"{TWO_RUNS} Top: every difference between label sets lies within the run-to-run SD (56 to 151); the "
                  "random-boundary marker lies 133 to 188 below the label sets. Quark/gluon: differences between label "
                  "sets are comparable to the run spread (SD 0.8 to 2.2, up to 1.7 between 43 and 17 classes); the "
                  "no-flavor-boundary marker lies 2.4 and 2.7 above the b-quark and random-boundary markers, more than "
                  "any label-set SD."),
    C.Page(
        id="F2-app",
        title="Random partitions beside the hierarchy",
        precedent=HONG,
        precedent_label="Hong Fig. 2",
        caption="Linear probes against the number of pretraining classes, as in the main-text figure: accuracy on "
                "class-balanced JetClass-II and on JetClass, and rejection R30, the background rejection at 30% "
                "signal efficiency, on top tagging and quark/gluon tagging. The line joins the four nested label "
                "sets; the five markers at 17 classes are random partitions of the 188 classes. The leftmost point is a "
                "network trained from scratch. The line shows means over runs; each marker is one run.",
        differs="As F2a, with the five random partitions as five markers of one colour (C7) at 17 classes in place "
                "of the controlled partitions. An appendix figure, and Hong has no controls.",
        draw=draw_f2_app,
        data_note=f"Each random partition has one run, so a marker is that run, not a mean. {SUBST_NOTE}"),
    C.Page(
        id="F3",
        title="Cost class by class",
        precedent=HEATMAPS,
        precedent_label="CMS Fig. 7, JetParticle-JEPA Fig. 12",
        caption="Difference in per-class AUC between each model and the 188-class model, in percentage points, for "
                "the 188 classes of class-balanced JetClass-II, from linear probes. Columns are the 162-, 43- and 17-class label sets "
                "and the three random partitions with controlled flavor boundaries. Rows are grouped by decay topology, "
                "with the QCD classes at the bottom. Values are means over runs; red marks an AUC below that of "
                "the 188-class model.",
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
        id="F3-app",
        title="Cost class by class, random partitions",
        precedent=HEATMAPS,
        precedent_label="CMS Fig. 7, JetParticle-JEPA Fig. 12",
        caption="Difference in per-class AUC between each of five random partitions into 17 classes and the 188-class "
                "model, in percentage points, for the 188 classes of class-balanced JetClass-II, from linear probes. Rows are grouped "
                "by decay topology as in the main-text figure, with the QCD classes at the bottom, and the color "
                "scale is the same; red marks an AUC below that of the 188-class model. Each column is one run "
                "against the 188-class run of the same index.",
        differs="As F3, with five columns, one per random partition. Same colour scale as F3, so the two can be read "
                "together.",
        draw=draw_f3_app,
        data_note="Each random partition has one run; its column is that run against the 188-class run of the same "
                  "index."),
    C.Page(
        id="F3-alt",
        title="Cost class by class: alternative to the heatmap",
        precedent=["1902.09914/fig5.png"],
        precedent_label="None for this form; shown beside landscape Fig. 5 for the theme",
        caption="Each of the 188 classes of class-balanced JetClass-II, from linear probes: one minus the class's area "
                "under the curve for a coarser label set, against the same for the 188-class model. On the dashed line "
                "the two agree; above it the class does worse than with 188 classes. Colors are decay families. Values "
                "are means over runs.",
        differs="Replaces the heatmap's 188 rows with one point per class, so the 188-class model is drawn as the "
                "reference line. Class names are not readable here; they are in the per-class appendix table.",
        draw=draw_f3_alt),
    C.Page(
        id="F3-app-alt",
        title="Cost class by class: alternative, the other models (appendix)",
        precedent=["1902.09914/fig5.png"],
        precedent_label="None for this form; shown beside landscape Fig. 5 for the theme",
        caption="As the main-text figure, for the 162-class label set, the random partition with a random flavor boundary, "
                "the five plain random partitions averaged, and the 17-class model with mass regression: one minus the "
                "class's area under the curve against the same for the 188-class model, for each of the 188 classes of "
                "class-balanced JetClass-II, from linear probes.",
        differs="As F3-alt.",
        draw=draw_f3_app_alt),
    C.Page(
        id="F8",
        title="Pretraining against the number of training jets",
        precedent=["2401.13537/fig3.png", "2401.13536/fig1.png"],
        precedent_label="Golling Fig. 3, Vigl Fig. 1 top",
        caption="Accuracy on class-balanced JetClass-II and on JetClass, and rejection R30, the background rejection at "
                "30% signal efficiency, on top tagging and quark/gluon tagging, against the number of training jets. "
                "Series are the 188-class model fine-tuned, the same model with a linear probe, and a network trained "
                "from scratch; fine-tuning and from scratch start at 10,000 jets on the two JetClass tasks. Points "
                "are means over runs; bands show the standard deviation over runs. Stars mark published ParT.",
        differs="Four panels, one per task, against Golling's one. Fine-tuned in the 188-class colour, the linear probe "
                "in tab10 C8 (C1 means 162 classes across our figures) and from scratch black, against Golling's green, "
                "red and blue. Sophon's font, landscape's grid and framed legend, "
                "where Golling has no grid and no frame. Lines 1.1 pt against about 0.75. Log y on R30. Band as Golling: "
                "same colour, alpha 0.2, no edge. Against Vigl Fig. 1: no ratio panel, no data-efficiency side panel, no dashed variant, no "
                "second legend.",
        draw=draw_f8,
        data_note="Quark/gluon fine-tuned and from scratch are one point each: the 90 cells at the smaller sizes were "
                  "unfinished at the read-out. On JetClass-II and JetClass the plan's sizes drop 1,000 jets for "
                  "fine-tuning and from scratch, so those curves start at 10,000 by design. Sizes below the full set use the fixed step budget with early "
                  "stopping; the full JetClass, top and quark/gluon points use the full-set recipe. JetClass-II at "
                  "1,000,000: all three from-scratch runs and 3 of 16 pretrained runs reached the 40-epoch limit, so "
                  f"from scratch is probably still improving there. {SUBST_NOTE}"),
    C.Page(
        id="A10",
        title="Linear probes against the number of training jets",
        precedent=["2401.13537/fig3.png"],
        precedent_label="Golling Fig. 3",
        caption="Linear probes of the four label sets against the number of training jets: accuracy on class-balanced "
                "JetClass-II and on JetClass, and rejection R30, the background rejection at 30% signal efficiency, "
                "on top tagging and quark/gluon tagging, with a network trained from scratch in black and published ParT as "
                "stars. Points are means over runs. The colors are those of the label sets in the other figures; the 188- and "
                "162-class curves coincide on JetClass and nearly so on JetClass-II.",
        differs="Four panels against Golling's one; four label sets against three training modes. No bands, where "
                "Golling has them: four would overlap (plan). tab10 in the order of the label sets. Sophon's font, "
                "dashed grid and framed legend, where Golling has no grid and no frame. Log y on R30.",
        draw=draw_a10,
        data_note=SUBST_NOTE),
]
