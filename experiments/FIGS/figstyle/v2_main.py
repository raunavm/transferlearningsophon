#!/usr/bin/env python3
"""The rewrite's five main figures (PI 2026-11-24, LEDGER RW-3: "design; frozen-feature summary; two class pairs, with
the label sets and the random groupings only; fine-tuning against training size; the X->bb / X->bs benchmark"), drawn
from the gallery's parts with the rewrite's names (RW-5) and no flavor-boundary partition in any of them:

    python3 experiments/FIGS/figstyle/v2_main.py [--outdir paper/journal/v2/figures] [--check]

  fig1_design         the label sets, the nesting of one example group (GO-6), pretraining and use
  fig2_frozen         linear probes: accuracy on JetClass-II and JetClass, Rej30% on top and quark/gluon tagging
  fig3_pairs          two class pairs, two panels, six curves (RW-8): the four label sets and the random groupings
  fig4_training_size  fine-tuned against the number of training jets, beside training from scratch
  fig5_resonance      X->bb and X->bs: transfer learning on the frozen features, the output scores, from scratch

and the five of the appendix (figA1 ... figA5, the stems paper/journal/v2/appendix.tex includes):

  figA1  recall of each JetClass-II class under a linear probe, 162, 43 and 17 against 188 classes (GO-9: probes only)
  figA2  the four class pairs Fig. 3 does not show, in its form
  figA3  linear probes against the number of training jets, Table 2's pretrained rows
  figA4  the mass output's response and resolution against jet pT and generator-level mass
  figA5  validation accuracy of the pretraining runs against epoch

Every number is the build's, through the gallery's loaders (gallery_common: the probe fits, the complete fine-tuning
read-out budget/fixed_metrics.json, the curve files); a bar or band is the standard deviation over runs, as in the
tables. --check: figcheck's measured findings per figure (<stem>_check.json) and its print-size rasters.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import gallery_common as C  # noqa: E402
import gallery_labelset as G  # noqa: E402
import gallery_perclass as PC  # noqa: E402
import gallery_physics as PH  # noqa: E402
import gallery_roc as R  # noqa: E402
import lines_v6 as L  # noqa: E402
import style_v5 as S  # noqa: E402

M = C.M
OUT = C.P.REPO / "paper/journal/v2/figures"
# RW-5: one name each; the random groupings of a class pair by whether they merge it (LEDGER PS-5)
NAMES = {"RAND": "17 classes, random grouping", "RAND_MERGED": "17 classes, random grouping, pair merged",
         "RAND_APART": "17 classes, random grouping, pair apart", "L162_MASS": "162 classes + mass regression",
         "R16_Q1_MASS": "17 classes + mass regression", "SCRATCH": "trained from scratch", "SOPHON": "Sophon (released)",
         "INIT_BN": "untrained network"}
OUTPUT_SCORES = "(output scores)"        # Sophon's X->bb discriminant, summed from the network's output scores
REJ30 = r"$\mathrm{Rej}_{30\%}$"         # ParT's notation
SPREAD = "mean ± s.d. over runs"    # RW-8: legends in plain words

# Fig. 2: Table 2's pretrained rows (the untrained network, far below every row, stays in the table) and columns'
# headline metric; Herwig is in the appendix (RW-3)
SUMMARY_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "L162_MASS", "R16_Q1_MASS")
SUMMARY_REFS = (M.SOPHON,)
SUMMARY_PANELS = tuple((key, n, ds, met, {"qg": "Quark/gluon"}.get(key, title)) for key, n, ds, met, title in G.HV_PANELS
                       if key != "qg_herwig")
SUMMARY_X = {"jc2": "Accuracy, JetClass-II", "jc1": "Accuracy, JetClass", "top": f"{REJ30}, top tagging",
             "qg": f"{REJ30}, quark/gluon"}
# Fig. 3 (RW-8: two panels, at most six curves)
PAIR_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND_MERGED", "RAND_APART")
# Fig. 4: the label sets beside training from scratch, the panels of Fig. 2
SIZE_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "SCRATCH")
SIZE_PANELS = tuple((key, ds, met, {"qg": "Quark/gluon"}.get(key, title)) for key, ds, met, title in G.SIZE5
                    if key != "qg_herwig")
SIZE_Y = {"accuracy": "Accuracy", "r30": f"Rejection ({REJ30})"}
# Fig. 5
XBX_NETS = ("L188", "L162", "R42_Q1", "R16_Q1")
X_EPS_B = r"Background efficiency ($\varepsilon_B$)"
# Fig. A1: each coarser label set against 188 classes; Fig. A2: the pairs of gallery_roc.OTHER, two a row
RECALL_ROWS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))
RECALL_W = 6.15                           # cm: a square panel, two a row, the legend in the fourth slot
VAL_LEGEND_W = 5.6                        # cm: "162 classes + mass regression" and its key beside Fig. A5's panel


@contextlib.contextmanager
def names():
    """The rewrite's names in every legend the gallery's parts draw, for the length of a drawing."""
    legend, tag = dict(S.LEGEND), PH.DIRECT_TAG
    S.LEGEND.update(NAMES)
    PH.DIRECT_TAG = OUTPUT_SCORES
    try:
        yield
    finally:
        S.LEGEND.clear()
        S.LEGEND.update(legend)
        PH.DIRECT_TAG = tag


def draw_fig1(path: pathlib.Path) -> None:
    import fig1_design
    fig = fig1_design.build(nested=True)
    fig.exact_size = True
    S.save(fig, path)


def _row_name(row: str) -> str:
    return S.LEGEND[G.HV_KEY.get(row, row)]


FIG2_DOT, FIG2_BAR = 3.4, 1.1     # pt: a smaller mean and a heavier bar, so short bars show beside the point (round 2)


def draw_fig2(path: pathlib.Path) -> None:
    """FIG2_DOT and FIG2_BAR for the length of the drawing (round-2 cold readers: "the error bars are invisible at this
    size"); a bar shorter than the point stays hidden by it, and the caption says so."""
    dot, bar = G.MEAN_DOT, L.BAR
    G.MEAN_DOT, L.BAR = FIG2_DOT, FIG2_BAR
    try:
        _draw_fig2(path)
    finally:
        G.MEAN_DOT, L.BAR = dot, bar


def _draw_fig2(path: pathlib.Path) -> None:
    """gallery_labelset._draw_hv's dot plot (a row a network, its mean over runs and standard deviation; the 188-class
    mean and band down the panel; one relative range about the 188-class mean for the panels of a metric, FG-2) for
    SUMMARY_ROWS and Sophon (released) in SUMMARY_PANELS, two a row, the legend under them."""
    cells = [(key, n, ds, met) for key, n, ds, met, _ in SUMMARY_PANELS]
    gets = [G._probe_value(*cell) for cell in cells]
    order = SUMMARY_ROWS + SUMMARY_REFS
    devs = {}
    for cell, get in zip(cells, gets):
        m = float(np.mean(G._hv_values("L188", get, cell)))
        devs.setdefault(cell[3], []).extend(v / m - 1 for r in order for vals in [G._hv_values(r, get, cell)] if vals
                                            for v in vals + G._sd_ends(vals))
    rel = {met: (min(d), max(d)) for met, d in devs.items()}
    probe = plt.figure()
    rend = probe.canvas.get_renderer()
    width = max(probe.text(0, 0, _row_name(r), fontsize=S.TICK_PT5).get_window_extent(rend).width for r in order)
    plt.close(probe)
    name_x = 0.1
    left = name_x + width / probe.dpi * 2.54 + 0.2
    w = (S.PRINT_W - left - G.HV_RIGHT - G.HV_WGAP) / 2
    rows = {r: float(i) for i, r in enumerate(SUMMARY_ROWS)}
    rows.update({r: len(SUMMARY_ROWS) + 0.5 + i for i, r in enumerate(SUMMARY_REFS)})
    dividers = [len(SUMMARY_ROWS) - 0.25]
    h = max((max(rows.values()) + 1.2) * G.HV_UNIT, G.HV_ASPECT * w)
    top, hgap, legend_h = 0.6, 0.95 + 0.8, 0.9
    height = top + 2 * h + hgap + 1.0 + legend_h
    fig = S.figure_cm(S.PRINT_W, height)
    panels, one = [], False
    for k, ((key, _, _, met, title), cell, get) in enumerate(zip(SUMMARY_PANELS, cells, gets)):
        i, j = divmod(k, 2)
        x, y0 = left + j * (w + G.HV_WGAP), top + i * (h + hgap)
        series = [(r, G._hv_values(r, get, cell), G._hv_colour(r), r in SUMMARY_REFS) for r in order]
        one = one or any(len(vals) == 1 for _, vals, _, _ in series)
        parts, main, bars, hidden = G._hv_panel(fig, x, y0, w, h, rows, dividers, series,
                                                G._hv_values("L188", get, cell), met, rel[met])
        print(f"{path.name} ({'abcd'[k]}): {hidden} of {bars} standard-deviation bars shorter than the point")
        main.set_title(f"({'abcd'[k]}) {title}", fontsize=G.HV_TITLE_PT, pad=4.0)
        S.labels(main, SUMMARY_X[key], None)
        if j == 0:
            main.yaxis.set_major_formatter(ticker.FixedFormatter([_row_name(r) for r in rows]))
            main.tick_params(axis="y", which="both", pad=(left - name_x) / 2.54 * 72, labelsize=S.TICK_PT5)
            for lab in main.get_yticklabels():
                lab.set_horizontalalignment("left")
        panels.append((parts, x, w))
    handles = [Line2D([], [], color="#000000", linewidth=L.BAR, marker="o", markersize=G.MEAN_DOT,
                      markerfacecolor="#000000", markeredgecolor="white", markeredgewidth=0.5),
               (Patch(facecolor=S.SERIES["L188"], alpha=L.BAND, linewidth=0),
                Line2D([], [], color=S.SERIES["L188"], linewidth=S.LW_THIN))]
    labels = [SPREAD, f"188 classes, {SPREAD}"]
    if one:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=G.MEAN_DOT, markerfacecolor="white",
                              markeredgecolor="#000000", markeredgewidth=1.0))
        labels.append("one run")
    fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 0.1 / height), ncol=len(handles),
               frameon=False, fontsize=S.LEGEND_PT5, borderpad=0.2, columnspacing=1.6, handlelength=S.HANDLE)
    G._hv_ticks(fig, panels)
    S.save(fig, path)


def draw_fig3(path: pathlib.Path) -> None:
    """The class-pair form (gallery_roc._roc_grid: rejection against signal efficiency, the ratio to 188 classes under
    each panel) for the two pairs of the old Fig. 5, one row: one y range and one ratio range. Every curve keeps its
    band in the ratio panels (the cold readers of draft e6e3b22: "the curves carry no bands except the 188-class
    one")."""
    _pairs(path, [list(R.MECHANISM)])


def _pairs(path: pathlib.Path, rows: list) -> None:
    docs = R._family("pairs")
    R._roc_grid(path, [[(docs, k, PAIR_NETS, R.PAIR_TITLE[k]) for k in row] for row in rows], w=R.PAIR_W[2],
                aspect=0.72, ratio_scale="log", own_y=True, wgap=R.PAIR_WGAP, left=R.PAIR_LEFT - 0.35, one_x=len(rows) > 1,
                bands=PAIR_NETS)


def _size_figure(path: pathlib.Path, makers: list, nets: tuple, ncol: int) -> None:
    """gallery_labelset._draw_size's panels (the mean over runs against the number of training jets, a band its
    standard deviation) for the series of makers in SIZE_PANELS, two a row, the legend of nets above in ncol columns."""
    rows = -(-len(nets) // ncol)
    w, h = G.SIZE_W, G.SIZE_ASPECT * G.SIZE_W
    top, hgap = S.LEGEND_TOP + 0.45 * rows + 0.6, G.SIZE_HGAP + G.SIZE_XLABEL
    fig = S.figure_cm(S.PRINT_W, top + 2 * h + hgap + 1.0)
    axes, handles = [], {}
    for k, (key, ds, met, title) in enumerate(SIZE_PANELS):
        i, j = divmod(k, 2)
        ax = S.axes_cm(fig, G.SIZE_LEFT + j * (w + G.SIZE_WGAP), top + i * (h + hgap), w, h)
        axes.append(ax)
        panel = [m(key, ds, met) for m in makers]
        ns = [n for series, *_ in panel for n in series]
        ax.set_xscale("log")
        ax.set_xlim(min(ns) / G.SIZE_PAD, max(ns) * G.SIZE_PAD)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, title)
        S.labels(ax, G.X_JETS, SIZE_Y[met])
        S.fine_grid(ax)
        for n_, (series, line, mark, label) in reversed(list(enumerate(panel))):      # 188 classes on top
            G._curve(ax, series, line, mark, label, True, L.SIZE, n_, len(panel))
        if met == "r30":
            G._log_y(ax)
        else:
            S.data_ylim(ax)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    labels = [S.LEGEND[r] for r in nets]
    S.fig_legend(fig, [handles[lab] for lab in labels], top_cm=S.LEGEND_TOP, ncol=ncol, labels=labels)
    S.letter_panels(axes)
    S.save(fig, path)


def draw_fig4(path: pathlib.Path) -> None:
    """Fine-tuned (every point the fixed-schedule cell at its row's recorded rate, gallery_labelset._settled)."""
    _size_figure(path, [G._tuned(r) for r in SIZE_NETS], SIZE_NETS, 3)


def draw_fig5(path: pathlib.Path) -> None:
    """The resonance benchmark as the text quotes it, signal efficiency against background efficiency (both
    logarithmic), from the curves of gallery_physics._xbx_series: the classifier on the frozen features for XBX_NETS,
    the same classifier trained from scratch, and for X->bb the output scores of the label sets that keep it as a
    class. The files hold eps_S/sqrt(eps_B) on the eps_B grid; eps_S is that times sqrt(eps_B). Every curve its band,
    the standard deviation over runs."""
    XR = C.P._xr()
    panels = [PH._xbx_series(XR, sig, XBX_NETS, PH.CLASSIFIER, True, True) for sig, _ in XR.SIGNALS]
    cols = [["L188", "L162"], ["R42_Q1", "R16_Q1", "SCRATCH"], [f"{n} direct" for n in XBX_NETS if XR.direct_ok(M, n)]]
    rows = max(len(c) for c in cols)
    # the gap holds the right panel's y label and its tick numbers ("0.05"), 1 mm clear of the left panel (figcheck)
    fig, grid, _ = S.grid_cm(1, 2, w=6.1, aspect=0.85, top=S.LEGEND_TOP + 0.45 * rows + 0.45, left=1.25, wgap=1.45,
                             bottom=1.15)
    axes, handles, low, xs = list(grid[0]), {}, [], []
    for ax, series in zip(axes, panels):
        for key, label, style, rs in series:
            x, y, _ = C.P.xbx_curves(M, rs)
            eff = y * np.sqrt(x)
            mean = eff.mean(axis=0)
            sd = eff.std(axis=0, ddof=1) if len(eff) > 1 else np.zeros_like(mean)
            ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            ax.plot(x, mean, **{**style, "linewidth": S.LW_CURVE})
            handles.setdefault(key, (Line2D([], [], **{**style, "linewidth": S.LW_CURVE}), label))
            low.append(float(np.min(mean - sd)))
            xs += [float(np.min(x)), float(np.max(x))]
    for ax, what in zip(axes, (r"$X\rightarrow b\bar{b}$", r"$X\rightarrow bs$")):
        S.fine_grid(ax)
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ax.set_yscale("log")
        ax.set_ylim(min(low) / 1.2, 1.25)
        S.log_axis(ax, "x")
        S.log_axis(ax, "y", fine=True)
        S.title(ax, what)
        S.labels(ax, xlabel=X_EPS_B, ylabel=S.X_EPS)
    S.letter_panels(axes)
    blank = (Line2D([], [], linestyle="none", marker="none"), "")
    items = [handles.get(k, blank) if k else blank for c in cols for k in c + [None] * (rows - len(c))]
    S.fig_legend(fig, [h for h, _ in items], top_cm=S.LEGEND_TOP, ncol=len(cols), labels=[t for _, t in items])
    S.save(fig, path)


def draw_figA1(path: pathlib.Path) -> None:
    """gallery_perclass._draw_grid's panel for linear probes only (GO-9): each class's recall under a coarser label
    set against its recall under 188 classes, means over runs with bars of the run spread and the bootstrap over test
    jets, coloured by what the panel's labels merge the class with; a class within three standard deviations of the
    188-class recall is open. Three square panels, two a row, the legend in the fourth slot."""
    get = PC.SOURCES["jc2"][0]
    if not PC._mean(get, "L188"):
        raise C.Absent(PC.ABSENT)

    def side(row):
        m, sd, bt = PC._mean(get, row), PC._sd(get, row), PC._boot_mean("jc2", 0, row)
        return {c: (m[c], float(np.hypot(sd[c], bt.get(c, 0.0)))) for c in m}
    ref = {b: side(b) for b in {b for _, b in RECALL_ROWS}}
    pts = [[(ref[b][c][0], v[0], c, ref[b][c][1], v[1]) for c, v in side(a).items() if c in ref[b]] for a, b in RECALL_ROWS]
    vals = [v for panel in pts for x, y, _, ex, ey in panel for v in (x - ex, y - ey, x + ex, y + ey)]
    pad = 0.04 * (max(vals) - min(vals))
    lim = (min(vals) - pad, max(vals) + pad)
    looks = [PC._look("jc2", a) for a, _ in RECALL_ROWS]
    handles = PC._handles("jc2", set().union(*(kinds for _, kinds in looks)))
    handles += [Line2D([], [], color="#777777", marker="o", markersize=3, markerfacecolor="white", markeredgewidth=0.5,
                       linestyle="none", label=PC.WITHIN), Line2D([], [], **S.REFERENCE, label="equal recall")]
    w, left, wgap, top, hgap = RECALL_W, 1.25, 1.3, 0.6, 1.5
    fig = S.figure_cm(S.PRINT_W, top + 2 * w + hgap + 0.95)
    for k, ((a, b), panel, (look, _)) in enumerate(zip(RECALL_ROWS, pts, looks)):
        i, j = divmod(k, 2)
        ax = S.axes_cm(fig, left + j * (w + wgap), top + i * (w + hgap), w, w)
        S.fine_grid(ax)
        ax.plot(lim, lim, **S.REFERENCE)
        marked = PC._significant("jc2", 0, (a, b))
        for x, y, c, ex, ey in panel:
            colour, marker, _, z = look(c)
            ax.errorbar(x, y, xerr=ex or None, yerr=ey or None, fmt="none", ecolor=colour, elinewidth=0.4, capsize=0,
                        alpha=0.6, zorder=z - 0.1)
            within = marked is not None and c not in marked
            ax.plot(x, y, color=colour, marker=marker, markersize=PC.DOT[2], linestyle="none", zorder=z,
                    markerfacecolor="white" if within else colour, markeredgewidth=0.5 if within else 0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        S.title(ax, f"({'abc'[k]}) {S.LEGEND[a]}")
        S.labels(ax, xlabel=f"Recall, {S.LEGEND[b]}", ylabel=f"Recall, {S.LEGEND[a]}")
    S.slot_legend(fig, handles, left + w + wgap, top + w + hgap, w, w)
    S.save(fig, path)


def draw_figA2(path: pathlib.Path) -> None:
    """Fig. 3's form for the four other pairs (the old F4-other and F4-qcd), one x, y and ratio range."""
    _pairs(path, [list(R.OTHER[:2]), list(R.OTHER[2:])])


def draw_figA3(path: pathlib.Path) -> None:
    """Fig. 4's form for linear probes on the frozen features of Table 2's pretrained rows (the old A10)."""
    _size_figure(path, [G._probe(r) for r in SUMMARY_ROWS], SUMMARY_ROWS, 2)


def draw_figA4(path: pathlib.Path) -> None:
    """The old F9b (gallery_physics._binned_2x2): the mass output of the two models with mass regression beside the
    soft-drop mass, median response and resolution against jet pT and generator-level mass."""
    series = PH._mass_series()
    PH._binned_2x2(path, "all", series, 2, cols=[[PH.GROOMED], [label for *_, label in series]])


def draw_figA5(path: pathlib.Path) -> None:
    """The old VAL (gallery_labelset._draw_val) for Table 2's pretrained rows: each model's validation accuracy on
    its own label set against the pretraining epoch."""
    width, G.VAL_LEGEND_W = G.VAL_LEGEND_W, VAL_LEGEND_W
    try:
        G._draw_val(path, SUMMARY_ROWS)
    finally:
        G.VAL_LEGEND_W = width


# (file stem, its drawing, printed width in cm, figcheck's least panel width in cm)
FIGURES = (("fig1_design", draw_fig1, 13.0, 6.1), ("fig2_frozen", draw_fig2, 15.0, 4.5),
           ("fig3_pairs", draw_fig3, 15.0, 6.1), ("fig4_training_size", draw_fig4, 15.0, 6.1),
           ("fig5_resonance", draw_fig5, 15.0, 6.1),
           ("figA1", draw_figA1, 15.0, 4.5), ("figA2", draw_figA2, 15.0, 6.1), ("figA3", draw_figA3, 15.0, 6.1),
           ("figA4", draw_figA4, 15.0, 6.05), ("figA5", draw_figA5, 15.0, 6.1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="figcheck's findings and print-size rasters beside each figure")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.check:
        import figcheck
        import style_v3
        import style_v4
        spec = {stem: (cm, least) for stem, _, cm, least in FIGURES}
        original = style_v3.save

        def save(fig, path):
            stem = pathlib.Path(path).stem
            rep = figcheck.check(fig, stem, *spec[stem])
            out = original(fig, path)
            figcheck.rasters(pathlib.Path(path).with_suffix(".pdf"), rep["print_cm"])
            (a.outdir / f"{stem}_check.json").write_text(json.dumps(rep, indent=1))
            print(f"{stem}: {rep['n_findings']} measured findings {({k: v for k, v in rep['findings'].items() if v})}")
            return out

        for mod in (style_v3, style_v4, S):
            mod.save = save
    with names():
        for stem, draw, _, _ in FIGURES:
            S.use_style5(simplify=True)
            draw(a.outdir / stem)
            print(f"{stem}: drawn")
    print("V2 FIGURES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
