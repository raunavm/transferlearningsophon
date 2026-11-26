#!/usr/bin/env python3
"""The rewrite's main and appendix figures (PI 2026-11-26, ledger 34.3, after 2026-11-24 RW-3, RW-5, RW-8, GO-6, GO-9),
drawn from the gallery's parts with the rewrite's names (RW-5) and no flavor-boundary partition in any of them:

    python3 experiments/FIGS/figstyle/v2_main.py [--outdir paper/journal/v2/figures] [--check]

  fig1_design         Fig. 1: the label sets, the nesting of one example group (GO-6); one flow of pretraining and use
  fig2_accuracy       Fig. 2: accuracy against the label set, linear probe and fine-tuned, trained from scratch dashed
  fig3_classes        Fig. 3: per-class recall of the 43- and 17-class models against the 188-class one; 1 - AUC of the
                      pairs of classes, one row a pair
  fig5_resonance      Fig. 4: X->bb and X->bs, signal efficiency against background efficiency, the ratio to 188 classes
  fig4_training_size  Fig. 5: fine-tuned JetClass-II and JetClass accuracy against the number of training jets, with
                      the difference to 188 classes
  figA1  recall of each JetClass-II class under a linear probe, 162, 43 and 17 against 188 classes (GO-9: probes only)
  figA2  the pair curves of the resonance pairs; figA6 those of the QCD pairs (ledger 34.3: G2 split in two)
  figA3  linear probes against the number of training jets, Table 2's pretrained rows
  figA4  the mass output's response and resolution against jet pT and generator-level mass
  figA5  validation accuracy of the pretraining runs against epoch
  figA7  fine-tuned top tagging and quark/gluon against the number of training jets (moved from Fig. 5)

Every number is the build's, through the gallery's loaders (gallery_common: the probe fits, the complete fine-tuning
read-out budget/fixed_metrics.json, the curve files, stats/ledger33_exports.json); a bar or band is the standard
deviation over runs, as in the tables. Where curves overlap, a lower panel gives them relative to 188 classes.
--check: figcheck's measured findings per figure (<stem>_check.json) and its print-size rasters.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
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
PERCENT = 100.0                          # PI 2026-11-25 (ledger 32.3): accuracy in percent on every accuracy axis
SUMMARY_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "L162_MASS", "R16_Q1_MASS")     # Table 2's pretrained rows
PAIR_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND_MERGED", "RAND_APART")
SIZE_NETS = ("L188", "L162", "R42_Q1", "R16_Q1", "SCRATCH")
SIZE_PANELS = tuple((key, ds, met, {"qg": "Quark/gluon"}.get(key, title)) for key, ds, met, title in G.SIZE5
                    if key != "qg_herwig")
SIZE_Y = {"accuracy": "Accuracy (%)", "r30": f"Rejection ({REJ30})"}
XBX_NETS = ("L188", "L162", "R42_Q1", "R16_Q1")
X_EPS_B = r"Background efficiency ($\varepsilon_B$)"
RECALL_ROWS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))
VAL_LEGEND_W = 5.6                        # cm: "162 classes + mass regression" and its key beside Fig. A5's panel
LOWER = 0.4                               # a lower panel's height over its panel's (relative to 188 classes)
DOT, BAR = 4.0, 1.0                       # pt: a mean's point and its standard-deviation bar in the point plots


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


@contextlib.contextmanager
def before_save(hook):
    """hook(fig) runs on a figure the gallery's parts draw, before it is saved."""
    save = S.save

    def with_hook(fig, out):
        hook(fig)
        return save(fig, out)
    S.save = with_hook
    try:
        yield
    finally:
        S.save = save


def _pct(value):
    """A value function of gallery_labelset (_hv_values' value(m)) in percent."""
    return lambda m: None if (x := value(m)) is None else PERCENT * x


def _pct_series(maker):
    """A training-size series maker (gallery_labelset._tuned, _probe) with its accuracy panels in percent."""
    def make(key, ds, met):
        series, line, mark, label = maker(key, ds, met)
        if met == "accuracy":
            series = {n: [PERCENT * v for v in vals] for n, vals in series.items()}
        return series, line, mark, label
    return make


def _msd(v) -> tuple:
    v = [float(x) for x in v]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _point(ax, x, y, v, colour, filled=True, marker="o", horizontal=False, single_open=True) -> None:
    """A mean over runs with a bar of their standard deviation; filled False (or, with single_open, a single run) an open
    marker."""
    m, sd = _msd(v)
    if sd:
        kw = {"xerr" if horizontal else "yerr": sd}
        ax.errorbar(x if not horizontal else m, y if horizontal else m, **kw, fmt="none", ecolor=colour,
                    elinewidth=BAR, capsize=0, zorder=2.5)
    open_ = (single_open and len(v) == 1) or not filled
    size = DOT * (S.STAR_SCALE if marker == "*" else 1.0)
    ax.plot([m if horizontal else x], [y if horizontal else m], linestyle="none", marker=marker, markersize=size,
            markerfacecolor="white" if open_ else colour, markeredgecolor=colour, markeredgewidth=0.9, zorder=2.7)


# ------------------------------------------------------------------ Fig. 1
def draw_fig1(path: pathlib.Path) -> None:
    import fig1_design
    fig = fig1_design.build(nested=True)
    fig.exact_size = True
    S.save(fig, path)


# ------------------------------------------------------------------ Fig. 2: accuracy against the label set
ACC_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND")
ACC_X = (0.0, 1.0, 2.0, 3.0, 4.4)
ACC_TICKS = ("188", "162", "43", "17", "17,\nrandom")
ACC_PANELS = (("jc2", 1_000_000, "jc2_188", "JetClass-II"), ("jc1", 1_000_000, "jc1", "JetClass"))
ACC_STACKED = ("jc1",)      # the linear probe and the fine-tuned models far apart: two stacked parts, the axis broken
ACC_MODES = {"ft": ("fine-tuned", "#000000", True), "probe": ("linear probe", "#888888", False)}
ACC_W, ACC_LEFT, ACC_WGAP, ACC_GAP = 6.1, 1.35, 1.45, 0.25     # cm; ACC_GAP between the parts of a broken axis


def accuracy_values(key: str, n: int, ds: str) -> dict:
    """{"probe": {row: runs}, "ft": {row: runs}, "scratch": runs}, in percent: Table 2's linear probes and Table 3's
    fine-tuned cells (at n jets) and the network trained from scratch, through the tables' value functions."""
    cell = (key, n, ds, "accuracy")
    out = {mode: {r: G._hv_values(r, _pct(fn(*cell)), cell) for r in ACC_ROWS}
           for mode, fn in (("probe", G._probe_value), ("ft", G._ft_value))}
    out["scratch"] = [PERCENT * v for v in G._scratch_value(*cell)]
    if not all(out["scratch"]) or not all(v for d in (out["probe"], out["ft"]) for v in d.values()):
        raise C.Absent(f"accuracy against the label set, {key}: a cell without runs")
    return out


def _acc_mode(ax, vals: dict, mode: str) -> list:
    """One mode's points (filled fine-tuned, open linear probe) in each label set's colour, the nested sets joined by a
    thin line; returns the values that set the axis range."""
    _, line, filled = ACC_MODES[mode]
    means = [_msd(vals[r])[0] for r in ACC_ROWS]
    ax.plot(ACC_X[:4], means[:4], color=line, linewidth=0.8, zorder=2.0)
    ends = []
    for x, r in zip(ACC_X, ACC_ROWS):
        _point(ax, x, None, vals[r], S.SERIES[r], filled=filled)
        m, sd = _msd(vals[r])
        ends += [m - sd, m + sd]
    return ends


def _scratch_line(ax, v: list) -> list:
    m, sd = _msd(v)
    ax.axhspan(m - sd, m + sd, color=S.SERIES["SCRATCH"], alpha=L.BAND, linewidth=0, zorder=0.9)
    ax.axhline(m, color=S.SERIES["SCRATCH"], linestyle="--", linewidth=S.LW_CURVE, zorder=1.0)
    return [m - sd, m + sd]


def _ylim(ax, vals: list, pad: float = 0.12) -> None:
    lo, hi = min(vals), max(vals)
    d = (hi - lo or 1.0) * pad
    ax.set_ylim(lo - d, hi + d)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))


def _acc_x(ax, labels: bool) -> None:
    ax.set_xlim(-0.5, ACC_X[-1] + 0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ACC_X))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ACC_TICKS) if labels else ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.axvline((ACC_X[3] + ACC_X[4]) / 2, color=S.GREY_MID, linewidth=S.LW_THIN, zorder=0.5)


def _break_y(upper, lower) -> None:
    """The two slashes of a broken y axis on each side, at the upper part's bottom and the lower part's top."""
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False)
    lower.tick_params(axis="x", which="both", top=False)
    for ax, y in ((upper, 0.0), (lower, 1.0)):
        ax.plot([0, 1], [y, y], transform=ax.transAxes, marker=[(-1, -0.6), (1, 0.6)], markersize=4.0,
                linestyle="none", color="#000000", markeredgewidth=S.LW_THIN, clip_on=False, zorder=5)


def draw_fig2(path: pathlib.Path) -> None:
    """Accuracy (%) against the label set, 188, 162, 43, 17 classes and the random grouping, for the linear probe on the
    frozen features and the fine-tuned models (10^6 jets), the network trained from scratch a dashed line with its
    band; JetClass-II one axis, JetClass two stacked parts, the axis broken between the two modes."""
    h = 0.72 * ACC_W
    top = S.LEGEND_TOP + 0.45 + 0.6
    fig = S.figure_cm(S.PRINT_W, top + h + 1.55)
    for k, (key, n, ds, title) in enumerate(ACC_PANELS):
        vals = accuracy_values(key, n, ds)
        x = ACC_LEFT + k * (ACC_W + ACC_WGAP)
        if key in ACC_STACKED:
            hp = (h - ACC_GAP) / 2
            upper = S.axes_cm(fig, x, top, ACC_W, hp)
            lower = S.axes_cm(fig, x, top + hp + ACC_GAP, ACC_W, hp, sharex=upper)
            lower.set_label("<break>")                  # figcheck: a part of its panel, beyond the break
            _ylim(upper, _acc_mode(upper, vals["ft"], "ft") + _scratch_line(upper, vals["scratch"]))
            _ylim(lower, _acc_mode(lower, vals["probe"], "probe"))
            _break_y(upper, lower)
            upper.tick_params(axis="x", which="both", labelbottom=False)     # shared x: the labels on the lower part only
            parts, main = (upper, lower), upper
            upper.yaxis.set_label_coords(-0.14, -ACC_GAP / 2 / hp)
        else:
            main = S.axes_cm(fig, x, top, ACC_W, h)
            _ylim(main, _acc_mode(main, vals["ft"], "ft") + _acc_mode(main, vals["probe"], "probe")
                  + _scratch_line(main, vals["scratch"]))
            parts = (main,)
        for ax in parts:
            S.fine_grid(ax)
            _acc_x(ax, ax is parts[-1])
        for ax in parts[:-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        S.title(main, f"({'ab'[k]}) {title}")
        S.labels(main, ylabel="Accuracy (%)")
        S.labels(parts[-1], xlabel="Pretraining label set (classes)")
    handles = [Line2D([], [], color=ACC_MODES[m][1], linewidth=0.8, marker="o", markersize=DOT,
                      markerfacecolor="#000000" if ACC_MODES[m][2] else "white", markeredgecolor="#000000",
                      markeredgewidth=0.9) for m in ("ft", "probe")]
    handles.append((Patch(facecolor=S.SERIES["SCRATCH"], alpha=L.BAND, linewidth=0),
                    Line2D([], [], color=S.SERIES["SCRATCH"], linestyle="--", linewidth=S.LW_CURVE)))
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=3,
                 labels=[ACC_MODES["ft"][0], ACC_MODES["probe"][0], S.LEGEND["SCRATCH"]])
    S.save(fig, path)


# ------------------------------------------------------------------ Fig. 3 and Fig. A1: classes and pairs of classes
def _recall(fig, slots: list, rows: tuple) -> list:
    """gallery_perclass._draw_grid's panel for linear probes only (GO-9), one per (compared, reference) of rows at
    slots [(x, top, w)] cm: each class's recall (%) against its recall under the reference, means over runs with bars of
    the run spread and the bootstrap over test jets, coloured by what the panel's labels merge the class with; a class
    within three standard deviations of the reference is open. One range for every panel. Returns the legend's handles."""
    get = PC.SOURCES["jc2"][0]
    if not PC._mean(get, "L188"):
        raise C.Absent(PC.ABSENT)

    def side(row):
        m, sd, bt = PC._mean(get, row), PC._sd(get, row), PC._boot_mean("jc2", 0, row)
        return {c: (PERCENT * m[c], PERCENT * float(np.hypot(sd[c], bt.get(c, 0.0)))) for c in m}
    ref = {b: side(b) for b in {b for _, b in rows}}
    pts = [[(ref[b][c][0], v[0], c, ref[b][c][1], v[1]) for c, v in side(a).items() if c in ref[b]] for a, b in rows]
    vals = [v for panel in pts for x, y, _, ex, ey in panel for v in (x - ex, y - ey, x + ex, y + ey)]
    pad = 0.04 * (max(vals) - min(vals))
    lim = (min(vals) - pad, max(vals) + pad)
    looks = [PC._look("jc2", a) for a, _ in rows]
    for k, ((a, b), panel, (look, _), (x0, y0, w)) in enumerate(zip(rows, pts, looks, slots)):
        ax = S.axes_cm(fig, x0, y0, w, w)
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
        S.title(ax, f"({'abcdefg'[k]}) {S.LEGEND[a]}")
        S.labels(ax, xlabel=f"Recall (%), {S.LEGEND[b]}", ylabel=f"Recall (%), {S.LEGEND[a]}")
    handles = PC._handles("jc2", set().union(*(kinds for _, kinds in looks)))
    return handles + [Line2D([], [], color="#777777", marker="o", markersize=3, markerfacecolor="white",
                             markeredgewidth=0.5, linestyle="none", label=PC.WITHIN),
                      Line2D([], [], **S.REFERENCE, label="equal recall")]


# PI 2026-11-25 (ledger 33.2(b)): X->gg against X->qq, a binary linear probe on the class-balanced test jets; the build
# exports its AUC per model and no curve (stats/ledger33_exports.json)
GG_QQ = C.P.DATA / "stats" / "ledger33_exports.json"
GG_QQ_TITLE = r"$X\to gg$ vs $q\bar{q}$"
GG_QQ_WHERE = {"L188": ("arms", "L188"), "L162": ("arms", "L162"), "R42_Q1": ("arms", "R42_Q1"),
               "R16_Q1": ("arms", "R16_Q1"), "RAND_MERGED": ("random_merged",), "RAND_APART": ("random_apart",)}
# Fig. 3(c): the pairs, one row each, in the family's order with X->gg vs X->qq after the resonance pairs
PAIRS_C = ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq", "gg_vs_qq", "qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light")


def gg_qq_values() -> dict:
    """{network: AUCs of its runs} of X->gg vs X->qq from the export, each list checked against its per-model values."""
    doc = json.loads(GG_QQ.read_text())["gg_vs_qq"]
    per = set(doc["per_model"].values())
    out = {}
    for net, where in GG_QQ_WHERE.items():
        node = doc
        for k in where:
            node = node[k]
        runs = [float(v) for v in node["runs"]]
        if not runs or not set(runs) <= per:
            raise SystemExit(f"FATAL: {GG_QQ.name}: gg_vs_qq {'/'.join(where)} runs not among its per-model AUCs")
        out[net] = runs
    return out


def pair_errors() -> dict:
    """{pair: {network: 1 - AUC of each run}}: the binary linear probe's AUC of the pair curves (pairs family, the
    curves of Figs. A2 and A6) and of X->gg vs X->qq (gg_qq_values); a network with no run of a kind (no random grouping
    keeps a QCD pair apart) is left out."""
    docs, out = R._family("pairs"), {}
    for k in PAIRS_C:
        if k == "gg_vs_qq":
            out[k] = {net: [1 - v for v in runs] for net, runs in gg_qq_values().items()}
            continue
        out[k] = {net: v for net in PAIR_NETS
                  if (v := [1 - float(docs[m]["curves"][k]["auc"]) for m in R._models(docs, net, k)
                            if m in docs and k in docs[m]["curves"]])}
    return out


def pair_merged(k: str, net: str) -> bool:
    """Whether a network's labels put the pair's two sides in one class: the label maps (mlst_results.merge_status,
    "merged"; "partly" is not), the export's status for X->gg vs X->qq, the random groupings by their curve."""
    if net in ("RAND_MERGED", "RAND_APART"):
        return net == "RAND_MERGED"
    if k == "gg_vs_qq":
        return json.loads(GG_QQ.read_text())["gg_vs_qq"]["status_of_label_sets"][net] == "merged"
    return R.PAIR_STATUS[net][k] == "merged"


def _pair_title(k: str) -> str:
    return GG_QQ_TITLE if k == "gg_vs_qq" else R.PAIR_TITLE[k]


def _pairs_panel(ax) -> list:
    """Fig. 3(c): one row a pair, 1 - AUC on a log axis, one marker a network (its shape and colour), dodged within the
    row; the mean over runs with its standard deviation, a single grouping open. Returns the legend's handles."""
    errs = pair_errors()
    dodge = 0.13
    xs = []
    for i, k in enumerate(PAIRS_C):
        for j, net in enumerate(PAIR_NETS):
            if net not in errs[k]:
                continue
            key = R._style(net)
            _point(ax, None, i + (j - (len(PAIR_NETS) - 1) / 2) * dodge, errs[k][net], S.SERIES[key],
                   marker=S.MARKERS.get(net, S.MARKERS[key]), horizontal=True, filled=not pair_merged(k, net),
                   single_open=False)
            m, sd = _msd(errs[k][net])
            xs += [m - sd if m - sd > 0 else m, m + sd]
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.4, max(xs) * 1.4)
    S.log_axis(ax, "x", force=True)
    ax.set_ylim(len(PAIRS_C) - 0.5, -0.5)
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(PAIRS_C))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([_pair_title(k) for k in PAIRS_C]))
    ax.tick_params(axis="y", which="both", length=0, labelsize=S.TICK_PT5)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    for y in np.arange(0.5, len(PAIRS_C) - 0.5):
        ax.axhline(y, color=S.GREY_LIGHT, linewidth=S.LW_THIN, zorder=0.5)
    ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.35, color="#b0b0b0", alpha=0.3)
    ax.set_axisbelow(True)
    S.labels(ax, xlabel=r"$1 - \mathrm{AUC}$ (binary linear probe)")
    return [Line2D([], [], linestyle="none", marker=S.MARKERS.get(net, S.MARKERS[R._style(net)]), markersize=DOT,
                   markerfacecolor=S.SERIES[R._style(net)], markeredgecolor=S.SERIES[R._style(net)],
                   label=R._name(net)) for net in PAIR_NETS] + [
        Line2D([], [], linestyle="none", marker="o", markersize=DOT, markerfacecolor="white", markeredgecolor="#000000",
               markeredgewidth=0.9, label="open: the labels merge the pair")]


FIG3_ROWS = (("R42_Q1", "L188"), ("R16_Q1", "L188"))


def draw_fig3(path: pathlib.Path) -> None:
    """(a, b) Fig. A1's recall panels for the 43- and 17-class models; (c) the pairs of classes (_pairs_panel), the row
    names left of it; the colour legend of (a, b) above them, the network legend of (c) between the rows."""
    w, left, wgap = 6.1, 1.35, 1.45
    top_leg = S.LEGEND_TOP + 0.45 * 3
    y_ab = top_leg + 0.6
    y_leg2 = y_ab + w + 1.05
    y_c = y_leg2 + 0.45 * 2 + 0.75
    xc, wc = 5.1, S.PRINT_W - 5.1 - 0.15
    hc = 0.72 * wc
    H = y_c + hc + 1.05
    fig = S.figure_cm(S.PRINT_W, H)
    handles = _recall(fig, [(left, y_ab, w), (left + w + wgap, y_ab, w)], FIG3_ROWS)
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=3, columnspacing=0.8, handletextpad=0.4)
    ax = S.axes_cm(fig, xc, y_c, wc, hc)
    pair_handles = _pairs_panel(ax)
    S.title(ax, "(c) Pairs of classes")
    fig.legend(handles=pair_handles, loc="upper center", bbox_to_anchor=(0.5, 1 - y_leg2 / H), ncol=4, frameon=False,
               fontsize=S.LEGEND_PT5, borderpad=0.2, labelspacing=0.3, columnspacing=1.0, handletextpad=0.3)
    S.save(fig, path)


def draw_figA1(path: pathlib.Path) -> None:
    """Three square recall panels (162, 43, 17 against 188 classes), two a row, the legend in the fourth slot."""
    w, left, wgap, top, hgap = 6.15, 1.25, 1.3, 0.6, 1.5
    fig = S.figure_cm(S.PRINT_W, top + 2 * w + hgap + 0.95)
    slots = [(left + j * (w + wgap), top + i * (w + hgap), w) for i, j in ((0, 0), (0, 1), (1, 0))]
    handles = _recall(fig, slots, RECALL_ROWS)
    S.slot_legend(fig, handles, left + w + wgap, top + w + hgap, w, w)
    S.save(fig, path)


# ------------------------------------------------------------------ Figs. A2 and A6: the pair curves
A2_RATIO = 0.5          # ledger 34.3: taller ratio panels than the main figures' (0.35)


def _start_at_188(fig) -> None:
    """Each pair panel's x range starts where its 188-class curve does (ledger 34.3), every curve cut there."""
    ref = S.SERIES["L188"].lower()
    for ax in fig.axes:
        if not re.match(r"\([a-z]\) ", ax.get_title()):
            continue
        xs = [float(np.nanmin(ln.get_xdata())) for ln in ax.lines
              if ln.get_color().lower() == ref and ln.get_transform() == ax.transData and len(ln.get_xdata())]
        if not xs:
            continue
        lo = math.floor(min(xs) * 20 + 1e-9) / 20
        pad = 0.02 * (1 - lo)
        group = [o for o in fig.axes if o is ax or ax.get_shared_x_axes().joined(ax, o)]
        for o in group:
            for ln in o.lines:
                if ln.get_transform() == o.transData and len(ln.get_xdata()) > 1:
                    x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                    keep = x >= lo - pad
                    ln.set_data(x[keep], y[keep])
        ax.set_xlim(lo - pad, 1 + pad)
        step = 0.1 if 1 - lo > 0.35 else 0.05
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(step / 2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda t, _, d=1 if step == 0.1 else 2: f"{t:.{d}f}" if t <= 1 + 1e-9 else ""))


def _pair_figure(path: pathlib.Path, keys: tuple) -> None:
    """The class-pair form (gallery_roc._roc_grid: rejection against signal efficiency, the ratio to 188 classes under
    each panel with every curve's band) for three pairs, two a row, the legend in the fourth slot."""
    docs = R._family("pairs")
    grid = [[(docs, k, PAIR_NETS, R.PAIR_TITLE[k]) for k in keys[:2]], [(docs, keys[2], PAIR_NETS, R.PAIR_TITLE[keys[2]]),
                                                                        None]]
    with before_save(_start_at_188):
        R._roc_grid(path, grid, w=R.PAIR_W[2], aspect=0.72, ratio_scale="log", own_y=True, wgap=R.PAIR_WGAP,
                    left=R.PAIR_LEFT - 0.35, legend_slot=(1, 1), bands=PAIR_NETS, ratio=A2_RATIO)


def draw_figA2(path: pathlib.Path) -> None:
    _pair_figure(path, ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq"))


def draw_figA6(path: pathlib.Path) -> None:
    _pair_figure(path, ("qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light"))


# ------------------------------------------------------------------ Fig. 4: the resonance benchmark
def draw_fig5(path: pathlib.Path) -> None:
    """Signal efficiency against background efficiency (both logarithmic) of X->bb and X->bs, from the curves of
    gallery_physics._xbx_series: the classifier on the frozen features for XBX_NETS, the same classifier trained from
    scratch, and for X->bb the output scores of the label sets that keep it as a class; every curve its band. Under each
    panel the ratio to the 188-class transfer learning on a linear scale; a dotted line at 1% background efficiency."""
    XR = C.P._xr()
    panels = [PH._xbx_series(XR, sig, XBX_NETS, PH.CLASSIFIER, True, True) for sig, _ in XR.SIGNALS]
    cols = [["L188", "L162"], ["R42_Q1", "R16_Q1", "SCRATCH"], [f"{n} direct" for n in XBX_NETS if XR.direct_ok(M, n)]]
    rows = max(len(c) for c in cols)
    fig, grid, rgrid = S.grid_cm(1, 2, w=6.1, aspect=0.85, ratio=0.35, top=S.LEGEND_TOP + 0.45 * rows + 0.45,
                                 left=1.25, wgap=1.45, bottom=1.15)
    axes, raxes, handles, low, xs, rlo = list(grid[0]), list(rgrid[0]), {}, [], [], []
    for ax, rx, series in zip(axes, raxes, panels):
        curves = {}
        for key, label, style, rs in series:
            x, y, _ = C.P.xbx_curves(M, rs)
            eff = y * np.sqrt(x)
            curves[key] = (x, eff.mean(axis=0), eff.std(axis=0, ddof=1) if len(eff) > 1 else np.zeros(eff.shape[1]))
            handles.setdefault(key, (Line2D([], [], **{**style, "linewidth": S.LW_CURVE}), label))
        rx_, rm, _ = curves["L188"]
        for key, label, style, _ in reversed(series):
            x, mean, sd = curves[key]
            ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            ax.plot(x, mean, **{**style, "linewidth": S.LW_CURVE})
            low.append(float(np.min(mean - sd)))
            xs += [float(np.min(x)), float(np.max(x))]
            common, i, j = np.intersect1d(np.round(x, 12), np.round(rx_, 12), return_indices=True)
            r, rs_ = mean[i] / rm[j], sd[i] / rm[j]
            rx.fill_between(common, r - rs_, r + rs_, color=style["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            rx.plot(common, r, **{**style, "linewidth": S.LW_CURVE})
            rlo += list(r - rs_) + list(r + rs_)
        for a in (ax, rx):
            a.axvline(1e-2, **S.REFERENCE, zorder=1.5)
        rx.axhline(1.0, color="#000000", linewidth=S.LW_THIN, zorder=1.4)
    for ax, rx, what in zip(axes, raxes, (r"$X\rightarrow b\bar{b}$", r"$X\rightarrow bs$")):
        S.fine_grid(ax)
        S.fine_grid(rx)
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ax.set_yscale("log")
        ax.set_ylim(min(v for v in low if v > 0) / 1.2, 1.25)
        S.log_axis(ax, "x")
        S.log_axis(ax, "y", fine=True)
        S.clear_edge_labels(ax, low=True)            # its lowest label clear of the ratio panel under it
        S.title(ax, what)
        S.labels(ax, ylabel=S.X_EPS)
        S.labels(rx, xlabel=X_EPS_B, ylabel="Ratio")
        rx.tick_params(labelbottom=True)
    v = np.array([x for x in rlo if np.isfinite(x)] + [1.0])
    d = 0.08 * (v.max() - v.min())
    for rx in raxes:
        rx.set_ylim(v.min() - d, v.max() + d)
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        S.clear_edge_labels(rx, high=True)
    S.letter_panels(axes)
    blank = (Line2D([], [], linestyle="none", marker="none"), "")
    items = [handles.get(k, blank) if k else blank for c in cols for k in c + [None] * (rows - len(c))]
    S.fig_legend(fig, [h for h, _ in items], top_cm=S.LEGEND_TOP, ncol=len(cols), labels=[t for _, t in items])
    S.save(fig, path)


# ------------------------------------------------------------------ Fig. 5, Figs. A3 and A7: against training size
def _relative(series: dict, ref: dict, met: str) -> tuple:
    """(n, value, its uncertainty) of a series relative to 188 classes at the sizes both hold: the difference in
    percentage points for accuracy, the ratio for rejection; the two run spreads combined."""
    ns = sorted(set(series) & set(ref))
    a = np.array([_msd(series[n]) for n in ns])
    b = np.array([_msd(ref[n]) for n in ns])
    if met == "accuracy":
        return ns, a[:, 0] - b[:, 0], np.hypot(a[:, 1], b[:, 1])
    r = a[:, 0] / b[:, 0]
    return ns, r, r * np.hypot(a[:, 1] / a[:, 0], b[:, 1] / b[:, 0])


def _size_figure(path: pathlib.Path, makers: list, nets: tuple, panels: tuple, ncol: int) -> None:
    """gallery_labelset._draw_size's panels (the mean over runs against the number of training jets, a band its
    standard deviation) for the series of makers in panels, two a row, the legend of nets above in ncol columns; under
    each panel the label sets relative to 188 classes (_relative), trained from scratch left out of it."""
    rows = -(-len(nets) // ncol)
    w, h = G.SIZE_W, G.SIZE_ASPECT * G.SIZE_W
    lh, nr = LOWER * h, -(-len(panels) // 2)
    top, hgap = S.LEGEND_TOP + 0.45 * rows + 0.6, G.SIZE_HGAP + G.SIZE_XLABEL
    fig = S.figure_cm(S.PRINT_W, top + nr * (h + 0.1 + lh) + (nr - 1) * hgap + 1.0)
    axes, handles = [], {}
    for k, (key, ds, met, title) in enumerate(panels):
        i, j = divmod(k, 2)
        x, y = G.SIZE_LEFT + j * (w + G.SIZE_WGAP), top + i * (h + 0.1 + lh + hgap)
        ax = S.axes_cm(fig, x, y, w, h)
        rx = S.axes_cm(fig, x, y + h + 0.1, w, lh, sharex=ax)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
        panel = [m(key, ds, met) for m in makers]
        ns = [n for series, *_ in panel for n in series]
        ax.set_xscale("log")
        ax.set_xlim(min(ns) / G.SIZE_PAD, max(ns) * G.SIZE_PAD)
        rx.xaxis.set_major_formatter(ticker.FuncFormatter(S.pow10))
        rx.xaxis.set_minor_formatter(ticker.NullFormatter())
        S.title(ax, title)
        S.labels(ax, ylabel=SIZE_Y[met])
        S.labels(rx, G.X_JETS, r"$\Delta$ (points)" if met == "accuracy" else "Ratio")
        for a in (ax, rx):
            S.fine_grid(a)
        ref = panel[nets.index("L188")][0]
        for n_, ((series, line, mark, label), net) in reversed(list(enumerate(zip(panel, nets)))):
            G._curve(ax, series, line, mark, label, True, L.SIZE, n_, len(panel))
            if net not in ("SCRATCH", "L188"):
                rel_n, rel, rel_sd = _relative(series, ref, met)
                L.scan(rx, rel_n, rel, rel_sd, rel_sd, color=line["color"], linestyle=line.get("linestyle", "-"),
                       probe=mark.get("markerfacecolor") == "white", label="_nolegend_", style="C")
        base = 0.0 if met == "accuracy" else 1.0
        rx.axhline(base, color=S.SERIES["L188"], linewidth=S.LW_CURVE, zorder=2.0)
        if met == "r30":
            G._log_y(ax)
        else:
            S.data_ylim(ax)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        S.data_ylim(rx, pad=0.12)
        lo_, hi_ = rx.get_ylim()
        span = hi_ - lo_
        rx.set_ylim(min(lo_, base - 0.1 * span), max(hi_, base + 0.1 * span))      # the 188-class line inside
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        S.clear_edge_labels(rx, high=True)
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    labels = [S.LEGEND[r] for r in nets]
    S.fig_legend(fig, [handles[lab] for lab in labels], top_cm=S.LEGEND_TOP, ncol=ncol, labels=labels)
    S.letter_panels(axes)
    S.save(fig, path)


def draw_fig4(path: pathlib.Path) -> None:
    """Fine-tuned JetClass-II and JetClass (every point the fixed-schedule cell at its row's recorded rate,
    gallery_labelset._settled), beside training from scratch."""
    _size_figure(path, [_pct_series(G._tuned(r)) for r in SIZE_NETS], SIZE_NETS, SIZE_PANELS[:2], 3)


def draw_figA7(path: pathlib.Path) -> None:
    """Fig. 5's form for fine-tuned top tagging and quark/gluon (ledger 34.3: moved from the main figure)."""
    _size_figure(path, [G._tuned(r) for r in SIZE_NETS], SIZE_NETS, SIZE_PANELS[2:], 3)


def draw_figA3(path: pathlib.Path) -> None:
    """Fig. 5's form for linear probes on the frozen features of Table 2's pretrained rows, the four tasks."""
    _size_figure(path, [_pct_series(G._probe(r)) for r in SUMMARY_ROWS], SUMMARY_ROWS, SIZE_PANELS, 2)


# ------------------------------------------------------------------ Figs. A4 and A5
def draw_figA4(path: pathlib.Path) -> None:
    """The old F9b (gallery_physics._binned_2x2): the mass output of the two models with mass regression beside the
    soft-drop mass, median response and resolution against jet pT and generator-level mass."""
    series = PH._mass_series()
    PH._binned_2x2(path, "all", series, 2, cols=[[PH.GROOMED], [label for *_, label in series]])


def draw_figA5(path: pathlib.Path) -> None:
    """The old VAL (gallery_labelset._draw_val's form and data, compute/) for Table 2's pretrained rows: each model's
    validation accuracy (%) on its own label set against the pretraining epoch, the mean over runs and a band of their
    standard deviation; the panel at a one-panel figure's size, the legend beside it."""
    docs, runs = M._load(C.P.DATA / "compute"), M._pooled(C.RUNS)
    top, w = 0.5, S.SINGLE_W - 1.5 - 0.25
    x0 = (S.PRINT_W - (1.5 + w + 0.4 + VAL_LEGEND_W)) / 2 + 1.5
    fig = S.figure_cm(S.PRINT_W, top + 0.85 * w + 1.0)
    ax = S.axes_cm(fig, x0, top, w, 0.85 * w)
    S.fine_grid(ax)
    for row in SUMMARY_ROWS:
        have = PERCENT * np.array([[e["val_acc"] for e in docs[m]["epochs"]] for m in runs[row]
                                   if m in docs and "val_acc" in docs[m]["epochs"][0]])
        if not len(have):
            raise C.Absent(f"no validation accuracy per epoch for {row} (compute/)")
        x, mean = np.arange(1, have.shape[1] + 1), have.mean(axis=0)
        if len(have) > 1:
            sd = have.std(axis=0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=S.SERIES[row], alpha=L.BAND, linewidth=0, zorder=1.0)
        ax.plot(x, mean, **S.look(row), linewidth=S.LW_CURVE, label=S.LEGEND[row])
    S.labels(ax, "Pretraining epoch", "Validation accuracy (%)")
    S.slot_legend(fig, ax.get_legend_handles_labels()[0], x0 + w + 0.4, top, VAL_LEGEND_W, 0.85 * w)
    S.save(fig, path)


# the tallest figure the paper's page holds at the text width with a caption of up to 80 words: A4 less 2.5 cm top and
# bottom margins (24.7 cm), less five 12 pt caption lines and the caption skips (about 3.1 cm); figcheck sets a taller
# figure at this height (as it does at the results book's 18.5 cm), so its text would print smaller
PAPER_MAX_H_CM = 21.6
# (file stem, its drawing, printed width in cm, figcheck's least panel width in cm), in the paper's order
FIGURES = (("fig1_design", draw_fig1, 13.0, 6.1), ("fig2_accuracy", draw_fig2, 15.0, 6.1),
           ("fig3_classes", draw_fig3, 15.0, 6.1), ("fig5_resonance", draw_fig5, 15.0, 6.1),
           ("fig4_training_size", draw_fig4, 15.0, 6.1),
           ("figA1", draw_figA1, 15.0, 4.5), ("figA2", draw_figA2, 15.0, 6.1), ("figA6", draw_figA6, 15.0, 6.1),
           ("figA3", draw_figA3, 15.0, 6.1), ("figA7", draw_figA7, 15.0, 6.1), ("figA4", draw_figA4, 15.0, 6.05),
           ("figA5", draw_figA5, 15.0, 6.1))


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
        figcheck.BOOK_MAX_H_CM = PAPER_MAX_H_CM      # the paper's page, not the results book's
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
