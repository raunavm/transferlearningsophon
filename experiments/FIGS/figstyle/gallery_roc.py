#!/usr/bin/env python3
"""The gallery's ROC mock-ups (docs/figstyle/PLAN.md): FJ, FJ-ctrl, FJ-s, FJ-app, FJ-ctrl-app, A3R, A3R-ft (the ROC family's
multi-class form), the class-pair family F4-mech, F4-mech-ft, F4-other, F4-other-ft, F4-ee, F4-rand, F4-rand-ft, and the
ROC family's two-class form F6, F6-part. Every ROC is
rejection (log, from 1) against signal efficiency (linear, 0 to 1), the mean over runs from
gallery_common.mean_over_runs, on style_v4's grid, with the y axis stopping where the curves stop so
they come in from the top of the frame, and the legend in the free lower left."""
from __future__ import annotations

import collections
import math
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MultipleLocator, NullFormatter, NullLocator

from gallery_bands import rejection_rel_error
from gallery_common import Absent, M, P, Page, ROC, RUNS, mean_over_runs
import lines_v6 as L
import style_v5 as S

# rejection ticks as powers of ten (PI, 2026-11-11: log axes labelled 10^n everywhere)
PLAIN = FuncFormatter(S.pow10)
ROC_LEFT, ROC_WGAP = 1.3, 0.5     # cm: room for the tick labels beside the y label; the right column's y labels are off
GRID = {"left": 0.1, "right": 0.965, "top": 0.92, "bottom": 0.06, "hspace": 0.1}
STACKED_GRID = {"top": 0.98, "bottom": 0.06, "hspace": 0.35}
FAMILIES = ("QCD", "Two-prong quarks and gluons", "Two-prong leptons", "Three-prong", "Four-prong",
            "Lepton-containing")
# one colour per family, from CLASS10 (no network colour): black, vermillion, brown, purple, dark teal, lime; the two
# orange shades of CLASS10[2] and CLASS10[5] no longer meet in one figure, and the pale lime goes to the largest family,
# visible by its number (blind page review 2026-11-13; palette_check.separation >= 10.4 in every pair and vision)
FAMILY_COLOURS = tuple(S.CLASS10[k] for k in (0, 2, 7, 3, 4, 1))


def family_label(f: str) -> str:
    """A family as a legend entry: lower case, as every legend of the paper (QCD stays)."""
    return f if f == "QCD" else f[0].lower() + f[1:]


def family_of(class_name: str) -> str:
    """The topology family of a JetClass-II class (configs/labelmaps/rung_label_maps.v1.csv)."""
    name = class_name.removeprefix("label_")
    if name.startswith("QCD_"):
        return FAMILIES[0]
    if re.fullmatch(r"X_[bcsqg]{2}", name):
        return FAMILIES[1]
    if re.fullmatch(r"X_(ee|mm|tauhtau[emh])", name):
        return FAMILIES[2]
    m = re.fullmatch(r"X_YY_([bcsqg]+)(ee|mm|e|m|ev|mv|tauhtau[emh]|tau[emh]v)?", name)
    if m and m[2]:
        return FAMILIES[5]
    if m and len(m[1]) in (3, 4):
        return FAMILIES[len(m[1])]
    raise ValueError(f"no family for class {class_name!r}")


def _family(fam: str) -> dict:
    docs = ROC.get(fam)
    if not docs:
        raise Absent(f"no ROC curves under curves_v2/{fam}")
    return docs


def _axes_2x2(sharey: bool, wspace: float, top: float = 1.0):
    """Round 4: four 6.5 cm panels (style_v5.grid_cm), the y label once per row, the x label under the lower row."""
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=top, left=ROC_LEFT, wgap=ROC_WGAP)
    if sharey:
        for ax in axes.flat[1:]:
            ax.sharey(axes.flat[0])
    for ax in axes.flat:
        _roc_axes(ax)
    S.labels(axes[0, 0], ylabel=S.Y_REJ)
    S.labels(axes[1, 0], ylabel=S.Y_REJ)
    for ax in axes[1]:
        S.labels(ax, xlabel=S.X_EPS)
    return fig, axes


def _roc_axes(ax, x_step: float = 0.2) -> None:
    """style_v4's faint grid at major y ticks; x ticks every x_step, minor every 0.1. x from 0 to 1 widened by 2% on
    each side, as _x_ticks, so no curve runs along the frame (L7)."""
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    S.fine_grid(ax)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 else f"{v:.1f}"))   # no "0" in the corner
    ax.xaxis.set_minor_locator(MultipleLocator(0.1 if x_step > 0.1 else 0.05))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PLAIN)


def _right_labels_off(axes) -> None:
    """A row whose two panels show one y range gives its tick labels once, on the left (the plain-number labels leave
    no room between the columns); the left panel's "1.0" is left to the right panel's axis, which starts beside it."""
    for left, right in axes:
        left.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 else f"{v:.1f}"))     # 1.0 on every panel
        if np.allclose(left.get_ylim(), right.get_ylim()):
            right.tick_params(labelleft=False)
        else:
            raise SystemExit(f"FATAL: {right.get_title()!r}: its own y range needs its tick labels, with no room for them")


def _y_range(axes) -> None:
    """From just below 1 to just above the highest point of any curve, so every curve is whole inside the frame, its
    end at rejection 1 off the lower frame (PI 2026-11-11)."""
    highest = max(float(np.nanmax(line.get_ydata())) for ax in axes for line in ax.lines)
    for ax in axes:
        ax.set_ylim(0.8, highest * 2.5)        # about 0.4 decade above the highest curve start (L7)


def _legend(ax, **kw):
    kw.setdefault("loc", "lower left")
    return S.legend(ax, handlelength=S.HANDLE, handletextpad=0.4, columnspacing=0.4, **kw)


def _plot(ax, s, **kw) -> None:
    """A mean curve from mean_over_runs; none where no run, or a single grid point, survives the cut."""
    if s is not None and len(s[0]) >= 2:
        ax.plot(s[0], s[1], **kw)


def _jc1_curves(ax, docs: dict, models: list, auc_in_legend: bool) -> float:
    """Ten class-against-the-rest curves, solid (every ROC figure, PI 2026-11-07); returns the mean AUC over classes."""
    aucs = []
    for c in S.JC1_CLASSES:
        s = mean_over_runs(docs, models, c)
        aucs.append(s[2])
        _plot(ax, s, color=S.JC1_COLOUR[c], linewidth=S.LW_CURVE,
              label=f"{S.JC1_TEX[c]} ({s[2]:.3f})" if auc_in_legend else S.JC1_TEX[c])
    return float(np.mean(aucs))


def _class_handles(aucs: dict | None = None) -> list:
    """One handle per JetClass class, in the class colours; with aucs, the AUC after the name."""
    return [Line2D([], [], color=S.JC1_COLOUR[c], linewidth=S.LW_CURVE,
                   label=(f"{S.JC1_TEX[c]} ({_auc(aucs[c])})") if aucs else S.JC1_TEX[c]) for c in S.JC1_CLASSES]


def _auc(v: float) -> str:
    """Three decimals; above 0.9995, where that would print 1.000, ">0.999"."""
    return ">0.999" if v >= 0.9995 else f"{v:.3f}"


def _jc1_one(path: pathlib.Path, models: list) -> None:
    """One panel across the text width, the class legend with each class's AUC above it (PI 2026-11-11: every figure
    the text width; the legend beside the panel left a third of the width empty)."""
    w = S.PRINT_W - 1.5 - 0.2
    h = 0.7 * w
    top = 1.95
    fig = S.figure_cm(S.PRINT_W, top + h + 1.0)
    ax = S.axes_cm(fig, 1.5, top, w, h)
    _roc_axes(ax, x_step=0.2)
    docs = _family("jc1_ft_ovr")
    aucs = {c: mean_over_runs(docs, models, c)[2] for c in S.JC1_CLASSES}
    _jc1_curves(ax, docs, models, auc_in_legend=False)
    _y_range([ax])
    S.labels(ax, S.X_EPS, S.Y_REJ)
    S.fig_legend(fig, _class_handles(aucs), top_cm=S.LEGEND_TOP, ncol=4)
    S.save(fig, path)


def _jc1_four(path: pathlib.Path, family: str, panels: list, legend: bool = True) -> None:
    """panels: (models, title), two a row (an odd last slot left empty). One class legend above the panels, the same in
    every JetClass figure."""
    docs = _family(family)
    # three rows at the two-row aspect overrun the page and the book shrinks the figure (FJ-mass-app): flatter panels,
    # and the class legend in the free sixth slot instead of above (L20: no panel under the aspect rule's 0.7)
    odd = len(panels) % 2
    aspect = 0.7 if len(panels) > 4 else 0.85
    fig, axes, _ = S.grid_cm(math.ceil(len(panels) / 2), 2, aspect=aspect, top=0.6 if odd else 1.9, left=ROC_LEFT,
                             wgap=ROC_WGAP)
    for ax in axes.flat[1:]:
        ax.sharey(axes.flat[0])
    for ax in axes.flat:
        _roc_axes(ax)
    for row in axes:
        S.labels(row[0], ylabel=S.Y_REJ)
        row[1].tick_params(labelleft=False)      # one y range: its labels once, clear of the "1.0" beside them
    for ax in axes[-1]:
        S.labels(ax, xlabel=S.X_EPS)
    for ax, (models, name) in zip(axes.flat, panels):
        auc = _jc1_curves(ax, docs, models, auc_in_legend=False)
        S.title(ax, f"{name}, mean AUC {auc:.3f}")
    _y_range(list(axes.flat)[:len(panels)])
    _right_labels_off(axes)
    if odd:                             # the last slot holds the class legend
        slot = axes.flat[-1]
        slot.set_visible(False)
        x0, y1 = slot.get_position().x0 * S.PRINT_W, (1 - slot.get_position().y1) * fig.get_figheight() / S.CM
        S.slot_legend(fig, _class_handles(), x0, y1, slot.get_position().width * S.PRINT_W,
                      slot.get_position().height * fig.get_figheight() / S.CM, ncol=2)
        S.labels(axes.flat[-3], xlabel=S.X_EPS)     # the panel above the legend is the lowest of its column
    else:
        S.fig_legend(fig, _class_handles(), top_cm=S.LEGEND_TOP, ncol=5)
    S.letter_panels(axes.flat[:len(panels)])
    S.save(fig, path)


def _label_sets() -> list:
    return [(RUNS[level], S.NAME[level]) for level in S.LEVELS]


def _controls() -> list:
    """The three controlled partitions, and the five random partitions together (mean AUC over classes within 3e-4)."""
    return [(RUNS[a], S.NAME[a]) for a in S.CONTROLS] + [([m for r in M.RAND for m in RUNS[r]], "Random partitions")]     # L5's name


def draw_fj(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", _label_sets())


def draw_fj_ctrl(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", _controls())


def draw_fj_s(path: pathlib.Path) -> None:
    models = P._scratch_models(_family("jc1_ft_ovr"))
    if not models:
        raise Absent("no from-scratch model in curves_v2/jc1_ft_ovr")
    _jc1_one(path, models)


def draw_fj_app(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_probe_ovr", _label_sets(), legend=False)     # probe curves lie lower: a legend would cover Z->qq


def _mass_sets() -> list:
    """Each mass network beside its own label set (Brief B)."""
    return [(RUNS[k], S.LEGEND[k][0].upper() + S.LEGEND[k][1:]) for k in ("L162", "L162_MASS", "R16_Q1", "R16_Q1_MASS")]


def draw_fj_mass(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", _mass_sets())


def draw_fj_mass_app(path: pathlib.Path) -> None:
    """The mass networks as linear probes, and the random network (as initialized), the probes' reference."""
    _jc1_four(path, "jc1_probe_ovr", _mass_sets() + [(RUNS["INIT"], "Random network")], legend=False)


def draw_fj_ctrl_app(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_probe_ovr", _controls(), legend=False)


# ------------------------------------------------------------------ the two-class ROC forms: class pairs (F4-*) and top
# and quark/gluon tagging (F6, F6-part). One form (plan "Class pairs" and "ROC curve", PI 2026-11-13): background
# rejection against signal efficiency, each network's mean over its runs a solid curve in its colour (one mode per
# panel, L6), under each panel a ratio panel to the 188-class curve with each curve's band (its SD over runs and the
# 68% binomial uncertainty of its passing background count, in quadrature); the rows of panels named turned on the left.
NETWORKS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND",
            "INIT", "SCRATCH")          # L5: every legend in this order
LABEL_SETS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
PARTITIONS = ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "L188", "R16_Q1")       # 188 and 17 classes for reference
RANDOM = ("RAND", "L188", "R16_Q1")
STYLE_KEY = {"INIT": "INIT_BN"}          # the random network (as initialized, Table 1's row) in style_v5's colour and name
POOLED = M._pooled(RUNS)
# the pairs in the family's one order (plan "Class pairs"): the main-text two, the four other flavor pairs, X->ee vs mumu
PAIR_ORDER = ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq", "qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light", "ee_vs_mm")
PAIR_TITLE = {k: h for k, h, *_ in M.PAIRS}
MECHANISM, OTHER = PAIR_ORDER[:2], PAIR_ORDER[2:6]
RATIO_HEIGHT = 0.35     # a ratio panel's height over its panel's (OmniLearned Fig. 3's form)
RATIO_RANGE_FROM = 0.1  # a ratio is drawn at efficiencies from 0.1: below, a few background jets swing it
RATIO_LOG = 5.0         # a ratio range spanning more than this factor is logarithmic (the 17-class probes on the pairs)
NAME_X = 0.2            # cm: the turned row names
RATIO_WORDS = 2.75      # cm: "Ratio to 188 classes" at 9 pt; a shorter ratio panel says "Ratio" (the caption says to what)
Y_REJ_WORDS = 4.3       # cm: "Background rejection (1/eps_B)" at 9 pt; a shorter panel says "Rejection (1/eps_B)
TITLE_LINE = 0.4        # cm: a second title line
# the legend's columns, one group of networks a column (L5): 188 and 162 classes, 43 and 17 classes, the partitions,
# the references (_legend_cols)
LEGEND_GROUPS = (("L188", "L162", "L162_MASS"), ("R42_Q1", "R16_Q1", "R16_Q1_MASS"),
                 ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND"), ("INIT", "SCRATCH"))


def _legend_cols(present) -> list:
    """The legend's columns: a LEGEND_GROUPS group of the networks present a column, neighbouring groups merged into one
    while they fit the tallest, at most three columns (four of these names are wider than the page: the references
    then go under the partitions)."""
    cols = [[n for n in g if n in present] for g in LEGEND_GROUPS]
    cols = [c for c in cols if c]
    rows = max(len(c) for c in cols)
    k = 0
    while k < len(cols) - 1:
        if len(cols[k]) + len(cols[k + 1]) <= rows:
            cols[k:k + 2] = [cols[k] + cols[k + 1]]
        else:
            k += 1
    return cols[:2] + [sum(cols[2:], [])] if len(cols) > 3 else cols


def _group_legend(fig, handles: dict) -> None:
    """The figure legend above the panels in _legend_cols's columns, blank rows below a short column."""
    cols = _legend_cols(handles)
    rows = max(len(c) for c in cols)
    blank = Line2D([], [], linestyle="none", marker="none")
    items = [(handles[n], S.LEGEND[_style(n)]) if n else (blank, "") for c in cols for n in c + [None] * (rows - len(c))]
    S.fig_legend(fig, [h for h, _ in items], top_cm=S.LEGEND_TOP, ncol=len(cols), labels=[t for _, t in items])


def _two_lines(title: str) -> str:
    """A pair's name on two lines, broken before "vs", for a panel too narrow for one."""
    return title.replace(" vs ", "\nvs ", 1)


def _models(docs: dict, net: str) -> list:
    """The runs of a network in a curve family; from scratch: its trainings in that family."""
    if net == "SCRATCH":
        return P._scratch_models(docs) or [m for m in M.FT_SCRATCH_MODELS if m in docs]
    return POOLED[net]


def _style(net: str) -> str:
    return STYLE_KEY.get(net, net)


def _rel_error(docs: dict, models: list, key: str):
    """Relative uncertainty of mean_over_runs' curve at its points (gallery_bands.rejection_rel_error): the runs'
    rejection and passing background count where every run has at least 20 background jets, mean_over_runs' cut."""
    have = [m for m in models if m in docs and key in docs[m]["curves"]]
    if not have:
        return None
    rej = np.array([M.V4.roc_series(docs[m], key)[1] for m in have])
    ok = np.isfinite(rej).all(axis=0)
    k = np.array([np.asarray(docs[m]["curves"][key]["n_bkg_pass"], float) for m in have])
    return rejection_rel_error(rej[:, ok], k[:, ok], docs[have[0]]["curves"][key]["n_background"])


def _ratio(rx, s, ref, rel=None, **kw) -> list:
    """A curve over the 188-class curve where both have points, from RATIO_RANGE_FROM; with rel (the curve's relative
    uncertainty at its points) a band of that uncertainty around the ratio. The ratios drawn and the band's edges."""
    if s is None or ref is None or len(s[0]) < 2 or len(ref[0]) < 2:
        return []
    common, i, j = np.intersect1d(np.round(s[0], 9), np.round(ref[0], 9), return_indices=True)
    far = common >= RATIO_RANGE_FROM
    if far.sum() < 2:
        return []
    r = s[1][i] / ref[1][j]
    rx.plot(common[far], r[far], **kw)
    if rel is None:
        return list(r[far])
    lo, hi = r * (1 - rel[i]), r * (1 + rel[i])
    rx.fill_between(common[far], lo[far], hi[far], color=kw["color"], alpha=0.2, linewidth=0,
                    zorder=kw.get("zorder", 2) - 0.1)
    return list(r[far]) + list(lo[far]) + list(hi[far])


def _ratio_axis(axes: list, values: list) -> None:
    """One ratio range for axes from every ratio and band edge drawn in them: logarithmic where the values span more
    than RATIO_LOG (labels at 1, 2 and 5 times a power of ten), linear otherwise (the widest round step that leaves two
    labels, none within a tenth of the height of either edge, as F6 had it); 1 always inside."""
    v = np.array([x for x in values if np.isfinite(x)] + [1.0], float)
    if v.size == 1:
        return
    if v[v > 0].max() / v[v > 0].min() > RATIO_LOG:
        lo, hi = v[v > 0].min() / 1.25, v.max() * 1.25
        ticks = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
                 for m in (1, 2, 5) if lo * 1.05 <= m * 10.0 ** e <= hi / 1.05]
        if len([t for t in ticks if abs(np.log10(t) - round(np.log10(t))) < 1e-9]) >= 2:
            ticks = [t for t in ticks if abs(np.log10(t) - round(np.log10(t))) < 1e-9]       # decades where two fit
        for rx in axes:
            rx.set_yscale("log")
            rx.set_ylim(lo, hi)
            rx.yaxis.set_major_locator(FixedLocator(ticks))
            rx.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}"))
            rx.yaxis.set_minor_locator(NullLocator())
        return
    lo, hi = v.min(), v.max()
    pad = 0.08 * (hi - lo or 0.1)
    span = hi - lo + 2 * pad
    a, b = lo - pad + 0.1 * span, hi + pad - 0.1 * span
    inside = lambda q: math.floor(b / q + 1e-9) - math.ceil(a / q - 1e-9) + 1      # noqa: E731
    step = max(q for q in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if inside(q) >= 2)
    for rx in axes:
        rx.set_ylim(lo - pad, hi + pad)
        rx.yaxis.set_major_locator(MultipleLocator(step))
        rx.yaxis.set_major_formatter(FuncFormatter(lambda t, _, a=a, b=b, d=max(0, -int(np.floor(np.log10(step) + 1e-9))):
                                                   f"{t:.{max(d, 1)}f}" if a - 1e-9 <= t <= b + 1e-9 else ""))


def _x_ticks(axes: list, lo: float, w: float, last: bool = True) -> None:
    """x from lo to 1, ticks every 0.1 or 0.2 (the finer while its labels, 0.6 cm each, fit the panel), no label at lo,
    where it would meet the last label of the panel to the left, and (last False) none at 1, where it would meet the
    next panel. The range is widened by 2% of its span on each side, so no curve ends on the frame (L7)."""
    step = 0.1 if (1 - lo) / 0.1 * 0.6 <= w else 0.2
    last = last or step / (1 - lo) * w >= 0.6       # labels a step apart fit, so "1.0" clears the next panel's first
    pad = 0.02 * (1 - lo)
    for ax in axes:
        ax.set_xlim(lo - pad, 1 + pad)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda t, _, lo=lo: "" if abs(t - lo) < 1e-9 or (not last and abs(t - 1) < 1e-9) else f"{t:.1f}"))


def _roc_grid(path: pathlib.Path, grid: list, *, w: float, aspect: float, row_names: tuple = (), wgap: float = 0.25,
              share=lambda i, j: 0, own_y: bool = False, fit_x: bool = True, legend_slot: tuple | None = None) -> None:
    """grid: rows of panels (docs, curve key, networks, title); a panel None is left out (its slot stays empty). Each
    panel w cm wide and aspect * w high, its ratio panel RATIO_HEIGHT of that under it. share(i, j): panels of one key
    share their y range and their ratio range (in a row only the first shows its tick labels unless own_y). fit_x: each
    column's x range fitted to its curves (pairs), else 0 to 1. One legend above the panels for every network drawn,
    a column per group (_group_legend), or (legend_slot (i, j), L20: two panels a row) one column in that free slot.
    Titles with a line break take a second line's room."""
    rows, cols = len(grid), max(len(r) for r in grid)
    left = (0.65 if row_names else 0.1) + 1.25
    h = aspect * w
    rh = RATIO_HEIGHT * h
    handles = {}
    nets = {n for r in grid for p in r if p for n in p[2]}
    legend_rows = max(len(c) for c in _legend_cols(nets))
    two = any("\n" in p[3] for r in grid for p in r if p)
    top = (0.0 if legend_slot else S.LEGEND_TOP + 0.45 * legend_rows) + 0.65 + (TITLE_LINE if two else 0)
    # a panel with the legend's slot under it carries the x label: room for it under that row
    hgap = 1.35 + (TITLE_LINE if two else 0) + (0.45 if legend_slot else 0)
    fig = S.figure_cm(S.PRINT_W, top + rows * (h + 0.1 + rh) + (rows - 1) * hgap + 1.0)
    mains, ratios, values = {}, {}, collections.defaultdict(list)
    for i, row in enumerate(grid):
        y0 = top + i * (h + 0.1 + rh + hgap)
        for j, panel in enumerate(row):
            if panel is None:
                continue
            docs, key, nets, name = panel
            x = left + j * (w + wgap)
            ax = S.axes_cm(fig, x, y0, w, h)
            rx = S.axes_cm(fig, x, y0 + h + 0.1, w, rh, sharex=ax)
            ax.tick_params(labelbottom=False)
            _roc_axes(ax)
            S.fine_grid(rx)
            mains[i, j], ratios[i, j] = ax, rx
            ref = mean_over_runs(docs, POOLED["L188"], key)
            for net in sorted(nets, key=lambda n: -NETWORKS.index(n)):        # 188 classes last, on top
                models = _models(docs, net)
                s = mean_over_runs(docs, models, key)
                if s is None or len(s[0]) < 2:
                    continue       # a network whose runs keep fewer than 20 background jets everywhere: no curve
                z = 1.9 if net in ("SCRATCH", "INIT") else 2.0       # the references under the networks
                # the random flavor boundary often runs on the b-quark boundary's curve (F4-mech (d)): drawn over
                # it and thinner, so both show
                z, lw = (2.05, 0.6 * S.LW_CURVE) if net == "FLAV_F1R" else (z, S.LW_CURVE)
                look = S.look(_style(net), linewidth=lw, zorder=z)
                ax.plot(s[0], s[1], **look)
                # L21: in a panel of more than three networks only 188 classes and the mode's reference keep a band
                rel = _rel_error(docs, models, key) if L.keeps_band(net, len(nets)) else None
                values[share(i, j)] += _ratio(rx, s, ref, rel, **look)
                handles.setdefault(net, Line2D([], [], **S.look(_style(net)), linewidth=lw,
                                               label=S.LEGEND[_style(net)]))
            S.title(ax, name)
    groups = collections.defaultdict(list)
    for ij in mains:
        groups[share(*ij)].append(ij)
    for g, ijs in groups.items():
        drawn = [mains[ij] for ij in ijs if mains[ij].lines]
        if drawn:
            _y_range(drawn)
            for ij in ijs:
                mains[ij].set_ylim(drawn[0].get_ylim())
        _ratio_axis([ratios[ij] for ij in ijs], values[g])
    for j in range(cols):
        col = [(i, j) for i in range(rows) if (i, j) in mains]
        lines = [ln for ij in col for ln in mains[ij].lines]
        lowest = min((float(np.nanmin(ln.get_xdata())) for ln in lines), default=0.0) if fit_x else 0.0
        lo = np.floor(lowest * 10 - 1e-9) / 10 if lowest >= 0.1 else 0.0
        _x_ticks([mains[ij] for ij in col], lo, w, last=own_y or j == cols - 1)
    for (i, j), ax in mains.items():
        first = j == 0 or share(i, j) != share(i, j - 1) or own_y
        ax.tick_params(labelleft=first)
        ratios[i, j].tick_params(labelleft=first)
        if first:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: "" if abs(t - 1) < 1e-9 else S.pow10(t)))
        if j == 0:
            # the full label is about 4.6 cm at 9 pt: a shorter panel says "Rejection", so it stays beside its panel
            S.labels(ax, ylabel=S.Y_REJ if h >= Y_REJ_WORDS else r"Rejection ($1/\varepsilon_B$)")
            S.labels(ratios[i, j], ylabel="Ratio to 188 classes" if rh + 0.3 >= RATIO_WORDS else "Ratio")
        if i == rows - 1 or (i + 1, j) not in mains:
            S.labels(ratios[i, j], xlabel=S.X_EPS)
    for i, name in enumerate(row_names):
        y0 = top + i * (h + 0.1 + rh + hgap)
        fig.text(NAME_X / S.PRINT_W, 1 - (y0 + (h + rh) / 2) / (fig.get_figheight() / S.CM), name, rotation=90,
                 ha="center", va="center", fontsize=S.LABEL_PT5)
    if legend_slot:
        i, j = legend_slot
        order = sorted(handles, key=NETWORKS.index)
        S.slot_legend(fig, [handles[n] for n in order], left + j * (w + wgap), top + i * (h + 0.1 + rh + hgap), w,
                      h + 0.1 + rh, ncol=1 if len(order) <= 8 else 2, labels=[S.LEGEND[_style(n)] for n in order])
    else:
        _group_legend(fig, handles)
    S.letter_panels([mains[ij] for ij in sorted(mains)])
    S.save(fig, path)


def _pair_grid(path: pathlib.Path, family: str, pairs: tuple, w: float, aspect: float) -> None:
    """Columns the pairs, upper row the label sets with the mass networks (fine-tuned: and from scratch), lower row the
    flavor-boundary partitions with 188 and 17 classes for reference; one y range for the figure, one ratio range a row."""
    docs = _family(family)
    ref = ("SCRATCH",) if family == "pairs_ft" else ()
    name = (lambda t: _two_lines(t)) if w < 3.5 else (lambda t: t)
    _roc_grid(path, [[(docs, k, LABEL_SETS + ref, name(PAIR_TITLE[k])) for k in pairs],
                     [(docs, k, PARTITIONS, name(PAIR_TITLE[k])) for k in pairs]],
              w=w, aspect=aspect, row_names=("Label sets", "17-class partitions"), share=lambda i, j: i)


PAIR_W = {2: 6.3, 3: 4.15, 4: 3.0}      # cm: a panel of two, three or four in a row beside the turned row names


def draw_f4_mech(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs", MECHANISM, PAIR_W[2], 0.72)


def draw_f4_mech_ft(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs_ft", MECHANISM, PAIR_W[2], 0.72)


# L20 (PI 2026-11-13): two panels a row, so the four other flavor pairs are two figures of F4-mech's form, the
# resonance pair and QCD bb vs cc (F4-other) and QCD with heavy quarks vs light (F4-light)
def draw_f4_other(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs", OTHER[:2], PAIR_W[2], 0.72)


def draw_f4_other_ft(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs_ft", OTHER[:2], PAIR_W[2], 0.72)


def draw_f4_light(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs", OTHER[2:], PAIR_W[2], 0.72)


def draw_f4_light_ft(path: pathlib.Path) -> None:
    _pair_grid(path, "pairs_ft", OTHER[2:], PAIR_W[2], 0.72)


def draw_f4_rand(path: pathlib.Path, family: str = "pairs", keys: tuple = PAIR_ORDER[:3]) -> None:
    """The pooled random partitions with 188 and 17 classes for reference on the six pairs any network leaves curves on
    (every random partition separates X->ee vs mumu completely, the caption says so): L20, two a row, so two figures,
    the three resonance pairs (F4-rand) and the three QCD pairs (F4-rand-qcd), the legend in the fourth slot."""
    docs = _family(family)
    _roc_grid(path, [[(docs, k, RANDOM, PAIR_TITLE[k]) for k in keys[:2]], [(docs, keys[2], RANDOM, PAIR_TITLE[keys[2]])]],
              w=PAIR_W[2], aspect=0.72, share=lambda i, j: 0, legend_slot=(1, 1))


def draw_f4_ee(path: pathlib.Path) -> None:
    """X->ee vs mumu, linear probes read from every test jet of the pair (pairs_sib2): the networks that keep a curve
    (43, 17 and 17 + mass classes, the random network). 188 classes separate it completely, so there is no ratio
    panel's reference; fine-tuned, every network separates it completely (the caption says so)."""
    docs = ROC.get("pairs_sib2") or _family("pairs")
    nets = [n for n in LABEL_SETS + PARTITIONS[:3] + ("RAND", "INIT")
            if (s := mean_over_runs(docs, _models(docs, n), "ee_vs_mm")) is not None and len(s[0]) >= 2]
    if not nets:
        raise Absent("no curve of X->ee vs X->mumu")
    w = S.SINGLE_W - 1.5 - 0.25
    fig = S.figure_cm(S.SINGLE_W, S.LEGEND_TOP + 0.45 * math.ceil(len(nets) / 2) + 0.55 + 0.85 * w + 1.0)
    ax = S.axes_cm(fig, 1.5, S.LEGEND_TOP + 0.45 * math.ceil(len(nets) / 2) + 0.55, w, 0.85 * w)
    _roc_axes(ax)
    for net in sorted(nets, key=lambda n: -NETWORKS.index(n)):
        s = mean_over_runs(docs, _models(docs, net), "ee_vs_mm")
        ax.plot(s[0], s[1], **S.look(_style(net)), linewidth=S.LW_CURVE, label=S.LEGEND[_style(net)],
                zorder=1.9 if net == "INIT" else 2.0)
    _y_range([ax])
    _x_ticks([ax], 0.0, w)
    S.labels(ax, S.X_EPS, S.Y_REJ)
    h, lab = ax.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda k: NETWORKS.index(next(n for n in nets if S.LEGEND[_style(n)] == lab[k])))
    S.fig_legend(fig, [h[k] for k in order], top_cm=S.LEGEND_TOP, ncol=2)
    S.save(fig, path)


BENCH = (("top", "Top tagging"), ("qg", "Quark/gluon, Pythia"), ("qg_herwig", "Quark/gluon, Herwig"))


def _bench_grid(path: pathlib.Path, mode: str, nets: tuple) -> None:
    """One mode a figure (L20: two panels a row; the two modes in one figure would overrun the page): top tagging and
    quark/gluon on Pythia over quark/gluon on Herwig and the legend; top on its own y range, the two quark/gluon panels
    on one (PI: matching axes); every panel its tick labels."""
    fams = {"probe": (_family("top_probe"), _family("qg_probe")), "ft": (_family("top_ft"), _family("qg_ft"))}
    panels = [(fams[mode][0 if key == "top" else 1], key, nets, name) for key, name in BENCH]
    _roc_grid(path, [panels[:2], panels[2:]], w=6.3, aspect=0.72, wgap=0.85, share=lambda i, j: int((i, j) != (0, 0)),
              own_y=True, fit_x=False, legend_slot=(1, 1))


def draw_f6(path: pathlib.Path) -> None:
    """The label sets and the mass networks as linear probes (the random network beside the probes waits for its
    curves, L14); F6-ft: fine-tuned, from scratch beside them."""
    _bench_grid(path, "probe", LABEL_SETS)


def draw_f6_ft(path: pathlib.Path) -> None:
    _bench_grid(path, "ft", LABEL_SETS + ("SCRATCH",))


def draw_f6_part(path: pathlib.Path) -> None:
    _bench_grid(path, "probe", PARTITIONS + ("RAND",))


def draw_f6_part_ft(path: pathlib.Path) -> None:
    _bench_grid(path, "ft", PARTITIONS + ("RAND",))


EVERY_CLASS_LW = 1.0     # 188 curves a panel: thin, opaque, and the legend handles as thin


def _jc2_every_class(path: pathlib.Path, family: str, panels: list | None = None) -> None:
    """panels: (models, title), four in a 2 x 2 grid or one across the text width (from scratch, as FJ-s); the label
    sets by default."""
    docs = _family(family)
    panels = panels or [(RUNS[level], S.NAME[level]) for level in S.LEVELS]
    names = {r["jet_label"]: r["class_name"] for r in M._class_rows(P.REPO)}
    colour = {label: FAMILY_COLOURS[FAMILIES.index(family_of(name))] for label, name in names.items()}
    if len(panels) == 1:
        w = S.PRINT_W - 1.5 - 0.2
        fig = S.figure_cm(S.PRINT_W, 1.9 + 0.7 * w + 1.0)
        axes = np.array([[S.axes_cm(fig, 1.5, 1.9, w, 0.7 * w)]])
        _roc_axes(axes[0, 0])
        S.labels(axes[0, 0], S.X_EPS, S.Y_REJ)
    else:
        fig, axes = _axes_2x2(sharey=True, wspace=0.1, top=1.9)
    for ax, (models, name) in zip(axes.flat, panels):
        aucs = []
        size = collections.Counter(colour.values())
        # the largest families first, underneath, so that the small ones stay visible on top
        for label in sorted(names, key=lambda k: -size[colour[k]]):
            s = mean_over_runs(docs, models, label)
            aucs.append(s[2])
            _plot(ax, s, color=colour[label], linewidth=EVERY_CLASS_LW)
        S.title(ax, f"{name}, mean AUC {np.mean(aucs):.3f}")
    _y_range(list(axes.flat))
    if len(panels) > 1:
        _right_labels_off(axes)
    S.fig_legend(fig, [Line2D([], [], color=FAMILY_COLOURS[i], linewidth=EVERY_CLASS_LW, label=family_label(f)) for i, f in enumerate(FAMILIES)],
                 top_cm=S.LEGEND_TOP, ncol=3)        # above the panels, off the curves (PI, 2026-11-07)
    if len(panels) > 1:
        S.letter_panels(axes.flat)
    S.save(fig, path)


def draw_a3r(path: pathlib.Path) -> None:
    _jc2_every_class(path, "jc2_ovr")


def draw_a3r_ft(path: pathlib.Path) -> None:
    _jc2_every_class(path, "jc2_ft_ovr")


def draw_a3r_ctrl_ft(path: pathlib.Path) -> None:
    """The 17-class partitions fine-tuned (the long-budget cells, gallery_common.ROC), as FJ-ctrl."""
    _jc2_every_class(path, "jc2_ft_ovr", _controls())


def draw_a3r_s(path: pathlib.Path) -> None:
    """Training from scratch on JetClass-II (the full-schedule cells, gallery_common.ROC), as FJ-s."""
    models = sorted(m for m in _family("jc2_ft_ovr") if m.startswith("scratch-s"))     # the full-schedule cells
    if not models:
        raise Absent("no from-scratch model in curves_v2/jc2_ft_ovr")
    _jc2_every_class(path, "jc2_ft_ovr", [(models, "From scratch")])



PAIR_DIFFERS = ("The family's form (PI 2026-11-13): columns the pairs in one order, upper row the label sets with the mass "
                "networks, lower row the flavor-boundary partitions with 188 and 17 classes, a ratio panel to 188 classes "
                "under each panel with each curve's band; every curve solid (one mode a figure).")
PAIR_NOTE = ("A network whose runs keep fewer than 20 background jets at every efficiency has no curve. Means over runs: "
             "three per label set, two per mass network and flavor-boundary partition.")
SUBST_NOTE = ("Top-tagging linear probes fill the missing inputs with random draws and quark/gluon probes with the "
              "training mean, the substitution chosen on validation.")

PAGES = [
    Page(id="FJ", title="JetClass after fine-tuning",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="The networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, one per panel, "
                 "fine-tuned on one million JetClass jets: background rejection against signal efficiency with "
                 "each JetClass class as signal against the other nine, the mean over three pretraining runs. "
                 "Curves end where fewer than 20 background jets remain. Titles give the mean area under the curve "
                 "over the ten classes.",
         differs="Against landscape: ten classes of one network, not fifteen networks; solid tab10. Against the poster: "
                 "rejection against efficiency (PI); four panels, not two; legend names only, as the AUCs do not fit.",
         draw=draw_fj),
    Page(id="FJ-ctrl", title="JetClass after fine-tuning, random partitions",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Networks pretrained on JetClass-II with 17-class partitions, with no flavor boundary, a b-quark "
                 "boundary or a random flavor boundary (two runs each) and five random partitions (one run each, "
                 "averaged), fine-tuned on one million JetClass jets: rejection against efficiency for each class "
                 "against the other nine. Curves end where fewer than 20 background jets remain. Titles give the mean "
                 "area under the curve.",
         differs="As FJ. The five plain random partitions share one panel, to stay at four: their mean AUCs over classes lie within "
                 "3e-4 of each other. Eight panels, one per partition, is the alternative.",
         draw=draw_fj_ctrl),
    Page(id="FJ-s", title="JetClass trained from scratch",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (AUC in legend)",
         caption="A network trained from scratch on one million JetClass jets: background "
                 "rejection against signal efficiency with each of the ten JetClass classes as signal and the "
                 "other nine as background, the mean over three trainings. Curves end where fewer than 20 background "
                 "jets remain. The legend gives each class's area under the curve.",
         differs="As FJ, one panel; the class AUCs in a legend beside it on the right (PI), full width.",
         draw=draw_fj_s),
    Page(id="FJ-app", title="JetClass with linear probes (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes, trained on one million JetClass jets: background rejection against signal "
                 "efficiency with each class as signal against the other nine, the mean over three pretraining "
                 "runs. Curves end where fewer than 20 background jets remain. Titles give the mean area under the "
                 "curve.",
         differs="As FJ, with no legend: probe curves lie lower, so it would cover Z to qq; the colors are FJ's. The "
                 "poster's left panel is also a frozen network, with an MLP on it; here the probe is linear.",
         draw=draw_fj_app),
    Page(id="FJ-ctrl-app", title="JetClass with linear probes, random partitions (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 17-class "
                 "partitions, with no flavor boundary, a b-quark boundary or a random flavor boundary (two runs "
                 "each) and five random partitions (one run each, averaged), trained on one million JetClass "
                 "jets: rejection against efficiency for each class against the other nine. Curves end where fewer "
                 "than 20 background jets remain. Titles give the mean area under the curve.",
         differs="As FJ-ctrl, with no legend, as FJ-app. The five plain partitions' mean AUCs over classes lie within 5e-4 of each other.",
         draw=draw_fj_ctrl_app),
    Page(id="FJ-mass", title="JetClass after fine-tuning, mass networks",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="",
         differs="FJ's form for the two mass networks, each beside its own label set (Brief B, round-1 fix 2026-11-13).",
         data_note="Reads curves_v2/jc1_ft_ovr (mtx-l162mass-s1/2, mtx-r16q1mass-s1/2).",
         draw=draw_fj_mass),
    Page(id="FJ-mass-app", title="JetClass with linear probes, mass networks and the random network",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="",
         differs="FJ-mass as linear probes, with the random network (as initialized) as a fifth panel; no legend, as "
                 "FJ-app.",
         data_note="Reads curves_v2/jc1_probe_ovr (the mass networks, init-s1 to s3).",
         draw=draw_fj_mass_app),
    Page(id="F4-mech", title="Class pairs, linear probes, X→bb̄ vs cc̄ and X→bc vs bq, cs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs=PAIR_DIFFERS,
         data_note=PAIR_NOTE,
         draw=draw_f4_mech),
    Page(id="F4-mech-ft", title="Class pairs, fine-tuned, X→bb̄ vs cc̄ and X→bc vs bq, cs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-mech's form after fine-tuning; from scratch (grey) in the upper row.",
         data_note="Reads curves_v2/pairs_ft (the fine-tuned networks on natural-mix test jets: the signal side's summed "
                   "softmax over the pair's summed softmax). " + PAIR_NOTE,
         draw=draw_f4_mech_ft),
    Page(id="F4-other", title="Class pairs, linear probes, the four other flavor pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-mech's form on X→YY→bb̄qq vs cc̄qq and QCD bb̄ vs cc̄ (PI 2026-11-13: X→ee vs μμ to the "
                 "supplement); L20: two 6.3 cm panels a row, so the QCD pairs with b and with c against light are "
                 "F4-light.",
         data_note=PAIR_NOTE,
         draw=draw_f4_other),
    Page(id="F4-light", title="Class pairs, linear probes, QCD with heavy quarks vs light",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-mech's form on QCD with b vs light and QCD with c, no b, vs light (L20: the second half of the "
                 "old four-pair F4-other).",
         data_note=PAIR_NOTE,
         draw=draw_f4_light),
    Page(id="F4-other-ft", title="Class pairs, fine-tuned, the four other flavor pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-other's form after fine-tuning (new: the fine-tuned counterpart of F4-other); from scratch in the "
                 "upper row.",
         data_note="Reads curves_v2/pairs_ft. " + PAIR_NOTE,
         draw=draw_f4_other_ft),
    Page(id="F4-light-ft", title="Class pairs, fine-tuned, QCD with heavy quarks vs light",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-light's form after fine-tuning; from scratch in the upper row.",
         data_note="Reads curves_v2/pairs_ft. " + PAIR_NOTE,
         draw=draw_f4_light_ft),
    Page(id="F4-ee", title="Class pairs, X→ee vs μμ, linear probes",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="One panel: the networks that keep a curve. No ratio panel, since 188 classes separate the pair "
                 "completely; no fine-tuned panel, since every fine-tuned network does.",
         data_note="Read from every test jet of the pair (pairs_sib2), so a curve can reach a rejection of 6,442. The "
                   "random network as initialized (init-s1 to s3), three initializations.",
         draw=draw_f4_ee),
    Page(id="F4-rand", title="Class pairs, linear probes, random partitions, resonance pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="The family's form for the five random partitions pooled, with 188 and 17 classes; L20: two panels a "
                 "row, so the three resonance pairs here, the legend in the fourth slot, and the three QCD pairs in "
                 "F4-rand-qcd (no random partition leaves a curve on X→ee vs μμ).",
         data_note="The pooled curve mixes partitions that merge a pair with ones that keep it apart (tab:pairsfull "
                   "marks which).",
         draw=draw_f4_rand),
    Page(id="F4-rand-qcd", title="Class pairs, linear probes, random partitions, QCD pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-rand's form on the three QCD pairs.",
         data_note="The pooled curve mixes partitions that merge a pair with ones that keep it apart (tab:pairsfull "
                   "marks which).",
         draw=lambda path: draw_f4_rand(path, "pairs", PAIR_ORDER[3:6])),
    Page(id="F4-rand-ft", title="Class pairs, fine-tuned, random partitions, resonance pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-rand after fine-tuning.",
         data_note="Reads curves_v2/pairs_ft.",
         draw=lambda path: draw_f4_rand(path, "pairs_ft")),
    Page(id="F4-rand-qcd-ft", title="Class pairs, fine-tuned, random partitions, QCD pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F4-rand-qcd after fine-tuning.",
         data_note="Reads curves_v2/pairs_ft.",
         draw=lambda path: draw_f4_rand(path, "pairs_ft", PAIR_ORDER[3:6])),
    Page(id="F6", title="ROC curves, top and quark/gluon tagging, linear probes",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="",
         differs="The family's two-class form (PI 2026-11-13): top, quark/gluon Pythia and Herwig, a ratio panel to 188 "
                 "classes under each; the label sets with the mass networks; the kinematic-inputs run out. L20: two "
                 "6.3 cm panels a row, so one mode a figure, the legend in the fourth slot; the fine-tuned networks "
                 "are F6-ft.",
         data_note="The random network's probe curves on top and quark/gluon tagging need inference (approved, queued) "
                   "and are not drawn. " + SUBST_NOTE,
         draw=draw_f6),
    Page(id="F6-ft", title="ROC curves, top and quark/gluon tagging, fine-tuned",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="",
         differs="F6's form for the fine-tuned networks, from scratch beside them (L20: the old F6's lower row).",
         draw=draw_f6_ft),
    Page(id="F6-part", title="ROC curves, top and quark/gluon tagging, 17-class partitions, linear probes",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F6's form for the four 17-class partitions, with 188 and 17 classes for reference.",
         data_note=SUBST_NOTE,
         draw=draw_f6_part),
    Page(id="F6-part-ft", title="ROC curves, top and quark/gluon tagging, 17-class partitions, fine-tuned",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="",
         differs="F6-part's form for the fine-tuned networks (L20: the old F6-part's lower row).",
         draw=draw_f6_part_ft),
    Page(id="A3R", title="Every JetClass-II class, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes, trained on one million class-balanced JetClass-II jets: rejection against "
                 "efficiency for each of the 188 classes against the rest of the test set, the mean over three "
                 "pretraining runs, colored by decay family. Curves end where fewer than 20 background jets remain. "
                 "Each title gives the mean area under the curve over all classes.",
         differs="Axes as FJ (PI). Linear probes on JetClass-II in all four panels (1M jets), where the poster has a "
                 "frozen MLP and full fine-tuning on JetClass (10M). 188 thin curves in six family colours, not ten "
                 "labelled classes; no AUC per curve, as there are too many. 0.4 pt lines at alpha 0.7: over the limit "
                 "of six series and off the line convention (one colour per series, 1.1 pt), not named in this brief, for "
                 "the PI to decide. No dashed diagonal. One-line titles, not two. Sophon's font and round 3's faint y grid, not the "
                 "poster's serif font and faint grid. The family legend is above the panels, off the curves, unframed, "
                 "with lines of normal thickness. Four panels, not two.",
         draw=draw_a3r),
    Page(id="A3R-ft", title="Every JetClass-II class after fine-tuning, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="The networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, fine-tuned on one "
                 "million class-balanced JetClass-II jets: rejection against efficiency for each of the 188 "
                 "classes against the rest of the test set, the mean over three pretraining runs, colored by "
                 "decay family. Curves end where fewer than 20 background jets remain. Each title gives the mean area "
                 "under the curve over all classes.",
         differs="As A3R, fine-tuned.",
         draw=draw_a3r_ft),
    Page(id="A3R-ctrl-ft", title="Every JetClass-II class after fine-tuning, 17-class partitions",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="",
         differs="A3R-ft's form for the 17-class partitions, as FJ-ctrl (round-1 fix 2026-11-13).",
         data_note="Reads curves_v2/jc2_ft_ovr_long (the partitions' long-budget cells).",
         draw=draw_a3r_ctrl_ft),
    Page(id="A3R-s", title="Every JetClass-II class, trained from scratch",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="",
         differs="A3R-ft's form for training from scratch, one panel, as FJ-s (round-1 fix 2026-11-13).",
         data_note="Reads curves_v2/jc2_ft_ovr_fulldecay (the full-schedule from-scratch cells).",
         draw=draw_a3r_s),
]

# the supplement members' captions, filled like the paper's (gallery_captions holds the main-text and appendix ones)
SUPPLEMENT_CAPTIONS = {
    "F4-ee": (
        "Linear probes on the frozen features separate X→ee from X→μμ completely with the networks pretrained with 188 "
        "classes, 162 classes and 162 classes + mass and with every 17-class partition, which keep fewer than 20 X→μμ jets at every "
        "signal efficiency and so have no curve. The curves show background rejection against signal efficiency, with "
        "X→ee as signal and every test jet of the pair scored, for the networks pretrained with 43 classes, 17 "
        "classes and 17 classes + mass, + mass marking an added mass regression, and the random network. Each curve is "
        "the mean over three pretraining runs, two for 17 classes + mass, or three random initializations, and ends "
        "where fewer than 20 X→μμ jets remain. Fine-tuned, every network separates the pair completely, so no "
        "fine-tuned panel is drawn, and the areas under the receiver operating characteristic curve are in table "
        "{tab:pairs}."
    ),
    "F4-rand": (
        "Linear probes on the frozen features of the networks pretrained with the five random 17-class partitions, the "
        "mean over the five with one run each, beside 188 and 17 classes with three runs each, on the three resonance "
        "pairs of figures {fig:F4-mech} and {fig:F4-other}, the QCD pairs being in figure {fig:F4-rand-qcd}. Some random partitions merge a pair into one class and others keep it apart, so the mean "
        "mixes the two. Each panel shows background rejection against signal efficiency with the first-named class as "
        "signal, and the lower panels the ratio to 188 classes with each curve's standard deviation over runs and "
        "binomial uncertainty in quadrature. Curves end where fewer than 20 background jets remain. Every random "
        "partition separates X→ee from X→μμ completely, and the areas under the receiver operating characteristic "
        "curve are in table {tab:pairsfull}."
    ),
    "F4-rand-ft": (
        "The panels of figure {fig:F4-rand} for networks fine-tuned on one million class-balanced JetClass-II jets and "
        "scored on natural-composition test jets, the five random 17-class partitions with one run each beside 188 and 17 "
        "classes with three runs each. Some random partitions merge a pair into one class and others keep it apart, so "
        "the mean mixes the two. The lower panels show the ratio to 188 classes with each curve's standard deviation "
        "over runs and binomial uncertainty in quadrature. Fine-tuning on JetClass-II ran with a cap of 160 epochs for "
        "every partition and the first run of 17 classes, and of 40 epochs for 188 classes and the other two runs of "
        "17 classes. Curves end where fewer than 20 background jets remain, and the areas under the receiver "
        "operating characteristic curve are in table {tab:pairsfull}."
    ),
    "F4-rand-qcd": (
        "The panels of figure {fig:F4-rand} on the three pairs of QCD jets of figures {fig:F4-other} and {fig:F4-light}, "
        "linear probes on the frozen features of the networks pretrained with the five random 17-class partitions, the "
        "mean over the five with one run each, beside 188 and 17 classes with three runs each. Some random partitions "
        "merge a pair into one class and others keep it apart, so the mean mixes the two. The lower panels show the "
        "ratio to 188 classes with each curve's standard deviation over runs and binomial uncertainty in quadrature. "
        "Curves end where fewer than 20 background jets remain, and the areas under the receiver operating "
        "characteristic curve are in table {tab:pairsfull}."
    ),
    "F4-rand-qcd-ft": (
        "The panels of figure {fig:F4-rand-qcd} for networks fine-tuned on one million class-balanced JetClass-II jets "
        "and scored on natural-composition test jets, the five random 17-class partitions with one run each beside 188 "
        "and 17 classes with three runs each. Some random partitions merge a pair into one class and others keep it "
        "apart, so the mean mixes the two. The lower panels show the ratio to 188 classes with each curve's standard "
        "deviation over runs and binomial uncertainty in quadrature. Fine-tuning on JetClass-II ran with a cap of 160 "
        "epochs for every partition and the first run of 17 classes, and of 40 epochs for 188 classes and the other "
        "two runs of 17 classes. Curves end where fewer than 20 background jets remain, and the areas under the "
        "receiver operating characteristic curve are in table {tab:pairsfull}."
    ),
    "F6-part": (
        "The four 17-class partitions in the form of figure {fig:F6}, as linear probes on frozen features trained on "
        "the full top and quark/gluon training sets, with quark/gluon scored on Pythia and on Herwig jets, and 188 and "
        "17 classes for reference, and figure {fig:F6-part-ft} shows them fine-tuned. Curves are means over two "
        "pretraining runs for each flavor-boundary partition, over the five random partitions with one run each, and "
        "over three runs for 188 and 17 classes. The lower panels show the ratio to 188 classes, the band of 188 "
        "classes its standard deviation over runs and the binomial uncertainty of its passing background count in "
        "quadrature, the others' being in the tables. R30, the inverse background efficiency at 30% signal efficiency, "
        "and the area under the receiver operating characteristic curve are in tables {tab:summary} and "
        "{tab:summaryauc}, the rejection at 50% signal efficiency in table {tab:probes}."
    ),
    "F6-part-ft": (
        "The four 17-class partitions in the form of figure {fig:F6-ft}, fine-tuned on the full top and quark/gluon "
        "training sets, with quark/gluon scored on Pythia and on Herwig jets, and 188 and 17 classes for reference. "
        "Curves are means over two pretraining runs for each flavor-boundary partition, over the five random partitions"
        " with one run each, and over three runs for 188 and 17 classes. The lower panels show the ratio to 188 "
        "classes, the band of 188 classes its standard deviation over runs and the binomial uncertainty of its passing "
        "background count in quadrature, the others' being in the tables. R30, the inverse background efficiency at 30%"
        " signal efficiency, and the area under the receiver operating characteristic curve are in tables {tab:summary}"
        " and {tab:summaryauc}, the rejection at 50% signal efficiency in table {tab:labelsets}."
    ),
    # the JetClass and JetClass-II class-against-rest pages (their figures lost the heading in this round): the brief's
    # form, no colons
    "FJ": (
        "Background rejection against signal efficiency with each JetClass class as signal against the other nine, for "
        "the networks pretrained with (a) 188, (b) 162, (c) 43 and (d) 17 classes and fine-tuned on one million "
        "JetClass jets. Each curve is the mean over three pretraining runs and ends where fewer than 20 background jets "
        "remain. The spread over runs is not drawn. Each panel title gives the area under the receiver operating "
        "characteristic curve averaged over the ten classes. Each class's area, with its standard deviation, is in "
        "table {tab:jc1_class_auc}, and table {tab:jetclassrej} gives a different quantity, the rejection of QCD jets "
        "alone at fixed signal efficiency. Figure {fig:FJ-mass} shows the mass networks."
    ),
    "FJ-ctrl": (
        "The panels of figure {fig:FJ} for the networks pretrained with the 17-class partitions, (a) no flavor boundary, "
        "(b) a b-quark boundary, (c) a random flavor boundary and (d) random partitions, fine-tuned on one million "
        "JetClass jets. Each curve is the mean over two pretraining runs, or over the five random partitions with one "
        "run each, and ends where fewer than 20 background jets remain. Each panel title gives the area under the receiver "
        "operating characteristic curve averaged over the ten classes. Each class's area is in table "
        "{tab:jc1_class_auc}, and table {tab:jetclassall} gives the rejection of QCD jets alone."
    ),
    "FJ-app": (
        "The panels of figure {fig:FJ} for linear probes on the frozen features of the networks pretrained with (a) 188, "
        "(b) 162, (c) 43 and (d) 17 classes, trained on one million JetClass jets. Each curve is the mean over three "
        "pretraining runs and ends where fewer than 20 background jets remain. The spread over runs is not drawn. Each "
        "panel title gives the area under the receiver operating characteristic curve averaged over the ten classes. "
        "Each class's area, with its standard deviation, is in table {tab:jc1_class_auc}, and table "
        "{tab:jetclassprobe} gives the rejection of QCD jets alone. Figure {fig:FJ-mass-app} "
        "shows the mass networks and the random network."
    ),
    "FJ-ctrl-app": (
        "The panels of figure {fig:FJ-ctrl} for linear probes on the frozen features of the networks pretrained with "
        "the 17-class partitions, trained on one million JetClass jets. Each curve is the mean over two pretraining "
        "runs, or over the five random partitions with one run each, and ends where fewer than 20 background jets "
        "remain. Each panel title gives the area under the receiver operating characteristic curve averaged over the "
        "ten classes, each class's area is in table {tab:jc1_class_auc}, and table {tab:jetclassprobe} gives the "
        "rejection of QCD jets alone."
    ),
    "FJ-s": (
        "Background rejection against signal efficiency with each JetClass class as signal against the other nine, for "
        "the network trained from scratch on one million JetClass jets. Each curve is the mean over three trainings and "
        "ends where fewer than 20 background jets remain, and the legend gives each class's area under the receiver "
        "operating characteristic curve. The spread over trainings is not drawn. Each class's area, with its standard "
        "deviation, is in table {tab:jc1_class_auc}, and table {tab:jetclassrej} gives a different quantity, the "
        "rejection of QCD jets alone at fixed signal efficiency."
    ),
    "FJ-mass": (
        "The panels of figure {fig:FJ} for the networks pretrained with (a) 162 classes, (b) 162 classes + mass, (c) 17 "
        "classes and (d) 17 classes + mass, + mass marking an added mass regression, fine-tuned on one million JetClass "
        "jets. Each curve is the mean over three pretraining runs, two for + mass, and ends where fewer than 20 "
        "background jets remain. The spread over runs is not drawn. Each panel title gives the area under the receiver "
        "operating characteristic curve averaged over the ten classes, each class's area is in table "
        "{tab:jc1_class_auc}, and table {tab:jetclassall} gives the rejection of QCD jets alone."
    ),
    "FJ-mass-app": (
        "The panels of figure {fig:FJ-mass} for linear probes on the frozen features, trained on one million JetClass "
        "jets, with (e) the random network, an untrained network, for reference. Each curve is the mean over three "
        "pretraining runs, two for + mass, or three random initializations, and ends where fewer than 20 background "
        "jets remain. The spread over runs is not drawn. Each panel title gives the area under the receiver operating "
        "characteristic curve averaged over the ten classes, each class's area is in table {tab:jc1_class_auc}, and "
        "table {tab:jetclassprobe} gives the rejection of QCD jets alone."
    ),
    "A3R-ctrl-ft": (
        "The panels of figure {fig:A3R-ft} for the networks pretrained with the 17-class partitions, (a) no flavor "
        "boundary, (b) a b-quark boundary, (c) a random flavor boundary and (d) random partitions, fine-tuned on one "
        "million class-balanced JetClass-II jets. Colors are decay families, and each curve is the mean over two "
        "pretraining runs, or over the five random partitions with one run each, and ends where fewer than 20 "
        "background jets remain. Every partition was fine-tuned with a cap of 160 epochs, while the networks of "
        "figure {fig:A3R-ft} had 40, except the first run of 17 classes. Each panel title gives the area under the "
        "receiver operating characteristic curve averaged over the classes, and each class's area is in table "
        "{tab:perclass}."
    ),
    "A3R-s": (
        "Background rejection against signal efficiency with each of the 188 JetClass-II classes as signal against the "
        "rest of the test jets, for the network trained from scratch on one million class-balanced JetClass-II jets. "
        "Colors are decay families, and each curve is the mean over three trainings and ends where fewer than 20 "
        "background jets remain. The title gives the area under the receiver operating characteristic curve averaged "
        "over the classes, which table {tab:summaryauc} gives with its spread over trainings."
    ),
    "A3R": (
        "Background rejection against signal efficiency with each of the 188 JetClass-II classes as signal against "
        "the rest of the test jets, for linear probes trained on one million class-balanced JetClass-II jets on the "
        "frozen features of the networks pretrained with (a) 188, (b) 162, (c) 43 and (d) 17 classes. Colors are "
        "decay families, and each curve is the mean over three pretraining runs and ends where fewer than 20 "
        "background jets remain. Each panel title gives the area under the receiver operating characteristic curve "
        "averaged over the classes, and each class's area is in table {tab:perclass}. The mass networks and the "
        "17-class partitions are not drawn."
    ),
    "A3R-ft": (
        "The panels of figure {fig:A3R} for the networks pretrained with (a) 188, (b) 162, (c) 43 and (d) 17 classes "
        "and fine-tuned on one million class-balanced JetClass-II jets. Colors are decay families, and each curve is "
        "the mean over three pretraining runs and ends where fewer than 20 background jets remain. Fine-tuning ran "
        "with a cap of 40 epochs, except the first run of 17 classes, which had the 160-epoch cap. Each panel title "
        "gives the area under the receiver operating characteristic curve averaged over the classes, and each class's "
        "area is in table {tab:perclass}. Figures {fig:A3R-ctrl-ft} and {fig:A3R-s} show the 17-class partitions and "
        "training from scratch, and the mass networks are not drawn."
    ),
}

import dataclasses  # noqa: E402
import gallery_captions  # noqa: E402  the paper's captions replace those above (PI 2026-11-12)
PAGES = [dataclasses.replace(p, caption=gallery_captions.fill(SUPPLEMENT_CAPTIONS[p.id])) if p.id in SUPPLEMENT_CAPTIONS
         else p for p in gallery_captions.apply(PAGES)]
