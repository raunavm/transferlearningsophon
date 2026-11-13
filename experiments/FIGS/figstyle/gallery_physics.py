#!/usr/bin/env python3
"""Gallery mock-ups of the design schematic, the mass regression, the mass sculpting, the resonance
tagging and performance against jet pT, in style_v4 only, with Sophon's Fig. 2 redrawn first as the check of that style. Layouts:
docs/figstyle/PLAN.md. Data come through paper_figures' selection code (gallery_common); nothing is drawn at
import."""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from gallery_bands import spread_band
from gallery_common import HERE, RUNS, Absent, I, M, P, Page, S
from gallery_roc import POOLED, _group_legend, _legend_cols, _style
import lines_v6 as L

GROOMED = "soft-drop mass"
MASS_ROWS = (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))     # a mass model and its label set's name
NAMES = {**S.NAME, **{row: S.LEGEND[row] for row, _ in MASS_ROWS}}      # "162 classes + mass", as every legend and table
RATIO_ATLAS, RATIO_CMS = 0.44, 0.33     # lower axes over main axes: ATLAS 2024-015 Figs. 7-8, CMS Fig. 8 (catalogue)
STACK_GAP, COLUMN_GAP = 0.08, 0.32      # a main and its lower axes; two columns with their own y axes
SHOWN_FROM = 0.05       # a response axis covers the bins with this share of the largest bin of any series
HIST_ROOM = 1.45        # top of a histogram's y axis over its peak: room for the legend above the curves
# F10a, after CMS Fig. 8: black inclusive, then two working points in CLASS10 colours (orange, plum), which reuse no
# network colour (L6, palette_check), so no working point reads as a network; dark, mid and light in greyscale
SELECTION_COLOUR = {1: S.CLASS10[9], 2: S.CLASS10[3]}
SCULPT_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")      # L5: each mass network under its label set
SCULPT_PARTITIONS = ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")     # with 188 and 17 for reference
F10B_FROM = 5e-3        # F10b: the first point is the 5.2e-3 grid point; CMS Fig. 9 left stops at 0.5%


def _fit_y(ax, values: list, room: float = 0.0, pad: float = 0.08) -> None:
    """y limits from the finite values given. By hand since an ax.stairs line does not autoscale alike across
    matplotlib versions; room is empty space on top for a legend, as a share of the data's span."""
    v = np.concatenate([np.ravel(x) for x in values])
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    span = hi - lo or abs(hi) or 1.0
    ax.set_ylim(lo - pad * span, hi + (pad + room) * span)


# ------------------------------------------------------------------ the check of the style
# Sophon's eight curves in our palette (round 3): Sophon, trained on 188 classes, takes the 188 colour in both panels
# (its X->bq discriminant too, dotted as Sophon draws it); Sophon* (42 classes) the 43 colour; ParT and ParticleNet,
# taggers trained from scratch, the from-scratch dark grey and the mid grey. Line styles stay Sophon's.
SOPHON_COLOUR = {"Sophon": S.SERIES["L188"], "Sophon* (42 class)": S.SERIES["R42_Q1"],
                 "Sophon (transfer learning)": S.SERIES["L188"], "Sophon_{discr. = X->bq vs. QCD}": S.SERIES["L188"],
                 "ParT_{X->bb vs. QCD}": S.GREY_DARK, "ParT_{X->bs vs. QCD}": S.GREY_DARK,
                 "ParticleNet_{X->bb vs. QCD}": S.GREY_MID, "ParticleNet_{X->bs vs. QCD}": S.GREY_MID}


def draw_sophon_fig2(path: pathlib.Path) -> None:
    """Sophon Fig. 2 from the points read off its native PDF, through style_v4 and our palette; Sophon's figure size,
    panel gap (hspace 0.2, where style_v4 sets 0.25), limits, ticks, line widths and line styles."""
    d = json.loads((HERE / "sophon_fig2_points.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH, S.HEIGHT[2]), sharex=True, gridspec_kw={"hspace": 0.2})
    for k, (ax, p) in enumerate(zip(axes, d["panels"])):
        S.grid(ax)
        for c in p["curves"]:
            xy = np.array(c["points_xy"])
            ax.plot(xy[:, 0], xy[:, 1], color=SOPHON_COLOUR[c["name_printed"]], linewidth=S.LW,
                    linestyle=":" if c["dash_pdf"] else "-", label=c["name_mathtext"])
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 5.4)
        ax.set_yticks(range(6))
        S.title(ax, p["title_mathtext"])
        S.labels(ax, ylabel=p["y"]["label"])
        S.legend(ax, ncol=2 if k == 0 else 1)   # upper panel: two columns keep the frameless legend above the curves
    S.labels(axes[1], xlabel=d["panels"][1]["x"]["label"])
    S.save(fig, path)


# ------------------------------------------------------------------ F1
def draw_f1(path: pathlib.Path) -> None:
    """The PI's drawing (2026-11-12), experiments/FIGS/fig1_design.py: 13 cm wide, its label sets read from the label
    maps."""
    import fig1_design
    fig = fig1_design.build()
    fig.exact_size = True
    S.save(fig, path)


# ------------------------------------------------------------------ F9a, F9b, A17
ATLAS_FINE = M._load(P.DATA / "head" / "atlas_res_fine")     # ten pT bins (atlas_rebin.py); the same jets and inclusive values


def _atlas() -> dict:
    if not (ATLAS_FINE or I["atlas"]):
        raise Absent("no mass resolution files (head/atlas_res)")
    return ATLAS_FINE or I["atlas"]


GROOMED_LOOK = {"color": "#000000", "linestyle": "-", "linewidth": S.LW_CURVE}    # solid black: the reference


def _response_series(atlas: dict, group: str) -> tuple:
    """(bin edges, [(values, look, label)]) of the response histograms of the soft-drop mass and the two mass outputs
    (solid: nothing else shares the panel)."""
    g = P._groomed(atlas, group)["hist"]
    e = np.asarray(g["edges"], float)
    out = [(P._frac(g), GROOMED_LOOK, GROOMED)]
    for row, _ in MASS_ROWS:
        hs = [P._frac(b["hist"]) for m in RUNS[row] if (b := P._block(atlas, m, "regressed", group))]
        if hs:
            out.append((np.mean(hs, axis=0), {**S.look(row), "linewidth": S.LW_CURVE}, S.LEGEND[row]))
    return e, out


def draw_f9a(path: pathlib.Path) -> None:
    """Two panels side by side, no ratio sub-panels (a 2x1 stack with them exceeds the page); one y range and one
    response range, the bins in which any series of either panel reaches SHOWN_FROM of its peak; only those bins are
    drawn, so no histogram runs off the axes."""
    atlas = _atlas()
    fig, axes, _ = S.grid_cm(1, 2, w=6.5, aspect=0.85, top=1.4, left=1.5, wgap=0.3, bottom=1.2)
    panels = [(_response_series(atlas, g), t) for g, t in (("resonant", "Resonant jets"), ("qcd", "QCD jets"))]
    e = panels[0][0][0]
    on = np.flatnonzero(np.any([v >= SHOWN_FROM * v.max() for (_, series), _ in panels for v, _, _ in series], axis=0))
    i, j = on[0], on[-1] + 1
    peak = 0.0
    for ax, ((edges, series), title) in zip(axes[0], panels):
        for v, look, label in series:
            ax.stairs(v[i:j], edges[i:j + 1], baseline=None, label=label, **look)
            peak = max(peak, float(v[i:j].max()))
        ax.set_xlim(e[i], e[j])
        S.title(ax, title)
        S.labels(ax, xlabel=r"Response $r = m/m_{\mathrm{gen}}$")
        S.fine_grid(ax)
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 0].set_ylim(0, 1.08 * peak)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))     # no 0 in the corner
    axes[0, 1].tick_params(labelleft=False)
    S.labels(axes[0, 0], ylabel="Fraction of jets")
    S.letter_panels(axes[0])
    S.fig_legend(fig, axes[0, 0].get_legend_handles_labels()[0], top_cm=S.LEGEND_TOP, ncol=3)
    S.save(fig, path)


def _mass_series() -> list:
    """(runs, estimator, line, marker shape, label) of the two mass networks' outputs, in their lighter shades, solid
    and filled (L6)."""
    return [(RUNS[row], "regressed", S.look(row), None, S.LEGEND[row]) for row, _ in MASS_ROWS]


PROBE_ESTS = ("mlp", "ridge")      # regressions on frozen features, against the networks' own mass output


def _quantity(ax, rx, atlas, group, axis, series, which: str) -> tuple:
    """One column of one row: the median (which="median") or the resolution against axis, binned, so points at the bin
    centres with +-1 SD bars over runs and no lines (PI 2026-11-11), the soft-drop mass first in black, the series
    with a bar across the bin (lines_v6.binned); below, the ratio to the soft-drop mass. The y range holds every point and
    bar (PI 2026-11-11: nothing off the axes); the resolution is on a log axis. Returns (soft-drop values, bin edges)."""
    g = P._groomed(atlas, group)[axis]
    e = np.asarray(g["edges"], float)
    ref = P._arr(g[which])
    drawn = [(runs, est, line, mark, label) for runs, est, line, mark, label in series
             if any(P._block(atlas, m, est, group) for m in runs)]
    # a probe beside a network's own output is open; one mode a panel stays filled
    mixed = len({est in PROBE_ESTS for _, est, _, _, _ in drawn}) > 1
    zero = np.zeros_like(ref)
    L.binned(ax, e, ref, zero, zero, color=GROOMED_LOOK["color"], probe=False, label=GROOMED)
    rx.axhline(1.0, **S.REFERENCE, zorder=1.5)       # the ratio's unity line: a reference, dotted (L6)
    ends, lows = [ref], [np.array([1.0])]
    for runs, est, line, mark, label in drawn:
        per = np.array([P._arr(b[axis][which]) for m in runs if (b := P._block(atlas, m, est, group))], float)
        st = [P.F._spread(list(per[:, i])) for i in range(per.shape[1])]
        mean = np.array([s_[0] for s_ in st])
        lo = np.array([s_[1] or 0.0 for s_ in st])
        hi = np.array([s_[2] or 0.0 for s_ in st])
        probe, shape = mixed and est in PROBE_ESTS, mark or "o"
        L.binned(ax, e, mean, lo, hi, color=line["color"], probe=probe, label=label, shape=shape)
        L.binned(rx, e, mean / ref, lo / ref, hi / ref, color=line["color"], probe=probe, label="_nolegend_", shape=shape)
        ends += [mean - lo, mean + hi]
        lows += [(mean - lo) / ref, (mean + hi) / ref]
    if which == "resolution":
        v = np.concatenate([np.ravel(x) for x in ends])
        v = v[np.isfinite(v) & (v > 0)]
        ax.set_ylim(v.min() / 1.3, v.max() * 1.15)
    else:
        _fit_y(ax, ends, pad=0.12)       # a point's marker clear of the frame (2026-11-12 review)
    _fit_y(rx, lows, pad=0.18)       # the reference at 1 clear of the frame, its tick labelled
    ax.set_xlim(e[0], e[-1])
    for a_ in (ax, rx):
        S.fine_grid(a_)
    return ref, e


# PI 2026-11-12: panel titles say what each panel shows
F9B_TITLE = {"median": "Response", "resolution": "Resolution", "vs_pt": r"jet $p_{\mathrm{T}}$",
             "vs_mgen": "generator mass"}
X_PT, X_MGEN = r"Jet $p_{\mathrm{T}}$ [GeV]", r"Generator mass ($m_{\mathrm{gen}}$) [GeV]"


def _share_row(row) -> None:
    """One y range for the two panels of a row, the union of theirs, with the tick labels once, on the left: matching
    axes carry the same labels (PI 2026-11-11)."""
    (a0, a1), (b0, b1) = row[0].get_ylim(), row[1].get_ylim()
    for ax in row:
        ax.set_ylim(min(a0, b0), max(a1, b1))
    row[1].tick_params(labelleft=False)


def _binned_2x2(path: pathlib.Path, group: str, series: list, ncol: int, cols: list | None = None) -> None:
    """Median response (upper row) and resolution (lower row, log axis) against jet pT (left) and generator mass
    (right), 6.5 cm panels, each with the ratio to the soft-drop mass below; one y range per row; the legend above
    the panels, in ncol columns or, given cols (lists of legend labels, one a column), in those columns."""
    atlas = _atlas()
    rows = max(len(c) for c in cols) if cols else math.ceil((len(series) + 1) / ncol)
    refs = np.empty((2, 2), object)
    fig, axes, raxes = S.grid_cm(2, 2, w=6.1, aspect=0.72, ratio=RATIO_ATLAS, top=1.0 + 0.45 * rows, left=1.85,
                                 wgap=0.6, hgap=1.0, bottom=1.25)      # Latin Modern: room for the y label and the last x label
    for i, which in enumerate(("median", "resolution")):
        for j, (axis, xlabel) in enumerate((("vs_pt", X_PT), ("vs_mgen", X_MGEN))):
            refs[i, j] = _quantity(axes[i, j], raxes[i, j], atlas, group, axis, series, which)
            S.title(axes[i, j], F9B_TITLE[which] + " against " + F9B_TITLE[axis])
            raxes[i, j].tick_params(labelbottom=True)
            if i == 1:
                S.labels(raxes[i, j], xlabel=xlabel)
        for row_axes in (axes[i], raxes[i]):
            _share_row(row_axes)
        S.labels(axes[i, 0], ylabel=r"Median response ($r$)" if which == "median" else r"Resolution ($\sigma_r$)")
        S.labels(raxes[i, 0], ylabel="Ratio")
        for j in range(2):
            # at least two labelled ticks inside a ratio range that ends just past 0.8 and 1.2 (A17-ridge)
            raxes[i, j].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], prune="both"))
            if which == "median":
                axes[i, j].yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], prune="lower"))
            else:
                S.log_axis(axes[i, j], force=True, fine=True)
                # PI 2026-11-12: "1.0", not "1", beside 0.2 and 0.5 (one decimal from 0.1 up)
                axes[i, j].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}" if v >= 0.1 else f"{v:g}"))
            S.clear_edge_labels(axes[i, j], low=True)       # clear of the ratio panel below
            S.clear_edge_labels(raxes[i, j], high=True)
    fig.align_ylabels([axes[0, 0], raxes[0, 0], axes[1, 0], raxes[1, 0]])
    S.letter_panels(list(axes.flat))
    handles = axes[0, 0].get_legend_handles_labels()[0]
    if cols:
        by = {h.get_label(): h for h in handles}
        blank = Line2D([], [], linestyle="none", marker="none", label="")
        handles = [by.get(t, blank) for c in cols for t in c + [None] * (rows - len(c))]
        ncol = len(cols)
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=ncol)
    S.save(fig, path)


def draw_f9b(path: pathlib.Path) -> None:
    _binned_2x2(path, "all", _mass_series(), 3)


def _a17(path: pathlib.Path, group: str) -> None:
    _binned_2x2(path, group, _mass_series(), 3)


# the probe members of the mass family (PI 2026-11-13): one regression on the frozen features of every label set, mass
# network and the random network (as initialized, three initializations), one mode a figure, so filled; the random
# network's points are crosses, apart from the black soft-drop points they overlap
PROBE_NETS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "INIT")      # L5


def _probe_figure(path: pathlib.Path, est: str) -> None:
    """The 2x2 of F9b for one regression on frozen features (est "mlp" or "ridge"); the legend in L5's columns: 188
    and 162 classes, 43 and 17 classes, the references."""
    series = [(RUNS[n], est, S.look(_style(n)), "x" if n == "INIT" else None, S.LEGEND[_style(n)]) for n in PROBE_NETS]
    names = [S.LEGEND[_style(n)] for n in PROBE_NETS]
    _binned_2x2(path, "all", series, 3, cols=[names[:3], names[3:6], [GROOMED, names[6]]])


def draw_a17_probe(path: pathlib.Path) -> None:
    _probe_figure(path, "mlp")


def draw_a17_ridge(path: pathlib.Path) -> None:
    _probe_figure(path, "ridge")


# ------------------------------------------------------------------ F10a, F10b
def _sculpt() -> dict:
    if not I["sculpt"]:
        raise Absent("no mass sculpting curves (head/curves_v2/*_sculpting_v2.json)")
    return I["sculpt"]


def _sculpt_panels(path: pathlib.Path, rows: tuple) -> None:
    """One panel per model, two per row at 6.5 cm or three in one row at 4.3 cm, each with its ratio to the inclusive
    distribution (CMS's height); step histograms over the files' whole mass range, the inclusive one black and the
    working points orange and plum (SELECTION_COLOUR), no model colour, since the title names the model; one y range."""
    sc = _sculpt()
    shown = [(a, ms) for a in rows if (ms := [m for m in RUNS[a] if m in sc])]
    if not shown:
        raise Absent("no mass sculpting curve of the models shown")
    ncols = 3 if len(shown) == 3 else 2         # three panels in one row (PI 2026-11-11: no empty slot)
    nrows = math.ceil(len(shown) / ncols)
    fig, axes, raxes = S.grid_cm(nrows, ncols, w=S.PANEL_3 if ncols == 3 else 6.4, aspect=0.75 if ncols == 2 else 0.95,
                                 ratio=RATIO_CMS, top=1.45, left=1.3, wgap=0.55, hgap=1.0)
    panels = list(zip(axes.flat, raxes.flat))
    peak, ratios = 0.0, [[1.0]]
    for k_panel, ((ax, rx), (a, ms)) in enumerate(zip(panels, shown)):
        e = np.asarray(sc[ms[0]]["bins_jet_sdmass"], float)
        before = np.asarray(sc[ms[0]]["hist_before"], float)
        before = before / before.sum()
        # step histograms over the whole mass range, no markers (PI 2026-11-11)
        ax.stairs(before, e, baseline=None, color="#000000", linewidth=S.LW_CURVE, label="inclusive", zorder=2.1)
        peak = max(peak, before.max())
        rx.axhline(1.0, **S.REFERENCE, zorder=1)          # the unity line: a reference, dotted (L6)
        for k_wp, (wp, _) in enumerate(P.WPS, start=1):
            after = np.mean([np.asarray(sc[m]["working_points"][wp]["hist_after"], float)
                             / np.sum(sc[m]["working_points"][wp]["hist_after"]) for m in ms], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            label = f"$\\varepsilon_B = {100 * float(wp):g}\\%$"
            ax.stairs(after, e, baseline=None, color=SELECTION_COLOUR[k_wp], linewidth=S.LW_CURVE, label=label)
            rx.stairs(r, e, baseline=None, color=SELECTION_COLOUR[k_wp], linewidth=S.LW_CURVE)
            peak = max(peak, after.max())
            ratios.append(r)
        ax.set_xlim(e[0], e[-1])
        ax.tick_params(labelbottom=False)
        S.title(ax, NAMES[a])
        if k_panel >= len(shown) - ncols:
            S.labels(rx, xlabel=r"Jet mass ($m_{\mathrm{SD}}$) [GeV]")
        S.fine_grid(ax)
        S.fine_grid(rx)
    for i in range(nrows):
        axes[i, 0].set_ylim(0, 1.1 * peak)
        _fit_y(raxes[i, 0], ratios, pad=0.1)
        for j in range(1, ncols):
            axes[i, j].sharey(axes[i, 0])
            raxes[i, j].sharey(raxes[i, 0])
            axes[i, j].tick_params(labelleft=False)
            raxes[i, j].tick_params(labelleft=False)
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        raxes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 5, 10], prune="both"))     # 1.0 a tick
        S.clear_edge_labels(raxes[i, 0], high=True)     # clear of the panel above
        S.labels(axes[i, 0], ylabel="Fraction of jets")
        S.labels(raxes[i, 0], ylabel="Ratio to\ninclusive")
    S.letter_panels([ax for ax, _ in panels[:len(shown)]])
    S.fig_legend(fig, axes[0, 0].get_legend_handles_labels()[0], top_cm=S.LEGEND_TOP, ncol=3)
    S.save(fig, path)


def draw_f10a(path: pathlib.Path) -> None:
    _sculpt_panels(path, tuple(S.LEVELS))


def draw_f10a_mass(path: pathlib.Path) -> None:
    _sculpt_panels(path, ("L162_MASS", "R16_Q1_MASS"))


def draw_f10a_flav(path: pathlib.Path) -> None:
    _sculpt_panels(path, tuple(S.CONTROLS))


def _floor(sc: dict, m: str) -> dict:
    """{efficiency: the distance a random selection of as many QCD jets shows} on run m's curve (mlst_results._curve,
    the tables' floor): the same at every point for every run, which pass the same number of jets."""
    return {e: f for e, (_, f) in M._curve(sc, m).items()}


def _sculpt_curves(path: pathlib.Path, sc: dict, rows: tuple, runs: dict | None = None) -> None:
    """Exact size, one panel SINGLE_W wide (PI 2026-11-12), the legend just above it; each row's mean over runs with a band of
    the spread over runs (gallery_bands.spread_band), every network solid in its colour (the mass networks in their
    label sets' lighter shades, L6); the floor black dotted; both axes logarithmic, labelled 10^n. runs: {row: models}, RUNS by default."""
    runs = RUNS if runs is None else runs
    legend_rows = math.ceil((sum(any(m in sc for m in runs[r]) for r in rows) + 1) / 2)
    top, w = 0.5 + 0.45 * legend_rows, S.SINGLE_W - 1.75 - 0.25
    h = 0.85 * w
    fig = S.figure_cm(S.SINGLE_W, top + h + 1.0)
    ax = S.axes_cm(fig, 1.75, top, w, h)
    S.fine_grid(ax)
    drawn = []
    for row in rows:
        have = [sc[m]["curve"] for m in runs[row] if m in sc]
        if not have:
            continue
        eff = np.array([c["eff"] for c in have[0]], float)
        per = np.array([[c["js_distance"] for c in h] for h in have], float)
        keep = eff >= F10B_FROM
        mean = per.mean(axis=0)
        band = spread_band(per)
        lo, hi = (np.zeros_like(mean), np.zeros_like(mean)) if band is None else (mean - band[0], band[1] - mean)
        L.scan(ax, eff[keep], mean[keep], lo[keep], hi[keep], color=S.SERIES[row],
               linestyle=S.look(row)["linestyle"], probe=False, label=S.LEGEND[row], style=L.SCAN)
        drawn.append(row)
    if not drawn:
        raise Absent("no mass sculpting curve of the models shown")
    ref = _floor(sc, next(m for m in runs[drawn[0]] if m in sc))
    eff = np.array(sorted(ref))
    keep = eff >= F10B_FROM
    ax.plot(eff[keep], [ref[e] for e in eff[keep]], **S.REFERENCE, zorder=1.5, label="random selection")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.autoscale_view()
    S.log_axis(ax, "x", force=True)
    S.log_axis(ax, "y", force=True)
    S.labels(ax, r"QCD efficiency ($\varepsilon_B$)", "Jensen–Shannon distance")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=S.LEGEND_TOP, ncol=2)
    S.save(fig, path)


def draw_f10b(path: pathlib.Path) -> None:
    """The label sets and the two mass models (PI, 2026-11-08); the flavor boundaries on F10b-flav."""
    _sculpt_curves(path, _sculpt(), SCULPT_ROWS)


def draw_f10b_flav(path: pathlib.Path) -> None:
    """The flavor-boundary partitions with 188 and 17 classes for reference; the random partitions have no head
    outputs (their extraction is queued inference)."""
    sc = _sculpt()
    if not any(m in sc for a in S.CONTROLS for m in RUNS[a]):
        raise Absent("no mass sculpting curve of the flavor-boundary partitions")
    _sculpt_curves(path, sc, SCULPT_PARTITIONS, POOLED)


SCULPT_FT = "head/sculpting_ft"     # the fine-tuned networks' sculpting files, <model>.json (EVAL/sculpting_ft.py)


def _sculpt_ft() -> dict:
    sc = {p.stem.removesuffix("_sculpting"): json.loads(p.read_text()) for p in sorted((P.DATA / SCULPT_FT).glob("*.json"))}
    if not sc:
        raise Absent(f"no fine-tuned sculpting files ({SCULPT_FT}/<model>.json)")
    return sc


def draw_f10_ft(path: pathlib.Path) -> None:
    """F10b's form after fine-tuning on JetClass-II, scored on the natural-mix test jets the pretraining heads are
    (unweighted), so the floor is F10b's (_floor)."""
    sc = _sculpt_ft()
    scratch = [f"scratch-s{k}" for k in (1, 2, 3)]
    if not all(m in sc for m in scratch):
        raise Absent(f"the three from-scratch networks' sculpting files ({SCULPT_FT}/scratch-s<k>.json)")
    _sculpt_curves(path, sc, SCULPT_ROWS + ("SCRATCH",), {**RUNS, "SCRATCH": scratch})


def draw_f10_ft_part(path: pathlib.Path) -> None:
    """F10-ft for the 17-class partitions (the five random partitions one curve, the band their SD), with 188 and 17
    classes for reference."""
    sc = _sculpt_ft()
    if not any(m in sc for a in S.CONTROLS for m in RUNS[a]):
        raise Absent("no fine-tuned sculpting file of the flavor-boundary partitions")
    _sculpt_curves(path, sc, SCULPT_PARTITIONS, POOLED)


# ------------------------------------------------------------------ F10c (DisCo Fig. 3's plane)
F10C_KEY = "sig_eff_50"     # per run (head/sculpting_v3): {"rejection": 1/eps_B, "jsd": Jensen-Shannon divergence} at 50% eps_S
F10C_ROWS = SCULPT_ROWS + tuple(S.CONTROLS)       # L5; three legend columns: 188 and 162, 43 and 17, the partitions


def draw_f10c(path: pathlib.Path) -> None:
    """One point per model: the mean over runs of R50 (x) and 1/JSD50 (y), with the spread over pretraining runs on
    both axes (mlst_figures._spread: the SD over runs); DisCo Fig. 3's axes, both logarithmic."""
    sc = I["sculpt"]
    top, w = 0.5 + 0.45 * 3, S.PRINT_W - 1.5 - 0.2      # one panel across the text width, the legend just above it
    fig = S.figure_cm(S.PRINT_W, top + 0.7 * w + 1.0)
    ax = S.axes_cm(fig, 1.5, top, w, 0.7 * w)
    S.fine_grid(ax)
    drawn = 0
    for row in F10C_ROWS:
        pts = [sc[m][F10C_KEY] for m in RUNS[row] if F10C_KEY in sc.get(m, {})]
        if not pts:
            continue
        (x, xl, xh), (y, yl, yh) = (P.F._spread(v) for v in ([p["rejection"] for p in pts], [1 / p["jsd"] for p in pts]))
        ax.errorbar(x, y, xerr=None if xl is None else [[xl], [xh]], yerr=None if yl is None else [[yl], [yh]],
                    fmt="none", ecolor=S.SERIES[row], elinewidth=L.BAR, capsize=0)
        ax.plot([x], [y], linestyle="none", **L.dot(S.SERIES[row], False), label=S.LEGEND[row], zorder=2.6)
        drawn += 1
    if not drawn:
        raise Absent(f"no {F10C_KEY} in the sculpting files (head/sculpting_v3/*_sculpting.json)")
    S.log_axis(ax, "x")         # the autoscaled limits hold the bars
    S.log_axis(ax)
    S.labels(ax, r"QCD rejection at 50% resonant efficiency ($R_{50}$)", r"$1/\mathrm{JSD}_{50}$")
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=S.LEGEND_TOP, ncol=3)      # three rows, clear of the panel
    S.save(fig, path)


# ------------------------------------------------------------------ F11, F11-part, F11-probe: the resonance family
# PI 2026-11-13 (L11): two panels, X->bb and X->bs on Sophon's samples; the label sets with the mass networks (main
# text), the 17-class partitions with 188 and 17 classes for reference (appendix), the linear probe on the same
# frozen features with the random network (supplement). No pretrained network is fine-tuned on these samples.
XBX_LABEL_SETS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")      # L5
XBX_PARTITIONS = ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")
CLASSIFIER, LINEAR = "head", "linear"       # xbx readouts: the frozen-feature classifier (the tables'), the linear probe
DIRECT = "zero_shot_xbb"                    # the direct discriminant (Sophon Eq. 4), X->bb only, where X->bb is a class


def _xbx_runs(XR, signal: str, net: str, readout: str) -> list:
    """The readouts of a network's runs (the random partitions pooled; the random network's three initializations)."""
    fits = I["xbx"]
    return [r for m in POOLED[net] if (r := XR.readout(fits.get(m, {}).get(signal, {}), readout))]


def _xbx_series(XR, signal: str, nets: tuple, readout: str, direct: bool, scratch: bool) -> list:
    """(key, label, look, readouts per run) of one signal: each network's readout, solid in its colour (the mass
    networks in their lighter shades); from scratch grey solid; for X->bb the direct discriminant dash-dot for every
    network whose label set keeps X->bb as a class of its own (xbx_report.direct_ok: 188, 162, 162 + mass)."""
    out = [(net, S.LEGEND[_style(net)], S.look(_style(net)), rs) for net in nets
           if (rs := _xbx_runs(XR, signal, net, readout))]
    if scratch and (scr := I["xbx_scratch"].get(signal, [])):
        out.append(("SCRATCH", S.LEGEND["SCRATCH"], S.look("SCRATCH"), scr))
    if direct and signal == "xbb":
        out += [(f"{net} direct", f"{S.LEGEND[net]} (direct discriminant)", S.look(net, "direct"), rs) for net in nets
                if XR.direct_ok(M, net) and (rs := _xbx_runs(XR, signal, net, DIRECT))]
    return out


def _xbx_figure(path: pathlib.Path, nets: tuple, readout: str, direct: bool = False, scratch: bool = False) -> None:
    """Two panels side by side at the text width (PI 2026-11-11), eps_S/sqrt(eps_B) against eps_B in the mass window,
    each curve the mean over runs with a +-1 SD band (PI 2026-11-12; seeds for from scratch, the five partitions for
    random partitions); one y range; the legend above in _legend_cols's columns, the direct discriminants a column of
    their own and from scratch under the first column."""
    XR = P._xr()
    panels = [(sig, _xbx_series(XR, sig, nets, readout, direct, scratch)) for sig, _ in XR.SIGNALS]
    if not all(series for _, series in panels):
        raise Absent(f"X→bb and X→bs fits on Sophon's samples ({P.DATA / 'xbx'}): not pulled")
    handles = {}
    for _, series in panels:
        for key, label, style, _ in series:
            handles.setdefault(key, (Line2D([], [], **{**style, "linewidth": S.LW_CURVE}), label))
    cols = _legend_cols([k for k in handles if k in nets or k == "SCRATCH"])
    if len(cols) > 2 and set(cols[-1]) <= {"SCRATCH", "INIT"} and any(k.endswith(" direct") for k in handles):
        cols[0] += cols.pop()       # three columns at most: the reference under the first one
    cols.append([k for k in handles if k.endswith(" direct")])
    cols = [c for c in cols if c]
    rows = max(len(c) for c in cols)
    fig, grid, _ = S.grid_cm(1, 2, w=6.35, aspect=0.85, top=S.LEGEND_TOP + 0.45 * rows + 0.45, left=1.2, wgap=0.7,
                             bottom=1.15)
    axes = list(grid[0])
    top, xs = 0.0, []
    for ax, (_, series) in zip(axes, panels):
        for _, _, style, rs in series:
            x, y, _ = P.xbx_curves(M, rs)
            mean = y.mean(axis=0)
            sd = y.std(axis=0, ddof=1) if len(y) > 1 else np.zeros_like(mean)
            ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=L.BAND, linewidth=0, zorder=1.0)
            ax.plot(x, mean, **{**style, "linewidth": S.LW_CURVE})
            top = max(top, float(np.nanmax(mean + sd)))
            xs += [float(np.min(x)), float(np.max(x))]
    for ax, what in zip(axes, (r"$X\rightarrow b\bar{b}$", r"$X\rightarrow bs$")):
        S.fine_grid(ax)
        ax.set_xlim(min(xs) / 1.3, max(xs))
        ax.set_ylim(0, 1.08 * top)
        S.log_axis(ax, "x")
        S.title(ax, what)
        S.labels(ax, xlabel=r"QCD efficiency ($\varepsilon_B$)")
    axes[1].tick_params(labelleft=False)
    S.clear_edge_labels(axes[0], low=True)          # its 0 clear of the first x label in the corner
    S.labels(axes[0], ylabel=r"Figure of merit ($\varepsilon_S/\varepsilon_B^{1/2}$)")
    S.letter_panels(axes)
    blank = (Line2D([], [], linestyle="none", marker="none"), "")
    items = [handles[k] if k else blank for c in cols for k in c + [None] * (rows - len(c))]
    S.fig_legend(fig, [h for h, _ in items], top_cm=S.LEGEND_TOP, ncol=len(cols), labels=[t for _, t in items])
    S.save(fig, path)


def draw_f11(path: pathlib.Path) -> None:
    _xbx_figure(path, XBX_LABEL_SETS, CLASSIFIER, direct=True, scratch=True)


def draw_f11_part(path: pathlib.Path) -> None:
    _xbx_figure(path, XBX_PARTITIONS, CLASSIFIER)


def draw_f11_probe(path: pathlib.Path) -> None:
    _xbx_figure(path, XBX_LABEL_SETS + ("INIT",), LINEAR)


# ------------------------------------------------------------------ the jet-pT family (PI 2026-11-13, L4-L6); CONF is
# now the per-class family's confusion slice (gallery_perclass)
# experiments/EVAL/perf_vs_pt.py's files, pulled by pull_mlst_data.py: per pT bin, the linear probes on class-balanced
# JetClass-II (jc2) and on JetClass (jc1_probe), and the networks fine-tuned on them (jc2_ft, jc1)
PERF = P.DATA / "perf_vs_pt"
PT_MARKER = 3.5
# the family's panels, (perf_vs_pt folder, panel title): rows linear probe and fine-tuned, columns JetClass-II (ten
# bins, 200-2500 GeV) and JetClass (five bins, 500-1000 GeV). Only these two test sets: the top and quark/gluon samples
# span about 50-100 GeV of jet pT, and the resonance samples have no measured spectrum (L4)
PT_PANELS = (("jc2", "JetClass-II, linear probe"), ("jc1_probe", "JetClass, linear probe"),
             ("jc2_ft", "JetClass-II, fine-tuned"), ("jc1", "JetClass, fine-tuned"))
PT_BBCC = (("jc2", r"Linear probe, $X\to b\bar{b}$ vs $c\bar{c}$"), ("jc2_ft", r"Fine-tuned, $X\to b\bar{b}$ vs $c\bar{c}$"))
PT_ROWS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")       # L5: each mass network under its label set
PT_PARTITIONS = ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")     # with 188 and 17 classes for reference


def _runs_in(docs: dict, row: str) -> list:
    """The docs of a row's runs (the five random partitions pooled under RAND); every run must have its file."""
    have = [docs[m] for m in POOLED[row] if m in docs]
    if len(have) != len(POOLED[row]):
        raise Absent(f"perf_vs_pt holds {len(have)} of the {len(POOLED[row])} runs of {row}")
    return have


def _edges(docs: dict) -> np.ndarray:
    edges = {tuple(d["pt_edges"]) for d in docs.values()}
    if len(edges) != 1:
        raise SystemExit(f"FATAL: the perf_vs_pt files hold {len(edges)} different pT binnings")
    return np.asarray(edges.pop(), float)


def _accuracy(doc: dict) -> np.ndarray:
    return np.array([np.nan if b["accuracy"] is None else b["accuracy"] for b in doc["bins"]])


def _bb_cc(doc: dict) -> np.ndarray:
    return np.array([np.nan if b["bb_vs_cc"] is None else b["bb_vs_cc"]["auc"] for b in doc["bins"]])


def _pt_series(ax, e, values: list, key: str, label: str, band: bool = True, style: str = "B", probe: bool = False) -> None:
    """Mean over runs per bin of edges e. Style A: joined at the bin centres, the series' marker, band the SD,
    as F8. Styles B and C: binned points (lines_v6.binned), the family's form."""
    st = [P.F._spread(list(v)) for v in np.asarray(values).T]
    mean = np.array([s[0] for s in st])
    if style != "A":
        L.binned(ax, e, mean, np.array([s[1] or 0.0 for s in st]), np.array([s[2] or 0.0 for s in st]),
                 color=S.SERIES[key], probe=probe, label=label)
        return
    c = 0.5 * (e[1:] + e[:-1])
    style = S.look(key, linewidth=S.LW_CURVE)
    ax.plot(c, mean, **style, **S.marker(key, size=PT_MARKER), label=label)
    if band and len(values) > 1:
        ax.fill_between(c, mean - np.array([s[1] for s in st]), mean + np.array([s[2] for s in st]),
                        color=style["color"], alpha=0.2, linewidth=0)


def _pt_panel(ax, docs: dict, rows: tuple, value, ylabel: str, style: str = "B", probe: bool = False) -> None:
    """value(doc) per pT bin for each row's runs, absolute; drawn in reverse, so that the first row (188 classes)
    lies on top. One mode a panel: filled points (PI 2026-11-11)."""
    e = _edges(docs)
    for row in reversed(rows):
        _pt_series(ax, e, [value(d) for d in _runs_in(docs, row)], row, S.LEGEND[row], style=style, probe=probe)
    ax.set_xlim(e[0], e[-1])
    if ylabel:
        S.labels(ax, ylabel=ylabel)


def _pt_figure(path: pathlib.Path, panels: tuple, rows: tuple, value, ylabel: str, style: str = "B") -> None:
    """The family's form: panels (perf_vs_pt folder, title) two to a row, each on its own y axis (the test sets and
    modes differ in level), every axis labelled; unjoined points with bin bars; the figure legend above in the
    columns of the networks' groups (gallery_roc._group_legend)."""
    docs = {sub: M._load(PERF / sub) for sub, _ in panels}
    if not all(docs.values()):
        raise Absent("no performance against jet pT (perf_vs_pt/" + ", ".join(s for s, d in docs.items() if not d) + ")")
    n_rows = (len(panels) + 1) // 2
    legend_rows = max(len(c) for c in _legend_cols(rows))
    fig, axes, _ = S.grid_cm(n_rows, 2, w=6.1, aspect=0.87, top=0.95 + 0.42 * legend_rows, left=1.3, wgap=1.5,
                             hgap=1.6)
    handles = {}
    for ax, (sub, text) in zip(axes.flat, panels):
        S.title(ax, text)
        S.fine_grid(ax)
        _pt_panel(ax, docs[sub], rows, value, ylabel, style)
        S.labels(ax, xlabel=X_PT)
        S.data_ylim(ax)              # every bar clear of the frame (L7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))     # no 2.5 step: its labels need a digit more
        S.clear_edge_labels(ax)      # the decimals the tick step needs, the same on every tick
        hi = _edges(docs[sub])[-1]   # the last label at 2500 would run into the next panel or off the page
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, hi=hi: "" if v >= hi - 1e-9 else f"{v:g}"))
        for h_, row in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(row, h_)
    fig.align_ylabels(axes[:, 0])
    by_name = {S.LEGEND[r]: r for r in rows}
    _group_legend(fig, {by_name[k]: h for k, h in handles.items()})
    S.letter_panels(axes.flat)
    S.save(fig, path)


def draw_pt(path: pathlib.Path, style: str = "B") -> None:
    """The family (PI 2026-11-13, L4): accuracy per jet-pT bin, linear probes over fine-tuned networks, JetClass-II
    beside JetClass, for the label sets and the mass networks (lighter shades); nothing else. style: the PI's
    comparison pages (gallery_compare)."""
    _pt_figure(path, PT_PANELS, PT_ROWS, _accuracy, "Accuracy", style)


def draw_pt_part(path: pathlib.Path) -> None:
    """The family's form for the 17-class partitions, the five random partitions pooled (one point, the bar their
    SD), with 188 and 17 classes for reference."""
    _pt_figure(path, PT_PANELS, PT_PARTITIONS, _accuracy, "Accuracy")


def draw_pt_bbcc(path: pathlib.Path) -> None:
    """The slice of the old panel (d), in both modes: the area under the ROC curve of X->bb against X->cc per
    JetClass-II jet-pT bin, p_bb / (p_bb + p_cc) of the 188 outputs, for the label sets and the mass networks."""
    _pt_figure(path, PT_BBCC, PT_ROWS, _bb_cc, "AUC")


# ------------------------------------------------------------------ the pages
# the supplement members' captions (gallery_captions holds the main-text and appendix ones), filled like those
SUPPLEMENT_CAPTIONS = {
    "A17-probe": (
        "A multilayer perceptron trained to regress the jet mass from frozen features (an MLP probe) gives a better "
        "mass resolution on the networks pretrained with an added mass regression (+ mass) than on the same label "
        "sets without it. Panels (a) and (b) show the median of r = m/m_gen, the ratio of estimated to generator mass, "
        "against jet pT and generator mass, and panels (c) and (d) the resolution sigma_r, half "
        "the central 68% range of r divided by its median, with the soft-drop mass in black and the random network, "
        "a network with its initial random weights, as crosses. The lower panels show the ratio to the soft-drop "
        "value. Horizontal bars span the bin, and vertical bars show the standard deviation over three pretraining "
        "runs, two for + mass, or three random initializations. The values are in tables {tab:mass} and "
        "{tab:massdetail}."
    ),
    "A17-ridge": (
        "A linear regression of the jet mass on frozen features (a ridge probe) also gives a better mass resolution "
        "on the networks pretrained with an added mass regression (+ mass) than on the same label sets without it. "
        "The panels have the form of figure {fig:A17-probe}, with the median of r = m/m_gen in (a) and (b), the "
        "resolution sigma_r in (c) and (d), the soft-drop mass in black and the random network as crosses. Horizontal "
        "bars span the bin, and vertical bars show the standard deviation over three pretraining runs, two for "
        "+ mass, or three random initializations. The values are in table {tab:massdetail}."
    ),
    "F9a": (
        "The panels show the response r = m/m_gen, a jet's estimated mass over its generator-level mass, of the mass "
        "output of the networks pretrained with 162 and 17 classes and an added mass regression (+ mass), for (a) "
        "resonant and (b) QCD JetClass-II test jets, with the soft-drop mass in black. Each histogram is the mean of "
        "two pretraining runs, the two networks nearly coincide, and both panels share their axes."
    ),
    **{f"A17-{key}": (
        f"The panels show the mass output of the networks pretrained with 162 and 17 classes and an added mass "
        f"regression (+ mass) on {group} JetClass-II test jets, in the form of figure {{fig:F9b}}, the median of "
        f"r = m/m_gen in (a) and (b) and the resolution sigma_r in (c) and (d) against jet transverse momentum pT and "
        f"generator mass, with the soft-drop mass in black. Horizontal bars span the bin, vertical bars show the "
        f"standard deviation over two pretraining runs, and the lower panels show the ratio to the soft-drop value. "
        f"The values are in table {{tab:massdetail}}."
    ) for key, group in (("res", "resonant"), ("qcd", "QCD"))},
    **{pid: (
        f"The panels show the soft-drop mass of JetClass-II QCD test jets before any selection (black) and after a "
        f"cut on 1 - g_QCD, one minus the summed QCD score of the network output used directly, at QCD efficiencies "
        f"of 10% (orange) and 1% (plum), for the networks pretrained with {nets}, each named above its panel. Each "
        f"histogram is the mean over {runs} pretraining runs, and the lower panels show the ratio to the inclusive "
        f"distribution. Figure {{fig:{dist}}} summarizes each distortion as a distance."
    ) for pid, nets, runs, dist in (
        ("F10a", "188, 162, 43 and 17 classes", "three", "F10b"),
        ("F10a-mass", "162 and 17 classes and an added mass regression (+ mass)", "two", "F10b"),
        ("F10a-flav", "the three flavor-boundary partitions of 17 classes", "two", "F10b-flav"))},
    "F10b-flav": (
        "The curves show the Jensen-Shannon distance between the soft-drop mass distributions of all JetClass-II QCD "
        "test jets and of those passing a cut on 1 - g_QCD, the network output used directly, against QCD efficiency "
        "eps_B, for the three flavor-boundary partitions with 188 and 17 classes for reference. The random partitions "
        "are not shown, because their output scores on these test jets have not been computed. Bands show ±1 standard "
        "deviation over two pretraining runs per partition or three for 188 and 17 classes, and the dotted line is a "
        "random selection of as many QCD jets. Values are in tables {tab:sculpt} and {tab:sculptboot}."
    ),
    "F10c": (
        "Each point shows the inverse Jensen-Shannon divergence between the soft-drop mass distributions of all "
        "JetClass-II QCD test jets and of those passing a cut on 1 - g_QCD, the network output used directly, against "
        "the QCD rejection, both at the cut that keeps 50% of resonant test jets, as in figure 3 of the DisCo paper. "
        "Points higher up distort the QCD mass less and points further right reject more QCD jets. Points are means "
        "over three pretraining runs, two for networks with an added mass regression (+ mass, lighter shades) and the "
        "flavor-boundary partitions, and bars show the standard deviation over runs. The values are in table {tab:sculpt}, "
        "at 50% signal efficiency."
    ),
    "F10-ft": (
        "The curves show the Jensen-Shannon distance between the soft-drop mass distributions of all QCD test jets and "
        "of those passing a cut on 1 - g_QCD, against QCD efficiency eps_B, for networks fine-tuned on one million "
        "class-balanced JetClass-II jets and scored on two million natural-mix test jets, with training from scratch "
        "in grey. Fine-tuning ran with a cap of 40 epochs except three cells with the fourfold cap of 160 epochs, the "
        "first run of 17 classes and both runs of 17 classes + mass, and from scratch ran its full schedule. Bands show "
        "±1 standard deviation over three pretraining runs, two for networks with an added mass regression (+ mass), "
        "or three trainings from scratch, and the dotted line is a random selection of as many QCD jets."
    ),
    "F10-ft-part": (
        "As in figure {fig:F10-ft}, the curves show the Jensen-Shannon distance between the soft-drop mass "
        "distributions of all QCD test jets and of those passing a cut on 1 - g_QCD, against QCD efficiency eps_B, "
        "here for the 17-class partitions fine-tuned on one million class-balanced JetClass-II jets, with 188 and 17 "
        "classes for reference. Every partition ran with the 160-epoch cap, 188 classes and the second and third runs "
        "of 17 classes with the 40-epoch cap. Bands show ±1 standard deviation over two pretraining runs per "
        "flavor-boundary partition, three for 188 and 17 classes, or the five random partitions, and the dotted line "
        "is a random selection of as many QCD jets."
    ),
    "PT-part": (
        "With linear probes on JetClass-II, every 17-class partition is less accurate than the network pretrained with "
        "188 classes and more accurate than the network pretrained with 17 classes in every jet pT bin. The panels "
        "repeat figure {fig:PT} for the 17-class partitions, each a regrouping of the 188 classes into 17, named by "
        "its flavor boundary, a class boundary between decays that differ only in quark flavor, with the 188- and "
        "17-class networks for reference. Points are means over two pretraining runs per flavor-boundary partition, "
        "three for 188 and 17 classes, or the five random partitions, horizontal bars span the bin, and vertical bars "
        "show ±1 standard deviation. Values are in table {tab:ptbins}."
    ),
    "PT-bbcc": (
        "The panels show the area under the ROC curve for separating X→bb̄ from X→cc̄ jets in each JetClass-II jet "
        "pT bin, with the discriminant p(bb̄)/(p(bb̄) + p(cc̄)) of the 188 class outputs, for linear probes (a) and "
        "fine-tuned networks (b) of figure {fig:PT}, the label sets and the networks with an added mass regression "
        "(+ mass, lighter shades). A bin holds 240 to 930 jets of each kind. Points are means over three pretraining "
        "runs, two for + mass, horizontal bars span the bin, and vertical bars show ±1 standard deviation over runs, "
        "without the uncertainty from the few test jets per bin, so no bin is compared. Values are in table "
        "{tab:ptbins}."
    ),
    "F11-probe": (
        "With a linear probe in place of the classifier of figure {fig:F11}, the random network, a network with its "
        "initial random weights, stays far below every pretrained network on both signals. The panels show the "
        "figure of merit against QCD efficiency for (a) X→bb̄ and (b) X→bs jets against QCD jets in the soft-drop "
        "mass window of figure {fig:F11}, for a linear probe trained on the frozen features of each label set, of "
        "the networks with an added mass regression (+ mass) and of the random network. Lines are means and bands ±1 "
        "standard deviation over three pretraining runs, two for + mass, or three random initializations. Values at "
        "1% QCD efficiency are in table {tab:xbx}."
    ),
}

SOPHON_FIG2_CAPTION = (
    "FIG. 2. Benchmark of Sophon model’s performance in the X→bb and X→bs jet tagging tasks, with the signal "
    "originated from a BSM resonance X0 with mX0 = 200 GeV that decays to bb and bs, and backgrounds corresponding "
    "to the 40 fb−1 of the full SM processes. The performance of various DNN models is compared in terms of the "
    "search’s discovery significance versus SM background efficiency calculated within the mass window "
    "150 < mSD < 230 GeV. A major conclusion is that the Sophon model’s direct tagging discriminant (for X→bb "
    "tagging) and its transfer learning version (for X→bs tagging) both outperform the current best results "
    "achievable in the LHC experiment using a ParT or ParticleNet tagger. It also confirms that the model improves "
    "performance when trained by large-scale classification. [Validation page, not one of our figures: Sophon's "
    "own caption, verbatim.]")
F1_CAPTION = (
    "(a) Six example classes as schematic jets and the 17-class group holding each (j: a quark "
    "or gluon); below, the 188 classes as cells, a bar spanning several cells being one class of that label set: 162 "
    "classes merge the QCD classes, 43 keep the heaviest quark flavor, 17 merge flavors. (b) One network is "
    "pretrained per label set on the same jets from the same initial weights, with three seeds, then used frozen "
    "or fine-tuned on five tasks.")
F9B_CAPTION = (
    "The mass output of networks pretrained on JetClass-II with 162 and 17 classes and an auxiliary mass "
    "regression (two runs): median response r = m/m_gen (a, b) and resolution sigma_r, half the 15.9-84.1% "
    "quantile range of r over its median (c, d), against pT (a, c) and generator mass (b, d), with the soft-drop "
    "mass (black). Points at bin centers; horizontal bars: the bin, vertical: SD over runs. Lower panels: ratio to "
    "the soft-drop mass.")
F10B_CAPTION = (
    "Cuts on the pretraining head of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (three "
    "runs) and with 162 and 17 classes plus mass regression (lighter shades, two runs), used without further "
    "training: Jensen-Shannon distance between the soft-drop mass of all QCD test jets and of those passing a "
    "cut on 1 - g_QCD, against its QCD efficiency eps_B. Bands: SD over runs. Black dotted: a "
    "random selection of as many QCD jets.")
F11_CAPTION = (
    "Networks pretrained on JetClass-II, on Sophon's (a) X→bb̄ and (b) X→bs samples against QCD jets in the "
    "soft-drop mass window: signal efficiency over the square root of QCD efficiency, against QCD efficiency, for a "
    "classifier on each network's frozen features (transfer), the 188-class head's direct discriminant (X→bb̄) and "
    "networks trained from scratch. Means over three pretraining runs (from scratch: three trainings), bands the SD. "
    "Curves end where fewer than 20 background jets remain.")
PT_CAPTION = (
    "Networks pretrained on JetClass-II, in bins of jet transverse momentum: (a) accuracy of linear probes on "
    "class-balanced JetClass-II, with 188, 162, 43 and 17 pretraining classes; (b) the same for 17-class partitions with controlled flavor boundaries, with 188 and 17 classes for "
    "reference; (c) accuracy on JetClass after fine-tuning on one million jets; (d) the probes' AUC of X→bb̄ "
    "against X→cc̄. Points: means at bin centers; horizontal bars: the bin, vertical: SD over pretraining runs.")
ATLAS_LOOK = ("Sophon's font and sizes in place of ATLAS's; inward ticks on four sides and no legend frame "
              "(2603.04084, round 3); no experiment label, and ATLAS's text block (sqrt(s), jet definition, pT and mass selection) is dropped, so our selection appears on no figure")
A17_DIFFERS = (
    "As F9b, on one jet group and with a second row against generator mass, which ATLAS does not draw (its Fig. 9 has "
    "pT response instead); the legend appears once, in the first panel, to keep four panels light. Eight axes in all, a lower axes under each "
    "of four panels: within the limit of four panels only if lower axes do not count, for the PI to decide. 5.6 in high, not "
    "the 4.8 in of a plain 2x2, because each panel carries a lower axes. The rows hold different jets: the lower covers "
    "generator mass 20 to 500 GeV, so 24.5% of QCD and 4.7% of resonant jets are in the upper row only. No window on the "
    "regressed mass, where ATLAS keeps 40 < m^Reg < 300 GeV. " + ATLAS_LOOK + ".")
MASS_RUNS_NOTE = ("Mean of two runs for each mass regression model (the third runs were cut on 2026-10-22); "
                  "the soft-drop mass is the same in every file. Nothing else is absent.")

PAGES = [
    Page("SOPHON-FIG2", "Sophon Fig. 2 redrawn through the paper's style (validation)", ["2405.12972/fig2.png"],
         "Sophon Fig. 2", SOPHON_FIG2_CAPTION,
         "Round 4: only the palette (style_v5) and the background (grid, ticks, legend frame: 2603.04084's, round 3) "
         "differ from Sophon's figure; the data, axes, sizes, line styles and text are Sophon's, so this page is exempt "
         "from the round-4 layout checks. In detail, against Sophon's native settings (sophon_native.json): grid a faint dashed "
         "#b0b0b0 line at major y ticks only (0.35 pt, alpha 0.25), where Sophon's is 0.5 pt at full strength at major "
         "and minor x ticks and major y ticks; spines 0.9 pt, not 0.8; ticks pointing in, 5.3 and 2.7 pt, mirrored on all "
         "four sides with minor ticks on y too, where Sophon's point out (3.5 and 2 pt) on the left and bottom only, with "
         "no y minor ticks; no legend frame, where Sophon's is framed (#cccccc, alpha 0.8, rounded); palette ours: Sophon "
         "(and its X→bq discriminant, dotted) in the 188 colour, Sophon* (42 classes) in the 43 colour, ParT dark grey and "
         "ParticleNet mid grey, where Sophon has tab10 blue, orange and green, pure blue and grey. Unchanged and checked "
         "against the native PDF by compare_fig2.py: the data points, axes box (279 x 98.84 pt each), scales, limits, "
         "y ticks, eight curves at 1.5 pt with Sophon's solid and dotted dashes, every vertex inside the axes at the "
         "native position (0.0000 pt off); also STIX text, its sizes, the figure size and the panel gap (hspace 0.2, kept "
         "against style_v4's 0.25). The file encoding differs (pdf.fonttype 42). Without its frame the upper legend's "
         "ParticleNet key lies on the curves near 3e-3, where Sophon's alpha-0.8 frame covered them. It breaks our own "
         "rules on purpose (dotted series, long caption), because this page checks the style against Sophon.",
         draw_sophon_fig2,
         "The points are those of the native PDF after its path simplification; a curve starts where the PDF's does."),
    Page("F1", "Design of the study", ["2405.12972/fig1.png"], "Sophon Fig. 1; finish after arXiv:2610.41889 Fig. 1", F1_CAPTION,
         "The PI's drawing (2026-11-12), replacing the Sophon-shaped card: example jets with a fine-to-coarse tree, all "
         "188 classes as a strip of cells read from the label maps, and a pipeline with a task matrix (open: frozen "
         "features, filled: fine-tuned). Drawn 13 cm wide. Boxes only for the network and its head, white; one accent "
         "(Tol muted wine) for the pretrained network, its features and the two uses.", draw_f1,
         data_note="X→bb̄ and X→bs use frozen features only. The jets are drawings, not events."),
    Page("F9a", "Mass regression: response", ["ATL-PHYS-PUB-2024-015/fig7.png"], "ATLAS 2024-015 Fig. 7", "",
         "Two jet groups instead of mass and pT responses, as we regress mass only, with panel titles ATLAS lacks. Three "
         "series instead of two: a thin solid black reference and the 162- and 17-class colours, solid, instead of purple dashed and green. "
         "Fraction of jets on the y axis instead of a.u., Response (m/m_gen) on x instead of m_Reco/m_Truth. x range from "
         "the data, per panel, instead of a fixed 0.7 to 1.3, because the soft-drop mass here is far wider. No vertical "
         "dotted line at 1, a second reference the PI limit forbids. No bands: ATLAS's are thinner than its lines. Same "
         "ratio height (0.44), but ratio ranges of about 0.1 to 2.6 and 0.2 to 3.1 against ATLAS's 0.35 to 1.55. "
         + ATLAS_LOOK + ".",
         draw_f9a, MASS_RUNS_NOTE),
    Page("F9b", "Mass regression: across jet pT", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8", F9B_CAPTION,
         "All jets in ten pT bins to 2500 GeV in reconstructed pT (PI: at least eight, ATLAS's density), instead of "
         "ATLAS's bins to 1500 GeV in truth pT. Connected steps instead of one disconnected segment per bin, and no bands (PI limit), though "
         "ATLAS draws them. Three series instead of two, colours as F9a, each model's marker (v, P) at the bin centres. No "
         "dotted line at 1 in the median panel. Faint dashed grid at major y ticks (round 3), where ATLAS has none. Median response (r) and Resolution "
         "(sigma_r) instead of ATLAS's longer axis words. Both lower panels are the ratio to the soft-drop mass (2026-11-11; "
         "ATLAS's lower resolution panel is an RSD), at ATLAS's height (0.44); the soft-drop mass is inside both ranges, "
         "the resolution on a log axis; step lines carry no markers. " + ATLAS_LOOK + ".",
         draw_f9b, MASS_RUNS_NOTE),
    Page("A17-res", "Mass regression on resonant jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         "", A17_DIFFERS, lambda path: _a17(path, "resonant"), MASS_RUNS_NOTE),
    Page("A17-qcd", "Mass regression on QCD jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         "", A17_DIFFERS, lambda path: _a17(path, "qcd"), MASS_RUNS_NOTE),
    Page("A17-probe", "Mass from frozen features, MLP probe (supplement)", ["ATL-PHYS-PUB-2024-015/fig8.png"],
         "ATLAS 2024-015 Fig. 8", "",
         "As F9b, with seven series and the soft-drop mass: the MLP probe on the frozen features of the label sets, the "
         "mass networks (lighter shades) and the random network (dark grey crosses, apart from the black soft-drop "
         "points), all filled, the figure's only mode (PI 2026-11-13); the mass outputs are F9b's. " + ATLAS_LOOK + ".",
         draw_a17_probe, "Three runs per label set, two per mass network, three random initializations."),
    Page("A17-ridge", "Mass from frozen features, linear (ridge) probe (supplement)", ["ATL-PHYS-PUB-2024-015/fig8.png"],
         "ATLAS 2024-015 Fig. 8", "",
         "A17-probe's form for the ridge regression (atlas_res_fine estimator \"ridge\"), new (PI 2026-11-13).",
         draw_a17_ridge, "Three runs per label set, two per mass network, three random initializations."),
    Page("F10a", "Mass sculpting: QCD mass", ["2604.09809/fig8.png"], "CMS JME-25-001 Fig. 8", "",
         "Six panels, one per model (PI), all jet pT, instead of CMS's six (two models, three pT ranges), with "
         "one y axis across them. Two working points instead of three, because the files hold 10% and "
         "1%. Mass from 20 to 500 GeV in 20 GeV bins, the files' whole range (CMS: 30 to 250 GeV), as step histograms; previously to 260 GeV, where the files run to 500 GeV, "
         "and the twelve bins cut hold 5.9% of the inclusive jets and 6.4 to 7.6% of the selected ones (fractions stay "
         "those of all jets to 500 GeV). Colours are black, orange and plum instead of black, blue, red, orange: "
         "CLASS10 colours, none of which means a network (PI 2026-11-13, L6). "
         "Markers as CMS's first three, at 2.5 pt against 1.59 to 2.12 pt so that the bars show, inclusive on top. No "
         "vertical error bars: the PI's limit, where CMS's are statistical. Bars at 1.0 pt instead of 0.4. Dashed reference "
         "at 1 instead of solid, and \"Ratio\" instead of a stacked fraction. Same ratio height (0.33). Jet mass (m_SD) "
         "instead of m_SD, fraction of jets instead of A.U., titles instead of a label block, no legend frame, as CMS; "
         "STIX at 10 and 8 pt instead of Helvetica at 6.9 and 5.8 pt; inward ticks, as CMS.",
         draw_f10a,
         "Mean of three runs. Round 4: the four label sets in a 2x2, working points in black and greys; the mass "
         "models move to F10a-mass."),
    Page("F10a-mass", "Mass sculpting: the mass-regression models (appendix)", ["2604.09809/fig8.png"],
         "CMS JME-25-001 Fig. 8", "", "F10a's form for the two mass-regression models (PI, round 4).",
         draw_f10a_mass, "Mean of two runs."),
    Page("F10a-flav", "Mass sculpting: the flavor-boundary partitions (appendix)", ["2604.09809/fig8.png"],
         "CMS JME-25-001 Fig. 8", "",
         "F10a's form for the three 17-class flavor-boundary partitions (PI, 2026-11-07, M4).", draw_f10a_flav,
         "Mean of two runs."),
    Page("F10b", "Mass sculpting: how much", ["2604.09809/fig9.png"], "CMS JME-25-001 Fig. 9 left", F10B_CAPTION,
         "One panel: CMS's right panel (against pT) is not drawn, since our curves hold all jet pT. Twelve efficiencies "
         "on a log axis rising to the right, the files' grid and the direction of the efficiency axis in Sophon Fig. 2 "
         "and F11, instead of five categories falling to the right. Starts at the 5.2e-3 grid point (0.5%), where CMS's "
         "panel stops; the four tighter points are cut, because the distance there is mostly finite statistics (at 0.1%, "
         "a random selection of as many jets gives 0.108 against the curves' 0.119 to 0.125). Distance over 20 to 500 GeV "
         "and all pT instead of CMS's 30 to 250 GeV and 600 to 1000 GeV. Six series (PI) instead of two, one marker shape each. "
         "Label sets in their colours; the mass regression models in their label sets' colours with their own markers (v, P). "
         "Markers joined by a 1.1 pt line, not step markers: each point is a cumulative selection, so a segment would span "
         "a bin that does not exist (for the PI to decide). Faint dashed grid at major y ticks (round 3), where "
         "CMS has none; no major y tick falls in this log range, so no grid line shows. Jensen-Shannon distance (d_JS) and QCD efficiency instead of JSD and eps_B [%]; y limits from the "
         "data instead of 0.007 to 1. The random-selection floor drawn as a black dotted line, where CMS has none (PI, 2026-11-07). No legend frame and no label "
         "block; STIX at 10 and 8 pt with inward ticks. Bands of the spread over runs (PI, 2026-11-08), where CMS draws "
         "none; the flavor-boundary partitions moved to F10b-flav.",
         draw_f10b,
         "Mean of two runs for the two mass regression models, three for the others. Appendix; main text only if the "
         "sculpting bootstrap (v2_head/sculpting_boot.json, on the map) gives at 1% 17 - 188 > 0 and 17 + mass - 17 < 0, "
         "each |diff| > 3 sigma (gallery_placement.F10B_RULE)."),
    Page("F10c", "Mass sculpting against rejection", ["2001.05310/fig3.png"], "DisCo Fig. 3", "",
         "DisCo's plane (1/JSD at 50% signal efficiency against R50), both axes logarithmic; one point per model in its "
         "color and marker with bars of the spread over pretraining runs on both axes, where DisCo draws envelopes over "
         "six trainings per method. Legend above the panel; no ideal-tagger arrow.",
         draw_f10c,
         "New page. Reads sig_eff_50 {rejection, jsd} from each run's sculpting file (head/sculpting_v3/<model>_sculpting.json, "
         "the 22 models of the sculpting rerun, the flavor-boundary partitions among them)."),
    Page("F10b-flav", "Mass sculpting: the flavor-boundary partitions, how much (appendix)", ["2604.09809/fig9.png"],
         "CMS JME-25-001 Fig. 9 left", "",
         "F10b's form for the three flavor-boundary partitions, with 188 classes for reference (PI, 2026-11-08: out of "
         "the main-text F10b). A separate page rather than a panel on F10a-flav, whose 2x2 of mass distributions has no "
         "room for a distance panel.",
         draw_f10b_flav, "Two runs per partition, three for 188 classes."),
    Page("F10-ft", "Mass sculpting after fine-tuning (appendix)", ["2604.09809/fig9.png"],
         "CMS JME-25-001 Fig. 9 left", "",
         "F10b's form after fine-tuning (PI, 2026-11-08), on the same natural-mix test jets as F10b, unweighted, so "
         "the floor is F10b's: a random selection of as many QCD jets.",
         draw_f10_ft,
         "Reads head/sculpting_ft/<model>.json (experiments/EVAL/sculpting_ft.py): the curve's js_distance and n_pass; "
         "absent until pulled. From scratch: the three full-schedule networks (scratch-s1..3), grey solid. 17 classes "
         "run 1 and both 17 + mass runs are the long-budget cells (160-epoch cap), the others the 40-epoch cells."),
    Page("F10-ft-part", "Mass sculpting after fine-tuning, 17-class partitions (supplement)", ["2604.09809/fig9.png"],
         "CMS JME-25-001 Fig. 9 left", "",
         "F10-ft's form for the three flavor-boundary partitions and the five random partitions (one curve, the band "
         "their SD), with 188 and 17 classes for reference (PI 2026-11-13).", draw_f10_ft_part,
         "Every partition cell is a long-budget cell (160-epoch cap); 188 classes and 17 classes runs 2 and 3 the "
         "40-epoch cells, 17 classes run 1 the long-budget cell."),
    Page("F11", "Resonance tagging", ["2405.12972/fig2.png"], "Sophon Fig. 2", F11_CAPTION,
         "Same layout, limits on x, fonts and legend place; round 3's faint grid and thinner lines. y axis eps_S/sqrt(eps_B), in words figure of merit, instead of the "
         "significance Z, and x axis QCD efficiency instead of SM background efficiency, because Z needs the SM backgrounds, which "
         "we do not have. Both panels on Sophon's samples. Roles: the direct discriminants of 188, 162 and 162 + mass are "
         "Sophon's direct curve, dash-dot in their colours (Sophon: grey, dotted); the classifiers on frozen features of the "
         "label sets and the mass networks (lighter shades) take Sophon's transfer role, on both signals; "
         "Sophon* (42 classes, separately trained) has no counterpart; the from-scratch networks (grey) take the ParT and "
         "ParticleNet role. y limits from the data. Curves have 26 points or fewer (the fits' "
         "efficiency grid) and end at 20 background jets, so they are coarser and shorter than Sophon's.",
         draw_f11,
         "Reads xbx/<signal>/<model>.json and xbx_scratch/<signal>/s<k>.json (three from-scratch seeds on both signals). "
         "The significance Z is not computed: it needs the SM backgrounds. X→bs has no direct discriminant (no "
         "pretraining class), and 43 and 17 classes merge X→bb̄ with other decays."),
    Page("F11-part", "Resonance tagging, 17-class partitions (appendix)", ["2405.12972/fig2.png"], "Sophon Fig. 2", "",
         "F11's form for the three flavor-boundary partitions and the five random partitions (one curve, the band their "
         "SD), with 188 and 17 classes for reference; no direct discriminant (every partition merges X→bb̄ with other "
         "decays) and no from scratch.", draw_f11_part, "Two runs per flavor-boundary partition, three for 188 and 17."),
    Page("F11-probe", "Resonance tagging, linear probes (supplement)", ["2405.12972/fig2.png"], "Sophon Fig. 2", "",
         "F11's form for a linear probe on the same frozen features (readout \"linear\" of the xbx fits): the label sets, "
         "the mass networks and the random network (three initializations), solid, the figure's only mode.",
         draw_f11_probe, "The random network's value in the resonance table is this linear probe."),
    Page("PT", "Performance against jet pT", ["2004.08262/fig17.png", "1808.07858/fig2.png"],
         "CMS DeepAK8 Fig. 17, ATLAS top/W Fig. 2", PT_CAPTION,
         "Absolute accuracy per pT bin, not the precedents' mistag rate or rejection at a working point, which would be "
         "a bound in many bins of the 188-class runs. Unjoined points at the bin centres with a horizontal bar across "
         "the bin and a vertical bar of the SD over runs, as the precedents. Rows linear probe and fine-tuned, columns "
         "JetClass-II (ten bins of reconstructed jet pT) and JetClass (five bins), each panel on its own y axis (PI "
         "2026-11-11: every axis labelled). Panel titles and a figure legend instead of the precedents' text block; no "
         "experiment label; the label sets and the mass networks only (PI 2026-11-12: nothing else mixed in).",
         draw_pt,
         "Three runs per label set, two per mass network. JetClass-II fine-tuned: 17 classes run 1 and both 17 + mass "
         "runs are the long-budget cells (160-epoch cap), the others the 40-epoch cells, as Table 1 reads them; "
         "JetClass fine-tuned: the long-budget cells (budget_long/leg2) of Table 1."),
    Page("PT-part", "Performance against jet pT, 17-class partitions (supplement)", ["2004.08262/fig17.png"],
         "CMS DeepAK8 Fig. 17", PT_CAPTION,
         "PT's form for the three flavor-boundary partitions and the five random partitions (one point, the bar their "
         "SD), with 188 and 17 classes for reference.", draw_pt_part,
         "Two runs per flavor-boundary partition, three for 188 and 17 classes; every partition's fine-tuned JetClass-II "
         "cell is a long-budget cell (160-epoch cap)."),
    Page("PT-bbcc", "X→bb̄ vs X→cc̄ against jet pT (supplement)", ["2004.08262/fig17.png"], "CMS DeepAK8 Fig. 17",
         PT_CAPTION,
         "The old PT panel (d) in both modes: the area under the ROC curve of X→bb̄ against X→cc̄ per JetClass-II jet-pT "
         "bin, for the label sets and the mass networks, on PT's axes.", draw_pt_bbcc,
         "Discriminant p_bb / (p_bb + p_cc); 5,000 jets of each class in all, 240 to 930 in a bin."),
]

import dataclasses  # noqa: E402
import gallery_captions  # noqa: E402  the paper's captions replace those above (PI 2026-11-12)
PAGES = [dataclasses.replace(p, caption=gallery_captions.fill(SUPPLEMENT_CAPTIONS[p.id])) if p.id in SUPPLEMENT_CAPTIONS
         else p for p in gallery_captions.apply(PAGES)]
