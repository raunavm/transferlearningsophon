#!/usr/bin/env python3
"""The MLST paper's figures, from the same pulled data and the same selection functions as the
tables (experiments/FIGS/mlst_results.py), so a figure and its table cannot show different
numbers. The style is mlst_style.py's (mplhep's CMS style, LaTeX-set Computer Modern in the
figure job, colour and marker per configuration fixed there).

  label_sets               the class of a selection of decays in every label set (label maps)
  probes_vs_label_set      linear probes at the largest training set, every configuration
  probes_vs_training_size  linear probes against the training-set size, the hierarchy
  pairs                    binary probes between classes; open markers where a label set
                           merges the pair, half filled where it merges part of it
  per_class_jc2            JetClass-II: each class's (1 - AUC) over the 188-class model's, run by
                           run, averaged; rows grouped by topology, flavor siblings adjacent
  finetune_vs_training_size  fine-tuning against the training-set size, with the ratio to the
                           network trained from scratch
  mass_response            mass resolution and scale in bins of pT and generated mass
  sculpting                QCD soft-drop mass before and after a cut on the tagger score

Every quantity compared in ratios is drawn on a logarithmic axis. Points are the mean over the
pretraining runs present and error bars their standard deviation (two runs: their range); the
runs themselves are small dots. A figure whose data are not pulled yet is skipped.

    python3 experiments/FIGS/mlst_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
from matplotlib import ticker, transforms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mlst_style as style  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "experiments/FIGS/data/v2"
CLASSES = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
ROW_NAME = {**{k: f"{v} classes" for k, v in CLASSES.items()},
            "L162_MASS": "162 classes, mass regression", "R16_Q1_MASS": "17 classes, mass regression",
            **{f"RAND2_p{k}": f"Random partition {k}" for k in range(1, 6)},
            "FLAV_F0": "No flavor boundary", "FLAV_F1": r"$b$-quark boundary",
            "FLAV_F1R": "Random flavor boundary", "INIT": "Randomly initialized", "scratch-v2": "From scratch"}
# (dataset, metric, panel label)
PROBE_PANELS = (("jc2_188", "macro_auc_ovr", "JetClass-II"), ("jc2_188", "macro_auc_qcd", "JetClass-II, QCD classes"),
                ("jc1", "macro_auc_ovr", "JetClass"), ("top", "auc", "Top tagging"),
                ("qg", "auc", "Quark/gluon tagging"), ("qg_herwig", "auc", "Quark/gluon, Herwig"))
SIZE_PANELS = (PROBE_PANELS[0], PROBE_PANELS[2], PROBE_PANELS[3], PROBE_PANELS[4])
FILL = {"apart": "full", "partly": "left", "merged": "none"}
# The schematic's classes, top to bottom, and the columns: (label map, column, heading).
SCHEMATIC_CLASSES = (("X_bb", r"$X\to b\bar b$"), ("X_bc", r"$X\to bc$"), ("X_bq", r"$X\to bq$"),
                     ("X_cc", r"$X\to c\bar c$"), ("X_cs", r"$X\to cs$"), ("X_ss", r"$X\to s\bar s$"),
                     ("X_qq", r"$X\to q\bar q$"), ("X_gg", r"$X\to gg$"), ("X_ee", r"$X\to ee$"),
                     ("X_mm", r"$X\to \mu\mu$"), ("X_YY_bbqq", r"$X\to YY\to bbqq$"),
                     ("X_YY_ccqq", r"$X\to YY\to ccqq$"), ("X_YY_qqqq", r"$X\to YY\to qqqq$"),
                     ("QCD_bb", "QCD"))
# rows that differ only in quark flavor (the flavor orbits of flavour_pair_map.v2.csv)
SCHEMATIC_ORBITS = (("X_bb", "X_qq"), ("X_YY_bbqq", "X_YY_qqqq"))
# A heading of None is the label set's number of classes, counted from its label map.
SCHEMATIC_COLUMNS = (("rung_label_maps.v1.csv", "L188", None), ("rung_label_maps.v1.csv", "L162", None),
                     ("rung_label_maps.v1.csv", "R42_Q1", None), ("rung_label_maps.v1.csv", "R16_Q1", None),
                     ("rand_label_map.v2.csv", "RAND2_p1", "Random partition 1"),
                     ("flavour_pair_map.v2.csv", "FLAV_F0", "No flavor boundary"),
                     ("flavour_pair_map.v2.csv", "FLAV_F1", "$b$-quark boundary"),
                     ("flavour_pair_map.v2.csv", "FLAV_F1R", "Random flavor boundary"))
# Paul Tol's pale and light schemes: tints no data series uses, so a colour in the schematic
# means only "a shared class within this column"
SCHEMATIC_TINTS = ("#BBCCEE", "#EEEEBB", "#FFCCCC", "#CCDDAA", "#CCEEFF", "#DDDDDD", "#EEDD88",
                   "#FFAABB", "#99DDFF", "#BBCC33")
# per_class_jc2's columns: (configuration, heading)
HEAT_COLUMNS = (("L162", "162"), ("R42_Q1", "43"), ("R16_Q1", "17"), ("RAND", "Random\npartitions"),
                ("FLAV_F0", "No flavor\nboundary"))


def _mlst():
    spec = importlib.util.spec_from_file_location("mlst_results", REPO / "experiments/FIGS/mlst_results.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _look(M, row: str) -> dict:
    """Colour, marker and line style of a configuration, the same in every figure."""
    if row in CLASSES:
        return {"color": style.LEVEL_COLOURS[CLASSES[row]], "marker": style.LEVEL_MARKERS[CLASSES[row]],
                "linestyle": "-"}
    if row in M.MASS:
        # the colour of its label set, a plus for the added regression
        return {"color": style.LEVEL_COLOURS[CLASSES[row[:-len("_MASS")]]], "marker": "P", "linestyle": "--"}
    if row in M.RAND:
        return {"color": style.RAND_COLOUR, "marker": "o", "linestyle": "none"}
    if row in M.FLAV:
        return {"color": "black", "marker": "D", "linestyle": "none"}
    if row == M.SCRATCH:
        return dict(style.SCRATCH_STYLE)
    return dict(style.INIT_STYLE)


def _log_axis(axis, lo: float, hi: float) -> None:
    """A logarithmic axis with 1-2-5 ticks over [lo, hi], tick labels as plain numbers."""
    loc = ticker.LogLocator(subs=(1.0, 2.0, 5.0)) if hi / lo > 3 else ticker.LogLocator(subs=tuple(range(1, 10)))
    axis.set_major_locator(loc)
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def _spread(v: list) -> tuple[float, float | None, float | None]:
    """(mean, lower and upper error): the SD for three or more runs, the range for two."""
    m = float(np.mean(v))
    if len(v) > 2:
        s = float(np.std(v, ddof=1))
        return m, s, s
    if len(v) == 2:
        return m, m - min(v), max(v) - m
    return m, None, None


# ------------------------------------------------------------------ probes against the training size
def series(M, fits, runs, ds: str, metric: str, rows) -> dict:
    """{row: [(jets, mean, low err, high err, runs)]} of the probes' 1 - AUC."""
    sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
    out = {}
    for row in rows:
        pts = []
        for n in sizes:
            v = [float(x) for x in (M._oma(M._probe(fits, m, ds, n, metric)) for m in runs[row]) if x is not None]
            if v:
                pts.append((int(n), *_spread(v), v))
        out[row] = pts
    return out


def _lines(ax, M, rows: dict, scale: float = 1.0) -> None:
    for k, (row, pts) in enumerate(rows.items()):
        if not pts:
            continue
        dodge = 10 ** (0.03 * (k - (len(rows) - 1) / 2))
        look = _look(M, row)
        x = [p[0] * dodge for p in pts]
        y = [scale * p[1] for p in pts]
        err = [[scale * (p[2] or 0) for p in pts], [scale * (p[3] or 0) for p in pts]]
        ax.errorbar(x, y, yerr=err, label=ROW_NAME[row], markersize=10, linewidth=style.LINEWIDTH,
                    elinewidth=style.LINEWIDTH - 1, capsize=0, **look)


def figure_sizes(root, outdir) -> list:
    M = _mlst()
    fits = M._load(DATA / "linprobe/fits_v3")
    if not fits:
        return []
    runs = M._paper(root)
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIZE_PANELS), figsize=(style.PANEL * len(SIZE_PANELS), style.PANEL))
    for ax, (ds, met, title) in zip(axes, SIZE_PANELS):
        rows = series(M, fits, runs, ds, met, tuple(CLASSES) + (M.INIT,))
        _lines(ax, M, rows)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ys = [p[1] for pts in rows.values() for p in pts]
        _log_axis(ax.yaxis, min(ys), max(ys))
        ax.set_xlabel("Training jets")
        ax.set_title(title, loc="left", fontsize="medium")
    axes[0].set_ylabel(r"$1-\mathrm{AUC}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    return style.save(fig, "probes_vs_training_size", outdir)


# ------------------------------------------------------------------ configurations
def _groups(M, with_init: bool) -> tuple:
    return (tuple(CLASSES), M.RAND, M.FLAV) + (((M.INIT,),) if with_init else ())


def _positions(groups) -> dict:
    """Row -> y, top to bottom, with a gap between the groups."""
    pos, y = {}, 0.0
    for g in groups:
        for r in g:
            pos[r] = y
            y += 1
        y += 0.6
    return pos


def probe_rows(M, fits, runs, ds: str, metric: str) -> dict:
    """{row: [(1 - AUC) x 10^3 of each run]} on the task's largest training set (Table tab:probes);
    every row empty while the task has no fits yet."""
    if not any(ck.get(ds) for f in fits.values() for ck in f["cells"].values()):
        return {row: [] for g in _groups(M, True) for row in g}
    n = M._largest(fits, ds)
    return {row: [1e3 * float(v) for v in (M._oma(M._probe(fits, m, ds, n, metric)) for m in runs[row])
                  if v is not None] for g in _groups(M, True) for row in g}


def pair_rows(M, fits, runs, key: str) -> dict:
    """{row: [(1 - AUC) x 10^3 of each run]} of a binary probe (Table tab:pairs)."""
    return {row: [1e3 * (1 - c["auc"]) for c in
                  (fits.get(m, {}).get("cells", {}).get(M.PRIMARY, {}).get(M.PAIR_DS, {})
                   .get(M.PAIR_N, {}).get(key) for m in runs[row]) if c is not None]
            for g in _groups(M, False) for row in g}


def _dots(ax, M, pos: dict, rows: dict, fill=lambda row: "full") -> None:
    """One configuration per row: mean with its error bar, each run as a small dot just below,
    a dashed line at the mean of each level of the hierarchy. Logarithmic axis."""
    span = [x for row, v in rows.items() if row != M.INIT for x in v]
    lo, hi = min(span), max(span)
    for row, v in rows.items():
        if not v:
            continue
        look = _look(M, row)
        mean, el, eh = _spread(v)
        if row == M.INIT and mean > hi * 1.5:
            # off the scale of the pretrained models: an arrow at the edge with its value
            t = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.plot(0.97, pos[row], marker=">", markersize=11, color=look["color"], transform=t, clip_on=False)
            ax.text(0.92, pos[row], f"{mean:.3g}", transform=t, ha="right", va="center", color=look["color"],
                    fontsize="small")
            continue
        lo, hi = min(lo, mean - (el or 0)), max(hi, mean + (eh or 0))
        if len(v) > 1:
            ax.plot(v, [pos[row] + 0.28] * len(v), linestyle="none", marker="o", markersize=4,
                    color=look["color"], alpha=0.7)
        ax.errorbar(mean, pos[row], xerr=None if el is None else [[el], [eh]], marker=look["marker"],
                    markersize=12, color=look["color"], linestyle="none", fillstyle=fill(row), markeredgewidth=2.5,
                    elinewidth=style.LINEWIDTH - 1, capsize=0,
                    markerfacecolor="white" if fill(row) == "none" else look["color"])
    for row in CLASSES:
        if rows.get(row):
            ax.axvline(float(np.mean(rows[row])), color=_look(M, row)["color"], linestyle="--",
                       linewidth=1.5, alpha=0.6, zorder=0)
    ax.set_xscale("log")
    pad = max(hi / lo, 1.5) ** 0.08
    ax.set_xlim(lo / pad, hi * pad)
    _log_axis(ax.xaxis, lo / pad, hi * pad)
    ax.set_ylim(max(pos.values()) + 0.8, -0.8)
    ax.set_yticks(list(pos.values()))
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def _dot_figure(M, panels, rows_of, pos, ncols, width, fill=lambda key, row: "full"):
    """Panels of `width` x PANEL in, `ncols` to a row, the row names on the left column."""
    import matplotlib.pyplot as plt
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols + 2.5, 0.95 * style.PANEL * nrows),
                             sharey=True, squeeze=False)
    for ax, (key, title) in zip(axes.flat, panels):
        _dots(ax, M, pos, rows_of(key), fill=lambda row, key=key: fill(key, row))
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$(1-\mathrm{AUC})\times 10^{3}$")
    for ax in axes[:, 0]:
        ax.set_yticklabels([ROW_NAME[r] for r in pos])
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def figure_label_sets(root, outdir) -> list:
    """Table tab:probes as a figure: every configuration on the largest training set of each task."""
    M = _mlst()
    fits = M._load(DATA / "linprobe/fits_v3")
    if not fits:
        return []
    runs = M._paper(root)
    style.use_style()
    panels = [((ds, met), title) for ds, met, title in PROBE_PANELS]
    fig = _dot_figure(M, panels, lambda k: probe_rows(M, fits, runs, *k), _positions(_groups(M, True)),
                      ncols=3, width=0.62 * style.PANEL)
    return style.save(fig, "probes_vs_label_set", outdir)


def figure_pairs(root, outdir) -> list:
    """Table tab:pairs as a figure; open markers where the label set merges the pair."""
    M = _mlst()
    fits = M._load(DATA / "linprobe/fits_v3")
    if not fits:
        return []
    runs = M._paper(root)
    status = M.merge_status(root)
    style.use_style()
    fig = _dot_figure(M, [(k, h) for k, h, _, _ in M.PAIRS], lambda k: pair_rows(M, fits, runs, k),
                      _positions(_groups(M, False)), ncols=3, width=0.62 * style.PANEL,
                      fill=lambda k, row: FILL[status.get(row, {}).get(k, "apart")])
    return style.save(fig, "pairs", outdir)


# ------------------------------------------------------------------ per class
def heat_matrix(M, fits, runs, root) -> tuple[list, list, np.ndarray]:
    """(class names in row order, group label per row, (rows, columns) array of the mean over
    runs of (1 - AUC of the class) / (1 - AUC of the same class, 188-class run of the same seed))
    at the largest JetClass-II training set. Rows: the 17-class topology groups, and inside a
    group the flavor orbits together (flavour_pair_map.v2.csv)."""
    maps = root / "configs/labelmaps"
    rung = {int(r["jet_label"]): r for r in csv.DictReader((maps / "rung_label_maps.v1.csv").open())}
    orbit = {int(r["jet_label"]): r["orbit"] for r in csv.DictReader((maps / "flavour_pair_map.v2.csv").open())}
    order = sorted(rung, key=lambda c: (int(rung[c]["R16_Q1"]), orbit.get(c, rung[c]["class_name"]), c))
    n = M.PAIR_N

    def oma(m):
        ck = "init" if m.startswith("init-") else M.PRIMARY
        c = fits.get(m, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(n)
        return None if c is None else {int(k): 1 - v for k, v in c["per_class_auc"].items()}
    full = {m.rsplit("-s", 1)[1]: oma(m) for m in runs["L188"]}
    cols = []
    for key, _ in HEAT_COLUMNS:
        models = [m for a in M.RAND for m in runs[a]] if key == "RAND" else runs[key]
        per = [(oma(m), full.get(m.rsplit("-s", 1)[1])) for m in models]
        per = [(a, b) for a, b in per if a is not None and b is not None]
        cols.append([float(np.mean([a[c] / b[c] for a, b in per])) if per else np.nan for c in order])
    return ([rung[c]["class_name"].removeprefix("label_") for c in order],
            [rung[c]["R16_Q1_name"] for c in order], np.array(cols).T)


def figure_per_class(root, outdir) -> list:
    M = _mlst()
    fits = M._load(DATA / "linprobe/fits_v3")
    if not fits:
        return []
    runs = M._paper(root)
    names, groups, R = heat_matrix(M, fits, runs, root)
    if np.isnan(R).all():
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(0.75 * style.PANEL, 1.9 * style.PANEL))
    L = np.log2(R)
    lim = float(np.nanmax(np.abs(L)))
    im = ax.imshow(L, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), interpolation="none")
    ax.set_xticks(range(len(HEAT_COLUMNS)))
    ax.set_xticklabels([h for _, h in HEAT_COLUMNS], fontsize="x-small")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # one tick per topology group, at its middle, and a line between groups
    starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
    ends = starts[1:] + [len(groups)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
    ax.set_yticklabels([groups[a].replace("_", " ") for a in starts], fontsize="xx-small")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(length=0)
    for a in starts[1:]:
        ax.axhline(a - 0.5, color="black", linewidth=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    ticks = [t for t in (-2, -1, 0, 1, 2, 3) if abs(t) <= lim + 1e-9]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{2.0 ** t:g}" for t in ticks])
    cb.set_label(r"$(1-\mathrm{AUC})$ over the 188-class model's")
    fig.tight_layout()
    return style.save(fig, "per_class_jc2", outdir)


# ------------------------------------------------------------------ fine-tuning
def figure_finetune(root, outdir) -> list:
    M = _mlst()
    tag, per = M.load_ft(DATA)
    if tag is None:
        return []
    runs = M._paper(root)
    style.use_style()
    import matplotlib.pyplot as plt
    tasks = [t for t in M.FT_TASKS if t[0] in ("jc2", "jc1", "top", "qg")]
    fig, axes = plt.subplots(2, len(tasks), figsize=(style.PANEL * len(tasks), 1.35 * style.PANEL),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for j, (key, _, _, met, title) in enumerate(tasks):
        ax, rx = axes[0, j], axes[1, j]
        sizes = M._sizes(per, key)
        ref = per[key].get(M.SCRATCH, {})
        sc = {n: float(np.mean([1 - r[met] for r in ref[n]])) for n in sizes if n in ref}
        rows = {}
        for row in tuple(CLASSES) + M.MASS:
            pts = []
            for n in sizes:
                v = [float(x) for x in (M._oma(M._ft(per, key, m, n, met)) for m in runs[row]) if x is not None]
                if v:
                    pts.append((n, *_spread(v), v))
            rows[row] = pts
        rows[M.SCRATCH] = [(n, *_spread([1 - r[met] for r in ref[n]]), None) for n in sizes if n in ref]
        _lines(ax, M, rows)
        for k, (row, pts) in enumerate(rows.items()):
            if row == M.SCRATCH or not pts:
                continue
            dodge = 10 ** (0.03 * (k - (len(rows) - 1) / 2))
            look = _look(M, row)
            rx.plot([p[0] * dodge for p in pts if p[0] in sc], [p[1] / sc[p[0]] for p in pts if p[0] in sc],
                    markersize=8, linewidth=style.LINEWIDTH - 1, **look)
        rx.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
        for a in (ax, rx):
            a.set_xscale("log")
        ax.set_yscale("log")
        ys = [p[1] for pts in rows.values() for p in pts]
        _log_axis(ax.yaxis, min(ys), max(ys))
        rx.set_yscale("log")
        rs = [p[1] / sc[p[0]] for row, pts in rows.items() if row != M.SCRATCH for p in pts if p[0] in sc]
        if rs:
            rx.set_ylim(min(rs) / 1.15, max(max(rs), 1.0) * 1.15)
            _log_axis(rx.yaxis, min(rs) / 1.15, max(max(rs), 1.0) * 1.15)
        rx.set_xlabel("Training jets")
        ax.set_title(title, loc="left", fontsize="medium")
    axes[0, 0].set_ylabel(r"$1-\mathrm{AUC}$")
    axes[1, 0].set_ylabel("Ratio to\nfrom scratch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    return style.save(fig, "finetune_vs_training_size", outdir)


# ------------------------------------------------------------------ mass
def figure_mass(root, outdir) -> list:
    """sigma_eff and the median m_reg / m_gen in bins of jet pT and of the generated mass: the
    uncorrected soft-drop mass and the mass node of the models pretrained with the regression."""
    M = _mlst()
    head = M._load(DATA / "head")
    mass = {k.removesuffix("_mass"): v for k, v in head.items() if k.endswith("_mass")}
    runs = M._paper(root)
    have = [m for a in M.MASS for m in runs[a] if m in mass and mass[m].get("heads")]
    if not have:
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(1.6 * style.PANEL, 1.4 * style.PANEL), sharex="col")
    any_m = have[0]
    for j, (var, xlabel) in enumerate((("jet_pt", r"$p_{\mathrm T}$ [GeV]"),
                                       ("genjet_sdmass", r"$m_{\mathrm{SD}}^{\mathrm{gen}}$ [GeV]"))):
        bins = list(mass[any_m]["uncorrected"]["binned"][var])
        edges = [tuple(float(x) for x in b.split("-")) for b in bins]
        mid = [np.sqrt(a * b) for a, b in edges]
        for i, key in enumerate(("sigma_eff", "median_response")):
            ax = axes[i, j]
            unc = [mass[any_m]["uncorrected"]["binned"][var][b].get(key) for b in bins]
            ax.plot(mid, unc, label="Uncorrected", **dict(style.INIT_STYLE, linestyle="--"))
            for arm in M.MASS:
                vals = [[mass[m]["heads"]["best70"]["binned"][var][b].get(key) for m in runs[arm]
                         if m in mass and mass[m].get("heads")] for b in bins]
                if not all(vals):
                    continue
                ax.plot(mid, [np.mean(v) for v in vals], label=f"Mass node, {ROW_NAME[arm]}",
                        markersize=9, linewidth=style.LINEWIDTH, **_look(M, arm))
            ax.set_xscale("log")
            if key == "median_response":
                ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
            if i == 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].set_ylabel(r"$\sigma_{\mathrm{eff}}$")
    axes[1, 0].set_ylabel(r"$m_{\mathrm{reg}}/m_{\mathrm{SD}}^{\mathrm{gen}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    return style.save(fig, "mass_response", outdir)


def figure_sculpting(root, outdir) -> list:
    """The soft-drop mass of QCD test jets, all and passing a cut on 1 - p_QCD at 1% QCD
    efficiency, for the hierarchy's runs of seed 1 and the mass-regression run of seed 1."""
    M = _mlst()
    head = M._load(DATA / "head")
    sculpt = {k.removesuffix("_sculpting"): v for k, v in head.items() if k.endswith("_sculpting")}
    rows = [(a, f"mtx-{a.lower().replace('_', '')}-s1") for a in ("L188", "R16_Q1", "R16_Q1_MASS")]
    rows = [(a, m) for a, m in rows if m in sculpt]
    if not rows:
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    d0 = sculpt[rows[0][1]]
    edges = np.asarray(d0["bins_jet_sdmass"])
    fig, ax = plt.subplots(figsize=(style.PANEL, 0.8 * style.PANEL))
    ax.stairs(d0["hist_before"], edges, label="All QCD jets", color="grey", linewidth=2.5, fill=True, alpha=0.3)
    for arm, m in rows:
        look = _look(M, arm)
        ax.stairs(sculpt[m]["working_points"]["0.01"]["hist_after"], edges, label=ROW_NAME[arm],
                  color=look["color"], linewidth=2.5, linestyle="--" if arm in M.MASS else "-")
    ax.set_xlabel(r"$m_{\mathrm{SD}}$ [GeV]")
    ax.set_ylabel("Fraction of jets")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return style.save(fig, "sculpting", outdir)


# ------------------------------------------------------------------ schematic
def schematic_groups(root: pathlib.Path) -> dict:
    """{heading: [class of each schematic row]} read from the label maps."""
    maps = root / "configs/labelmaps"
    out = {}
    for f, col, head in SCHEMATIC_COLUMNS:
        by = {r["class_name"]: r for r in csv.DictReader((maps / f).open())}
        head = head or f"{len({r[col] for r in by.values()})} classes"
        out[head] = [by[f"label_{c}"][col] for c, _ in SCHEMATIC_CLASSES]
    return out


def figure_schematic(root: pathlib.Path, outdir: pathlib.Path) -> list:
    """The class of each selected decay in every label set: a shared tint is a shared class,
    and adjacent rows of one class are one box. The first column is Sophon's: every decay is
    its own class, and the QCD jets are split by their quark content."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    groups = schematic_groups(root)
    n_qcd = sum(1 for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())
                if r["class_name"].startswith("label_QCD"))
    style.use_style()
    rows = [name for _, name in SCHEMATIC_CLASSES]
    fig, ax = plt.subplots(figsize=(1.3 * style.PANEL + 1.0, 1.25 * style.PANEL))
    gap = 0.5  # between the hierarchy and the controls
    for j, (head, g) in enumerate(groups.items()):
        x = j + (gap if j >= 4 else 0)
        own = j < 2  # Sophon's and the 162-class label set: every resonant decay is its own class
        colour = {}
        for k in g:
            colour.setdefault(k, SCHEMATIC_TINTS[len(colour) % len(SCHEMATIC_TINTS)])
        i = 0
        while i < len(g):
            k = i
            while k + 1 < len(g) and g[k + 1] == g[i]:
                k += 1
            face = "white" if own else colour[g[i]]
            ax.add_patch(FancyBboxPatch((x + 0.1, i - 0.4), 0.8, k - i + 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=face, edgecolor="black", linewidth=1.5))
            i = k + 1
        ax.text(x + 0.35, -0.6, head, ha="left", va="bottom", fontsize="small", rotation=45,
                rotation_mode="anchor")
    ax.text(0.5, len(rows) - 1, str(n_qcd), ha="center", va="center", fontsize="small")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_ylim(len(rows) - 0.4, -0.5)
    ax.set_xlim(-0.05, len(groups) + gap + 0.05)
    # brackets at the left: the decays that differ only in quark flavor
    names = [c for c, _ in SCHEMATIC_CLASSES]
    for a, b in SCHEMATIC_ORBITS:
        i, k = names.index(a), names.index(b)
        ax.plot([-0.02, -0.06, -0.06, -0.02], [i - 0.35, i - 0.35, k + 0.35, k + 0.35], color="black",
                linewidth=2, clip_on=False, transform=transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ax.tick_params(axis="y", pad=34, length=0, which="both")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    return style.save(fig, "label_sets", outdir)


FIGURES = (figure_schematic, figure_label_sets, figure_sizes, figure_pairs, figure_per_class,
           figure_finetune, figure_mass, figure_sculpting)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES)
    a = ap.parse_args(argv)
    for f in FIGURES:
        paths = f(REPO, a.outdir)
        print(f"{f.__name__}: " + (", ".join(str(p) for p in paths) if paths else "skipped (no data yet)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
