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
  per_class_ranked         the fifteen classes each merge of the hierarchy hurts most
  roc_pairs, roc_bench(_ft)  rejection against signal efficiency with the ratio to the 188-class
                           model: flavor-sibling pairs; top and quark/gluon (linear probes; full
                           fine-tuning with the network trained from scratch)
  missing_features         balanced JetClass-II with the top or quark/gluon inputs zeroed (filled)
                           or imputed (open), over all features, same run
  finetune_vs_training_size  fine-tuning against the training-set size, with the ratio to the
                           network trained from scratch
  mass_response            mass resolution and scale in bins of pT and generated mass
  mass_response_hist       ln(m / m_gen), resonant and QCD: soft-drop mass and the mass node
  xbx                      eps_S / sqrt(eps_B) for X->bb and X->bs (experiments/XBS/xbx_report.py)
  sculpting                QCD soft-drop mass before and after a cut on the tagger score, and the
                           Jensen-Shannon divergence against the QCD efficiency

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
FT_TITLE = {"jc2": "JetClass-II", "jc1": "JetClass", "top": "Top tagging", "qg": "Quark/gluon tagging"}
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
                ("FLAV_F0", "No flavor\nboundary"), ("FLAV_F1", "$b$-quark\nboundary"),
                ("FLAV_F1R", "Random flavor\nboundary"))
# per_class_ranked's panels: each merge of the hierarchy, (coarser, finer, title)
MERGES = (("L162", "L188", "188 to 162 classes"), ("R42_Q1", "L162", "162 to 43 classes"),
          ("R16_Q1", "R42_Q1", "43 to 17 classes"))
RANKED_N = 15
# decay-product tokens of a class name, longest first, and their typesetting
DECAY_TOKENS = (("tauh", r"\tau_h"), ("taue", r"\tau_e"), ("taum", r"\tau_\mu"), ("b", "b"), ("c", "c"),
                ("s", "s"), ("q", "q"), ("g", "g"), ("e", "e"), ("m", r"\mu"), ("v", r"\nu"))


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mlst():
    spec = importlib.util.spec_from_file_location("mlst_results", REPO / "experiments/FIGS/mlst_results.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fits(M) -> dict:
    """The linear probes: the v4 fits (balanced JetClass-II, the wider regularization grid) once
    pulled, else fits_v3."""
    v4 = M.V4.load_fits(DATA)[0]
    return v4 or M._load(DATA / "linprobe/fits_v3")


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
        return {"color": style.FLAV_COLOUR, "marker": style.FLAV_MARKERS[row.removeprefix("FLAV_")],
                "linestyle": "none"}
    if row == M.SCRATCH:
        return dict(style.SCRATCH_STYLE)
    return dict(style.INIT_STYLE)


def _log_axis(axis, lo: float, hi: float) -> None:
    """A logarithmic axis with 1-2-5 ticks over [lo, hi], tick labels as plain numbers."""
    # a wide span: 1-2-5 ticks; a narrow one: a few round values, so the labels do not run together
    loc = (ticker.LogLocator(subs=(1.0, 2.0, 5.0)) if hi / lo > 3 else
           ticker.FixedLocator([v for v in ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
                                if lo <= v <= hi]))
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
    fits = _fits(M)
    if not fits:
        return []
    runs = M._paper(root)
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIZE_PANELS), figsize=(style.PANEL * len(SIZE_PANELS), style.PANEL))
    for ax, (ds, met, title) in zip(axes, SIZE_PANELS):
        rows = series(M, fits, runs, ds, met, tuple(CLASSES) + (M.INIT,))
        _lines(ax, M, rows, 1e3)          # (1 - AUC) x 10^3, the scale of every table
        ax.set_xscale("log")
        ax.set_yscale("log")
        ys = [1e3 * p[1] for pts in rows.values() for p in pts]
        _log_axis(ax.yaxis, min(ys), max(ys))
        ax.set_xlabel("Training jets")
        ax.set_title(title, loc="left", fontsize="medium")
    axes[0].set_ylabel(r"$(1-\mathrm{AUC})\times10^{3}$")
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
    fits = _fits(M)
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
    fits = _fits(M)
    if not fits:
        return []
    runs = M._paper(root)
    status = M.merge_status(root)
    style.use_style()
    fig = _dot_figure(M, [(k, h) for k, h, _, _ in M.PAIRS], lambda k: pair_rows(M, fits, runs, k),
                      _positions(_groups(M, False)), ncols=3, width=0.62 * style.PANEL,
                      fill=lambda k, row: FILL[status.get(row, {}).get(k, "apart")])
    return style.save(fig, "pairs", outdir)


# ------------------------------------------------------------------ ROC curves
# (curves dataset, panel title); rows: the configurations drawn ("RAND" pools the five partitions)
ROC_PAIRS = (("jc2bal_pairs.bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$"),
             ("jc2bal_pairs.qcd_bb_vs_cc", r"QCD $bb$ vs $cc$"),
             ("jc2bal_pairs.bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$"))
ROC_BENCH = (("top", "Top tagging"), ("qg", "Quark/gluon tagging"), ("qg_herwig", "Quark/gluon, Herwig"))
ROC_ROWS_PAIRS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "FLAV_F0", "FLAV_F1")
ROC_ROWS_BENCH = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "INIT")
ROC_ROWS_FT = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "SCRATCH")
# the full training sets of the fine-tuning curves (experiments/FT/ft_curves.py cells, curves-full)
FT_FULL = {"leg_top": 1_200_000, "leg_qg": 1_600_000}
ROC_MIN_EPS = 0.1
FLAV_DASHES = {"FLAV_F0": (0, (6, 2)), "FLAV_F1": (0, (2, 1.5)), "FLAV_F1R": (0, (6, 2, 2, 2))}


def _probe_curves() -> dict:
    """cells of the probes' ROC file: v4 once pulled, else v3 (experiments/FT/ft_curves.py probes)."""
    for name in ("linprobe_fits_v4", "linprobe_fits_v3"):
        f = DATA / "curves" / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text())["cells"]
    return {}


def _ft_curves() -> dict:
    """The full-set fine-tuning curves in the probes' layout, {model: {checkpoint: {dataset: block}}},
    the from-scratch seeds as scratch-v2-s<k>; empty until pulled."""
    out = {}
    for tree, n in FT_FULL.items():
        for name in (f"full_best70_{tree}", f"full_ref_v100_bench_{tree}"):
            f = DATA / "curves" / f"{name}.json"
            if not f.exists():
                return {}
            for init, by_n in json.loads(f.read_text())["cells"].items():
                for seed, blocks in by_n.get(f"N{n}", {}).items():
                    key = f"{init}-{seed}" if init == "scratch-v2" else f"mtx-{init}"
                    out.setdefault(key, {}).setdefault("best70", {}).update(blocks)
    return out


def roc_rows(M, runs, cells, ds: str, rows) -> dict:
    """{row: (eps_s, [rejection of each run])} of a dataset's ROC points, runs with the curve only."""
    out = {}
    for row in rows:
        models = [m for a in M.RAND for m in runs[a]] if row == "RAND" else \
            list(M.INIT_MODELS) if row == "INIT" else \
            sorted(m for m in cells if m.startswith("scratch-v2-")) if row == "SCRATCH" else runs[row]
        per, eps = [], None
        for m in models:
            r = cells.get(m, {}).get(M._ck(m), {}).get(ds, {}).get("roc")
            if r:
                eps, per = r["eps_s"], per + [r["rejection"]]
        if per:
            out[row] = (eps, per)
    return out


def _roc_figure(M, runs, cells, panels, rows, name, outdir) -> list:
    got = [(ds, title, roc_rows(M, runs, cells, ds, rows)) for ds, title in panels]
    got = [g for g in got if "L188" in g[2]]
    if not got:
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(got), figsize=(0.75 * style.PANEL * len(got) + 0.5, 1.05 * style.PANEL),
                             sharex=True, squeeze=False, gridspec_kw={"height_ratios": (3, 1.2)})
    for j, (ds, title, rr) in enumerate(got):
        top, bot = axes[0, j], axes[1, j]
        eps_ref, ref = rr["L188"]
        ref_mean = np.mean(ref, axis=0)
        for row, (eps, per) in rr.items():
            eps = np.asarray(eps)
            keep = eps >= ROC_MIN_EPS
            per = np.asarray(per)[:, keep]
            mean = per.mean(axis=0)
            look = {"color": style.RAND_COLOUR, "linestyle": "-"} if row == "RAND" else \
                {"color": style.INIT_STYLE["color"], "linestyle": ":"} if row == "INIT" else \
                {"color": style.SCRATCH_STYLE["color"], "linestyle": style.SCRATCH_STYLE["linestyle"]} if row == "SCRATCH" else \
                {"color": style.FLAV_COLOUR, "linestyle": FLAV_DASHES[row]} if row in FLAV_DASHES else _look(M, row)
            label = "Random partitions" if row == "RAND" else "From scratch" if row == "SCRATCH" else ROW_NAME[row]
            ls = look["linestyle"]
            top.plot(eps[keep], mean, color=look["color"], linestyle=ls, linewidth=2, label=label)
            ratio = per / ref_mean[keep]
            bot.plot(eps[keep], ratio.mean(axis=0), color=look["color"], linestyle=ls, linewidth=2)
            if len(per) > 1:
                top.fill_between(eps[keep], per.min(axis=0), per.max(axis=0), color=look["color"], alpha=0.15,
                                 linewidth=0)
                bot.fill_between(eps[keep], ratio.min(axis=0), ratio.max(axis=0), color=look["color"],
                                 alpha=0.15, linewidth=0)
        top.set_yscale("log")
        top.set_title(title, fontsize="small", loc="left")
        bot.axhline(1.0, color="black", linewidth=0.8)
        bot.set_xlabel(r"Signal efficiency $\varepsilon_S$")
    axes[0, 0].set_ylabel(r"$1/\varepsilon_B$")
    axes[1, 0].set_ylabel("Ratio to 188")
    fig.align_ylabels(axes[:, 0])
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=min(4, len(ls)), frameon=False, fontsize="x-small")
    fig.tight_layout(rect=(0, 0.05 * ((len(ls) + 3) // 4) + 0.02, 1, 1))
    return style.save(fig, name, outdir)


def figure_roc_pairs(root, outdir) -> list:
    """Linear probes between flavor siblings on the balanced JetClass-II pairs."""
    M = _mlst()
    return _roc_figure(M, M._paper(root), _probe_curves(), ROC_PAIRS, ROC_ROWS_PAIRS, "roc_pairs", outdir)


def figure_roc_bench_ft(root, outdir) -> list:
    """Full fine-tuning on the full top and quark/gluon training sets, against training from scratch."""
    M = _mlst()
    return _roc_figure(M, M._paper(root), _ft_curves(), ROC_BENCH, ROC_ROWS_FT, "roc_bench_ft", outdir)


def figure_roc_bench(root, outdir) -> list:
    """Linear probes on top and quark/gluon tagging (Pythia and Herwig test sets), largest training set."""
    M = _mlst()
    return _roc_figure(M, M._paper(root), _probe_curves(), ROC_BENCH, ROC_ROWS_BENCH, "roc_bench", outdir)


# ------------------------------------------------------------------ missing features
# (zeroed variant, imputed variant or None, panel title): input_variants.py
MISSING_PANELS = (("mtop", "itop", "Top-tagging inputs (10 of 17 features zero)"),
                  ("mqg", "iqg", "Quark/gluon inputs (4 displacement features zero)"))
IMPUTED_ROWS = ("L188", "R16_Q1")


def missing_rows(M, raw, runs, variant: str, rows) -> dict:
    """{row: [(1 - AUC) with the variant's inputs over (1 - AUC) with all of them, per run]}: the
    macro AUC of the linear probe on balanced JetClass-II at its largest training set (the only
    size the variants are probed at)."""
    out = {}
    for row in rows:
        vals = []
        for m in runs[row]:
            c = raw.get(m, {}).get("cells", {}).get(M._ck(m), {})
            allf, var = c.get("jc2bal", {}), c.get(f"jc2bal.{variant}", {})
            if not var:
                continue
            n = max(var, key=int)
            if n in allf:
                vals.append((1 - var[n]["macro_auc_ovr"]) / (1 - allf[n]["macro_auc_ovr"]))
        out[row] = vals
    return out


def figure_missing_features(root, outdir) -> list:
    M = _mlst()
    raw = M.V4.load_fits(DATA)[1]
    if not any("." in k for d in raw.values() for c in d["cells"].values() for k in c):
        return []
    runs = M._paper(root)
    groups = _groups(M, True)
    pos = _positions(groups)
    rows = [r for g in groups for r in g]
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MISSING_PANELS), figsize=(2 * 0.62 * style.PANEL + 2.5, 0.95 * style.PANEL),
                             sharey=True, squeeze=False)
    for ax, (zero, imp, title) in zip(axes.flat, MISSING_PANELS):
        _dots(ax, M, pos, missing_rows(M, raw, runs, zero, rows))
        for row, v in missing_rows(M, raw, runs, imp, IMPUTED_ROWS).items():
            if v:
                look = _look(M, row)
                mean, el, eh = _spread(v)
                ax.errorbar(mean, pos[row] - 0.3, xerr=None if el is None else [[el], [eh]], marker=look["marker"],
                            markersize=10, color=look["color"], markerfacecolor="white", linestyle="none",
                            markeredgewidth=2, capsize=0)
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$(1-\mathrm{AUC})$ over all features")
    axes[0, 0].set_yticklabels([ROW_NAME[r] for r in pos])
    fig.tight_layout()
    return style.save(fig, "missing_features", outdir)


# ------------------------------------------------------------------ per class
def _class_order(root) -> tuple[dict, list]:
    """(label-map rows by class, classes in row order): the 17-class topology groups, and inside a
    group the flavor orbits together (flavour_pair_map.v2.csv)."""
    maps = root / "configs/labelmaps"
    rung = {int(r["jet_label"]): r for r in csv.DictReader((maps / "rung_label_maps.v1.csv").open())}
    orbit = {int(r["jet_label"]): r["orbit"] for r in csv.DictReader((maps / "flavour_pair_map.v2.csv").open())}
    return rung, sorted(rung, key=lambda c: (int(rung[c]["R16_Q1"]), orbit.get(c, rung[c]["class_name"]), c))


def _one_minus_auc(M, fits, model: str) -> dict | None:
    """{class: 1 - AUC} of a model's linear probe at the largest JetClass-II training set."""
    c = fits.get(model, {}).get("cells", {}).get(M._ck(model), {}).get("jc2_188", {}).get(M.PAIR_N)
    return None if c is None else {int(k): 1 - v for k, v in c["per_class_auc"].items()}


def _seed_ratios(M, fits, runs, num: list, den: str, classes: list) -> np.ndarray:
    """(runs, classes) array of (1 - AUC) of each model in num over that of the same seed's den
    model; rows only for seeds where both exist."""
    base = {m.rsplit("-s", 1)[1]: _one_minus_auc(M, fits, m) for m in runs[den]}
    rows = []
    for m in num:
        a, b = _one_minus_auc(M, fits, m), base.get(m.rsplit("-s", 1)[1])
        if a is not None and b is not None:
            rows.append([a[c] / b[c] for c in classes])
    return np.array(rows).reshape(-1, len(classes))


def heat_matrix(M, fits, runs, root) -> tuple[list, list, np.ndarray]:
    """(class names in row order, group label per row, (rows, columns) array of the mean over
    runs of (1 - AUC of the class) / (1 - AUC of the same class, 188-class run of the same seed))
    at the largest JetClass-II training set, rows in _class_order."""
    rung, order = _class_order(root)
    cols = []
    for key, _ in HEAT_COLUMNS:
        models = [m for a in M.RAND for m in runs[a]] if key == "RAND" else runs[key]
        r = _seed_ratios(M, fits, runs, models, "L188", order)
        cols.append(r.mean(axis=0) if len(r) else np.full(len(order), np.nan))
    return ([rung[c]["class_name"].removeprefix("label_") for c in order],
            [rung[c]["R16_Q1_name"] for c in order], np.array(cols).T)


def class_tex(name: str) -> str:
    """A class name of the label map typeset: X_YY_bbqq -> $X\\to YY\\to bbqq$, QCD_bb -> QCD ($bb$)."""
    name = name.removeprefix("label_")
    head, _, decay = name.partition("_")
    if head == "QCD":
        return "QCD" if not decay else f"QCD ({decay})" if decay == "light" else f"QCD (${decay}$)"
    steps = decay.split("_")
    out = []
    for i, step in enumerate(steps):
        if step == "YY" or i < len(steps) - 1:
            out.append(step)
            continue
        toks, j = [], 0
        while j < len(step):
            t = next((tex for k, tex in DECAY_TOKENS if step.startswith(k, j)), None)
            if t is None:
                raise ValueError(f"unknown decay product in {name!r} at {step[j:]!r}")
            toks.append(t)
            j += next(len(k) for k, tex in DECAY_TOKENS if step.startswith(k, j))
        out.append(" ".join(toks))
    return "$" + r"\to ".join(["X", *out]) + "$"


def ranked_merges(M, fits, runs, root) -> list:
    """Per merge in MERGES: [(class name, mean ratio, ratio of each seed)] of the RANKED_N classes
    whose (1 - AUC) grows most from the finer to the coarser label set, seed by seed."""
    rung, order = _class_order(root)
    out = []
    for coarse, fine, _ in MERGES:
        r = _seed_ratios(M, fits, runs, runs[coarse], fine, order)
        if not len(r):
            out.append([])
            continue
        mean = r.mean(axis=0)
        top = np.argsort(-mean)[:RANKED_N]
        out.append([(rung[order[i]]["class_name"], float(mean[i]), r[:, i].tolist()) for i in top])
    return out


def figure_per_class_ranked(root, outdir) -> list:
    M = _mlst()
    fits = _fits(M)
    if not fits:
        return []
    ranked = ranked_merges(M, fits, M._paper(root), root)
    if not any(ranked):
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MERGES), figsize=(2 * style.PANEL, 0.9 * style.PANEL))
    for ax, rows, (coarse, _, title) in zip(axes, ranked, MERGES):
        ax.set_title(title, fontsize="small")
        if not rows:
            ax.set_axis_off()
            continue
        y = np.arange(len(rows))[::-1]
        look = _look(M, coarse)
        ax.barh(y, [m for _, m, _ in rows], color=look["color"], height=0.7, left=0.0)
        for yi, (_, _, per) in zip(y, rows):
            ax.plot(per, [yi] * len(per), "o", color="black", markersize=2, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([class_tex(n) for n, _, _ in rows], fontsize="xx-small")
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(r"$(1-\mathrm{AUC})$, coarser over finer", fontsize="x-small")
    fig.tight_layout()
    return style.save(fig, "per_class_ranked", outdir)


def figure_per_class(root, outdir) -> list:
    M = _mlst()
    fits = _fits(M)
    if not fits:
        return []
    runs = M._paper(root)
    names, groups, R = heat_matrix(M, fits, runs, root)
    if np.isnan(R).all():
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(0.95 * style.PANEL, 1.9 * style.PANEL))
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
    tag, per, _ = M.load_ft(DATA)
    if tag is None:
        return []
    runs = M._paper(root)
    style.use_style()
    import matplotlib.pyplot as plt
    # a task not yet read out has no sizes and no panel
    tasks = [t for t in M.FT_TASKS if t[0] in ("jc2", "jc1", "top", "qg") and M._sizes(per, t[0])]
    fig, axes = plt.subplots(2, len(tasks), figsize=(style.PANEL * len(tasks), 1.35 * style.PANEL),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col", squeeze=False)
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
        _lines(ax, M, rows, 1e3)
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
        ys = [1e3 * p[1] for pts in rows.values() for p in pts]
        _log_axis(ax.yaxis, min(ys), max(ys))
        rs = [p[1] / sc[p[0]] for row, pts in rows.items() if row != M.SCRATCH for p in pts if p[0] in sc]
        if rs:
            # a linear ratio axis with a few ticks, 1 included
            rx.set_ylim(min(min(rs), 1.0) - 0.05, max(max(rs), 1.0) + 0.05)
            rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        rx.set_xlabel("Training jets")
        ax.set_title(FT_TITLE.get(key, title), loc="left", fontsize="medium")
    axes[0, 0].set_ylabel(r"$(1-\mathrm{AUC})\times10^{3}$")
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


def _curves(kind: str) -> dict:
    """{model: curve file} of head/curves_v2 (scripts/build_jc2bal_jobs.py masscurves)."""
    d = DATA / "head/curves_v2"
    return {p.stem.removesuffix(f"_{kind}"): json.loads(p.read_text())
            for p in sorted(d.glob(f"*_{kind}.json"))} if d.is_dir() else {}


SCULPT_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS")
FLOOR_DRAWS = 200


def jsd_floor(before: list, n_pass: list, seed: int = 0) -> list:
    """The mean JSD (mass_sculpting.jsd) between the QCD mass histogram and that of n random jets
    drawn from it, for each n: what a cut that does not sculpt at all gives at that many jets."""
    ms = _module("experiments/EVAL/mass_sculpting.py", "mass_sculpting")
    p = np.asarray(before, float) / np.sum(before)
    rng = np.random.default_rng(seed)
    return [float(np.mean([ms.jsd(p, rng.multinomial(n, p) / n) for _ in range(FLOOR_DRAWS)])) for n in n_pass]


def sculpting_curves(M, runs, curves) -> dict:
    """{row: (target efficiencies, mean JSD over runs, [JSD of each run])} of the QCD soft-drop
    mass after a cut on 1 - p_QCD, against the QCD efficiency the cut is set to."""
    out = {}
    for row in SCULPT_ROWS:
        have = [curves[m]["curve"] for m in runs[row] if m in curves]
        if have:
            eff = [c["eff"] for c in have[0]]
            per = [[c["jsd"] for c in h] for h in have]
            out[row] = (eff, np.mean(per, axis=0).tolist(), per)
    return out


def figure_sculpting(root, outdir) -> list:
    """QCD soft-drop mass, all and after the cut at 10% and 1% QCD efficiency (seed 1 of 188, 17
    and 17 + mass), and the Jensen-Shannon divergence against the efficiency, every level."""
    M = _mlst()
    curves = _curves("sculpting_v2")
    runs = M._paper(root)
    seed1 = [(a, next((m for m in runs[a] if m.endswith("-s1")), None)) for a in ("L188", "R16_Q1", "R16_Q1_MASS")]
    seed1 = [(a, m) for a, m in seed1 if m in curves]
    if not seed1:
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(2.1 * style.PANEL, 0.75 * style.PANEL))
    d0 = curves[seed1[0][1]]
    edges = np.asarray(d0["bins_jet_sdmass"])
    for ax, wp, title in ((axes[0], "0.1", "10\\% QCD efficiency"), (axes[1], "0.01", "1\\% QCD efficiency")):
        ax.stairs(d0["hist_before"], edges, label="All QCD jets", color="grey", linewidth=2, fill=True, alpha=0.3)
        for arm, m in seed1:
            look = _look(M, arm)
            ax.stairs(curves[m]["working_points"][wp]["hist_after"], edges, label=ROW_NAME[arm],
                      color=look["color"], linewidth=2, linestyle="--" if arm in M.MASS else "-")
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$m_{\mathrm{SD}}$ [GeV]")
    axes[0].set_ylabel("Fraction of jets")
    ax = axes[2]
    pts = d0["curve"]
    ax.plot([c["eff"] for c in pts], jsd_floor(d0["hist_before"], [c["n_pass"] for c in pts]), color="grey",
            linestyle=":", linewidth=2, label="Random subset, same size")
    for row, (eff, mean, per) in sculpting_curves(M, runs, curves).items():
        look = _look(M, row)
        ax.plot(eff, mean, color=look["color"], marker=look["marker"], markersize=5,
                linestyle=look["linestyle"], linewidth=2, label=ROW_NAME[row])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("QCD efficiency")
    ax.set_ylabel("JS divergence [bits]")
    # one legend below the panels: the histograms' entries, then the curves'
    hs, ls = [], []
    for a in (axes[0], axes[2]):
        for h, lab in zip(*a.get_legend_handles_labels()):
            if lab not in ls:
                hs.append(h)
                ls.append(lab)
    fig.legend(hs, ls, loc="lower center", ncol=4, frameon=False, fontsize="x-small")
    fig.tight_layout(rect=(0, 0.06 * (len(ls) + 3) / 4, 1, 1))
    return style.save(fig, "sculpting", outdir)


def figure_mass_response_hist(root, outdir) -> list:
    """ln(m / m_gen) of matched test jets, resonant and QCD: the uncorrected soft-drop mass and the
    mass node, each the mean over the runs of a mass-regression model of its normalized histogram."""
    M = _mlst()
    curves = _curves("response")
    runs = M._paper(root)
    have = {a: [curves[m] for m in runs[a] if m in curves and "node/resonant" in curves[m]["hist"]] for a in M.MASS}
    if not any(have.values()):
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(1.6 * style.PANEL, 0.75 * style.PANEL), sharey=True)
    ref = next(c for v in have.values() for c in v)
    edges = np.asarray(ref["bins"])
    # mass_response_hist.py clips values outside the range into the two edge bins: only the
    # interior is drawn, and each histogram's share in the edge bins is given in its label
    inner = edges[1:-1]

    def norm(h):
        h = np.asarray(h, float)
        return h[1:-1] / h.sum() / np.diff(inner), (h[0] + h[-1]) / h.sum()
    for ax, kind, title in ((axes[0], "resonant", "Resonant jets"), (axes[1], "qcd", "QCD jets")):
        unc = [norm(c["hist"][f"uncorrected/{kind}"]) for v in have.values() for c in v]
        out = 100 * np.mean([o for _, o in unc])
        ax.stairs(np.mean([d for d, _ in unc], axis=0), inner, color="grey", fill=True, alpha=0.3,
                  label=rf"Soft-drop mass ({out:.1f}\% outside)")
        for arm, cs in have.items():
            if cs:
                look = _look(M, arm)
                node = [norm(c["hist"][f"node/{kind}"]) for c in cs]
                out = 100 * np.mean([o for _, o in node])
                ax.stairs(np.mean([d for d, _ in node], axis=0), inner, color=look["color"], linewidth=2,
                          linestyle=look["linestyle"], label=rf"Mass node, {ROW_NAME[arm]} ({out:.1f}\% outside)")
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$\ln(m / m^{\mathrm{gen}})$")
        ax.legend(frameon=False, fontsize="xx-small", loc="upper left", bbox_to_anchor=(0, -0.28))
    axes[0].set_ylabel("Density")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return style.save(fig, "mass_response_hist", outdir)


# ------------------------------------------------------------------ resonance tagging
def figure_xbx(root, outdir) -> list:
    """experiments/XBS/xbx_report.py's figure from the pulled fits (data/v2/xbx/), the code the dry
    run drew on the stand-in."""
    if not (DATA / "xbx").is_dir():
        return []
    XR = _module("experiments/XBS/xbx_report.py", "xbx_report")
    M = _mlst()
    return XR.figure(M, style, M._paper(root), XR.load(DATA / "xbx"), outdir)


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
           figure_per_class_ranked, figure_missing_features, figure_roc_pairs, figure_roc_bench,
           figure_roc_bench_ft,
           figure_finetune, figure_mass, figure_mass_response_hist, figure_sculpting, figure_xbx)


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
