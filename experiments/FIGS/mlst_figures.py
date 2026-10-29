#!/usr/bin/env python3
"""The MLST paper's figures of the linear probes: against the training-set size, against the
pretraining configuration, and for the pairs of classes.

Reads the same probe fits as mlst_results.py (experiments/FIGS/data/v2/linprobe/fits3000,
loaded and selected by its own functions, so the figure and Table tab:probesn cannot show
different numbers) and draws 1 - AUC against the number of training jets: JetClass-II
(macro AUC) and top tagging, each label set of the hierarchy and the randomly initialized
network. Points are the mean over the pretraining runs present, error bars their standard
deviation; the series are offset slightly along x so that coinciding points stay visible.
The style and the encoding are mlst_style.py's (mplhep's CMS style; colour and marker =
label-set size).

The configuration figures (probes_vs_label_set, pairs) draw Tables tab:probes and tab:pairs:
one row per pretraining configuration in the tables' order, the mean over its runs with
their standard deviation, a short tick per run, and a dashed line at the mean of each level
of the hierarchy. In the pairs figure the marker is open where the label set merges the pair
and half filled where it merges part of it (the tables' daggers).

The schematic (label_sets) is drawn from the label maps in configs/labelmaps, not by hand: for
a selection of classes, the class each one has in every label set of the hierarchy and in the
controls, a shared colour meaning a shared class.

    python3 experiments/FIGS/mlst_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys

import numpy as np
from matplotlib import ticker, transforms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mlst_style as style  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLASSES = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
NAME = "probes_vs_training_size"
ROW_NAME = {**{k: f"{v} classes" for k, v in CLASSES.items()},
            **{f"RAND2_p{k}": f"Random partition {k}" for k in range(1, 6)},
            "FLAV_F0": "No flavor boundary", "FLAV_F1": r"$b$-quark boundary",
            "FLAV_F1R": "Random flavor boundary", "INIT": "Randomly initialized"}
PROBE_PANELS = (("jc2", "JetClass-II"), ("jc1", "JetClass"), ("top", "Top tagging"),
                ("qg", "Quark/gluon tagging"))
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


def _mlst():
    spec = importlib.util.spec_from_file_location("mlst_results", REPO / "experiments/FIGS/mlst_results.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def series(M, fits, runs, ds: str, metric: str) -> dict:
    """{row: [(jets, mean 1-AUC, sd or None)]} for the label sets of the hierarchy and INIT."""
    sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
    out = {}
    for row in tuple(CLASSES) + (M.INIT,):
        pts = []
        for n in sizes:
            v = [float(x) for x in (M._oma(M._probe(fits, m, ds, n, metric)) for m in runs[row]) if x is not None]
            if v:
                pts.append((int(n), float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None))
        out[row] = pts
    return out


def figure(root: pathlib.Path, outdir: pathlib.Path) -> list[pathlib.Path]:
    M = _mlst()
    fits = M._load(root / "experiments/FIGS/data/v2/linprobe/fits3000")
    if not fits:
        raise SystemExit("FATAL: no linear-probe fits in experiments/FIGS/data/v2/linprobe/fits3000")
    runs = M._paper(root)
    style.use_style()
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    panels = ((M.TASKS[0], "JetClass-II, macro AUC"), (M.TASKS[2], "Top tagging"))
    fig, axes = plt.subplots(1, len(panels), figsize=(style.PANEL * len(panels), style.PANEL))
    for ax, ((ds, metric, _), title) in zip(axes, panels):
        rows = series(M, fits, runs, ds, metric)
        for k, (row, pts) in enumerate(rows.items()):
            if not pts:
                continue
            dodge = 10 ** (0.03 * (k - (len(rows) - 1) / 2))
            kw = (style.INIT_STYLE if row == M.INIT else
                  {"color": style.LEVEL_COLOURS[CLASSES[row]], "marker": style.LEVEL_MARKERS[CLASSES[row]],
                   "linestyle": "-"})
            label = "Randomly initialized" if row == M.INIT else f"{CLASSES[row]} classes"
            x, y, e = zip(*pts)
            ax.errorbar([xi * dodge for xi in x], y, yerr=[s or 0.0 for s in e], label=label,
                        markersize=12, linewidth=style.LINEWIDTH, elinewidth=style.LINEWIDTH - 1,
                        **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        hi = hi * (hi / lo) ** 0.25  # headroom for the panel label
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0) if hi / lo > 3 else
                                                     tuple(range(1, 10))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Training jets")
        ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top")
    axes[0].set_ylabel(r"$1-\mathrm{AUC}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97),
               frameon=False)
    fig.tight_layout()
    return style.save(fig, NAME, outdir)


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


def _look(M, row: str) -> tuple:
    if row in CLASSES:
        return style.LEVEL_COLOURS[CLASSES[row]], style.LEVEL_MARKERS[CLASSES[row]]
    if row in M.RAND:
        return style.PETROFF6[5], "o"
    if row in M.FLAV:
        return "black", "D"
    return style.INIT_STYLE["color"], style.INIT_STYLE["marker"]


def probe_rows(M, fits, runs, ds: str, metric: str) -> dict:
    """{row: [(1 - AUC) x 10^3 of each run]} on the task's largest training set (Table tab:probes)."""
    n = M._largest(fits, ds)
    out = {}
    for g in _groups(M, True):
        for row in g:
            out[row] = [1e3 * float(v) for v in (M._oma(M._probe(fits, m, ds, n, metric)) for m in runs[row])
                        if v is not None]
    return out


def pair_rows(M, fits, runs, key: str) -> dict:
    """{row: [(1 - AUC) x 10^3 of each run]} of a binary probe (Table tab:pairs)."""
    out = {}
    for g in _groups(M, False):
        for row in g:
            out[row] = [1e3 * (1 - c["auc"]) for c in
                        (fits.get(m, {}).get("cells", {}).get(M.PRIMARY, {}).get("jc2_pairs", {})
                         .get(M.PAIR_N, {}).get(key) for m in runs[row]) if c is not None]
    return out


def _dots(ax, M, pos: dict, rows: dict, fill=lambda row: "full") -> None:
    """One configuration per row: mean +- SD, a tick per run, a dashed line per hierarchy level.
    The axis is logarithmic when the values span more than a factor of three."""
    span = [x for row, v in rows.items() if row != M.INIT for x in v]
    lo, hi = min(span), max(span)
    for row, v in rows.items():
        if not v:
            continue
        c, mk = _look(M, row)
        mean = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
        if row == M.INIT and mean > hi * 1.5:
            # off the scale of the pretrained models: an arrow at the edge with its value
            t = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.plot(0.97, pos[row], marker=">", markersize=12, color=c, transform=t, clip_on=False)
            ax.text(0.92, pos[row], f"{mean:.3g}", transform=t, ha="right", va="center", color=c,
                    fontsize="small")
            continue
        lo, hi = min(lo, mean - (sd or 0)), max(hi, mean + (sd or 0))
        if len(v) > 1:
            ax.plot(v, [pos[row]] * len(v), linestyle="none", marker="|", markersize=14,
                    markeredgewidth=2, color=c, alpha=0.6)
        ax.errorbar(mean, pos[row], xerr=sd, marker=mk, markersize=13, color=c, linestyle="none",
                    fillstyle=fill(row), markeredgewidth=2.5, elinewidth=style.LINEWIDTH - 1,
                    markerfacecolor="white" if fill(row) == "none" else c)
    for row in CLASSES:
        if rows.get(row):
            ax.axvline(float(np.mean(rows[row])), color=_look(M, row)[0], linestyle="--",
                       linewidth=1.5, alpha=0.6, zorder=0)
    if hi / lo > 3:
        ax.set_xscale("log")
        pad = (hi / lo) ** 0.08
        ax.set_xlim(lo / pad, hi * pad)
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0) if hi / lo > 3 else
                                                     tuple(range(1, 10))))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_ylim(max(pos.values()) + 0.8, -0.8)
    ax.set_yticks(list(pos.values()))
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def _dot_figure(M, panels, rows_of, pos, ncols, width, title_size="medium", fill=lambda key, row: "full"):
    """Panels of `width` x PANEL in, `ncols` to a row, with the row names on the left column."""
    import matplotlib.pyplot as plt
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols + 2.5, 0.95 * style.PANEL * nrows),
                             sharey=True, squeeze=False)
    for ax, (key, title) in zip(axes.flat, panels):
        _dots(ax, M, pos, rows_of(key), fill=lambda row, key=key: fill(key, row))
        ax.set_title(title, fontsize=title_size, loc="left")
        ax.set_xlabel(r"$(1-\mathrm{AUC})\times 10^{3}$")
    for ax in axes[:, 0]:
        ax.set_yticklabels([ROW_NAME[r] for r in pos])
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def figure_label_sets(root: pathlib.Path, outdir: pathlib.Path) -> list[pathlib.Path]:
    """Table tab:probes as a figure: every configuration on the largest training set of each task."""
    M = _mlst()
    fits = M._load(root / "experiments/FIGS/data/v2/linprobe/fits3000")
    runs = M._paper(root)
    style.use_style()
    metric = {ds: met for ds, met, _ in M.TASKS}
    fig = _dot_figure(M, PROBE_PANELS, lambda ds: probe_rows(M, fits, runs, ds, metric[ds]),
                      _positions(_groups(M, True)), ncols=2, width=0.75 * style.PANEL)
    return style.save(fig, "probes_vs_label_set", outdir)


def figure_pairs(root: pathlib.Path, outdir: pathlib.Path) -> list[pathlib.Path]:
    """Table tab:pairs as a figure; open markers where the label set merges the pair."""
    M = _mlst()
    fits = M._load(root / "experiments/FIGS/data/v2/linprobe/fits3000")
    runs = M._paper(root)
    status = M.merge_status(root)
    style.use_style()
    fig = _dot_figure(M, [(k, h) for k, h, _, _ in M.PAIRS], lambda k: pair_rows(M, fits, runs, k),
                      _positions(_groups(M, False)), ncols=3, width=0.6 * style.PANEL, title_size="small",
                      fill=lambda k, row: FILL[status.get(row, {}).get(k, "apart")])
    return style.save(fig, "pairs", outdir)


def schematic_groups(root: pathlib.Path) -> dict:
    """{heading: [class of each schematic row]} read from the label maps."""
    maps = root / "configs/labelmaps"
    out = {}
    for f, col, head in SCHEMATIC_COLUMNS:
        by = {r["class_name"]: r for r in csv.DictReader((maps / f).open())}
        head = head or f"{len({r[col] for r in by.values()})} classes"
        out[head] = [by[f"label_{c}"][col] for c, _ in SCHEMATIC_CLASSES]
    return out


def figure_schematic(root: pathlib.Path, outdir: pathlib.Path) -> list[pathlib.Path]:
    """The class of each selected decay in every label set: a shared colour is a shared class,
    and adjacent rows of one class are one box. The first column is Sophon's: every decay is
    its own class, and the QCD jets are split by their quark content."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    groups = schematic_groups(root)
    n_qcd = sum(1 for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())
                if r["class_name"].startswith("label_QCD"))
    style.use_style()
    palette = __import__("matplotlib").style.library["petroff10"]["axes.prop_cycle"].by_key()["color"]
    rows = [name for _, name in SCHEMATIC_CLASSES]
    fig, ax = plt.subplots(figsize=(1.3 * style.PANEL + 1.0, 1.25 * style.PANEL))
    gap = 0.5  # between the hierarchy and the controls
    for j, (head, g) in enumerate(groups.items()):
        x = j + (gap if j >= 4 else 0)
        own = j < 2  # Sophon's and the 162-class label set: every resonant decay is its own class
        colour = {}
        for k in g:
            colour.setdefault(k, palette[len(colour) % len(palette)])
        i = 0
        while i < len(g):
            k = i
            while k + 1 < len(g) and g[k + 1] == g[i]:
                k += 1
            face = "white" if own else colour[g[i]]
            ax.add_patch(FancyBboxPatch((x + 0.1, i - 0.4), 0.8, k - i + 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=face, edgecolor="black", linewidth=1.5, alpha=1 if own else 0.55))
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES)
    a = ap.parse_args(argv)
    for p in (figure_schematic(REPO, a.outdir) + figure(REPO, a.outdir) + figure_label_sets(REPO, a.outdir)
              + figure_pairs(REPO, a.outdir)):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
