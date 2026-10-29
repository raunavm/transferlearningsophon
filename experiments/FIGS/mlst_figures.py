#!/usr/bin/env python3
"""The MLST paper's figure of the linear probes against the training-set size.

Reads the same probe fits as mlst_results.py (experiments/FIGS/data/v2/linprobe/fits3000,
loaded and selected by its own functions, so the figure and Table tab:probesn cannot show
different numbers) and draws 1 - AUC against the number of training jets: JetClass-II
(macro AUC) and top tagging, each label set of the hierarchy and the randomly initialized
network. Points are the mean over the pretraining runs present, error bars their standard
deviation; the series are offset slightly along x so that coinciding points stay visible.
The style and the encoding are mlst_style.py's (mplhep's CMS style; colour and marker =
label-set size).

    python3 experiments/FIGS/mlst_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mlst_style as style  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLASSES = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
NAME = "probes_vs_training_size"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES)
    a = ap.parse_args(argv)
    for p in figure(REPO, a.outdir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
