#!/usr/bin/env python3
"""Color-coded scaling curves for the slide deck, in the poster's own style.

Reuses plots/style.py (Okabe-Ito palette, serif fonts, STRATEGY_COLORS) so the
figures match the poster exactly. Adds two curves the poster didn't have:
linear probe (frozen Sophon, blue) and from-scratch/ParT recipe (purple star).

Reads:  results/sweep_results.csv  (frozen / partial_ft / full_ft sweep)
Writes: figures/scaling_auc.{pdf,png}, figures/scaling_acc.{pdf,png}
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import apply_style, save_fig, STRATEGY_COLORS, OKABE_ITO

PARTICLE_TRANSFORMER_AUC = 0.988

# Linear probe (frozen Sophon -> Linear head), mean over 3 seeds, from PVC pull.
LP_SIZES = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]
LP_AUC   = [0.9560, 0.9618, 0.9593, 0.9650, 0.9665, 0.9673, 0.9675, 0.9675, 0.9675]
LP_ACC   = [0.7341, 0.7490, 0.7416, 0.7581, 0.7637, 0.7668, 0.7679, 0.7679, 0.7679]

# From-scratch (Sophon arch, ParT recipe), single point at 10M seed 42.
FS_SIZE, FS_AUC, FS_ACC = 1e7, 0.9842, 0.8396

# Per-strategy display: (label, color, marker) — colors from the poster palette.
DISPLAY = {
    "linear_probe": ("Linear probe (frozen)",  OKABE_ITO["blue"],           "^"),
    "frozen":       ("Frozen + MLP",           STRATEGY_COLORS["frozen"],    "o"),
    "partial_ft":   ("Partial FT",             STRATEGY_COLORS["partial_ft"], "s"),
    "full_ft":      ("Full FT",                STRATEGY_COLORS["full_ft"],   "D"),
    "from_scratch": ("From-scratch (ParT)",    STRATEGY_COLORS["from_scratch"], "*"),
}


def curve(ax, x, y, key, metric):
    label, color, marker = DISPLAY[key]
    ax.plot(x, y, color=color, lw=1.6, marker=marker, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=4)


def make(metric, lp_y, fs_y, ylabel, ylim, out):
    apply_style()
    df = pd.read_csv("results/sweep_results.csv")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # linear probe (own arrays)
    curve(ax, LP_SIZES, lp_y, "linear_probe", metric)

    # frozen / partial / full from the sweep CSV
    for key in ("frozen", "partial_ft", "full_ft"):
        sub = df[df.strategy == key]
        if not len(sub):
            continue
        agg = sub.groupby("train_size")[metric].mean().reset_index().sort_values("train_size")
        curve(ax, agg["train_size"].values.astype(float), agg[metric].values, key, metric)

    # from-scratch single point — big star
    label, color, marker = DISPLAY["from_scratch"]
    ax.plot([FS_SIZE], [fs_y], color=color, marker=marker, markersize=17,
            markeredgecolor="white", markeredgewidth=0.8, linestyle="",
            label=label, zorder=6)

    if metric == "test_auc":
        ax.axhline(PARTICLE_TRANSFORMER_AUC, color="#444", lw=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.text(1.0e8, PARTICLE_TRANSFORMER_AUC + 0.0008,
                r"published ParT (AUC $\approx$ 0.988)",
                ha="right", va="bottom", fontsize=9, color="#444")

    ax.set_xscale("log")
    ax.set_xlim(8e3, 1.5e8)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"Number of fine-tuning jets $N$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["10K", "100K", "1M", "10M", "100M"])
    ax.minorticks_off()
    ax.legend(loc="lower right", ncol=1)
    ax.grid(True, which="major", alpha=1.0)

    save_fig(fig, out)
    print("Saved", out)


if __name__ == "__main__":
    make("test_auc", LP_AUC, FS_AUC,
         r"Macro AUC (10-class JetClass)", (0.93, 0.995),
         "figures/scaling_auc")
    make("test_acc", LP_ACC, FS_ACC,
         "Test accuracy", (0.64, 0.87),
         "figures/scaling_acc")
