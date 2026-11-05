#!/usr/bin/env python3
"""Two layouts for FJ, to choose from: one panel, four panels. Both stop the y axis at the
test set's limit, so every curve comes in from the top of the frame, and put the legend in the free lower left."""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

from gallery_common import M, Page, RUNS, mean_over_runs
from gallery_roc import _family, _plot
import style_v3 as S

TOP = 1e5       # 20 of the 1.8M background jets: 9e4
LANDSCAPE = ["1902.09914/fig5.png", "map2026_poster/fig5.png"]
LABEL = "Landscape Fig. 5 (size, grid, axis range), MAP 2026 poster Fig. 5 (curves per panel)"
CAPTION = ("Background rejection against signal efficiency on JetClass after fine-tuning on one million jets. Each "
           "curve takes one of the ten JetClass classes as signal and the other nine as background, and is the mean "
           "over three pretraining runs. Curves begin where 20 background jets remain. {extra}")


def _axes(ax, x_step: float) -> None:
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, TOP)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1 if x_step > 0.1 else 0.05))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="both", linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)


def _curves(ax, docs: dict, level: str, auc_in_legend: bool, legend: bool, fontsize: float, ncol: int) -> float:
    """The ten curves and their legend; returns the macro AUC."""
    aucs = []
    for c in S.JC1_CLASSES:
        s = mean_over_runs(docs, RUNS[level], c)
        aucs.append(s[2])
        name = f"{S.JC1_TEX[c]} ({s[2]:.3f})" if auc_in_legend else S.JC1_TEX[c]
        _plot(ax, s, color=S.JC1_COLOUR[c], linewidth=1.1, label=name)
    if legend:
        S.legend(ax, loc="lower left", title="Signal class (AUC)" if auc_in_legend else "Signal class",
                 title_fontsize=fontsize, fontsize=fontsize, ncol=ncol, handlelength=1.5, handletextpad=0.4,
                 columnspacing=0.8)
    return float(np.mean(aucs))


def _report_overlap(fig, axes) -> None:
    """Prints how many curve points lie under each legend."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        box = leg.get_window_extent(r)
        n = sum(box.count_contains(ax.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])) for ln in ax.lines)
        print(f"{ax.get_title()!r}: {n} curve points under the legend")


def draw_one(path: pathlib.Path) -> None:
    docs = _family("jc1_ft_ovr")
    fig, ax = plt.subplots(figsize=(14 * S.CM, 11.5 * S.CM))
    _axes(ax, 0.1)
    auc = _curves(ax, docs, "L188", True, True, 9, 2)
    S.title(ax, f"Pretrained on 188 classes, fine-tuned on $10^6$ JetClass jets\nmacro AUC {auc:.3f}")
    S.labels(ax, S.X_EPS, S.Y_REJ)
    _report_overlap(fig, [ax])
    S.save(fig, path)


def draw_four(path: pathlib.Path) -> None:
    docs = _family("jc1_ft_ovr")
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(S.FULL, 18 * S.CM),
                             gridspec_kw={"left": 0.1, "right": 0.97, "top": 0.92, "bottom": 0.06, "wspace": 0.08,
                                          "hspace": 0.1})
    for i, (ax, level) in enumerate(zip(axes.flat, S.LEVELS)):
        _axes(ax, 0.2)
        auc = _curves(ax, docs, level, False, i == 0, 8, 2)
        S.title(ax, f"Pretrained on {S.NAME[level]}, macro AUC {auc:.3f}")
    for ax in axes[1]:
        S.labels(ax, xlabel=S.X_EPS)
    for ax in axes[:, 0]:
        S.labels(ax, ylabel=S.Y_REJ)
    fig.suptitle("Fine-tuned on $10^6$ JetClass jets", fontsize=S.TITLE_PT, y=0.975)
    _report_overlap(fig, axes.flat)
    S.save(fig, path)


PAGES = [
    Page(id="FJ-1", title="FJ option 1: one large panel", precedent=LANDSCAPE, precedent_label=LABEL,
         caption=CAPTION.format(extra="Pretraining on 188 classes; the other label sets are in the appendix. The legend "
                                      "gives each class's area under the curve; the title gives their mean."),
         differs="One panel, larger than the landscape figure's printed size. The other three label sets move to the appendix.",
         draw=draw_one),
    Page(id="FJ-2", title="FJ option 2: all four label sets", precedent=LANDSCAPE, precedent_label=LABEL,
         caption=CAPTION.format(extra="One panel per pretraining label set; the title gives the mean area under the curve "
                                      "over the ten classes."),
         differs="All four label sets, at full width and 18 cm tall. One legend, names only: with the class AUCs it "
                 "does not fit under the curves at this panel width, so the class AUCs need a table.",
         draw=draw_four),
]
