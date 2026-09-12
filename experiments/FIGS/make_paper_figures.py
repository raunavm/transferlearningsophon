#!/usr/bin/env python3
"""The two headline figures for paper 1, from the committed result artifacts.

FIG 1  label recovery across the contraction tree -- the crossover.
FIG 2  the fine-tuning curves, in-domain (leg 1) beside domain-shifted (leg 2).

WHAT THE FIGURES ARE ALLOWED TO SAY. L162 has exactly ONE pretraining seed
(`l162-s1b`); `mtx-l162-s2..s5` are still pretraining. So L162 is drawn as a bare
line with NO band anywhere, and every shaded band belongs to R16_Q1, whose three
or four pretraining seeds are the only pretraining-seed scatter measured. Drawing
a band on both would imply a symmetry of evidence that does not exist. The
caption says so; the figure is built so it cannot accidentally stop saying so.

WHY N=1e4 IS DRAWN BUT ANNOTATED IN FIG 2. The pretrained-vs-scratch contrast at
N=1e4 confounds initialisation with learning rate: weaver ran scratch at
start_lr 5e-4 and every pretrained init at 1e-4, per the documented recipe. The
GRANULARITY contrast (L162 vs R16_Q1) uses 1e-4 on both sides and is unaffected,
which is why the scratch curve is drawn dashed and marked rather than omitted --
hiding it would be worse than labelling it.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
R16_SEEDS = ["r16q1-s2", "r16q1-s3", "r16q1-s4", "r16q1-s5"]
C_L162, C_R16, C_SOPH, C_SCRATCH = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def fig1_label_recovery(path: pathlib.Path):
    d = json.loads(path.read_text())
    A = d["arms"]
    seeds = [s for s in R16_SEEDS if s in A]
    l162 = np.array([A["l162-s1b"]["rungs"][r]["linear"] for r in RUNGS])
    r16 = np.array([[A[s]["rungs"][r]["linear"] for r in RUNGS] for s in seeds])
    mean, sd = r16.mean(0), r16.std(0, ddof=1)
    gap = l162 - mean
    x = np.arange(len(RUNGS))

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(x, l162, "-o", color=C_L162, lw=2, ms=5,
            label="L162 pretraining (n = 1 seed, no band)")
    ax.plot(x, mean, "-s", color=C_R16, lw=2, ms=5,
            label=f"R16_Q1 pretraining (mean of {len(seeds)} seeds)")
    ax.fill_between(x, mean - sd, mean + sd, color=C_R16, alpha=0.22, lw=0,
                    label="R16_Q1 pretraining-seed SD")
    for s in range(r16.shape[0]):
        ax.plot(x, r16[s], color=C_R16, alpha=0.30, lw=0.8, zorder=1)
    ax.set_ylabel("label-recovery balanced accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    bx.axhline(0, color="k", lw=1)
    bx.plot(x, gap, "-o", color="#9467bd", lw=2, ms=5)
    flip = next(i for i in range(len(gap)) if gap[i] < 0)
    bx.axvline(flip, ls=":", color="k", lw=1)
    # Headroom FIRST, so the per-point sigma labels and the callout have space
    # that does not collide with the rung tick labels below the axis.
    lo, hi = gap.min(), gap.max()
    pad = 0.34 * (hi - lo)
    bx.set_ylim(lo - pad, hi + pad)
    bx.annotate(f"sign flip at {RUNGS[flip]}\nR16_Q1's own rung",
                xy=(flip, gap[flip]), xytext=(len(RUNGS) - 1.15, hi * 0.72),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    for i, g in enumerate(gap):
        bx.annotate(f"{g/sd[i]:+.1f}$\\sigma$", (i, g), textcoords="offset points",
                    xytext=(0, 8 if g > 0 else -14), ha="center", fontsize=7)
    bx.set_ylabel("L162 − R16_Q1")
    bx.set_xticks(x, RUNGS, rotation=30, ha="right", fontsize=8)
    bx.set_xlabel("contraction rung the probe must recover  (finer → coarser)")
    bx.grid(alpha=0.3)

    fig.suptitle("Which distinctions survive vocabulary compression\n"
                 "frozen linear probe; arms differ ONLY in pretraining label "
                 "vocabulary (I1); σ is R16_Q1 pretraining-seed SD", fontsize=9, y=0.985)
    fig.tight_layout()
    fig.subplots_adjust(top=0.895, hspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"label_recovery_crossover.{ext}", dpi=180)
    plt.close(fig)
    return {RUNGS[i]: (float(gap[i]), float(gap[i] / sd[i])) for i in range(len(RUNGS))}


def _curve(summary, init, ns):
    m = [summary[init][n]["accuracy_mean"] for n in ns]
    s = [summary[init][n]["accuracy_sd"] or 0.0 for n in ns]
    return np.array(m), np.array(s)


def fig2_transfer(p1: pathlib.Path, p2: pathlib.Path):
    ns = ["N10000", "N100000", "N1000000"]
    xs = np.array([1e4, 1e5, 1e6])
    legs = [("leg 1 — in-domain\nJetClass-II, 162-way", json.loads(p1.read_text())["summary"]),
            ("leg 2 — domain shift\nJetClass-I, 10-class", json.loads(p2.read_text())["summary"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (title, S) in zip(axes, legs):
        r16 = np.array([[S[s][n]["accuracy_mean"] for n in ns]
                        for s in R16_SEEDS if s in S])
        ax.errorbar(xs, *_curve(S, "l162-s1b", ns), fmt="-o", color=C_L162, lw=2,
                    ms=5, capsize=3, label="L162 (n = 1 pretraining seed)")
        ax.plot(xs, r16.mean(0), "-s", color=C_R16, lw=2, ms=5,
                label=f"R16_Q1 (mean of {r16.shape[0]})")
        ax.fill_between(xs, r16.min(0), r16.max(0), color=C_R16, alpha=0.22, lw=0)
        ax.errorbar(xs, *_curve(S, "sophon-public", ns), fmt="-^", color=C_SOPH,
                    lw=1.5, ms=5, capsize=3, label="Sophon public (188)")
        ax.errorbar(xs, *_curve(S, "scratch", ns), fmt="--v", color=C_SCRATCH,
                    lw=1.5, ms=5, capsize=3, label="scratch (LR 5e-4, see note)")
        ax.set_xscale("log")
        ax.set_xlabel("fine-tuning jets $N$")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.axvspan(6e3, 1.7e4, color="orange", alpha=0.10, lw=0)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Pretraining vocabulary granularity governs transfer, in-domain "
                 "and under domain shift\n"
                 "shaded orange: at $N=10^4$ the scratch curve alone runs at LR 5e-4 "
                 "vs 1e-4 pretrained — that contrast is confounded; "
                 "L162-vs-R16_Q1 is not", fontsize=8.5, y=0.985)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"transfer_curves.{ext}", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g = fig1_label_recovery(HERE / "data/label_recovery_v3.json")
    print("fig 1 gaps (value, sigma):")
    for r, (v, s) in g.items():
        print(f"  {r:8} {v:+.4f}  {s:+6.1f} sigma")
    fig2_transfer(HERE / "data/leg1_metrics.json", HERE / "data/leg2_metrics.json")
    print(f"\nwrote {OUT}/label_recovery_crossover.{{pdf,png}} and "
          f"{OUT}/transfer_curves.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
