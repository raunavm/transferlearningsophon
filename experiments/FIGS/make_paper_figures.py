#!/usr/bin/env python3
"""The two headline figures for paper 1, from the committed result artifacts.

FIG 1  label recovery across the contraction tree -- the crossover.
FIG 2  the fine-tuning curves, in-domain (leg 1) beside domain-shifted (leg 2).

FIG 1 READS THE FIVE-SEED MEASUREMENT (decided 2026-09-22). It used to read
`data/label_recovery_v3.json`: ONE 162-class pretraining seed against four
17-class seeds, so only one side had a band. The seed-level analysis of all four
label sets at five seeds each now exists and measures the same quantity with
equal evidence on every side, so FIG 1 reads that and every line carries a band.
The two agree on the headline -- the 162-class model's lead shrinks level by
level and changes sign at the 17-class model's own level -- and disagree on a
secondary shape claim the older file supported (where the gap is deepest below
the crossover). The paper makes no claim the five-seed file does not support.
The older file stays on disk as the record of what was drawn first.

WHY N=1e4 IS DRAWN BUT ANNOTATED IN FIG 2. The pretrained-vs-scratch contrast at
N=1e4 confounds initialisation with learning rate: weaver ran scratch at
start_lr 5e-4 and every pretrained init at 1e-4, per the documented recipe. The
GRANULARITY contrast (162-class vs 17-class) uses 1e-4 on both sides and is unaffected,
which is why the scratch curve is drawn dashed and marked rather than omitted --
hiding it would be worse than labelling it.
"""
from __future__ import annotations

import csv
import json
import pathlib

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style                                                       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures"
LABEL_RECOVERY = HERE / "data/label_recovery_ladder_v1/analysis/s9_label_recovery.json"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
# Rendered tick labels: the number of classes at each contraction level, counted
# from the committed label map. RUNGS are internal keys into the result JSON and
# must never reach a figure.
with (HERE.parents[1] / "configs/labelmaps/rung_label_maps.v1.csv").open() as _f:
    _MAP_ROWS = list(csv.DictReader(_f))
RUNG_NCLASS = [len({row[r] for row in _MAP_ROWS}) for r in RUNGS]
R16_SEEDS = ["r16q1-s2", "r16q1-s3", "r16q1-s4", "r16q1-s5"]
C_L162, C_R16, C_SOPH, C_SCRATCH = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def fig1_label_recovery(path: pathlib.Path):
    """Label recovery at the eight contraction levels, four pretraining label sets,
    five pretraining seeds each, both probes -- read from the seed-level analysis
    (experiments/STATS/seed_level.py), whose paired contrasts are plotted as the
    analysis computed them rather than recomputed here.

    Top row: balanced accuracy of recovering each level's labels from frozen
    features, mean over seeds with a +-1 SD band. Bottom row: the paired
    162-class minus 17-class difference by seed index with its 95% interval;
    filled where Holm rejects within the analysis's own table, open where not."""
    d = json.loads(path.read_text())
    s9 = d["secondary"]["S9"]
    levels = s9["levels_fine_to_coarse"]
    assert s9["rungs_fine_to_coarse"] == RUNGS, s9["rungs_fine_to_coarse"]
    rows = d["table"]
    x = np.arange(len(RUNGS))
    style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(style.FIG_W_TWO_COLUMN, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.1]})
    contrasts = {}
    for col, probe in enumerate(("linear", "mlp")):
        ax, bx = axes[0, col], axes[1, col]
        first = {}
        for lvl in levels:
            acc = np.array([[next(r["accuracy"] for r in rows if r["probe"] == probe
                                  and r["level"] == lvl and r["seed"] == s and r["rung"] == g)
                             for g in RUNGS] for s in s9["seeds_used"]])
            mean, sd = acc.mean(0), acc.std(0, ddof=1)
            c = style.LEVEL_COLOURS[lvl]
            ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.18, lw=0)
            ax.plot(x, mean, ls=style.PROBE_LINESTYLES[probe], color=c,
                    marker=style.LEVEL_MARKERS[lvl], fillstyle=style.PROBE_FILLSTYLES[probe],
                    label=f"pretrained on {style.level_label(lvl)}")
            first[lvl] = mean[0]
        # Direct labels at the finest level, where the curves separate. Curves
        # that coincide there (188 and 162 do) share one label rather than
        # overprinting each other.
        groups = []
        for lvl in sorted(first, key=first.get, reverse=True):
            if groups and abs(first[groups[-1][-1]] - first[lvl]) < 0.02:
                groups[-1].append(lvl)
            else:
                groups.append([lvl])
        for g in groups:
            y = float(np.mean([first[v] for v in g]))
            ax.annotate(" & ".join(str(v) for v in g) + "-class", (0, y),
                        textcoords="offset points", xytext=(-7, 0), ha="right",
                        va="center", fontsize=6.5)
        ax.set_title(style.PROBE_LABELS[probe])
        ax.set_xlim(-1.9, len(RUNGS) - 0.6)
        if col == 0:
            ax.set_ylabel("label-recovery balanced accuracy")
            ax.legend(loc="lower right")

        pair = s9["pairs"]["162_vs_17"][probe]
        cells = {c["rung"]: c for c in pair["rungs"]}
        contrasts[probe] = cells
        m = np.array([cells[g]["mean_diff"] for g in RUNGS])
        lo = m - np.array([cells[g]["ci95"][0] for g in RUNGS])
        hi = np.array([cells[g]["ci95"][1] for g in RUNGS]) - m
        bx.axhline(0, color="k", lw=0.8)
        own = RUNGS.index(s9["own_rung"]["17"])
        bx.axvline(own, ls=":", color="k", lw=0.8)
        for i, g in enumerate(RUNGS):
            bx.errorbar(i, m[i], yerr=[[lo[i]], [hi[i]]], fmt="o", color="#444444",
                        mfc="#444444" if cells[g]["holm_reject"] else "white", capsize=2)
        bx.annotate("17-class model's\nown level", (own, bx.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -10), fontsize=6.5, va="top")
        bx.set_xticks(x, [str(n) for n in RUNG_NCLASS])
        if col == 0:
            bx.set_ylabel("162-class \u2212 17-class")
    fig.suptitle("Label recovery from frozen features: four pretraining label sets, "
                 f"{len(s9['seeds_used'])} pretraining seeds each", fontsize=9)
    fig.supxlabel("classes in the label set the probe must recover (finer \u2192 coarser)",
                  fontsize=8)
    fig.tight_layout()
    style.save(fig, "label_recovery_crossover", OUT)
    return {"contrasts": contrasts,
            "crossover": {p: s9["pairs"]["162_vs_17"][p]["crossover"] for p in ("linear", "mlp")},
            "composite_verdict": s9["composite_verdict"]}


def _curve(summary, init, ns):
    m = [summary[init][n]["accuracy_mean"] for n in ns]
    s = [summary[init][n]["accuracy_sd"] or 0.0 for n in ns]
    return np.array(m), np.array(s)


def fig2_transfer(p1: pathlib.Path, p2: pathlib.Path):
    ns = ["N10000", "N100000", "N1000000"]
    xs = np.array([1e4, 1e5, 1e6])
    legs = [("in-domain\nJetClass-II, 162-way", json.loads(p1.read_text())["summary"]),
            ("domain shift\nJetClass-I, 10-class", json.loads(p2.read_text())["summary"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (title, S) in zip(axes, legs):
        r16 = np.array([[S[s][n]["accuracy_mean"] for n in ns]
                        for s in R16_SEEDS if s in S])
        ax.errorbar(xs, *_curve(S, "l162-s1b", ns), fmt="-o", color=C_L162, lw=2,
                    ms=5, capsize=3, label="162-class (n = 1 pretraining seed)")
        ax.plot(xs, r16.mean(0), "-s", color=C_R16, lw=2, ms=5,
                label=f"17-class (mean of {r16.shape[0]})")
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
                 "162-class vs 17-class is not", fontsize=8.5, y=0.985)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"transfer_curves.{ext}", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g = fig1_label_recovery(LABEL_RECOVERY)
    print("fig 1: 162-class minus 17-class, linear probe, paired by seed:")
    for r in RUNGS:
        c = g["contrasts"]["linear"][r]
        print(f"  {r:8} {c['mean_diff']:+.4f}  p={c['p']:.2g}  {c['advantage']}")
    fig2_transfer(HERE / "data/leg1_metrics.json", HERE / "data/leg2_metrics.json")
    print(f"\nwrote {OUT}/label_recovery_crossover.{{pdf,png}} and "
          f"{OUT}/transfer_curves.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
