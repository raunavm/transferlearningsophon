#!/usr/bin/env python3
"""The results figures of the journal draft that the ladder figures do not cover.

  finetune_curves       S3 and S4: fine-tuning on JetClass-II and JetClass.
  anomaly_sensitivity   section 5: ln sigma_min per signal, detector family and label set.
  mass_tradeoff         C5 (b vs c with and without the mass output) beside S7
                        (jet-mass resolution from frozen features).
  realdata_top_yield    section 6: fitted top-quark yield in CMS open data.

Same rules as make_ladder_figures.py: every coordinate comes from a committed
analysis file, the encoding lives in style.py (colour = pretraining label-set
size), and nothing is recomputed that the analysis already states. Means over
seeds are drawn beside the per-seed points they are the mean of.

Usage:
    python3 experiments/FIGS/make_results_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
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
DATA = HERE / "data"
INPUTS = {
    "finetune": DATA / "finetune_s3_s4/analysis_v2/s3_s4_finetune.json",
    "anomaly": DATA / "anomaly_merged_v4/analysis_v2/anomaly_s5.json",
    "mass": DATA / "mass_resolution/analysis_holm/s7_mass_resolution.json",
    "mass2x2": sorted((DATA / "probe_ladder_mass2x2").glob("s*.json")),
    "realdata": DATA / "aoj_full_v1/analysis_labelled/aoj_top.json",
}
LEVELS = [188, 162, 43, 17]
REFERENCES = {"scratch": ("random initialisation", "#7f7f7f", "x"),
              "mpm-s1": ("self-supervised, seed 1", "#CC79A7", "v"),
              "rand-d1-s1b": ("random-label control, draw 1", "#D55E00", "P")}
SIGNALS = {"label_X_bb": r"$X\to b\bar b$", "label_X_qq": r"$X\to q\bar q$",
           "label_X_YY_bbb": r"$YY\to bbb$", "label_X_YY_bbbb": r"$YY\to bbbb$",
           "label_X_YY_qqq": r"$YY\to qqq$", "label_X_YY_qqqq": r"$YY\to qqqq$"}
FAMILIES = {"class_sum": "class sum (uses labels)", "mahalanobis": "Mahalanobis",
            "knn": "nearest neighbours", "iad_hgb": "classifier-based"}
GROUPS = ["188", "162", "43", "17", "162+mass", "17+mass"]


def n_of(key: str) -> int:
    return int(key[1:])


def group_style(g: str) -> dict:
    """Colour by label-set size; the mass-output models are the same colour, open."""
    lv = int(g.split("+")[0])
    return {"color": style.LEVEL_COLOURS[lv], "marker": style.LEVEL_MARKERS[lv],
            "fillstyle": "none" if "+mass" in g else "full"}


def save(fig, outdir: pathlib.Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=style.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_finetune(F: dict, outdir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(style.FIG_W_TWO_COLUMN, 5.6), sharex=True)
    for col, (key, title) in enumerate((("S4", "JetClass-II, 162 classes"),
                                        ("S3", "JetClass, 10 classes"))):
        S = F["secondary"][key]
        sizes = list(S["per_size"])
        x = np.array([n_of(n) for n in sizes])
        top, bottom = axes[0, col], axes[1, col]
        for lv in LEVELS:
            y = [next(r["mean"] for r in S["per_size"][n]["levels"] if r["level"] == lv)
                 for n in sizes]
            top.plot(x, y, color=style.LEVEL_COLOURS[lv], marker=style.LEVEL_MARKERS[lv],
                     label=f"{lv} classes")
        for arm, (label, colour, marker) in REFERENCES.items():
            pts = [(n_of(n), np.log1p(-S["reference_rows"][n][arm]["macro_auc"]))
                   for n in sizes if arm in S["reference_rows"][n]]
            if pts:
                top.plot(*zip(*pts), color=colour, marker=marker, linestyle=":", label=label)
        for lv in LEVELS[1:]:
            rows = [next(r for r in S["per_size"][n]["pairwise"]
                         if (r["fine"], r["coarse"]) == (LEVELS[0], lv)) for n in sizes]
            m = np.array([r["mean_diff"] for r in rows])
            ci = np.array([r["ci95"] for r in rows])
            bottom.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], capsize=2,
                            color=style.LEVEL_COLOURS[lv], marker=style.LEVEL_MARKERS[lv],
                            label=f"{lv} minus {LEVELS[0]} classes")
        bottom.axhline(0, color="#999999", linewidth=0.8)
        top.set_title(title)
        top.set_xscale("log")
        margin = np.sqrt(x[1] / x[0])
        top.set_xlim(x.min() / margin, x.max() * margin)
        bottom.set_xlabel("fine-tuning jets")
    axes[0, 0].set_ylabel(r"$\ln(1-$macro AUC$)$")
    axes[1, 0].set_ylabel("paired difference, 95% interval")
    axes[0, 0].legend(fontsize="x-small", loc="upper right")
    axes[1, 0].legend(fontsize="x-small", loc="lower right")
    fig.tight_layout()
    save(fig, outdir, "finetune_curves")


def fig_anomaly(S5: dict, outdir: pathlib.Path) -> None:
    s = S5["section5"]
    inj = s["injection"]
    fams = [f for f in FAMILIES if any(k.startswith(f + "|") for k in s["tests"])]
    sigs = [g for g in SIGNALS if any(k.endswith("|" + g) for k in s["tests"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(style.FIG_W_TWO_COLUMN, 3.0), sharey=True)
    offsets = np.linspace(-0.24, 0.24, len(LEVELS))
    xs = np.arange(len(sigs))
    for ax, fam in zip(axes, fams):
        for j, lv in enumerate(LEVELS):
            pts = [(i + offsets[j], s["level_means_all_injections"][f"{fam}|{g}|{inj}"][str(lv)])
                   for i, g in enumerate(sigs)
                   if s["tests"][f"{fam}|{g}"]["trend"].get("run")]
            ax.plot(*zip(*pts), linestyle="none", color=style.LEVEL_COLOURS[lv],
                    marker=style.LEVEL_MARKERS[lv], markersize=4, label=f"{lv} classes")
        r = s["clause1_per_family"][fam]["n_signals_rejected"]
        ax.set_title(f"{FAMILIES[fam]}\ntrend rejects: {r} of {len(sigs)}", fontsize="x-small")
        ax.set_xticks(xs)
        ax.set_xticklabels([SIGNALS[g] for g in sigs], rotation=60, fontsize="x-small")
    axes[0].set_ylabel(r"$\ln\sigma_{\min}$ (lower is more sensitive)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(LEVELS), fontsize="x-small",
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, outdir, "anomaly_sensitivity")


def mass2x2_points(paths) -> dict:
    """Per-seed b-vs-c log(1-AUC), linear probe, for the four corners of the 2x2."""
    corner = {"l162": "162", "l162mass": "162+mass", "r16q1": "17", "r16q1mass": "17+mass"}
    out = {g: [] for g in corner.values()}
    for p in paths:
        arms = json.loads(pathlib.Path(p).read_text())["tasks"]["bvc_resonant"]["arms"]
        for a, v in arms.items():
            out[corner[a.split("-s")[0]]].append(v["linear"]["log1m_auc"])
    return out


def fig_mass(M: dict, pts2x2: dict, outdir: pathlib.Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.FIG_W_TWO_COLUMN, 3.2))
    corners, gap = ["162", "162+mass", "17", "17+mass"], 0.3
    pos = [i // 2 + (i % 2) * gap for i in range(len(corners))]
    for x, g in zip(pos, corners):
        y = pts2x2[g]
        left.plot([x] * len(y), y, linestyle="none", **group_style(g))
        left.plot([x], [np.mean(y)], marker="_", markersize=18, color="#333333")
    left.set_xticks(pos)
    left.set_xticklabels([g.replace("+mass", "\n+ mass") for g in corners], fontsize="x-small")
    left.set_ylabel(r"$b$ vs $c$ probe, $\ln(1-$AUC$)$")
    left.set_title("classification (frozen linear probe)", fontsize="small")
    for i, g in enumerate(GROUPS):
        y = [r["sigma_eff"] for r in M["table"] if r["cell"] == g and r["probe"] == "ridge"]
        right.plot([i] * len(y), y, linestyle="none", **group_style(g))
        right.plot([i], [np.mean(y)], marker="_", markersize=18, color="#333333")
    right.set_xticks(np.arange(len(GROUPS)))
    right.set_xticklabels([g.replace("+mass", "\n+ mass") for g in GROUPS], fontsize="x-small")
    right.set_ylabel(r"$\sigma_{\mathrm{eff}}$ of $\ln(m_{\mathrm{pred}}/m_{\mathrm{true}})$")
    right.set_title("jet-mass regression (frozen ridge probe)", fontsize="small")
    fig.tight_layout()
    save(fig, outdir, "mass_tradeoff")


def fig_realdata(J: dict, outdir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(style.FIG_W_ONE_COLUMN * 1.4, 3.0))
    pub = J["reference_models"]["sophon-public"]
    ax.axhspan(pub["signal_yield"] - pub["signal_yield_err"],
               pub["signal_yield"] + pub["signal_yield_err"], color="#dddddd",
               label="published 188-class checkpoint")
    for i, g in enumerate(GROUPS):
        rows = [r for r in J["table"] if str(r["level"]) == g]
        ax.errorbar([i] * len(rows), [r["signal_yield"] for r in rows],
                    yerr=[r["signal_yield_err"] for r in rows], linestyle="none", capsize=2,
                    **group_style(g))
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels([g.replace("+mass", "\n+ mass") for g in GROUPS], fontsize="x-small")
    ax.set_ylabel("fitted top-quark yield\n(1% data efficiency)")
    ax.set_xlabel("pretraining label set (one point per seed)")
    ax.legend(fontsize="x-small", loc="lower left")
    fig.tight_layout()
    save(fig, outdir, "realdata_top_yield")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES)
    a = ap.parse_args(argv)
    load = lambda p: json.loads(pathlib.Path(p).read_text())           # noqa: E731
    fig_finetune(load(INPUTS["finetune"]), a.outdir)
    fig_anomaly(load(INPUTS["anomaly"]), a.outdir)
    fig_mass(load(INPUTS["mass"]), mass2x2_points(INPUTS["mass2x2"]), a.outdir)
    fig_realdata(load(INPUTS["realdata"]), a.outdir)
    print(f"wrote four figures to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
