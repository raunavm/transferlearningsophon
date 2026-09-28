#!/usr/bin/env python3
"""The results figures of the journal draft that the ladder figures do not cover.

  finetune_curves       S3 and S4: fine-tuning on JetClass-II and JetClass.
  anomaly_sensitivity   section 5: ln sigma_min per signal, detector family and label set.
  mass_tradeoff         C5 (b vs c with and without the mass output) beside S7
                        (jet-mass resolution from frozen features).
  realdata_top_yield    section 6: fitted top-quark yield in CMS open data.

Same rules as make_ladder_figures.py: every coordinate comes from a committed
file, the encoding lives in style.py (colour = pretraining label-set size), and
nothing is recomputed that the analysis already states. The spread shown is the
standard deviation over the five pretraining seeds (ddof=1): a band around a
mean line, or an error bar on a mean marker with the per-seed points beside it.
No figure carries a test result or an interval over seeds; those are in the tables.

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
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style                                                       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
INPUTS = {
    # The metrics files the S3/S4 analysis read (its provenance names them):
    # leg 1 is the JetClass-II 162-class task, leg 2 the JetClass 10-class task.
    # A later metrics file (the rerun from-scratch and self-supervised
    # fine-tunes) is appended to its task's list and named in FT_REFERENCES.
    "finetune": {"JetClass-II, 162 classes": [DATA / "w2b_leg1_metrics_v2.json"],
                 "JetClass, 10 classes": [DATA / "w2b_leg2_metrics_v2.json"]},
    "anomaly": DATA / "anomaly_merged_v4/analysis_v3/anomaly_summary.json",
    "mass": DATA / "mass_resolution/analysis_holm/s7_mass_resolution.json",
    "mass2x2": sorted((DATA / "probe_ladder_mass2x2_mlp2").glob("s*.json")),
    "realdata": DATA / "aoj_full_v1/analysis_v4/aoj_top.json",
}
LEVELS = [188, 162, 43, 17]
ARMS = {"l188": 188, "l162": 162, "r42q1": 43, "r16q1": 17}
ARM_ALIAS = {"l162-s1b": "l162-s1"}      # seed index 1 of the 162-class model is a rerun
FT_SEED = "s1"                           # the pre-specified fine-tuning seed
# References drawn as a dashed line with seed-spread error bars: label, colour,
# marker, the pretrained models averaged.
FT_REFERENCES = {"random-label control": ("#CC79A7", "P", ("rand-d1-s1b", "rand-d2-s2", "rand-d3-s3"))}
SEED_OFFSET = 0.1                        # per-seed points sit this far left of their mean
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


def ft_cells(paths) -> dict:
    """1 - macro AUC (one-vs-rest) at fine-tuning seed 1, keyed (initialisation, jets)."""
    out = {}
    for p in paths:
        for init, per_n in json.loads(pathlib.Path(p).read_text())["cells"].items():
            for n, per_seed in per_n.items():
                if FT_SEED not in per_seed:
                    continue
                if (init, n) in out:
                    raise SystemExit(f"FATAL: {init}/{n} appears in two fine-tuning files")
                out[(init, n)] = 1.0 - per_seed[FT_SEED]["macro_auc_ovr"]
    return out


def ft_by_seed(cells: dict) -> dict:
    """{label set: {training jets: {pretraining seed: 1 - macro AUC}}}."""
    out = {lv: {} for lv in LEVELS}
    for (init, n), v in cells.items():
        arm, _, seed = ARM_ALIAS.get(init, init).partition("-s")
        if arm in ARMS and seed.isdigit():
            out[ARMS[arm]].setdefault(n_of(n), {})[int(seed)] = v
    return out


def fig_finetune(legs: dict, outdir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(style.FIG_W_TWO_COLUMN, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": (2, 1)})
    for col, (title, paths) in enumerate(legs.items()):
        cells = ft_cells(paths)
        by = ft_by_seed(cells)
        top, bottom = axes[0, col], axes[1, col]
        for lv in LEVELS:
            x = sorted(by[lv])
            y = [list(by[lv][n].values()) for n in x]
            m = np.array([np.mean(v) for v in y])
            sd = np.array([np.std(v, ddof=1) for v in y])
            kw = {"color": style.LEVEL_COLOURS[lv], "marker": style.LEVEL_MARKERS[lv]}
            top.plot(x, m, label=f"{lv} classes", **kw)
            top.fill_between(x, m - sd, m + sd, color=kw["color"], alpha=0.2, lw=0)
            if lv == LEVELS[0]:
                continue
            # Ratio to the 188-class model of the same pretraining seed, then
            # mean and spread of the ratio over seeds.
            r = [[by[lv][n][s] / by[LEVELS[0]][n][s] for s in by[lv][n]] for n in x]
            m = np.array([np.mean(v) for v in r])
            sd = np.array([np.std(v, ddof=1) for v in r])
            bottom.plot(x, m, **kw)
            bottom.fill_between(x, m - sd, m + sd, color=kw["color"], alpha=0.2, lw=0)
        for label, (colour, marker, inits) in FT_REFERENCES.items():
            per = {}
            for (i, n), v in cells.items():
                if i in inits:
                    per.setdefault(n_of(n), []).append(v)
            xs = sorted(n for n in per if len(per[n]) == len(inits))
            if xs:
                top.errorbar(xs, [np.mean(per[n]) for n in xs],
                             yerr=[np.std(per[n], ddof=1) for n in xs], color=colour,
                             marker=marker, linestyle="--", capsize=2,
                             label=f"{label} ({len(inits)} draws)")
        bottom.axhline(1, color="#999999", linewidth=0.8)
        top.set_title(title)
        top.set_xscale("log")
        top.set_yscale("log")
        margin = np.sqrt(x[1] / x[0])
        top.set_xlim(x[0] / margin, x[-1] * margin)
        bottom.set_xlabel("fine-tuning jets")
    axes[0, 0].set_ylabel(r"$1-$macro AUC (one-vs-rest)")
    axes[1, 0].set_ylabel(f"ratio to {LEVELS[0]} classes\n(same pretraining seed)")
    axes[0, 0].legend(fontsize="x-small", loc="upper right")
    fig.tight_layout()
    save(fig, outdir, "finetune_curves")


def fig_anomaly(S: dict, outdir: pathlib.Path) -> None:
    """sigma_min per signal and label set for the two feature-based detectors: mean
    +- SD over seeds (each seed the median over resamplings). Signals no detector sees at any label set are left out; the table lists them."""
    inj = S["conventions"]["primary_injection"]
    nd = set(S["not_detected_rule"]["not_detected"])
    fams = [f for f in FAMILIES if f in ("mahalanobis", "knn")]
    sigs = [g for g in SIGNALS if any(f"{f}|{g}" not in nd for f in fams)]
    fig, axes = plt.subplots(1, len(fams), figsize=(style.FIG_W_TWO_COLUMN, 2.8), sharey=True)
    offsets = np.linspace(-0.24, 0.24, len(LEVELS))
    xs = np.arange(len(sigs))
    for ax, fam in zip(axes, fams):
        for j, lv in enumerate(LEVELS):
            per = [np.exp(S["families"][fam][g][inj]["levels"][str(lv)]["ln_sigma_min"]) for g in sigs]
            x = xs + offsets[j]
            ax.errorbar(x, [np.mean(v) for v in per], yerr=[np.std(v, ddof=1) for v in per],
                        linestyle="none", color=style.LEVEL_COLOURS[lv],
                        marker=style.LEVEL_MARKERS[lv], markersize=4, capsize=2,
                        label=f"{lv} classes", zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%g"))
        ax.set_title(FAMILIES[fam], fontsize="small")
        ax.set_xticks(xs)
        ax.set_xticklabels([SIGNALS[g] for g in sigs], fontsize="small")
    axes[0].set_ylabel(r"$\sigma_{\min}$  (lower is more sensitive)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(LEVELS), fontsize="small",
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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


def seeds_and_mean(ax, x: float, y, g: str) -> None:
    """Per-seed points, light, just left of a mean marker with a +-1 SD bar."""
    ax.plot([x - SEED_OFFSET] * len(y), y, linestyle="none", alpha=0.4, markersize=4,
            **group_style(g))
    ax.errorbar([x], [np.mean(y)], yerr=[np.std(y, ddof=1)], capsize=3, **group_style(g))


def fig_mass(M: dict, pts2x2: dict, outdir: pathlib.Path) -> None:
    corners, gap = ["162", "162+mass", "17", "17+mass"], 0.4
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.FIG_W_TWO_COLUMN, 3.2),
                                      gridspec_kw={"width_ratios": (len(corners), len(GROUPS))})
    pos = [i // 2 + (i % 2) * gap for i in range(len(corners))]
    for x, g in zip(pos, corners):
        seeds_and_mean(left, x, np.exp(pts2x2[g]), g)      # stored as ln(1 - AUC)
    left.set_yscale("log")
    # About one decade: label the 2-3-4-6 minor ticks too, or only one tick is labelled.
    left.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
    left.set_xticks(pos)
    left.set_xticklabels([g.replace("+mass", "\n+ mass") for g in corners], fontsize="x-small")
    left.set_ylabel(r"$b$ vs $c$ probe, $1-$AUC")
    left.set_title("classification (frozen linear probe)", fontsize="small")
    for i, g in enumerate(GROUPS):
        y = [r["sigma_eff"] for r in M["table"] if r["cell"] == g and r["probe"] == "mlp"]
        seeds_and_mean(right, i, y, g)
    target = {r["target_sigma_eff"] for r in M["table"]}
    if len(target) != 1:
        raise SystemExit(f"FATAL: the rows disagree on the class-mean-only sigma_eff: {target}")
    target = target.pop()
    right.axhline(target, color="#999999", linestyle="--", linewidth=0.8)
    right.annotate("class mean only", xy=(1, target), xycoords=("axes fraction", "data"),
                   xytext=(-3, -3), textcoords="offset points", ha="right", va="top",
                   fontsize="x-small", color="#777777")
    right.set_xticks(np.arange(len(GROUPS)))
    right.set_xticklabels([g.replace("+mass", "\n+ mass") for g in GROUPS], fontsize="x-small")
    right.set_ylabel(r"$\sigma_{\mathrm{eff}}$ of $\ln(m_{\mathrm{pred}}/m_{\mathrm{true}})$")
    right.set_title("jet-mass regression (frozen nonlinear (MLP) probe)", fontsize="small")
    fig.tight_layout()
    save(fig, outdir, "mass_tradeoff")


def fig_realdata(J: dict, outdir: pathlib.Path) -> None:
    """Fitted top yield per pretrained model with its fit error (light), and the
    mean +- SD over the five seeds of each label set (bold)."""
    fig, ax = plt.subplots(figsize=(style.FIG_W_ONE_COLUMN * 1.4, 3.0))
    pub = J["reference_models"]["sophon-public"]
    ax.axhspan(pub["signal_yield"] - pub["signal_yield_err"],
               pub["signal_yield"] + pub["signal_yield_err"], color="#dddddd",
               label="published 188-class checkpoint")
    for i, g in enumerate(GROUPS):
        rows = [r for r in J["table"] if str(r["level"]) == g]
        y = [r["signal_yield"] for r in rows]
        ax.errorbar([i - SEED_OFFSET] * len(rows), y, yerr=[r["signal_yield_err"] for r in rows],
                    linestyle="none", capsize=0, alpha=0.4, markersize=3, **group_style(g))
        ax.errorbar([i + SEED_OFFSET], [np.mean(y)], yerr=[np.std(y, ddof=1)], linestyle="none",
                    capsize=3, markersize=6, **group_style(g))
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels([g.replace("+mass", "\n+ mass") for g in GROUPS], fontsize="small")
    ax.set_ylabel("fitted top-quark yield\n(1% data efficiency)")
    ax.set_xlabel("pretraining label set")
    ax.legend(fontsize="small", loc="lower right")
    fig.tight_layout()
    save(fig, outdir, "realdata_top_yield")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES)
    a = ap.parse_args(argv)
    load = lambda p: json.loads(pathlib.Path(p).read_text())           # noqa: E731
    style.use_style()
    fig_finetune(INPUTS["finetune"], a.outdir)
    fig_anomaly(load(INPUTS["anomaly"]), a.outdir)
    fig_mass(load(INPUTS["mass"]), mass2x2_points(INPUTS["mass2x2"]), a.outdir)
    fig_realdata(load(INPUTS["realdata"]), a.outdir)
    print(f"wrote four figures to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
