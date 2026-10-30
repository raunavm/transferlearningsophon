#!/usr/bin/env python3
"""Figures and numbers of the group results deck: the four label sets of the class hierarchy,
the auxiliary mass regression, the randomly initialized network and training from scratch (no
random partitions), drawn for 16:9 slides.

Reads the paper's inputs through mlst_results.py and draws with mlst_figures.py's helpers, so
a slide and a paper table cannot disagree. Every value a slide quotes that no paper table or
macro holds is written to <outdir>/deck_numbers.json, formatted as the slide prints it.

Fine-tuning comes from the interim read-out (mlst_results.load_ft: best70 when it exists, else
best70_done18), so its labels and the JSON carry the number of runs behind each value.

    python3 experiments/FIGS/deck_figures.py [--outdir figures/deck]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_tables as MT  # noqa: E402
import mlst_figures as MF  # noqa: E402
import mlst_style as style  # noqa: E402

REPO = MF.REPO
DATA = REPO / "experiments/FIGS/data/v2"
BG = "#FBFBF8"  # the slides' background
DPI = 200
TITLE = {"jc2": "JetClass-II", "jc1": "JetClass", "top": "Top tagging", "qg": "Quark/gluon tagging",
         "qg_herwig": "Quark/gluon, Herwig"}
LEVELS = tuple(MF.CLASSES)


def pm(mean, sd=None) -> str:
    """mean +- sd as the paper rounds it (PDG), in plain text; a single value to 3 figures."""
    s = MT.fmt_pm(mean, sd) if sd is not None else MT.fmt_one(mean)
    return s.replace("{,}", ",").replace("\\,$\\pm$\\,", " ± ")


def _stats(v: list):
    v = [float(x) for x in v if x is not None]
    if not v:
        return None
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return {"mean": float(np.mean(v)), "sd": sd, "n": len(v), "text": pm(np.mean(v), sd)}


def _start():
    style.use_style()
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
    return plt


def _save(fig, name: str, outdir: pathlib.Path) -> pathlib.Path:
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return p


def _lines(ax, rows: dict, look: dict, dodge_step=0.03):
    for k, (row, pts) in enumerate(rows.items()):
        if not pts:
            continue
        dodge = 10 ** (dodge_step * (k - (len(rows) - 1) / 2))
        x, y, e = zip(*pts)
        ax.errorbar([xi * dodge for xi in x], y, yerr=[s or 0.0 for s in e], markersize=11,
                    linewidth=style.LINEWIDTH, elinewidth=style.LINEWIDTH - 1, **look[row])


def _logy(ax):
    from matplotlib import ticker
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0) if hi / lo > 3 else tuple(range(1, 10))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _decades(ax, x):
    """A labelled tick at every power of ten from the smallest to the largest training set."""
    from matplotlib import ticker
    lo, hi = int(np.floor(np.log10(min(x)))), int(np.floor(np.log10(max(x))))
    ax.xaxis.set_major_locator(ticker.FixedLocator([10.0 ** e for e in range(lo, hi + 1)]))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())


def _look(M, row, label):
    if row in (M.INIT, M.SCRATCH):
        return {**style.INIT_STYLE, "label": label}
    c = MF.CLASSES[row]
    return {"color": style.LEVEL_COLOURS[c], "marker": style.LEVEL_MARKERS[c], "linestyle": "-", "label": label}


def _ratio_panel(ax, rows: dict, ref_row, look: dict) -> dict:
    """Each label set over the reference row, the reference's spread as a grey band."""
    ref = {n: (y, s) for n, y, s in rows[ref_row]}
    x = sorted(ref)
    ax.fill_between(x, [1 - (ref[n][1] or 0) / ref[n][0] for n in x], [1 + (ref[n][1] or 0) / ref[n][0] for n in x],
                    color=style.INIT_STYLE["color"], alpha=0.25, linewidth=0)
    ax.axhline(1, color=style.INIT_STYLE["color"], linestyle="--", linewidth=2)
    ratio = {r: [(n, y / ref[n][0], (s / ref[n][0]) if s else None) for n, y, s in pts if n in ref]
             for r, pts in rows.items() if r != ref_row}
    _lines(ax, ratio, look)
    ax.set_xscale("log")
    ax.set_xlim(x[0] / 1.8, x[-1] * 1.8)
    _decades(ax, x)
    ax.margins(y=0.15)
    ax.set_xlabel("Training jets")
    return ratio


def probe_series(M, fits, runs, ds):
    metric = {d: met for d, met, _ in M.TASKS}
    return MF.series(M, fits, runs, ds, metric[ds])


def ft_series(M, per, runs, key):
    """{row: [(jets, mean 1-AUC, sd)]} for the hierarchy and from scratch; and the run counts."""
    met = {k: m for k, _, _, m, _ in M.FT_TASKS}[key]
    sizes = M._sizes(per, key)
    rows, count = {}, {}
    for row in LEVELS:
        pts, ns = [], set()
        for n in sizes:
            v = [float(x) for x in (M._oma(M._ft(per, key, m, n, met)) for m in runs[row]) if x is not None]
            if v:
                pts.append((n, float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None))
                ns.add(len(v))
        rows[row], count[row] = pts, max(ns) if ns else 0
    ref = per[key].get(M.SCRATCH, {})
    rows[M.SCRATCH] = [(n, float(np.mean([1 - r[met] for r in ref[n]])), float(np.std([1 - r[met] for r in ref[n]], ddof=1)))
                       for n in sizes if n in ref and len(ref[n]) > 1]
    count[M.SCRATCH] = max((len(ref[n]) for n in sizes if n in ref), default=0)
    return rows, count


def probe_lines(M, fits, runs, keys, name, outdir, out):
    """Linear-probe 1 - AUC against the training-set size, hierarchy and random initialization."""
    plt = _start()
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.9), squeeze=False)
    out[name] = {}
    for ax, ds in zip(axes[0], keys):
        rows = probe_series(M, fits, runs, ds)
        look = {r: _look(M, r, "Randomly initialized" if r == M.INIT else f"{MF.CLASSES[r]} classes") for r in rows}
        _lines(ax, rows, look)
        _logy(ax)
        _decades(ax, [n for pts in rows.values() for n, _, _ in pts])
        ax.set_xlabel("Training jets")
        ax.set_title(TITLE[ds], loc="left", fontsize="medium")
        out[name][ds] = {r: [{"n": n, "oma": y, "sd": s} for n, y, s in pts] for r, pts in rows.items()}
    axes[0][0].set_ylabel(r"$1-\mathrm{AUC}$")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize="small")
    fig.tight_layout()
    return _save(fig, name, outdir)


def ft_lines(M, per, runs, keys, name, outdir, out):
    """Fine-tuned 1 - AUC against the training-set size, with its ratio to training from scratch."""
    plt = _start()
    fig, axes = plt.subplots(2, len(keys), figsize=(4.0 * len(keys), 6.4), squeeze=False,
                             gridspec_kw={"height_ratios": (2.6, 1)}, sharex="col")
    out[name] = {}
    for j, key in enumerate(keys):
        rows, count = ft_series(M, per, runs, key)
        look = {r: _look(M, r, f"From scratch ({count[r]} seeds)" if r == M.SCRATCH else
                         f"{MF.CLASSES[r]} classes ({count[r]} of {len(runs[r])} runs)") for r in rows}
        _lines(axes[0][j], rows, look)
        _logy(axes[0][j])
        axes[0][j].set_title(TITLE[key], loc="left", fontsize="medium")
        ratio = _ratio_panel(axes[1][j], rows, M.SCRATCH, look)
        out[name][key] = {"runs": count, "values": {r: [{"n": n, "oma": y, "sd": s} for n, y, s in pts] for r, pts in rows.items()},
                          "ratio_to_scratch": {r: [{"n": n, "ratio": y, "text": pm(y)} for n, y, _ in pts]
                                               for r, pts in ratio.items()}}
    axes[0][0].set_ylabel(r"$1-\mathrm{AUC}$")
    axes[1][0].set_ylabel("Ratio to\nfrom scratch", fontsize="small")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize="small")
    fig.tight_layout()
    return _save(fig, name, outdir)


def herwig(M, fits, per, runs, outdir, out):
    """Quark/gluon tagging on the Herwig sample: linear probes over the randomly initialized
    network (left) and fine-tuning over training from scratch (right), against training-set size."""
    plt = _start()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), gridspec_kw={"height_ratios": (2.6, 1)}, sharex="col")
    probe = probe_series(M, fits, runs, "qg_herwig")
    ft, count = ft_series(M, per, runs, "qg_herwig")
    look = {r: _look(M, r, "Reference" if r in (M.INIT, M.SCRATCH) else f"{MF.CLASSES[r]} classes")
            for r in tuple(probe) + (M.SCRATCH,)}
    out["herwig"] = {"ft_runs": count}
    for j, (rows, ref, title, ylab) in enumerate(((probe, M.INIT, "Linear probe", "Ratio to random\ninitialization"),
                                                  (ft, M.SCRATCH, "Fine-tuning", "Ratio to\nfrom scratch"))):
        _lines(axes[0][j], rows, look)
        _logy(axes[0][j])
        axes[0][j].set_title(title, loc="left", fontsize="medium")
        ratio = _ratio_panel(axes[1][j], rows, ref, look)
        axes[1][j].set_ylabel(ylab, fontsize="small")
        out["herwig"]["probe" if j == 0 else "ft"] = {r: [{"n": n, "ratio": y, "text": pm(y)} for n, y, _ in pts]
                                                      for r, pts in ratio.items()}
    axes[0][0].set_ylabel(r"$1-\mathrm{AUC}$")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize="small")
    fig.tight_layout()
    return _save(fig, "herwig", outdir)


def pairs(M, fits, runs, outdir, out):
    """Binary probes between pairs of JetClass-II classes (Table tab:pairs), hierarchy only;
    open markers where the label set merges the pair, half filled where it merges part of it."""
    _start()
    status = M.merge_status(REPO)
    titles = {"bc_vs_bq_cs": r"$X\to bc$ vs $bq$, $cs$", "bb_vs_cc": r"$X\to b\bar b$ vs $c\bar c$",
              "bbqq_vs_ccqq": r"$X\to YY\to bbqq$ vs $ccqq$"}
    fig = MF._dot_figure(M, [(k, titles[k]) for k, _, _, _ in M.PAIRS],
                         lambda k: {r: v for r, v in MF.pair_rows(M, fits, runs, k).items() if r in LEVELS},
                         MF._positions((LEVELS,)), ncols=3, width=0.6 * style.PANEL, title_size="small",
                         fill=lambda k, row: MF.FILL[status.get(row, {}).get(k, "apart")])
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, 0.55 * h)  # four rows instead of the paper figure's twelve
    fig.tight_layout()
    return _save(fig, "pairs", outdir)


def mass_dots(M, runs, outdir, out):
    """sigma_eff of the MLP mass regression from frozen features, hierarchy, mass regression and
    random initialization, with that of a constant prediction (the target's own) as a line."""
    plt = _start()
    mass = M._load(DATA / "linprobe/mass")
    groups = (LEVELS, M.MASS, (M.INIT,))
    rows, target = {}, set()
    for g in groups:
        for row in g:
            v = []
            for m in runs[row]:
                a = mass[m]["arms"]["init" if m.startswith("init-") else M.PRIMARY]
                v.append(a["mlp"]["sigma_eff"])
                target.add(round(a["target"]["sigma_eff"], 12))
            rows[row] = v
    if len(target) != 1:
        raise SystemExit(f"FATAL: the mass target's sigma_eff differs between runs: {sorted(target)}")
    const = target.pop()
    pos = MF._positions(groups)
    fig, ax = plt.subplots(figsize=(0.9 * style.PANEL + 2.5, 0.6 * style.PANEL))
    MF._dots(ax, M, pos, rows)
    ax.axvline(const, color="black", linestyle=":", linewidth=2.5)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, const + 0.006))
    ax.text(const - 0.002, pos["L162_MASS"] + 0.5, "constant\nprediction", ha="right", va="center", fontsize="small")
    ax.set_yticklabels([MF.ROW_NAME[r] for r in pos])
    ax.set_xlabel(r"$\sigma_{\mathrm{eff}}$ of $\ln(m_{\mathrm{SD}}^{\mathrm{gen}}/m_{\mathrm{SD}}^{\mathrm{reco}})$")
    fig.tight_layout()
    out["mass_resolution"] = {"constant_prediction": pm(const), "rows": {row: _stats(v) for row, v in rows.items()}}
    return _save(fig, "mass_resolution", outdir)


def batchnorm(M, fits, runs, out):
    """(1 - AUC) with the BatchNorm statistics recomputed over (1 - AUC) as trained, every run
    of the hierarchy and of the mass regression, largest training set: the range per task."""
    res = {}
    for ds, met, _ in M.TASKS:
        n = M._largest(fits, ds)
        r = []
        for arm in LEVELS + M.MASS:
            for m in runs[arm]:
                a, b = fits[m]["cells"][M.PRIMARY][ds][n], fits[m]["cells"]["best70_bn"][ds][n]
                if not b["converged"]:
                    raise SystemExit(f"FATAL: {m} {ds}: the BatchNorm-recomputed probe did not converge")
                r.append((1 - b[met]) / (1 - a[met]))
        res[ds] = {"n_runs": len(r), "min": f"{min(r):.3f}", "max": f"{max(r):.3f}"}
    out["batchnorm_ratio"] = res


def probe_over_init(M, fits, runs, out):
    """Linear-probe (1 - AUC) of each label set over that of the randomly initialized network of
    the same seed, largest training set, top and quark/gluon tagging (the paper quotes only the
    188- and 17-class rows)."""
    res = {}
    for ds, met, _ in M.TASKS[2:]:
        n = M._largest(fits, ds)
        init = {m.rsplit("-s", 1)[1]: 1 - fits[m]["cells"]["init"][ds][n][met] for m in runs[M.INIT]}
        res[ds] = {}
        for arm in LEVELS + M.MASS:
            r = [(1 - fits[m]["cells"][M.PRIMARY][ds][n][met]) / init[m.rsplit("-s", 1)[1]] for m in runs[arm]]
            res[ds][arm] = {**_stats(r), "per_seed": [round(x, 3) for x in r]}
    out["probe_over_init"] = res


def ft_rejection(M, per, runs, out):
    """R50 and R30 after fine-tuning on the full training set (Table tab:ftrej), as the slide
    prints them: mean +- SD, or a lone run to three figures; and the fewest background jets
    passing behind any value."""
    res, fewest = {}, []
    for key, k in M.FT_REJ:
        n = M._sizes(per, key)[-1]
        col = {}
        for arm in LEVELS + M.MASS:
            v = []
            for m in runs[arm]:
                recs = per[key].get(m[len("mtx-"):], {}).get(n)
                if recs:
                    v.append(recs[0][k])
                    fewest.append(recs[0][f"{k}_n_bkg_pass"])
            col[arm] = _stats(v)
        ref = per[key].get(M.SCRATCH, {}).get(n, [])
        fewest += [r[f"{k}_n_bkg_pass"] for r in ref]
        col[M.SCRATCH] = _stats([r[k] for r in ref])
        res[f"{key}_{k}"] = col
    res["fewest_background_jets"] = min(fewest)
    out["ft_rejection"] = res


def ft_status(M, runs, out):
    """Fine-tuning runs DONE at the last data pull (ft_mlst/manifests.json) for the runs of the
    hierarchy and the mass regression, out of 16 per pretraining run; and their GPU types."""
    man = json.loads((DATA / "ft_mlst/manifests.json").read_text())
    models = [m[len("mtx-"):] for arm in LEVELS + M.MASS for m in runs[arm]]
    cells = [c for k, c in man.items() if k.startswith(f"{M.PRIMARY}/") and c["init"] in models]
    done = collections.Counter(c["init"] for c in cells)
    out["ft_status"] = {"runs": len(models), "done": len(cells), "expected": 16 * len(models),
                        "incomplete": {m: done[m] for m in models if done[m] < 16},
                        "gpus": sorted({c["gpu_device_name"] for c in cells})}


def compute(M, runs, out):
    """GPU-hours per configuration (training and validation passes of every epoch)."""
    comp = M._load(DATA / "compute")
    _, hours = M.table_compute(comp, runs, MT.fmt_int)
    res, total = {}, collections.Counter()

    def row(per, n):
        tot = sum(per.values())
        return {"runs": n, "gpu_hours": f"{tot:,.0f}", "per_run": f"{tot / n:,.0f}",
                "types": ", ".join(f"{g} {100 * h / tot:.0f}%" for g, h in sorted(per.items(), key=lambda x: -x[1]))}
    for arm in LEVELS + M.MASS:
        total.update(hours[arm])
        res[arm] = row(hours[arm], len(runs[arm]))
    res["all"] = row(total, sum(len(runs[a]) for a in LEVELS + M.MASS))
    out["compute"] = res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=REPO / "figures/deck")
    a = ap.parse_args(argv)
    M = MF._mlst()
    fits = M._load(DATA / "linprobe/fits3000")
    runs = M._paper(REPO)
    tag, per = M.load_ft(DATA)
    out = {"fine_tuning_readout": tag}
    paths = [probe_lines(M, fits, runs, ("jc2", "jc1"), "probes_n_jc", a.outdir, out),
             probe_lines(M, fits, runs, ("top", "qg"), "probes_n_bench", a.outdir, out),
             pairs(M, fits, runs, a.outdir, out),
             mass_dots(M, runs, a.outdir, out),
             ft_lines(M, per, runs, ("jc2", "jc1"), "ft_n_jc", a.outdir, out),
             ft_lines(M, per, runs, ("top", "qg"), "ft_n_bench", a.outdir, out),
             herwig(M, fits, per, runs, a.outdir, out)]
    batchnorm(M, fits, runs, out)
    probe_over_init(M, fits, runs, out)
    ft_rejection(M, per, runs, out)
    ft_status(M, runs, out)
    compute(M, runs, out)
    (a.outdir / "deck_numbers.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for p in paths + [a.outdir / "deck_numbers.json"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
