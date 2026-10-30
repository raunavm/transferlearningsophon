#!/usr/bin/env python3
"""Figures and numbers of the group results deck: the four label sets of the class hierarchy
(188, 162, 43 and 17 classes), the auxiliary mass regression, the randomly initialized network
and training from scratch (no random partitions), drawn for 16:9 slides.

Reads the paper's inputs through mlst_results.py and draws with mlst_figures.py's helpers, so
a slide and a paper table cannot disagree: the linear probes from linprobe/fits_v3 (JetClass-II
at its 188 classes), the mass regression from head/<model>_mass.json, the fine-tuning from the
read-out tag mlst_results.load_ft finds (FT_TAGS; JetClass-II from leg188 at 188 classes), and
compute/ and ft_mlst/manifests.json. Points are the mean over the runs present, error bars
mlst_figures' spread (the SD, or the range of two runs); a value quoted as text is the paper
table's (two runs: both values). Every value a slide quotes that no paper table or macro holds
is written to <outdir>/deck_numbers.json, formatted as the slide prints it, with the number of
runs behind it. A figure or number whose data are not pulled yet is skipped, with a line saying so.

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
# probe datasets (jc2_188, ...) and fine-tuning tasks (jc2, ...)
TITLE = {"jc2_188": "JetClass-II", "jc2": "JetClass-II", "jc1": "JetClass", "top": "Top tagging",
         "qg": "Quark/gluon tagging", "qg_herwig": "Quark/gluon, Herwig"}
# the probe metric of each dataset: its first panel in the paper (JetClass-II: all 188 classes)
METRIC = {ds: met for ds, met, _ in MF.PROBE_PANELS[::-1]}
LEVELS = tuple(MF.CLASSES)
# the fine-tuning legs the paper reads (ft_mlst/manifests.json: best70/<leg>/<init>/N<n>/s1), four sizes each
LEGS = ("leg1_188", "leg2", "leg_top", "leg_qg")


def pm(mean, sd=None) -> str:
    """mean +- sd as the paper rounds it (PDG), in plain text; a single value to 3 figures."""
    s = MT.fmt_pm(mean, sd) if sd is not None else MT.fmt_one(mean)
    return s.replace("{,}", ",").replace("\\,$\\pm$\\,", " ± ")


def _stats(M, v: list):
    """Mean, SD and the text a paper table prints for these runs (mlst_results._cell, which
    marks an unconverged value, so it gets the values as they are)."""
    v = [x for x in v if x is not None]
    if not v:
        return None
    f = [float(x) for x in v]
    sd = float(np.std(f, ddof=1)) if len(f) > 1 else None
    text = M._cell(v, len(v), MT.fmt_pm).replace("{,}", ",").replace("\\,$\\pm$\\,", " ± ")
    return {"mean": float(np.mean(f)), "sd": sd, "n": len(f), "text": text}


def _point(p) -> dict:
    """A point (jets, mean, lower error, upper error, values) as deck_numbers.json holds it."""
    return {"n": p[0], "oma": p[1], "err": [p[2], p[3]], "runs": len(p[4])}


def _ready(rows: dict, ref=None) -> bool:
    """Something to draw: a row other than the reference has points, and so has the reference."""
    return any(pts for r, pts in rows.items() if r != ref) and (ref is None or bool(rows.get(ref)))


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
        ax.errorbar([p[0] * dodge for p in pts], [p[1] for p in pts],
                    yerr=[[p[2] or 0.0 for p in pts], [p[3] or 0.0 for p in pts]], markersize=11,
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
    if row == M.SCRATCH:
        return {**style.SCRATCH_STYLE, "label": label}
    if row == M.INIT:
        return {**style.INIT_STYLE, "label": label}
    c = MF.CLASSES[row]
    return {"color": style.LEVEL_COLOURS[c], "marker": style.LEVEL_MARKERS[c], "linestyle": "-", "label": label}


def _ratio_panel(ax, rows: dict, ref_row, look: dict) -> dict:
    """Each label set over the reference row, the reference's spread as a grey band."""
    ref = {p[0]: p for p in rows[ref_row]}
    x = sorted(ref)
    ax.fill_between(x, [1 - (ref[n][2] or 0) / ref[n][1] for n in x], [1 + (ref[n][3] or 0) / ref[n][1] for n in x],
                    color=style.INIT_STYLE["color"], alpha=0.25, linewidth=0)
    ax.axhline(1, color=style.INIT_STYLE["color"], linestyle="--", linewidth=2)
    ratio = {}
    for r, pts in rows.items():
        if r == ref_row:
            continue
        ratio[r] = []
        for n, y, lo, hi, v in pts:
            if n in ref:
                d = ref[n][1]
                ratio[r].append((n, y / d, None if lo is None else lo / d, None if hi is None else hi / d, v))
    _lines(ax, ratio, look)
    ax.set_xscale("log")
    ax.set_xlim(x[0] / 1.8, x[-1] * 1.8)
    _decades(ax, x)
    ax.margins(y=0.15)
    ax.set_xlabel("Training jets")
    return ratio


def probe_series(M, fits, runs, ds):
    """{row: [(jets, mean 1-AUC, lower err, upper err, values)]}: the hierarchy and random initialization."""
    return MF.series(M, fits, runs, ds, METRIC[ds], LEVELS + (M.INIT,))


def ft_series(M, per, runs, key):
    """{row: [(jets, mean 1-AUC, lower err, upper err, values)]} for the hierarchy and from
    scratch; and the most runs behind a point of each row."""
    met = {k: m for k, _, _, m, _ in M.FT_TASKS}[key]
    sizes = M._sizes(per, key)
    rows = {}
    for row in LEVELS:
        pts = []
        for n in sizes:
            v = [float(x) for x in (M._oma(M._ft(per, key, m, n, met)) for m in runs[row]) if x is not None]
            if v:
                pts.append((n, *MF._spread(v), v))
        rows[row] = pts
    ref = per[key].get(M.SCRATCH, {})
    rows[M.SCRATCH] = []
    for n in sizes:
        if n in ref:
            v = [1 - r[met] for r in ref[n]]
            rows[M.SCRATCH].append((n, *MF._spread(v), v))
    count = {r: max((len(p[4]) for p in pts), default=0) for r, pts in rows.items()}
    return rows, count


def probe_lines(M, fits, runs, keys, name, outdir, out):
    """Linear-probe 1 - AUC against the training-set size, hierarchy and random initialization."""
    if not fits:
        return None
    series = {ds: probe_series(M, fits, runs, ds) for ds in keys}
    if not all(_ready(rows) for rows in series.values()):
        return None
    plt = _start()
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.9), squeeze=False)
    out[name] = {}
    for ax, ds in zip(axes[0], keys):
        rows = series[ds]
        look = {r: _look(M, r, "Randomly initialized" if r == M.INIT else f"{MF.CLASSES[r]} classes") for r in rows}
        _lines(ax, rows, look)
        _logy(ax)
        _decades(ax, [p[0] for pts in rows.values() for p in pts])
        ax.set_xlabel("Training jets")
        ax.set_title(TITLE[ds], loc="left", fontsize="medium")
        out[name][ds] = {r: [_point(p) for p in pts] for r, pts in rows.items()}
    axes[0][0].set_ylabel(r"$1-\mathrm{AUC}$")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize="small")
    fig.tight_layout()
    return _save(fig, name, outdir)


def ft_lines(M, per, runs, keys, name, outdir, out):
    """Fine-tuned 1 - AUC against the training-set size, with its ratio to training from scratch."""
    if not per:
        return None
    series = {key: ft_series(M, per, runs, key) for key in keys}
    if not all(_ready(rows, M.SCRATCH) for rows, _ in series.values()):
        return None
    plt = _start()
    fig, axes = plt.subplots(2, len(keys), figsize=(4.0 * len(keys), 6.4), squeeze=False,
                             gridspec_kw={"height_ratios": (2.6, 1)}, sharex="col")
    out[name] = {}
    for j, key in enumerate(keys):
        rows, count = series[key]
        look = {r: _look(M, r, f"From scratch ({count[r]} seeds)" if r == M.SCRATCH else
                         f"{MF.CLASSES[r]} classes ({count[r]} of {len(runs[r])} runs)") for r in rows}
        _lines(axes[0][j], rows, look)
        _logy(axes[0][j])
        axes[0][j].set_title(TITLE[key], loc="left", fontsize="medium")
        ratio = _ratio_panel(axes[1][j], rows, M.SCRATCH, look)
        out[name][key] = {"runs": count, "values": {r: [_point(p) for p in pts] for r, pts in rows.items()},
                          "ratio_to_scratch": {r: [{"n": p[0], "ratio": p[1], "text": pm(p[1])} for p in pts]
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
    if not fits or not per:
        return None
    probe = probe_series(M, fits, runs, "qg_herwig")
    ft, count = ft_series(M, per, runs, "qg_herwig")
    if not (_ready(probe, M.INIT) and _ready(ft, M.SCRATCH)):
        return None
    plt = _start()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), gridspec_kw={"height_ratios": (2.6, 1)}, sharex="col")
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
        out["herwig"]["probe" if j == 0 else "ft"] = {r: [{"n": p[0], "ratio": p[1], "text": pm(p[1])} for p in pts]
                                                      for r, pts in ratio.items()}
    axes[0][0].set_ylabel(r"$1-\mathrm{AUC}$")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize="small")
    fig.tight_layout()
    return _save(fig, "herwig", outdir)


def pairs(M, fits, runs, outdir, out):
    """Binary probes between pairs of JetClass-II classes (Table tab:pairs, titles from
    mlst_results.PAIRS), hierarchy only; open markers where the label set merges the pair, half
    filled where it merges part of it."""
    if not fits:
        return None
    rows = {k: {r: v for r, v in MF.pair_rows(M, fits, runs, k).items() if r in LEVELS} for k, _, _, _ in M.PAIRS}
    if not all(any(r.values()) for r in rows.values()):
        return None
    _start()
    status = M.merge_status(REPO)
    fig = MF._dot_figure(M, [(k, h) for k, h, _, _ in M.PAIRS], lambda k: rows[k],
                         MF._positions((LEVELS,)), ncols=3, width=0.6 * style.PANEL,
                         fill=lambda k, row: MF.FILL[status.get(row, {}).get(k, "apart")])
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, 0.55 * h)  # four rows instead of the paper figure's twelve
    fig.tight_layout()
    return _save(fig, "pairs", outdir)


def mass_dots(M, runs, outdir, out):
    """sigma_eff of the MLP mass regression from frozen features, hierarchy, mass regression and
    random initialization, with that of a constant prediction (the target's own) as a line."""
    head = M._load(DATA / "head")
    mass = {k.removesuffix("_mass"): v for k, v in head.items() if k.endswith("_mass")}
    groups = (LEVELS, M.MASS, (M.INIT,))
    rows, target = {}, set()
    for g in groups:
        for row in g:
            rows[row] = [M._mass(mass, m, "mlp") for m in runs[row] if m in mass]
            target |= {round(M._mass(mass, m, "target"), 12) for m in runs[row] if m in mass}
    if not any(v for r, v in rows.items() if r != M.INIT):
        return None
    if len(target) != 1:
        raise SystemExit(f"FATAL: the mass target's sigma_eff differs between runs: {sorted(target)}")
    const = target.pop()
    plt = _start()
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
    out["mass_resolution"] = {"constant_prediction": pm(const), "rows": {row: _stats(M, v) for row, v in rows.items()}}
    return _save(fig, "mass_resolution", outdir)


def _has(fits, ds) -> bool:
    """Some fits file has this dataset (mlst_results._largest needs one)."""
    return any(ds in ck for f in fits.values() for ck in f["cells"].values())


def batchnorm(M, fits, runs, out) -> bool:
    """(1 - AUC) with the BatchNorm statistics recomputed over (1 - AUC) as trained, every run
    of the hierarchy and of the mass regression, largest training set: the range per task."""
    if not fits:
        return False
    res = {}
    for ds, met, _ in M.TASKS:
        if not _has(fits, ds):
            continue
        n = M._largest(fits, ds)
        r = []
        for arm in LEVELS + M.MASS:
            for m in runs[arm]:
                cells = fits.get(m, {}).get("cells", {})
                a, b = cells.get(M.PRIMARY, {}).get(ds, {}).get(n), cells.get("best70_bn", {}).get(ds, {}).get(n)
                if a is None or b is None:
                    continue
                if not b["converged"]:
                    raise SystemExit(f"FATAL: {m} {ds}: the BatchNorm-recomputed probe did not converge")
                r.append((1 - b[met]) / (1 - a[met]))
        if r:
            res[f"{ds}/{met}"] = {"n_runs": len(r), "min": f"{min(r):.3f}", "max": f"{max(r):.3f}"}
    if res:
        out["batchnorm_ratio"] = res
    return bool(res)


def probe_over_init(M, fits, runs, out) -> bool:
    """Linear-probe (1 - AUC) of each label set over that of the randomly initialized network of
    the same seed, largest training set, top and quark/gluon tagging (the paper quotes only the
    188- and 17-class rows)."""
    if not fits:
        return False
    res = {}
    for ds, met, _ in (t for t in M.TASKS if t[0] in ("top", "qg", "qg_herwig")):
        if not _has(fits, ds):
            continue
        n = M._largest(fits, ds)
        init = {m.rsplit("-s", 1)[1]: M._oma(M._probe(fits, m, ds, n, met)) for m in runs[M.INIT]}
        res[ds] = {}
        for arm in LEVELS + M.MASS:
            r = []
            for m in runs[arm]:
                x, y = M._oma(M._probe(fits, m, ds, n, met)), init.get(m.rsplit("-s", 1)[1])
                if x is not None and y is not None:
                    unc = isinstance(x, M.Unconverged) or isinstance(y, M.Unconverged)
                    r.append(M.Unconverged(x / y) if unc else x / y)
            if r:
                res[ds][arm] = {**_stats(M, r), "per_seed": [round(x, 3) for x in r]}
        if not res[ds]:
            del res[ds]
    if res:
        out["probe_over_init"] = res
    return bool(res)


def ft_rejection(M, per, runs, out) -> bool:
    """R50 and R30 after fine-tuning on the full training set (Table tab:ftrej), as the paper
    table prints them; and the fewest background jets passing behind any value. A rejection that
    is only a bound (fewer than one background jet passes) is left out and listed in only_a_bound
    (the paper's table refuses it)."""
    if not per:
        return False
    res, fewest, bound = {}, [], []

    def rej(rec, k, name):
        if rec[f"{k}_is_bound"]:
            bound.append(f"{key} {k} {name}")
            return None
        fewest.append(rec[f"{k}_n_bkg_pass"])
        return rec[k]

    for key, k in M.FT_REJ:
        n = M._sizes(per, key)[-1]
        col = {}
        for arm in LEVELS + M.MASS:
            v = []
            for m in runs[arm]:
                recs = per[key].get(m[len("mtx-"):], {}).get(n)
                if recs:
                    v.append(rej(recs[0], k, m))
            col[arm] = _stats(M, v)
        ref = per[key].get(M.SCRATCH, {}).get(n, [])
        col[M.SCRATCH] = _stats(M, [rej(r, k, r.get("run", M.SCRATCH)) for r in ref])
        res[f"{key}_{k}"] = col
    res["fewest_background_jets"] = min(fewest) if fewest else None
    res["only_a_bound"] = bound
    out["ft_rejection"] = res
    return True


def ft_status(M, runs, out) -> bool:
    """Fine-tuning cells DONE at the last data pull (ft_mlst/manifests.json) for the runs of the
    hierarchy and the mass regression, counting only the legs the paper reads (LEGS), out of
    4 legs x 4 training-set sizes = 16 per pretraining run; and their GPU types."""
    src = DATA / "ft_mlst/manifests.json"
    if not src.exists():
        return False
    man = json.loads(src.read_text())
    models = [m[len("mtx-"):] for arm in LEVELS + M.MASS for m in runs[arm]]
    cells = [c for k, c in man.items() if k.split("/")[0] == M.PRIMARY and k.split("/")[1] in LEGS
             and c["init"] in models]
    done = collections.Counter(c["init"] for c in cells)
    per_run = 4 * len(LEGS)
    out["ft_status"] = {"runs": len(models), "done": len(cells), "expected": per_run * len(models),
                        "incomplete": {m: done[m] for m in models if done[m] < per_run},
                        "gpus": sorted({c["gpu_device_name"] for c in cells})}
    return True


def compute(M, runs, out) -> bool:
    """GPU-hours per configuration (training and validation passes of every epoch), over the
    runs with a compute record."""
    comp = M._load(DATA / "compute")
    if not comp:
        return False
    _, hours = M.table_compute(comp, runs, MT.fmt_int)
    res, total = {}, collections.Counter()

    def row(per, n):
        tot = sum(per.values())
        return {"runs": n, "gpu_hours": f"{tot:,.0f}", "per_run": f"{tot / n:,.0f}",
                "types": ", ".join(f"{g} {100 * h / tot:.0f}%" for g, h in sorted(per.items(), key=lambda x: -x[1]))}
    for arm in LEVELS + M.MASS:
        if arm in hours:
            total.update(hours[arm])
            res[arm] = row(hours[arm], sum(m in comp for m in runs[arm]))
    if not res:
        return False
    res["all"] = row(total, sum(m in comp for a in LEVELS + M.MASS for m in runs[a]))
    out["compute"] = res
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=REPO / "figures/deck")
    a = ap.parse_args(argv)
    M = MF._mlst()
    fits = M._load(DATA / M.FITS)
    runs = M._paper(REPO)
    tag, per = M.load_ft(DATA)
    out = {"fine_tuning_readout": tag}
    figures = (("probes_n_jc", lambda: probe_lines(M, fits, runs, ("jc2_188", "jc1"), "probes_n_jc", a.outdir, out)),
               ("probes_n_bench", lambda: probe_lines(M, fits, runs, ("top", "qg"), "probes_n_bench", a.outdir, out)),
               ("pairs", lambda: pairs(M, fits, runs, a.outdir, out)),
               ("mass_resolution", lambda: mass_dots(M, runs, a.outdir, out)),
               ("ft_n_jc", lambda: ft_lines(M, per, runs, ("jc2", "jc1"), "ft_n_jc", a.outdir, out)),
               ("ft_n_bench", lambda: ft_lines(M, per, runs, ("top", "qg"), "ft_n_bench", a.outdir, out)),
               ("herwig", lambda: herwig(M, fits, per, runs, a.outdir, out)))
    numbers = (("batchnorm_ratio", lambda: batchnorm(M, fits, runs, out)),
               ("probe_over_init", lambda: probe_over_init(M, fits, runs, out)),
               ("ft_rejection", lambda: ft_rejection(M, per, runs, out)),
               ("ft_status", lambda: ft_status(M, runs, out)),
               ("compute", lambda: compute(M, runs, out)))
    for name, draw in figures:
        p = draw()
        if not p:
            (a.outdir / f"{name}.png").unlink(missing_ok=True)  # a stale figure would look current
        print(f"{name}: {p if p else 'skipped (no data yet)'}")
    a.outdir.mkdir(parents=True, exist_ok=True)
    for name, count in numbers:
        print(f"{name}: " + ("in deck_numbers.json" if count() else "skipped (no data yet)"))
    (a.outdir / "deck_numbers.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(a.outdir / "deck_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
