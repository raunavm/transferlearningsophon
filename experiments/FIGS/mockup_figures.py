#!/usr/bin/env python3
"""Mock-ups of the MLST paper's main-text figures in the layouts of docs/FIGURE_PLAN.md (local),
drawn from today's committed data and watermarked "[preliminary data]". Each figure says under
its axes what is a stand-in or old. Not paper figures: the finals come from mlst_figures.py.

  F1  design: the label-set schematic (mlst_figures.figure_schematic) as panel (a)
  F2  ratio of 1 - AUC (188 classes, one against the rest) to the 188-class model of the same
      seed against the number of pretraining classes (hong et al. Fig. 5 form); natural-composition
      JetClass-II probes (fits_v3), linear probe only
  F3  per-class heatmap, eleven columns (three hierarchy levels, five random partitions, three
      flavor-controlled partitions); dot = the class shares its group with a decay differing only
      in quark flavor; hatched = fewer than 1,000 jets in the natural test subsample
  F4  three flavor pairs in the F2 form; half-filled = the label set merges the pair
  F5  missing inputs: relative drop in JetClass-II accuracy, and R30 on top and quark/gluon
  F7  fine-tuning against training jets, 188 classes and from scratch, with the 188-class linear
      probe; below the full set the old (under-trained) fine-tuning settings
  F8a mass: response, and resolution against jet pT and generator soft-drop mass
  F8b sculpting: QCD soft-drop mass after cuts, and the Jensen-Shannon divergence

    python3 experiments/FIGS/mockup_figures.py [--outdir figures/mockups]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from matplotlib import ticker

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mlst_figures as F  # noqa: E402
import mlst_style as style  # noqa: E402

REPO = F.REPO
DATA = F.DATA
N_JC2 = "1000000"
X_OF = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
# controls drawn at x = 17, each a little to the right of the 17-class point
OFFSET = {"RAND": 1.10, "FLAV_F0": 1.22, "FLAV_F1": 1.30, "FLAV_F1R": 1.38}
HEAT = (("L162", "162"), ("R42_Q1", "43"), ("R16_Q1", "17"),
        *((f"RAND2_p{k}", f"Random {k}") for k in range(1, 6)),
        ("FLAV_F0", "No flavor\nboundary"), ("FLAV_F1", "$b$-quark\nboundary"), ("FLAV_F1R", "Random flavor\nboundary"))
PAIRS = (("bb_vs_cc", r"$X\to b\bar b$ vs $X\to c\bar c$"),
         ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $X\to YY\to ccqq$"),
         ("bc_vs_bq_cs", r"$X\to bc$ vs $X\to bq$, $X\to cs$"))
HALF = {"apart": "full", "partly": "left", "merged": "bottom"}


def preliminary(fig, title: str, note: str) -> None:
    """The mock-up marks: title with "[preliminary data]", a light diagonal watermark, and a note
    under the axes saying what is a stand-in."""
    fig.suptitle(f"{title} [preliminary data]", fontsize="medium", y=1.02)
    fig.text(0.5, 0.5, "[preliminary data]", rotation=25, ha="center", va="center", fontsize=60,
             color="grey", alpha=0.12, zorder=0)
    fig.text(0.0, -0.03, note, ha="left", va="top", fontsize="x-small", color="dimgray", wrap=True)


def _seed(model: str) -> str:
    return model.rsplit("-s", 1)[1]


def _ref(runs, model: str) -> str:
    """The 188-class run of a model's seed."""
    return next(m for m in runs["L188"] if _seed(m) == _seed(model))


def _jc2(M, fits, model: str):
    v = M._probe(fits, model, "jc2_188", N_JC2, "macro_auc_ovr")
    return None if v is None else 1 - float(v)


def _pair(fits, model: str, key: str):
    c = fits.get(model, {}).get("cells", {}).get("best70", {}).get("jc2_188_pairs", {}).get(N_JC2, {}).get(key)
    return None if c is None else 1 - c["auc"]


def _paired(runs, models, value) -> list:
    """value(model) / value(188-class run of the same seed), for the models where both exist."""
    out = []
    for m in models:
        a, b = value(m), value(_ref(runs, m))
        if a is not None and b is not None:
            out.append(a / b)
    return out


def _point(ax, M, x, row, vals, label=None, fill="full", size=12):
    """Each run a small dot, the mean a large marker with its spread (F._spread)."""
    look = F._look(M, row)
    mean, el, eh = F._spread(vals)
    ax.plot([x] * len(vals), vals, linestyle="none", marker="o", markersize=4, color=look["color"], alpha=0.6)
    ax.errorbar(x, mean, yerr=None if el is None else [[el], [eh]], marker=look["marker"], markersize=size,
                color=look["color"], linestyle="none", fillstyle=fill, markerfacecolor=look["color"],
                markerfacecoloralt="white", markeredgewidth=2, elinewidth=2, capsize=0, label=label)
    return mean


def _label_axis(ax, ys) -> None:
    ax.set_xscale("log")
    ax.set_xlim(12, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yscale("log")
    lo, hi = min(ys + [1.0]) / 1.1, max(ys) * 1.1
    ax.set_ylim(lo, hi)
    F._log_axis(ax.yaxis, lo, hi)
    ax.axhline(1.0, color="grey", linestyle="-.", linewidth=1.5, zorder=0)
    ax.set_xlabel("Number of classes in pretraining")


def _hierarchy(ax, M, runs, value, labels=True) -> list:
    """The hierarchy's points and the dashed line through their means (linear probe)."""
    xs, means, ys = [], [], []
    for row, x in X_OF.items():
        v = _paired(runs, runs[row], value)
        if v:
            means.append(_point(ax, M, x, row, v, label=F.ROW_NAME[row] if labels else None))
            xs.append(x)
            ys += v
    ax.plot(xs, means, color="black", linestyle="--", linewidth=1.5, zorder=0,
            label="Linear probe" if labels else None)
    return ys


def figure_f2(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    style.use_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(style.PANEL, 0.75 * style.PANEL))
    value = lambda m: _jc2(M, fits, m)
    ys = _hierarchy(ax, M, runs, value)
    rand = [m for a in M.RAND for m in runs[a]]
    v = _paired(runs, rand, value)
    for k, x in enumerate(v):
        ax.plot(17 * OFFSET["RAND"] * 1.015 ** (k - 2), x, "o", color=style.RAND_COLOUR, markersize=5, alpha=0.7)
    mean, el, eh = F._spread(v)
    ax.errorbar(17 * OFFSET["RAND"], mean, yerr=[[el], [eh]], marker="o", markersize=13, color=style.RAND_COLOUR,
                linestyle="none", elinewidth=2, capsize=0, label="Random partitions (5)")
    _label_axis(ax, ys + v)
    ax.set_ylabel(r"$(1-\mathrm{AUC})$ / 188-class model ($\downarrow$ better)")
    ax.legend(frameon=False, fontsize="x-small", loc="upper right")
    preliminary(fig, "F2 JetClass-II linear probe against label granularity",
                "Natural-composition test set (final: class-balanced). Fine-tuning line absent: waits on the "
                "class-balanced JetClass-II fine-tuning runs.")
    return style.save(fig, "F2_label_granularity", outdir)


def figure_f4(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    mlp = M._load(DATA / "linprobe/fits_v3_mlp2")
    status = M.merge_status(REPO)
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.4 * len(PAIRS), 0.75 * style.PANEL), sharey=True)
    ys_all = []
    for j, (ax, (key, title)) in enumerate(zip(axes, PAIRS)):
        value = lambda m, key=key: _pair(fits, m, key)
        ys = _hierarchy(ax, M, runs, value, labels=j == 0)
        for k, a in enumerate(M.RAND):
            for m in runs[a]:
                r = _paired(runs, [m], value)
                if r:
                    ax.plot(17 * OFFSET["RAND"] * 1.015 ** (k - 2), r[0], "o", color=style.RAND_COLOUR, markersize=7,
                            fillstyle=HALF[status[a][key]], markerfacecoloralt="white",
                            label="Random partitions" if j == 0 and k == 0 else None)
                    ys += r
        for a in M.FLAV:
            v = _paired(runs, runs[a], value)
            if v:
                _point(ax, M, 17 * OFFSET[a], a, v, label=F.ROW_NAME[a] if j == 0 else None,
                       fill=HALF[status[a][key]], size=10)
                ys += v
        if key == "bc_vs_bq_cs":
            # the 162 vs 188 null: MLP probe beside the linear one (D6), open markers
            for row in ("L188", "L162"):
                mv = lambda m: (lambda c: None if c is None else 1 - c["auc"])(
                    mlp.get(m, {}).get("cells", {}).get("best70", {}).get(key))
                v = _paired(runs, runs[row], mv)
                if v:
                    look = F._look(M, row)
                    ax.errorbar(X_OF[row] * 1.06, np.mean(v), yerr=None if len(v) < 3 else np.std(v, ddof=1),
                                marker=look["marker"], markersize=8, color=look["color"], markerfacecolor="white",
                                linestyle="none", markeredgewidth=2, capsize=0,
                                label="MLP probe" if row == "L188" else None)
                    ys += v
            ax.legend(frameon=False, fontsize="xx-small", loc="upper right")
        ax.set_title(title, fontsize="small", loc="left")
        ys_all += ys
    for ax in axes:
        _label_axis(ax, ys_all)
    axes[0].set_ylabel(r"$(1-\mathrm{AUC})$ on the pair / 188-class model ($\downarrow$ better)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize="x-small")
    preliminary(fig, "F4 Class pairs that differ only in flavor",
                "Natural-composition pair test sets (final: class-balanced pairs). Half-filled marker: the label "
                "set merges the pair. The controlled-flavor partitions have two runs each.")
    return style.save(fig, "F4_flavor_pairs", outdir)


def _groups(root) -> dict:
    """{configuration: {class name: group}} from the label maps (as mlst_results.merge_status)."""
    maps = root / "configs/labelmaps"
    out: dict = {}
    for f, cols in (("rung_label_maps.v1.csv", ("L162", "R42_Q1", "R16_Q1")),
                    ("rand_label_map.v2.csv", tuple(f"RAND2_p{k}" for k in range(1, 6))),
                    ("flavour_pair_map.v2.csv", ("FLAV_F0", "FLAV_F1", "FLAV_F1R"))):
        for r in csv.DictReader((maps / f).open()):
            for c in cols:
                out.setdefault(c, {})[int(r["jet_label"])] = r[c]
    return out


def figure_f3(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    rung, order = F._class_order(REPO)
    qcd = lambda c: rung[c]["class_name"].startswith("label_QCD")
    order = [c for c in order if not qcd(c)] + [c for c in order if qcd(c)]
    orbit = {int(r["jet_label"]): r["orbit"]
             for r in csv.DictReader((REPO / "configs/labelmaps/flavour_pair_map.v2.csv").open())}
    groups = _groups(REPO)
    counts = json.loads((DATA / "jc2_test/counts.json").read_text())["counts_auc"]
    R = np.array([F._seed_ratios(M, fits, runs, runs[key], "L188", order).mean(axis=0) for key, _ in HEAT]).T
    style.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(9.0, 15.0))
    L = np.clip(np.log2(R), -2, 2)
    im = ax.imshow(L, aspect="auto", cmap="RdBu_r", norm=Normalize(-2, 2), interpolation="none")
    for i, c in enumerate(order):
        if counts[c] < 1000:
            ax.add_patch(Rectangle((-0.5, i - 0.5), len(HEAT), 1, fill=False, hatch="////", edgecolor="grey",
                                   linewidth=0, alpha=0.5))
        for j, (key, _) in enumerate(HEAT):
            g = groups[key]
            if orbit.get(c) and any(d != c and orbit.get(d) == orbit[c] and g[d] == g[c] for d in g):
                ax.plot(j, i, "o", color="black", markersize=1.8)
    ax.set_xticks(range(len(HEAT)))
    ax.set_xticklabels([h for _, h in HEAT], fontsize="xx-small", rotation=90)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for x in (2.5, 7.5):
        ax.axvline(x, color="black", linewidth=1.2)
    names = ["QCD" if qcd(c) else rung[c]["R16_Q1_name"] for c in order]
    starts = [i for i in range(len(names)) if i == 0 or names[i] != names[i - 1]]
    ends = starts[1:] + [len(names)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
    ax.set_yticklabels([names[a].replace("_", " ") for a in starts], fontsize="xx-small")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(length=0)
    for a in starts[1:]:
        ax.axhline(a - 0.5, color="black", linewidth=0.6 if names[a] != "QCD" else 2.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["1/4", "1/2", "1", "2", "4"])
    cb.set_label(r"$(1-\mathrm{AUC})$ of the class / 188-class model")
    preliminary(fig, "F3 JetClass-II per-class cost of each label set",
                "Natural-composition test subsample: hatched rows have fewer than 1,000 test jets here (final: "
                "class-balanced set, all jets). Dot: the class shares its group with a decay that differs only "
                "in quark flavor. Random partitions: one run each.")
    return style.save(fig, "F3_per_class", outdir)


def figure_f5(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    initbn = M._load(DATA / "linprobe/fits_v3_initbn")
    mlp = M._load(DATA / "linprobe/fits_v3_mlp2")
    mlp_bn = M._load(DATA / "linprobe/fits_v3_mlp2_initbn")
    fill = {mask: M._load(DATA / "fillcontrol" / mask) for mask in M.MASKS}
    acc = lambda d, m, ck: d.get(m, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(N_JC2, {}).get("accuracy")
    groups = (("L188", runs["L188"]), ("R16_Q1", runs["R16_Q1"]), ("INIT", list(M.INIT_MODELS)))
    treat = (("zero", None), ("mean", "//"), ("imputed", ".."))
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(2 * 5.2, 2 * 3.6))

    def bars(ax, rows):
        """rows: [(name, colour, {treatment: [values of each run] or None})], horizontal bars."""
        y = 0
        ticks = []
        for name, colour, vals in rows:
            for k, (t, hatch) in enumerate(treat):
                v = vals.get(t)
                yy = y + k * 0.8
                if v:
                    mean, el, eh = F._spread(v)
                    ax.barh(yy, mean, height=0.7, color=colour if hatch is None else "white", edgecolor=colour,
                            hatch=hatch, linewidth=1.5)
                    ax.plot(v, [yy] * len(v), "o", color="black", markersize=3)
                    if el is not None:
                        ax.errorbar(mean, yy, xerr=[[el], [eh]], color="black", linewidth=1.5, capsize=0)
                    ax.text(mean * 1.02 + (eh or 0), yy, f"{mean:.3g}", va="center", fontsize="xx-small")
                else:
                    ax.text(0.01, yy, f"{t}: waits on probes", va="center", fontsize="xx-small", color="dimgray",
                            transform=ax.get_yaxis_transform())
            ticks.append((y + 0.8, name))
            y += 3 * 0.8 + 0.8
        ax.set_yticks([t for t, _ in ticks])
        ax.set_yticklabels([n for _, n in ticks], fontsize="x-small")
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.invert_yaxis()
        ax.set_xlim(left=0)

    for ax, mask, title in ((axes[0, 0], "TopReference", "JetClass-II, top-tagging inputs zero"),
                            (axes[0, 1], "EnergyFlowQG", "JetClass-II, quark/gluon inputs zero")):
        rows = []
        for row, models in groups:
            drop = []
            for m in models:
                a, z = acc(fits, m, M._ck(m)), acc(fill[mask], m, M._ck(m))
                if a is not None and z is not None:
                    drop.append(100 * (a - z) / a)
            name = "Randomly initialized$^*$" if row == "INIT" else F.ROW_NAME[row]
            rows.append((name, F._look(M, row)["color"], {"zero": drop}))
        bars(ax, rows)
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"Relative drop in accuracy [\%] ($\leftarrow$ smaller is better)", fontsize="x-small")
    for ax, ds, n, title in ((axes[1, 0], "top", "1200000", "Top tagging, linear probe"),
                             (axes[1, 1], "qg", "1600000", "Quark/gluon, linear probe")):
        r30 = lambda d, m, ck: d.get(m, {}).get("cells", {}).get(ck, {}).get(ds, {}).get(n, {}).get("r30")
        rows = [(F.ROW_NAME[row], F._look(M, row)["color"], {"zero": [r30(fits, m, "best70") for m in runs[row]]})
                for row in ("L188", "R16_Q1")]
        rows.append(("Randomly initialized,\nBatchNorm recomputed", style.INIT_STYLE["color"],
                     {"zero": [x for x in (r30(initbn, m, "init_bn") for m in M.INIT_MODELS) if x is not None]}))
        if ds == "top":
            rows.append(("188 classes,\nkinematic inputs only", style.LEVEL_COLOURS[188], {}))
        bars(ax, rows)
        # MLP probe on the zero bars (D6): open circles
        for i, (src, models, ck) in enumerate(((mlp, runs["L188"], "best70"), (None, None, None),
                                               (mlp_bn, list(M.INIT_MODELS), "init_bn"))):
            if src is None:
                continue
            v = [c["r30"] for c in (src.get(m, {}).get("cells", {}).get(ck, {}).get(ds) for m in models) if c]
            if v:
                ax.plot(np.mean(v), i * 3.2, "o", markersize=9, markerfacecolor="white", markeredgecolor="black",
                        markeredgewidth=1.5, label="MLP probe" if i == 0 else None)
        ax.legend(frameon=False, fontsize="xx-small", loc="lower right")
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$R_{30}$ ($\rightarrow$ better)", fontsize="x-small")
    fig.tight_layout()
    preliminary(fig, "F5 Missing input features: zero, mean and imputed",
                "Zero bars only; mean and imputed bars wait on the fill probes (jc2bal-xfill, then jc2bal-fitfill). "
                "Top row: natural-composition JetClass-II; *randomly initialized network without recomputed "
                "BatchNorm (final: recomputed). Kinematic-only bar waits on the run (epoch 36 of 80).")
    return style.save(fig, "F5_missing_inputs", outdir)


FT_PANELS = (("jc2", "leg1_metrics", None, "accuracy", "jc2_188", "JetClass-II", "Accuracy"),
             ("jc1", "leg2_metrics", None, "accuracy", "jc1", "JetClass", "Accuracy"),
             ("top", "bench_metrics", "top", "r30", "top", "Top tagging", r"$R_{30}$"),
             ("qg", "bench_metrics", "qg", "r30", "qg", "Quark/gluon tagging", r"$R_{30}$"))


def figure_f7(M, runs, outdir) -> list:
    """The raw read-outs at every size (not mlst_results.load_ft, which keeps only the fixed-budget
    cells below the full set, none of them run yet)."""
    fits = M._load(DATA / "linprobe/fits_v3")
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(2 * 5.5, 2 * 5.0), sharex="col",
                             gridspec_kw={"height_ratios": (3, 1, 3, 1)})
    for k, (key, f, ds, met, pds, title, ylab) in enumerate(FT_PANELS):
        ax, rx = axes[2 * (k // 2), k % 2], axes[2 * (k // 2) + 1, k % 2]
        cells = json.loads((DATA / f"ft_mlst/best70_{f}.json").read_text())["cells"]
        cells = cells if ds is None else cells[ds]
        per = lambda init: {int(n[1:]): [r[met] for _, r in sorted(s.items())] for n, s in cells.get(init, {}).items()}
        ft = {}
        for m in runs["L188"]:
            for n, v in per(m[len("mtx-"):]).items():
                ft.setdefault(n, []).extend(v)
        sc = per(M.SCRATCH)
        probe = {}
        for m in runs["L188"]:
            for n, c in fits.get(m, {}).get("cells", {}).get("best70", {}).get(pds, {}).items():
                probe.setdefault(int(n), []).append(c[met])
        look = F._look(M, "L188")
        for vals, kw, lab in ((ft, dict(color=look["color"], marker="o", linestyle="-"), "188 classes, fine-tuned"),
                              (probe, dict(color=look["color"], marker="o", linestyle="--", markerfacecolor="white"),
                               "188 classes, linear probe"),
                              (sc, dict(color="black", marker="s", linestyle="--", markerfacecolor="white"),
                               "From scratch")):
            xs = sorted(vals)
            mean = np.array([np.mean(vals[n]) for n in xs])
            sd = np.array([np.std(vals[n], ddof=1) if len(vals[n]) > 1 else 0 for n in xs])
            ax.plot(xs, mean, markersize=7, linewidth=2, label=lab, **kw)
            ax.fill_between(xs, mean - sd, mean + sd, color=kw["color"], alpha=0.15, linewidth=0)
            if vals is not sc:
                ns = [n for n in xs if n in sc]
                rx.plot(ns, [np.mean(vals[n]) / np.mean(sc[n]) for n in ns], markersize=6, linewidth=2, **kw)
        rx.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set_xscale("log")
        if met == "r30":
            ax.set_yscale("log")
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_ylabel(ylab, fontsize="small")
        rx.set_ylabel("Ratio to\nfrom scratch", fontsize="xx-small")
        rx.set_xlabel("Training jets", fontsize="small")
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize="x-small")
    fig.tight_layout()
    preliminary(fig, "F7 Fine-tuning against the number of training jets",
                "Below the full set: old fine-tuning settings (19 steps per epoch, under-trained; final: fixed "
                "budget after the learning-rate sweep). Full-set points valid. JetClass-II: natural composition "
                "(final: class-balanced). Probe: zero inputs on top and quark/gluon.")
    return style.save(fig, "F7_finetune_vs_size", outdir)


def _edges(bins) -> np.ndarray:
    return np.array([tuple(float(x) for x in b.split("-")) for b in bins])


def figure_f8a(M, runs, outdir) -> list:
    head = M._load(DATA / "head")
    mass = {k.removesuffix("_mass"): v for k, v in head.items() if k.endswith("_mass")}
    curves = F._curves("response")
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.2), gridspec_kw={"height_ratios": (3, 1)}, sharex="col")
    # (a) response, all matched jets (resonant + QCD), mean over the runs of each model
    ref = next(c for a in M.MASS for m in runs[a] for c in [curves.get(m)] if c)
    edges = np.asarray(ref["bins"])[1:-1]
    dens = lambda h: np.asarray(h, float)[1:-1] / np.sum(h) / np.diff(edges)
    allj = lambda c, what: dens(np.add(c["hist"][f"{what}/resonant"], c["hist"][f"{what}/qcd"]))
    groomed = np.mean([allj(curves[m], "uncorrected") for a in M.MASS for m in runs[a] if m in curves], axis=0)
    ax, rx = axes[0, 0], axes[1, 0]
    ax.stairs(groomed, edges, color="black", linewidth=1.5, label="Groomed (soft-drop) mass")
    for a in M.MASS:
        look = F._look(M, a)
        h = np.mean([allj(curves[m], "node") for m in runs[a] if m in curves], axis=0)
        ax.stairs(h, edges, color=look["color"], linewidth=2, linestyle=look["linestyle"], label=f"Regression, {F.ROW_NAME[a]}")
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.stairs(np.where(groomed > 0, h / groomed, np.nan), edges, color=look["color"], linewidth=1.5,
                      linestyle=look["linestyle"])
    rx.axhline(1, color="black", linewidth=1, linestyle=":")
    rx.set_yscale("log")
    rx.set_ylabel("Ratio to\ngroomed", fontsize="xx-small")
    rx.set_xlabel(r"$\ln(m/m^{\mathrm{gen}}_{\mathrm{SD}})$", fontsize="small")
    ax.set_ylabel("Density", fontsize="small")
    ax.set_title("Response, all matched jets", fontsize="small", loc="left")
    # (b), (c) resolution in bins
    for j, (var, xlabel) in enumerate((("jet_pt", r"Jet $p_{\mathrm{T}}$ [GeV]"),
                                       ("genjet_sdmass", r"Generator soft-drop mass [GeV]")), start=1):
        ax, rx = axes[0, j], axes[1, j]
        m0 = next(m for a in M.MASS for m in runs[a] if m in mass)
        bins = list(mass[m0]["uncorrected"]["binned"][var])
        e = _edges(bins)
        mid, half = e.mean(axis=1), np.diff(e, axis=1)[:, 0] / 2
        unc = np.array([mass[m0]["uncorrected"]["binned"][var][b]["sigma_eff"] for b in bins])
        ax.errorbar(mid, unc, xerr=half, color="black", linestyle="none", linewidth=1.5)
        for a in M.MASS:
            look = F._look(M, a)
            v = np.array([[mass[m]["heads"]["best70"]["binned"][var][b]["sigma_eff"] for b in bins]
                          for m in runs[a] if m in mass and mass[m].get("heads")])
            ax.errorbar(mid, v.mean(axis=0), xerr=half, color=look["color"], linestyle="none", linewidth=2)
            ax.fill_between(mid, v.min(axis=0), v.max(axis=0), color=look["color"], alpha=0.2, linewidth=0)
            rx.errorbar(mid, v.mean(axis=0) / unc, xerr=half, color=look["color"], linestyle="none", linewidth=2)
        rx.axhline(1, color="black", linewidth=1, linestyle=":")
        rx.set_ylabel("Ratio to\ngroomed", fontsize="xx-small")
        rx.set_xlabel(xlabel, fontsize="small")
        ax.set_ylabel(r"Resolution $\sigma_{\mathrm{eff}}$", fontsize="small")
        ax.set_title("Resolution against " + ("jet $p_{\\mathrm{T}}$" if var == "jet_pt" else "generator mass"),
                     fontsize="small", loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize="x-small")
    fig.tight_layout()
    preliminary(fig, "F8a Jet-mass regression with the auxiliary target",
                "Resolution here: half the smallest interval holding 68 per cent of ln(m/m_gen), as stored (decision "
                "P7 may switch to the ATLAS inter-quantile definition). Two runs per model: band = their range. "
                "No per-bin statistical bars yet.")
    return style.save(fig, "F8a_mass_regression", outdir)


def figure_f8b(M, runs, outdir) -> list:
    curves = F._curves("sculpting_v2")
    shown = ("L188", "R16_Q1", "R16_Q1_MASS")
    style.use_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(3, 3, height_ratios=(3, 1, 3.4), hspace=0.08)
    for j, a in enumerate(shown):
        m = next(m for m in runs[a] if m.endswith("-s1"))
        d = curves[m]
        edges = np.asarray(d["bins_jet_sdmass"])
        before = np.asarray(d["hist_before"], float)
        before = before / before.sum()
        ax = fig.add_subplot(gs[0, j])
        rx = fig.add_subplot(gs[1, j], sharex=ax)
        ax.stairs(before, edges, color="grey", fill=True, alpha=0.3, label="All QCD jets")
        look = F._look(M, a)
        for wp, ls, lab in (("0.1", "-", "After cut, 10 per cent QCD efficiency"),
                            ("0.01", ":", "After cut, 1 per cent QCD efficiency")):
            after = np.asarray(d["working_points"][wp]["hist_after"], float)
            after = after / after.sum()
            ax.stairs(after, edges, color=look["color"], linewidth=2, linestyle=ls, label=lab)
            with np.errstate(divide="ignore", invalid="ignore"):
                rx.stairs(np.where(before > 0, after / before, np.nan), edges, color=look["color"], linewidth=1.5,
                          linestyle=ls)
        rx.axhline(1, color="black", linewidth=1, linestyle=":")
        ax.set_title(F.ROW_NAME[a], fontsize="small", loc="left")
        rx.set_xlabel(r"Soft-drop mass $m_{\mathrm{SD}}$ [GeV]", fontsize="small")
        if j == 0:
            ax.set_ylabel("Fraction of jets", fontsize="small")
            rx.set_ylabel("Selected / all", fontsize="xx-small")
            ax.legend(frameon=False, fontsize="xx-small", loc="upper right")
        plt.setp(ax.get_xticklabels(), visible=False)
    ax = fig.add_subplot(gs[2, :2])
    d0 = curves[next(m for m in runs["L188"] if m.endswith("-s1"))]
    pts = d0["curve"]
    eff = [c["eff"] for c in pts]
    ax.fill_between(eff, 1e-7, js_distance_floor(d0["hist_before"], [c["n_pass"] for c in pts]), color="grey",
                    alpha=0.25, linewidth=0, label="Random selection, same size")
    for row in F.SCULPT_ROWS:
        have = [curves[m]["curve"] for m in runs[row] if m in curves]
        if not have:
            continue
        e = [c["eff"] for c in have[0]]
        per = np.array([[c["js_distance"] for c in h] for h in have])
        mean = per.mean(axis=0)
        look = F._look(M, row)
        ax.plot(e, mean, color=look["color"], marker=look["marker"], markersize=5, linestyle=look["linestyle"],
                linewidth=2, label=F.ROW_NAME[row])
        if len(per) > 1:
            lo, hi = (per.min(axis=0), per.max(axis=0)) if len(per) == 2 else \
                (per.mean(axis=0) - per.std(axis=0, ddof=1), per.mean(axis=0) + per.std(axis=0, ddof=1))
            ax.fill_between(e, lo, hi, color=look["color"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e-1)
    ax.set_xlabel(r"QCD efficiency $\varepsilon_{\mathrm{QCD}}$", fontsize="small")
    ax.set_ylabel("Jensen-Shannon distance (base 2)", fontsize="small")
    ax.legend(frameon=False, fontsize="xx-small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    preliminary(fig, "F8b Mass sculpting",
                "Committed data. Top row: seed 1 of each model (final: the same, or the mean over runs). Bottom: "
                "Jensen-Shannon distance as in CMS JME-25-001 Fig. 9; grey band = a random selection of the same size.")
    return style.save(fig, "F8b_sculpting", outdir)


def js_distance_floor(before: list, n_pass: list, draws: int = 200, seed: int = 0) -> list:
    """Mean Jensen-Shannon distance between the QCD mass histogram and that of n jets drawn from it
    at random, for each n (mlst_figures.jsd_floor for the distance, the quantity CMS plots)."""
    ms = F._module("experiments/EVAL/mass_sculpting.py", "mass_sculpting")
    p = np.asarray(before, float) / np.sum(before)
    rng = np.random.default_rng(seed)
    return [float(np.mean([np.sqrt(ms.jsd(p, rng.multinomial(n, p) / n)) for _ in range(draws)])) for n in n_pass]


FIGURES = (figure_f2, figure_f3, figure_f4, figure_f5, figure_f7, figure_f8a, figure_f8b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=style.FIGURES / "mockups")
    a = ap.parse_args(argv)
    M = F._mlst()
    runs = M._paper(REPO)
    print("figure_schematic:", F.figure_schematic(REPO, a.outdir))
    rc = 0
    for f in FIGURES:
        try:
            print(f"{f.__name__}:", f(M, runs, a.outdir))
        except Exception as e:  # one broken mock-up must not hide the others
            import traceback
            traceback.print_exc()
            print(f"{f.__name__}: FAILED {type(e).__name__}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
