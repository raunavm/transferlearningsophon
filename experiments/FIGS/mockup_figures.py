#!/usr/bin/env python3
"""Mock-ups of the MLST paper's main-text figures in the layouts of docs/FIGURE_PLAN.md (local),
drawn from today's committed data and watermarked "[preliminary data]". What is a stand-in or old
is said in each figure's caption in paper/mockups/mockups.tex, not on the figure. Not paper
figures: the finals come from mlst_figures.py.

  F1  design: the label-set schematic (mlst_figures.figure_schematic) as panel (a)
  F2  ratio of 1 - AUC (188 classes, one against the rest) to the 188-class model of the same
      seed against the number of pretraining classes (hong et al. Fig. 5 form); natural-composition
      JetClass-II probes (fits_v3), linear probe only
  F3  per-class heatmap, eleven columns (three hierarchy levels, five random partitions, three
      flavor-controlled partitions); dot = the class shares its group with a decay differing only
      in quark or lepton flavor; hatched = fewer than 1,000 jets in the natural test subsample
  F4  three flavor pairs in the F2 form; open marker = the label set merges the pair
  F5  missing inputs: relative drop in JetClass-II accuracy, and R30 on top and quark/gluon
  F7  fine-tuning against training jets, 188 classes and from scratch, with the 188-class linear
      probe (not on JetClass-II); below the full set the old (under-trained) fine-tuning settings
  F8a mass: response, and resolution against jet pT and generator soft-drop mass
  F8b sculpting: QCD soft-drop mass after cuts, and the Jensen-Shannon distance

    python3 experiments/FIGS/mockup_figures.py [--outdir figures/mockups]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
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
# F4's controls drawn beside x = 17: the random partitions' runs around 17 * RAND, the flavor-controlled ones
OFFSET = {"RAND": 1.22, "FLAV_F0": 1.55, "FLAV_F1": 1.80, "FLAV_F1R": 2.10}
HEAT = (("L162", "162"), ("R42_Q1", "43"), ("R16_Q1", "17"),
        *((f"RAND2_p{k}", f"Random {k}") for k in range(1, 6)),
        ("FLAV_F0", "No flavor\nboundary"), ("FLAV_F1", "$b$-quark\nboundary"), ("FLAV_F1R", "Random flavor\nboundary"))
PAIRS = ("bb_vs_cc", "bbqq_vs_ccqq", "bc_vs_bq_cs")
# merge status of a pair -> marker fill: kept apart filled, merged open, partly merged half
HALF = {"apart": "full", "partly": "left", "merged": "none"}
# lepton flavor taken out of a quark-flavor orbit name (flavour_pair_map.v2.csv), first match only
LEPTON = ((r"tauhtau[em]$", "tauhtauL"), (r"tau[em]v$", "tauLv"), (r"[em]v$", "Lv"), (r"(ee|mm)$", "LL"),
          (r"[em]$", "L"))
# F3's row groups (17-class parents) in words; the three small two-prong leptonic groups share one label
GROUP_TEX = {"2P_HAD_2PARTON": r"$X\to$ 2 partons",
             "2P_HAD_TAUH_TAUH": r"$X\to\tau_h\tau_h,\,\tau_h\tau_\ell,\,\ell\ell$",
             "2P_SEMILEP_TAUH_TAUL": r"$X\to\tau_h\tau_h,\,\tau_h\tau_\ell,\,\ell\ell$",
             "2P_LEP_LL": r"$X\to\tau_h\tau_h,\,\tau_h\tau_\ell,\,\ell\ell$",
             "3P_HAD_3PARTON": r"$YY\to$ 3 partons",
             "3P_HAD_1PARTON_TAUH_TAUH": r"$YY\to$ parton $+\,\tau_h\tau_h$",
             "3P_SEMILEP_2PARTON_L": r"$YY\to$ 2 partons $+\,\ell$",
             "3P_SEMILEP_1PARTON_LL": r"$YY\to$ parton $+\,\ell\ell$",
             "3P_SEMILEP_1PARTON_TAUH_TAUL": r"$YY\to$ parton $+\,\tau_h\tau_\ell$",
             "4P_HAD_4PARTON": r"$YY\to$ 4 partons",
             "4P_HAD_2PARTON_TAUH_TAUH": r"$YY\to$ 2 partons $+\,\tau_h\tau_h$",
             "4P_SEMILEP_2PARTON_LL": r"$YY\to$ 2 partons $+\,\ell\ell$",
             "4P_SEMILEP_2PARTON_TAUH_TAUL": r"$YY\to$ 2 partons $+\,\tau_h\tau_\ell$",
             "4P_SEMILEP_NU_2PARTON_L": r"$YY\to$ 2 partons $+\,\ell\nu$",
             "4P_SEMILEP_NU_2PARTON_TAUL": r"$YY\to$ 2 partons $+\,\tau_\ell\nu$",
             "4P_HADVIS_NU_2PARTON_TAUH": r"$YY\to$ 2 partons $+\,\tau_h\nu$",
             "QCD_ALL": "QCD"}
# the two mass-regression models: line dash (F8a) and marker face
MASS_DASH = {"L162_MASS": (0, (6, 2)), "R16_Q1_MASS": (0, (2, 1.5))}
MASS_FACE = {"L162_MASS": None, "R16_Q1_MASS": "white"}
Y_RATIO = "$(1-\\mathrm{AUC})$\n/ 188-class model\n($\\downarrow$ better)"


def _top(fig) -> float:
    """Top of everything drawn (axes with titles, tick labels and legends; figure legends), in
    figure fraction."""
    r = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(r) for a in fig.axes] + [lg.get_window_extent(r) for lg in fig.legends]
    return max(b.y1 for b in boxes if b is not None) / fig.bbox.height


def _top_legend(fig, handles, labels, ncol: int) -> None:
    """A figure legend above every axes and its title."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, _top(fig) + 0.005), ncol=ncol,
               frameon=False, fontsize="small")


def preliminary(fig, title: str) -> None:
    """The mock-up marks: a short title with "[preliminary data]" above everything, and a light
    diagonal watermark that does not count toward the saved size."""
    fig.suptitle(f"{title} [preliminary data]", fontsize="small", y=_top(fig) + 0.01, va="bottom")
    w = fig.text(0.5, 0.5, "[preliminary data]", rotation=25, ha="center", va="center",
                 fontsize=min(60, 7 * fig.get_figwidth()), color="grey", alpha=0.12, zorder=0)
    w.set_in_layout(False)


def _look(M, row: str) -> dict:
    """mlst_figures._look, with the plan's pentagon for the mass-regression models ("P" is the
    b-quark-boundary partition's marker)."""
    d = F._look(M, row)
    if row in M.MASS:
        d["marker"] = "p"
    return d


def _face(fill: str, colour: str) -> dict:
    """Marker face for a merge status fill: full, half (left) or open (white)."""
    if fill == "none":
        return {"fillstyle": "full", "markerfacecolor": "white"}
    return {"fillstyle": fill, "markerfacecolor": colour, "markerfacecoloralt": "white"}


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


def _point(ax, M, x, row, vals, fill="full", size=10, dx=0.93):
    """The mean as a large marker with its spread (F._spread), each run a small white-edged dot
    drawn on top at x * dx, beside the marker."""
    look = _look(M, row)
    mean, el, eh = F._spread(vals)
    ax.errorbar(x, mean, yerr=None if el is None else [[el], [eh]], marker=look["marker"], markersize=size,
                color=look["color"], linestyle="none", markeredgewidth=1.5, elinewidth=2, capsize=0, zorder=2,
                **_face(fill, look["color"]))
    ax.plot([x * dx] * len(vals), vals, linestyle="none", marker="o", markersize=4, color=look["color"],
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    return mean


def _label_axis(ax, ys, xlabel=True) -> None:
    ax.set_xscale("log")
    ax.set_xlim(12, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # 162 and 188 sit 0.065 decades apart: their labels lean away from each other
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left")):
        lab.set_horizontalalignment(ha)
    ax.set_yscale("log")
    lo, hi = min(ys + [1.0]) / 1.1, max(ys) * 1.1
    ax.set_ylim(lo, hi)
    F._log_axis(ax.yaxis, lo, hi)
    ax.axhline(1.0, color="grey", linestyle="-.", linewidth=1.5, zorder=0)
    if xlabel:
        ax.set_xlabel("Number of classes in pretraining")


def _hierarchy(ax, M, runs, value, fills=None, line_label=None) -> list:
    """The hierarchy's points (fill per level from fills) and the dashed line through their means
    (linear probe). The run dots of 188 go to its right, the others' to their left."""
    xs, means, ys = [], [], []
    for row, x in X_OF.items():
        v = _paired(runs, runs[row], value)
        if v:
            means.append(_point(ax, M, x, row, v, fill=(fills or {}).get(row, "full"),
                                dx=1.07 if row == "L188" else 0.93))
            xs.append(x)
            ys += v
    ax.plot(xs, means, color="black", linestyle="--", linewidth=1.5, zorder=0, label=line_label)
    return ys


def figure_f2(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    style.use_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(style.PANEL, 0.75 * style.PANEL))
    value = lambda m: _jc2(M, fits, m)
    ys = _hierarchy(ax, M, runs, value, line_label="Hierarchy, linear probe")
    rand = [m for a in M.RAND for m in runs[a]]
    v = _paired(runs, rand, value)
    mean, el, eh = F._spread(v)
    ax.errorbar(17 * 1.10, mean, yerr=[[el], [eh]], marker="o", markersize=10, color=style.RAND_COLOUR,
                linestyle="none", elinewidth=2, capsize=0, zorder=2, label="Random partitions")
    # each partition a dot to the right of the mean marker
    for k, x in enumerate(v):
        ax.plot(17 * 1.30 * 1.03 ** (k - 2), x, "o", color=style.RAND_COLOUR, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    _label_axis(ax, ys + v)
    ax.set_ylabel(Y_RATIO, fontsize="small")
    # one column: two columns at "small" would be wider than the axes and widen the saved figure
    ax.legend(frameon=False, fontsize="small", loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, borderaxespad=0)
    preliminary(fig, "F2 Label granularity")
    return style.save(fig, "F2_label_granularity", outdir)


def figure_f4(M, runs, outdir) -> list:
    from matplotlib.lines import Line2D
    fits = M._load(DATA / "linprobe/fits_v3")
    mlp = M._load(DATA / "linprobe/fits_v3_mlp2")
    status = M.merge_status(REPO)
    heading = {k: h for k, h, *_ in M.PAIRS}
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(13.2, 4.6), sharey=True)
    ys_all = []
    for j, (ax, key) in enumerate(zip(axes, PAIRS)):
        value = lambda m, key=key: _pair(fits, m, key)
        ys = _hierarchy(ax, M, runs, value, fills={row: HALF[status[row][key]] for row in X_OF})
        # random partitions: one run each, no mean marker (it would mix merged and kept-apart partitions)
        for k, a in enumerate(M.RAND):
            for m in runs[a]:
                r = _paired(runs, [m], value)
                if r:
                    ax.plot(17 * OFFSET["RAND"] * 1.045 ** (k - 2), r[0], linestyle="none", marker="o",
                            color=style.RAND_COLOUR, markersize=7, markeredgewidth=1.5,
                            **_face(HALF[status[a][key]], style.RAND_COLOUR))
                    ys += r
        for a in M.FLAV:
            v = _paired(runs, runs[a], value)
            if v:
                _point(ax, M, 17 * OFFSET[a], a, v, fill=HALF[status[a][key]], size=9, dx=1.0)
                ys += v
        if key == "bc_vs_bq_cs":
            # the 162 vs 188 null (D6): MLP probe of the 162-class model, open, left of its linear point;
            # its ratio is to the 188-class MLP probe of the same seed
            mv = lambda m: (lambda c: None if c is None else 1 - c["auc"])(
                mlp.get(m, {}).get("cells", {}).get("best70", {}).get(key))
            v = _paired(runs, runs["L162"], mv)
            if v:
                look = _look(M, "L162")
                mean, el, eh = F._spread(v)
                ax.errorbar(X_OF["L162"] / 1.2, mean, yerr=None if el is None else [[el], [eh]],
                            marker=look["marker"], markersize=7, color=look["color"], markerfacecolor="white",
                            linestyle="none", markeredgewidth=1.5, capsize=0)
                ys += v
        ax.set_title(f"({'abc'[j]}) {heading[key]}", fontsize="small", loc="left")
        ys_all += ys
    for ax in axes:
        _label_axis(ax, ys_all, xlabel=False)
    axes[1].set_xlabel("Number of classes in pretraining", loc="center")
    axes[0].set_ylabel(Y_RATIO.replace("$(1-\\mathrm{AUC})$", "$(1-\\mathrm{AUC})$ on the pair"), fontsize="small")
    fig.tight_layout()
    grey = "0.35"
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=1.5),
               Line2D([], [], color=style.RAND_COLOUR, marker="o", linestyle="none", markersize=7),
               *(Line2D([], [], color=style.FLAV_COLOUR, marker=_look(M, a)["marker"], linestyle="none",
                        markersize=9) for a in M.FLAV),
               Line2D([], [], color=_look(M, "L162")["color"], marker="s", markerfacecolor="white",
                      linestyle="none", markersize=7, markeredgewidth=1.5),
               Line2D([], [], color=grey, marker="o", linestyle="none", markersize=8),
               Line2D([], [], color=grey, marker="o", linestyle="none", markersize=8, markeredgewidth=1.5,
                      **_face("left", grey)),
               Line2D([], [], color=grey, marker="o", linestyle="none", markersize=8, markeredgewidth=1.5,
                      **_face("none", grey))]
    labels = ["Hierarchy, linear probe", "Random partitions", *(F.ROW_NAME[a] for a in M.FLAV),
              "MLP probe, 162 classes", "Pair kept apart", "Pair partly merged", "Pair merged"]
    _top_legend(fig, handles, labels, ncol=3)
    preliminary(fig, "F4 Flavor pairs")
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


def _flavor_key(orbit: str) -> str:
    """A quark-flavor orbit with the lepton flavor also taken out (e, mu -> L; tau_e, tau_mu ->
    tau_L): classes with one key differ only in quark or lepton flavor."""
    for pat, rep in LEPTON:
        new = re.sub(pat, rep, orbit)
        if new != orbit:
            return new
    return orbit


def figure_f3(M, runs, outdir) -> list:
    fits = M._load(DATA / "linprobe/fits_v3")
    rung, order = F._class_order(REPO)
    qcd = lambda c: rung[c]["class_name"].startswith("label_QCD")
    order = [c for c in order if not qcd(c)] + [c for c in order if qcd(c)]
    key = {int(r["jet_label"]): _flavor_key(r["orbit"])
           for r in csv.DictReader((REPO / "configs/labelmaps/flavour_pair_map.v2.csv").open())}
    groups = _groups(REPO)
    counts = json.loads((DATA / "jc2_test/counts.json").read_text())["counts_auc"]
    R = np.array([F._seed_ratios(M, fits, runs, runs[k], "L188", order).mean(axis=0) for k, _ in HEAT]).T
    print(f"F3: {(R > 4).sum()} cells above 4 (largest {R.max():.3g}), {(R < 0.9).sum()} below 0.9")
    style.use_style()
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    n_res = sum(not qcd(c) for c in order)
    row = lambda i: i + (i >= n_res)      # a blank row between the resonant and the QCD blocks
    L = np.clip(np.log2(R), -2, 2)
    fig, ax = plt.subplots(figsize=(8.0, 12.5))
    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("white")
    im = ax.imshow(np.insert(L, n_res, np.nan, axis=0), aspect="auto", cmap=cmap, norm=Normalize(-2, 2),
                   interpolation="none")
    for i, c in enumerate(order):
        if counts[c] < 1000:
            ax.add_patch(Rectangle((-0.5, row(i) - 0.5), len(HEAT), 1, fill=False, hatch="////", edgecolor="grey",
                                   linewidth=0, alpha=0.5))
    # dot (right third of the cell): shares its group with a flavor sibling; dash (left third): ratio below 0.9.
    # White on dark cells.
    ink = lambda i, j: "white" if abs(L[i, j]) > 1.3 else "black"
    dots, dashes = ([], [], []), ([], [], [])
    for j, (k, _) in enumerate(HEAT):
        g = groups[k]
        n = 0
        for i, c in enumerate(order):
            if any(d != c and key.get(d) == key.get(c) and g[d] == g[c] for d in g):
                n += 1
                for lst, v in zip(dots, (j + 0.3, row(i), ink(i, j))):
                    lst.append(v)
            if R[i, j] < 0.9:
                for lst, v in zip(dashes, (j - 0.3, row(i), ink(i, j))):
                    lst.append(v)
        print(f"F3 dotted cells, {k}: {n}")
    ax.scatter(dots[0], dots[1], s=6, c=dots[2], marker="o", linewidths=0, zorder=3)
    ax.scatter(dashes[0], dashes[1], s=10, c=dashes[2], marker="_", linewidths=1.0, zorder=3)
    ax.set_xticks(range(len(HEAT)))
    ax.set_xticklabels([h for _, h in HEAT], fontsize="x-small", rotation=90)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for x in (2.5, 7.5):
        ax.axvline(x, color="black", linewidth=1.2)
    names = [rung[c]["R16_Q1_name"] for c in order]
    starts = [i for i in range(len(names)) if i == 0 or names[i] != names[i - 1]]
    for a in starts[1:]:
        if a != n_res:
            ax.axhline(row(a) - 0.5, color="black", linewidth=0.6)
    # one bracket and label per group label (the three small two-prong leptonic groups share one)
    labs = [GROUP_TEX[n] for n in names]
    blocks = [i for i in range(len(labs)) if i == 0 or labs[i] != labs[i - 1]]
    ends = blocks[1:] + [len(labs)]
    left = blended_transform_factory(ax.transAxes, ax.transData)
    for a, b in zip(blocks, ends):
        ax.plot([-0.015, -0.015], [row(a) - 0.4, row(b - 1) + 0.4], color="black", linewidth=1.0,
                transform=left, clip_on=False)
    ax.set_yticks([(row(a) + row(b - 1)) / 2 for a, b in zip(blocks, ends)])
    ax.set_yticklabels([labs[a] for a in blocks], fontsize="x-small")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, extend="max")
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["1/4", "1/2", "1", "2", "4"])
    cb.set_label(r"$(1-\mathrm{AUC})$ of the class / 188-class model")
    preliminary(fig, "F3 Per-class cost")
    return style.save(fig, "F3_per_class", outdir)


def figure_f5(M, runs, outdir) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fits = M._load(DATA / "linprobe/fits_v3")
    initbn = M._load(DATA / "linprobe/fits_v3_initbn")
    mlp = M._load(DATA / "linprobe/fits_v3_mlp2")
    mlp_bn = M._load(DATA / "linprobe/fits_v3_mlp2_initbn")
    fill = {mask: M._load(DATA / "fillcontrol" / mask) for mask in M.MASKS}
    acc = lambda d, m, ck: d.get(m, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(N_JC2, {}).get("accuracy")
    groups = (("L188", runs["L188"]), ("R16_Q1", runs["R16_Q1"]), ("INIT", list(M.INIT_MODELS)))
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4))

    def bars(ax, rows):
        """rows: [(name, look, [zero-input value of each run], text at the bar end or None,
        [MLP value of each run] or None)]: one horizontal bar per row (mean, SD bar, each run a
        dot); the MLP probe in open markers just below its bar, each run small, the mean large with
        its SD. A log x axis when the MLP is several times the bars."""
        vals = [x for _, _, v, _, mv in rows for x in v + (mv or [])]
        log = max(vals) > 3 * max(np.mean(v) for _, _, v, _, _ in rows)
        lo = 10 ** np.floor(np.log10(min(vals) / 1.5)) if log else 0
        if log:
            ax.set_xscale("log")
        ends = []
        for y, (name, look, v, txt, mv) in enumerate(rows):
            mean, el, eh = F._spread(v)
            ax.barh(y, mean - lo, left=lo, height=0.5, color=look["color"], edgecolor=look["color"], linewidth=1.5)
            ax.plot(v, [y] * len(v), "o", color="black", markersize=3, zorder=3)
            if el is not None:
                ax.errorbar(mean, y, xerr=[[el], [eh]], color="black", linewidth=1.5, capsize=0, zorder=3)
            ends.append((max(max(v), mean + (eh or 0)), y, f"{mean:.3g}" if txt is None else txt))
            if mv:
                mm, ml, mh = F._spread(mv)
                kw = dict(marker=look["marker"], color=look["color"], markerfacecolor="white", linestyle="none",
                          markeredgewidth=1.5)
                ax.errorbar(mm, y + 0.38, xerr=None if ml is None else [[ml], [mh]], markersize=9, capsize=0,
                            zorder=2, **kw)
                ax.plot(mv, [y + 0.38] * len(mv), markersize=4, zorder=3, **kw)
        hi = max(vals + [e for e, _, _ in ends])
        xmax = hi * 2.0 if log else hi * 1.5
        for e, y, txt in ends:
            ax.text(e * 1.08 if log else e + 0.02 * xmax, y, txt, va="center", fontsize="x-small")
        ax.set_xlim(lo, xmax)
        if log:
            # a few decade-and-three ticks: 1-2-5 ticks run together on this short axis
            ax.xaxis.set_major_locator(ticker.FixedLocator([t for t in (1, 3, 10, 30, 100, 300, 1000, 3000)
                                                            if lo <= t <= xmax]))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize="x-small")
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(len(rows) - 0.4, -0.5)

    for ax, mask, title in ((axes[0, 0], "TopReference", "(a) JetClass-II, top-tagging inputs zero"),
                            (axes[0, 1], "EnergyFlowQG", "(b) JetClass-II, quark/gluon inputs zero")):
        rows = []
        for row, models in groups:
            drop, zero = [], []
            for m in models:
                a, z = acc(fits, m, M._ck(m)), acc(fill[mask], m, M._ck(m))
                if a is not None and z is not None:
                    drop.append(100 * (a - z) / a)
                    zero.append(z)
            name = "Randomly initialized" if row == "INIT" else F.ROW_NAME[row]
            rows.append((name, _look(M, row), drop, f"acc. {np.mean(zero):.3f}", None))
        bars(ax, rows)
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel("Relative drop in accuracy [\\%]\n($\\leftarrow$ smaller is better)", fontsize="small")
    for ax, ds, n, title in ((axes[1, 0], "top", "1200000", "(c) Top tagging"),
                             (axes[1, 1], "qg", "1600000", "(d) Quark/gluon tagging")):
        r30 = lambda d, m, ck: d.get(m, {}).get("cells", {}).get(ck, {}).get(ds, {}).get(n, {}).get("r30")
        mlp30 = lambda d, models, ck: [c["r30"] for c in (d.get(m, {}).get("cells", {}).get(ck, {}).get(ds)
                                                          for m in models) if c] or None
        rows = [(F.ROW_NAME["L188"], _look(M, "L188"), [r30(fits, m, "best70") for m in runs["L188"]], None,
                 mlp30(mlp, runs["L188"], "best70")),
                (F.ROW_NAME["R16_Q1"], _look(M, "R16_Q1"), [r30(fits, m, "best70") for m in runs["R16_Q1"]], None,
                 None),
                ("Randomly initialized,\nBatchNorm recomputed", _look(M, "INIT"),
                 [x for x in (r30(initbn, m, "init_bn") for m in M.INIT_MODELS) if x is not None], None,
                 mlp30(mlp_bn, list(M.INIT_MODELS), "init_bn"))]
        bars(ax, rows)
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"$R_{30}$ ($\rightarrow$ better)", fontsize="small")
    fig.tight_layout()
    handles = [Patch(facecolor="0.55"), Line2D([], [], color="black", marker="o", linestyle="none", markersize=3),
               Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none", markersize=8,
                      markeredgewidth=1.5)]
    _top_legend(fig, handles, ["Linear probe, inputs zero", "One run", "MLP probe"], ncol=3)
    preliminary(fig, "F5 Missing input features")
    return style.save(fig, "F5_missing_inputs", outdir)


# (task, read-out file, data set in it, metric, probe data set or None, title, y label, smallest N drawn)
FT_PANELS = (("jc2", "leg1_metrics", None, "accuracy", None, "(a) JetClass-II, 162-group accuracy", "Accuracy",
              10_000),
             ("jc1", "leg2_metrics", None, "accuracy", "jc1", "(b) JetClass", "Accuracy", 10_000),
             ("top", "bench_metrics", "top", "r30", "top", "(c) Top tagging", r"$R_{30}$", 0),
             ("qg", "bench_metrics", "qg", "r30", "qg", "(d) Quark/gluon tagging", r"$R_{30}$", 0))
FT_LABELS = ("188 classes, fine-tuned", "188 classes, linear probe", "From scratch")


def figure_f7(M, runs, outdir) -> list:
    """The raw read-outs at every size (not mlst_results.load_ft, which keeps only the fixed-budget
    cells below the full set, none of them run yet). No probe line on JetClass-II: the fine-tuning
    read-out there is 162-group accuracy, the probe 188-class."""
    fits = M._load(DATA / "linprobe/fits_v3")
    style.use_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11.0, 9.0))
    outer = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
    look = _look(M, "L188")
    for k, (key, f, ds, met, pds, title, ylab, nmin) in enumerate(FT_PANELS):
        inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.0)
        ax = fig.add_subplot(inner[0])
        rx = fig.add_subplot(inner[1], sharex=ax)
        cells = json.loads((DATA / f"ft_mlst/best70_{f}.json").read_text())["cells"]
        cells = cells if ds is None else cells[ds]
        per = lambda init: {int(n[1:]): [r[met] for _, r in sorted(s.items())] for n, s in cells.get(init, {}).items()
                            if int(n[1:]) >= nmin}
        ft = {}
        for m in runs["L188"]:
            for n, v in per(m[len("mtx-"):]).items():
                ft.setdefault(n, []).extend(v)
        sc = per(M.SCRATCH)
        series = [(ft, dict(color=look["color"], marker="o", linestyle="-"), FT_LABELS[0])]
        if pds:
            probe = {}
            for m in runs["L188"]:
                for n, c in fits.get(m, {}).get("cells", {}).get("best70", {}).get(pds, {}).items():
                    if int(n) >= nmin:
                        probe.setdefault(int(n), []).append(c[met])
            series.append((probe, dict(color=look["color"], marker="o", linestyle="--"), FT_LABELS[1]))
        series.append((sc, dict(color="black", marker="s", linestyle="--", markerfacecolor="white"), FT_LABELS[2]))
        lows, highs = [], []
        for vals, kw, lab in series:
            xs = sorted(vals)
            mean = np.array([np.mean(vals[n]) for n in xs])
            sd = np.array([np.std(vals[n], ddof=1) if len(vals[n]) > 1 else 0 for n in xs])
            ax.plot(xs, mean, markersize=7, linewidth=2, label=lab, **kw)
            ax.fill_between(xs, mean - sd, mean + sd, color=kw["color"], alpha=0.15, linewidth=0)
            lows.append(min(mean - sd))
            highs.append(max(mean + sd))
        # the ratio panel: fine-tuning over from scratch at the same N
        ns = [n for n in sorted(ft) if n in sc]
        rx.plot(ns, [np.mean(ft[n]) / np.mean(sc[n]) for n in ns], markersize=6, linewidth=2, **series[0][1])
        rx.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set_xscale("log")
        if met == "r30":
            lo, hi = min(lows) / 1.15, max(highs) * 1.15
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            F._log_axis(ax.yaxis, lo, hi)
        ax.tick_params(labelbottom=False)
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_ylabel(ylab, fontsize="small")
        rx.set_ylabel("Ratio", fontsize="small")
        rx.set_xlabel("Training jets", fontsize="small")
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    seen = {}
    for a in fig.axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    _top_legend(fig, [seen[lab] for lab in FT_LABELS], list(FT_LABELS), ncol=3)
    preliminary(fig, "F7 Fine-tuning against training jets")
    return style.save(fig, "F7_finetune_vs_size", outdir)


def _edges(bins) -> np.ndarray:
    return np.array([tuple(float(x) for x in b.split("-")) for b in bins])


def figure_f8a(M, runs, outdir) -> list:
    from matplotlib.lines import Line2D
    head = M._load(DATA / "head")
    mass = {k.removesuffix("_mass"): v for k, v in head.items() if k.endswith("_mass")}
    curves = F._curves("response")
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6), sharex="col",
                             gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05, "wspace": 0.55})
    # (a) response: fraction of jets per bin, resonant and QCD weighted by their jet counts (n of
    # binned.group in head/<model>_mass.json), mean over the runs of each model; the first and last
    # stored bins hold the clipped overflow and are not drawn
    have = {a: [m for m in runs[a] if m in curves and m in mass] for a in M.MASS}
    edges = np.asarray(curves[next(m for a in M.MASS for m in have[a])]["bins"])[1:-1]

    def frac(m, what):
        g = mass[m]["uncorrected"]["binned"]["group"]
        h = sum(g[k]["n"] * np.asarray(curves[m]["hist"][f"{what}/{k}"], float) / np.sum(curves[m]["hist"][f"{what}/{k}"])
                for k in ("resonant", "qcd"))
        return h / h.sum()

    groomed = np.mean([frac(m, "uncorrected") for a in M.MASS for m in have[a]], axis=0)
    print(f"F8a: outside the axis of (a), groomed {100 * (groomed[0] + groomed[-1]):.1f}%")
    ax, rx = axes[0, 0], axes[1, 0]
    ax.stairs(groomed[1:-1], edges, color="black", linewidth=1.5)
    hi = 0.0
    for a in M.MASS:
        look = _look(M, a)
        h = np.mean([frac(m, "node") for m in have[a]], axis=0)
        print(f"F8a: outside the axis of (a), {a} {100 * (h[0] + h[-1]):.1f}%")
        ax.stairs(h[1:-1], edges, color=look["color"], linewidth=2, linestyle=MASS_DASH[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(groomed > 0, h / groomed, np.nan)[1:-1]
        hi = max(hi, np.nanmax(r))
        rx.stairs(r, edges, baseline=None, color=look["color"], linewidth=1.5, linestyle=MASS_DASH[a])
    rx.axhline(1, color="black", linewidth=1, linestyle=":")
    rx.set_ylim(0, 1.1 * hi)
    rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    rx.set_ylabel("Ratio", fontsize="small")
    rx.set_xlabel(r"$\ln(m/m^{\mathrm{gen}}_{\mathrm{SD}})$", fontsize="small")
    ax.set_ylabel("Fraction of jets", fontsize="small")
    ax.set_title("(a) Response", fontsize="small", loc="left")
    # (b), (c) resolution in bins; the two runs' range as a band per bin; the two models offset
    # within the bin and told apart by marker face
    for j, (var, xlabel) in enumerate((("jet_pt", r"Jet $p_{\mathrm{T}}$ [GeV]"),
                                       ("genjet_sdmass", r"Generator soft-drop mass [GeV]")), start=1):
        ax, rx = axes[0, j], axes[1, j]
        m0 = next(m for a in M.MASS for m in runs[a] if m in mass)
        bins = list(mass[m0]["uncorrected"]["binned"][var])
        e = _edges(bins)
        mid, half = e.mean(axis=1), np.diff(e, axis=1)[:, 0] / 2
        edges_b = np.r_[e[:, 0], e[-1, 1]]
        unc = np.array([mass[m0]["uncorrected"]["binned"][var][b]["sigma_eff"] for b in bins])
        ax.errorbar(mid, unc, xerr=half, color="black", linestyle="none", linewidth=1.5)
        for a in M.MASS:
            look = _look(M, a)
            v = np.array([[mass[m]["heads"]["best70"]["binned"][var][b]["sigma_eff"] for b in bins]
                          for m in runs[a] if m in mass and mass[m].get("heads")])
            mean = v.mean(axis=0)
            sgn = -1 if a == "L162_MASS" else 1
            ax.stairs(v.max(axis=0), edges_b, baseline=v.min(axis=0), fill=True, color=look["color"], alpha=0.3,
                      linewidth=0)
            for axis, y in ((ax, mean), (rx, mean / unc)):
                axis.errorbar(mid, y, xerr=half, color=look["color"], linestyle="none", linewidth=2)
                axis.plot(mid + sgn * 0.15 * half, y, linestyle="none", marker="p", markersize=8,
                          color=look["color"], markerfacecolor=MASS_FACE[a] or look["color"], markeredgewidth=1.5)
        rx.axhline(1, color="black", linewidth=1, linestyle=":")
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        rx.set_ylabel("Ratio", fontsize="small")
        rx.set_xlabel(xlabel, fontsize="small")
        ax.set_ylabel(r"Resolution $\sigma_{\mathrm{eff}}$", fontsize="small")
        ax.set_title("(b) Against jet $p_{\\mathrm{T}}$" if var == "jet_pt" else
                     "(c) Against generator mass", fontsize="small", loc="left")
    handles = [Line2D([], [], color="black", linewidth=1.5)]
    for a in M.MASS:
        look = _look(M, a)
        handles.append(Line2D([], [], color=look["color"], linewidth=2, linestyle=MASS_DASH[a], marker="p",
                              markersize=8, markerfacecolor=MASS_FACE[a] or look["color"], markeredgewidth=1.5))
    _top_legend(fig, handles, ["Groomed (soft-drop) mass", *(F.ROW_NAME[a] for a in M.MASS)], ncol=3)
    preliminary(fig, "F8a Jet-mass regression")
    return style.save(fig, "F8a_mass_regression", outdir)


def figure_f8b(M, runs, outdir) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    curves = F._curves("sculpting_v2")
    shown = ("L188", "R16_Q1", "R16_Q1_MASS")
    style.use_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.5, 9.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(4, 3.4), hspace=0.5)
    top = outer[0].subgridspec(2, 3, height_ratios=(3, 1), hspace=0.05, wspace=0.3)
    ratio_axes, hi = [], 0.0
    for j, a in enumerate(shown):
        m = next(m for m in runs[a] if m.endswith("-s1"))
        d = curves[m]
        edges = np.asarray(d["bins_jet_sdmass"])
        before = np.asarray(d["hist_before"], float)
        before = before / before.sum()
        ax = fig.add_subplot(top[0, j])
        rx = fig.add_subplot(top[1, j], sharex=ax)
        ax.stairs(before, edges, color="grey", fill=True, alpha=0.3)
        look = _look(M, a)
        for wp, ls in (("0.1", "-"), ("0.01", ":")):
            after = np.asarray(d["working_points"][wp]["hist_after"], float)
            after = after / after.sum()
            ax.stairs(after, edges, baseline=None, color=look["color"], linewidth=2, linestyle=ls)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            hi = max(hi, np.nanmax(r))
            rx.stairs(r, edges, baseline=None, color=look["color"], linewidth=1.5, linestyle=ls)
        rx.axhline(1, color="black", linewidth=1, linestyle=":")
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelbottom=False)
        ax.set_title(f"({'abc'[j]}) {F.ROW_NAME[a]}", fontsize="small", loc="left")
        rx.set_xlabel(r"Soft-drop mass $m_{\mathrm{SD}}$ [GeV]", fontsize="small")
        if j == 0:
            ax.set_ylabel("Fraction of jets", fontsize="small")
            rx.set_ylabel("Ratio", fontsize="small")
        ratio_axes.append(rx)
    for rx in ratio_axes:
        rx.set_ylim(0, 1.1 * hi)
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax = fig.add_subplot(outer[1].subgridspec(1, 3, wspace=0.3)[0, :2])
    d0 = curves[next(m for m in runs["L188"] if m.endswith("-s1"))]
    eff = np.array([c["eff"] for c in d0["curve"]])
    fmean, fsd = js_distance_floor(d0["hist_before"], [c["n_pass"] for c in d0["curve"]])
    ax.fill_between(eff, fmean - fsd, fmean + fsd, color="grey", alpha=0.25, linewidth=0,
                    label=r"Random selection, $\pm 1$ SD")
    ax.plot(eff, fmean, color="grey", linewidth=1.5, label="Random selection, mean")
    for i, row in enumerate(F.SCULPT_ROWS):
        have = [curves[m]["curve"] for m in runs[row] if m in curves]
        if not have:
            continue
        e = np.array([c["eff"] for c in have[0]]) * 1.04 ** (i - 2.5)   # small offsets so markers do not overlap
        per = np.array([[c["js_distance"] for c in h] for h in have])
        look = _look(M, row)
        ax.plot(e, per.mean(axis=0), color=look["color"], marker=look["marker"], markersize=7,
                linestyle=look["linestyle"], linewidth=2, label=F.ROW_NAME[row])
        if len(per) > 1:
            lo, hi = (per.min(axis=0), per.max(axis=0)) if len(per) == 2 else \
                (per.mean(axis=0) - per.std(axis=0, ddof=1), per.mean(axis=0) + per.std(axis=0, ddof=1))
            ax.fill_between(e, lo, hi, color=look["color"], alpha=0.25, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2e-2, 2e-1)
    F._log_axis(ax.yaxis, 2e-2, 2e-1)
    ax.set_xlabel(r"QCD efficiency $\varepsilon_{\mathrm{QCD}}$", fontsize="small")
    ax.set_ylabel("JS distance (base 2)", fontsize="small")
    ax.set_title("(d) Sculpting against the QCD efficiency", fontsize="small", loc="left")
    # the third column's space beside (d) is empty: the legend goes there
    ax.legend(frameon=False, fontsize="small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    handles = [Patch(facecolor="grey", alpha=0.3), Line2D([], [], color="black", linewidth=2, linestyle="-"),
               Line2D([], [], color="black", linewidth=2, linestyle=":")]
    _top_legend(fig, handles, ["All QCD jets", r"$\varepsilon_{\mathrm{QCD}} = 10\%$",
                               r"$\varepsilon_{\mathrm{QCD}} = 1\%$"], ncol=3)
    preliminary(fig, "F8b Mass sculpting")
    return style.save(fig, "F8b_sculpting", outdir)


def js_distance_floor(before: list, n_pass: list, draws: int = 200, seed: int = 0) -> tuple:
    """(mean, SD) over random draws of the Jensen-Shannon distance between the QCD mass histogram
    and that of n jets drawn from it at random, for each n (mlst_figures.jsd_floor for the
    distance, the quantity CMS JME-25-001 plots)."""
    ms = F._module("experiments/EVAL/mass_sculpting.py", "mass_sculpting")
    p = np.asarray(before, float) / np.sum(before)
    rng = np.random.default_rng(seed)
    d = np.array([[np.sqrt(ms.jsd(p, rng.multinomial(n, p) / n)) for _ in range(draws)] for n in n_pass])
    return d.mean(axis=1), d.std(axis=1)


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
