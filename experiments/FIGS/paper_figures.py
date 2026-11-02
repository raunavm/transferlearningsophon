#!/usr/bin/env python3
"""The MLST paper's figures, final (docs/FIGURE_PLAN.md as amended by the PI decisions of
2026-11-02): drawn at their printed size, 8.5 or 15 cm wide with the text between 8 and 12 pt, in
Helvetica (mlst_style.use_style(final=True)), into figures/mlst/ as PDF (vector) and PNG. Every
number is selected by the generator's own functions (mlst_results.py, mlst_v4.py) from the pulled
data under experiments/FIGS/data/v2/, so a figure and the table or macro beside it cannot differ.
The JetClass-II probes are the class-balanced v4 fits (mlst_v4.load_fits), never fits_v3.
Nothing but data is drawn: what a caption must say is said in the caption.

Main text
  F2   ratio of 1 - AUC (JetClass-II, AUC averaged over the 188 classes, each against the rest) to
       the 188-class model of the same seed against the number of pretraining classes (Hong et al.,
       Fig. 5 form): the hierarchy joined (linear probe dashed, fine-tuning on class-balanced
       JetClass-II solid, open markers); the random partitions and the random partitions with
       controlled flavor boundaries as their own markers at 17
  F3   the same ratio class by class (linear probe), eleven columns; filled dot: the class shares
       its group with a decay that differs only in quark flavor, ring: only in lepton flavor; dash:
       ratio below 1 (the sign in greyscale); hatched: fewer than THIN class-balanced test jets
  F4   F2's form for three class pairs; marker fill: pair kept apart (filled), partly merged
       (half), merged (open) by that label set; the MLP probe of the 162-class model beside the
       162 null on X->bc
  F5   missing input features set to zero, to the training mean or to random draws: relative drop
       in JetClass-II accuracy with the top (quark/gluon) inputs substituted, and R30 of the linear
       probe on top and quark/gluon; the substitution chosen on validation
       (mlst_results.choose_substitution) outlined in black; MLP probe on the zero bars
  F7   fine-tuning, linear probe and training from scratch against the number of training jets,
       188-class model, with the ratio of fine-tuning to training from scratch
  F8a  auxiliary mass regression (head/atlas_res): response histogram, ATLAS resolution (half the
       15.9-84.1 % inter-quantile range over the median) and median response against jet pT and
       generator soft-drop mass; groomed mass with its bootstrap bars, the two runs of each mass
       model as their range
  F8b  CMS JME-25-001 pair: QCD soft-drop mass after cuts at 10 % and 1 % QCD efficiency, and the
       Jensen-Shannon distance (base 2) against the QCD efficiency with the random-selection floor
  F9   X->bb and X->bs: eps_S / sqrt(eps_B) against eps_B and the rejection 1/eps_B against eps_S
Appendix
  A_roc  rejection against signal efficiency after fine-tuning, top and quark/gluon (Pythia,
         Herwig), the four label sets and training from scratch, ratio to training from scratch
  A10    linear probes against the number of training jets, every label set and the randomly
         initialized network (BatchNorm recomputed)
  A17    the mass regression on resonant and on QCD jets: response histograms, and resolution and
         median response against jet pT and generator mass

A figure or panel whose input is not pulled yet is skipped with one line naming the input. main()
prints what it drew and skipped and exits 0 when at least one figure was drawn.

    python3 experiments/FIGS/paper_figures.py [--outdir figures/mlst]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import traceback

import numpy as np
from matplotlib import ticker

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mlst_figures as F  # noqa: E402
import mlst_style as style  # noqa: E402
import mockup_figures as MK  # noqa: E402

REPO, DATA = F.REPO, F.DATA
OUTDIR = REPO / "figures" / "mlst"
FULL, ONE, CM = style.FULL_WIDTH, style.ONE_COLUMN, style.CM
SCRATCH, INIT_BN, KIN = "scratch-v2", "INIT_BN", "KIN"      # mlst_results.SCRATCH, .INIT_BN; KIN: one model
NAME = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
        "L162_MASS": "162 classes, mass regression", "R16_Q1_MASS": "17 classes, mass regression",
        "RAND": "Random partitions", "FLAV_F0": "No flavor boundary", "FLAV_F1": "$b$-quark boundary",
        "FLAV_F1R": "Random flavor boundary", INIT_BN: "Randomly initialized,\nBatchNorm recomputed",
        KIN: "188 classes,\nkinematic inputs only", SCRATCH: "Trained from scratch"}
LEVELS = ("L188", "L162", "R42_Q1", "R16_Q1")
X_OF = MK.X_OF
# positions at the class-count axis, as factors of x: fine-tuning beside its probe (188 to the right,
# the others to the left, so 162 and 188 do not collide), the controls to the right of 17 and within
# 0.12 decades of it (17 x 1.32), so that none reads as another class count; the offsets carry no meaning
FT_X = {"L188": 1.08}
FT_X_OTHER = 1 / 1.08
RAND_X, RAND_DOTS, DOT_STEP = 17 * 1.06, 17 * 1.11, 1.012     # mean; the five partitions' runs around RAND_DOTS
FLAV_X = {"FLAV_F0": 17 * 1.19, "FLAV_F1": 17 * 1.255, "FLAV_F1R": 17 * 1.32}
RATIO = r"$(1-\mathrm{AUC})$ / 188-class model"
Y_RATIO = RATIO + "\n" + r"($\downarrow$ better)"
THIN = 1000                 # F3: classes with fewer class-balanced test jets are hatched (MlstJcBalNThin)
SUBST = ("zero", "mean", "draws")
SUBST_NAME = {"zero": "Set to zero", "mean": "Training mean", "draws": "Random draws"}
SUBST_HATCH = {"zero": "", "mean": "//////", "draws": "....", "none": ""}
# F7 and A10 panels: (fine-tuning task, probe data set, metric, title, y label)
SIZE_PANELS = (("jc2", "jc2_188", "accuracy", "JetClass-II", "Accuracy"),
               ("jc1", "jc1", "accuracy", "JetClass", "Accuracy"),
               ("top", "top", "r30", "Top tagging", r"$R_{30}$"),
               ("qg", "qg", "r30", "Quark/gluon tagging", r"$R_{30}$"))
F8B_SHOWN = ("L188", "R16_Q1", "R16_Q1_MASS")
WPS = (("0.1", "-"), ("0.01", ":"))
ROC_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "SCRATCH")
ROC_MIN_EPS = 0.1


class Skip(Exception):
    """A figure whose input is not pulled yet; the message names the input."""


def absent(what: str) -> None:
    """One line naming an input a panel lacks; the panel or series is left out."""
    print(f"  absent: {what}")


def inputs(M) -> dict:
    """Every input of the figures, read once; empty where not pulled yet."""
    fits, raw = M.V4.load_fits(DATA)
    census = DATA / "jc2bal" / "census.json"
    xbx = DATA / "xbx"
    return {"fits": fits, "raw": raw, "mlp": M.load_mlp(DATA), "fill": M._load(DATA / M.FITS_FILL),
            "per": M.load_ft(DATA)[1], "atlas": M._load(DATA / "head" / "atlas_res"),
            "sculpt": F._curves("sculpting_v2"),
            "census": json.loads(census.read_text()) if census.exists() else None,
            "xbx": _xr().load(xbx) if xbx.is_dir() else {}, "xbx_scratch": M.xbx_scratch(DATA),
            "ft_curves": F._ft_curves()}


def _xr():
    return F._module("experiments/XBS/xbx_report.py", "xbx_report")


# ------------------------------------------------------------------ drawing helpers
def look(M, row: str) -> dict:
    """Colour, marker and line style (mlst_figures._look) with the plan's variants: a mass-regression
    model in its label set's colour with a pentagon, dashed; the kinematic-only model in the
    188-class colour with a hexagon, dash-dot; the random partitions pooled as "RAND"."""
    if row == KIN:
        return {"color": style.LEVEL_COLOURS[188], "marker": "h", "linestyle": "-."}
    if row == "RAND":
        return {"color": style.RAND_COLOUR, "marker": "o", "linestyle": "none"}
    d = F._look(M, row)
    if row in M.MASS:
        d.update(marker="p", linestyle="--")
    return d


def _fig(width: float, height: float):
    """A figure at its printed size, laid out by matplotlib's constrained layout."""
    style.use_style(final=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.02, hspace=0.02)
    return fig


def _save(fig, name: str, outdir: pathlib.Path) -> list:
    return style.save(fig, name, outdir, tight=False, dpi=300)


def _pct() -> str:
    import matplotlib.pyplot as plt
    return r"\%" if plt.rcParams["text.usetex"] else "%"


def _panel(ax, k: int, text: str = "") -> None:
    ax.set_title(f"({'abcdefghij'[k]}) {text}".rstrip(), loc="left", fontsize="small")


def _legend(fig, handles, labels, ncol: int, **kw) -> None:
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False, fontsize="small", **kw)


def _fmt3(x: float) -> str:
    """A value at a bar end: three significant figures, integers from 100."""
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _point(ax, M, x, row, vals, fill="full", size=5.5):
    """The mean as a marker with its spread (mlst_figures._spread: SD for three or more runs, the
    range for two), each run a small white-edged dot on it. Returns the mean."""
    lk = look(M, row)
    mean, el, eh = F._spread(vals)
    ax.errorbar(x, mean, yerr=None if el is None else [[el], [eh]], marker=lk["marker"], markersize=size,
                color=lk["color"], linestyle="none", elinewidth=1.0, capsize=0, zorder=3, **MK._face(fill, lk["color"]))
    ax.plot([x] * len(vals), vals, linestyle="none", marker="o", markersize=2.0, color=lk["color"],
            markeredgecolor="white", markeredgewidth=0.3, zorder=4)
    return mean


def _over_full(M, runs, arm: str, value) -> list:
    """value of each run of arm over value of the 188-class run of the same seed (mlst_results._ratio_vals)."""
    r, _ = M._ratio_vals(runs, arm, "L188", value)
    return [x for x in r if x is not None]


def _hierarchy(ax, M, runs, value, fill_of=lambda row: "full", ft: bool = False) -> list:
    """The four levels' points and the line through their means: the linear probe at x, dashed;
    fine-tuning (ft) beside it with open markers, solid. Returns every plotted ratio."""
    xs, means, ys = [], [], []
    for row, x in X_OF.items():
        v = _over_full(M, runs, row, value)
        if not v:
            continue
        xx = x * (FT_X.get(row, FT_X_OTHER) if ft else 1.0)
        means.append(_point(ax, M, xx, row, v, fill="none" if ft else fill_of(row)))
        xs.append(xx)
        ys += v
    if xs:
        ax.plot(xs, means, color="black", linestyle="-" if ft else "--", linewidth=0.9, zorder=1)
    return ys


def _classes_axis(ax, ys: list, xlabel: bool = True) -> None:
    """Number of pretraining classes on a log axis (ticks 17, 43, 162, 188); the ratio on a log axis
    from the data, the 188-class model's line at 1 (Hong et al.'s baseline, dash-dot)."""
    ax.set_xscale("log")
    ax.set_xlim(12, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # 162 and 188 sit 0.065 decades apart: their labels lean away from each other
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left")):
        lab.set_horizontalalignment(ha)
    ax.set_yscale("log")
    lo, hi = min(ys + [1.0]) / 1.1, max(ys + [1.0]) * 1.1
    ax.set_ylim(lo, hi)
    F._log_axis(ax.yaxis, lo, hi)
    ax.axhline(1.0, color="0.5", linestyle="-.", linewidth=0.8, zorder=0)
    if xlabel:
        ax.set_xlabel("Number of classes in pretraining")


def _has(fits: dict, ds: str) -> bool:
    return any(ck.get(ds) for f in fits.values() for ck in f.get("cells", {}).values())


def _jc2_probe(M, fits):
    """1 - AUC of a run's linear probe on class-balanced JetClass-II at the largest training set."""
    n = M._largest(fits, "jc2_188")
    return lambda m: M._oma(M._probe(fits, m, "jc2_188", n, "macro_auc_ovr"))


def _jc2_ft(M, per):
    """1 - AUC of a run's fine-tuning on class-balanced JetClass-II at its largest size, or None."""
    sizes = M._sizes(per, "jc2") if per.get("jc2") else []
    if not sizes:
        return None
    n = max(sizes)

    def value(m):
        x = M._ft(per, "jc2", m, n, "macro_auc_ovr")
        return None if x is None else 1 - x
    return value


# ------------------------------------------------------------------ F2
def figure_f2(M, runs, I, outdir) -> list:
    from matplotlib.lines import Line2D
    fits = I["fits"]
    if not _has(fits, "jc2_188"):
        raise Skip("class-balanced JetClass-II linear probes (linprobe/fits_v4)")
    probe, ft = _jc2_probe(M, fits), _jc2_ft(M, I["per"])
    fig = _fig(ONE, 8.0 * CM)
    ax = fig.add_subplot()
    ys = _hierarchy(ax, M, runs, probe)
    if ft:
        ys += _hierarchy(ax, M, runs, ft, ft=True)
    else:
        absent("F2: fine-tuning on class-balanced JetClass-II (budget/budget_metrics.json, leg_jc2b)")
    per = [_over_full(M, runs, a, probe) for a in M.RAND]
    v = [x for p in per for x in p]
    if v:
        mean, el, eh = F._spread(v)
        ax.errorbar(RAND_X, mean, yerr=None if el is None else [[el], [eh]], marker="o", markersize=5.5,
                    color=style.RAND_COLOUR, linestyle="none", elinewidth=1.0, capsize=0, zorder=3)
        for k, p in enumerate(per):         # one x per partition, its runs on it
            ax.plot([RAND_DOTS * DOT_STEP ** (k - 2)] * len(p), p, "o", color=style.RAND_COLOUR, markersize=2.5,
                    markeredgecolor="white", markeredgewidth=0.3, zorder=4)
        ys += v
    else:
        absent("F2: linear probes of the random partitions")
    for a in M.FLAV:
        v = _over_full(M, runs, a, probe)
        if v:
            _point(ax, M, FLAV_X[a], a, v)
            ys += v
        else:
            absent(f"F2: linear probes of {a}")
    _classes_axis(ax, ys)
    ax.set_ylabel(Y_RATIO)
    grey = "0.45"
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=0.9, marker="o", markersize=4,
                      markerfacecolor=grey, markeredgecolor=grey),
               Line2D([], [], color="black", linestyle="-", linewidth=0.9, marker="o", markersize=4,
                      markerfacecolor="white", markeredgecolor=grey),
               Line2D([], [], color=style.RAND_COLOUR, marker="o", linestyle="none", markersize=4),
               *(Line2D([], [], color=style.FLAV_COLOUR, marker=look(M, a)["marker"], linestyle="none", markersize=4)
                 for a in M.FLAV)]
    labels = ["Hierarchy, linear probe", "Hierarchy, fine-tuned", NAME["RAND"], *(NAME[a] for a in M.FLAV)]
    if not ft:
        handles.pop(1)
        labels.pop(1)
    _legend(fig, handles, labels, ncol=2, handlelength=1.5, columnspacing=0.8)
    return _save(fig, "F2_label_granularity", outdir)


# ------------------------------------------------------------------ F3
def flavor_merges(M, root: pathlib.Path) -> dict:
    """{configuration: {class name: (merged with a quark-flavor sibling, with a lepton-flavor sibling)}}:
    the class shares its group with a decay that differs from it only in quark (lepton) flavor
    (mlst_results._orbits and _groups, the definitions of the per-class macros)."""
    q, lep = M._orbits(root)
    groups = M._groups(root)
    out = {}
    for arm, g in groups.items():
        out[arm] = {n: (any(d != n and q[d] == q[n] and g[d] == g[n] for d in q),
                        any(d != n and lep[d] == lep[n] and g[d] == g[n] for d in lep)) for n in q}
    return out


def heat(M, fits, runs, root: pathlib.Path) -> tuple[list, np.ndarray]:
    """(label-map rows in F3's order: the resonant classes by 17-class group with flavor siblings
    adjacent, then the QCD classes; (classes, columns) array of the mean over runs of the per-class
    ratio of 1 - AUC to the 188-class run of the same seed, NaN where absent)."""
    rows = M._class_rows(root)
    rows = [r for r in rows if not r["class_name"].startswith("label_QCD")] + \
           [r for r in rows if r["class_name"].startswith("label_QCD")]
    R = np.full((len(rows), len(MK.HEAT)), np.nan)
    for j, (arm, _) in enumerate(MK.HEAT):
        rat = M.per_class_ratios(fits, runs, arm)
        if not rat:
            absent(f"F3: per-class linear probes of {arm}")
        for i, r in enumerate(rows):
            v = rat.get(int(r["jet_label"]))
            if v:
                R[i, j] = float(np.mean(v))
    return rows, R


def figure_f3(M, runs, I, outdir) -> list:
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import blended_transform_factory
    fits = I["fits"]
    if not _has(fits, "jc2_188"):
        raise Skip("class-balanced JetClass-II linear probes (linprobe/fits_v4)")
    rows, R = heat(M, fits, runs, REPO)
    if np.isnan(R[:, :3]).all():
        raise Skip("per-class linear probes of the hierarchy (linprobe/fits_v4)")
    merges = flavor_merges(M, REPO)
    census = I["census"]
    if census is None:
        absent("F3 hatching: class-balanced test counts (jc2bal/census.json)")
    counts = census["counts"]["ft_test"] if census else None
    n_res = sum(not r["class_name"].startswith("label_QCD") for r in rows)
    row = lambda i: i + (i >= n_res)      # a blank row between the resonant and the QCD classes
    L = np.clip(np.log2(R), -2, 2)
    fig = _fig(FULL, 21.0 * CM)
    ax = fig.add_subplot()
    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("white")
    im = ax.imshow(np.insert(L, n_res, np.nan, axis=0), aspect="auto", cmap=cmap, norm=Normalize(-2, 2),
                   interpolation="none")
    if counts:
        for i, r in enumerate(rows):
            if counts[int(r["jet_label"])] < THIN:
                ax.add_patch(Rectangle((-0.5, row(i) - 0.5), len(MK.HEAT), 1, fill=False, hatch="//////",
                                       edgecolor="0.3", linewidth=0))
    # marks, white on dark cells: filled dot (right third) quark-flavor sibling in the group, a ring at
    # the same place (around the dot when both) lepton-flavor sibling, dash (left third) ratio below 1
    ink = lambda i, j: "white" if np.isfinite(L[i, j]) and abs(L[i, j]) > 1.3 else "black"
    marks = {}
    for j, (arm, _) in enumerate(MK.HEAT):
        for i, r in enumerate(rows):
            qm, lm = merges.get(arm, {}).get(r["class_name"], (False, False))
            if qm:
                marks.setdefault(("q", ink(i, j)), []).append((j + 0.3, row(i)))
            if lm:
                marks.setdefault(("l", ink(i, j)), []).append((j + 0.3, row(i)))
            if R[i, j] < 1:
                marks.setdefault(("d", ink(i, j)), []).append((j - 0.3, row(i)))
    for (kind, c), pts in marks.items():
        x, y = zip(*pts)
        if kind == "q":
            ax.scatter(x, y, s=2.5, c=c, marker="o", linewidths=0, zorder=3)
        elif kind == "l":
            ax.scatter(x, y, s=4, facecolors="none", edgecolors=c, marker="o", linewidths=0.4, zorder=3)
        else:
            ax.scatter(x, y, s=6, c=c, marker="_", linewidths=0.7, zorder=3)
    ax.set_xticks(range(len(MK.HEAT)))
    ax.set_xticklabels([h for _, h in MK.HEAT], rotation=90, fontsize="small")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for x in (2.5, 7.5):
        ax.axvline(x, color="black", linewidth=0.9)
    names = [r["R16_Q1_name"] for r in rows]
    starts = [i for i in range(len(names)) if i == 0 or names[i] != names[i - 1]]
    for a in starts[1:]:
        if a != n_res:
            ax.axhline(row(a) - 0.5, color="black", linewidth=0.4)
    labs = [MK.GROUP_TEX.get(n, n.replace("_", " ")) for n in names]
    blocks = [i for i in range(len(labs)) if i == 0 or labs[i] != labs[i - 1]]
    ends = blocks[1:] + [len(labs)]
    left = blended_transform_factory(ax.transAxes, ax.transData)
    for a, b in zip(blocks, ends):
        ax.plot([-0.012, -0.012], [row(a) - 0.4, row(b - 1) + 0.4], color="black", linewidth=0.7,
                transform=left, clip_on=False)
    ax.set_yticks([(row(a) + row(b - 1)) / 2 for a, b in zip(blocks, ends)])
    ax.set_yticklabels([labs[a] for a in blocks], fontsize="small")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="both")
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["1/4", "1/2", "1", "2", "4"])
    cb.minorticks_off()
    cb.set_label(RATIO + ", per class")
    handles = [Line2D([], [], marker="o", color="black", linestyle="none", markersize=2.5),
               Line2D([], [], marker="o", color="black", markerfacecolor="none", linestyle="none", markersize=3,
                      markeredgewidth=0.6),
               Line2D([], [], marker="_", color="black", linestyle="none", markersize=5, markeredgewidth=0.9)]
    labels = ["Merged with a quark-flavor sibling", "Merged with a lepton-flavor sibling", "Ratio below 1"]
    if counts:
        handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="//////", linewidth=0))
        labels.append(f"Fewer than {THIN:,} test jets")
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize="small")
    return _save(fig, "F3_per_class", outdir)


# ------------------------------------------------------------------ F4
def figure_f4(M, runs, I, outdir) -> list:
    from matplotlib.lines import Line2D
    fits, mlp = I["fits"], I["mlp"]
    if not _has(fits, M.PAIR_DS):
        raise Skip("class-balanced JetClass-II pair probes (linprobe/fits_v4)")
    status = M.merge_status(REPO)
    heading = {k: h for k, h, *_ in M.PAIRS}
    fig = _fig(FULL, 7.4 * CM)
    axes = fig.subplots(1, len(MK.PAIRS), sharey=True)
    ys_all = []
    for j, (ax, key) in enumerate(zip(axes, MK.PAIRS)):
        value = lambda m, key=key: (lambda a: None if a is None else 1 - a)(M._pair(fits, m, key))
        ys = _hierarchy(ax, M, runs, value, fill_of=lambda row, key=key: MK.HALF[status[row][key]])
        # the random partitions one run each, no mean marker: it would mix merged and kept-apart partitions
        for k, a in enumerate(M.RAND):
            for x in _over_full(M, runs, a, value):
                ax.plot(RAND_DOTS * DOT_STEP ** (k - 2), x, linestyle="none", marker="o", color=style.RAND_COLOUR,
                        markersize=4, **MK._face(MK.HALF[status[a][key]], style.RAND_COLOUR))
                ys.append(x)
        for a in M.FLAV:
            v = _over_full(M, runs, a, value)
            if v:
                _point(ax, M, FLAV_X[a], a, v, fill=MK.HALF[status[a][key]])
                ys += v
        if key == "bc_vs_bq_cs":
            # beside the 162 vs 188 null (D6): the MLP probe of the 162-class model over the 188-class
            # MLP probe of the same seed, open, left of the linear point
            mv = lambda m: M._mlp(mlp, m, key)
            v = _over_full(M, runs, "L162", mv)
            if v:
                _point(ax, M, X_OF["L162"] * FT_X_OTHER, "L162", v, fill="none")
                ys += v
            else:
                absent("F4: MLP probes of the 162- and 188-class models on X->bc (linprobe/fits_v4_mlp)")
        if not ys:
            absent(f"F4: pair probes {key}")
        _panel(ax, j, heading[key])
        ys_all += ys
    for ax in axes:
        _classes_axis(ax, ys_all, xlabel=False)
    axes[1].set_xlabel("Number of classes in pretraining", loc="center")
    axes[0].set_ylabel(r"$(1-\mathrm{AUC})$ on the pair" + "\n/ 188-class model " + r"($\downarrow$ better)")
    grey = "0.4"
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=0.9),
               Line2D([], [], color=style.RAND_COLOUR, marker="o", linestyle="none", markersize=4),
               *(Line2D([], [], color=style.FLAV_COLOUR, marker=look(M, a)["marker"], linestyle="none", markersize=4)
                 for a in M.FLAV),
               Line2D([], [], color=look(M, "L162")["color"], marker="s", markerfacecolor="white",
                      linestyle="none", markersize=4),
               *(Line2D([], [], color=grey, marker="o", linestyle="none", markersize=4, **MK._face(f, grey))
                 for f in ("full", "left", "none"))]
    labels = ["Hierarchy, linear probe", NAME["RAND"], *(NAME[a] for a in M.FLAV), "162 classes, MLP probe",
              "Pair kept apart", "Pair partly merged", "Pair merged"]
    _legend(fig, handles, labels, ncol=5)
    return _save(fig, "F4_flavor_pairs", outdir)


# ------------------------------------------------------------------ F5
def _jc2_acc(M, raw, fill, m, task, s):
    """Absolute JetClass-II accuracy of a run's linear probe with the task's missing slots substituted."""
    c = M._top_of(M._raw_cells(raw if s == "zero" else fill, m), M.JC2_SUBST[task][s])
    return None if c is None else c["accuracy"]


def _bench_r30(M, fits, fill, m, task, s):
    """R30 of a run's linear probe on top or quark/gluon under substitution s, largest training set;
    with zero, the fits_v4 cell while the substitution fits are not pulled for that run."""
    v = M._fill_r30(fill, m, task, s) if fill else None
    if v is None and s == "zero":
        v = M._rej(M._probe_rec(fits, m, task), "r30")
    return v


def _bars(ax, M, groups: list, chosen) -> None:
    """Horizontal bars, one group per model, one bar per substitution (colour per model, hatch per
    substitution), the mean with the SD (range for two runs) and each run a dot; the substitution
    chosen on validation outlined in black; the MLP probe (open, black edge) on the zero bar; the
    value of the bar at its end. groups: [(row, label, {substitution: (values, text)}, MLP values,
    slots)]: a group takes `slots` rows and each substitution its own row of them whatever the group
    holds, so panels sharing their rows align."""
    y, ticks, ends, xs = 0.0, [], [], []
    for row, label, vals, mlpv, slots in groups:
        lk, first = look(M, row), y
        for s, (v, txt) in vals.items():
            y = first + (SUBST.index(s) if s in SUBST else 0)
            mean, el, eh = F._spread(v)
            solid = s in ("zero", "none")
            ax.barh(y, mean, height=0.78, color=lk["color"] if solid else "white", edgecolor=lk["color"],
                    hatch=SUBST_HATCH[s], linewidth=0.8, linestyle="-." if row == KIN else "-", zorder=2)
            if s == chosen:
                ax.barh(y, mean, height=0.78, fill=False, edgecolor="black", linewidth=1.4, zorder=3)
            ax.plot(v, [y] * len(v), "o", color="black", markersize=1.8, zorder=4)
            if el is not None:
                ax.errorbar(mean, y, xerr=[[el], [eh]], color="black", linewidth=0.8, capsize=0, zorder=4)
            ends.append((max(max(v), mean + (eh or 0)), y, txt))
            xs += v
            if s == "zero" and mlpv:
                mm, ml, mh = F._spread(mlpv)
                ax.errorbar(mm, y, xerr=None if ml is None else [[ml], [mh]], marker=lk["marker"], markersize=4.5,
                            color="black", markerfacecolor="white", linestyle="none", capsize=0, zorder=5)
                xs += mlpv
        ticks.append((first + (slots - 1) / 2, label))
        y = first + slots + 0.5
    if not xs:
        ax.set_axis_off()
        return
    xmax = 1.45 * max(xs + [e for e, _, _ in ends])
    for e, yy, txt in ends:
        if txt:
            ax.text(e + 0.015 * xmax, yy, txt, va="center", ha="left", fontsize="small")
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([lab for _, lab in ticks], fontsize="small")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 1.0, -0.6)


def figure_f5(M, runs, I, outdir) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fits, raw, fill, mlp = I["fits"], I["raw"], I["fill"], I["mlp"]
    if not raw:
        raise Skip("class-balanced JetClass-II linear probes with the inputs set to zero (linprobe/fits_v4)")
    chosen, _ = M.chosen_substitutions(fill, runs)
    if not fill:
        absent("F5: the training-mean and random-draw substitutions (linprobe/fits_v4_fill)")
    if chosen["top"] is None:
        absent("F5: the substitution chosen on validation (fits_v4_fill of every model)")
    rows = ("L188", "R16_Q1", INIT_BN)
    kin = {t: M._rej(M._probe_rec(fits, M.KIN_MODEL, t), "r30") for t in ("top", "qg")}
    if kin["top"] is None:
        absent("F5: linear probe of the 188-class model with kinematic inputs only on top tagging")
    fig = _fig(FULL, 12.5 * CM)
    # the panels of a row share their models: the model names on the left column only
    axes = fig.subplots(2, 2, sharey="row")
    for k, (ax, task, title) in enumerate(((axes[0, 0], "top", "JetClass-II, top inputs"),
                                           (axes[0, 1], "qg", "JetClass-II, quark/gluon inputs"))):
        groups = []
        for row in rows:
            vals = {}
            for s in SUBST:
                v = [x for m in runs[row] if (x := M._jc2_drop(raw, fill, m, task, s)) is not None]
                acc = [x for m in runs[row] if (x := _jc2_acc(M, raw, fill, m, task, s)) is not None]
                if v:
                    vals[s] = (v, f"{np.mean(acc):.3f}" if acc else None)
            groups.append((row, NAME[row], vals, None, len(SUBST)))
        _bars(ax, M, groups, chosen[task])
        _panel(ax, k, title)
        ax.set_xlabel(f"Relative drop in accuracy ({_pct()})\n" + r"($\leftarrow$ smaller is better)")
    for k, (ax, task, title) in enumerate(((axes[1, 0], "top", "Top tagging"),
                                           (axes[1, 1], "qg", "Quark/gluon tagging")), start=2):
        groups = []
        for row in rows:
            vals = {}
            for s in SUBST:
                v = [x for m in runs[row] if (x := _bench_r30(M, fits, fill, m, task, s)) is not None]
                if v:
                    vals[s] = (v, _fmt3(float(np.mean(v))))
            mv = [x for m in runs[row] if (x := M._mlp_val(mlp, m, task, "r30")) is not None] \
                if row in ("L188", INIT_BN) else None
            groups.append((row, NAME[row], vals, mv, len(SUBST)))
        if kin["top"] is not None:
            # nothing is missing for it: one bar, no substitution
            groups.append((KIN, NAME[KIN], {"none": ([kin[task]], _fmt3(kin[task]))} if kin[task] is not None
                           else {}, None, 1))
        _bars(ax, M, groups, chosen[task])
        _panel(ax, k, title)
        ax.set_xlabel(r"$R_{30}$ of the linear probe" + "\n" + r"($\rightarrow$ better)")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    g = "0.45"
    handles = [Patch(facecolor=g, edgecolor=g), Patch(facecolor="white", edgecolor=g, hatch=SUBST_HATCH["mean"]),
               Patch(facecolor="white", edgecolor=g, hatch=SUBST_HATCH["draws"]),
               Patch(facecolor="none", edgecolor="black", linewidth=1.4),
               Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none", markersize=4),
               Line2D([], [], color="black", marker="o", linestyle="none", markersize=1.8)]
    labels = [f"Missing inputs: {SUBST_NAME[s].lower()}" for s in SUBST] + \
             ["Chosen on validation", "MLP probe, inputs set to zero", "One run"]
    _legend(fig, handles, labels, ncol=3)
    return _save(fig, "F5_missing_inputs", outdir)


# ------------------------------------------------------------------ F7, A10
def ft_series(M, per, runs, key: str, met: str, row: str) -> dict:
    """{training jets: [fine-tuned value of each run of row]} (mlst_results._ft: one fine-tuning per run)."""
    if not per.get(key):
        return {}
    out = {}
    for n in M._sizes(per, key):
        v = [x for m in runs[row] if (x := M._ft(per, key, m, n, met)) is not None]
        if v:
            out[n] = v
    return out


def scratch_series(M, per, key: str, met: str) -> dict:
    """{training jets: [value of each from-scratch seed]}."""
    return {n: [r[met] for r in recs if r.get(met) is not None]
            for n, recs in sorted(per.get(key, {}).get(M.SCRATCH, {}).items()) if recs}


def probe_series(M, I, runs, ds: str, met: str, row: str, chosen: dict) -> dict:
    """{training jets: [linear-probe value of each run of row]}: JetClass-II and JetClass from the
    v4 fits; top and quark/gluon R30 under the substitution chosen on validation (zero while it is
    pending), as the tables (mlst_results._bench_rec, _bench_sizes)."""
    fits, fill = I["fits"], I["fill"]
    if ds in ("top", "qg"):
        sizes = M._bench_sizes(fits, fill, chosen, ds)
        get = lambda m, n: M._rej(M._bench_rec(fits, fill, chosen, m, ds, n), "r30")
    else:
        sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
        get = lambda m, n: (lambda r: None if r is None else r.get(met))(M._probe_rec(fits, m, ds, n))
    out = {}
    for n in sizes:
        v = [x for m in runs[row] if (x := get(m, n)) is not None]
        if v:
            out[int(n)] = v
    return out


def _line(ax, series: dict, kw: dict, label: str, dodge: float = 1.0) -> dict:
    """Mean over runs with markers and a band of the SD (range for two runs); {N: mean}."""
    xs = sorted(series)
    if not xs:
        return {}
    st = [F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    lo, hi = mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st])
    x = np.array(xs, float) * dodge
    ax.plot(x, mean, label=label, markersize=3.5, **kw)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.18, linewidth=0)
    return dict(zip(xs, mean))


def _size_axis(ax, met: str, ys: list) -> None:
    ax.set_xscale("log")
    if met == "r30" and ys and min(ys) > 0:
        lo, hi = min(ys) / 1.15, max(ys) * 1.15
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        F._log_axis(ax.yaxis, lo, hi)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))


def _band_values(series: list) -> list:
    return [x for s in series for v in s.values() for x in v]


def figure_f7(M, runs, I, outdir) -> list:
    per = I["per"]
    chosen, _ = M.chosen_substitutions(I["fill"], runs)
    lk = look(M, "L188")
    kw_ft = dict(color=lk["color"], marker="o", linestyle="-")
    kw_lp = dict(color=lk["color"], marker="o", linestyle="--", markerfacecolor="white")
    kw_sc = dict(color="black", marker="s", linestyle="--", markerfacecolor="white")
    got = []
    for key, ds, met, title, ylab in SIZE_PANELS:
        ft, sc = ft_series(M, per, runs, key, met, "L188"), scratch_series(M, per, key, met)
        lp = probe_series(M, I, runs, ds, met, "L188", chosen) if I["fits"] else {}
        for what, s in (("fine-tuning", ft), ("from scratch", sc), ("linear probe", lp)):
            if not s:
                absent(f"F7 {title}: {what}")
        got.append((key, met, title, ylab, ft, lp, sc))
    if not any(ft or lp or sc for *_, ft, lp, sc in got):
        raise Skip("fine-tuning read-outs (ft_mlst/, budget/budget_metrics.json) and linear probes (fits_v4)")
    fig = _fig(FULL, 13.0 * CM)
    outer = fig.add_gridspec(2, 2)
    for k, (key, met, title, ylab, ft, lp, sc) in enumerate(got):
        inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=(3, 1))
        ax = fig.add_subplot(inner[0])
        rx = fig.add_subplot(inner[1], sharex=ax)
        _panel(ax, k, title)
        if not (ft or lp or sc):
            ax.set_axis_off()
            rx.set_axis_off()
            continue
        mft = _line(ax, ft, kw_ft, "188 classes, fine-tuned")
        _line(ax, lp, kw_lp, "188 classes, linear probe")
        msc = _line(ax, sc, kw_sc, NAME[SCRATCH])
        ns = [n for n in sorted(mft) if n in msc]
        if ns:
            rx.plot(ns, [mft[n] / msc[n] for n in ns], markersize=3.5, **kw_ft)
        rx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        _size_axis(ax, met, _band_values([ft, lp, sc]))
        rx.set_xscale("log")
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.tick_params(labelbottom=False)
        ax.set_ylabel(ylab)
        rx.set_ylabel("Fine-tuned /\nfrom scratch")
        rx.set_xlabel("Number of training jets")
    seen = {}
    for a in fig.axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    _legend(fig, list(seen.values()), list(seen), ncol=3)
    return _save(fig, "F7_finetune_vs_size", outdir)


def figure_a10(M, runs, I, outdir) -> list:
    """Linear probes against the number of training jets, every label set and the randomly
    initialized network with BatchNorm recomputed (F7's axes)."""
    if not I["fits"]:
        raise Skip("linear probes (linprobe/fits_v4)")
    chosen, _ = M.chosen_substitutions(I["fill"], runs)
    fig = _fig(FULL, 11.0 * CM)
    axes = fig.subplots(2, 2)
    rows = LEVELS + (INIT_BN,)
    for k, (ax, (key, ds, met, title, ylab)) in enumerate(zip(axes.flat, SIZE_PANELS)):
        ser = []
        for i, row in enumerate(rows):
            s = probe_series(M, I, runs, ds, met, row, chosen)
            lk = look(M, row)
            _line(ax, s, dict(color=lk["color"], marker=lk["marker"], linestyle=lk["linestyle"]),
                  NAME[row].replace("\n", " "), dodge=10 ** (0.02 * (i - 2)))
            ser.append(s)
        _panel(ax, k, title)
        if not any(ser):
            absent(f"A10 {title}: linear probes")
            ax.set_axis_off()
            continue
        _size_axis(ax, met, _band_values(ser))
        ax.set_ylabel(ylab)
        ax.set_xlabel("Number of training jets")
    seen = {}
    for a in fig.axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    _legend(fig, list(seen.values()), list(seen), ncol=3)
    return _save(fig, "A10_probes_vs_size", outdir)


# ------------------------------------------------------------------ F8a, A17
def _block(atlas: dict, m: str, est: str, group: str):
    try:
        return atlas[m]["estimators"][est][group]
    except (KeyError, TypeError):
        return None


def _groomed(atlas: dict, group: str):
    """The groomed mass's block (the same in every file, mlst_results._groomed checks it)."""
    return next((b for m in sorted(atlas) if (b := _block(atlas, m, "groomed", group)) is not None), None)


def _arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in x])


def _frac(h: dict) -> np.ndarray:
    """Fraction of all jets per histogram bin (the jets outside the edges count in the total)."""
    c = np.asarray(h["counts"], float)
    return c / (c.sum() + h["underflow"] + h["overflow"])


def mass_hist(ax, rx, M, runs, atlas, group: str, tag: str) -> bool:
    """The response m/m_gen of the groomed mass and of the two mass models' regression outputs (mean of
    each model's runs, the runs' range as a band), the ratio to the groomed mass below."""
    g = _groomed(atlas, group)
    if g is None:
        absent(f"{tag}: groomed-mass histogram, {group} jets (head/atlas_res)")
        return False
    e = np.asarray(g["hist"]["edges"], float)
    gr = _frac(g["hist"])
    ax.stairs(gr, e, color=style.GROOMED_STYLE["color"], linewidth=1.0)
    shown = [gr]
    for a in M.MASS:
        hs = np.array([_frac(b["hist"]) for m in runs[a] if (b := _block(atlas, m, "regressed", group))])
        if not len(hs):
            absent(f"{tag}: regression output of {a}, {group} jets")
            continue
        lk, mean = look(M, a), hs.mean(axis=0)
        ax.stairs(mean, e, color=lk["color"], linewidth=1.1, linestyle=MK.MASS_DASH[a])
        if len(hs) > 1:
            ax.stairs(hs.max(axis=0), e, baseline=hs.min(axis=0), fill=True, color=lk["color"], alpha=0.25, linewidth=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.stairs(np.where(gr > 0, mean / gr, np.nan), e, baseline=None, color=lk["color"], linewidth=1.0,
                      linestyle=MK.MASS_DASH[a])
        shown.append(mean)
    rx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    # the range from the data: the bins where any histogram holds at least 0.2 % of its largest bin
    on = np.flatnonzero(np.any([s >= 2e-3 * s.max() for s in shown if s.max() > 0], axis=0))
    if len(on):
        ax.set_xlim(e[on[0]], e[on[-1] + 1])
    rx.set_ylim(bottom=0)
    rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Fraction of jets")
    rx.set_ylabel("Ratio to\ngroomed")
    rx.set_xlabel(r"Response $m/m^{\mathrm{gen}}_{\mathrm{SD}}$")
    return True


def mass_bins(axs, M, runs, atlas, group: str, axis: str, xlabel: str, tag: str) -> bool:
    """In bins of axis ("vs_pt", "vs_mgen"): the ATLAS resolution (top), its ratio to the groomed
    mass's (middle) and the median response (bottom). The groomed mass: one value per bin with its
    jet-level bootstrap SD (resolution_sd, median_sd) and the bin width; each mass model: the mean of
    its runs (pentagon, offset within the bin) and the runs' range as a band. Bins below the jet
    minimum carry null and are not drawn."""
    a0, a1, a2 = axs
    g = _groomed(atlas, group)
    if g is None or axis not in g:
        absent(f"{tag}: groomed mass {axis}, {group} jets (head/atlas_res)")
        return False
    b = g[axis]
    e = np.asarray(b["edges"], float)
    mid, half = (e[:-1] + e[1:]) / 2, np.diff(e) / 2
    rg = _arr(b["resolution"])
    kw = dict(color=style.GROOMED_STYLE["color"], linestyle="none", linewidth=0.9, capsize=0)
    a0.errorbar(mid, rg, xerr=half, yerr=_arr(b["resolution_sd"]), **kw)
    a2.errorbar(mid, _arr(b["median"]), xerr=half, yerr=_arr(b["median_sd"]), **kw)
    for j, a in enumerate(M.MASS):
        bl = [x[axis] for m in runs[a] if (x := _block(atlas, m, "regressed", group)) and axis in x]
        if not bl:
            absent(f"{tag}: regression output of {a}, {group} jets {axis}")
            continue
        lk = look(M, a)
        off = mid + (-1 if j == 0 else 1) * 0.3 * half
        res = np.array([_arr(x["resolution"]) for x in bl])
        for axx, v in ((a0, res), (a1, res / rg), (a2, np.array([_arr(x["median"]) for x in bl]))):
            lo, hi = v.min(axis=0), v.max(axis=0)
            if len(v) > 1:
                for i in np.flatnonzero(np.isfinite(lo) & np.isfinite(hi)):
                    axx.fill_between([e[i], e[i + 1]], lo[i], hi[i], color=lk["color"], alpha=0.25, linewidth=0)
            axx.plot(off, v.mean(axis=0), linestyle="none", marker="p", markersize=4, color=lk["color"],
                     markerfacecolor=MK.MASS_FACE[a] or lk["color"], zorder=3)
    for axx in (a1, a2):
        axx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        axx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    a0.set_xlim(e[0], e[-1])
    for axx in (a0, a1):
        axx.tick_params(labelbottom=False)
    a0.set_ylabel("Resolution")
    a1.set_ylabel("Ratio to\ngroomed")
    a2.set_ylabel("Median\nresponse")
    a2.set_xlabel(xlabel)
    return True


BIN_AXES = (("vs_pt", r"Jet $p_{\mathrm{T}}$ [GeV]"), ("vs_mgen", r"Generator soft-drop mass [GeV]"))


def _mass_legend(fig, M) -> None:
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=style.GROOMED_STYLE["color"], linewidth=1.0)]
    for a in M.MASS:
        lk = look(M, a)
        handles.append(Line2D([], [], color=lk["color"], linewidth=1.1, linestyle=MK.MASS_DASH[a], marker="p",
                              markersize=4, markerfacecolor=MK.MASS_FACE[a] or lk["color"]))
    _legend(fig, handles, ["Groomed (soft-drop) mass", *(NAME[a] for a in M.MASS)], ncol=3)


def _mass_bin_columns(fig, spec, M, runs, atlas, group, tag, k0) -> int:
    """One column of mass_bins per BIN_AXES entry in spec (a gridspec cell per column); panel labels from k0."""
    drawn = 0
    for j, (axis, xlabel) in enumerate(BIN_AXES):
        inner = spec[j].subgridspec(3, 1, height_ratios=(2.2, 1, 1.3))
        a0 = fig.add_subplot(inner[0])
        axs = (a0, fig.add_subplot(inner[1], sharex=a0), fig.add_subplot(inner[2], sharex=a0))
        if mass_bins(axs, M, runs, atlas, group, axis, xlabel, tag):
            drawn += 1
        else:
            for a in axs:
                a.set_axis_off()
        _panel(a0, k0 + j)
    return drawn


def figure_f8a(M, runs, I, outdir) -> list:
    atlas = I["atlas"]
    if not atlas or _groomed(atlas, "all") is None:
        raise Skip("ATLAS mass resolution (head/atlas_res)")
    fig = _fig(FULL, 10.5 * CM)
    gs = fig.add_gridspec(1, 3)
    inner = gs[0].subgridspec(2, 1, height_ratios=(3, 1))
    ax = fig.add_subplot(inner[0])
    rx = fig.add_subplot(inner[1], sharex=ax)
    mass_hist(ax, rx, M, runs, atlas, "all", "F8a")
    _panel(ax, 0)
    _mass_bin_columns(fig, [gs[1], gs[2]], M, runs, atlas, "all", "F8a", 1)
    _mass_legend(fig, M)
    return _save(fig, "F8a_mass_regression", outdir)


def figure_a17_response(M, runs, I, outdir) -> list:
    atlas = I["atlas"]
    if not atlas:
        raise Skip("ATLAS mass resolution (head/atlas_res)")
    fig = _fig(FULL, 7.5 * CM)
    gs = fig.add_gridspec(1, 2)
    drawn = 0
    for k, (group, title) in enumerate((("resonant", "Resonant jets"), ("qcd", "QCD jets"))):
        inner = gs[k].subgridspec(2, 1, height_ratios=(3, 1))
        ax = fig.add_subplot(inner[0])
        rx = fig.add_subplot(inner[1], sharex=ax)
        drawn += mass_hist(ax, rx, M, runs, atlas, group, "A17")
        _panel(ax, k, title)
    if not drawn:
        raise Skip("ATLAS mass resolution by jet group (head/atlas_res)")
    _mass_legend(fig, M)
    return _save(fig, "A17_mass_response_groups", outdir)


def _a17_bins(M, runs, I, outdir, group: str, title: str, name: str) -> list:
    atlas = I["atlas"]
    if not atlas or _groomed(atlas, group) is None:
        raise Skip(f"ATLAS mass resolution, {title} (head/atlas_res)")
    fig = _fig(FULL, 9.5 * CM)
    gs = fig.add_gridspec(1, 2)
    if not _mass_bin_columns(fig, [gs[0], gs[1]], M, runs, atlas, group, f"A17 {title}", 0):
        raise Skip(f"binned ATLAS mass resolution, {title} (head/atlas_res)")
    _mass_legend(fig, M)
    return _save(fig, name, outdir)


def figure_a17_bins_resonant(M, runs, I, outdir) -> list:
    return _a17_bins(M, runs, I, outdir, "resonant", "resonant jets", "A17_mass_bins_resonant")


def figure_a17_bins_qcd(M, runs, I, outdir) -> list:
    return _a17_bins(M, runs, I, outdir, "qcd", "QCD jets", "A17_mass_bins_qcd")


# ------------------------------------------------------------------ F8b
def sculpt_floor(M, d: dict) -> tuple:
    """(efficiencies, mean, SD) of the Jensen-Shannon distance a random selection of as many QCD jets
    shows, on a run's curve: mlst_results._floor_distance (8 n ln2 JSD ~ chi^2 with k - 1 degrees of
    freedom, finite-population corrected) and the SD of the same chi distribution."""
    k = int(np.count_nonzero(d["hist_before"]))
    eff, mean, sd = [], [], []
    for p in d["curve"]:
        n, nref = p["n_pass"], d.get("n_qcd")
        if n <= 0:
            continue
        eff.append(p["eff"])
        mean.append(M._floor_distance(k, n, nref))
        sd.append(M._floor_sd(k, n, nref))
    return np.array(eff), np.array(mean), np.array(sd)


def figure_f8b(M, runs, I, outdir) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sc = I["sculpt"]
    ref = next((m for m in runs["L188"] if m in sc), None)
    if ref is None:
        raise Skip("mass sculpting curves (head/curves_v2/*_sculpting_v2.json)")
    fig = _fig(FULL, 13.5 * CM)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.15, 1))
    top = outer[0].subgridspec(2, 3, height_ratios=(3, 1))
    ratio_axes, hi = [], 0.0
    for j, a in enumerate(F8B_SHOWN):
        ax = fig.add_subplot(top[0, j])
        rx = fig.add_subplot(top[1, j], sharex=ax)
        ms = [m for m in runs[a] if m in sc]
        _panel(ax, j, NAME[a])
        if not ms:
            absent(f"F8b: sculpting curves of {a}")
            ax.set_axis_off()
            rx.set_axis_off()
            continue
        edges = np.asarray(sc[ms[0]]["bins_jet_sdmass"], float)
        before = np.asarray(sc[ms[0]]["hist_before"], float)
        before = before / before.sum()
        ax.stairs(before, edges, color="0.5", fill=True, alpha=0.35, linewidth=0)
        lk = look(M, a)
        for wp, ls in WPS:
            after = np.mean([np.asarray(sc[m]["working_points"][wp]["hist_after"], float)
                             / np.sum(sc[m]["working_points"][wp]["hist_after"]) for m in ms], axis=0)
            ax.stairs(after, edges, baseline=None, color=lk["color"], linewidth=1.1, linestyle=ls)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            hi = max(hi, float(np.nanmax(r)))
            rx.stairs(r, edges, baseline=None, color=lk["color"], linewidth=1.0, linestyle=ls)
        rx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelbottom=False)
        rx.set_xlabel(r"$m_{\mathrm{SD}}$ [GeV]")
        if j == 0:
            ax.set_ylabel("Fraction of jets")
            rx.set_ylabel("After cut /\nall")
        ratio_axes.append(rx)
    for rx in ratio_axes:
        rx.set_ylim(0, 1.1 * hi)
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    bottom = outer[1].subgridspec(1, 3)
    ax = fig.add_subplot(bottom[0, :2])
    eff, fmean, fsd = sculpt_floor(M, sc[ref])
    ax.fill_between(eff, np.maximum(fmean - fsd, 1e-12), fmean + fsd, color="0.5", alpha=0.3, linewidth=0)
    ax.plot(eff, fmean, color="0.4", linewidth=1.0)
    ys = list(fmean - fsd) + list(fmean + fsd)
    handles = [(Patch(facecolor="0.5", alpha=0.3), Line2D([], [], color="0.4", linewidth=1.0))]
    labels = ["Random selection, $\\pm 1$ SD"]
    for i, row in enumerate(F.SCULPT_ROWS):
        have = [sc[m]["curve"] for m in runs[row] if m in sc]
        if not have:
            absent(f"F8b: sculpting curve of {row}")
            continue
        per = np.array([[c["js_distance"] for c in h] for h in have])
        e = np.array([c["eff"] for c in have[0]]) * 1.04 ** (i - 2.5)   # small offsets so markers do not overlap
        lk = look(M, row)
        mean, lo, hi_ = per.mean(axis=0), per.min(axis=0), per.max(axis=0)
        if len(per) > 2:
            s = per.std(axis=0, ddof=1)
            lo, hi_ = mean - s, mean + s
        ax.plot(e, mean, color=lk["color"], marker=lk["marker"], markersize=3.5, linestyle=lk["linestyle"],
                linewidth=1.0)
        if len(per) > 1:
            ax.fill_between(e, lo, hi_, color=lk["color"], alpha=0.2, linewidth=0)
        ys += list(lo) + list(hi_)
        handles.append(Line2D([], [], color=lk["color"], marker=lk["marker"], markersize=3.5,
                              linestyle=lk["linestyle"]))
        labels.append(NAME[row])
    ax.set_xscale("log")
    ax.set_yscale("log")
    pos = [y for y in ys if y > 0]
    lo, hi_ = min(pos) / 1.3, max(pos) * 1.3
    ax.set_ylim(lo, hi_)
    F._log_axis(ax.yaxis, lo, hi_)
    ax.set_xlabel(r"QCD efficiency $\varepsilon_{\mathrm{QCD}}$")
    ax.set_ylabel("Jensen-Shannon distance")
    _panel(ax, 3)
    lax = fig.add_subplot(bottom[0, 2])
    lax.set_axis_off()
    lax.legend(handles, labels, loc="center left", frameon=False, fontsize="small")
    wp_handles = [Patch(facecolor="0.5", alpha=0.35)] + [Line2D([], [], color="black", linewidth=1.1, linestyle=ls)
                                                         for _, ls in WPS]
    _legend(fig, wp_handles, ["All QCD jets", f"$\\varepsilon_{{\\mathrm{{QCD}}}} = 10{_pct()}$",
                              f"$\\varepsilon_{{\\mathrm{{QCD}}}} = 1{_pct()}$"], ncol=3)
    return _save(fig, "F8b_sculpting", outdir)


# ------------------------------------------------------------------ F9
EPS_S_GRID = np.linspace(0.05, 1.0, 96)


def xbx_curves(XR, rs: list) -> tuple:
    """(eps_B, eps_S/sqrt(eps_B) of each run) on the points with at least XR.MIN_BKG_PASS QCD jets
    passing, and (eps_S grid, rejection 1/eps_B of each run interpolated in ln there, NaN outside)."""
    x, keep = XR._kept(rs[0])
    y = np.array([np.asarray(r["curve"]["eps_s_over_sqrt_eps_b"], float)[keep] for r in rs])
    rej = []
    for r in rs:
        xb, kp = XR._kept(r)
        es = np.asarray(r["curve"]["eps_s_over_sqrt_eps_b"], float)[kp] * np.sqrt(xb)
        o = np.argsort(es)
        rej.append(np.exp(np.interp(EPS_S_GRID, es[o], -np.log(xb[o]), left=np.nan, right=np.nan)))
    return x, y, np.array(rej)


MARK_SLOTS, MARKS = 5, 6


def _marks(x, log: bool, slot: int) -> list:
    """Indices of MARKS points evenly spaced along x (in log10 when log), shifted by slot/MARK_SLOTS of
    a step so that the label sets' markers on nearby curves do not sit on top of each other."""
    u = np.log10(np.asarray(x, float)) if log else np.asarray(x, float)
    lo, hi = np.nanmin(u), np.nanmax(u)
    at = lo + (hi - lo) / MARKS * (np.arange(MARKS) + (slot + 0.5) / MARK_SLOTS)
    return sorted({int(np.nanargmin(np.abs(u - a))) for a in at})


def _mean_band(ax, x, v, kw: dict, label: str | None, slot: int | None = None) -> None:
    """Mean over runs with the SD (three or more) or range (two) as a band. With slot, the curve's
    marker (kw) is drawn at a few points (_marks), so that curves differ by more than colour."""
    mean = v.mean(axis=0)
    if slot is not None:
        kw = dict(kw, markevery=_marks(x, ax.get_xscale() == "log", slot))
    ax.plot(x, mean, label=label, **kw)
    if len(v) > 1:
        lo, hi = (v.min(axis=0), v.max(axis=0)) if len(v) == 2 else \
            (mean - v.std(axis=0, ddof=1), mean + v.std(axis=0, ddof=1))
        ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.18, linewidth=0)


def figure_f9(M, runs, I, outdir) -> list:
    xfits = I["xbx"]
    XR = _xr()
    have = [(s, t) for s, t in XR.SIGNALS if any(s in v for v in xfits.values())]
    if not have:
        raise Skip("X->bb and X->bs fits (xbx/)")
    fig = _fig(FULL, 12.0 * CM)
    axes = fig.subplots(2, len(have), squeeze=False)
    for j, (s, title) in enumerate(have):
        top, bot = axes[0, j], axes[1, j]
        top.set_xscale("log")           # before the curves: their markers are spaced on this scale
        got = lambda row, name, s=s: [r for m in runs[row] if (r := XR.readout(xfits.get(m, {}).get(s, {}), name))]
        # each label set its marker (line style: the method), so that colour is not the only cue
        mk = lambda row, **kw: dict(color=look(M, row)["color"], marker=look(M, row)["marker"], markersize=3,
                                    linewidth=1.1, **kw)
        series = [(got(row, "head"), mk(row, linestyle="-"), f"{NAME[row]}, transfer", LEVELS.index(row))
                  for row in LEVELS]
        if s == "xbb":
            # Sophon's direct discriminant (Eq. 4), only where X->bb is a class of its own
            series += [(got(row, "zero_shot_xbb"), mk(row, linestyle=":", markerfacecolor="white"),
                        f"{NAME[row]}, direct discriminant", LEVELS.index(row))
                       for row in ("L188", "L162") if XR.direct_ok(M, row)]
        else:
            series.append((got("L188", "zero_shot_xbq"), dict(style.NODE_CONTROL_STYLE, linewidth=1.1),
                           r"188 classes, $X\to bq$ output node", None))
            scr = I["xbx_scratch"].get(s, [])
            if not scr:
                absent(f"F9 {s}: training from scratch (xbx_scratch/)")
            series.append((scr, dict(color="black", linestyle="--", linewidth=1.1), NAME[SCRATCH], None))
        for rs, kw, label, slot in series:
            if not rs:
                absent(f"F9 {s}: {label}")
                continue
            x, y, rej = xbx_curves(XR, rs)
            _mean_band(top, x, y, kw, label, slot)
            ok = np.isfinite(rej).all(axis=0)
            if ok.any():
                _mean_band(bot, EPS_S_GRID[ok], rej[:, ok], kw, None, slot)
        top.axvline(1e-3, color="0.6", linewidth=0.7, linestyle="-.", zorder=0)
        for e in (0.3, 0.5):
            bot.axvline(e, color="0.6", linewidth=0.7, linestyle="-.", zorder=0)
        top.set_xlim(1e-4, 1)
        top.set_ylim(bottom=0)
        top.set_xlabel(r"QCD efficiency $\varepsilon_B$")
        bot.set_yscale("log")
        bot.set_xlim(0, 1)
        bot.set_xlabel(r"Signal efficiency $\varepsilon_S$")
        _panel(top, j, title)
        _panel(bot, len(have) + j, title)
    axes[0, 0].set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")
    axes[1, 0].set_ylabel(r"Rejection $1/\varepsilon_B$")
    seen = {}
    for a in axes[0]:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    _legend(fig, list(seen.values()), list(seen), ncol=3)
    return _save(fig, "F9_xbx", outdir)


# ------------------------------------------------------------------ appendix ROC curves
def figure_a_roc(M, runs, I, outdir) -> list:
    """Rejection against signal efficiency of full fine-tuning on the full top and quark/gluon
    training sets (Pythia and Herwig test jets), the four label sets, the kinematic-only model on top
    and training from scratch; below, the ratio to training from scratch."""
    cells = I["ft_curves"]
    got = [(ds, t, F.roc_rows(M, runs, cells, ds, ROC_ROWS)) for ds, t in F.ROC_BENCH]
    got = [g for g in got if "L188" in g[2] and "SCRATCH" in g[2]]
    if not got:
        raise Skip("fine-tuning ROC curves (curves/full_best70_leg_*.json, full_ref_v100_bench_leg_*.json)")
    fig = _fig(FULL, 9.0 * CM)
    axes = fig.subplots(2, len(got), sharex="col", squeeze=False, gridspec_kw={"height_ratios": (3, 1.2)})
    for j, (ds, title, rr) in enumerate(got):
        top, bot = axes[0, j], axes[1, j]
        kin = cells.get(M.KIN_MODEL, {}).get(M.PRIMARY, {}).get(ds, {}).get("roc")
        if kin:
            rr = {**rr, KIN: (kin["eps_s"], [kin["rejection"]])}
        elif ds == "top":
            absent("A_roc: fine-tuning of the 188-class model with kinematic inputs only on top tagging")
        eps0 = np.asarray(rr["SCRATCH"][0])
        ref = np.asarray(rr["SCRATCH"][1], float)[:, eps0 >= ROC_MIN_EPS].mean(axis=0)
        for row, (eps, per) in rr.items():
            eps = np.asarray(eps)
            keep = eps >= ROC_MIN_EPS
            v = np.asarray(per, float)[:, keep]
            key = SCRATCH if row == "SCRATCH" else row
            lk = look(M, key)
            kw = dict(color=lk["color"], linestyle=lk["linestyle"], linewidth=1.1)
            # the label sets and the kinematic-only model carry their marker: colour is not the only cue
            slot = LEVELS.index(key) if key in LEVELS else len(LEVELS) if key == KIN else None
            if slot is not None:
                kw.update(marker=lk["marker"], markersize=3)
            _mean_band(top, eps[keep], v, kw, NAME[key].replace("\n", " "), slot)
            if len(eps[keep]) == len(ref):
                _mean_band(bot, eps[keep], v / ref, kw, None, slot)
        top.set_yscale("log")
        bot.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        bot.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        bot.set_xlabel(r"Signal efficiency $\varepsilon_S$")
        top.tick_params(labelbottom=False)
        _panel(top, j, title)
    axes[0, 0].set_ylabel(r"Rejection $1/\varepsilon_B$")
    axes[1, 0].set_ylabel("Ratio to\nfrom scratch")
    seen = {}
    for a in axes[0]:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    _legend(fig, list(seen.values()), list(seen), ncol=3)
    return _save(fig, "A_roc_benchmarks", outdir)


FIGURES = (figure_f2, figure_f3, figure_f4, figure_f5, figure_f7, figure_f8a, figure_f8b, figure_f9,
           figure_a_roc, figure_a10, figure_a17_response, figure_a17_bins_resonant, figure_a17_bins_qcd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=pathlib.Path, default=OUTDIR)
    a = ap.parse_args(argv)
    M = F._mlst()
    runs = M._paper(REPO)
    I = inputs(M)
    drawn, skipped, failed = [], [], []
    for f in FIGURES:
        try:
            paths = f(M, runs, I, a.outdir)
        except Skip as e:
            print(f"{f.__name__}: skipped, no input: {e}")
            skipped.append(f.__name__)
            continue
        except Exception as e:  # one broken figure must not hide the others
            traceback.print_exc()
            print(f"{f.__name__}: FAILED {type(e).__name__}: {e}")
            failed.append(f.__name__)
            continue
        print(f"{f.__name__}: " + ", ".join(str(p) for p in paths))
        drawn.append(f.__name__)
    print(f"\ndrawn {len(drawn)}, skipped {len(skipped)}, failed {len(failed)}"
          + (f" ({', '.join(failed)})" if failed else ""))
    return 0 if drawn else 1


if __name__ == "__main__":
    raise SystemExit(main())
