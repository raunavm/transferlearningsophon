#!/usr/bin/env python3
"""The MLST paper's figures, final (docs/FIGURE_PLAN.md as amended by the PI decisions of
2026-11-02 and the ROC redesign of the same day): drawn at their printed size, 15 cm wide with the
text between 8 and 12 pt, in Sophon's style (mlst_style.use_style(final=True)), into figures/mlst/
as PDF (vector) and PNG. Every number is selected by the generator's own functions
(mlst_results.py, mlst_v4.py) from the pulled data under experiments/FIGS/data/v2/, so a figure and
the table or macro beside it cannot differ. The JetClass-II probes are the class-balanced v4 fits
(mlst_v4.load_fits), never fits_v3. 1 - AUC is on no axis or colour bar (PI). Nothing but data is
drawn: what a caption must say is said in the caption.

Every ROC panel (F4, F6J, F6, A6J, A3R, F9 left) is background rejection 1/eps_B (log) against the
signal efficiency eps_S, from the curves_v2 files (mlst_v4.load_roc): each curve is drawn only where
at least mlst_v4.JC1_MIN_BKG background jets pass (the black dashed line N_B / that number marks the
limit) and only when at least mlst_results.ROC_MIN_POINTS points survive, as the mean over the
pretraining runs at fixed eps_S where every run survives, with the SD (range for two runs) as a
band; below, the ratio to the 188-class model, run by run against the
188-class run of the same index for the label sets, mass models, flavor partitions and the randomly
initialized network (it holds the initial weights of the 188-class run of its index), and against
the 188-class mean for training from scratch (its seed k trains on subset k) and the kinematic-only
model (one run).

Main text
  F2   accuracy (JetClass-II 188-class task on the class-balanced test set; JetClass) and R30 (top;
       quark/gluon, Pythia) against the number of pretraining classes (Hong et al., Fig. 5 form): the
       hierarchy joined (linear probe dashed, filled markers; fine-tuning solid, open markers); the
       random partitions and the partitions with controlled flavor boundaries at 17
  F3   per class (linear probe), the AUC difference to the 188-class model of the same index in
       percentage points (diverging, centred at 0), eleven columns; filled dot: the class shares its
       group with a decay that differs only in quark flavor, ring: only in lepton flavor; dash: AUC
       above the 188-class model's; hatched: fewer than THIN_TEST class-balanced test jets
  F4   ROC per class pair (linear probes): X->bb vs cc, X->YY->bbqq vs ccqq, X->bc vs bq+cs and
       X->ee vs mumu; the label sets and the three flavor partitions
  F5   missing input features set to zero, to the training mean or to random draws: relative drop
       in JetClass-II accuracy with the top (quark/gluon) inputs substituted, and R30 of the linear
       probe on top and quark/gluon; the substitution chosen on validation
       (mlst_results.choose_substitution) outlined in black; MLP probe on the zero bars
  F6J  JetClass after fine-tuning at 10^6 jets, ROC of each signal class against QCD (ParT Eq. 2):
       the label sets and training from scratch
  F6   top and quark/gluon (Pythia, Herwig) after fine-tuning on the full sets: the label sets,
       training from scratch and, on top, the 188-class model with kinematic inputs only
  F7   fine-tuning, linear probe and training from scratch against the number of training jets,
       188-class model, with the ratio of fine-tuning to training from scratch
  F8a  auxiliary mass regression (head/atlas_res): response histogram, ATLAS resolution (half the
       15.9-84.1 % inter-quantile range over the median) and median response against jet pT and
       generator soft-drop mass; groomed mass (filled) with its bootstrap bars, the two runs of each
       mass model as their range; the ratio to the groomed mass only in bins holding at least
       MASS_MIN_JETS groomed jets
  F8b  CMS JME-25-001 pair: QCD soft-drop mass after cuts at 10 % and 1 % QCD efficiency, and the
       Jensen-Shannon distance (base 2) against the QCD efficiency with the random-selection floor
  F9   X->bb and X->bs, one row each: the ROC with the ratio to the 188-class transfer head, and
       eps_S / sqrt(eps_B) against eps_B
Appendix
  A6J  F6J's layout for the linear probes, the randomly initialized network with BatchNorm
       recomputed as the reference curve
  A3R  JetClass-II one-vs-rest ROC of the linear probes for the classes mlst_results.jc2_roc_classes
       selects (most hurt at 17 classes, least changed by any merge), label sets only
  A10  linear probes against the number of training jets, every label set and the randomly
       initialized network (BatchNorm recomputed)
  A17  the mass regression on resonant and on QCD jets: response histograms, and resolution and
       median response against jet pT and generator mass

A figure or panel whose input is not pulled yet is skipped with one line naming the input; main()
removes a skipped figure's older files from the output folder (FILES), so the paper prints its
pending marker rather than a superseded render. main() prints what it drew and skipped and exits 0
when at least one figure was drawn.

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
# the smallest text MLST allows: tick labels and legends of the style (8 pt)
SMALL = style.SOPHON_RC["legend.fontsize"]
SUBST = ("zero", "mean", "draws")
SUBST_NAME = {"zero": "Set to zero", "mean": "Training mean", "draws": "Random draws"}
SUBST_HATCH = {"zero": "", "mean": "//////", "draws": "....", "none": ""}
# F2 panels: (fine-tuning task, its full training set, probe data set, metric, title, y label); the
# fine-tuned JetClass-II point at 10^6 jets, the probes' size (ROC interface decision 9.9)
F2_PANELS = (("jc2", 1_000_000, "jc2_188", "accuracy", "JetClass-II", "Accuracy"),
             ("jc1", 1_000_000, "jc1", "accuracy", "JetClass", "Accuracy"),
             ("top", 1_200_000, "top", "r30", "Top tagging", r"$R_{30}$"),
             ("qg", 1_600_000, "qg", "r30", "Quark/gluon tagging, Pythia", r"$R_{30}$"))
# F7 and A10 panels: (fine-tuning task, probe data set, metric, title, y label)
SIZE_PANELS = (("jc2", "jc2_188", "accuracy", "JetClass-II", "Accuracy"),
               ("jc1", "jc1", "accuracy", "JetClass", "Accuracy"),
               ("top", "top", "r30", "Top tagging", r"$R_{30}$"),
               ("qg", "qg", "r30", "Quark/gluon tagging", r"$R_{30}$"))
F8B_SHOWN = ("L188", "R16_Q1", "R16_Q1_MASS")
WPS = (("0.1", "-"), ("0.01", ":"))
# ROC panels: main and ratio axes' heights; the configurations compared with the 188-class mean,
# not run by run: training from scratch (seed k is no pretraining run k) and the kinematic-only model
# (one run). The randomly initialized network init(bn)-sK pairs with the 188-class run sK, whose
# initial weights it holds (scripts/build_extract_jobs.py V2_INIT_RUNS)
ROC_HEIGHTS = (3, 1.2)
UNPAIRED = (SCRATCH, KIN)
X_EPS = r"Signal efficiency ($\varepsilon_S$)"
Y_REJ = r"Background rejection ($1/\varepsilon_B$)"
Y_REJ_RATIO = "Ratio to\n188 classes"
Y_REJ_SHORT = r"Rejection ($1/\varepsilon_B$)"     # A3R's panels are shorter than Y_REJ
ROC_ZOOM_FROM = 0.9     # every curve of a panel starts above this: the axis shows only what they reach


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
            "roc": M.V4.load_roc(DATA)}


def _xr():
    return F._module("experiments/XBS/xbx_report.py", "xbx_report")


# ------------------------------------------------------------------ drawing helpers
def look(M, row: str) -> dict:
    """Colour, line style and marker of a configuration (style.CONFIG, the same in every figure): the
    label sets solid, dashed, dash-dot and dotted, a variant of a model (mass output, kinematic inputs
    only) a long-dash pattern in its colour, the flavor partitions long dashes, long dash-dot and
    dash-dot-dot, training from scratch black dash-dot-dot with open squares, the randomly initialized
    network grey long dashes; the random partitions pooled as "RAND"."""
    return dict(style.CONFIG["RAND" if row in M.RAND else row])


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
    """The panel letter inside the centred title (Sophon's titles); the text cites panels by letter."""
    ax.set_title(f"({'abcdefghijklmnop'[k]}) {text}".rstrip())


def _legend(fig, handles, labels, ncol: int, **kw) -> None:
    """One framed legend (the style's box) above the panels."""
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, **kw)


def _fmt3(x: float) -> str:
    """A value at a bar end: three significant figures, integers from 100."""
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _point(ax, M, x, row, vals, fill="full", size=5.5) -> list:
    """The mean as a marker with its spread (mlst_figures._spread: SD for three or more runs, the
    range for two), each run a small white-edged dot on it. Returns the values the marker, its bar
    and the dots reach."""
    lk = look(M, row)
    mean, el, eh = F._spread(vals)
    ax.errorbar(x, mean, yerr=None if el is None else [[el], [eh]], marker=lk["marker"], markersize=size,
                color=lk["color"], linestyle="none", elinewidth=1.0, capsize=0, zorder=3, **MK._face(fill, lk["color"]))
    ax.plot([x] * len(vals), vals, linestyle="none", marker="o", markersize=2.0, color=lk["color"],
            markeredgecolor="white", markeredgewidth=0.3, zorder=4)
    return list(vals) + [mean - (el or 0), mean + (eh or 0)]


def _vals(runs, row: str, value) -> list:
    """value of each run of row, the runs without one left out."""
    return [x for m in runs[row] if (x := value(m)) is not None]


def _hierarchy(ax, M, runs, value, ft: bool = False) -> list:
    """The four levels' points and the line through their means: the linear probe at x, dashed, filled
    markers; fine-tuning (ft) beside it with open markers, solid. Returns every plotted value."""
    xs, means, ys = [], [], []
    for row, x in X_OF.items():
        v = _vals(runs, row, value)
        if not v:
            continue
        xx = x * (FT_X.get(row, FT_X_OTHER) if ft else 1.0)
        ys += _point(ax, M, xx, row, v, fill="none" if ft else "full")
        means.append(float(np.mean(v)))
        xs.append(xx)
    if xs:
        ax.plot(xs, means, color="black", linestyle="-" if ft else "--", linewidth=0.9, zorder=1)
    return ys


def _controls(ax, M, runs, value, ft: bool, tag: str) -> list:
    """The random partitions (mean with its spread, each partition's runs as dots beside it) and the
    three partitions with controlled flavor boundaries at 17, filled for the linear probe and open for
    fine-tuning; a control without a value is left out (logged for the probes). Returns every value."""
    fill = "none" if ft else "full"
    per = [_vals(runs, a, value) for a in M.RAND]
    ys = [x for p in per for x in p]
    if ys:
        mean, el, eh = F._spread(ys)
        ax.errorbar(RAND_X, mean, yerr=None if el is None else [[el], [eh]], marker="o", markersize=5.5,
                    color=style.RAND_COLOUR, linestyle="none", elinewidth=1.0, capsize=0, zorder=3,
                    **MK._face(fill, style.RAND_COLOUR))
        for k, p in enumerate(per):         # one x per partition, its runs on it
            ax.plot([RAND_DOTS * DOT_STEP ** (k - 2)] * len(p), p, "o", color=style.RAND_COLOUR, markersize=2.5,
                    markeredgecolor="white", markeredgewidth=0.3, zorder=4)
        ys += [mean - (el or 0), mean + (eh or 0)]
    elif not ft:
        absent(f"{tag}: linear probes of the random partitions")
    for a in M.FLAV:
        v = _vals(runs, a, value)
        if v:
            ys += _point(ax, M, FLAV_X[a], a, v, fill=fill)
        elif not ft:
            absent(f"{tag}: linear probes of {a}")
    return ys


def _classes_axis(ax, ys: list, xlabel: bool = True, log: bool = False) -> None:
    """Number of pretraining classes on a log axis (ticks 17, 43, 162, 188); the metric on a linear
    axis zoomed to the data (8 % of the range on either side), or with log on a log axis (a rejection:
    the probe and the fine-tuned models lie more than a decade apart on top tagging)."""
    ax.set_xscale("log")
    ax.set_xlim(12, 260)
    ax.xaxis.set_major_locator(ticker.FixedLocator([17, 43, 162, 188]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # 162 and 188 sit 0.065 decades apart: their labels lean away from each other
    for lab, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left")):
        lab.set_horizontalalignment(ha)
    lo, hi = min(ys), max(ys)
    if log and lo > 0:
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.15, hi * 1.15)
        _log_ticks(ax.yaxis, lo / 1.15, hi * 1.15)
    else:
        pad = 0.08 * (hi - lo) or 0.08 * abs(hi) or 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    if xlabel:
        ax.set_xlabel(r"Number of classes in pretraining ($N_{\mathrm{out}}$)")


def _has(fits: dict, ds: str) -> bool:
    return any(ck.get(ds) for f in fits.values() for ck in f.get("cells", {}).values())


def _f2_probe(M, fits, fill, chosen, ds: str, met: str):
    """A run's linear-probe value at the largest training set, or None when the task has no fits:
    accuracy on JetClass-II and JetClass, R30 on top and quark/gluon under the substitution chosen on
    validation (zero while it is pending), as the tables (mlst_results._bench_rec)."""
    if ds in ("top", "qg"):
        sizes = M._bench_sizes(fits, fill, chosen, ds)
        if not sizes:
            return None
        return lambda m: M._rej(M._bench_rec(fits, fill, chosen, m, ds, sizes[-1]), met)
    if not _has(fits, ds):
        return None
    n = M._largest(fits, ds)
    return lambda m: (lambda r: None if r is None else r.get(met))(M._probe_rec(fits, m, ds, n))


def _f2_ft(M, per, key: str, n: int, met: str):
    """A run's fine-tuned value on n training jets (its one fine-tuning), or None when the task has
    no fine-tuning at n."""
    if not per.get(key) or n not in M._sizes(per, key):
        return None
    return lambda m: M._ft(per, key, m, n, met)


# ------------------------------------------------------------------ F2
def figure_f2(M, runs, I, outdir) -> list:
    from matplotlib.lines import Line2D
    fits, fill, per = I["fits"], I["fill"], I["per"]
    chosen, _ = M.chosen_substitutions(fill, runs)
    values = [(_f2_probe(M, fits, fill, chosen, ds, met), _f2_ft(M, per, key, n, met))
              for key, n, ds, met, *_ in F2_PANELS]
    if not any(p or f for p, f in values):
        raise Skip("linear probes (linprobe/fits_v4) and fine-tuning read-outs (ft_mlst/, budget/)")
    fig = _fig(FULL, 11.0 * CM)
    axes = fig.subplots(2, 2)
    for k, (ax, (key, n, ds, met, title, ylab), (probe, ft)) in enumerate(zip(axes.flat, F2_PANELS, values)):
        _panel(ax, k, title)
        ys = []
        if probe:
            ys += _hierarchy(ax, M, runs, probe)
            ys += _controls(ax, M, runs, probe, False, f"F2 {title}")
        else:
            absent(f"F2 {title}: linear probes")
        if ft:
            ys += _hierarchy(ax, M, runs, ft, ft=True)
            ys += _controls(ax, M, runs, ft, True, f"F2 {title}")
        elif key == "jc2":
            absent("F2: fine-tuning on class-balanced JetClass-II (budget/budget_metrics.json, leg_jc2b)")
        else:
            absent(f"F2 {title}: fine-tuning on {n:,} jets")
        if not ys:
            ax.set_axis_off()
            continue
        _classes_axis(ax, ys, xlabel=k >= 2, log=met == "r30")
        if k % 2 == 0:
            ax.set_ylabel(ylab)
    grey = "0.45"
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=0.9, marker="o", markersize=4,
                      markerfacecolor=grey, markeredgecolor=grey),
               Line2D([], [], color="black", linestyle="-", linewidth=0.9, marker="o", markersize=4,
                      markerfacecolor="white", markeredgecolor=grey),
               Line2D([], [], color=style.RAND_COLOUR, marker="o", linestyle="none", markersize=4),
               *(Line2D([], [], color=look(M, a)["color"], marker=look(M, a)["marker"], linestyle="none",
                        markersize=4) for a in M.FLAV)]
    labels = ["Hierarchy, linear probe", "Hierarchy, fine-tuned", NAME["RAND"], *(NAME[a] for a in M.FLAV)]
    _legend(fig, handles, labels, ncol=3, handlelength=1.5, columnspacing=0.8)
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
    adjacent, then the QCD classes; (classes, columns) array of the mean over indices of the per-class
    AUC difference to the 188-class run of the same index in percentage points
    (mlst_results.per_class_auc_diff), NaN where absent)."""
    rows = M._class_rows(root)
    rows = [r for r in rows if not r["class_name"].startswith("label_QCD")] + \
           [r for r in rows if r["class_name"].startswith("label_QCD")]
    R = np.full((len(rows), len(MK.HEAT)), np.nan)
    for j, (arm, _) in enumerate(MK.HEAT):
        diff = M.per_class_auc_diff(fits, runs, arm)
        if not diff:
            absent(f"F3: per-class linear probes of {arm}")
        for i, r in enumerate(rows):
            v = diff.get(int(r["jet_label"]))
            if v:
                R[i, j] = float(np.mean(v))
    return rows, R


def _nice_up(x: float) -> float:
    """x rounded up to 1, 2 or 5 times a power of ten (1 for no positive x)."""
    if not x > 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(x))
    return next(k * e for k in (1, 2, 5, 10) if k * e >= x * (1 - 1e-12))


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
    # symmetric about 0 (negative: below the 188-class model, red), to the 99th percentile of |diff|
    L = _nice_up(float(np.percentile(np.abs(R[np.isfinite(R)]), 99)))
    fig = _fig(FULL, 21.0 * CM)
    ax = fig.add_subplot()
    ax.grid(False)
    cmap = colormaps["RdBu"].copy()
    cmap.set_bad("white")
    im = ax.imshow(np.insert(R, n_res, np.nan, axis=0), aspect="auto", cmap=cmap, norm=Normalize(-L, L),
                   interpolation="none")
    if counts:
        for i, r in enumerate(rows):
            if counts[int(r["jet_label"])] < M.THIN_TEST:
                ax.add_patch(Rectangle((-0.5, row(i) - 0.5), len(MK.HEAT), 1, fill=False, hatch="//////",
                                       edgecolor="0.3", linewidth=0))
    # marks, white on dark cells: filled dot (right third) quark-flavor sibling in the group, a ring at
    # the same place (around the dot when both) lepton-flavor sibling, dash (left third) AUC above the
    # 188-class model's (the sign in greyscale)
    ink = lambda i, j: "white" if np.isfinite(R[i, j]) and abs(R[i, j]) > 0.65 * L else "black"
    marks = {}
    for j, (arm, _) in enumerate(MK.HEAT):
        for i, r in enumerate(rows):
            qm, lm = merges.get(arm, {}).get(r["class_name"], (False, False))
            if qm:
                marks.setdefault(("q", ink(i, j)), []).append((j + 0.3, row(i)))
            if lm:
                marks.setdefault(("l", ink(i, j)), []).append((j + 0.3, row(i)))
            if R[i, j] > 0:
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
    ax.set_xticklabels([h for _, h in MK.HEAT], rotation=90)
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
    ax.set_yticklabels([labs[a] for a in blocks])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="both")
    cb.minorticks_off()
    cb.ax.grid(False)
    cb.set_label(r"$\Delta\mathrm{AUC}_c$ to the 188-class model (percentage points)")
    handles = [Line2D([], [], marker="o", color="black", linestyle="none", markersize=2.5),
               Line2D([], [], marker="o", color="black", markerfacecolor="none", linestyle="none", markersize=3,
                      markeredgewidth=0.6),
               Line2D([], [], marker="_", color="black", linestyle="none", markersize=5, markeredgewidth=0.9)]
    labels = ["Merged with a quark-flavor sibling", "Merged with a lepton-flavor sibling",
              "AUC above the 188-class model"]
    if counts:
        handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="//////", linewidth=0))
        labels.append(f"Fewer than {M.THIN_TEST:,} test jets")
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    return _save(fig, "F3_per_class", outdir)


# ------------------------------------------------------------------ ROC panels
def _band(V: np.ndarray) -> tuple:
    """(mean, low, high) over the runs (rows) of V: the SD for three or more runs, the range for two
    (mlst_figures._spread), no band (None) for one."""
    mean = V.mean(axis=0)
    if len(V) > 2:
        sd = V.std(axis=0, ddof=1)
        return mean, mean - sd, mean + sd
    if len(V) == 2:
        return mean, V.min(axis=0), V.max(axis=0)
    return mean, None, None


def _mean_band(ax, x, v, kw: dict, label: str | None) -> list:
    """Mean over runs as a line with _band as a band (alpha 0.2, no edge); returns the values drawn."""
    mean, lo, hi = _band(np.asarray(v, float))
    ax.plot(x, mean, label=label, **kw)
    if lo is None:
        return list(mean)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.2, linewidth=0)
    return list(lo) + list(hi)


def _ratio(rx, M, eps, ref: dict, rows: list) -> bool:
    """The ratio panel below a ROC: ref {model: rejection} of the 188-class runs, rows [(configuration,
    {model: rejection}, line style)]. Run by run against the 188-class run of the same index
    (mlst_results._idx; the randomly initialized network included), except the UNPAIRED
    configurations against the 188-class mean; drawn where every run and the reference survive and at
    least mlst_results.ROC_MIN_POINTS points do (the generator's 'not reached' rule, _roc_cut), the
    reference flat at 1 in its colour, y from the data (5 % padding, 1 included, no clipping). False,
    with nothing drawn, when the reference has fewer surviving points."""
    if not ref:
        return False
    R = np.array(list(ref.values()))
    ref_ok = np.isfinite(R).all(axis=0)
    if ref_ok.sum() < M.ROC_MIN_POINTS:
        return False
    ref_mean, by_idx = R.mean(axis=0), {M._idx(m): r for m, r in ref.items()}
    rx.plot(eps[ref_ok], np.ones(ref_ok.sum()), color=look(M, "L188")["color"])
    ys = [1.0]
    for row, got, kw in rows:
        if row == "L188":
            continue
        if row in UNPAIRED:
            Q = np.array([r / ref_mean for r in got.values()])
        else:
            Q = np.array([r / by_idx[M._idx(m)] for m, r in got.items() if M._idx(m) in by_idx])
        if not len(Q):
            continue
        ok = np.isfinite(Q).all(axis=0) & ref_ok
        if ok.sum() >= M.ROC_MIN_POINTS:
            ys += _mean_band(rx, eps[ok], Q[:, ok], kw, None)
    lo, hi = min(ys), max(ys)
    pad = 0.05 * (hi - lo) or 0.05
    rx.set_ylim(lo - pad, hi + pad)
    rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=RATIO_STEPS))
    return True


# ratio-panel ticks at steps of 1, 2, 2.5 or 5 x 10^k (matplotlib's default steps include 1.5,
# which gave 1.05 and 1.20 around a ratio of 1)
RATIO_STEPS = (1, 2, 2.5, 5, 10)


def _rej_axis(ax, ys: list) -> None:
    """Rejection on a log axis with 10^k labels over [min / 1.3, max x 1.3] of what is drawn; eps_S
    from 0 to 1."""
    pos = [y for y in ys if np.isfinite(y) and y > 0]
    ax.set_yscale("log")
    ax.set_ylim(min(pos) / 1.3, max(pos) * 1.3)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xlim(0, 1)


def _log_ticks(axis, lo: float, hi: float) -> None:
    """Ticks of a log axis over [lo, hi]: 10^k labels only (Sophon; matplotlib's log locator with
    LogFormatterSciNotation, as _rej_axis) when two or more powers of ten lie in the range; on a
    narrower span, a few round plain numbers (30, 40, 60, 100), where 10^k would leave one label or
    none and labelled minor ticks would read 6x10^1. Minor ticks are never labelled."""
    axis.set_minor_locator(ticker.LogLocator(subs="auto"))
    axis.set_minor_formatter(ticker.NullFormatter())
    if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) >= 1:
        axis.set_major_locator(ticker.LogLocator())
        axis.set_major_formatter(ticker.LogFormatterSciNotation())
        return
    vals = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    axis.set_major_locator(ticker.FixedLocator([v for v in vals if lo <= v <= hi]))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def roc_panel(ax, rx, M, fam: dict, key: str, configs: list) -> tuple[bool, list]:
    """One ROC panel (module docstring) of the curve key in a curves_v2 family {model: doc}; configs
    [(configuration, models)]. A curve is drawn only where at least mlst_results.ROC_MIN_POINTS grid
    points survive the cut in every run (fewer draw no visible line; the generator calls such a
    configuration 'not reached', _roc_cut). Returns (whether the ratio panel was drawn, the
    configurations with a curve file, which keep their legend entry even when no point survives the
    cut)."""
    got = {}
    for row, models in configs:
        c = {m: s[1] for m in models if m in fam and (s := M.V4.roc_series(fam[m], key)) is not None}
        if c:
            got[row] = c
    if not got:
        return False, []
    first = next(m for c in got.values() for m in c)
    eps = np.asarray(M.V4.roc_series(fam[first], key)[0], float)
    limit = M.V4.roc_limit(fam[first], key)
    ax.axhline(limit, **style.LIMIT_STYLE)
    ys, rows, starts = [limit], [], []
    for row, c in got.items():
        lk = look(M, row)
        kw = dict(color=lk["color"], linestyle=lk["linestyle"])
        c = {m: np.asarray(r, float) for m, r in c.items()}
        V = np.array(list(c.values()))
        ok = np.isfinite(V).all(axis=0)
        if ok.sum() >= M.ROC_MIN_POINTS:
            ys += _mean_band(ax, eps[ok], V[:, ok], kw, None)
            starts.append(eps[ok][0])
        rows.append((row, c, kw))
    if not starts:      # no configuration reaches two grid points: say so, an empty axis reads as missing data
        ys.append(limit / 100)
        ax.text(0.5, 0.45, f"No curve: fewer than\n{M.V4.JC1_MIN_BKG} background jets pass\nat all but at most\n"
                r"one $\varepsilon_S$ of the grid", transform=ax.transAxes, ha="center", va="center", fontsize=SMALL)
    _rej_axis(ax, ys)
    if starts and min(starts) >= ROC_ZOOM_FROM:
        ax.set_xlim(math.floor(min(starts) * 20) / 20, 1)
    ratio = _ratio(rx, M, eps, next((c for row, c, _ in rows if row == "L188"), {}), rows)
    return ratio, list(got)


def roc_figure(M, fig, panels: list, ncols: int, name: str, outdir, tag: str, ylabel: str = Y_REJ) -> list:
    """panels [(family {model: doc}, curve key, title, configs)] in rows of ncols, each a ROC panel with
    its ratio panel below (ROC_HEIGHTS). Shared labels: x on the bottom row, y on the left column; a
    panel whose 188-class reference has no surviving point has no ratio axes and carries the x labels
    itself. One framed legend above the panels: every configuration with a curve file and the limit
    line. Raises Skip when no panel has a curve."""
    from matplotlib.lines import Line2D
    nrows = math.ceil(len(panels) / ncols)
    outer = fig.add_gridspec(nrows, ncols)
    shown = {}
    for k, (fam, key, title, configs) in enumerate(panels):
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=ROC_HEIGHTS)
        ax = fig.add_subplot(inner[0])
        rx = fig.add_subplot(inner[1], sharex=ax)
        _panel(ax, k, title)
        ratio, rows = roc_panel(ax, rx, M, fam, key, configs)
        if not rows:
            absent(f"{tag}: ROC curves of {key}")
            ax.set_axis_off()
            rx.set_axis_off()
            continue
        shown.update(dict.fromkeys(rows))
        low = rx if ratio else ax
        if ratio:
            ax.tick_params(labelbottom=False)
        else:
            rx.set_axis_off()
        # the same ticks on every panel; a zoomed panel (roc_panel) in steps it can show
        step = 0.02 if ax.get_xlim()[0] >= ROC_ZOOM_FROM else 0.5 if ncols >= 4 else 0.25
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if k + ncols >= len(panels):
            low.set_xlabel(X_EPS)
        if k % ncols == 0:
            ax.set_ylabel(ylabel)
            if ratio:
                rx.set_ylabel(Y_REJ_RATIO)
    if not shown:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise Skip(f"{tag}: ROC curves (curves_v2)")
    # one shared legend above the panels for the configurations every panel is meant to show (also when
    # a curve is cut away entirely, as on X->ee vs X->mumu); a configuration only some panels show (the
    # kinematic-only model on top tagging) in a framed legend inside those panels (PI 2026-11-02)
    line = lambda r: Line2D([], [], color=look(M, r)["color"], linestyle=look(M, r)["linestyle"])
    everywhere = set.intersection(*(set(r for r, _ in p[3]) for p in panels))
    rows = [r for r in dict.fromkeys(r for p in panels for r, _ in p[3]) if r in shown and r in everywhere]
    for k, p in enumerate(panels):
        own = [r for r, _ in p[3] if r not in everywhere and r in shown]
        if own:
            fig.axes[2 * k].legend([line(r) for r in own], [NAME[r].replace("\n", " ") for r in own], loc="best")
    handles = [line(r) for r in rows] + [Line2D([], [], **style.LIMIT_STYLE)]
    labels = [NAME[r].replace("\n", " ") for r in rows] + [f"{M.V4.JC1_MIN_BKG} background jets pass"]
    _legend(fig, handles, labels, ncol=len(labels) if len(labels) <= 5 else 3)   # wider overflows the figure
    return _save(fig, name, outdir)


def _scratch_models(fam: dict) -> list:
    """The from-scratch trainings of a curves_v2 family (scratch-v2-s1, ...)."""
    return [m for m in sorted(fam) if m.startswith(SCRATCH + "-s")]


# ------------------------------------------------------------------ F4
def figure_f4(M, runs, I, outdir) -> list:
    fam = I["roc"].get("pairs")
    if not fam:
        raise Skip("ROC curves of the pair probes (curves_v2/pairs)")
    heading = {k: h for k, h, *_ in M.PAIRS}
    # the label sets and the flavor partitions; the random partitions and mass models are in tab:pairs only
    configs = [(row, list(runs[row])) for row in LEVELS + M.FLAV]
    fig = _fig(FULL, 13.0 * CM)
    return roc_figure(M, fig, [(fam, key, heading[key], configs) for key in M.ROC_PAIRS], 2, "F4_flavor_pairs",
                      outdir, "F4")


# ------------------------------------------------------------------ F6J, A6J, F6, A3R
def _jc1_roc(M, runs, I, outdir, family: str, ref: tuple, name: str, what: str) -> list:
    """F6J and A6J: one ROC panel per JetClass signal class against QCD (ParT Table 1's order), the
    label sets and the reference configuration ref = (configuration, models of the family)."""
    fam = I["roc"].get(family)
    if not fam:
        raise Skip(f"JetClass ROC curves, {what} (curves_v2/{family})")
    configs = [(row, list(runs[row])) for row in LEVELS] + [(ref[0], ref[1](fam))]
    fig = _fig(FULL, 18.5 * CM)
    return roc_figure(M, fig, [(fam, c, f"{tex} vs QCD", configs) for c, tex in M.V4.JC1_COLS], 3, name, outdir,
                      name.split("_")[0])


def figure_f6j(M, runs, I, outdir) -> list:
    return _jc1_roc(M, runs, I, outdir, "jc1_ft", (SCRATCH, _scratch_models), "F6J_roc_jetclass", "fine-tuning")


def figure_a6j(M, runs, I, outdir) -> list:
    return _jc1_roc(M, runs, I, outdir, "jc1_probe", (INIT_BN, lambda fam: list(M.INIT_BN_MODELS)),
                    "A6J_roc_jetclass_probe", "linear probes")


def figure_f6(M, runs, I, outdir) -> list:
    top, qg = I["roc"].get("top_ft", {}), I["roc"].get("qg_ft", {})
    if not (top or qg):
        raise Skip("top and quark/gluon fine-tuning ROC curves (curves_v2/top_ft, qg_ft)")
    base = lambda fam: [(row, list(runs[row])) for row in LEVELS] + [(SCRATCH, _scratch_models(fam))]
    kin = []
    if M.KIN_MODEL in top:
        kin = [(KIN, [M.KIN_MODEL])]
    else:
        absent("F6: fine-tuning of the 188-class model with kinematic inputs only on top tagging")
    panels = [(top, "top", "Top tagging", base(top) + kin), (qg, "qg", "Quark/gluon, Pythia", base(qg)),
              (qg, "qg_herwig", "Quark/gluon, Herwig", base(qg))]
    fig = _fig(FULL, 8.0 * CM)
    return roc_figure(M, fig, panels, 3, "F6_roc_benchmarks", outdir, "F6")


def figure_a3r(M, runs, I, outdir) -> list:
    fam, fits = I["roc"].get("jc2_ovr"), I["fits"]
    if not fam or not _has(fits, "jc2_188"):
        raise Skip("JetClass-II one-vs-rest ROC curves (curves_v2/jc2_ovr) and per-class probes (linprobe/fits_v4)")
    sel = M.jc2_roc_classes(fits, runs, REPO, I["census"])
    names, tex = {int(r["jet_label"]): r["class_name"] for r in M._class_rows(REPO)}, M._native_tex(REPO)
    configs = [(row, list(runs[row])) for row in LEVELS]
    # the curve keys of jc2_ovr are the native labels as strings
    panels = [(fam, str(c), tex(names[c]), configs) for c in list(sel["hurt"]) + list(sel["stable"])]
    fig = _fig(FULL, 19.0 * CM)
    return roc_figure(M, fig, panels, 4, "A3R_roc_jetclass2", outdir, "A3R", Y_REJ_SHORT)


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
    chosen on validation outlined in black; the MLP probe (open, black edge) on the zero bar, its value
    not printed; the value of the bar (the linear probe) at its end, or inside the bar, white, left of
    its dots, where the MLP marker would lie under the number (LABEL_SPAN). groups: [(row, label,
    {substitution: (values, text)}, MLP values, slots)]: a group takes `slots` rows and each
    substitution its own row of them whatever the group holds, so panels sharing their rows align."""
    y, ticks, ends, xs = 0.0, [], [], []
    for row, label, vals, mlpv, slots in groups:
        lk, first = look(M, row), y
        for s, (v, txt) in vals.items():
            y = first + (SUBST.index(s) if s in SUBST else 0)
            mean, el, eh = F._spread(v)
            solid = s in ("zero", "none")
            ax.barh(y, mean, height=0.78, color=lk["color"] if solid else "white", edgecolor=lk["color"],
                    hatch=SUBST_HATCH[s], linewidth=0.8, linestyle=":" if row == KIN else "-", zorder=2)
            if s == chosen:
                ax.barh(y, mean, height=0.78, fill=False, edgecolor="black", linewidth=1.4, zorder=3)
            ax.plot(v, [y] * len(v), "o", color="black", markersize=1.8, zorder=4)
            if el is not None:
                ax.errorbar(mean, y, xerr=[[el], [eh]], color="black", linewidth=0.8, capsize=0, zorder=4)
            end, start, mlp = max(max(v), mean + (eh or 0)), min(min(v), mean - (el or 0)), None
            xs += v
            if s == "zero" and mlpv:
                mm, ml, mh = F._spread(mlpv)
                ax.errorbar(mm, y, xerr=None if ml is None else [[ml], [mh]], marker="o", markersize=4.5,
                            color="black", markerfacecolor="white", linestyle="none", capsize=0, zorder=5)
                xs += mlpv + [mm + (mh or 0)]
                mlp = (mm - (ml or 0), mm + (mh or 0))    # the marker with its bar
            ends.append((end, start, mlp, y, txt))
        ticks.append((first + (slots - 1) / 2, label))
        y = first + slots + 0.5
    if not xs:
        ax.set_axis_off()
        return
    xmax = 1.45 * max(xs + [e for e, *_ in ends])
    for e, start, mlp, yy, txt in ends:
        if not txt:
            continue
        pad = 0.02 * xmax       # half the MLP marker
        if mlp is not None and mlp[1] + pad >= e and mlp[0] - pad <= e + LABEL_SPAN * xmax:
            if start - 0.015 * xmax >= LABEL_SPAN * xmax:
                # the MLP marker would lie under the number: the number inside the (solid) zero bar
                ax.text(start - 0.015 * xmax, yy, txt, va="center", ha="right", fontsize=SMALL, color="white",
                        zorder=6)
            else:
                # a zero bar too short to hold it: the number past the marker and its error bar
                ax.text(mlp[1] + pad + 0.015 * xmax, yy, txt, va="center", ha="left", fontsize=SMALL)
        else:
            ax.text(e + 0.015 * xmax, yy, txt, va="center", ha="left", fontsize=SMALL)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([lab for _, lab in ticks])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    ax.set_ylim(y - 1.0, -0.6)


# the width a bar-end number takes, as a fraction of the axis (five characters at 8 pt on a panel of
# about 5 cm, with the gap before it)
LABEL_SPAN = 0.16


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


def _line(ax, series: dict, kw: dict, label: str, dodge: float = 1.0) -> tuple[dict, list]:
    """Mean over runs with markers and a band of the SD (range for two runs); ({N: mean}, the values
    the band reaches), the second for the axis limits, as _point's."""
    xs = sorted(series)
    if not xs:
        return {}, []
    st = [F._spread(series[n]) for n in xs]
    mean = np.array([s[0] for s in st])
    lo, hi = mean - np.array([s[1] or 0 for s in st]), mean + np.array([s[2] or 0 for s in st])
    x = np.array(xs, float) * dodge
    ax.plot(x, mean, label=label, markersize=3.5, **kw)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.18, linewidth=0)
    return dict(zip(xs, mean)), list(lo) + list(hi)


def _size_axis(ax, met: str, ys: list) -> None:
    """Log x; R30 on a log y over [min / 1.15, max x 1.15] of ys (the runs and the bands' extents; a
    band reaching zero or below is cut at the axis), accuracy on a linear y."""
    ax.set_xscale("log")
    pos = [y for y in ys if y > 0]
    if met == "r30" and pos:
        lo, hi = min(pos) / 1.15, max(pos) * 1.15
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        _log_ticks(ax.yaxis, lo, hi)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))


def _band_values(series: list) -> list:
    return [x for s in series for v in s.values() for x in v]


def figure_f7(M, runs, I, outdir) -> list:
    per = I["per"]
    chosen, _ = M.chosen_substitutions(I["fill"], runs)
    lk = look(M, "L188")
    kw_ft = dict(color=lk["color"], marker="o", linestyle="-", markerfacecolor="white")
    kw_lp = dict(color=lk["color"], marker="o", linestyle="--")
    kw_sc = look(M, SCRATCH)
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
        mft, eft = _line(ax, ft, kw_ft, "188 classes, fine-tuned")
        _, elp = _line(ax, lp, kw_lp, "188 classes, linear probe")
        msc, esc = _line(ax, sc, kw_sc, NAME[SCRATCH])
        ns = [n for n in sorted(mft) if n in msc]
        r = [mft[n] / msc[n] for n in ns]
        if r:
            rx.plot(ns, r, markersize=3.5, **kw_ft)
            lo, hi = min(r + [1.0]), max(r + [1.0])
            pad = max(0.15 * (hi - lo), 0.02)
            rx.set_ylim(lo - pad, hi + pad)
        else:
            rx.set_ylim(0.9, 1.1)
            rx.tick_params(labelleft=False)
        rx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        _size_axis(ax, met, _band_values([ft, lp, sc]) + eft + elp + esc)
        rx.set_xscale("log")
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=RATIO_STEPS))
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
        ser, ext = [], []
        for i, row in enumerate(rows):
            s = probe_series(M, I, runs, ds, met, row, chosen)
            lk = look(M, row)
            ext += _line(ax, s, dict(color=lk["color"], marker=lk["marker"], linestyle=lk["linestyle"]),
                         NAME[row].replace("\n", " "), dodge=10 ** (0.02 * (i - 2)))[1]
            ser.append(s)
        _panel(ax, k, title)
        if not any(ser):
            absent(f"A10 {title}: linear probes")
            ax.set_axis_off()
            continue
        _size_axis(ax, met, _band_values(ser) + ext)
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
    """The response m/m_gen of the groomed mass (filled, Sophon's reference histogram) and of the two
    mass models' regression outputs (mean of each model's runs, the runs' range as a band), the ratio to
    the groomed mass below, only in bins holding at least MASS_MIN_JETS groomed jets (elsewhere the
    groomed fraction is too small for a ratio)."""
    g = _groomed(atlas, group)
    if g is None:
        absent(f"{tag}: groomed-mass histogram, {group} jets (head/atlas_res)")
        return False
    e = np.asarray(g["hist"]["edges"], float)
    gr = _frac(g["hist"])
    enough = np.asarray(g["hist"]["counts"], float) >= M.MASS_MIN_JETS
    ax.stairs(gr, e, fill=True, facecolor=style.PASTEL, edgecolor="black", linewidth=0.5)
    shown = [gr]
    for a in M.MASS:
        hs = np.array([_frac(b["hist"]) for m in runs[a] if (b := _block(atlas, m, "regressed", group))])
        if not len(hs):
            absent(f"{tag}: regression output of {a}, {group} jets")
            continue
        lk, mean = look(M, a), hs.mean(axis=0)
        ax.stairs(mean, e, color=lk["color"], linewidth=1.1, linestyle=lk["linestyle"])
        if len(hs) > 1:
            ax.stairs(hs.max(axis=0), e, baseline=hs.min(axis=0), fill=True, color=lk["color"], alpha=0.25, linewidth=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.stairs(np.where(enough, mean / gr, np.nan), e, baseline=None, color=lk["color"], linewidth=1.0,
                      linestyle=lk["linestyle"])
        shown.append(mean)
    rx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.grid(False)
    rx.grid(False)
    # the range from the data: the bins where any histogram holds at least 0.2 % of its largest bin
    on = np.flatnonzero(np.any([s >= 2e-3 * s.max() for s in shown if s.max() > 0], axis=0))
    if len(on):
        ax.set_xlim(e[on[0]], e[on[-1] + 1])
    rx.set_ylim(bottom=0)
    rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=RATIO_STEPS))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Fraction of jets", loc="top")
    rx.set_ylabel("Ratio to\nsoft-drop mass")
    rx.set_xlabel(r"Response ($r$)")
    return True


def mass_bins(axs, M, runs, atlas, group: str, axis: str, xlabel: str, tag: str) -> bool:
    """In bins of axis ("vs_pt", "vs_mgen"): the ATLAS resolution (top), its ratio to the groomed
    mass's (middle) and the median response (bottom). The groomed mass: one value per bin with its
    jet-level bootstrap SD (resolution_sd, median_sd) and the bin width; each mass model: the mean of
    its runs (its marker in style.CONFIG, filled or open as MK.MASS_FACE, offset within the bin) and the
    runs' range as a band. Bins below the jet minimum carry null and are not drawn."""
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
            axx.plot(off, v.mean(axis=0), linestyle="none", marker=lk["marker"], markersize=4, color=lk["color"],
                     markerfacecolor=MK.MASS_FACE[a] or lk["color"], zorder=3)
    for axx in (a1, a2):
        axx.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        axx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=RATIO_STEPS))
    a0.set_xlim(e[0], e[-1])
    for axx in (a0, a1):
        axx.tick_params(labelbottom=False)
    a0.set_ylabel(r"Resolution ($\sigma_r$)")
    a1.set_ylabel("Ratio to\nsoft-drop mass")
    a2.set_ylabel("Median\nresponse")
    a2.set_xlabel(xlabel)
    return True


# x labels at the right end of the axis (the style), short enough not to reach the next panel's
BIN_AXES = (("vs_pt", r"$p_{\mathrm{T}}$ [GeV]"), ("vs_mgen", r"$m^{\mathrm{gen}}_{\mathrm{SD}}$ [GeV]"))


def _mass_legend(fig, M) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=style.PASTEL, edgecolor="black", linewidth=0.5)]
    for a in M.MASS:
        lk = look(M, a)
        handles.append(Line2D([], [], color=lk["color"], linewidth=1.1, linestyle=lk["linestyle"],
                              marker=lk["marker"], markersize=4, markerfacecolor=MK.MASS_FACE[a] or lk["color"]))
    _legend(fig, handles, ["Reconstructed soft-drop mass", *(NAME[a] for a in M.MASS)], ncol=3)


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
        ax.stairs(before, edges, fill=True, facecolor=style.PASTEL, edgecolor="black", linewidth=0.5)
        ax.grid(False)
        rx.grid(False)
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
            ax.set_ylabel("Fraction of jets", loc="top")
            rx.set_ylabel("After cut /\nall")
        ratio_axes.append(rx)
    for rx in ratio_axes:
        rx.set_ylim(0, 1.1 * hi)
        rx.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=RATIO_STEPS))
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
    _log_ticks(ax.yaxis, lo, hi_)
    ax.set_xlabel(r"QCD efficiency ($\varepsilon_B$)")
    ax.set_ylabel(r"Jensen-Shannon distance ($d_{\mathrm{JS}}$)")
    _panel(ax, 3)
    lax = fig.add_subplot(bottom[0, 2])
    lax.set_axis_off()
    lax.legend(handles, labels, loc="center left")
    wp_handles = [Patch(facecolor=style.PASTEL, edgecolor="black", linewidth=0.5)] + \
        [Line2D([], [], color="black", linewidth=1.1, linestyle=ls) for _, ls in WPS]
    _legend(fig, wp_handles, ["All QCD jets", r"$\varepsilon_B = 10\%$", r"$\varepsilon_B = 1\%$"], ncol=3)
    return _save(fig, "F8b_sculpting", outdir)


# ------------------------------------------------------------------ F9
EPS_S_GRID = np.linspace(0.05, 1.0, 96)


def _kept(M, r: dict) -> tuple:
    """(eps_B, mask) of a readout's curve points with at least mlst_v4.JC1_MIN_BKG QCD jets passing,
    the rule of every ROC figure (xbx_report.MIN_BKG_PASS is that report's own figure's)."""
    x = np.asarray(r["curve"]["eps_b"], float)
    keep = x * r["n_b"] >= M.V4.JC1_MIN_BKG
    return x[keep], keep


def xbx_curves(M, rs: list) -> tuple:
    """(eps_B, eps_S/sqrt(eps_B) of each run) on the points _kept, and (eps_S grid, rejection 1/eps_B of
    each run there, NaN outside), eps_B interpolated linearly between the kept points. The points are
    xbx's eps_B grid (xbx.EPS_B_GRID, five per decade), not the ROC vertices of roc_points' curves, so a
    curve ends at the last grid point with at least JC1_MIN_BKG QCD jets, up to a grid step (10^0.2)
    below the limit line, not at the limit itself (the F9 caption says so)."""
    x, keep = _kept(M, rs[0])
    y = np.array([np.asarray(r["curve"]["eps_s_over_sqrt_eps_b"], float)[keep] for r in rs])
    rej = []
    for r in rs:
        xb, kp = _kept(M, r)
        es = np.asarray(r["curve"]["eps_s_over_sqrt_eps_b"], float)[kp] * np.sqrt(xb)
        o = np.argsort(es)
        rej.append(1 / np.interp(EPS_S_GRID, es[o], xb[o], left=np.nan, right=np.nan))
    return x, y, np.array(rej)


def figure_f9(M, runs, I, outdir) -> list:
    """One row per signal: left the ROC (common ROC panel, the ratio to the 188-class transfer head
    below), right eps_S/sqrt(eps_B) against eps_B (Sophon Fig. 2's axis)."""
    from matplotlib.lines import Line2D
    xfits = I["xbx"]
    XR = _xr()
    have = [(s, t) for s, t in XR.SIGNALS if any(s in v for v in xfits.values())]
    if not have:
        raise Skip("X->bb and X->bs fits (xbx/)")
    fig = _fig(FULL, 15.0 * CM)
    outer = fig.add_gridspec(len(have), 2)
    for i, (s, title) in enumerate(have):
        inner = outer[i, 0].subgridspec(2, 1, height_ratios=ROC_HEIGHTS)
        ax = fig.add_subplot(inner[0])
        rx = fig.add_subplot(inner[1], sharex=ax)
        sx = fig.add_subplot(outer[i, 1])
        got = lambda row, name, s=s: {m: r for m in runs[row] if (r := XR.readout(xfits.get(m, {}).get(s, {}), name))}
        kw = lambda row: dict(color=look(M, row)["color"], linestyle=look(M, row)["linestyle"])
        # (configuration for the ratio, {model: readout}, line, label): the transfer head of each label
        # set in its line style; Sophon's direct discriminant (Eq. 4) where X->bb is a class of its own;
        # for X->bs the X->bq output node (Sophon's control; style.NODE_CONTROL_STYLE, grey, a sparse
        # dash-dot-dot with crosses) and training from scratch. The
        # direct discriminant and the output node come from the pretraining runs themselves, so their
        # keys (neither "L188" nor UNPAIRED) pair them in the ratio with the 188-class transfer head of
        # the same run (_ratio); training from scratch against the 188-class mean
        series = [(row, got(row, "head"), kw(row), f"{NAME[row]}, transfer") for row in LEVELS]
        if s == "xbb":
            # the direct discriminant: its label set's line, thin, with open markers of that label set,
            # so it stays apart from the transfer heads without colour
            series += [(f"{row}:direct", got(row, "zero_shot_xbb"),
                        dict(kw(row), linewidth=0.8, marker=look(M, row)["marker"], markersize=3,
                             markerfacecolor="white", markevery=0.12), f"{NAME[row]}, direct discriminant")
                       for row in ("L188", "L162") if XR.direct_ok(M, row)]
        else:
            series.append(("L188:node", got("L188", "zero_shot_xbq"), dict(style.NODE_CONTROL_STYLE),
                           r"188 classes, $X\to bq$ output node"))
            scr = I["xbx_scratch"].get(s, [])
            if not scr:
                absent(f"F9 {s}: training from scratch (xbx_scratch/)")
            lk = look(M, SCRATCH)
            series.append((SCRATCH, {f"{SCRATCH}-s{k + 1}": r for k, r in enumerate(scr)},
                           dict(color=lk["color"], linestyle=lk["linestyle"]), NAME[SCRATCH]))
        ys, rows, ref, limit, xlo = [], [], {}, None, []
        for row, rs, line, label in series:
            if not rs:
                absent(f"F9 {s}: {label}")
                continue
            x, y, rej = xbx_curves(M, list(rs.values()))
            _mean_band(sx, x, y, line, label)
            xlo += [float(x.min())] if len(x) else []        # the smallest kept eps_B
            ok = np.isfinite(rej).all(axis=0)
            if ok.sum() >= M.ROC_MIN_POINTS:
                ys += _mean_band(ax, EPS_S_GRID[ok], rej[:, ok], line, None)
            rows.append((row, dict(zip(rs, rej)), line))
            if row == "L188":
                ref = dict(zip(rs, rej))
            limit = limit or next(iter(rs.values()))["n_b"] / M.V4.JC1_MIN_BKG
        if limit is not None:
            ax.axhline(limit, **style.LIMIT_STYLE)
            ys.append(limit)
            _rej_axis(ax, ys)
        ratio = _ratio(rx, M, EPS_S_GRID, ref, rows)
        low = rx if ratio else ax
        if ratio:
            ax.tick_params(labelbottom=False)
            rx.set_ylabel(Y_REJ_RATIO)
        else:
            rx.set_axis_off()
        sx.set_xscale("log")
        # from the smallest eps_B any curve of the row keeps (padded) to 1
        sx.set_xlim(min(xlo) / 1.3 if xlo else 1e-4, 1)
        sx.set_ylim(bottom=0)
        ax.set_ylabel(Y_REJ)
        sx.set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")
        if i == len(have) - 1:
            low.set_xlabel(X_EPS)
            sx.set_xlabel(r"QCD efficiency ($\varepsilon_B$)")
        ax.set_title(title)
        sx.set_title(title)
    # the transfer heads are in every row: one shared legend above; the curves of one row only (the
    # direct discriminant for X->bb, the output node and from scratch for X->bs) in a framed legend
    # inside that row's right panel (PI 2026-11-02)
    per_row = [dict(zip(*a.get_legend_handles_labels()[::-1])) for a in fig.axes[2::3]]
    common = set.intersection(*(set(d) for d in per_row)) if per_row else set()
    for a, d in zip(fig.axes[2::3], per_row):
        own = [lab for lab in d if lab not in common]
        if own:
            a.legend([d[lab] for lab in own], own, loc="best")
    seen = {lab: h for d in per_row for lab, h in d.items() if lab in common}
    seen[f"{M.V4.JC1_MIN_BKG} background jets pass"] = Line2D([], [], **style.LIMIT_STYLE)
    _legend(fig, list(seen.values()), list(seen), ncol=4)
    return _save(fig, "F9_xbx", outdir)


FIGURES = (figure_f2, figure_f3, figure_f4, figure_f5, figure_f6j, figure_f6, figure_f7, figure_f8a, figure_f8b,
           figure_f9, figure_a6j, figure_a3r, figure_a10, figure_a17_response, figure_a17_bins_resonant,
           figure_a17_bins_qcd)
# the files each figure saves (its _save names): removed from --outdir when the figure is skipped, so
# that main.tex's \paperfig prints \pending instead of an older render
FILES = {figure_f2: ("F2_label_granularity",), figure_f3: ("F3_per_class",), figure_f4: ("F4_flavor_pairs",),
         figure_f5: ("F5_missing_inputs",), figure_f6j: ("F6J_roc_jetclass",), figure_f6: ("F6_roc_benchmarks",),
         figure_f7: ("F7_finetune_vs_size",), figure_f8a: ("F8a_mass_regression",),
         figure_f8b: ("F8b_sculpting",), figure_f9: ("F9_xbx",), figure_a6j: ("A6J_roc_jetclass_probe",),
         figure_a3r: ("A3R_roc_jetclass2",), figure_a10: ("A10_probes_vs_size",),
         figure_a17_response: ("A17_mass_response_groups",),
         figure_a17_bins_resonant: ("A17_mass_bins_resonant",), figure_a17_bins_qcd: ("A17_mass_bins_qcd",)}


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
            for stem in FILES[f]:
                for ext in ("pdf", "png"):
                    old = a.outdir / f"{stem}.{ext}"
                    if old.exists():
                        old.unlink()
                        print(f"  removed {old} (superseded)")
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
