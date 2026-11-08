#!/usr/bin/env python3
"""The gallery's ROC mock-ups (docs/figstyle/PLAN.md): FJ, FJ-ctrl, FJ-s, FJ-app, FJ-ctrl-app, F4-mech, F4-mech-ft, F4-other, F6, F6-app, A3R, A3R-ft. Every ROC is
rejection (log, from 1) against signal efficiency (linear, 0 to 1), the mean over runs from
gallery_common.mean_over_runs, on style_v4's grid, with the y axis stopping where the curves stop so
they come in from the top of the frame, and the legend in the free lower left."""
from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, MultipleLocator, NullFormatter

from gallery_bands import rejection_rel_error
from gallery_common import Absent, M, P, Page, ROC, RUNS, mean_over_runs
import style_v5 as S

# rejection ticks in plain numbers (PI, 2026-11-07: every tick label at least 8 pt; a 10^n exponent prints at about 70%)
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
ROC_LEFT, ROC_WGAP = 1.5, 0.25     # cm: room for "100000" beside the y label; the right column's y labels are off
GRID = {"left": 0.1, "right": 0.965, "top": 0.92, "bottom": 0.06, "hspace": 0.1}
STACKED_GRID = {"top": 0.98, "bottom": 0.06, "hspace": 0.35}
FAMILIES = ("QCD", "Two-prong quarks and gluons", "Two-prong leptons", "Three-prong", "Four-prong",
            "Lepton-containing")


def family_of(class_name: str) -> str:
    """The topology family of a JetClass-II class (configs/labelmaps/rung_label_maps.v1.csv)."""
    name = class_name.removeprefix("label_")
    if name.startswith("QCD_"):
        return FAMILIES[0]
    if re.fullmatch(r"X_[bcsqg]{2}", name):
        return FAMILIES[1]
    if re.fullmatch(r"X_(ee|mm|tauhtau[emh])", name):
        return FAMILIES[2]
    m = re.fullmatch(r"X_YY_([bcsqg]+)(ee|mm|e|m|ev|mv|tauhtau[emh]|tau[emh]v)?", name)
    if m and m[2]:
        return FAMILIES[5]
    if m and len(m[1]) in (3, 4):
        return FAMILIES[len(m[1])]
    raise ValueError(f"no family for class {class_name!r}")


def _family(fam: str) -> dict:
    docs = ROC.get(fam)
    if not docs:
        raise Absent(f"no ROC curves under curves_v2/{fam}")
    return docs


def _axes_2x2(sharey: bool, wspace: float, heading: str, top: float = 1.0):
    """Round 4: four 6.5 cm panels (style_v5.grid_cm), the y label once per row, the x label under the lower row."""
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=top, left=ROC_LEFT, wgap=ROC_WGAP)
    if sharey:
        for ax in axes.flat[1:]:
            ax.sharey(axes.flat[0])
    for ax in axes.flat:
        _roc_axes(ax)
    S.labels(axes[0, 0], ylabel=S.Y_REJ)
    S.labels(axes[1, 0], ylabel=S.Y_REJ)
    for ax in axes[1]:
        S.labels(ax, xlabel=S.X_EPS)
    S.heading(fig, heading)
    return fig, axes


def _roc_axes(ax, x_step: float = 0.2) -> None:
    """style_v4's faint grid at major y ticks; x ticks every x_step, minor every 0.1."""
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    S.fine_grid(ax)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 else f"{v:.1f}"))   # no "0" in the corner
    ax.xaxis.set_minor_locator(MultipleLocator(0.1 if x_step > 0.1 else 0.05))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PLAIN)


def _right_labels_off(axes) -> None:
    """A row whose two panels show one y range gives its tick labels once, on the left (the plain-number labels leave
    no room between the columns); the left panel's "1.0" is left to the right panel's axis, which starts beside it."""
    for left, right in axes:
        left.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 or v > 1 - 1e-9 else f"{v:.1f}"))
        if np.allclose(left.get_ylim(), right.get_ylim()):
            right.tick_params(labelleft=False)
        else:
            raise SystemExit(f"FATAL: {right.get_title()!r}: its own y range needs its tick labels, with no room for them")


def _y_range(axes) -> None:
    """From 1 to where the highest curve stops, or the decade just above it, so curves leave through the frame."""
    highest = max(float(line.get_ydata().max()) for ax in axes for line in ax.lines)
    decade = 10.0 ** np.ceil(np.log10(highest))
    axes[0].set_ylim(1, decade if decade / highest < 1.3 else 1.1 * highest)


def _legend(ax, **kw):
    kw.setdefault("loc", "lower left")
    return S.legend(ax, handlelength=1.2, handletextpad=0.4, columnspacing=0.4, **kw)


def _plot(ax, s, **kw) -> None:
    """A mean curve from mean_over_runs; none where no run, or a single grid point, survives the cut."""
    if s is not None and len(s[0]) >= 2:
        ax.plot(s[0], s[1], **kw)


def _jc1_curves(ax, docs: dict, models: list, auc_in_legend: bool) -> float:
    """Ten class-against-the-rest curves, solid (every ROC figure, PI 2026-11-07); returns the macro AUC."""
    aucs = []
    for c in S.JC1_CLASSES:
        s = mean_over_runs(docs, models, c)
        aucs.append(s[2])
        _plot(ax, s, color=S.JC1_COLOUR[c], linewidth=S.LW_CURVE,
              label=f"{S.JC1_TEX[c]} ({s[2]:.3f})" if auc_in_legend else S.JC1_TEX[c])
    return float(np.mean(aucs))


def _class_handles(aucs: dict | None = None) -> list:
    """One handle per JetClass class, in the class colours; with aucs, the AUC after the name."""
    return [Line2D([], [], color=S.JC1_COLOUR[c], linewidth=S.LW_CURVE,
                   label=(f"{S.JC1_TEX[c]} ({_auc(aucs[c])})") if aucs else S.JC1_TEX[c]) for c in S.JC1_CLASSES]


def _auc(v: float) -> str:
    """Three decimals; above 0.9995, where that would print 1.000, ">0.999"."""
    return ">0.999" if v >= 0.9995 else f"{v:.3f}"


def _jc1_one(path: pathlib.Path, models: list, heading: str) -> None:
    """One panel, the class AUCs in a legend beside it on the right (PI, 2026-11-07): not above, where it took a
    tall band, and not inside, where ten curves leave no empty corner. Full width, as the other figures."""
    w = 7.3
    h = 0.85 * w
    top = 0.9
    fig = S.figure_cm(S.PRINT_W, top + h + 1.0)
    ax = S.axes_cm(fig, 1.5, top, w, h)
    _roc_axes(ax, x_step=0.2)
    docs = _family("jc1_ft_ovr")
    aucs = {c: mean_over_runs(docs, models, c)[2] for c in S.JC1_CLASSES}
    _jc1_curves(ax, docs, models, auc_in_legend=False)
    _y_range([ax])
    S.labels(ax, S.X_EPS, S.Y_REJ)
    S.heading(fig, f"{heading}, macro AUC {np.mean(list(aucs.values())):.3f}")
    S.legend(ax, handles=_class_handles(aucs), loc="center left", bbox_to_anchor=(1.03, 0.5), handlelength=1.6,
             title="Signal class (AUC)", title_fontsize=S.LEGEND_PT5)
    S.save(fig, path)


def _jc1_four(path: pathlib.Path, family: str, heading: str, panels: list, legend: bool = True) -> None:
    """panels: (models, title). One class legend above the panels, the same in every JetClass figure."""
    docs = _family(family)
    scratch = _family("jc1_ft_ovr")
    reference = np.mean([mean_over_runs(scratch, P._scratch_models(scratch), c)[2] for c in S.JC1_CLASSES])
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=2.35, left=ROC_LEFT, wgap=ROC_WGAP)
    for ax in axes.flat[1:]:
        ax.sharey(axes.flat[0])
    for ax in axes.flat:
        _roc_axes(ax)
    for row in axes:
        S.labels(row[0], ylabel=S.Y_REJ)
        row[1].tick_params(labelleft=False)      # one y range: its labels once, clear of the "1.0" beside them
    for ax in axes[1]:
        S.labels(ax, xlabel=S.X_EPS)
    S.heading(fig, f"{heading}; fully trained from scratch, macro AUC {reference:.3f}")
    for ax, (models, name) in zip(axes.flat, panels):
        auc = _jc1_curves(ax, docs, models, auc_in_legend=False)
        S.title(ax, f"{name}, macro AUC {auc:.3f}")
    _y_range(list(axes.flat))
    _right_labels_off(axes)
    S.fig_legend(fig, _class_handles(), top_cm=0.6, ncol=5)
    S.letter_panels(axes.flat)
    S.save(fig, path)


def _label_sets() -> list:
    return [(RUNS[level], S.NAME[level]) for level in S.LEVELS]


def _controls() -> list:
    """The three controlled partitions, and the five random partitions together (macro AUC within 3e-4)."""
    return [(RUNS[a], S.NAME[a]) for a in S.CONTROLS] + [([m for r in M.RAND for m in RUNS[r]], "Five random partitions")]


def draw_fj(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", "Fine-tuned on $10^6$ JetClass jets, per pretraining label set", _label_sets())


def draw_fj_ctrl(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", "Fine-tuned on $10^6$ JetClass jets, 17-class random partitions", _controls())


def draw_fj_s(path: pathlib.Path) -> None:
    models = P._scratch_models(_family("jc1_ft_ovr"))
    if not models:
        raise Absent("no from-scratch model in curves_v2/jc1_ft_ovr")
    _jc1_one(path, models, "Trained from scratch on $10^6$ JetClass jets")


def draw_fj_app(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_probe_ovr", "Linear probes on $10^6$ JetClass jets, per pretraining label set", _label_sets(),
              legend=False)     # probe curves lie lower: a legend would cover Z->qq


def draw_fj_ctrl_app(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_probe_ovr", "Linear probes on $10^6$ JetClass jets, 17-class random partitions",
              _controls(), legend=False)


F4_YMAX = 1e4          # one y range on all four panels (PI 2026-11-05)
PAIR_LABEL_SETS = tuple(S.LEVELS[::-1])              # the 188-class curve last, so it shows where the 162-class one lies
PAIR_BOUNDARIES = tuple(S.CONTROLS) + ("L188",)       # the three flavor boundaries, 188 classes for reference


def _pairs(path: pathlib.Path, family: str, panels: list, heading: str) -> None:
    """panels: (models, pair key, title), up to four in a 2x2, each model a solid curve in its colour. X->ee vs X->mumu
    of the linear probes is read from every test jet of the pair (pairs_sib2) where that exists. Every panel runs from
    1 to F4_YMAX, and a model whose runs keep fewer than 20 background jets at every efficiency is named in the panel
    as complete separation. One legend above the panels, off the curves, for every model drawn as a curve."""
    docs = _family(family)
    order = [S.LEGEND[a] for a in S.LEVELS + tuple(S.CONTROLS)]
    shown = {a for rows, *_ in panels for a in rows}
    fig, axes = _axes_2x2(sharey=False, wspace=0.2, heading=heading, top=1.9 if len(shown) <= 4 else 2.35)
    handles = {}
    for ax, (rows, key, name) in zip(axes.flat, panels):
        d = ROC["pairs_sib2"] if family == "pairs" and key == "ee_vs_mm" and ROC.get("pairs_sib2") else docs
        separated = []
        for a in rows:
            s = mean_over_runs(d, RUNS[a], key)
            _plot(ax, s, **S.look(a), linewidth=S.LW_CURVE, label=S.LEGEND[a])
            if s is not None and len(s[0]) < 2:
                separated.append(S.LEGEND[a])
            elif s is not None:          # a legend entry only for a curve drawn in some panel
                handles.setdefault(S.LEGEND[a], Line2D([], [], **S.look(a), linewidth=S.LW_CURVE, label=S.LEGEND[a]))
        if separated:
            separated.sort(key=order.index)
            words = (" and ".join(w.removesuffix(" classes") for w in separated) + " classes"
                     if all(w.endswith(" classes") for w in separated) else ", ".join(separated))
            ax.text(0.97, 0.95, "Complete separation:\n" + words, transform=ax.transAxes, ha="right",
                    va="top", fontsize=S.LEGEND_PT5)     # no frame, as the legends (style_v4)
        S.title(ax, name)
    for ax in axes.flat:
        ax.set_ylim(1, F4_YMAX)
    _right_labels_off(axes)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    if len(panels) == 3:        # the lower-right cell is empty: (b) takes the x label, and (c) its "1.0" back
        S.labels(axes[0, 1], xlabel=S.X_EPS)
        axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 else f"{v:.1f}"))
    S.fig_legend(fig, [handles[w] for w in order if w in handles], top_cm=0.6, ncol=4)
    S.letter_panels(list(axes.flat)[:len(panels)])
    S.save(fig, path)


def _mechanism(path: pathlib.Path, family: str, heading: str) -> None:
    """Columns X->bb vs cc and X->bc vs bq, cs; upper row the four label sets, lower row the flavor boundaries."""
    title = {k: h for k, h, *_ in M.PAIRS}
    _pairs(path, family, [(rows, key, f"{title[key]}: {word}") for rows, word in
                          ((PAIR_LABEL_SETS, "label sets"), (PAIR_BOUNDARIES, "boundaries"))
                          for key in ("bb_vs_cc", "bc_vs_bq_cs")], heading)


def draw_f4_mech(path: pathlib.Path) -> None:
    _mechanism(path, "pairs", "Linear probes on JetClass-II class pairs")


def draw_f4_mech_ft(path: pathlib.Path) -> None:
    _mechanism(path, "pairs_ft", "Fine-tuned on JetClass-II, class pairs")


def draw_f4_other(path: pathlib.Path) -> None:
    title = {k: h for k, h, *_ in M.PAIRS}
    _pairs(path, "pairs", [(PAIR_LABEL_SETS, key, title[key]) for key in ("bbqq_vs_ccqq", "ee_vs_mm", "qcd_bb_vs_cc")],
           "Linear probes on JetClass-II class pairs")


RATIO_HEIGHT = 0.35     # F6's ratio panels, one height for all three (OmniLearned Fig. 3's form)
RATIO_RANGE_FROM = 0.1  # the ratio range is set by efficiencies from 0.1: below, a few background jets swing it
STACKED_RATIO = (14 * S.CM, 22 * S.CM)
BOOK_H = 18.5           # cm: the tallest figure the results book prints at full size (make_results_book.MAX_H_CM); F6 and
                        # F6-app are drawn at it, so their text prints at its drawn size (PI 2026-11-07: at least 8 pt)


def _rel_error(docs: dict, models: list, key: str):
    """Relative uncertainty of mean_over_runs' curve at its points (gallery_bands.rejection_rel_error): the runs'
    rejection and passing background count where every run has at least 20 background jets, mean_over_runs' cut."""
    have = [m for m in models if m in docs and key in docs[m]["curves"]]
    if not have:
        return None
    rej = np.array([M.V4.roc_series(docs[m], key)[1] for m in have])
    ok = np.isfinite(rej).all(axis=0)
    k = np.array([np.asarray(docs[m]["curves"][key]["n_bkg_pass"], float) for m in have])
    return rejection_rel_error(rej[:, ok], k[:, ok], docs[have[0]]["curves"][key]["n_background"])


def _ratio(rx, s, ref, rel=None, band: bool = False, **kw) -> list:
    """A curve over the 188-class curve where both have points; the ratios drawn, and with rel (the curve's relative
    uncertainty at its points) a band of that curve's own uncertainty around its ratio. The ratios returned, with the
    band's edges when band."""
    if s is None or ref is None or len(s[0]) < 2:
        return []
    common, i, j = np.intersect1d(np.round(s[0], 9), np.round(ref[0], 9), return_indices=True)
    if common.size < 2:
        return []
    r = s[1][i] / ref[1][j]
    rx.plot(common, r, **kw)
    far = common >= RATIO_RANGE_FROM
    if rel is None:
        return list(r[far])
    lo, hi = r * (1 - rel[i]), r * (1 + rel[i])
    rx.fill_between(common, lo, hi, color=kw["color"], alpha=0.2, linewidth=0, zorder=kw.get("zorder", 2) - 0.1)
    return list(r[far]) + (list(lo[far]) + list(hi[far]) if band else [])


def _benchmarks(path: pathlib.Path, top, qg, word: str, kinematic: bool) -> None:
    """Round 4: top tagging alone at full width on the first row, quark/gluon Pythia and Herwig side by side on the
    second; a ratio panel to the 188-class curve under each, 0.35 of its height. The ratio range is set by the label
    sets (and the kinematic-inputs model); from scratch is drawn and may leave it. From scratch always from the
    fine-tuned families."""
    top_ft, qg_ft = _family("top_ft"), _family("qg_ft")
    left, gap, top_cm, rgap, row_gap, bottom = ROC_LEFT, ROC_WGAP, 1.2, 0.1, 1.2, 0.95
    w1, w2 = S.PRINT_W - left - 0.3, S.PANEL_W
    h2 = 0.7 * w2 + 0.01     # clear of the aspect rule's 0.7 at print
    H = BOOK_H          # the first row's height takes what the second leaves (item 5, PI 2026-11-07)
    h1 = (H - top_cm - 2 * rgap - row_gap - h2 * (1 + RATIO_HEIGHT) - bottom) / (1 + RATIO_HEIGHT)
    fig = S.figure_cm(S.PRINT_W, H)
    y2 = top_cm + h1 * (1 + RATIO_HEIGHT) + rgap + row_gap
    boxes = [(left, top_cm, w1, h1), (left, y2, w2, h2), (left + w2 + gap, y2, w2, h2)]
    axes, ratios = [], []
    for x, y, w, h in boxes:
        ax = S.axes_cm(fig, x, y, w, h)
        rx = S.axes_cm(fig, x, y + h + rgap, w, RATIO_HEIGHT * h, sharex=ax)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
        ratios.append(rx)
    axes[2].sharey(axes[1])                 # Pythia and Herwig: one y range, its tick labels once
    ratios[2].sharey(ratios[1])
    axes[2].tick_params(labelleft=False)
    ratios[2].tick_params(labelleft=False)
    panels = ((top, top_ft, "top", "Top tagging"), (qg, qg_ft, "qg", "Quark/gluon tagging, Pythia"), (qg, qg_ft, "qg_herwig", "Quark/gluon tagging, Herwig"))
    for k, (ax, rx, (docs, ft, key, name)) in enumerate(zip(axes, ratios, panels)):
        _roc_axes(ax)
        S.fine_grid(rx)
        rx.xaxis.set_major_locator(MultipleLocator(0.2))
        rx.xaxis.set_major_formatter(FuncFormatter(lambda v, _, k=k: "" if abs(v) < 1e-9 or (k == 1 and v > 1 - 1e-9)
                                                   else f"{v:.1f}"))     # (b)'s "1.0" left to (c)'s axis beside it
        ref = mean_over_runs(docs, RUNS["L188"], key)
        shown = [1.0]
        for level in S.LEVELS[::-1]:            # the 188-class curve last, on top
            s = mean_over_runs(docs, RUNS[level], key)
            _plot(ax, s, **S.look(level), linewidth=S.LW_CURVE, label=S.LEGEND[level])
            shown += _ratio(rx, s, ref, _rel_error(docs, RUNS[level], key), band=True, **S.look(level),
                            linewidth=S.LW_CURVE)
        models = P._scratch_models(ft)
        s = mean_over_runs(ft, models, key)
        _plot(ax, s, **S.look("SCRATCH"), linewidth=S.LW_CURVE, label=S.LEGEND["SCRATCH"] + ("" if word == "fine-tuned" else f" {S.FINE_TUNED}"), zorder=1.9)  # under the label sets
        scratch = _ratio(rx, s, ref, _rel_error(ft, models, key), band=True, **S.look("SCRATCH"), linewidth=S.LW_CURVE,
                         zorder=1.9)
        if word == "fine-tuned":            # after fine-tuning from scratch is near the label sets: its ratio in range
            shown += scratch
        if kinematic and key == "top" and M.KIN_MODEL in docs:
            s = mean_over_runs(docs, [M.KIN_MODEL], key)
            _plot(ax, s, **S.look("KIN"), linewidth=S.LW_CURVE, label=S.LEGEND["KIN"])
            shown += _ratio(rx, s, ref, _rel_error(docs, [M.KIN_MODEL], key), band=True, **S.look("KIN"),
                            linewidth=S.LW_CURVE)
        _y_range([ax] if k == 0 else axes[1:])
        top_ax = ax if k == 0 else axes[1]
        top_ax.set_ylim(1, 10.0 ** np.ceil(np.log10(top_ax.get_ylim()[1])))     # from 10^0 to a labelled decade
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v - 1) < 1e-9 else f"{v:g}"))
        rx.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], prune="upper"))   # 1.0 a tick; none at the top edge
        rx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))        # one format in every ratio panel
        v = np.array(shown, float)
        v = v[np.isfinite(v)]
        lo, hi = v.min(), v.max()
        pad = 0.08 * (hi - lo or 0.1)
        if k == 2:                          # the shared ratio range holds both quark/gluon panels
            lo, hi = min(lo, ratios[1].get_ylim()[0] + pad), max(hi, ratios[1].get_ylim()[1] - pad)
        rx.set_ylim(lo - pad, hi + pad)
        rx.set_xlim(0, 1)
        S.title(ax, f"({'abc'[k]}) {name}")
        if k < 2:
            S.labels(ax, ylabel=S.Y_REJ)
            S.labels(rx, ylabel="Ratio to 188")      # "classes" does not fit the quark/gluon ratio panel; the caption says it
        S.labels(rx, xlabel=S.X_EPS)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(S.LEGEND[a]) for a in S.LEVELS] + [i for i, lab in enumerate(labels)
                                                             if lab not in [S.LEGEND[a] for a in S.LEVELS]]
    _legend(axes[0], handles=[handles[i] for i in order], labels=[labels[i] for i in order], ncol=2)
    S.heading(fig, f"{word.capitalize()} on the full benchmark training sets")
    S.save(fig, path)


def draw_f6(path: pathlib.Path) -> None:
    _benchmarks(path, _family("top_ft"), _family("qg_ft"), "fine-tuned", True)


def draw_f6_app(path: pathlib.Path) -> None:
    _benchmarks(path, _family("top_probe"), _family("qg_probe"), "linear probes", True)


def _jc2_every_class(path: pathlib.Path, family: str, heading: str) -> None:
    docs = _family(family)
    names = {r["jet_label"]: r["class_name"] for r in M._class_rows(P.REPO)}
    colour = {label: S.TAB10[FAMILIES.index(family_of(name))] for label, name in names.items()}
    fig, axes = _axes_2x2(sharey=True, wspace=0.1, heading=heading, top=2.35)
    for ax, level in zip(axes.flat, S.LEVELS):
        aucs = []
        for label in names:
            s = mean_over_runs(docs, RUNS[level], label)
            aucs.append(s[2])
            _plot(ax, s, color=colour[label], linewidth=0.4, alpha=0.7)
        S.title(ax, f"{S.NAME[level]}, macro AUC {np.mean(aucs):.3f}")
    _y_range(list(axes.flat))
    _right_labels_off(axes)
    S.fig_legend(fig, [Line2D([], [], color=S.TAB10[i], linewidth=S.LW_CURVE, label=f) for i, f in enumerate(FAMILIES)],
                 top_cm=0.6, ncol=3)        # above the panels, off the curves (PI, 2026-11-07)
    S.letter_panels(axes.flat)
    S.save(fig, path)


def draw_a3r(path: pathlib.Path) -> None:
    _jc2_every_class(path, "jc2_ovr", "Linear probes on $10^6$ JetClass-II jets, per pretraining label set")


def draw_a3r_ft(path: pathlib.Path) -> None:
    _jc2_every_class(path, "jc2_ft_ovr", "Fine-tuned on $10^6$ JetClass-II jets, per pretraining label set")



PAGES = [
    Page(id="FJ", title="JetClass after fine-tuning",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="The networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, one per panel, "
                 "fine-tuned on one million JetClass jets: background rejection against signal efficiency with "
                 "each JetClass class as signal against the other nine, the mean over three pretraining runs, "
                 "ending where fewer than 20 background jets remain. Titles give the mean area under the curve "
                 "over the ten classes; the heading, that of a network without pretraining trained from scratch.",
         differs="Against landscape: ten classes of one network, not fifteen networks; solid tab10. Against the poster: "
                 "rejection against efficiency (PI); four panels, not two; legend names only, as the AUCs do not fit.",
         draw=draw_fj),
    Page(id="FJ-ctrl", title="JetClass after fine-tuning, random partitions",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Networks pretrained on JetClass-II with 17-class partitions, with no flavor boundary, a b-quark "
                 "boundary or a random flavor boundary (two runs each) and five random partitions (one run each, "
                 "averaged), fine-tuned on one million JetClass jets: rejection against efficiency for each class "
                 "against the other nine, ending where fewer than 20 background jets remain. Titles give the mean "
                 "area under the curve; the heading, that of a network trained from scratch.",
         differs="As FJ. The five plain random partitions share one panel, to stay at four: their macro AUCs lie within "
                 "3e-4 of each other. Eight panels, one per partition, is the alternative.",
         draw=draw_fj_ctrl),
    Page(id="FJ-s", title="JetClass trained from scratch",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (AUC in legend)",
         caption="A network without pretraining, trained from scratch on one million JetClass jets: background "
                 "rejection against signal efficiency with each of the ten JetClass classes as signal and the "
                 "other nine as background, the mean over three trainings, ending where fewer than 20 background "
                 "jets remain. The legend gives each class's area under the curve; the heading gives their mean.",
         differs="As FJ, one panel; the class AUCs in a legend beside it on the right (PI), full width.",
         draw=draw_fj_s),
    Page(id="FJ-app", title="JetClass with linear probes (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes, trained on one million JetClass jets: background rejection against signal "
                 "efficiency with each class as signal against the other nine, the mean over three pretraining "
                 "runs, ending where fewer than 20 background jets remain. Titles give the mean area under the "
                 "curve; the heading, that of a network trained from scratch, for reference.",
         differs="As FJ, with no legend: probe curves lie lower, so it would cover Z to qq; the colors are FJ's. The "
                 "poster's left panel is also a frozen network, with an MLP on it; here the probe is linear.",
         draw=draw_fj_app),
    Page(id="FJ-ctrl-app", title="JetClass with linear probes, random partitions (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 17-class "
                 "partitions, with no flavor boundary, a b-quark boundary or a random flavor boundary (two runs "
                 "each) and five random partitions (one run each, averaged), trained on one million JetClass "
                 "jets: rejection against efficiency for each class against the other nine, ending where fewer "
                 "than 20 background jets remain. Titles give the mean area under the curve; the heading, from "
                 "scratch.",
         differs="As FJ-ctrl, with no legend, as FJ-app. The five plain partitions' macro AUCs lie within 5e-4 of each other.",
         draw=draw_fj_ctrl_app),
    Page(id="F4-mech", title="Flavor separation: what the labels merge",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II, trained and scored on "
                 "two JetClass-II class pairs differing only in flavor (signal first): upper row, 188, 162, 43 and 17 "
                 "classes; lower row, 17-class partitions with no, a b-quark and a random flavor boundary, with 188 "
                 "classes for reference. Means over three pretraining runs (partitions: two); curves end where fewer "
                 "than 20 background jets remain.",
         differs="F4's (a) and (c) and F4-app's (a) and (c) in one 2x2 (PI, 2026-11-08): columns the pairs, rows the label "
                 "sets and the flavor boundaries; every panel from 1 to 10^4, the legend above the panels, off the curves. "
                 "Otherwise as F4: solid curves, plain-number ticks, landscape's axes.",
         draw=draw_f4_mech),
    Page(id="F4-mech-ft", title="Flavor separation after fine-tuning (appendix)",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Networks pretrained on JetClass-II, fine-tuned on one million class-balanced JetClass-II jets and "
                 "scored on the natural-mix test jets of two class pairs differing only in flavor (signal first) by the "
                 "signal side's share of the summed softmax of the pair's classes: upper row, 188, 162, 43 and 17 "
                 "classes; lower row, 17-class partitions with no, a b-quark and a random flavor boundary, with 188 for "
                 "reference. Means over three runs (partitions: two); curves end at 20 background jets.",
         differs="F4-mech's form after fine-tuning (PI, 2026-11-08), on F4-mech's axes (rejection 1 to 10^4); the main "
                 "text quotes these numbers from the pair table's fine-tuned columns.",
         data_note="Reads curves_v2/pairs_ft (the fine-tuned pairs job on the natural-mix test jets: the signal side's "
                   "summed softmax over the pair's classes' summed softmax), in curves_v2/pairs's per-model schema; absent "
                   "until pulled.",
         draw=draw_f4_mech_ft),
    Page(id="F4-other", title="Flavor separation, the other pairs (appendix)",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and "
                 "17 classes, trained and scored on three pairs of JetClass-II classes differing only in flavor "
                 "(signal first): (a) X→YY→bb̄qq̄ vs cc̄qq̄, (b) X→ee vs μμ and (c) QCD bb̄ vs cc̄. Means over three "
                 "pretraining runs; curves end where fewer than 20 background jets remain; a network with no curve "
                 "is named in its panel.",
         differs="F4's (b) and (d) with QCD bb̄ vs cc̄ (PI, 2026-11-08); as F4-mech otherwise, the fourth cell empty.",
         data_note="X→ee vs X→μμ is read from every test jet of the pair (pairs_sib2), so a curve can reach a rejection "
                   "of 6,442; the 188- and 162-class models keep fewer than 20 X→μμ jets at every efficiency and are "
                   "named as complete separation.",
         draw=draw_f4_other),
    Page(id="F6", title="Benchmarks after fine-tuning",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="The networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and a network without "
                 "pretraining, fine-tuned on the full top and quark/gluon training sets: rejection against "
                 "efficiency, means over three runs; dotted, one run with 188 classes and kinematic inputs only. "
                 "Lower panels: ratio to 188 classes; bands: each curve's SD over runs and 68% binomial "
                 "uncertainty of the passing background count, in quadrature. On Herwig jets, 43 and 17 classes "
                 "agree with 188 within uncertainty.",
         differs="Three stacked panels on one x axis (Sophon Fig. 3's arrangement), as landscape and ParticleNet draw "
                 "one per figure. Rejection on y as landscape, not ParticleNet's background efficiency; no AUC in the "
                 "legend, unlike ParticleNet (PI limit, numbers are in tables). The y axis starts at 1, not at the data; "
                 "Sophon's font, 1.1 pt lines; round 3's faint y grid and palette: label sets in viridis, from scratch dark "
                 "grey dashed and, on top tagging, the kinematic-inputs model dotted in the 188 colour, not landscape's "
                 "fifteen in mixed line styles. Legend without frame, first panel only, lower left in two columns for its six entries, as curves fill the other "
                 "corners of that panel. Stacked panels are larger than Sophon's, with a wider gap: the y-axis words "
                 "are longer than Sophon's 'Events' and repeat on each panel. Under each panel the ratio to the 188-class "
                 "curve (PI, OmniLearned Fig. 3's form), one height (0.35 of the panel) for all three; the paper's only "
                 "ratio-to-a-model panel.",
         data_note="The 188-class kinematic-inputs model has one run, so the dotted curve is that run, not a mean.",
         draw=draw_f6),
    Page(id="F6-app", title="Benchmarks with linear probes (appendix)",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes, trained on the full top and quark/gluon training sets, and a network without "
                 "pretraining, fine-tuned from scratch (f.t.): rejection against efficiency, means over three "
                 "runs; dotted, one run with 188 classes and kinematic inputs only. Lower panels: ratio to 188 "
                 "classes; bands: each curve's SD over runs and 68% binomial uncertainty of the passing "
                 "background count, in quadrature.",
         differs="As F6, with linear probes; the ratio is to the 188-class probe.",
         data_note="The kinematic-inputs model's probe is one run (curves_v2/top_probe/mtx-l188kin-s1.json), so its "
                   "dotted curve is that run, not a mean.",
         draw=draw_f6_app),
    Page(id="A3R", title="Every JetClass-II class, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes, trained on one million class-balanced JetClass-II jets: rejection against "
                 "efficiency for each of the 188 classes against the rest of the test set, the mean over three "
                 "pretraining runs, colored by decay family, ending where fewer than 20 background jets remain. "
                 "Each title gives the mean area under the curve over all classes.",
         differs="Axes as FJ (PI). Linear probes on JetClass-II in all four panels (1M jets), where the poster has a "
                 "frozen MLP and full fine-tuning on JetClass (10M). 188 thin curves in six family colours, not ten "
                 "labelled classes; no AUC per curve, as there are too many. 0.4 pt lines at alpha 0.7: over the limit "
                 "of six series and off the line convention (one colour per series, 1.1 pt), not named in this brief, for "
                 "the PI to decide. No dashed diagonal. One-line titles, not two. Sophon's font and round 3's faint y grid, not the "
                 "poster's serif font and faint grid. The family legend is above the panels, off the curves, unframed, "
                 "with lines of normal thickness. Four panels, not two.",
         draw=draw_a3r),
    Page(id="A3R-ft", title="Every JetClass-II class after fine-tuning, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="The networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, fine-tuned on one "
                 "million class-balanced JetClass-II jets: rejection against efficiency for each of the 188 "
                 "classes against the rest of the test set, the mean over three pretraining runs, colored by "
                 "decay family, ending where fewer than 20 background jets remain. Each title gives the mean area "
                 "under the curve over all classes.",
         differs="As A3R, fine-tuned.",
         draw=draw_a3r_ft),
]
