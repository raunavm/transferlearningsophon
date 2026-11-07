#!/usr/bin/env python3
"""The gallery's ROC mock-ups (docs/figstyle/PLAN.md): FJ, FJ-ctrl, FJ-s, FJ-app, FJ-ctrl-app, F4, F4-app, F6, F6-app, A3R, A3R-ft. Every ROC is
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

from gallery_common import Absent, M, P, Page, ROC, RUNS, mean_over_runs
import style_v5 as S

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


def _axes_2x2(sharey: bool, wspace: float, heading: str):
    """Round 4: four 6.5 cm panels (style_v5.grid_cm), the y label once per row, the x label under the lower row."""
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=1.0)
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


def _floor(ax, docs: dict, models: list, key: str) -> None:
    """The y axis starts at 1; a panel with no curve gets the range its test set allows."""
    if ax.has_data():
        _y_range([ax])
    else:
        ax.set_ylim(1, M.V4.roc_limit(docs[models[0]], key))


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
    """One panel at its printed width (FJ-s: 9 cm), the class AUCs in a legend inside it (PI, 2026-11-07): ten curves
    leave no empty corner, so the y axis runs three decades above them and the legend sits in that band."""
    width, w = 9.0, 7.3
    h = 1.15 * w
    top = 0.9
    fig = S.figure_cm(width, top + h + 1.0)
    ax = S.axes_cm(fig, 1.4, top, w, h)
    _roc_axes(ax, x_step=0.2)
    docs = _family("jc1_ft_ovr")
    aucs = {c: mean_over_runs(docs, models, c)[2] for c in S.JC1_CLASSES}
    _jc1_curves(ax, docs, models, auc_in_legend=False)
    _y_range([ax])
    ax.set_ylim(1, 1000 * ax.get_ylim()[1])           # the legend's band, clear of every curve
    S.labels(ax, S.X_EPS, S.Y_REJ)
    S.heading(fig, f"{heading}, macro AUC {np.mean(list(aucs.values())):.3f}")
    _legend(ax, handles=_class_handles(aucs), ncol=2, title="Signal class (AUC)", title_fontsize=S.LEGEND_PT5,
            loc="upper center")
    S.save(fig, path)


def _jc1_four(path: pathlib.Path, family: str, heading: str, panels: list, legend: bool = True) -> None:
    """panels: (models, title). One class legend above the panels, the same in every JetClass figure."""
    docs = _family(family)
    scratch = _family("jc1_ft_ovr")
    reference = np.mean([mean_over_runs(scratch, P._scratch_models(scratch), c)[2] for c in S.JC1_CLASSES])
    fig, axes, _ = S.grid_cm(2, 2, aspect=0.85, top=2.35)
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


def _pairs(path: pathlib.Path, series: list, keys: tuple, ymax: float | None = None, legend_order: list | None = None) -> None:
    """Four class pairs; series [(models, plot keywords)]. The series are the same in every panel, so the
    legend is in the first (the fourth may hold none of them). X->ee vs X->mumu is read from every test jet of
    the pair (pairs_sib2) where that exists. With ymax, every panel runs from 1 to ymax, and a model whose runs
    keep fewer than 20 background jets at every efficiency is named in the panel as complete separation."""
    docs = _family("pairs")
    heading = {k: h for k, h, *_ in M.PAIRS}
    fig, axes = _axes_2x2(sharey=False, wspace=0.2, heading="Linear probes on JetClass-II class pairs")
    for ax, key in zip(axes.flat, keys):
        d = ROC["pairs_sib2"] if key == "ee_vs_mm" and ROC.get("pairs_sib2") else docs
        separated = []
        for models, kw in series:
            s = mean_over_runs(d, models, key)
            _plot(ax, s, **kw)
            if s is not None and len(s[0]) < 2:
                separated.append(kw["label"])
        if ymax:
            ax.set_ylim(1, ymax)
        else:
            _floor(ax, d, series[0][0], key)
        if ymax and separated:
            if legend_order:
                separated.sort(key=legend_order.index)
            words = (" and ".join(w.removesuffix(" classes") for w in separated) + " classes"
                     if all(w.endswith(" classes") for w in separated) else ", ".join(separated))
            ax.text(0.97, 0.95, "Complete separation:\n" + words, transform=ax.transAxes, ha="right",
                    va="top", fontsize=S.LEGEND_PT5)     # no frame, as the legends (style_v4)
        S.title(ax, heading[key])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: legend_order.index(labels[i])) if legend_order else range(len(labels))
    _legend(axes[0, 0], handles=[handles[i] for i in order], labels=[labels[i] for i in order],
            title="Pretrained on", title_fontsize=S.LEGEND_PT5)
    S.letter_panels(axes.flat)
    S.save(fig, path)


def draw_f4(path: pathlib.Path) -> None:
    """Linear probes, solid (a figure of probes only). The 188-class curve is drawn last, so it shows where the 162-class curve lies on it."""
    levels = list(S.LEVELS)
    _pairs(path, [(RUNS[a], {**S.look(a), "linewidth": S.LW_CURVE, "label": S.NAME[a]}) for a in levels[::-1]],
           M.ROC_PAIRS, ymax=F4_YMAX, legend_order=[S.NAME[a] for a in levels])


def draw_f4_app(path: pathlib.Path) -> None:
    _pairs(path, [(RUNS[a], {**S.look(a), "linewidth": S.LW_CURVE, "label": S.NAME[a]}) for a in M.FLAV]
           + [(RUNS["L188"], {**S.look("L188"), "linewidth": S.LW_CURVE, "label": S.NAME["L188"]})],
           ("bb_vs_cc", "bbqq_vs_ccqq", "bc_vs_bq_cs", "qcd_bb_vs_cc"))     # X->ee vs mumu: no curve to draw here


RATIO_HEIGHT = 0.35     # F6's ratio panels, one height for all three (OmniLearned Fig. 3's form)
RATIO_RANGE_FROM = 0.1  # the ratio range is set by efficiencies from 0.1: below, a few background jets swing it
STACKED_RATIO = (14 * S.CM, 22 * S.CM)


def _ratio(rx, s, ref, **kw) -> list:
    """A curve over the 188-class curve where both have points; the ratios drawn."""
    if s is None or ref is None or len(s[0]) < 2:
        return []
    common, i, j = np.intersect1d(np.round(s[0], 9), np.round(ref[0], 9), return_indices=True)
    if common.size < 2:
        return []
    r = s[1][i] / ref[1][j]
    rx.plot(common, r, **kw)
    return list(r[common >= RATIO_RANGE_FROM])


def _benchmarks(path: pathlib.Path, top, qg, word: str, kinematic: bool) -> None:
    """Round 4: top tagging alone at full width on the first row, quark/gluon Pythia and Herwig side by side on the
    second; a ratio panel to the 188-class curve under each, 0.35 of its height. The ratio range is set by the label
    sets (and the kinematic-inputs model); from scratch is drawn and may leave it. From scratch always from the
    fine-tuned families."""
    top_ft, qg_ft = _family("top_ft"), _family("qg_ft")
    left, gap, top_cm, rgap, row_gap, bottom = 1.15, 0.35, 1.2, 0.1, 1.2, 0.95
    w1 = S.PRINT_W - left - 0.3
    h1, w2 = 0.7 * w1, S.PANEL_W
    h2 = 0.7 * w2
    H = top_cm + h1 * (1 + RATIO_HEIGHT) + rgap + row_gap + h2 * (1 + RATIO_HEIGHT) + rgap + bottom
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
        rx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-9 else f"{v:.1f}"))
        ref = mean_over_runs(docs, RUNS["L188"], key)
        shown = [1.0]
        for level in S.LEVELS[::-1]:            # the 188-class curve last, on top
            s = mean_over_runs(docs, RUNS[level], key)
            _plot(ax, s, **S.look(level), linewidth=S.LW_CURVE, label=S.NAME[level])
            shown += _ratio(rx, s, ref, **S.look(level), linewidth=S.LW_CURVE)
        s = mean_over_runs(ft, P._scratch_models(ft), key)
        _plot(ax, s, **S.look("SCRATCH"), linewidth=S.LW_CURVE, label=S.NAME["SCRATCH"], zorder=1.9)  # under the label sets
        scratch = _ratio(rx, s, ref, **S.look("SCRATCH"), linewidth=S.LW_CURVE, zorder=1.9)
        if word == "fine-tuned":            # after fine-tuning from scratch is near the label sets: its ratio in range
            shown += scratch
        if kinematic and key == "top" and M.KIN_MODEL in docs:
            s = mean_over_runs(docs, [M.KIN_MODEL], key)
            _plot(ax, s, **S.look("KIN"), linewidth=S.LW_CURVE, label=S.NAME["KIN"])
            shown += _ratio(rx, s, ref, **S.look("KIN"), linewidth=S.LW_CURVE)
        _y_range([ax] if k == 0 else axes[1:])
        top_ax = ax if k == 0 else axes[1]
        top_ax.set_ylim(1, 10.0 ** np.ceil(np.log10(top_ax.get_ylim()[1])))     # from 10^0 to a labelled decade
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v - 1) < 1e-9 else f"$10^{{{round(np.log10(v))}}}$"))
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
    order = [labels.index(S.NAME[a]) for a in S.LEVELS] + [i for i, lab in enumerate(labels)
                                                           if lab not in [S.NAME[a] for a in S.LEVELS]]
    _legend(axes[0], handles=[handles[i] for i in order], labels=[labels[i] for i in order], ncol=2)
    S.heading(fig, f"{word.capitalize()} on the full benchmark training sets")
    S.save(fig, path)


def draw_f6(path: pathlib.Path) -> None:
    _benchmarks(path, _family("top_ft"), _family("qg_ft"), "fine-tuned", True)


def draw_f6_app(path: pathlib.Path) -> None:
    _benchmarks(path, _family("top_probe"), _family("qg_probe"), "linear probes", False)


def _jc2_every_class(path: pathlib.Path, family: str, heading: str) -> None:
    docs = _family(family)
    names = {r["jet_label"]: r["class_name"] for r in M._class_rows(P.REPO)}
    colour = {label: S.TAB10[FAMILIES.index(family_of(name))] for label, name in names.items()}
    fig, axes = _axes_2x2(sharey=True, wspace=0.1, heading=heading)
    for ax, level in zip(axes.flat, S.LEVELS):
        aucs = []
        for label in names:
            s = mean_over_runs(docs, RUNS[level], label)
            aucs.append(s[2])
            _plot(ax, s, color=colour[label], linewidth=0.4, alpha=0.7)
        S.title(ax, f"{S.NAME[level]}, macro AUC {np.mean(aucs):.3f}")
    _y_range(list(axes.flat))
    _legend(axes[0, 0], handles=[Line2D([], [], color=S.TAB10[i], linewidth=S.LW_CURVE, label=f)
                                 for i, f in enumerate(FAMILIES)], fontsize=8)
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
         caption="Background rejection against signal efficiency on JetClass after fine-tuning on one million jets. "
                 "Each curve takes one of the ten JetClass classes as signal and the other nine as background, and is "
                 "the mean over three pretraining runs. Curves begin where 20 background jets remain. One panel per "
                 "pretraining label set; its title gives the mean area under the curve over the ten classes, and the heading "
                 "that of the network trained from scratch.",
         differs="Against landscape: ten classes of one network, not fifteen networks; solid tab10. Against the poster: "
                 "rejection against efficiency (PI); four panels, not two; legend names only, as the AUCs do not fit.",
         draw=draw_fj),
    Page(id="FJ-ctrl", title="JetClass after fine-tuning, random partitions",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Background rejection against signal efficiency on JetClass after fine-tuning on one million jets, "
                 "for networks pretrained on 17-class random partitions: with no flavor boundary, a b-quark boundary "
                 "or a random flavor boundary (two runs each), and five plain random partitions (averaged). Each curve "
                 "takes one class as signal against the other nine and begins where 20 background jets remain. Titles "
                 "give the mean area under the curve; the heading, from scratch.",
         differs="As FJ. The five plain random partitions share one panel, to stay at four: their macro AUCs lie within "
                 "3e-4 of each other. Eight panels, one per partition, is the alternative.",
         draw=draw_fj_ctrl),
    Page(id="FJ-s", title="JetClass trained from scratch",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (AUC in legend)",
         caption="Background rejection against signal efficiency on JetClass for the network trained from scratch on "
                 "one million jets, with no pretraining. Each curve takes one of the ten JetClass classes as signal "
                 "and the other nine as background, and is the mean over three trainings. Curves begin where 20 "
                 "background jets remain. The legend gives each class's area under the curve; the heading gives their "
                 "mean.",
         differs="As FJ, one panel, with room for the class AUCs in the legend.",
         draw=draw_fj_s),
    Page(id="FJ-app", title="JetClass with linear probes (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Background rejection against signal efficiency on JetClass from linear probes, trained on one million "
                 "jets, on frozen pretrained networks. Each curve takes one of the ten classes as signal and the other "
                 "nine as background, and is the mean over three pretraining runs. Curves begin where 20 background jets "
                 "remain. Titles give the mean area under the curve; the heading gives the fully trained network from "
                 "scratch for reference.",
         differs="As FJ, with no legend: probe curves lie lower, so it would cover Z to qq; the colors are FJ's. The "
                 "poster's left panel is also a frozen network, with an MLP on it; here the probe is linear.",
         draw=draw_fj_app),
    Page(id="FJ-ctrl-app", title="JetClass with linear probes, random partitions (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Background rejection against signal efficiency on JetClass from linear probes trained on one million "
                 "jets, on frozen networks pretrained on 17-class random partitions: with no flavor boundary, a "
                 "b-quark boundary or a random flavor boundary (two runs each), and five plain ones (averaged). Each "
                 "curve takes one class as signal against the other nine and begins where 20 background jets remain. "
                 "Titles give the mean area under the curve; the heading, from scratch.",
         differs="As FJ-ctrl, with no legend, as FJ-app. The five plain partitions' macro AUCs lie within 5e-4 of each other.",
         draw=draw_fj_ctrl_app),
    Page(id="F4", title="Flavor separation for four class pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Background rejection against signal efficiency for four pairs of JetClass-II classes that differ "
                 "only in flavor, from linear probes on networks pretrained with 188, 162, 43 and 17 classes. The "
                 "first class named is the signal. Curves are means over three runs and end where fewer than 20 "
                 "background jets remain; models with fewer at every efficiency are named in the panel. In (a, b) the "
                 "three finest label sets nearly coincide.",
         differs="Four panels, not one, for four class pairs. Four solid curves, not fifteen in mixed line styles (PI "
                 "limit), at 1.1 pt, not 0.955 pt. The y axis starts at 1, not at the data, with its range set per "
                 "panel. x ticks every 0.2 over 0 to 1, not every 0.1 over about -0.01 to 1.02. Axis words at the axis "
                 "end, with the rejection written inline, not as a fraction. The legend is in the first panel only, "
                 "lower left, as the curves fill the upper right; it holds for every panel. Sophon's fonts; round 3's "
                 "faint y grid, frameless legend and palette (label sets in viridis, darkest for 188). The complete-"
                 "separation note is plain text, without the box it had.",
         data_note="X→ee vs X→μμ is read from every test jet of the pair (pairs_sib2: 135,245 X→ee and 128,833 X→μμ "
                   "jets), so a curve can reach a rejection of 6,442 (128,833 / 20). The 188- and 162-class models keep "
                   "fewer than 20 X→μμ jets at every efficiency in some run even so, and are named as complete "
                   "separation; the 43- and 17-class models draw full curves. All four panels run from 1 to 10^4 (PI).",
         draw=draw_f4),
    Page(id="F4-app", title="Flavor separation for random partitions with controlled flavor boundaries",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Background rejection against signal efficiency for four pairs of JetClass-II classes that differ "
                 "only in flavor, (d) of QCD jets, from linear probes on networks pretrained on 17-class random "
                 "partitions with no flavor boundary, a b-quark boundary or a random flavor boundary, and on 188 "
                 "classes for reference. The first class named is the signal. Curves are means over two runs (188 "
                 "classes: three) and end where fewer than 20 background jets remain.",
         differs="As F4, with the three random partitions with controlled flavor boundaries (rose, green, cyan) "
                 "and the 188-class model in its colour, all solid, in place of F4's four label sets. "
                 "One panel differs from F4's: QCD bb vs cc in place of X→ee vs X→μμ, which has no curve here.",
         data_note="QCD bb vs cc replaces F4's X→ee vs X→μμ here: neither run of any controlled partition, and no run "
                   "of the 188-class model, keeps 20 background jets at any efficiency on that pair's grid, so its panel "
                   "would be empty.",
         draw=draw_f4_app),
    Page(id="F6", title="Benchmarks after fine-tuning",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="Background rejection against signal efficiency after fine-tuning on the full training sets: top "
                 "tagging, and quark/gluon separation tested on Pythia and Herwig jets. Curves are networks pretrained "
                 "with 188, 162, 43 and 17 classes and the network trained from scratch, each the mean over three "
                 "runs; in (a), dotted, one run of the 188-class model with kinematic inputs only. Curves end where "
                 "fewer than 20 background jets remain. Lower panels: ratio to 188 classes.",
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
         caption="Background rejection against signal efficiency from linear probes on the full training sets (missing "
                 "inputs filled by random draws on top tagging, by the training mean on quark/gluon), on networks "
                 "pretrained with 188, 162, 43 and 17 classes, and for reference the network fully trained from "
                 "scratch (three runs each). Curves end where fewer than 20 background jets remain. Lower panels: "
                 "ratio to the 188-class probe; from scratch falls outside it in (a), (b) and partly (c).",
         differs="As F6, with linear probes; the ratio is to the 188-class probe.",
         draw=draw_f6_app),
    Page(id="A3R", title="Every JetClass-II class, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="Background rejection against signal efficiency for each of the 188 JetClass-II classes against the "
                 "rest of the class-balanced test set, from linear probes on frozen networks pretrained with 188, 162, "
                 "43 and 17 classes. Thin curves are means over three pretraining runs, colored by decay family, and "
                 "end where fewer than 20 background jets remain. Each title gives the mean area under the curve over "
                 "all classes.",
         differs="Axes as FJ (PI). Linear probes on JetClass-II in all four panels (1M jets), where the poster has a "
                 "frozen MLP and full fine-tuning on JetClass (10M). 188 thin curves in six family colours, not ten "
                 "labelled classes; no AUC per curve, as there are too many. 0.4 pt lines at alpha 0.7: over the limit "
                 "of six series and off the line convention (one colour per series, 1.1 pt), not named in this brief, for "
                 "the PI to decide. No dashed diagonal. One-line titles, not two. Sophon's font and round 3's faint y grid, not the "
                 "poster's serif font and faint grid. The family legend is in the first panel only, lower left, unframed, "
                 "with no heading, and with lines of normal thickness. Four panels, not two.",
         draw=draw_a3r),
    Page(id="A3R-ft", title="Every JetClass-II class after fine-tuning, per label set",
         precedent=["map2026_poster/fig5.png"],
         precedent_label="MAP 2026 poster Fig. 5 (form)",
         caption="Background rejection against signal efficiency for each of the 188 JetClass-II classes against the "
                 "rest of the class-balanced test set, after fine-tuning on one million jets, for networks pretrained with "
                 "188, 162, 43 and 17 classes. Thin curves are means over runs, colored by decay family, and end where "
                 "fewer than 20 background jets remain. Each title gives the mean area under the curve over all classes.",
         differs="As A3R, fine-tuned.",
         draw=draw_a3r_ft),
]
