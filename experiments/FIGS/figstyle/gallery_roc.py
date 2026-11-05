#!/usr/bin/env python3
"""The gallery's ROC mock-ups (docs/figstyle/PLAN.md): FJ, FJ-s, FJ-all, FJ-app, F4, F4-app, F6, A3R. Every ROC is
rejection (log, from 1) against signal efficiency (linear, 0 to 1), the mean over runs from
gallery_common.mean_over_runs, on landscape Fig. 5's grid, with the y axis stopping where the curves stop so
they come in from the top of the frame, and the legend in the free lower left."""
from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

from gallery_common import Absent, M, P, Page, ROC, RUNS, mean_over_runs
import style_v3 as S

ROC_LW = 1.1
ONE_PANEL = (14 * S.CM, 11.5 * S.CM)
FOUR_PANELS = (S.FULL, 18 * S.CM)
GRID = {"left": 0.1, "right": 0.97, "top": 0.92, "bottom": 0.06, "hspace": 0.1}
STACKED = (14 * S.CM, 17 * S.CM)
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
    fig = plt.figure(figsize=FOUR_PANELS)
    axes = fig.subplots(2, 2, sharex=True, sharey=sharey, gridspec_kw={**GRID, "wspace": wspace})
    for ax in axes.flat:
        _roc_axes(ax)
    for ax in axes[:, 0]:
        S.labels(ax, ylabel=S.Y_REJ)
    for ax in axes[1]:
        S.labels(ax, xlabel=S.X_EPS)
    fig.suptitle(heading, fontsize=S.TITLE_PT, y=0.975)
    return fig, axes


def _roc_axes(ax, x_step: float = 0.2) -> None:
    """Landscape Fig. 5's grid: major and minor lines on both axes, x every 0.1."""
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1 if x_step > 0.1 else 0.05))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="both", linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)


def _y_range(axes) -> None:
    """From 1 to where the highest curve stops, or the decade just above it, so curves leave through the frame."""
    highest = max(float(line.get_ydata().max()) for ax in axes for line in ax.lines)
    decade = 10.0 ** np.ceil(np.log10(highest))
    axes[0].set_ylim(1, decade if decade / highest < 1.3 else 1.1 * highest)


def _legend(ax, **kw):
    return S.legend(ax, loc="lower left", handlelength=1.2, handletextpad=0.4, columnspacing=0.6, **kw)


def _save(fig, path: pathlib.Path) -> None:
    """Saves, and prints how many curve points lie under each legend."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None:
            box = leg.get_window_extent(r)
            n = sum(box.count_contains(ax.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])) for ln in ax.lines)
            print(f"{path.name}, {ax.get_title()!r}: {n} curve points under the legend")
    S.save(fig, path)


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
    """Ten class-against-the-rest curves; returns the macro AUC."""
    aucs = []
    for c in S.JC1_CLASSES:
        s = mean_over_runs(docs, models, c)
        aucs.append(s[2])
        _plot(ax, s, color=S.JC1_COLOUR[c], linewidth=ROC_LW,
              label=f"{S.JC1_TEX[c]} ({s[2]:.3f})" if auc_in_legend else S.JC1_TEX[c])
    return float(np.mean(aucs))


def _jc1_one(path: pathlib.Path, models: list, heading: str) -> None:
    """One large panel; the class AUCs in the legend."""
    fig, ax = plt.subplots(figsize=ONE_PANEL)
    _roc_axes(ax, x_step=0.1)
    auc = _jc1_curves(ax, _family("jc1_ft_ovr"), models, auc_in_legend=True)
    _y_range([ax])
    S.title(ax, f"{heading}\nmacro AUC {auc:.3f}")
    S.labels(ax, S.X_EPS, S.Y_REJ)
    _legend(ax, title="Signal class (AUC)", title_fontsize=8.5, fontsize=8.5, ncol=2)
    _save(fig, path)


def _jc1_four(path: pathlib.Path, family: str, heading: str) -> None:
    """One panel per label set; one legend, names only: with the AUCs it does not fit under the curves."""
    docs = _family(family)
    fig, axes = _axes_2x2(sharey=True, wspace=0.08, heading=heading)
    for ax, level in zip(axes.flat, S.LEVELS):
        auc = _jc1_curves(ax, docs, RUNS[level], auc_in_legend=False)
        S.title(ax, f"Pretrained on {S.NAME[level]}, macro AUC {auc:.3f}")
    _y_range(list(axes.flat))
    _legend(axes[0, 0], title="Signal class", title_fontsize=7.5, fontsize=7.5, ncol=2)
    _save(fig, path)


def draw_fj(path: pathlib.Path) -> None:
    _jc1_one(path, RUNS["L188"], "Pretrained on 188 classes, fine-tuned on $10^6$ JetClass jets")


def draw_fj_s(path: pathlib.Path) -> None:
    models = P._scratch_models(_family("jc1_ft_ovr"))
    if not models:
        raise Absent("no from-scratch model in curves_v2/jc1_ft_ovr")
    _jc1_one(path, models, "Trained from scratch on $10^6$ JetClass jets")


def draw_fj_all(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_ft_ovr", "Fine-tuned on $10^6$ JetClass jets")


def draw_fj_app(path: pathlib.Path) -> None:
    _jc1_four(path, "jc1_probe_ovr", "Linear probes trained on $10^6$ JetClass jets")


def _pairs(path: pathlib.Path, series: list, keys: tuple) -> None:
    """Four class pairs; series [(models, plot keywords)]. The series are the same in
    every panel, so the legend is in the first (the fourth may hold none of them)."""
    docs = _family("pairs")
    heading = {k: h for k, h, *_ in M.PAIRS}
    fig, axes = _axes_2x2(sharey=False, wspace=0.2, heading="Linear probes on JetClass-II class pairs")
    for ax, key in zip(axes.flat, keys):
        for models, kw in series:
            _plot(ax, mean_over_runs(docs, models, key), **kw)
        _floor(ax, docs, series[0][0], key)
        S.title(ax, heading[key])
    _legend(axes[0, 0], title="Pretrained on", title_fontsize=S.LEGEND_PT)
    _save(fig, path)


def draw_f4(path: pathlib.Path) -> None:
    _pairs(path, [(RUNS[a], {"color": S.SERIES[a], "linewidth": ROC_LW, "label": S.NAME[a]}) for a in S.LEVELS], M.ROC_PAIRS)


def draw_f4_app(path: pathlib.Path) -> None:
    _pairs(path, [(RUNS[a], {"color": S.SERIES[a], "linewidth": ROC_LW, "label": S.NAME[a]}) for a in M.FLAV]
           + [(RUNS["L188"], {**S.REFERENCE, "label": S.NAME["L188"]})],
           ("bb_vs_cc", "bbqq_vs_ccqq", "bc_vs_bq_cs", "qcd_bb_vs_cc"))     # X->ee vs mumu: no curve to draw here


def draw_f6(path: pathlib.Path) -> None:
    top, qg = _family("top_ft"), _family("qg_ft")
    fig = plt.figure(figsize=STACKED)
    axes = fig.subplots(3, 1, sharex=True, gridspec_kw=STACKED_GRID)
    panels = ((top, "top", "Top tagging, fine-tuned"), (qg, "qg", "Quark/gluon, Pythia, fine-tuned"),
              (qg, "qg_herwig", "Quark/gluon, Herwig, fine-tuned"))
    for ax, (docs, key, name) in zip(axes, panels):
        _roc_axes(ax, x_step=0.1)
        for level in S.LEVELS:
            _plot(ax, mean_over_runs(docs, RUNS[level], key), color=S.SERIES[level], linewidth=ROC_LW,
                  label=S.NAME[level])
        _plot(ax, mean_over_runs(docs, P._scratch_models(docs), key), color=S.SERIES["SCRATCH"], linewidth=ROC_LW,
              label=S.NAME["SCRATCH"], zorder=1.9)      # under the label sets, or it hides them where they agree
        if key == "top" and M.KIN_MODEL in docs:
            _plot(ax, mean_over_runs(docs, [M.KIN_MODEL], key), **S.REFERENCE, label=S.NAME["KIN"])
        _y_range([ax])
        S.title(ax, name)
        S.labels(ax, ylabel=S.Y_REJ)
    S.labels(axes[-1], xlabel=S.X_EPS)
    _legend(axes[0], ncol=2 if M.KIN_MODEL in top else 1, fontsize=S.LEGEND_PT)
    _save(fig, path)


def draw_a3r(path: pathlib.Path) -> None:
    docs = _family("jc2_ovr")
    names = {r["jet_label"]: r["class_name"] for r in M._class_rows(P.REPO)}
    colour = {label: S.TAB10[FAMILIES.index(family_of(name))] for label, name in names.items()}
    fig, axes = _axes_2x2(sharey=True, wspace=0.08, heading="Linear probes on $10^6$ JetClass-II jets")
    for ax, level in zip(axes.flat, S.LEVELS):
        aucs = []
        for label in names:
            s = mean_over_runs(docs, RUNS[level], label)
            aucs.append(s[2])
            _plot(ax, s, color=colour[label], linewidth=0.4, alpha=0.7)
        S.title(ax, f"Pretrained on {S.NAME[level]}, macro AUC {np.mean(aucs):.3f}")
    _y_range(list(axes.flat))
    _legend(axes[0, 0], handles=[Line2D([], [], color=S.TAB10[i], linewidth=S.LW, label=f)
                                 for i, f in enumerate(FAMILIES)], fontsize=7.5)
    _save(fig, path)


PAGES = [
    Page(id="FJ", title="JetClass after fine-tuning",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (size, grid, axes), MAP 2026 poster Fig. 5 (curves per panel, AUC in legend)",
         caption="Background rejection against signal efficiency on JetClass after fine-tuning on one million jets, "
                 "for the network pretrained on 188 classes. Each curve takes one of the ten JetClass classes as signal "
                 "and the other nine as background, and is the mean over three pretraining runs. Curves begin where "
                 "20 background jets remain. The legend gives each class's area under the curve; the title gives "
                 "their mean.",
         differs="Against landscape: ten classes of one network, not fifteen networks on one task; solid curves in "
                 "tab10; y range 1 to 1e5 against about 4 to 4e4; axis words at the axis end, in our word form; title "
                 "and legend heading added, to say what the curves are. Against the poster: rejection against signal "
                 "efficiency, not true against false positive rate (PI); one panel, not two; the other label sets in "
                 "FJ-all. Ten curves: over the limit of six series, the exception the PI asked for.",
         draw=draw_fj),
    Page(id="FJ-s", title="JetClass trained from scratch",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (size, grid, axes), MAP 2026 poster Fig. 5 (curves per panel, AUC in legend)",
         caption="Background rejection against signal efficiency on JetClass for the network trained from scratch on "
                 "one million jets, with no pretraining. Each curve takes one of the ten JetClass classes as signal "
                 "and the other nine as background, and is the mean over three trainings. Curves begin where 20 "
                 "background jets remain. The legend gives each class's area under the curve; the title gives their "
                 "mean.",
         differs="As FJ.",
         draw=draw_fj_s),
    Page(id="FJ-all", title="JetClass after fine-tuning, every label set (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (grid, axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Background rejection against signal efficiency on JetClass after fine-tuning on one million jets. "
                 "Each curve takes one of the ten JetClass classes as signal and the other nine as background, and is "
                 "the mean over three pretraining runs. Curves begin where 20 background jets remain. One panel per "
                 "pretraining label set; the title gives the mean area under the curve over the ten classes.",
         differs="As FJ, with four panels, one per label set, 15 by 18 cm. One legend, names only: with the class AUCs it "
                 "does not fit under the curves at this panel width.",
         draw=draw_fj_all),
    Page(id="FJ-app", title="JetClass with linear probes, every label set (appendix)",
         precedent=["1902.09914/fig5.png", "map2026_poster/fig5.png"],
         precedent_label="Landscape Fig. 5 (grid, axes), MAP 2026 poster Fig. 5 (one panel per model)",
         caption="Background rejection against signal efficiency on JetClass from linear probes, trained on one million "
                 "jets, on frozen pretrained networks. Each curve takes one of the ten JetClass classes as signal and "
                 "the other nine as background, and is the mean over three pretraining runs. Curves begin where 20 "
                 "background jets remain. One panel per pretraining label set; the title gives the mean area under the "
                 "curve over the ten classes.",
         differs="As FJ-all. The poster's left panel is also a frozen network, with an MLP on it; here the probe is linear.",
         draw=draw_fj_app),
    Page(id="F4", title="Flavor separation for four class pairs",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Background rejection against signal efficiency for four pairs of JetClass-II classes that differ only "
                 "in flavor, from linear probes on frozen networks pretrained with 188, 162, 43 and 17 classes. The "
                 "first class named is the signal. Each curve is the mean over three pretraining runs and ends where "
                 "fewer than 20 background jets remain; the 188- and 162-class models separate the electron and muon "
                 "pair completely, so they have no curve there, and elsewhere nearly coincide.",
         differs="Four panels, not one, for four class pairs. Four solid curves, not fifteen in mixed line styles (PI "
                 "limit), at 1.1 pt, not 0.955 pt. The y axis starts at 1, not at the data, with its range set per "
                 "panel. x ticks every 0.2 over 0 to 1, not every 0.1 over about -0.01 to 1.02. Axis words at the axis "
                 "end, with the rejection written inline, not as a fraction. The legend is in the first panel only, "
                 "lower left, as the curves fill the upper right; it holds for every panel. Landscape's grid, Sophon's "
                 "fonts, and the plan's tab10 order.",
         data_note="X→ee vs X→μμ has 5,000 background test jets against 200,000, so a run's curve stops at a rejection of "
                   "250 against 10,000. It has no curve for 188 or 162 classes: all six runs separate the pair completely "
                   "(AUC 1.0), so fewer than 20 background jets remain at every efficiency on the grid. The 43-class "
                   "curve starts at 0.145 and the 17-class curve at 0.54, where the first of their runs drops below 20 "
                   "jets (the other two continue to about 0.02 to 0.04). 17 classes lie above 43 because of one run, "
                   "the second 17-class run: at efficiency 0.8 the runs give 27.9, 131.6, 23.9 for 17 classes and 28.4, 29.2, 32.5 "
                   "for 43. The gap is one run, not the label set.",
         draw=draw_f4),
    Page(id="F4-app", title="Flavor separation with random partitions with controlled flavor boundaries",
         precedent=["1902.09914/fig5.png"],
         precedent_label="Landscape Fig. 5",
         caption="Background rejection against signal efficiency for four pairs of JetClass-II classes that differ only "
                 "in flavor, three resonant and one of QCD jets, from linear probes on frozen networks pretrained on "
                 "17-class random partitions with no flavor boundary, a b-quark boundary or a random flavor boundary, "
                 "with the 188-class model dashed. The first class named is the signal. Curves are means over two "
                 "runs, three for 188 classes, and end where fewer than 20 background jets remain.",
         differs="As F4, with three solid curves (random partitions with controlled flavor boundaries) and one dashed "
                 "reference in place of F4's four solid curves. Landscape has no such reference: a single black 1.0 pt "
                 "line (Sophon Fig. 5). "
                 "One panel differs from F4's: QCD bb vs cc in place of X→ee vs X→μμ, which has no curve here.",
         data_note="QCD bb vs cc replaces F4's X→ee vs X→μμ here: neither run of any controlled partition, and no run "
                   "of the 188-class model, keeps 20 background jets at any efficiency on that pair's grid, so its panel "
                   "would be empty.",
         draw=draw_f4_app),
    Page(id="F6", title="Benchmarks after fine-tuning",
         precedent=["1902.09914/fig5.png", "1902.08570/fig3.png", "1902.08570/fig4.png"],
         precedent_label="Landscape Fig. 5, ParticleNet Figs. 3-4",
         caption="Background rejection against signal efficiency after fine-tuning on the full training sets: top "
                 "tagging, and quark/gluon separation tested on Pythia and Herwig jets. Curves are networks "
                 "pretrained with 188, 162, 43 and 17 classes and the network trained from scratch, each the mean "
                 "over three runs; on quark/gluon they overlap at this scale. The dashed curve on top tagging is "
                 "one run of the 188-class model with kinematic inputs only. Curves end where fewer than 20 "
                 "background jets remain.",
         differs="Three stacked panels on one x axis (Sophon Fig. 3's arrangement), as landscape and ParticleNet draw "
                 "one per figure. Rejection on y as landscape, not ParticleNet's background efficiency; no AUC in the "
                 "legend, unlike ParticleNet (PI limit, numbers are in tables). The y axis starts at 1, not at the data; "
                 "landscape's grid, Sophon's font, 1.1 pt lines; tab10 in the plan's order. Five solid curves, and on top tagging "
                 "one black dashed 1.0 pt reference (Sophon Fig. 5), not landscape's fifteen in mixed line styles. Legend "
                 "framed, first panel only, lower left in two columns for its six entries, as curves fill the other "
                 "corners of that panel. Stacked panels are larger than Sophon's, with a wider gap: the y-axis words "
                 "are longer than Sophon's 'Events' and repeat on each panel.",
         data_note="The 188-class kinematic-inputs model has one run, so the dashed curve is that run, not a mean.",
         draw=draw_f6),
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
                 "of six series and off the line convention (one colour per series, 1.5 pt), not named in this brief, for "
                 "the PI to decide. No dashed diagonal. One-line titles, not two. Sophon's font and landscape's grid, not the "
                 "poster's serif font and faint grid. The family legend is in the first panel only, lower left, framed, "
                 "with no heading, and with lines of normal thickness. Four panels, not two.",
         draw=draw_a3r),
]
