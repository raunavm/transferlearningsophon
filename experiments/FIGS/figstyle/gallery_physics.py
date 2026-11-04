#!/usr/bin/env python3
"""Gallery mock-ups of the design schematic, the mass regression, the mass sculpting and the resonance
tagging figures, in style_v3 only, with Sophon's Fig. 2 redrawn first as the check of that style. Layouts:
docs/figstyle/PLAN.md. Data come through paper_figures' selection code (gallery_common); nothing is drawn at
import."""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from gallery_common import HERE, RUNS, Absent, I, M, P, Page, S

GROOMED = "Groomed mass"
MASS_ROWS = (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))     # a mass model and its label set's colour, name
NAMES = {**S.NAME, **{row: f"{S.NAME[base]}, mass regression" for row, base in MASS_ROWS}}
RATIO_ATLAS, RATIO_CMS = 0.44, 0.33     # lower axes over main axes: ATLAS 2024-015 Figs. 7-8, CMS Fig. 8 (catalogue)
STACK_GAP, COLUMN_GAP = 0.08, 0.32      # a main and its lower axes; two columns with their own y axes
SHOWN_FROM = 0.05       # a response axis covers the bins with this share of the largest bin of any series
HIST_ROOM = 1.45        # top of a histogram's y axis over its peak: room for the legend above the curves
SELECTION_COLOUR = {1: S.TAB10[7], 2: S.TAB10[9]}     # F10a, as CMS Fig. 8's: black inclusive; not the label sets' colours
MASS_17_COLOUR = S.TAB10[9]     # F10b: the 17-class model and its mass variant are on one axes
DIRECT_COLOUR = S.TAB10[7]      # F11: Sophon's grey for a discriminant built from the output scores
FITS = pathlib.Path("/data/results/eval/xbx/fits")      # the dry-run fits on the stand-in jets: cluster only
STANDIN = "xbb_jc2"


def _fit_y(ax, values: list, room: float = 0.0, pad: float = 0.08) -> None:
    """y limits from the finite values given. By hand since an ax.stairs line does not autoscale alike across
    matplotlib versions; room is empty space on top for a legend, as a share of the data's span."""
    v = np.concatenate([np.ravel(x) for x in values])
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    span = hi - lo or abs(hi) or 1.0
    ax.set_ylim(lo - pad * span, hi + (pad + room) * span)


# ------------------------------------------------------------------ the check of the style
def draw_sophon_fig2(path: pathlib.Path) -> None:
    """Sophon Fig. 2 from the points read off its native PDF, every setting from style_v3."""
    d = json.loads((HERE / "sophon_fig2_points.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH, S.HEIGHT[2]), sharex=True)
    for ax, p in zip(axes, d["panels"]):
        S.grid(ax)
        for c in p["curves"]:
            xy = np.array(c["points_xy"])
            ax.plot(xy[:, 0], xy[:, 1], color=c["colour"], linewidth=S.LW, linestyle=":" if c["dash_pdf"] else "-",
                    label=c["name_mathtext"])
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 5.4)
        ax.set_yticks(range(6))
        S.title(ax, p["title_mathtext"])
        S.labels(ax, ylabel=p["y"]["label"])
        S.legend(ax)
    S.labels(axes[1], xlabel=d["panels"][1]["x"]["label"])
    S.save(fig, path)


# ------------------------------------------------------------------ F1
# Schematic rows of the four label sets, top to bottom, not to scale: each level's boxes are unions of the boxes
# to its left, the last row group is the QCD jets
BOXES = {"L188": (1,) * 14, "L162": (1,) * 11 + (3,), "R42_Q1": (2, 2, 2, 2, 2, 1, 3), "R16_Q1": (4, 4, 3, 3)}
MERGES = ("QCD quark\ncontent", "quark counts,\ngluon vs quark,\n$e$ vs $\\mu$", "quark\nflavor")
TRAINED, FROZEN = "#cfe6f7", "#e6e6e6"
FAN = "#cccccc"
NEW_LAYER = "#9aa7e0"


def _network(ax, x: float, y: float, w: float, h: float, fill: str, words: str, size: float) -> list:
    """Sophon's network box with its feature column on the right edge; the column's node centres."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec="black", lw=0.8,
                                zorder=2))
    ax.text(x + (w - 0.12) / 2, y + h / 2, words, ha="center", va="center", fontsize=size, zorder=3)
    ax.add_patch(FancyBboxPatch((x + w - 0.14, y + 0.02), 0.14, h - 0.04, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec="black", lw=0.8, zorder=3))
    nodes = [(x + w - 0.07, v) for v in np.linspace(y + 0.12, y + h - 0.12, max(int(h / 0.14), 3))]
    for xy in nodes:
        ax.add_patch(Circle(xy, 0.04, fc="white", ec="black", lw=0.4, zorder=4))
    return nodes


def _use(ax, x: float, fill: str, word: str, action: str, name: str) -> None:
    """One use of the features in (b): the network, a new layer of two outputs, what is trained, the name."""
    nodes = _network(ax, x, 0.72, 0.92, 0.84, fill, word, S.LEGEND_PT)
    outs = [(x + 1.8, 0.92), (x + 1.8, 1.36)]
    for a in nodes:
        for b in outs:
            ax.plot(*zip(a, b), color=NEW_LAYER, linewidth=0.4, zorder=1.5)
    for xy in outs:
        ax.add_patch(Circle(xy, 0.05, fc="white", ec="black", lw=0.5, zorder=4))
    ax.text(x + 1.36, 1.14, action, ha="center", va="center", fontsize=S.LEGEND_PT, zorder=5,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#808080", lw=0.5))
    ax.text(x + 1.0, 0.36, name, ha="center", va="center", fontsize=S.LABEL_PT, zorder=5)


def draw_f1(path: pathlib.Path) -> None:
    w, h = S.FULL, 4.9     # inches: one data unit is one inch on the page
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    bold = dict(fontsize=S.LABEL_PT, fontweight="bold", va="center")
    ax.text(0, 4.72, "(a) Pretraining", **bold)
    ax.text(3.9, 4.68, "Four nested label sets, one pretraining each", ha="center", va="center", fontsize=S.LABEL_PT)
    # the jets go into the one network
    ax.add_patch(Polygon([(0.05, 3.82), (0.05, 3.2), (0.5, 3.5)], fc="#d0d0d0", ec="none", zorder=2))
    ax.text(0.28, 3.05, "same jets", ha="center", va="center", fontsize=S.LEGEND_PT)
    ax.annotate("", xy=(0.72, 3.5), xytext=(0.5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0))
    nodes = _network(ax, 0.72, 2.95, 1.14, 1.1, TRAINED, "One network", S.LABEL_PT)
    ax.text(0.72 + 0.51, 3.2, "same initial\nweights", ha="center", va="center", fontsize=S.LEGEND_PT, zorder=3)
    ax.text(1.79, 4.13, "features", ha="center", va="center", fontsize=S.LEGEND_PT)
    # the label sets: one column each, a merged class spans the rows it merges
    top, bottom, cw, pitch, x0 = 4.35, 2.75, 0.62, 0.9, 2.24
    rows = (top - bottom) / sum(BOXES["L188"])
    for k, (row, spans) in enumerate(BOXES.items()):
        x = x0 + k * pitch
        ax.text(x + cw / 2, top + 0.11, S.NAME[row], ha="center", va="center", fontsize=S.LABEL_PT)
        y = top
        for n in spans:
            ax.add_patch(Rectangle((x, y - n * rows), cw, n * rows, fc=to_rgba(S.SERIES[row], 0.25), ec="black", lw=0.5,
                                   zorder=2))
            y -= n * rows
        if spans[-1] > 1:
            ax.text(x + cw / 2, bottom + spans[-1] * rows / 2, "QCD", ha="center", va="center",
                    fontsize=S.LEGEND_PT, zorder=3)
    for a in nodes:
        for r in range(sum(BOXES["L188"])):
            ax.plot(*zip(a, (x0, top - (r + 0.5) * rows)), color=FAN, linewidth=0.3, zorder=1)
    # what each merge removes
    ax.text(x0 + 0.18, 2.45, "Each merge\nremoves:", ha="right", va="center", fontsize=S.LEGEND_PT)
    for k, words in enumerate(MERGES):
        x = x0 + k * pitch + cw / 2
        ax.annotate("", xy=(x + pitch, 2.69), xytext=(x, 2.69),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(x + pitch / 2, 2.63, words, ha="center", va="top", fontsize=S.LEGEND_PT)
    # the uses, on one card (Sophon's shadowed card and callout), since both start from the features
    ax.text(0, 2.02, "(b) Use", **bold)
    for dx, dy in ((0.05, 0.05), (0, 0)):
        ax.add_patch(Rectangle((0.25 + dx, 0.15 + dy), 5.4, 1.7, fc="#f0f0f0", ec="#808080", lw=0.6, zorder=0.5))
    tip, a, b = (1.79, 2.95), (1.3, 1.855), (2.2, 1.855)
    ax.add_patch(Polygon([tip, a, b], fc="#f0f0f0", ec="none", zorder=0.6))
    for corner in (a, b):
        ax.plot(*zip(tip, corner), color="#808080", linewidth=0.6, zorder=0.7)
    _use(ax, 0.45, FROZEN, "frozen", "Train a\nlinear layer", "Frozen features, linear probe")
    _use(ax, 3.2, TRAINED, "trained", "Train all\nweights", "Fine-tuning")
    S.save(fig, path)


# ------------------------------------------------------------------ F9a, F9b, A17
def _atlas() -> dict:
    if not I["atlas"]:
        raise Absent("no mass resolution files (head/atlas_res)")
    return I["atlas"]


def _response(ax, rx, atlas: dict, group: str, title: str) -> None:
    g = P._groomed(atlas, group)["hist"]
    e = np.asarray(g["edges"], float)
    gr = P._frac(g)
    enough = np.asarray(g["counts"], float) >= M.MASS_MIN_JETS
    ax.stairs(gr, e, baseline=None, label=GROOMED, **S.REFERENCE)
    rx.axhline(1.0, **S.REFERENCE)
    shown, ratios = [gr], []
    for row, base in MASS_ROWS:
        hs = [P._frac(b["hist"]) for m in RUNS[row] if (b := P._block(atlas, m, "regressed", group))]
        if not hs:
            continue
        mean = np.mean(hs, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(enough, mean / gr, np.nan)
        ax.stairs(mean, e, baseline=None, color=S.SERIES[base], linewidth=S.LW, label=NAMES[row])
        rx.stairs(r, e, baseline=None, color=S.SERIES[base], linewidth=S.LW)
        shown.append(mean)
        ratios.append(r)
    on = np.flatnonzero(np.any([s >= SHOWN_FROM * s.max() for s in shown], axis=0))
    ax.set_xlim(e[on[0]], e[on[-1] + 1])
    ax.set_ylim(0, HIST_ROOM * max(s.max() for s in shown))
    _fit_y(rx, [[1.0]] + [r[on] for r in ratios])
    S.title(ax, title)
    S.labels(ax, ylabel="Fraction of jets")
    S.labels(rx, xlabel=r"Response ($m/m_{\mathrm{gen}}$)", ylabel="Ratio")
    S.legend(ax)


def draw_f9a(path: pathlib.Path) -> None:
    atlas = _atlas()
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL, 3.4), sharex="col",
                             gridspec_kw=dict(height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP))
    _response(axes[0, 0], axes[1, 0], atlas, "resonant", "Resonant jets")
    _response(axes[0, 1], axes[1, 1], atlas, "qcd", "QCD jets")
    S.save(fig, path)


def _binned(axes, atlas: dict, group: str, axis: str, xlabel: str, legends: tuple, room: float = 0.0) -> None:
    """ATLAS Fig. 8's four axes: [[median, resolution], [its ratio, its root square difference]] to the groomed
    mass, in the bins of axis. legends: the columns with a legend, room above the curves for it."""
    (a0, a1), (r0, r1) = axes
    g = P._groomed(atlas, group)[axis]
    e = np.asarray(g["edges"], float)
    med0, res0 = P._arr(g["median"]), P._arr(g["resolution"])
    a0.stairs(med0, e, baseline=None, label=GROOMED, **S.REFERENCE)
    a1.stairs(res0, e, baseline=None, label=GROOMED, **S.REFERENCE)
    r0.axhline(1.0, **S.REFERENCE)
    r1.axhline(0.0, **S.REFERENCE)
    meds, ress, ratios, rsds = [med0], [res0], [[1.0]], [[0.0]]
    for row, base in MASS_ROWS:
        bl = [b[axis] for m in RUNS[row] if (b := P._block(atlas, m, "regressed", group))]
        if not bl:
            continue
        med = np.mean([P._arr(b["median"]) for b in bl], axis=0)
        res = np.mean([P._arr(b["resolution"]) for b in bl], axis=0)
        ratio = med / med0
        rsd = np.sign(res - res0) * np.sqrt(np.abs(res ** 2 - res0 ** 2))   # ATLAS 2024-015's root square difference
        kw = dict(baseline=None, color=S.SERIES[base], linewidth=S.LW)
        a0.stairs(med, e, label=NAMES[row], **kw)
        a1.stairs(res, e, label=NAMES[row], **kw)
        r0.stairs(ratio, e, **kw)
        r1.stairs(rsd, e, **kw)
        meds.append(med)
        ress.append(res)
        ratios.append(ratio)
        rsds.append(rsd)
    for col, ax, values in ((0, a0, meds), (1, a1, ress)):
        _fit_y(ax, values, room=room if col in legends else 0.0)
        if col in legends:
            S.legend(ax)
    _fit_y(r0, ratios)
    _fit_y(r1, rsds)
    for ax in (a0, a1, r0, r1):
        S.grid(ax)
    a0.set_xlim(e[0], e[-1])
    a1.set_xlim(e[0], e[-1])
    S.labels(a0, ylabel=r"Median response ($r$)")
    S.labels(a1, ylabel=r"Resolution ($\sigma_r$)")
    S.labels(r0, xlabel=xlabel, ylabel="Ratio")
    S.labels(r1, xlabel=xlabel, ylabel="RSD")


X_PT, X_MGEN = r"Jet $p_{\mathrm{T}}$ [GeV]", r"Generator mass ($m_{\mathrm{gen}}$) [GeV]"


def draw_f9b(path: pathlib.Path) -> None:
    atlas = _atlas()
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL, 3.4), sharex="col",
                             gridspec_kw=dict(height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP))
    _binned(axes, atlas, "all", "vs_pt", X_PT, legends=(0, 1), room=0.55)
    S.save(fig, path)


def _a17(path: pathlib.Path, group: str) -> None:
    """Median and resolution (columns) against jet pT and generator mass (rows); the legend once."""
    atlas = _atlas()
    fig = plt.figure(figsize=(S.FULL, 5.6))
    outer = fig.add_gridspec(2, 1, hspace=0.3)
    for k, (axis, xlabel) in enumerate((("vs_pt", X_PT), ("vs_mgen", X_MGEN))):
        inner = outer[k].subgridspec(2, 2, height_ratios=(1, RATIO_ATLAS), hspace=STACK_GAP, wspace=COLUMN_GAP)
        _binned(inner.subplots(sharex="col"), atlas, group, axis, xlabel, legends=(0,) if k == 0 else (), room=0.8)
    S.save(fig, path)


# ------------------------------------------------------------------ F10a, F10b
def _sculpt() -> dict:
    if not I["sculpt"]:
        raise Absent("no mass sculpting curves (head/curves_v2/*_sculpting_v2.json)")
    return I["sculpt"]


def _points(ax, x, y, half, marker: str, colour: str, hollow: bool, width: float, label: str | None = None) -> None:
    """CMS's points: a marker on the bin's centre and a horizontal bar across the bin, nothing joined."""
    ax.errorbar(x, y, xerr=half, fmt=marker, color=colour, markerfacecolor="white" if hollow else colour,
                markersize=3.5, elinewidth=width, capsize=0, label=label)


def draw_f10a(path: pathlib.Path) -> None:
    sc = _sculpt()
    shown = [(a, ms) for a in P.F8B_SHOWN if (ms := [m for m in RUNS[a] if m in sc])]
    if not shown:
        raise Absent("no mass sculpting curve of the models shown")
    fig, axes = plt.subplots(2, len(shown), figsize=(S.FULL, 3.4), sharex=True, sharey="row", squeeze=False,
                             gridspec_kw=dict(height_ratios=(1, RATIO_CMS), hspace=STACK_GAP, wspace=0.08))
    # the markers of CMS Fig. 8: open circle before the selection, filled circle, open square
    looks = (("o", True), ("o", False), ("s", True))
    peak, ratios = 0.0, [[1.0]]
    for (ax, rx), (a, ms) in zip(axes.T, shown):
        e = np.asarray(sc[ms[0]]["bins_jet_sdmass"], float)
        mid, half = (e[:-1] + e[1:]) / 2, np.diff(e) / 2
        before = np.asarray(sc[ms[0]]["hist_before"], float)
        before = before / before.sum()
        _points(ax, mid, before, half, looks[0][0], "#000000", looks[0][1], S.LW_THIN, "Inclusive")
        peak = max(peak, before.max())
        rx.axhline(1.0, **S.REFERENCE)
        for k, (wp, _) in enumerate(P.WPS, start=1):
            after = np.mean([np.asarray(sc[m]["working_points"][wp]["hist_after"], float)
                             / np.sum(sc[m]["working_points"][wp]["hist_after"]) for m in ms], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(before > 0, after / before, np.nan)
            label = f"$\\varepsilon_B = {100 * float(wp):g}\\%$"
            _points(ax, mid, after, half, looks[k][0], SELECTION_COLOUR[k], looks[k][1], S.LW_THIN, label)
            _points(rx, mid, r, half, looks[k][0], SELECTION_COLOUR[k], looks[k][1], S.LW_THIN)
            peak = max(peak, after.max())
            ratios.append(r)
        ax.set_xlim(e[0], e[-1])
        S.title(ax, NAMES[a])
        S.labels(rx, xlabel=r"Jet mass ($m_{\mathrm{SD}}$) [GeV]")
        S.legend(ax)
    axes[0, 0].set_ylim(0, 1.1 * peak)
    _fit_y(axes[1, 0], ratios)
    S.labels(axes[0, 0], ylabel="Fraction of jets")
    S.labels(axes[1, 0], ylabel="Ratio")
    S.save(fig, path)


def draw_f10b(path: pathlib.Path) -> None:
    sc = _sculpt()
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.HEIGHT[1]))
    S.grid(ax)
    colours = (S.SERIES["L188"], S.SERIES["R16_Q1"], MASS_17_COLOUR)
    drawn = 0
    for row, colour in zip(P.F8B_SHOWN, colours):
        have = [sc[m]["curve"] for m in RUNS[row] if m in sc]
        if not have:
            continue
        eff = np.array([c["eff"] for c in have[0]], float)
        mean = np.mean([[c["js_distance"] for c in h] for h in have], axis=0)
        ax.plot(eff, mean, color=colour, linewidth=S.LW, marker="o", markersize=3.5, label=NAMES[row])
        drawn += 1
    if not drawn:
        raise Absent("no mass sculpting curve of the models shown")
    ax.set_xscale("log")
    ax.set_yscale("log")
    S.labels(ax, r"QCD efficiency ($\varepsilon_B$)", r"Jensen-Shannon distance ($d_{\mathrm{JS}}$)")
    S.legend(ax)
    S.save(fig, path)


# ------------------------------------------------------------------ F11
def _standin_series(XR, fits: dict) -> list:
    """(label, colour, readouts per run) on the stand-in jets: the 188-class output scores (Sophon's direct
    discriminant, Eq. 4), then each label set's classifier on the frozen features (the fits' "head"; the linear
    probe where a fit has none)."""
    of = lambda row, key: [r for m in RUNS[row] if (r := XR.readout(fits.get(m, {}).get(STANDIN, {}), key))]
    out = []
    if direct := of("L188", "zero_shot_xbb"):
        out.append((f"{S.NAME['L188']}, direct discriminant", DIRECT_COLOUR, direct))
    for row in S.LEVELS:
        for key, word in (("head", "transfer"), ("linear", "linear probe")):
            if rs := of(row, key):
                out.append((f"{S.NAME[row]}, {word}", S.SERIES[row], rs))
                break
    return out


def draw_f11(path: pathlib.Path) -> None:
    if not FITS.is_dir():
        raise Absent(f"the stand-in fits ({FITS}) are on the cluster only")
    XR = P._xr()
    series = _standin_series(XR, XR.load(FITS))
    if not series:
        raise Absent(f"no {STANDIN} readout of the label sets under {FITS}")
    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH, S.HEIGHT[2]), sharex=True)
    top = 0.0
    for label, colour, rs in series:
        x, y, _ = P.xbx_curves(M, rs)
        mean = y.mean(axis=0)
        axes[0].plot(x, mean, color=colour, linewidth=S.LW, label=label)
        top = max(top, float(np.nanmax(mean)))
    S.legend(axes[0])
    for ax, what in zip(axes, (r"$X\rightarrow bb$ tagging performance, stand-in jets",
                               r"$X\rightarrow bs$ tagging performance")):
        S.grid(ax)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 1.25 * top)
        S.title(ax, what)
        S.labels(ax, ylabel=r"Figure of merit ($\varepsilon_S/\sqrt{\varepsilon_B}$)")
    S.labels(axes[1], xlabel=r"QCD efficiency ($\varepsilon_B$)")
    S.save(fig, path)


# ------------------------------------------------------------------ the pages
SOPHON_FIG2_CAPTION = (
    "FIG. 2. Benchmark of Sophon model’s performance in the X→bb and X→bs jet tagging tasks, with the signal "
    "originated from a BSM resonance X0 with mX0 = 200 GeV that decays to bb and bs, and backgrounds corresponding "
    "to the 40 fb−1 of the full SM processes. The performance of various DNN models is compared in terms of the "
    "search’s discovery significance versus SM background efficiency calculated within the mass window "
    "150 < mSD < 230 GeV. A major conclusion is that the Sophon model’s direct tagging discriminant (for X→bb "
    "tagging) and its transfer learning version (for X→bs tagging) both outperform the current best results "
    "achievable in the LHC experiment using a ParT or ParticleNet tagger. It also confirms that the model improves "
    "performance when trained by large-scale classification. [Validation page, not one of our figures: Sophon's "
    "own caption, verbatim.]")
F1_CAPTION = (
    "Design of the study. (a) One network is pretrained on the same jets four times, once for each of four nested "
    "label sets with 188, 162, 43 and 17 classes; below each merge is the distinction it removes. (b) The pretrained "
    "network is then used in two ways: its frozen features feed a linear probe, or all of its weights are fine-tuned.")
F9A_CAPTION = (
    "Distribution of the jet mass response, the ratio of a jet's mass to its generator-level mass, for resonant jets "
    "(left) and QCD jets (right): the groomed mass and the mass regression output of the models pretrained on 162 and "
    "17 classes, each the mean over runs. The lower panels show the ratio of each distribution to the groomed mass.")
F9B_CAPTION = (
    "Median jet mass response (left) and relative mass resolution (right) against the jet transverse momentum, for all "
    "jets: the groomed mass and the mass regression output of the models pretrained on 162 and 17 classes, each the "
    "mean over runs. Below are the ratio of each median to the groomed mass and the root square difference (RSD) of "
    "each resolution to the groomed mass.")
A17_CAPTION = (
    "Median jet mass response (left) and relative mass resolution (right) for {group} jets, against the jet transverse "
    "momentum (upper) and the generator-level jet mass (lower): the groomed mass and the mass regression output of the "
    "models pretrained on 162 and 17 classes, each the mean over runs. Below each panel are the ratio of the median "
    "and the root square difference (RSD) of the resolution to the groomed mass.")
F10A_CAPTION = (
    "Soft-drop mass distribution of QCD jets before any selection (inclusive) and after selections at QCD efficiencies "
    "of 10 and 1 per cent on the score of the model pretrained on 188 classes (left), 17 classes (centre) and 17 "
    "classes with mass regression (right), each the mean over runs. Horizontal bars show the bin width. The lower "
    "panels show the ratio to the inclusive distribution.")
F10B_CAPTION = (
    "Jensen-Shannon distance between the soft-drop mass distributions of QCD jets before and after a selection on the "
    "model's score, against the QCD efficiency of the selection, for the models pretrained on 188 classes, on 17 "
    "classes and on 17 classes with mass regression. Each point is the mean over runs.")
F11_CAPTION = (
    "Signal efficiency over the square root of the background efficiency, against the QCD efficiency, for X→bb on "
    "stand-in JetClass-II jets (upper) and for X→bs (lower), which has no samples yet. Curves show the direct "
    "discriminant of the 188-class model and a classifier on the frozen features of each label set, each the mean "
    "over runs; a curve ends where fewer than 20 background jets remain.")

ATLAS_LOOK = "Sophon's font, sizes, ticks and framed legend in place of ATLAS's; no experiment label"
A17_DIFFERS = (
    "As F9b, on one jet group and with a second row against generator mass, which ATLAS does not draw (its Fig. 9 has "
    "pT response instead); the legend appears once, in the first panel, to keep four panels light. Eight axes in all, a lower axes under each "
    "of four panels: within the limit of four panels only if lower axes do not count, for the PI to decide. 5.6 in high, not "
    "the 4.8 in of a plain 2x2, because each panel carries a lower axes. " + ATLAS_LOOK + ".")
MASS_RUNS_NOTE = ("Mean of two runs for each mass regression model (the third runs were cut on 2026-10-22); "
                  "the groomed mass is the same in every file. Nothing else is absent.")

PAGES = [
    Page("SOPHON-FIG2", "Sophon Fig. 2 redrawn through style_v3 (validation)", ["2405.12972/fig2.png"], "Sophon Fig. 2",
         SOPHON_FIG2_CAPTION,
         "Nothing but the file encoding (pdf.fonttype 42, embedded TrueType) and the data, which are the points read "
         "off Sophon's native PDF rather than its original arrays. It breaks our own rules on purpose (dotted series, "
         "long caption), because this page checks the style against Sophon, not the figure.",
         draw_sophon_fig2,
         "The points are those of the native PDF after its path simplification; a curve starts where the PDF's does."),
    Page("F1", "Design of the study", ["2405.12972/fig1.png"], "Sophon Fig. 1", F1_CAPTION,
         "Our content in Sophon's shape: (a) above (b), jets into a network box with its feature column, grey fan, "
         "shadowed card and callout. Four output columns instead of Sophon's one, with a merged class spanning the "
         "rows it merges, to show the nesting; rows are schematic, not to scale. One card with two uses and one callout "
         "instead of two cards, because both uses start from the features (Sophon's discriminants start from the output "
         "layer). Drawn with matplotlib patches in STIX instead of a vector drawing with jet icons, and "
         "no latent or output size formulae, which belong to the methods.",
         draw_f1),
    Page("F9a", "Mass regression: response", ["ATL-PHYS-PUB-2024-015/fig7.png"], "ATLAS 2024-015 Fig. 7", F9A_CAPTION,
         "Two jet groups instead of mass and pT responses, as we regress mass only. Three series instead of two: "
         "black dashed reference and two solid colours, instead of purple dashed and green. Fraction of jets on the "
         "y axis instead of a.u. x range from the data, per panel, instead of a fixed 0.7 to 1.3, because the groomed "
         "mass here is far wider. No vertical dotted line at 1, a second reference the PI limit forbids. No bands: "
         "ATLAS's are thinner than its lines. Same ratio height (0.44). " + ATLAS_LOOK + ".",
         draw_f9a, MASS_RUNS_NOTE),
    Page("F9b", "Mass regression: across jet pT", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8", F9B_CAPTION,
         "All jets on our bins: four unequal pT bins to 2500 GeV in reconstructed pT, instead of ATLAS's finer bins to 1500 "
         "GeV in truth pT. Connected steps instead of one disconnected segment per bin, and no bands (PI limit), though "
         "ATLAS draws them. Three series instead of two, colours as F9a. No dotted line at 1 in the median panel. Dashed "
         "grid at the ticks, our rule for curve figures, where ATLAS has none. The lower panels keep ATLAS's ratio "
         "and its RSD, sgn(s'-s) sqrt|s'^2-s^2|, at its height (0.44). " + ATLAS_LOOK + ".",
         draw_f9b, MASS_RUNS_NOTE),
    Page("A17-res", "Mass regression on resonant jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         A17_CAPTION.format(group="resonant"), A17_DIFFERS, lambda path: _a17(path, "resonant"), MASS_RUNS_NOTE),
    Page("A17-qcd", "Mass regression on QCD jets", ["ATL-PHYS-PUB-2024-015/fig8.png"], "ATLAS 2024-015 Fig. 8",
         A17_CAPTION.format(group="QCD"), A17_DIFFERS, lambda path: _a17(path, "qcd"), MASS_RUNS_NOTE),
    Page("F10a", "Mass sculpting: QCD mass", ["2604.09809/fig8.png"], "CMS JME-25-001 Fig. 8", F10A_CAPTION,
         "Three panels side by side (three models, all jet pT) instead of CMS's six (two models, three pT ranges), with "
         "one y axis across them. Two working points instead of three, because the files hold 10% and 1%. Mass from 20 "
         "to 500 GeV in 20 GeV bins, the files' range, instead of CMS's 30 to 250 GeV. Colours are "
         "black, grey and cyan instead of black, blue, red, orange (colours that mean no label set elsewhere in the paper). "
         "Markers as CMS's first three. No vertical error bars, as the histograms are means over runs and seed spread "
         "lives in tables. Bars at 1.0 pt instead of 0.4. Dashed reference at 1 instead of solid, and \"Ratio\" instead "
         "of a stacked fraction. Same ratio height (0.33). Fraction of jets instead of A.U., titles instead of a label "
         "block, STIX at 10 and 8 pt with outward ticks instead of Helvetica at 6.9 and 5.8 pt with inward ones.",
         draw_f10a,
         "Mean of two runs for the 17-class model with mass regression, three for the others. Nothing else is absent."),
    Page("F10b", "Mass sculpting: how much", ["2604.09809/fig9.png"], "CMS JME-25-001 Fig. 9 left", F10B_CAPTION,
         "One panel: CMS's right panel (against pT) is not drawn, since our curves hold all jet pT. Sixteen efficiencies "
         "on a log axis rising to the right, the files' grid and the direction of the efficiency axis in Sophon Fig. 2 "
         "and F11, instead of five categories falling to the right. Three series instead of two, one filled circle "
         "in each colour instead of an open square and a filled circle. Markers joined by a 1.5 pt line, not the plan's "
         "step markers: CMS's segments each span one column of a categorical axis, ours would span a bin that does not "
         "exist, since each point is a cumulative selection at one efficiency (for the PI to decide). Dashed grid at the ticks (our rule for curve figures), where CMS has none. Limits from the "
         "data instead of 0.007 to 1. No random-selection band or floor: CMS has none (plan). Framed legend and no "
         "label block; STIX at 10 and 8 pt with outward ticks.",
         draw_f10b,
         "Mean of two runs for the 17-class model with mass regression, three for the others. Nothing else is absent."),
    Page("F11", "Resonance tagging", ["2405.12972/fig2.png"], "Sophon Fig. 2", F11_CAPTION,
         "Same layout, limits on x, grid, fonts and legend place. y axis eps_S/sqrt(eps_B), in words figure of merit, instead of the "
         "significance Z, and x axis QCD efficiency instead of SM background efficiency, because Z needs the SM backgrounds, which "
         "we do not have. Upper panel on stand-in jets from JetClass-II, not Sophon's samples. Lower panel empty, "
         "not invented curves. Roles as far as the data have them: Sophon's direct discriminant "
         "becomes the 188-class one, in Sophon's grey for a discriminant from the output scores; Sophon's 42-class curve "
         "becomes the four label sets' transfer classifiers; ParT and ParticleNet, trained from scratch, have no "
         "counterpart. Solid lines only (PI limit), where Sophon's are dotted for two series. y limits from the data, "
         "not 0 to 5.4. Curves have 26 points or fewer (the fits' efficiency grid) and end at 20 background jets, so "
         "they are coarser and shorter than Sophon's.",
         draw_f11,
         "Stand-in only: X→bb from JetClass-II jets, /data/results/eval/xbx/fits (cluster), so the page is absent off "
         "the cluster. X→bs: no samples exist, the panel is empty. The significance Z is not computed: it needs the SM "
         "backgrounds. From scratch is not drawn: only a two-epoch dry run exists."),
]
