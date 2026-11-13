#!/usr/bin/env python3
"""The per-class family (plan "Per class", PI brief 2026-11-12/13, L10): every class's AUC against the rest for a
compared network (vertical) against its reference network (horizontal), one point per class, the mean over runs, the
diagonal a black dotted reference line, one axis range per grid; two rows, linear probe over fine-tuned, written turned
on the left as in Fig. 2. Slices: the label sets (162, 43, 17 classes against 188; JetClass-II in the main text,
JetClass in the appendix), the 17-class partitions against 188 and the mass networks against their own label set
(supplement), and the JetClass confusion matrices in the same three slices (supplement).

One colour rule for the whole family (L6): a point is coloured by what it is, never by network. On JetClass-II its
decay family, on JetClass its class, from style_v5.CLASS10, which reuses no network colour (palette_check); the network
is named in the panel title. Sources: linear probes of JetClass-II from the v4 fits (mlst_results._per_class), fine-tuned
JetClass-II from the fine-tuning read-out the paper reads (paper_figures.inputs "per", mlst_results._ft: the same
values as curves_v2/jc2_ft_ovr where both hold a run, plus the mass networks), JetClass from curves_v2 jc1_probe_ovr and
jc1_ft_ovr (tab:jc1_class_auc's source), the confusion matrices from confusion_jc1 (experiments/EVAL/perf_vs_pt.py)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

import gallery_common as C
from gallery_roc import FAMILIES, FAMILY_COLOURS, family_label, family_of
import style_v5 as S

P, M, RUNS, I = C.P, C.M, C.RUNS, C.I
POOLED = M._pooled(RUNS)
N_FT = 1_000_000

# the slices: (compared, reference) per column
LABEL_SETS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))
PARTITIONS = (("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"), ("RAND", "L188"))
# L20: four rows of two panels overrun the page, so two figures: the named flavor boundaries (-part) and the random
# controls, the random flavor boundary and the random partitions (-rand)
MASS = (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
MODES = ("Linear probe", "Fine-tuned")
# panel width (cm) by rows of networks (L20, points only: at least 4.5 cm): two rows at a two-column figure's width,
# three as wide as the page's height allows (make_results_book.MAX_H_CM)
PANEL = {2: 6.3, 3: 4.6}
DOT = {2: 3.0, 3: 2.6}                  # pt: 188 points in a panel stay apart
JC1_DOT = 4.0
JC1_SHAPES = ("o", "s", "^", "v", "D", "P", "X", "p", "h", "*")    # with the colour, so the ten classes part in greyscale
LEFT = 1.3                               # cm: the y label and the tick labels
WGAP = 0.35                              # cm between panels of a row: a tick label at AUC 1 clears the next panel
TITLE_LINE = 0.4                         # cm: a second title line (four columns: a partition's name in two lines)
HGAP = 0.95                              # cm: the upper row's x tick labels are shared, so only the next titles
XLABEL = 0.9                             # cm: x tick labels and label under each row (rows with their own reference)
ABSENT = "no class-against-rest AUCs of the 188-class network"


# ------------------------------------------------------------------ the values: {class: AUC} of one run
def _probe_jc2(m: str):
    return M._per_class(I["fits"], m)


def _ft_jc2(m: str):
    p = M._ft(I["per"], "jc2", m, N_FT, "per_class_auc")
    return None if p is None else {int(c): v for c, v in p.items() if v is not None}


def _jc1(family: str):
    docs = C.ROC.get(family) or {}
    return lambda m: {c: v["auc"] for c, v in docs[m]["curves"].items()} if m in docs else None


SOURCES = {"jc2": (_probe_jc2, _ft_jc2), "jc1": (_jc1("jc1_probe_ovr"), _jc1("jc1_ft_ovr"))}


def _mean(get, row: str) -> dict:
    """{class: AUC, the mean over the row's runs that hold it} (the five random partitions pooled)."""
    per = [p for m in POOLED[row] if (p := get(m))]
    return {c: float(np.mean([p[c] for p in per if c in p])) for c in set().union(*per)} if per else {}


# ------------------------------------------------------------------ the scatter grids
def _look(test: str):
    """(class -> (colour, marker, size, zorder), legend handles) of the family's one colour rule."""
    if test == "jc1":
        def look(c):
            k = S.JC1_CLASSES.index(c)
            return S.CLASS10[k], JC1_SHAPES[k], JC1_DOT, 3
        handles = [Line2D([], [], color=S.CLASS10[k], marker=JC1_SHAPES[k], markersize=JC1_DOT, linestyle="none",
                          label=S.JC1_TEX[c]) for k, c in enumerate(S.JC1_CLASSES)]
        return look, handles, 6
    family = {int(r["jet_label"]): family_of(r["class_name"]) for r in M._class_rows(P.REPO)}
    size = {f: sum(v == f for v in family.values()) for f in FAMILIES}

    def look(c):
        k = FAMILIES.index(family[c])
        return FAMILY_COLOURS[k], "o", None, 3.1 if size[family[c]] < 10 else 3     # a small family over the large ones
    handles = [Line2D([], [], color=FAMILY_COLOURS[k], marker="o", markersize=3, linestyle="none", label=family_label(f))
               for k, f in enumerate(FAMILIES)]
    return look, handles, 3


def _draw_grid(path, test: str, rows: tuple) -> None:
    """One row per (compared, reference) pair, linear probe and fine-tuned beside each other (L20: points-only panels at
    least 4.5 cm, so the networks are rows and the two modes the columns), on one axis range: the lowest AUC of any point
    and the points at AUC 1 clear of the frame. Rows with one reference share the x labels under the lowest row."""
    gets = SOURCES[test]
    if not all(_mean(get, "L188") for get in gets):
        raise C.Absent(ABSENT)
    look, handles, ncol = _look(test)
    handles = handles + [Line2D([], [], **S.REFERENCE, label="equal AUC")]
    n = len(rows)
    w = PANEL[n]
    pts = [[[(ref[c], cmp_[c], c) for c in ref if c in cmp_] for cmp_, ref in ((_mean(get, a), _mean(get, b)) for get in gets)]
           for a, b in rows]
    lo = min(v for row in pts for panel in row for x, y, _ in panel for v in (x, y))
    lim = (lo - 0.08 * (1 - lo), 1.0 + 0.08 * (1 - lo))      # the points clear of the frames, the lowest too (L7)
    ncol = math.ceil(len(handles) / 2)                       # two legend rows: three rows of panels fit the page
    one_ref = len({b for _, b in rows}) == 1
    two = any(len(S.LEGEND[a]) > 16 for a, _ in rows)        # a partition's name and the mode on two title lines
    row_cm = 0.5 if test == "jc1" else 0.45          # a legend row of class names in mathtext is taller
    top = S.LEGEND_TOP + row_cm * 2 + 0.6 + (TITLE_LINE if two else 0)
    hgap = HGAP + (TITLE_LINE if two else 0) + (0 if one_ref else XLABEL)
    fig = S.figure_cm(S.PRINT_W, top + n * w + (n - 1) * hgap + 1.0)
    x0 = (S.PRINT_W - LEFT - 2 * w - WGAP) / 2 + LEFT       # the block of labels and panels centred
    span = lim[1] - lim[0]
    # the finest round step whose labels fit (0.8 cm each) and whose first tick clears the previous panel's last label
    step = next(t for t in (0.02, 0.04, 0.05, 0.1)
                if span / t * 0.8 <= w and (math.ceil(lim[0] / t - 1e-9) * t - lim[0]) / span * w >= 0.3)
    dot = DOT[n]
    for i, (row, (a, b)) in enumerate(zip(pts, rows)):
        y0 = top + i * (w + hgap)
        last = i == n - 1 or not one_ref
        for j, (panel, mode) in enumerate(zip(row, MODES)):
            ax = S.axes_cm(fig, x0 + j * (w + WGAP), y0, w, w)
            S.fine_grid(ax)
            ax.plot(lim, lim, **S.REFERENCE)
            for x, y, c in panel:
                colour, marker, size, z = look(c)
                ax.plot(x, y, color=colour, marker=marker, markersize=size or dot, linestyle="none", clip_on=False,
                        zorder=z, markeredgewidth=0.5 if marker in "+x*" else 0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(ticker.MultipleLocator(step))
            ax.tick_params(labelleft=j == 0, labelbottom=last)
            S.title(ax, f"({'abcdefgh'[2 * i + j]}) {S.LEGEND[a]},{chr(10) if two else ' '}{mode.lower()}")
            if j == 0:
                S.labels(ax, ylabel="AUC, the panel's network")
            if last:
                S.labels(ax, xlabel=f"AUC, {S.LEGEND[b]}")
    # the six decay families and the diagonal in two rows across 15 cm: closer columns
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=ncol, columnspacing=0.8, handletextpad=0.4)
    S.save(fig, path)


def draw_jc2(path) -> None:
    _draw_grid(path, "jc2", LABEL_SETS)


def draw_jc1(path) -> None:
    _draw_grid(path, "jc1", LABEL_SETS)


def draw_jc2_part(path) -> None:
    _draw_grid(path, "jc2", PARTITIONS[:2])


def draw_jc1_part(path) -> None:
    _draw_grid(path, "jc1", PARTITIONS[:2])


def draw_jc2_rand(path) -> None:
    _draw_grid(path, "jc2", PARTITIONS[2:])


def draw_jc1_rand(path) -> None:
    _draw_grid(path, "jc1", PARTITIONS[2:])


def draw_jc2_mass(path) -> None:
    _draw_grid(path, "jc2", MASS)


def draw_jc1_mass(path) -> None:
    _draw_grid(path, "jc1", MASS)


# ------------------------------------------------------------------ the JetClass confusion matrices
CONF = P.DATA / "confusion_jc1"
CONF_DIFF = (LABEL_SETS, PARTITIONS, MASS)       # every difference of the three pages: one colour scale for all


def _matrix(doc: dict) -> np.ndarray:
    """Row-normalised [true][predicted] in S.JC1_CLASSES order."""
    k = [doc["classes"].index(c) for c in S.JC1_CLASSES]
    rows = doc["row_normalised"]
    return np.array([[np.nan] * len(k) if rows[i] is None else [rows[i][j] for j in k] for i in k], float)


def _seed(model: str) -> str:
    return model.rsplit("-s", 1)[1]


def _diff(docs: dict, a: str, b: str):
    """100 x the mean over runs of (run of a - run of b with the same pretraining seed), or None without a pair."""
    ref = {_seed(m): _matrix(docs[m]) for m in POOLED[b] if m in docs}
    d = [_matrix(docs[m]) - ref[_seed(m)] for m in POOLED[a] if m in docs and _seed(m) in ref]
    return 100 * np.mean(d, axis=0) if d else None


def _draw_conf(path, pairs: tuple, base: bool, bar_word: str) -> None:
    """2 x 2 (base: the 188-class matrix and three differences) or one row of differences, 6.5 cm squares; the
    colour bars run horizontally above the panels, one per column with the base matrix, one across the row otherwise.
    One difference scale over every page of the slice (CONF_DIFF); no numbers in the cells (L7)."""
    docs = M._load(CONF)
    if not any(m in docs for m in RUNS["L188"]):
        raise C.Absent("no JetClass confusion matrices of the 188-class network (confusion_jc1)")
    every = [d for cols in CONF_DIFF for a, b in cols if (d := _diff(docs, a, b)) is not None]
    limit = max(float(np.nanmax(np.abs(d))) for d in every)
    n = len(S.JC1_CLASSES)
    names = [S.JC1_TEX[c] for c in S.JC1_CLASSES]
    w, left, gap, vgap = 6.5, 1.75, 0.15, 0.55
    cb_top, top = (1.4, 2.3) if base else (1.0, 1.9)       # a colour-bar label of two lines or one
    slots = (1 if base else 0) + len(pairs)
    nrows = math.ceil(slots / 2)
    fig = S.figure_cm(S.PRINT_W, top + nrows * w + (nrows - 1) * vgap + 2.25)
    xs = (left, left + w + gap)
    panels = [S.axes_cm(fig, xs[k % 2], top + (k // 2) * (w + vgap), w, w) for k in range(slots)]
    images = []
    if base:
        mean = 100 * np.mean([_matrix(docs[m]) for m in RUNS["L188"] if m in docs], axis=0)
        images.append(panels[0].imshow(mean, cmap="Blues", vmin=0, vmax=100, interpolation="none"))
        S.title(panels[0], S.LEGEND["L188"])
    for ax, (a, b) in zip(panels[1:] if base else panels, pairs):
        S.title(ax, f"{S.LEGEND[a]} minus {S.LEGEND[b].removesuffix(' classes') if b == 'L188' else S.LEGEND[b]}")
        d = _diff(docs, a, b)
        if d is not None:
            images.append(ax.imshow(d, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none"))
    for k, ax in enumerate(panels):
        last = k + 2 >= slots
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names if last else [], rotation=90)
        ax.set_yticklabels(names if k % 2 == 0 else [])
        ax.tick_params(which="both", length=0, top=False, right=False)
        ax.minorticks_off()
        S.labels(ax, "Predicted class" if last else None, "True class" if k % 2 == 0 else None)
    bars = ([(xs[0], w, images[0], "(a) fraction of the true class\n(percent)"),
             (xs[1], w, images[-1], f"(b)–(d) {bar_word}\n(percentage points)")] if base
            else [(xs[0], 2 * w + gap, images[-1], f"{bar_word} (percentage points)")])
    for x, width, image, word in bars:
        cax = S.axes_cm(fig, x, cb_top, width, 0.28)
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.ax.minorticks_off()
        if image is images[0] and base:
            cb.set_ticks([0, 20, 40, 60, 80])       # no "100" at the end, where it would meet the other bar
        else:                                   # one difference scale, the same ticks on every page (CONF_DIFF)
            cb.set_ticks([t for t in range(-int(limit), int(limit) + 1)])
        cb.set_label(word, fontsize=S.LABEL_PT5, labelpad=4)
    S.letter_panels(panels)
    S.save(fig, path)


def draw_conf(path) -> None:
    _draw_conf(path, LABEL_SETS, True, "difference from 188 classes")


def draw_conf_part(path) -> None:
    _draw_conf(path, PARTITIONS, False, "difference from 188 classes")


def draw_conf_mass(path) -> None:
    _draw_conf(path, MASS, False, "difference from the network's label set")


# ------------------------------------------------------------------ the pages
HEATMAPS = ["2604.09809/fig7.png", "2606.14813/fig12.png"]
HEAT_LABEL = "None for this form; CMS Fig. 7 and JetParticle-JEPA Fig. 12 are the dropped heatmap's"
SCATTER_DIFFERS = ("One layout for the family (PI 2026-11-12; L20 2026-11-13: networks as rows): linear probe beside "
                   "fine-tuned, a row per compared network, one axis range per grid; points coloured by decay family (JetClass-II) or class "
                   "(JetClass) in colours that mean no network (L6).")
RUN_NOTE = ("Means over runs: three per label set, two per mass network and flavor-boundary partition, the five random "
            "partitions one run each (pooled).")
CONF_DIFFERS = ("Ten classes instead of 34; the 188-class matrix in CMS's Blues, every other panel a difference on one "
                "diverging RdBu_r scale shared by the three confusion pages; no numbers in the cells (L7).")
CONF_NOTE = ("Each run against the run of its reference with the same pretraining seed. From scratch is not drawn; its "
             "full-schedule matrices (confusion_jc1_fulldecay) are in tab:jc1conf.")


def _page(pid, title, draw, precedent=HEATMAPS, label=HEAT_LABEL, differs=SCATTER_DIFFERS, note=RUN_NOTE) -> C.Page:
    return C.Page(id=pid, title=title, precedent=precedent, precedent_label=label, caption="", differs=differs,
                  draw=draw, data_note=note)


PAGES = [
    _page("PC-jc2", "Per class, JetClass-II, label sets", draw_jc2),
    _page("PC-jc1", "Per class, JetClass, label sets", draw_jc1),
    _page("PC-jc2-part", "Per class, JetClass-II, partitions with no and a b-quark flavor boundary", draw_jc2_part),
    _page("PC-jc1-part", "Per class, JetClass, partitions with no and a b-quark flavor boundary", draw_jc1_part),
    _page("PC-jc2-rand", "Per class, JetClass-II, random flavor boundary and random partitions", draw_jc2_rand),
    _page("PC-jc1-rand", "Per class, JetClass, random flavor boundary and random partitions", draw_jc1_rand),
    _page("PC-jc2-mass", "Per class, JetClass-II, mass networks against their label set", draw_jc2_mass),
    _page("PC-jc1-mass", "Per class, JetClass, mass networks against their label set", draw_jc1_mass),
    _page("PC-conf", "JetClass confusion matrices, label sets", draw_conf, ["2604.09809/fig7.png"],
          "CMS JME-25-001 Fig. 7", CONF_DIFFERS, CONF_NOTE),
    _page("PC-conf-part", "JetClass confusion matrices, 17-class partitions", draw_conf_part,
          ["2604.09809/fig7.png"], "CMS JME-25-001 Fig. 7", CONF_DIFFERS, CONF_NOTE),
    _page("PC-conf-mass", "JetClass confusion matrices, mass networks against their label set", draw_conf_mass,
          ["2604.09809/fig7.png"], "CMS JME-25-001 Fig. 7", CONF_DIFFERS, CONF_NOTE),
]

# the supplement members' captions, filled like the paper's (gallery_captions holds the main-text and appendix ones)
SUPPLEMENT_CAPTIONS = {
    "PC-jc2-part": (
        "The networks pretrained with the 17-class partitions with no and with a b-quark flavor boundary in the form of "
        "figure {fig:PC-jc2}. Each point is one JetClass-II class, at its area under the receiver operating "
        "characteristic curve (AUC) for separating it from all other classes, with (a, b) no flavor boundary and "
        "(c, d) a b-quark boundary (vertical) against 188 classes (horizontal), as linear probes (left column) or "
        "fine-tuned (right column) on one million class-balanced JetClass-II jets. Points below the dotted diagonal "
        "are classes the partition separates less well, and colors are decay families. Each point is the mean over "
        "two pretraining runs, and the 188-class values over three runs. Fine-tuning on JetClass-II ran with a cap of "
        "160 epochs for every partition and of 40 epochs for 188 classes. Figure {fig:PC-jc2-rand} shows the random "
        "flavor boundary and the random partitions. The points are in table {tab:perclass} and their averages in "
        "table {tab:summaryauc}."
    ),
    "PC-jc2-rand": (
        "The panels of figure {fig:PC-jc2-part} for (a, b) the 17-class partition with a random flavor boundary and "
        "(c, d) the five random 17-class partitions, each point one JetClass-II class at its area under the receiver "
        "operating characteristic curve (AUC) for separating it from all other classes (vertical) against 188 classes "
        "(horizontal), as linear probes (left column) or fine-tuned (right column) on one million class-balanced "
        "JetClass-II jets. Points below the dotted diagonal are classes the partition separates less well, and colors "
        "are decay families. Each point is the mean over two pretraining runs, or over the five random partitions "
        "with one run each, and the 188-class values over three runs. Fine-tuning on JetClass-II ran with a cap of "
        "160 epochs for every partition and of 40 epochs for 188 classes. The points are in table {tab:perclass} and "
        "their averages in table {tab:summaryauc}."
    ),
    "PC-jc1-part": (
        "The networks pretrained with the 17-class partitions with no and with a b-quark flavor boundary in the form of "
        "figure {fig:PC-jc1}. Each point is one JetClass class, at its area under the receiver operating "
        "characteristic curve (AUC) for separating it from the other nine, with (a, b) no flavor boundary and (c, d) a "
        "b-quark boundary (vertical) against 188 classes (horizontal), as linear probes (left column) or fine-tuned "
        "(right column) on one million JetClass jets. Each point is the mean over two pretraining runs, and the "
        "188-class values over three runs. Figure {fig:PC-jc1-rand} shows the random flavor boundary and the random "
        "partitions, and values are in table {tab:jc1_class_auc}."
    ),
    "PC-jc1-rand": (
        "The panels of figure {fig:PC-jc1-part} for (a, b) the 17-class partition with a random flavor boundary and "
        "(c, d) the five random 17-class partitions, each point one JetClass class at its area under the receiver "
        "operating characteristic curve (AUC) for separating it from the other nine (vertical) against 188 classes "
        "(horizontal), as linear probes (left column) or fine-tuned (right column) on one million JetClass jets. Each "
        "point is the mean over two pretraining runs, or over the five random partitions with one run each, and the "
        "188-class values over three runs. Values are in table {tab:jc1_class_auc}."
    ),
    "PC-jc2-mass": (
        "The networks pretrained with an added mass regression against the same label set without it, in the form of "
        "figure {fig:PC-jc2}. Each point is one JetClass-II class, at its area under the receiver operating characteristic curve (AUC) for separating it "
        "from all other classes, with (a, b) 162 classes + mass against 162 classes and (c, d) 17 classes + mass "
        "against 17 classes, as linear probes (left column) or fine-tuned (right column) on one million class-balanced "
        "JetClass-II jets. Points below the dotted diagonal are classes the mass network separates less well, and "
        "colors are decay families. Each point is the mean over two pretraining runs for a mass network and three for "
        "its label set. Fine-tuning on JetClass-II ran with a cap of 40 epochs, except the first run of 17 classes "
        "and both runs of 17 classes + mass, which had the 160-epoch cap. The points are in table {tab:perclass} and "
        "their averages in table {tab:summaryauc}."
    ),
    "PC-jc1-mass": (
        "The networks pretrained with an added mass regression against the same label set without it, in the form of "
        "figure {fig:PC-jc1}. Each point is one JetClass class, at its area under the receiver operating characteristic curve (AUC) for separating it "
        "from the other nine, with (a, b) 162 classes + mass against 162 classes and (c, d) 17 classes + mass against "
        "17 classes, as linear probes (left column) or fine-tuned (right column) on one million JetClass jets. Each point "
        "is the mean over two pretraining runs for a mass network and three for its label set. Values are in table "
        "{tab:jc1_class_auc}."
    ),
    "PC-conf": (
        "Networks pretrained on JetClass-II and fine-tuned on one million JetClass jets. Panel (a) shows the "
        "percentage of each true class that the network pretrained with 188 classes assigns to each predicted class, "
        "and panels (b)-(d) the change in that percentage with 162, 43 and 17 classes, each run against the "
        "188-class run of the same pretraining seed. Red marks a larger share and blue a smaller one, so a loss "
        "shows as blue on the diagonal. Each panel is the mean over three pretraining runs, and every difference "
        "shares one color scale with figures {fig:PC-conf-part} and {fig:PC-conf-mass}. The percentages are in table "
        "{tab:jc1conf}."
    ),
    "PC-conf-part": (
        "The networks pretrained with the four 17-class partitions in the form of figure {fig:PC-conf}, fine-tuned on "
        "one million JetClass jets. Each panel shows the change in the percentage of each true class assigned to each "
        "predicted class with (a) no flavor boundary, (b) a b-quark boundary, (c) a random flavor boundary and (d) "
        "random partitions, each run against the 188-class run of the same pretraining seed. Each panel is the mean "
        "over two pretraining runs, or over the five random partitions with one run each, on the color scale of "
        "figure {fig:PC-conf}. The changes are in table {tab:jc1conf}."
    ),
    "PC-conf-mass": (
        "The networks pretrained with an added mass regression in the form of figure {fig:PC-conf}, fine-tuned on one "
        "million JetClass jets. Each panel shows the change in the percentage of each true class assigned to each "
        "predicted class with (a) 162 classes + mass against 162 classes and (b) 17 classes + mass against 17 "
        "classes, each run against the run of its label set with the same pretraining seed. Each panel is the mean "
        "over two pretraining runs, on the color scale of figure {fig:PC-conf}. The changes are in table "
        "{tab:jc1conf}."
    ),
}

import gallery_captions  # noqa: E402
PAGES = [dataclasses.replace(p, caption=gallery_captions.fill(SUPPLEMENT_CAPTIONS[p.id])) if p.id in SUPPLEMENT_CAPTIONS
         else p for p in gallery_captions.apply(PAGES)]
