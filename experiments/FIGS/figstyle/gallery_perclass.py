#!/usr/bin/env python3
"""The per-class family (plan "Per class", PI brief 2026-11-12/13, L10; recall, PI 2026-11-18): every class's recall, the
fraction of its test jets the network labels as that class (argmax), for a compared network (vertical) against its
reference network (horizontal), one point per class, the mean over runs, the diagonal a black dotted reference line, one
axis range per grid; linear probe beside fine-tuned, a row per compared network. Slices: Fig. 4 (PC-jc2), the label sets
and the mass networks against 188 classes on JetClass-II (FIG4), the same on JetClass in the appendix (PC-jc1), the
17-class partitions against 188 and the mass networks against their own label set (supplement), and the JetClass
confusion matrices (supplement).

One colour rule for the whole family (L6): a point is coloured by what it is, never by network. On JetClass-II by what
the 17-class labels merged the class with (MERGE_KINDS: a decay differing only in quark flavor, only in lepton flavor,
both, neither; the QCD classes apart), on JetClass its class, from style_v5.CLASS10, which reuses no network colour
(palette_check); the network is named in the panel title. Source: experiments/EVAL/perclass_recall.py, pulled to
perclass_recall/ (probe_<test>.json: the v5 linear probes at 10^6 jets; ft_<test>_N1000000.json: the one-rule
fine-tuning cells at 10^6 jets), which holds each network's recalls, their bootstrap SDs over test jets, and decision
L3's differences; a missing file leaves its column pending. The confusion matrices come from confusion_jc1
(experiments/EVAL/perf_vs_pt.py)."""
from __future__ import annotations

import dataclasses
import functools
import json
import math

import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

import gallery_common as C
import style_v5 as S

P, M, RUNS = C.P, C.M, C.RUNS
POOLED = M._pooled(RUNS)
N_FT = 1_000_000
RECALL = P.DATA / "perclass_recall"

# the slices: (compared, reference) per row
LABEL_SETS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))
PARTITIONS = (("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"), ("RAND", "L188"))
# L20: four rows of two panels overrun the page, so two figures: the named flavor boundaries (-part) and the random
# controls, the random flavor boundary and the random partitions (-rand)
MASS = (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
# Fig. 4 and A1 (PI 2026-11-18): rows 162, 43 and 17 classes, each against 188; five rows (with the mass networks) fit
# the page only at 2.65 cm, below L20's 4.5 cm for points-only panels, so the mass networks stay on PC-jc2-mass /
# PC-jc1-mass (supplement), as NETWORK_RULE has it: the per-class figures show the label sets only
FIG4 = LABEL_SETS
MODES = ("Linear probe", "Fine-tuned")
# panel width (cm) by rows of networks (L20, points only: at least 4.5 cm): two rows at a two-column figure's width,
# three as wide as the page's height allows (make_results_book.MAX_H_CM); five (FIG4) fit 18.5 cm only at 2.65 cm
PANEL = {2: 6.3, 3: 4.6, 5: 2.65}
DOT = {2: 3.0, 3: 2.6, 5: 1.8}          # pt: 188 points in a panel stay apart
JC1_DOT = 4.0
JC1_SHAPES = ("o", "s", "^", "v", "D", "P", "X", "p", "h", "*")    # with the colour, so the ten classes part in greyscale
LEFT = 1.3                               # cm: the y label and the tick labels
WGAP = {5: 3.4}                          # cm between panels of a row: the titles clear; default 0.35
TITLE_LINE = 0.4                         # cm: a second title line (four columns: a partition's name in two lines)
HGAP = {5: 0.6}                          # cm between rows: the next titles only (shared x tick labels); default 0.95
XLABEL = 0.9                             # cm: x tick labels and label under each row (rows with their own reference)
ABSENT = "no per-class recalls of the 188-class network (perclass_recall/, experiments/EVAL/perclass_recall.py)"

# what the 17-class labels merged a JetClass-II class with (mlst_results._orbits, the per_class_macros merge rule)
MERGE_KINDS = ("qcd", "quark", "lepton", "both", "apart")
MERGE_LABEL = {"qcd": "QCD", "quark": "quark flavor merged", "lepton": "lepton flavor merged",
               "both": "quark and lepton flavor merged", "apart": "none merged"}
MERGE_COLOURS = dict(zip(MERGE_KINDS, (S.CLASS10[k] for k in (0, 1, 4, 3, 2))))
MERGE_SHAPES = dict(zip(MERGE_KINDS, ("o", "s", "^", "D", "v")))


@functools.lru_cache(maxsize=None)
def merge_kind() -> dict:
    """{jet_label: kind}: qcd for the 27 QCD classes; for a resonant class, whether its 17-class group also holds a
    decay that differs from it only in quark flavor (a shared orbit of flavour_pair_map.v2.csv), only in lepton flavor,
    both, or neither (apart)."""
    q, lep = M._orbits(P.REPO)
    g = M._groups(P.REPO)["R16_Q1"]
    out = {}
    for r in M._class_rows(P.REPO):
        n = r["class_name"]
        if n.startswith("label_QCD"):
            out[int(r["jet_label"])] = "qcd"
            continue
        mq = any(o != n and q[o] == q[n] and g[o] == g[n] for o in q)
        ml = any(o != n and lep[o] == lep[n] and g[o] == g[n] for o in lep)
        out[int(r["jet_label"])] = "both" if mq and ml else "quark" if mq else "lepton" if ml else "apart"
    return out


# ------------------------------------------------------------------ the values: {class: recall} of one run
@functools.lru_cache(maxsize=None)
def _doc(test: str, mode: int) -> dict:
    """perclass_recall.py's file of a test and mode (0 linear probe, 1 fine-tuned at N_FT jets), {} when not pulled."""
    f = RECALL / (f"probe_{test}.json" if mode == 0 else f"ft_{test}_N{N_FT}.json")
    return json.loads(f.read_text()) if f.exists() else {}


def _key(test: str, doc: dict):
    """class index of the file -> the family's class key: the jet_label (JetClass-II) or the class name (JetClass)."""
    return (lambda k: k) if test == "jc2" else (lambda k: doc["classes"][k])


def _source(test: str, mode: int):
    def get(m: str):
        doc = _doc(test, mode)
        r = doc.get("models", {}).get(m)
        return None if r is None else {_key(test, doc)(k): v for k, v in enumerate(r["recall"])}
    return get


SOURCES = {t: (_source(t, 0), _source(t, 1)) for t in ("jc2", "jc1")}


def _mean(get, row: str) -> dict:
    """{class: recall, the mean over the row's runs that hold it} (the five random partitions pooled)."""
    per = [p for m in POOLED[row] if (p := get(m))]
    return {c: float(np.mean([p[c] for p in per if c in p])) for c in set().union(*per)} if per else {}


def _sd(get, row: str) -> dict:
    """{class: SD of the recall over the row's runs} (0 for one run)."""
    per = [p for m in POOLED[row] if (p := get(m))]
    out = {}
    for c in set().union(*per) if per else ():
        v = [p[c] for p in per if c in p]
        out[c] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def _boot_mean(test: str, mode: int, row: str) -> dict:
    """{class: bootstrap SD over test jets of the row's mean recall}, {} when the file lacks the row."""
    doc = _doc(test, mode)
    g = doc.get("groups", {}).get(row)
    return {} if g is None else {_key(test, doc)(k): v for k, v in enumerate(g["boot_sd_mean"])}


# ------------------------------------------------------------------ decision L3 per class (PI 2026-11-14)
# A class counts as lower (higher) for a network than for its reference when |delta| > 3 sigma, sigma^2 = the SD^2 of
# each side over its runs + the bootstrap SD^2 over test jets of the difference of the two means (perclass_recall.py,
# its "diffs": the resamples shared by every network, as experiments/STATS/paired_bootstrap.py).
def l3_counts(test: str, pair: tuple, mode: int) -> dict | None:
    """{"lower", "higher", "lower_qcd", "n"}: the classes whose recall for pair[0] lies below (above) pair[1]'s by more
    than three L3 standard deviations, the QCD classes among the lower ones (JetClass-II), the classes compared; None
    when the file of the mode lacks the difference."""
    doc = _doc(test, mode)
    d = doc.get("diffs", {}).get(f"{pair[0]}-{pair[1]}")
    if d is None:
        return None
    lower = [_key(test, doc)(k) for k in d["lower"]]
    qcd = {c for c, k in merge_kind().items() if k == "qcd"} if test == "jc2" else {"label_QCD"}
    return {"lower": len(lower), "higher": len(d["higher"]), "lower_qcd": sum(c in qcd for c in lower),
            "n": len(d["delta"])}


# ------------------------------------------------------------------ the scatter grids
def _look(test: str):
    """(class -> (colour, marker, size, zorder), legend handles) of the family's one colour rule."""
    if test == "jc1":
        def look(c):
            k = S.JC1_CLASSES.index(c)
            return S.CLASS10[k], JC1_SHAPES[k], JC1_DOT, 3
        return look, [Line2D([], [], color=S.CLASS10[k], marker=JC1_SHAPES[k], markersize=JC1_DOT, linestyle="none",
                             label=S.JC1_TEX[c]) for k, c in enumerate(S.JC1_CLASSES)]
    kind = merge_kind()
    size = {k: sum(v == k for v in kind.values()) for k in MERGE_KINDS}

    def look(c):
        k = kind[c]
        return MERGE_COLOURS[k], MERGE_SHAPES[k], None, 3.1 if size[k] < 30 else 3     # a small kind over the large ones
    return look, [Line2D([], [], color=MERGE_COLOURS[k], marker=MERGE_SHAPES[k], markersize=3, linestyle="none",
                         label=MERGE_LABEL[k]) for k in MERGE_KINDS]


def _draw_grid(path, test: str, rows: tuple) -> None:
    """One row per (compared, reference) pair, linear probe and fine-tuned beside each other (L20: the networks are rows
    and the two modes the columns), on one linear axis range: the lowest and highest recall of any point and its bars,
    clear of the frame. Each point has bars of its side's run SD and the bootstrap SD over test jets of the mean (L3).
    A panel without values (its mode's file not pulled, or the network not run) is drawn empty with "pending" in its
    title. Rows with one reference share the x labels under the lowest row."""
    gets = SOURCES[test]
    if not any(_mean(get, "L188") for get in gets):
        raise C.Absent(ABSENT)
    look, handles = _look(test)
    handles = handles + [Line2D([], [], **S.REFERENCE, label="equal recall")]
    n = len(rows)
    w = PANEL[n]

    def side(k, row):
        m, sd, bt = _mean(gets[k], row), _sd(gets[k], row), _boot_mean(test, k, row)
        return {c: (m[c], math.hypot(sd[c], bt.get(c, 0.0))) for c in m}
    pts = [[[(ref[c][0], cmp_[c][0], c, ref[c][1], cmp_[c][1]) for c in ref if c in cmp_]
            for cmp_, ref in ((side(k, a), side(k, b)) for k in (0, 1))] for a, b in rows]
    vals = [v for row in pts for panel in row for x, y, _, ex, ey in panel for v in (x - ex, y - ey, x + ex, y + ey)]
    if not vals:
        raise C.Absent(f"no per-class recalls of {', '.join(a for a, _ in rows)}")
    pad = 0.04 * (max(vals) - min(vals))
    lim = (min(vals) - pad, max(vals) + pad)                 # the points and bars clear of the frames (L7)
    ncol = math.ceil(len(handles) / 2)                       # two legend rows
    one_ref = len({b for _, b in rows}) == 1
    two = any(len(S.LEGEND[a]) > 16 for a, _ in rows) and n < 5      # a long name and the mode on two title lines
    row_cm = 0.5 if test == "jc1" else 0.45          # a legend row of class names in mathtext is taller
    top = S.LEGEND_TOP + row_cm * 2 + 0.6 + (TITLE_LINE if two else 0)
    hgap = HGAP.get(n, 0.95) + (TITLE_LINE if two else 0) + (0 if one_ref else XLABEL)
    fig = S.figure_cm(S.PRINT_W, top + n * w + (n - 1) * hgap + 1.0)
    wgap = WGAP.get(n, 0.35)
    x0 = (S.PRINT_W - LEFT - 2 * w - wgap) / 2 + LEFT       # the block of labels and panels centred
    dot = DOT[n]
    for i, (row, (a, b)) in enumerate(zip(pts, rows)):
        y0 = top + i * (w + hgap)
        last = i == n - 1 or not one_ref
        for j, (panel, mode) in enumerate(zip(row, MODES)):
            ax = S.axes_cm(fig, x0 + j * (w + wgap), y0, w, w)
            S.fine_grid(ax)
            ax.plot(lim, lim, **S.REFERENCE)
            for x, y, c, ex, ey in panel:
                colour, marker, size, z = look(c)
                ax.errorbar(x, y, xerr=ex or None, yerr=ey or None, fmt="none", ecolor=colour, elinewidth=0.4,
                            capsize=0, alpha=0.6, zorder=z - 0.1)
                ax.plot(x, y, color=colour, marker=marker, markersize=size or dot, linestyle="none", clip_on=False,
                        zorder=z, markeredgewidth=0.5 if marker in "+x*" else 0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(ticker.MaxNLocator(3 if w < 4 else 5, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(labelleft=j == 0, labelbottom=last)
            S.title(ax, f"({'abcdefghij'[2 * i + j]}) {S.LEGEND[a]},{chr(10) if two else ' '}{mode.lower()}"
                    + ("" if panel else " (pending)"))
            if j == 0:
                S.labels(ax, ylabel="Recall" if n == 5 else "Recall, the panel's network")
            if last:
                S.labels(ax, xlabel=f"Recall, {S.LEGEND[b]}")
    S.fig_legend(fig, handles, top_cm=S.LEGEND_TOP, ncol=ncol, columnspacing=0.8, handletextpad=0.4)
    S.save(fig, path)


def draw_jc2(path) -> None:
    _draw_grid(path, "jc2", FIG4)


def draw_jc1(path) -> None:
    _draw_grid(path, "jc1", FIG4)


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
                   "fine-tuned, a row per compared network, one linear axis range per grid; per-class recall, not AUC "
                   "(PI 2026-11-18); points coloured by what the 17-class labels merged (JetClass-II) or by class "
                   "(JetClass) in colours that mean no network (L6). Fig. 4 and A1 (five rows) have 2.65 cm panels, "
                   "under L20's 4.5 cm: five rows of two do not fit 18.5 cm otherwise.")
RUN_NOTE = ("Means over runs: three per label set, two per mass network and flavor-boundary partition, the five random "
            "partitions one run each (pooled); bars of the run SD and the test-jet bootstrap SD of the mean in quadrature.")


def _note(test: str) -> str:
    """RUN_NOTE, and which mode's recalls are not pulled yet (its column drawn pending)."""
    miss = [f"{MODES[k].lower()} ({'probe_' + test if k == 0 else f'ft_{test}_N{N_FT}'}.json)"
            for k in (0, 1) if not _doc(test, k)]
    return RUN_NOTE + (f" Pending, not in perclass_recall/: {' and '.join(miss)}." if miss else "")


CONF_DIFFERS = ("Ten classes instead of 34; the 188-class matrix in CMS's Blues, every other panel a difference on one "
                "diverging RdBu_r scale shared by the three confusion pages; no numbers in the cells (L7).")
CONF_NOTE = ("Each run against the run of its reference with the same pretraining seed. From scratch is not drawn; its "
             "full-schedule matrices (confusion_jc1_fulldecay) are in tab:jc1conf.")


def _page(pid, title, draw, precedent=HEATMAPS, label=HEAT_LABEL, differs=SCATTER_DIFFERS, note=None) -> C.Page:
    return C.Page(id=pid, title=title, precedent=precedent, precedent_label=label, caption="", differs=differs,
                  draw=draw, data_note=_note(pid.split("-")[1]) if note is None else note)


PAGES = [
    _page("PC-jc2", "Per class, JetClass-II, label sets and mass networks", draw_jc2),
    _page("PC-jc1", "Per class, JetClass, label sets and mass networks", draw_jc1),
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
        "figure {fig:PC-jc2}. Each point is one JetClass-II class at its recall, the fraction of its test jets the "
        "network assigns to it, with (a, b) no flavor boundary and (c, d) a b-quark boundary (vertical) against 188 "
        "classes (horizontal), as linear probes (left column) or fine-tuned (right column) on one million "
        "class-balanced JetClass-II jets. Points below the dotted diagonal are classes the partition recalls less "
        "often, and colors give what the 17-class labels merge each class with. Each point is the mean over two "
        "pretraining runs, and the 188-class values over three runs, with bars of ±1 standard deviation over runs and "
        "test jets. Figure {fig:PC-jc2-rand} shows the random flavor boundary and the random partitions. The points "
        "are in table {tab:perclass}."
    ),
    "PC-jc2-rand": (
        "The panels of figure {fig:PC-jc2-part} for (a, b) the 17-class partition with a random flavor boundary and "
        "(c, d) the five random 17-class partitions, each point one JetClass-II class at its recall, the fraction of "
        "its test jets the network assigns to it (vertical), against 188 classes (horizontal), as linear probes (left "
        "column) or fine-tuned (right column) on one million class-balanced JetClass-II jets. Points below the dotted "
        "diagonal are classes the partition recalls less often, and colors give what the 17-class labels merge each "
        "class with. Each point is the mean over two pretraining runs, or over the five random partitions with one run "
        "each, and the 188-class values over three runs, with bars of ±1 standard deviation over runs and test jets. "
        "The points are in table {tab:perclass}."
    ),
    "PC-jc1-part": (
        "The networks pretrained with the 17-class partitions with no and with a b-quark flavor boundary in the form of "
        "figure {fig:PC-jc1}. Each point is one JetClass class at its recall, the fraction of its test jets the network "
        "assigns to it, with (a, b) no flavor boundary and (c, d) a b-quark boundary (vertical) against 188 classes "
        "(horizontal), as linear probes (left column) or fine-tuned (right column) on one million JetClass jets. Points "
        "below the dotted diagonal are classes the partition recalls less often. Each point is the mean over two "
        "pretraining runs, and the 188-class values over three runs, with bars of ±1 standard deviation over runs and "
        "test jets. Figure {fig:PC-jc1-rand} shows the random flavor boundary and the random partitions, and values "
        "are in table {tab:jc1_class_aucfull}."
    ),
    "PC-jc1-rand": (
        "The panels of figure {fig:PC-jc1-part} for (a, b) the 17-class partition with a random flavor boundary and "
        "(c, d) the five random 17-class partitions, each point one JetClass class at its recall, the fraction of its "
        "test jets the network assigns to it (vertical), against 188 classes (horizontal), as linear probes (left "
        "column) or fine-tuned (right column) on one million JetClass jets. Each point is the mean over two pretraining "
        "runs, or over the five random partitions with one run each, and the 188-class values over three runs, with "
        "bars of ±1 standard deviation over runs and test jets. Values are in table {tab:jc1_class_aucfull}."
    ),
    "PC-jc2-mass": (
        "The networks pretrained with an added mass regression against the same label set without it, in the form of "
        "figure {fig:PC-jc2}. Each point is one JetClass-II class at its recall, the fraction of its test jets the "
        "network assigns to it, with (a, b) 162 classes + mass against 162 classes and (c, d) 17 classes + mass "
        "against 17 classes, as linear probes (left column) or fine-tuned (right column) on one million class-balanced "
        "JetClass-II jets. Points below the dotted diagonal are classes the mass network recalls less often, and colors "
        "give what the 17-class labels merge each class with. Each point is the mean over two pretraining runs for a "
        "mass network and three for its label set, with bars of ±1 standard deviation over runs and test jets. The "
        "points are in table {tab:perclass}."
    ),
    "PC-jc1-mass": (
        "The networks pretrained with an added mass regression against the same label set without it, in the form of "
        "figure {fig:PC-jc1}. Each point is one JetClass class at its recall, the fraction of its test jets the network "
        "assigns to it, with (a, b) 162 classes + mass against 162 classes and (c, d) 17 classes + mass against 17 "
        "classes, as linear probes (left column) or fine-tuned (right column) on one million JetClass jets. Each point "
        "is the mean over two pretraining runs for a mass network and three for its label set, with bars of ±1 "
        "standard deviation over runs and test jets. Values are in table {tab:jc1_class_auc}."
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

def counts_sentence(test: str) -> str:
    """The finding sentence of PC-jc2 and PC-jc1 from l3_counts (PI 2026-11-14: the counts follow L3, the run and
    test-jet spread, and are computed, never typed): how many classes each network of FIG4 recalls less often than 188
    classes, as linear probes and, once its file is pulled, fine-tuned; "" without the linear probes."""
    c = {(a, k): l3_counts(test, (a, b), k) for a, b in FIG4 for k in (0, 1)}
    if any(c[a, 0] is None for a, _ in FIG4):
        return ""
    seq = lambda k: (", ".join(str(c[a, k]["lower"]) for a, _ in FIG4[:-1])      # noqa: E731
                     + f" and {c[FIG4[-1][0], k]['lower']}")
    ft = all(c[a, 1] is not None for a, _ in FIG4)
    raised = sum(v["higher"] for v in c.values() if v)
    qcd = f" ({c['L162', 0]['lower_qcd']} of the 162-class ones QCD classes)" if test == "jc2" else ""
    what = "JetClass-II" if test == "jc2" else "JetClass"
    return (f"Beyond three standard deviations of the spread over runs and test jets, 162, 43 and 17 classes, 162 "
            f"classes + mass and 17 classes + mass have a lower recall than 188 classes on {seq(0)} of the "
            f"{c[FIG4[0][0], 0]['n']} {what} classes as linear probes{qcd}"
            + (f" and on {seq(1)} after fine-tuning" if ft else "")
            + (", and none a higher one." if not raised else f", and a higher one on {raised} in all."))


import gallery_captions  # noqa: E402
gallery_captions.COUNTS.update({"PC-jc2": counts_sentence("jc2"), "PC-jc1": counts_sentence("jc1")})
PAGES = [dataclasses.replace(p, caption=gallery_captions.fill(SUPPLEMENT_CAPTIONS[p.id])) if p.id in SUPPLEMENT_CAPTIONS
         else p for p in gallery_captions.apply(PAGES)]
