#!/usr/bin/env python3
"""Fig. 1: label sets, pretraining and use (the PI's drawing, 2026-11-12).

The label sets are read from configs/labelmaps/rung_label_maps.v1.csv (the label maps the pretrainings used), never
written here: the asserts below stop the script unless the rows count 188, 162, 43 and 17 classes, every class of every
label set is contiguous, and the groups are 15 two-prong, 146 three- and four-prong and 27 QCD classes.
The jets are schematic drawings, not data.

    python3 experiments/FIGS/fig1_design.py [out.pdf]     (default figures/mlst/fig1_design.pdf, and a .png beside it)

Drawn at 13 cm wide, in centimetres; all text 8 pt (panel titles 9 pt) in the paper's body font.

build(nested=True) is the rewrite's Fig. 1 (PI 2026-11-24, LEDGER GO-6: "Fig. 1 shows the nesting for one example
group"): in place of the six example jets, the ten classes of the first 17-class group (X to two quarks or gluons) as
jets, with the class each falls in under the 162, 43 and 17 classes, read from the label maps.
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAP = ROOT / "configs/labelmaps/rung_label_maps.v1.csv"
FONTS = HERE / "figstyle" / "fonts"

C = {188: "#332288", 162: "#AA4499", 43: "#999933", 17: "#44AA99"}
COLUMN = {188: "L188", 162: "L162", 43: "R42_Q1", 17: "R16_Q1"}      # the map's column of each label set
GREY, INK, LG = "#6B6B6B", "#4A4A4A", "#C9C9C9"
# PI 2026-11-13 (L8): colour only for the four label sets; the jets, features and task markers in grey or slate.
# SLATE marks the features and the task markers, SLATE_TINT fills the features; the jets keep their detector regions
# apart by lightness: tracks in the tracker, deposits in the inner (electromagnetic) and the outer (hadronic) band
SLATE, SLATE_TINT = "#4F5B6B", "#E4E8ED"
TRK, ECAL, HCAL = "#56606E", "#7D8896", "#AAB3BE"
BAND = "#F3F3F3"                                   # the calorimeter bands themselves stay neutral
W, H = 13.0, 13.1
FS, FS_T = 8, 9


NESTED_EXTRA = 2.5      # cm: the nested example's two more rows of classes and their names, its taller strip and key
# the nested example's strip (round-2 cold readers: "the colored bars are too thin to count"): taller rows, wider gaps
# between the cells of 188 and 162 classes, the example group outlined, and a key under the strip
NESTED_STRIP = {"rh": 0.38, "gap": 0.13, "cell_gap": 0.014}
FLAVOR_VERTEX = {"b": 0.27, "c": 0.17}       # a prong's displaced vertex, as a share of the jet's radius; light: none
# the 43 classes of the example group, by the name the label maps give them
NESTED_43 = {"2P_HAD_2PARTON|B": r"with a $b$ quark", "2P_HAD_2PARTON|C": r"with a $c$ quark, no $b$",
             "2P_HAD_2PARTON|LG": "light quarks and gluons only"}
NESTED_17 = {"2P_HAD_2PARTON": r"$X\!\to$two quarks or gluons"}


def build(nested: bool = False):
    """The figure, at its printed size. nested: the rewrite's example (the module's docstring)."""
    H = globals()["H"] + (NESTED_EXTRA if nested else 0)
    for f in sorted(FONTS.glob("lmroman10-*.otf")):          # the paper's body font
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 8,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    fig.height_cm = H
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def arrow(p, q, lw=0.9, ms=6, ls="-", z=3):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=GREY,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))

    def line(xs, ys, ls="-", lw=0.9):
        ax.plot(xs, ys, color=GREY, lw=lw, ls=ls, solid_capstyle="butt", zorder=3)

    def block(x, y, w, h, txt, rot=0, ec=INK, lw=0.6):
        """A plain outlined box: white, no tint (boxes only for the network and its head)."""
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=ec, lw=lw, joinstyle="miter", zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", rotation=rot, fontsize=FS, zorder=4, linespacing=1.05)

    def jet(ox, oy, R, prongs, half=26, seed=0, layers=4, lw=0.32, cells=(2, 3)):
        """Schematic jet in a detector wedge: tracker layers and two calorimeter bands of cells.
        prongs: (offset_rad, n_tracks, sigma, displaced_vertex_as_fraction_of_R). Each calorimeter band is a
        polar grid of cells (cells = radial layers per band); a cell is drawn when the energy that the
        particles leave in it passes a threshold, so deposits are contiguous clusters on the grid."""
        rng = np.random.default_rng(seed)
        a0, a1 = 90 - half, 90 + half
        rt = 0.60 * R
        tracker = ax.add_patch(Wedge((ox, oy), rt, a0, a1, fc="#FAFAFA", ec=LG, lw=0.5, zorder=0))
        for k in range(1, layers + 1):
            ax.add_patch(Wedge((ox, oy), rt * k / (layers + 0.6), a0, a1, fc="none", ec="#DCDCDC", lw=0.4, zorder=0))
        # (inner radius, outer radius, layers, colour, energy share per layer, lateral width in cells)
        bands = ((0.65 * R, 0.78 * R, cells[0], ECAL, np.linspace(1.0, 0.55, cells[0]), 0.4),
                 (0.82 * R, 1.00 * R, cells[1], HCAL, np.array([0.6, 1.0, 0.6, 0.35])[:cells[1]], 0.55))
        for r0, r1, *_ in bands:
            ax.add_patch(Wedge((ox, oy), r1, a0, a1, width=r1 - r0, fc=BAND, ec=LG, lw=0.5, zorder=0))
        lim = np.radians(half) - 0.03
        clip = lambda v: float(np.clip(v, np.pi / 2 - lim, np.pi / 2 + lim))
        inner, outer = [], []                        # (angle, energy) of what reaches each calorimeter band
        for off, n, sig, dv in prongs:
            axis = np.pi / 2 + off
            vx, vy = ox + dv * R * np.cos(axis), oy + dv * R * np.sin(axis)
            if dv > 0:
                ax.plot([ox, vx], [oy, vy], color=INK, lw=0.4, ls=(0, (1.2, 1.2)), zorder=1)
                ax.add_patch(Circle((vx, vy), 0.014 * R + 0.012, fc=INK, ec="none", zorder=4))
            for _ in range(n):                       # charged hadrons: a track, then both calorimeters
                d = float(np.clip(rng.normal(0, sig), -2.2 * sig, 2.2 * sig))
                aend = clip(axis + d + rng.normal(0, 0.03))
                ex, ey = ox + rt * np.cos(aend), oy + rt * np.sin(aend)
                t = np.linspace(0, 1, 40)
                bend = rng.normal(0, 0.05) * R
                nx, ny = -(ey - vy), (ex - vx)
                nn = np.hypot(nx, ny) + 1e-9
                track, = ax.plot(vx + (ex - vx) * t + bend * nx / nn * 2 * t * (1 - t),
                                 vy + (ey - vy) * t + bend * ny / nn * 2 * t * (1 - t),
                                 color=TRK, lw=lw, alpha=0.95, zorder=1, solid_capstyle="round")
                track.set_clip_path(tracker)         # a bent track stays inside the tracker (PI 2026-11-13)
                e = 0.35 + rng.exponential(0.6)
                inner.append((aend, 0.35 * e)); outer.append((aend, e))
            for _ in range(n // 2 + 1):              # photons: no track, inner calorimeter only
                inner.append((clip(axis + rng.normal(0, sig)), 0.5 + rng.exponential(0.5)))
        span = np.radians(a1 - a0)
        for (r0, r1, nl, colour, share, width), ends in zip(bands, (inner, outer)):
            dr = (r1 - r0) / nl
            nphi = int(round(span * (r0 + r1) / 2 / dr))                 # cells square at the band's middle
            dphi = span / nphi
            phi = np.radians(a0) + dphi * (np.arange(nphi) + 0.5)
            energy = sum(e * np.outer(share, np.exp(-0.5 * ((phi - a) / (width * dphi)) ** 2)) for a, e in ends)
            g = 0.1                                                      # gap around a cell, in cells
            for i, j in zip(*np.nonzero(energy > 0.8)):
                ra, rb = r0 + dr * (i + g), r0 + dr * (i + 1 - g)
                pa, pb = phi[j] - dphi * (0.5 - g), phi[j] + dphi * (0.5 - g)
                ax.add_patch(Polygon([(ox + r * np.cos(p), oy + r * np.sin(p))
                                      for r, p in ((ra, pa), (rb, pa), (rb, pb), (ra, pb))],
                                     closed=True, fc=colour, ec="none", zorder=2))
        ax.add_patch(Circle((ox, oy), 0.018 * R + 0.012, fc=INK, ec="none", zorder=5))

    # ------------------------------------------------------------------ label map (the released label maps)
    # sorted so that every class of every label set is contiguous: two prongs, three and four prongs, QCD
    group = lambda r: 2 if r["class_name"].startswith("label_QCD") else 0 if r["R16_Q1_name"].startswith("2P") else 1
    lm = sorted(csv.DictReader(MAP.open()),
                key=lambda r: (group(r), int(r["R16_Q1"]), int(r["R42_Q1"]), int(r["L162"]), int(r["L188"])))
    assert len(lm) == 188
    names = [r["class_name"] for r in lm]
    is_qcd = [n.startswith("label_QCD") for n in names]
    levels = {n: [int(r[col]) for r in lm] for n, col in COLUMN.items()}
    g17 = levels[17]
    LV = [188, 162, 43, 17]

    def runs(g):
        out, s = [], 0
        for i in range(1, len(g) + 1):
            if i == len(g) or g[i] != g[s]:
                out.append((s, i - s)); s = i
        return out

    for n, g in levels.items():                              # 188, 162, 43, 17 cells; every class contiguous
        assert len(set(g)) == n and len(runs(g)) == n, n
    N2 = sum(group(r) == 0 for r in lm); NQ = sum(is_qcd)
    assert (N2, 188 - N2 - NQ, NQ) == (15, 146, 27), (N2, NQ)
    assert all(is_qcd[-NQ:]) and not any(is_qcd[:-NQ])
    assert [k for _, k in runs(g17[:N2])] == [10, 1, 2, 2], [k for _, k in runs(g17[:N2])]

    # ------------------------------------------------------------------ (a) example: fine to coarse
    ax.text(0.2, H - 0.3, "(a) Label sets", fontsize=FS_T, va="center")
    SQ = 0.15
    if nested:
        # the first 17-class group: its ten classes, the same ten under the 162 classes, three of the 43, one of the 17
        n1 = runs(g17)[0][1]
        flav = [names[i].removeprefix("label_X_") for i in range(n1)]
        assert lm[0]["R16_Q1_name"] in NESTED_17 and all(len(f) == 2 for f in flav), flav
        assert [k for _, k in runs(levels[162][:n1])] == [1] * n1, "the example's classes are single under 162 classes"
        tex = lambda f: rf"$X\!\to {f[0]}\bar {f[1]}$" if f[0] == f[1] and f[0] != "g" else rf"$X\!\to {f}$"
        xs = list(np.linspace(2.35, W - 0.75, n1))
        yjet, ylab = H - 1.95, H - 2.2
        y188, y162, y43 = H - 2.62, H - 3.12, H - 3.82
        ycoarse = y43 - 1.05
        for k, (f, x) in enumerate(zip(flav, xs)):
            jet(x, yjet, 1.25, [(o, 3, 0.03, FLAVOR_VERTEX.get(q, 0.0)) for o, q in zip((-0.19, 0.19), f)], half=17,
                seed=20 + k, layers=3, lw=0.3)
            ax.text(x, ylab, tex(f), ha="center", va="center", fontsize=FS)
            ax.plot([x, x], [y188, y162], color="#9AA3B5", lw=0.6, zorder=1)
            for y, n in ((y188, 188), (y162, 162)):
                ax.add_patch(Rectangle((x - SQ / 2, y - SQ / 2), SQ, SQ, fc=C[n], ec="none", zorder=3))

        def merge(xa: list, ya: float, xc: float, yb: float) -> None:
            ymid = (ya + yb) / 2
            for x in xa:
                ax.plot([x, x, xc, xc], [ya - SQ / 2, ymid, ymid, yb + SQ / 2], color="#9AA3B5", lw=0.6,
                        solid_joinstyle="miter", zorder=1)

        x43 = []
        for s0, k in runs(levels[43][:n1]):
            xc = float(np.mean(xs[s0:s0 + k]))
            x43.append(xc)
            merge(xs[s0:s0 + k], y162, xc, y43)
            ax.add_patch(Rectangle((xc - SQ / 2, y43 - SQ / 2), SQ, SQ, fc=C[43], ec="none", zorder=3))
            ax.text(xc, y43 - 0.3, NESTED_43[lm[s0]["R42_Q1_name"]], ha="center", va="center", fontsize=FS)
        xc = float(np.mean(xs))
        merge(x43, y43 - 0.42, xc, ycoarse)
        ax.add_patch(Rectangle((xc - SQ / 2, ycoarse - SQ / 2), SQ, SQ, fc=C[17], ec="none", zorder=3))
        ax.text(xc, ycoarse - 0.3, NESTED_17[lm[0]["R16_Q1_name"]], ha="center", va="center", fontsize=FS)
        for y, n in ((y188, 188), (y162, 162), (y43, 43), (ycoarse, 17)):
            ax.text(0.2, y, f"{n} classes", ha="left", va="center", fontsize=FS)
    else:
        p2 = lambda dv: [(-0.19, 3, 0.03, dv), (0.19, 3, 0.03, dv)] if dv != 0.14 else [(-0.19, 3, 0.03, 0.17), (0.19, 3, 0.03, 0.17)]
        p4 = lambda dvs: [(o, 2, 0.018, d) for o, d in zip((-0.39, -0.13, 0.13, 0.39), dvs)]
        # (class name in the label map, label, x, prongs, wedge half-angle)
        EX = [("label_X_bb", r"$X\!\to b\bar b$", 1.95, p2(0.27), 18),
              ("label_X_cc", r"$X\!\to c\bar c$", 3.3, p2(0.14), 18),
              ("label_X_qq", r"$X\!\to q\bar q$", 4.65, p2(0.0), 18),
              ("label_X_YY_bbbb", r"$X\!\to YY\!\to bbbb$", 7.0, p4((0.3, 0.3, 0.3, 0.3)), 29),
              ("label_X_YY_ccqq", r"$X\!\to YY\!\to ccqq$", 9.25, p4((0.17, 0.17, 0.0, 0.0)), 29),
              ("label_X_YY_qqqq", r"$X\!\to YY\!\to qqqq$", 11.5, p4((0.0, 0.0, 0.0, 0.0)), 29)]
        # j: any quark or gluon (the 17-class groups of two and of four quarks or gluons)
        COARSE = {"label_X_bb": r"$X\!\to jj$", "label_X_YY_bbbb": r"$X\!\to YY\!\to jjjj$"}
        ei = [names.index(e[0]) for e in EX]
        gf = [g17[i] for i in ei]
        assert len(set(ei)) == len(ei) and [k for _, k in runs(gf)] == [3, 3], "the examples must form two 17-class groups"
        xs = [e[2] for e in EX]
        yjet, ylab = H - 2.3, H - 2.56
        yfine, ycoarse = H - 2.98, H - 3.68
        for k, (_, lab, x, prongs, half) in enumerate(EX):
            jet(x, yjet, 1.4, prongs, half=half, seed=20 + k, layers=3, lw=0.3)
            ax.text(x, ylab, lab, ha="center", va="center", fontsize=FS)
            ax.add_patch(Rectangle((x - SQ / 2, yfine - SQ / 2), SQ, SQ, fc=C[188], ec="none", zorder=3))
        for s0, k in runs(gf):
            xc = float(np.mean(xs[s0:s0 + k])); ymid = (yfine + ycoarse) / 2
            for x in xs[s0:s0 + k]:
                ax.plot([x, x, xc, xc], [yfine - SQ / 2, ymid, ymid, ycoarse + SQ / 2], color="#9AA3B5", lw=0.6,
                        solid_joinstyle="miter", zorder=1)
            ax.add_patch(Rectangle((xc - SQ / 2, ycoarse - SQ / 2), SQ, SQ, fc=C[17], ec="none", zorder=3))
            ax.text(xc, ycoarse - 0.3, COARSE[EX[s0][0]], ha="center", va="center", fontsize=FS)
        ax.text(0.2, yfine, "Fine", ha="left", va="center", fontsize=FS)
        ax.text(0.2, ycoarse, "Coarse", ha="left", va="center", fontsize=FS)

    # ------------------------------------------------------------------ (a) all 188 classes
    rh, gap = (NESTED_STRIP["rh"], NESTED_STRIP["gap"]) if nested else (0.28, 0.11)
    ytop = ycoarse - 1.5
    rowy = [ytop - j * (rh + gap) for j in range(4)]
    x0, x1 = 1.0, 12.8; pitch = (x1 - x0) / 188
    for y, n in zip(rowy, LV):
        ax.add_patch(Rectangle((0.2, y + rh / 2 - 0.07), 0.14, 0.14, fc=C[n], ec="none"))
        ax.text(0.43, y + rh / 2 - 0.01, str(n), va="center", ha="left", fontsize=FS)
        g = (NESTED_STRIP["cell_gap"] if nested else 0.009) if n >= 162 else 0.017
        for s0, k in runs(levels[n]):
            gg = 0.017 if (n == 162 and k > 1) else g
            ax.add_patch(Rectangle((x0 + s0 * pitch + gg, y), k * pitch - 2 * gg, rh, fc=C[n], ec="none", zorder=2))
    for name, a_, b_ in (("2 prongs", 0, N2), ("3 and 4 prongs", N2, 188 - NQ), ("QCD", 188 - NQ, 188)):
        xa, xb, yy = x0 + a_ * pitch + 0.02, x0 + b_ * pitch - 0.02, ytop + rh + 0.13
        line([xa, xa, xb, xb], [yy - 0.06, yy, yy, yy - 0.06], lw=0.6)
        ax.text((xa + xb) / 2, yy + 0.22, name, ha="center", va="center", fontsize=FS)
    ybot = rowy[-1]
    if nested:
        # the group of the tree above, outlined across the four rows, and what a cell and a bar are
        xa, xb = x0 + 0.004, x0 + n1 * pitch - 0.004
        ax.add_patch(Rectangle((xa, ybot - 0.05), xb - xa, ytop + rh - ybot + 0.1, fc="none", ec=INK, lw=0.8,
                               zorder=4))
        line([(xa + xb) / 2] * 2, [ybot - 0.05, ybot - 0.22], lw=0.6)
        ax.text(xa, ybot - 0.36, "the group drawn above", ha="left", va="center", fontsize=FS)
        ax.text(x0, ybot - 0.74, "Each cell is one of the 188 classes; each bar is one class of that label set.",
                ha="left", va="center", fontsize=FS)
        ybot -= 0.85

    # ------------------------------------------------------------------ (b) pretraining
    yt = ybot - 0.85
    ax.text(0.2, yt, "(b) Pretraining and use", fontsize=FS_T, va="center")
    if nested:
        # PI 2026-11-26 (ledger 34.3): one flow, jets, network, features, then linear probe or fine-tuning, then the
        # tasks; the row of output squares and the task matrix go
        y0 = yt - 1.75
        jet(1.35, y0 - 1.0, 1.95, [(-0.23, 5, 0.045, 0.0), (0.23, 5, 0.045, 0.0)], half=27, seed=12, lw=0.3, cells=(3, 4))
        ax.text(1.35, y0 + 1.13, "JetClass-II jets", ha="center", va="center", fontsize=FS)
        arrow((2.65, y0), (3.3, y0))
        xb0, wb = 3.35, 1.0
        block(xb0, y0 - 1.0, wb, 2.0, "Particle\nTransformer", rot=90)
        ax.text(xb0 + wb / 2, y0 + 1.25, "pretrained on\none label set", ha="center", va="center", fontsize=FS,
                linespacing=1.0)
        arrow((xb0 + wb + 0.15, y0), (5.3, y0))
        xf = 5.52
        for k in range(6):
            ax.add_patch(Circle((xf, y0 - 0.5 + 0.2 * k), 0.065, fc=SLATE_TINT, ec=SLATE, lw=0.8, zorder=3))
        ax.text(xf, y0 + 0.8, "Features", ha="center", va="center", fontsize=FS)
        # three uses (editor 8f085fe, item 30): a classifier trained on the frozen features (Methods' words) for the
        # resonance benchmark,
        # a linear probe on them and fine-tuning for the four tasks; one box of tasks for each
        xu, wu, hu = 6.4, 2.6, 0.5
        ytl, yp, yft = y0 + 0.88, y0 + 0.05, y0 - 0.8
        htl = 0.72                                       # two lines
        block(xu, ytl - htl / 2, wu, htl, "Classifier trained\non frozen features")
        block(xu, yp - hu / 2, wu, hu, "Linear probe")
        block(xu, yft - hu / 2, wu, hu, "Fine-tuning")
        arrow((xf + 0.15, y0 + 0.45), (xu - 0.05, ytl - 0.12))     # both frozen uses read the features
        arrow((xf + 0.2, yp), (xu - 0.05, yp))
        arrow((xb0 + wb, yft), (xu - 0.05, yft))         # fine-tuning adapts the whole network
        xt, wt = 9.5, W - 9.5 - 0.15
        block(xt, y0 - 1.1, wt, 1.45, "JetClass-II\nJetClass\nTop tagging\nQuark/gluon")
        block(xt, ytl - htl / 2, wt, htl, r"$X\!\to b\bar b$, $X\!\to bs$")
        arrow((xu + wu + 0.05, ytl), (xt - 0.05, ytl))
        for yy in (yp, yft):
            arrow((xu + wu + 0.05, yy), (xt - 0.05, yy))
        bottom = y0 - 1.1 - 0.25                         # the figure ends under the flow
        ax.set_ylim(bottom, H)
        fig.set_size_inches(W / 2.54, (H - bottom) / 2.54)
        fig.height_cm = H - bottom
        return fig
    y0 = yt - 1.9
    # input: the particles of one jet (tracks and energy deposits; no detector hits)
    jet(1.35, y0 - 1.0, 1.95, [(-0.23, 5, 0.045, 0.0), (0.23, 5, 0.045, 0.0)], half=27, seed=12, lw=0.3, cells=(3, 4))
    ax.text(1.35, y0 + 1.13, "JetClass-II jet", ha="center", va="center", fontsize=FS)
    arrow((2.65, y0), (3.3, y0))
    xb0, wb = 3.35, 1.0                      # the network that is reused
    block(xb0, y0 - 1.0, wb, 2.0, "Particle\nTransformer", rot=90)      # the architecture (L8); the caption names Sophon's head
    arrow((xb0 + wb + 0.2, y0), (5.3, y0))
    xf = 5.52
    for k in range(6):
        ax.add_patch(Circle((xf, y0 - 0.5 + 0.2 * k), 0.065, fc=SLATE_TINT, ec=SLATE, lw=0.8, zorder=3))
    ax.text(xf, y0 + 0.8, "Features", ha="center", va="center", fontsize=FS)
    arrow((xf + 0.22, y0), (xf + 0.92, y0))
    # the head: a shape of its own, tapering toward the outputs
    xh, wh = xf + 0.97, 1.15 if nested else 0.95       # "Output layer" on two lines needs the wider shape
    ax.add_patch(Polygon([(xh, y0 - 0.42), (xh + wh, y0 - 0.26), (xh + wh, y0 + 0.26), (xh, y0 + 0.42)], closed=True,
                         fc="white", ec=INK, lw=0.6, joinstyle="miter", zorder=2))
    ax.text(xh + wh / 2, y0, "Output\nlayer" if nested else "Head", ha="center", va="center", fontsize=FS, zorder=4,
            linespacing=1.0)
    # one pretraining per label set: the head's output is ONE of the four rows (brace, not a fan of arrows)
    xe = xh + wh + 0.63
    arrow((xh + wh, y0), (xe - 0.12, y0))

    def brace(x, ya, yb, w=0.16):
        ymid = (ya + yb) / 2; t = np.linspace(0, 1, 30)

        def q(p0, p1, p2):
            return ((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t ** 2 * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t ** 2 * p2[1])
        for seg in (((x + w, ya), (x + w / 2, ya), (x + w / 2, ya - 0.14 * np.sign(ya - yb))),
                    ((x + w / 2, ymid + 0.14 * np.sign(ya - yb)), (x + w / 2, ymid), (x, ymid)),
                    ((x, ymid), (x + w / 2, ymid), (x + w / 2, ymid - 0.14 * np.sign(ya - yb))),
                    ((x + w / 2, yb + 0.14 * np.sign(ya - yb)), (x + w / 2, yb), (x + w, yb))):
            ax.plot(*q(*seg), color=GREY, lw=0.9, zorder=3, solid_capstyle="round")
        sg = np.sign(ya - yb)
        ax.plot([x + w / 2] * 2, [ya - 0.14 * sg, ymid + 0.14 * sg], color=GREY, lw=0.9, zorder=3)
        ax.plot([x + w / 2] * 2, [ymid - 0.14 * sg, yb + 0.14 * sg], color=GREY, lw=0.9, zorder=3)

    brace(xe, y0 + 0.72, y0 - 0.72)
    for j, (n, k) in enumerate([(188, 22), (162, 19), (43, 9), (17, 4)]):
        yy = y0 + 0.6 - 0.4 * j
        for i in range(k):
            ax.add_patch(Rectangle((xe + 0.3 + 0.17 * i, yy - 0.065), 0.13, 0.13, fc=C[n], ec="none", zorder=3))

    # ------------------------------------------------------------------ (b) use
    REN = fig.canvas.get_renderer()

    def tw(txt):                                   # text width in cm
        o = ax.text(0, 0, txt, fontsize=FS); wd = o.get_window_extent(REN).width / fig.dpi * 2.54; o.remove()
        return wd

    TASKS = [("JetClass-II", ("188 classes",), True),
             ("JetClass", (r"$H$", r"$W$", r"$Z$", r"$t$", "QCD"), True),
             ("Top tagging", (r"$t$", "QCD"), True),
             ("Quark/gluon", (r"$q$", r"$g$"), True),
             (r"$X\!\to b\bar b$, $X\!\to bs$", (r"$b\bar b$", r"$bs$", "QCD"), False)]   # fine-tuned on the first four
    colw = [tw(nm) for nm, _, _ in TASKS]          # the task names only (PI 2026-11-12: no lines under them)
    BX, BW, BH = 0.85, 2.2, 0.48                   # the two row labels, in the same plain box as the network and head
    XL, XR = BX + BW + 0.45, W - 0.2
    GAP = (XR - XL - sum(colw)) / (len(colw) - 1)  # equal gaps between the five columns
    cx, x = [], XL
    for cw_ in colw:
        cx.append(x + cw_ / 2); x += cw_ + GAP
    yb_, ya_ = y0 - 1.28, y0 - 1.52                # feed lines: block (outer), features (inner)
    ynam = ya_ - 0.62
    yF = ynam - 0.66; yT = yF - 0.72
    # fine-tuning starts from the pretrained network; frozen features come out of the Features column
    xbk = xb0 + wb / 2
    line([xbk, xbk, 0.14, 0.14], [y0 - 1.0, yb_, yb_, yT]); arrow((0.14, yT), (BX - 0.03, yT), ms=8)
    line([xf, xf, 0.38, 0.38], [y0 - 0.62, ya_, ya_, yF]); arrow((0.38, yF), (BX - 0.03, yF), ms=8)
    block(BX, yF - BH / 2, BW, BH, "Frozen features")
    block(BX, yT - BH / 2, BW, BH, "Fine-tuning")
    MS = 0.17
    line([BX + BW, cx[-1]], [yF, yF], lw=0.6)
    line([BX + BW, max(c for c, t in zip(cx, TASKS) if t[2])], [yT, yT], lw=0.6)
    for c, (nm, toks, ft) in zip(cx, TASKS):
        ax.text(c, ynam, nm, ha="center", va="center", fontsize=FS)
        ax.add_patch(Rectangle((c - MS / 2, yF - MS / 2), MS, MS, fc="white", ec=SLATE, lw=0.9, zorder=4))     # frozen: open
        if ft:
            ax.add_patch(Rectangle((c - MS / 2, yT - MS / 2), MS, MS, fc=SLATE, ec=SLATE, lw=0.9, zorder=4))     # fine-tuned: filled
    return fig


def draw(path: pathlib.Path, nested: bool = False) -> None:
    fig = build(nested)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {path} and its .png ({W:g} x {fig.height_cm:g} cm)")


if __name__ == "__main__":
    draw(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures/mlst/fig1_design.pdf")
