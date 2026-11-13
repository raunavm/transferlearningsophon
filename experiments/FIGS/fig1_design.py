#!/usr/bin/env python3
"""Fig. 1: label sets, pretraining and use (the PI's drawing, 2026-11-12).

The label sets are read from configs/labelmaps/rung_label_maps.v1.csv (the label maps the pretrainings used), never
written here: the asserts below stop the script unless the rows count 188, 162, 43 and 17 classes, every class of every
label set is contiguous, and the groups are 15 two-prong, 146 three- and four-prong and 27 QCD classes.
The jets are schematic drawings, not data.

    python3 experiments/FIGS/fig1_design.py [out.pdf]     (default figures/mlst/fig1_design.pdf, and a .png beside it)

Drawn at 13 cm wide, in centimetres; all text 8 pt (panel titles 9 pt) in the paper's body font.
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAP = ROOT / "configs/labelmaps/rung_label_maps.v1.csv"
FONTS = HERE / "figstyle" / "fonts"

C = {188: "#332288", 162: "#AA4499", 43: "#999933", 17: "#44AA99"}
COLUMN = {188: "L188", 162: "L162", 43: "R42_Q1", 17: "R16_Q1"}      # the map's column of each label set
SLATE, BLOCK, GREY, INK, LG = "#5B6B8A", "#DDE4EE", "#6B6B6B", "#4A4A4A", "#C9C9C9"
W, H = 13.0, 12.4
FS, FS_T = 8, 9


def build():
    """The figure, at its printed size."""
    for f in sorted(FONTS.glob("lmroman10-*.otf")):          # the paper's body font
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 8,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def arrow(p, q, lw=0.9, ms=6, ls="-", z=3):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=GREY,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))

    def line(xs, ys, ls="-", lw=0.9):
        ax.plot(xs, ys, color=GREY, lw=lw, ls=ls, solid_capstyle="butt", zorder=3)

    def block(x, y, w, h, txt, rot=0, stack=0):
        for k in range(stack, -1, -1):
            ax.add_patch(Rectangle((x + 0.07 * k, y + 0.07 * k), w, h, fc=BLOCK, ec=SLATE, lw=0.7,
                                   joinstyle="miter", zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", rotation=rot, fontsize=FS, zorder=4, linespacing=1.05)

    def jet(ox, oy, R, prongs, half=26, seed=0, layers=4, hits=True, lw=0.32, sq=1.0, dep=0.035):
        """Schematic jet in a detector wedge: tracker layers and two calorimeter bands.
        prongs: (offset_rad, n_tracks, sigma, displaced_vertex_as_fraction_of_R[, kind]);
        kind 'h' hadronic (default), 'e' electron (one track, compact inner deposit),
        'm' muon (one track that leaves the detector)."""
        rng = np.random.default_rng(seed)
        a0, a1 = 90 - half, 90 + half
        rt = 0.60 * R
        ax.add_patch(Wedge((ox, oy), rt, a0, a1, fc="#FAFAFA", ec=LG, lw=0.5, zorder=0))
        for k in range(1, layers + 1):
            ax.add_patch(Wedge((ox, oy), rt * k / (layers + 0.6), a0, a1, fc="none", ec="#DCDCDC", lw=0.4, zorder=0))
        ax.add_patch(Wedge((ox, oy), 0.78 * R, a0, a1, width=0.13 * R, fc="#EEEEEE", ec=LG, lw=0.5, zorder=0))
        ax.add_patch(Wedge((ox, oy), R, a0, a1, width=0.18 * R, fc="#F3F3F3", ec=LG, lw=0.5, zorder=0))
        lim = np.radians(half) - 0.03
        clip = lambda v: float(np.clip(v, np.pi / 2 - lim, np.pi / 2 + lim))

        def square(r, ang, size):
            ax.add_patch(Rectangle((ox + r * np.cos(ang) - size / 2, oy + r * np.sin(ang) - size / 2), size, size,
                                   fc=INK, ec="none", zorder=2))

        for pr in prongs:
            off, n, sig, dv = pr[:4]; kind = pr[4] if len(pr) > 4 else "h"
            axis = np.pi / 2 + off
            vx, vy = ox + dv * R * np.cos(axis), oy + dv * R * np.sin(axis)
            if dv > 0:
                ax.plot([ox, vx], [oy, vy], color=INK, lw=0.4, ls=(0, (1.2, 1.2)), zorder=1)
                ax.add_patch(Circle((vx, vy), 0.014 * R + 0.012, fc=INK, ec="none", zorder=4))
            for _ in range(n):
                d = float(np.clip(rng.normal(0, sig), -2.2 * sig, 2.2 * sig))
                aend = clip(axis + d + (rng.normal(0, 0.03) if kind == "h" else 0.0))
                ex, ey = ox + rt * np.cos(aend), oy + rt * np.sin(aend)
                t = np.linspace(0, 1, 40)
                bend = rng.normal(0, 0.05) * R if kind == "h" else 0.0
                nx, ny = -(ey - vy), (ex - vx)
                nn = np.hypot(nx, ny) + 1e-9
                x = vx + (ex - vx) * t + bend * nx / nn * 2 * t * (1 - t)
                y = vy + (ey - vy) * t + bend * ny / nn * 2 * t * (1 - t)
                if kind == "m":                      # a muon crosses the calorimeters and leaves
                    x = np.append(x, ox + 1.13 * R * np.cos(aend)); y = np.append(y, oy + 1.13 * R * np.sin(aend))
                ax.plot(x, y, color=INK, lw=lw * (1.6 if kind != "h" else 1.0), alpha=0.9, zorder=1, solid_capstyle="round")
                if hits:
                    r = np.hypot(x - ox, y - oy)
                    for k in range(1, layers + 1):
                        j = int(np.argmin(np.abs(r - rt * k / (layers + 0.6))))
                        ax.add_patch(Circle((x[j], y[j]), 0.008 * R, fc=INK, ec="none", zorder=2))
                if kind == "e":                      # an electron stops in the inner calorimeter
                    for _ in range(6):
                        square(R * rng.uniform(0.68, 0.75), clip(aend + rng.normal(0, 0.02)), R * 0.026 * sq)
                elif kind == "h":
                    for (r0, r1), sizes, nmax in (((0.665, 0.765), (0.016, 0.022), 3), ((0.84, 0.98), (0.024, 0.034), 4)):
                        for _ in range(rng.integers(1, nmax)):
                            square(R * rng.uniform(r0, r1), clip(aend + rng.normal(0, dep)), R * rng.choice(sizes) * sq)
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
    yjet, ylab = H - 1.82, H - 2.04
    yfine, ycoarse = H - 2.42, H - 3.12
    SQ = 0.15
    for k, (_, lab, x, prongs, half) in enumerate(EX):
        jet(x, yjet, 1.08, prongs, half=half, seed=20 + k, layers=3, hits=False, lw=0.3, sq=1.3, dep=0.02)
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
    rh, gap = 0.28, 0.11; ytop = ycoarse - 1.5
    rowy = [ytop - j * (rh + gap) for j in range(4)]
    x0, x1 = 1.0, 12.8; pitch = (x1 - x0) / 188
    for y, n in zip(rowy, LV):
        ax.add_patch(Rectangle((0.2, y + rh / 2 - 0.07), 0.14, 0.14, fc=C[n], ec="none"))
        ax.text(0.43, y + rh / 2 - 0.01, str(n), va="center", ha="left", fontsize=FS)
        g = 0.009 if n >= 162 else 0.017
        for s0, k in runs(levels[n]):
            gg = 0.017 if (n == 162 and k > 1) else g
            ax.add_patch(Rectangle((x0 + s0 * pitch + gg, y), k * pitch - 2 * gg, rh, fc=C[n], ec="none", zorder=2))
    for name, a_, b_ in (("2 prongs", 0, N2), ("3 and 4 prongs", N2, 188 - NQ), ("QCD", 188 - NQ, 188)):
        xa, xb, yy = x0 + a_ * pitch + 0.02, x0 + b_ * pitch - 0.02, ytop + rh + 0.13
        line([xa, xa, xb, xb], [yy - 0.06, yy, yy, yy - 0.06], lw=0.6)
        ax.text((xa + xb) / 2, yy + 0.22, name, ha="center", va="center", fontsize=FS)
    ybot = rowy[-1]

    # ------------------------------------------------------------------ (b) pretraining
    yt = ybot - 0.62
    ax.text(0.2, yt, "(b) Pretraining and use", fontsize=FS_T, va="center")
    y0 = yt - 1.5
    # input: the particles of one jet (tracks and energy deposits; no detector hits)
    jet(1.35, y0 - 1.0, 1.95, [(-0.23, 5, 0.045, 0.0), (0.23, 5, 0.045, 0.0)], half=27, seed=12, lw=0.3, hits=False)
    ax.text(1.35, y0 + 1.13, "Jet", ha="center", va="center", fontsize=FS)
    arrow((2.65, y0), (3.3, y0))
    xb0, wb = 3.35, 1.0                      # the network that is reused
    block(xb0, y0 - 1.0, wb, 2.0, "Sophon's\nnetwork", rot=90, stack=2)      # the tables' name for it
    arrow((xb0 + wb + 0.2, y0), (5.3, y0))
    xf = 5.52
    for k in range(6):
        ax.add_patch(Circle((xf, y0 - 0.5 + 0.2 * k), 0.065, fc="white", ec=SLATE, lw=0.7, zorder=3))
    ax.text(xf, y0 + 0.8, "Features", ha="center", va="center", fontsize=FS)
    arrow((xf + 0.22, y0), (xf + 0.92, y0))
    block(xf + 0.97, y0 - 0.3, 0.95, 0.6, "Head")
    # one pretraining per label set: the head's output is ONE of the four rows (brace, not a fan of arrows)
    xe = xf + 2.55
    arrow((xf + 1.92, y0), (xe - 0.12, y0))

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

    PH, PAD, TG = 0.38, 0.1, 0.06                  # pill height, side padding, gap between pills
    TASKS = [("JetClass-II", ("188 classes",), True),
             ("JetClass", (r"$H$", r"$W$", r"$Z$", r"$t$", "QCD"), True),
             ("Top tagging", (r"$t$", "QCD"), True),
             ("Quark/gluon", (r"$q$", r"$g$"), True),
             (r"$X\!\to b\bar b$, $X\!\to bs$", (r"$b\bar b$", r"$bs$", "QCD"), False)]   # fine-tuned on the first four
    pw = lambda t: max(0.44, tw(t) + 2 * PAD)
    tokw = [[pw(t) for t in toks] for _, toks, _ in TASKS]
    grpw = [sum(ws) + TG * (len(ws) - 1) for ws in tokw]
    colw = [max(g, tw(nm)) for g, (nm, _, _) in zip(grpw, TASKS)]
    BX, BW, BH = 0.34, 1.7, 0.74                   # the two row-label blocks
    XL, XR = BX + BW + 0.3, W - 0.13
    GAP = (XR - XL - sum(colw)) / (len(colw) - 1)  # equal gaps between the five columns
    cx, x = [], XL
    for cw_ in colw:
        cx.append(x + cw_ / 2); x += cw_ + GAP
    yb_, ya_ = y0 - 1.28, y0 - 1.52                # feed lines: block (outer), features (inner)
    ysym = ya_ - 0.5; ynam = ysym - 0.47
    yF = ynam - 0.6; yT = yF - 0.86
    # fine-tuning starts from the pretrained network; frozen features come out of the Features column
    xbk = xb0 + wb / 2
    line([xbk, xbk, 0.14, 0.14], [y0 - 1.0, yb_, yb_, yT]); arrow((0.14, yT), (BX - 0.03, yT), ms=5)
    xfc = BX + BW / 2
    line([xf, xf, xfc], [y0 - 0.62, ya_, ya_]); arrow((xfc, ya_), (xfc, yF + BH / 2 + 0.03), ms=5)
    block(BX, yF - BH / 2, BW, BH, "Frozen\nfeatures")
    block(BX, yT - BH / 2, BW, BH, "Fine-tuning")
    MS = 0.17
    line([BX + BW, cx[-1]], [yF, yF], lw=0.7)
    line([BX + BW, max(c for c, t in zip(cx, TASKS) if t[2])], [yT, yT], lw=0.7)
    for c, ws, (nm, toks, ft) in zip(cx, tokw, TASKS):
        xx = c - (sum(ws) + TG * (len(ws) - 1)) / 2
        for wd, t in zip(ws, toks):
            ax.add_patch(Rectangle((xx, ysym - PH / 2), wd, PH, fc=BLOCK, ec=SLATE, lw=0.7, joinstyle="miter", zorder=2))
            ax.text(xx + wd / 2, ysym - 0.01, t, ha="center", va="center", fontsize=FS, zorder=3)
            xx += wd + TG
        ax.text(c, ynam, nm, ha="center", va="center", fontsize=FS)
        ax.add_patch(Rectangle((c - MS / 2, yF - MS / 2), MS, MS, fc="white", ec=INK, lw=0.8, zorder=4))     # frozen: open
        if ft:
            ax.add_patch(Rectangle((c - MS / 2, yT - MS / 2), MS, MS, fc=INK, ec=INK, lw=0.8, zorder=4))     # fine-tuned: filled
    return fig


def draw(path: pathlib.Path) -> None:
    fig = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {path} and its .png ({W:g} x {H:g} cm)")


if __name__ == "__main__":
    draw(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures/mlst/fig1_design.pdf")
