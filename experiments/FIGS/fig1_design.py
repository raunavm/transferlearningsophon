#!/usr/bin/env python3
"""Fig. 1, the design of the study (PI brief and sketch 2026-11-12; finish: Fig. 1 of arXiv:2610.41889).

(a) left   three real JetClass-II test jets (experiments/FIGS/data/v2/fig1_jets.json, written by pick_fig1_jets.py)
           as a transverse event display: charged particles as tracks to the inner circle, every particle a square
           in the ring (size by pT), each quark of the decay a short arrow. Schematic: a particle's azimuth is its
           dphi from the jet axis, stretched, about the jet's place on the ring; its depth in the ring is its deta.
(a) right  the four label sets as squares: every one of the 188 classes one cell of the top row of the strip, the
           classes a coarser label set merges one bar; above it the two-prong classes enlarged. Read from
           configs/labelmaps/rung_label_maps.v1.csv; the script stops unless the rows count 188, 162, 43 and 17 and
           every class of every label set is contiguous.
(b)        pretraining and use: particles -> Particle Transformer -> features -> head -> one output layer per label
           set; from the features, a linear probe or fine-tuning, on five tasks.

    python3 experiments/FIGS/fig1_design.py [out.pdf]     (default figures/mlst/fig1_design.pdf, and a .png beside it)

Placed in centimetres on a 15 cm page (the text width). No text under 8 pt, none bold, and no words but short labels.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAP = ROOT / "configs/labelmaps/rung_label_maps.v1.csv"
JETS = HERE / "data/v2/fig1_jets.json"
FONTS = HERE / "figstyle" / "fonts"
CM = 1 / 2.54
W, H = 15.0, 10.5
PT, TITLE_PT = 8.0, 9.0
SETS = (("L188", 188, "#332288"), ("L162", 162, "#AA4499"), ("R42_Q1", 43, "#999933"), ("R16_Q1", 17, "#44AA99"))
BLOCK_FC, BLOCK_EC = "#DDE4EE", "#5B6B8A"
GREY = "#707070"            # arrows
SLATE = "#46526E"           # tracks and deposits
RING_EC, RING_FC = "#C8C8C8", "#F4F4F4"
LW_BLOCK, LW_ARROW = 0.7, 0.9
# the three jets of the display: (class in fig1_jets.json, its place on the ring in degrees, its label)
SHOWN = (("label_X_bb", 90.0, r"$X\to b\bar{b}$"), ("label_X_YY_qqqq", 205.0, r"$X\to YY\to 4q$"),
         ("label_QCD_light", 335.0, "QCD"))
STRETCH = 62.0              # degrees on the ring per unit of dphi


# ------------------------------------------------------------------ the label sets, from the map
def label_sets():
    """(rows, bars, counts): the 188 classes sorted so that every class of every label set is contiguous, grouped as
    two prongs, three and four prongs, QCD; per label set the (first, last) cell of each of its classes; the number
    of classes in each of the three groups."""
    rows = list(csv.DictReader(MAP.open()))
    group = lambda r: 2 if r["R16_Q1_name"].startswith("QCD") else 0 if r["R16_Q1_name"].startswith("2P") else 1
    rows.sort(key=lambda r: (group(r), int(r["R16_Q1"]), int(r["R42_Q1"]), int(r["L162"]), int(r["L188"])))
    bars = []
    for col, n, _ in SETS:
        ids = [r[col] for r in rows]
        runs, k = [], 0
        while k < len(ids):
            j = k
            while j + 1 < len(ids) and ids[j + 1] == ids[k]:
                j += 1
            runs.append((k, j))
            k = j + 1
        if len(runs) != n or len(set(ids)) != n:
            raise SystemExit(f"FATAL: {col} has {len(set(ids))} classes in {len(runs)} runs of cells, not {n} contiguous ones")
        bars.append(runs)
    groups = [group(r) for r in rows]
    return rows, bars, [groups.count(g) for g in (0, 1, 2)]


def symbol(class_name: str) -> str:
    """The short symbol of a two-prong class, from its name in the map (label_X_bb -> b b-bar)."""
    s = class_name.removeprefix("label_X_")
    tau = {"tauhtaue": r"$\tau e$", "tauhtaum": r"$\tau\mu$", "tauhtauh": r"$\tau\tau$"}
    if s in tau:
        return tau[s]
    if s == "mm":
        return r"$\mu\mu$"
    if len(s) == 2 and s[0] == s[1] and s[0] in "bcsq":
        return rf"${s[0]}\bar{{{s[0]}}}$"
    if len(s) == 2:
        return f"${s}$"
    raise SystemExit(f"FATAL: no symbol for {class_name}")


# ------------------------------------------------------------------ helpers
def block(ax, x, y, w, h, text="", rotation=0, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09", fc=BLOCK_FC, ec=BLOCK_EC,
                                lw=LW_BLOCK, zorder=z))
    if text:
        return ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=PT, rotation=rotation, zorder=z + 1,
                       linespacing=1.05)


def line(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=GREY, lw=LW_ARROW, solid_capstyle="butt", solid_joinstyle="miter", zorder=2)


def arrow(ax, pts):
    if len(pts) > 2:
        line(ax, pts[:-1])
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=3.0,head_width=1.5", mutation_scale=1.0,
                                 color=GREY, lw=LW_ARROW, shrinkA=0, shrinkB=0, zorder=2))


def at(c, r, deg):
    a = math.radians(deg)
    return c[0] + r * math.cos(a), c[1] + r * math.sin(a)


# ------------------------------------------------------------------ (a) left: the event display
def display(ax, c, r_in, r_out) -> None:
    ax.add_patch(Circle(c, r_out, fc=RING_FC, ec=RING_EC, lw=0.7, zorder=1))
    ax.add_patch(Circle(c, r_in, fc="white", ec=RING_EC, lw=0.7, zorder=1.1))
    jets = {j["class_name"]: j for j in json.loads(JETS.read_text())["jets"]}
    for name, place, _ in SHOWN:
        j = jets[name]
        pt = np.hypot(j["part_px"], j["part_py"])
        dphi, deta, q = np.array(j["part_dphi"]), np.array(j["part_deta"]), np.array(j["part_charge"])
        for p, a, e, ch in sorted(zip(pt, dphi, deta, q)):
            deg = place + STRETCH * a
            if ch != 0 and p > 1.0:                 # a charged particle: a curved track to the inner circle
                end = at(c, r_in, deg)
                bend = ch * min(0.09, 0.012 + 0.5 / p)
                mid = ((c[0] + end[0]) / 2 - bend * (end[1] - c[1]), (c[1] + end[1]) / 2 + bend * (end[0] - c[0]))
                ax.add_patch(PathPatch(Path([c, mid, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", ec=SLATE,
                                       lw=0.4, zorder=2, capstyle="round"))
            depth = min(max(0.5 + e / 1.0, 0.14), 0.86)
            x, y = at(c, r_in + depth * (r_out - r_in), deg)
            s = 0.04 + 0.09 * math.sqrt(p / pt.max())
            ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=SLATE, ec="none", zorder=3))
        # the quarks of the decay (for QCD: the hard partons), as short arrows along their directions
        gp = zip(j["aux_genpart_pt"], j["aux_genpart_phi"], j["aux_genpart_pid"], j["aux_genpart_isResDecayProd"],
                 j["aux_genpart_isQcdParton"])
        for gpt, gphi, pid, prod, qcd in gp:
            if not (prod or qcd) or abs(pid) not in (1, 2, 3, 4, 5, 21) or gpt < 100:
                continue
            d = (gphi - j["jet_phi"] + math.pi) % (2 * math.pi) - math.pi
            ax.add_patch(FancyArrowPatch(c, at(c, 0.6 * r_in, place + STRETCH * d),
                                         arrowstyle="-|>,head_length=3.4,head_width=1.9", mutation_scale=1.0, color="#1E1E1E",
                                         lw=1.5, shrinkA=0, shrinkB=0, zorder=4))


# ------------------------------------------------------------------ the figure
def draw(path: pathlib.Path) -> None:
    for f in sorted(FONTS.glob("lmroman10-*.otf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "pdf.fonttype": 42, "font.size": PT})
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    boxed = []

    # ---------------- (a)
    ax.text(0.25, H - 0.15, "(a) Label sets", ha="left", va="top", fontsize=TITLE_PT)
    c, r_in, r_out = (2.45, 7.2), 1.3, 1.78
    display(ax, c, r_in, r_out)
    ax.text(c[0], c[1] + r_out + 0.12, SHOWN[0][2], ha="center", va="bottom", fontsize=PT)
    ax.text(c[0] - 1.25, c[1] - r_out - 0.02, SHOWN[1][2], ha="center", va="top", fontsize=PT)
    ax.text(*at(c, r_out + 0.1, 323.0), SHOWN[2][2], ha="left", va="top", fontsize=PT)

    rows, bars, counts = label_sets()
    x0, x1 = 5.85, 14.75
    n2 = counts[0]
    # the zoom: the two-prong classes
    zw = (x1 - x0) / n2
    zh, zpitch, ztop = 0.3, 0.46, 9.0
    for k in range(n2):
        ax.text(x0 + (k + 0.5) * zw - 0.03, ztop + zh + 0.12, symbol(rows[k]["class_name"]), ha="center", va="bottom",
                fontsize=PT)
    for i, ((_, n, colour), runs) in enumerate(zip(SETS, bars)):
        y = ztop - i * zpitch
        ax.add_patch(Rectangle((x0 - 0.98, y + zh / 2 - 0.09), 0.18, 0.18, fc=colour, ec="none"))
        ax.text(x0 - 0.7, y + zh / 2, str(n), ha="left", va="center", fontsize=PT)
        for a, b in runs:
            if a < n2:
                ax.add_patch(Rectangle((x0 + a * zw, y), (min(b, n2 - 1) - a + 1) * zw - 0.06, zh, fc=colour, ec="none"))
    # the strip: all 188
    cw = (x1 - 0.046 - x0) / len(rows)        # its last cell ends where the zoom's last bar ends
    sh, spitch, stop = 0.22, 0.31, 6.55
    zbot = ztop - 3 * zpitch
    ax.add_patch(Polygon([(x0, zbot - 0.1), (x1 - 0.06, zbot - 0.1), (x0 + n2 * cw, stop + sh + 0.08), (x0, stop + sh + 0.08)],
                         closed=True, fc="#EFEFEF", ec=RING_EC, lw=0.5, zorder=0.5))
    for i, ((_, _, colour), runs) in enumerate(zip(SETS, bars)):
        y = stop - i * spitch
        for a, b in runs:
            ax.add_patch(Rectangle((x0 + a * cw, y), (b - a + 1) * cw - 0.014, sh, fc=colour, ec="none"))
    ax.text(x0 - 0.98, stop + sh / 2, "all 188", ha="left", va="center", fontsize=PT)
    yb = stop - 3 * spitch - 0.12
    edge = 0
    for n, text in zip(counts, ("2 prongs", "3 and 4 prongs", "QCD")):
        a, b = x0 + edge * cw + 0.02, x0 + (edge + n) * cw - 0.03
        ax.plot([a, a, b, b], [yb, yb - 0.08, yb - 0.08, yb], color=GREY, lw=0.6, solid_joinstyle="miter")
        ax.text(a if edge == 0 else (a + b) / 2, yb - 0.14, text, ha="left" if edge == 0 else "center", va="top", fontsize=PT)
        edge += n

    # ---------------- (b)
    ax.text(0.25, 4.42, "(b) Pretraining and use", ha="left", va="top", fontsize=TITLE_PT)
    mid = 2.95
    xp = 0.95
    for i, s in enumerate((0.04, 0.055, 0.04, 0.07, 0.045, 0.055, 0.04)):
        ax.add_patch(Circle((xp, mid + (i - 3) * 0.19), s, fc="#444444", ec="none", zorder=3))
    ax.text(xp, mid - 0.78, "Particles", ha="center", va="top", fontsize=PT)
    xt, wt, ht = 2.0, 0.95, 1.72
    for d in (0.16, 0.08):
        block(ax, xt + d, mid - ht / 2 + d, wt, ht, z=2)
    boxed.append((block(ax, xt, mid - ht / 2, wt, ht, "Particle\nTransformer", rotation=90), ht - 0.16, "h"))
    arrow(ax, [(xp + 0.22, mid), (xt - 0.04, mid)])
    xf = 4.05
    yf = [mid + (i - 2.5) * 0.22 for i in range(6)]
    for y in yf:
        ax.add_patch(Circle((xf, y), 0.075, fc="white", ec=BLOCK_EC, lw=LW_BLOCK, zorder=3))
    ax.text(xf, yf[-1] + 0.2, "Features", ha="center", va="bottom", fontsize=PT)
    arrow(ax, [(xt + wt + 0.2, mid), (xf - 0.16, mid)])
    xh, wh, hh = 4.95, 1.05, 0.62
    boxed.append((block(ax, xh, mid - hh / 2, wh, hh, "Head"), wh - 0.2, "w"))
    arrow(ax, [(xf + 0.16, mid), (xh - 0.04, mid)])
    xo = xh + wh + 0.75
    y_out = [mid + (1.5 - k) * 0.46 for k in range(4)]
    line(ax, [(xh + wh, mid), (xo - 0.4, mid)])
    line(ax, [(xo - 0.4, y_out[0]), (xo - 0.4, y_out[-1])])
    for (_, _, colour), n, y in zip(SETS, (12, 10, 6, 3), y_out):
        arrow(ax, [(xo - 0.4, y), (xo - 0.06, y)])
        for i in range(n):
            ax.add_patch(Rectangle((xo + i * 0.19, y - 0.065), 0.13, 0.13, fc=colour, ec="none", zorder=3))
    # the two uses
    uh, uy = 0.62, 0.78
    uses = (("Linear probe", 4.75, 1.95), ("Fine-tuning", 7.1, 1.95))
    bus = uy + uh + 0.34
    line(ax, [(xf, yf[0] - 0.13), (xf, bus), (uses[1][1] + uses[1][2] / 2, bus)])
    for text, x, w in uses:
        boxed.append((block(ax, x, uy, w, uh, text), w - 0.24, "w"))
        arrow(ax, [(x + w / 2, bus), (x + w / 2, uy + uh + 0.03)])
    # the five tasks, each with its own arrow
    tasks = ("JetClass-II", "JetClass", "Top tagging", "Quark/gluon tagging", r"$X\to b\bar{b}$ and $X\to bs$")
    xtr, yt = 10.75, [3.87 - k * 0.72 for k in range(5)]
    low = 0.36
    for _, x, w in uses:
        line(ax, [(x + w / 2, uy), (x + w / 2, low)])
    line(ax, [(uses[0][1] + uses[0][2] / 2, low), (xtr, low), (xtr, yt[0])])
    for text, y in zip(tasks, yt):
        arrow(ax, [(xtr, y), (xtr + 0.36, y)])
        ax.text(xtr + 0.5, y, text, ha="left", va="center", fontsize=PT)

    # ---------------- checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    small = [t.get_text() for t in ax.texts if t.get_fontsize() < 8.0]
    if small:
        raise SystemExit(f"FATAL: text under 8 pt: {small}")
    for t, room, side in boxed:
        e = t.get_window_extent(rend)
        size = (e.width if side == "w" else e.height) / fig.dpi / CM
        if size > room:
            raise SystemExit(f"FATAL: {t.get_text()!r} takes {size:.2f} cm of {room:.2f} cm")
    for t in ax.texts:
        e = t.get_window_extent(rend)
        if e.x0 < 0 or e.y0 < 0 or e.x1 > fig.bbox.width or e.y1 > fig.bbox.height:
            raise SystemExit(f"FATAL: {t.get_text()!r} leaves the page")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    print(f"wrote {path} and its .png ({W:g} x {H:g} cm); label sets {[len(b) for b in bars]}, groups {counts}")


if __name__ == "__main__":
    draw(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures/mlst/fig1_design.pdf")
