#!/usr/bin/env python3
"""The gallery's palette page (figure round 4): the PI's Tol muted palette (2026-11-07) in four views, its pairwise
CIEDE2000 (palette_check.pi_palette), the mode encoding of style_v5, and F2a and F4-mech in colour and under deuteranopia
and protanopia."""
from __future__ import annotations

import pathlib
import tempfile

import fitz
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import gallery_common as C
import palette_check as PC
import style_v5 as S

MODELS = ("L188", "L162", "R42_Q1", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND", "INIT_BN", "SCRATCH")
CHECKED = MODELS[:7]                    # the seven hues; the greys are told apart by lightness
KINDS = ("normal", "greyscale", "deuteranopia", "protanopia")


def _seen(colours: list, kind: str) -> np.ndarray:
    rgb = np.array([PC.rgb(h) for h in colours])
    if kind == "greyscale":
        g = PC.to_srgb(PC.to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722]))
        return np.repeat(g[:, None], 3, 1)
    return rgb if kind == "normal" else np.clip(PC.simulate(rgb, kind), 0, 1)


def _render(draw, tmp: pathlib.Path, stem: str) -> np.ndarray:
    draw(tmp / stem)
    page = fitz.open(tmp / f"{stem}.pdf")[0]
    zoom = 7.0 / 2.54 * 150 / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3) / 255.0


def _swatches(fig, top: float) -> None:
    """One row per model: its name, its colour in the four views, its lightness L*."""
    ax = S.axes_cm(fig, 0.2, top, 14.6, 0.6 + 0.55 * len(MODELS))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0.6 + 0.55 * len(MODELS), 0)
    ax.axis("off")
    for j, kind in enumerate(KINDS):
        ax.text(5.6 + 1.9 * j, 0.3, kind, ha="center", va="center", fontsize=S.LEGEND_PT5)
    ax.text(13.3, 0.3, "$L^*$", ha="center", va="center", fontsize=S.LEGEND_PT5)
    for i, key in enumerate(MODELS):
        y = 0.6 + 0.55 * i
        ax.text(0.0, y + 0.22, S.NAME[key], ha="left", va="center", fontsize=S.LEGEND_PT5)
        for j, kind in enumerate(KINDS):
            ax.add_patch(Rectangle((5.0 + 1.9 * j, y + 0.02), 1.2, 0.4, fc=_seen([S.SERIES[key]], kind)[0], ec="none"))
        ax.text(13.3, y + 0.22, f"{PC.lab(PC.rgb(S.SERIES[key]))[0]:.0f}", ha="center", va="center",
                fontsize=S.LEGEND_PT5)


def _matrix(fig, x: float, top: float, kinds: tuple, title: str) -> None:
    """Lower triangle of the pairwise CIEDE2000 among the seven hues (the smaller over kinds); rows and columns named
    by colour patches in the swatch order; a pair below the rule (25 in normal vision, 10 simulated) in bold, boxed."""
    n, cell = len(CHECKED), 0.85
    ax = S.axes_cm(fig, x, top, 0.5 + n * cell, 0.5 + n * cell)
    ax.set_xlim(-0.5, n * cell)
    ax.set_ylim(n * cell, -0.5)
    ax.axis("off")
    rule = PC.DE_MIN if kinds == ("normal",) else PC.DE_CVD
    for i, key in enumerate(CHECKED):
        ax.add_patch(Rectangle((-0.45, i * cell + 0.25), 0.35, 0.35, fc=S.SERIES[key], ec="none"))
        ax.add_patch(Rectangle((i * cell + 0.25, -0.45), 0.35, 0.35, fc=S.SERIES[key], ec="none"))
        for j in range(i):
            d = PC.separation(S.SERIES[key], S.SERIES[CHECKED[j]])
            v = min(d[k] for k in kinds)
            low = v < rule
            ax.text(j * cell + cell / 2, i * cell + cell / 2, f"{v:.1f}", ha="center", va="center",
                    fontsize=S.LEGEND_PT5, fontweight="bold" if low else "normal")
            if low:
                ax.add_patch(Rectangle((j * cell + 0.05, i * cell + 0.05), cell - 0.1, cell - 0.1, fc="none",
                                       ec="#000000", lw=0.6))
    ax.set_title(title, fontsize=S.LEGEND_PT5, pad=2)


def draw_palette(path: pathlib.Path) -> None:
    import gallery_labelset
    import gallery_roc
    check = PC.pi_palette()
    swatch_h = 0.6 + 0.55 * len(MODELS)
    m_top = 0.9 + swatch_h + 0.6
    m_h = 0.5 + len(CHECKED) * 0.85
    modes_top = m_top + m_h + 1.4
    render_top = modes_top + 1.6
    fig = S.figure_cm(S.PRINT_W, render_top + 2 * 7.2 + 0.3)
    S.heading(fig, "Palette (Tol muted): the models' colors, their separation, and F2a and F4-mech in it")
    _swatches(fig, 0.9)
    _matrix(fig, 0.9, m_top, ("normal",), "(a) CIEDE2000, normal vision")
    _matrix(fig, 8.4, m_top, ("deuteranopia", "protanopia"), "(b) smaller of deuteranopia and protanopia")
    swap = "; ".join(f"{w['control'].capitalize()}: {w['from']} swapped for {w['to']}" for w in check["swaps"]) or "No swap"
    fig.text(0.5, 1 - (m_top + m_h + 0.55) / (fig.get_figheight() / S.CM),
             f"Boxed: below the rule (25; 10 simulated). {swap}.", ha="center", va="top", fontsize=S.LEGEND_PT5)
    ax2 = S.axes_cm(fig, 0.2, modes_top, 14.6, 1.3)
    ax2.set_xlim(0, 14.6)
    ax2.set_ylim(1.3, 0)
    ax2.axis("off")
    modes = [("Fine-tuned, or one mode per panel: solid, filled", S.look("L188", "ft"), S.marker("L188", "ft")),
             ("Linear probe beside fine-tuned: dashed, hollow", S.look("L188", "probe"), S.marker("L188", "probe")),
             ("Mass regression beside its label set: dashed, hollow", S.look("L162_MASS", "mass"), S.marker("L162_MASS", "mass")),
             ("From scratch: dashed, dark grey", S.look("SCRATCH"), {}),
             ("Kinematic inputs only: dotted", S.look("KIN"), {}),
             ("F8 only: probe tint of 188", {"color": S.TINT_188, "linestyle": "--"},
              {"marker": "s", "markerfacecolor": "white", "markeredgecolor": S.TINT_188})]
    for k, (word, line, mark) in enumerate(modes):
        x, y = 0.1 + 7.3 * (k % 2), 0.25 + 0.42 * (k // 2)
        ax2.add_line(Line2D([x, x + 0.9], [y, y], linewidth=S.LW_CURVE, **line))
        if mark:
            ax2.add_line(Line2D([x + 0.45], [y], markersize=4, linestyle="none", **mark))
        ax2.text(x + 1.05, y, word, va="center", fontsize=S.LEGEND_PT5)
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        imgs = [_render(gallery_labelset.draw_f2a, tmp, "F2a"), _render(gallery_roc.draw_f4_mech, tmp, "F4-mech")]
    for i, img in enumerate(imgs):
        for j, kind in enumerate(("normal", "deuteranopia", "protanopia")):
            a = S.axes_cm(fig, 0.2 + 4.9 * j, render_top + 7.2 * i, 4.7, 6.8)
            a.imshow(img if kind == "normal" else np.clip(PC.simulate(img, kind), 0, 1), interpolation="lanczos")
            a.set_aspect("equal")
            a.axis("off")
            a.set_title(f"{('F2a', 'F4-mech')[i]}, {kind}", fontsize=S.LEGEND_PT5)
    S.save(fig, path)


PAGES = [
    C.Page(
        id="PALETTE",
        title="Palette (round 4)",
        precedent=[],
        precedent_label="None: the palette check of round 4",
        caption="The palette. Upper: each model's colour in normal vision, greyscale and simulated deuteranopia and "
                "protanopia, with its lightness. Middle: pairwise CIEDE2000 among the seven hues, and the line and marker "
                "of each mode. Lower: F2a and F4-mech in this palette, in colour and under the two simulations.",
        differs="Not a paper figure. Palette (PI, 2026-11-07): Tol muted indigo, purple, olive and teal for 188, 162, 43 "
                "and 17 classes; controls rose, green and sand, a colliding control swapped for Tol cyan or wine. Rule for "
                "a collision: the label sets' own (CIEDE2000 at least 25; at least 10 under either simulation, Machado "
                "2009). Sand is 16 from olive, so the random flavor boundary is cyan; wine would sit 15 from purple. "
                "Three pairs stay below 25 with no spare that helps: rose and purple 19.8, cyan and teal 23.2, green and "
                "teal 24.6. Among the label sets, indigo and purple are 24.8.",
        draw=draw_palette),
]
