#!/usr/bin/env python3
"""The gallery's palette page: every network colour of style_v5 (the PI's Tol muted palette of 2026-11-07 with the
colours of L6, 2026-11-13) in four views, the pairwise CIEDE2000 of the hues, the line rule of style_v5, and F2a and
F4-mech in colour and under deuteranopia and protanopia."""
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

MODELS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND",
          "KIN", "SOPHON", "INIT_BN", "SCRATCH")       # the common row order (L5), then the references
CHECKED = tuple(k for k in MODELS if k not in ("RAND", "INIT_BN", "SCRATCH"))     # the hues; greys differ by lightness
CELL = 0.62                             # cm, a cell of the separation matrices (eleven hues in half the page)
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
        ax.text(0.0, y + 0.22, S.LEGEND[key], ha="left", va="center", fontsize=S.LEGEND_PT5)
        for j, kind in enumerate(KINDS):
            ax.add_patch(Rectangle((5.0 + 1.9 * j, y + 0.02), 1.2, 0.4, fc=_seen([S.SERIES[key]], kind)[0], ec="none"))
        ax.text(13.3, y + 0.22, f"{PC.lab(PC.rgb(S.SERIES[key]))[0]:.0f}", ha="center", va="center",
                fontsize=S.LEGEND_PT5)


def _matrix(fig, x: float, top: float, kinds: tuple, title: str) -> None:
    """Lower triangle of the pairwise CIEDE2000 among the hues (the smaller over kinds); rows and columns named by colour
    patches in the swatch order; a pair below the rule (25 in normal vision, 10 simulated) in bold, boxed."""
    n, cell = len(CHECKED), CELL
    ax = S.axes_cm(fig, x, top, 0.5 + n * cell, 0.5 + n * cell)
    ax.set_xlim(-0.5, n * cell)
    ax.set_ylim(n * cell, -0.5)
    ax.axis("off")
    rule = PC.DE_MIN if kinds == ("normal",) else PC.DE_CVD
    for i, key in enumerate(CHECKED):
        ax.add_patch(Rectangle((-0.45, i * cell + 0.15), 0.32, 0.32, fc=S.SERIES[key], ec="none"))
        ax.add_patch(Rectangle((i * cell + 0.15, -0.45), 0.32, 0.32, fc=S.SERIES[key], ec="none"))
        for j in range(i):
            d = PC.separation(S.SERIES[key], S.SERIES[CHECKED[j]])
            v = min(d[k] for k in kinds)
            low = v < rule
            ax.text(j * cell + cell / 2, i * cell + cell / 2, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold" if low else "normal")
            if low:
                ax.add_patch(Rectangle((j * cell + 0.05, i * cell + 0.05), cell - 0.1, cell - 0.1, fc="none",
                                       ec="#000000", lw=0.6))
    ax.set_title(title, fontsize=S.LEGEND_PT5, pad=2)


def draw_palette(path: pathlib.Path) -> None:
    import gallery_labelset
    import gallery_roc
    swatch_h = 0.6 + 0.55 * len(MODELS)
    m_top = 0.9 + swatch_h + 0.6
    m_h = 0.5 + len(CHECKED) * CELL
    modes_top = m_top + m_h + 1.4
    render_top = modes_top + 1.6
    fig = S.figure_cm(S.PRINT_W, render_top + 2 * 7.2 + 0.3)
    _swatches(fig, 0.9)
    _matrix(fig, 0.2, m_top, ("normal",), "(a) CIEDE2000, normal vision")
    _matrix(fig, 7.6, m_top, ("deuteranopia", "protanopia"), "(b) smaller of deuteranopia and protanopia")
    fig.text(0.5, 1 - (m_top + m_h + 0.55) / (fig.get_figheight() / S.CM),
             "Boxed: below the label sets' rule (25; 10 simulated). A tint is close to its own label set by design.",
             ha="center", va="top", fontsize=S.LEGEND_PT5)
    ax2 = S.axes_cm(fig, 0.2, modes_top, 14.6, 1.3)
    ax2.set_xlim(0, 14.6)
    ax2.set_ylim(1.3, 0)
    ax2.axis("off")
    modes = [("Fine-tuned or only use: solid, filled", S.look("L188", "ft"), S.marker("L188", "ft")),
             ("Probe beside fine-tuned: dashed, hollow", S.look("L188", "probe"), S.marker("L188", "probe")),
             ("Mass network: lighter shade, solid", S.look("L162_MASS"), S.marker("L162_MASS")),
             ("Direct discriminant: dash-dot", S.look("L188", "direct"), {}),
             ("From scratch: grey, solid", S.look("SCRATCH"), {}),
             ("Reference line: black, dotted", S.REFERENCE, {})]
    for k, (word, line, mark) in enumerate(modes):
        x, y = 0.1 + 7.3 * (k % 2), 0.25 + 0.42 * (k // 2)
        ax2.add_line(Line2D([x, x + 0.9], [y, y], **{"linewidth": S.LW_CURVE, **line}))
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
        caption="The palette. Upper: each network's colour in normal vision, greyscale and simulated deuteranopia and "
                "protanopia, with its lightness. Middle: pairwise CIEDE2000 among the hues, and the line and marker "
                "of each use. Lower: F2a and F4-mech in this palette, in colour and under the two simulations.",
        differs="Not a paper figure. Palette (PI, 2026-11-07): Tol muted indigo, purple, olive and teal for 188, 162, 43 "
                "and 17 classes; controls rose, green and sand, a colliding control swapped for Tol cyan or wine. Rule for "
                "a collision: the label sets' own (CIEDE2000 at least 25; at least 10 under either simulation, Machado "
                "2009). Sand is 16 from olive, so the random flavor boundary is cyan; wine would sit 15 from purple. "
                "Three pairs stay below 25 with no spare that helps: rose and purple 19.8, cyan and teal 23.2, green and "
                "teal 24.6. Among the label sets, indigo and purple are 24.8.",
        draw=draw_palette),
]
