#!/usr/bin/env python3
"""The gallery's inventory page (figure round 4, docs/figstyle/ROUND4_BRIEF.md Part 5): every figure of the paper,
main text and appendix in paper order, with its panels, its series and the encoding it uses. Colour encodes the
model everywhere except F10a's working points and the class and family figures (JetClass classes, JetClass-II
families), which use the fixed tab10 set."""
from __future__ import annotations

import pathlib

import gallery_common as C
import style_v5 as S

LS = "label sets: palette colours, solid, filled"
ROC = "solid, no markers"
# (figure, place, panels, series, encoding)
ROWS = [
    ("F1", "main", "2: design, the two uses", "four label-set columns", "label-set colours (tint)"),
    ("F2a", "main", "4 tasks", "4 label sets, 3 controls, 5 random, scratch", LS + "; controls own colours left of 17"),
    ("F2b", "main", "4 tasks", "4 label sets, 3 controls, scratch", "as F2a"),
    ("F3", "main", "1 heatmap", "6 models × 188 classes", "diverging map; no model colour"),
    ("F4", "main", "4 class pairs", "4 label sets", "label-set colours; " + ROC),
    ("FJ", "main", "4 label sets", "10 JetClass classes", "tab10 classes; " + ROC),
    ("FJ-ctrl", "main", "4 controls", "10 JetClass classes", "tab10 classes; " + ROC),
    ("PT", "main", "4: JetClass-II, controls, JetClass, X→bb̄ vs cc̄", "4 label sets, 3 controls", LS + "; SD bands"),
    ("F6", "main", "3 + ratios", "4 label sets, scratch, kinematic", ROC + "; scratch dashed, kinematic dotted"),
    ("F8", "main", "4 tasks", "188 fine-tuned, probe, scratch", "188 colour; probe tint dashed hollow; SD bands"),
    ("F9a", "main", "2 jet groups", "soft-drop, 2 mass outputs", "black reference; label-set colours, solid"),
    ("F9b", "main", "2×2 + ratio/RSD", "soft-drop, 2 mass outputs", "as F9a, markers filled"),
    ("F10a", "main", "4 label sets + ratios", "inclusive, 10%, 1%", "working points black and greys (exception)"),
    ("F10b", "main", "1", "4 label sets, 2 mass outputs", LS + "; mass outputs dashed hollow beside them"),
    ("F11", "main", "placeholder", "waits for the X→bb̄ and X→bs samples", "-"),
    ("VAL", "appendix", "1", "4 label sets", "label-set colours, solid"),
    ("A10", "appendix", "4 tasks", "4 label-set probes, scratch", LS + "; scratch dashed"),
    ("A8", "appendix", "4 tasks", "4 label sets fine-tuned, scratch", LS + "; scratch dashed"),
    ("F6-app", "appendix", "3 + ratios", "4 label-set probes, scratch", ROC + "; scratch dashed"),
    ("FJ-app", "appendix", "4 label sets", "10 JetClass classes", "tab10 classes; " + ROC),
    ("FJ-ctrl-app", "appendix", "4 controls", "10 JetClass classes", "tab10 classes; " + ROC),
    ("FJ-s", "appendix", "1", "10 JetClass classes", "tab10 classes; " + ROC),
    ("CONF", "appendix", "4 matrices", "188; 162/43/17 minus 188", "sequential and diverging maps"),
    ("F4-app", "appendix", "4 class pairs", "3 controls, 188", "palette colours; " + ROC),
    ("F3-scatter", "appendix", "2", "188 classes per panel", "tab10 families, filled circles"),
    ("F3-app", "appendix", "4", "188 classes per panel", "tab10 families"),
    ("F3-app2", "appendix", "3", "188 classes per panel", "tab10 families"),
    ("F3-app3", "appendix", "2", "188 classes per panel", "tab10 families"),
    ("F3-ft", "appendix", "4", "188 classes per panel", "tab10 families"),
    ("F3-ft2", "appendix", "3", "188 classes per panel", "tab10 families"),
    ("A3R", "appendix", "4 label sets", "188 classes", "tab10 families, thin; " + ROC),
    ("A3R-ft", "appendix", "4 label sets", "188 classes", "tab10 families, thin; " + ROC),
    ("UMAP-jc1", "appendix", "2×2", "10 JetClass classes", "tab10 classes, points"),
    ("UMAP-jc2-had", "appendix", "2×2", "4 classes", "tab10, points"),
    ("UMAP-jc2-lep", "appendix", "2×2", "2 classes", "tab10, points"),
    ("UMAP-top", "appendix", "2×2", "top, QCD", "tab10, points"),
    ("UMAP-qg", "appendix", "2×2", "quark, gluon", "tab10, points"),
    ("A17-res", "appendix", "2×2 + ratio/RSD", "soft-drop, 2 mass outputs", "as F9b"),
    ("A17-qcd", "appendix", "2×2 + ratio/RSD", "soft-drop, 2 mass outputs", "as F9b"),
    ("A17-probe", "appendix", "2×2 + ratio/RSD", "4 MLP probes, 2 mass outputs", LS + "; mass outputs dashed hollow"),
    ("F10a-mass", "appendix", "2 + ratios", "inclusive, 10%, 1%", "as F10a"),
    ("F10a-flav", "appendix", "3 + ratios", "inclusive, 10%, 1%", "as F10a"),
]
COLS = (("Figure", 0.0), ("Place", 2.15), ("Panels", 3.5), ("Series", 6.45), ("Encoding", 10.1))
ROW_H = 0.5


def draw_inventory(path: pathlib.Path) -> None:
    import textwrap
    wrapped = [[textwrap.wrap(c, w) or [""] for c, w in zip(r, (12, 9, 20, 24, 30))] for r in ROWS]
    heights = [max(len(c) for c in r) * ROW_H * 0.82 + 0.12 for r in wrapped]
    H = 1.6 + sum(heights)
    fig = S.figure_cm(S.PRINT_W, H + 0.4)
    ax = S.axes_cm(fig, 0.3, 0.9, 14.5, H - 0.5)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(H - 0.5, 0)
    ax.axis("off")
    for name, x in COLS:
        ax.text(x, 0.2, name, fontsize=S.LEGEND_PT5, fontweight="bold", va="center")
    ax.axhline(0.45, color="#000000", linewidth=0.6)
    y = 0.6
    for r, h in zip(wrapped, heights):
        for (_, x), lines in zip(COLS, r):
            ax.text(x, y, "\n".join(lines), fontsize=8.0, va="top", linespacing=1.05)
        y += h
        ax.axhline(y - 0.06, color="#BBBBBB", linewidth=0.3)
    S.heading(fig, "Inventory: every figure of the paper, in paper order")
    S.save(fig, path)


PAGES = [
    C.Page(
        id="INVENTORY",
        title="Inventory (round 4)",
        precedent=[],
        precedent_label="None: the inventory the round-4 brief asks for",
        caption="Every figure of the paper in paper order, main text then appendix: its panels, its series and the "
                "colour and line encoding it uses. Colour names the model in every figure except the working points "
                "of the mass-sculpting figures, the heatmaps, and the class and family figures, which use one fixed ten-color set.",
        differs="Not a paper figure.",
        draw=draw_inventory),
]
