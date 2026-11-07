#!/usr/bin/env python3
"""Compare our redrawn Sophon Fig. 2 with the native PDF on the attributes that carry the data:

    python3 experiments/FIGS/figstyle/compare_fig2.py <figures-dir>/SOPHON-FIG2.pdf docs/figstyle/pdfs/2405.12972/src/roc_benchmark.pdf

Round 3 (2026-11-06) draws the page through style_v4, whose grid, spines, ticks, legend frame and palette are on purpose
not Sophon's, so a pixel count (this script until then, by the method of docs/figstyle/sophon_tools/replot_fig2_check.py)
would measure those choices. This reads the vector strokes of both PDFs instead and compares, panel by panel: the axes
box (width and height in pt, from the black spines), the number of curves, and for each curve in drawing order its line
width, dash pattern, and the count and positions (in axes coordinates, which fix the scales and limits of both axes) of
its vertices inside the axes. Vertices outside the axes are not compared: they are clipped away on the page, and
matplotlib 3.10 drops the few that lie beyond the figure edge where Sophon's 3.4.3 kept them. Colour, grid, ticks and
legend are not compared. Writes the result to fig2_check.txt beside the first PDF.
"""
import pathlib
import sys

import fitz
import numpy as np


def _segments(d) -> list:
    return [(it[1], it[2]) for it in d["items"] if it[0] == "l"]


def read(path: str) -> list:
    """[(axes box (x0, y0, x1, y1) in pt, y down; [(width, dashes, vertices inside the axes (n, 2), in axes
    coordinates)])], top panel first."""
    page = fitz.open(path)[0]
    drawings = page.get_drawings()
    spines = [(p, q) for d in drawings if d.get("color") and max(d["color"]) < 0.01 and d.get("dashes") in (None, "[] 0")
              for p, q in _segments(d) if abs(p.y - q.y) < 0.01 and abs(q.x - p.x) > 50]       # long black horizontals
    ys = sorted({round(p.y, 2) for p, _ in spines})
    x0, x1 = min(min(p.x, q.x) for p, q in spines), max(max(p.x, q.x) for p, q in spines)
    if len(ys) != 4:
        sys.exit(f"{path}: expected the top and bottom spines of two axes, found horizontal black lines at y = {ys}")
    boxes = [(x0, ys[0], x1, ys[1]), (x0, ys[2], x1, ys[3])]
    panels = [[] for _ in boxes]
    for d in drawings:
        seg = _segments(d)
        if d.get("width") is None or d["width"] < 1.4 or len(seg) < 20:       # curves only: not spines, ticks, grid or legend keys
            continue
        xy = np.array([[seg[0][0].x, seg[0][0].y]] + [[q.x, q.y] for _, q in seg])
        k = int(np.argmin([abs(np.median(xy[:, 1]) - (b[1] + b[3]) / 2) for b in boxes]))
        bx0, by0, bx1, by1 = boxes[k]
        uv = np.c_[(xy[:, 0] - bx0) / (bx1 - bx0), (by1 - xy[:, 1]) / (by1 - by0)]
        uv = uv[((uv >= -1e-6) & (uv <= 1 + 1e-6)).all(axis=1)]       # what the axes show
        panels[k].append((round(d["width"], 3), d.get("dashes") or "[] 0", uv))
    return list(zip(boxes, panels))


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ours_path = pathlib.Path(sys.argv[1])
    ours, native = read(sys.argv[1]), read(sys.argv[2])
    lines, ok = [], True
    for k, ((ob, oc), (nb, nc)) in enumerate(zip(ours, native)):
        ow, oh, nw, nh = ob[2] - ob[0], ob[3] - ob[1], nb[2] - nb[0], nb[3] - nb[1]
        same_box = abs(ow - nw) < 0.05 and abs(oh - nh) < 0.05
        ok &= same_box and len(oc) == len(nc)
        lines.append(f"panel {k + 1}: axes box ours {ow:.2f} x {oh:.2f} pt, native {nw:.2f} x {nh:.2f} pt; "
                     f"curves ours {len(oc)}, native {len(nc)}")
        for j, ((w, dash, uv), (nw_, ndash, nuv)) in enumerate(zip(oc, nc)):
            same = len(uv) == len(nuv)
            dev = float(np.max(np.abs((uv - nuv) * [nw, nh]))) if same else float("nan")
            ok &= w == nw_ and dash == ndash and same and dev < 0.05
            lines.append(f"  curve {j + 1}: width {w} / {nw_}, dashes {dash} / {ndash}, vertices inside the axes {len(uv)} / {len(nuv)}, "
                         f"largest vertex offset {dev:.4f} pt")
    lines.append("data-bearing attributes MATCH" if ok else "data-bearing attributes DIFFER")
    text = "\n".join(lines)
    print(text)
    (ours_path.parent / "fig2_check.txt").write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
