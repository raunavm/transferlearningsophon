#!/usr/bin/env python3
"""Figure round 4's render-and-check step (docs/figstyle/ROUND4_BRIEF.md, Part 1): draw gallery figures, render each at
its printed width and 300 dpi, write greyscale, deuteranopia and protanopia copies, and measure the checklist items
that can be measured. Everything else is looked at, in the PNGs.

    python3 experiments/FIGS/figstyle/figcheck.py --outdir <dir> --only F2a F4 ...

Per figure <id> in <dir>: <id>.pdf (the figure as the paper includes it), <id>_print.png, <id>_grey.png,
<id>_deutan.png, <id>_protan.png, and <id>_check.json with the findings:
  panel      a main panel narrower than MIN_PANEL_CM (MIN_PANEL_CM_OF for a figure the brief allows smaller) or with
             height/width outside 0.7-1.3; ratio sub-panels (under half the height of the panel above them), colour
             bars, axes with no axis and image (heatmap) panels are not held to the aspect rule
  font       a text under 8 pt at print size
  overlap    two texts whose boxes intersect; a text over another panel; a legend over drawn data (line paths, markers,
             filled bands, scatter points, images)
  notation   plain-text "to ee", "mu", "bb", "->", "LOOK:", "Data today", "PI" in any text or the caption
  letters    main panels whose titles do not start (a), (b), ... in reading order
  caption    outside 40-80 words
The print size is the paper's: \\linewidth = 15 cm, except PRINT_CM. The saved PDF is cut to its tight box
(style_v3.save), so sizes are measured times print width / saved width.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import sys

import fitz
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.collections import Collection, PolyCollection  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import palette_check as PC  # noqa: E402

PRINT_CM = {"FJ-s": 9.0, "VAL": 10.5}                        # \paperfig widths other than \linewidth (15 cm)
MIN_PANEL_CM, MIN_PANEL_CM_OF = 6.5, {"F3-app": 5.0, "F3-ft": 5.0}
MIN_PT, ASPECT, PAD_IN, DPI = 8.0, (0.7, 1.3), 0.1, 300
BANNED = [(re.compile(p), why) for p, why in (
    (r"\bto\s+(ee|mu|mumu|bb|cc|qq|bs|bc|b\b)", "plain 'to' for an arrow"), (r"\bmu\b", "plain 'mu'"),
    (r"\b(bb|cc|qq|mumu)\b(?![\u0304\u2032])", "plain parton pair"), (r"->", "ASCII arrow"), (r"LOOK:", "internal note"),
    (r"Data today", "internal note"), (r"\bPI\b", "internal note"))]


def plain(text: str) -> str:
    """text with $...$ removed."""
    return re.sub(r"\$[^$]*\$", " ", text)


def notation(text: str) -> list:
    return [f"{why}: {text!r}" for rx, why in BANNED if rx.search(plain(text))]


def drawn_texts(fig, r) -> list:
    """[(text artist, owner axes or None, kind)] of every text drawn with something in it."""
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis, kind in (((ax.xaxis, "xtick"), (ax.yaxis, "ytick")) if ax.axison else ()):
            if not axis.get_visible():
                continue
            for t in axis._update_ticks():
                for lab, on in ((t.label1, t.label1.get_visible()), (t.label2, t.label2.get_visible())):
                    if on and lab.get_text().strip():
                        out.append((lab, ax, kind))
            for lab in (axis.label, axis.get_offset_text()):
                if lab.get_visible() and lab.get_text().strip():
                    out.append((lab, ax, "axislabel"))
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_visible() and t.get_text().strip():
                out.append((t, ax, "title"))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append((t, ax, "text"))
        if (leg := ax.get_legend()) is not None and leg.get_visible():
            out += [(t, ax, "legend") for t in leg.get_texts() if t.get_text().strip()]
            if leg.get_title().get_text().strip():
                out.append((leg.get_title(), ax, "legend"))
    for leg in fig.legends:
        out += [(t, None, "figlegend") for t in leg.get_texts() if t.get_text().strip()]
    for t in list(dict.fromkeys(fig.texts + ([fig._suptitle] if fig._suptitle is not None else []))):
        if t.get_visible() and t.get_text().strip():
            out.append((t, None, "figtext"))
    return out


def data_hit(ax, box: Bbox) -> str | None:
    """What drawn data of ax lie in box (display units), or None."""
    pad = box.expanded(1.0, 1.0)
    for ln in ax.lines:
        if not ln.get_visible() or ln.get_label().startswith("_child"):
            continue
        path = ln.get_transform().transform_path(ln.get_path())
        if ln.get_linestyle() not in ("None", "none", "", " ") and path.intersects_bbox(box, filled=False):
            return f"line {ln.get_label()!r}"
        if ln.get_marker() not in (None, "None", "none", "", " "):
            r = ln.get_markersize() * ax.figure.dpi / 72 / 2
            if box.expanded(1 + 2 * r / max(box.width, 1), 1 + 2 * r / max(box.height, 1)).count_contains(path.vertices):
                return f"markers {ln.get_label()!r}"
    for c in ax.collections:
        if not c.get_visible() or not isinstance(c, Collection):
            continue
        if isinstance(c, PolyCollection):
            for p in c.get_paths():
                if c.get_transform().transform_path(p).intersects_bbox(box, filled=True):
                    return f"band {c.get_label()!r}"
        elif len(c.get_offsets()):
            xy = c.get_offset_transform().transform(c.get_offsets())
            if pad.count_contains(xy):
                return f"points {c.get_label()!r}"
    for im in ax.images:
        if isinstance(im, AxesImage) and im.get_window_extent().overlaps(box):
            return "image"
    return None


def classify(fig, r) -> tuple[list, dict]:
    """(main panels, {axes: why exempt from the aspect rule})."""
    axes = [ax for ax in fig.axes if ax.get_visible()]
    boxes = {ax: ax.get_window_extent(r) for ax in axes}
    main, exempt = [], {}
    for ax in axes:
        b = boxes[ax]
        if hasattr(ax, "_colorbar") or ax.get_label() == "<colorbar>":
            continue
        if not ax.axison:
            continue
        above = [o for o in axes if o is not ax and boxes[o].x0 < b.x1 - 1 and boxes[o].x1 > b.x0 + 1
                 and boxes[o].y0 >= b.y1 - 2 and b.height < 0.55 * boxes[o].height]
        if above:
            continue                        # a ratio sub-panel
        main.append(ax)
        if ax.images or any(type(c).__name__ == "QuadMesh" for c in ax.collections):
            exempt[ax] = "heatmap panel"
    return main, exempt


def check(fig, name: str, print_cm: float, min_panel: float) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    saved_in = fig.get_figwidth() if getattr(fig, "exact_size", False) else tight.width + 2 * PAD_IN

    scale = print_cm / 2.54 / saved_in
    px_per_cm = fig.dpi / scale / 2.54             # display pixels per printed centimetre
    found = {k: [] for k in ("panel", "font", "overlap", "notation", "letters")}
    main, exempt = classify(fig, r)
    for ax in main:
        b = ax.get_window_extent(r)
        w, h = b.width / px_per_cm, b.height / px_per_cm
        where = ax.get_title() or f"axes at ({b.x0:.0f}, {b.y0:.0f})"
        if w < min_panel - 0.05:
            found["panel"].append(f"{where!r}: {w:.1f} cm wide (< {min_panel})")
        if ax not in exempt and not ASPECT[0] <= h / w <= ASPECT[1]:
            found["panel"].append(f"{where!r}: height/width {h / w:.2f} ({w:.1f} x {h:.1f} cm)")
    texts = drawn_texts(fig, r)
    found["page"] = []
    if getattr(fig, "exact_size", False):
        W, H = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        for t, ax, kind in texts:
            b = t.get_window_extent(r)
            if b.x0 < -0.5 or b.x1 > W + 0.5 or b.y0 < -0.5 or b.y1 > H + 0.5:
                found["page"].append(f"{kind} {t.get_text()[:40]!r} leaves the page")
        for leg in fig.legends:
            b = leg.get_window_extent(r)
            if b.x0 < -0.5 or b.x1 > W + 0.5 or b.y0 < -0.5 or b.y1 > H + 0.5:
                found["page"].append("the figure legend leaves the page")
    for t, ax, kind in texts:
        pt = t.get_fontsize() * scale
        if pt < MIN_PT - 0.05:
            found["font"].append(f"{kind} {t.get_text()[:40]!r}: {pt:.1f} pt")
        found["notation"] += notation(t.get_text())
    boxes = [(t.get_window_extent(r), t, ax, kind) for t, ax, kind in texts]
    for (b1, t1, a1, k1), (b2, t2, a2, k2) in itertools.combinations(boxes, 2):
        if k1 in ("legend", "figlegend") and k2 == k1 and a1 is a2:
            continue
        i = Bbox.intersection(b1, b2)
        if i is not None and i.width > 1 and i.height > 1:
            found["overlap"].append(f"text over text: {k1} {t1.get_text()[:30]!r} / {k2} {t2.get_text()[:30]!r}")
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    for b, t, ax, kind in boxes:
        if kind in ("xtick", "ytick", "axislabel", "title", "figtext", "figlegend"):
            for o in panels:
                if o is ax or (ax is not None and (o in ax.child_axes or hasattr(o, "_colorbar"))):
                    continue
                ob = o.get_window_extent(r)
                near = 0.1 * px_per_cm              # within 1 mm of another panel reads as touching it
                i = Bbox.intersection(b, ob.expanded((ob.width + 2 * near) / ob.width, (ob.height + 2 * near) / ob.height))
                if i is not None and i.width > 0 and i.height > 0:
                    found["overlap"].append(f"{kind} {t.get_text()[:30]!r} within 1 mm of the panel {o.get_title()[:30]!r}")
        if kind == "text" and ax is not None and ax.axison and not ax.images:
            if (hit := data_hit(ax, b)):
                found["overlap"].append(f"text {t.get_text()[:30]!r} over {hit} in {ax.get_title()[:30]!r}")
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None and leg.get_visible():
            if (hit := data_hit(ax, leg.get_window_extent(r))):
                found["overlap"].append(f"legend of {ax.get_title()[:30]!r} over {hit}")
    for leg in fig.legends:
        lb = leg.get_window_extent(r)
        for o in panels:
            i = Bbox.intersection(lb, o.get_window_extent(r))
            if i is not None and i.width > 2 and i.height > 2:
                found["overlap"].append(f"figure legend over the panel {o.get_title()[:30]!r}")
    if len(main) > 1:
        order = sorted(main, key=lambda a: (-round(a.get_window_extent(r).y1 / 20), a.get_window_extent(r).x0))
        first = re.match(r"\(([a-z])\)", order[0].get_title())
        start = ord(first.group(1)) if first else 97          # a figure that continues another starts at its letter
        for i, ax in enumerate(order):
            if not ax.get_title().startswith(f"({chr(start + i)})"):
                found["letters"].append(f"panel {i + 1} in reading order is titled {ax.get_title()[:30]!r}")
    found = {k: sorted(set(v)) for k, v in found.items()}
    return {"figure": name, "print_cm": print_cm, "scale": round(scale, 3), "n_main_panels": len(main),
            "findings": found, "n_findings": sum(map(len, found.values()))}


def rasters(pdf: pathlib.Path, print_cm: float) -> list:
    """<stem>_print.png at print width and DPI, and its greyscale, deuteranopia and protanopia copies."""
    doc = fitz.open(pdf)
    page = doc[0]
    zoom = print_cm / 2.54 * DPI / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3) / 255.0
    lin = PC.to_linear(rgb)
    grey = PC.to_srgb(lin @ np.array([0.2126, 0.7152, 0.0722]))
    out = []
    for suffix, img in (("print", rgb), ("grey", np.repeat(grey[..., None], 3, -1)),
                        ("deutan", PC.simulate(rgb, "deuteranopia")), ("protan", PC.simulate(rgb, "protanopia"))):
        p = pdf.with_name(f"{pdf.stem}_{suffix}.png")
        a = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
        fitz.Pixmap(fitz.csRGB, pix.width, pix.height, a.tobytes(), False).save(p)
        out.append((p, a))
    # one sheet of the four at half size: print | grey over deutan | protan
    half = [x[::2, ::2] for _, x in out]
    sheet = np.concatenate([np.concatenate(half[:2], 1), np.concatenate(half[2:], 1)], 0)
    q = pdf.with_name(f"{pdf.stem}_sheet.png")
    fitz.Pixmap(fitz.csRGB, sheet.shape[1], sheet.shape[0], np.ascontiguousarray(sheet).tobytes(), False).save(q)
    return [p for p, _ in out] + [q]


def caption_findings(page) -> list:
    n = len(page.caption.split())
    out = [] if 40 <= n <= 80 else [f"caption has {n} words (40-80)"]
    return out + [f"caption: {x}" for x in notation(page.caption)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=pathlib.Path, required=True)
    ap.add_argument("--only", nargs="+", required=True)
    a = ap.parse_args(argv)
    import gallery_figures as G
    import style_v3
    import style_v4
    import style_v5
    a.outdir.mkdir(parents=True, exist_ok=True)
    original = style_v3.save
    results = {}

    def save(fig, path):
        stem = pathlib.Path(path).stem
        page_id = current[0]
        rep = check(fig, stem, PRINT_CM.get(page_id, 15.0), MIN_PANEL_CM_OF.get(page_id, MIN_PANEL_CM))
        out = original(fig, path)
        rasters(pathlib.Path(path).with_suffix(".pdf"), rep["print_cm"])
        results[stem] = rep
        return out

    for mod in (style_v3, style_v4, style_v5):
        mod.save = save
    current = [None]
    pages = {p.id: p for m in G.MODULES for p in m.PAGES}
    for pid in a.only:
        page = pages[pid]
        current[0] = pid
        style_v5.use_style5(simplify=pid != "SOPHON-FIG2")
        try:
            page.draw(a.outdir / pid)
        except G.C.Absent as e:          # a placeholder page (F11): nothing drawn to check
            print(f"{pid}: absent ({e})")
            continue
        rep = results.get(pid, {"figure": pid, "findings": {}, "n_findings": 0})
        rep["findings"]["caption"] = caption_findings(page)
        rep["n_findings"] = sum(map(len, rep["findings"].values()))
        (a.outdir / f"{pid}_check.json").write_text(json.dumps(rep, indent=1))
        print(f"{pid}: {rep['n_findings']} measured findings")
        for k, v in rep["findings"].items():
            for x in v:
                print(f"  {k}: {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
