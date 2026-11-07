#!/usr/bin/env python3
"""Assemble the side-by-side gallery from what gallery_figures.py left in <figures-dir>. Run from anywhere:

    python3 -I experiments/FIGS/figstyle/build_gallery.py <figures-dir> docs/figstyle/gallery.pdf

One landscape A4 page per record of <figures-dir>/pages.json: the precedent on the left, ours as vector on the
right with its greyscale render as a small inset under the precedent (round 4), the caption, notes, the number of
render-check passes and the independent reviewer's report under both. Uses only fitz and the standard library, because
-I drops this directory from sys.path.
"""
import functools
import json
import math
import pathlib
import sys

import fitz

PRECEDENTS = pathlib.Path(__file__).resolve().parents[3] / "docs/figstyle/precedents"
STIX = pathlib.Path("/System/Library/Fonts/Supplemental")
REGULAR = {"fontname": "stix", "fontfile": str(STIX / "STIXGeneral.otf")} if (STIX / "STIXGeneral.otf").exists() else {"fontname": "tiro"}
BOLD = {"fontname": "stixb", "fontfile": str(STIX / "STIXGeneralBol.otf")} if (STIX / "STIXGeneralBol.otf").exists() else {"fontname": "tibo"}
LINE = 1.15
SIZES = (9, 8.5, 8, 7.5)
HEADING_SIZES = (12, 11, 10, 9)
RED = (0.7, 0, 0)

PW, PH = fitz.paper_size("a4-l")
MARGIN, GUTTER, GAP = 28, 16, 4
COL = (PW - 2 * MARGIN - GUTTER) / 2
TOP = MARGIN + 24       # below the heading
BOTTOM = PH - MARGIN
LABEL = 16              # height of the one-line label above our mock-up (a line needs 14.6 at 9 pt)
FIG_TOP = TOP + LABEL
FIG_FLOOR = 0.6 * (BOTTOM - FIG_TOP)    # our figure may shrink to this before the text does
FIG_MIN = 120           # and to this when the text does not fit even at the smallest size

_scratch_doc = fitz.open()
_scratch = _scratch_doc.new_page()


@functools.lru_cache
def load_font(fontname: str, fontfile: str = None) -> fitz.Font:
    return fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(fontname)


def text_height(text: str, width: float, size: float, font: dict = REGULAR) -> float:
    """Height the text takes, from a dry run: the shape is never committed. insert_textbox's fit test counts
    lines at size * LINE apart but draws them size * ascender * LINE apart (STIX: 1.055 times wider), so its
    own figure would let long paragraphs overrun; only the line count is taken from it."""
    tall = 5000
    needed = tall - _scratch.new_shape().insert_textbox(fitz.Rect(0, 0, width, tall), text, fontsize=size, lineheight=LINE, **font)
    f = load_font(**font)
    lines = round((needed + f.descender * size) / (size * LINE))
    return size * f.ascender * (1 + (lines - 1) * LINE) - size * f.descender + 1


def write(page, x: float, y: float, width: float, height: float, text: str, size: float, font: dict = REGULAR, color=(0, 0, 0)) -> bool:
    """Write text into the box, cutting it at a word when it does not fit. False when it was cut."""
    words, cut = text.split(" "), False
    while words and text_height(text, width, size, font) > height:
        words.pop()
        text, cut = " ".join(words) + " [...]", True
    if words:
        page.insert_textbox(fitz.Rect(x, y, x + width, y + height + 1), text, fontsize=size, lineheight=LINE, color=color, **font)
    return not cut


def image_rects(sizes: list, x: float, y: float, width: float, height: float) -> list:
    """Top-left aligned rects for images in the grid that shows them largest, keeping their aspect."""
    n, best, best_area = len(sizes), [], -1.0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = (width - (cols - 1) * GAP) / cols
        cell_h = (height - (rows - 1) * GAP) / rows
        rects, area = [], 0.0
        for i, (w, h) in enumerate(sizes):
            s = min(cell_w / w, cell_h / h)
            x0 = x + (i % cols) * (cell_w + GAP)
            y0 = y + (i // cols) * (cell_h + GAP)
            rects.append(fitz.Rect(x0, y0, x0 + s * w, y0 + s * h))
            area += s * s * w * h
        if area > best_area:
            best, best_area = rects, area
    return best


INSET = 0.42           # the greyscale inset's width, as a share of the column


def grey_inset(page, figdir: pathlib.Path, rec: dict, bottom: float) -> float:
    """Our figure in greyscale, small, at the bottom right of the left column; returns where it begins."""
    pdf = figdir / f"{rec['id']}.pdf"
    if not (rec.get("drawn") and pdf.exists()):
        return bottom
    with fitz.open(pdf) as src:
        r = src[0].rect
        w = INSET * COL
        top = max(TOP + text_height("Precedent: " + rec.get("precedent_label", ""), COL, SIZES[0]), FIG_TOP)
        h = min(w * r.height / r.width, 0.45 * (bottom - top) - LABEL)     # the precedent keeps the rest
        if h < 40:
            return bottom
        w = h * r.width / r.height
        pix = src[0].get_pixmap(colorspace=fitz.csGRAY, dpi=150)
    x0, y0 = MARGIN + COL - w, bottom - h
    write(page, x0, y0 - LABEL, w, LABEL, "Greyscale", SIZES[-1])
    page.insert_image(fitz.Rect(x0, y0, x0 + w, bottom), pixmap=pix, keep_proportion=True)
    return y0 - LABEL


def left_column(page, rec: dict, warn: list, bottom: float) -> None:
    paths = [PRECEDENTS / p for p in rec.get("precedent", [])]
    missing = [p for p in paths if not p.exists()]
    label = "Precedent: " + rec.get("precedent_label", "")
    if missing:
        names = ", ".join(str(p.relative_to(PRECEDENTS)) for p in missing)
        label += f" (missing: {names})"
        warn.append(f"missing precedent {names}")
    height = text_height(label, COL, SIZES[0])
    write(page, MARGIN, TOP, COL, height, label, SIZES[0])
    found = [p for p in paths if p.exists()]
    if found:
        sizes = [(pm.width, pm.height) for pm in map(fitz.Pixmap, map(str, found))]
        y = max(TOP + height, FIG_TOP)
        if bottom - y < 20:
            return
        for path, rect in zip(found, image_rects(sizes, MARGIN, y, COL, bottom - y)):
            page.insert_image(rect, filename=str(path), keep_proportion=True)


def right_column(page, rec: dict, figdir: pathlib.Path, warn: list) -> float:
    """Ours at the top of the right column; the caption and notes below both figures, across the page, so that a
    long note costs little figure. Returns where the figures end."""
    x = MARGIN + COL + GUTTER
    wide = 2 * COL + GUTTER
    write(page, x, TOP, COL, LABEL, "Ours", SIZES[0])
    pdf = figdir / f"{rec['id']}.pdf"
    src = fitz.open(pdf) if rec.get("drawn") and pdf.exists() else None

    paragraphs = [(rec["change"], RED)] if rec.get("change") else []      # round 5: what changed since approval
    if src is None:
        if rec.get("absent"):
            why = f"Absent from today's data: {rec['absent']}"
        elif rec.get("failed"):
            why = f"Figure failed: {rec['failed']}"
        else:
            why = f"No figure: {pdf.name} was not found."
        paragraphs.append((why, RED))
    caption = rec.get("caption", "")
    paragraphs.append((f"Caption ({len(caption.split())} words): {caption}", (0, 0, 0)))
    paragraphs.append((f"Differs from the precedent: {rec.get('differs', '')}", (0, 0, 0)))
    if rec.get("data_note"):
        paragraphs.append((f"Data: {rec['data_note']}", (0, 0, 0)))
    if rec.get("passes") or rec.get("review"):
        paragraphs.append((f"Checks: {rec.get('passes', 0)} independent review rounds; the last two full renders "
                           f"passed the measured checks (figcheck.py). Final review: {rec.get('review') or 'not run'}",
                           (0, 0, 0)))
    check = figdir / "fig2_check.txt"
    if rec["id"] == "SOPHON-FIG2" and check.exists():
        paragraphs.append((check.read_text().strip(), (0, 0, 0)))

    # The largest text that leaves our figure its width, or at least FIG_FLOOR of height.
    full_h = COL * src[0].rect.height / src[0].rect.width if src else 0
    for size in SIZES:
        heights = [text_height(t, wide, size) for t, _ in paragraphs]
        avail = BOTTOM - FIG_TOP - sum(h + GAP for h in heights)
        if avail >= min(full_h, FIG_FLOOR):
            break
    fig_h = min(full_h, max(avail, FIG_MIN))
    if src:
        fig_w = fig_h * src[0].rect.width / src[0].rect.height
        page.show_pdf_page(fitz.Rect(x, FIG_TOP, x + fig_w, FIG_TOP + fig_h), src, 0)
        src.close()

    y = FIG_TOP + fig_h
    for (text, color), height in zip(paragraphs, heights):
        y += GAP
        if not write(page, MARGIN, y, wide, min(height, BOTTOM - y + 1e-6), text, size, color=color):     # y is a sum of floats
            warn.append(f"text overflows at {size} pt and was cut")
            break
        y += height
    return FIG_TOP + fig_h


def missing_glyphs(rec: dict, font: fitz.Font) -> str:
    text = "".join(str(rec.get(k, "")) for k in ("id", "title", "precedent_label", "caption", "differs", "data_note", "absent", "failed",
                                                  "change"))
    return "".join(sorted({c for c in text if not c.isspace() and not font.has_glyph(ord(c))}))


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    figdir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    records = json.loads((figdir / "pages.json").read_text())
    font = load_font(**REGULAR)
    doc = fitz.open()
    warned = 0
    for rec in records:
        warn = []
        page = doc.new_page(width=PW, height=PH)
        heading = f"{rec['change'].split(':')[0] + ' ' if rec.get('change') else ''}{rec['id']}: {rec.get('title', '')}"
        size = next((s for s in HEADING_SIZES if text_height(heading, PW - 2 * MARGIN, s, BOLD) == text_height("x", PW - 2 * MARGIN, s, BOLD)), HEADING_SIZES[-1])
        if not write(page, MARGIN, MARGIN, PW - 2 * MARGIN, text_height("x", PW - 2 * MARGIN, size, BOLD), heading, size, BOLD):
            warn.append("heading overflows and was cut")
        left_column(page, rec, warn, grey_inset(page, figdir, rec, right_column(page, rec, figdir, warn)))
        if lost := missing_glyphs(rec, font):
            warn.append(f"characters the font does not have: {lost}")
        for w in warn:
            print(f"WARNING {rec['id']}: {w}")
        warned += bool(warn)
    doc.save(out, garbage=4, deflate=True)
    print(f"{out}: {len(records)} pages, {warned} with warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
