#!/usr/bin/env python3
"""Render a precedent paper's pages, crop each captioned figure and measure it from the PDF itself.

    python3 -I experiments/FIGS/figstyle/extract_precedents.py <paper.pdf> <out-dir>
    python3 -I experiments/FIGS/figstyle/extract_precedents.py <paper.pdf> <out-dir> --crop FIG PAGE X0 Y0 X1 Y1 [--caption TEXT]

Writes pages/p<NN>.png (150 dpi), and per figure fig<N>.png, fig<N>.txt (its caption) and fig<N>.json:
width as a fraction of the text block, aspect ratio, the strokes (hex colour, width, dash pattern,
path count, rectilinear or curved), the fills, the fonts and sizes of the text inside the crop, the
embedded raster images, and the commonest non-grey colours of the crop's pixels (for raster figures).
The automatic crop takes the graphics above a caption in its column; --crop replaces it (points).
Run with -I: the input is a downloaded file.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

import fitz
import numpy as np
from PIL import Image

CAPTION = re.compile(r"^\s*(?:FIG(?:URE)?\.?|Figure|Fig\.)\s*(\d+)\s*[.:|—-]?", re.S)
DPI = 150


def hexc(c):
    return None if c is None else "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c[:3])


def text_block(page):
    """(x0, x1) of the body text: the span holding most body lines."""
    lines = [l["bbox"] for b in page.get_text("dict")["blocks"] if b["type"] == 0 for l in b["lines"]
             if l["bbox"][2] - l["bbox"][0] > 100]
    if not lines:
        return 0.0, page.rect.width
    return float(np.percentile([b[0] for b in lines], 5)), float(np.percentile([b[2] for b in lines], 95))


def measure(page, r, tb):
    strokes, fills = collections.Counter(), collections.Counter()
    for d in page.get_drawings():
        if not fitz.Rect(d["rect"]).intersects(r):
            continue
        curved = any(it[0] == "c" for it in d["items"]) or sum(it[0] == "l" for it in d["items"]) > 8
        if d.get("color") is not None and "s" in d.get("type", ""):
            strokes[(hexc(d["color"]), round(d.get("width") or 0, 3), str(d.get("dashes")), curved)] += 1
        if d.get("fill") is not None and "f" in d.get("type", ""):
            fills[hexc(d["fill"])] += 1
    fonts = collections.defaultdict(lambda: [0, ""])
    for b in page.get_text("dict", clip=r)["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                k = (s["font"], round(s["size"], 2), hexc(
                                                         tuple(((s["color"] >> i) & 255) / 255 for i in (16, 8, 0))))
                fonts[k][0] += len(s["text"])
                if len(fonts[k][1]) < 60:
                    fonts[k][1] += s["text"] + " "
    images = [{"bbox": [round(v, 1) for v in i["bbox"]], "w": i["width"], "h": i["height"]}
              for i in page.get_image_info() if fitz.Rect(i["bbox"]).intersects(r)]
    pix = page.get_pixmap(dpi=DPI, clip=r)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    q = (a // 8 * 8 + 4).reshape(-1, 3)
    q = q[(q.max(1).astype(int) - q.min(1).astype(int)) > 40]
    u, n = np.unique(q, axis=0, return_counts=True)
    px = [{"hex": "#%02x%02x%02x" % tuple(u[i]), "pixels": int(n[i])} for i in np.argsort(-n)[:12]]
    return {"bbox_pt": [round(v, 1) for v in r], "width_pt": round(r.width, 1), "height_pt": round(r.height, 1),
            "fraction_of_text_block": round(r.width / (tb[1] - tb[0]), 3), "aspect_w_over_h": round(r.width / r.height, 3),
            "text_block_pt": [round(v, 1) for v in tb], "page": page.number + 1,
            "strokes": [{"hex": k[0], "width_pt": k[1], "dashes": k[2], "curved": k[3], "paths": v}
                        for k, v in strokes.most_common(40)],
            "fills": dict(fills.most_common(25)),
            "fonts": [{"font": k[0], "size_pt": k[1], "hex": k[2], "chars": v[0], "sample": v[1][:60]}
                      for k, v in sorted(fonts.items(), key=lambda kv: -kv[1][0])][:25],
            "raster_images": images, "pixel_colours": px}


def save(page, r, name, caption, out, tb):
    page.get_pixmap(dpi=DPI, clip=r).save(out / f"{name}.png")
    (out / f"{name}.txt").write_text(caption + "\n")
    m = measure(page, r, tb)
    m["caption_words"] = len(caption.split())
    (out / f"{name}.json").write_text(json.dumps(m, indent=1))


def auto(doc, out):
    seen = []
    for page in doc:
        tb = text_block(page)
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        caps = [(b, CAPTION.match(b[4])) for b in blocks if CAPTION.match(b[4])]
        gfx = [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
        gfx += [fitz.Rect(c) for c in page.cluster_drawings()]
        gfx = [g for g in gfx if g.width > 20 and g.height > 20 and g.width < page.rect.width * 0.98]
        for b, m in caps:
            n = int(m.group(1))
            if n in seen:
                continue
            cap = fitz.Rect(b[:4])
            full = cap.width > (tb[1] - tb[0]) * 0.6
            col = (tb[0], tb[1]) if full else (cap.x0 - 8, cap.x1 + 8)
            above = [g for g in gfx if g.y1 <= cap.y0 + 6 and g.x1 > col[0] and g.x0 < col[1]]
            if not above:
                continue
            r = fitz.Rect(above[0])
            for g in above[1:]:
                r |= g
            r = fitz.Rect(min(r.x0, col[0]) - 2, r.y0 - 2, max(r.x1, col[1]) + 2, cap.y0)
            seen.append(n)
            save(page, r & page.rect, f"fig{n}", " ".join(b[4].split()), out, tb)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--crop", nargs=6)
    ap.add_argument("--caption", default="")
    a = ap.parse_args()
    doc = fitz.open(a.pdf)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.crop:
        fig, p, *box = a.crop
        page = doc[int(p) - 1]
        save(page, fitz.Rect(*map(float, box)), f"fig{fig}", a.caption, a.out, text_block(page))
        return 0
    (a.out / "pages").mkdir(exist_ok=True)
    for page in doc:
        page.get_pixmap(dpi=DPI).save(a.out / "pages" / f"p{page.number + 1:02d}.png")
    print(a.pdf.parent.name, "pages", doc.page_count, "figures", sorted(auto(doc, a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
