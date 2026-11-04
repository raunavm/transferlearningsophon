#!/usr/bin/env python3
"""Compare our redrawn Sophon Fig. 2 with the native PDF, by the method of docs/figstyle/sophon_tools/replot_fig2_check.py:

    python3 experiments/FIGS/figstyle/compare_fig2.py <figures-dir>/SOPHON-FIG2.pdf docs/figstyle/pdfs/2405.12972/src/roc_benchmark.pdf

Rasterises both at 4 px/pt, aligns them on the bottom-left corner, counts pixels of the overlap differing
by more than 32 of 255 in any channel, and writes the result to fig2_check.txt beside the first PDF.
"""
import pathlib
import sys

import fitz
import numpy as np


def raster(page) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n).astype(int)


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ours_path = pathlib.Path(sys.argv[1])
    ours, native = fitz.open(ours_path)[0], fitz.open(sys.argv[2])[0]
    a, b = raster(ours), raster(native)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    differ = np.abs(a[-h:, :w] - b[-h:, :w]).max(axis=2) > 32
    line = (f"Pixels differing from the native figure by more than 32/255 at 4 px/pt, aligned bottom-left: "
            f"{100 * differ.mean():.3f}% of {differ.size} "
            f"(page size: ours {ours.rect.width:.2f} x {ours.rect.height:.2f} pt, "
            f"native {native.rect.width:.2f} x {native.rect.height:.2f} pt)")
    print(line)
    (ours_path.parent / "fig2_check.txt").write_text(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
