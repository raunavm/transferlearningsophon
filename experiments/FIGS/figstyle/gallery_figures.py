#!/usr/bin/env python3
"""Draw the gallery's mock-ups and write pages.json beside them. One broken figure does not hide the others.

    python3 experiments/FIGS/figstyle/gallery_figures.py --outdir figures/figstyle [--only ID ...]
"""
import argparse
import dataclasses
import json
import pathlib
import traceback

import gallery_common as C
import gallery_labelset
import gallery_palette
import gallery_physics
import gallery_review
import gallery_roc
import gallery_inventory
import style_v5 as S

MODULES = (gallery_physics, gallery_palette, gallery_labelset, gallery_roc, gallery_inventory)
ORDER = ("SOPHON-FIG2", "PALETTE",       # gallery only; then the paper's order: main text, appendix
         "F1", "F2a", "F2b", "F3", "F4", "FJ", "FJ-ctrl", "PT", "F6", "F8", "F9a", "F9b", "F10a", "F10b", "F10c", "F11",
         "VAL", "A10", "A8", "F6-app", "FJ-app", "FJ-ctrl-app", "FJ-s", "CONF", "F4-app", "F3-scatter", "F3-app",
         "F3-app2", "F3-app3", "F3-ft", "F3-ft2", "A3R", "A3R-ft", "UMAP-jc1", "UMAP-jc2-had", "UMAP-jc2-lep", "UMAP-top",
         "UMAP-qg", "A17-res", "A17-qcd", "A17-probe", "F10a-mass", "INVENTORY")


# the figures the PI approved page by page (2026-11-05), drawn into the paper by the render job (--paper);
# F11 waits for its samples, and the pages still coming back through the gallery are not here
PAPER = ("F1", "F2a", "F2b", "F3", "F3-scatter", "F3-app", "F3-ft", "FJ", "FJ-ctrl", "FJ-s", "FJ-app", "FJ-ctrl-app", "F4",
         "F4-app", "F6", "F6-app", "F8", "A8", "A10", "F9a", "F9b", "A17-res", "A17-qcd", "A17-probe", "F10a", "F10b",
         "A3R", "A3R-ft", "VAL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("figures/figstyle"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--paper", action="store_true", help="only the approved figures (PAPER)")
    a = ap.parse_args()
    if a.paper:
        a.only = list(PAPER)
    a.outdir.mkdir(parents=True, exist_ok=True)
    pages, failed = [], []
    for page in sorted((p for m in MODULES for p in m.PAGES), key=lambda p: ORDER.index(p.id)):
        if a.only and page.id not in a.only:
            continue
        S.use_style5(simplify=page.id != "SOPHON-FIG2")
        rec = {k: v for k, v in dataclasses.asdict(page).items() if k != "draw"}
        rec["passes"], rec["review"] = gallery_review.REVIEW.get(page.id, (rec["passes"], rec["review"]))
        rec["change"] = gallery_review.CHANGES.get(page.id, "")       # "NEW: ..." / "CHANGED: ...", shown on the page
        try:
            page.draw(a.outdir / page.id)
            rec["drawn"] = True
            print(f"{page.id}: drawn")
        except C.Absent as e:
            rec["drawn"], rec["absent"] = False, str(e)
            print(f"{page.id}: absent: {e}")
        except (Exception, SystemExit) as e:      # the selection code exits on a fatal input (xbx_report.load)
            traceback.print_exc()
            rec["drawn"], rec["failed"] = False, f"{type(e).__name__}: {e}"
            failed.append(page.id)
            print(f"{page.id}: FAILED {type(e).__name__}: {e}")
        pages.append(rec)
    (a.outdir / "pages.json").write_text(json.dumps(pages, indent=1))
    print(f"\ndrawn {sum(r['drawn'] for r in pages)} of {len(pages)}; failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
