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
import gallery_perclass
import gallery_physics
import gallery_placement
import gallery_review
import gallery_roc
import gallery_inventory
import gallery_compare
import style_v5 as S

MODULES = (gallery_physics, gallery_palette, gallery_labelset, gallery_perclass, gallery_roc, gallery_inventory,
           gallery_compare)
ORDER = ("INVENTORY",            # the one-page map first; then the paper's order (gallery_placement): main text,
         "F1", "F2", "F2-ft", "PC-jc2", "F4-mech", "A8", "F11",
         # appendix
         "F2a", "F2b", "PC-jc1", "F4-mech-ft", "F4-other", "F4-light", "F4-other-ft", "F4-light-ft", "A10", "F6", "F6-ft",
         "PT", "F9b", "F10b", "F11-part",
         "UMAP-jc2-had", "VAL",
         # supplement
         "PC-jc2-part", "PC-jc2-rand", "PC-jc1-part", "PC-jc1-rand", "PC-jc2-mass", "PC-jc1-mass", "PC-conf",
         "PC-conf-part", "PC-conf-mass", "F4-ee", "F4-rand", "F4-rand-qcd", "F4-rand-ft", "F4-rand-qcd-ft", "F8", "F8-auc",
         "A8-auc", "A10-auc", "A8-part", "A10-part", "F6-part", "F6-part-ft",
         "FJ", "FJ-ctrl", "FJ-app", "FJ-ctrl-app", "FJ-mass", "FJ-mass-app", "FJ-s", "A3R", "A3R-ft", "A3R-ctrl-ft", "A3R-s", "PT-part", "PT-bbcc", "PT-bbcc-part", "F9a", "A17-res", "A17-qcd", "A17-probe",
         "A17-ridge", "F10a", "F10b-flav", "F10c", "F10a-flav", "F10-ft", "F10-ft-part", "F10a-mass", "F11-probe",
         "UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg", "UMAP-xbx", "UMAP-mass-jc2-had", "UMAP-mass-jc1",
         "UMAP-mass-jc2-lep", "UMAP-mass-top", "UMAP-mass-qg", "UMAP-mass-xbx", "VAL-part",
         "SOPHON-FIG2", "PALETTE",      # gallery only
         # the PI's line-graph style comparison (2026-11-11), gallery only
         "CMP-F7-B", "CMP-F7-C", "CMP-A4-B", "CMP-A4-C", "CMP-FONT-T", "CMP-FONT-LM")


# the figures the PI approved page by page (2026-11-05), drawn into the paper by the render job (--paper);
# the pages still coming back through the gallery are not here (the resonance, mass, sculpting and pretraining-curve
# families are, PI 2026-11-13, and the jet-pT family); F3, F3-scatter, F4 and
# F4-app were dropped (gallery_placement.DROPPED, PI 2026-11-08); the per-class family replaced F3-main, F3-app,
# F3-app2, F3-app3, F3-ft, F3-ft2 and CONF, and F6 took in F6-app (gallery_placement.SUPERSEDED, PI 2026-11-13)
PAPER = ("F1", "F2", "F2-ft", "F2a", "F2b", "PC-jc2", "PC-jc1", "PC-jc2-part", "PC-jc2-rand", "PC-jc1-part", "PC-jc1-rand", "PC-jc2-mass", "PC-jc1-mass",
         "PC-conf", "PC-conf-part", "PC-conf-mass", "F4-mech", "F4-mech-ft", "F4-other", "F4-other-ft", "F4-light", "F4-light-ft", "F4-ee", "F4-rand", "F4-rand-qcd",
         "F4-rand-ft", "F4-rand-qcd-ft", "FJ", "FJ-ctrl", "FJ-s", "FJ-app", "FJ-ctrl-app", "FJ-mass", "FJ-mass-app", "F6", "F6-ft", "F6-part", "F6-part-ft", "F8",
         "F8-auc", "A8", "A8-auc", "A8-part", "A10", "A10-auc", "A10-part", "F9a", "F9b", "A17-res", "A17-qcd", "A17-probe", "A17-ridge", "F10a", "F10b", "F10b-flav", "F10c", "F10a-flav",
         "F10-ft", "F10-ft-part", "F10a-mass", "F11", "F11-part", "F11-probe", "A3R", "A3R-ft", "A3R-ctrl-ft", "A3R-s", "PT", "PT-part", "PT-bbcc", "PT-bbcc-part",
         "VAL", "VAL-part")      # F10b-flav, F10c, F10a-flav: the appendix's sculpting subsection (PI 2026-11-09)


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
        if page.id in gallery_placement.TIER:                         # the paper's placement and the sentence it backs
            rec["place"], rec["backs"] = gallery_placement.place(page.id), gallery_placement.BACKS[page.id]
            rec["evidence"] = ", ".join(f"{gallery_placement.number(k)} ({k})" for k in gallery_placement.evidence(page.id))
            rec["title"] = rec["title"].removesuffix(" (appendix)")    # the place names the tier
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
