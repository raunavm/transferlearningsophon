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
import gallery_fj_options
import gallery_labelset
import gallery_physics
import gallery_roc
import style_v3 as S

MODULES = (gallery_physics, gallery_labelset, gallery_roc, gallery_fj_options)
ORDER = ("SOPHON-FIG2", "F1", "F2a", "F2b", "F2-app", "F3", "F3-app", "FJ", "FJ-1", "FJ-2", "FJ-3", "FJ-s", "FJ-app", "F4", "F4-app", "F6", "F7",
         "F8", "A10", "F9a", "F9b", "A17-res", "A17-qcd", "F10a", "F10b", "F11", "A3R")     # the paper's order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("figures/figstyle"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    pages, failed = [], []
    for page in sorted((p for m in MODULES for p in m.PAGES), key=lambda p: ORDER.index(p.id)):
        if a.only and page.id not in a.only:
            continue
        S.use_style(simplify=page.id != "SOPHON-FIG2")
        rec = {k: v for k, v in dataclasses.asdict(page).items() if k != "draw"}
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
