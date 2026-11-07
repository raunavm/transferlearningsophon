#!/usr/bin/env python3
"""What the gallery's figure modules share: the data the paper's figures read, through the paper's own
selection code (paper_figures.inputs, mlst_results), so a mock-up shows the numbers the paper has. Each
module (gallery_labelset, gallery_roc, gallery_physics) has PAGES, a list of Page; gallery_figures.py
draws them and writes pages.json for build_gallery.py."""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Callable

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import paper_figures as P  # noqa: E402
import style_v5 as S  # noqa: E402

M = P.F._mlst()
RUNS = M._paper(P.REPO)
I = P.inputs(M)
# jc1_ft_ovr: the long-budget JetClass cells once they hold every model (mlst_v4.load_family)
ROC = {**I["roc"], **{fam: docs for fam in ("jc1_ft_ovr", "jc1_probe_ovr", "top_probe", "qg_probe", "jc2_ft_ovr", "pairs_sib2")
                      if (docs := M.V4.load_family(P.DATA, fam))}}
# the long-budget JetClass-II cells at 1e6 replace the fixed-budget ones of the same model (one recipe per model)
if (_long := M._load(P.DATA / M.V4.ROC_DIR / "jc2_ft_ovr_long")):
    ROC["jc2_ft_ovr"] = {**ROC.get("jc2_ft_ovr", {}), **_long}


class Absent(Exception):
    """Today's data do not hold this figure at all: the gallery page says so."""


@dataclasses.dataclass
class Page:
    id: str                 # as in docs/figstyle/PLAN.md; also the file stem
    title: str              # the figure's name for the gallery page
    precedent: list         # paths under docs/figstyle/precedents/, e.g. "1902.09914/fig5.png"
    precedent_label: str    # e.g. "Landscape Fig. 5"
    caption: str            # final caption: 40 to 80 words, what is plotted only
    differs: str            # every attribute on which ours differs from the precedent, and why
    draw: Callable          # draw(path: pathlib.Path) -> None, saving through style_v4.save
    data_note: str = ""     # what is incomplete in today's data, if anything
    passes: int = 0         # round 4: independent review rounds (gallery_review.py)
    review: str = ""        # round 4: the independent reviewer's final report


def mean_over_runs(docs: dict, models: list, key: str):
    """(eps_S, mean rejection over the runs, drawn where every run has at least 20 background jets, mean AUC);
    the curves_v2 files' own cut, via mlst_v4.roc_series. None when no model holds the curve."""
    series = [s for m in models if m in docs and (s := M.V4.roc_series(docs[m], key)) is not None]
    if not series:
        return None
    eps = np.asarray(series[0][0], float)
    v = np.array([np.asarray(s[1], float) for s in series])
    ok = np.isfinite(v).all(axis=0)
    auc = float(np.mean([docs[m]["curves"][key]["auc"] for m in models if m in docs]))
    return eps[ok], v[:, ok].mean(axis=0), auc
