#!/usr/bin/env python3
"""Figure round 4: per gallery page, (independent review rounds, the final reviewer report). The reviewers saw only
the print-size PNG, its greyscale copy, the caption and the checklist (scratchpad r4/REVIEW_BRIEF*.md); each round's
findings were fixed and the figure re-rendered through figcheck.py before the next. gallery_figures.py puts these
on the pages. "Raised with the PI" marks a finding whose fix would change the figure's approved form."""

CLEAN = "no findings."
FIXED = "findings fixed after the last round and re-checked (layout, captions); "
ROC_NOTE = "raised with the PI: greyscale of the ten class colours."
SCATTER_NOTE = "raised with the PI: greyscale of the family colours, 1-AUC log axes."
BINNED_NOTE = ("raised with the PI: spread over runs; one jet selection for the pT and generator-mass panels (the pT "
               "panels hold every jet, the mass panels 20-500 GeV, which is why the soft-drop RSD falls far at high pT).")

REVIEW = {
    "PALETTE": (1, FIXED + "the separation pairs below 25 are the PI's palette, shown as such."),
    "F1": (3, CLEAN),
    "F2a": (3, FIXED + "raised with the PI: spread over runs."),
    "F2b": (3, FIXED + "raised with the PI: spread over runs."),
    "FJ": (3, CLEAN + " " + ROC_NOTE),
    "FJ-ctrl": (3, FIXED + ROC_NOTE),
    "PT": (2, FIXED + "raised with the PI: a 1-AUC log axis for (d), where the 17-class curve sets the range."),
    "F6": (3, FIXED.rstrip("; ") + "."),
    "F8": (3, FIXED.rstrip("; ") + "."),
    "F9a": (3, FIXED.rstrip("; ") + "."),
    "F9b": (3, FIXED + BINNED_NOTE),
    "F10a": (3, CLEAN + " Raised with the PI: spread over runs, log y on the upper panels."),
    "F10b": (3, FIXED + "raised with the PI: the scan has no point at exactly 1% and 10%; a statistical floor; "
                        "linear y."),
    "VAL": (3, CLEAN),
    "A10": (3, FIXED + "raised with the PI: spread over runs."),
    "A8": (3, FIXED + "raised with the PI: spread over runs."),
    "F6-app": (3, FIXED.rstrip("; ") + "."),
    "FJ-app": (3, CLEAN + " Raised with the PI: log x for the low-efficiency cluster; " + ROC_NOTE.split(": ")[1]),
    "FJ-ctrl-app": (3, FIXED + "raised with the PI: log x; greyscale of the ten class colours."),
    "FJ-s": (3, CLEAN),
    "CONF": (3, FIXED + "raised with the PI: blue means a high share in (a) and a loss in (b)-(d)."),
    "F3-app": (3, FIXED + SCATTER_NOTE),
    "F3-app2": (3, FIXED + SCATTER_NOTE),
    "F3-app3": (1, "new page, not yet reviewed."),
    "F3-ft": (3, FIXED + SCATTER_NOTE),
    "F3-ft2": (3, FIXED + SCATTER_NOTE),
    "A3R": (3, CLEAN),
    "A3R-ft": (3, CLEAN),
    "UMAP-jc1": (3, CLEAN + " Raised with the PI: per-class colours and greyscale."),
    "UMAP-jc2-had": (3, CLEAN + " Raised with the PI: per-class colours and greyscale."),
    "UMAP-jc2-lep": (3, CLEAN),
    "UMAP-top": (3, FIXED + "raised with the PI: greyscale of the two classes."),
    "UMAP-qg": (3, CLEAN + " Raised with the PI: greyscale of the two classes."),
    "A17-res": (3, FIXED + BINNED_NOTE),
    "A17-qcd": (3, FIXED + BINNED_NOTE),
    "A17-probe": (3, FIXED + BINNED_NOTE),
    "F10a-flav": (1, "new page, not yet reviewed."),
    "F10a-mass": (3, FIXED + "its ratio range differs slightly from F10a's (each set by its own points)."),
}

# Round 5 (PI, 2026-11-07): what changed on each page since the approved round 4, shown on the gallery page for page-by-page
# approval. Keyed by page id; a table page joins with its own id. "NEW: ..." for a page the PI has not seen.
NAMES = "short legend names (188 classes, no pretraining, ...) with a mode tag (LP), (MLP), (f.t.) where the figure mixes modes"
CAPTION = "caption opens with what was trained on what"
TICKS = "rejection ticks in plain numbers (no small exponents)"
CHANGES = {
    "INVENTORY": "CHANGED: the map of the story-based placement (PI, 2026-11-08): tier, number, draft sentence and "
                 "evidence of every item; the sculpting test; book numbers; printed at 100%.",
    "PALETTE": "CHANGED: F4-mech in place of the dropped F4 in the colour-vision renders.",
    "F1": f"CHANGED: {CAPTION}.",
    "F2a": f"CHANGED: bars of the spread over pretraining runs (SD over three, range over two); {NAMES}; {CAPTION}.",
    "F2b": f"CHANGED: bars of the spread over pretraining runs (SD over three, range over two); {NAMES}; {CAPTION}.",
    "F3-main": "NEW: 17 against 188 classes per class, linear probe (F3-scatter's b) beside fine-tuned (F3-ft's c) on one "
               "axis range, replacing the heatmap F3 and F3-scatter.",
    "F4-mech": "NEW: X→bb̄ vs cc̄ and X→bc vs bq, cs, the label sets above and the flavor-boundary partitions with 188 "
               "classes below, replacing F4 and F4-app.",
    "F4-mech-ft": "NEW: F4-mech's form after fine-tuning, on natural-mix test jets, on F4-mech's axes (absent until "
                  "the fine-tuned pairs job is read out).",
    "F4-other": "NEW: F4's other pairs, X→YY→bb̄qq vs cc̄qq and X→ee vs μμ, with QCD bb̄ vs cc̄, label sets (appendix).",
    "FJ": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-ctrl": f"CHANGED: {TICKS}; {CAPTION}.",
    "PT": f"CHANGED: {NAMES}; {CAPTION}.",
    "F6": f"CHANGED: uncertainty bands on the ratio panels only (SD over runs and 68% binomial, in quadrature); {TICKS}; "
          f"drawn at 18.5 cm height (8 pt text in the book); {NAMES}; {CAPTION}; Herwig: consistent within uncertainty.",
    "F8": f"CHANGED: {NAMES}; {CAPTION}, the bands said to be over pretraining runs.",
    "F9a": f"CHANGED: one hollow marker per mass-regression model; {NAMES}; {CAPTION}.",
    "F9b": f"CHANGED: soft-drop mass off the axis as arrows at the edge, values in the panel title; one hollow marker per "
           f"mass-regression model; {NAMES}; {CAPTION}.",
    "F10a": f"CHANGED: {CAPTION}.",
    "F10b": f"CHANGED: bands of the spread over pretraining runs (SD over three, range over two); the flavor-boundary "
            f"partitions moved to F10b-flav, leaving the label sets and the two mass models; random-selection floor "
            f"kept; moved to the appendix, main text if at 1% 17 - 188 > 0 and 17 + mass - 17 < 0, each beyond 3 sigma; "
            f"{NAMES}; {CAPTION}.",
    "F10b-flav": "NEW: F10b's form for the three flavor-boundary partitions with 188 classes for reference, moved out of "
                 "the main-text F10b (appendix).",
    "F10-ft": "NEW: F10b's form after fine-tuning on JetClass-II, on the same natural-mix test jets as F10b, with its "
              "floor (absent until the fine-tuned sculpting job is read out).",
    "F10c": "NEW: 1/JSD50 against R50 per model, DisCo Fig. 3's plane, bars of the spread over pretraining runs on both "
            "axes, from head/sculpting_v3's sig_eff_50; legend in three columns, clear of the panel.",
    "F11": f"CHANGED: {NAMES}; {CAPTION}.",
    "VAL": f"CHANGED: {NAMES}; {CAPTION}.",
    "A10": f"CHANGED: SD bands over pretraining runs, as F8; {NAMES}; {CAPTION}.",
    "A8": f"CHANGED: SD bands over pretraining runs, as F8; {NAMES}; {CAPTION}.",
    "F6-app": f"CHANGED: uncertainty bands on the ratio panels only; the kinematic-only probe curve in (a), dotted, in "
              f"the 188-class color; {TICKS}; drawn at 18.5 cm height (8 pt text in the book); {NAMES}; {CAPTION}.",
    "FJ-app": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-ctrl-app": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-s": f"CHANGED: {TICKS}; {CAPTION}.",
    "CONF": f"CHANGED: {CAPTION}.",
    "F3-app": f"CHANGED: panels in F3-ft's order (162, 43, 17 classes, random partitions); {CAPTION}.",
    "F3-app3": "NEW: the two mass-regression models, moved out of F3-app so probe and fine-tuning pages share one order.",
    "F3-app2": f"CHANGED: the three flavor-boundary partitions, (e)-(g) as in F3-ft2; {CAPTION}.",
    "F3-ft": f"CHANGED: {CAPTION}.",
    "F3-ft2": f"CHANGED: {CAPTION}.",
    "A3R": f"CHANGED: legend above the panels, off the curves; {TICKS}; caption run count checked (three) and {CAPTION}.",
    "A3R-ft": f"CHANGED: legend above the panels, off the curves; {TICKS}; caption gives the run count (three); {CAPTION}.",
    **{p: f"CHANGED: caption says the plots are qualitative; {CAPTION}." for p in
       ("UMAP-jc1", "UMAP-jc2-had", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg")},
    "A17-res": f"CHANGED: one hollow marker per mass-regression model in all three appendix mass figures; soft-drop arrows; "
               f"{NAMES}; {CAPTION}.",
    "A17-qcd": f"CHANGED: one hollow marker per mass-regression model in all three appendix mass figures; soft-drop arrows; "
               f"{NAMES}; {CAPTION}.",
    "A17-probe": f"CHANGED: one hollow marker per mass-regression model in all three appendix mass figures; soft-drop arrows; "
                 f"{NAMES}; {CAPTION}.",
    "F10a-mass": f"CHANGED: {CAPTION}.",
    "F10a-flav": "NEW: F10a's form for the three flavor-boundary partitions (M4).",
}
