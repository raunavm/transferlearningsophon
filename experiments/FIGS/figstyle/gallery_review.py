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
    "F3": (3, FIXED + "raised with the PI: the unused positive half of the colour scale and greyscale of the "
                      "diverging map."),
    "F4": (3, FIXED.rstrip("; ") + "."),
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
    "F4-app": (3, FIXED.rstrip("; ") + "."),
    "F3-scatter": (3, FIXED + SCATTER_NOTE),
    "F3-app": (3, FIXED + SCATTER_NOTE),
    "F3-app2": (3, FIXED + SCATTER_NOTE),
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
    "F10a-mass": (3, FIXED + "its ratio range differs slightly from F10a's (each set by its own points)."),
    "INVENTORY": (1, FIXED.rstrip("; ") + "."),
}
