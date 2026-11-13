#!/usr/bin/env python3
"""Figure round 4: per gallery page, (independent review rounds, the final reviewer report). The reviewers saw only
the print-size PNG, its greyscale copy, the caption and the checklist (scratchpad r4/REVIEW_BRIEF*.md); each round's
findings were fixed and the figure re-rendered through figcheck.py before the next. gallery_figures.py puts these
on the pages. "Raised with the PI" marks a finding whose fix would change the figure's approved form."""

CLEAN = "no findings."
FIXED = "findings fixed after the last round and re-checked (layout, captions); "
ROC_NOTE = "raised with the PI: greyscale of the ten class colours."
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
    "FJ-app": (3, CLEAN + " Raised with the PI: log x for the low-efficiency cluster; " + ROC_NOTE.split(": ")[1]),
    "FJ-ctrl-app": (3, FIXED + "raised with the PI: log x; greyscale of the ten class colours."),
    "FJ-s": (3, CLEAN),
    "F8-auc": (1, "new page, not yet reviewed."),
    "A8-auc": (1, "new page, not yet reviewed."),
    "A10-auc": (1, "new page, not yet reviewed."),
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
    "F2": "NEW: the one main-text figure of the headline family (PI 2026-11-13, L1): linear probes over fine-tuned "
          "networks, five tasks with Herwig, headline metric only; the mass networks under their label sets, the "
          "partitions under a subheading, the random network and from scratch (L5).",
    "F8-auc": "NEW: F8 in AUC (PI 2026-11-11): the AUC averaged over the classes on JetClass-II and JetClass, the AUC of "
              "top and quark/gluon tagging, with ParT's published AUCs.",
    "A8-auc": "NEW: A8 in AUC, as F8-auc; five tasks with Herwig and the mass networks.",
    "A10-auc": "NEW: A10 in AUC, as F8-auc; five tasks with Herwig, the mass networks and the random network.",
    "INVENTORY": "CHANGED: the map of the story-based placement (PI, 2026-11-08): tier, number, draft sentence and "
                 "evidence of every item; the sculpting test; book numbers; printed at 100%.",
    "PALETTE": "CHANGED: F4-mech in place of the dropped F4 in the colour-vision renders.",
    "F1": f"CHANGED: {CAPTION}.",
    "F2a": "CHANGED: the appendix full version of F2's linear probes (L1): five tasks with Herwig, headline metric over "
           "AUC, the rows of L5 with the random network and the released Sophon checkpoint.",
    "F2b": "CHANGED: the appendix full version of F2's fine-tuned networks (L1): five tasks with Herwig, headline metric "
           "over AUC, the rows of L5 with from scratch.",
    "PC-jc2": "NEW: the per-class family's main-text grid (PI 2026-11-12): linear probe over fine-tuned, 162, 43 and 17 "
              "classes each against 188, one axis range; replaces F3-main, F3-app and F3-ft (their label-set panels).",
    "PC-jc1": "NEW: the same grid on JetClass's ten classes, one colour and shape per class (appendix).",
    "PC-jc2-part": "NEW: the family's form for the four 17-class partitions against 188 classes, both modes; replaces "
                   "F3-app2, F3-ft2 and the random-partition panels of F3-app and F3-ft.",
    "PC-jc1-part": "NEW: the same for JetClass.",
    "PC-jc2-mass": "NEW: each mass network against its own label set (not against 188), both modes; replaces F3-app3, "
                   "which had no fine-tuned row.",
    "PC-jc1-mass": "NEW: the same for JetClass.",
    "PC-conf": "CHANGED: CONF as the family's confusion slice; the difference scale shared with PC-conf-part and "
               "PC-conf-mass.",
    "PC-conf-part": "NEW: CONF's differences for the four 17-class partitions against 188 classes.",
    "PC-conf-mass": "NEW: CONF's differences for each mass network against its own label set.",
    "F4-mech": "CHANGED: the class-pair family's form (PI 2026-11-13): the mass networks in the upper row, 17 classes "
               "beside 188 in the lower, a ratio panel to 188 classes with each curve's band under every panel, the rows "
               "named on the left.",
    "F4-mech-ft": "CHANGED: F4-mech's new form after fine-tuning, with from scratch in the upper row.",
    "F4-other": "CHANGED: the four other flavor pairs in F4-mech's form (X→ee vs μμ to F4-ee; QCD with b and with c "
                "against light added; the partitions' row, the mass networks and the ratio panels added).",
    "F4-other-ft": "NEW: F4-other after fine-tuning, the missing counterpart (appendix).",
    "F4-ee": "NEW: X→ee vs μμ as its own supplement page (PI: out of Fig. A1): the probes that keep a curve.",
    "F4-rand": "NEW: the random partitions pooled on the six pairs, linear probes, with 188 and 17 classes.",
    "F4-rand-ft": "NEW: F4-rand after fine-tuning.",
    "FJ": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-ctrl": f"CHANGED: {TICKS}; {CAPTION}.",
    "PT": f"CHANGED: {NAMES}; {CAPTION}.",
    "F6": "CHANGED: the ROC family's two-class form (PI 2026-11-13): linear probes (old F6-app) over fine-tuned networks, "
          "top, quark/gluon Pythia and Herwig, a ratio panel to 188 classes under each; the mass networks added; the "
          "kinematic-inputs run out.",
    "F6-part": "NEW: F6's form for the four 17-class partitions with 188 and 17 classes.",
    "F8": "CHANGED: a named slice of the training-size family in the supplement (L9): the five panels with Herwig; the "
          "probe dashed in the 188 colour; caption's top-tagging claim corrected.",
    "F9a": f"CHANGED: one hollow marker per mass-regression model; {NAMES}; {CAPTION}.",
    "F9b": f"CHANGED: soft-drop mass off the axis as arrows at the edge, values in the panel title; one hollow marker per "
           f"mass-regression model; {NAMES}; {CAPTION}.",
    "F10a": f"CHANGED: {CAPTION}.",
    "F10b": f"CHANGED: bands of the spread over pretraining runs (SD over three, range over two); the flavor-boundary "
            f"partitions moved to F10b-flav, leaving the label sets and the two mass models; random-selection floor "
            f"kept; in the appendix, as the sculpting test on the map page decides; "
            f"{NAMES}; {CAPTION}.",
    "F10b-flav": "CHANGED: 17 classes added for reference beside 188 (PI 2026-11-13); the random partitions wait for "
                 "their head outputs.",
    "F10-ft": "NEW: F10b's form after fine-tuning on JetClass-II, on the same natural-mix test jets as F10b, with its "
              "floor (absent until the fine-tuned sculpting job is read out).",
    "F10c": "NEW: 1/JSD50 against R50 per model, DisCo Fig. 3's plane, bars of the spread over pretraining runs on both "
            "axes, from head/sculpting_v3's sig_eff_50; legend in three columns, clear of the panel.",
    "F11": "CHANGED: the resonance family's main slice (PI 2026-11-13, L11): the label sets with the mass networks in "
           "their lighter shades, the direct discriminants of 188, 162 and 162 + mass dash-dot on X→bb̄, from scratch "
           "grey solid with three seeds on both signals.",
    "F11-part": "NEW: F11's form for the 17-class partitions with 188 and 17 classes (appendix).",
    "F11-probe": "NEW: F11's form for the linear probe, with the random network (supplement).",
    "A17-ridge": "NEW: A17-probe's form for the ridge regression on frozen features (supplement).",
    "F10-ft-part": "NEW: F10-ft's form for the 17-class partitions with 188 and 17 classes (supplement).",
    "VAL-part": "NEW: VAL's form for the 17-class partitions with 17 classes for reference (supplement).",
    "VAL": f"CHANGED: the mass networks in their lighter shades and SD bands (PI 2026-11-13); {NAMES}; {CAPTION}.",
    "A10": "CHANGED: the training-size family's form (PI 2026-11-13): five tasks with Herwig, the mass networks, the "
           "random network in place of from scratch.",
    "A8": "CHANGED: the training-size family's form (PI 2026-11-13): five tasks with Herwig, the mass networks; now the "
          "main-text slice.",
    "A8-part": "NEW: A8's form for the four 17-class partitions with 188 and 17 classes.",
    "A10-part": "NEW: A10's form for the four 17-class partitions with 188 and 17 classes.",
    "FJ-app": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-ctrl-app": f"CHANGED: {TICKS}; {CAPTION}.",
    "FJ-s": f"CHANGED: {TICKS}; {CAPTION}.",
    "A3R": f"CHANGED: legend above the panels, off the curves; {TICKS}; caption run count checked (three) and {CAPTION}.",
    "A3R-ft": f"CHANGED: legend above the panels, off the curves; {TICKS}; caption gives the run count (three); {CAPTION}.",
    **{p: f"CHANGED: caption says the plots are qualitative; {CAPTION}." for p in
       ("UMAP-jc1", "UMAP-jc2-had", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg")},
    "A17-res": f"CHANGED: one hollow marker per mass-regression model in all three appendix mass figures; soft-drop arrows; "
               f"{NAMES}; {CAPTION}.",
    "A17-qcd": f"CHANGED: one hollow marker per mass-regression model in all three appendix mass figures; soft-drop arrows; "
               f"{NAMES}; {CAPTION}.",
    "A17-probe": "CHANGED: the MLP probe on every label set, the mass networks and the random network, filled, the "
                 "mass outputs left to F9b (PI 2026-11-13).",
    "F10a-mass": f"CHANGED: {CAPTION}.",
    "F10a-flav": "NEW: F10a's form for the three flavor-boundary partitions (M4).",
}
