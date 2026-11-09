#!/usr/bin/env python3
"""Where every figure and table of the paper goes (PI, 2026-11-08, the story-based cut; plan "Placement"): main text,
appendix or supplement, and the one sentence each backs. A main-text item backs a sentence of the abstract or the
conclusions, an appendix item a sentence of the main text; a supplement item is listed with the sentence that cites it.
The single source of the placement: gallery_figures.py orders the gallery by it, gallery_inventory.py draws it as the
one-page map and make_results_book.py builds its lists from it. Free of data loading, so tests and the book import it.

Every sentence is a DRAFT for the PI's approval before the writing pass, worded to what today's numbers support. A
sentence names another item as [[key]], filled in by number() ("Fig. 4", "Table A1"), so no number is written by hand.
evidence(key): the items the sentence rests on, the item itself first."""
from __future__ import annotations

import re

MAIN, APPENDIX, SUPPLEMENT = "main", "appendix", "supplement"
TIERS = (MAIN, APPENDIX, SUPPLEMENT)
TIER_NAME = {MAIN: "Main text", APPENDIX: "Appendix", SUPPLEMENT: "Supplement"}
SAMPLES = "waits for the samples"
FT_READOUT = "waits for the fine-tuned read-out"
# F10b's promotion to the main text (plan "Sculpting condition", PI 2026-11-08), sign included: at 1% QCD efficiency
# the 17-class head sculpts more than the 188-class one (17 - 188 > 0) and the mass regression takes that back
# (17 + mass - 17 < 0), each by more than 3 sigma; (sculpting_boot.DIFFS name, required sign)
F10B_RULE = (("17-188", +1), ("17mass-17", -1))
F10B_WP = "eps_b=0.01"
# the sculpting differences beyond 3 sigma that F10b's sentence names (PI 2026-11-09: those and nothing else), as
# (sculpting_boot.DIFFS name, working point); tests/test_gallery_placement.py checks them against the bootstrap
F10B_STATED = (("17-188", "eps_b=0.01"), ("43-188", "eps_b=0.1"), ("162mass-162", "eps_b=0.1"))
# a placement that waits on a result; the two tables only while they hold pending cells (make_results_book sets a
# PROVISIONAL table with its pending cells, and a box for a PROVISIONAL figure not drawn yet)
PROVISIONAL = {"F11": SAMPLES, "F4-mech-ft": FT_READOUT, "F10-ft": FT_READOUT, "tab:xbx": SAMPLES, "tab:xbxrej": SAMPLES,
               "tab:pairs": "its fine-tuned columns wait for the fine-tuned read-out"}
SCULPT_FLAV = ("No flavor-boundary partition sculpts the QCD mass differently from 188 classes by more than three times "
               "the uncertainty.")
FJ_S = "Cited beside [[FJ]]: partitions, linear probes, from scratch."
SCATTER_S = "Cited from [[F3-main]]: the other scatter panels."
XBX = "X→bb̄ and X→bs transfer: written when the samples arrive."

# (gallery id, tier, draft sentence), in paper order within each tier; items that share a sentence share a row on the map
_FIGURES = (
    ("F1", MAIN, "We pretrain one network on the same jets, in the same order, and change only the label set."),
    ("F2a", MAIN, "On JetClass-II and JetClass, frozen features lose performance as the label set coarsens, most at 17 "
                  "classes; at a fixed class count, which classes merge matters."),
    ("F2b", MAIN, "After fine-tuning the coarsest labels still cost accuracy on JetClass-II and JetClass; on top and "
                  "quark/gluon tagging the label sets differ by about the run spread or less."),
    ("F3-main", MAIN, "The 17-class loss falls unevenly over the classes; fine-tuning shrinks it but does not remove it."),
    ("F4-mech", MAIN, "Linear probes separate quark flavors worse with 17 classes than with 188, and a flavor boundary "
                      "in a 17-class partition recovers part of the loss."),
    ("F8", MAIN, "Fine-tuning the 188-class network beats training from scratch on JetClass-II, JetClass and top "
                 "tagging, but not on the smallest quark/gluon sets; a linear probe keeps up only on JetClass-II."),
    ("A8", MAIN, "After fine-tuning, 17 classes stay lowest on JetClass-II and JetClass at every training-set size; on "
                 "top and quark/gluon tagging the label sets show no consistent order."),
    ("F9b", MAIN, "An auxiliary mass regression gives a better jet-mass resolution than the soft-drop mass, shown in "
                  "bins of jet pT and generator mass."),
    ("F11", MAIN, XBX),
    ("FJ", APPENDIX, "[[tab:jetclass]]'s rejections come from these per-class ROC curves."),
    ("F4-other", APPENDIX, "On the other flavor pairs the 17-class probes separate least, 43 classes as little on X→ee "
                           "vs μμ; 162 classes, merging QCD, lose on QCD bb̄ vs cc̄."),
    ("F4-mech-ft", APPENDIX, "The fine-tuned pairs are quoted from [[tab:pairs]]'s fine-tuned columns."),
    ("PT", APPENDIX, "In every jet-pT bin the JetClass-II probes keep the label-set order, and 17 classes stay lowest on "
                     "JetClass after fine-tuning and on X→bb̄ vs cc̄."),
    ("F6", APPENDIX, "Fine-tuned on top and quark/gluon tagging, Pythia and Herwig, the label sets differ by about the "
                     "run spread or less."),
    ("F6-app", APPENDIX, "On top and quark/gluon tagging every linear probe stays below fine-tuning."),
    ("A10", APPENDIX, "The JetClass-II probes keep the label-set order at every training-set size; on JetClass 17 "
                      "classes stay lowest."),
    ("F9a", APPENDIX, "The mass output's response is narrower than the soft-drop mass's."),
    ("F10a", APPENDIX, "Cuts on the pretraining head reshape the QCD mass at 10% and 1% efficiency."),
    # PI 2026-11-09: the main text reports exactly the differences beyond 3 sigma (tab:sculptboot), nothing else
    ("F10b", APPENDIX, "17 classes sculpt the QCD mass more than 188 at 1% QCD efficiency, 43 at 10%; mass regression "
                       "reduces it for 162 classes at 10%; every other difference (17 + mass, flavor boundaries) is "
                       "within the uncertainty."),
    ("F10b-flav", APPENDIX, SCULPT_FLAV),
    ("F10c", APPENDIX, "At 50% resonant efficiency no label-set difference in QCD mass sculpting exceeds three times "
                       "its uncertainty."),
    ("F10a-flav", APPENDIX, SCULPT_FLAV),
    ("F10-ft", APPENDIX, "After fine-tuning on JetClass-II the same comparison on natural-mix test jets; the sentence "
                         "waits for the read-out."),
    ("UMAP-jc2-had", APPENDIX, "Qualitatively, 188-class frozen features separate X→bb̄, cc̄ and qq̄ and 17-class ones "
                               "only order them; after fine-tuning both separate them."),
    ("VAL", APPENDIX, "Validation accuracy levels off for every label set during pretraining."),
    *((k, SUPPLEMENT, FJ_S) for k in ("FJ-ctrl", "FJ-app", "FJ-ctrl-app", "FJ-s")),
    ("CONF", SUPPLEMENT, "Cited beside [[tab:jetclass]]: JetClass confusion matrices."),
    *((k, SUPPLEMENT, SCATTER_S) for k in ("F3-app", "F3-app2", "F3-app3", "F3-ft", "F3-ft2")),
    *((k, SUPPLEMENT, "Cited from [[F3-main]]: every class's ROC curve.") for k in ("A3R", "A3R-ft")),
    *((k, SUPPLEMENT, "Cited from [[UMAP-jc2-had]]: the other tasks.")
      for k in ("UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg")),
    *((k, SUPPLEMENT, "Cited from [[F9b]]: by jet group; from frozen features.") for k in ("A17-res", "A17-qcd", "A17-probe")),
    ("F10a-mass", SUPPLEMENT, "Cited beside [[F10a]]: the mass-regression models."),
)
# (table label, tier, draft sentence); tab:benchtop and tab:benchqg become one table if it fits at the current type size
_TABLES = (
    ("tab:jetclass", MAIN, "Fine-tuned, the coarsest labels still lower the rejection for several JetClass classes."),
    ("tab:benchtop + tab:benchqg", MAIN, "On top and quark/gluon tagging the label sets differ by about the run "
                                         "spread or less."),
    ("tab:mass", MAIN, "An auxiliary mass regression puts the jet mass into the frozen features."),
    ("tab:xbx", MAIN, XBX),
    ("tab:xbxrej", MAIN, XBX),
    ("tab:probes", APPENDIX, "Every probe behind [[F2a]], with the released Sophon checkpoint."),
    ("tab:mlp", APPENDIX, "Nonlinear probes repeat the linear-probe comparisons."),
    ("tab:coarse", APPENDIX, "Scored on the 43 groups, 17-class features fall behind; on the 17 groups the label sets agree."),
    ("tab:qcdhf", APPENDIX, "Merging the 27 QCD classes (162) does not lower the X→bb̄ discriminant's rejection of "
                            "b-flavored QCD."),
    ("tab:fill", APPENDIX, "Filling the missing inputs with zeros, the training mean or random draws changes the "
                           "linear probes."),
    ("tab:inputs", APPENDIX, "Top and quark/gluon data lack some inputs, which are set to zero."),
    ("tab:bn", APPENDIX, "BatchNorm statistics recomputed, against the checkpoints as trained."),
    ("tab:partrepro", APPENDIX, "Our from-scratch training against ParT's published top-tagging result."),
    ("tab:lrsweep", APPENDIX, "The fine-tuning learning rate was chosen on validation."),
    ("tab:ft", APPENDIX, "Every label set fine-tuned at every training-set size."),
    ("tab:randprong", APPENDIX, "The random partitions mix prong counts, which likely limits H→4q."),
    ("tab:pairs", APPENDIX, "Pair separations, linear probe and fine-tuned, quoted from this table."),
    ("tab:sculpt", APPENDIX, "Sculpting at 10% and 1% QCD efficiency, beside the random-selection floor."),
    ("tab:sculptboot", APPENDIX, "The sculpting differences and their combined uncertainties that decide [[F10b]]'s "
                                 "tier."),
    ("tab:boot", APPENDIX, "Every \"consistent within uncertainty\" rests on these uncertainties."),
    ("tab:compute", APPENDIX, "The paper reports each run's GPU types and GPU-hours."),
    ("tab:perrun", SUPPLEMENT, "Cited from the uncertainties section: every run."),
    *((k, SUPPLEMENT, "Cited beside [[tab:jetclass]]: all models, passing QCD jets, probes, class AUCs.")
      for k in ("tab:jetclassall", "tab:jetclassnbkg", "tab:jetclassprobe", "tab:jc1_class_auc")),
    ("tab:massdetail", SUPPLEMENT, "Cited beside [[tab:mass]]: per-bin mass detail."),
)
# what a sentence rests on beyond its own item
EVIDENCE = {
    "F2a": ("tab:probes",), "F2b": ("tab:jetclass", "tab:benchtop + tab:benchqg"), "F3-main": ("tab:ft",),
    "F4-mech": ("tab:pairs",), "F8": ("tab:ft",), "A8": ("tab:ft",), "F9b": ("tab:massdetail",),
    "F4-other": ("tab:pairs",), "F4-mech-ft": ("tab:pairs",), "F6": ("tab:benchtop + tab:benchqg", "tab:boot"),
    "F6-app": ("tab:ft",), "A10": ("tab:probes",), "F9a": ("tab:massdetail",),
    **{k: ("tab:sculpt", "tab:sculptboot") for k in ("F10b", "F10b-flav", "F10c", "F10a-flav")}, "tab:mass": ("tab:massdetail",),
}
TIER = {k: t for k, t, _ in _FIGURES + _TABLES}
# the PI's book numbers (2026-11-08 message) -> gallery ids
BOOK_NAMES = (("F9", "F8"), ("F16", "A8"), ("F15", "A10"), ("F5", "FJ"), ("F7", "PT"), ("F8", "F6"), ("F17", "F6-app"),
              ("F10", "F9a"), ("F12", "F10a"), ("F14", "VAL"), ("F6, F18-F20", "FJ-ctrl, FJ-app, FJ-ctrl-app, FJ-s"),
              ("F21", "CONF"), ("F28, F29", "A3R, A3R-ft"), ("F23b", "F3-main (a)"), ("F26c", "F3-main (b)"),
              ("UMAP figure", "UMAP-jc2-had"))
DROPPED = ("F3", "F4", "F4-app", "F3-scatter")      # F3-scatter: its panels are F3-main (a) and F3-app (b)
# mlst_results tables the plan places nowhere: nothing cites them, so they are dropped (PI 2026-11-08); the plan's
# "missing inputs (fill)" is tab:fill and its "fine-tuning vs size" tab:ft, both placed above
DROPPED_TABLES = ("tab:balance", "tab:fill_main", "tab:ftrej", "tab:perclass", "tab:probesn")
GALLERY_ONLY = ("INVENTORY", "SOPHON-FIG2", "PALETTE")    # gallery pages that are not figures of the paper
REF = re.compile(r"\[\[(.+?)\]\]")


def number(key: str) -> str:
    """The item's number within its tier: "Fig. 4", "Fig. A3", "Table S2"."""
    table = key.startswith("tab:")
    tier = TIER[key]
    same = [k for k, t, _ in (_TABLES if table else _FIGURES) if t == tier]
    prefix = {MAIN: "", APPENDIX: "A", SUPPLEMENT: "S"}[tier]
    return f"{'Table' if table else 'Fig.'} {prefix}{same.index(key) + 1}"


def _fill(sentence: str) -> str:
    return REF.sub(lambda m: number(m[1]), sentence)


FIGURES = tuple((k, t, _fill(s)) for k, t, s in _FIGURES)
TABLES = tuple((k, t, _fill(s)) for k, t, s in _TABLES)
BACKS = {k: s for k, _, s in FIGURES + TABLES}


def evidence(key: str) -> tuple:
    return (key,) + EVIDENCE.get(key, ())


def f10b_main(differences: dict) -> bool:
    """F10b's promotion (F10B_RULE) from sculpting_boot.json's "differences": each rule difference at F10B_WP of its
    sign and beyond 3 sigma, sign x delta > 3 sigma."""
    ds = [(differences.get(name, {}).get(F10B_WP, {}).get("distance"), sign) for name, sign in F10B_RULE]
    return all(d is not None and d["sigma"] is not None and sign * d["delta"] > 3 * d["sigma"] for d, sign in ds)


def place(key: str) -> str:
    """"Appendix, Fig. A11; waits for the fine-tuned read-out"."""
    note = f"; {PROVISIONAL[key]}" if key in PROVISIONAL else ""
    return f"{TIER_NAME[TIER[key]]}, {number(key)}{note}"
