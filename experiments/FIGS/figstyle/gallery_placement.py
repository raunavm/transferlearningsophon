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
PROVISIONAL = {}     # nothing waits: the fine-tuned read-out (2026-11-09) and X→bb/X→bs on Sophon's samples (2026-11-11) are in
FJ_S = "Cited beside [[FJ]]: partitions, linear probes, from scratch."
SCATTER_S = "Cited from [[F3-main]]: the other scatter panels."
AUC_S = "[[F8]], [[A8]] and [[A10]] in AUC."
XBX = "17 classes keep fewer X→bb̄ and X→bs signal jets than 188 on Sophon's samples."
MASS_S = "Cited beside [[F9b]]: the response distributions, by jet group, and from frozen features."
SCULPT_S = "Cited beside [[F10b]]: the histograms, flavor boundaries, rejection plane, fine-tuned and mass models."

# (gallery id, tier, draft sentence), in paper order within each tier; items that share a sentence share a row on the map.
# PI 2026-11-12: the paper keeps the main-text figures and, in the appendix, F4-other, F4-mech-ft, PT, F6, A10, F10b,
# UMAP-jc2-had and VAL; everything else is in the supplement (nothing deleted).
_FIGURES = (
    ("F1", MAIN, "We pretrain one network on the same jets, in the same order, and change only the label set."),
    ("F2a", MAIN, "On JetClass-II, frozen features lose with each coarser label set, most where quark flavor merges, "
                  "and at a fixed class count by which classes merge; on JetClass only 17 below 43 is resolved."),
    ("F2b", MAIN, "Fine-tuned, 17 classes give about 3% more errors on JetClass, the label sets are consistent on top "
                  "and quark/gluon; JetClass-II shows partial recovery."),
    ("F3-main", MAIN, "The 17-class loss falls unevenly over the classes; fine-tuning shrinks it but does not remove it."),
    ("F4-mech", MAIN, "Linear probes separate quark flavors worse with 17 classes than with 188, and a flavor boundary "
                      "in a 17-class partition recovers part of the loss, except on X→bb̄ vs cc̄."),
    ("F8", MAIN, "Fine-tuning the 188-class network beats training from scratch on JetClass-II and JetClass at every "
                 "size; on top and quark/gluon tagging the two are consistent."),
    ("A8", MAIN, "Fine-tuned, 17 classes stay lowest on JetClass-II at every size and below 188 on JetClass at 10,000 "
                 "and a million jets; top and quark/gluon are consistent."),
    ("F9b", MAIN, "An auxiliary mass regression gives a better jet-mass resolution than the soft-drop mass, shown in "
                  "bins of jet pT and generator mass."),
    ("F11", MAIN, XBX),
    ("F4-other", APPENDIX, "On QCD flavor pairs 17-class probes separate least; on X→ee vs μμ 43 classes fall below 188; 162 "
                           "loses on QCD bb̄ vs cc̄."),
    ("F4-mech-ft", APPENDIX, "Fine-tuned, 17 classes still separate quark-flavor pairs worse than 188, by less than "
                             "probes; the flavor boundaries are consistent with none."),
    ("PT", APPENDIX, "Per jet-pT bin, 43 and 17 classes fall behind 188 on JetClass-II probes; elsewhere they agree "
                     "but in the lowest JetClass bin."),
    ("F6", APPENDIX, "Fine-tuned on top and quark/gluon tagging, Pythia and Herwig, the label sets are consistent."),
    ("A10", APPENDIX, "At every size the 43- and 17-class probes fall behind 188 on JetClass-II, 162 at all sizes but "
                      "10,000 jets; on JetClass 17 fall below 43 only."),
    # PI 2026-11-09: the main text reports exactly the differences beyond 3 sigma (tab:sculptboot), nothing else
    ("F10b", APPENDIX, "17 classes sculpt the QCD mass more than 188 at 1% QCD efficiency, 43 at 10%; mass regression "
                       "reduces it for 162 classes at 10%; every other difference (17 + mass, flavor boundaries) is "
                       "within the uncertainty."),
    ("UMAP-jc2-had", APPENDIX, "Qualitatively, 188-class frozen features separate X→bb̄, cc̄ and qq̄ and 17-class ones "
                               "only order them; after fine-tuning both separate them."),
    ("VAL", APPENDIX, "Validation accuracy levels off for every label set during pretraining."),
    ("FJ", SUPPLEMENT, "Cited beside [[tab:jetclassrej]]: the per-class ROC curves its rejections come from."),
    *((k, SUPPLEMENT, FJ_S) for k in ("FJ-ctrl", "FJ-app", "FJ-ctrl-app", "FJ-s")),
    ("CONF", SUPPLEMENT, "Cited beside [[tab:jetclassrej]]: JetClass confusion matrices."),
    *((k, SUPPLEMENT, SCATTER_S) for k in ("F3-app", "F3-app2", "F3-app3", "F3-ft", "F3-ft2")),
    *((k, SUPPLEMENT, "Cited from [[F3-main]]: every class's ROC curve.") for k in ("A3R", "A3R-ft")),
    ("F6-app", SUPPLEMENT, "Cited beside [[F6]]: the linear probes' ROC curves on top and quark/gluon tagging."),
    *((k, SUPPLEMENT, AUC_S) for k in ("F8-auc", "A8-auc", "A10-auc")),
    *((k, SUPPLEMENT, MASS_S) for k in ("F9a", "A17-res", "A17-qcd", "A17-probe")),
    *((k, SUPPLEMENT, SCULPT_S) for k in ("F10a", "F10b-flav", "F10c", "F10a-flav", "F10-ft", "F10a-mass")),
    *((k, SUPPLEMENT, "Cited from [[UMAP-jc2-had]]: the other tasks.")
      for k in ("UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg")),
)
# (table label, tier, draft sentence). PI 2026-11-12: one main-text table holds every value of Figs. 2 and 3
# (tab:summary); one compares our network, from scratch and fine-tuned from 188 classes, with the published models on
# JetClass, top and quark/gluon tagging (tab:published, in place of tab:jetclass, tab:benchtop and tab:benchqg); the
# other label sets fine-tuned on those tasks and every network scored on Herwig jets in one appendix table
# (tab:labelsets, which takes in tab:benchqgherwig); JetClass's per-class rejections in the appendix (tab:jetclassrej);
# the X->bb / X->bs rejections in the appendix; the trimmed appendix tables whole (every partition, rho) in the supplement
# (tab:*full); tab:boot nowhere (its probe spreads disagree with tab:probes; the file is kept).
_TABLES = (
    ("tab:summary", MAIN, "Every value of [[F2a]] and [[F2b]]."),
    ("tab:published", MAIN, "Fine-tuned from 188 classes, our network is more accurate on JetClass than ParT trained from "
                            "scratch on twice the jets; on top and quark/gluon tagging its rejection at 30% signal "
                            "efficiency lies between ParT trained from scratch and ParT fine-tuned."),
    ("tab:mass", MAIN, "Over all jets, the mass output of the pretraining head that [[F9b]] shows in bins has a "
                       "narrower response than the soft-drop mass; an auxiliary mass regression also puts the jet mass "
                       "into the frozen features."),
    ("tab:xbx", MAIN, XBX),
    ("tab:probes", APPENDIX, "Every probe behind [[F2a]], with the released Sophon checkpoint."),
    ("tab:coarse", APPENDIX, "Scored on the 43 groups, 17-class features fall behind; on the 17 groups the label sets "
                             "are consistent."),
    ("tab:qcdhf", APPENDIX, "With the 27 QCD classes merged (162), the X→bb̄ discriminant's rejection of b-flavored QCD "
                            "is consistent with 188."),
    ("tab:fill", APPENDIX, "Random draws for the missing top inputs beat zeros; the mean, and every quark/gluon fill, are "
                           "consistent with zeros."),
    ("tab:inputs", APPENDIX, "Top and quark/gluon data lack some inputs, which are set to zero."),
    ("tab:partrepro", APPENDIX, "ParT's training script on our cluster is consistent with its published top-tagging "
                                "rejection; with Sophon's jet-pT rescaling alone the rejection is 28% lower."),
    ("tab:lrsweep", APPENDIX, "The fine-tuning learning rate was chosen on validation."),
    ("tab:jetclassrej", APPENDIX, "Per class, the label sets are consistent on JetClass after fine-tuning."),
    ("tab:labelsets", APPENDIX, "Fine-tuned on a million JetClass jets, 17 classes give about 3% more errors than 188; on "
                                "top and quark/gluon tagging, Pythia and Herwig, the label sets are consistent."),
    ("tab:pairs", APPENDIX, "Pair separations, linear probe and fine-tuned, quoted from this table."),
    ("tab:sculpt", APPENDIX, "Sculpting at 10% and 1% QCD efficiency, beside the random-selection floor."),
    ("tab:sculptboot", APPENDIX, "The sculpting differences and their combined uncertainties that decide [[F10b]]'s "
                                 "tier."),
    ("tab:xbxrej", APPENDIX, "Rejections at 50% and 30% signal efficiency for the configurations of [[tab:xbx]]."),
    ("tab:perrun", SUPPLEMENT, "Cited from the uncertainties section: every run."),
    *((k, SUPPLEMENT, "Cited beside [[tab:jetclassrej]]: all models, passing QCD jets, probes, class AUCs.")
      for k in ("tab:jetclassall", "tab:jetclassnbkg", "tab:jetclassprobe", "tab:jc1_class_auc")),
    ("tab:massdetail", SUPPLEMENT, "Cited beside [[tab:mass]]: per-bin mass detail."),
    ("tab:mlp", SUPPLEMENT, "Cited beside [[tab:probes]]: nonlinear probes repeat the linear-probe comparisons."),
    ("tab:bn", SUPPLEMENT, "Cited beside [[tab:probes]]: BatchNorm statistics recomputed, against the checkpoints as trained."),
    ("tab:ft", SUPPLEMENT, "Cited beside [[F8]] and [[A8]]: every label set fine-tuned at every training-set size."),
    ("tab:randprong", SUPPLEMENT, "Cited beside [[tab:summary]]: the prong counts each random partition mixes."),
    ("tab:compute", SUPPLEMENT, "Cited from the methods: each run's GPU types and GPU-hours."),
    *((k, SUPPLEMENT, "Cited beside the appendix table of the same name: every random partition, the ratio to 188 "
                      "classes and every block.")
      for k in ("tab:probesfull", "tab:coarsefull", "tab:qcdhffull", "tab:pairsfull", "tab:fillfull")),
)
# what a sentence rests on beyond its own item
EVIDENCE = {
    "F2a": ("tab:summary", "tab:probes"), "F2b": ("tab:summary", "tab:published", "tab:labelsets"),
    "F3-main": ("tab:ft",),
    "F4-mech": ("tab:pairs",), "F8": ("tab:ft",), "F8-auc": ("tab:ft",), "A8": ("tab:ft",), "A8-auc": ("tab:ft",), "F9b": ("tab:mass", "tab:massdetail"),
    "F4-other": ("tab:pairs",), "F4-mech-ft": ("tab:pairs",), "F6": ("tab:published", "tab:labelsets"),
    "F6-app": ("tab:ft",), "A10": ("tab:probes",), "A10-auc": ("tab:probes",), "F9a": ("tab:massdetail",),
    **{k: ("tab:sculpt", "tab:sculptboot") for k in ("F10b", "F10b-flav", "F10c", "F10a-flav")}, "tab:mass": ("tab:massdetail",),
    "tab:labelsets": ("tab:published",),      # the 188-class and from-scratch rows its comparisons are against
    "tab:partrepro": ("tab:published",),      # ParT's published row and our from-scratch row
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
DROPPED_TABLES = ("tab:balance", "tab:fill_main", "tab:ftrej", "tab:perclass", "tab:probesn",
                  "tab:boot",     # tab:boot: PI 2026-11-12, its probe spreads disagree with tab:probes
                  # PI 2026-11-12: their rows are in tab:published and tab:labelsets
                  "tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig")
COMPARE = ("CMP-F7-B", "CMP-F7-C", "CMP-A4-B", "CMP-A4-C", "CMP-FONT-T", "CMP-FONT-LM")     # the PI's style comparison, 2026-11-11
GALLERY_ONLY = ("INVENTORY", "SOPHON-FIG2", "PALETTE") + COMPARE    # gallery pages that are not figures of the paper
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
