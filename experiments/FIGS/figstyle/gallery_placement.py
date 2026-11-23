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

import os
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
# (sculpting_boot.DIFFS name, working point). PI 2026-11-13 (L3): sigma^2 = each side's SD^2 over its runs + the
# bootstrap SD^2 over test jets; the paired-by-run test of sculpting_boot.json ("pass") is never the sole basis. Under
# L3 none passes (bootstrap/l3/sculpting_boot_l3.json, tab:sculptboot: 162 + mass - 162 at 10%, 2.76 sigma; 17 - 188
# at 1%, 2.55; 17 + mass - 17 at 1%, 2.45; 43 - 188 at 10%, 0.81)
F10B_STATED = ()
# a placement that waits on a result; the two tables only while they hold pending cells (make_results_book sets a
# PROVISIONAL table with its pending cells, and a box for a PROVISIONAL figure not drawn yet)
# the fine-tuned read-out (2026-11-09) and X→bb/X→bs on Sophon's samples (2026-11-11) are in; the partitions' mass probes
# are being fitted (PI 2026-11-14): their two supplement figures are boxed until then
PROVISIONAL = {k: "waits for the partitions' mass probes" for k in ("A17-probe-part", "A17-ridge-part")}
# PI 2026-11-23 (LEDGER SP-9, SP-11, SP-12 step 4): the stated differences without a paired bootstrap yet, and the
# JetClass rows of the published comparison until the 10^6 rerun is read (no superseded fine-tuning run is printed)
PROVISIONAL.update({"tab:tests": "waits for the paired bootstraps of its pending rows",
                    "tab:published": "waits for the JetClass rerun at a million jets"})
FJ_S = ("Each JetClass class's ROC curve against the other nine is in [[FJ-app]] and [[FJ]], and for the 17-class "
        "partitions, the mass networks and training from scratch in [[FJ-ctrl-app]], [[FJ-mass-app]], [[FJ-ctrl]], "
        "[[FJ-mass]] and [[FJ-s]].")
PER_CLASS_S = ("[[PC-jc2-part]] to [[PC-jc1-mass]] repeat the per-class comparison of [[PC-jc2]] and [[PC-jc1]] for the "
               "17-class partitions and for each mass network against its own label set.")
CONF_S = "The JetClass confusion matrices of every pretrained network are in [[PC-conf]], [[PC-conf-part]] and [[PC-conf-mass]]."
AUC_S = "[[F8-auc]] repeats [[F8]] in the area under the ROC curve."
PAIRS_S = ("The pair X→ee vs μμ is in [[F4-ee]], and the class pairs of [[F4-mech]] and [[F4-other]] for the random "
           "partitions are in [[F4-rand]] to [[F4-rand-qcd-ft]].")
SIZE_PART_S = "[[A10-part]] and [[A8-part]] show the 17-class partitions against training-set size."
XBX = "17 classes keep fewer X→bb̄ and X→bs signal jets than 188 on Sophon's samples."
MASS_S = ("The response distributions, the mass output by jet group and the ridge regression on frozen features are in "
          "[[F9a]] to [[A17-ridge]].")
SCULPT_S = ("The mass histograms, the flavor-boundary partitions, the rejection plane, the fine-tuned networks and the "
            "mass networks of the sculpting comparison of [[F10b]] are in [[F10a]] to [[F10a-mass]].")
# L3 for the resonance read-outs (tests/test_xbx_claims_l3.py, which fails the build if a claim stops holding): run SDs
# plus the test-sample variance of each side, the signal jets (binomial) and the QCD jets through the cut (ROC slope at
# eps_B = 1%): 17 - 188 is -4.9 sigma on X->bb and -5.8 on X->bs; each flavor-boundary partition - 188 at least -5.2;
# every partition - 17 at most 2.8 sigma (random partitions on X->bs)
XBX_PART = ("At 1% QCD efficiency, the partitions with no, a b-quark and a random flavor boundary each keep fewer X→bb̄ "
            "and X→bs signal jets than 188 classes, and no 17-class partition differs from 17 classes by more than three "
            "standard deviations.")

# (gallery id, tier, draft sentence), in paper order within each tier; items that share a sentence share a row on the map.
# PI 2026-11-23 (LEDGER SP-9): the main text keeps F1, F2, F2-ft, PC-jc2, F4-mech, A8 and F11, the appendix VAL, A10,
# F4-other and F9b; everything else is in the supplement, the data release (nothing deleted).
_FIGURES = (
    ("F1", MAIN, "We pretrain one network on the same jets, in the same order, and change only the label set."),
    # PI 2026-11-13 (L1): the headline family's main-text slice, the headline metric of both modes; L20 (points-only
    # panels at least 4.5 cm) splits it one mode a figure, linear probes (F2) and fine-tuned (F2-ft), one sentence; the
    # same in the AUC (F2a, F2b) opens the appendix
    # PI 2026-11-21: no "no difference on top" while any printed metric passes 3 sigma; the top and quark/gluon clause
    # is written by hand in F2's caption, R30 only, and tested against the bootstrap (gallery_labelset.TOP_QG_PROBE_CLAIMS)
    *((k, MAIN, "With linear probes, 43 and 17 classes and every 17-class partition are less accurate on JetClass-II "
                "than 188 classes, and after fine-tuning 17 classes stay below 188 on JetClass-II and JetClass.")
      for k in ("F2", "F2-ft")),
    # PI 2026-11-13 (L10): the 17-class loss is broad, not uneven; PI 2026-11-14: the counts follow L3 (run SDs and the
    # per-class test-jet bootstrap where stored) and are computed with the figure (gallery_perclass.l3_counts, in its
    # caption), so the sentence holds no number; the class averages are Table 1's JetClass-II AUC
    ("PC-jc2", MAIN, "The 17-class loss is broad, spread over most of the 188 JetClass-II classes with linear probes "
                     "and after fine-tuning, and fine-tuning shrinks the class-averaged loss but does not remove it."),
    # L3 with run SDs: b-quark boundary - 17 on X->bb vs cc 8 sigma and on X->bc 6 sigma; random flavor boundary - 17 on
    # X->bc 6 sigma, on X->bb vs cc 2.3 (not claimed); no flavor boundary: on X->bc 0.9 sigma from 17 and 7.6 below 188;
    # on X->bb vs cc its two runs (+-0.012) place it within 1.0 sigma of 17 and 1.6 of 188, so nothing is said there
    ("F4-mech", MAIN, "Linear probes separate quark flavors worse with 17 classes than with 188; the b-quark boundary "
                      "recovers part of the loss on both pairs and the random flavor boundary on X→bc vs bq, cs, while "
                      "the partition with no flavor boundary recovers none on X→bc vs bq, cs."),
    # PI 2026-11-13 (L9, L13): the training-size family's main slice is the fine-tuned figure; the 188-class network
    # against from scratch (F8) is a named slice in the supplement. L3: JetClass at 10,000 jets 3.5 sigma on run SDs
    # with no bootstrap stored at that size (not claimed, round-1 fix 2026-11-13), at 100,000 1.6 sigma (not claimed);
    # top and quark/gluon, Pythia and Herwig, |z| < 2.2 everywhere
    # PI 2026-11-18: Fig. 7 carries the AUC as its second row (A8-auc folded in)
    ("A8", MAIN, "Fine-tuned, 17 classes stay below 188 classes on JetClass-II at every size and on JetClass at a "
                 "million jets."),
    ("F11", MAIN, XBX),
    # PI 2026-11-23 (LEDGER SP-9, the structure approved as proposed, docs/STRUCTURE_PROPOSAL.md): the appendix
    # holds VAL, A10, F4-other and F9b, by family (APPENDIX_FAMILIES); every other figure placed before goes to the
    # data release (the supplement tier: the results book and data_release.py)
    ("VAL", APPENDIX, "Validation accuracy levels off for every label set and mass network during pretraining."),
    # the 162-class probe in accuracy is within 2.2 sigma of 188 at every size, so the old "162 at all sizes but
    # 10,000 jets" (an AUC reading) is retired
    ("A10", APPENDIX, "With linear probes, 43 and 17 classes fall behind 188 classes on JetClass-II at every training "
                      "size."),
    # PI 2026-11-23 (LEDGER SP-9, merge 8): F4-light merged into F4-other, its pairs as more panels and no partitions
    # row for the QCD pairs (every partition merges QCD)
    ("F4-other", APPENDIX, "With linear probes, 43 and 17 classes separate the four other flavor pairs worse than 188 "
                           "classes, and 162 classes, which merge the QCD classes, the three QCD pairs."),
    # PI 2026-11-13 (L9): Fig. 8 to the appendix; Table 5 (tab:mass) stays in the main text and carries the claim;
    # audit 3: in the 20-50 GeV generator-mass bin both mass outputs are more biased than the soft-drop mass
    ("F9b", APPENDIX, "An auxiliary mass regression gives a better jet-mass resolution than the soft-drop mass in every "
                      "bin of jet pT and generator mass, and less bias in most of them, but more in the lowest "
                      "generator-mass bin."),
    # the data release (PI 2026-11-23, LEDGER SP-9): the figures the appendix held before
    # the class-pair family (PI 2026-11-13): one panel order for both modes; L3 with the probes' test-jet bootstrap,
    # fine-tuned pairs on run SDs (no bootstrap stored). The two modes are scored on different test jets and with
    # different read-outs (audit 3, P4), so no sentence compares them
    ("F4-mech-ft", SUPPLEMENT, "Fine-tuned, 17 classes still separate X→bb̄ vs cc̄ and X→bc vs bq, cs worse than 188 "
                             "classes."),
    *((k, SUPPLEMENT, "Fine-tuned, 17 classes still separate X→YY→bb̄qq vs cc̄qq, QCD bb̄ vs cc̄ and QCD with c "
                    "vs light worse than 188 classes.")
      for k in ("F4-other-ft", "F4-light-ft")),
    # PI 2026-11-20 (meeting): to the appendix; PI 2026-11-18: ROC curves, one figure a mode: all nine JetClass signal classes against QCD jets
    # (ParT's convention) and top and quark/gluon tagging (F6 and F6-ft, now in the supplement)
    *((k, SUPPLEMENT, "On JetClass, 17 classes reject QCD jets less well than 188 classes in the flavour classes, as linear "
                "probes and after fine-tuning on a million jets, while on top and quark/gluon tagging the label sets' ROC "
                "curves agree in both modes.") for k in ("ROC-main", "ROC-main-ft")),
    # the jet-pT family (PI 2026-11-13, L4): L3 per bin with the run SDs and the binomial test-jet term of each side:
    # probes on JetClass-II 43 and 17 - 188 at least 9.5 sigma in every bin; fine-tuned 17 - 188 at least 4.5 sigma
    # in every JetClass-II bin, 7.1 and 3.4 in the two lowest JetClass bins, at most 2.6 above them; JetClass probes
    # 17 - 188 at 3.38 sigma in 900-1000 GeV (with the binomial term, so it passes L3; audit 3) and 2.4-3.0 below it
    ("PT", SUPPLEMENT, "Per jet-pT bin, 43 and 17 classes trail 188 on JetClass-II probes; fine-tuned, 17 classes trail "
                     "in every JetClass-II bin and the two lowest JetClass bins, and as a linear probe in the highest "
                     "JetClass bin."),
    # PI 2026-11-14 (step 5): the mass probe on frozen features next to F9b
    ("A17-probe", SUPPLEMENT, "On frozen features, a multilayer-perceptron probe resolves the jet mass better with an "
                            "added mass output than with the label set alone."),
    # PI 2026-11-13 (L3): no sculpting difference passes (F10B_STATED), so the sentence says so; PI 2026-11-14: (a) used
    # directly, (b) fine-tuned
    ("F10b", SUPPLEMENT, "Used directly, no label set or mass network sculpts the QCD mass differently from 188 classes, "
                       "or a mass network from its own label set, by more than three standard deviations at 10% or 1% "
                       "QCD efficiency or 50% signal efficiency."),
    ("F11-part", SUPPLEMENT, XBX_PART),
    # PI 2026-11-14 (step 5): the embeddings of every task, one grid per mode; qualitative, one run
    *((k, SUPPLEMENT, "In this UMAP projection, qualitatively, 188-class frozen features separate X→bb̄, cc̄ and qq̄ "
                    "more cleanly than 17-class ones, and after fine-tuning both separate them.")
      for k in ("UMAP-frozen", "UMAP-ft")),
    *((k, SUPPLEMENT, "[[F2a]] and [[F2b]] repeat [[F2]] and [[F2-ft]] in the area under the ROC curve.")
      for k in ("F2a", "F2b")),
    # PI 2026-11-20 (meeting): out of the paper, every point on the diagonal
    ("PC-jc1", SUPPLEMENT, "Fine-tuned on JetClass, 17 classes separate several of the ten classes less well than 188 "
                         "classes, and 162 and 43 classes agree with 188 on every class."),
    *((k, SUPPLEMENT, "The top and quark/gluon ROC curves of [[ROC-main]] are drawn with every label set and mass "
                      "network, as linear probes and fine-tuned, in [[F6]] and [[F6-ft]].") for k in ("F6", "F6-ft")),
    *((k, SUPPLEMENT, PER_CLASS_S) for k in ("PC-jc2-part", "PC-jc2-rand", "PC-jc1-part", "PC-jc1-rand", "PC-jc2-mass",
                                             "PC-jc1-mass")),
    *((k, SUPPLEMENT, CONF_S) for k in ("PC-conf", "PC-conf-part", "PC-conf-mass")),
    *((k, SUPPLEMENT, PAIRS_S) for k in ("F4-ee", "F4-rand", "F4-rand-qcd", "F4-rand-ft", "F4-rand-qcd-ft")),
    ("F8", SUPPLEMENT, "[[F8]] sets the network pretrained with 188 classes, fine-tuned and as a linear probe, against "
                       "training from scratch at every training-set size."),
    ("F8-auc", SUPPLEMENT, AUC_S),
    *((k, SUPPLEMENT, SIZE_PART_S) for k in ("A10-part", "A8-part")),
    *((k, SUPPLEMENT, "The ROC curves of [[F6]] and [[F6-ft]] for the 17-class partitions are in [[F6-part]] and "
                      "[[F6-part-ft]].") for k in ("F6-part", "F6-part-ft")),
    # PI 2026-11-18: every JetClass class against QCD is in ROC-main / ROC-main-ft, so the class-against-rest curves
    # are supplement items
    *((k, SUPPLEMENT, FJ_S) for k in ("FJ-app", "FJ", "FJ-ctrl-app", "FJ-mass-app", "FJ-ctrl", "FJ-mass", "FJ-s")),
    *((k, SUPPLEMENT, "Every JetClass-II class's ROC curve is in [[A3R]] to [[A3R-s]].")
      for k in ("A3R", "A3R-ft", "A3R-ctrl-ft", "A3R-s")),
    *((k, SUPPLEMENT, "[[PT-part]] to [[PT-bbcc-part]] repeat [[PT]] for the partitions and X→bb̄ vs cc̄.")
      for k in ("PT-part", "PT-bbcc", "PT-bbcc-part")),
    *((k, SUPPLEMENT, MASS_S) for k in ("F9a", "A17-res", "A17-qcd", "A17-ridge")),
    # PI 2026-11-14 (step 7): the partitions' mass probes, drawn once fitted (PROVISIONAL until then)
    *((k, SUPPLEMENT, "[[A17-probe-part]] and [[A17-ridge-part]] show the mass probes of [[A17-probe]] and "
                      "[[A17-ridge]] for the 17-class partitions.") for k in ("A17-probe-part", "A17-ridge-part")),
    *((k, SUPPLEMENT, SCULPT_S) for k in ("F10a", "F10b-flav", "F10c", "F10a-flav", "F10-ft", "F10-ft-part",
                                          "F10a-mass")),
    ("F11-probe", SUPPLEMENT, "[[F11-probe]] repeats [[F11]] with a linear probe on the same features and adds the "
                              "random network."),
    *((k, SUPPLEMENT, "[[UMAP-jc2-had]] to [[UMAP-xbx]] show the embeddings of [[UMAP-frozen]] and [[UMAP-ft]] task by "
                      "task, frozen beside fine-tuned, and [[UMAP-mass-jc2-had]] to [[UMAP-mass-xbx]] for the mass "
                      "networks beside their label sets.")
      for k in ("UMAP-jc2-had", "UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg", "UMAP-xbx", "UMAP-mass-jc2-had",
                "UMAP-mass-jc1", "UMAP-mass-jc2-lep", "UMAP-mass-top", "UMAP-mass-qg", "UMAP-mass-xbx")),
    ("VAL-part", SUPPLEMENT, "[[VAL-part]] shows the pretraining curves of [[VAL]] for the 17-class partitions."),
)
# (table label, tier, draft sentence). PI 2026-11-12: one main-text table holds every value of Figs. 2 and 3
# (tab:summary); one compares our network, from scratch and fine-tuned from 188 classes, with the published models on
# JetClass, top and quark/gluon tagging (tab:published, in place of tab:jetclass, tab:benchtop and tab:benchqg); the
# other label sets fine-tuned on those tasks and every network scored on Herwig jets in one appendix table
# (tab:labelsets, which takes in tab:benchqgherwig); JetClass's per-class rejections in the appendix (tab:jetclassrej);
# the X->bb / X->bs rejections in the appendix; the trimmed appendix tables whole (every partition, rho) in the supplement
# (tab:*full); tab:boot nowhere (its probe spreads disagree with tab:probes; the file is kept).
_TABLES = (
    # PI 2026-11-19 (B) and 2026-11-22 (Part 2): the compact table of label sets and controls, first in the main text
    ("tab:sets", MAIN, "[[tab:sets]] lists the label sets and the controls: each one's classes, what it merges and its "
                       "runs."),
    # PI 2026-11-23 (LEDGER SP-9; merge 4): tab:xbx's two transfer-head columns at 1% QCD efficiency join Table 1
    ("tab:summary", MAIN, "[[tab:summary]] gives every value of [[F2]] and [[F2-ft]], with the released Sophon "
                          "checkpoint's, and the X→bb̄ and X→bs signal efficiencies of [[F11]] at 1% QCD efficiency."),
    # L3 (round 2, package 3), sigma^2 = our run SD^2 + the published run SD^2 + our test-jet bootstrap SD^2
    # (bootstrap/l3, per network): JetClass accuracy 0.8468 - 0.836, 17 sigma; R30 against ParT trained from scratch,
    # top 1,800 - 1,602, 1.0 sigma, quark/gluon 107.3 - 101.2, 1.4 sigma, so the old "between ParT from scratch and ParT
    # fine-tuned" fails on its lower bound; top against ParT fine-tuned's 2,766 is 4.3 sigma with our test-jet term but
    # about 2.4 with a binomial term for ParT's 73 passing jets as well, so it is not claimed either
    ("tab:published", MAIN, "Fine-tuned from 188 classes, our network is more accurate on JetClass than ParT trained from "
                            "scratch on twice the jets, and its top and quark/gluon rejections at 30% signal efficiency "
                            "are consistent with ParT trained from scratch."),
    ("tab:mass", MAIN, "Over all jets, the mass output of the pretraining head that [[F9b]] shows in bins has a "
                       "narrower response than the soft-drop mass; an auxiliary mass regression also puts the jet mass "
                       "into the frozen features."),
    # PI 2026-11-23 (LEDGER SP-9): the appendix's tables, by family (APPENDIX_FAMILIES): R3a first (class map, inputs,
    # rate search, compute, the substitution of the missing inputs), then the evidence for the claims the main text
    # states but does not show (R3b)
    # merge 3: the class map takes in tab:app-levels, its 17 group names as block headers, the levels never pretrained
    # deleted
    ("tab:app-vocabulary", APPENDIX, "[[tab:app-vocabulary]] gives every JetClass-II class's group in each pretrained "
                                     "label set and partition, under the groups of the 17-class label set."),
    # PI 2026-11-20 (meeting, B): from the supplement into the appendix
    # audit 3 (B): only fine-tuning sets the missing inputs to zero; the linear probes use the substitution chosen on
    # validation, random draws on top and the training mean on quark/gluon
    ("tab:inputs", APPENDIX, "Top and quark/gluon data lack some inputs, which fine-tuning sets to zero and the linear "
                             "probes to random draws (top) or the training mean (quark/gluon)."),
    ("tab:lrsweep", APPENDIX, "The fine-tuning learning rate was chosen on validation."),
    # PI 2026-11-23 (LEDGER SP-9): one row per stage; each run's row is in the data release (tab:computefull)
    ("tab:compute", APPENDIX, "The GPU types and GPU-hours of pretraining and fine-tuning are in [[tab:compute]]."),
    # L3 (round 2, package 3), run SDs and a test-jet term scaled from bootstrap/l3: top draws - zero at least 4.0 sigma
    # for every pretrained network (162 classes), the random network -0.7; top mean - zero within 1.2; quark/gluon
    # within 2.9 except the b-quark boundary (mean 4.6, draws 4.0 on two runs each, about 3.6 and 3.2 with the term)
    ("tab:fill", APPENDIX, "Random draws for the missing top inputs beat zeros for every pretrained network; the mean, "
                           "and both quark/gluon fills but with the b-quark boundary, are consistent with zeros."),
    # PI 2026-11-20 (meeting, B): from the supplement into the appendix
    ("tab:mlp", APPENDIX, "Multilayer-perceptron probes for the comparisons of [[tab:summary]] are in [[tab:mlp]], the "
                       "random partitions pooled."),
    # L3 (round 2, package 3): single runs, so sigma is the binomial count of the background jets passing (118 and
    # 165): unchanged 1,710 against ParT's 1,602 +- 81, 0.6 sigma; rescaled 1,220 against unchanged, 2.7 sigma, not a
    # claim, so no "28% lower" (audit 3, C)
    ("tab:partrepro", APPENDIX, "ParT's training script on our cluster is consistent with its published top-tagging "
                                "rejection, with or without Sophon's output head or jet-pT rescaling."),
    # PI 2026-11-18: the AUC as the secondary metric, beside every headline value
    ("tab:summaryauc", APPENDIX, "[[tab:summaryauc]] gives the area under the ROC curve beside every value of "
                                 "[[tab:summary]], and on JetClass-II over its resonant and its QCD classes."),
    # PI 2026-11-23 (LEDGER SP-9; merge 6): tab:sculptboot and tab:boot become one table of every difference or null
    # the text states; PI SC-1 (1): sigma^2 = each side's run SD^2 + the paired test-jet bootstrap SD^2
    ("tab:tests", APPENDIX, "[[tab:tests]] gives every difference and null the text states, with each side's run "
                            "standard deviation, the paired test-jet bootstrap standard deviation and Δ/σ."),
    ("tab:coarse", APPENDIX, "Linear probes scored on the 43 and 17 merged classes are in [[tab:coarse]]."),
    # PI 2026-11-23 (LEDGER SP-9): the linear-probe half only, kept as its comparison passes 3 sigma
    # (scratchpad/part0/jc1_class_auc_check.md); the fine-tuned half is in tab:jc1_class_aucfull (the data release)
    ("tab:jc1_class_auc", APPENDIX, "With linear probes, every 17-class network recalls several JetClass classes less "
                                    "often than 188 classes, by more than three standard deviations, and 162 and 43 "
                                    "classes none."),
    # the data release (PI 2026-11-23, LEDGER SP-9): the tables the main text and the appendix held before; tab:randparts
    # comes back to the appendix only if a sentence states something about one partition (SP-9)
    ("tab:xbx", SUPPLEMENT, XBX),
    # PI 2026-11-19 (E): each random partition on its own row
    ("tab:randparts", SUPPLEMENT, "Each random partition's linear-probe and fine-tuned values are in [[tab:randparts]]."),
    # PI 2026-11-20 (meeting, B): from the supplement into the appendix
    ("tab:perclass", SUPPLEMENT, "Every JetClass-II class's recall in [[PC-jc2]], as a linear probe and fine-tuned, is "
                                 "in [[tab:perclass]]."),
    # L3 (run SDs, no bootstrap): 162, 43 and 17 against 188 within 2.2 standard deviations on every class; the class
    # AUCs of 17 classes are not (PC-jc1)
    ("tab:jetclassrej", SUPPLEMENT, "Fine-tuned on JetClass, 162, 43 and 17 classes reject QCD jets as 188 classes do "
                                  "in every class, within three standard deviations."),
    # PI 2026-11-20 (meeting, B): from the supplement into the appendix
    *((k, SUPPLEMENT, "Every network fine-tuned on JetClass, its QCD jets passing and the JetClass linear probes are "
                      "in [[tab:jetclassall]], [[tab:jetclassnbkg]] and [[tab:jetclassprobe]].")
      for k in ("tab:jetclassall", "tab:jetclassnbkg", "tab:jetclassprobe")),
    ("tab:pairs", SUPPLEMENT, "[[tab:pairs]] gives every class-pair separation of [[F4-mech]] to [[F4-light-ft]] as the "
                            "rejection at 90% signal efficiency, as a linear probe and fine-tuned, with a lower bound "
                            "where too few background jets pass to measure it."),
    # the training-size points below Table 1's (plan "tables": restored)
    ("tab:probesn", SUPPLEMENT, "[[tab:probesn]] gives every point of [[A10]] and [[A10-part]] below the training-set "
                              "sizes of [[tab:summary]]."),
    ("tab:ft", SUPPLEMENT, "[[tab:ft]] gives every point of [[A8]] and [[A8-part]] below the training-set sizes of "
                         "[[tab:summary]]."),
    # PI 2026-11-13 (plan "tables"): what Table 1 and tab:summaryauc lack of the top and quark/gluon probes
    ("tab:probes", SUPPLEMENT, "[[tab:probes]] gives the accuracy and the rejection at 50% signal efficiency of the top "
                             "and quark/gluon linear probes of [[ROC-main]], with the released Sophon checkpoint's."),
    ("tab:labelsets", SUPPLEMENT, "[[tab:labelsets]] gives the accuracy and the rejection at 50% signal efficiency of the "
                                "fine-tuned top and quark/gluon networks of [[ROC-main-ft]]."),
    # PI 2026-11-23 (LEDGER SP-13, step 4): the top and quark/gluon AUC columns that left tab:summaryauc
    ("tab:summaryaucfull", SUPPLEMENT, "[[tab:summaryaucfull]] repeats [[tab:summaryauc]] with the area under the ROC "
                                       "curve on top and quark/gluon tagging."),
    # PI 2026-11-20 (meeting, B): from the supplement into the appendix
    ("tab:massdetail", SUPPLEMENT, "The mass values of [[tab:mass]] per jet group and per bin are in [[tab:massdetail]]."),
    ("tab:sculpt", SUPPLEMENT, "[[tab:sculpt]] gives the sculpting of [[F10b]] at 10% and 1% QCD efficiency and at 50% "
                             "signal efficiency, beside the random-selection floor."),
    ("tab:xbxrej", SUPPLEMENT, "[[tab:xbxrej]] gives the QCD rejections at 50% and 30% signal efficiency for the networks "
                             "of [[tab:xbx]]."),
    ("tab:computefull", SUPPLEMENT, "Each run's GPU types and GPU-hours are in [[tab:computefull]]."),
    ("tab:mlpfull", SUPPLEMENT, "Every random partition of [[tab:mlp]] is in [[tab:mlpfull]]."),
    ("tab:perrun", SUPPLEMENT, "Every run's value is in [[tab:perrun]]."),
    # PI 2026-11-14 (step 5): to the supplement, after the mass tables
    ("tab:bn", SUPPLEMENT, "Linear probes with the BatchNorm statistics recomputed are in [[tab:bn]]."),
    ("tab:probesnfull", SUPPLEMENT, "Every training-set size and random partition of [[tab:probesn]], with the area "
                                    "under the ROC curve of [[A10]], is in [[tab:probesnfull]]."),
    ("tab:ftfull", SUPPLEMENT, "Every training-set size and random partition of [[tab:ft]], with the AUC points of [[A8]], "
                               "[[F8]] and [[F8-auc]], is in [[tab:ftfull]]."),
    ("tab:jc1conf", SUPPLEMENT, "The percentages of [[PC-conf]] to [[PC-conf-mass]] are in [[tab:jc1conf]]."),
    # L13: no sentence cites them, so the supplement (PI 2026-11-13: delete nothing; the pooled random partitions)
    ("tab:randprong", SUPPLEMENT, "The prong counts that each random partition mixes are in [[tab:randprong]]."),
    *((k, SUPPLEMENT, "Every random partition, the ratio to 188 classes and every block of [[tab:probes]], "
                      "[[tab:fill]], [[tab:pairs]] and [[tab:coarse]] are in [[tab:probesfull]] to [[tab:fillfull]].")
      for k in ("tab:probesfull", "tab:coarsefull", "tab:pairsfull", "tab:fillfull")),
    # the network rule (PI 2026-11-14): the partitions of the appendix tables of these families, whole copies
    ("tab:jc1_class_aucfull", SUPPLEMENT, "Every point of [[PC-jc1-part]] and [[PC-jc1-rand]] is in "
                                         "[[tab:jc1_class_aucfull]], with the rows of [[tab:jc1_class_auc]] and their "
                                         "fine-tuned recall."),
    ("tab:labelsetsfull", SUPPLEMENT, "The fine-tuned top and quark/gluon networks of [[tab:labelsets]] with the "
                                     "17-class partitions are in [[tab:labelsetsfull]]."),
    ("tab:sculptfull", SUPPLEMENT, "The sculpting of [[tab:sculpt]] with the 17-class partitions is in "
                                  "[[tab:sculptfull]]."),
    # tab:qcdhf repeated every row of this table (audit 3): this copy stays, with each random partition
    ("tab:qcdhffull", SUPPLEMENT, "The linear probes' X→bb̄ discriminant on QCD jets with b quarks is in "
                                  "[[tab:qcdhffull]]."),
)
# what a sentence rests on beyond its own item
EVIDENCE = {
    # Table 1 holds the headline metric of every network of the headline family, tab:summaryauc the AUC beside it
    # PI 2026-11-23 (LEDGER SP-9): a paper item's stated differences are in the tested-differences table
    "F2": ("tab:summary", "tab:tests"), "F2-ft": ("tab:summary", "tab:tests"), "F2a": ("tab:summaryauc", "tab:summary"),
    "F2b": ("tab:summaryauc", "tab:summary"),
    "PC-jc2": ("tab:summary", "tab:summaryauc", "tab:perclass"),      # the recalls averaged with the test jets of each class are the accuracy (JetClass-II: 11 QCD classes hold fewer than 5,000 jets); every point
    "ROC-main": ("tab:probes", "tab:summary"), "ROC-main-ft": ("tab:jetclassrej", "tab:labelsets", "tab:summary"),
    "PC-jc1": ("tab:jc1_class_auc",),
    # the training-size points: below the largest training set in tab:ft and tab:probesn, at it in Table 1; every size,
    # partition and AUC in the supplement's copies
    "F4-mech": ("tab:tests", "tab:pairs"), "F8": ("tab:ftfull", "tab:probesnfull"), "F8-auc": ("tab:ftfull", "tab:probesnfull"),
    "A8": ("tab:summary", "tab:tests", "tab:ft", "tab:ftfull"), "A8-part": ("tab:ft", "tab:summary"),
    "F9b": ("tab:mass", "tab:tests", "tab:massdetail"),
    # "PT": its values are the data release's jet_pt_bins.csv, no table (L18)
    "F4-other": ("tab:tests", "tab:pairs"), "F4-mech-ft": ("tab:pairs",), "F4-other-ft": ("tab:pairs",),
    "F4-ee": ("tab:pairs",), "F4-light-ft": ("tab:pairs",),
    **{k: ("tab:pairsfull",) for k in ("F4-rand", "F4-rand-qcd", "F4-rand-ft", "F4-rand-qcd-ft")},
    # R30 in Table 1, the AUC in tab:summaryauc
    # one sentence for both modes of the ROC family (L20 splits it a mode a figure), so one list of tables
    **{k: ("tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets") for k in ("F6", "F6-ft", "F6-part",
                                                                                    "F6-part-ft")},
    "A10": ("tab:summary", "tab:tests", "tab:probesn", "tab:probesnfull"), "A10-part": ("tab:probesn", "tab:summary"),
    "F9a": ("tab:massdetail",),
    "F11": ("tab:summary", "tab:tests", "tab:xbx"), "F11-part": ("tab:summary", "tab:xbx"), "F11-probe": ("tab:xbx",),
    **{k: ("tab:mass", "tab:massdetail") for k in ("A17-probe", "A17-ridge")},
    **{k: ("tab:massdetail",) for k in ("A17-probe-part", "A17-ridge-part")}, "A17-probe": ("tab:mass",),
    **{k: ("tab:sculpt", "tab:tests") for k in ("F10b", "F10b-flav", "F10c", "F10a-flav")},
    "tab:mass": ("tab:tests", "tab:massdetail"), "tab:summary": ("tab:tests",),
    "tab:labelsets": ("tab:summary", "tab:summaryauc"), "tab:probes": ("tab:summary", "tab:summaryauc"),
    "tab:partrepro": ("tab:published",),      # ParT's published row and our from-scratch row
}
TIER = {k: t for k, t, _ in _FIGURES + _TABLES}
# the preview book (scripts/build_render_job.py --book --preview, 2026-11-18): today's partial rerun results, so every
# placed item is PROVISIONAL (a figure not drawn keeps its number in a box, a table shows its pending cells); never set
# for the paper build
PREVIEW = os.environ.get("MLST_FT_PREVIEW") == "1"
if PREVIEW:
    PROVISIONAL = {**{k: "preview: waits for the rerun results" for k, t in TIER.items() if t != SUPPLEMENT}, **PROVISIONAL}

# PI 2026-11-14 (step 6): the appendix ordered by family, figures and tables together, each family under a one-line
# heading; within a family linear probes before fine-tuned networks, each table beside the figures it holds.
# _FIGURES and _TABLES list the appendix in this order (checked below), so number() and the book agree.
# PI 2026-11-23 (LEDGER SP-9): the appendix of the approved structure, in the order of its lists: what someone needs to
# reproduce the work (R3a), then the evidence for the claims the main text states but does not show (R3b)
APPENDIX_FAMILIES = (
    ("Training details", ("tab:app-vocabulary", "VAL", "tab:inputs", "tab:lrsweep", "tab:compute", "tab:fill")),
    ("Checks of the method", ("tab:mlp", "tab:partrepro")),
    ("Headline metrics", ("tab:summaryauc", "tab:tests", "tab:coarse")),
    ("Training size", ("A10",)),
    ("Class pairs", ("F4-other",)),
    ("Per class", ("tab:jc1_class_auc",)),
    ("Mass", ("F9b",)),
)
# the families of the coverage map (PI 2026-11-14, step 6), one row each whether or not the appendix holds one of their
# items
FAMILY_NAMES = ("Headline metrics", "Per class", "Class pairs", "Training size", "ROC curves", "Jet pT", "Mass",
                "Sculpting", "Resonance tagging", "Embeddings", "Training details")
_app = [k for _, ks in APPENDIX_FAMILIES for k in ks]
for _kind, _items in (("figures", _FIGURES), ("tables", _TABLES)):
    _have = [k for k, t, _ in _items if t == APPENDIX]
    if _have != [k for k in _app if k.startswith("tab:") == (_kind == "tables")]:
        raise SystemExit(f"FATAL: the appendix {_kind} are not in the order of APPENDIX_FAMILIES: {_have}")

# PI 2026-11-14 (step 6): one rule for which networks a figure shows, stated once (on the coverage map's page) and
# followed by every figure of the paper. Groups: the label sets (188, 162, 43, 17 classes), the mass networks
# (162 and 17 classes + mass), the 17-class partitions.
NETWORK_RULE = ("Every figure and table of the paper shows the label sets (188, 162, 43 and 17 classes) and the mass "
                "networks (162 and 17 classes + mass); the 17-class partitions appear in the paper for the headline "
                "metrics, class pairs and resonance tagging, and in the supplementary material for every other "
                "family; the per-class and embedding figures show the label sets only.")
PART_IN_PAPER = ("Headline metrics", "Class pairs", "Resonance tagging")
SETS_ONLY = ("Per class", "Embeddings")
LP, FT = "lp", "ft"
GROUPS = ("sets", "mass", "part")
# what each paper item shows: (family, modes, groups). The "lp" column is the frozen network: a linear probe, or the
# network's own output used directly (F9b, F10b (a)) or read by a classifier on frozen features (F11, A17-probe).
# F1 and the tables of inputs and of the published comparison's ParT reproduction show no group.
SHOWS = {
    "F2": ("Headline metrics", (LP,), GROUPS), "F2-ft": ("Headline metrics", (FT,), GROUPS),
    "tab:summary": ("Headline metrics", (LP, FT), GROUPS), "tab:summaryauc": ("Headline metrics", (LP, FT), GROUPS),
    "tab:published": ("Headline metrics", (FT,), ("sets",)), "tab:mlp": ("Headline metrics", (LP,), GROUPS),
    "tab:coarse": ("Headline metrics", (LP,), GROUPS),
    "PC-jc2": ("Per class", (LP, FT), ("sets",)), "tab:jc1_class_auc": ("Per class", (LP,), GROUPS),
    **{k: ("Class pairs", (LP,), GROUPS) for k in ("F4-mech", "F4-other")},
    "A8": ("Training size", (FT,), ("sets", "mass")), "A10": ("Training size", (LP,), ("sets", "mass")),
    "tab:mass": ("Mass", (LP,), ("sets", "mass")), "F9b": ("Mass", (LP,), ("mass",)),
    "F11": ("Resonance tagging", (LP,), ("sets", "mass")),
    "VAL": ("Training details", (LP, FT), ("sets", "mass")),       # pretraining, before either mode
    "tab:fill": ("Training details", (LP,), ("sets", "mass")), "tab:lrsweep": ("Training details", (FT,), ("sets",)),
}
# the cells no network of the group can fill, with the reason (PI: "does not apply" with the reason, no cell blank)
DOES_NOT_APPLY = {
    ("Mass", FT, "sets"): "no network is fine-tuned to regress the jet mass",
    ("Mass", FT, "mass"): "no network is fine-tuned to regress the jet mass",
    ("Mass", FT, "part"): "no network is fine-tuned to regress the jet mass",
    ("Resonance tagging", FT, "sets"): "the pretrained networks are not fine-tuned on these samples",
    ("Resonance tagging", FT, "mass"): "the pretrained networks are not fine-tuned on these samples",
    ("Resonance tagging", FT, "part"): "the pretrained networks are not fine-tuned on these samples",
    ("Embeddings", LP, "part"): "the embeddings compare the label sets",
    ("Embeddings", FT, "part"): "the embeddings compare the label sets",
}
GROUP_NAMES = {"sets": "label sets", "mass": "mass networks", "part": "partitions"}
MODE_NAMES = {LP: "linear probe", FT: "fine-tuned"}


def _numbers(keys) -> str:
    """"Figs. 2, A1; Tables 1, A1": the numbers of keys, figures then tables, each in paper order."""
    out = []
    for word, table in (("Fig", False), ("Table", True)):
        items = [k for k, _, _ in (_TABLES if table else _FIGURES) if k in keys]
        if items:
            nums = [number(k).split()[-1] for k in items]
            out.append(f"{word}{'s' if len(nums) > 1 else ''}{'.' if word == 'Fig' else ''} {', '.join(nums)}")
    return "; ".join(out)


def coverage() -> list:
    """The coverage map (PI 2026-11-14, step 6): one row per family, (family, [cell per (mode, group)]) in the order
    (lp, sets), (lp, mass), (lp, part), (ft, sets), (ft, mass), (ft, part); a cell is the numbers of the paper's
    figures and tables that show the group in that mode, else "does not apply (<reason>)", else "supplement"."""
    rows = []
    for family in FAMILY_NAMES:
        cells = []
        for mode in (LP, FT):
            for group in GROUPS:
                items = [k for k, (f, modes, groups) in SHOWS.items() if f == family and mode in modes and group in groups
                         and TIER[k] != SUPPLEMENT]
                if items:
                    cells.append(_numbers(items))
                elif (family, mode, group) in DOES_NOT_APPLY:
                    cells.append(f"does not apply ({DOES_NOT_APPLY[family, mode, group]})")
                else:
                    cells.append("supplement")
        rows.append((family, cells))
    return rows
# the PI's book numbers (2026-11-08 message) -> gallery ids
BOOK_NAMES = (("F9", "F8"), ("F16", "A8"), ("F15", "A10"), ("F5", "FJ"), ("F7", "PT"), ("F8", "F6"), ("F17", "F6 (upper row)"),
              ("F10", "F9a"), ("F12", "F10a"), ("F14", "VAL"), ("F6, F18-F20", "FJ-ctrl, FJ-app, FJ-ctrl-app, FJ-s"),
              ("F21", "PC-conf"), ("F28, F29", "A3R, A3R-ft"), ("F23b", "PC-jc2 (c)"), ("F26c", "PC-jc2 (f)"),
              ("UMAP figure", "UMAP-jc2-had"))
DROPPED = ("F3", "F4", "F4-app", "F3-scatter")      # F3-scatter: its panels are F3-main (a) and F3-app (b)
# the per-class pages the family replaced (PI 2026-11-12/13, plan "existing_items"): old id -> the page that holds its
# panels now; out of the gallery and the paper, their renders kept (F3-app3 compared the mass networks with 188 classes,
# the family compares each with its own label set)
SUPERSEDED = {"F3-main": "PC-jc2", "F3-app": "PC-jc2", "F3-ft": "PC-jc2", "F3-app2": "PC-jc2-part",
              "F3-ft2": "PC-jc2-part", "F3-app3": "PC-jc2-mass", "CONF": "PC-conf",
              # the ROC family's two-class form (PI 2026-11-13): the probes' ROC page is the upper row of F6
              "F6-app": "F6",
              "A8-auc": "A8", "A10-auc": "A10",      # PI 2026-11-18: the AUC is the second row of A8 and A10
              "F4-light": "F4-other"}                # PI 2026-11-23 (LEDGER SP-9, merge 8): its pairs are F4-other's panels
DROPPED += tuple(SUPERSEDED)
# mlst_results tables the plan places nowhere: nothing cites them, so they are dropped (PI 2026-11-08); the plan's
# "missing inputs (fill)" is tab:fill and its "fine-tuning vs size" tab:ft, both placed above
DROPPED_TABLES = ("tab:balance", "tab:fill_main", "tab:ftrej",
                  "tab:boot",     # tab:boot: PI 2026-11-12, its probe spreads disagree with tab:probes
                  # PI 2026-11-12: their rows are in tab:published and tab:labelsets
                  "tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig",
                  "tab:qcdhf",    # audit 3, 2026-11-14: every row is in tab:qcdhffull
                  # PI 2026-11-23 (LEDGER SP-9): merged (tab:app-levels into the class map, tab:sculptboot into
                  # tab:tests) or dropped as superseded (tab:usecase, tab:masseffect); their files stay generated, so the
                  # data release keeps them as tables placed nowhere
                  "tab:app-levels", "tab:sculptboot", "tab:usecase", "tab:masseffect")
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
