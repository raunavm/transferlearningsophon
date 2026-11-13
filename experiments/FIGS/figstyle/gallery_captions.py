"""The captions of the figures placed in the paper (PI brief 2026-11-12): the finding where one is resolved, what is
plotted, how to read it, the uncertainty and run counts, and the table that holds the values; two to five plain
sentences. They replace the Page captions of those figures (apply, called at the end of each gallery module). A table
is named by its label in braces and printed with its number of the day (gallery_placement.number)."""
import dataclasses
import re

import gallery_placement as G

CAPTIONS = {
    "F1": (
        "Panel (a) shows six JetClass-II classes as schematic jets (tracks as lines, calorimeter deposits as grey "
        "cells) above their fine (188-class) and coarse (17-class) labels, and below, each row a label set and each bar"
        " a merged class. Panel (b) shows the network, the Particle Transformer architecture with Sophon's output head,"
        " pretrained from scratch for each label set on the same jets in the same order. Each pretrained network is "
        "used frozen (open squares) or fine-tuned (filled) on five tasks, JetClass-II, JetClass, top tagging, "
        "quark/gluon tagging, and X→bb̄ and X→bs resonance tagging, the last frozen only. A label set has three "
        "pretraining runs, and the networks with an added mass regression (+ mass) and the 17-class partitions of later"
        " figures have two, except the five random partitions with one each (table {tab:summary})."
    ),
    "F2": (
        'With linear probes, 43 and 17 classes and every 17-class partition fall below 188 classes in JetClass-II '
        'accuracy. On top and quark/gluon tagging, Pythia and Herwig, no pretrained network differs from 188 classes '
        'by more than three standard deviations. Points are means ±1 standard deviation over the runs of figure '
        '{fig:F1} (random partitions pooled) or three runs of the untrained random network, the linear-probe '
        'reference, beyond the // breaks in (a) and (b). R30 is the inverse background efficiency at 30% signal '
        'efficiency, and quark/gluon tagging trains on Pythia jets. Values are in table {tab:summary}, figure '
        '{fig:F2-ft} shows the fine-tuned networks and figure {fig:F2a} the area under the receiver operating '
        'characteristic curve.'
    ),
    "F2-ft": (
        'After fine-tuning, 17 classes stay below 188 classes on JetClass-II and make about 3% more errors on '
        'JetClass. On top and quark/gluon tagging, Pythia and Herwig, no pretrained network differs from 188 classes '
        'by more than three standard deviations. Points are means ±1 standard deviation over the runs of figure '
        '{fig:F1} (random partitions pooled) or three trainings from scratch, the fine-tuned reference. R30 is the '
        'inverse background efficiency at 30% signal efficiency, and quark/gluon tagging trains on Pythia jets. '
        'Values are in table {tab:summary}, figure {fig:F2} shows the linear probes and figure {fig:F2b} the area '
        'under the receiver operating characteristic curve.'
    ),
    "F2a": (
        'Linear probes on JetClass-II separate the classes less well for 43 and 17 classes than for 188 classes in '
        'the area under the receiver operating characteristic curve (AUC), as in accuracy, and the AUC also places '
        '162 classes below 188, by about five standard deviations, within the markers. The panels show the AUC of '
        "figure {fig:F2}'s linear probes, on JetClass-II and JetClass averaged over the classes, each against the "
        'rest, with the random network beyond the // breaks in (a) and (b). Tasks, networks and run counts are '
        'those of figure {fig:F2}, and the line and band repeat the 188-class values. The released Sophon checkpoint, '
        'one network, appears on JetClass-II and JetClass only, as its top and quark/gluon probes set the missing '
        'inputs to zero. Values are in tables {tab:summary} and {tab:summaryauc}.'
    ),
    "F2b": (
        'After fine-tuning, every pretrained network is more accurate than training from scratch on JetClass-II and '
        'JetClass, in accuracy (figure {fig:F2-ft}) and in the area under the receiver operating characteristic curve '
        "(AUC). The panels show the AUC of figure {fig:F2-ft}'s fine-tuned networks, on JetClass-II and JetClass "
        'averaged over the classes, each against the rest. Tasks, networks and run counts are those of figure '
        '{fig:F2-ft}, the line and band repeat the 188-class values, and from scratch lies beyond an axis break (//) '
        'in (b) and (d). The random network has no fine-tuned version, '
        'since fine-tuning it is training from scratch, and the released Sophon checkpoint, not fine-tuned here, is '
        'in figure {fig:F2a}. Values are in tables {tab:summary} and {tab:summaryauc}.'
    ),
    "PC-jc2": (
        'Pretraining with fewer classes lowers the area under the receiver operating characteristic curve (AUC) of '
        'JetClass-II classes and raises none by more than three standard deviations over runs. It lowers 25 classes '
        '(24 QCD) as linear probes and 1 fine-tuned with 162 classes, 174 and 46 with 43, and 184 and 161 with 17. '
        "Each point is one class's AUC against all others for the panel's network against 188 classes, the mean over "
        'three pretraining runs, as linear probes (left column) or fine-tuned (right column) on one million '
        'class-balanced jets. Points below the dotted diagonal lose separation, colors mark decay families of X, and '
        'fine-tuning ran at most 40 epochs (160 for the first 17-class run). Values are in table {tab:perclass}, and '
        'figures {fig:PC-jc1}, {fig:PC-jc2-part}, {fig:PC-jc2-rand} and {fig:PC-jc2-mass} show JetClass, partitions '
        'and mass networks.'
    ),
    "PC-jc1": (
        'With linear probes, the network pretrained with 17 classes separates H→cc̄ jets visibly less well than 188 '
        'classes, while 162 and 43 classes agree with 188 classes on every class. After fine-tuning every difference '
        'is below 0.003, too small to see, and 17 classes stay below 188 classes by more than four standard '
        "deviations over runs on four of the ten classes. Each point is one class's area under the receiver operating "
        "characteristic curve against the other nine, for the panel's network (vertical) against 188 classes "
        '(horizontal), as linear probes (left column) or fine-tuned (right column) on one million '
        'JetClass jets. Points are means over three pretraining runs, whose standard deviations, at most 0.003, are '
        'not drawn. Values, also for the mass networks and partitions, are in table {tab:jc1_class_auc}.'
    ),
    "F4-mech": (
        'Linear probes on 17 classes separate decays differing only in quark flavor worse than 188 classes. The b-quark'
        ' boundary recovers part of the loss on both pairs and the random flavor boundary on X→bc vs bq, cs, while in '
        '(c) its rise and the dip of no flavor boundary stay within three standard deviations in the area under the '
        'curve. Panels show background rejection against signal efficiency for the label sets and mass networks (upper '
        'row, 43 classes under 188 in (a)) and the partitions (lower row), means over runs of figure {fig:F1} down to '
        '20 background jets. Lower panels give ratios to 188 classes, banded for 188 classes only by run spread and '
        'binomial uncertainty. Figures {fig:F4-mech-ft} and {fig:F4-rand} show the fine-tuned networks and random '
        'partitions, and table {tab:pairs} the values and other spreads.'
    ),
    "A8": (
        'After fine-tuning, 17 classes are less accurate than 188 classes on JetClass-II at every training-set size and'
        ' on JetClass at one million jets. On top and quark/gluon tagging, Pythia and Herwig, no label set or mass '
        'network differs from 188 classes by more than three standard deviations at any size. Panels show accuracy on '
        'class-balanced JetClass-II and JetClass (from 10,000 jets) and R30, the inverse background efficiency at 30% '
        'signal efficiency, against training-set size. Error bars are ±1 standard deviation over the runs of figure '
        '{fig:F1} or three trainings from scratch (grey), with points offset slightly, and open stars mark the '
        'published Particle Transformer, trained from scratch on 2, 1.2 and 1.6 million jets. Figures {fig:A10} and '
        '{fig:A8-part} show the linear probes and partitions, and values are in tables {tab:ft} and {tab:summary}.'
    ),
    "F9b": (
        'Each network with an added mass regression (+ mass) has a better resolution than the soft-drop mass in all 15 '
        'bins and less bias in 13, each by more than four standard deviations over runs, and the two + mass networks '
        'agree within two standard deviations in every bin. Panels (a) and (b) show the median of r = m/m_gen, mass '
        'output or soft-drop mass over generator-level soft-drop mass, against jet pT and generator mass, and (c) and '
        '(d) the resolution, half the central 68% range of r over its median. Lower panels give the ratio to the '
        'soft-drop value, and error bars are ±1 standard deviation over two pretraining runs, points offset slightly '
        'within each bin. Values are in tables {tab:mass} and {tab:massdetail}, and figures {fig:A17-probe} and '
        '{fig:A17-ridge} show frozen-feature probes.'
    ),
    "F11": (
        'At 1% QCD efficiency, 17 classes keep fewer X→bb̄ and X→bs jets than 188 classes, a gap of 0.6 and 1.0 in the '
        'figure of merit that widens below 0.1%. The figure of merit, signal efficiency over the square root of QCD '
        'efficiency, is shown for (a) X→bb̄ and (b) X→bs against QCD jets with soft-drop mass 150 to 230 GeV. The '
        'pretrained networks are not fine-tuned, solid curves being a multilayer-perceptron classifier on frozen '
        'features, grey curves training from scratch on each signal, and the near-coincident dash-dotted curves the '
        'X→bb̄ output against the QCD outputs (188 and 162 classes only). Bands, for 188 classes and from scratch only,'
        ' are ±1 standard deviation over the runs of figure {fig:F1} or three trainings. Table {tab:xbx} holds the '
        'values and other spreads, and figure {fig:F11-part} the partitions.'
    ),
    "F11-part": (
        "At 1% QCD efficiency, the partitions with no, a b-quark and a random flavor boundary each keep fewer X→bb̄ and"
        " X→bs jets than the network pretrained with 188 classes, and no 17-class partition differs from the 17-class "
        "network by more than three standard deviations (table {tab:xbx}). The panels show figure {fig:F11}'s figure of"
        " merit, signal efficiency over the square root of QCD efficiency, against QCD efficiency, for a "
        "multilayer-perceptron classifier on the frozen features of each partition and of 188 and 17 classes, the other"
        " networks being in figure {fig:F11}. Lines are means over two pretraining runs per flavor-boundary partition, "
        "three for 188 and 17 classes and the five random partitions, and the band, for 188 classes only, is ±1 "
        "standard deviation. Values and the other standard deviations are in table {tab:xbx}."
    ),
    "F4-other": (
        "With linear probes, 43 and 17 classes separate both pairs less well than 188 classes, and 162 classes, which "
        "merge the 27 QCD classes, separate the QCD pair (b) less well. Panels show background rejection against signal"
        " efficiency, the first-named decay the signal, for the label sets and mass networks (upper row) and the "
        "flavor-boundary partitions with 188 and 17 classes (lower row), as means over the runs of figure {fig:F1} "
        "ending where fewer than 20 background jets remain. The lower panels give the ratio to 188 classes, on a range "
        "set per row, with a band of run spread and binomial uncertainty for 188 classes only. Figure {fig:F4-light} "
        "shows the other two pairs, figures {fig:F4-rand} and {fig:F4-rand-qcd} the random partitions, and table "
        "{tab:pairs} each pair's AUC and the other spreads."
    ),
    "F4-light": (
        "With linear probes, 43 and 17 classes separate QCD jets with b quarks, and those with c but no b quarks, from "
        "light QCD jets less well than 188 classes, and so do 162 classes, which merge the 27 QCD classes. The panels "
        "are those of figure {fig:F4-other} with the jets with heavy quarks as the signal, the label sets and mass "
        "networks in the upper row and the flavor-boundary partitions with 188 and 17 classes in the lower row. Curves "
        "are means over the runs of figure {fig:F1} ending where fewer than 20 background jets remain, and the ratio "
        "panels have a band of run spread and binomial uncertainty for 188 classes only. Figure {fig:F4-rand-qcd} shows"
        " the random partitions, and table {tab:pairs} each pair's AUC and the other spreads."
    ),
    "F4-other-ft": (
        "After fine-tuning, 17 classes still separate both pairs worse than 188 classes, by less than with linear "
        "probes (figure {fig:F4-other}). The panels are those of figure {fig:F4-other} for networks fine-tuned on one "
        "million class-balanced JetClass-II jets and scored on natural-composition test jets, with training from "
        "scratch in the upper row. Curves end where fewer than 20 background jets remain, and bands, for 188 classes "
        "and from scratch only, combine the spread over the runs of figure {fig:F1} or three trainings from scratch "
        "with the binomial uncertainty. Figure {fig:F4-light-ft} shows the pairs of QCD jets with heavy quarks against "
        "light ones, figures {fig:F4-rand-ft} and {fig:F4-rand-qcd-ft} the random partitions and table {tab:pairs} each"
        " pair's AUC and the other spreads."
    ),
    "F4-light-ft": (
        "After fine-tuning, 17 classes still separate the pair of (b) worse than 188 classes, by less than with linear "
        "probes (figure {fig:F4-light}). Their dip in (a) near 90% signal efficiency is not claimed, as the AUC lies "
        "only 3.4 standard deviations over runs below, short of the four required without a test-jet term. The panels "
        "are those of figure {fig:F4-light} for networks fine-tuned on one million class-balanced JetClass-II jets and "
        "scored on natural-composition test jets, with training from scratch in the upper row. Curves end where fewer "
        "than 20 background jets remain, and bands, for 188 classes and from scratch only, combine the spread over the "
        "runs of figure {fig:F1} or three trainings from scratch with the binomial uncertainty. Figure "
        "{fig:F4-rand-qcd-ft} shows the random partitions and table {tab:pairs} each pair's AUC and the other spreads."
    ),
    "F4-mech-ft": (
        "After fine-tuning, 17 classes still separate X→bb̄ from cc̄ and X→bc from bq, cs worse than 188 classes, by "
        "less than with linear probes (figure {fig:F4-mech}). The panels are those of figure {fig:F4-mech} for networks"
        " fine-tuned on one million class-balanced JetClass-II jets and scored on natural-composition test jets, with "
        "training from scratch in the upper row and 17 classes in the lower row for reference. The ratio panels have a "
        "logarithmic axis, and curves end where fewer than 20 background jets remain. Bands, for 188 classes and from "
        "scratch only, combine the spread over the runs of figure {fig:F1} or three trainings from scratch with the "
        "binomial uncertainty. Figure {fig:F4-rand-ft} shows the random partitions, and table {tab:pairs} each pair's "
        "AUC and the other spreads."
    ),
    "PT": (
        'In every jet pT bin, linear probes with 43 and 17 classes and 17 classes fine-tuned are less accurate on '
        'JetClass-II than 188 classes, by more than three standard deviations of the run and test-jet spread combined. '
        'On JetClass, 17 classes do so only in the two lowest bins (500–700 GeV) fine-tuned and the highest (900–1000 '
        'GeV) as a linear probe. Panels show per-bin accuracy of linear probes (upper row) and fine-tuned networks '
        '(lower row) for label sets and mass networks on JetClass-II and JetClass only, as top and quark/gluon jet pT '
        'span about 50 to 100 GeV. Points, offset slightly within each bin, are means with ±1 standard deviation error '
        'bars over runs of figure {fig:F1} (test-jet part not drawn). Figure {fig:PT-part} shows the partitions, and '
        'values are in the supplementary file jet_pt_bins.csv.'
    ),
    "F6": (
        'With linear probes, no label set or mass network differs from 188 classes by more than three standard '
        'deviations in R30, the inverse background efficiency at 30% signal efficiency, on top and quark/gluon tagging,'
        ' Pythia and Herwig, the highest curve in (a), 17 classes + mass, lying about two above. Panels show background'
        ' rejection against signal efficiency, means over the runs of figure {fig:F1} down to 20 passing background '
        'jets, and the curves in (b) and (c) overlap. The lower panels give the ratio to 188 classes from a signal '
        'efficiency of 0.1, with a band of run spread and binomial uncertainty for 188 classes only. Figure {fig:F6-ft}'
        ' shows the fine-tuned networks and figure {fig:F6-part} the partitions, and values and the other standard '
        'deviations are in tables {tab:summary}, {tab:summaryauc} and {tab:probes}.'
    ),
    "F6-ft": (
        "After fine-tuning, no label set or mass network differs from 188 classes in R30, the inverse background "
        "efficiency at 30% signal efficiency, by more than three standard deviations on top and quark/gluon tagging, "
        "Pythia and Herwig. Panels show background rejection against signal efficiency for the networks of figure "
        "{fig:F6} fine-tuned and for training from scratch, means over runs of figure {fig:F1} or three trainings from "
        "scratch, down to 20 passing background jets, curves in (b) and (c) overlapping. Lower panels give the ratio to"
        " 188 classes from a signal efficiency of 0.1, with bands for the run spread and binomial uncertainty of 188 "
        "classes and from scratch only, the other networks' standard deviations being in the tables. Figure "
        "{fig:F6-part-ft} shows the partitions, and values are in tables {tab:summary}, {tab:summaryauc} and "
        "{tab:labelsets}."
    ),
    "A10": (
        'With linear probes, 43 and 17 classes are less accurate on JetClass-II than 188 classes at every training-set '
        'size. Panels show accuracy on class-balanced JetClass-II and JetClass and R30, the inverse background '
        'efficiency at 30% signal efficiency, on top and quark/gluon tagging, Pythia and Herwig, against training-set '
        'size, the 162- and 43-class and + mass curves lying under those of 188 and 17 classes on JetClass. Error bars '
        'are ±1 standard deviation over the runs of figure {fig:F1} or three initializations of the untrained random '
        'network (dark grey), with points offset slightly. Open stars in the upper corners of (b) to (d) mark the '
        'published Particle Transformer, trained from scratch on 2, 1.2 and 1.6 million jets, above every probe. Figure'
        ' {fig:A10-part} shows the partitions, and values are in tables {tab:probesn} and {tab:summary}.'
    ),
    "F10b": (
        'Used directly, no network pretrained with fewer classes or with an added mass regression (+ mass) distorts the'
        ' QCD mass distribution differently from 188 classes, nor a + mass network from its own label set, by more than'
        ' three standard deviations of run and test-jet spread at 10% or 1% QCD efficiency. Dash-dotted curves show the'
        ' Jensen-Shannon distance (base 2) between the soft-drop mass distributions of all JetClass-II QCD test jets '
        'and those passing a cut on the summed QCD output, against QCD efficiency eps_B. The band, for 188 classes '
        'only, is ±1 standard deviation over the runs of figure {fig:F1}, without the test-jet part, and the dotted '
        'line is a random selection of as many QCD jets. Values and other spreads are in tables {tab:sculpt} and '
        '{tab:sculptboot}, and figure {fig:F10b-flav} shows the flavor-boundary partitions.'
    ),
    "UMAP-jc2-had": (
        'The panels show two-dimensional embeddings, by uniform manifold approximation and projection (UMAP), of the '
        'class token, the 128-dimensional vector the classification head reads, for the networks pretrained with '
        '188, 162, 43 and 17 classes (rows), frozen (left) and fine-tuned on one million class-balanced JetClass-II '
        'jets (right). Every panel shows the same 2,000 test jets of each kind in the legend, QCD pooling its 27 '
        'classes, and as the 17-class set merges X→bb̄, X→cc̄ and X→qq̄, their partial separation in (g) comes from the '
        'features alone. All panels use pretraining run 1 and the same UMAP settings and seed. The grid is '
        'qualitative, and as each panel is fitted separately, positions cannot be compared between panels. Figures '
        '{fig:UMAP-jc1} to {fig:UMAP-mass-xbx} show the other tasks, X→bb̄ and X→bs against QCD, and the mass '
        'networks.'
    ),
    "VAL": (
        "The curves show the accuracy of each network's own classification output on the JetClass-II validation jets "
        'of pretraining against epoch, for 188, 162, 43 and 17 classes and for 162 and 17 classes with an added mass '
        'regression (+ mass, lighter shades), the only label sets given one, the 17 classes + mass curve lying on that '
        'of 17 classes. Each network is scored on its own classes, so the heights of the curves do not compare the '
        'label sets. The learning rate decays from epoch 57, where every curve rises slightly. Each curve is the mean '
        'over the runs of figure {fig:F1}, and the ±1 standard deviation bands over runs are too narrow to see. The '
        'values at the last epoch are in table {tab:compute}, and figure {fig:VAL-part} shows the 17-class partitions.'
    ),
}


def fill(raw: str) -> str:
    """raw with each {tab:...} and {fig:<page id>} as that item's number, without the word (the sentence says
    "table" or "figure")."""
    out = re.sub(r"\{(tab:[a-z_0-9]+)\}", lambda m: G.number(m.group(1)).removeprefix("Table "), raw)
    return re.sub(r"\{fig:([A-Za-z0-9-]+)\}", lambda m: G.number(m.group(1)).removeprefix("Fig. "), out)


def text(pid: str) -> str:
    """The caption of a main-text or appendix figure, filled (fill)."""
    return fill(CAPTIONS[pid])


def apply(pages: list) -> list:
    return [dataclasses.replace(p, caption=text(p.id)) if p.id in CAPTIONS else p for p in pages]
