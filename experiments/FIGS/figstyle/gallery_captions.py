"""The captions of the figures placed in the paper (PI brief 2026-11-12): the finding where one is resolved, what is
plotted, how to read it, the uncertainty and run counts, and the table that holds the values; two to five plain
sentences. They replace the Page captions of those figures (apply, called at the end of each gallery module). A table
is named by its label in braces and printed with its number of the day (gallery_placement.number). PI 2026-11-14
(step 8): a paper caption never names a supplement item by number (text() refuses one), only "the supplementary
material"; one plain finding sentence first; no sentence about a gap, an unclaimed result or an epoch cap.
{counts}: the computed per-class counts (gallery_perclass.counts_sentence, COUNTS)."""
import dataclasses
import re

import gallery_placement as G

# the mixed test sample's decay families (PI 2026-11-14), the fine-tuned class pairs' and sculpting's test jets
MIXED = "the mixed test sample (38.7% two-prong, 40.8% three- and four-prong, 20.5% QCD jets)"
# a figure without the random partitions in a class-pair figure (PI 2026-11-14): one clause why
NO_RAND = "Random partitions, which merge the pair's classes differently, are in the supplementary material."
COUNTS = {}       # PC-jc2, PC-jc1: the computed count sentence (gallery_perclass), "" without per-class values
# a sentence that depends on the data read, {name} in a caption (gallery_physics sets "probe_note": how the mass probe
# is centred, on the jet's true class in atlas_res, on one training median in the truth-free refit atlas_res_v3)
FILLS = {"probe_note": ""}

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
        'With linear probes, 43 and 17 classes and every 17-class partition are less accurate on JetClass-II than 188 '
        'classes, while on top and quark/gluon tagging, Pythia and Herwig, no pretrained network differs from 188 '
        'classes in R30 by more than three standard deviations. Points are means ±1 standard deviation over the runs '
        'of figure {fig:F1}, over the five random partitions, or over three initializations of the untrained random '
        'network, the linear-probe reference, beyond the // breaks in (a) and (b). The released Sophon checkpoint, one '
        'network, appears on JetClass-II and JetClass only. R30 is the inverse background efficiency at 30% signal '
        'efficiency, and quark/gluon tagging trains on Pythia jets. Values are in table {tab:summary}, figure '
        '{fig:F2-ft} shows the fine-tuned networks and table {tab:summaryauc} the area under the receiver '
        'operating characteristic curve.'
    ),
    "F2-ft": (
        'After fine-tuning, 17 classes stay below 188 classes on JetClass-II and make about 3% more errors on '
        'JetClass, while on top and quark/gluon tagging, Pythia and Herwig, no pretrained network differs from 188 '
        'classes in R30 by more than three standard deviations. Points are means ±1 standard deviation over the runs '
        'of figure {fig:F1}, over the five random partitions, or over three trainings from scratch, the fine-tuned '
        'reference. R30 is the inverse background efficiency at 30% signal efficiency, and quark/gluon tagging trains '
        'on Pythia jets. Values are in table {tab:summary}, figure {fig:F2} shows the linear probes and table '
        '{tab:summaryauc} the area under the receiver operating characteristic curve.'
    ),
    "PC-jc2": (
        "{counts} Each point is one JetClass-II class's recall, the fraction of its test jets the network assigns to "
        "it, for the panel's network (vertical) against 188 classes (horizontal), as linear probes (left) or "
        "fine-tuned (right) on one million class-balanced jets; rows are 162, 43 and 17 classes, and points below the "
        "dotted diagonal are classes the panel's network recalls less often. Averaged with each class's test jets, the "
        "recalls give the accuracy of figures {fig:F2} and {fig:F2-ft}. Points are means over three pretraining runs "
        "with bars of ±1 standard deviation over runs and test jets, colored by what the 17-class labels merge the "
        "class with: decays differing only in quark flavor, only in lepton flavor, both or neither, and the QCD "
        "classes. Values are in the supplementary material."
    ),
    "PC-jc1": (
        "{counts} Each point is one JetClass class's recall, the fraction of its test jets the network assigns to it, "
        "for the panel's network (vertical) against 188 classes (horizontal), rows as in figure {fig:PC-jc2}, as "
        "linear probes (left column) or fine-tuned (right column) on one million JetClass jets. Points below the "
        "dotted diagonal are classes the panel's network recalls less often. Points are means over three pretraining "
        "runs, with bars of ±1 standard deviation over runs and test jets. Values, also for the mass networks and "
        "partitions, are in table {tab:jc1_class_auc}."
    ),
    "F4-mech": (
        'Linear probes on 17 classes separate decays differing only in quark flavor worse than 188 classes; the b-quark '
        'boundary recovers part of the loss on both pairs, and the random flavor boundary on X→bc vs bq, cs. Panels '
        'show background rejection against signal efficiency for the label sets and mass networks (upper row) and the '
        'flavor-boundary partitions (lower row), means over the runs of figure {fig:F1} down to 20 background jets. '
        'Lower panels give ratios to 188 classes on a logarithmic axis, banded for 188 classes only by run spread and '
        'binomial uncertainty. ' + NO_RAND + ' Figure {fig:F4-mech-ft} shows the fine-tuned networks, and table '
        '{tab:pairs} the values and other spreads.'
    ),
    "ROC-main": (
        'Background rejection against signal efficiency with linear probes on frozen features, the mean over runs. '
        '(a)-(i) Every JetClass signal class against QCD jets, as the Particle Transformer reports JetClass; (j) top and '
        '(k) quark/gluon tagging (Pythia). Every panel shows the label sets and mass networks; only 188 classes carries a '
        'band of the standard deviation over runs. Panels (e) and (i), the leptonic decays, show signal efficiencies '
        'from 0.95. Curves end where fewer than 20 background jets remain. Figure '
        '{fig:ROC-main-ft} shows the fine-tuned networks, and tables {tab:probes} and {tab:summary} the values.'
    ),
    "ROC-main-ft": (
        'Background rejection against signal efficiency after fine-tuning, the mean over runs. (a)-(i) Every JetClass '
        'signal class against QCD jets, fine-tuned on one million JetClass jets, as the Particle Transformer reports '
        'JetClass; (j) top and (k) quark/gluon tagging (Pythia), fine-tuned on the full sets. Every panel shows the label '
        'sets and mass networks, with training from scratch (grey); only 188 classes and from scratch carry bands of the '
        'standard deviation over runs. Panels (e) and (i), the leptonic decays, show signal efficiencies from 0.95. '
        'Curves end where fewer than 20 background jets remain. Values are in tables '
        '{tab:jetclassrej}, {tab:labelsets} and {tab:summary}.'
    ),
    "A8": (
        'After fine-tuning, 17 classes are less accurate than 188 classes on JetClass-II at every training-set size and'
        ' on JetClass at one million jets, while on top and quark/gluon tagging, Pythia and Herwig, no label set or '
        'mass network differs from 188 classes in R30 by more than three standard deviations. The upper panels show '
        'accuracy and R30, the inverse background efficiency at 30% signal efficiency, and the lower panels the area '
        'under the ROC curve, averaged over the classes on JetClass-II and JetClass. Error bars are ±1 standard deviation over the '
        'runs of figure {fig:F1} or three trainings from scratch (grey); open stars mark the published Particle '
        'Transformer. Figure {fig:A10} shows the linear probes, and values are in tables {tab:ft} and {tab:summary}.'
    ),
    "F9b": (
        'Each network with an added mass regression (+ mass) resolves the jet mass better than the soft-drop mass in '
        'every bin of jet pT and generator mass, and is less biased in 13 of the 15 bins but more biased in the lowest '
        'generator-mass bin, 20 to 50 GeV. Panels (a) and (b) show the median of r = m/m_gen, estimated over '
        'generator-level soft-drop mass, with its distance from one below, and (c) and (d) the resolution, half the '
        'central 68% range of r over its median, on a logarithmic axis, with the ratio to the soft-drop value below on a linear one. '
        'Points sit in their bin, the horizontal bar spanning it; error bars combine the standard deviation over two '
        'pretraining runs with the bootstrap over test jets. Values are in table {tab:mass}.'
    ),
    "A17-probe": (
        'On frozen features, a multilayer-perceptron probe resolves the jet mass better on the networks with an added '
        'mass regression (+ mass) than on the same label sets without it. The panels have the form of figure '
        '{fig:F9b}, for a probe trained to regress the jet mass from the frozen features of each label set, mass '
        'network and the random network (crosses), with the soft-drop mass in black. {probe_note} Points sit in their bin, the horizontal bar spanning it, and error bars combine the '
        'standard deviation over three pretraining runs, two for + mass, or three random initializations, with the '
        'bootstrap over test jets. Values are in table {tab:mass}.'
    ),
    "F11": (
        'At 1% QCD efficiency, 17 classes keep fewer X→bb̄ and X→bs jets than 188 classes, a gap of 0.6 and 1.0 in the '
        'figure of merit, which widens below 0.1% for X→bs. The figure of merit, signal efficiency over the square '
        'root of QCD efficiency, is shown for (a) X→bb̄ and (b) X→bs against QCD, all jets with soft-drop mass 150 to '
        '230 GeV and efficiencies relative to them. Solid curves are a multilayer-perceptron classifier on frozen '
        'features starting from the pretrained output head, grey curves training from scratch, and dash-dotted curves '
        'the X→bb̄ output against the QCD outputs (188, 162 and 162 classes + mass, in (a)). Bands, for 188 classes '
        'and from scratch only, are ±1 standard deviation over the runs of figure {fig:F1} or three trainings. Table '
        '{tab:xbx} holds the values.'
    ),
    "F11-part": (
        "At 1% QCD efficiency, the partitions with no, a b-quark and a random flavor boundary each keep fewer X→bb̄ and"
        " X→bs jets than the network pretrained with 188 classes, and no 17-class partition differs from the 17-class "
        "network by more than three standard deviations (table {tab:xbx}). The panels show the figure of merit of "
        "figure {fig:F11} against QCD efficiency for the classifier on frozen features of every label set, mass network"
        " and partition. Lines are means over three pretraining runs per label set, two per mass network and "
        "flavor-boundary partition, and over the five random partitions with one run each; the band, for 188 classes "
        "only, is ±1 standard deviation. Values and the other standard deviations are in table {tab:xbx}."
    ),
    "F4-other": (
        "With linear probes, 43 and 17 classes separate both pairs less well than 188 classes, and 162 classes, which "
        "merge the 27 QCD classes, separate the QCD pair (b) less well. Panels show background rejection against signal"
        " efficiency, the first-named decay the signal, for the label sets and mass networks (upper row) and the "
        "flavor-boundary partitions (lower row), means over the runs of figure {fig:F1} ending where fewer than 20 "
        "background jets remain. Lower panels give the ratio to 188 classes on a logarithmic axis, banded for 188 "
        "classes only by run spread and binomial uncertainty. " + NO_RAND +
        " Figure {fig:F4-light} shows the other two pairs, and table {tab:pairs} each pair's rejection at 90% signal efficiency and the other spreads."
    ),
    "F4-light": (
        "With linear probes, 43 and 17 classes separate QCD jets with b quarks, and those with c but no b quarks, from "
        "light QCD jets less well than 188 classes, and so do 162 classes, which merge the 27 QCD classes. The panels "
        "are those of figure {fig:F4-other} with the jets with heavy quarks as the signal. Curves are means over the "
        "runs of figure {fig:F1} ending where fewer than 20 background jets remain, and the ratio panels, on a "
        "logarithmic axis, are banded for 188 classes only. "
        + NO_RAND + " Table {tab:pairs} gives each pair's rejection at 90% signal efficiency and the other spreads."
    ),
    "F4-mech-ft": (
        "After fine-tuning, 17 classes still separate X→bb̄ from cc̄ and X→bc from bq, cs worse than 188 classes. The "
        "panels are those of figure {fig:F4-mech} for networks fine-tuned on one million class-balanced JetClass-II "
        "jets and scored on " + MIXED + ", with training from scratch in the upper row and 17 classes in the lower row "
        "for reference. The ratio panels have a logarithmic axis, and curves end where fewer than 20 background jets "
        "remain. Bands, for 188 classes and from scratch only, combine run spread and binomial "
        "uncertainty. " + NO_RAND + " Table {tab:pairs} gives each pair's rejection at 90% signal efficiency and the other spreads."
    ),
    "F4-other-ft": (
        "After fine-tuning, 17 classes still separate both pairs worse than 188 classes. The panels are those of "
        "figure {fig:F4-other} for networks fine-tuned on one million class-balanced JetClass-II jets and scored on "
        + MIXED + ", with training from scratch in the upper row. The ratio panels have a logarithmic axis, and curves "
        "end where fewer than 20 background jets remain. Bands, for 188 classes and from scratch only, combine run "
        "spread and binomial uncertainty. " + NO_RAND + " Figure {fig:F4-light-ft} shows the other two pairs, and table {tab:pairs} each pair's rejection at 90% signal efficiency and "
        "the other spreads."
    ),
    "F4-light-ft": (
        "After fine-tuning, 17 classes still separate QCD jets with c but no b quarks from light QCD jets worse than "
        "188 classes. The panels are those of figure {fig:F4-light} for networks fine-tuned on one million "
        "class-balanced JetClass-II jets and scored on " + MIXED + ", with training from scratch in the upper row. The "
        "ratio panels have a logarithmic axis, and curves end where fewer than 20 background jets remain. Bands, for "
        "188 classes and from scratch only, combine run spread and binomial uncertainty. " + NO_RAND + " Table {tab:pairs} gives each pair's rejection at 90% signal efficiency and the "
        "other spreads."
    ),
    "PT": (
        'In every jet pT bin, linear probes with 43 and 17 classes and 17 classes fine-tuned are less accurate on '
        'JetClass-II than 188 classes, and on JetClass 17 classes are less accurate fine-tuned in the two lowest bins '
        '(500–700 GeV) and as a linear probe in the highest (900–1000 GeV). Panels show per-bin accuracy of linear '
        'probes (upper row) and fine-tuned networks (lower row) for label sets and mass networks on JetClass-II and '
        'JetClass only, as top and quark/gluon jet pT span about 50 to 100 GeV. Points sit in their bin, the '
        'horizontal bar spanning it, and error bars combine the standard deviation over the runs of figure {fig:F1} '
        'with the binomial uncertainty over test jets, the three-standard-deviation test applied. Values are in the '
        'supplementary file jet_pt_bins.csv.'
    ),
    "A10": (
        'With linear probes, 43 and 17 classes are less accurate on JetClass-II than 188 classes at every training-set '
        'size. The upper panels show accuracy on class-balanced JetClass-II and JetClass and R30, the inverse '
        'background efficiency at 30% signal efficiency, on top and quark/gluon tagging, Pythia and Herwig, and the '
        'lower panels the area under the ROC curve (AUC), averaged over the classes on JetClass-II and JetClass, '
        'against training-set size. Error bars are ±1 standard deviation over the runs of figure {fig:F1}, with points offset by at most 10% '
        'in size. Open stars mark the published Particle Transformer, trained from scratch on 2, 1.2 and 1.6 million '
        'jets, with its published standard deviation on R30, above every probe. The random network, far below every '
        'pretrained network, and every value are in tables {tab:probesn} and {tab:summary}.'
    ),
    "F10b": (
        'Used directly, no label set differs from 188 classes, nor a mass network from its own label set, in how much '
        'it distorts the QCD mass by more than three standard deviations at 10% or 1% QCD efficiency. '
        'The curves show '
        'the Jensen-Shannon distance (base 2) between the soft-drop mass distributions of all JetClass-II QCD test '
        'jets and those passing a cut on the summed QCD output, against QCD efficiency eps_B, of (a) the pretraining '
        'head used directly (dash-dotted) and (b) the networks fine-tuned on JetClass-II, with training from scratch '
        'in grey, on logarithmic axes but for the linear y axis of (b). Bands, for 188 classes and from scratch only, are ±1 standard deviation over '
        'runs; the dotted line in (a) is a random selection. Values are in table {tab:sculpt}.'
    ),
    **{k: (
        'Qualitatively and in this projection, the 188-class frozen features separate X→bb̄, X→cc̄ and X→qq̄ more '
        'cleanly than the 17-class ones' + (', and after fine-tuning both separate them' if k == "UMAP-ft" else '')
        + '. The panels show two-dimensional embeddings, by uniform manifold approximation and projection (UMAP), of '
        'the class token, the 128-dimensional vector the classification head reads, of the networks pretrained with '
        '188, 162, 43 and 17 classes (columns), ' + ('frozen' if k == "UMAP-frozen" else 'fine-tuned on each task')
        + ', for 2,000 test jets of each kind in the row\'s legend on each task (rows), the same jets in every panel '
        'of a row' + (', X→bb̄ and X→bs with soft-drop mass 150 to 230 GeV' if k == "UMAP-frozen" else '') + '. '
        'Every panel uses pretraining run 1 and the same UMAP settings and seed. Each panel is fitted separately, so '
        'positions cannot be compared between panels. ' + ('Figure {fig:UMAP-ft} shows the fine-tuned networks.'
                                                           if k == "UMAP-frozen" else
                                                           'The pretrained networks are not fine-tuned on X→bb̄ and '
                                                           'X→bs.')
    ) for k in ("UMAP-frozen", "UMAP-ft")},
    "VAL": (
        "Every network's pretraining accuracy has levelled off by the last epoch, and the learning-rate decay from "
        "epoch 57 adds a small rise to every curve. The curves show the accuracy of each network's own classification "
        "output on the JetClass-II validation jets of pretraining against epoch, for 188, 162, 43 and 17 classes and "
        "for 162 and 17 classes with an added mass regression (+ mass, lighter shades), the 17 classes + mass curve "
        "lying on that of 17 classes. Each network is scored on its own classes, so the heights of the curves do not "
        "compare the label sets. Each curve is the mean over the runs of figure {fig:F1}, and the ±1 standard "
        "deviation bands over runs are too narrow to see."
    ),
}


def fill(raw: str) -> str:
    """raw with each {tab:...} and {fig:<page id>} as that item's number, without the word (the sentence says
    "table" or "figure"), and each {name} of FILLS as its sentence."""
    for name, sentence in FILLS.items():
        raw = raw.replace("{" + name + "}", sentence)
    out = re.sub(r"\{(tab:[a-z_0-9]+)\}", lambda m: G.number(m.group(1)).removeprefix("Table "), raw)
    return re.sub(r"\{fig:([A-Za-z0-9-]+)\}", lambda m: G.number(m.group(1)).removeprefix("Fig. "), out)


def text(pid: str) -> str:
    """The caption of a main-text or appendix figure, filled (fill), with its computed counts (COUNTS). A paper caption
    that names a supplement item by number is refused (PI 2026-11-14: the paper must read without the supplement)."""
    raw = CAPTIONS[pid]
    sup = [k for k in re.findall(r"\{((?:tab|fig):[A-Za-z0-9_-]+)\}", raw)
           if G.TIER.get(k.removeprefix("fig:")) == G.SUPPLEMENT or G.TIER.get(k) == G.SUPPLEMENT]
    if G.TIER.get(pid) in (G.MAIN, G.APPENDIX) and sup:
        raise SystemExit(f"FATAL: the paper caption of {pid} names supplement items {sup}: write \"the supplementary "
                         f"material\"")
    return " ".join(fill(raw.replace("{counts}", COUNTS.get(pid, ""))).split())


def apply(pages: list) -> list:
    return [dataclasses.replace(p, caption=text(p.id)) if p.id in CAPTIONS else p for p in pages]
