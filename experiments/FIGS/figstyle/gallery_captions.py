"""The captions of the figures placed in the paper (PI brief 2026-11-12): the finding where one is resolved, what is
plotted, how to read it, the uncertainty and run counts, and the table that holds the values; two to five plain
sentences. They replace the Page captions of those figures (apply, called at the end of each gallery module). A table
is named by its label in braces and printed with its number of the day (gallery_placement.number)."""
import dataclasses
import re

import gallery_placement as G

CAPTIONS = {
    "F1": (
        "Panel (a) shows six of the 188 JetClass-II classes as jets above the class of the 17-class set holding each "
        "(j, any quark or gluon). Below, each cell is one class, each row is one label set, and a bar spanning "
        "several cells is one merged class. The 162-class set merges the 27 QCD classes, the 43-class set also keeps "
        "only the heaviest quark flavor and merges electrons with muons, and the 17-class set drops quark flavor. "
        "Panel (b) shows the network, the Particle Transformer architecture with Sophon's output head, pretrained "
        "from scratch three times for each label set, every label set seeing the same jets in the same order. Each "
        "pretrained network is then used with frozen features (open squares) or fine-tuned (filled squares) on five "
        "tasks, namely JetClass-II, JetClass, top tagging, quark/gluon tagging, and X→bb̄ and X→bs resonance tagging."
    ),
    "F2": (
        "With linear probes, the networks pretrained with 43 and 17 classes and with every 17-class partition, a "
        "regrouping of the 188 classes, are less accurate on JetClass-II than the 188-class network, and fine-tuned "
        "the 17-class network stays below it on JetClass-II and JetClass. On top and quark/gluon tagging, Pythia and "
        "Herwig, no label set differs from 188 classes by more than the spread over runs. Rows show the mean and ±1 "
        "standard deviation over three pretraining runs at each task's largest training set, two for networks with an "
        "added mass regression (+ mass) and for partitions named by their flavor boundary, between decays differing "
        "only in quark flavor, or five random partitions. R30 is the inverse background efficiency at 30% signal "
        "efficiency, and the random network is untrained. Figures {fig:F2a} and {fig:F2b} add the area under the ROC "
        "curve, with values in table {tab:summary}."
    ),
    "F2a": (
        "Linear probes on JetClass-II separate the classes less well for the networks pretrained with 43 and 17 "
        "classes than for 188 classes in the area under the ROC curve (AUC), as in accuracy, and the AUC also places "
        "162 classes below 188. The upper row repeats figure {fig:F2}'s linear probes, and the lower row shows the "
        "AUC, averaged over the classes, each against the rest, on JetClass-II and JetClass. Rows show the mean and "
        "±1 standard deviation over runs as in figure {fig:F2}, and the line and band repeat the 188-class values. "
        "The released Sophon checkpoint is one network, shown on JetClass-II and JetClass only because its top and "
        "quark/gluon probes set the missing inputs to zero, and the random network, which is untrained, lies beyond "
        "an axis break. Values are in tables {tab:summary} and {tab:summaryauc}."
    ),
    "F2b": (
        "After fine-tuning, every pretrained network is more accurate than training from scratch on JetClass-II "
        "and JetClass, in accuracy and in the area under the ROC curve (AUC). The upper row repeats figure "
        "{fig:F2}'s fine-tuned networks, and the lower row shows the AUC, averaged over the classes, each against the rest, "
        "on JetClass-II and JetClass. Rows show the mean and ±1 standard deviation over three pretraining runs, two "
        "for the + mass networks and the flavor-boundary partitions, five random partitions or three trainings from "
        "scratch, and the line and band repeat the 188-class values. Fine-tuning uses one million class-balanced "
        "JetClass-II jets, one million JetClass jets and the full top and quark/gluon training sets, and from "
        "scratch lies beyond an axis break where it is far below the rest. Values are in tables {tab:summary} and "
        "{tab:summaryauc}."
    ),
    "PC-jc2": (
        "Pretraining with 17 classes instead of 188 lowers the area under the ROC curve (AUC) of almost every "
        "JetClass-II class, by more than three standard deviations over runs for 184 of the 188 classes with linear "
        "probes and for 161 after fine-tuning, and raises none. Each point is one class, at its AUC for separating it "
        "from all other classes, for the network named above the panel (vertical) against the network pretrained "
        "with 188 classes (horizontal), both as linear probes (upper row) or both fine-tuned (lower row) on one "
        "million class-balanced JetClass-II jets. Points below the dotted diagonal are classes that the named network "
        "separates less well, and colors are decay families. Each point is the mean over three pretraining runs. The "
        "points are in table {tab:perclass} and their averages in table {tab:summaryauc}, and figure {fig:PC-jc1} "
        "repeats the comparison on JetClass."
    ),
    "PC-jc1": (
        "With linear probes, the network pretrained with 17 classes separates H→cc̄ jets visibly less well than the "
        "network pretrained with 188 classes, while 162 and 43 classes agree with 188 classes on every class. After "
        "fine-tuning every difference is below 0.003, too small to see here, and 17 classes stay below 188 classes by "
        "more than four standard deviations over runs on four of the ten classes. Each point is one class, at its "
        "area under the ROC curve for separating it from the other nine, for the network named above the panel "
        "(vertical) against the 188-class network (horizontal), as linear probes (upper row) or fine-tuned (lower "
        "row) on one million JetClass jets. Each point is the mean over three pretraining runs, and the values are in "
        "table {tab:jc1_class_auc}."
    ),
    "F4-mech": (
        "Linear probes separate decays that differ only in quark flavor worse on the network pretrained with 17 classes "
        "than on the network pretrained with 188 classes, and a flavor boundary in a 17-class partition recovers part "
        "of the loss on X→bc vs bq, cs but not on X→bb̄ vs cc̄. Each panel shows background rejection against signal "
        "efficiency for a linear probe on frozen features, with the first-named decay as signal. The upper row compares "
        "the label sets, with and without an added mass regression (+ mass), and the lower row the three "
        "flavor-boundary partitions with 188 and 17 classes. The lower panels show the ratio to 188 classes, each band "
        "combining a curve's standard deviation over three pretraining runs, two for + mass and the partitions, with "
        "its binomial uncertainty. Figure {fig:F4-mech-ft} shows the fine-tuned networks, and the values are in table "
        "{tab:pairs}."
    ),
    "A8": (
        "After fine-tuning, the network pretrained with 17 classes is less accurate than the network pretrained with "
        "188 classes on JetClass-II at every training-set size and on JetClass at 10,000 and one million jets, while "
        "on top and quark/gluon tagging, Pythia and Herwig, the label sets and mass networks agree with 188 classes "
        "within the spread over runs. The panels show accuracy on class-balanced JetClass-II and on JetClass, which "
        "start at 10,000 jets, and R30, the inverse background efficiency at 30% signal efficiency, against the "
        "number of training jets, with training from scratch in grey. Points are means and bands ±1 standard "
        "deviation over three pretraining runs, two for networks with an added mass regression (+ mass), or three "
        "trainings from scratch, and stars mark the published Particle Transformer (ParT). Figure {fig:A10} shows the "
        "linear probes, and the values are in tables {tab:ft} and {tab:summary}."
    ),
    "F9b": (
        "A mass output added in pretraining has a better resolution than the soft-drop mass in all 15 bins and less "
        "bias in 13 of them, each by more than four standard deviations over runs, and the networks pretrained with "
        "162 and 17 classes agree within two standard deviations in every bin. Panels (a) and (b) show the median of "
        "r = m/m_gen, the ratio of estimated to generator mass, against jet transverse momentum pT and generator "
        "mass. Panels (c) and (d) show the resolution sigma_r, half the central 68% range of that ratio divided by "
        "its median, and the lower panels the ratio to the soft-drop value. Horizontal bars span the bin, and "
        "vertical bars show the standard deviation over two pretraining runs. Values are in tables {tab:mass} and "
        "{tab:massdetail}, and figures {fig:A17-probe} and {fig:A17-ridge} show the mass regressed from frozen "
        "features."
    ),
    "F11": (
        "At 1% QCD efficiency, the network pretrained with 17 classes keeps fewer X→bb̄ and X→bs jets than the "
        "network pretrained with 188 classes. The figure of merit, signal efficiency over the square root of QCD "
        "efficiency, is plotted against QCD efficiency for (a) X→bb̄ and (b) X→bs against QCD jets with "
        "soft-drop mass between 150 and 230 GeV, so its value at 1% is the table value over 10. Solid curves are a "
        "classifier trained on frozen features, since the pretrained networks are not fine-tuned on these samples, "
        "and dash-dotted curves the direct discriminant from the output scores, where X→bb̄ is one of the network's "
        "classes. Lines are means and bands ±1 standard deviation over three pretraining runs, two for networks with "
        "an added mass regression (+ mass), or three trainings from scratch. Values are in table {tab:xbx}, "
        "and figure {fig:F11-part} shows the 17-class partitions."
    ),
    "F11-part": (
        "Each flavor-boundary partition keeps fewer X→bb̄ and X→bs jets at 1% QCD efficiency than the network "
        "pretrained with 188 classes, and no 17-class partition differs from the 17-class network by more than three "
        "standard deviations. The panels have the form of figure {fig:F11}, with a classifier trained on the frozen "
        "features of each partition and of 188 and 17 classes for reference. Lines are means and bands ±1 standard "
        "deviation over two pretraining runs per flavor-boundary partition, three for 188 and 17 classes, or the five "
        "random partitions. Values at 1% QCD efficiency are in table {tab:xbx}."
    ),
    "F4-other": (
        "With linear probes, the networks pretrained with 43 and 17 classes separate the four other flavor pairs less "
        "well than the network pretrained with 188 classes, and 162 classes, which merge the 27 QCD classes, the three "
        "QCD pairs. The panels have the form of figure {fig:F4-mech}, with the label sets and the networks with an "
        "added mass regression (+ mass) in the upper row and the flavor-boundary partitions with 188 and 17 classes in "
        "the lower row. Curves are means over three pretraining runs, two for + mass and the partitions, and end where "
        "fewer than 20 background jets remain. The lower panels show the ratio to 188 classes with each curve's "
        "standard deviation over runs and binomial uncertainty, and the values are in table {tab:pairs}."
    ),
    "F4-other-ft": (
        "After fine-tuning, the network pretrained with 17 classes still separates the pairs of panels (a), (b) and (d) "
        "worse than the network pretrained with 188 classes, by less than with linear probes. The panels are those of figure {fig:F4-other} for networks fine-tuned on one million "
        "class-balanced JetClass-II jets and scored on natural-mix test jets, with training from scratch added to the "
        "upper row. The lower panels show the ratio to 188 classes, each band combining a curve's standard deviation "
        "over runs, three per label set, two for + mass and the partitions and three trainings from scratch, with the "
        "binomial uncertainty of its passing background count. The values are in table {tab:pairs}."
    ),
    "F4-mech-ft": (
        "After fine-tuning, the network pretrained with 17 classes still separates X→bb̄ from cc̄ and X→bc from bq, cs "
        "worse than the network pretrained with 188 classes, by less than with linear probes. The panels are those of "
        "figure {fig:F4-mech} for networks fine-tuned on one million class-balanced JetClass-II jets and scored on "
        "natural-mix test jets, with training from scratch added to the upper row. The lower panels show the ratio to "
        "188 classes, each band combining a curve's standard deviation over runs, three per label set, two for + mass "
        "and the partitions and three trainings from scratch, with the binomial uncertainty of its passing background "
        "count. Curves end where fewer than 20 background jets remain, and the values are in table {tab:pairs}."
    ),
    "PT": (
        "In every bin of jet transverse momentum, linear probes on the networks pretrained with 43 and 17 "
        "classes are less accurate on JetClass-II than probes on the network pretrained with 188 classes. "
        "Panels (a) and (b) show linear-probe accuracy on JetClass-II, panel (c) accuracy on JetClass after "
        "fine-tuning on one million jets, and panel (d) the area under the receiver operating characteristic "
        "curve (AUC) of the linear probes separating X→bb̄ from X→cc̄ jets. Panel (b) uses the "
        "flavor-boundary partitions of figure 2. Points show the mean over three pretraining runs, two for "
        "each flavor-boundary partition, horizontal bars span the bin, and vertical bars show ±1 standard "
        "deviation over runs."
    ),
    "F6": (
        "On top and quark/gluon tagging, Pythia and Herwig, no label set or mass network differs "
        "from the network pretrained with 188 classes in R30, the inverse background efficiency at 30% signal "
        "efficiency, by more than three standard deviations, as linear probes or fine-tuned. The panels show "
        "background rejection against signal efficiency for linear probes on frozen features (upper row) and for "
        "fine-tuned networks beside training from scratch (lower row), on the full training sets. The lower panels "
        "show the ratio to 188 classes, each band combining a curve's standard deviation over three runs, two for "
        "networks with an added mass regression (+ mass), or three trainings from scratch, with the binomial "
        "uncertainty of its passing background count. R30 and the area under the ROC curve are in tables {tab:summary} and "
        "{tab:summaryauc}, the rejection at 50% signal efficiency in tables {tab:probes} and {tab:labelsets}."
    ),
    "A10": (
        "With linear probes, the networks pretrained with 43 and 17 classes are less accurate on JetClass-II than the "
        "network pretrained with 188 classes at every training-set size. The panels show linear-probe accuracy on "
        "class-balanced JetClass-II and on JetClass and R30, the inverse background efficiency at 30% signal "
        "efficiency, on top and quark/gluon tagging, Pythia and Herwig, against the number of training jets, with the "
        "random network, a network with its initial random weights, in dark grey. Points are means and bands ±1 "
        "standard deviation over three pretraining runs, two for networks with an added mass regression (+ mass), or "
        "three random initializations, and stars mark the published Particle Transformer (ParT). The values are in "
        "tables {tab:probesn} and {tab:summary}."
    ),
    "F10b": (
        "Used directly, no network pretrained with fewer classes or with an added mass regression (+ mass) distorts "
        "the QCD mass distribution differently from the network pretrained with 188 classes, nor a + mass network "
        "from its own label set, by more than three standard deviations of the spread over runs and test jets "
        "combined, at 10% or 1% QCD efficiency or 50% signal efficiency. Curves show the "
        "Jensen-Shannon distance between the soft-drop mass distributions of all JetClass-II QCD test jets and of "
        "those passing a cut on 1 - g_QCD, against QCD efficiency eps_B, where $g_{\\mathrm{QCD}}$ is the summed "
        "probability the network assigns to QCD. Bands show ±1 standard deviation over three pretraining runs, two "
        "for + mass, and the dotted line is the distance expected from a random selection of as many QCD jets. "
        "Values are in tables {tab:sculpt} and {tab:sculptboot}."
    ),
    "UMAP-jc2-had": (
        "The panels show two-dimensional embeddings of the class token, the 128-dimensional vector that the "
        "classification head reads, computed with uniform manifold approximation and projection (UMAP) for "
        "the networks pretrained with 188 and with 17 classes, frozen and fine-tuned with the 188 classes on "
        "one million class-balanced JetClass-II jets. Each point is a JetClass-II test jet, and every panel "
        "shows the same 2,000 jets of each kind in the legend. The 17-class label set merges X→bb̄, X→cc̄ and "
        "X→qq̄ into one class. One pretraining run of each network is shown and each panel is fitted "
        "separately, so only the grouping of points within a panel can be read."
    ),
    "VAL": (
        "The curves show the validation accuracy on JetClass-II against pretraining epoch for the networks "
        "pretrained with 188, 162, 43 and 17 classes and with 162 and 17 classes and an added mass regression "
        "(+ mass, lighter shades). Each network is scored on its own classes, so the heights of the curves do not "
        "compare the label sets. The learning rate decays from epoch 57, where every curve rises slightly. Each "
        "curve is the mean over three pretraining runs, two for + mass, and the bands show ±1 standard deviation "
        "over runs."
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
