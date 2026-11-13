"""The captions of the figures placed in the paper (PI brief 2026-11-12): the finding where one is resolved, what is
plotted, how to read it, the uncertainty and run counts, and the table that holds the values; two to five plain
sentences. They replace the Page captions of those figures (apply, called at the end of each gallery module). A table
is named by its label in braces and printed with its number of the day (gallery_placement.number)."""
import dataclasses
import re

import gallery_placement as G

CAPTIONS = {
    "F1": (
        "Panel (a) shows six of the 188 JetClass-II classes as jets above the class of the 17-class set "
        "holding each (j, any quark or gluon). Below, each cell is one class, each row is one label set, and "
        "a bar spanning several cells is one merged class. The 162-class set merges the 27 QCD classes, the "
        "43-class set also keeps only the heaviest quark flavor and merges electrons with muons, and the "
        "17-class set drops quark flavor. Panel (b) shows a network pretrained three times per label set, "
        "every label set seeing the same jets in the same order, then used with frozen features (open "
        "squares) or fine-tuned (filled squares)."
    ),
    "F2a": (
        "With linear probes on frozen features, the networks pretrained with 43 classes, 17 classes and every "
        "17-class partition are less accurate on JetClass-II than the network pretrained with 188 classes. "
        "Each pair of panels is one task, with its accuracy or R30, the inverse background efficiency at 30% "
        "signal efficiency, on the left and its area under the receiver operating characteristic curve (AUC) "
        "on the right. The four bracketed rows are other 17-class partitions of the 188 classes, "
        "and a flavor boundary is a class boundary between decays that differ only in quark flavor. Each row "
        "shows the mean and ±1 standard deviation over three pretraining runs, two per flavor-boundary "
        "partition, or five random partitions with one run each. Values are in table {tab:summary}."
    ),
    "F2b": (
        "After fine-tuning, the network pretrained with 17 classes makes about 3% more errors on JetClass "
        "than the network pretrained with 188 classes. In R30 on top and quark/gluon tagging, the networks "
        "pretrained with 162, 43 and 17 classes agree with it within the spread over pretraining runs. Every "
        "pretrained network is more accurate than training from scratch on JetClass-II and JetClass. Each row shows the "
        "mean over pretraining runs with a bar of ±1 standard deviation, and the vertical line and shaded "
        "band show the same for 188 classes. There are three runs per label set, two per flavor-boundary "
        "partition, five random partitions with one run each, and three from scratch. Fine-tuning uses one "
        "million class-balanced JetClass-II jets, one million JetClass jets and the full top and quark/gluon "
        "training sets. Each pair of panels is one task, with its accuracy or R30, the inverse background "
        "efficiency at 30% signal efficiency, on the left and its area under the receiver operating "
        "characteristic curve (AUC) on the right. Values are in table {tab:summary}."
    ),
    "F3-main": (
        "Pretraining with 17 classes instead of 188 lowers the area under the ROC curve (AUC), averaged over "
        "the 188 JetClass-II classes, by 0.0110 for a linear probe and by 0.0025 after fine-tuning. Each "
        "point is one class, at its AUC for separating it from all other classes, for the network pretrained "
        "with 17 classes against the network pretrained with 188 classes. Panel (a) shows linear probes and "
        "panel (b) fine-tuned networks, all trained on one million class-balanced JetClass-II jets. Each "
        "point is the mean over three pretraining runs, and the class-averaged linear-probe values are in "
        "table {tab:probes}."
    ),
    "F4-mech": (
        "Linear probes separate decays that differ only in quark flavor worse on the network pretrained with "
        "17 classes than on the network pretrained with 188 classes. Each panel shows background rejection "
        "against signal efficiency for a linear probe on the frozen features, with the first-named decay as "
        "signal. The upper row compares the four label sets, and the lower row compares the network "
        "pretrained with 188 classes with the three 17-class flavor-boundary partitions of figure 2. Curves "
        "are means over three pretraining runs, two for each partition, and end where fewer than 20 "
        "background jets pass in any run. Areas under the ROC curve are in table {tab:pairs}."
    ),
    "F8": (
        "Fine-tuning the network pretrained with 188 classes is more accurate than training from scratch on "
        "JetClass-II and JetClass at every training-set size, and on top and quark/gluon tagging the two are "
        "consistent within the spread over runs and test jets. The panels show accuracy or R30, the inverse "
        "background efficiency at 30% signal efficiency, against the number of training jets, with that "
        "network fine-tuned (f.t.) or as a linear probe (LP). Points are means and bands show ±1 standard "
        "deviation over three pretraining runs or three trainings from scratch. Stars mark the published "
        "Particle Transformer (ParT) trained from scratch. Values at the largest training size are in table "
        "{tab:summary}."
    ),
    "A8": (
        "After fine-tuning, the network pretrained with 17 classes has the lowest accuracy of the four "
        "pretrained networks on JetClass-II at every training-set size. The panels show accuracy on "
        "class-balanced JetClass-II and on JetClass, and R30, the inverse background efficiency at 30% signal "
        "efficiency, on top and quark/gluon tagging, against the number of training jets. Points are means "
        "and bands are ±1 standard deviation over three pretraining runs, or over three trainings from "
        "scratch. Stars mark the published results of the Particle Transformer (ParT). Values at the largest "
        "training size are in table {tab:summary}."
    ),
    "F9b": (
        "A mass output added in pretraining has better resolution than the soft-drop mass in every bin and "
        "less bias in 13 of 15, and the 162- and 17-class networks agree within the spread over runs. Panels "
        "(a) and (b) show the median of r = m/m_gen, the ratio of estimated to generator mass, against jet "
        "transverse momentum pT and generator mass. Panels (c) and (d) show the resolution sigma_r, half the "
        "central 68% range of that ratio, divided by its median. The lower panels show the ratio to the "
        "soft-drop value. Horizontal bars span the bin, and vertical bars show the standard deviation over "
        "two pretraining runs."
    ),
    "F11": (
        "At a QCD efficiency of 1%, the network pretrained with 17 classes keeps fewer X→bb̄ and X→bs jets "
        "than the network pretrained with 188 classes. The figure of merit is plotted against QCD efficiency "
        "for (a) X→bb̄ and (b) X→bs jets against QCD jets, all with soft-drop mass between 150 and 230 GeV. "
        "Transfer is a classifier trained on a network's frozen features, and the direct discriminant uses "
        "the network's output scores without further training. Lines are means and bands are ±1 standard "
        "deviation over three runs, and the single training from scratch in (a) has no band. Values at a QCD "
        "efficiency of 1% are in table {tab:xbx}."
    ),
    "F4-other": (
        "With 17 classes in (a) and (c), 162 and 43 classes in (c) and 43 classes in (b), where the "
        "pretraining label set merged the two classes into one, linear probes separate them less well than "
        "with 188 classes. Each panel shows background rejection against signal efficiency for a linear probe "
        "on frozen features, with the first-named class as signal. Each curve ends where fewer than 20 "
        "background jets remain, so the networks pretrained with 188 and 162 classes, which separate the pair "
        "in (b) almost completely, have no curve. Each curve is the mean over three pretraining runs. Areas "
        "under the ROC curve are in table {tab:pairs}."
    ),
    "F4-mech-ft": (
        "After fine-tuning, the network pretrained with 17 classes still separates X→bb̄ from cc̄ and X→bc "
        "from bq, cs worse than the network pretrained with 188 classes. Each panel shows background "
        "rejection against signal efficiency on test jets for one pair of JetClass-II classes, the "
        "first-named being signal, for networks fine-tuned on one million class-balanced JetClass-II jets. "
        "The upper row compares the four label sets and the lower row the three 17-class flavor-boundary "
        "partitions of figure 2, with 188 classes for reference. Curves are means over three pretraining "
        "runs, two for each partition, and end where fewer than 20 background jets remain. Areas under the "
        "ROC curve are in table {tab:pairs}."
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
        "After fine-tuning on top and quark/gluon tagging, no label set differs from 188 classes in "
        "background rejection by more than the combined uncertainty from the pretraining runs and the test "
        "sample. The upper panels show background rejection against signal efficiency, and panel (c) scores "
        "the networks of panel (b) on Herwig jets. Curves are means over three runs, except the dotted curve, "
        "which is one run. The lower panels show the ratio to the 188-class curve, and each curve's band "
        "combines its standard deviation over runs with the 68% binomial uncertainty of its passing "
        "background count. Values at 50% and 30% signal efficiency are in tables {tab:published} and "
        "{tab:labelsets}."
    ),
    "A10": (
        "On JetClass-II the linear probes of the networks pretrained with 43 and 17 classes are less accurate "
        "than the probe of the network pretrained with 188 classes at every training size. The panels show "
        "linear-probe accuracy on class-balanced JetClass-II and on JetClass and the rejection R30, the "
        "inverse background efficiency at 30% signal efficiency, on top and quark/gluon tagging, against the "
        "number of training jets. Stars mark published values for the Particle Transformer (ParT) trained "
        "from scratch. Points are means and bands show ±1 standard deviation over three pretraining runs or "
        "three trainings from scratch. Linear-probe values at the largest training size are in table "
        "{tab:probes}."
    ),
    "F10b": (
        "At 1% QCD efficiency, a cut on the network pretrained with 17 classes distorts the QCD mass "
        "distribution more than the same cut on the network pretrained with 188 classes. Curves show the "
        "Jensen-Shannon distance between the soft-drop mass distributions of all JetClass-II QCD test jets "
        "and of those passing a cut on 1 - g_QCD, against QCD efficiency eps_B, where $g_{\\mathrm{QCD}}$ is "
        "the summed probability the network assigns to QCD. Bands show ±1 standard deviation over three "
        "pretraining runs, two for networks with mass regression, and the dotted line is the distance "
        "expected from a random selection of as many QCD jets. Values are in tables {tab:sculpt} and "
        "{tab:sculptboot}."
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
        "pretrained with 188, 162, 43 and 17 classes. Each network is scored on its own classes, so the "
        "heights of the curves do not compare the label sets. The learning rate decays from epoch 57, where "
        "every curve rises slightly. Each curve is the mean over three pretraining runs, and the standard "
        "deviation over runs, below 0.006 at every epoch, is not drawn."
    ),
}


def text(pid: str) -> str:
    """The caption with each {tab:...} as that table's number, without the word (the sentence says "table")."""
    return re.sub(r"\{(tab:[a-z_0-9]+)\}", lambda m: G.number(m.group(1)).removeprefix("Table "), CAPTIONS[pid])


def apply(pages: list) -> list:
    return [dataclasses.replace(p, caption=text(p.id)) if p.id in CAPTIONS else p for p in pages]
