"""The captions of the figures placed in the paper (PI brief 2026-11-12): the finding where one is resolved, what is
plotted, how to read it, the uncertainty and run counts, and the table that holds the values; two to five plain
sentences. They replace the Page captions of those figures (apply, called at the end of each gallery module). A table
is named by its label in braces and printed with its number of the day (gallery_placement.number). PI 2026-11-14
(step 8): a paper caption never names a supplement item by number (text() refuses one), only "the supplementary
material"; one plain finding sentence first; no sentence about a gap, an unclaimed result or an epoch cap.
{counts}: the computed per-class counts (gallery_perclass.counts_sentence, COUNTS). {m:Name}: the paper's macro Name
(macro), so a page caption that follows its section caption prints the same counts and ranges."""
import dataclasses
import pathlib
import re

import gallery_placement as G

# the mixed test sample's decay families (PI 2026-11-14), the fine-tuned class pairs' and sculpting's test jets
MIXED = "the mixed test sample (38.7% two-prong, 40.8% three- and four-prong, 20.5% QCD jets)"
# the random partitions of a class-pair figure (PI 2026-11-21, LEDGER PS-5): those that merge the pair and those that
# keep it apart, drawn separately (gallery_roc.PAIR_RAND)
NO_RAND = "The random partitions, one run each, are two curves, those merging the pair and those keeping it apart."
COUNTS = {}       # PC-jc2, PC-jc1: the computed count sentence (gallery_perclass), "" without per-class values
# a sentence that depends on the data read, {name} in a caption (gallery_physics sets "probe_note": the mass probe is
# centred on one training median, the truth-free refit atlas_res_v3, the only mass files read)
FILLS = {"probe_note": "", "xbs_widen": "", "init_range": ""}
# the paper's numbers: make_tables.py writes them before the gallery draws, in the paper and the book render jobs
GENERATED = pathlib.Path(__file__).resolve().parents[3] / "paper" / "journal" / "results_generated.tex"


def macro(name: str) -> str:
    """The body of the paper's macro name, as results_generated.tex defines it."""
    m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*?)\}\s*(?:%|$)", GENERATED.read_text(), re.M)
    if m is None:
        raise SystemExit(f"FATAL: {name} is not in {GENERATED}; run experiments/FIGS/make_tables.py first")
    return m.group(1)


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
        # PI 2026-11-20 (C2): the random network off the plot (its range in one clause), no breaks; the released Sophon
        # checkpoint on every panel with the same substitutions (2026-11-22); run counts stated; at most 135 words
        'With linear probes, 43 and 17 classes and every 17-class partition are less accurate on JetClass-II than 188 '
        'classes; no label set, mass network or partition differs from it in R30 by more than three standard '
        'deviations on top tagging (top-quark against QCD jets) or quark/gluon tagging (quark against gluon jets, '
        'trained on Pythia). Points are means ±1 standard deviation over the pretraining runs of each network, counted '
        'in figure {fig:F1}; the released Sophon checkpoint (one run) is open.{init_range} R30 is one over the fraction of background jets kept when the cut '
        'keeps 30% of signal jets, the top-tagging working point. Values are in table {tab:summary}.'
    ),
    "F2-ft": (
        'After fine-tuning, 17 classes stay below 188 classes on JetClass-II and make about 3% more errors on JetClass; '
        'among the pretrained networks, no network differs from 188 classes in R30 by more than three standard '
        'deviations on top tagging (top-quark against QCD jets) or quark/gluon tagging (quark against gluon jets, '
        'trained on Pythia). Points are means ±1 standard deviation over the pretraining runs of each network, counted '
        'in figure {fig:F1}; three trainings from scratch, the reference, are grey. R30 is one '
        'over the fraction of background jets kept when the cut keeps 30% of signal jets, the top-tagging working '
        'point. Values are in table {tab:summary}.'
    ),
    "PC-jc2": (
        "{counts} Each point is one JetClass-II class's recall, the share of its test jets assigned to it, for the "
        "row's network against 188 classes, as linear probes (left) or fine-tuned (right) on one million "
        "class-balanced jets; points below the diagonal are recalled less often. Their test-jet-weighted mean is the "
        "accuracy of figures {fig:F2} and {fig:F2-ft}. Means over three pretraining runs with ±1 standard deviation "
        "bars over runs and test jets; colors give what the 17 classes merge the class with. Values are in the data "
        "release."
    ),
    # PI 2026-11-23 (SP-13): the page captions of F4-mech, F4-other and F9b follow the paper's section captions (1db8bc37);
    # F4-mech's, at 141 words as written there, loses six connecting words to stay within 135 (CA-1); PI 2026-11-22
    # (LEDGER CA-1, counted as printed): the paper's drops "label sets above and 17-class partitions below", which
    # the figure's row names say, and so does this one
    "F4-mech": (
        'Linear probes on 17 classes, which merge each pair, separate X→bb̄ from cc̄ and X→bc from bq, cs, the '
        'second-named the background, worse than 188 classes, and the b-quark boundary recovers part of the loss on '
        'both. Panels show one over the fraction of background jets kept against that of signal jets, with ratios to '
        '188 classes. Curves average {m:MlstNRunsWord} runs per label set, {m:MlstMassNRunsWord} per mass network and '
        'flavor-boundary partition, and {m:RandVtwoRuns} per random partition, split into those merging the pair '
        '({m:MlstPairRandMergedBbN}, {m:MlstPairRandMergedBcN}) and keeping it apart ({m:MlstPairRandApartBbN}, '
        '{m:MlstPairRandApartBcN}, one partial merge left out). They end where under {m:MlstMinBkgPass} background '
        'jets pass, and 188 classes\' one-standard-deviation band combines run spread and binomial uncertainty. The '
        'random network, not drawn, reaches {m:MlstPairInitRhalfMechMin} to {m:MlstPairInitRhalfMechMax} at 50% of '
        'signal jets. Table {tab:tests} has the tested differences.'
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
        'mass network differs from 188 classes in R30 by more than three standard deviations. The panels show '
        'accuracy and R30, one over the fraction of background jets kept when the cut keeps 30% of signal jets, the '
        'top-tagging working point. Lines are means and light bands ±1 standard deviation over the pretraining runs '
        '(three a label set, two a mass network) or three trainings from scratch (grey); open stars mark the published '
        'Particle Transformer. The values at one million jets are in table {tab:summary}, the others in the data '
        'release.'
    ),
    "F9b": (
        'Each network with an added mass output resolves the jet mass better than the soft-drop mass in every bin of '
        'jet pT and of generator-level mass. Panels show the median of the response r = m_est/m_gen and its distance '
        'from one, and the resolution, the width of the central 68% of r over twice its median, and its ratio to the '
        'soft-drop mass, against the reconstructed jet pT and the generator-level soft-drop mass m_gen. The estimates '
        'are the soft-drop mass and the mass output of the 162- and 17-class networks, on JetClass-II test jets matched '
        'at generator level. Points are means over {m:MlstMassNRunsWord} runs, horizontal bars span each bin, and '
        'vertical bars of one standard deviation combine the spread over runs with the bootstrap over test jets. '
        'Values over all jets are in table {tab:mass}.'
    ),
    "A17-probe": (
        'On frozen features, a multilayer-perceptron probe resolves the jet mass better on the networks with an added '
        'mass regression (+ mass) than on the same label sets without it. The panels have the form of figure '
        '{fig:F9b}, for a probe trained to regress the jet mass from the frozen features of each label set, mass '
        'network and the random network (dashed), with the soft-drop mass in black. {probe_note} Lines are means over '
        'three pretraining runs, two for + mass, or three random initializations, flat across each bin; the bands, for '
        '188 classes, the random network and the soft-drop mass only, are ±1 standard deviation combining the spread '
        'over runs with the bootstrap over test jets. Values, and the other standard deviations, are in table {tab:mass}.'
    ),
    "F11": (
        'At 1% QCD efficiency, 17 classes keep fewer X→bb̄ and X→bs jets than 188 classes, a gap of 0.6 and 1.0 in the '
        'figure of merit{xbs_widen}. The figure of merit, signal efficiency over the square '
        'root of QCD efficiency, is shown for (a) X→bb̄ and (b) X→bs against QCD, all jets with soft-drop mass 150 to '
        '230 GeV and efficiencies relative to them. Solid curves are a multilayer-perceptron classifier on frozen '
        'features starting from the pretrained output head, grey curves training from scratch, and dash-dotted curves '
        'the X→bb̄ output against the QCD outputs (188, 162 and 162 classes + mass, in (a)). Bands, for 188 classes '
        'and from scratch only, are ±1 standard deviation over the pretraining runs (counted in figure {fig:F1}) or three trainings. Table '
        '{tab:summary} holds the classifier\'s values at 1% QCD efficiency.'
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
    # PI 2026-11-23 (LEDGER SP-9, merge 8): F4-light's two pairs are this figure's second row; the QCD pairs have no
    # partitions panel, as every partition merges the QCD classes
    "F4-other": (
        "With linear probes, 43 and 17 classes separate four pairs that 17 classes merge, X→YY→bbqq from ccqq, QCD bb̄ "
        "from cc̄, and QCD jets with a b quark, or a c and no b, from light QCD jets, the second-named the background, "
        "less well than 188 classes, and 162 classes, which merge the QCD classes, the QCD pairs. Panels, drawn as in "
        "figure {fig:F4-mech}, show the label sets and, in (e), the 17-class partitions, which all merge the QCD "
        "classes. Curves average {m:MlstNRunsWord} runs per label set, {m:MlstMassNRunsWord} per mass network and "
        "flavor-boundary partition, and {m:RandVtwoRuns} per random partition, split into those merging the pair "
        "({m:MlstPairRandMergedBbqqN}) and keeping it apart ({m:MlstPairRandApartBbqqN}). The random network, not "
        "drawn, reaches {m:MlstPairInitRhalfOtherMin} to "
        "{m:MlstPairInitRhalfOtherMax} at 50% of signal jets. Table {tab:tests} has the tested differences."
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
        "188 classes. The panels are those of figure {fig:F4-other} for networks fine-tuned on one million "
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
        'horizontal bar spanning it, and error bars combine the standard deviation over the pretraining runs (counted in figure {fig:F1}) '
        'with the binomial uncertainty over test jets, the three-standard-deviation test applied. Values are in the '
        'data release file jet_pt_bins.csv.'
    ),
    "A10": (
        'With linear probes, 43 and 17 classes are less accurate on JetClass-II than 188 classes at every training-set '
        'size. The panels show accuracy on class-balanced JetClass-II and JetClass and R30, one over the fraction of '
        'background jets kept when the cut keeps 30% of signal jets, the top-tagging working point, on top and '
        'quark/gluon tagging, Pythia and Herwig, against training-set size. Lines are means and light bands ±1 '
        'standard deviation over the pretraining runs (three a label set, two a mass network). The random network '
        'and the values at the largest training sets are in table {tab:summary}, the others in the data release.'
    ),
    "F10b": (
        'Used directly, no label set differs from 188 classes, nor a mass network from its own label set, in how much '
        'it distorts the QCD mass by more than three standard deviations at 10% or 1% QCD efficiency. '
        'The curves show '
        'the Jensen-Shannon distance (base 2) between the soft-drop mass distributions of all JetClass-II QCD test '
        'jets and those passing a cut on the summed QCD output, against QCD efficiency eps_B, of (a) the pretraining '
        'head used directly (dash-dotted) and (b) the networks fine-tuned on JetClass-II, with training from scratch '
        'in grey, on logarithmic axes. Bands, for 188 classes and from scratch only, are ±1 standard deviation over '
        'runs; the dotted line in each panel is a random selection of as many QCD jets. Values are in table {tab:sculpt}.'
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
        "compare the label sets. Each curve is the mean over the pretraining runs (counted in figure {fig:F1}), and the ±1 standard "
        "deviation bands over runs are too narrow to see."
    ),
}


# PI 2026-11-23 (LEDGER SP-9): the figures the main text and the appendix held before are in the data release (the
# supplement tier); their pages keep these captions, which may name any item
RELEASE = ('ROC-main', 'ROC-main-ft', 'A17-probe', 'F11-part', 'F4-mech-ft', 'F4-other-ft', 'F4-light-ft', 'PT', 'F10b', 'UMAP-frozen', 'UMAP-ft')
RELEASE_CAPTIONS = {k: CAPTIONS.pop(k) for k in RELEASE}


def fill(raw: str) -> str:
    """raw with each {tab:...} and {fig:<page id>} as that item's number, without the word (the sentence says
    "table" or "figure"), each {name} of FILLS as its sentence and each {m:Name} as the paper's macro Name."""
    for name, sentence in FILLS.items():
        raw = raw.replace("{" + name + "}", sentence)
    raw = re.sub(r"\{m:([A-Za-z]+)\}", lambda m: macro(m.group(1)), raw)
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
    return [dataclasses.replace(p, caption=text(p.id)) if p.id in CAPTIONS else
            dataclasses.replace(p, caption=" ".join(fill(RELEASE_CAPTIONS[p.id]).split())) if p.id in RELEASE_CAPTIONS
            else p for p in pages]
