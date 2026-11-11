"""The PI's line-graph comparison (2026-11-11): Fig. 7 (A8), Fig. A4 (PT) and Fig. 2 (F2a) in styles B and C
(lines_v6; style A is each figure's own page), and one panel of Fig. A4 in the current font and in Latin Modern, the
book's body font. Gallery only: nothing here enters the book until the PI picks a style and a font."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

import gallery_labelset as LS
import gallery_physics as PH
import lines_v6 as L
from gallery_common import HERE, Page, S

FONTS = HERE / "fonts"          # lmroman10 regular, italic, bold (GUST Font License), from the TeX distribution
LM = "Latin Modern Roman"
B_NOTE = " Style B: " + L.OFFSET_TEXT


def latin_modern():
    """rc settings that set every text of a figure in Latin Modern, mathematics in Computer Modern (its source)."""
    for f in sorted(FONTS.glob("lmroman10-*.otf")):
        font_manager.fontManager.addfont(str(f))
    return plt.rc_context({"font.family": LM, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})


def _font_panel(path, lm: bool) -> None:
    """Panel (a) of Fig. A4 (JetClass-II, linear probes, accuracy against jet pT) at its printed size in the book."""
    jc2 = PH.M._load(PH.PERF / "jc2")
    if not jc2:
        raise PH.Absent("no performance against jet pT (perf_vs_pt/jc2)")
    w, h, left, top, bottom = 6.25, 0.85 * 6.25, 1.25, 1.6, 1.1      # the legend above the panel, off the bands
    fig = S.figure_cm(left + w + 0.3, top + h + bottom)
    ax = S.axes_cm(fig, left, top, w, h)
    S.title(ax, "(a) JetClass-II, linear probes")
    S.fine_grid(ax)
    PH._pt_panel(ax, jc2, S.LEVELS, PH._accuracy, "Accuracy")
    S.labels(ax, xlabel=PH.X_PT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, hi=ax.get_xlim()[1]: "" if v >= hi - 1e-9 else f"{v:g}"))
    S.fig_legend(fig, ax.get_legend_handles_labels()[0], top_cm=0.1, ncol=2)
    S.save(fig, path)


def draw_font_current(path) -> None:
    _font_panel(path, False)


def draw_font_lm(path) -> None:
    with latin_modern():
        _font_panel(path, True)


F7 = ("Fine-tuned networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, and a network trained from "
      "scratch (dashed), on class-balanced JetClass-II and JetClass (accuracy) and top and quark/gluon tagging (R30), "
      "against the number of training jets. Points are means over three pretraining runs; ")
A4 = ("Networks pretrained on JetClass-II, in bins of jet transverse momentum: linear-probe accuracy on JetClass-II for "
      "the label sets (a) and the controls (b), JetClass accuracy after fine-tuning (c), and the probes' X→bb̄ "
      "against X→cc̄ AUC (d). Points are means at the bin centres, bars the SD over pretraining runs, no lines.")
F2 = ("Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and "
      "the 17-class controls, scored on class-balanced JetClass-II and JetClass (accuracy) and top and quark/gluon "
      "tagging (R30), against the pretraining label set. Markers are means, bars the SD over pretraining runs; dashed, "
      "trained from scratch.")
DIFFERS = "The PI's style comparison (2026-11-11); gallery only."
PAGES = [
    Page(id="CMP-F7-B", title="Fig. 7, style B (error bars)", precedent=[], precedent_label="", differs=DIFFERS, caption=F7 + "bars the SD." + B_NOTE,
          draw=lambda path: LS.draw_a8(path, style="B")),
    Page(id="CMP-F7-C", title="Fig. 7, style C (light bands)", precedent=[], precedent_label="", differs=DIFFERS, caption=F7 + "bands the SD at 15% opacity.",
          draw=lambda path: LS.draw_a8(path, style="C")),
    Page(id="CMP-A4-B", title="Fig. A4, style B", precedent=[], precedent_label="", differs=DIFFERS, caption=A4 + " Within a bin the series are set slightly apart.",
          draw=lambda path: PH.draw_pt(path, style="B")),
    Page(id="CMP-A4-C", title="Fig. A4, style C (binned: as B)", precedent=[], precedent_label="", differs=DIFFERS, caption=A4 + " Within a bin the series are set slightly apart.",
          draw=lambda path: PH.draw_pt(path, style="C")),
    Page(id="CMP-F2-B", title="Fig. 2, style B (categorical)", precedent=[], precedent_label="", differs=DIFFERS, caption=F2,
          draw=lambda path: LS.draw_f2a(path, style="B")),
    Page(id="CMP-F2-C", title="Fig. 2, style C (categorical: as B)", precedent=[], precedent_label="", differs=DIFFERS, caption=F2,
          draw=lambda path: LS.draw_f2a(path, style="C")),
    Page(id="CMP-F2-DOT", title="Fig. 2 as a horizontal dot plot", precedent=[], precedent_label="", differs=DIFFERS,
          caption=F2.replace("against the pretraining label set", "for each pretraining label set (rows)")
          .replace("bars the SD", "horizontal bars the SD").replace("dashed,", "vertical dashed line,"),
          draw=lambda path: LS.draw_f2a_dot(path)),
    Page(id="CMP-F3-DOT", title="Fig. 3 as a horizontal dot plot", precedent=[], precedent_label="", differs=DIFFERS,
          caption=F2.replace("Linear probes on the frozen features of networks", "Fine-tuned networks")
          .replace("against the pretraining label set", "for each pretraining label set (rows)")
          .replace("bars the SD", "horizontal bars the SD").replace("dashed,", "vertical dashed line,"),
          draw=lambda path: LS.draw_f2b_dot(path)),
    Page(id="CMP-F2-REL", title="Fig. 2 relative to 188 classes", precedent=[], precedent_label="", differs=DIFFERS,
          caption="For each pretraining label set (rows) and the 17-class controls, the change in linear-probe performance from the network pretrained with 188 classes and the same seed, in percent: class-balanced JetClass-II and JetClass accuracy, top and quark/gluon R30. Markers are means, horizontal bars the SD over seeds (two for the flavor boundaries; random partitions one run each). The vertical line is 188 classes.",
          draw=lambda path: LS.draw_f2a_rel(path)),
    Page(id="CMP-F3-REL", title="Fig. 3 relative to 188 classes", precedent=[], precedent_label="", differs=DIFFERS,
          caption="For each pretraining label set (rows) and the 17-class controls, the change in fine-tuned performance from the network pretrained with 188 classes and the same seed, in percent: class-balanced JetClass-II and JetClass accuracy, top and quark/gluon R30. Markers are means, horizontal bars the SD over seeds (two for the flavor boundaries; random partitions one run each). The vertical line is 188 classes.",
          draw=lambda path: LS.draw_f2b_rel(path)),
    Page(id="CMP-F2-HEAT", title="Fig. 2 as a table of changes", precedent=[], precedent_label="", differs=DIFFERS,
          caption="For each pretraining label set (rows) and the 17-class controls, the change in linear-probe performance from the network pretrained with 188 classes and the same seed, in percent, mean and SD over seeds, for the four tasks (columns). Red: worse than 188 classes, blue: better, on one scale; random partitions: the mean over the five, one run each.",
          draw=lambda path: LS.draw_f2a_heat(path)),
    Page(id="CMP-F3-HEAT", title="Fig. 3 as a table of changes", precedent=[], precedent_label="", differs=DIFFERS,
          caption="For each pretraining label set (rows) and the 17-class controls, the change in fine-tuned performance from the network pretrained with 188 classes and the same seed, in percent, mean and SD over seeds, for the four tasks (columns). Red: worse than 188 classes, blue: better, on one scale; random partitions: the mean over the five, one run each.",
          draw=lambda path: LS.draw_f2b_heat(path)),
    Page(id="CMP-FONT-T", title="Fig. A4 (a), current font", precedent=[], precedent_label="", differs=DIFFERS, caption="Panel (a) of Fig. A4 in the current Times-like font (STIX), at "
          "its printed size, for comparison with the same panel in Latin Modern, the body font of the book. Accuracy of "
          "linear probes on class-balanced JetClass-II against jet transverse momentum for the four label sets.",
          draw=draw_font_current),
    Page(id="CMP-FONT-LM", title="Fig. A4 (a), Latin Modern", precedent=[], precedent_label="", differs=DIFFERS, caption="Panel (a) of Fig. A4 in Latin Modern, the body font of the book, "
          "at its printed size, for comparison with the same panel in the current Times-like font (STIX). Accuracy of "
          "linear probes on class-balanced JetClass-II against jet transverse momentum for the four label sets.",
          draw=draw_font_lm),
]
