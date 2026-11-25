"""The rewrite's five main and five appendix figures (experiments/FIGS/figstyle/v2_main.py; PI 2026-11-24, LEDGER
RW-3, RW-5, RW-8, GO-6, GO-9): the list, the names, no flavor-boundary partition, the pair figure's two panels and six curves, the nesting of
one example group in Fig. 1, and the render mode that draws them. Draws the figures from the committed data."""
import importlib.util
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments/FIGS/figstyle"), str(ROOT / "experiments/FIGS")]
import style_v5 as S  # noqa: E402
import v2_main as V  # noqa: E402

# names of the old paper and of the code that no label of the rewrite may show (RW-1, RW-5)
INTERNAL = re.compile(r"partition|flavor boundary|random network|checkpoint|direct discriminant|transfer head|"
                      r"\bL1\d\d\b|R16|R42|\+ mass$|^from scratch|\bSD\b|\$R_\{30\}\$")


@pytest.fixture(scope="module")
def figures():
    """{file stem: the figure as drawn}, caught at the save."""
    caught, save = {}, S.save
    S.save = lambda fig, path: caught.__setitem__(pathlib.Path(path).stem, fig)
    try:
        with V.names():
            for stem, draw, _, _ in V.FIGURES:
                S.use_style5(simplify=True)
                draw(pathlib.Path(stem))
    finally:
        S.save = save
    return caught


def _texts(fig) -> list:
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]


def _legend(fig) -> list:
    return [t.get_text() for leg in fig.legends for t in leg.get_texts() if t.get_text()]


def _panels(fig) -> list:
    return [ax for ax in fig.axes if re.match(r"\([a-z]\) ", ax.get_title())]


def test_the_five_figures_of_the_list_and_no_flavor_boundary_partition():
    assert [stem for stem, *_ in V.FIGURES] == ["fig1_design", "fig2_frozen", "fig3_pairs", "fig4_training_size",
                                                "fig5_resonance", "figA1", "figA2", "figA3", "figA4", "figA5"]
    nets = V.SUMMARY_ROWS + V.PAIR_NETS + V.SIZE_NETS + V.XBX_NETS + tuple(a for a, _ in V.RECALL_ROWS)
    assert not [n for n in nets if n.startswith("FLAV")]
    assert [p[0] for p in V.SUMMARY_PANELS] == [p[0] for p in V.SIZE_PANELS] == ["jc2", "jc1", "top", "qg"]


def test_the_names_are_the_rewrites_and_the_gallery_keeps_its_own():
    before = dict(S.LEGEND)
    with V.names():
        assert S.LEGEND["RAND"] == "17 classes, random grouping" and S.LEGEND["SOPHON"] == "Sophon (released)"
        assert S.LEGEND["L162_MASS"] == "162 classes + mass regression" and S.LEGEND["SCRATCH"] == "trained from scratch"
        assert V.PH.DIRECT_TAG == "(output scores)"
    assert S.LEGEND == before and V.PH.DIRECT_TAG == "(direct discriminant)"


def test_no_label_of_any_figure_is_an_internal_name(figures):
    assert set(figures) == {stem for stem, *_ in V.FIGURES}
    bad = {stem: hits for stem, fig in figures.items() if (hits := [t for t in _texts(fig) if INTERNAL.search(t)])}
    assert not bad


def test_fig1_nests_one_example_group_through_the_four_label_sets(figures):
    texts = _texts(figures["fig1_design"])
    assert all(f"{n} classes" in texts for n in (188, 162, 43, 17))
    assert sum(bool(re.fullmatch(r"\$X\\!\\to [a-z]\S*( [a-z])?\$", t)) for t in texts) == 10      # its ten classes as jets
    assert r"$X\!\to$two quarks or gluons" in texts
    assert "Output\nlayer" in texts and "Head" not in texts and "Fine" not in texts
    # round-2 cold readers: the strip says what a cell and a bar are, and outlines the group drawn above
    assert "the group drawn above" in texts and any(t.startswith("Each cell is one of the 188 classes") for t in texts)


def test_fig2_rows_are_table_2s_pretrained_rows_in_four_panels(figures):
    fig = figures["fig2_frozen"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass", "(c) Top tagging", "(d) Quark/gluon"]
    assert list(panels[0].yaxis.get_major_formatter().seq) == [
        "188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
        "162 classes + mass regression", "17 classes + mass regression", "Sophon (released)"]
    assert _legend(fig)[0] == "mean \u00b1 s.d. over runs"
    # round-2 cold readers: the mean's point smaller and its bar heavier than the gallery's, and the gallery unchanged
    assert V.G.MEAN_DOT == 4.6 and V.L.BAR == 0.8
    bars = [c for ax in panels for c in ax.collections if hasattr(c, "get_segments") and c.get_segments()]
    assert bars and all(abs(c.get_linewidth()[0] - V.FIG2_BAR) < 1e-9 for c in bars)
    # PI 2026-11-25 (ledger 32.3): accuracy in percent, and each panel its own range: the JetClass gap visible
    assert [ax.get_xlabel() for ax in panels[:2]] == ["Accuracy (%), JetClass-II", "Accuracy (%), JetClass"]
    lo, hi = panels[1].get_xlim()
    jc1 = [float(np.mean(v)) for v in ([V.G._hv_values(r, V._pct(V.G._probe_value("jc1", 1_000_000, "jc1", "accuracy")),
                                                      ("jc1", 1_000_000, "jc1", "accuracy")) for r in ("L188", "R16_Q1")])]
    assert 70 < lo < hi < 80 and (jc1[0] - jc1[1]) / (hi - lo) > 0.2      # the 188-17 gap a fifth of the axis or more
    assert [ax.get_xlabel() for ax in panels[2:]] == [r"$\mathrm{Rej}_{30\%}$, top tagging",
                                                      r"$\mathrm{Rej}_{30\%}$, quark/gluon"]


def test_fig3_has_two_panels_and_six_curves(figures):
    fig = figures["fig3_pairs"]
    panels = _panels(fig)
    assert len(panels) == 2 and len(V.PAIR_NETS) == 6
    assert _legend(fig) == ["188 classes", "162 classes", "43 classes", "17 classes",
                            "17 classes, random grouping, pair merged", "17 classes, random grouping, pair apart"]
    colours = {S.SERIES[V.R._style(n)].lower() for n in V.PAIR_NETS}
    for ax in panels:
        assert len({ln.get_color().lower() for ln in ax.lines} & colours) == 6     # every curve drawn, none else
    # every curve its band in the ratio panels (the cold readers of draft e6e3b22)
    ratios = [ax for ax in fig.axes if ax.get_ylabel().startswith("Ratio")]
    assert len(ratios) == 2 and all(len(ax.collections) == 6 for ax in ratios)


def test_fig4_is_fine_tuning_against_training_size_beside_from_scratch(figures):
    fig = figures["fig4_training_size"]
    assert _legend(fig) == ["188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch"]
    panels = _panels(fig)
    assert len(panels) == 4 and all(ax.get_xlabel() == "Number of training jets" for ax in panels)
    assert [ax.get_ylabel() for ax in panels[:2]] == ["Accuracy (%)"] * 2
    assert 40 < max(panels[0].lines[-1].get_ydata()) < 50       # 188 classes, JetClass-II, in percent
    assert all(len(ax.lines) == 5 and all(len(ln.get_xdata()) >= 3 for ln in ax.lines) for ax in panels)


def test_fig5_is_the_resonance_benchmark_with_the_output_scores_for_xbb(figures):
    fig = figures["fig5_resonance"]
    assert len(_panels(fig)) == 2
    legend = _legend(fig)
    assert set(legend) == {"188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch",
                           "188 classes (output scores)", "162 classes (output scores)"}
    # the axes the text quotes: signal efficiency at a background efficiency, every curve with its band
    for ax in _panels(fig):
        assert ax.get_ylabel().startswith("Signal efficiency") and ax.get_xlabel().startswith("Background efficiency")
        assert ax.get_yscale() == ax.get_xscale() == "log" and len(ax.collections) == len(ax.lines)
        assert max(float(max(ln.get_ydata())) for ln in ax.lines) <= 1.0 + 1e-9


def test_the_appendix_figures_keep_the_stems_the_appendix_includes(figures):
    appendix = (ROOT / "paper/journal/v2/appendix.tex").read_text()
    assert re.findall(r"\\vfig(?:\[[^]]*\])?\{(figA\d)\}", appendix) == ["figA1", "figA2", "figA3", "figA4", "figA5"]
    a1 = _panels(figures["figA1"])       # GO-9: linear probes only, one panel a coarser label set
    assert [ax.get_title() for ax in a1] == ["(a) 162 classes", "(b) 43 classes", "(c) 17 classes"]
    assert all(ax.get_xlabel() == "Recall (%), 188 classes" for ax in a1) and not figures["figA1"].texts
    assert len(_panels(figures["figA2"])) == 5 and set(_legend(figures["figA2"])) == set(_legend(figures["fig3_pairs"]))
    # ledger 33.2(b): X->gg vs X->qq as AUCs per model from the build's export, in the fifth panel
    e = _panels(figures["figA2"])[4]
    assert e.get_title().startswith("(e) $X\\to gg$ vs $q\\bar{q}$") and e.get_xlabel().startswith("AUC")
    rows = V.gg_qq_values()
    assert [n for n, _, _ in rows] == ["188 classes", "162 classes", "43 classes", "17 classes",
                                       "17 classes, random grouping, pair merged", "17 classes, random grouping, pair apart"]
    assert [len(v) for _, _, v in rows] == [3, 3, 3, 3, 1, 4]
    means = sorted(float(ln.get_xdata()[0]) for ln in e.lines if ln.get_marker() == "o")
    assert np.allclose(means, sorted(float(np.mean(v)) for _, _, v in rows))
    names = ["188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
             "162 classes + mass regression", "17 classes + mass regression"]
    assert sorted(_legend(figures["figA3"])) == sorted(names) and len(_panels(figures["figA3"])) == 4
    assert all(len(ax.lines) == 7 for ax in _panels(figures["figA3"]))
    assert set(_legend(figures["figA4"])) == {"soft-drop mass", "162 classes + mass regression",
                                              "17 classes + mass regression"}
    assert len(_panels(figures["figA4"])) == 4
    assert sorted(_legend(figures["figA5"])) == sorted(names)
    # accuracy in percent on every accuracy axis, recall too (ledger 32.3)
    assert all(ax.get_xlabel() == "Recall (%), 188 classes" for ax in a1)
    assert [ax.get_ylabel() for ax in _panels(figures["figA3"])[:2]] == ["Accuracy (%)"] * 2
    a5 = [ax for ax in figures["figA5"].axes if ax.lines][0]
    assert a5.get_ylabel() == "Validation accuracy (%)" and max(max(ln.get_ydata()) for ln in a5.lines) > 50


def test_the_render_mode_draws_them_and_stops_without_all_five():
    spec = importlib.util.spec_from_file_location("build_render_job_v2figs_t", ROOT / "scripts/build_render_job.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    job = B.SPEC.format(short="abc1234", commit="abc1234" * 5, image=B.IMAGE, pins=B.PINS, tectonic=B.TECTONIC, line=90,
                        **B.V2FIGS)
    assert "name: render-v2figs-abc1234-raunav" in job and "backoffLimit: 0" in job
    assert "python3 experiments/FIGS/figstyle/v2_main.py --check" in job
    assert 'grep -q "^V2 FIGURES COMPLETE" ${OUT}/figures.txt || exit 3' in job
    assert "tectonic -X compile figures_sheet.tex" in job
    sheet = (ROOT / "paper/journal/v2/figures_sheet.tex").read_text()
    assert [m for m in re.findall(r"\\mainfig\{\d+cm\}\{(\w+)\}", sheet)] == [stem for stem, *_ in V.FIGURES]
    for caption in re.findall(r"\\mainfig\{\d+cm\}\{\w+\}\{(.*?)\}\n(?=\\mainfig|\\end)", sheet, re.S):
        assert len(caption.split()) <= 80, caption[:60]      # RW-8
