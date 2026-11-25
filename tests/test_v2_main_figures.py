"""The rewrite's five main figures (experiments/FIGS/figstyle/v2_main.py; PI 2026-11-24, LEDGER RW-3, RW-5, RW-8,
GO-6): the list, the names, no flavor-boundary partition, the pair figure's two panels and six curves, the nesting of
one example group in Fig. 1, and the render mode that draws them. Draws the figures from the committed data."""
import importlib.util
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
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
                                                "fig5_resonance"]
    nets = V.SUMMARY_ROWS + V.PAIR_NETS + V.SIZE_NETS + V.XBX_NETS
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


def test_fig2_rows_are_table_2s_pretrained_rows_in_four_panels(figures):
    fig = figures["fig2_frozen"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass", "(c) Top tagging", "(d) Quark/gluon"]
    assert list(panels[0].yaxis.get_major_formatter().seq) == [
        "188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
        "162 classes + mass regression", "17 classes + mass regression", "Sophon (released)"]
    assert _legend(fig)[0] == "mean \u00b1 s.d. over runs"
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


def test_fig4_is_fine_tuning_against_training_size_beside_from_scratch(figures):
    fig = figures["fig4_training_size"]
    assert _legend(fig) == ["188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch"]
    panels = _panels(fig)
    assert len(panels) == 4 and all(ax.get_xlabel() == "Number of training jets" for ax in panels)
    assert all(len(ax.lines) == 5 and all(len(ln.get_xdata()) >= 3 for ln in ax.lines) for ax in panels)


def test_fig5_is_the_resonance_benchmark_with_the_output_scores_for_xbb(figures):
    fig = figures["fig5_resonance"]
    assert len(_panels(fig)) == 2
    legend = _legend(fig)
    assert set(legend) == {"188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch",
                           "188 classes (output scores)", "162 classes (output scores)"}


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
