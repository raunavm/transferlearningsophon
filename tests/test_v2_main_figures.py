"""The rewrite's main and appendix figures (experiments/FIGS/figstyle/v2_main.py; PI 2026-11-26, ledger 34.3, after
2026-11-24 RW-3, RW-5, RW-8, GO-6, GO-9): the list, the names, no flavor-boundary partition, the pair figure's two panels and six curves, the nesting of
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


STEMS = ["fig1_design", "fig2_accuracy", "fig3_classes", "fig5_resonance", "fig4_training_size", "figA1", "figA2",
         "figA6", "figA3", "figA7", "figA4", "figA5"]


def test_the_figures_of_the_list_and_no_flavor_boundary_partition():
    assert [stem for stem, *_ in V.FIGURES] == STEMS
    nets = V.SUMMARY_ROWS + V.PAIR_NETS + V.SIZE_NETS + V.XBX_NETS + V.ACC_ROWS + tuple(a for a, _ in V.RECALL_ROWS)
    assert not [n for n in nets if n.startswith("FLAV")]


def test_the_names_are_the_rewrites_and_the_gallery_keeps_its_own():
    before = dict(S.LEGEND)
    with V.names():
        assert S.LEGEND["RAND"] == "17 classes, random grouping" and S.LEGEND["SOPHON"] == "Sophon (released)"
        assert S.LEGEND["L162_MASS"] == "162 classes + mass regression" and S.LEGEND["SCRATCH"] == "trained from scratch"
        assert V.PH.DIRECT_TAG == "(output scores)"
    assert S.LEGEND == before and V.PH.DIRECT_TAG == "(direct discriminant)"


def test_no_label_of_any_figure_is_an_internal_name(figures):
    assert set(figures) == set(STEMS)
    bad = {stem: hits for stem, fig in figures.items() if (hits := [t for t in _texts(fig) if INTERNAL.search(t)])}
    assert not bad


def test_every_figure_fits_the_papers_page(figures):
    assert {stem: round(fig.get_figheight() / S.CM, 1) for stem, fig in figures.items()
            if fig.get_figheight() / S.CM > V.PAPER_MAX_H_CM} == {}


def test_fig1_nests_one_example_group_and_shows_one_flow(figures):
    texts = _texts(figures["fig1_design"])
    assert all(f"{n} classes" in texts for n in (188, 162, 43, 17))
    assert sum(bool(re.fullmatch(r"\$X\\!\\to [a-z]\S*( [a-z])?\$", t)) for t in texts) == 10      # its ten classes as jets
    assert r"$X\!\to$two quarks or gluons" in texts
    assert "the group drawn above" in texts and any(t.startswith("Each cell is one of the 188 classes") for t in texts)
    # ledger 34.3: one flow, network, features, then linear probe or fine-tuning, then the tasks; no row of squares
    assert {"Linear probe", "Fine-tuning", "Features"} <= set(texts) and "Frozen features" not in texts
    assert "Output\nlayer" not in texts and any(t.startswith("JetClass-II\nJetClass\nTop tagging") for t in texts)


def test_fig2_is_accuracy_against_the_label_set_with_the_tables_values(figures):
    fig = figures["fig2_accuracy"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass"]
    assert _legend(fig) == ["fine-tuned", "linear probe", "trained from scratch"]
    for ax, (key, n, ds, _) in zip(panels, V.ACC_PANELS):
        vals = V.accuracy_values(key, n, ds)
        parts = [ax] + [o for o in fig.axes if o.get_label() == "<break>" and o.get_shared_x_axes().joined(ax, o)]
        drawn = sorted(round(float(ln.get_ydata()[0]), 9) for a in parts for ln in a.lines
                       if ln.get_marker() == "o" and len(ln.get_ydata()) == 1)
        want = sorted(round(float(np.mean(v)), 9) for mode in ("probe", "ft") for v in vals[mode].values())
        assert drawn == want                                      # every point a table's mean, in percent
        assert 30 < min(want) and max(want) < 90
        assert len(parts) == (2 if key in V.ACC_STACKED else 1)
    # JetClass: the axis broken between the fine-tuned models (upper part) and the linear probes (lower part)
    upper = panels[1]
    lower = [o for o in fig.axes if o.get_label() == "<break>"][0]
    assert upper.get_ylim()[0] > lower.get_ylim()[1]


def test_fig3_shows_recall_and_the_pairs_of_classes(figures):
    fig = figures["fig3_classes"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) 43 classes", "(b) 17 classes", "(c) Pairs of classes"]
    c = panels[2]
    assert c.get_xscale() == "log" and c.get_xlabel().startswith(r"$1 - \mathrm{AUC}$")
    assert list(c.yaxis.get_major_formatter().seq) == [V._pair_title(k) for k in V.PAIRS_C]
    errs = V.pair_errors()
    assert "gg_vs_qq" in errs and set(errs["gg_vs_qq"]) == set(V.PAIR_NETS)        # ledger 33.2(b)'s pair included
    assert all("RAND_APART" not in errs[k] for k in V.PAIRS_C if k.startswith("qcd_"))   # every grouping merges them
    n_points = sum(len(e) for e in errs.values())
    assert len([ln for ln in c.lines if ln.get_linestyle() == "None"]) == n_points
    with V.names():
        assert _legend(fig)[-6:] == [V.R._name(n) for n in V.PAIR_NETS]


def test_fig4_resonance_has_ratio_panels_and_the_working_point(figures):
    fig = figures["fig5_resonance"]
    panels = _panels(fig)
    assert len(panels) == 2
    assert set(_legend(fig)) == {"188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch",
                                 "188 classes (output scores)", "162 classes (output scores)"}
    ratios = [ax for ax in fig.axes if ax.get_ylabel() == "Ratio"]
    assert len(ratios) == 2 and all(ax.get_yscale() == "linear" and ax.get_xlabel().startswith("Background") for ax in ratios)
    for ax in panels + ratios:          # a vertical line at 1% background efficiency in every panel
        assert any(np.allclose(ln.get_xdata(), 1e-2) for ln in ax.lines if ln.get_transform() == ax.get_xaxis_transform())
    for ax in panels:
        assert ax.get_yscale() == ax.get_xscale() == "log" and ax.get_ylabel().startswith("Signal efficiency")


def test_fig5_is_fine_tuning_on_the_two_multiclass_tasks_with_the_difference_to_188(figures):
    fig = figures["fig4_training_size"]
    assert _legend(fig) == ["188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass"]
    assert [ax.get_ylabel() for ax in panels] == ["Accuracy (%)"] * 2
    lowers = [ax for ax in fig.axes if ax.get_ylabel() == r"$\Delta$ (points)"]
    assert len(lowers) == 2 and all(len([ln for ln in ax.lines if len(ln.get_xdata()) >= 3]) == 3 for ax in lowers)
    a7 = _panels(figures["figA7"])
    assert [ax.get_title() for ax in a7] == ["(a) Top tagging", "(b) Quark/gluon"]
    assert len([ax for ax in figures["figA7"].axes if ax.get_ylabel() == "Ratio"]) == 2


def test_the_appendix_figures(figures):
    a1 = _panels(figures["figA1"])       # GO-9: linear probes only, one panel a coarser label set
    assert [ax.get_title() for ax in a1] == ["(a) 162 classes", "(b) 43 classes", "(c) 17 classes"]
    assert all(ax.get_xlabel() == "Recall (%), 188 classes" for ax in a1) and not figures["figA1"].texts
    # ledger 34.3: the pair curves of the old Fig. 3 join G2 (figA2, resonance pairs; figA6, QCD pairs), taller ratio
    # panels, each x range starting where the 188-class curve does
    ref = S.SERIES["L188"].lower()
    for stem, n in (("figA2", 3), ("figA6", 3)):
        fig = figures[stem]
        panels = _panels(fig)
        assert len(panels) == n
        for ax in panels:
            start = min(float(np.min(ln.get_xdata())) for ln in ax.lines if ln.get_color().lower() == ref)
            assert ax.get_xlim()[0] <= start < ax.get_xlim()[0] + 0.08
            assert all(float(np.min(ln.get_xdata())) >= ax.get_xlim()[0] for ln in ax.lines if len(ln.get_xdata()))
    names = ["188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
             "162 classes + mass regression", "17 classes + mass regression"]
    assert sorted(_legend(figures["figA3"])) == sorted(names) and len(_panels(figures["figA3"])) == 4
    assert [ax.get_ylabel() for ax in _panels(figures["figA3"])[:2]] == ["Accuracy (%)"] * 2
    assert set(_legend(figures["figA4"])) == {"soft-drop mass", "162 classes + mass regression",
                                              "17 classes + mass regression"}
    assert sorted(_legend(figures["figA5"])) == sorted(names)
    a5 = [ax for ax in figures["figA5"].axes if ax.lines][0]
    assert a5.get_ylabel() == "Validation accuracy (%)" and max(max(ln.get_ydata()) for ln in a5.lines) > 50


def test_the_render_mode_draws_them_all():
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
