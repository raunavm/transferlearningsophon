"""The rewrite's main and appendix figures (experiments/FIGS/figstyle/v2_main.py; PI 2026-11-26, ledger 34.3, after
2026-11-24 RW-3, RW-5, RW-8, GO-6, GO-9): the list, the names, no flavor-boundary partition, the pair figure's two panels and six curves, the nesting of
one example group in Fig. 1, and the render mode that draws them. Draws the figures from the committed data."""
import importlib.util
import pathlib
import re
import sys

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments/FIGS/figstyle"), str(ROOT / "experiments/FIGS")]
import style_v5 as S  # noqa: E402
import v2_main as V  # noqa: E402

# names of the old paper and of the code that no label of the rewrite may show (RW-1, RW-5)
INTERNAL = re.compile(r"partition|random network|checkpoint|direct discriminant|transfer head|"
                      r"\bL1\d\d\b|R16|R42|\+ mass$|^from scratch|\bSD\b|\$R_\{30\}\$")


@pytest.fixture(scope="module")
def figures():
    """{file stem: the figure as drawn}, caught at the save."""
    caught, save = {}, S.save
    S.save = lambda fig, path: caught.__setitem__(pathlib.Path(path).stem, fig)
    try:
        with V.names():
            for stem, draw, _, _ in V.FIGURES:
                V.v2_style()
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


STEMS = ["fig1_design", "fig2_accuracy", "fig_separation", "fig3_classes", "fig5_resonance", "fig4_training_size",
         "figA1", "figA2", "figA6", "figA3", "figA7", "figA4", "figA5"]


def test_the_figures_of_the_list_and_their_names():
    assert [stem for stem, *_ in V.FIGURES] == STEMS
    assert list(V.FIGURE_NAMES) == STEMS and len(set(V.FIGURE_NAMES.values())) == len(STEMS)


def test_the_names_and_font_are_the_rewrites_and_the_gallery_keeps_its_own():
    before, fonts = dict(S.LEGEND), {k: getattr(S, k) for k in V.FONT_KEYS}
    with V.names():
        assert S.LEGEND["RAND"] == "17 classes, random grouping" and S.LEGEND["SOPHON"] == "Sophon (released)"
        assert S.LEGEND["SCRATCH"] == "trained from scratch" and V.PH.DIRECT_TAG == "(output scores)"
        assert all(getattr(S, k) == V.FONT for k in V.FONT_KEYS)
    assert S.LEGEND == before and {k: getattr(S, k) for k in V.FONT_KEYS} == fonts


def test_no_label_of_any_figure_is_an_internal_name(figures):
    assert set(figures) == set(STEMS)
    bad = {stem: hits for stem, fig in figures.items() if (hits := [t for t in _texts(fig) if INTERNAL.search(t)])}
    assert not bad
    # ledger 35, section 4: the flavor-boundary groupings appear in the main text only in Figs. 2 and the separation one
    flav = {stem for stem, fig in figures.items() if any("boundary" in t for t in _texts(fig))}
    assert {s_ for s_ in flav if not s_.startswith("figA")} == {"fig2_accuracy", "fig_separation"}


def test_every_figure_fits_the_papers_page_and_prints_at_the_figure_font(figures):
    assert {stem: round(fig.get_figheight() / S.CM, 1) for stem, fig in figures.items()
            if fig.get_figheight() / S.CM > V.PAPER_MAX_H_CM} == {}
    small = {stem: sorted({round(t.get_fontsize(), 1) for t in fig.findobj(matplotlib.text.Text)
                           if t.get_text().strip() and t.get_fontsize() < V.FONT - 1e-6})
             for stem, fig in figures.items() if not stem.startswith("figA")}
    assert not {k: v for k, v in small.items() if v}


def test_fig1_is_a_tree_and_one_row_in_under_60_words(figures):
    fig = figures["fig1_design"]
    texts = _texts(fig)
    assert all(f"{n} classes" in texts for n in (188, 162, 43, 17))
    assert [t for t in texts if t in [V._class_tex(c) for c, _ in V.fig1_classes()]] == [V._class_tex(c) for c, _ in
                                                                                       V.fig1_classes()]
    assert {"Features", "Fine-tuning", "Frozen-feature\nclassifier", "JetClass-II jets"} <= set(texts)
    assert sum(len(t.split()) for t in texts) < 60 and V.fig1_words() < 60
    # ledger 35: no jet glyphs, no 188-cell strip, no notes
    assert not [t for t in texts if "group drawn" in t or "Each cell" in t or "prong" in t]
    assert len([p_ for ax in fig.axes for p_ in ax.patches if isinstance(p_, matplotlib.patches.Rectangle)]) < 40


def test_fig2_is_a_dot_plot_with_the_tables_values_and_no_joining_lines(figures):
    fig = figures["fig2_accuracy"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass"]
    for ax, (key, n, ds, _) in zip(panels, V.ACC_PANELS):
        vals = V.accuracy_values(key, n, ds)
        parts = [ax] + [o for o in fig.axes if o.get_label() == "<break>" and o.get_shared_x_axes().joined(ax, o)]
        assert len(parts) == 2
        for part, (mode, _, filled) in zip(parts, V.ACC_MODES):
            means = sorted(round(float(ln.get_ydata()[0]), 9) for ln in part.lines
                           if ln.get_marker() == "o" and len(ln.get_ydata()) == 1 and ln.get_markersize() == V.MEAN_DOT)
            assert means == sorted(round(float(np.mean(v)), 9) for v in vals[mode].values())
            # no line joins two columns: every drawn line is a single point, a set of run dots, or a guide
            joined = [ln for ln in part.lines if ln.get_linestyle() not in ("None", "none") and ln.get_transform() == part.transData]
            assert not joined
            runs = sum(len(v) for v in vals[mode].values())
            assert sum(len(ln.get_xdata()) for ln in part.lines if ln.get_markersize() == V.RUN_DOT) == runs
    assert _legend(fig)[-1] == "trained from scratch"
    # draft_2ab77ad: short horizontal ticks under one family label (the caption names the three groupings)
    ticks = [t for t in [o for o in fig.axes if o.get_label() == "<break>"][0].get_xticklabels()]
    assert {"none", "$b$", "random"} <= {t.get_text() for t in ticks} and all(t.get_rotation() == 0 for t in ticks)
    fam = [t for t in fig.texts if t.get_text() in ("nested", "random\ngrouping", "flavor\nboundary")]
    assert len(fam) == 6 and len({round(t.get_position()[1], 6) for t in fam}) == 1        # one baseline
    assert fig.get_figheight() / S.CM < 9.6


def test_separation_figure_draws_the_twelve_label_sets_of_the_build(figures):
    fig = figures["fig_separation"]
    ax = [a for a in fig.axes if a.lines][0]
    pts = sorted((round(float(ln.get_xdata()[0]), 6), round(float(ln.get_ydata()[0]), 6)) for ln in ax.lines
                 if len(ln.get_xdata()) == 1)
    assert pts == sorted((round(s, 6), round(a, 6)) for _, _, s, a, _ in V.separation_rows()) and len(pts) == 12
    with V.names():
        assert [" ".join(t.split()).replace(", one run each", "") for t in _legend(fig)] == [S.LEGEND[k] for k in V.SEP_ORDER]
        assert _legend(fig)[V.SEP_ORDER.index("RAND")].endswith("one run each")       # review of 4c045c9
    assert not fig.texts and not [t for t in ax.texts]                     # rho is in the caption, not the plot


def test_fig3_shows_recall_by_merge_kind_and_the_pairs(figures):
    fig = figures["fig3_classes"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) Per-class recall", "(b) Pairs of classes"]
    a, c = panels
    data = V.recall_by_merge()
    import importlib
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "experiments/FIGS"))
    RV = importlib.import_module("v2_rev35")
    import json
    docs = {"lp": json.loads((V.C.P.DATA / "perclass_recall/probe_jc2.json").read_text())}
    macros, _ = RV.recall_by_merge(V.M, ROOT, docs, V.C.RUNS)
    for arm, w in (("R42_Q1", "Mid"), ("R16_Q1", "Coarse")):     # the build's VRecall macros
        for k, _ in V.MERGE_KINDS:
            assert f"{data[arm][k][0]:.1f}" == macros[f"VRecall{w}{k}"] and str(data[arm][k][2]) == macros[f"VRecall{w}{k}N"]
    pts = sorted(round(float(ln.get_ydata()[0]), 9) for ln in a.lines if len(ln.get_ydata()) == 1)
    assert pts == sorted(round(m, 9) for arm in data for m, _, _ in data[arm].values())
    errs = V.pair_errors()
    assert c.get_xscale() == "log" and c.get_xlabel() == r"$1 - \mathrm{AUC}$, relative to 188 classes"
    assert list(c.yaxis.get_major_formatter().seq) == [f"{V._pair_title(k)} ({np.mean(errs[k]['L188']):.2g})"
                                                       for k in V.PAIRS_C]
    assert V.WITHIN_FADED not in _legend(fig)                    # ledger 35: not in the main figure
    # review of 4c045c9, F09: (a)'s legend lists only the 43- and 17-class markers, (b)'s the pair markers
    with V.names():
        assert [t.get_text() for t in a.get_legend().get_texts()] == [S.LEGEND["R42_Q1"], S.LEGEND["R16_Q1"]]
    assert len(fig.legends) == 1 and fig.get_figheight() / S.CM < 15.5          # draft_2ab77ad: (a) across the width
    assert V.pair_status("bc_vs_bq_cs", "R42_Q1") == "partly"
    assert len([ln for ln in c.lines if ln.get_fillstyle() == "left"]) == sum(
        V.pair_status(k, n) == "partly" for k in V.PAIRS_C for n in errs[k])


def test_fig4_resonance_is_rejection_against_signal_efficiency(figures):
    fig = figures["fig5_resonance"]
    panels = _panels(fig)
    assert len(panels) == 2
    assert set(_legend(fig)) == {"188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch",
                                 "output scores"} and len(_legend(fig)) <= 6
    for ax in panels:
        assert ax.get_yscale() == "log" and ax.get_ylabel().startswith("Rejection")
        assert any(np.allclose(ln.get_ydata(), 100.0) for ln in ax.lines if ln.get_transform() == ax.get_yaxis_transform())
    ratios = [ax for ax in fig.axes if ax.get_ylabel() == "Ratio"]
    assert len(ratios) == 2 and all(ax.get_xlabel().startswith("Signal efficiency") for ax in ratios)


def test_fig5_is_fine_tuning_on_the_two_multiclass_tasks_with_the_difference_to_188(figures):
    fig = figures["fig4_training_size"]
    assert _legend(fig) == ["188 classes", "162 classes", "43 classes", "17 classes", "trained from scratch"]
    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["(a) JetClass-II", "(b) JetClass"]
    lowers = [ax for ax in fig.axes if ax.get_ylabel() == r"$\Delta$ (points)"]
    assert len(lowers) == 2


def test_the_appendix_figures(figures):
    a1 = _panels(figures["figA1"])       # ledger 35: the full recall scatters, moved from Fig. 3
    assert [ax.get_title() for ax in a1] == ["(a) 162 classes", "(b) 43 classes", "(c) 17 classes"]
    assert V.WITHIN_FADED in _legend(figures["figA1"])
    ref = S.SERIES["L188"].lower()
    for stem in ("figA2", "figA6"):
        for ax in _panels(figures[stem]):
            start = min(float(np.min(ln.get_xdata())) for ln in ax.lines if ln.get_color().lower() == ref)
            assert all(float(np.min(ln.get_xdata())) >= start - 1e-9 for ln in ax.lines if len(ln.get_xdata()) > 1)
    a5 = [ax for ax in figures["figA5"].axes if ax.lines][0]
    chosen = V.chosen_epochs()
    ticks = [ln for ln in a5.lines if ln.get_marker() == "|"]
    runs = V.M._pooled(V.C.RUNS)
    assert sorted(int(ln.get_xdata()[0]) for ln in ticks) == sorted(chosen[m] + 1 for r in V.SUMMARY_ROWS for m in runs[r])


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
