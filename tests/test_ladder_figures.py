"""The ladder figures: generated from the JSONs, or not shipped.

A figure is the part of the paper nobody diffs, so the tests here are about
provenance rather than pixels: the script must run end to end on a synthetic
ladder, it must not contain a number that came from a person, a saturated cell
must never become a point, and the palette must survive a greyscale printer.

The fixture builder is shared with tests/test_make_tables.py -- one synthetic
ladder, so the two halves of the pipeline are tested against the same schema.
"""
import ast
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


F = _mod("make_ladder_figures", REPO / "experiments/FIGS/make_ladder_figures.py")
S = _mod("figs_style", REPO / "experiments/FIGS/style.py")
T = _mod("fixture_tables", REPO / "tests/test_make_tables.py")

# Plotting calls whose positional arguments carry data. A literal there is a
# number somebody typed, which is the thing this pipeline exists to remove.
# 0 and 1 stay allowed: a reference line at zero is not a result.
DATA_CALLS = {"plot", "errorbar", "fill_between", "scatter", "bar", "hlines", "vlines",
              "axhline", "axvline", "axvspan", "axhspan", "set_xticks", "set_yticks",
              "set_xlim", "set_ylim"}
ALLOWED = {0, 1}


@pytest.fixture
def ladder(tmp_path):
    files = T.write_ladder(tmp_path)
    analysis = T.write_analysis(tmp_path, files)
    rung_map = T.write_rung_map(tmp_path)
    return {"root": tmp_path, "analysis": analysis, "rung_map": rung_map,
            "A": json.loads(analysis.read_text())}


def test_both_figures_are_written_end_to_end_from_the_fixture(ladder, tmp_path):
    out = tmp_path / "figs"
    assert F.main(["--analysis", str(ladder["analysis"]), "--rung-map", str(ladder["rung_map"]),
                   "--outdir", str(out)]) == 0
    stems = ["probe_ladder_granularity", "probe_ladder_paired_linear", "probe_ladder_paired_mlp"]
    for stem in stems:
        for ext in ("pdf", "png"):
            assert (out / f"{stem}.{ext}").stat().st_size > 5000, f"{stem}.{ext} is empty"


def test_no_number_in_the_figure_script_reaches_a_plotting_call():
    """Every coordinate must come from a file. This greps the syntax tree rather
    than the text, so a literal cannot hide inside a keyword argument list."""
    tree = ast.parse((REPO / "experiments/FIGS/make_ladder_figures.py").read_text())
    bad = []

    def numeric(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value not in ALLOWED
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            return all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                       for e in node.elts) and \
                any(e.value not in ALLOWED for e in node.elts)
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        args = []
        if node.func.attr in DATA_CALLS:
            args = list(node.args)
        elif node.func.attr == "annotate":
            args = [k.value for k in node.keywords if k.arg == "xy"]
        bad += [f"line {node.lineno}: {node.func.attr}" for a in args if numeric(a)]
    assert not bad, f"hard-coded numbers in plotting calls: {bad}"


def test_a_saturated_cell_is_a_bound_and_never_a_point(ladder):
    """The fixture saturates one task at its two finest vocabularies, which is
    what the real electron/muon task does. Those cells must leave the line."""
    A = ladder["A"]
    levels = A["levels_fine_to_coarse"]
    s = F.series(A, "beta", "linear", levels)
    assert s["bounds"] == levels[:2]
    assert s["x"] == levels[2:]
    assert not any(x in s["bounds"] for x, _ in s["points"])
    clean = F.series(A, "alpha", "linear", levels)
    assert clean["bounds"] == [] and clean["x"] == levels


def test_the_paired_figure_reads_the_stored_interval_and_flips_its_sign_once(ladder):
    """Half the pairs against the reference store it as the COARSER member, so
    their difference and interval are negated. If that flip were wrong, the
    figure would show the reference winning every contrast it actually loses."""
    A = ladder["A"]
    reference = A["levels_fine_to_coarse"][1]
    rows = F.paired_rows(A, "alpha", "linear", reference)
    assert [r["level"] for r in rows] == [lv for lv in A["levels_fine_to_coarse"]
                                          if lv != reference]
    stored = {(p["fine"], p["coarse"]): p for p in A["pairwise_exploratory"]["alpha"]["linear"]}
    finer = next(r for r in rows if r["level"] > reference)     # reference is the coarser here
    assert finer["mean"] == pytest.approx(-stored[(finer["level"], reference)]["mean_diff"])
    assert finer["ci95"] == sorted(-x for x in stored[(finer["level"], reference)]["ci95"])
    coarser = next(r for r in rows if r["level"] < reference)
    assert coarser["mean"] == pytest.approx(stored[(reference, coarser["level"])]["mean_diff"])
    for r in rows:
        assert len(r["diffs"]) == len(A["seeds_used"])


def test_per_seed_points_that_do_not_reproduce_the_stored_mean_stop_the_run(ladder):
    """The one way a paired figure goes silently wrong is a mismatched pairing:
    right-looking points, an interval from a different contrast."""
    A = ladder["A"]
    A["pairwise_exploratory"]["alpha"]["linear"][0]["mean_diff"] += 1.0
    with pytest.raises(SystemExit) as e:
        F.paired_rows(A, "alpha", "linear", A["levels_fine_to_coarse"][1])
    assert "do not reproduce" in str(e.value)


def test_the_merge_annotation_is_computed_from_the_label_map(ladder):
    """Where a distinction disappears is read off the tree, not typed in."""
    sizes = F.vocabulary_sizes(ladder["rung_map"])
    names, collapsed = F.task_names(ladder["analysis"].parent.parent)
    merges = F.merge_levels(ladder["rung_map"], names, sizes)
    # alpha's two classes share an R42_Q1 group; beta's survive to the last rung.
    assert merges["alpha"]["rung"] == "R42_Q1"
    assert merges["alpha"]["level"] == sizes["R42_Q1"]
    assert merges["beta"]["rung"] == "R1_Q1"
    assert collapsed["alpha"][0] == "R42_Q1"


def test_a_label_map_that_contradicts_the_probe_file_stops_the_run(ladder, tmp_path):
    """Two answers to 'where is this distinction merged' means one is wrong, and
    the figure would annotate the wrong vocabulary."""
    p = ladder["root"] / "experiments/FIGS/data/probe_ladder_v2/s1.json"
    d = json.loads(p.read_text())
    d["tasks"]["alpha"]["collapsed_at"] = ["R16_Q1"]
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        F.main(["--analysis", str(ladder["analysis"]), "--rung-map", str(ladder["rung_map"]),
                "--outdir", str(tmp_path / "figs")])
    assert "read the tree wrong" in str(e.value)


def test_the_palette_is_still_readable_in_greyscale():
    """Colour-blind-safe is not enough: the journal prints some figures in grey,
    and four hues of the same luminance become one line."""
    lums = sorted(S.relative_luminance(c) for c in S.LEVEL_COLOURS.values())
    gaps = [b - a for a, b in zip(lums, lums[1:])]
    assert min(gaps) > 0.08, f"luminances too close to separate in grey: {lums}"
    assert len(set(S.LEVEL_COLOURS.values())) == len(S.LEVEL_COLOURS)
    assert len(set(S.LEVEL_MARKERS.values())) == len(S.LEVEL_MARKERS)
    assert S.PROBE_LINESTYLES["linear"] != S.PROBE_LINESTYLES["mlp"]


def test_save_writes_both_formats_at_print_resolution(tmp_path):
    import matplotlib.pyplot as plt
    S.use_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = S.save(fig, "unit_test_figure", tmp_path)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert S.DPI == 300


def test_colour_cycle_refuses_to_recycle_a_colour():
    """Two series in one colour is a figure that lies about how many things it shows."""
    with pytest.raises(SystemExit):
        S.colour_cycle([str(i) for i in range(len(S.SERIES_COLOURS) + 1)])
