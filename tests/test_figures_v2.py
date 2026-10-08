"""The figure scripts on the second grid (tests/v2_fixture.py): the levels and run counts are
the grid's, the files are the ones the tables read, and a level without a colour stops the
figure rather than borrowing one."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


F = _mod("make_ladder_figures_v2", REPO / "experiments/FIGS/make_ladder_figures.py")
R = _mod("make_results_figures_v2", REPO / "experiments/FIGS/make_results_figures.py")
FX = _mod("v2_fixture_fig", REPO / "tests/v2_fixture.py")
T1 = _mod("test_make_tables_figfx", REPO / "tests/test_make_tables.py")


def _root(tmp_path, tier3):
    FX.write_design(tmp_path)
    FX.write_frozen(tmp_path, tier3=tier3)
    FX.write_ft(tmp_path)
    FX.seed_level_v2(tmp_path)
    return tmp_path


def test_the_ladder_figure_takes_the_grids_levels_and_draws_no_test_interval(tmp_path):
    root = _root(tmp_path / "r", tier3=True)
    a = F.default_analysis(root)
    assert a.parts[-3:] == ("best70", "features", "seed_level_results.json")
    A = json.loads(a.read_text())
    assert A["levels_fine_to_coarse"] == [188, 162, 64, 43, 17]
    # the |V_cb| window probe pins its own working points and is left to the text
    assert "bc_vs_rest" not in F.plotted_tasks(A) and "bvc_resonant" in F.plotted_tasks(A)
    out = tmp_path / "figs"
    assert F.main(["--analysis", str(a), "--rung-map", str(T1.write_rung_map(tmp_path)),
                   "--outdir", str(out)]) == 0
    assert (out / "probe_ladder_granularity.pdf").stat().st_size > 5000
    assert not list(out.glob("probe_ladder_paired_*"))
    # without v2 frozen probes the first grid's analysis is the default
    assert F.default_analysis(tmp_path / "empty") == F.ANALYSIS


def test_the_results_figures_read_the_v2_fine_tuning_and_mass_of_the_freeze(tmp_path):
    root = _root(tmp_path / "r", tier3=False)
    legs, levels, arms = R.v2_finetune(root)
    assert levels == [188, 162, 43, 17] and arms["r16q1"] == 17 and "r63q1" not in arms
    by = R.ft_by_seed(R.ft_cells(legs["JetClass-II, 162 classes"]), levels, arms)
    assert {lv: len(v[1000]) for lv, v in by.items()} == {188: 3, 162: 3, 43: 3, 17: 3}
    M, pts, groups = R.v2_mass(root)
    assert groups == ["188", "162", "43", "17", "162+mass", "17+mass"]
    assert pts["17+mass"] == pytest.approx([FX.log1m("R16_Q1_MASS", k, "best70", "features", "bvc_resonant")
                                           for k in (1, 2, 3)])
    out = tmp_path / "figs"
    assert R.main(["--outdir", str(out), "--root", str(root)]) == 0
    for stem in ("finetune_curves", "mass_tradeoff"):
        assert (out / f"{stem}.pdf").stat().st_size > 5000


def test_the_tier_three_levels_have_colours_and_a_level_without_one_stops_the_figure(tmp_path):
    # 64 and 30 classes (A12) have their own colour and marker, every greyscale gap above 0.08
    S = _mod("figs_style_v2", REPO / "experiments/FIGS/style.py")
    assert {64, 30} <= set(S.LEVEL_COLOURS) and {64, 30} <= set(S.LEVEL_MARKERS)
    lums = sorted(S.relative_luminance(c) for c in S.LEVEL_COLOURS.values())
    assert min(b - a for a, b in zip(lums, lums[1:])) > 0.08
    root = _root(tmp_path / "r", tier3=True)
    assert R.v2_mass(root)[2][:5] == ["188", "162", "64", "43", "17"]
    with pytest.raises(SystemExit, match="no colour for \\[99\\]"):
        R.coloured([188, 99])
