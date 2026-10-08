"""experiments/STATS/seed_level.py --v2: the descriptive tables of the second grid's frozen
readouts, per checkpoint and readout, with the levels and run counts read from the grid
(tests/v2_fixture.py writes a synthetic one), never from the first grid's constants."""
import importlib.util
import json
import pathlib
import shutil

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


SL = _mod("seed_level_v2t", "experiments/STATS/seed_level.py")
FX = _mod("v2_fixture_sl", "tests/v2_fixture.py")


@pytest.fixture
def root(tmp_path):
    FX.write_design(tmp_path)
    FX.write_frozen(tmp_path)
    return tmp_path


def _read(dest, tag, ro, name="seed_level_results.json"):
    return json.loads((dest / tag / ro / name).read_text())


def test_the_levels_and_runs_come_from_the_grid(root):
    dest = FX.seed_level_v2(root)
    A = _read(dest, "best70", "features")
    # the tree arms present, fine -> coarse, at the grid's class counts: the 64-class
    # level sits between 162 and 43 because the grid says so
    assert A["levels_fine_to_coarse"] == [188, 162, 64, 43, 17]
    assert A["seeds_used"] == [1, 2, 3] and A["missing_runs"] == {}
    assert A["runs_by_arm"]["L188"] == [1, 2, 3] and A["expected_runs"]["L162_MASS"] == 3
    assert A["level_arms"] == {"188": "L188", "162": "L162", "64": "R63_Q1", "43": "R42_Q1", "17": "R16_Q1"}
    # the mean over runs of ln(1 - AUC) is the closed form's: runs 1-3 add 0.02 on average
    lv = {r["level"]: r for r in A["levels"]["bvc_resonant"]["linear"]}
    for arm, level in A["level_arms"].items():
        want = FX.BASE["bvc_resonant"] + FX.STEP * FX.RANK[level] + 0.02
        assert lv[int(arm)]["mean"] == pytest.approx(want, abs=1e-12)
        assert lv[int(arm)]["n_seeds"] == 3 and lv[int(arm)]["seed_sd"] == pytest.approx(0.01)
    # every arm has its summary, the mass-output and self-supervised arms included, by label
    arms = A["arms"]["bvc_resonant"]["linear"]
    assert set(arms) == {n for n, *_, o, _ in FX.ARMS if o != "mpm"}       # MPM: pooled only
    assert {r["cell"] for r in A["table"] if r["arm"] == "L162_MASS"} == {"162+mass"}
    assert A["tasks"]["bc_vs_rest"]["eps_s"] == [0.6, 0.4]
    assert A["tasks"]["bvc_resonant"]["n_background_test"] == FX.N_BKG
    # the 90 % working point is the headline, as for the first grid
    assert lv[188]["headline_eps_s"] == "0.90" and "0.90" in lv[188]["rejection_points"]


def test_every_tag_and_readout_has_its_table_and_the_references_theirs(root):
    dest = FX.seed_level_v2(root)
    got = {(p.parents[1].name, p.parent.name) for p in dest.glob("*/*/seed_level_results.json")}
    assert got == {(t, r) for t in FX.TAGS for r in ("features", "pooled")} | \
        {("init", "features"), ("init", "pooled")}
    # the self-supervised model has the pooled readout only
    assert "MPM" in _read(dest, "best70", "pooled")["runs_by_arm"]
    assert "MPM" not in _read(dest, "best70", "features")["runs_by_arm"]
    # the untrained trunk is a reference: no level, its own arm, three runs
    I = _read(dest, "init", "pooled")
    assert I["levels_fine_to_coarse"] == [] and I["runs_by_arm"] == {"INIT": [1, 2, 3]}
    assert {r["cell"] for r in I["table"]} == {"untrained trunk"}
    # the BatchNorm twin is read as its own tag
    T = _read(dest, "best70_bn", "features")
    assert {r["model"] for r in T["table"] if r["arm"] == "L188"} == {f"l188-s{k}@best70_bn" for k in (1, 2, 3)}
    # mass regression and label recovery, the same way
    M = _read(dest, "best70", "features", "mass_resolution_table.json")
    assert {r["cell"] for r in M["table"]} == {FX.LABELS[n] for n, *_, o, _ in FX.ARMS if o != "mpm"}
    assert {"188", "64", "162+mass", "17+mass, matched lambda"} <= {r["cell"] for r in M["table"]}
    assert M["provenance"]["n_classes_used"] == 150
    R = _read(dest, "best70", "features", "label_recovery_curve_summary.json")
    assert sorted({r["level"] for r in R["table"]}) == [17, 43, 64, 162, 188]
    assert R["sizes"] == FX.SIZES and not R["unconverged"]
    cap = {r["level"]: r for r in R["mlp_minus_linear_at_largest"]}
    assert cap[17]["per_run"] == pytest.approx([0.02] * 3)


def test_a_linked_tag_is_read_as_its_own_tag_and_a_stray_file_is_refused(root):
    dest = FX.seed_level_v2(root)
    B = _read(dest, "bestval", "features")
    # run 1's bestval is a copy of its best70 file, which names best70
    assert {r["model"] for r in B["table"] if r["arm"] == "L188"} == \
        {"l188-s1@best70", "l188-s2@bestval", "l188-s3@bestval"}
    assert any(x["path"].endswith("mtx-l188-s1/bestval/features/probe_results.json")
               for x in B["provenance"]["inputs"])
    # a bestval file naming best70 that is NOT best70's bytes is not a link: refused
    p = FX.v2(root, "probe_ladder") / "probe/mtx-l188-s1/bestval/features/probe_results.json"
    d = json.loads(p.read_text())
    d["tasks"]["bvc_resonant"]["arms"]["l188-s1@best70"]["linear"]["auc"] = 0.5
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not a link to it"):
        FX.seed_level_v2(root, "analysis_again")


def test_a_wrong_readout_a_foreign_run_and_two_alignments_are_refused(root):
    sec = FX.v2(root, "probe_ladder") / "probe"
    p = sec / "mtx-l162-s2/wavg/pooled/probe_results.json"
    good = p.read_text()
    d = json.loads(good)
    d["readout"] = "features"
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="records readout"):
        FX.seed_level_v2(root, "a1")
    p.write_text(good)
    shutil.copytree(sec / "mtx-l162-s2", sec / "mtx-l162-s9")
    with pytest.raises(SystemExit, match="not a run or reference"):
        FX.seed_level_v2(root, "a2")
    shutil.rmtree(sec / "mtx-l162-s9")
    d = json.loads(good)
    d["row_alignment_sha256"] = "z" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="row_alignment_sha256"):
        FX.seed_level_v2(root, "a3")


def test_a_missing_run_is_reported_and_an_analysis_is_never_overwritten(root):
    shutil.rmtree(FX.v2(root, "probe_ladder") / "probe/mtx-r16q1-s3")
    dest = FX.seed_level_v2(root)
    A = _read(dest, "best70", "features")
    assert A["missing_runs"] == {"R16_Q1": [3]}
    assert A["runs_by_arm"]["R16_Q1"] == [1, 2]
    with pytest.raises(SystemExit, match="exists"):
        FX.seed_level_v2(root)


def test_without_tier_three_the_ladder_is_the_tier_one_levels(tmp_path):
    FX.write_design(tmp_path)
    FX.write_frozen(tmp_path, tier3=False)
    A = _read(FX.seed_level_v2(tmp_path), "best70", "features")
    assert A["levels_fine_to_coarse"] == [188, 162, 43, 17]
    assert not np.isin(64, [r["level"] for r in A["table"]])


def test_the_first_grid_ladder_is_unchanged_by_the_levels_parameter():
    # level_summary defaults to the first grid's four levels
    cells = {("t", "linear", lv, s): {"log1m_auc": -5.0 + 0.1 * s, "auc": 0.99, "rejection": 10.0,
                                       "rejection_eps_s": 0.5, "rejection_is_bound": False, "censored": False}
             for lv in SL.LEVELS for s in (1, 2)}
    assert [r["level"] for r in SL.level_summary(cells, "t", "linear", [1, 2])] == list(SL.LEVELS)
