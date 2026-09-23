"""experiments/AOJ/discriminants.py: the two- and three-prong scores must be the
SAME physical discriminant whichever head they are read from."""
import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


disc = _load("experiments/AOJ/discriminants.py", "aoj_discriminants")
ROWS = list(csv.DictReader(MAP.open()))
HEADS = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
EXPECTED_NATIVE = {"two_prong": 10, "three_prong": 29}


def _native(rung, nodes):
    return {int(r["jet_label"]) for r in ROWS if int(r[rung]) in set(nodes)}


@pytest.mark.parametrize("which", sorted(disc.STRUCTURES))
def test_member_sets_are_nested_consistently_across_granularities(which):
    """188 -> 162 -> 43 -> 17 classes: the members at each head must expand to the
    IDENTICAL set of native classes, or the heads are scored on different physics."""
    structure = disc.STRUCTURES[which]
    expanded = {rung: _native(rung, disc.members(rung, structure))
                for rung in [*HEADS, "R63_Q1", "R29_Q1"]}
    assert len(expanded["L188"]) == EXPECTED_NATIVE[which]
    assert all(v == expanded["L188"] for v in expanded.values()), {k: len(v) for k, v in expanded.items()}
    assert expanded["L188"] == disc.native_classes(structure)
    # and coarsening only ever MERGES members: the count cannot grow down the ladder
    sizes = [len(disc.members(r, structure)) for r in ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] == 1


@pytest.mark.parametrize("which", sorted(disc.STRUCTURES))
def test_members_are_exactly_the_nodes_carrying_the_structural_prefix(which):
    structure = disc.STRUCTURES[which]
    for rung in disc.NAMED_RUNGS:
        by_prefix = {int(r[rung]) for r in ROWS if r[f"{rung}_name"].split("|")[0] == structure}
        assert set(disc.members(rung, structure)) == by_prefix
    name16 = {r["R16_Q1_name"] for r in ROWS if int(r["R16_Q1"]) in disc.members("R16_Q1", structure)}
    name43 = {r["R42_Q1_name"] for r in ROWS if int(r["R42_Q1"]) in disc.members("R42_Q1", structure)}
    assert name16 == {structure}
    assert name43 == {f"{structure}|{s}" for s in ("B", "C", "LG")}


def test_qcd_denominator_is_the_27_native_qcd_classes_at_every_head():
    qcd = {int(r["jet_label"]) for r in ROWS if r["class_name"].startswith("label_QCD_")}
    assert len(qcd) == 27
    for rung, k in HEADS.items():
        assert disc.n_outputs(rung) == k
        assert _native(rung, disc.qcd_nodes(rung)) == qcd
        assert len(disc.qcd_nodes(rung)) == (27 if rung == "L188" else 1)
        if rung != "L188":
            assert {r[f"{rung}_name"] for r in ROWS if int(r[rung]) in disc.qcd_nodes(rung)} == {"QCD_ALL"}


def test_the_score_is_identical_at_every_granularity_under_class_division():
    """Contract 188 probabilities exactly (sum within groups, Sophon Property 1).
    The discriminant read from each contracted head must equal the 188-class one."""
    rng = np.random.default_rng(0)
    p188 = rng.dirichlet(np.full(188, 0.3), size=500)
    ref = {w: disc.score(np.log(p188), "L188", s) for w, s in disc.STRUCTURES.items()}
    for rung, k in HEADS.items():
        node = np.array([int(r[rung]) for r in sorted(ROWS, key=lambda r: int(r["jet_label"]))])
        pk = np.zeros((len(p188), k))
        np.add.at(pk, (slice(None), node), p188)
        for w, s in disc.STRUCTURES.items():
            np.testing.assert_allclose(disc.score(np.log(pk), rung, s), ref[w], rtol=1e-10)
    # and it IS sum P(sig) / (sum P(sig) + sum P(QCD)), not some other ratio
    sig = sorted(disc.native_classes(disc.STRUCTURES["two_prong"]))
    want = p188[:, sig].sum(1) / (p188[:, sig].sum(1) + p188[:, 161:188].sum(1))
    np.testing.assert_allclose(ref["two_prong"], want, rtol=1e-10)


def test_log_odds_is_shift_invariant_and_survives_saturated_logits():
    x = np.zeros((2, 17)); x[0, 0], x[1, 16] = 80.0, 80.0          # softmax would round to 1.0 / 0.0
    lo = disc.log_odds(x, "R16_Q1", "2P_HAD_2PARTON")
    assert lo[0] == pytest.approx(80.0) and lo[1] == pytest.approx(-80.0)
    np.testing.assert_allclose(disc.log_odds(x + 1e3, "R16_Q1", "2P_HAD_2PARTON"), lo)


def test_heads_that_cannot_express_the_discriminant_are_refused():
    for rung in ("R3_VIS", "R1_Q1"):
        with pytest.raises(SystemExit, match="not constructible"):
            disc.members(rung, "2P_HAD_2PARTON")
    with pytest.raises(SystemExit, match="not a L162 head"):
        disc.log_odds(np.zeros((3, 188)), "L162", "2P_HAD_2PARTON")


def _fake_extraction(tmp_path, monkeypatch, corrupt=False, num_reg=0, extra=()):
    sa = _load("scripts/stage_aoj.py", "stage_aoj")
    t = _load("tests/test_stage_aoj.py", "t_stage_aoj")
    import awkward as ak
    paths = []
    for i in range(2):
        rec, _ = sa.convert(*t.synth(n_jets=20, seed=20 + i))
        paths.append(tmp_path / f"f{i}.parquet")
        ak.to_parquet(ak.Array(rec), paths[-1])
    jets = disc.staged_jets(paths)
    ext = tmp_path / "extract"; ext.mkdir()
    np.save(ext / "logits.npy",
            np.random.default_rng(1).normal(size=(40, 162 + num_reg)).astype(np.float32))
    obs = {k: jets[k].astype(np.float32) for k in disc.JET_FLOATS}
    if corrupt:
        obs["jet_sdmass"] = obs["jet_sdmass"][::-1].copy() + 1.0
    np.savez(ext / "observers.npz", **obs)
    man = dict(checkpoint="x", checkpoint_sha256="y")
    if num_reg:
        man["logit_columns"] = {"class_logits": [0, 162], "regression": [162, 162 + num_reg]}
    (ext / "extract_manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(sys, "argv", ["discriminants.py", "--name", "m", "--rung", "L162",
                                      "--extract-dir", str(ext), "--out", str(tmp_path / "out"),
                                      "--staged", *map(str, paths), *extra])
    return jets


def test_main_persists_float16_log_odds_and_exact_event_numbers(tmp_path, monkeypatch):
    jets = _fake_extraction(tmp_path, monkeypatch)
    assert disc.main() == 0
    s = np.load(tmp_path / "out" / "scores_m.npz")
    assert sorted(s.files) == ["three_prong_logodds", "two_prong_logodds"]
    assert all(s[k].dtype == np.float16 and s[k].shape == (40,) for k in s.files)
    j = np.load(tmp_path / "out" / "jets.npz")
    assert j["event"].dtype == np.int64 and np.array_equal(j["event"], jets["event"])
    assert j["event"].max() > 2**32, "fixture must exceed float32's exact-integer range"
    assert disc.main() == 0, "a second model over the same jets must be accepted"


def test_main_refuses_logits_that_are_not_row_aligned_with_the_staged_files(tmp_path, monkeypatch):
    _fake_extraction(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(SystemExit, match="not row-aligned"):
        disc.main()


def test_a_mass_output_head_is_scored_on_its_class_columns_only(tmp_path, monkeypatch):
    """The last column of a mass-output model is the mass regression. The score
    must equal the one computed from the 162 class columns alone."""
    _fake_extraction(tmp_path, monkeypatch, num_reg=1)
    assert disc.main() == 0
    full = np.load(tmp_path / "extract" / "logits.npy")
    want = disc.log_odds(full[:, :162], "L162", "3P_HAD_3PARTON").astype(np.float16)
    assert np.array_equal(np.load(tmp_path / "out" / "scores_m.npz")["three_prong_logodds"], want)


def test_the_withdrawn_two_prong_score_can_be_left_unwritten(tmp_path, monkeypatch):
    _fake_extraction(tmp_path, monkeypatch, extra=("--structures", "three_prong"))
    assert disc.main() == 0
    assert np.load(tmp_path / "out" / "scores_m.npz").files == ["three_prong_logodds"]
