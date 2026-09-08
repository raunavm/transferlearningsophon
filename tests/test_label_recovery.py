"""The label-recovery probe (docs/DOWNSTREAM_SUITE.md Core, STATISTICS P4).

It asks directly whether a distinction survived compression, so its own control
-- an arm recovering its OWN vocabulary -- has to work, or no cell in that arm's
row means anything.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    s = importlib.util.spec_from_file_location(
        "label_recovery", ROOT / "experiments" / "EVAL" / "label_recovery.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


lr = _load()


def test_rung_widths_match_the_committed_map():
    """188 = 161+27; every _Q1 rung is <resonant>+1."""
    m = lr.rung_maps()
    want = {"L188": 188, "L162": 162, "R63_Q1": 64, "R42_Q1": 43,
            "R29_Q1": 30, "R16_Q1": 17, "R3_VIS": 4, "R1_Q1": 2}
    for r, k in want.items():
        assert len(set(m[r].values())) == k, f"{r}: {len(set(m[r].values()))} != {k}"


def test_rungs_are_ordered_fine_to_coarse():
    """is_finer_than_own is decided by position in this list, so the order is
    load-bearing rather than cosmetic."""
    m = lr.rung_maps()
    ks = [len(set(m[r].values())) for r in lr.RUNGS]
    assert ks == sorted(ks, reverse=True), f"RUNGS not fine->coarse: {ks}"


def test_a_coarser_rung_is_a_function_of_a_finer_one():
    """Why coarser cells are a floor, not a result: they are recoverable by
    construction. If this ever fails the tree is not nested and the whole
    interpretation of the matrix changes."""
    m = lr.rung_maps()
    for fine, coarse in zip(lr.RUNGS, lr.RUNGS[1:]):
        seen = {}
        for lab in range(188):
            f, c = m[fine][lab], m[coarse][lab]
            assert seen.setdefault(f, c) == c, (
                f"{fine} group {f} maps to more than one {coarse} group")


def _cache(d, n, labels, rng, sep):
    """Features that carry the native label with strength `sep`."""
    d.mkdir(parents=True, exist_ok=True)
    F = rng.normal(size=(n, 12)).astype(np.float32)
    F[:, 0] += sep * labels.astype(np.float32) / 188.0
    F[:, 1] += sep * (labels % 7).astype(np.float32)
    np.save(d / "features.npy", F)
    np.save(d / "label188.npy", labels.astype(np.int16))
    np.savez(d / "observers.npz", jet_pt=rng.uniform(200, 2500, n).astype(np.float32))
    (d / "extract_manifest.json").write_text('{"arm": "t"}')


def test_main_runs_and_the_own_rung_control_is_reported(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    lab = rng.integers(0, 188, size=n)
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, sep=30.0)
    out = tmp_path / "o"
    lr.main(["--features", f"a={d}", "--own-rung", "a=R16_Q1", "--out", str(out),
             "--n", "3000", "--rungs", "R16_Q1", "R3_VIS", "R1_Q1"])
    res = json.loads((out / "label_recovery.json").read_text())
    cells = res["arms"]["a"]["rungs"]
    assert cells["R16_Q1"]["is_own_rung"] is True
    assert cells["R3_VIS"]["is_own_rung"] is False
    # R3_VIS and R1_Q1 are COARSER than R16_Q1, so not finer
    assert cells["R3_VIS"]["is_finer_than_own"] is False
    for c in cells.values():
        assert "linear" in c and "mlp" in c, "D6: both probes, always"
        assert c["chance"] == pytest.approx(1.0 / c["n_groups"])


def test_finer_than_own_is_flagged_on_the_right_cells(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    lab = rng.integers(0, 188, size=n)
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, sep=30.0)
    out = tmp_path / "o"
    lr.main(["--features", f"a={d}", "--own-rung", "a=R16_Q1", "--out", str(out),
             "--n", "2500", "--rungs", "R3_VIS", "R16_Q1", "R29_Q1"])
    cells = json.loads((out / "label_recovery.json").read_text())["arms"]["a"]["rungs"]
    # R29_Q1 (K=30) is FINER than R16_Q1 (K=17): that is the measurement
    assert cells["R29_Q1"]["is_finer_than_own"] is True
    assert cells["R3_VIS"]["is_finer_than_own"] is False


def test_a_null_needs_both_probes_at_chance(tmp_path):
    """D6. A linear probe lower-bounds mutual information, so a linear null
    cannot distinguish 'absent' from 'present but not linearly decodable'."""
    rng = np.random.default_rng(2)
    n = 3000
    lab = rng.integers(0, 188, size=n)
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, sep=0.0)      # features carry NOTHING
    out = tmp_path / "o"
    lr.main(["--features", f"a={d}", "--own-rung", "a=R1_Q1", "--out", str(out),
             "--n", "2500", "--rungs", "R1_Q1"])
    c = json.loads((out / "label_recovery.json").read_text())["arms"]["a"]["rungs"]["R1_Q1"]
    assert c["not_recovered"] is True, "pure noise features must read as not recovered"
    # and the flag is conjunctive: raising only the linear score must clear it
    assert c["not_recovered"] == bool(
        c["linear"] <= c["chance"] + lr.CHANCE_MARGIN
        and c["mlp"] <= c["chance"] + lr.CHANCE_MARGIN)


def test_informative_features_are_not_called_a_null(tmp_path):
    rng = np.random.default_rng(3)
    n = 4000
    lab = rng.integers(0, 188, size=n)
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, sep=60.0)
    out = tmp_path / "o"
    lr.main(["--features", f"a={d}", "--own-rung", "a=R3_VIS", "--out", str(out),
             "--n", "3000", "--rungs", "R3_VIS"])
    c = json.loads((out / "label_recovery.json").read_text())["arms"]["a"]["rungs"]["R3_VIS"]
    assert c["not_recovered"] is False
    assert max(c["linear"], c["mlp"]) > c["chance"] + lr.CHANCE_MARGIN
