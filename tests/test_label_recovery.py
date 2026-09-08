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
        c["linear"] <= c["chance"] + c["chance_margin"]
        and c["mlp"] <= c["chance"] + c["chance_margin"])


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
    assert max(c["linear"], c["mlp"]) > c["chance"] + c["chance_margin"]


def test_chance_margin_scales_with_k_and_n():
    """A flat absolute margin is 3.8x chance at L188 and 0.08x at R3_VIS --
    most permissive exactly on the finer cells the module calls the actual
    measurement. The margin must be a statement about the null's width."""
    lr = _load()
    N = 400_000
    frac = {}
    for k in (188, 162, 64, 43, 30, 17, 4, 2):
        m = lr.chance_margin(k, [N // k] * k)
        frac[k] = m / (1.0 / k)
        assert 0 < m < 1.0 / k, f"k={k}: margin must be well inside chance"
    # never again more permissive on the fine end than on the coarse end
    assert frac[188] < 0.25, "L188 margin must not approach chance itself"
    assert max(frac.values()) / min(frac.values()) < 20, \
        "margin/chance must not swing by orders of magnitude across rungs"


def test_chance_margin_tightens_with_more_data():
    lr = _load()
    small = lr.chance_margin(17, [1_000] * 17)
    big = lr.chance_margin(17, [100_000] * 17)
    assert big < small, "more test jets must narrow the null band"


def test_the_mlp_probe_reports_whether_it_converged():
    """D6's argument needs a CONVERGED MLP, and the first live run had none.

    A linear probe only lower-bounds mutual information, so D6 requires the
    nonlinear probe beside any null. That reasoning collapses if the MLP simply
    ran out of iterations: at max_iter=300 with no early stopping, the live
    eval-labelrec job hit the cap on every cell and returned mlp 0.2813 against
    linear 0.3454 for l162-s1b/L188 -- impossible for a converged model whose
    hypothesis class CONTAINS the linear one. Nothing in the output said so, so
    the number read as "no nonlinear structure". These three fields make the
    failure visible instead.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    n, d, k = 600, 12, 3
    Xtr = rng.normal(size=(n, d)); ytr = rng.integers(0, k, n)
    Xte = rng.normal(size=(200, d)); yte = rng.integers(0, k, 200)
    out = lr.fit_pair(Xtr, ytr, Xte, yte)
    for key in ("mlp_n_iter", "mlp_converged", "mlp_below_linear"):
        assert key in out, f"fit_pair no longer reports {key}"
    assert isinstance(out["mlp_converged"], bool)
    assert all(i <= lr.MLP_MAX_ITER for i in out["mlp_n_iter"])
    assert len(out["mlp_n_iter"]) == len(lr.MLP_SEEDS)


def test_the_mlp_uses_early_stopping_not_a_bare_iteration_cap():
    """The cap alone is what produced an unconverged probe; a raised cap with no
    stopping rule would just be a slower way to hit it."""
    src = (ROOT / "experiments" / "EVAL" / "label_recovery.py").read_text()
    # Executable lines only. The comment above fit_pair quotes the old
    # max_iter=300 to explain what went wrong, and a naive substring check on
    # the whole file fails on that prose rather than on the code.
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "early_stopping=True" in code
    assert "max_iter=300" not in code, "the unconverged setting is back"
