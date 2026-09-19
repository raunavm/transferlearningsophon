"""The probe's task definitions and its two correctness guards.

The most valuable test here is test_collapse_claims_match_the_label_map: the
probe asserts, in prose that reaches the paper, that b-vs-c resonant survives at
R42_Q1 and dies at R16_Q1, and that QCD b-vs-c survives ONLY at L188. Those
claims are what make each rung load-bearing, and they are checked here against
configs/labelmaps/rung_label_maps.v1.csv rather than believed.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MAPS = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def probe():
    return _probe()


@pytest.fixture(scope="module")
def rungs():
    if not MAPS.exists():
        pytest.skip("label map absent")
    rows = list(csv.DictReader(MAPS.open()))
    return {int(r["jet_label"]): r for r in rows}


def test_collapse_claims_match_the_label_map(probe, rungs):
    """Each task must be COLLAPSED exactly at the rungs it claims, and distinct
    at every other rung. This is the paper's rung argument, checked."""
    all_arms = ["L188", "L162", "R42_Q1", "R16_Q1"]
    for task, spec in probe.TASKS.items():
        sig_ids, bkg_ids = spec["signal"], spec["background"]
        assert rungs[sig_ids[0]]["class_name"] == spec["names"][0]
        # A one-class background names itself; a multi-class background (the
        # published discriminants have several) names the SET, so only the
        # single-class case can be checked against the map by name.
        if len(bkg_ids) == 1:
            assert rungs[bkg_ids[0]]["class_name"] == spec["names"][1]
        for arm in all_arms:
            # Collapsed at a rung iff the signal shares its group with ANY
            # background class: that is exactly when the arm can no longer
            # separate the two sides, whatever the background's size.
            sg = {rungs[i][arm] for i in sig_ids}
            bg = {rungs[i][arm] for i in bkg_ids}
            collapsed = bool(sg & bg)
            claimed = arm in spec["collapsed_at"]
            assert collapsed == claimed, (
                f"{task}: at {arm} the sides are "
                f"{'collapsed' if collapsed else 'distinct'} "
                f"(signal groups {sorted(sg)} vs background {sorted(bg)}) but "
                f"the task claims {'collapsed' if claimed else 'distinct'}")


def test_tasks_are_arm_independent(probe):
    """A task defined per-arm would not be a controlled contrast."""
    for spec in probe.TASKS.values():
        assert all(isinstance(v, int) for v in spec["signal"] + spec["background"])
        assert set(spec["signal"]).isdisjoint(spec["background"])


def test_alignment_guard_fires(probe):
    import hashlib
    a = np.arange(100, dtype=np.int16)
    arms = {
        "A": {"L": a, "label_sha": hashlib.sha256(a.tobytes()).hexdigest()},
        "B": {"L": a[::-1].copy(),
              "label_sha": hashlib.sha256(a[::-1].copy().tobytes()).hexdigest()},
    }
    with pytest.raises(SystemExit) as e:
        probe.check_alignment(arms)
    assert e.value.code == 2
    same = {"A": arms["A"], "B": dict(arms["A"])}
    assert probe.check_alignment(same) == arms["A"]["label_sha"]


def test_rejection_past_the_cap_is_flagged_a_bound(probe):
    """Beyond 1/N_bkg the value is an artefact of sample size, never a value."""
    y = np.r_[np.ones(500, int), np.zeros(500, int)]
    perfect = np.r_[np.ones(500), np.zeros(500)] * 1.0
    r, eps_b, bound, npass, rel = probe.rejection_at(y, perfect)
    assert bound is True and r <= 500.0
    rng = np.random.default_rng(0)
    r2, _, bound2, npass2, rel2 = probe.rejection_at(y, rng.random(1000))
    assert bound2 is False and r2 < 10
    # the rejection is 1/eps_B and eps_B comes from a COUNT, so the surviving
    # count and its Poisson band travel with the number
    assert npass2 > 0 and rel2 == pytest.approx(1.0 / np.sqrt(npass2), rel=0.05)


def test_rejection_carries_its_poisson_band(probe):
    """A rejection resting on a handful of surviving background jets is not as
    precise as it looks; anchors.py bands the same quantity."""
    y = np.r_[np.ones(2000, int), np.zeros(2000, int)]
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(1.5, 1, 2000), rng.normal(0, 1, 2000)]
    r, eps_b, bound, npass, rel = probe.rejection_at(y, s, 0.5)
    assert npass == pytest.approx(eps_b * 2000, abs=1)
    assert rel == pytest.approx(1.0 / np.sqrt(npass), rel=1e-6)
    # tighter working point -> fewer survivors -> wider band
    _, _, _, npass_t, rel_t = probe.rejection_at(y, s, 0.2)
    assert npass_t < npass and rel_t > rel


def test_splits_are_deterministic_and_partition(probe):
    """Same jets must land in the same split for every arm, or the comparison
    is confounded by the split rather than by the arm."""
    tr, va, te = probe.make_splits(1000)
    tr2, va2, te2 = probe.make_splits(1000)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2)
    allidx = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allidx), np.arange(1000)), "splits must partition"


def test_alignment_check_surfaces_the_checkpoint(probe, capsys):
    """label188 is a property of the DATA, so two caches of the SAME arm at
    different checkpoints share a label sha and pass the row-alignment check.
    Mixing features_v2 (best epoch) with features_e79 would then report an
    epoch difference as a vocabulary effect, with nothing erroring."""
    L = np.arange(100, dtype=np.int16)
    sha = hashlib.sha256(L.tobytes()).hexdigest()
    arms = {
        "A": {"L": L, "label_sha": sha, "manifest": {"checkpoint_sha256": "a" * 64}},
        "B": {"L": L, "label_sha": sha, "manifest": {"checkpoint_sha256": "b" * 64}},
    }
    assert probe.check_alignment(arms) == sha
    out = capsys.readouterr().out
    assert "aaaaaaaaaaaaaaaa" in out and "bbbbbbbbbbbbbbbb" in out, \
        "each arm's checkpoint must be printed so the confound is auditable"


def test_alignment_check_warns_when_a_manifest_predates_the_field(probe, capsys):
    L = np.arange(100, dtype=np.int16)
    sha = hashlib.sha256(L.tobytes()).hexdigest()
    arms = {
        "A": {"L": L, "label_sha": sha, "manifest": {"checkpoint_sha256": "a" * 64}},
        "B": {"L": L, "label_sha": sha, "manifest": {}},
    }
    probe.check_alignment(arms)
    err = capsys.readouterr().err
    assert "record no checkpoint" in err


def test_alignment_check_reads_the_legacy_digest_key(probe, capsys):
    L = np.arange(100, dtype=np.int16)
    sha = hashlib.sha256(L.tobytes()).hexdigest()
    arms = {
        "A": {"L": L, "label_sha": sha, "manifest": {"sha256": "c" * 64}},
        "B": {"L": L, "label_sha": sha, "manifest": {"sha256": "c" * 64}},
    }
    probe.check_alignment(arms)
    assert "record no checkpoint" not in capsys.readouterr().err


def test_mlp_is_reproducible_across_thread_counts(probe):
    """torch.manual_seed fixes the weights and dropout masks but NOT the order
    in which a CPU matmul reduces partial sums -- that follows the thread count,
    which follows the pod's CPU allocation. Two runs of the same job over the
    same cached features, differing only in cpu 4 vs cpu 8, gave 25 of 48 MLP
    cells different AUCs (max 0.0009) while all 48 linear cells were identical.
    0.0009 exceeds label_recovery's 5-sigma chance margin at the finest rungs,
    so an unpinned thread count could flip a null call between runs.
    """
    import torch
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 16)).astype(np.float32)
    y = (X[:, 0] + 0.4 * rng.normal(size=n) > 0).astype(np.int64)
    tr, va, te = probe.make_splits(n)

    outs = []
    for nt in (1, 4):
        torch.set_num_threads(nt)
        s, _ = probe.fit_mlp(X[tr], y[tr], X[va], y[va], X[te])
        outs.append(np.asarray(s, dtype=np.float64))
    # fit_mlp pins the count internally, so the caller's setting is irrelevant
    np.testing.assert_allclose(outs[0], outs[1], rtol=0, atol=0)


def test_mlp_thread_count_is_restored_and_recorded(probe):
    import torch
    torch.set_num_threads(3)
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    tr, va, te = probe.make_splits(n)
    probe.fit_mlp(X[tr], y[tr], X[va], y[va], X[te])
    assert torch.get_num_threads() == 3, "the caller's thread setting must survive"
    assert probe.MLP_THREADS == 4


def test_perfect_separation_is_floored_at_the_samples_resolution():
    """AUC == 1.0 is censored, not a measurement of 1 - AUC == 0.

    The floor must be one discordant pair out of n_sig * n_bkg -- what the
    sample can actually express -- not a constant chosen for convenience.
    """
    import numpy as np
    p = _probe()
    y = np.array([0] * 40 + [1] * 60)
    s = np.array([0.0] * 40 + [1.0] * 60)          # perfectly separated
    value, censored, auc = p.log1m_auc(y, s)
    assert auc == 1.0
    assert censored is True
    assert value == pytest.approx(np.log(1.0 / (60 * 40)))


def test_an_ordinary_auc_is_left_alone():
    """Below the ceiling the floor must not touch the number."""
    import numpy as np
    p = _probe()
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([rng.normal(0, 1, 500), rng.normal(1, 1, 500)])
    value, censored, auc = p.log1m_auc(y, s)
    assert censored is False
    assert 0.5 < auc < 1.0
    assert value == pytest.approx(np.log(1 - auc))


def test_the_old_epsilon_floor_overstated_the_separation():
    """Regression pin for the ee_vs_mm defect measured 2026-09-08.

    l162-s1b reached AUC = 1.0 on the ee-vs-mumu probe. The retired 1e-12 floor
    reported log(1-AUC) = -27.63 and a contrast of about -24 whose CI excluded
    zero by a margin the epsilon invented. The honest floor is ~4.3e-8 here, so
    the corrected number is far closer to zero. Anything that moves it back
    toward -27 is reintroducing the bug.
    """
    import numpy as np
    p = _probe()
    n_sig, n_bkg = 4933, 4693                      # the measured test split
    y = np.array([0] * n_bkg + [1] * n_sig)
    s = np.array([0.0] * n_bkg + [1.0] * n_sig)
    value, censored, _ = p.log1m_auc(y, s)
    assert censored is True
    assert value == pytest.approx(np.log(1.0 / (n_sig * n_bkg)), rel=1e-9)
    assert -18.0 < value < -16.0, value
    assert value > np.log(1e-12), "the floor must not sink below the resolution"


# ---------------------------------------------------------------------------
# The operating-point flag (docs/PRESPEC_2026-09.md, final section).
#
# 50 % signal efficiency is unusable on bvc_resonant: zero of 11,876 test
# background jets survive it at the three finest vocabularies, and on the MLP
# probe at all four, so the headline physics number is a statement about the
# size of the test split. The prespec added 70 % and 90 %. These tests bind the
# two things that could go wrong while adding them -- silently MOVING the
# working point instead of adding to it, and moving the one working point that
# is pinned to a published table.
# ---------------------------------------------------------------------------

def _fixture_arm(d: pathlib.Path) -> pathlib.Path:
    """One arm's feature cache: bvc_resonant perfectly separable (so its cells
    are censored at every operating point), bc_vs_rest ordinary (so a resolved
    cell is in the same run), both above MIN_PER_CLASS on the test split."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    per = 6000
    labels = np.concatenate([
        np.full(per, 0), np.full(per, 1),                       # bvc_resonant
        np.full(per, 4),                                        # bc_vs_rest sig
        np.repeat([5, 6, 70], per // 3)]).astype(np.int16)       # bc_vs_rest bkg
    n = labels.size
    X = rng.normal(size=(n, 8)).astype(np.float32)
    X[labels == 0, 0] = 1.0          # exact, so the split is perfect
    X[labels == 1, 0] = -1.0
    X[labels == 4, 0] = 0.8 * rng.normal(size=int((labels == 4).sum())) + 0.8
    np.save(d / "features.npy", X)
    np.save(d / "label188.npy", labels)
    (d / "extract_manifest.json").write_text('{"checkpoint_sha256": "%s"}' % ("d" * 64))
    # bc_vs_rest is defined only inside its published window
    np.savez(d / "observers.npz",
             jet_pt=np.full(n, 500.0), jet_sdmass=np.full(n, 110.0),
             jet_eta=np.zeros(n))
    return d


def _run(probe, tmp: pathlib.Path, extra: list[str]) -> dict:
    import json as _json
    feats = _fixture_arm(tmp / "feat")
    out = tmp / ("out" + "".join(extra).replace(" ", "").replace("-", "").replace(".", ""))
    argv = ["probe.py", "--features", f"A={feats}", "--out", str(out),
            "--tasks", "bvc_resonant", "bc_vs_rest"] + extra
    import sys as _sys
    old = _sys.argv
    _sys.argv = argv
    try:
        assert probe.main() == 0
    finally:
        _sys.argv = old
    return _json.loads((out / "probe_results.json").read_text())


@pytest.fixture(scope="module")
def runs(probe, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eps")
    return {"default": _run(probe, tmp, []),
            "flagged": _run(probe, tmp, ["--eps-s", "0.5", "0.7", "0.9"])}


def test_the_default_reproduces_todays_behaviour_exactly(runs):
    """No flag must mean what it meant before, or every committed v1 result
    stops being reproducible from the committed code."""
    d = runs["default"]
    assert d["eps_s_default"] == 0.5
    assert d["tasks"]["bvc_resonant"]["eps_s"] == [0.5]
    for entry in d["tasks"]["bvc_resonant"]["arms"]["A"].values():
        assert set(entry["rejection_at"]) == {"0.50"}
        assert entry["rejection_eps_s"] == 0.5


def test_the_flag_adds_operating_points_and_moves_none(runs):
    """The 50 % cell -- and every flat field mirroring it -- must come back bit
    for bit, so v2 contains v1 rather than replacing it."""
    a = runs["default"]["tasks"]["bvc_resonant"]["arms"]["A"]
    b = runs["flagged"]["tasks"]["bvc_resonant"]["arms"]["A"]
    assert runs["flagged"]["eps_s_default"] == 0.5
    assert runs["flagged"]["tasks"]["bvc_resonant"]["eps_s"] == [0.5, 0.7, 0.9]
    for kind in ("linear", "mlp"):
        assert set(b[kind]["rejection_at"]) == {"0.50", "0.70", "0.90"}
        assert b[kind]["rejection_at"]["0.50"] == a[kind]["rejection_at"]["0.50"]
        for k in ("auc", "log1m_auc", "log1m_auc_censored", "rejection", "eps_b",
                  "rejection_is_bound", "n_bkg_pass", "rel_stat_err",
                  "rejection_eps_s"):
            assert b[kind][k] == a[kind][k], k


def test_the_published_anchors_operating_points_are_untouched(probe, runs):
    """bc_vs_rest quotes 60 % / 40 % to match arXiv:2503.00118's table. A task
    pinned to a published number must not drift with a command-line default."""
    assert probe.TASKS["bc_vs_rest"]["eps_s"] == [0.60, 0.40]
    for r in runs.values():
        t = r["tasks"]["bc_vs_rest"]
        assert not t.get("skipped"), t
        assert t["eps_s"] == [0.60, 0.40]
        for entry in t["arms"]["A"].values():
            assert set(entry["rejection_at"]) == {"0.60", "0.40"}
            assert entry["rejection_eps_s"] == 0.60
    # and it is a RESOLVED cell, so the censoring test below is not vacuous
    lin = runs["flagged"]["tasks"]["bc_vs_rest"]["arms"]["A"]["linear"]
    assert lin["n_bkg_pass"] > 0 and not lin["rejection_is_bound"]


def test_a_censored_cell_carries_its_flag_and_its_cap_at_every_point(runs):
    """A bound is a statement about the size of the test split. Every operating
    point must return the full tuple, so no censored value can print as a bare
    number, and the cap must be readable without inferring it from the cells
    that happen to be sitting at it."""
    t = runs["flagged"]["tasks"]["bvc_resonant"]
    cap = t["n_background_test"]
    assert cap > 0
    lin = t["arms"]["A"]["linear"]
    for point in ("0.50", "0.70", "0.90"):
        r = lin["rejection_at"][point]
        assert set(r) == {"rejection", "eps_b", "rejection_is_bound",
                          "n_bkg_pass", "rel_stat_err"}
        assert r["rejection_is_bound"] is True, point
        assert r["n_bkg_pass"] == 0 and r["eps_b"] == 0.0
        assert r["rejection"] == float(cap), (point, r["rejection"], cap)


def test_the_cap_is_recorded_for_every_measured_task(runs):
    """It is derivable from a BOUNDED cell (rejection == cap there) and from
    nowhere else, so an uncensored cell could not say how much headroom it had
    left. `n` and `n_signal` are full-sample counts, not test-split counts."""
    for r in runs.values():
        for task, t in r["tasks"].items():
            if t.get("skipped"):
                continue
            assert t["n_background_test"] > 0 and t["n_signal_test"] > 0
            assert t["n_background_test"] < t["n"] - t["n_signal"], \
                "the cap is the TEST split's background count, not the sample's"


def test_an_out_of_range_operating_point_is_rejected(probe, tmp_path):
    """np.interp clamps rather than raising, so `--eps-s 50 70 90` would return
    a rejection of 1.0 in every cell and nothing would error."""
    with pytest.raises(SystemExit) as e:
        _run(probe, tmp_path, ["--eps-s", "50", "70", "90"])
    assert "0, 1" in str(e.value)


def test_a_contrast_touching_a_censored_arm_is_flagged_as_a_bound():
    """The magnitude is a lower bound, so the code must say so in the JSON."""
    import ast
    src = (REPO / "experiments" / "EVAL" / "probe.py").read_text()
    assert '"delta_is_bound"' in src
    assert "log1m_auc_censored" in src
    # Walk the AST rather than the text: the docstring explains the retired
    # epsilon on purpose, and a text search cannot tell that from a live one.
    floors = [n.value for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Constant) and isinstance(n.value, float)
              and 0 < n.value <= 1e-9]
    assert not floors, f"a sub-resolution epsilon floor is live in the code: {floors}"
