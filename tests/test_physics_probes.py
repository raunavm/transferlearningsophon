"""The two published physics discriminants must be the published ones.

docs/PRD_PLAN.md 3.1(b) makes the paper's D4 trigger depend on the arm ordering
of the |V_cb| discriminant matching the physics figure. That only means
anything if the probe reproduces arXiv:2503.00118's construction: its class
set, its kinematic window, and its working points. Any of the three drifting
would still fit, still report an AUC, and answer a different question.

The collapse rungs here are DERIVED from the committed label map at all eight
rungs, not copied from prose. Doing that corrected docs/PRD_PLAN.md 3.1(b),
which states the ee/mumu split merges "from R42_Q1 down" when it actually
merges one rung earlier, at R63_Q1, into 2P_LEP_LL|nb0_nc0.
"""
import csv
import importlib.util
import json
import pathlib

import sys
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]


def _probe():
    s = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


probe = _probe()
rows = {int(r["jet_label"]): r for r in csv.DictReader(MAP.open())}
by_name = {r["class_name"]: int(r["jet_label"]) for r in rows.values()}


def test_qcd_set_is_read_from_the_map_and_is_27():
    q = probe.qcd_indices()
    assert len(q) == 27, "docs/GROUND_TRUTH.md: 161 resonant + 27 QCD"
    assert all(rows[i]["class_name"].startswith("label_QCD_") for i in q)


def test_bc_discriminant_uses_the_published_class_set():
    """arXiv:2503.00118 Eq. 1: D_bc = g_bc/(g_bc+g_bq+g_cs+g_bqq+g_QCD)."""
    spec = probe.TASKS["bc_vs_rest"]
    assert spec["signal"] == [by_name["label_X_bc"]]
    bkg = set(spec["background"])
    assert by_name["label_X_bq"] in bkg
    assert by_name["label_X_cs"] in bkg
    assert by_name["label_X_YY_qqb"] in bkg, "bqq is the 3-prong q,q,b class"
    assert set(probe.qcd_indices()) <= bkg, "QCD is in the published denominator"
    assert len(bkg) == 3 + 27
    assert by_name["label_X_bb"] not in bkg, "bb is not in Eq. 1's denominator"


def test_bc_window_is_the_published_one():
    w = probe.TASKS["bc_vs_rest"]["window"]
    assert w["jet_pt"] == [450.0, 600.0]
    assert w["jet_sdmass"] == [90.0, 140.0]
    # arXiv:2503.00118 App. A states THREE cuts. The eta cut was missing and a
    # test pinned the two-cut window as "the published one"; JetClass-II runs
    # to |eta| < 2.5, so the band it drops is populated.
    assert w["jet_eta"] == [-2.4, 2.4], "the published |eta| < 2.4 cut"
    assert set(w) == {"jet_pt", "jet_sdmass", "jet_eta"}
    assert probe.TASKS["bc_vs_rest"]["eps_s"] == [0.60, 0.40]


@pytest.mark.parametrize("task", sorted(probe.TASKS))
def test_collapse_rungs_derived_at_all_eight_rungs(task):
    spec = probe.TASKS[task]
    derived = []
    for g in RUNGS:
        sg = {rows[i][g] for i in spec["signal"]}
        bg = {rows[i][g] for i in spec["background"]}
        if sg & bg:
            derived.append(g)
    assert derived == spec["collapsed_at"], (
        f"{task}: map says collapsed at {derived}, task claims "
        f"{spec['collapsed_at']}")


def test_ee_mm_merges_at_r63_not_r42():
    """The correction to docs/PRD_PLAN.md 3.1(b), pinned so it cannot regress."""
    ee, mm = by_name["label_X_ee"], by_name["label_X_mm"]
    assert rows[ee]["R63_Q1"] == rows[mm]["R63_Q1"], "they merge at R63_Q1"
    assert rows[ee]["L162"] != rows[mm]["L162"], "and are still distinct at L162"
    assert "R63_Q1" in probe.TASKS["ee_vs_mm"]["collapsed_at"]


def _cache(d, n, labels, pt, msd, rng, eta=None):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "features.npy", rng.normal(size=(n, 8)).astype(np.float32))
    np.save(d / "label188.npy", labels.astype(np.int16))
    np.savez(d / "observers.npz", jet_pt=pt, jet_sdmass=msd,
             jet_eta=np.zeros(n, dtype=np.float32) if eta is None else eta)
    (d / "extract_manifest.json").write_text(json.dumps({"arm": d.name}))


def test_window_actually_filters(tmp_path):
    """Half the jets are outside the window; the task must use only the half in."""
    rng = np.random.default_rng(0)
    n = 4000
    lab = np.where(np.arange(n) % 2 == 0, by_name["label_X_bc"], by_name["label_X_bq"])
    pt = np.where(np.arange(n) < n // 2, 500.0, 900.0)      # half inside 450-600
    msd = np.full(n, 100.0)                                  # all inside 90-140
    d = tmp_path / "arm"
    _cache(d, n, lab, pt, msd, rng)
    # Drive probe.main() rather than re-deriving the cut here. Re-implementing
    # the window in the test means deleting it from probe.py would fail
    # nothing -- the test would keep passing on its own copy of the logic.
    res = _run_probe(tmp_path, {"arm": d}, ["bc_vs_rest"])["tasks"]["bc_vs_rest"]
    assert res.get("n") == n // 2, (
        f"the window must drop the out-of-window half: got {res.get('n')} "
        f"of {n}")


def _run_probe(tmp_path, arms, tasks):
    """Run probe.main() end to end on synthetic caches and return its JSON."""
    out = tmp_path / "probe_out"
    argv = ["probe.py", "--features"] + [f"{k}={v}" for k, v in arms.items()] + \
           ["--out", str(out), "--tasks"] + tasks + ["--bootstrap", "50"]
    old = sys.argv
    try:
        sys.argv = argv
        probe.main()
    finally:
        sys.argv = old
    return json.loads(next(out.glob("*.json")).read_text())


def test_eta_cut_is_applied_by_probe_not_just_declared(tmp_path):
    """Jets in the 2.4-2.5 band that JetClass-II populates must be dropped."""
    rng = np.random.default_rng(0)
    n = 4000
    lab = np.where(np.arange(n) % 2 == 0, by_name["label_X_bc"], by_name["label_X_bq"])
    pt = np.full(n, 500.0)
    msd = np.full(n, 100.0)
    eta = np.where(np.arange(n) < n // 2, 0.5, 2.45)   # half in the forward band
    d = tmp_path / "arm"
    _cache(d, n, lab, pt, msd, rng, eta=eta)
    res = _run_probe(tmp_path, {"arm": d}, ["bc_vs_rest"])["tasks"]["bc_vs_rest"]
    assert res.get("n") == n // 2, (
        f"|eta| < 2.4 must drop the forward half: got {res.get('n')} of {n}")


def test_guard_is_per_class_not_on_the_union(tmp_path):
    """A huge background with a tiny signal must SKIP, not report a number."""
    rng = np.random.default_rng(0)
    n = 40_000
    lab = np.full(n, by_name["label_X_bq"])       # all background ...
    lab[:50] = by_name["label_X_bc"]              # ... except 50 signal jets
    d = tmp_path / "arm"
    _cache(d, n, lab, np.full(n, 500.0), np.full(n, 100.0), rng)
    res = _run_probe(tmp_path, {"arm": d}, ["bc_vs_rest"])["tasks"]["bc_vs_rest"]
    assert res.get("skipped") is True, (
        "the union clears the floor while the signal count does not; a union "
        "guard cannot see that")
    assert res["n_signal"] == 50


def test_load_arm_rejects_observers_of_the_wrong_length(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "arm"
    d.mkdir()
    np.save(d / "features.npy", rng.normal(size=(100, 8)).astype(np.float32))
    np.save(d / "label188.npy", np.zeros(100, dtype=np.int16))
    np.savez(d / "observers.npz", jet_pt=np.zeros(50))
    (d / "extract_manifest.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        probe.load_arm(d)
    assert "observer" in str(e.value)


def test_rejection_reported_at_both_published_working_points():
    y = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(int)
    s = np.concatenate([np.random.default_rng(1).normal(1, 1, 2000),
                        np.random.default_rng(2).normal(0, 1, 2000)])
    r60 = probe.rejection_at(y, s, 0.60)[0]
    r40 = probe.rejection_at(y, s, 0.40)[0]
    assert r40 > r60, "a tighter signal efficiency must reject more background"
