"""Merging per-arm anomaly runs. Every test here pins a failure that would
otherwise produce a clean-looking table rather than an error."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _payload(arm, sigma_min, *, sha="abc", trainings=10):
    return {
        "row_alignment_sha256": sha, "sigma_t": 5.0, "stat_cut": 0.2,
        "min_bkg_pass": 25, "trainings": trainings,
        "n_bkg": 200000, "n_template": 200000,
        "arms": {arm: {"signals": {"label_X_bb": {"100": {
            "knn": {"sigma_min": sigma_min, "argos": 0.01},
        }}}}},
    }


def test_cross_arm_regret_is_recomputed_over_all_arms_not_inherited():
    """A per-arm run normalises against itself and writes regret 1.000. If the
    merge kept those, the vocabulary ablation would report no regret for every
    arm -- the null under test, manufactured by the aggregation."""
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    an = _mod("anomaly", "experiments/EVAL/anomaly.py")
    a = _payload("l162-s1b", 2.0)
    b = _payload("r16q1-s2", 6.0)
    # simulate what a lone per-arm run leaves behind
    for p in (a, b):
        for v in p["arms"][list(p["arms"])[0]]["signals"]["label_X_bb"]["100"].values():
            v["regret"], v["regret_n_arms"] = 1.0, 1
    m = mg.merge([("a.json", a), ("b.json", b)])
    an.cross_arm_regret(m)
    got = {arm: ad["signals"]["label_X_bb"]["100"]["knn"]
           for arm, ad in m["arms"].items()}
    assert got["l162-s1b"]["regret"] == pytest.approx(1.0)
    assert got["r16q1-s2"]["regret"] == pytest.approx(3.0)
    assert all(v["regret_n_arms"] == 2 for v in got.values())


def test_a_different_row_ordering_is_fatal():
    """Arms that scored different jets cannot share a cross-arm minimum."""
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    with pytest.raises(SystemExit, match="row_alignment_sha256"):
        mg.merge([("a.json", _payload("x", 1.0, sha="aaa")),
                  ("b.json", _payload("y", 1.0, sha="bbb"))])


def test_a_different_configuration_is_fatal():
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    with pytest.raises(SystemExit, match="trainings"):
        mg.merge([("a.json", _payload("x", 1.0)),
                  ("b.json", _payload("y", 1.0, trainings=3))])


def test_a_duplicated_arm_is_fatal_rather_than_last_one_wins():
    """Two runs of the same arm are two different draws. Keeping one silently
    hides which run the published number came from."""
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    with pytest.raises(SystemExit, match="appears in both"):
        mg.merge([("a.json", _payload("same", 1.0)),
                  ("b.json", _payload("same", 9.0))])


def test_merge_records_which_file_each_arm_came_from():
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    m = mg.merge([("a.json", _payload("x", 1.0)), ("b.json", _payload("y", 2.0))])
    assert m["merged_from"] == {"x": "a.json", "y": "b.json"}


def test_a_single_arm_merge_warns_that_regret_is_degenerate(tmp_path, capsys):
    """Not fatal -- a one-arm merge is a legitimate intermediate -- but the
    regret column it yields is all 1.000 and must not be published."""
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    d = tmp_path / "in"; d.mkdir()
    (d / "anomaly_results.json").write_text(json.dumps(_payload("only", 1.0)))
    mg.main(["--inputs", str(d), "--out", str(tmp_path / "out")])
    assert "not publishable" in capsys.readouterr().out


def test_the_merged_artifact_carries_the_null_guard(tmp_path):
    """The merged file must record the same null bookkeeping a single-process
    run would have applied, or a reader cannot tell a passed null from an
    untested one."""
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    a, b = _payload("x", 1.0), _payload("y", 2.0)
    for p in (a, b):
        arm = list(p["arms"])[0]
        p["arms"][arm]["signals"]["label_X_bb"]["0"] = {"knn": {"argos": 0.9}}
    for i, p in enumerate((a, b)):
        d = tmp_path / f"in{i}"; d.mkdir()
        (d / "anomaly_results.json").write_text(json.dumps(p))
    out = tmp_path / "out"
    mg.main(["--inputs", str(tmp_path / "in0"), str(tmp_path / "in1"), "--out", str(out)])
    res = json.loads((out / "anomaly_results.json").read_text())
    assert len(res["null_not_flat"]) == 2      # ARGOS 0.9 on pure background
    assert "null_unmeasured" in res


# --------------------------------------------------------------------------
# Ragged grids. anomaly.py writes a complete-LOOKING results file after every
# signal, so an existence check cannot tell a finished arm from a partial one,
# and any arm killed by its deadline merges in silently. The l162-s1b payload
# the committed merge job reads holds 5 of 6 signals for exactly this reason.

def _ragged_payload(arms, signals, sigma=1.0):
    def cell():
        return {"250": {"knn": {"sigma_min": sigma, "max_sic": 2.0}}}
    return {
        "row_alignment_sha256": "deadbeef", "sigma_t": 2.0, "stat_cut": 0.5,
        "min_bkg_pass": 10, "trainings": 10, "n_bkg": 200000,
        "n_template": 200000,
        "arms": {a: {"signals": {s: cell() for s in signals}} for a in arms},
    }


def _run_merge(tmp_path, payloads):
    m = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    outs = []
    for i, p in enumerate(payloads):
        d = tmp_path / f"in{i}"
        d.mkdir()
        (d / "anomaly_results.json").write_text(json.dumps(p))
        outs.append(str(d))
    out = tmp_path / "merged"
    m.main(["--inputs", *outs, "--out", str(out)])
    return json.loads((out / "anomaly_results.json").read_text())


def test_a_ragged_grid_is_reported_not_silently_merged(tmp_path, capsys):
    merged = _run_merge(tmp_path, [
        _ragged_payload(["l162-s1b"], ["bb", "qq", "qqqq"], sigma=1.0),
        _ragged_payload(["r16q1-s2"], ["bb", "qq"], sigma=2.0),
    ])
    c = merged["completeness"]
    assert c["arms_missing_signals"] == {"r16q1-s2": ["qqqq"]}
    assert ["qqqq", "250"] in c["cells_normalised_against_one_arm"]
    assert "RAGGED" in capsys.readouterr().out


def test_a_single_arm_cell_is_flagged_even_though_its_regret_is_finite(tmp_path):
    merged = _run_merge(tmp_path, [
        _ragged_payload(["l162-s1b"], ["bb", "qqqq"], sigma=1.0),
        _ragged_payload(["r16q1-s2"], ["bb"], sigma=4.0),
    ])
    lone = merged["arms"]["l162-s1b"]["signals"]["qqqq"]["250"]["knn"]
    assert lone["regret"] == 1.0 and lone["regret_n_arms"] == 1, (
        "a cell only one arm reached normalises against itself")
    shared = merged["arms"]["r16q1-s2"]["signals"]["bb"]["250"]["knn"]
    assert shared["regret_n_arms"] == 2 and shared["regret"] == 4.0
    assert ["qqqq", "250"] in merged["completeness"]["cells_normalised_against_one_arm"]
    assert ["bb", "250"] not in merged["completeness"]["cells_normalised_against_one_arm"]


def test_a_complete_grid_reports_no_raggedness(tmp_path):
    merged = _run_merge(tmp_path, [
        _ragged_payload(["l162-s1b"], ["bb", "qq"], sigma=1.0),
        _ragged_payload(["r16q1-s2"], ["bb", "qq"], sigma=2.0),
    ])
    c = merged["completeness"]
    assert c["arms_missing_signals"] == {}
    assert c["cells_normalised_against_one_arm"] == []
    assert c["signals_seen"] == ["bb", "qq"]


# --------------------------------------------- one RUNG is not one ARM (2026-09-15)

def _rung_payload(arm, rung, signals):
    """A per-arm payload carrying an explicit rung, as anomaly.py writes."""
    cell = {"class_sum": {"max_sic": 2.0, "regret_n_arms": 3, "argos": 0.01}}
    return {"row_alignment_sha256": "x", "sigma_t": 5, "stat_cut": 0.2,
            "min_bkg_pass": 25, "trainings": 10, "n_bkg": 1000, "n_template": 1000,
            "arms": {arm: {"rung": rung,
                           "signals": {s: {"0": cell, "250": cell} for s in signals}}}}


def test_a_signal_carried_by_one_rung_is_flagged(capsys):
    """THE DEFECT THIS EXISTS FOR. Three R16_Q1 seeds satisfy regret_n_arms >= 2 --
    the filter this module tells the reader to publish on -- while carrying NO
    L162 arm, so the cell's regret is seed variation wearing a vocabulary
    ablation's clothes. Real: label_X_YY_qqqq in the 2026-09-15 merge."""
    both, coarse_only = "shared_sig", "coarse_only_sig"
    payloads = [
        (f"s{i}", _rung_payload(f"r16q1-s{i}", "R16_Q1", [both, coarse_only]))
        for i in (2, 3, 4)
    ] + [("f", _rung_payload("l162-s1b", "L162", [both]))]
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    merged = mg.merge(payloads)

    per = {}
    for arm, ad in merged["arms"].items():
        for sig in ad["signals"]:
            per.setdefault(sig, set()).add(ad["rung"])
    assert per[coarse_only] == {"R16_Q1"}, "fixture must exercise the defect"
    assert len(per[both]) == 2


def test_the_arm_count_alone_would_not_have_caught_it():
    """Why the existing guard is insufficient rather than merely incomplete."""
    payloads = [
        (f"s{i}", _rung_payload(f"r16q1-s{i}", "R16_Q1", ["only_coarse"]))
        for i in (2, 3, 4)
    ]
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    merged = mg.merge(payloads)
    n_arms = len(merged["arms"])
    rungs = {ad["rung"] for ad in merged["arms"].values()}
    assert n_arms >= 2, "passes the arm-count filter"
    assert len(rungs) == 1, "but there is only one rung, so no coarsening is measured"


def test_rung_falls_back_to_the_arm_name_when_absent():
    """Older payloads predate the `rung` field; the check must not crash on them."""
    p = _rung_payload("a1", "R16_Q1", ["s"])
    del p["arms"]["a1"]["rung"]
    mg = _mod("anomaly_merge", "experiments/EVAL/anomaly_merge.py")
    merged = mg.merge([("x", p)])
    assert merged["arms"]["a1"].get("rung", "a1") == "a1"
