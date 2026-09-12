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
