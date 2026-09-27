"""Leg 2 log parsing. A parser that fails quietly moves a published mean."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "leg2_metrics", REPO / "experiments/FT/leg2_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(root, parts, text):
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "predict.log").write_text(text)


def test_an_unparseable_cell_is_fatal_not_dropped(tmp_path):
    """Dropping it shrinks the denominator and shifts the mean with nothing
    visible to show for it."""
    m = _mod()
    _log(tmp_path, ("init", "N10000", "s1"), "no metric here")
    with pytest.raises(SystemExit, match="must not be silently dropped"):
        m.main(["--root", str(tmp_path), "--out", str(tmp_path / "o")])


def test_partial_directories_are_skipped_by_name(tmp_path):
    m = _mod()
    _log(tmp_path, ("init", "N10000", "s1"), "Test metric 0.5")
    _log(tmp_path, ("init", "N10000", "s1.partial.99"), "Test metric 0.9")
    got = m.discover(tmp_path)
    assert [c[2] for c in got] == ["s1"]


def test_the_last_metric_wins_when_a_pass_was_retried(tmp_path):
    """A retried predict pass appends rather than truncating."""
    m = _mod()
    _log(tmp_path, ("init", "N10000", "s1"), "Test metric 0.10\nTest metric 0.80\n")
    assert m.cell_metric(tmp_path / "init/N10000/s1/predict.log") == pytest.approx(0.80)


def test_reference_runs_one_level_shallower_are_collected(tmp_path):
    m = _mod()
    _log(tmp_path, ("ref_e1arms-s1",), "Test metric 0.8177")
    got = m.discover(tmp_path)
    assert got and got[0][0] == "ref_e1arms-s1" and got[0][1] == "ref"


def test_summary_uses_ddof_1_and_reports_none_for_a_single_seed(tmp_path):
    m = _mod()
    for s, v in (("s1", 0.80), ("s2", 0.82), ("s3", 0.84)):
        _log(tmp_path, ("init", "N10000", s), f"Test metric {v}")
    _log(tmp_path, ("solo", "N10000", "s1"), "Test metric 0.5")
    out = tmp_path / "o"
    m.main(["--root", str(tmp_path), "--out", str(out)])
    d = json.loads((out / "leg2_metrics.json").read_text())["summary"]
    assert d["init"]["N10000"]["accuracy_sd"] == pytest.approx(0.02)
    assert d["solo"]["N10000"]["accuracy_sd"] is None


def _pred_root(m, monkeypatch, truth, scores, names=("QCD", "Hbb", "Hcc")):
    """Stand in for weaver's pred.root: the reader is the only uproot call."""
    import numpy as np
    monkeypatch.setattr(m, "read_pred_root",
                        lambda path: (list(names), np.eye(len(names))[truth].astype(np.int32),
                                      scores.astype(np.float32)))


def test_macro_auc_comes_from_pred_root_and_must_agree_with_the_log(tmp_path, monkeypatch):
    """PRESPEC 2.3's JetClass endpoint is log(1 - macro AUC); the log only has
    accuracy. The recomputed accuracy must match the log's, or the file is not
    the pass the log describes."""
    import numpy as np
    m = _mod()
    rng = np.random.default_rng(0)
    truth = np.tile([0, 1, 2], 400)
    scores = rng.random((truth.size, 3)) + 2.0 * np.eye(3)[truth]
    scores /= scores.sum(1, keepdims=True)
    acc = float((scores.argmax(1) == truth).mean())
    d = tmp_path / "init" / "N10000" / "s1"
    d.mkdir(parents=True)
    (d / "predict.log").write_text(f"Test metric {acc:.6f}\n")
    _pred_root(m, monkeypatch, truth, scores)
    assert m.main(["--root", str(tmp_path), "--out", str(tmp_path / "o"), "--macro-auc"]) == 0
    c = json.loads((tmp_path / "o" / "leg2_metrics.json").read_text())["cells"]["init"]["N10000"]["s1"]
    assert 0.9 < c["macro_auc_ovr"] <= 1.0 and c["classes"] == ["QCD", "Hbb", "Hcc"]
    (d / "predict.log").write_text(f"Test metric {acc - 0.01:.6f}\n")
    with pytest.raises(SystemExit, match="not the same pass"):
        m.main(["--root", str(tmp_path), "--out", str(tmp_path / "o2"), "--macro-auc"])
