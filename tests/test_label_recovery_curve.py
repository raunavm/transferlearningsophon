"""experiments/EVAL/label_recovery_curve.py: the audit's B8 protocol -- nested
training sizes up to the whole pool, a class-weighted linear probe, the MLP as
the capacity check, and a resumable output."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location(
    "label_recovery_curve", ROOT / "experiments" / "EVAL" / "label_recovery_curve.py")
lc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lc)


def _cache(d, labels, rng, sep):
    d.mkdir(parents=True, exist_ok=True)
    F = rng.normal(size=(labels.size, 10)).astype(np.float32)
    F[:, 0] += sep * (labels % 17)
    F[:, 1] += sep * (labels % 5)
    np.save(d / "features.npy", F)
    np.save(d / "label188.npy", labels.astype(np.int16))
    (d / "extract_manifest.json").write_text('{"arm": "t"}')


def test_split_is_fixed_nested_and_disjoint():
    te, va, pool, sizes = lc.split(10_000, [100, 1000, 0])
    assert te.size == 2000 and va.size == 1000 and pool.size == 7000
    assert sizes == [100, 1000, 7000]
    assert not (set(te) & set(va)) and not (set(te) | set(va)) & set(pool)
    te2, _, pool2, _ = lc.split(10_000, [100])
    assert (te == te2).all() and (pool == pool2).all()
    with pytest.raises(SystemExit):
        lc.split(1000, [5000])


def test_default_curve_has_five_sizes_ending_at_the_whole_pool():
    assert len(set(lc.DEFAULT_SIZES)) >= 5 and 0 in lc.DEFAULT_SIZES


def test_linear_probe_is_class_weighted():
    # 95/5 imbalance with overlapping classes: weighting must raise the balanced
    # accuracy over an unweighted fit, which is what the metric rewards.
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) < 0.05).astype(int)
    X = rng.normal(size=(n, 4)) + 1.0 * y[:, None]
    w = lc.fit_linear(X[:4000], y[:4000], X[4000:], y[4000:])
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    u = LogisticRegression().fit(X[:4000], y[:4000])
    assert w["balanced_accuracy"] > balanced_accuracy_score(y[4000:], u.predict(X[4000:])) + 0.05
    assert w["converged"]


def test_main_writes_a_rising_curve_and_the_mlp_check_and_resumes(tmp_path):
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 188, size=6000)
    _cache(tmp_path / "arm", lab, rng, sep=0.6)
    out = tmp_path / "o"
    argv = ["--features", f"a={tmp_path / 'arm'}", "--own-rung", "a=R16_Q1",
            "--out", str(out), "--sizes", "60", "300", "1200", "0",
            "--rungs", "R16_Q1", "R1_Q1", "--mlp-rungs", "R16_Q1", "--threads", "1"]
    assert lc.main(argv) == 0
    res = json.loads((out / "label_recovery_curve.json").read_text())
    assert res["sizes"] == [60, 300, 1200, 4200]
    assert res["linear"]["class_weight"] == "balanced"
    cell = res["arms"]["a"]["rungs"]["R16_Q1"]
    curve = [c["balanced_accuracy"] for c in cell["curve"]]
    assert [c["n_train"] for c in cell["curve"]] == res["sizes"]
    assert curve[-1] > curve[0] + 0.02
    assert cell["mlp"]["n_train"] == 4200 and "converged" in cell["mlp"]
    assert "mlp" not in res["arms"]["a"]["rungs"]["R1_Q1"]
    # resume: a finished cell is not refitted, and a changed size list is refused
    before = (out / "label_recovery_curve.json").read_text()
    assert lc.main(argv) == 0
    assert (out / "label_recovery_curve.json").read_text() == before
    with pytest.raises(SystemExit, match="other jets or sizes"):
        lc.main(argv[:7] + ["60", "0"] + argv[11:])
