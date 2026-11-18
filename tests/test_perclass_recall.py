"""experiments/EVAL/perclass_recall.py (per-class recall, PI 2026-11-18): the recall of each class, its identity with
the accuracy, the test-jet bootstrap and decision L3's differences; the linear probe's logits rebuilt from the saved
probe; the fine-tuning cells it refuses."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PR = _load("perclass_recall", "experiments/EVAL/perclass_recall.py")
PB = _load("paired_bootstrap", "experiments/STATS/paired_bootstrap.py")


def _preds(y, k, rate, seed):
    """A prediction right with probability rate[c] on class c, else a wrong class."""
    rng = np.random.default_rng(seed)
    ok = rng.random(y.size) < np.asarray(rate)[y]
    return np.where(ok, y, (y + 1 + rng.integers(0, k - 1, y.size)) % k)


def test_recall_is_the_class_by_class_breakdown_of_the_accuracy_on_a_balanced_set():
    k, n = 4, 2000
    y = np.repeat(np.arange(k), n)
    rate = [0.9, 0.7, 0.5, 0.3]
    preds = {f"a-s{s}": _preds(y, k, rate, s) for s in (1, 2, 3)}
    preds |= {f"b-s{s}": _preds(y, k, [0.9, 0.7, 0.2, 0.3], 10 + s) for s in (1, 2, 3)}
    groups = {"L188": [f"a-s{s}" for s in (1, 2, 3)], "R16_Q1": [f"b-s{s}" for s in (1, 2, 3)]}
    d = PR.recall_doc(preds, y, k, groups, 200, 7, PB.counts)
    assert d["balanced"] and d["class_counts"] == [n] * k
    for m, p in preds.items():
        r = d["models"][m]
        assert r["recall"] == pytest.approx([float((p[y == c] == c).mean()) for c in range(k)], abs=0)
        assert r["accuracy"] == pytest.approx(r["mean_recall"], abs=1e-12) and r["n_correct"] == int((p == y).sum())
        # within-class resampling: the binomial SD of a recall, to the bootstrap's precision at B = 200
        binom = np.sqrt(np.array(r["recall"]) * (1 - np.array(r["recall"])) / n)
        assert np.array(r["boot_sd"]) == pytest.approx(binom, rel=0.2)
    diff = d["diffs"]["R16_Q1-L188"]
    assert diff["lower"] == [2] and diff["higher"] == []
    assert diff["delta"][2] == pytest.approx(np.mean([d["models"][m]["recall"][2] for m in groups["R16_Q1"]])
                                             - np.mean([d["models"][m]["recall"][2] for m in groups["L188"]]))
    assert set(d["groups"]) == {"L188", "R16_Q1"} and len(d["groups"]["L188"]["boot_sd_mean"]) == k


def test_an_unequal_test_set_keeps_the_jet_weighted_identity_and_records_the_plain_mean():
    """The class-balanced JetClass-II test set holds 224 to 5,000 jets per class: there the accuracy is the
    jet-weighted mean of the recalls, not their plain mean, and both are recorded."""
    k = 3
    y = np.concatenate([np.zeros(3000, int), np.ones(3000, int), np.full(300, 2)])
    p = _preds(y, k, [0.9, 0.8, 0.1], 0)
    d = PR.recall_doc({"m": p}, y, k, {}, 50, 1, PB.counts)
    r = d["models"]["m"]
    assert not d["balanced"] and r["accuracy"] == pytest.approx(float((p == y).mean()), abs=0)
    assert r["accuracy"] == pytest.approx(sum(c * x for c, x in zip(d["class_counts"], r["recall"])) / y.size, abs=1e-12)
    assert abs(r["mean_recall"] - r["accuracy"]) > 0.05


def test_a_side_of_one_run_has_no_sigma_and_counts_no_class():
    k, y = 2, np.repeat(np.arange(2), 500)
    preds = {"a": _preds(y, k, [0.9, 0.9], 0), "b-s1": _preds(y, k, [0.1, 0.1], 1), "b-s2": _preds(y, k, [0.1, 0.1], 2)}
    d = PR.recall_doc(preds, y, k, {"RAND": ["a"], "L188": ["b-s1", "b-s2"]}, 20, 1, PB.counts)
    diff = d["diffs"]["RAND-L188"]
    assert diff["sigma"] is None and diff["sd_a"] is None and diff["lower"] == diff["higher"] == []


def test_an_incomplete_group_gives_no_difference():
    k, y = 2, np.repeat(np.arange(2), 100)
    d = PR.recall_doc({"a-s1": y.copy(), "b-s1": y.copy()}, y, k, {"L162": ["a-s1", "a-s2"], "L188": ["b-s1"]}, 10, 1,
                      PB.counts)
    assert "L162" not in d["groups"] and d["diffs"] == {}


def _probe_files(tmp_path, monkeypatch, n=400, k=5, stored_shift=0):
    """A saved v5 probe and its stored test features, laid out as on /data, under tmp_path."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 128)).astype(np.float32)
    y = rng.integers(0, k, n)
    coef, b = rng.normal(size=(k, 128)), rng.normal(size=k)
    mu, sc = rng.normal(size=128) * 0.1, rng.random(128) + 0.5
    a = x.copy()
    a -= mu
    a /= sc
    acc = float((((a.astype(np.float64) @ coef.T) + b).argmax(1) == y).mean())
    bal = tmp_path / "bal"
    fdir = bal / "mtx-l188-s1" / "jc2bal" / "test" / "best70"
    fdir.mkdir(parents=True)
    np.save(fdir / "features.npy", x)
    np.save(fdir / "label188.npy", y.astype(np.int64))
    np.save(fdir / "rows.npy", np.arange(n))
    pdir = bal / "fits_v5_probs" / "mtx-l188-s1" / "best70"
    pdir.mkdir(parents=True)
    np.savez(pdir / f"jc2bal_N{PR.N_PROBE}.npz", rows=np.arange(n), y=y.astype(np.int16), coef=coef, intercept=b,
             classes=np.arange(k), scaler_mean=mu, scaler_scale=sc, C=np.float64(1.0),
             p=np.zeros((n, k), np.float16))
    (bal / "fits_v5").mkdir()
    (bal / "fits_v5" / "mtx-l188-s1.json").write_text(json.dumps(
        {"cells": {"best70": {"jc2bal": {str(PR.N_PROBE): {"accuracy": acc + stored_shift / n}}}}}))
    monkeypatch.setattr(PR, "PROBE_BAL", bal)
    monkeypatch.setitem(PR.TASKS, "jc2", (k, "jc2bal", bal, "leg_jc2b"))
    return x, y, acc


def test_the_probe_logits_are_rebuilt_from_the_saved_probe_and_match_the_stored_accuracy(tmp_path, monkeypatch):
    _, y, acc = _probe_files(tmp_path, monkeypatch)
    pred, y2, rec = PR.probe_pred("jc2", "mtx-l188-s1", "best70")
    assert np.array_equal(y2, y) and float((pred == y).mean()) == acc and rec["accuracy_stored"] == acc


def test_a_probe_whose_correct_count_differs_from_the_stored_accuracy_is_refused(tmp_path, monkeypatch):
    _probe_files(tmp_path, monkeypatch, stored_shift=1)
    with pytest.raises(SystemExit) as e:
        PR.probe_pred("jc2", "mtx-l188-s1", "best70")
    assert e.value.code == 42


def test_fine_tuning_cells_without_done_are_listed_missing_never_read(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "FT_ROOT", tmp_path)
    for rel, done in (("leg2/l188-s1/N1000000/s1", True), ("leg2/l162-s1/N1000000/s1", False),
                      ("leg2/scratch/N1000000/s2", True)):
        (tmp_path / rel).mkdir(parents=True)
        if done:
            (tmp_path / rel / "DONE").write_text("")
    cells, missing = PR.ft_cells("jc1", 1_000_000, ["mtx-l188-s1", "mtx-l162-s1", "mtx-r16q1-s1"])
    assert set(cells) == {"mtx-l188-s1", "scratch-s2"}
    assert set(missing) == {"leg2/l162-s1/N1000000/s1", "leg2/r16q1-s1/N1000000/s1", "leg2/scratch/N1000000/s1",
                            "leg2/scratch/N1000000/s3"}


def test_the_groups_are_the_paper_runs_with_the_random_partitions_pooled():
    g = PR.paper_groups()
    assert len(g["L188"]) == 3 and len(g["L162_MASS"]) == 2 and len(g["RAND"]) == 5
    assert sum(len(v) for a, v in g.items() if a != "RAND") == 27
    assert {a for pair in PR.DIFFS for a in pair} <= set(g)
