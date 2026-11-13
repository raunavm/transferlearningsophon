"""The JetClass accuracy bootstrap (PI 2026-11-09): paired by seed, the same resample for every run, replica 0
equal to the stored read-out, pass only beyond 3 sigma with sigma^2 = run SD^2 + bootstrap SD^2."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jc1_acc_boot", ROOT / "experiments/FT/jc1_acc_boot.py")
jb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jb)
N = 20000


def _setup(tmp_path, monkeypatch, acc_of, bad_labels=None):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, N)
    data, stored = {}, {}
    for init, ks in jb.ARM_RUNS.items():
        for k in ks:
            key = f"leg2/{init}-s{k}/N1000/s1"
            (tmp_path / key).mkdir(parents=True)
            (tmp_path / key / "DONE").write_text("")
            c = rng.random(N) < acc_of(init, k)
            data[str(tmp_path / key / "pred.root")] = (c, y if key != bad_labels else (y + 1) % 10)
            stored[key] = {"accuracy": float(c.mean())}
    monkeypatch.setattr(jb, "correct", lambda p: data[str(p)])
    m = tmp_path / "metrics.json"
    m.write_text(json.dumps({"cells": stored}))
    return m


def test_a_resolved_drop_passes_and_no_drop_does_not(tmp_path, monkeypatch):
    acc = {"l188": 0.85, "l162": 0.85, "r42q1": 0.85, "r16q1": 0.80, "l162mass": 0.85, "r16q1mass": 0.80}
    m = _setup(tmp_path, monkeypatch, lambda init, k: acc[init])
    out = tmp_path / "boot.json"
    jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--out", str(out)])
    d = json.loads(out.read_text())["diffs"]
    assert d["r16q1-l188"]["accuracy"]["pass"] and d["r16q1-l188"]["accuracy"]["delta"] < 0
    assert d["r16q1-l188"]["error_ratio"]["delta"] > 0
    assert not d["l162-l188"]["accuracy"]["pass"]
    for q in ("accuracy", "error_ratio"):
        s = d["r16q1-l188"][q]
        assert s["sigma"] == pytest.approx(np.hypot(s["run_sd"], s["boot_sd"])) and s["boot_sd"] > 0
    with pytest.raises(SystemExit):           # never overwritten
        jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--out", str(out)])


def test_it_refuses_unpaired_jets_a_wrong_replica_zero_and_few_replicas(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, lambda init, k: 0.8, bad_labels="leg2/r16q1-s2/N1000/s1")
    with pytest.raises(SystemExit, match="test jets differ"):
        jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--out", str(tmp_path / "o.json")])
    doc = json.loads(m.read_text())
    doc["cells"]["leg2/l188-s1/N1000/s1"]["accuracy"] += 1e-6
    m.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="stored"):
        jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="replicas"):
        jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--replicas", "50",
                 "--out", str(tmp_path / "o.json")])


def test_the_mass_pairs_are_read_paired_on_their_two_runs_and_unpaired_by_decision_l3(tmp_path, monkeypatch):
    acc = {"l188": 0.85, "l162": 0.85, "r42q1": 0.85, "r16q1": 0.85, "l162mass": 0.85, "r16q1mass": 0.80}
    m = _setup(tmp_path, monkeypatch, lambda init, k: acc[init])
    out = tmp_path / "boot.json"
    jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--out", str(out)])
    doc = json.loads(out.read_text())
    d = doc["diffs"]
    assert set(d) == {"l162-l188", "r42q1-l188", "r16q1-l188", "l162mass-l162", "r16q1mass-r16q1"}
    assert d["r16q1mass-r16q1"]["runs"] == [1, 2] and d["l162-l188"]["runs"] == [1, 2, 3]
    u = d["r16q1mass-r16q1"]["l3"]["accuracy"]
    a = doc["accuracy"]
    c, r = ([a[f"leg2/{x}-s{k}/N1000/s1"] for k in jb.ARM_RUNS[x]] for x in ("r16q1mass", "r16q1"))
    assert (u["n_config"], u["n_reference"]) == (2, 3) and u["delta"] == pytest.approx(np.mean(c) - np.mean(r))
    assert u["sd_config"] == pytest.approx(np.std(c, ddof=1)) and u["sd_reference"] == pytest.approx(np.std(r, ddof=1))
    assert u["sigma"] == pytest.approx(np.sqrt(u["sd_config"] ** 2 + u["sd_reference"] ** 2 + u["boot_sd"] ** 2))
    assert u["delta"] < 0 and u["pass"] and u["boot_sd"] > 0
    assert not d["l162mass-l162"]["l3"]["accuracy"]["pass"]
    # one pair only: no other cells are read
    only = tmp_path / "only.json"
    jb.main(["--root", str(tmp_path), "--n", "1000", "--metrics", str(m), "--diffs", "l162mass-l162", "--out", str(only)])
    assert set(json.loads(only.read_text())["accuracy"]) == {f"leg2/{x}-s{k}/N1000/s1" for x in ("l162mass", "l162")
                                                             for k in jb.ARM_RUNS[x]}
