"""experiments/AOJ/sim_scores.py: the real-data scores on simulation must be the
same function of the logits as on data, and every model must be on the same jets."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sim = _load("experiments/AOJ/sim_scores.py", "aoj_sim_scores")


def _extract(tmp, name, n=50, k=17, num_reg=0, seed=0, pt_shift=0.0):
    rng = np.random.default_rng(seed)
    ext = tmp / name
    ext.mkdir()
    np.save(ext / "logits.npy", rng.normal(size=(n, k + num_reg)).astype(np.float32))
    obs_rng = np.random.default_rng(99)                       # the jets are the same for every model
    np.savez(ext / "observers.npz", jet_pt=obs_rng.uniform(500, 2500, n).astype(np.float32) + pt_shift,
             jet_eta=obs_rng.uniform(-2.4, 2.4, n).astype(np.float32),
             jet_sdmass=obs_rng.uniform(20, 500, n).astype(np.float32))
    np.save(ext / "label188.npy", obs_rng.integers(0, 188, n).astype(np.int16))
    man = dict(checkpoint=f"/c/{name}.pt", checkpoint_sha256="x", data_config="d", data_config_sha256="y")
    if num_reg:
        man["logit_columns"] = {"class_logits": [0, k], "regression": [k, k + num_reg]}
    (ext / "extract_manifest.json").write_text(json.dumps(man))
    return ext


def _run(monkeypatch, ext, out, name, rung="R16_Q1"):
    monkeypatch.setattr(sys, "argv", ["sim_scores.py", "--name", name, "--rung", rung,
                                      "--extract-dir", str(ext), "--out", str(out)])
    return sim.main()


def test_scores_are_the_discriminants_functions_of_the_class_logits(tmp_path, monkeypatch):
    ext = _extract(tmp_path, "a", num_reg=1)
    assert _run(monkeypatch, ext, tmp_path / "out", "a") == 0
    s = np.load(tmp_path / "out" / "scores_a.npz")
    x = np.load(ext / "logits.npy")[:, :17]
    for k in ("three_prong", "prong_only"):
        np.testing.assert_array_equal(s[f"{k}_logodds"], sim.disc.contrast(x, "R16_Q1", k).astype(np.float32))
    j = np.load(tmp_path / "out" / "jets.npz")
    assert sorted(j.files) == ["jet_eta", "jet_pt", "jet_sdmass", "label"] and j["label"].dtype == np.int16


def test_a_second_model_on_the_same_jets_is_accepted_and_on_other_jets_refused(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert _run(monkeypatch, _extract(tmp_path, "a", seed=1), out, "a") == 0
    assert _run(monkeypatch, _extract(tmp_path, "b", seed=2), out, "b") == 0
    with pytest.raises(SystemExit, match="scored on other jets"):
        _run(monkeypatch, _extract(tmp_path, "c", seed=3, pt_shift=1.0), out, "c")


def test_a_head_of_the_wrong_width_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="not a L188 head"):
        _run(monkeypatch, _extract(tmp_path, "a", k=17), tmp_path / "out", "a", rung="L188")


def test_every_model_records_the_digest_of_the_jets_it_was_scored_on(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert _run(monkeypatch, _extract(tmp_path, "a", seed=1), out, "a") == 0
    assert _run(monkeypatch, _extract(tmp_path, "b", seed=2), out, "b") == 0
    got = {json.loads((out / f"scores_{m}.json").read_text())["jets_sha256"] for m in "ab"}
    assert got == {sim.jets_digest(dict(np.load(out / "jets.npz")))}
    assert not list(out.glob(".jets.*")), "the private file must be renamed into place"
