"""experiments/EVAL/probe_sophon.py (PI 2026-11-23, LEDGER SH-13): Sophon's CPU probes are our functions pointed at its
stored features; the mass view holds exactly the observers' jets."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_sophon", REPO / "experiments/EVAL/probe_sophon.py")
PS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PS)


def _fixture(tmp, n=50, extra=20, bad=None):
    src, obs = tmp / "massreg", tmp / "obs.npz"
    src.mkdir()
    r = np.random.default_rng(0)
    lab = r.integers(0, 188, n + extra).astype(np.int64)
    ob = {k: r.random(n + extra).astype(np.float32) for k in ("jet_pt", "jet_sdmass", "genjet_sdmass", "jet_eta")}
    np.save(src / "label188.npy", lab)
    np.save(src / "features.npy", r.normal(size=(n + extra, 128)).astype(np.float32))
    np.savez(src / "observers.npz", **ob)
    (src / "extract_manifest.json").write_text(json.dumps({"checkpoint_sha256": "abc"}))
    o = {k: v[:n].copy() for k, v in ob.items()}
    if bad:
        o[bad][3] += 1.0
    np.savez(obs, rows=np.arange(n), label188=lab[:n].astype(np.int16), **o)
    return src, obs


def test_the_mass_view_is_the_first_rows_in_the_observers_dtype(tmp_path):
    src, obs = _fixture(tmp_path)
    out = tmp_path / "view"
    man = PS.run_massview(out, src, obs, 50)
    f = np.load(out / "features.npy")
    assert f.shape == (50, 128) and np.array_equal(f, np.load(src / "features.npy")[:50])
    lab = np.load(out / "label188.npy")
    assert lab.dtype == np.int16 and np.array_equal(np.load(out / "rows.npy"), np.arange(50))
    assert man["prefix_features"] == man["n_stream"] == 50 and json.loads((out / "manifest.json").read_text())["n_stream"] == 50


@pytest.mark.parametrize("bad", ["jet_sdmass", "genjet_sdmass", "jet_pt", "jet_eta"])
def test_the_mass_view_refuses_other_jets(tmp_path, bad):
    src, obs = _fixture(tmp_path, bad=bad)
    with pytest.raises(SystemExit):
        PS.run_massview(tmp_path / "view", src, obs, 50)


def test_the_mass_view_passes_mass_resolutions_prefix_and_alignment(tmp_path):
    src, obs = _fixture(tmp_path)
    out = tmp_path / "view"
    PS.run_massview(out, src, obs, 50)
    spec = importlib.util.spec_from_file_location("mass_resolution", REPO / "experiments/EVAL/mass_resolution.py")
    MR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MR)
    assert np.array_equal(MR.v2_prefix(out, 50), np.arange(50))
    import hashlib
    sha = hashlib.sha256(np.load(obs)["label188"].tobytes()).hexdigest()
    assert MR.check_alignment("released", out, sha, 50)["label188_sha256"] == sha


def test_probe_bal_reads_sophons_root(tmp_path):
    pb = PS.sophon_probe_bal(tmp_path)
    assert pb.NEW_ROOT == pb.OLD_ROOT == tmp_path
    assert pb.load.keywords["root"] == tmp_path
