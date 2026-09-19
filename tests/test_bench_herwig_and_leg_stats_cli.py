"""The two read-out changes of 2026-09-18: bench_metrics.py --herwig and the
generalised leg_stats.py CLI. Both must be no-ops on what already exists."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("bench_metrics", "experiments/FT/bench_metrics.py")
L = _load("leg_stats", "experiments/FT/leg_stats.py")


def _cell(root, y, z, dataset="qg", init="l162-s1b", n="N1000", seed="s1", herwig=None):
    cell = root / f"leg_{dataset}" / init / n / seed
    fd = cell / "features"
    fd.mkdir(parents=True)
    z = np.asarray(z, dtype=np.float32)
    np.save(fd / "label188.npy", np.asarray(y).astype(np.int16))
    np.save(fd / "logits.npy", np.stack([np.zeros_like(z), z], axis=1))
    if herwig is not None:
        yh, zh = herwig
        zh = np.asarray(zh, dtype=np.float32)
        np.save(fd / "label_herwig.npy", np.asarray(yh).astype(np.int16))
        np.save(fd / "logits_herwig.npy", np.stack([np.zeros_like(zh), zh], axis=1))
    (cell / "DONE").touch()
    return cell


def _sep(n, sep, seed):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    return y, rng.normal(0.0, 1.0, n) + sep * y


def test_herwig_reads_the_other_pair_through_the_same_checks_and_writes_its_own_file(tmp_path):
    y, z = _sep(4000, 3.0, 0)
    yh, zh = _sep(2000, 1.0, 1)                       # worse under generator shift
    for s in (1, 2):
        _cell(tmp_path, y, z, seed=f"s{s}", herwig=(yh, zh))
    _cell(tmp_path, y, z, dataset="top")               # top has no Herwig pair
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out"), "--herwig"]) == 0
    py = json.loads((tmp_path / "out/bench_metrics.json").read_text())
    hw = json.loads((tmp_path / "out/bench_metrics_herwig.json").read_text())
    assert py["test_set"] == "pythia" and hw["test_set"] == "herwig"
    assert set(py["cells"]) == {"top", "qg"} and set(hw["cells"]) == {"qg"}, "top is never read for Herwig"
    p, h = py["cells"]["qg"]["l162-s1b"]["N1000"]["s1"], hw["cells"]["qg"]["l162-s1b"]["N1000"]["s1"]
    assert p["n_jets"] == 4000 and h["n_jets"] == 2000
    assert h["auc"] < p["auc"]
    assert hw["row_alignment_sha256"]["qg"] != py["row_alignment_sha256"]["qg"]
    assert hw["summary"]["qg"]["l162-s1b"]["N1000"]["auc"]["n"] == 2


def test_herwig_is_fatal_on_a_done_cell_without_the_pair_and_on_misaligned_cells(tmp_path):
    y, z = _sep(400, 2.0, 0)
    _cell(tmp_path / "a", y, z)                              # no Herwig files
    with pytest.raises(SystemExit, match="logits_herwig.npy"):
        M.main(["--root", str(tmp_path / "a"), "--out", str(tmp_path / "a/out"), "--herwig"])
    yh, zh = _sep(400, 1.0, 1)
    _cell(tmp_path / "b", y, z, init="p", herwig=(yh, zh))
    _cell(tmp_path / "b", y, z, init="q", herwig=(yh[::-1].copy(), zh[::-1].copy()))
    with pytest.raises(SystemExit, match="not paired"):
        M.main(["--root", str(tmp_path / "b"), "--out", str(tmp_path / "b/out"), "--herwig"])
    assert not (tmp_path / "b/out/bench_metrics_herwig.json").exists()


def test_lock_directories_are_not_cells(tmp_path):
    y, z = _sep(400, 2.0, 0)
    _cell(tmp_path, y, z)
    (tmp_path / "leg_qg/l162-s1b/N1000/s2.lock").mkdir()
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out/bench_metrics.json").read_text())
    assert doc["skipped"] == [] and list(doc["cells"]["qg"]["l162-s1b"]["N1000"]) == ["s1"]


# ------------------------------------------------------------- leg_stats CLI

def test_leg_stats_gives_the_same_json_with_and_without_the_new_flag(tmp_path):
    """The generalisation must not move a number on the two committed inputs."""
    d = REPO / "experiments/FIGS/data"
    a = subprocess.run([PY, "experiments/FT/leg_stats.py", "--out", str(tmp_path / "a.json")],
                       cwd=REPO, capture_output=True, text=True)
    b = subprocess.run([PY, "experiments/FT/leg_stats.py", "--out", str(tmp_path / "b.json"),
                        "--leg", f"leg1={d / 'leg1_metrics.json'}",
                        "--leg", f"leg2={d / 'leg2_metrics.json'}"],
                       cwd=REPO, capture_output=True, text=True)
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    doc = json.loads((tmp_path / "a.json").read_text())
    assert set(doc["legs"]) == {"leg1", "leg2"}
    assert sorted({r["size"] for r in doc["legs"]["leg1"]}) == sorted(L.SIZES)
    assert doc["pretraining_seed_inference_blocked_rows"] == 18


def test_sizes_are_discovered_from_the_file_and_never_include_the_reference_row():
    cells = L.read_cells(json.loads((REPO / "experiments/FIGS/data/leg2_metrics.json").read_text()))
    assert L.sizes_in(cells, ["l162-s1b", "r16q1-s2"]) == L.SIZES
    assert "ref" in cells["ref_e1arms-s1"]
    # the union: a size one arm lacks is still asked for, and is then fatal
    cells = {"a": {"N1000": {}, "N10000": {}}, "b": {"N10000": {}}}
    assert L.sizes_in(cells, ["a", "b"]) == ["N1000", "N10000"]
    with pytest.raises(SystemExit, match="has no N1000"):
        L.arm_seed_values(cells, "b", "N1000", "accuracy")


def test_a_bench_metrics_file_is_read_by_dataset_name(tmp_path):
    y, z = _sep(400, 2.0, 0)
    for init in ("l162-s1b", "r16q1-s2", "r16q1-s3", "r16q1-s4"):
        for s in (1, 2, 3):
            _cell(tmp_path, y, _sep(400, 2.0, s)[1], dataset="top", init=init, n="N1000", seed=f"s{s}")
    M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")])
    r = subprocess.run([PY, "experiments/FT/leg_stats.py", "--metric", "auc",
                        "--leg", f"top={tmp_path / 'out/bench_metrics.json'}",
                        "--out", str(tmp_path / "ls.json")], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads((tmp_path / "ls.json").read_text())
    rows = doc["legs"]["top"]
    assert [r["size"] for r in rows] == ["N1000"] and rows[0]["contrast"] == "granularity"
    bad = subprocess.run([PY, "experiments/FT/leg_stats.py", "--leg", "nonsense"],
                         cwd=REPO, capture_output=True, text=True)
    assert bad.returncode != 0 and "NAME=PATH" in bad.stderr
