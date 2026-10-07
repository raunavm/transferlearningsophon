"""experiments/AOJ/v2_checkpoints.py: a v2 real-data shard reads the files extract_v2's
own rule resolves, from finished runs only, and never two files for one model."""
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


V = _load("v2_checkpoints", "experiments/AOJ/v2_checkpoints.py")
XV = _load("extract_v2", "experiments/EVAL/extract_v2.py")


def _finished_run(d: pathlib.Path, vals: dict) -> pathlib.Path:
    """A finished v2 run as pretrain_v2.py and bn_twins_v2.py leave it: every epoch's record,
    the global and window best, epochs 70-79, the weight average, best70's twin, DONE."""
    pv = _load("pretrain_v2_for_aoj", "experiments/MTX/pretrain_v2.py")
    full = {e: vals.get(e, 0.1) for e in range(80)}
    (d / "metrics").mkdir(parents=True)
    for e, v in full.items():
        (d / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"epoch": e, "selection": {"metric": "val.acc", "value": v}}))
        (d / f"net_epoch-{e}_state.pt").write_text(f"state{e}")
    top = max(full.values())
    (d / "best_epoch.json").write_text(json.dumps({"epoch": min(e for e, v in full.items() if v == top)}))
    pv.write_window_best(d, 80, {"driver": "test"})
    (d / XV.WAVG_FILE).write_text("wavg")
    e = XV.best_window_epoch(d)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (d / f"net_epoch-{e}_bn_state.pt").write_text(f"bn{e}")
    (d / f"net_epoch-{e}_bn.json").write_text(json.dumps(
        {"epoch": e, "tags": ["best70"], "inputs": {str(e): sha(d / f"net_epoch-{e}_state.pt")},
         "sha256": sha(d / f"net_epoch-{e}_bn_state.pt")}))
    (d / "DONE").touch()
    return d


def _args(run, tmp_path, tags=("best70", "wavg", "best70_bn")):
    return ["--links", str(tmp_path / "ckpt"), "--record", str(tmp_path / "out" / "checkpoints.json"),
            *[f"r16q1-s1-{t}={run}:{t}" for t in tags]]


def test_each_model_links_the_file_extract_v2_resolves_and_the_record_says_which(tmp_path):
    run = _finished_run(tmp_path / "mtx-r16q1-s1", {40: 0.9, 73: 0.8, 76: 0.8})
    assert V.main(_args(run, tmp_path)) == 0
    want = dict(XV.resolve_checkpoints(run, ["best70", "wavg", "best70_bn"]))
    assert want["best70"].name == "net_epoch-73_state.pt" and want["best70_bn"].name == "net_epoch-73_bn_state.pt"
    for tag, path in want.items():
        link = tmp_path / "ckpt" / f"r16q1-s1-{tag}.pt"
        assert link.is_symlink() and link.resolve() == path.resolve()
    rec = json.loads((tmp_path / "out" / "checkpoints.json").read_text())
    assert rec["r16q1-s1-wavg"] == {"run_dir": str(run), "tag": "wavg", "checkpoint": str(run / XV.WAVG_FILE)}
    assert V.main(_args(run, tmp_path)) == 0, "a retried shard resolves the same files again"


def test_an_unfinished_run_and_a_missing_twin_are_refused(tmp_path):
    run = _finished_run(tmp_path / "mtx-r16q1-s1", {74: 0.9})
    (run / "DONE").unlink()
    with pytest.raises(SystemExit, match="has no DONE"):
        V.main(_args(run, tmp_path))
    (run / "DONE").touch()
    (run / "net_epoch-74_bn.json").unlink()
    with pytest.raises(SystemExit, match="no BatchNorm twin of epoch 74"):
        V.main(_args(run, tmp_path))
    assert not (tmp_path / "ckpt").exists() and not (tmp_path / "out").exists()


def test_a_retry_that_resolves_another_file_links_nothing(tmp_path):
    run = _finished_run(tmp_path / "mtx-r16q1-s1", {73: 0.8})
    V.main(_args(run, tmp_path, ["best70"]))
    rec = (tmp_path / "out" / "checkpoints.json").read_text()
    (tmp_path / "ckpt" / "r16q1-s1-best70.pt").unlink()
    # the run's window record now names another epoch: a shard half scored from epoch 73
    # must not finish from epoch 75
    (run / "metrics" / "epoch-075.json").write_text(json.dumps(
        {"epoch": 75, "selection": {"metric": "val.acc", "value": 0.95}}))
    (run / "best_window_epoch.json").write_text(json.dumps(
        {**json.loads((run / "best_window_epoch.json").read_text()), "epoch": 75}))
    with pytest.raises(SystemExit, match="records other checkpoints"):
        V.main(_args(run, tmp_path, ["best70"]))
    assert not (tmp_path / "ckpt" / "r16q1-s1-best70.pt").exists()
    assert (tmp_path / "out" / "checkpoints.json").read_text() == rec


def test_a_global_best_at_best70s_epoch_is_marked_as_that_file_and_never_scored_twice(tmp_path):
    """The extraction's own rule (extract_v2.aliases): bestval at best70's epoch is best70's
    file, and then bestval_bn is best70_bn's twin. A run whose global best lies outside the
    window keeps five files."""
    tags = ["best70", "wavg", "bestval", "best70_bn", "bestval_bn"]
    run = _finished_run(tmp_path / "same" / "mtx-r16q1-s1", {74: 0.9})
    V.main(_args(run, tmp_path / "same", tags))
    rec = json.loads((tmp_path / "same" / "out" / "checkpoints.json").read_text())
    assert {n: r.get("same_file_as") for n, r in rec.items()} == {
        "r16q1-s1-best70": None, "r16q1-s1-wavg": None, "r16q1-s1-bestval": "r16q1-s1-best70",
        "r16q1-s1-best70_bn": None, "r16q1-s1-bestval_bn": "r16q1-s1-best70_bn"}
    links = tmp_path / "same" / "ckpt"
    assert sorted(p.name for p in links.glob("*.same")) == ["r16q1-s1-bestval.same", "r16q1-s1-bestval_bn.same"]
    assert (links / "r16q1-s1-bestval.same").read_text() == "r16q1-s1-best70"
    assert (links / "r16q1-s1-bestval.pt").resolve() == (links / "r16q1-s1-best70.pt").resolve()

    run = _finished_run(tmp_path / "apart" / "mtx-r16q1-s1", {40: 0.9, 74: 0.8})
    e = XV.best_epoch(run)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (run / f"net_epoch-{e}_bn_state.pt").write_text(f"bn{e}")
    (run / f"net_epoch-{e}_bn.json").write_text(json.dumps(
        {"epoch": e, "tags": ["bestval"], "inputs": {str(e): sha(run / f"net_epoch-{e}_state.pt")},
         "sha256": sha(run / f"net_epoch-{e}_bn_state.pt")}))
    V.main(_args(run, tmp_path / "apart", tags))
    rec = json.loads((tmp_path / "apart" / "out" / "checkpoints.json").read_text())
    assert not any("same_file_as" in r for r in rec.values()) and len({r["checkpoint"] for r in rec.values()}) == 5
    assert not list((tmp_path / "apart" / "ckpt").glob("*.same"))
