"""BatchNorm-recomputed twins of a finished v2 run's reported checkpoints
(experiments/MTX/bn_twins_v2.py; PRESPEC A14's BatchNorm rule, which fired 2026-10-03).

A CPU run of pretrain_v2 on the synthetic files of tests/test_pretrain_v2.py, then the
twins: the BatchNorm pass is the weight average's own (applied to the averaged weights
it reproduces the run's net_wavg file), each twin changes nothing but the BatchNorm
statistics, and the sample must be the weight average's.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "MTX"))

torch = pytest.importorskip("torch")
pytest.importorskip("weaver")

from test_pretrain_v2 import _args, data  # noqa: E402,F401  (data is a fixture)
import bn_twins_v2 as bt  # noqa: E402
import pretrain_v2 as pv  # noqa: E402

BN_KEYS = ("running_mean", "running_var", "num_batches_tracked")


@pytest.fixture(scope="module")
def done(data, tmp_path_factory):
    """A complete 3-epoch run with window retention (every epoch is in its window)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(pv, "BN_JETS", 256)
    mp.setattr(pv, "_GRAD_DIAG", False)
    out = tmp_path_factory.mktemp("T") / "t"
    try:
        assert pv.main(_args(data, out, extra=["--keep-checkpoints", "window"])) == 0
    finally:
        mp.undo()
    return out


@pytest.fixture
def run(done, tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "BN_JETS", 256)
    out = tmp_path / done.name
    shutil.copytree(done, out)
    return out


def test_the_pass_is_the_weight_averages_own(run):
    """Applied to the averaged weights, the twin pass reproduces net_wavg0-2: the same
    sample, record and statistics."""
    rec, a = bt.run_args(run)
    dev = torch.device("cpu")
    model, seeds, side, dc = bt.build(run, a, dev)
    model.load_state_dict(pv.average_states([run / f"net_epoch-{e}_state.pt" for e in range(3)]))
    bn = bt.bn_recompute(a, model, pv.train_stream(a, side, seeds), seeds, dev, False,
                         pv.loader_kwargs(a, dev), list(dc.input_names))
    assert bn == json.loads((run / "net_wavg0-2.json").read_text())["bn_recompute"]
    want, got = torch.load(run / "net_wavg0-2_state.pt"), model.state_dict()
    assert want.keys() == got.keys()
    for k in want:
        if "running_" in k:
            torch.testing.assert_close(got[k], want[k], rtol=1e-5, atol=1e-7)
        else:
            assert torch.equal(got[k], want[k]), k


def test_each_reported_epoch_gets_one_twin_that_changes_only_batchnorm(run):
    by_epoch = bt.tag_epochs(run)
    assert sorted(t for ts in by_epoch.values() for t in ts) == ["best70", "bestval"]
    assert bt.main(["--out", str(run)]) == 0
    wavg = json.loads((run / "net_wavg0-2.json").read_text())["bn_recompute"]
    for e, tags in by_epoch.items():
        state, meta = bt.twin_paths(run, e)
        m = json.loads(meta.read_text())
        assert m["epoch"] == e and m["tags"] == tags and m["bn_recompute"] == wavg
        src = run / f"net_epoch-{e}_state.pt"
        assert m["inputs"] == {str(e): pv.sha256_file(src)} and m["sha256"] == pv.sha256_file(state)
        assert m["code"]["driver"] == "experiments/MTX/bn_twins_v2.py"
        old, new = torch.load(src), torch.load(state)
        assert old.keys() == new.keys()
        stats = [k for k in old if k.endswith(BN_KEYS)]
        assert stats and all(torch.equal(old[k], new[k]) for k in old if k not in stats)
        assert any(not torch.equal(old[k], new[k]) for k in stats if "running_" in k)
    assert len(list(run.glob("net_epoch-*_bn_state.pt"))) == len(by_epoch)


def test_a_written_twin_is_left_alone(run):
    assert bt.main(["--out", str(run)]) == 0
    before = {p.name: p.stat().st_mtime_ns for p in run.glob("net_epoch-*_bn*")}
    assert bt.main(["--out", str(run)]) == 0
    assert {p.name: p.stat().st_mtime_ns for p in run.glob("net_epoch-*_bn*")} == before


def test_a_run_that_is_not_done_is_refused(run, capsys):
    (run / "DONE").unlink()
    assert bt.main(["--out", str(run)]) == pv.EXIT_HALT
    assert "is not DONE" in capsys.readouterr().out
    assert not list(run.glob("net_epoch-*_bn*"))


def test_a_sample_that_is_not_the_weight_averages_writes_nothing(run, monkeypatch, capsys):
    monkeypatch.setattr(pv, "BN_JETS", 128)
    assert bt.main(["--out", str(run)]) == pv.EXIT_HALT
    assert "is not the weight average's" in capsys.readouterr().out
    assert not list(run.glob("net_epoch-*_bn*"))
