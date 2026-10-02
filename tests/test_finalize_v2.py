"""The recovery entry point for a v2 run whose end step failed (experiments/MTX/finalize_v2.py).

A CPU run of pretrain_v2 on the synthetic files of tests/test_pretrain_v2.py, its end
step (window checkpoint, weight average, DONE) removed, then redone by finalize_v2
under another code version: the same files as pretrain_v2 wrote, with finalize_v2's
own code version recorded.
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
import finalize_v2 as fv  # noqa: E402
import pretrain_v2 as pv  # noqa: E402

END = ("best_window_epoch.json", "net_wavg0-2.json", "net_wavg0-2_state.pt", "DONE")


@pytest.fixture(scope="module")
def done(data, tmp_path_factory):
    """A complete 3-epoch run with window retention (every epoch is in its window)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(pv, "BN_JETS", 256)
    mp.setattr(pv, "_GRAD_DIAG", False)
    out = tmp_path_factory.mktemp("F") / "f"
    try:
        assert pv.main(_args(data, out, extra=["--keep-checkpoints", "window"])) == 0
    finally:
        mp.undo()
    return out


def _broken(done, tmp_path, keep=()):
    """The run as an end-step failure leaves it: epochs complete, end files absent."""
    out = tmp_path / done.name
    shutil.copytree(done, out)
    for name in END:
        if name not in keep:
            (out / name).unlink()
    return out


def test_finalize_redoes_the_end_step_as_the_run_would_have(done, tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "BN_JETS", 256)
    monkeypatch.setenv("REPO_REF", "mtx-s9.99")              # other code: no code-version check
    out = _broken(done, tmp_path)
    assert fv.main(["--out", str(out)]) == 0
    assert all((out / n).exists() for n in END)
    a, b = torch.load(done / "net_wavg0-2_state.pt"), torch.load(out / "net_wavg0-2_state.pt")
    assert a.keys() == b.keys()
    for k in a:
        if "running_" in k:                                   # recomputed BatchNorm statistics
            torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-7)
        else:                                                 # averaged weights: exact
            assert torch.equal(a[k], b[k]), k
    ma, mb = (json.loads((r / "net_wavg0-2.json").read_text()) for r in (done, out))
    assert ma["inputs"] == mb["inputs"]
    bn_a, bn_b = (dict(m["bn_recompute"]) for m in (ma, mb))
    assert bn_a == bn_b and bn_a["n_jets"] == 256              # the same jets, trimmed the same way
    code = {"repo_ref": "mtx-s9.99", "driver": "experiments/MTX/finalize_v2.py"}
    assert {k: mb["code"][k] for k in code} == code and mb["code"]["commit"]
    wa, wb = (json.loads((r / "best_window_epoch.json").read_text()) for r in (done, out))
    assert {k: v for k, v in wa.items() if k != "code"} == {k: v for k, v in wb.items() if k != "code"}
    assert {k: wb["code"][k] for k in code} == code
    d = json.loads((out / "DONE").read_text())
    assert d["epoch"] == json.loads((done / "best_epoch.json").read_text())["epoch"]
    assert {k: d["finalized_by"][k] for k in code} == code


def test_finalize_keeps_a_weight_average_the_run_already_wrote(done, tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "BN_JETS", 256)
    out = _broken(done, tmp_path, keep=("net_wavg0-2.json", "net_wavg0-2_state.pt"))
    assert fv.main(["--out", str(out)]) == 0
    assert (out / "net_wavg0-2.json").read_text() == (done / "net_wavg0-2.json").read_text()
    assert (out / "DONE").exists() and (out / "best_window_epoch.json").exists()


def test_finalize_refuses_a_done_run_or_a_missing_window_state(done, tmp_path, capsys):
    assert fv.main(["--out", str(done)]) == pv.EXIT_HALT
    assert "is DONE" in capsys.readouterr().out
    out = _broken(done, tmp_path)
    (out / "net_epoch-1_state.pt").unlink()
    assert fv.main(["--out", str(out)]) == pv.EXIT_HALT
    assert "net_epoch-1_state.pt" in capsys.readouterr().out
    assert not any((out / n).exists() for n in END)
