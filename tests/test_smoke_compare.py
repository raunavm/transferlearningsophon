"""experiments/MTX/smoke_compare.py on synthetic smoke-run directories."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("smoke_compare", ROOT / "experiments" / "MTX" / "smoke_compare.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

VAL = {"loss": 2.0, "acc": 0.5, "head_top1_acc": 0.4, "p_qcd_resonant": 0.1, "p_qcd_qcd": 0.8}


def _run(root, name, loss=(1.5, 1.0, 0.9), val_scale=1.0, w_shift=0.0, streams=("s0", "s1", "s2")):
    d = root / name
    (d / "metrics").mkdir(parents=True)
    (d / "stream").mkdir()
    for e, (l, s) in enumerate(zip(loss, streams)):
        val = {k: v * (val_scale if k == "acc" else 1.0) for k, v in VAL.items()}
        (d / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"epoch": e, "train": {"loss": l}, "val": val, "device": "GPU"}))
        (d / "stream" / f"epoch-{e:03d}.json").write_text(json.dumps({"sha256": s}))
    torch.save({"trunk": {"w": torch.ones(3)}}, d / "init_trunk.pt")
    torch.save({"w": torch.ones(3) + w_shift, "b": torch.zeros(2)}, d / f"net_epoch-{len(loss) - 1}_state.pt")


def test_identical_runs_have_zero_differences(tmp_path):
    _run(tmp_path, "a")
    _run(tmp_path, "a2")
    r = sc.compare(tmp_path, "a", "a2")
    assert all(r["stream_equal"].values()) and r["init_trunk_bitwise_equal"]
    assert set(r["train_loss_reldiff"].values()) == {0.0} and set(r["val_reldiff_max"].values()) == {0.0}
    assert r["weights_max_abs_diff"] == 0.0 and r["weights_epoch"] == 2


def test_differences_ratios_and_stream_mismatch(tmp_path):
    _run(tmp_path, "a")
    _run(tmp_path, "b", loss=(1.5, 1.0, 0.9 * (1 + 1e-6)), w_shift=1e-4)        # a resume-like difference
    _run(tmp_path, "g", loss=(1.5 * (1 + 3e-6), 1.0, 0.9), val_scale=1 + 2e-4, w_shift=4e-4,
         streams=("s0", "s1", "X"))
    ref, test = sc.compare(tmp_path, "a", "b"), sc.compare(tmp_path, "a", "g")
    assert ref["train_loss_reldiff"][2] == pytest.approx(1e-6) and ref["train_loss_reldiff"][0] == 0.0
    assert test["stream_equal"] == {0: True, 1: True, 2: False}
    assert test["val_reldiff"][1]["acc"] == pytest.approx(2e-4) and test["val_reldiff"][1]["loss"] == 0.0
    rat = sc.ratios(test, ref)
    assert rat["train_loss_reldiff"] == pytest.approx(3.0, rel=1e-6)
    assert rat["weights_max_abs_diff"] == pytest.approx(4.0, rel=1e-3)
    assert rat["val_reldiff_max"] is None                                       # the reference has none


def test_runs_with_different_state_dicts_are_refused(tmp_path):
    _run(tmp_path, "a")
    _run(tmp_path, "c")
    torch.save({"w": torch.ones(4)}, tmp_path / "c" / "net_epoch-2_state.pt")
    with pytest.raises(ValueError):
        sc.compare(tmp_path, "a", "c")
