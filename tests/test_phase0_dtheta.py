"""Phase 0 part B: the trunk-displacement measure must mean what it claims.

The measurement decides whether the LR plateau is the overwrite regime, so the
ways it could quietly be wrong are the ways that conclusion could be wrong:
counting the re-initialised head as "movement", letting an integer BatchNorm
counter into a float norm, or silently comparing tensors that do not correspond.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location(
        "phase0_dtheta", ROOT / "experiments" / "FT" / "phase0_dtheta.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sd(scale=1.0, head_scale=1.0):
    return {
        "mod.embed.weight": torch.ones(4, 4) * scale,
        "mod.blocks.0.attn.weight": torch.ones(2, 2) * scale,
        "mod.norm.running_mean": torch.zeros(4),
        "mod.norm.num_batches_tracked": torch.tensor(7),      # integer counter
        "mod.fc.0.weight": torch.ones(3, 4) * head_scale,     # re-initialised head
    }


def test_the_head_is_excluded(m):
    """weaver re-initialises mod.fc.*, and its shape differs between the
    pretraining vocabulary and the 2-class task. Counting it would report
    movement that is not movement."""
    keys = m.trunk_tensors(_sd())
    assert not any(k.startswith("mod.fc.") for k in keys)
    assert "mod.embed.weight" in keys


def test_integer_buffers_are_excluded(m):
    """num_batches_tracked is a counter. It is often in the thousands, so
    admitting it would dominate a norm over weights of order 1."""
    assert "mod.norm.num_batches_tracked" not in m.trunk_tensors(_sd())


def test_zero_displacement_for_an_identical_checkpoint(m):
    r = m.displacement(_sd(), _sd())
    assert r["d_theta"] == pytest.approx(0.0)
    assert r["relative"] == pytest.approx(0.0)


def test_the_head_may_change_without_moving_the_trunk(m):
    """THE LOAD-BEARING TEST. Every cell re-initialises the head, so if head
    change leaked in, every cell would report large displacement and the
    overwrite story would be confirmed by construction."""
    r = m.displacement(_sd(head_scale=1.0), _sd(head_scale=99.0))
    assert r["d_theta"] == pytest.approx(0.0)


def test_relative_displacement_is_scale_free(m):
    """Doubling every trunk weight is a relative displacement of exactly 1."""
    r = m.displacement(_sd(scale=1.0), _sd(scale=2.0))
    assert r["relative"] == pytest.approx(1.0)
    r2 = m.displacement(_sd(scale=10.0), _sd(scale=20.0))
    assert r2["relative"] == pytest.approx(1.0)
    assert r2["d_theta"] > r["d_theta"], "absolute norm must still scale"


def test_mismatched_shapes_are_skipped_not_subtracted(m):
    a = _sd()
    b = _sd()
    b["mod.blocks.0.attn.weight"] = torch.ones(3, 3)
    r = m.displacement(a, b)
    assert r["n_tensors"] == 2, "the mismatched tensor must be dropped, not broadcast"


def test_a_totally_different_architecture_fails_loudly(m):
    with pytest.raises(SystemExit, match="no shared trunk tensors"):
        m.displacement({"a.weight": torch.ones(2)}, {"b.weight": torch.ones(2)})


def test_per_tensor_breakdown_is_reported(m):
    r = m.displacement(_sd(scale=1.0), _sd(scale=2.0))
    assert r["per_tensor"]["mod.embed.weight"]["rel"] == pytest.approx(1.0)


def test_the_cell_name_pattern_parses_the_probe_layout(m):
    got = m.CELL.match("r16q1-s2_lr3e-3")
    assert got and got["arm"] == "r16q1-s2" and float(got["lr"]) == 3e-3
    assert m.CELL.match("l162-s1b_lr1e-2")["arm"] == "l162-s1b"


# ---------------------------------------------- weights vs BatchNorm buffers

def _bn_sd(w=1.0, mean=0.0, var=1.0):
    return {
        "mod.embed.weight": torch.ones(4, 4) * w,
        "mod.norm.running_mean": torch.ones(4) * mean,
        "mod.norm.running_var": torch.ones(4) * var,
        "mod.fc.0.weight": torch.ones(3, 4),
    }


def test_buffers_are_separated_from_weights(m):
    """THE FINDING THIS SCRIPT EXISTS TO REPORT. BN running stats update by
    momentum on every forward pass, independently of the learning rate, so a
    total-norm view shows a flat ~0.022 floor across a 33x LR range and reads as
    'the trunk is pinned'. Measured separately, the weights move monotonically
    and the buffers are what is flat."""
    r = m.displacement(_bn_sd(w=1.0, var=1.0), _bn_sd(w=1.0, var=3.0))
    assert r["relative_weight"] == pytest.approx(0.0), "no weight moved"
    assert r["relative_buffer"] > 0.0
    assert r["buffer_share_of_d2"] == pytest.approx(1.0)


def test_a_pure_weight_change_has_no_buffer_component(m):
    r = m.displacement(_bn_sd(w=1.0), _bn_sd(w=2.0))
    assert r["relative_weight"] == pytest.approx(1.0)
    assert r["relative_buffer"] == pytest.approx(0.0)
    assert r["buffer_share_of_d2"] == pytest.approx(0.0)


def test_the_total_still_combines_both(m):
    r = m.displacement(_bn_sd(w=1.0, var=1.0), _bn_sd(w=2.0, var=3.0))
    assert r["d_theta"] > 0
    assert 0.0 < r["buffer_share_of_d2"] < 1.0


def test_only_running_stats_count_as_buffers(m):
    """num_batches_tracked is already excluded as an integer; weights whose name
    merely contains 'var' or 'mean' must not be swept in."""
    assert m.BUFFER.search("mod.norm.running_mean")
    assert m.BUFFER.search("mod.norm.running_var")
    assert not m.BUFFER.search("mod.blocks.0.attn.weight")
    assert not m.BUFFER.search("mod.running_mean_projection.weight")
