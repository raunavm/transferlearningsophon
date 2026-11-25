"""experiments/EVAL/head_accuracy.py (PI 2026-11-25, LEDGER §28 1.2): the output layer is ParticleTransformer's fc,
hidden blocks Linear -> ReLU then one Linear, applied to the stored fc input; the majority class is the commonest."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("head_accuracy", ROOT / "experiments/EVAL/head_accuracy.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def test_the_output_layer_is_hidden_relu_blocks_then_a_linear_layer():
    rng = np.random.default_rng(0)
    w0, b0, w1, b1 = rng.normal(size=(6, 4)), rng.normal(size=6), rng.normal(size=(3, 6)), rng.normal(size=3)
    st = {"mod.fc.1.bias": b1, "mod.fc.0.0.weight": w0, "mod.fc.1.weight": w1, "mod.fc.0.0.bias": b0, "mod.x": 1}
    layers = H.fc_layers(st)
    x = rng.normal(size=(10, 4))
    want = np.maximum(x @ w0.T + b0, 0) @ w1.T + b1
    assert np.allclose(H.logits(x, layers), want, atol=1e-5)
    y = want.argmax(1)
    r = H.head_accuracy(x.astype(np.float32), y, layers, chunk=3)
    assert r["n_jets"] == 10 and r["n_out"] == 3 and r["accuracy"] == 1.0
    with pytest.raises(SystemExit):
        H.fc_layers({"mod.x": 1})


def test_the_majority_class():
    m = H.majority(np.array([2, 2, 5, 2, 0]))
    assert m == {"class": 2, "accuracy": 0.6, "n_classes_present": 3}


def test_the_bias_refit_recovers_a_shift_of_the_class_biases_and_moves_nothing_else():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    k, n = 5, 20000
    y = rng.integers(0, k, n)
    z = rng.normal(size=(n, k)).astype(np.float32)
    z[np.arange(n), y] += 2.0                      # logits calibrated for balanced classes
    shift = np.array([1.5, -1.0, 0.0, 0.5, -1.0])  # ... then biased as by another class mix
    d, rec = H.refit_bias(z + shift.astype(np.float32), y)
    assert rec["converged"] and rec["train_cross_entropy_after"] < rec["train_cross_entropy_before"]
    assert abs(d.mean()) < 1e-9 and np.allclose(d, -(shift - shift.mean()), atol=0.08)
    assert ((z + shift + d).argmax(1) == y).mean() > ((z + shift).argmax(1) == y).mean()
