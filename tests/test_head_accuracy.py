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
