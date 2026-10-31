"""The v4 probes (experiments/EVAL/probe_bal.py): grid, stride, ParT's AUC and the hidden layer."""
import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_bal", ROOT / "experiments/EVAL/probe_bal.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def test_v4_grid_extends_two_decades_and_scores_every_jet():
    assert pb.lp.C_GRID == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    assert pb.lp.AUC_STRIDE == 1


def test_ovo_auc_is_sklearns_macro_one_vs_one():
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(0)
    y = r.integers(0, 4, 2000)
    p = r.random((2000, 4)) + np.eye(4)[y]
    assert pb.ovo_auc(p, y) == pytest.approx(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovo"))


def test_hidden_layer_is_relu_of_the_checkpoints_first_head_layer(tmp_path):
    r = np.random.default_rng(1)
    w, b = r.normal(size=(512, 128)), r.normal(size=512)
    sd = {"mod.fc.0.0.weight": torch.tensor(w), "mod.fc.0.0.bias": torch.tensor(b),
          "mod.fc.1.weight": torch.zeros(188, 512)}
    path = tmp_path / "ckpt.pt"
    torch.save(sd, path)
    x = r.normal(size=(10, 128)).astype(np.float32)
    h = pb.hidden(x, {"checkpoint": str(path)})
    assert h.shape == (10, 512) and (h >= 0).all()
    assert np.allclose(h, np.maximum(x.astype(np.float64) @ w.T + b, 0), atol=1e-4)


def test_logit_takes_inputs_wider_than_the_class_token():
    r = np.random.default_rng(2)
    x = r.normal(size=(400, 512))
    y = (x[:, 0] > 0).astype(np.int64)
    clf = pb.lp.Logit(np.unique(y), "cpu")
    clf.fit(x, y, 1.0)
    assert clf.coef_.shape == (1, 512) and (clf.predict_proba(x).argmax(1) == y).mean() > 0.9
