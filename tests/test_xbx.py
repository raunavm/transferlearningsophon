"""X->bb / X->bs helpers (experiments/XBS/xbx.py): Sophon's Eq. 4 discriminant, the Cowan
significance against background efficiency, and the transfer head, on synthetic data."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xbx", ROOT / "experiments/XBS/xbx.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)


def test_eq4_is_the_signal_share_of_signal_plus_qcd():
    logits = np.log(np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.1, 0.4, 0.4]]))
    d = X.xbb_discriminant(logits, 0, [2, 3])
    assert d == pytest.approx([0.5 / 0.8, 0.1 / 0.9])
    # a merged vocabulary: the signal is the group of nodes
    assert X.xbb_discriminant(logits, [0, 1], [2, 3]) == pytest.approx([0.7 / 1.0, 0.2 / 1.0])
    with pytest.raises(ValueError):
        X.xbb_discriminant(logits, [0, 2], [2, 3])
    # invariant to a shift of the logits
    assert X.xbb_discriminant(logits + 7.0, 0, [2, 3]) == pytest.approx(d)


def test_cowan_z():
    assert X.cowan_z(10, 100) == pytest.approx(np.sqrt(2 * (110 * np.log(1.1) - 10)))
    assert X.cowan_z(1e-3, 1e4) == pytest.approx(1e-3 / 100, rel=1e-3)     # s << b: s / sqrt(b)
    with pytest.raises(ValueError):
        X.cowan_z(1, 0)


def test_z_against_background_efficiency():
    rng = np.random.default_rng(0)
    sb, ss = rng.normal(0, 1, 100_000), rng.normal(2, 1, 10_000)
    wb, ws = np.full(sb.size, 0.5), np.full(ss.size, 0.01)
    r = X.z_vs_eps_b(ss, ws, sb, wb, [1.0, 0.1, 0.01])
    assert r["eps_b"] == pytest.approx([1.0, 0.1, 0.01], rel=1e-3)
    assert r["B"][0] == pytest.approx(wb.sum()) and r["S"][0] == pytest.approx(ws.sum())
    assert r["S"][1] == pytest.approx(ws[ss >= r["cut"][1]].sum())
    assert np.all(np.diff(r["S"]) <= 0) and r["Z"] == pytest.approx(X.cowan_z(r["S"], r["B"]))
    # tighter cuts help here: a well separated signal gains significance
    assert r["Z"][2] > r["Z"][0]


def test_the_head_preloads_the_first_output_layer_and_learns_a_separable_task():
    torch = pytest.importorskip("torch")
    sd = {"mod.fc.0.0.weight": torch.randn(512, 128), "mod.fc.0.0.bias": torch.randn(512)}
    h = X.sophon_head(sd)
    assert torch.equal(h[0].weight, sd["mod.fc.0.0.weight"]) and torch.equal(h[0].bias, sd["mod.fc.0.0.bias"])
    assert h[2].weight.shape == (2, 512)
    with pytest.raises(ValueError):
        X.sophon_head({"mod.fc.0.0.weight": torch.randn(512, 64), "mod.fc.0.0.bias": torch.randn(512)})
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    f = rng.normal(0, 1, (4000, 128)) + y[:, None] * 0.5
    head = X.train_head(f, y, None, seed=3, n_samples=409_600)
    s = X.head_scores(head, f)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, s) > 0.95
    # the same seed gives the same head
    s2 = X.head_scores(X.train_head(f, y, None, seed=3, n_samples=409_600), f)
    assert np.allclose(s, s2)
