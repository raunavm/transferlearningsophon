"""CI for the class + mass-regression hybrid loop (experiments/MTX/hybrid_mass.py)
and the mass network config (experiments/MTX/ParT_sophon_arch_mass.py).

The failure these guard against is silent: a K+1 head trained by weaver's
stock loop is a K+1-way classifier whose last class is never true, and its
loss curve looks fine. So the tests check that (1) the loop trains the mass
node toward the target and the classifier toward the labels, (2) the invalid
(unmatched) rows cannot reach the regression loss, (3) the evaluate path hands
weaver exactly K scores, and (4) the arch refuses to build without the loop.

Run:  python3 -m pytest tests/test_hybrid_mass.py -v
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hm():
    return _load("hybrid_mass_under_test", ROOT / "experiments" / "MTX" / "hybrid_mass.py")


# ------------------------------------------------------------------ the loss
def test_logcosh_is_log_cosh_and_does_not_overflow(hm):
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64)
    assert torch.allclose(hm.logcosh(x), torch.log(torch.cosh(x)), atol=1e-12)
    big = hm.logcosh(torch.tensor([1e3, -1e3], dtype=torch.float64))
    assert torch.isfinite(big).all()
    # log cosh(x) -> |x| - log 2 for large |x|
    assert torch.allclose(big, torch.tensor([1e3, 1e3], dtype=torch.float64) - math.log(2.0))


def test_masked_logcosh_ignores_invalid_rows_entirely(hm):
    pred = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([0.0, 0.5, 0.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    clean = hm.masked_logcosh(pred, target, valid)
    # garbage in the masked rows: the unmatched-jet value of log(1e-6/m) ~ -20
    poisoned = target.clone()
    poisoned[~valid] = -20.0
    assert torch.allclose(hm.masked_logcosh(pred, poisoned, valid), clean)
    # it is the MEAN over valid rows, not the sum and not the mean over all
    expect = (hm.logcosh(pred[:2] - target[:2])).mean()
    assert torch.allclose(clean, expect)
    # no valid rows: zero, not nan
    assert hm.masked_logcosh(pred, target, torch.zeros(4, dtype=torch.bool)) == 0.0


def test_split_output_refuses_a_plain_k_wide_head(hm):
    with pytest.raises(RuntimeError, match="mass node"):
        hm.split_output(torch.zeros(4, 17), 17)
    logits, pred = hm.split_output(torch.zeros(4, 18), 17)
    assert logits.shape == (4, 17) and pred.shape == (4,)
    # WHICH column is the mass node, not just the shapes: GloParT regresses on
    # the LAST nodes (tools.py: logits = out[:, :-n_reg]; preds = out[:, -n_reg:]).
    # Every downstream reader takes logits[:, :K], so this convention is load-bearing.
    logits, pred = hm.split_output(torch.arange(18.0)[None], 17)
    assert pred.item() == 17.0 and logits[0].tolist() == list(range(17))


def test_lambda_scales_only_the_mass_term(hm):
    torch.manual_seed(0)
    k = 5
    out = torch.randn(8, k + 1)
    y = {"truth_label": torch.randint(0, k, (8,)),
         "mass_target": torch.randn(8),
         "mass_valid": torch.tensor([1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.bool)}
    ce = torch.nn.CrossEntropyLoss()
    dev = torch.device("cpu")
    a_tot, a_cls, a_reg, _, _ = hm.hybrid_loss(ce, out, y, k, 1.0, dev)
    b_tot, b_cls, b_reg, _, _ = hm.hybrid_loss(ce, out, y, k, 5.0, dev)
    assert torch.allclose(a_cls, b_cls) and torch.allclose(a_reg, b_reg)
    assert torch.allclose(a_tot, a_cls + 1.0 * a_reg, atol=1e-6)
    assert torch.allclose(b_tot, b_cls + 5.0 * b_reg, atol=1e-6)
    assert not torch.allclose(a_tot, b_tot)


def test_num_cls_refuses_a_model_without_the_attribute(hm):
    with pytest.raises(RuntimeError, match="num_cls"):
        hm.num_cls(torch.nn.Linear(3, 4))


# --------------------------------------------------------- a tiny weaver-alike
class _Cfg:
    input_names = ("x",)
    label_names = ("truth_label", "mass_target", "mass_valid")


class _DS:
    config = _Cfg()


class _Loader:
    """Iterable of (X, y, Z) batches with the .dataset.config weaver reads."""

    def __init__(self, batches):
        self.batches = batches
        self.dataset = _DS()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Head(torch.nn.Module):
    def __init__(self, k, d=8):
        super().__init__()
        self.lin = torch.nn.Linear(d, k + 1)
        self.num_cls = k

    def forward(self, x):
        return self.lin(x)


def _batches(k=3, d=8, n_batches=40, bs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        y = torch.randint(0, k, (bs,), generator=g)
        x = torch.randn(bs, d, generator=g)
        x[:, 0] = y.float()                      # class is readable from x[:,0]
        x[:, 1] = torch.randn(bs, generator=g)   # mass ratio is x[:,1] * 0.5
        target = 0.5 * x[:, 1]
        valid = torch.rand(bs, generator=g) > 0.3
        target = torch.where(valid, target, torch.full_like(target, -20.0))
        out.append(({"x": x}, {"truth_label": y, "mass_target": target, "mass_valid": valid}, {}))
    return out


def test_train_fn_fits_both_heads_and_ignores_masked_targets(hm):
    torch.manual_seed(0)
    k = 3
    model = _Head(k)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    loader = _Loader(_batches(k))
    train = hm.make_train_fn(5.0)
    assert train.hybrid_mass_lambda == 5.0
    for epoch in range(3):
        train(model, torch.nn.CrossEntropyLoss(), opt, None, loader, torch.device("cpu"), epoch)
    X, y, _ = loader.batches[0]
    with torch.no_grad():
        out = model(X["x"])
    logits, pred = hm.split_output(out, k)
    acc = (logits.argmax(1) == y["truth_label"]).float().mean().item()
    assert acc > 0.9, acc
    valid = y["mass_valid"]
    err = (pred[valid] - y["mass_target"][valid]).abs().mean().item()
    assert err < 0.15, err
    # the masked rows carried -20 and the node must NOT have learned them
    assert (pred[~valid] - 0.5 * X["x"][~valid, 1]).abs().mean().item() < 0.5


def test_train_fn_refuses_a_config_without_the_mass_labels(hm):
    class _Cfg2:
        input_names = ("x",)
        label_names = ("truth_label",)
    loader = _Loader(_batches(3))
    loader.dataset.config = _Cfg2()
    model = _Head(3)
    with pytest.raises(RuntimeError, match="mass_target"):
        hm.make_train_fn(1.0)(model, torch.nn.CrossEntropyLoss(),
                              torch.optim.SGD(model.parameters(), lr=0.1),
                              None, loader, torch.device("cpu"), 0)


def test_install_replaces_weavers_loops_and_evaluate_hands_back_k_scores(hm):
    tools = pytest.importorskip("weaver.utils.nn.tools")
    stock_train, stock_eval = tools.train_classification, tools.evaluate_classification
    try:
        hm.install(2.5)
        assert os.environ.get(hm.ENV_FLAG) == "1"
        assert tools.train_classification is not stock_train
        assert tools.train_classification.hybrid_mass_lambda == 2.5
        assert tools.evaluate_classification is hm.evaluate_hybrid
        # what seed_weaver's --lean-val-metrics patch needs to find
        params = inspect.signature(tools.evaluate_classification).parameters
        assert "eval_metrics" in params and isinstance(params["eval_metrics"].default, list)
        # weaver's stock evaluate, through the K-wide view: accuracy over K nodes
        k = 3
        model = _Head(k)
        with torch.no_grad():
            model.lin.weight.zero_()
            model.lin.bias.zero_()
            model.lin.weight[:k, 0] = torch.tensor([0.0, 0.0, 0.0])
            # make class = round(x0): logits_c = -(x0 - c)^2 is not linear, so
            # instead give class c a bias ramp on a one-hot-ish feature: use
            # the trained route -- simplest is to train briefly.
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        loader = _Loader(_batches(k))
        for epoch in range(3):
            tools.train_classification(model, torch.nn.CrossEntropyLoss(), opt, None,
                                       loader, torch.device("cpu"), epoch)
        acc = tools.evaluate_classification(model, loader, torch.device("cpu"), 0,
                                            for_training=True, eval_metrics=[])
        assert 0.9 < acc <= 1.0, acc
        # test-mode: scores are K wide, the mass node is not in them
        _, scores, labels, _ = tools.evaluate_classification(
            model, loader, torch.device("cpu"), None, for_training=False, eval_metrics=[])
        assert scores.shape[1] == k
        assert set(labels) == {"truth_label", "mass_target", "mass_valid"}
    finally:
        tools.train_classification, tools.evaluate_classification = stock_train, stock_eval
        os.environ.pop(hm.ENV_FLAG, None)
        hm._STOCK_EVAL = None


# ------------------------------------------------------------------ the arch
def test_mass_arch_refuses_without_the_loop_and_builds_k_plus_one_with_it(hm):
    pytest.importorskip("weaver.utils.data.config")
    from weaver.utils.data.config import DataConfig
    cfg_path = ROOT / "configs" / "arms" / "R16_Q1_MASS.yaml"
    if not cfg_path.exists():
        pytest.skip("run scripts/build_arm_configs.py first")
    dc = DataConfig.load(str(cfg_path))
    arch = _load("mass_arch_under_test", ROOT / "experiments" / "MTX" / "ParT_sophon_arch_mass.py")
    os.environ.pop(hm.ENV_FLAG, None)
    with pytest.raises(RuntimeError, match="hybrid"):
        arch.get_model(dc, num_classes=17, fc_params=[(512, 0.1)])
    try:
        os.environ[hm.ENV_FLAG] = "1"
        model, _ = arch.get_model(dc, num_classes=17, fc_params=[(512, 0.1)])
        assert model.num_cls == 17 and model.num_reg == 1
        last = [m for m in model.mod.fc.modules() if isinstance(m, torch.nn.Linear)][-1]
        assert last.out_features == 18
        # the no-mass twin's head is 17 wide from the same arch chain, so the
        # trunk keys are identical and only the head differs (extract_features
        # is_head() is what separates them)
        keys = [k for k in model.state_dict() if not k.startswith("mod.fc.")]
        mtx = _load("mtx_arch_under_test", ROOT / "experiments" / "MTX" / "ParT_sophon_arch_mtx.py")
        twin, _ = mtx.get_model(dc, num_classes=17, fc_params=[(512, 0.1)])
        assert keys == [k for k in twin.state_dict() if not k.startswith("mod.fc.")]
        # -o allow_without_hybrid True is the inspection escape hatch
        os.environ.pop(hm.ENV_FLAG)
        arch.get_model(dc, num_classes=17, fc_params=[(512, 0.1)], allow_without_hybrid=True)
    finally:
        os.environ.pop(hm.ENV_FLAG, None)


def test_seed_weaver_installs_the_loop_before_the_lean_val_patch():
    """Order matters: the lean-val patch wraps whatever evaluate_classification
    is at that moment. If --mass-lambda were handled after it, validation would
    softmax over K+1 nodes."""
    text = (ROOT / "experiments" / "E1" / "seed_weaver.py").read_text()
    i_mass = text.index("if mass_lambda is not None:")
    i_lean = text.index("if lean_val:")
    assert i_mass < i_lean
    assert '_pop(rest, "--mass-lambda")' in text
