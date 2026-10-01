"""experiments/AOJ/fit_v6.py and peak_fit.pooled_shape: one peak shape for the pretrained
models, from their summed likelihood, each at its own F-test order at that shape."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit_v6", ROOT / "experiments/AOJ/fit_v6.py")
F6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F6)
P = F6.P
DATA = ROOT / "experiments/FIGS/data/aoj_full_v1"
SUBSET = ["l162-s2", "l162-s4", "r16q1-s1"]


@pytest.fixture(scope="module")
def sub(tmp_path_factory):
    out = tmp_path_factory.mktemp("v6")
    F6.main(["--bins", str(DATA / "fit_v3/bins.npz"), "--previous", str(DATA / "fit_v5/results.json"),
             "--out", str(out / "fit"), "--analysis-out", str(out / "an"), "--workers", "1", "--models", *SUBSET])
    return json.loads((out / "fit" / "results.json").read_text()), out


def test_the_pooled_shape_minimises_the_summed_loss_at_each_models_own_order_there(sub):
    res, _ = sub
    z = np.load(DATA / "fit_v3/bins.npz")
    ps = res["pooled_shape"]
    shape = (ps["mean"], ps["width"])
    tops = P.tops_from_reference(F6._bins(z, "reference", "main"), res["reference"]["top"])
    bins = {n: F6._bins(z, n, "main") for n in SUBSET}
    win = P.PEAKS["top"]["window"]
    orders = {n: tuple(res["models"][n]["top"]["tf_order"]) for n in SUBSET}
    for n in SUBSET:
        assert orders[n] == tuple(P._choose_order(bins[n], P._tf_norm(bins[n], win), *shape, tops)[0])
    total = lambda m, w: sum(P._Model(bins[n], orders[n], P._tf_norm(bins[n], win), m, w, tops).fit()[1] for n in SUBSET)
    t0 = total(*shape)
    for dm, dw in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)):
        assert total(shape[0] + dm, shape[1] + dw) > t0 - 1e-6


def test_every_model_is_fitted_at_the_pooled_shape_given_the_tops_with_its_systematics(sub):
    res, _ = sub
    ps = res["pooled_shape"]
    for n in SUBSET:
        f = res["models"][n]["top"]
        assert (f["mean"], f["width"]) == (ps["mean"], ps["width"]) and f["shape_source"] == "pooled"
        assert set(f["shape_systematic"]) == set(F6.SHAPE_KEYS)
        assert f["shape_systematic"]["pooled_width_up"] > f["signal_yield"] > f["shape_systematic"]["pooled_width_down"]
        assert 0 < f["leak_systematic"]["shift"]["0.6"] < f["leak_systematic"]["shift"]["0.4"]
        assert f["shape_variations"]["own_shape"]["signal_yield"] == f["own_shape_fit"]["signal_yield"]
    assert res["reference"]["top"]["signal_yield"] == json.loads((DATA / "fit_v5/results.json").read_text())[
        "reference"]["top"]["signal_yield"]


def test_the_pooled_shape_of_samples_sharing_one_peak_is_that_peak():
    """Two synthetic scores, one true peak (173 / 14 GeV): the pooled shape finds it."""
    tp = importlib.util.spec_from_file_location("tpf", ROOT / "tests/test_aoj_peak_fit.py")
    T = importlib.util.module_from_spec(tp)
    tp.loader.exec_module(T)
    bins = {f"s{k}": T._top_bins(40 + k, n_sig=6000)[0] for k in range(2)}
    shape, orders, trail = P.pooled_shape(bins, list(bins), P.PEAKS["top"]["window"], (180.0, 17.3))
    assert abs(shape[0] - 173.0) < 1.5 and abs(shape[1] - 14.0) < 1.5
    assert trail[-1]["orders"] == {n: list(o) for n, o in orders.items()}


def test_a_result_is_never_overwritten(sub):
    _, out = sub
    with pytest.raises(SystemExit, match="exists"):
        F6.main(["--out", str(out / "fit"), "--analysis-out", str(out / "an2")])
