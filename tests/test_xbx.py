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
    assert isinstance(h[2], torch.nn.Dropout) and h[2].p == 0.1 and h[3].weight.shape == (2, 512)
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


def test_the_seed_sets_the_second_layer_and_the_preloaded_first_layer_is_kept_at_lr_zero():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    f, y = rng.normal(0, 1, (3000, 128)), np.tile([0, 1], 1500)
    sd = {"mod.fc.0.0.weight": torch.randn(512, 128), "mod.fc.0.0.bias": torch.randn(512)}
    h1 = X.train_head(f, y, sd, seed=1, n_samples=4096, batch=1024, lr=0.0)
    h2 = X.train_head(f, y, sd, seed=2, n_samples=4096, batch=1024, lr=0.0)
    assert torch.equal(h1[0].weight, sd["mod.fc.0.0.weight"]) and torch.equal(h2[0].weight, h1[0].weight)
    assert not torch.equal(h1[3].weight, h2[3].weight)
    with pytest.raises(ValueError):
        X.train_head(f, np.zeros(3000, int), None, seed=1, n_samples=1024)


def _arm_labels(arm):
    """The output index each native label trains on, from the arm's own data config."""
    import yaml
    path = ROOT / ("configs/arms/v2" if arm.startswith(("RAND2", "FLAV")) else "configs/arms") / f"{arm}.yaml"
    cfg = yaml.safe_load(path.read_text())
    expr = list(cfg["labels"]["value"].values())[0]
    env = {"np": np, "jet_label": np.arange(188)}
    for k, v in (cfg.get("new_variables") or {}).items():
        if "jet_label" in str(v):
            env[k] = eval(str(v), env)
    return np.broadcast_to(np.asarray(eval(str(expr), env)), (188,))


@pytest.mark.parametrize("arm", ["L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS",
                                 *[f"RAND2_p{k}" for k in range(1, 6)], "FLAV_F0", "FLAV_F1", "FLAV_F1R"])
def test_the_zero_shot_nodes_are_the_nodes_each_vocabulary_trained_on(arm):
    import csv
    g = X.node_groups(arm)
    lab = _arm_labels(arm)
    names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(
        (ROOT / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    assert g["xbb"] == lab[0] and g["xbq"] == lab[6] and names[0] == "label_X_bb" and names[6] == "label_X_bq"
    assert g["k"] == lab.max() + 1
    # the QCD nodes are exactly the nodes the 27 QCD classes train on, and hold nothing else
    assert set(g["qcd"]) == set(lab[161:188].tolist()) and not set(g["qcd"]) & set(lab[:161].tolist())


def test_the_zero_shot_nodes_merge_as_the_vocabulary_does():
    full, mid = X.node_groups("L188"), X.node_groups("R42_Q1")
    assert len(full["qcd"]) == 27 and full["xbb"] != full["xbq"]
    assert len(mid["qcd"]) == 1 and mid["xbb"] == mid["xbq"]     # {bb, bc, bq} at 43 classes
    with pytest.raises(ValueError):
        X.node_groups("R63_Q1")


def test_the_stored_columns_give_eq4():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 3, (500, 188)).astype(np.float32)
    g = X.node_groups("L188")
    c = X.xbx_columns(z, g)
    assert X.zero_shot(c["logp_xbb"], c["logp_qcd"]) == pytest.approx(X.xbb_discriminant(z, g["xbb"], g["qcd"]), rel=1e-5)
    assert X.zero_shot(c["logp_xbq"], c["logp_qcd"]) == pytest.approx(X.xbb_discriminant(z, g["xbq"], g["qcd"]), rel=1e-5)
    assert np.array_equal(c["argmax"], z.argmax(1))


def test_the_matched_draw_gives_both_classes_the_same_cells():
    rng = np.random.default_rng(4)
    edges = (np.array([200.0, 400, 800, 2500]), np.array([20.0, 100, 200, 500]))
    ps, ms = rng.uniform(200, 2500, 5000), rng.normal(125, 15, 5000)          # a peaked signal
    pb, mb = 200 + rng.exponential(300, 40000), 20 + rng.exponential(80, 40000)
    a, b, info = X.matched_draw(ps, ms, pb, mb, 1500, np.random.default_rng(0), edges)
    assert a.size == b.size == info["n_per_class"] == 1500
    assert np.unique(a).size == a.size and np.unique(b).size == b.size          # no jet twice
    ca, cb = X._cells(ps[a], ms[a], edges), X._cells(pb[b], mb[b], edges)
    assert np.array_equal(np.bincount(ca, minlength=9), np.bincount(cb, minlength=9))
    a2, b2, _ = X.matched_draw(ps, ms, pb, mb, 1500, np.random.default_rng(0), edges)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    # a pool too small: every jet the classes can share, no more
    a3, b3, info3 = X.matched_draw(ps, ms, pb, mb, 10**9, np.random.default_rng(0), edges)
    full = np.minimum(np.bincount(X._cells(ps, ms, edges)[X._cells(ps, ms, edges) >= 0], minlength=9),
                      np.bincount(X._cells(pb, mb, edges)[X._cells(pb, mb, edges) >= 0], minlength=9))
    assert info3["cap"] is None and a3.size == b3.size == full.sum()


def test_the_grid_is_the_base_configs():
    ep, em = X.grid_edges()
    assert ep[0] == 200 and ep[-1] == 2500 and em[0] == 20 and em[-1] == 500 and em.size == 49


def test_the_endpoints_take_a5s_working_point_from_the_qcd_count():
    rng = np.random.default_rng(5)
    s, b = rng.normal(1.5, 1, 20_000), rng.normal(0, 1, 100_000)
    e = X.endpoints(s, b)
    assert e["eps_b_star"] == 1e-2 and e["eps_b_at_star"] == pytest.approx(1e-2, rel=0.01)
    # eps_S at the cut that keeps 1 % of the background, from the normal tails
    from scipy.stats import norm
    assert e["eps_s_at_star"] == pytest.approx(norm.sf(norm.isf(0.01) - 1.5), abs=0.01)
    assert e["neg_ln_eps_s"] == pytest.approx(-np.log(e["eps_s_at_star"]))
    assert e["auc"] == pytest.approx(norm.cdf(1.5 / np.sqrt(2)), abs=0.005)
    assert e["r50"] == pytest.approx(1 / norm.sf(1.5), rel=0.05) and e["r50_n_bkg_pass"] > 0
    assert X.endpoints(s, np.r_[b, b, b, b])["eps_b_star"] == 1e-3                # 400,000 QCD jets


def _extract_xbx():
    s = importlib.util.spec_from_file_location("extract_xbx_t", ROOT / "experiments/XBS/extract_xbx.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_the_extraction_wrapper_selects_every_jet_with_extract_v2s_own_flags(tmp_path, monkeypatch):
    pytest.importorskip("weaver")
    E = _extract_xbx()
    seen = {}

    class Stop(Exception):
        pass

    def capture(*args, **kw):
        seen["args"] = args
        raise Stop
    monkeypatch.setattr(E.EX, "Selector", capture)
    with pytest.raises(Stop):
        E.main(["--arm", "R16_Q1", "--run-dir", str(tmp_path), "--rung", "R16_Q1", "--num-classes", "17",
                "--checkpoints", "best70", "--data-test", "x.parquet", "--out", str(tmp_path / "o")])
    fcls, prefix, hcls, head_prefix, stride = seen["args"][:5]
    assert fcls == [] and hcls == [] and prefix >= 2_000_000_000 and head_prefix >= 2_000_000_000 and stride == 1
    with pytest.raises(SystemExit, match="17 classes, the model 188"):
        E.main(["--arm", "R16_Q1", "--num-classes", "188", "--out", "o"])
    with pytest.raises(SystemExit, match="set by extract_xbx"):
        E.main(["--arm", "R16_Q1", "--num-classes", "17", "--diag-stride", "40", "--out", "o"])


def test_the_extraction_wrapper_swaps_in_the_xbx_columns_for_every_jet():
    torch = pytest.importorskip("torch")
    E = _extract_xbx()
    g = X.node_groups("R16_Q1")
    E.EX.head_score_columns = lambda logits, rung, signals: X.xbx_columns(logits, g)  # as main() does
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.Linear(128, 17)).eval()

    class Tap:
        def __init__(self, m):
            self.buf = None
            self.h = m[1].register_forward_pre_hook(lambda _m, i: setattr(self, "buf", i[0]))

        def close(self):
            self.h.remove()

    f = np.random.default_rng(2).normal(size=(300, 3)).astype(np.float32)
    y = np.tile([0, 1], 150)                     # xbx_class is the label, not a native class
    batches = [({"x": torch.from_numpy(f[i:i + 64])}, y[i:i + 64]) for i in range(0, 300, 64)]
    to_in = lambda Xb, need: [Xb["x"][torch.from_numpy(np.flatnonzero(need))]]
    sel = E.EX.Selector([], 2_000_000_000, [], 2_000_000_000, 1)
    res = E.EX.run(iter(batches), {"best70": model}, sel, "R16_Q1", 17, {}, Tap, to_in)
    c = res["checkpoints"]["best70"]
    assert np.array_equal(c["rows"], np.arange(300)) and np.array_equal(c["head_rows"], np.arange(300))
    with torch.no_grad():
        want = X.xbx_columns(model(torch.from_numpy(f)).numpy(), g)
    assert set(c["head"]) == set(want) and all(np.allclose(c["head"][k], want[k], atol=1e-6) for k in want)
    assert np.array_equal(c["label188"], y)
