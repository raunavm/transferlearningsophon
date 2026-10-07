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


# ---- what the cluster run wrote (aoj_full_v1/fit_v6, analysis_v6) ----
V6 = DATA / "fit_v6" / "results.json"


def test_the_committed_fit_v6_is_what_its_code_and_inputs_make():
    import hashlib
    import subprocess
    res = json.loads(V6.read_text())
    r = res["refit"]
    for key, path in (("bins_sha256", r["bins"]), ("previous_sha256", r["previous"])):
        assert r[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for key, path in (("peak_fit_sha256", "experiments/AOJ/peak_fit.py"), ("script_sha256", "experiments/AOJ/fit_v6.py")):
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"mtx-s1.91:{path}"], capture_output=True).stdout
        assert r[key] == hashlib.sha256(blob).hexdigest(), f"{path} at mtx-s1.91 is not the code that wrote fit_v6"


def test_fit_v6_fits_every_model_at_the_pooled_shape_at_the_order_the_pool_settled_on():
    res = json.loads(V6.read_text())
    ps = res["pooled_shape"]
    last = ps["passes"][-1]
    assert len(res["models"]) == 31 and len(ps["pool"]) == 30 and F6.PUBLISHED not in ps["pool"]
    for n, m in res["models"].items():
        f = m["top"]
        assert (f["mean"], f["width"]) == (ps["mean"], ps["width"]) and f["converged"]
        if n in last["orders"]:
            assert f["tf_order"] == last["orders"][n]
    assert last["shape"] == pytest.approx([ps["mean"], ps["width"]], abs=1e-9)


def test_the_v6_readout_is_the_mean_and_sd_over_seeds_of_the_v6_yields():
    v6 = json.loads(V6.read_text())
    an = json.loads((DATA / "analysis_v6/aoj_top.json").read_text())["per_label_set"]["label_sets"]
    for level, s in an.items():
        ys = [v6["models"][n]["top"]["signal_yield"] for n in s["models"]]
        assert len(ys) == 5 and s["signal_yield"]["mean"] == pytest.approx(np.mean(ys))
        assert s["signal_yield"]["sd"] == pytest.approx(np.std(ys, ddof=1))


# ---- the v2 grid's run (scripts/build_aoj_jobs.py --v2) ----
RB = F6.RB
S = RB._load("seed_level", RB.REPO / "experiments/STATS/seed_level.py")
V1_TO_ARM = {"188": "L188", "162": "L162", "43": "R42_Q1", "17": "R16_Q1", "162+mass": "L162_MASS",
             "17+mass": "R16_Q1_MASS"}


def _v2_name(v1, tag):
    """A first-run model under the v2 grid's name: l162-s1b is the 162-class run 1."""
    return f"{v1.removesuffix('b')}-{tag}" if v1 != F6.PUBLISHED else v1


def test_the_first_runs_readout_is_what_it_was():
    """per_label_set gained a grouping argument for the v2 grid; on the first run's fit it
    must still give the committed analysis_v6, all but the script's own hash."""
    got = json.loads(json.dumps(RB.per_label_set(json.loads(V6.read_text()), S)))
    want = json.loads((DATA / "analysis_v6/aoj_top.json").read_text())["per_label_set"]
    assert {k: v for k, v in got.items() if k != "script_sha256"} == \
           {k: v for k, v in want.items() if k != "script_sha256"}


def test_the_v2_readout_is_the_same_mean_and_sd_per_grid_arm_at_each_checkpoint():
    """The first run's fits under v2 names, at two checkpoints (the second a copy): each
    grid arm's row is the first run's row of its label set, at each checkpoint."""
    v6 = json.loads(V6.read_text())
    models = {}
    for n, m in v6["models"].items():
        models[_v2_name(n, "best70")] = m
        if n != F6.PUBLISHED:
            models[_v2_name(n, "wavg")] = m
    per = RB.per_arm_v2(dict(v6, models=models))
    want = json.loads((DATA / "analysis_v6/aoj_top.json").read_text())["per_label_set"]["label_sets"]
    assert list(per) == ["best70", "wavg"]
    for tag, summary in per.items():
        assert list(summary["label_sets"]) == list(V1_TO_ARM.values()), "grid order"
        for level, arm in V1_TO_ARM.items():
            s, w = summary["label_sets"][arm], want[level]
            assert s["models"] == [_v2_name(n, tag) for n in w["models"]]
            assert s["signal_yield"] == w["signal_yield"] and s["by_shape"] == w["by_shape"]
    with pytest.raises(SystemExit, match="names no arm"):
        RB.per_arm_v2(dict(v6, models={**models, "x188-s1-best70": models["l188-s1-best70"]}))


def _v2_world(d, as_v1):
    """bins.npz and a --previous of a v2 run, from the first run's bins and fit_v5 under v2
    names; the --previous, a v2 run's own peak_fit.py fit, has no fit_v4 to carry."""
    d.mkdir()
    z = np.load(DATA / "fit_v3/bins.npz")
    np.savez(d / "bins.npz", **{k: z[k] for k in z.files if k.startswith("reference|")},
             **{f"{new}|{part}|{k}": z[f"{old}|{part}|{k}"] for new, old in as_v1.items()
                for part in ("main", "validation") for k in F6.KEYS})
    v5 = json.loads((DATA / "fit_v5/results.json").read_text())
    v5["models"] = {new: {"top": {k: v for k, v in v5["models"][old]["top"].items() if k != "fit_v4_signal_yield"}}
                    for new, old in as_v1.items()}
    (d / "previous.json").write_text(json.dumps(v5))
    return ["--v2", "--bins", str(d / "bins.npz"), "--previous", str(d / "previous.json"),
            "--out", str(d / "fit_v6"), "--analysis-out", str(d / "an"), "--workers", "1"]


@pytest.fixture(scope="module")
def freeze(tmp_path_factory):
    """A freeze-like v2 run: two runs of the 162-class vocabulary at three checkpoints."""
    d = tmp_path_factory.mktemp("v2") / "t12"
    as_v1 = {F6.PUBLISHED: F6.PUBLISHED, "l162-s2-best70": "l162-s2", "l162-s4-best70": "l162-s4",
             "l162-s2-wavg": "l162-s3", "l162-s4-wavg": "l162-s5", "l162-s2-best70_bn": "r16q1-s2",
             "l162-s4-best70_bn": "r16q1-s4"}
    assert F6.main(_v2_world(d, as_v1)) == 0
    return d, as_v1


def test_fit_v6_runs_a_v2_run_from_its_own_fit_and_reads_it_out_per_arm(freeze):
    d, as_v1 = freeze
    res = json.loads((d / "fit_v6/results.json").read_text())
    assert set(res["models"]) == set(as_v1)
    assert not any("fit_v4_signal_yield" in m["top"] for m in res["models"].values())
    an = json.loads((d / "an/aoj_top.json").read_text())
    assert an["provenance"]["input_sha256"] == RB._sha(d / "fit_v6/results.json")
    assert list(an["per_checkpoint"]) == ["best70", "wavg", "best70_bn"]
    for tag in an["per_checkpoint"]:
        s = an["per_checkpoint"][tag]["label_sets"]["L162"]
        ys = [res["models"][n]["top"]["signal_yield"] for n in s["models"]]
        assert s["models"] == [f"l162-s2-{tag}", f"l162-s4-{tag}"]
        assert s["signal_yield"]["mean"] == pytest.approx(np.mean(ys)) and s["signal_yield"]["n"] == 2


def test_a_v2_pooled_shape_is_built_from_the_primary_entries_alone_and_every_checkpoint_fitted_at_it(freeze):
    d, _ = freeze
    res = json.loads((d / "fit_v6/results.json").read_text())
    ps = res["pooled_shape"]
    assert ps["pool"] == ["l162-s2-best70", "l162-s4-best70"]
    own = json.loads((d / "previous.json").read_text())["models"]
    assert ps["systematic_step"]["mean"] == pytest.approx(np.std([own[n]["top"]["mean"] for n in ps["pool"]], ddof=1))
    for f in res["models"].values():
        assert (f["top"]["mean"], f["top"]["width"]) == (ps["mean"], ps["width"])


def test_tier_three_holds_the_freeze_shape_and_changes_no_frozen_file(freeze, tmp_path):
    d, _ = freeze
    frozen = {p: p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
    held = d / "fit_v6/results.json"
    args = _v2_world(tmp_path / "t3", {F6.PUBLISHED: F6.PUBLISHED, "r63q1-s1-best70": "r42q1-s1",
                                       "r63q1-s2-best70": "r42q1-s2", "r63q1-s1-wavg": "r42q1-s3",
                                       "r63q1-s2-wavg": "r42q1-s4"})
    assert F6.main([*args, "--shape-from", str(held)]) == 0
    res = json.loads((tmp_path / "t3/fit_v6/results.json").read_text())
    ps, fz = res["pooled_shape"], json.loads(held.read_text())["pooled_shape"]
    assert (ps["mean"], ps["width"], ps["systematic_step"], ps["pool"]) == \
           (fz["mean"], fz["width"], fz["systematic_step"], fz["pool"])
    assert ps["held_from"] == str(held) and ps["held_from_sha256"] == RB._sha(held)
    for n, m in res["models"].items():
        f = m["top"]
        assert (f["mean"], f["width"]) == (fz["mean"], fz["width"])
        assert f["shape_systematic"]["pooled_width_up"] == pytest.approx(
            P.fit_binned(F6._bins(np.load(tmp_path / "t3/bins.npz"), n, "main"), "top", fz["mean"],
                         fz["width"] + fz["systematic_step"]["width"], order=tuple(f["tf_order"]),
                         tops=P.tops_from_reference(F6._bins(np.load(tmp_path / "t3/bins.npz"), "reference", "main"),
                                                    res["reference"]["top"]))[0]["signal_yield"])
    assert {p: p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()} == frozen
    assert "R63_Q1" in json.loads((tmp_path / "t3/an/aoj_top.json").read_text())["per_checkpoint"]["best70"]["label_sets"]


def test_a_shape_from_other_jets_is_refused(freeze, tmp_path):
    d, _ = freeze
    other = json.loads((d / "fit_v6/results.json").read_text())
    other["reference"]["top"]["signal_yield"] += other["reference"]["top"]["signal_yield_err"]
    (tmp_path / "other.json").write_text(json.dumps(other))
    args = _v2_world(tmp_path / "t3", {F6.PUBLISHED: F6.PUBLISHED, "r63q1-s1-best70": "r42q1-s1",
                                       "r63q1-s2-best70": "r42q1-s2"})
    with pytest.raises(SystemExit, match="not the same jets"):
        F6.main([*args, "--shape-from", str(tmp_path / "other.json")])
    assert not (tmp_path / "t3/fit_v6").exists()
