"""experiments/AOJ/refit_from_bins.py and what it wrote (aoj_full_v1/fit_v4, analysis_v4):
fit_v3's procedure replayed from the committed bins must be fit_v3, a replay that is
not is refused, and the new fits and the per-label-set numbers re-derive from their
inputs."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import statistics

import numpy as np
import pytest
from scipy import optimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refit_from_bins", ROOT / "experiments/AOJ/refit_from_bins.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
P = R.P

DATA = ROOT / "experiments/FIGS/data/aoj_full_v1"
BINS, V3, V4, AN4, Q4 = (DATA / "fit_v3/bins.npz", DATA / "fit_v3/results.json", DATA / "fit_v4/results.json",
                         DATA / "analysis_v4/aoj_top.json", DATA / "fit_v4/fit_quality.json")
STEMS = {"l188": "188", "l162": "162", "r42q1": "43", "r16q1": "17", "l162mass": "162+mass",
         "r16q1mass": "17+mass"}


def _job(z, name, start, n_toys=0):
    return (name, *({k: z[f"{name}|{p}|{k}"] for k in R.KEYS} for p in ("main", "validation")), start, n_toys)


def _fits(res):
    return {"reference": res["reference"]["top"], **{n: m["top"] for n, m in res["models"].items()}}


@pytest.fixture(scope="module")
def v4():
    return json.loads(V4.read_text())


def test_fit_v3s_procedure_replayed_from_its_bins_is_fit_v3_spurious_floated_peak_included():
    z, v3 = np.load(BINS), json.loads(V3.read_text())
    stored = {"reference": v3["reference"]["top"], "l162-s2": v3["models"]["l162-s2"]["top"]}
    ref = R.old_fit(_job(z, "reference", None))[1]
    old = {"reference": ref, "l162-s2": R.old_fit(_job(z, "l162-s2", ref["floated"]))[1]}
    rows = R.reproduction(old, stored)
    assert all(r["ok"] for r in rows.values()), rows
    # the diagnostic float at START_ORDER: l162-s2's "peak" at 145 GeV, the width on its bound
    assert old["l162-s2"]["floated"][0] < 150 and old["l162-s2"]["floated"][1] > 29.9
    moved = dict(stored["l162-s2"], signal_yield=stored["l162-s2"]["signal_yield"] + 0.01 * stored["l162-s2"]["signal_yield_err"])
    assert not R.reproduction(old, dict(stored, **{"l162-s2": moved}))["l162-s2"]["ok"]


def test_the_refit_writes_nothing_when_fit_v3_is_not_reproduced(tmp_path):
    z, v3 = np.load(BINS), json.loads(V3.read_text())
    np.savez(tmp_path / "bins.npz", **{k: z[k] for k in z.files if k.split("|")[0] in ("reference", "l162-s2")})
    mini = copy.deepcopy(dict(v3, models={"l162-s2": v3["models"]["l162-s2"]}, n_toys=0))
    top = mini["models"]["l162-s2"]["top"]
    top["signal_yield"] += 0.01 * top["signal_yield_err"]
    (tmp_path / "results.json").write_text(json.dumps(mini))
    with pytest.raises(SystemExit, match="does not reproduce"):
        R.main(["--bins", str(tmp_path / "bins.npz"), "--results", str(tmp_path / "results.json"),
                "--out", str(tmp_path / "out"), "--analysis-out", str(tmp_path / "an"), "--workers", "1"])
    assert not (tmp_path / "out").exists() and not (tmp_path / "an").exists()


@pytest.mark.parametrize("record", ["results.json", "histograms.npz", "fit_quality.json"])
def test_the_refit_never_overwrites_a_result(tmp_path, record):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / record).write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        R.main(["--out", str(tmp_path / "out"), "--analysis-out", str(tmp_path / "an")])
    assert [p.name for p in (tmp_path / "out").iterdir()] == [record]


def test_fit_v4_keeps_fit_v3s_schema_and_records_that_it_reproduced_fit_v3(v4):
    v3 = json.loads(V3.read_text())
    assert set(v3) <= set(v4) and list(v3["models"]) == list(v4["models"])
    for name, old in _fits(v3).items():
        new = _fits(v4)[name]
        assert set(old) <= set(new) and set(old["validation"]) <= set(new["validation"])
        # the band's background-only fit and its toys do not involve the signal shape
        assert new["validation"]["toy_p"] == old["validation"]["toy_p"]
    r = v4["refit"]
    assert r["reproduces_fit_v3"] and all(x["ok"] for x in r["reproduction"].values())
    assert r["bins_sha256"] == hashlib.sha256(BINS.read_bytes()).hexdigest()
    assert r["fit_v3_results_sha256"] == hashlib.sha256(V3.read_bytes()).hexdigest()


def test_every_fit_floats_its_shape_to_one_answer_from_either_start_inside_the_bounds(v4):
    for name, f in _fits(v4).items():
        assert f["shape_floated"] and f["start_check"]["one_answer"] and f["profile_error_ok"], name
        assert f["start_check"]["start"] == v4["shape_variations"]["top"]["pooled"], name
        assert not f["width_at_bound"] and not f["mean_at_bound"], name
        assert f["floated_mean"] == f["mean"], "the peak position is read from the fit the yield comes from"
        assert f["signal_yield_err"] >= f["signal_yield_err_hessian"], name
        assert f["converged"] and f["n_tf_at_floor"] == 0, name
        # and the old reference shape describes every peak worse than its own
        assert f["shape_variations"]["old_reference"]["delta_deviance_vs_fitted_shape"] > 0, name


def test_a_quoted_error_is_where_the_deviance_profiled_over_tf_and_shape_rises_by_one(v4):
    """Re-derived for one real fit with another minimiser over the shape."""
    name, z = "l162-s2", np.load(BINS)
    f = v4["models"][name]["top"]
    b = {k: z[f"{name}|main|{k}"] for k in R.KEYS}
    tf_norm = P._tf_norm(b, P.PEAKS["top"]["window"])
    model = P._Model(b, tuple(f["tf_order"]), tf_norm, f["mean"], f["width"])
    x, half = model.fit()

    def prof(y):
        at = lambda v: P._Model(b, tuple(f["tf_order"]), tf_norm, v[0], v[1]).fit_at_yield(y, x)[1]
        return optimize.minimize(at, (f["mean"] + 0.7, f["width"] - 0.7), method="Nelder-Mead",
                                 options=dict(xatol=1e-4, fatol=1e-10)).fun
    f0 = prof(f["signal_yield"])
    assert abs(half - f0) < 1e-5 and abs(half - f["deviance"] / 2) < 1e-6
    for sign, e in ((-1, f["signal_yield_err_lo"]), (1, f["signal_yield_err_hi"])):
        assert 2 * (prof(f["signal_yield"] + sign * e) - f0) == pytest.approx(1.0, abs=0.01)


def test_the_fit_with_two_answers_under_the_iteration_has_one_now(v4):
    """r16q1mass-s3 ended at order (2, 3) from the reference's shape and at (2, 2) from
    the pooled one when the order and shape were iterated (the review of 2026-09-28).
    With the shape profiled at every order, a search started at either answer's shape
    gives the stored answer."""
    name, z = "r16q1mass-s3", np.load(BINS)
    f = v4["models"][name]["top"]
    b = {k: z[f"{name}|main|{k}"] for k in R.KEYS}
    window = P.PEAKS["top"]["window"]
    for start in ((182.42, 12.52), (182.20, 11.83)):
        order, shape, _ = P._choose_shape_and_order(b, P._tf_norm(b, window), window, [start])
        assert list(order) == f["tf_order"], start
        assert max(abs(shape[0] - f["mean"]), abs(shape[1] - f["width"])) < R.START_TOL["shape_gev"], start


def test_fit_quality_is_what_the_fits_say(v4):
    q, fits = json.loads(Q4.read_text()), _fits(v4)
    assert q["n_fits"] == len(fits) == 32
    assert q["all_converged"] == all(f["converged"] for f in fits.values()) is True
    assert q["max_edm"] == max(f["edm"] for f in fits.values())
    assert q["tf_order"] == {n: f["tf_order"] for n, f in fits.items()}
    assert q["n_width_or_mean_at_bound"] == sum(f["width_at_bound"] or f["mean_at_bound"] for f in fits.values())
    assert q["n_start_dependent"] == sum(not f["start_check"]["one_answer"] for f in fits.values()) == 0
    rises = [r for c in q["profile_error_check"]["per_fit"].values() for r in c["twice_rise"]]
    assert q["profile_error_check"]["twice_rise_range"] == [min(rises), max(rises)]
    assert all(abs(r - 1) < 0.05 for r in rises), "a quoted error is where the profiled deviance rises by 1"


def test_the_pooled_shape_minimises_the_summed_loss_of_the_thirty_models(v4):
    z, sv = np.load(BINS), v4["shape_variations"]["top"]
    pool, (m0, w0) = sv["pool"], sv["pooled"]
    assert len(pool) == 30 and "sophon-public" not in pool
    assert sv["old_reference"] == v4["reference"]["top"]["shape_start"]
    bins = {n: {k: z[f"{n}|main|{k}"] for k in R.KEYS} for n in pool}
    norm = {n: P._tf_norm(b, P.PEAKS["top"]["window"]) for n, b in bins.items()}
    total = lambda m, w: sum(P._Model(bins[n], tuple(v4["models"][n]["top"]["tf_order"]), norm[n], m, w).fit()[1]
                             for n in pool)
    f0 = total(m0, w0)
    for dm, dw in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)):
        assert total(m0 + dm, w0 + dw) > f0
    for n in pool:
        assert v4["models"][n]["top"]["shape_variations"]["pooled"]["signal_yield"] == pytest.approx(
            P.fit_binned(bins[n], "top", m0, w0, order=tuple(v4["models"][n]["top"]["tf_order"]))[0]["signal_yield"],
            rel=1e-6)


def test_the_per_label_set_numbers_are_the_mean_and_sd_over_seeds_of_the_fits(v4):
    doc = json.loads(AN4.read_text())
    pls = doc["per_label_set"]
    for stem, level in STEMS.items():
        fits = [v4["models"][f"{stem}-s{s}b" if (stem, s) == ("l162", 1) else f"{stem}-s{s}"]["top"]
                for s in range(1, 6)]
        got = pls["label_sets"][level]
        ys = [f["signal_yield"] for f in fits]
        assert got["signal_yield"]["mean"] == pytest.approx(statistics.mean(ys), rel=1e-12)
        assert got["signal_yield"]["sd"] == pytest.approx(statistics.stdev(ys), rel=1e-12)
        assert got["median_stat_err"] == pytest.approx(statistics.median(f["signal_yield_err"] for f in fits))
        for k in ("pooled", "old_reference"):
            v = [f["shape_variations"][k]["signal_yield"] for f in fits]
            assert got["by_shape"][k]["mean"] == pytest.approx(statistics.mean(v), rel=1e-12)
            assert got["by_shape"][k]["sd"] == pytest.approx(statistics.stdev(v), rel=1e-12)
    assert pls["reference"]["signal_yield"] == v4["reference"]["top"]["signal_yield"]
    assert pls["published_188_class_checkpoint"]["signal_yield"] == v4["models"]["sophon-public"]["top"]["signal_yield"]


def test_analysis_v4_is_analysis_v3s_readout_made_from_fit_v4(v4):
    doc = json.loads(AN4.read_text())
    old = json.loads((DATA / "analysis_v3/aoj_top.json").read_text())
    assert set(doc) == set(old) | {"per_label_set"}
    assert set(doc["secondary"]["real_data_top"]) == set(old["secondary"]["real_data_top"])
    assert doc["provenance"]["input"] == "experiments/FIGS/data/aoj_full_v1/fit_v4/results.json"
    assert doc["provenance"]["input_sha256"] == hashlib.sha256(V4.read_bytes()).hexdigest()
    assert {r["model"]: r["signal_yield"] for r in doc["table"]} == {
        n: m["top"]["signal_yield"] for n, m in v4["models"].items() if n != "sophon-public"}
    assert doc["reference_models"]["sophon-public"]["signal_yield"] == v4["models"]["sophon-public"]["top"]["signal_yield"]
    assert doc["reference"]["signal_yield"] == v4["reference"]["top"]["signal_yield"]
