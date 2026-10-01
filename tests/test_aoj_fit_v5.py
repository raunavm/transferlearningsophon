"""experiments/AOJ/fit_v5.py: the fits with the tops that fail each cut in the fail
region, from the committed bins. The reference must be fit_v4's, every yield must rise
by what the tops in the fail region carried, and nothing is overwritten."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit_v5", ROOT / "experiments/AOJ/fit_v5.py")
F5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F5)
P = F5.P
DATA = ROOT / "experiments/FIGS/data/aoj_full_v1"


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    out = tmp_path_factory.mktemp("v5")
    F5.main(["--bins", str(DATA / "fit_v3/bins.npz"), "--previous", str(DATA / "fit_v4/results.json"),
             "--out", str(out / "fit"), "--analysis-out", str(out / "an"), "--workers", "1",
             "--models", "l162-s2", "--toys", "0"])
    return json.loads((out / "fit" / "results.json").read_text()), out


def test_the_reference_is_fit_v4s_and_the_tops_are_its_fitted_signal(one):
    res, _ = one
    v4 = json.loads((DATA / "fit_v4/results.json").read_text())
    ref, old = res["reference"]["top"], v4["reference"]["top"]
    assert ref["tf_order"] == old["tf_order"]
    assert abs(ref["signal_yield"] - old["signal_yield"]) < 1e-3 * old["signal_yield_err"]
    assert res["fail_tops"]["eps_ref"] == 1.0
    assert res["fail_tops"]["total"] == pytest.approx(ref["signal_yield"], rel=1e-6)


def test_the_yield_rises_by_the_tops_the_fail_region_carried_and_more_for_fewer_reference_passed_tops(one):
    res, _ = one
    f = res["models"]["l162-s2"]["top"]
    assert f["signal_yield"] - f["fit_v4_signal_yield"] > 0.3 * f["signal_yield_err"]
    shift = f["leak_systematic"]["shift"]
    assert 0 < shift["0.6"] < shift["0.4"]
    assert f["start_check"]["one_answer"]


def test_a_result_is_never_overwritten(one):
    _, out = one
    with pytest.raises(SystemExit, match="exists"):
        F5.main(["--out", str(out / "fit"), "--analysis-out", str(out / "an2")])


# ---- what the cluster run wrote (aoj_full_v1/fit_v5, analysis_v5) ----
V5 = DATA / "fit_v5" / "results.json"


def test_the_committed_fit_v5_is_what_its_code_and_bins_make():
    import hashlib
    import subprocess
    res = json.loads(V5.read_text())
    r = res["refit"]
    assert r["bins_sha256"] == hashlib.sha256((ROOT / r["bins"]).read_bytes()).hexdigest()
    pin = "mtx-s1.86"
    for key, path in (("peak_fit_sha256", "experiments/AOJ/peak_fit.py"), ("script_sha256", "experiments/AOJ/fit_v5.py")):
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{pin}:{path}"], capture_output=True).stdout
        assert r[key] == hashlib.sha256(blob).hexdigest(), f"{path} at {pin} is not the code that wrote fit_v5"


def test_fit_v5_has_the_reference_of_fit_v4_and_every_yield_raised_by_the_tops_in_the_fail_region():
    v4, v5 = json.loads((DATA / "fit_v4/results.json").read_text()), json.loads(V5.read_text())
    assert v5["reference"]["top"]["signal_yield"] == pytest.approx(v4["reference"]["top"]["signal_yield"], rel=1e-7)
    assert v5["fail_tops"]["total"] == pytest.approx(v4["reference"]["top"]["signal_yield"], rel=1e-6)
    assert set(v5["models"]) == set(v4["models"]) and len(v5["models"]) == 31
    for n, m in v5["models"].items():
        f = m["top"]
        assert f["fit_v4_signal_yield"] == v4["models"][n]["top"]["signal_yield"]
        assert 0.4 * f["signal_yield_err"] < f["signal_yield"] - f["fit_v4_signal_yield"] < 1.2 * f["signal_yield_err"], n
        assert f["profile_error_ok"] and not f["width_at_bound"] and not f["mean_at_bound"] and f["converged"]
        assert 0 < f["leak_systematic"]["shift"]["0.6"] < f["leak_systematic"]["shift"]["0.4"]
        assert f["start_check"]["one_answer"]


def test_the_v5_readout_is_the_mean_and_sd_over_seeds_of_the_v5_yields():
    v5 = json.loads(V5.read_text())
    an = json.loads((DATA / "analysis_v5/aoj_top.json").read_text())["per_label_set"]["label_sets"]
    for level, s in an.items():
        ys = [v5["models"][n]["top"]["signal_yield"] for n in s["models"]]
        assert len(ys) == 5 and s["signal_yields"] == ys
        assert s["signal_yield"]["mean"] == pytest.approx(np.mean(ys)) and s["signal_yield"]["sd"] == pytest.approx(np.std(ys, ddof=1))
