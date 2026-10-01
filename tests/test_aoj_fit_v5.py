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
