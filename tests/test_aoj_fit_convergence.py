"""experiments/AOJ/fit_convergence_check.py: the rebuilt model must reproduce the
fit it checks, and the diagnosis must tell a fit at its minimum from a stall."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("fit_convergence_check", "experiments/AOJ/fit_convergence_check.py")
T = _mod("test_aoj_peak_fit_helpers", "tests/test_aoj_peak_fit.py")


def test_the_rebuilt_fit_reproduces_the_stored_yield_and_sits_at_its_minimum():
    mass, pt, score, _ = T.sample(3, n_sig=5000, peak="top")
    passed = C.P.passes(score, mass, pt, C.P.build_map(score, mass, pt, T.EFF))
    fit, _, _ = C.P.fit_peak(mass, pt, passed, "top", *T.SHAPES["top"])
    model = C.rebuild(mass, pt, score, "top", T.EFF, fit["tf_order"], fit["mean"], fit["width"])
    d = C.diagnose(model)
    assert abs(d["signal_yield"] - fit["signal_yield"]) <= 1e-6 * abs(fit["signal_yield"])
    assert d["at_minimum"] and d["restart_best_half_deviance_drop"] < 1e-3


def test_a_stalled_fit_is_not_called_a_minimum():
    mass, pt, score, _ = T.sample(4, n_sig=5000, peak="top")
    passed = C.P.passes(score, mass, pt, C.P.build_map(score, mass, pt, T.EFF))
    fit, _, _ = C.P.fit_peak(mass, pt, passed, "top", *T.SHAPES["top"])
    model = C.rebuild(mass, pt, score, "top", T.EFF, fit["tf_order"], fit["mean"], fit["width"])
    old = C.P.FIT_OPTIONS, C.P.MAX_RESTARTS
    try:                                    # stop the checked fit early, as the first full run's did
        C.P.FIT_OPTIONS, C.P.MAX_RESTARTS = dict(maxiter=2, ftol=1e-12, gtol=1e-8), 1
        d = C.diagnose(model)
    finally:
        C.P.FIT_OPTIONS, C.P.MAX_RESTARTS = old
    assert not d["success"]
    assert not d["at_minimum"]


def test_the_restarting_minimiser_reaches_the_minimum_a_single_capped_call_misses():
    """A 20-dimensional Rosenbrock valley: one L-BFGS-B call capped at 50 iterations
    stops short; restarting from its end point reaches the minimum at 1."""
    from scipy import optimize
    from scipy.optimize import rosen, rosen_der

    def loss(x):
        return rosen(x), rosen_der(x)
    x0 = np.full(20, -1.2)
    old = C.P.FIT_OPTIONS
    try:
        C.P.FIT_OPTIONS = dict(maxiter=50, ftol=1e-15, gtol=1e-10)
        one = optimize.minimize(loss, x0, jac=True, method="L-BFGS-B", options=C.P.FIT_OPTIONS)
        x, f, converged, n = C.P._minimize(loss, x0)
    finally:
        C.P.FIT_OPTIONS = old
    assert not one.success and one.fun > 1e-3
    assert converged and n > 1 and f < 1e-8 and np.allclose(x, 1.0, atol=1e-3)


def test_a_fit_records_that_it_converged_and_its_distance_to_the_minimum():
    mass, pt, score, _ = T.sample(3, n_sig=5000, peak="top")
    passed = C.P.passes(score, mass, pt, C.P.build_map(score, mass, pt, T.EFF))
    fit, _, _ = C.P.fit_peak(mass, pt, passed, "top", *T.SHAPES["top"])
    assert fit["converged"] and fit["n_restarts"] >= 1 and fit["edm"] < 1e-3


D = _mod("fit_minimum_diagnostic", "experiments/AOJ/fit_minimum_diagnostic.py")


BINS = ROOT / "experiments/FIGS/data/aoj_full_v1/fit_v2/bins.npz"
RESULTS = ROOT / "experiments/FIGS/data/aoj_full_v1/fit_v2/results.json"


def _real_fit(name):
    """One real-data top fit at its stored order, from the committed bins."""
    import json
    z = np.load(BINS)
    st = json.loads(RESULTS.read_text())["models"][name]["top"]
    b = {k: z[f"{name}|main|{k}"] for k in D.KEYS}
    fit, model, x = D.replay(b, C.P._Model, mean=st["mean"], width=st["width"], order=tuple(st["tf_order"]))
    return fit, model, x


@pytest.mark.parametrize("name", ["l162-s3", "r42q1-s2"])
def test_a_high_order_real_fit_is_at_its_minimum_and_its_error_is_the_profile_likelihood_error(name):
    """The two real fits the v2 minimiser left furthest from their minimum, at orders
    (4, 3) and (3, 3). Newton steps must find nothing lower, and the quoted error must
    be the yield change that raises the deviance by 1."""
    fit, model, x = _real_fit(name)
    xn, fn, edm, _ = D.newton(model, x)
    assert model.loss(x)[0] - fn < 1e-6 and edm < 1e-6
    assert abs(D.yield_and_error(model, xn)[0] - fit["signal_yield"]) < 1e-3 * fit["signal_yield_err"]
    assert D.profile_error(model, x, fit["signal_yield_err"]) == pytest.approx(fit["signal_yield_err"], rel=0.01)
    assert fit["n_tf_at_floor"] == 0


def test_the_gradient_is_the_derivative_of_the_loss_even_where_the_transfer_factor_is_floored():
    _, model, x = _real_fit("l162-s3")
    y = x.copy()
    y[1] -= 3.0                                     # drive the polynomial below zero in part of the plane
    assert model.n_at_floor(y) > 0
    g = model.loss(y)[1]
    num = np.array([(model.loss(y + e)[0] - model.loss(y - e)[0]) / 2e-7
                    for e in np.eye(len(y)) * 1e-7])
    assert np.allclose(g, num, rtol=1e-4, atol=1e-2)


def test_an_order_whose_fit_floors_the_transfer_factor_is_not_admissible(monkeypatch):
    mass, pt, score, _ = T.sample(5, n_sig=5000, peak="top")
    passed = C.P.passes(score, mass, pt, C.P.build_map(score, mass, pt, T.EFF))
    free, _, _ = C.P.fit_peak(mass, pt, passed, "top", *T.SHAPES["top"])
    step = [t["order"] for t in free["f_test"][1:] if t.get("f_test_p", 1.0) < C.P.F_ALPHA]
    assert step, "the fixture must raise the order at least once"
    banned = tuple(step[0])
    monkeypatch.setattr(C.P._Model, "n_at_floor", lambda self, x: 4 if tuple(self.args[0]) == banned else 0)
    fit, _, _ = C.P.fit_peak(mass, pt, passed, "top", *T.SHAPES["top"])
    marked = [t for t in fit["f_test"] if tuple(t["order"]) == banned]
    assert marked and all(t.get("admissible") is False and t["n_tf_at_floor"] == 4 for t in marked)
    assert tuple(fit["tf_order"]) != banned
