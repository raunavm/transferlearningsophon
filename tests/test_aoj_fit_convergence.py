"""experiments/AOJ/fit_convergence_check.py: the rebuilt model must reproduce the
fit it checks, and the diagnosis must tell a fit at its minimum from a stall."""
import importlib.util
import pathlib

import numpy as np

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
