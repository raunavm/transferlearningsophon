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
    old = C.OPTIONS
    try:
        C.OPTIONS = dict(maxiter=2, ftol=1e-12, gtol=1e-8)     # stop the first fit early
        d = C.diagnose(model)
    finally:
        C.OPTIONS = old
    assert not d["success"]
    assert not d["at_minimum"]
