#!/usr/bin/env python3
"""Are the real-data top fits at their minimum? Redo every fit from its exported bins
with a second, better-conditioned minimiser and compare.

WHY. The v2 fits (peak_fit.py at mtx-s1.61) restart L-BFGS-B until the loss stops
falling, and their check (fit_v2/check.json) still finds 16 of 32 fits with an
estimated distance to the minimum (EDM) above 1e-3, up to 1.3, although no restart
moves a yield by more than 0.09 sigma. The transfer factor is a polynomial in the
monomials r^k p^l on [0, 1]^2, a badly conditioned basis, so either L-BFGS-B stalls
in its valleys or the finite-difference Hessian behind the EDM is poor. This script
tells the two apart.

HOW. peak_fit.fit_peak is run unchanged on the exported (m_SD, pT) bins
(export_fit_bins.py), twice:
  as_run       with peak_fit._Model as it is -- must reproduce results.json, or
               nothing here is about those fits;
  orthonormal  with the same model minimised in an orthonormal basis of the SAME
               polynomial space (QR of the design matrix). The model, its deviance and
               its minimum are unchanged; only the conditioning of the search is.
The whole procedure is redone either way: the floated peak shape of the reference
(as fit_v2 and fit_v3 floated it, at START_ORDER; see replay), the F-test choice of
order, the fit, the background-only fit, the validation fit.
Then each orthonormal fit is polished with Newton steps on a Hessian taken in the
orthonormal coordinates, its gradient is checked against finite differences, and
the yield's error is measured by a profile-likelihood scan (the yield change that
raises the deviance by 1), the reference any Hessian-based error must match.

Nothing here changes a stored result. Toys are not redone.

Usage:
    python3 experiments/AOJ/fit_minimum_diagnostic.py --bins bins.npz \\
        --results results.json --out diagnostic.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import optimize

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("peak_fit", HERE / "peak_fit.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

KEYS = ("m_edges", "i", "j", "n_pass", "n_fail", "rho", "pt")
PEAK = "top"
AS_RUN = P._Model


class Orthonormal(P._Model):
    """peak_fit._Model minimised in an orthonormal basis of its transfer-factor space.

    x = T u with T = diag(R^-1, 1) from X = Q R, so X x_tf = Q u_tf: the polynomial is
    the same function of u as of x, the search sees orthonormal directions. Parameters
    go in and come out in the monomial basis, so embed(), expect() and the covariance
    are the parent's."""

    def transform(self):
        _, r = np.linalg.qr(self.X)
        t = np.eye(len(self.x0))
        t[:self.n_tf, :self.n_tf] = np.linalg.inv(r)
        return t

    def loss_u(self, t):
        return lambda u: (lambda v, g: (v, t.T @ g))(*self.loss(t @ u))

    def fit(self, start=None):
        t = self.transform()
        x0 = self.x0 if start is None else start
        u, f, self.converged, self.n_restarts = P._minimize(self.loss_u(t), np.linalg.solve(t, x0))
        return t @ u, f


@contextlib.contextmanager
def patched(**attrs):
    old = {k: getattr(P, k) for k in attrs}
    for k, v in attrs.items():
        setattr(P, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(P, k, v)


def replay(b, model_cls, **kw):
    """peak_fit.fit_peak on these bins with this model class. float_shape=True is
    replayed as the procedure of fit_v2 and fit_v3 ran it -- the shape floated at
    START_ORDER, then the order by F-test at that shape (refit_from_bins.old_fit) --
    not as peak_fit now floats it (the shape profiled at every order)."""
    with patched(_bins=lambda *a, **k: b, _Model=model_cls):
        if kw.pop("float_shape", False):
            window = P.PEAKS[PEAK]["window"]
            kw["mean"], kw["width"] = P._float_shape(b, P._tf_norm(b, window), P.START_ORDER, window)
        out, _, (model, x) = P.fit_peak(None, None, None, PEAK, **kw)
    return out, model, x


def hessian_u(loss_u, u):
    h = np.zeros((len(u), len(u)))
    for k in range(len(u)):
        d = np.zeros(len(u)); d[k] = 1e-5 * max(1.0, abs(u[k]))
        h[k] = (loss_u(u + d)[1] - loss_u(u - d)[1]) / (2 * d[k])
    return 0.5 * (h + h.T)


def newton(model, x, n_iter=100):
    """Newton steps with backtracking in the orthonormal coordinates, from x. Returns
    (x, loss, edm, n_steps); edm = 1/2 g^T H^-1 g at the end point."""
    t = Orthonormal.transform(model)
    lu = Orthonormal.loss_u(model, t)
    u = np.linalg.solve(t, x)
    f, g = lu(u)
    steps = 0
    for steps in range(n_iter):
        w, v = np.linalg.eigh(hessian_u(lu, u))
        step = -v @ ((v.T @ g) / np.maximum(w, 1e-12 * max(w.max(), 1e-300)))
        if -0.5 * g @ step < 1e-12:
            break
        a = 1.0
        while a > 1e-10:
            f1, g1 = lu(u + a * step)
            if f1 <= f + 1e-4 * a * (g @ step):
                break
            a *= 0.5
        if a <= 1e-10:
            break
        u, f, g = u + a * step, f1, g1
    w, v = np.linalg.eigh(hessian_u(lu, u))
    keep = w > 1e-12 * w.max()                       # pinv: flat directions carry no distance
    edm = float(0.5 * np.sum((v.T @ g)[keep] ** 2 / w[keep]))
    return t @ u, float(f), edm, steps


def yield_and_error(model, x):
    """Signal yield in the fitted bins and its error from the Hessian taken in the
    orthonormal coordinates (the quantity peak_fit quotes, better conditioned)."""
    t = Orthonormal.transform(model)
    lu = Orthonormal.loss_u(model, t)
    cov = t @ np.linalg.pinv(hessian_u(lu, np.linalg.solve(t, x))) @ t.T
    v = model.G.sum(axis=0)
    c = cov[model.n_tf:, model.n_tf:]
    return float(v @ x[model.n_tf:]), float(np.sqrt(max(v @ c @ v, 0.0)))


def gradient_check(model, x):
    """Largest |analytic - central difference| gradient component, orthonormal coordinates."""
    t = Orthonormal.transform(model)
    lu = Orthonormal.loss_u(model, t)
    u = np.linalg.solve(t, x)
    g = lu(u)[1]
    num = np.zeros_like(u)
    for k in range(len(u)):
        d = np.zeros(len(u)); d[k] = 1e-6 * max(1.0, abs(u[k]))
        num[k] = (lu(u + d)[0] - lu(u - d)[0]) / (2 * d[k])
    return float(np.max(np.abs(g - num)))


def clipped(model, x):
    """Bins where the transfer factor or the pass expectation sits on peak_fit's floor."""
    raw_t = model.tf_norm * (model.X @ x[:model.n_tf])
    s = model.G @ x[model.n_tf:] if model.signal else np.zeros_like(raw_t)
    _, _, q, _ = model.expect(x)
    return dict(min_transfer_factor=float(raw_t.min()), n_tf_clipped=int((raw_t <= 1e-12).sum()),
                n_mu_clipped=int((raw_t.clip(1e-12) * q + s <= 1e-9).sum()))


def profile_error(model, x, sigma_guess):
    """Profile-likelihood error of the total yield Y = v.s at the minimum x, as MINOS
    defines it: where 2 x the loss, minimised over everything else at fixed Y, has risen
    by 1. Measured at Y +- sigma_guess and scaled by the local parabola on each side,
    the two sides averaged. None if the transfer factor sits on its floor at x."""
    if clipped(model, x)["n_tf_clipped"]:
        return None
    t = Orthonormal.transform(model)
    n, v = model.n_tf, model.G.sum(axis=0)
    rtf = t[:n, :n]
    wc = np.linalg.svd(v[None, :])[2][1:].T                   # the signal directions orthogonal to v
    join = lambda u, y, w: np.r_[rtf @ u, y * v / (v @ v) + wc @ w]
    f0 = model.loss(x)[0]
    y0 = float(v @ x[n:])
    p0 = np.r_[np.linalg.solve(rtf, x[:n]), wc.T @ x[n:]]

    def prof(y):
        # tighter than the fit's own settings: at +-0.5 sigma the deviance rises by only
        # 0.25, so a profile point left 0.01 short moves the error by several per cent
        def f(p):
            val, g = model.loss(join(p[:n], y, p[n:]))
            return val, np.r_[rtf.T @ g[:n], wc.T @ g[n:]]
        p, best = p0, np.inf
        for _ in range(200):
            r = optimize.minimize(f, p, jac=True, method="L-BFGS-B",
                                  options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
            if best - r.fun < 1e-10:
                return min(best, float(r.fun))
            p, best = r.x, float(r.fun)
        return best
    d2 = [2 * (prof(y0 + k * sigma_guess) - f0) for k in (-1.0, 1.0)]
    if min(d2) <= 0:
        return None
    return float(np.mean([sigma_guess / np.sqrt(d) for d in d2]))


def one(job):
    name, b, bv, stored = job
    kw = dict(float_shape=True) if name == "reference" else dict(mean=stored["mean"], width=stored["width"])
    row = {}
    fits = {}
    for label, cls in (("as_run", AS_RUN), ("orthonormal", Orthonormal)):
        out, model, x = replay(b, cls, **kw)
        fits[label] = (out, model, x)
        val, vmodel, vx = replay(bv, cls)
        row[label] = dict(tf_order=out["tf_order"], mean=out["mean"], width=out["width"],
                          signal_yield=out["signal_yield"], signal_yield_err=out["signal_yield_err"],
                          deviance=out["deviance"], edm=out["edm"], n_restarts=out["n_restarts"],
                          delta_deviance_vs_background_only=out["delta_deviance_vs_background_only"],
                          validation_tf_order=val["tf_order"], validation_deviance=val["deviance"],
                          validation_asymptotic_p=val["asymptotic_p"], validation_edm=val["edm"])
    v2 = row["as_run"]
    # Across machines an ill-conditioned minimiser stops at slightly different points,
    # so "reproduces" means the same orders and the yield within 0.1 of its error;
    # the size of the difference is kept, it is itself a diagnostic.
    row["replay_yield_shift_over_err"] = (v2["signal_yield"] - stored["signal_yield"]) / stored["signal_yield_err"]
    row["replay_deviance_shift"] = v2["deviance"] - stored["deviance"]
    row["reproduces_stored"] = bool(
        v2["tf_order"] == stored["tf_order"]
        and v2["validation_tf_order"] == stored["validation"]["tf_order"]
        and abs(row["replay_yield_shift_over_err"]) < 0.1)
    out, model, x = fits["orthonormal"]
    xn, fn, edm_n, steps = newton(model, x)
    y_o, err_o = yield_and_error(model, x)
    y_n, _ = yield_and_error(model, xn)
    _, model2, x2 = fits["as_run"]
    y_2, err_2 = yield_and_error(model2, x2)
    row["newton"] = dict(deviance=2 * fn, edm=edm_n, n_steps=steps, signal_yield=y_n,
                         deviance_drop_from_orthonormal=out["deviance"] - 2 * fn)
    row["orthonormal_err_from_orthonormal_hessian"] = err_o
    row["as_run_err_from_orthonormal_hessian"] = err_2
    row["profile_err_at_minimum"] = profile_error(model, xn, err_o)
    row["gradient_check_max_abs"] = gradient_check(model, x)
    row["clipping_orthonormal"] = clipped(model, x)
    row["clipping_as_run"] = clipped(model2, x2)
    same_order = row["orthonormal"]["tf_order"] == v2["tf_order"]
    row["summary"] = dict(
        same_order=same_order,
        same_validation_order=row["orthonormal"]["validation_tf_order"] == v2["validation_tf_order"],
        deviance_as_run_minus_orthonormal=v2["deviance"] - row["orthonormal"]["deviance"] if same_order else None,
        yield_shift_over_err=(row["orthonormal"]["signal_yield"] - v2["signal_yield"]) / err_2,
        newton_yield_shift_over_err=(y_n - y_o) / err_o,
        err_ratio_orthonormal_over_as_run=row["orthonormal"]["signal_yield_err"] / v2["signal_yield_err"],
        profile_err_over_as_run_err=(row["profile_err_at_minimum"] / v2["signal_yield_err"]
                                     if row["profile_err_at_minimum"] else None),
        validation_deviance_as_run_minus_orthonormal=(v2["validation_deviance"] - row["orthonormal"]["validation_deviance"]
                                                  if row["orthonormal"]["validation_tf_order"] == v2["validation_tf_order"]
                                                  else None))
    print(f"{name:14s} reproduced {row['reproduces_stored']!s:5s} order {v2['tf_order']}->"
          f"{row['orthonormal']['tf_order']}  dDev {row['summary']['deviance_as_run_minus_orthonormal']}  "
          f"dY/err {row['summary']['yield_shift_over_err']:+.3f}  Newton dY/err "
          f"{row['summary']['newton_yield_shift_over_err']:+.1e} edm {edm_n:.1e}", flush=True)
    return name, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bins", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="fit names to run (default all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    res = json.loads(pathlib.Path(a.results).read_text())
    z = np.load(a.bins)
    stored = {"reference": res["reference"][PEAK], **{n: m[PEAK] for n, m in res["models"].items()}}
    names = [n for n in stored if not a.only or n in a.only]
    jobs = [(n, {k: z[f"{n}|main|{k}"] for k in KEYS}, {k: z[f"{n}|validation|{k}"] for k in KEYS}, stored[n])
            for n in names]
    if a.workers == 1:
        rows = dict(map(one, jobs))
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = dict(ex.map(one, jobs))
    s = [r["summary"] for r in rows.values()]
    out = {"results": a.results, "bins": a.bins, "fits": rows, "summary": {
        "n_fits": len(rows),
        "n_as_run_reproduced": sum(r["reproduces_stored"] for r in rows.values()),
        "max_abs_replay_yield_shift_over_err": max(abs(r["replay_yield_shift_over_err"]) for r in rows.values()),
        "n_same_order": sum(x["same_order"] for x in s),
        "n_same_validation_order": sum(x["same_validation_order"] for x in s),
        "max_abs_yield_shift_over_err": max(abs(x["yield_shift_over_err"]) for x in s),
        "max_deviance_as_run_minus_orthonormal": max((x["deviance_as_run_minus_orthonormal"] for x in s
                                                  if x["deviance_as_run_minus_orthonormal"] is not None), default=None),
        "min_deviance_as_run_minus_orthonormal": min((x["deviance_as_run_minus_orthonormal"] for x in s
                                                  if x["deviance_as_run_minus_orthonormal"] is not None), default=None),
        "max_newton_edm": max(r["newton"]["edm"] for r in rows.values()),
        "max_abs_newton_yield_shift_over_err": max(abs(x["newton_yield_shift_over_err"]) for x in s),
        "err_ratio_range": [min(x["err_ratio_orthonormal_over_as_run"] for x in s),
                            max(x["err_ratio_orthonormal_over_as_run"] for x in s)],
        "profile_over_as_run_err_range": [min((x["profile_err_over_as_run_err"] for x in s
                                               if x["profile_err_over_as_run_err"]), default=None),
                                          max((x["profile_err_over_as_run_err"] for x in s
                                               if x["profile_err_over_as_run_err"]), default=None)],
        "max_n_tf_clipped": max(r["clipping_orthonormal"]["n_tf_clipped"] for r in rows.values())}}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
