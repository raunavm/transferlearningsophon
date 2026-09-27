#!/usr/bin/env python3
"""Do the real-data top fits that report `converged: false` sit at their minimum?

WHY. The full real-data run (experiments/FIGS/data/aoj_full_v1/results.json)
records `converged` from scipy's L-BFGS-B `success` flag, and it is false for 21
of 32 fits, the CMS reference among them. With ftol=1e-12 and gtol=1e-8 that
flag is also false when the line search simply cannot improve a loss that is
already at its minimum to machine precision, so the flag alone cannot say
whether a quoted yield is a minimum or a stall. This script answers it for
every fit, with no change to how the fits were made:

  1. Re-run each fit exactly as peak_fit.py made it: the same cut, bins,
     transfer-factor order (read from results.json, so the F-test is not
     re-run), signal shape and minimiser settings. The yield must reproduce the
     stored one, or this check is not checking those fits.
  2. Record why the minimiser stopped, the largest gradient component, and the
     estimated distance to the minimum, EDM = 1/2 g^T H^-1 g in units of the
     loss (half the deviance), as MINUIT reports it.
  3. Restart from the end point, and from perturbed starts around it. A fit at
     its minimum returns to the same loss and the same yield; a stalled one
     finds a lower loss.

Nothing here changes a stored result. The output is a report beside them.

Usage (in the fit job's environment, after merge_shards.py):
    python3 experiments/AOJ/fit_convergence_check.py --jets merged/jets.npz \\
        --results results.json --scores NAME=scores_NAME.npz ... --out check.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from scipy import optimize

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("peak_fit", HERE / "peak_fit.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

OPTIONS = dict(maxiter=2000, ftol=1e-12, gtol=1e-8)       # peak_fit._Model.fit
N_PERTURBED = 8
PERTURBATION = 0.1                                          # relative, per parameter


def rebuild(mass, pt, score, peak, eff, order, mean, width):
    """The final signal-plus-background model of one fit, as fit_peak builds it."""
    cfg = P.PEAKS[peak]
    passed = P.passes(score, mass, pt, P.build_map(score, mass, pt, eff))
    b = P._bins(mass, pt, passed, cfg["fit_range"])
    side = ~P.in_windows(P._bin_centres(b), [cfg["window"]])
    tf_norm = b["n_pass"][side].sum() / max(b["n_fail"][side].sum(), 1.0)
    return P._Model(b, tuple(order), tf_norm, mean, width)


def signal_yield(model, x):
    return float(model.G.sum(axis=0) @ x[model.n_tf:])


def diagnose(model, seed=0) -> dict:
    """Reproduce the fit with the fit's own minimiser, then its gradient, EDM, and
    what single L-BFGS-B restarts from its end point and from nearby starts find.
    (peak_fit.py at mtx-s1.56, the tag of the first full run, fits with one
    L-BFGS-B call of OPTIONS; from mtx-s1.61 it restarts until the loss stops
    falling. model.fit() is whichever the checked fit used.)"""
    x, f = model.fit()
    g = model.loss(x)[1]
    cov = model.covariance(x)                                # pinv of the Hessian
    edm = float(0.5 * g @ cov @ g)
    y = signal_yield(model, x)
    v = model.G.sum(axis=0)
    y_err = float(np.sqrt(max(v @ cov[model.n_tf:, model.n_tf:] @ v, 0.0)))
    rng = np.random.default_rng(seed)
    starts = [x] + [x * (1 + PERTURBATION * rng.standard_normal(len(x))) for _ in range(N_PERTURBED)]
    runs = []
    for s in starts:
        rr = optimize.minimize(model.loss, s, jac=True, method="L-BFGS-B", options=OPTIONS)
        runs.append((float(rr.fun), signal_yield(model, rr.x)))
    best_f, best_y = min(runs)
    return {"success": bool(model.converged), "n_restarts": getattr(model, "n_restarts", 0),
            "half_deviance": f, "grad_max_abs": float(np.max(np.abs(g))), "edm": edm,
            "signal_yield": y, "signal_yield_err": y_err,
            "restart_best_half_deviance_drop": f - best_f,
            "restart_yield_shift_at_best": best_y - y,
            "restart_max_yield_shift": max(abs(ry - y) for _, ry in runs),
            "at_minimum": bool(f - best_f < 1e-3 and edm < 1e-3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jets", required=True)
    ap.add_argument("--results", required=True, help="results.json of the run being checked")
    ap.add_argument("--scores", nargs="+", required=True, metavar="NAME=scores.npz")
    ap.add_argument("--peak", default="top", choices=list(P.PEAKS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = json.loads(pathlib.Path(a.results).read_text())
    eff = res["eff"]
    j = np.load(a.jets)
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    mass, pt = mass[ok], pt[ok]
    if int(ok.sum()) != res["n_jets"]:
        raise SystemExit(f"FATAL: {int(ok.sum())} jets in the fit region, results.json has "
                         f"{res['n_jets']}; these are not the jets those fits used")
    key = dict(W="two_prong_logodds", top="three_prong_logodds")[a.peak]
    cms_key = dict(W="aoj_pn_WvsQCD", top="aoj_pn_TvsQCD")[a.peak]
    ref = res["reference"][a.peak]
    fits = {"reference": (P.logit(j[cms_key])[ok], ref)}
    for item in a.scores:
        name, path = item.split("=", 1)
        fits[name] = (np.load(path)[key].astype(float)[ok], res["models"][name][a.peak])
    out = {"results": a.results, "peak": a.peak, "options": OPTIONS,
           "n_perturbed": N_PERTURBED, "perturbation": PERTURBATION, "fits": {}}
    for name, (score, stored) in fits.items():
        model = rebuild(mass, pt, score, a.peak, eff, stored["tf_order"], stored["mean"], stored["width"])
        d = diagnose(model)
        d["stored_signal_yield"] = stored["signal_yield"]
        d["stored_converged"] = stored["converged"]
        d["reproduces_stored_yield"] = bool(abs(d["signal_yield"] - stored["signal_yield"])
                                            <= 1e-6 * max(1.0, abs(stored["signal_yield"])))
        out["fits"][name] = d
        print(f"{name:14s} stored {stored['signal_yield']:9.1f}  refit {d['signal_yield']:9.1f}  "
              f"success {d['success']!s:5s} edm {d['edm']:.2e}  restart drop "
              f"{d['restart_best_half_deviance_drop']:.2e}  at_minimum {d['at_minimum']}", flush=True)
    f = out["fits"].values()
    out["summary"] = {"n_fits": len(out["fits"]),
                      "n_reproduce_stored_yield": sum(x["reproduces_stored_yield"] for x in f),
                      "n_stored_converged": sum(bool(x["stored_converged"]) for x in f),
                      "n_at_minimum": sum(x["at_minimum"] for x in f),
                      "max_restart_yield_shift_over_err": max(
                          x["restart_max_yield_shift"] / x["signal_yield_err"] for x in f
                          if x["signal_yield_err"] > 0)}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
