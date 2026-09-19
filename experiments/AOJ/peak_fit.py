#!/usr/bin/env python3
"""Find the W and top mass peaks in UNLABELLED data with a tagger score, the way
CMS does it: a fixed-DATA-efficiency cut, then a simultaneous pass/fail fit.

There is no truth label in AspenOpenJets, so "does this score tag W bosons on
real data" can only be answered by whether a resonance appears in the soft-drop
mass of the jets it selects -- and a tagger that merely prefers heavy jets also
makes a bump. Two steps keep those apart.

(a) THE MAP. The score is cut at a threshold t(rho, pT), rho = ln(m_SD^2/pT^2),
    chosen so a fixed fraction `eff` of DATA passes everywhere in the
    (rho, pT) plane. The per-cell quantiles are fitted with a LOW-ORDER
    polynomial, and jets in the 65-105 and 140-220 GeV windows are MASKED when
    it is built: an unmasked data-driven map would raise the threshold exactly
    where the signal sits and erase it. Restricted to -5.5 < rho < -2.
    LIMIT, stated because no test can remove it: a background whose score has
    structure NARROWER than a masked window cannot be told from signal by any
    data-driven map. The shipped non-mass-decorrelated ParticleNet scores are
    the likeliest to do that; the validation fit below is the check on it.

(b) THE FIT, per (m_SD, pT) bin:   fail = q,   pass = TF(rho, pT) * q + S_j * G(m)
    q     free per bin, profiled analytically (it is the fail spectrum)
    TF    (pass/fail outside the peak window) times a polynomial in (rho, pT);
          order by F-test from (2, 1)
    G     Gaussian core integrated over the mass bin. Mean and width FLOAT only
          for the reference selection (the shipped CMS score) and are FIXED from
          it for every other score
    S_j   one signal yield per pT bin, so no signal pT spectrum is assumed; the
          reported yield is the fitted signal IN THE FITTED BINS, summed over pT
          with the full covariance
    Only bins lying fully inside the rho window are used. Signal left in the
    fail region is absorbed by q, which biases the yield LOW by about
    TF * (1 - eff_S) / eff_S -- 2 % for a good tagger, the safe direction.

    VALIDATION. The same machinery, background only, in a slice of the FAIL
    region: pass' = the `eff`-wide score band starting 50 * eff below the cut
    (50-51 % for eff = 1 %), where a usable tagger has left almost no signal.
    Its goodness of fit -- saturated deviance against toys -- says whether map
    plus polynomial describe background THROUGH the mass windows. See
    validation() for how to read a failure.

Run:  python3 experiments/AOJ/peak_fit.py --jets OUT/jets.npz --closure OUT/closure.json \
          --scores sophon-public=OUT/scores_sophon-public.npz mtx-l162-s1b=... --out OUT
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import optimize, stats
from scipy.special import xlogy

RHO_RANGE = (-5.5, -2.0)
PT_RANGE = (500.0, 2500.0)
PEAKS = dict(
    W=dict(window=(65.0, 105.0), fit_range=(40.0, 140.0),
           mean_ok=(75.0, 90.0), min_s_sqrt_b=5.0, reference_z=10.0),
    top=dict(window=(140.0, 220.0), fit_range=(105.0, 300.0),
             mean_ok=(160.0, 185.0), min_s_sqrt_b=3.0, reference_z=5.0))
MASKED = tuple(p["window"] for p in PEAKS.values())
MASS_BIN = 5.0
PT_EDGES = np.array([500, 550, 600, 675, 750, 850, 1000, 1200, 2500], dtype=float)
MAP_ORDER, MAP_CELLS, MAP_MIN_PASS, MAP_ITERATIONS = (3, 2), (10, 5), 10, 6
START_ORDER, MAX_ORDER, F_ALPHA = (2, 1), (4, 3), 0.05
SHAPE_WIDTHS = (5.0, 7.0, 9.0, 12.0, 15.0, 20.0)
# The validation band starts this many `eff` below the cut: 50-51 % for eff = 1 %.
# NOT just under the cut. Signal in a band scales with the ROC slope there, which
# for a ParticleNet-class tagger is ~2 at 10 % background efficiency -- the band
# would hold twice the inclusive W fraction and fail for that reason alone
# (measured on synthetic taggers: p < 0.05 three times in sixteen at offset 10).
# At the median the slope is ~0.1. The price: a band this deep tests the map and
# the polynomial on real QCD, not the sculpting of the tight working point.
VALIDATION_OFFSET = 50
MIN_AUC, MIN_VALIDATION_P = 0.8, 0.05


# ------------------------------------------------------------------ (a) the map
def rho_of(mass, pt):
    return 2.0 * np.log(mass / pt)


def _unit(rho, pt):
    return ((rho - RHO_RANGE[0]) / (RHO_RANGE[1] - RHO_RANGE[0]),
            np.log(pt / PT_RANGE[0]) / np.log(PT_RANGE[1] / PT_RANGE[0]))


def _design(rho, pt, order):
    r, p = _unit(np.asarray(rho, float), np.asarray(pt, float))
    return np.stack([r ** k * p ** l for k in range(order[0] + 1) for l in range(order[1] + 1)], axis=1)


def logit(p, tiny=1e-7):
    """Probability-like score -> log-odds. The threshold surface is a polynomial
    in the SCORE, so the score must live on an unbounded, roughly linear scale:
    a probability saturating at 1 does not, its log-odds does. (A rank transform
    was tried and is worse: when the score depends strongly on rho it maps a
    linear threshold onto an exponential one that no low-order surface follows.)"""
    p = np.clip(np.asarray(p, float), tiny, 1.0 - tiny)
    return np.log(p) - np.log1p(-p)


def in_windows(mass, windows):
    out = np.zeros(len(mass), dtype=bool)
    for lo, hi in windows:
        out |= (mass > lo) & (mass < hi)
    return out


def build_map(z, mass, pt, eff, masked=MASKED, order=MAP_ORDER, cells=MAP_CELLS):
    """Polynomial threshold surface at data efficiency `eff`, windows masked.

    ITERATED on the residual z - t(rho, pT). One pass is biased: where the score
    has a gradient across a cell, the cell's top `eff` comes from its high end,
    so the cell quantile sits above the threshold at the cell's centre (measured
    on a score linear in rho: 0.46 % passed for a 1 % target). The residual of a
    correct surface has no gradient left, so re-fitting it converges.
    """
    rho = rho_of(mass, pt)
    ok = ~in_windows(mass, masked)
    rho_edges = np.linspace(*RHO_RANGE, cells[0] + 1)
    pt_edges = np.unique(np.quantile(pt[ok], np.linspace(0, 1, cells[1] + 1)))
    pt_edges[0], pt_edges[-1] = PT_RANGE
    ir, ip = np.digitize(rho, rho_edges) - 1, np.digitize(pt, pt_edges) - 1
    sels = [ok & (ir == i) & (ip == j) for i in range(cells[0]) for j in range(len(pt_edges) - 1)]
    sels = [np.flatnonzero(sel) for sel in sels if sel.sum() * eff >= MAP_MIN_PASS]
    n_coef = (order[0] + 1) * (order[1] + 1)
    if len(sels) < 2 * n_coef:
        raise SystemExit(f"FATAL: only {len(sels)} populated (rho, pT) cells for a {n_coef}-term "
                         f"map at eff = {eff}; too few jets")
    centre = _design([rho[k].mean() for k in sels], [np.exp(np.log(pt[k]).mean()) for k in sels], order)
    w = np.sqrt([len(k) for k in sels])
    full, coef = _design(rho, pt, order), np.zeros(n_coef)
    for _ in range(MAP_ITERATIONS):
        resid = z - full @ coef
        q = np.array([np.quantile(resid[k], 1.0 - eff) for k in sels])
        coef += np.linalg.lstsq(centre * w[:, None], q * w, rcond=None)[0]
    return dict(coef=coef, order=tuple(order), eff=eff, n_cells=len(sels))


def passes(z, mass, pt, m):
    return z > _design(rho_of(mass, pt), pt, m["order"]) @ m["coef"]


# ------------------------------------------------------------------ (b) the fit
def _bins(mass, pt, passed, fit_range):
    """Pass/fail counts in the (m_SD, pT) bins lying FULLY inside the rho window."""
    m_edges = np.arange(fit_range[0], fit_range[1] + 1e-9, MASS_BIN)
    h = lambda sel: np.histogram2d(mass[sel], pt[sel], bins=(m_edges, PT_EDGES))[0]
    n_pass, n_fail = h(passed), h(~passed)
    sum_rho = np.histogram2d(mass, pt, bins=(m_edges, PT_EDGES), weights=rho_of(mass, pt))[0]
    sum_pt = np.histogram2d(mass, pt, bins=(m_edges, PT_EDGES), weights=pt)[0]
    m_lo, m_hi = m_edges[:-1, None], m_edges[1:, None]
    p_lo, p_hi = PT_EDGES[None, :-1], PT_EDGES[None, 1:]
    inside = (rho_of(m_lo, p_hi) > RHO_RANGE[0]) & (rho_of(m_hi, p_lo) < RHO_RANGE[1])
    use = inside & (n_pass + n_fail > 0)
    n = np.maximum(n_pass + n_fail, 1)
    ii, jj = np.nonzero(use)
    return dict(m_edges=m_edges, i=ii, j=jj, n_pass=n_pass[use], n_fail=n_fail[use],
                rho=(sum_rho / n)[use], pt=(sum_pt / n)[use])


def _bin_centres(b):
    return 0.5 * (b["m_edges"][:-1] + b["m_edges"][1:])[b["i"]]


def _gauss_bins(m_edges, mean, width):
    return np.diff(stats.norm.cdf(m_edges, mean, width))


def _half_deviance(n, mu):
    return mu - n + xlogy(n, n) - xlogy(n, mu)


class _Model:
    """pass = tf_norm * poly(rho, pT) * q + S_j * G ;  fail = q  (q profiled)."""

    def __init__(self, b, order, tf_norm, mean=None, width=None):
        self.b, self.tf_norm, self.args = b, tf_norm, (order, tf_norm, mean, width)
        self.X = _design(b["rho"], b["pt"], order)
        self.n_tf = self.X.shape[1]
        self.signal = mean is not None
        self.pt_bins = np.unique(b["j"]) if self.signal else np.array([], dtype=int)
        if self.signal:
            g = _gauss_bins(b["m_edges"], mean, width)[b["i"]]
            # yields are fitted in units of ~sqrt(pass in that pT bin), so O(1)
            self.scale = np.array([max(np.sqrt(b["n_pass"][b["j"] == j].sum()), 3.0) for j in self.pt_bins])
            self.G = np.stack([g * (b["j"] == j) for j in self.pt_bins], axis=1) * self.scale
        # START AT THE SIDEBAND-SUBTRACTED EXCESS, not at zero signal. tf_norm is the
        # pass/fail ratio outside the peak window, so a flat TF is already close; with
        # S = 0 and a peak of S/B >> 1 the first steps drive the polynomial negative,
        # where it is clipped and has no gradient, and the fit stalls at a deviance
        # ten times the minimum (seen on the synthetic reference).
        s0 = []
        if self.signal:
            excess = b["n_pass"] - tf_norm * b["n_fail"]
            core = np.abs(_bin_centres(b) - mean) < 2 * width
            for k in range(len(self.pt_bins)):
                norm = self.G[core, k].sum()
                s0.append(max(excess[core & (self.G[:, k] > 0)].sum(), 0.0) / norm if norm > 0 else 0.0)
        self.x0 = np.r_[1.0, np.zeros(self.n_tf - 1), s0]

    def expect(self, x):
        t = np.maximum(self.tf_norm * (self.X @ x[:self.n_tf]), 1e-12)
        s = self.G @ x[self.n_tf:] if self.signal else np.zeros_like(t)
        p, f = self.b["n_pass"], self.b["n_fail"]
        a, bq, c = (1 + t) * t, (1 + t) * s - (p + f) * t, -f * s
        q = (-bq + np.sqrt(np.maximum(bq * bq - 4 * a * c, 0.0))) / (2 * a)
        return t, s, np.maximum(q, 1e-12), np.maximum(t * q + s, 1e-9)

    def loss(self, x):
        """Half the saturated deviance, and its gradient (q is at its optimum, so
        only the explicit dependence on the parameters contributes)."""
        t, s, q, mu = self.expect(x)
        p, f = self.b["n_pass"], self.b["n_fail"]
        val = _half_deviance(p, mu).sum() + _half_deviance(f, q).sum()
        d_mu = 1.0 - p / mu
        grad = np.r_[(d_mu * q * self.tf_norm) @ self.X, d_mu @ self.G if self.signal else []]
        return val, grad

    def fit(self, start=None):
        r = optimize.minimize(self.loss, self.x0 if start is None else start, jac=True,
                              method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
        self.converged = bool(r.success)
        return r.x, float(r.fun)

    def embed(self, x, order):
        """Parameters of a lower-order fit, laid out for THIS model's order -- a
        nested model started there can only improve on it."""
        out = np.zeros(len(self.x0))
        out[self.n_tf:] = x[(order[0] + 1) * (order[1] + 1):]
        mine = self.args[0]
        for k in range(order[0] + 1):
            for l in range(order[1] + 1):
                out[k * (mine[1] + 1) + l] = x[k * (order[1] + 1) + l]
        return out

    def covariance(self, x):
        h, eps = np.zeros((len(x), len(x))), 1e-4
        for k in range(len(x)):
            d = np.zeros(len(x)); d[k] = eps * max(1.0, abs(x[k]))
            h[k] = (self.loss(x + d)[1] - self.loss(x - d)[1]) / (2 * d[k])
        return np.linalg.pinv(0.5 * (h + h.T))


def _choose_order(b, tf_norm, mean, width):
    """F-test up from START_ORDER: raise an order only if it buys a significant
    drop in deviance. Each candidate is started from the current best fit."""
    n_par = lambda o: (o[0] + 1) * (o[1] + 1)
    n_data, n_sig = len(b["n_pass"]), (len(np.unique(b["j"])) if mean is not None else 0)
    order = START_ORDER
    x, dev = _Model(b, order, tf_norm, mean, width).fit()
    trail = [dict(order=order, deviance=2 * dev)]
    while True:
        best = None
        for cand in ((order[0] + 1, order[1]), (order[0], order[1] + 1)):
            dof = n_data - n_par(cand) - n_sig
            if cand[0] > MAX_ORDER[0] or cand[1] > MAX_ORDER[1] or dof <= 0:
                continue
            model = _Model(b, cand, tf_norm, mean, width)
            x_c, dev_c = model.fit(model.embed(x, order))
            if dev_c <= 0:
                continue
            f = ((dev - dev_c) / (n_par(cand) - n_par(order))) / (dev_c / dof)
            p = float(stats.f.sf(max(f, 0.0), n_par(cand) - n_par(order), dof))
            trail.append(dict(order=cand, deviance=2 * dev_c, f_test_p=p))
            if p < F_ALPHA and (best is None or p < best[3]):
                best = (cand, x_c, dev_c, p)
        if best is None:
            return order, trail
        order, x, dev = best[:3]


def fit_peak(mass, pt, passed, peak, mean=None, width=None, float_shape=False, order=None):
    """Simultaneous pass/fail fit of one peak. mean/width None -> background only."""
    cfg = PEAKS[peak]
    b = _bins(mass, pt, passed, cfg["fit_range"])
    side = ~in_windows(_bin_centres(b), [cfg["window"]])
    tf_norm = b["n_pass"][side].sum() / max(b["n_fail"][side].sum(), 1.0)
    if float_shape:
        # GRID FIRST, then polish. A wrong (mean, width) lets the polynomial contort
        # itself into a bump, so the profile has local minima a line search falls into.
        outer = lambda v: _Model(b, order or START_ORDER, tf_norm, v[0], v[1]).fit()[1]
        grid = [(m, w) for m in np.arange(cfg["window"][0] + 5, cfg["window"][1] - 4, 2.5)
                for w in SHAPE_WIDTHS]
        losses = [outer(v) for v in grid]
        best = grid[int(np.argmin(losses))]
        r = optimize.minimize(outer, best, method="Powell", bounds=[cfg["window"], (3.0, 30.0)],
                              options=dict(xtol=1e-2, ftol=1e-6))
        mean, width = (float(r.x[0]), float(r.x[1])) if r.fun <= min(losses) else best
    trail = None
    if order is None:
        order, trail = _choose_order(b, tf_norm, mean, width)
    model = _Model(b, order, tf_norm, mean, width)
    x, half_dev = model.fit()
    t, s, q, mu = model.expect(x)
    n_par = len(x)
    out = dict(peak=peak, tf_order=list(order), f_test=trail, n_bins=int(len(t)), n_parameters=n_par,
               converged=model.converged,
               deviance=2 * half_dev, n_pass=float(b["n_pass"].sum()), n_fail=float(b["n_fail"].sum()),
               asymptotic_p=float(stats.chi2.sf(2 * half_dev, max(len(t) - n_par, 1))))
    hist = {k: np.bincount(b["i"], weights=v, minlength=len(b["m_edges"]) - 1)
            for k, v in dict(n_pass=b["n_pass"], n_fail=b["n_fail"], background=t * q, signal=s).items()}
    hist["m_edges"] = b["m_edges"]
    if model.signal:
        # THE YIELD IS THE SIGNAL IN THE FITTED BINS, not the Gaussian's full norm:
        # near rho = -2 part of the top peak lies outside the window, and quoting
        # the extrapolated norm would credit jets no cut was ever applied to.
        cov = model.covariance(x)[model.n_tf:, model.n_tf:]
        v = model.G.sum(axis=0)
        yields = v * x[model.n_tf:]
        err = float(np.sqrt(max(v @ cov @ v, 0.0)))
        win = in_windows(_bin_centres(b), [cfg["window"]])
        s_win, b_win = float(s[win].sum()), float((t * q)[win].sum())
        _, dev0 = _Model(b, order, tf_norm).fit()
        out.update(mean=mean, width=width, shape_floated=bool(float_shape),
                   signal_yield=float(yields.sum()), signal_yield_err=err,
                   yield_per_pt_bin={f"{PT_EDGES[j]:g}-{PT_EDGES[j + 1]:g}": float(y)
                                     for j, y in zip(model.pt_bins, yields)},
                   z_wald=float(yields.sum() / err) if err > 0 else 0.0,
                   delta_deviance_vs_background_only=float(2 * (dev0 - half_dev)),
                   s_in_window=s_win, b_in_window=b_win,
                   s_over_sqrt_b=s_win / np.sqrt(b_win) if b_win > 0 else 0.0)
    return out, hist, (model, x)


def toy_p_value(model, x, n_toys, seed=0):
    """Saturated-deviance goodness of fit against parametric-bootstrap toys."""
    rng = np.random.default_rng(seed)
    _, _, q, mu = model.expect(x)
    observed = model.loss(x)[0]
    worse = 0
    for _ in range(n_toys):
        toy = dict(model.b, n_pass=rng.poisson(mu).astype(float), n_fail=rng.poisson(q).astype(float))
        worse += _Model(toy, *model.args).fit()[1] >= observed
    return (worse + 1) / (n_toys + 1)


def validation(z, mass, pt, peak, eff, n_toys, shape=None):
    """Background-only fit in a signal-depleted slice of the FAIL region.

    HOW TO READ A FAILURE. The band is depleted for a GOOD tagger; a weak one
    leaves real signal in it and the background-only fit can then fail for that
    reason. `band_signal_z` is the significance of a peak of the reference shape
    fitted in the band: background-only p < 0.05 WITH a significant band signal
    means leftover resonance, WITHOUT one it means the map plus polynomial do not
    describe the background."""
    outer = build_map(z, mass, pt, (VALIDATION_OFFSET + 1) * eff)
    inner = build_map(z, mass, pt, VALIDATION_OFFSET * eff)
    region = ~passes(z, mass, pt, inner)
    passed = passes(z, mass, pt, outer)[region]
    res, hist, (model, x) = fit_peak(mass[region], pt[region], passed, peak)
    res["toy_p"] = toy_p_value(model, x, n_toys) if n_toys else None
    res["band"] = [VALIDATION_OFFSET * eff, (VALIDATION_OFFSET + 1) * eff]
    if shape is not None:
        with_signal, _, _ = fit_peak(mass[region], pt[region], passed, peak, *shape, order=tuple(res["tf_order"]))
        res["band_signal_z"] = with_signal["z_wald"]
    return res, hist


def auc(score, label):
    """Mann-Whitney AUC of `score` for a boolean `label`."""
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return None
    return float((stats.rankdata(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def analyse(score, mass, pt, peak, eff, n_toys, shape=None):
    """Map -> cut -> fit (+ floating-shape diagnostic) -> validation, for one score
    given on a log-odds-like scale."""
    z = np.asarray(score, float)
    passed = passes(z, mass, pt, build_map(z, mass, pt, eff))
    floated, hist_f, _ = fit_peak(mass, pt, passed, peak, float_shape=True)
    if shape is None:                                   # this IS the reference
        fit, hist = floated, hist_f
    else:
        fit, hist, _ = fit_peak(mass, pt, passed, peak, mean=shape[0], width=shape[1])
        fit["floated_mean"], fit["floated_width"] = floated["mean"], floated["width"]
    fit["data_efficiency"] = float(passed.mean())
    fit["validation"], hist_v = validation(z, mass, pt, peak, eff, n_toys, shape=(fit["mean"], fit["width"]))
    return fit, passed, dict(hist, **{f"validation_{k}": v for k, v in hist_v.items()})


# ---------------------------------------------------------------------- verdict
def decide(per_peak, hard_flags, pipeline_ok):
    """(verdict, failed criteria) for one model.

    PIPELINE-INVALID outranks NO-GO: if the shipped CMS scores do not show the
    peaks through this pipeline, a missing peak for OUR score says nothing."""
    failed = [f"{pk}:{c}" for pk, f in per_peak.items() for c, good in f["criteria"].items() if not good]
    return ("PIPELINE-INVALID" if not pipeline_ok else
            "NO-GO" if (failed or hard_flags) else "GO"), failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jets", required=True, help="jets.npz from discriminants.py")
    ap.add_argument("--scores", nargs="+", required=True, metavar="NAME=scores.npz")
    ap.add_argument("--closure", default=None, help="closure.json; hard flags veto a GO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eff", type=float, default=0.01, help="data efficiency of every cut")
    ap.add_argument("--toys", type=int, default=200)
    a = ap.parse_args()

    j = np.load(a.jets)
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = rho_of(mass, pt)
    ok = (rho > RHO_RANGE[0]) & (rho < RHO_RANGE[1]) & (pt > PT_RANGE[0]) & (pt < PT_RANGE[1])
    mass, pt = mass[ok], pt[ok]
    print(f"{ok.sum():,} of {len(ok):,} jets inside {RHO_RANGE[0]} < rho < {RHO_RANGE[1]}", flush=True)

    cms = dict(W=logit(j["aoj_pn_WvsQCD"])[ok], top=logit(j["aoj_pn_TvsQCD"])[ok])
    ours = {}
    for item in a.scores:
        name, path = item.split("=", 1)
        s = np.load(path)
        ours[name] = dict(W=s["two_prong_logodds"].astype(float)[ok],
                          top=s["three_prong_logodds"].astype(float)[ok])

    results, hists, ref_pass = dict(reference={}, models={n: {} for n in ours}), {}, {}
    for peak, cfg in PEAKS.items():
        print(f"\n===== {peak}: reference (shipped CMS ParticleNet) =====", flush=True)
        ref, ref_pass[peak], h = analyse(cms[peak], mass, pt, peak, a.eff, a.toys)
        ref["ok"] = bool(ref["z_wald"] >= cfg["reference_z"])
        results["reference"][peak] = ref
        hists.update({f"reference_{peak}_{k}": v for k, v in h.items()})
        print(json.dumps({k: ref[k] for k in ("mean", "width", "signal_yield", "signal_yield_err",
                                              "z_wald", "s_over_sqrt_b", "ok")}), flush=True)
        in_win = in_windows(mass, [cfg["window"]])
        for name, sc in ours.items():
            print(f"===== {peak}: {name} =====", flush=True)
            fit, _, h = analyse(sc[peak], mass, pt, peak, a.eff, a.toys, shape=(ref["mean"], ref["width"]))
            fit["efficiency_relative_to_reference"] = (
                fit["signal_yield"] / ref["signal_yield"] if ref["signal_yield"] > 0 else None)
            fit["auc_vs_cms_proxy"] = auc(sc[peak][in_win], ref_pass[peak][in_win])
            val_p = fit["validation"]["toy_p"] if a.toys else fit["validation"]["asymptotic_p"]
            fit["criteria"] = dict(
                peak_position=bool(cfg["mean_ok"][0] <= fit["floated_mean"] <= cfg["mean_ok"][1]),
                s_over_sqrt_b=bool(fit["s_over_sqrt_b"] >= cfg["min_s_sqrt_b"]),
                validation_p=bool(val_p > MIN_VALIDATION_P),
                auc=bool((fit["auc_vs_cms_proxy"] or 0.0) >= MIN_AUC))
            results["models"][name][peak] = fit
            hists.update({f"{name}_{peak}_{k}": v for k, v in h.items()})
            print(json.dumps({k: fit[k] for k in ("signal_yield", "signal_yield_err", "s_over_sqrt_b",
                                                  "floated_mean", "auc_vs_cms_proxy", "criteria")}), flush=True)

    hard = json.loads(pathlib.Path(a.closure).read_text())["hard_flags"] if a.closure else []
    pipeline_ok = all(r["ok"] for r in results["reference"].values())
    verdict = {}
    for name, per_peak in results["models"].items():
        verdict[name], failed = decide(per_peak, hard, pipeline_ok)
        print(f"\nVERDICT {name}: {verdict[name]}"
              + (f"   failed: {', '.join(failed)}" if failed else "")
              + (f"   closure hard flags: {hard}" if hard else ""))
    if not pipeline_ok:
        print("PIPELINE-INVALID: the shipped CMS scores do not reach "
              + ", ".join(f"{pk} >= {c['reference_z']} sigma" for pk, c in PEAKS.items())
              + " through this pipeline, so a missing peak for OUR scores says nothing about them.")
    results.update(verdict=verdict, closure_hard_flags=hard, pipeline_ok=pipeline_ok,
                   eff=a.eff, n_jets=int(ok.sum()), n_toys=a.toys)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "histograms.npz", **hists)
    print(f"\nwrote {out/'results.json'} and {out/'histograms.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
