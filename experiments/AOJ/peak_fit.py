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
          order by Fisher F-test up from (2, 1), as CMS chooses it in its pass/fail
          fits (arXiv:1709.05543, 1710.00159). An order whose best fit puts TF on
          its floor (zero) in any bin is not admissible: a zero background under a
          populated fail bin is outside the model's physical range
    G     Gaussian core integrated over the mass bin. Mean and width FLOAT in every
          fit, the reference's and each score's, and are profiled at every order the
          F-test compares (_choose_shape_and_order). CMS takes the peak shape from
          simulation, with the jet-mass scale and resolution as constrained nuisance
          parameters fitted in situ (arXiv:1705.10532, CMS-EXO-16-030); with no
          simulated template here, the Gaussian mean and width are free parameters
    S_j   one signal yield per pT bin, so no signal pT spectrum is assumed; the
          reported yield is the fitted signal IN THE FITTED BINS, summed over pT.
          Its error is the profile-likelihood (MINOS) error with the TF and the
          shape re-minimised; the Hessian error at the fitted shape is kept beside it
    SHAPE SYSTEMATIC, label-set blind (shape_variations): every fit again at its own
          order with one pooled (mean, width), and with the reference's old shape
    Only bins lying fully inside the rho window are used. Signal left in the
    fail region is absorbed by q, which biases the yield LOW by about
    TF * (1 - eff_S) / eff_S of it -- 2 % for eff_S = 0.3, but these taggers keep a few
    per cent of the data tops, and then it is about one error: the fit takes the tops
    in each bin (`tops`, _Model), at least those the CMS reference passes.

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
WIDTH_BOUNDS = (3.0, 30.0)
PUBLISHED = "sophon-public"     # the published 188-class checkpoint: a reference row, never pooled
CROSSING_TOL, MAX_CROSSING_STEPS = 0.01, 6      # 2 x the profile rise within 1 +- 0.01
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


FIT_OPTIONS = dict(maxiter=2000, ftol=1e-12, gtol=1e-8)
MAX_RESTARTS = 500
RESTART_TOL = 1e-7          # half-deviance units
TF_FLOOR, MU_FLOOR = 1e-12, 1e-9


def _minimize(loss, x0):
    """L-BFGS-B, restarted from its own end point until a restart stops lowering the loss.

    ONE CALL IS NOT A MINIMUM. With these options a single call stopped at its
    iteration limit in 9 of the 32 real-data top fits of 2026-09-23, and a
    restart from its end point found a lower loss in every one of them, moving
    the yield by up to 0.77 of its error (experiments/AOJ/fit_convergence_check.py,
    2026-09-27; experiments/FIGS/data/aoj_full_v1/fit_convergence_check/). The
    loss has long, shallow valleys along the transfer-factor polynomial, and a
    restart discards the curvature memory that stalls there. Returns
    (x, loss, converged, n_restarts); converged means the last restart lowered
    the loss by less than RESTART_TOL. _Model.fit calls it in orthonormal
    coordinates: restarts alone left 31 of 32 fits short of their minimum."""
    x, f = np.asarray(x0, dtype=float), np.inf
    for k in range(MAX_RESTARTS):
        r = optimize.minimize(loss, x, jac=True, method="L-BFGS-B", options=FIT_OPTIONS)
        drop = f - float(r.fun)
        if drop > 0:
            x, f = r.x, float(r.fun)
        if drop < RESTART_TOL:
            return x, f, True, k
    return x, f, False, MAX_RESTARTS


class _Model:
    """pass = tf_norm * poly(rho, pT) * q + S_j * G ;  fail = q + F  (q profiled), where
    F = max(tops - S_j * G, 0) per bin: the tops that FAIL the cut, when `tops` (the tops
    in each bin, passing or failing) is given, else F = 0.

    WHY F (2026-10-01). Tops in the fail region are part of the fail counts, and with F = 0
    q absorbs them, so the pass background TF * q carries TF * F of them and the yield comes
    out low by up to sum(TF * F) (less where the TF polynomial takes up F's broad part). On
    data these taggers keep a few per cent of the tops (their yields are 0.03-0.11 of the
    CMS reference's at the same data efficiency), so nearly every top fails the cut; with
    tops = the reference's fitted signal, the fewest there can be, sum(TF * F) is 255-305
    jets for every score, about one error (fit_v5: the yields with and without F). F
    cannot float: with q free in every bin a fail-region signal is degenerate with q,
    constrained only through the variation of the TF across the peak."""

    def __init__(self, b, order, tf_norm, mean=None, width=None, tops=None):
        self.b, self.tf_norm, self.args = b, tf_norm, (order, tf_norm, mean, width)
        self.tops = None if tops is None else np.asarray(tops, float)
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
        # MINIMISED AND DIFFERENTIATED IN AN ORTHONORMAL BASIS (2026-09-28). The monomials
        # r^k p^l on [0, 1]^2 are nearly collinear at orders 3-4. In that basis L-BFGS-B
        # stopped short of the minimum by up to 2.4 in deviance (yields off by up to 1 sigma)
        # and finite-difference Hessians gave yield errors off by up to a factor 2
        # (experiments/AOJ/fit_minimum_diagnostic.py on the v2 fits). With X = Q R the
        # search and the Hessian use u = R x_tf, whose design is Q. The polynomial, the
        # deviance and its minimum are unchanged; x goes in and comes out in monomials.
        _, r = np.linalg.qr(self.X)
        self.T = np.eye(len(self.x0))
        self.T[:self.n_tf, :self.n_tf] = np.linalg.inv(r)

    def fail_signal(self, s):
        """F, the tops failing the cut in each bin, for signal s passing it."""
        return np.zeros_like(s) if self.tops is None else np.maximum(self.tops - s, 0.0)

    def expect(self, x):
        """(TF, pass signal, q, pass expectation). q maximises the likelihood of the bin:
        (1 + t)(t q + s)(q + F) = t p (q + F) + f (t q + s), a quadratic in q."""
        t = np.maximum(self.tf_norm * (self.X @ x[:self.n_tf]), TF_FLOOR)
        s = self.G @ x[self.n_tf:] if self.signal else np.zeros_like(t)
        F = self.fail_signal(s)
        p, f = self.b["n_pass"], self.b["n_fail"]
        a, bq, c = (1 + t) * t, (1 + t) * (s + t * F) - (p + f) * t, (1 + t) * s * F - t * p * F - f * s
        q = (-bq + np.sqrt(np.maximum(bq * bq - 4 * a * c, 0.0))) / (2 * a)
        return t, s, np.maximum(q, 1e-12), np.maximum(t * q + s, MU_FLOOR)

    def loss(self, x):
        """Half the saturated deviance, and its gradient (q is at its optimum, so
        only the explicit dependence on the parameters contributes). Where TF or the
        pass expectation sits on its floor the loss does not depend on them, and
        neither does the gradient; F = tops - s moves against the signal where it is
        not clipped at zero."""
        t, s, q, mu = self.expect(x)
        p, f = self.b["n_pass"], self.b["n_fail"]
        F = self.fail_signal(s)
        val = _half_deviance(p, mu).sum() + _half_deviance(f, q + F).sum()
        d_mu = (1.0 - p / mu) * (t * q + s > MU_FLOOR)
        on = self.tf_norm * (self.X @ x[:self.n_tf]) > TF_FLOOR
        d_sig = d_mu if self.tops is None else d_mu - (1.0 - f / (q + F)) * (self.tops > s)
        grad = np.r_[(d_mu * q * self.tf_norm * on) @ self.X, d_sig @ self.G if self.signal else []]
        return val, grad

    def _loss_u(self, u):
        val, g = self.loss(self.T @ u)
        return val, self.T.T @ g

    def fit(self, start=None):
        x0 = self.x0 if start is None else start
        u, f, self.converged, self.n_restarts = _minimize(self._loss_u, np.linalg.solve(self.T, x0))
        return self.T @ u, f

    def fit_at_yield(self, y, start):
        """Minimum of the loss with the total yield v.x_sig held at y, from `start`: the
        signal moves only in the directions orthogonal to v, the TF in orthonormal
        coordinates (fit_minimum_diagnostic.profile_error, which this generalises)."""
        n, v = self.n_tf, self.G.sum(axis=0)
        wc = np.linalg.svd(v[None, :])[2][1:].T
        rtf = self.T[:n, :n]
        join = lambda p: np.r_[rtf @ p[:n], y * v / (v @ v) + wc @ p[n:]]

        def loss(p):
            val, g = self.loss(join(p))
            return val, np.r_[rtf.T @ g[:n], wc.T @ g[n:]]
        p, f, _, _ = _minimize(loss, np.r_[np.linalg.solve(rtf, start[:n]), wc.T @ start[n:]])
        return join(p), f

    def n_at_floor(self, x):
        """Bins where the transfer factor sits on its floor."""
        return int((self.tf_norm * (self.X @ x[:self.n_tf]) <= TF_FLOOR).sum())

    def edm(self, x):
        """Estimated distance to the minimum, 1/2 g^T H^-1 g, in units of the loss."""
        g = self.loss(x)[1]
        return float(0.5 * g @ self.covariance(x) @ g)

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
        """pinv of the Hessian, by central differences of the gradient in the orthonormal
        coordinates, transformed back. Matches the profile-likelihood error of the yield
        to 0.2% on the real-data fits (fit_minimum_diagnostic.py)."""
        u = np.linalg.solve(self.T, x)
        h = np.zeros((len(u), len(u)))
        for k in range(len(u)):
            d = np.zeros(len(u)); d[k] = 1e-5 * max(1.0, abs(u[k]))
            h[k] = (self._loss_u(u + d)[1] - self._loss_u(u - d)[1]) / (2 * d[k])
        return self.T @ np.linalg.pinv(0.5 * (h + h.T)) @ self.T.T


def _f_test_up(fit, n_data, n_other):
    """F-test up from START_ORDER: raise an order only if it buys a significant drop
    in deviance. fit(order, parent) -> (half deviance, n bins with the TF on its floor,
    state, extra trail fields) is the maximum-likelihood fit at `order`; `parent` is
    (order, state) of the fit it is compared to, None at START_ORDER. n_other counts
    the parameters besides the TF's. A candidate whose fit puts the transfer factor on
    its floor in any bin is not admissible (module docstring); it is kept in the
    trail, marked. Returns (order, trail, state)."""
    n_par = lambda o: (o[0] + 1) * (o[1] + 1)
    order = START_ORDER
    dev, floor, state, extra = fit(order, None)
    trail = [dict(order=order, deviance=2 * dev, n_tf_at_floor=floor, **extra)]
    while True:
        best = None
        for cand in ((order[0] + 1, order[1]), (order[0], order[1] + 1)):
            dof = n_data - n_par(cand) - n_other
            if cand[0] > MAX_ORDER[0] or cand[1] > MAX_ORDER[1] or dof <= 0:
                continue
            dev_c, floor_c, state_c, extra = fit(cand, (order, state))
            if dev_c <= 0:
                continue
            if floor_c:
                trail.append(dict(order=cand, deviance=2 * dev_c, admissible=False, n_tf_at_floor=floor_c, **extra))
                continue
            f = ((dev - dev_c) / (n_par(cand) - n_par(order))) / (dev_c / dof)
            p = float(stats.f.sf(max(f, 0.0), n_par(cand) - n_par(order), dof))
            trail.append(dict(order=cand, deviance=2 * dev_c, f_test_p=p, **extra))
            if p < F_ALPHA and (best is None or p < best[3]):
                best = (cand, state_c, dev_c, p)
        if best is None:
            return order, trail, state
        order, state, dev = best[:3]


def _choose_order(b, tf_norm, mean, width, tops=None):
    """The F-test order at a FIXED signal shape (or background only, mean None). Each
    candidate is started from the fit it is compared to."""
    def fit(order, parent):
        model = _Model(b, order, tf_norm, mean, width, tops)
        x, dev = model.fit(None if parent is None else model.embed(parent[1], parent[0]))
        return dev, model.n_at_floor(x), x, {}
    n_sig = len(np.unique(b["j"])) if mean is not None else 0
    order, trail, _ = _f_test_up(fit, len(b["n_pass"]), n_sig)
    return order, trail


def _peak_cfg(peak):
    return PEAKS[peak] if isinstance(peak, str) else peak


def _tf_norm(b, window):
    side = ~in_windows(_bin_centres(b), [window])
    return b["n_pass"][side].sum() / max(b["n_fail"][side].sum(), 1.0)


def _minimize_shape(f, start, f_start, window, xtol, ftol):
    """((mean, width), f) minimising f inside the bounds from `start`, by Powell -- or,
    when scipy's bounded Powell ends ABOVE its start, by a simplex from the start. It
    did so on a peak narrower than a mass bin, whose minimum sits on the width floor
    (tests/test_aoj_peak_fit.py); without this the start came back as the answer."""
    bounds = [window, WIDTH_BOUNDS]
    r = optimize.minimize(f, start, method="Powell", bounds=bounds, options=dict(xtol=xtol, ftol=ftol))
    if r.fun > f_start:
        r = optimize.minimize(f, start, method="Nelder-Mead", bounds=bounds, options=dict(xatol=xtol, fatol=1e-9))
    if r.fun > f_start:
        return tuple(map(float, start)), f_start
    return (float(r.x[0]), float(r.x[1])), float(r.fun)


def _float_shape(b, tf_norm, order, window, starts=(), tops=None):
    """(mean, width) of the signal minimising the loss at this TF order.
    GRID FIRST, then polish. A wrong (mean, width) lets the polynomial contort
    itself into a bump, so the profile has local minima a line search falls into.
    The polish runs from the best grid point and from each shape in `starts`; the
    lowest end point wins."""
    outer = lambda v: _Model(b, order, tf_norm, v[0], v[1], tops).fit()[1]
    grid = [(m, w) for m in np.arange(window[0] + 5, window[1] - 4, 2.5) for w in SHAPE_WIDTHS]
    losses = [outer(v) for v in grid]
    k = int(np.argmin(losses))
    best = _minimize_shape(outer, grid[k], losses[k], window, 1e-2, 1e-6)
    for s in starts:
        polished = _minimize_shape(outer, tuple(s), outer(s), window, 1e-2, 1e-6)
        if polished[1] < best[1]:
            best = polished
    return best[0]


def _choose_shape_and_order(b, tf_norm, window, starts=(), tops=None):
    """TF order and signal shape by the nested-model comparison with the shape PROFILED
    at every order: each order the F-test visits gets its own floated (mean, width)
    (_float_shape, its polish started also from `starts` and from the shape of the
    order it is compared to), and the F-test compares those maximised likelihoods,
    the two shape parameters counted in every model. The order then depends on the
    data, not on where the shape search starts. Returns (order, shape, F-test trail,
    each order's floated mean and width in it).

    WHY (2026-09-28). Until then every score was fitted with the reference's shape,
    180.05 / 17.32 GeV, floated at START_ORDER. No score's peak has that shape: floated
    at its own order each sits near 183 / 11.5 GeV, the wider Gaussian took in
    background and the yields came out about 1.5x too high. And a shape floated at
    START_ORDER is not the shape at the fitted order: at (2, 1) the polynomial cannot
    follow the background, and in fit_v3 l162-s2's floated peak went to 145 GeV and
    r16q1mass-s4's to 190 GeV, both with the width on its 30 GeV bound. The first fix
    alternated {F-test order at the current shape; shape floated at that order} until
    the order held. Its F-test compared orders at a shape floated for one of them,
    which favours that order, and r16q1mass-s3 had two stable answers -- (2, 3) from
    the reference's shape, (2, 2) from the pooled one."""
    def fit(order, parent):
        shape = _float_shape(b, tf_norm, order, window, [*starts, *([parent[1]] if parent else [])], tops)
        model = _Model(b, order, tf_norm, *shape, tops)
        x, dev = model.fit()
        return dev, model.n_at_floor(x), shape, dict(mean=shape[0], width=shape[1])
    order, trail, shape = _f_test_up(fit, len(b["n_pass"]), len(np.unique(b["j"])) + 2)
    return order, shape, trail


def _profile_yield_error(model, x, window, guess):
    """(lo, hi) MINOS errors of the total yield with the TF AND the signal shape
    profiled: on each side, the yield change at which 2 x the loss, minimised over
    everything else at that yield, has risen by 1 -- the profile likelihood ratio of
    Cowan et al., arXiv:1007.1727 eq. (7), whose nuisance parameters here are the TF,
    the split of the yield over pT and the Gaussian's mean and width. Each side steps
    to its parabola's crossing until the rise is within CROSSING_TOL of 1. The shape
    is minimised by Powell (the loss is smooth in it near the fitted shape), the rest
    by the fit's own minimiser. The TF order is not profiled: it is a choice of model.
    None when the profile falls BELOW the fit's minimum -- with no peak the shape is
    not measured, and a dip placed elsewhere does better -- or when MAX_CROSSING_STEPS
    end with the rise still not within CROSSING_TOL of 1."""
    order, tf_norm, mean, width = model.args
    y0 = float(model.G.sum(axis=0) @ x[model.n_tf:])

    def prof(y):
        at = lambda v: _Model(model.b, order, tf_norm, v[0], v[1], model.tops).fit_at_yield(y, x)[1]
        return _minimize_shape(at, (mean, width), at((mean, width)), window, 1e-3, 1e-12)[1]
    f0 = prof(y0)
    out = []
    for sign in (-1.0, 1.0):
        d = guess
        for _ in range(MAX_CROSSING_STEPS):
            rise = 2 * (prof(y0 + sign * d) - f0)
            if rise <= 0:
                return None
            d, done = d / np.sqrt(rise), abs(rise - 1) < CROSSING_TOL
            if done:
                break
        else:
            return None
        out.append(float(d))
    return tuple(out)


def fit_peak(mass, pt, passed, peak, mean=None, width=None, float_shape=False, order=None, tops=None):
    """Simultaneous pass/fail fit of one peak. mean/width None -> background only."""
    return fit_binned(_bins(mass, pt, passed, _peak_cfg(peak)["fit_range"]), peak, mean, width, float_shape, order,
                      tops)


def fit_binned(b, peak, mean=None, width=None, float_shape=False, order=None, tops=None):
    """fit_peak on bins already made. float_shape: the mean and width float, profiled
    at every order the F-test compares (_choose_shape_and_order); `order` is then not
    used, and (mean, width) -- or when none is given the shape floated at START_ORDER
    -- is recorded as shape_start and also starts the shape search at every order.
    `peak` is a key of PEAKS or a config of the same form (window, fit_range) -- the
    signal-free pseudo-peak of experiments/AOJ/realdata_checks.py is one. `tops`: the
    tops in each bin, passing or failing (_Model); None: no signal in the fail region."""
    cfg = _peak_cfg(peak)
    tf_norm = _tf_norm(b, cfg["window"])
    trail = None
    if float_shape:
        start = (mean, width) if mean is not None else _float_shape(b, tf_norm, START_ORDER, cfg["window"], tops=tops)
        order, (mean, width), trail = _choose_shape_and_order(b, tf_norm, cfg["window"], [start], tops)
    elif order is None:
        order, trail = _choose_order(b, tf_norm, mean, width, tops)
    model = _Model(b, order, tf_norm, mean, width, tops)
    x, half_dev = model.fit()
    t, s, q, mu = model.expect(x)
    n_par = len(x) + (2 if float_shape else 0)
    out = dict(peak=peak if isinstance(peak, str) else cfg.get("name", "custom"), tf_order=list(order), f_test=trail, n_bins=int(len(t)), n_parameters=n_par,
               converged=model.converged, n_restarts=model.n_restarts, edm=model.edm(x),
               n_tf_at_floor=model.n_at_floor(x),
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
        _, dev0 = _Model(b, order, tf_norm, tops=tops).fit()
        if float_shape:
            lo, hi = _profile_yield_error(model, x, cfg["window"], err) or (None, None)
            at_bound = lambda v, bounds: bool(min(v - bounds[0], bounds[1] - v) < 0.05)
            out.update(signal_yield_err_lo=lo, signal_yield_err_hi=hi, signal_yield_err_hessian=err,
                       profile_error_ok=lo is not None,
                       shape_start=[float(v) for v in start],
                       mean_at_bound=at_bound(mean, cfg["window"]), width_at_bound=at_bound(width, WIDTH_BOUNDS))
            err = err if lo is None else 0.5 * (lo + hi)    # no profile error: the Hessian's, flagged
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
    _, s, q, mu = model.expect(x)
    fail = q + model.fail_signal(s)
    observed = model.loss(x)[0]
    worse = 0
    for _ in range(n_toys):
        toy = dict(model.b, n_pass=rng.poisson(mu).astype(float), n_fail=rng.poisson(fail).astype(float))
        worse += _Model(toy, *model.args, model.tops).fit()[1] >= observed
    return (worse + 1) / (n_toys + 1)


def validation(z, mass, pt, peak, eff, n_toys, shape=None):
    """Background-only fit in a signal-depleted slice of the FAIL region.

    HOW TO READ A FAILURE. The band is depleted for a GOOD tagger; a weak one
    leaves real signal in it and the background-only fit can then fail for that
    reason. `band_signal_z` is the significance of a peak of the score's own fitted
    shape in the band: background-only p < 0.05 WITH a significant band signal
    means leftover resonance, WITHOUT one it means the map plus polynomial do not
    describe the background."""
    outer = build_map(z, mass, pt, (VALIDATION_OFFSET + 1) * eff)
    inner = build_map(z, mass, pt, VALIDATION_OFFSET * eff)
    region = ~passes(z, mass, pt, inner)
    passed = passes(z, mass, pt, outer)[region]
    res, hist = validation_binned(_bins(mass[region], pt[region], passed, _peak_cfg(peak)["fit_range"]),
                                  peak, n_toys, shape)
    res["band"] = [VALIDATION_OFFSET * eff, (VALIDATION_OFFSET + 1) * eff]
    return res, hist


def validation_binned(b, peak, n_toys, shape=None):
    """validation() on the band's bins already made."""
    res, hist, (model, x) = fit_binned(b, peak)
    res["toy_p"] = toy_p_value(model, x, n_toys) if n_toys else None
    if shape is not None:
        with_signal, _, _ = fit_binned(b, peak, *shape, order=tuple(res["tf_order"]))
        res["band_signal_z"] = with_signal["z_wald"]
    return res, hist


def auc(score, label):
    """Mann-Whitney AUC of `score` for a boolean `label`."""
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return None
    return float((stats.rankdata(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def analyse(score, mass, pt, peak, eff, n_toys, shape=None, tops=None):
    """Map -> cut -> fit with the shape floating -> validation, for one score given on
    a log-odds-like scale. `shape` also starts the shape search (the reference's
    fitted shape); None for the reference itself, whose search also starts from its
    own shape floated at START_ORDER. `tops`: the tops in each bin (tops_from_reference)."""
    z = np.asarray(score, float)
    passed = passes(z, mass, pt, build_map(z, mass, pt, eff))
    fit, hist, _ = fit_peak(mass, pt, passed, peak, *(shape or (None, None)), float_shape=True, tops=tops)
    # floated_* was a second fit, at START_ORDER, until 2026-09-28; now it is this fit's
    fit["floated_mean"], fit["floated_width"] = fit["mean"], fit["width"]
    fit["data_efficiency"] = float(passed.mean())
    # THE WORKING POINT IS DEFINED ON THE SIDEBANDS: build_map sets 1 % of the jets
    # OUTSIDE the masked windows to pass. data_efficiency above counts all jets, so a
    # score that passes more inside the windows -- signal, or background it sculpts
    # there -- reads above 1 % (the CMS reference: 1.38 %, 2.2 % in the top window;
    # audit 2026-09-29). Both are recorded so the two are never confused.
    side = ~in_windows(mass, MASKED)
    fit["data_efficiency_sidebands"] = float(passed[side].mean())
    fit["data_efficiency_top_window"] = float(passed[in_windows(mass, [PEAKS["top"]["window"]])].mean())
    fit["validation"], hist_v = validation(z, mass, pt, peak, eff, n_toys, shape=(fit["mean"], fit["width"]))
    return fit, passed, dict(hist, **{f"validation_{k}": v for k, v in hist_v.items()})


# ONE PEAK SHAPE FOR THE PRETRAINED MODELS (2026-10-01). Floated per score, the Gaussian's
# mean and width trade against the transfer factor: on toys around each fit (experiments/AOJ/
# injection_test.py) the procedure's pulls had a standard deviation of 2.1 at 1,000 injected
# jets and 1.15 at 2,000-4,000, the shape running to a narrow width on a fluctuation (yield
# and error both small) or to the window's edge with a negative yield, where it stands in
# for TF curvature and the F-test stops at too low an order. At the injected shape the same
# toys gave pulls of mean 0 and width 0.95-1.13. The scores' own floated shapes are
# consistent with one shape (fit_v4: summed deviance change 45.5 for 62 degrees of freedom),
# as a property of the top peak and the detector rather than of the tagger would be.
POOL_MAX_ITERATIONS = 5


def pooled_shape(bins, pool, window, start, tops=None, mapper=map):
    """(shape, orders, iterations): the one (mean, width) minimising the summed loss of the
    scores in `pool`, each at its own TF order chosen by F-test AT that shape, alternated
    until the orders hold (at most POOL_MAX_ITERATIONS). `tops`: name -> tops or None;
    `mapper` runs the per-score fits (a process pool's map)."""
    tops = tops or {}
    norm = {n: _tf_norm(bins[n], window) for n in pool}
    shape, orders, trail = tuple(map(float, start)), None, []
    for _ in range(POOL_MAX_ITERATIONS):
        new = dict(zip(pool, mapper(_order_at, [(bins[n], norm[n], shape, tops.get(n)) for n in pool])))
        trail.append(dict(shape=list(shape), orders={n: list(o) for n, o in new.items()}))
        if new == orders:
            return shape, orders, trail
        orders = new

        def total(v):
            return sum(mapper(_loss_at, [(bins[n], orders[n], norm[n], (v[0], v[1]), tops.get(n)) for n in pool]))
        shape = _minimize_shape(total, shape, total(shape), window, 1e-2, 1e-9)[0]
    raise SystemExit(f"FATAL: the pooled shape's orders did not settle in {POOL_MAX_ITERATIONS} iterations: {trail}")


def _order_at(job):
    b, tf_norm, shape, tops = job
    return tuple(_choose_order(b, tf_norm, *shape, tops)[0])


def _loss_at(job):
    b, order, tf_norm, shape, tops = job
    return _Model(b, tuple(order), tf_norm, *shape, tops).fit()[1]


def shape_variations(bins, fits, pool, old_reference, peak, tops=None):
    """The shape systematic, blind to the label set: every fit in `fits` again, at its
    own TF order, with two fixed shapes --
      pooled         one (mean, width) minimising the summed loss of the fits named
                     in `pool`, each at its own order (main(): every pretrained
                     model; the reference and the published checkpoint are not pooled)
      old_reference  the reference's shape floated at START_ORDER, which every score
                     was fitted with until 2026-09-28
    `tops`: name -> the tops in each bin given that fit (None or absent: none).
    Adds fit["shape_variations"]; returns the two shapes."""
    cfg = PEAKS[peak]
    tops = tops or {}
    norm = {n: _tf_norm(b, cfg["window"]) for n, b in bins.items()}
    total = lambda v: sum(_Model(bins[n], tuple(fits[n]["tf_order"]), norm[n], v[0], v[1], tops.get(n)).fit()[1]
                          for n in pool)
    start = np.median([[fits[n]["mean"], fits[n]["width"]] for n in pool], axis=0)
    pooled = _minimize_shape(total, start, total(start), cfg["window"], 1e-2, 1e-9)[0]
    shapes = dict(pooled=list(pooled), old_reference=[float(v) for v in old_reference])
    for n, fit in fits.items():
        fit["shape_variations"] = {}
        for k, v in shapes.items():
            var = fit_binned(bins[n], peak, *v, order=tuple(fit["tf_order"]), tops=tops.get(n))[0]
            fit["shape_variations"][k] = dict(
                mean=v[0], width=v[1], signal_yield=var["signal_yield"], signal_yield_err=var["signal_yield_err"],
                deviance=var["deviance"], delta_deviance_vs_fitted_shape=var["deviance"] - fit["deviance"])
    return dict(shapes, pool=list(pool))


# THE TOPS IN THE FAIL REGION (2026-10-01, _Model). The fewest there can be are those the
# CMS reference passes: its efficiency on data tops is at most 1. EPS_REF = 1 takes that
# bound; a smaller value scales the tops up (the one-sided systematic of fit_v5).
EPS_REF = 1.0


def tops_from_reference(b_ref, ref_fit, eps_ref=EPS_REF):
    """The tops in each bin of the top fit: the reference's fitted signal there, refitted at
    its stored order and shape, over its efficiency on data tops eps_ref. Every score's
    top fit has the same (m_SD, pT) bins, so the array serves all of them."""
    model = _Model(b_ref, tuple(ref_fit["tf_order"]), _tf_norm(b_ref, PEAKS["top"]["window"]),
                   ref_fit["mean"], ref_fit["width"])
    x, _ = model.fit()
    s = model.expect(x)[1]
    if abs(s.sum() - ref_fit["signal_yield"]) > 1e-3 * ref_fit["signal_yield_err"]:
        raise SystemExit(f"FATAL: the reference refitted at its order and shape gives {s.sum():.1f}, "
                         f"not its yield {ref_fit['signal_yield']:.1f}")
    return s / eps_ref


def same_cells(a, b) -> bool:
    """Do two sets of bins hold the same (m_SD, pT) cells in the same order?"""
    return all(np.array_equal(a[k], b[k]) for k in ("m_edges", "i", "j"))


def criteria(fit, peak, toys):
    """The four criteria of one model's fit (a GO needs all)."""
    cfg = PEAKS[peak]
    val_p = fit["validation"]["toy_p"] if toys else fit["validation"]["asymptotic_p"]
    return dict(peak_position=bool(cfg["mean_ok"][0] <= fit["floated_mean"] <= cfg["mean_ok"][1]),
                s_over_sqrt_b=bool(fit["s_over_sqrt_b"] >= cfg["min_s_sqrt_b"]),
                validation_p=bool(val_p > MIN_VALIDATION_P),
                auc=bool((fit["auc_vs_cms_proxy"] or 0.0) >= MIN_AUC))


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
    # The full run fits the top peak only: the W channel was WITHDRAWN as a design
    # error (docs/PRESPEC_2026-09.md, amendment 2026-09-19) and is not re-fitted.
    ap.add_argument("--peaks", nargs="+", choices=list(PEAKS), default=list(PEAKS))
    ap.add_argument("--shape-pool", nargs="+", default=None, metavar="NAME",
                    help="models whose fits define the pooled shape of the shape systematic (default: every "
                         f"score but the published checkpoint, {PUBLISHED}; the reference is never pooled)")
    a = ap.parse_args()
    peaks = {pk: cfg for pk, cfg in PEAKS.items() if pk in a.peaks}
    score_key = dict(W="two_prong_logodds", top="three_prong_logodds")
    cms_key = dict(W="aoj_pn_WvsQCD", top="aoj_pn_TvsQCD")

    j = np.load(a.jets)
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = rho_of(mass, pt)
    ok = (rho > RHO_RANGE[0]) & (rho < RHO_RANGE[1]) & (pt > PT_RANGE[0]) & (pt < PT_RANGE[1])
    mass, pt = mass[ok], pt[ok]
    print(f"{ok.sum():,} of {len(ok):,} jets inside {RHO_RANGE[0]} < rho < {RHO_RANGE[1]}", flush=True)

    cms = {pk: logit(j[cms_key[pk]])[ok] for pk in peaks}
    ours = {}
    for item in a.scores:
        name, path = item.split("=", 1)
        s = np.load(path)
        ours[name] = {pk: s[score_key[pk]].astype(float)[ok] for pk in peaks}

    results = dict(reference={}, models={n: {} for n in ours}, shape_variations={})
    hists, ref_pass = {}, {}
    for peak, cfg in peaks.items():
        print(f"\n===== {peak}: reference (shipped CMS ParticleNet) =====", flush=True)
        ref, ref_pass[peak], h = analyse(cms[peak], mass, pt, peak, a.eff, a.toys)
        ref["ok"] = bool(ref["z_wald"] >= cfg["reference_z"])
        bins = {"reference": _bins(mass, pt, ref_pass[peak], cfg["fit_range"])}
        # the tops failing each score's cut: at least those the reference passes (_Model)
        tops = tops_from_reference(bins["reference"], ref) if peak == "top" else None
        results["reference"][peak] = ref
        hists.update({f"reference_{peak}_{k}": v for k, v in h.items()})
        print(json.dumps({k: ref[k] for k in ("mean", "width", "signal_yield", "signal_yield_err",
                                              "z_wald", "s_over_sqrt_b", "ok")}), flush=True)
        in_win = in_windows(mass, [cfg["window"]])
        for name, sc in ours.items():
            print(f"===== {peak}: {name} =====", flush=True)
            fit, passed, h = analyse(sc[peak], mass, pt, peak, a.eff, a.toys, shape=(ref["mean"], ref["width"]),
                                     tops=tops)
            bins[name] = _bins(mass, pt, passed, cfg["fit_range"])
            if tops is not None and not same_cells(bins[name], bins["reference"]):
                raise SystemExit(f"FATAL: {name}'s top bins are not the reference's cells")
            fit["efficiency_relative_to_reference"] = (
                fit["signal_yield"] / ref["signal_yield"] if ref["signal_yield"] > 0 else None)
            fit["auc_vs_cms_proxy"] = auc(sc[peak][in_win], ref_pass[peak][in_win])
            fit["criteria"] = criteria(fit, peak, a.toys)
            results["models"][name][peak] = fit
            hists.update({f"{name}_{peak}_{k}": v for k, v in h.items()})
            print(json.dumps({k: fit[k] for k in ("signal_yield", "signal_yield_err", "s_over_sqrt_b",
                                                  "floated_mean", "auc_vs_cms_proxy", "criteria")}), flush=True)
        fits = {"reference": ref, **{n: results["models"][n][peak] for n in ours}}
        pool = a.shape_pool or [n for n in ours if n != PUBLISHED]
        results["shape_variations"][peak] = shape_variations(bins, fits, pool, ref["shape_start"], peak,
                                                             {n: tops for n in ours})
        if tops is not None:
            results["fail_tops"] = dict(source="the reference's fitted signal per bin", eps_ref=EPS_REF,
                                        total=float(tops.sum()))

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
              + ", ".join(f"{pk} >= {c['reference_z']} sigma" for pk, c in peaks.items())
              + " through this pipeline, so a missing peak for OUR scores says nothing about them.")
    results.update(verdict=verdict, closure_hard_flags=hard, pipeline_ok=pipeline_ok,
                   eff=a.eff, n_jets=int(ok.sum()), n_toys=a.toys, peaks=list(peaks))

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "histograms.npz", **hists)
    print(f"\nwrote {out/'results.json'} and {out/'histograms.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
