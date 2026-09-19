"""Multiple-contrast trend test over ordered label granularities (STATISTICS §2).

Design: k ordered levels (188, 162, 43, 17 pretraining classes), one scalar
endpoint per (level, seed) — log(1−AUC) or log background rejection — with
seeds PAIRED across levels by seed index: a randomised complete block design
whose block is the seed. The expected effect is non-uniform (flat between some
levels, a step between others), so a single linear contrast is mis-specified.
The test is the maximum over a family of monotone-trend contrasts (max-T;
Hothorn, Bretz & Westfall 2008, Biom. J. 50:346), with isotonic regression as
the shape-free companion (Barlow, Bartholomew, Bremner & Brunk 1972).

Reference distribution: level labels permuted WITHIN each seed block
(`blocks=None`: across all units), enumerated exactly when (k!)^b is small
enough, Monte-Carlo otherwise with a caller-supplied generator or seed. There
is no default seed and no global seed.

RESOLUTION. 5 seeds x 4 levels has 24^5 = 7,962,624 arrangements, but the
smallest attainable p (`p_min`) is the share of arrangements TIED at the
maximum, not 1/N: relabelling levels inside a pooled set leaves that contrast's
t unchanged (data-dependent; for the default family typically 2/N one-sided
and 4/N two-sided, and 1/N for the isotonic statistic). Only data ordered
across all four levels in every block get there. A pure step at one end of the ladder cannot: the 6^b arrangements
that keep the same level extreme in every block share the numerator, so p is
roughly uniform on (0, (1/4)^b) however large the step — below 9.8e-4 at 5
seeds, below 1.6e-2 at 3.

ASSUMPTION. Under the null the k endpoints of a block are exchangeable across
levels, and blocks are independent. This is the sharp null — granularity does
not change the endpoint's distribution — not merely "equal means": a seed
variance that differs between levels breaks exchangeability with equal means.

NOT LICENSED. This is a trend test over level means. The arg-max contrast says
where on the ladder the step sits; it says nothing about WHICH physical
distinction merged between those levels caused it. The per-contrast adjusted
p-values are single-step max-T (Westfall & Young 1993): familywise error is
controlled exactly under the complete null; strong control needs subset
pivotality, which a joint permutation of all levels gives only approximately.
The contrast family and the direction must be fixed before looking at the data.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

EXACT_MAX = 10_000_000   # enumerate when (k!)^b <= this; 5 blocks x 4 levels = 7,962,624
N_PERM_DEFAULT = 99_999
_CHUNK = 200_000         # Monte-Carlo arrangements held in memory at once
_RTOL = 1e-9             # float slack so the observed arrangement counts as >= itself


def _pairs(k, family):
    """(i, j) index pairs: levels 0..i pooled, against levels j..k-1 pooled."""
    families = {"williams": [(0, j) for j in range(k - 1, 0, -1)],
                "changepoint": [(i, i + 1) for i in range(k - 1)],
                "marcus": [(i, j) for i in range(k - 1) for j in range(i + 1, k)]}
    if family not in families:
        raise ValueError(f"family must be one of {sorted(families)}, got {family!r}")
    return families[family]


def contrast_matrix(k=4, family="marcus", n=None) -> np.ndarray:
    """Monotone-trend contrasts for k ordered groups; rows sum to 0, unit norm.

    Rows are positive for means that INCREASE along the level order. `n` =
    per-level sample sizes; pooled sets are n-weighted (Bretz 2006, CSDA
    50:1735, the extension to unbalanced designs). Families:

    williams     first level (the control) vs the pooled top levels — the
                 contrast form of Williams 1971 (Biometrics 27:103) / 1972
                 (Biometrics 28:519) given by Bretz 2006. k-1 rows.
    marcus       pooled first levels vs pooled last levels, every such pair
                 (Marcus 1976, Biometrika 63:177; contrast form Bretz 2006).
                 k(k-1)/2 rows; contains the other two families. No level is
                 privileged as control, hence the default for this ladder.
    changepoint  all levels below a cut vs all levels above it (step contrasts,
                 Hirotsu type; `multcomp::contrMat(type="Changepoint")`).
                 k-1 rows; the arg-max row is the step location.
    """
    n = np.ones(k) if n is None else np.asarray(n, dtype=float)
    pairs = _pairs(k, family)
    c = np.zeros((len(pairs), k))
    for row, (i, j) in zip(c, pairs):
        row[:i + 1] = -n[:i + 1] / n[:i + 1].sum()
        row[j:] = n[j:] / n[j:].sum()
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def _design(y, levels, blocks, order, inference=True) -> dict:
    """Centred responses laid out for permutation, plus the invariants.

    Blocked: rows = complete blocks x levels, centred within block; a block
    with a missing (or non-finite) cell is dropped whole and reported.
    Unblocked: one row of all units, centred on the grand mean.
    """
    y, levels = np.asarray(y, dtype=float), np.asarray(levels)
    order = np.unique(levels) if order is None else np.asarray(order)
    k = len(order)
    pos = {v: j for j, v in enumerate(order.tolist())}
    col = np.array([pos[v] for v in levels.tolist()])
    ok = np.isfinite(y)
    if blocks is None:
        y, col = y[ok], col[ok]
        onehot = np.eye(k)[col]
        n = onehot.sum(0)
        rows, sums = (y - y.mean())[None, :], (y - y.mean()) @ onehot
        means, df, n_blocks, dropped = (y @ onehot) / n, len(y) - k, None, []
    else:
        ids, bi = np.unique(np.asarray(blocks), return_inverse=True)
        count = np.zeros((len(ids), k), dtype=int)
        np.add.at(count, (bi[ok], col[ok]), 1)
        if count.max() > 1:
            raise ValueError("more than one endpoint in a (block, level) cell")
        table = np.full((len(ids), k), np.nan)
        table[bi[ok], col[ok]] = y[ok]
        keep = count.all(1)
        table, dropped = table[keep], ids[~keep].tolist()
        rows = table - table.mean(1, keepdims=True)
        onehot, sums, means = None, rows.sum(0), table.mean(0)
        n_blocks = int(keep.sum())
        n, df = np.full(k, float(n_blocks)), (n_blocks - 1) * (k - 1)
    if n.min() < 1 or (inference and df < 1):
        raise ValueError("need every level observed and, for a test, >= 2 complete "
                         "blocks (blocked) or more units than levels (unblocked)")
    return {"order": order, "rows": rows, "onehot": onehot, "n": n, "sums": sums,
            "means": means, "ss": float((rows ** 2).sum()), "df": df,
            "n_blocks": n_blocks, "blocks_dropped": dropped}


def _enumerate(rows):
    """Level sums of every within-block relabelling, in k! chunks of (k!)^(b-1)."""
    k = rows.shape[1]
    perms = np.array(list(itertools.permutations(range(k))))
    tables = rows[:, perms]                          # (b, k!, k)
    head = tables[0]
    for t in tables[1:-1]:
        head = (head[:, None, :] + t[None, :, :]).reshape(-1, k)
    for last in tables[-1]:
        yield head + last


def _monte_carlo(rows, onehot, n_perm, gen):
    """Level sums of n_perm random relabellings; every row shuffled independently."""
    for start in range(0, n_perm, _CHUNK):
        m = min(_CHUNK, n_perm - start)
        sums = gen.permuted(np.repeat(rows[None], m, axis=0), axis=2).sum(1)
        yield sums if onehot is None else sums @ onehot


def _reference(d, stat, observed, exact, n_perm, rng) -> dict:
    """P(max statistic over the family >= each observed statistic) under relabelling.

    `stat` maps (m, k) level sums to (m, n_stats). Exact: p = hits / N, with the
    observed arrangement among the N. Monte-Carlo: p = (hits + 1) / (n_perm + 1).
    """
    rows, onehot = d["rows"], d["onehot"]
    if exact is None:
        exact = onehot is None and math.factorial(rows.shape[1]) ** rows.shape[0] <= EXACT_MAX
    if exact and onehot is not None:
        raise ValueError("exact enumeration is implemented for the blocked design only")
    if exact:
        batches, seed = _enumerate(rows), None
    else:
        if rng is None:
            raise ValueError("Monte-Carlo needs an explicit rng (numpy Generator or "
                             "seed); there is no default and no global seed")
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        seed = {"seed": None if gen is rng else rng, "state": gen.bit_generator.state}
        batches = _monte_carlo(rows, onehot, n_perm, gen)
    slack = _RTOL * np.maximum(1.0, np.abs(observed))
    hits, total, tops = np.zeros(len(observed), dtype=np.int64), 0, []
    for sums in batches:
        tmax = stat(sums).max(1)
        hits += (tmax[:, None] >= observed - slack).sum(0)
        total += len(tmax)
        top = tmax.max()
        tops.append((top, int((tmax >= top - _RTOL * max(1.0, abs(top))).sum())))
    if exact:   # smallest attainable p = share of arrangements tied at the maximum
        best = max(t for t, _ in tops)
        n_best = sum(c for t, c in tops if t >= best - _RTOL * max(1.0, abs(best)))
        p, p_min = hits / total, n_best / total
    else:
        p, p_min = (hits + 1) / (total + 1), 1 / (total + 1)
    return {"p": p, "method": "exact" if exact else "monte-carlo",
            "n_arrangements": total, "p_min": float(p_min), "rng": seed,
            "n_blocks": d["n_blocks"], "blocks_dropped": d["blocks_dropped"]}


def max_t_trend(y, levels, blocks, alternative="decreasing", family="marcus",
                n_perm=N_PERM_DEFAULT, exact=None, rng=None, order=None) -> dict:
    """Max-T multiple-contrast trend test with a permutation reference.

    y, levels, blocks: one entry per run — endpoint, level value, seed index.
    Levels are ordered ascending unless `order` lists them explicitly;
    `alternative` ("decreasing" | "increasing" | "two-sided") is the direction
    of the endpoint ALONG that order. `blocks=None` is the unblocked variant
    for independent groups (pairing broken, e.g. by a hardware mismatch):
    labels are permuted across all units, Monte-Carlo only.

    Statistic: max over the family of t_c = c'm / sqrt(s² Σ c_j²/n_j), m the
    level means and s² the residual mean square of the additive block + level
    model (one-way model if unblocked), recomputed in every arrangement.
    `exact=None` enumerates iff the design is blocked and (k!)^b <= EXACT_MAX;
    otherwise `rng` is required and recorded.

    Returns stat (observed max-T), p (global), argmax and step (the arg-max
    contrast's pooled lower | upper level sets — where the step is), contrasts
    (per contrast: lower, upper, difference = upper mean − lower mean, t, and
    the single-step max-T adjusted p), method, n_arrangements, p_min (smallest
    attainable p), n_blocks, blocks_dropped, rng. See the module docstring for
    the assumption and for what this does not license.
    """
    signs = {"increasing": 1.0, "decreasing": -1.0, "two-sided": None}
    if alternative not in signs:
        raise ValueError(f"alternative must be one of {sorted(signs)}, got {alternative!r}")
    sign = signs[alternative]
    d = _design(y, levels, blocks, order)
    n, ss, df, lv = d["n"], d["ss"], d["df"], d["order"]
    pairs = _pairs(len(lv), family)
    c = contrast_matrix(len(lv), family, n)
    scale = np.sqrt((c ** 2 / n).sum(1))

    def t_stats(sums):
        m = sums / n            # centred data: grand mean 0, so Σ n m² is the level SS
        s2 = np.maximum(ss - (n * m ** 2).sum(-1, keepdims=True), 1e-30 * ss) / df
        return (m @ c.T) / (np.sqrt(s2) * scale)

    def directed(sums):
        t = t_stats(sums)       # large = evidence for the stated alternative
        return np.abs(t) if sign is None else sign * t

    t = t_stats(d["sums"][None])[0]
    observed = directed(d["sums"][None])[0]
    ref = _reference(d, directed, observed, exact, n_perm, rng)
    p_adj = ref.pop("p")
    diff = (c @ d["means"]) / c.clip(min=0).sum(1)
    a = int(np.argmax(observed))
    contrasts = [{"lower": lv[:i + 1].tolist(), "upper": lv[j:].tolist(),
                  "difference": float(diff[r]), "t": float(t[r]), "p_adj": float(p_adj[r])}
                 for r, (i, j) in enumerate(pairs)]
    return {"stat": float(observed[a]), "p": float(p_adj[a]), "argmax": a,
            "step": (contrasts[a]["lower"], contrasts[a]["upper"]),
            "contrasts": contrasts, "family": family, "alternative": alternative,
            "levels": lv.tolist(), **ref}


def _isotonic(m, w):
    """Weighted non-decreasing isotonic regression along the last axis.

    Max-min formula fit_j = max_{s<=j} min_{t>=j} Av(s, t), Av = w-weighted mean
    of levels s..t (Barlow et al. 1972, ch. 1). It is the pool-adjacent-violators
    solution — checked against scikit-learn in the tests — and, unlike the
    pooling loop, vectorises over arrangements.
    """
    k = m.shape[-1]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cm = np.concatenate([np.zeros_like(m[..., :1]), np.cumsum(m * w, axis=-1)], axis=-1)
    av = {(s, t): (cm[..., t + 1] - cm[..., s]) / (cw[t + 1] - cw[s])
          for s in range(k) for t in range(s, k)}
    return np.stack([np.max([np.min([av[s, t] for t in range(j, k)], axis=0)
                             for s in range(j + 1)], axis=0) for j in range(k)], axis=-1)


def _fit_summary(d, increasing) -> dict:
    sign = 1.0 if increasing else -1.0
    m, n, lv = d["means"], d["n"], d["order"]
    fit = sign * _isotonic(sign * m, n)
    grand = np.average(m, weights=n)
    tied = np.isclose(fit[1:], fit[:-1], rtol=0.0, atol=1e-12 * np.ptp(m))
    return {"levels": lv.tolist(), "n": n.tolist(), "means": m.tolist(),
            "fitted": fit.tolist(), "increasing": bool(increasing),
            "pooled": [g.tolist() for g in np.split(lv, np.flatnonzero(~tied) + 1)],
            "explained": float((n * (fit - grand) ** 2).sum() / (n * (m - grand) ** 2).sum())}


def isotonic_fit(y, levels, increasing=False, order=None) -> dict:
    """Pool-adjacent-violators fit of the level means, weights = n per level.

    Returns levels, n, means, fitted (monotone means), pooled (the sets of
    adjacent levels the fit ties together) and explained = isotonic
    between-level sum of squares / raw between-level sum of squares: 1 when the
    means are already monotone in the stated direction, 0 when they run fully
    against it. Descriptive; `isotonic_trend` is the test.
    """
    return _fit_summary(_design(y, levels, None, order, inference=False), increasing)


def isotonic_trend(y, levels, blocks, increasing=False, n_perm=N_PERM_DEFAULT,
                   exact=None, rng=None, order=None) -> dict:
    """Permutation test of the isotonic likelihood-ratio-type statistic Ē².

    Ē² = isotonic between-level SS / total SS about the block means (about the
    grand mean if `blocks=None`) — Barlow et al. 1972; Robertson, Wright &
    Dykstra 1988. The denominator is invariant under relabelling, so the test
    orders arrangements by the isotonic SS alone. Same permutation scheme,
    arguments, assumption and caveats as `max_t_trend`; `fit` is the
    `isotonic_fit` of the retained data. Shape-free: it assumes monotonicity in
    the stated direction and nothing about where the steps are.
    """
    d = _design(y, levels, blocks, order)
    n, ss, sign = d["n"], d["ss"], 1.0 if increasing else -1.0

    def e2(sums):               # isotonic fit keeps the (zero) weighted mean
        return ((n * _isotonic(sign * sums / n, n) ** 2).sum(-1) / ss)[:, None]

    observed = e2(d["sums"][None])[0]
    ref = _reference(d, e2, observed, exact, n_perm, rng)
    return {"stat": float(observed[0]), "p": float(ref.pop("p")[0]),
            "fit": _fit_summary(d, increasing), **ref}
