"""Paired superiority + TOST equivalence + Holm multiplicity (PLAN §16.3).

Primary decision (P0): two-sided paired t (α=0.05) AND TOST equivalence
(α=0.05 ⇔ 90% CI within ±ln(1.1)). "No effect" language is permitted only on a
TOST pass; otherwise "inconclusive at the pre-registered MEI". Secondaries
S1–S8: Holm at family α=0.05 over exactly eight p-values.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .mde import MEI_LOG


def paired_t(diffs, alpha=0.05) -> dict:
    """Two-sided one-sample t on paired differences d_s (H0: mean 0)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    res = stats.ttest_1samp(d, 0.0)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(1 - alpha, n - 1, loc=d.mean(), scale=se)
    return {"mean": float(d.mean()), "t": float(res.statistic), "p": float(res.pvalue),
            "reject_null": bool(res.pvalue < alpha), "ci": (float(lo), float(hi))}


def tost(diffs, bound=MEI_LOG, alpha=0.05) -> dict:
    """Two one-sided tests for equivalence within ±bound.

    Equivalent iff the (1-2α) CI ⊂ (-bound, +bound), the CI-duality of TOST;
    p = max of the two one-sided p-values.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    df = n - 1
    p_lower = stats.t.sf((mean + bound) / se, df)   # H0: mean <= -bound
    p_upper = stats.t.cdf((mean - bound) / se, df)   # H0: mean >= +bound
    p = float(max(p_lower, p_upper))
    lo, hi = stats.t.interval(1 - 2 * alpha, df, loc=mean, scale=se)  # (1-2α) CI
    return {"equivalent": bool(-bound < lo and hi < bound), "p": p,
            "ci_1m2a": (float(lo), float(hi)), "bound": float(bound)}


def holm(pvalues, alpha=0.05) -> np.ndarray:
    """Holm–Bonferroni step-down. Returns a boolean reject mask in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: first failure stops all larger p-values
    return reject
