"""Kish effective sample size and the FPR working-point gate (PLAN §16.2).

N_eff = (Σw)² / Σw², computed at event level AFTER split, matching-category
selection, and reweighting. FPR=1e-4 is eligible iff N_eff ≥ 4e6 (⇒ ~400
expected background survivors, ~5% relative binomial error); otherwise the cell
demotes to 1e-3. Eligibility comes from these numbers, never raw file counts
(prohibition P4).
"""
from __future__ import annotations

import numpy as np

KISH_THRESHOLD_1EM4 = 4e6  # N_eff floor for reporting FPR = 1e-4 (PLAN §16.2)


def n_eff(weights) -> float:
    """Kish effective sample size. Unweighted (all-equal) ⇒ N_eff = N."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s * s / np.square(w).sum())


def n_eff_clustered(weights, event_ids) -> float:
    """Event-clustered N_eff: jets sharing an event UID are one unit.

    Per-event weight = sum of its jet weights; N_eff computed on those event
    weights (the cluster correction of PLAN §16.2/§6-A.5a).
    """
    w = np.asarray(weights, dtype=float)
    ids = np.asarray(event_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    ev_w = np.zeros(len(uniq))
    np.add.at(ev_w, inv, w)
    return n_eff(ev_w)


def kish_gate(weights, fpr_grid=(1e-4, 1e-3), event_ids=None,
              threshold=KISH_THRESHOLD_1EM4) -> dict:
    """Tightest grid FPR whose cell passes the Kish gate (PLAN §12 wp_rule).

    Returns the reported working point plus the N_eff evidence. With event_ids,
    uses the clustered N_eff.
    """
    ne = n_eff_clustered(weights, event_ids) if event_ids is not None else n_eff(weights)
    eligible_1em4 = ne >= threshold
    grid = sorted(fpr_grid)  # tightest = smallest FPR first
    reported = grid[0] if eligible_1em4 else next((f for f in grid if f >= 1e-3), grid[-1])
    # relative binomial error at the reported WP: 1/sqrt(N_eff * p)
    rel_err = float("inf") if ne * reported == 0 else 1.0 / np.sqrt(ne * reported)
    return {
        "n_eff": ne,
        "eligible_1e-4": bool(eligible_1em4),
        "fpr_reported": reported,
        "expected_survivors": ne * reported,
        "rel_binomial_error": rel_err,
    }
