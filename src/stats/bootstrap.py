"""Event-level bootstrap (PLAN §16.1).

Resampling unit = hard-scatter event UID: all jets of an event, all cross-μ
twins, and LHCO jet pairs move together. B=2000, RNG seed 7777 by default,
per-analysis stream ids recorded by the caller. Joint cross-μ resampling is
obtained by passing the shared event UID for every μ level so twin rows are
drawn together.
"""
from __future__ import annotations

import numpy as np

B_DEFAULT = 2000
SEED_DEFAULT = 7777


def _event_groups(event_ids):
    """Row-index groups, one per unique event UID (stable order)."""
    ids = np.asarray(event_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(uniq))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def event_bootstrap(event_ids, statistic, b=B_DEFAULT, seed=SEED_DEFAULT):
    """Bootstrap distribution of `statistic(row_indices)` resampling by event.

    `statistic` maps an array of original row indices to a scalar. Events are
    drawn with replacement; all rows of a drawn event are included together.
    Deterministic for a fixed seed.
    """
    ids = np.asarray(event_ids)
    singleton = len(np.unique(ids)) == len(ids)
    groups = None if singleton else _event_groups(ids)
    n_ev = len(ids) if singleton else len(groups)
    rng = np.random.default_rng(seed)
    out = np.empty(b, dtype=float)
    for i in range(b):
        pick = rng.integers(0, n_ev, n_ev)
        if singleton:
            # every event is one row: groups would be singletons in sorted-id
            # order, so concatenating groups[pick] IS pick (identical draws,
            # identical statistic) — skip the per-replica list concat.
            rows = pick
        else:
            rows = np.concatenate([groups[k] for k in pick])
        out[i] = statistic(rows)
    return out


def paired_bootstrap_diff(event_ids, values_a, values_b, reduce=np.mean,
                          b=B_DEFAULT, seed=SEED_DEFAULT):
    """Bootstrap of reduce(A) − reduce(B) on the SAME resampled events.

    values_a/values_b are per-row arrays scored on identical rows in identical
    order (the paired-bootstrap contract). Returns (point_estimate, dist).
    """
    a = np.asarray(values_a, dtype=float)
    b_ = np.asarray(values_b, dtype=float)
    point = float(reduce(a) - reduce(b_))
    dist = event_bootstrap(event_ids, lambda r: reduce(a[r]) - reduce(b_[r]),
                           b=b, seed=seed)
    return point, dist


def ci(dist, alpha=0.05):
    """Percentile CI at level 1-alpha."""
    d = np.asarray(dist, dtype=float)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
