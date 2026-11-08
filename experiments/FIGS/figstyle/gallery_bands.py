#!/usr/bin/env python3
"""The gallery's uncertainty arithmetic, kept free of data loading so tests can import it: the relative uncertainty of a
mean rejection curve (F6, F6-app ratio panels), the band of the spread over runs (F10b) and the soft-drop reference's
off-axis rule (F9b, A17)."""
from __future__ import annotations

import numpy as np


def rejection_rel_error(rej, n_pass, n_background: int) -> np.ndarray:
    """Relative uncertainty of the mean over runs of a rejection 1/eps_B, per efficiency point: the SD over runs
    (ddof 1; none for one run) over the mean, and the 68% binomial uncertainty of eps_B = k/N for the mean passing
    count k, sqrt((1 - k/N) / k) (normal approximation; drawn points have k >= 20), in quadrature. The binomial term is
    not divided by the number of runs: every run is scored on the same test jets. rej, n_pass: (runs, points)."""
    rej, k = np.atleast_2d(np.asarray(rej, float)), np.atleast_2d(np.asarray(n_pass, float)).mean(axis=0)
    mean = rej.mean(axis=0)
    seed = rej.std(axis=0, ddof=1) / mean if len(rej) > 1 else np.zeros_like(mean)
    binom = np.sqrt((1 - k / n_background) / k)
    return np.sqrt(seed ** 2 + binom ** 2)


def extend_to(lo: float, hi: float, ref, keep: float = 0.5) -> tuple | None:
    """(lo, hi) widened to hold every finite reference value, when the plotted data's span [lo, hi] still takes at
    least keep of the widened axis (the differences between the series stay readable); None when it would not."""
    r = np.asarray(ref, float)
    r = r[np.isfinite(r)]
    a, b = min(lo, r.min(initial=lo)), max(hi, r.max(initial=hi))
    return (a, b) if hi - lo >= keep * (b - a) else None


def spread_band(per) -> tuple | None:
    """(lower, upper) of the spread over runs at each point, per (runs, points): mean -+ the SD (ddof 1) over three or
    more runs, the range over two (F2a's rule); None for one run (F10b, F10b-flav, F10-ft)."""
    per = np.asarray(per, float)
    if len(per) > 2:
        mean, sd = per.mean(axis=0), per.std(axis=0, ddof=1)
        return mean - sd, mean + sd
    return (per.min(axis=0), per.max(axis=0)) if len(per) == 2 else None
