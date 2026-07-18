"""Minimum detectable effect for the paired-seed design (PLAN §16.4).

MDE_total = sqrt( [ (t_{0.975,n-1} + t_{0.80,n-1}) · σ̂_d / sqrt(n) ]² + σ_test,Δ² )

σ̂_d = paired-seed SD of d_s (log-rejection); n fixed per §9; σ_test,Δ = the
test-resampling floor from §6-A.5(b). The seed-term multiplier is 3.10 at n=3
and 1.66 at n=5 (the plan's reference values — used as the analytic unit test).
MEI = 10% relative rejection ⇔ ln(1.1) = 0.09531.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

MEI_LOG = float(np.log(1.1))  # 0.09531 — 10% relative rejection (PLAN §16.4)


def seed_term_multiplier(n: int, alpha=0.05, power=0.80) -> float:
    """(t_{1-α/2, n-1} + t_{power, n-1}) / sqrt(n). 3.10 at n=3, 1.66 at n=5."""
    df = n - 1
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) / np.sqrt(n))


def mde_total(sigma_d: float, n: int, sigma_test: float = 0.0,
              alpha=0.05, power=0.80) -> float:
    """Total MDE (seed term ⊕ test floor, added in quadrature)."""
    seed = seed_term_multiplier(n, alpha, power) * sigma_d
    return float(np.hypot(seed, sigma_test))


def sigma_d_ceiling_for_mei(n: int, mei=MEI_LOG, alpha=0.05, power=0.80) -> float:
    """Largest σ̂_d for which the seed term alone covers the MEI at this n.

    At n=5 this is ≈5.7% (PLAN §16.4), i.e. σ_d ≲ 0.057 in log-rejection.
    """
    return float(mei / seed_term_multiplier(n, alpha, power))
