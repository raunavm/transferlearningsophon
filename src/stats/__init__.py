"""Statistical procedures for the granularity/pileup analysis (PLAN §16).

Every public function here is unit-tested against an analytic case in
`tests/test_stats.py` before first confirmatory use (AGENT §8).
"""
from .bootstrap import ci, event_bootstrap, paired_bootstrap_diff
from .inference import holm, paired_t, tost
from .kish import kish_gate, n_eff, n_eff_clustered
from .mde import MEI_LOG, mde_total, seed_term_multiplier, sigma_d_ceiling_for_mei

__all__ = [
    "ci", "event_bootstrap", "paired_bootstrap_diff",
    "holm", "paired_t", "tost",
    "kish_gate", "n_eff", "n_eff_clustered",
    "MEI_LOG", "mde_total", "seed_term_multiplier", "sigma_d_ceiling_for_mei",
]
