"""Paired ratios of two models' metrics, with a run + test-sample error.

WHY (audit 2026-09-29, B3 and must-fix 5). The paper quoted ratios of seed means
with no error, and called the finite test sample "common to all models". It is
common, and that is exactly why it does not cancel: two models scored on the
same 11,876 background jets still disagree on WHICH jets they get wrong, so the
ratio of their 1 - AUC carries a test-sample error of its own. For b vs c
two-prong 43/162 that error (95 % half-width 0.088 on the log) is six times the
spread over runs (SD 0.015). A ratio is therefore quoted here as

    r = exp( mean_k ln( m_coarse,k / m_fine,k ) )        (paired geometric mean)

over runs k paired by run index, with its run range, and one error that adds

    run term    SD_k(ln r_k) / sqrt(n)                   (which runs were drawn)
    test term   SD_b( mean_k ln r_k^(b) )                 (which test jets were drawn)

in quadrature. The test term is a bootstrap over test jets in which replicate b
applies ONE resampling of the jets to every model, both sides and every run, so
the pairing across models and the correlation across runs are both kept. The
run SD already holds the part of the test-sample noise that is independent
between runs, so the sum is conservative by that part; the audit's B3 table used
the same sum.

HOW. Everything is built on replicate vectors: for one model, element 0 is the
metric on the test sample as it is, and elements 1..B are the metric under
bootstrap weights w^(b) (multinomial counts, `boot_counts`). Because w^(b) is a
function of (n, seed, b) only, two models scored separately -- in different
jobs, on different days -- see the same resampling as long as they were scored
on the same jets in the same order. `paired_ratio` then needs only the vectors.

    replicates(AucScorer(y, s), n, b, seed)       -> 1 - AUC        (probes)
    replicates(MacroAucScorer(y, P), ...)         -> 1 - macro AUC  (fine-tuning)
    replicates(EpsBScorer(y, s, 0.9), ...)        -> eps_B at 90 %  (rejection)
    replicates(SigmaEffScorer(res), ...)          -> sigma_eff      (mass probes)
    paired_ratio({run: vec}, {run: vec})          -> the ratio and its errors

A rejection 1/eps_B at a fixed signal efficiency rests on a COUNT of passing
background jets, so every rejection also gets a Poisson interval on that count
(Garwood, `rejection_interval`).

PAIRING OF v2 RUNS. Two v2 runs are paired at an epoch only if the sha256 of
their realised training stream at that epoch agrees (<run>/stream/epoch-EEE.json,
written by the training code). `paired_ratio(..., run_dirs=...)` refuses a pair
whose streams differ. v1 runs recorded no stream; they are paired by run index
for their shared initialisation only, and the result says so.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
from typing import Callable, Mapping

import numpy as np

B_DEFAULT = 1000
SEED_DEFAULT = 20260929
CL_68 = 0.682689492137086      # the 1-sigma central interval


# ----------------------------------------------------------------- resampling
def boot_counts(n: int, b: int, seed: int = SEED_DEFAULT):
    """Yield B multinomial count vectors over n jets; replicate r depends only on
    (n, seed, r), so separately scored models share it."""
    for r in range(1, b + 1):
        rng = np.random.default_rng([int(seed), int(n), r])
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def replicates(metric: Callable[[np.ndarray | None], float], n: int,
               b: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> np.ndarray:
    """[metric(unweighted), metric(w^(1)), ..., metric(w^(B))]."""
    out = np.empty(b + 1)
    out[0] = metric(None)
    for i, w in enumerate(boot_counts(n, b, seed), start=1):
        out[i] = metric(w)
    return out


# ------------------------------------------------------------ weighted metrics
class AucScorer:
    """Weighted AUC with ties counted one half, the Mann-Whitney form.

    Sorting is done once; each weighted evaluation is O(n). With unit weights it
    equals sklearn's roc_auc_score; with integer weights it equals the AUC of the
    sample in which jet i appears w_i times (tests/test_paired_stats.py).
    """

    def __init__(self, y, s):
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, dtype=np.float64)
        if y.shape != s.shape or y.ndim != 1:
            raise ValueError("y and s must be 1-d and of equal length")
        self.n = y.size
        self.order = np.argsort(s, kind="stable").astype(np.int32)
        ss = s[self.order]
        self.pos = y[self.order].astype(np.float64)
        self.gid = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]).astype(np.int32)
        self.g = int(self.gid[-1]) + 1 if self.n else 0

    def auc(self, w=None) -> float:
        ws = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        p = np.bincount(self.gid, weights=ws * self.pos, minlength=self.g)
        q = np.bincount(self.gid, weights=ws * (1.0 - self.pos), minlength=self.g)
        wp, wq = p.sum(), q.sum()
        if wp <= 0 or wq <= 0:
            return float("nan")
        below = np.cumsum(q) - q
        return float(np.dot(p, below + 0.5 * q) / (wp * wq))

    def resolution(self, w=None) -> float:
        """The smallest non-zero 1 - AUC the (weighted) sample can express: one
        discordant pair, probe.log1m_auc's floor."""
        ws = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        wp = float(np.dot(ws, self.pos))
        wq = float(ws.sum() - wp)
        return 1.0 / max(wp * wq, 1.0)

    def __call__(self, w=None) -> float:
        """1 - AUC, floored at the sample's resolution (a censored value is a bound)."""
        return max(1.0 - self.auc(w), self.resolution(w))


class MacroAucScorer:
    """1 - macro one-vs-rest AUC, exactly as eval_arm.metrics computes
    macro_auc_ovr: over the classes present in `y`, and, when some class of the
    K columns is absent, on the scores of the present classes renormalised to
    sum to one (a renormalisation changes the per-class ranking, so it is kept).
    `P` is (n, K) class probabilities."""

    def __init__(self, y, P):
        y = np.asarray(y)
        P = np.asarray(P, dtype=np.float64)
        present = np.unique(y)
        if present.size < P.shape[1]:
            P = P[:, present]
            P = P / P.sum(axis=1, keepdims=True)
            cols = range(present.size)
        else:
            cols = present
        self.n = y.size
        if present.size == 2:
            self.scorers = [AucScorer(y == present[1], P[:, 1])]
        else:
            self.scorers = [AucScorer(y == c, P[:, j]) for c, j in zip(present, cols)]

    def auc(self, w=None) -> float:
        return float(np.mean([s.auc(w) for s in self.scorers]))

    def __call__(self, w=None) -> float:
        return 1.0 - self.auc(w)


class EpsBScorer:
    """Background efficiency at a fixed signal efficiency, on the ROC by linear
    interpolation, as probe.rejection_at computes it (np.interp over the ROC)."""

    def __init__(self, y, s, eps_s: float):
        if not 0.0 < eps_s < 1.0:
            raise ValueError(f"eps_s must be in (0, 1), got {eps_s}")
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, dtype=np.float64)
        self.n = y.size
        self.eps_s = float(eps_s)
        self.order = np.argsort(-s, kind="stable")
        ss = s[self.order]
        self.pos = y[self.order].astype(np.float64)
        self.gid = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])
        self.g = int(self.gid[-1]) + 1 if self.n else 0

    def roc(self, w=None):
        ws = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        tp = np.cumsum(np.bincount(self.gid, weights=ws * self.pos, minlength=self.g))
        fp = np.cumsum(np.bincount(self.gid, weights=ws * (1.0 - self.pos), minlength=self.g))
        return np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]], float(fp[-1])

    def eps_b(self, w=None) -> float:
        fpr, tpr, _ = self.roc(w)
        return float(np.interp(self.eps_s, tpr, fpr))

    def n_bkg(self, w=None) -> float:
        return self.roc(w)[2]

    def __call__(self, w=None) -> float:
        return self.eps_b(w)


class SigmaEffScorer:
    """Half the smallest interval holding 68 % of the (weighted) residuals --
    mass_resolution.sigma_eff, extended to integer bootstrap weights."""

    def __init__(self, res, frac: float = 0.68):
        res = np.asarray(res, dtype=np.float64)
        self.order = np.argsort(res, kind="stable")
        self.x = res[self.order]
        self.n = self.x.size
        self.frac = frac

    def __call__(self, w=None) -> float:
        if w is None:
            c = np.ones(self.n)
        else:
            c = np.asarray(w, dtype=np.float64)[self.order]
        tot = c.sum()
        k = int(np.ceil(self.frac * tot))   # mass_resolution.sigma_eff's k
        cum = np.cumsum(c)
        start = np.flatnonzero(c > 0)
        before = cum[start] - c[start]
        end = np.searchsorted(cum, before + k - 1e-9, side="left")
        ok = end < self.n
        if not ok.any():
            return float((self.x[-1] - self.x[0]) / 2.0)
        width = self.x[end[ok]] - self.x[start[ok]]
        return float(width.min() / 2.0)


# --------------------------------------------------------------- the ratio
def _stream_ids():
    """experiments/MTX/stream_ids.py (the training-code agent's reader), if present."""
    p = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "MTX" / "stream_ids.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("stream_ids", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_stream(run_dir) -> dict[int, str]:
    """{epoch: sha256} from <run_dir>/stream/epoch-EEE.json; {} for a v1 run."""
    m = _stream_ids()
    if m is not None:
        return dict(m.load_stream(run_dir))
    d = pathlib.Path(run_dir) / "stream"
    out = {}
    for f in sorted(d.glob("epoch-*.json")) if d.is_dir() else []:
        rec = json.loads(f.read_text())
        out[int(rec["epoch"])] = rec["sha256"]
    return out


def stream_pairing(run_dir_a, run_dir_b) -> str:
    """'v1' when neither run recorded its stream; 'identical' when both did and
    every common epoch agrees. Anything else is refused."""
    a, b = load_stream(run_dir_a), load_stream(run_dir_b)
    if not a and not b:
        return "v1"
    if not a or not b:
        raise SystemExit(f"FATAL: {run_dir_a if not a else run_dir_b} recorded no training "
                         "stream while its partner did; they are not a pair")
    m = _stream_ids()
    if m is not None and hasattr(m, "assert_paired"):
        try:
            m.assert_paired(run_dir_a, run_dir_b)
        except AssertionError as e:
            raise SystemExit(f"FATAL: not a pair -- {e}") from None
    if set(a) != set(b):
        raise SystemExit(f"FATAL: {run_dir_a} and {run_dir_b} recorded different epochs "
                         f"({sorted(set(a) ^ set(b))[:5]} ...)")
    bad = [e for e in sorted(a) if a[e] != b[e]]
    if bad:
        raise SystemExit(f"FATAL: {run_dir_a} and {run_dir_b} saw different training "
                         f"streams from epoch {bad[0]} ({len(bad)} of {len(a)} epochs); "
                         "they are not a pair")
    return "identical"


def _t975(dof: float) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, dof)) if math.isfinite(dof) else 1.959963984540054


def paired_ratio(fine: Mapping, coarse: Mapping, *, pairs: Mapping | None = None,
                 run_dirs: Mapping | None = None) -> dict:
    """The paired geometric-mean ratio coarse/fine over runs, and its errors.

    fine, coarse  {run: replicate vector}, vectors from `replicates` with the same
                  n, B and seed (checked by length only; the caller keys the jets).
    pairs         {coarse run: fine run}; default: the runs both sides share.
    run_dirs      {run: run directory}; when given, every pair must pass
                  `stream_pairing` (v2 runs with different realised streams are
                  refused).
    """
    if pairs is None:
        pairs = {k: k for k in coarse if k in fine}
    pairs = dict(pairs)
    if not pairs:
        raise SystemExit("FATAL: no paired runs")
    lengths = {len(np.asarray(v)) for v in list(fine.values()) + list(coarse.values())}
    if len(lengths) != 1:
        raise SystemExit(f"FATAL: replicate vectors of different lengths {sorted(lengths)}")
    pairing = "unchecked"
    if run_dirs is not None:
        kinds = {stream_pairing(run_dirs[f], run_dirs[c]) for c, f in pairs.items()}
        pairing = kinds.pop() if len(kinds) == 1 else "mixed"
    ln = np.array([np.log(np.asarray(coarse[c], float)) - np.log(np.asarray(fine[f], float))
                   for c, f in pairs.items()])            # runs x (1 + B)
    if not np.all(np.isfinite(ln)):
        raise SystemExit("FATAL: a metric is zero or negative; its log ratio is undefined")
    point = ln[:, 0]
    n = point.size
    mean = float(point.mean())
    reps = ln[:, 1:].mean(axis=0)
    test_se = float(reps.std(ddof=1)) if reps.size > 1 else float("nan")
    run_sd = float(point.std(ddof=1)) if n > 1 else float("nan")
    run_se = run_sd / math.sqrt(n) if n > 1 else float("nan")
    if n > 1:
        comb = math.sqrt(run_se ** 2 + test_se ** 2)
        dof = comb ** 4 / (run_se ** 4 / (n - 1)) if run_se > 0 else float("inf")
    else:
        comb, dof = test_se, float("inf")
    t = _t975(dof)
    fm = np.array([np.asarray(fine[f], float)[0] for f in pairs.values()])
    cm = np.array([np.asarray(coarse[c], float)[0] for c in pairs])
    lo_b, hi_b = np.quantile(reps, [0.025, 0.975]) if reps.size > 1 else (math.nan, math.nan)
    return {
        "pairs": {str(c): str(f) for c, f in pairs.items()},
        "stream_pairing": pairing,
        "n_runs": n,
        "per_run_ratio": [float(math.exp(x)) for x in point],
        "ratio": math.exp(mean),
        "ln_ratio": mean,
        "run_range": [float(math.exp(point.min())), float(math.exp(point.max()))],
        "ln_run_sd": run_sd,
        "ln_run_se": run_se,
        "ln_test_se": test_se,
        "ln_combined_se": comb,
        "dof": dof,
        "ci95": [math.exp(mean - t * comb), math.exp(mean + t * comb)],
        "ci95_test_only_percentile": [float(math.exp(lo_b)), float(math.exp(hi_b))],
        "z": mean / comb if comb > 0 else float("inf"),
        "ratio_of_means": float(cm.mean() / fm.mean()),
        "n_boot": int(ln.shape[1] - 1),
    }


# -------------------------------------------------------------- rejections
def poisson_interval(k: float, cl: float = CL_68) -> tuple[float, float]:
    """Garwood's central interval on a Poisson mean given k observed counts."""
    from scipy.stats import chi2
    a = 1.0 - cl
    lo = 0.0 if k <= 0 else float(chi2.ppf(a / 2, 2 * k) / 2)
    hi = float(chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2)
    return lo, hi


def rejection_interval(n_bkg: float, k_pass: float, cl: float = CL_68) -> dict:
    """1/eps_B = n_bkg / k with the Garwood interval on k. k = 0 is a bound:
    the rejection is above n_bkg / k_hi, and no upper end exists."""
    lo, hi = poisson_interval(k_pass, cl)
    return {"n_bkg": float(n_bkg), "n_bkg_pass": float(k_pass), "cl": cl,
            "rejection": float(n_bkg / k_pass) if k_pass > 0 else float("inf"),
            "interval": [float(n_bkg / hi), float(n_bkg / lo) if lo > 0 else float("inf")],
            "is_bound": bool(k_pass <= 0)}
