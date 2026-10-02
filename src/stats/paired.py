"""Paired ratios of two models' metrics, with a run + test-sample error.

WHY (audit 2026-09-29, B3 and must-fix 5). The paper quoted ratios of seed means
with no error, and called the finite test sample "common to all models". It is
common, and that is exactly why it does not cancel: two models scored on the
same 11,876 background jets still disagree on WHICH jets they get wrong, so the
ratio of their 1 - AUC carries a test-sample error of its own. For b vs c
two-prong 43/162 a single run's test-sample SD of ln r is 0.045 (95 % half-width
0.088), three times the SD of ln r over the five runs (0.015). That part of the
test noise is almost independent between runs (correlation 0.06), so the test
error of the five-run MEAN is 0.022, half the single-run value. A ratio is
therefore quoted here as

    r = exp( mean_k ln( m_coarse,k / m_fine,k ) )        (paired geometric mean)

over runs k paired by run index, with its run range and one error. Write a run's
log ratio on test sample T as ln r_k(T) = mu + a_k + e(T) + e_k(T): a_k the run
(which runs were drawn), e the test noise every run shares, e_k the test noise
of run k alone. Then

    var(mean_k ln r_k) = (var a + var e_k) / n + var e .

The bootstrap over test jets, in which replicate b applies ONE resampling of the
jets to every model, both sides and every run, measures the test terms: across
runs, the covariance of ln r_k^(b) is var e off the diagonal and var e + var e_k
on it, so

    v_shared = mean off-diagonal covariance        (clipped at 0)
    v_ind    = mean diagonal - v_shared

and the test SE of the mean is measured directly, test_se^2 = v_ind / n +
v_shared. s^2 = SD_k(ln r_k)^2 estimates var a + var e_k, so the run variance
beyond the test noise is v_run = max(s^2 - v_ind, 0) (it cannot be negative),
and the error is

    var(mean) = v_run / n + test_se^2  =  max(s^2, v_ind) / n + v_shared .

Adding the run SD and the test SE of the mean in quadrature, as the audit's B3
table and this module did until 2026-10-01, counts var e_k / n twice. With one
run the error is the full test variance of that run.

The 95 % interval is Student t at the Welch-Satterthwaite degrees of freedom,
the observed spread s^2 / n carrying n - 1 and the bootstrap terms none.

ANY CONTRAST w . y over units u in groups (`combined_error`): the runs of one
model for a paired ratio; the runs of two vocabularies, or the partitions that
merge a pair and those that split it, for a two-group comparison. Then

    var = sum(w_u^2) max(s^2 - v_ind, 0) + w' C w ,

s^2 the units' spread about their own group's mean, pooled (or, `separate`,
each group's own with a Welch-Satterthwaite t: groups that need not vary alike,
five runs against three in A13); C the bootstrap covariance among the units, so
w' C w is the contrast's own test variance (ln_test_se^2), whatever test noise
the units share; v_ind the part of s^2 the
test noise explains, taken from the covariances WITHIN each group. Units of one
group can share more test noise than units of different groups -- runs of one
vocabulary get the same jets wrong more often than runs of two (within-group
minus cross-group covariance 0.09 of the diagonal at the median over the v1
probe cells) -- so no term may assume one shared covariance for all units: a
difference of two groups keeps 2 (cov_within - cov_across) of test variance,
which an exchangeable v_shared cancels (2026-10-01: the combined error then fell
below the contrast's own test SE in 99 of 124 v1 two-group cells, to 0.42 of
it). For one group this is the formula above; the error is never below the
contrast's test SE. `fixed_effects` applies the same split to a regression,
with a run variance and degrees of freedom of each stratum's own.

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

PAIRING OF v2 RUNS (amendment A7). Two v2 runs are a pair for a checkpoint only
if the sha256 of their realised training stream (<run>/stream/epoch-EEE.json,
written by the training code) agrees at every epoch that checkpoint was trained
on: up to the later of the two selected epochs within 70-79
(best_window_epoch.json) for `best70`, the primary (A14); up to the later of the
two global best-validation epochs (best_epoch.json) for `bestval`; up to epoch 79
for `wavg`, the average of epochs 70-79; up to the latest of them for a
comparison between checkpoints (A8). With
`run_dirs`, every directory and its stream records must exist; a pair that
fails is excluded, reported with its first differing epoch, and the ratio is
formed from the others. Without `run_dirs` (v1, which recorded no stream) runs
pair by index for their shared initialisation only, and the result says
"unchecked".

SMALL SAMPLES (A14, v2 only: `small_sample_rule`). With two runs the spread
between runs has one degree of freedom, and the error above can fall to the
test variance alone when the two runs happen to agree (v_run clipped at 0), with
a large Welch-Satterthwaite degrees of freedom: a true null was excluded 7-11 %
of the time at a test SD of 0.1-0.7 times the run SD. So a spread with one
degree of freedom gives var >= sum(w^2) s^2 and the interval takes Student t at
1 degree of freedom (n - 1); and a paired contrast left with one run pair is not
computed. In a one-group contrast the error already exceeds s^2 / n unless the
shared test covariance is negative, so the degrees of freedom are what change
the coverage (src/stats/tests/test_paired.py). The rule is conservative: under
the null it excluded 0 in at most 2.5 % of the trials of that test, and in none
once each run's own test SD is 0.3 of the run SD or more, so a result formed
under it carries two_run_rule and is counted apart from the 5 % reference. The
v1 results are formed without the rule and do not change.
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
        """Mean over the classes present in the (resampled) sample: a rare class
        can be absent from a resampling, whose macro AUC then averages over the
        rest, as eval_arm.metrics does for a sample without it. (The present-class
        renormalisation is kept as on the full sample; the difference is one
        class's probability mass in a class absent from the resampling.)"""
        return float(np.nanmean([s.auc(w) for s in self.scorers]))

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


WAVG_EPOCHS = range(70, 80)     # net_wavg70-79_state.pt, written by experiments/MTX/pretrain_v2.py


SELECTED_EPOCH = {"best70": ("best_window_epoch.json", "selected epoch within 70-79"),
                  "bestval": ("best_epoch.json", "best-validation epoch")}


def checkpoint_epoch(run_dir, checkpoint: str) -> int:
    """The last training epoch a v2 checkpoint was trained on: the selected epoch
    within 70-79 (best_window_epoch.json) for 'best70', the global best-validation
    epoch (best_epoch.json) for 'bestval', the last averaged epoch for 'wavg'."""
    if checkpoint == "wavg":
        return WAVG_EPOCHS[-1]
    if checkpoint in SELECTED_EPOCH:
        name, what = SELECTED_EPOCH[checkpoint]
        f = pathlib.Path(run_dir) / name
        if not f.exists():
            raise SystemExit(f"FATAL: {f} does not exist; the {what} is unknown")
        return int(json.loads(f.read_text())["epoch"])
    raise SystemExit(f"FATAL: unknown checkpoint {checkpoint!r} (best70, bestval or wavg)")


def stream_check(run_dirs, checkpoint) -> dict | None:
    """None when the runs in `run_dirs` drew the same training stream at every
    epoch up to the latest epoch any of their `checkpoint`s was trained on, so
    that their checkpoints differ only in what the runs differ in. `checkpoint`
    is one checkpoint or several (a comparison between checkpoints, A8).
    Otherwise {first_bad_epoch, upto_epoch, runs}: the first epoch whose sha256
    differs or that a run has not recorded. A directory that does not exist, or
    holds no stream record, is fatal: that is a wrong path, not a pair that
    failed."""
    cks = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    dirs = list(dict.fromkeys(pathlib.Path(d) for d in run_dirs))
    streams = []
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"FATAL: run directory {d} does not exist")
        s = load_stream(d)
        if not s:
            raise SystemExit(f"FATAL: {d} holds no stream record (stream/epoch-EEE.json); "
                             "a v2 pair cannot be checked without one")
        streams.append(s)
    upto = max(checkpoint_epoch(d, c) for d in dirs for c in cks)
    for e in range(upto + 1):
        if streams[0].get(e) is None or len({s.get(e) for s in streams}) > 1:
            return {"first_bad_epoch": e, "upto_epoch": upto, "runs": [str(d) for d in dirs]}
    return None


def _stream_filter(keys, run_dirs: Mapping | None, checkpoint: str | None):
    """(kept keys, excluded records); every key kept when run_dirs is None."""
    if run_dirs is None:
        return list(keys), None
    if checkpoint is None:
        raise SystemExit("FATAL: a stream check needs the checkpoint compared (best70, bestval or wavg)")
    kept, excluded = [], []
    for k in keys:
        bad = stream_check(run_dirs[k], checkpoint)
        if bad:
            excluded.append((k, bad))
        else:
            kept.append(k)
    return kept, excluded


def _dirs(x) -> list:
    return [x] if isinstance(x, (str, pathlib.PurePath)) else list(x)


def _t975(dof: float) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, dof)) if math.isfinite(dof) else 1.959963984540054


def bounds(est: float, se: float, dof: float, level: float = 0.95) -> tuple[float, float]:
    """The central `level` Student-t interval est -/+ t se (normal at infinite dof)."""
    from scipy.stats import norm, t
    q = 0.5 + level / 2
    k = float(t.ppf(q, dof)) if math.isfinite(dof) else float(norm.ppf(q))
    return est - k * se, est + k * se


# ------------------------------------------------------------------ the error
def _small_sample_floor(terms: list, observed: list, rule: bool) -> bool:
    """A14's two-run rule for the groups (separate) or pools of a Welch error: a group
    whose spread has one degree of freedom keeps its term at least its observed term
    (sum w^2) s^2. Changes `terms` in place; True when any group was floored, and the
    caller then takes dof = min(dof, 1)."""
    if not rule:
        return False
    small = [i for i, (d, _) in enumerate(observed) if d == 1]
    for i in small:
        terms[i] = max(terms[i], observed[i][1])
    return bool(small)


def combined_error(ln, w, groups=None, separate: bool = False, pools=None,
                   small_sample_rule: bool = False) -> dict:
    """The contrast w . ln[:, 0] over units and its error, by the decomposition in
    the module docstring.

    ln        units x (1 + B) log metrics: element 0 on the test sample as it is,
              1..B under the shared resamplings
    w         the contrast's weight on each unit
    groups    lists of unit indices; the observed spread s^2 is the units' spread
              about their own group's mean, pooled, with n - len(groups) degrees
              of freedom. Default: one group of every unit.
    separate  each group keeps its own spread (Welch): for groups that need not
              share a run variance, such as runs that leave a family out against
              their parent's, five against three (A13). Every group needs two
              units.

    var = sum(w^2) v_run + ln_test_se^2, v_run = max(s^2 - v_ind, 0), with
    ln_test_se^2 = w' C w the contrast's bootstrap variance. v_ind is pooled over
    the groups with their degrees of freedom: per group of k units the mean
    diagonal of C less its mean off-diagonal clipped at 0, the expected test part
    of that group's spread. v_shared(_unclipped) is the mean off-diagonal over
    pairs of units in one group. With one unit the error is that unit's test
    error. With B = 0 (no resampling) the test noise each unit has alone is
    inside s^2 and nothing is subtracted; noise that units share is not measured
    (v_ind = 0, ln_test_se NaN).

    separate: var = sum_g (sum_{u in g} w_u^2) max(s_g^2 - v_ind,g, 0) +
    ln_test_se^2, Student t at the Welch-Satterthwaite degrees of freedom with
    group g's observed term (sum w_u^2) s_g^2 carrying k_g - 1; ln_spread_sd,
    v_ind and v_run are then lists, one per group. A pooled spread with five
    runs against three, the three's run variance three times the five's, covered
    0.91 (verification of 2026-10-01).

    pools     lists of group indices (A14, P1): the spread is taken about each
              group's own mean and pooled within each pool, and each pool keeps
              its own run variance (Welch). For the random partitions: a group is
              one partition's runs, a pool the partitions that merge a pair (or
              those that split it), so each side's run variance comes from the
              runs that replicate one partition. Each pool needs one group of two
              units. ln_spread_sd, v_ind, v_run and pool_dof are lists, one per
              pool; with one group per pool this is `separate`.

    small_sample_rule  (A14) when the pooled spread has one degree of freedom
              (two runs), var >= sum(w^2) s^2 and dof = 1, and the result
              carries two_run_rule: the module docstring. With `separate` or
              `pools`, the same holds per group (pool) whose spread has one degree
              of freedom: its term is at least its observed term, and dof <= 1."""
    L = np.atleast_2d(np.asarray(ln, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64)
    n, nb = L.shape[0], L.shape[1] - 1
    est = float(w @ L[:, 0])
    test_var = float((w @ L[:, 1:]).var(ddof=1)) if nb > 1 else math.nan
    out = {"estimate": est, "ln_test_se": math.sqrt(test_var)}
    if n == 1:
        if nb < 2:
            return {**out, "not_computed": "one unit and no resampling: no error can be formed"}
        return {**out, "ln_spread_sd": math.nan, "v_shared": math.nan,
                "v_shared_unclipped": math.nan, "v_ind": math.nan, "v_run": math.nan,
                "ln_combined_se": out["ln_test_se"], "dof": math.inf}
    groups = [list(range(n))] if groups is None else [list(g) for g in groups]
    dof_s = n - len(groups)
    if dof_s < 1:
        return {**out, "not_computed": "no group holds two units, so the spread between "
                                       "runs cannot be measured"}
    if separate and min(len(g) for g in groups) < 2:
        return {**out, "not_computed": "a group holds one unit, so its own spread cannot "
                                       "be measured"}
    pt = L[:, 0]
    C = np.cov(L[:, 1:]).reshape(n, n) if nb > 1 else None
    per = []                       # (k, sum of squares about the mean, v_ind,g, off-diagonal, sum w^2)
    for g in groups:
        k = len(g)
        ss = float(((pt[g] - pt[g].mean()) ** 2).sum())
        vig, og = 0.0, math.nan
        if C is not None and k > 1:
            Cg = C[np.ix_(g, g)]
            d = float(np.trace(Cg)) / k
            og = (float(Cg.sum()) - k * d) / (k * (k - 1))
            vig = d - max(og, 0.0)
        per.append((k, ss, vig, og, float((w[g] ** 2).sum())))
    multi = [x for x in per if x[0] > 1]
    off = (sum(k * (k - 1) * o for k, _, _, o, _ in multi) / sum(k * (k - 1) for k, *_ in multi)
           if C is not None else math.nan)
    tv = test_var if nb > 1 else 0.0
    shared = {"v_shared": max(off, 0.0) if nb > 1 else math.nan, "v_shared_unclipped": off}
    if pools is not None:
        pool = []                  # (degrees of freedom, s^2, v_ind, sum w^2)
        for p in pools:
            d = sum(per[i][0] - 1 for i in p)
            if d < 1:
                return {**out, "not_computed": "a pool holds no group of two units, so its "
                                               "run variance cannot be measured"}
            pool.append((d, sum(per[i][1] for i in p) / d,
                         sum((per[i][0] - 1) * per[i][2] for i in p) / d,
                         sum(per[i][4] for i in p)))
        v_run = [max(s2 - vi, 0.0) for _, s2, vi, _ in pool]
        terms = [c * vr for (*_, c), vr in zip(pool, v_run)]
        small = _small_sample_floor(terms, [(d, c * s2) for d, s2, _, c in pool], small_sample_rule)
        var = sum(terms) + tv
        obs2 = sum((c * s2) ** 2 / d for d, s2, _, c in pool)
        dof = var ** 2 / obs2 if obs2 > 0 else math.inf
        return {**out, "ln_spread_sd": [math.sqrt(x[1]) for x in pool], **shared,
                "v_ind": [x[2] for x in pool], "v_run": v_run, "pool_dof": [x[0] for x in pool],
                "ln_combined_se": math.sqrt(var), "dof": min(dof, 1.0) if small else dof,
                **({"two_run_rule": True} if small else {})}
    if separate:
        s2g = [ss / (k - 1) for k, ss, *_ in per]
        v_run = [max(x - vig, 0.0) for x, (_, _, vig, _, _) in zip(s2g, per)]
        terms = [cg * vr for (*_, cg), vr in zip(per, v_run)]
        small = _small_sample_floor(terms, [(k - 1, cg * x) for x, (k, *_, cg) in zip(s2g, per)],
                                    small_sample_rule)
        var = sum(terms) + tv
        obs2 = sum((cg * x) ** 2 / (k - 1) for x, (k, _, _, _, cg) in zip(s2g, per))
        dof = var ** 2 / obs2 if obs2 > 0 else math.inf
        return {**out, "ln_spread_sd": [math.sqrt(x) for x in s2g], **shared,
                "v_ind": [x[2] for x in per], "v_run": v_run,
                "ln_combined_se": math.sqrt(var), "dof": min(dof, 1.0) if small else dof,
                **({"two_run_rule": True} if small else {})}
    s2 = sum(ss for _, ss, *_ in per) / dof_s
    v_ind = sum((k - 1) * vig for k, _, vig, _, _ in per) / dof_s
    v_run = max(s2 - v_ind, 0.0)
    c = float((w ** 2).sum())
    var = c * v_run + tv
    obs = c * s2
    if small_sample_rule and dof_s == 1:
        var = max(var, obs)
        return {**out, "ln_spread_sd": math.sqrt(s2), **shared, "v_ind": v_ind, "v_run": v_run,
                "ln_combined_se": math.sqrt(var), "dof": 1.0, "two_run_rule": True}
    return {**out, "ln_spread_sd": math.sqrt(s2), **shared, "v_ind": v_ind, "v_run": v_run,
            "ln_combined_se": math.sqrt(var),
            "dof": var ** 2 / (obs ** 2 / dof_s) if obs > 0 else math.inf}


def contrast(ln, w, groups=None, separate: bool = False, pools=None,
             small_sample_rule: bool = False) -> dict:
    """exp of the contrast with its error, 95 % interval and test-only percentile
    interval: the fields every ratio row carries."""
    L = np.atleast_2d(np.asarray(ln, dtype=np.float64))
    if not np.all(np.isfinite(L)):
        raise SystemExit("FATAL: a metric is zero or negative; its log is undefined")
    e = combined_error(L, w, groups, separate, pools, small_sample_rule)
    if "not_computed" in e:
        return {"ln_test_se": e["ln_test_se"], "not_computed": e["not_computed"]}
    mean, comb, t = e["estimate"], e["ln_combined_se"], _t975(e["dof"])
    reps = np.asarray(w, dtype=np.float64) @ L[:, 1:]
    lo_b, hi_b = np.quantile(reps, [0.025, 0.975]) if reps.size > 1 else (math.nan, math.nan)
    return {"ratio": math.exp(mean), "ln_ratio": mean,
            **{k: e[k] for k in ("ln_test_se", "v_shared", "v_ind", "v_shared_unclipped",
                                 "v_run", "pool_dof", "ln_combined_se", "dof", "two_run_rule")
               if k in e},
            "ci95": [math.exp(mean - t * comb), math.exp(mean + t * comb)],
            "ci95_test_only_percentile": [float(math.exp(lo_b)), float(math.exp(hi_b))],
            "z": mean / comb if comb > 0 else math.inf,
            "n_boot": int(L.shape[1] - 1)}


ONE_PAIR = ("one run pair (after the stream check): a paired contrast needs two, and is "
            "not computed from one (A14)")


def paired_log(ln: Mapping, *, run_dirs: Mapping | None = None,
               checkpoint: str | None = None, small_sample_rule: bool = False) -> dict:
    """The paired mean over runs of a per-run log contrast (for a ratio,
    ln m_coarse,k - ln m_fine,k; for a ratio of ratios, the difference of two),
    and its errors.

    ln          {run: replicate vector of that run's log contrast}
    run_dirs    {run: the run directories the contrast reads} (v2): they must
                share their training stream up to `checkpoint` (stream_check); a
                run that does not is excluded, reported, and the rest are used.
    small_sample_rule  (A14) one run pair is not computed; two take the floor
                and the one degree of freedom of combined_error."""
    if not ln:
        raise SystemExit("FATAL: no paired runs")
    if len({len(np.asarray(v)) for v in ln.values()}) != 1:
        raise SystemExit("FATAL: replicate vectors of different lengths")
    keys, excluded = _stream_filter(list(ln), run_dirs, checkpoint)
    out = {"stream_pairing": "unchecked" if run_dirs is None else "identical" if keys else "differs"}
    if excluded is not None:
        out["excluded_runs"] = [{"run": str(k), **bad} for k, bad in excluded]
    if not keys:
        return {**out, "n_runs": 0, "not_computed": "every run failed the stream check (A7)"}
    if small_sample_rule and len(keys) < 2:
        return {**out, "n_runs": len(keys), "not_computed": ONE_PAIR}
    L = np.array([np.asarray(ln[k], dtype=np.float64) for k in keys])
    n = len(keys)
    point = L[:, 0]
    run_sd = float(point.std(ddof=1)) if n > 1 else math.nan
    return {**out, "n_runs": n,
            "per_run_ratio": [float(math.exp(x)) for x in point],
            "run_range": [float(math.exp(point.min())), float(math.exp(point.max()))],
            "ln_run_sd": run_sd,
            "ln_run_se": run_sd / math.sqrt(n) if n > 1 else math.nan,
            **contrast(L, np.full(n, 1.0 / n), small_sample_rule=small_sample_rule)}


def paired_ratio(fine: Mapping, coarse: Mapping, *, pairs: Mapping | None = None,
                 run_dirs: Mapping | None = None, checkpoint: str | None = None,
                 small_sample_rule: bool = False) -> dict:
    """The paired geometric-mean ratio coarse/fine over runs, and its errors.

    fine, coarse  {run: replicate vector}, vectors from `replicates` with the same
                  n, B and seed (checked by length only; the caller keys the jets).
    pairs         {coarse run: fine run}; default: the runs both sides share.
    run_dirs      {run: run directory, or a list of them} (v2). Each pair's
                  directories must share their training stream up to
                  `checkpoint` ('best70', 'bestval' or 'wavg'); a pair that does
                  not is excluded and listed in `excluded_pairs` with its first
                  differing epoch, and the ratio is formed from the others.
    small_sample_rule  as for paired_log (A14).
    """
    if pairs is None:
        pairs = {k: k for k in coarse if k in fine}
    pairs = dict(pairs)
    if not pairs:
        raise SystemExit("FATAL: no paired runs")
    lengths = {len(np.asarray(v)) for v in list(fine.values()) + list(coarse.values())}
    if len(lengths) != 1:
        raise SystemExit(f"FATAL: replicate vectors of different lengths {sorted(lengths)}")
    dirs = (None if run_dirs is None else
            {c: _dirs(run_dirs[c]) + _dirs(run_dirs[f]) for c, f in pairs.items()})
    kept, excluded = _stream_filter(list(pairs), dirs, checkpoint)
    out = {"pairs": {str(c): str(pairs[c]) for c in kept}}
    if excluded is not None:
        out["excluded_pairs"] = [{"pair": [str(c), str(pairs[c])], **bad} for c, bad in excluded]
    if not kept:
        return {**out, "stream_pairing": "differs", "n_runs": 0,
                "not_computed": "every pair failed the stream check (A7)"}
    ln = {c: np.log(np.asarray(coarse[c], float)) - np.log(np.asarray(fine[pairs[c]], float))
          for c in kept}
    res = paired_log(ln, small_sample_rule=small_sample_rule)
    res["stream_pairing"] = "unchecked" if run_dirs is None else "identical"
    fm = np.array([np.asarray(fine[pairs[c]], float)[0] for c in kept])
    cm = np.array([np.asarray(coarse[c], float)[0] for c in kept])
    return {**out, **res, "ratio_of_means": float(cm.mean() / fm.mean())}


def fixed_effects(ln, X, strata, report: Mapping[str, int], run_var=None) -> dict:
    """Weighted least squares of the units' log metrics on the design X, with an
    error for every coefficient by the decomposition of the module docstring.

    ln      units x (1 + B) log metrics, as for `combined_error`
    X       units x p design; it must hold an intercept for every stratum
    strata  each unit's stratum (one probe task), resampled on its own: C holds
            the test noise the units of one stratum share, not noise that two
            strata share through common jets (v2: the b vs c two-prong and
            retained-topology probes share X->bb jets, the four-prong and
            visible-content probes X->YY->bbqq, and the two X->bc probes X->bc)
    report  {name: column of X} of the coefficients to report
    run_var each unit's run variance, measured apart from this fit (A14: from
            the runs that replicate one partition); a unit is then weighted by
            1 / (run_var + v_ind), its run-plus-test variance

    Each unit is weighted by 1 / v_ind of its stratum, the independent test
    variance of one unit there (bootstrap covariance among the stratum's
    units), so a noisier metric counts for less; with run_var, by
    1 / (run_var + v_ind). The weights only set the estimator: the error
    below holds for any fixed weights. The run variance is each
    stratum's own: it differs between probe tasks by about 35x on the v1
    17-class runs, and one value pooled over the tasks covered 0.77 to 0.998
    task by task (verification of 2026-10-01). With L the least-squares rows,
    M = I - X L the residual maker, r = M y the residuals, C the bootstrap
    covariance among all units and W_s the weights of stratum s's units alone,
    stratum s's weighted residual sum of squares Q_s = r' W_s r has expectation

        E[Q_s] = sum_s' G_ss' v_s' + tau_s,   G_ss' = sum_{u in s, u' in s'} W_u M_uu'^2,
                                              tau_s = tr(M' W_s M C),

    and v = G^-1 (Q - tau), each v_s clipped at 0, is reported as each
    stratum's run variance. A coefficient's variance is

        var_j = sum_s c_js v_s + L_j C L_j' ,    c_js = sum_{u in s} L_ju^2 ,

    the second term its own bootstrap variance (ln_test_se^2). The unbiased
    estimate of the first term, g_j'(Q - tau) with g_j = G^-T c_j, has negative
    weights on the strata whose run noise reaches stratum j's residuals through
    the partition effects (more than it reaches the coefficient): it subtracts
    that leakage, estimated from those strata's own spreads. With about three
    degrees of freedom per stratum that difference is unstable: when a noisy
    task's spread came out high and the task's own low, the degrees of freedom
    fell to 0.2 (t near 10^4) in 0.5 % of coefficients and math.exp overflowed
    (verification of 2026-10-01, v1 17-class variances at the v2 test
    fraction). The run term is therefore formed from the positive weights only,
    g_j+ = max(g_j, 0):

        var_j = max(g_j+'(Q - tau), 0) + L_j C L_j' .

    The leakage is then not subtracted. For every v >= 0 the expectation of
    this term can only exceed the run term, by (G' g_j-)'v (G >= 0, g_j- =
    max(-g_j, 0)): 2.5 to 14 % of var_j on the
    v1 17-class variances, up to 22 % with the smaller test noise of the v2
    test fraction. The clip at 0 is max(s^2 - v_ind, 0) of one contrast, and the
    error is never below the coefficient's test error. Nothing assumes the
    units of a stratum share their test noise alike (a partition that merges a
    pair may err on the same jets as another that does). A stratum whose units
    carry no independent test noise (an AUC of 1 floored at one pair in every
    resampling) cannot be weighted, and a design that leaves a stratum no
    residual of its own cannot separate its run variance; either way the fit is
    not computed.

    The interval is Student t at the Welch-Satterthwaite degrees of freedom of
    that positive combination, the observed term g_js+ Q_s carrying Q_s's own
    k_s = tr(A_s S)^2 / tr(A_s S A_s S) (A_s = M' W_s M, S = diag(v) + C: about
    3 for a task of five partitions) and the bootstrap terms none:
    dof_j = var_j^2 / sum_s (g_js+ Q_s)^2 / k_s. For one stratum and an
    intercept this is the form of `combined_error`, not its value: the test
    part subtracted is tau (d - o per unit, d and o the stratum's mean diagonal
    and off-diagonal bootstrap covariance) where combined_error takes
    d - max(o, 0), and k comes from S where combined_error takes n - 1; the two
    agree when o >= 0 and C is exchangeable.

    Coverage per task (src/stats/tests/test_paired.py: the committed A10 merge
    design, the v1 17-class per-task variances) is that of the per-pair
    two-group rule on the same draws within simulation error: 0.95 where the
    run variance dominates, and above it, as that rule is, where the test noise
    is as large (0.96 to 0.99: neither goes below the test error). The lowest,
    X->bc vs X->cs, covers 0.9455 over 20,000 trials against 0.948 for the
    true variance and a normal quantile: Welch-Satterthwaite's own error at
    about three degrees of freedom. Test noise that two strata share, not
    measured here, did not lower it (correlation 0.3 and 0.6 between such
    strata). Without the leakage term no coefficient's degrees of freedom fell
    below 1 in 140,000 (v1 variances at the v2 test fraction), where the
    subtracted form fell below 1 in 0.5 % of them."""
    Lm = np.atleast_2d(np.asarray(ln, dtype=np.float64))
    X = np.asarray(X, dtype=np.float64)
    strata = list(strata)
    n, p = X.shape
    if not np.all(np.isfinite(Lm)):
        raise SystemExit("FATAL: a metric is zero or negative; its log is undefined")
    if np.linalg.matrix_rank(X) < p:
        raise SystemExit(f"FATAL: the design is not of full rank ({np.linalg.matrix_rank(X)} < {p})")
    if n - p < 1:
        raise SystemExit(f"FATAL: {n} units for {p} coefficients leave no residual")
    v_ind = np.empty(n)
    rv = np.zeros(n) if run_var is None else np.asarray(run_var, dtype=np.float64)
    for s in dict.fromkeys(strata):
        idx = [i for i, x in enumerate(strata) if x == s]
        if len(idx) < 2:
            raise SystemExit(f"FATAL: stratum {s!r} has one unit; its independent test "
                             "variance cannot be separated from the shared one")
        C = np.cov(Lm[idx, 1:])
        k = len(idx)
        diag = float(np.trace(C)) / k
        off = (float(C.sum()) - k * diag) / (k * (k - 1))
        v_ind[idx] = diag - max(off, 0.0)
        if not v_ind[idx[0]] + rv[idx[0]] > 0:
            return {"not_computed": f"the units of stratum {s!r} carry no independent test "
                                    "noise, so they cannot be weighted"}
    W = 1.0 / (v_ind if run_var is None else v_ind + rv)
    A = np.linalg.inv(X.T @ (W[:, None] * X))
    rows = A @ (X.T * W)                                   # p x n least-squares rows
    beta = rows @ Lm[:, 0]
    reps = rows @ Lm[:, 1:]
    M = np.eye(n) - X @ rows
    resid = M @ Lm[:, 0]
    C = np.cov(Lm[:, 1:])
    names = list(dict.fromkeys(strata))
    D = np.array([[x == s for x in strata] for s in names], dtype=np.float64)   # strata x units
    As = [M.T @ ((W * d)[:, None] * M) for d in D]                              # A_s = M' W_s M
    G = np.array([np.diag(a) @ D.T for a in As])
    if np.linalg.matrix_rank(G) < len(names):
        return {"not_computed": "the design leaves some stratum no residual of its own, so its "
                                "run variance cannot be measured"}
    Q = D @ (W * resid ** 2)
    tau = np.array([(a * C).sum() for a in As])           # the test part of each Q_s
    v = np.clip(np.linalg.solve(G, Q - tau), 0.0, None)
    S = np.diag(D.T @ v) + C
    k = np.array([float(np.trace(x)) ** 2 / float((x * x.T).sum())
                  for x in (a @ S for a in As)])           # each Q_s's degrees of freedom
    dof_r = n - p
    rss = float((W * resid ** 2).sum())
    out = {"resid_dof": dof_r, "dispersion": rss / dof_r,
           "v_run": {s: float(x) for s, x in zip(names, v)},
           "stratum_dof": {s: float(x) for s, x in zip(names, k)}, "coefficients": {}}
    for name, j in report.items():
        c = D @ rows[j] ** 2                               # each stratum's share of the run term
        g = np.clip(np.linalg.solve(G.T, c), 0.0, None)    # the leakage is not subtracted
        var = max(float(g @ (Q - tau)), 0.0) + float(reps[j].var(ddof=1))
        comb = math.sqrt(var)
        obs = float((((g * Q) ** 2) / k).sum())
        dof = var ** 2 / obs if obs > 0 else math.inf
        t = _t975(dof)
        b = float(beta[j])
        out["coefficients"][name] = {
            "ratio": math.exp(b), "ln_ratio": b,
            "ln_test_se": float(reps[j].std(ddof=1)), "ln_combined_se": comb, "dof": dof,
            "ci95": [math.exp(b - t * comb), math.exp(b + t * comb)],
            "z": b / comb if comb > 0 else math.inf}
    return out


def fieller(num, den, small_sample_rule: bool = False) -> dict:
    """The ratio of two paired means, f = mean_k N_k / mean_k D_k over runs k, with
    its 95 % Fieller interval: every f for which the contrast mean_k (N_k - f D_k)
    lies within its own 95 % interval of 0, that error formed by combined_error
    (run + test, Student t at its degrees of freedom) like every other contrast.
    The denominator's error so enters the interval, which a ratio of the two
    point values would drop, and the interval does not rest on resampling five
    runs, which a bootstrap over runs would (126 distinct resamples, no
    small-sample correction). When the denominator's own 95 % interval holds 0
    the set is unbounded and no interval is given.

    num, den   runs x (1 + B) replicate vectors of each run's numerator and
               denominator (log contrasts), on one resampling of the test jets.
    The edges are found by bisection; the set is taken to be one interval about f."""
    N = np.atleast_2d(np.asarray(num, dtype=np.float64))
    D = np.atleast_2d(np.asarray(den, dtype=np.float64))
    if N.shape != D.shape:
        raise SystemExit(f"FATAL: numerator {N.shape} and denominator {D.shape} differ in shape")
    n = N.shape[0]
    if small_sample_rule and n < 2:
        return {"n_runs": n, "not_computed": ONE_PAIR}
    w = np.full(n, 1.0 / n)
    out = {"n_runs": n}
    for name, M in (("numerator", N), ("denominator", D)):
        e = combined_error(M, w, small_sample_rule=small_sample_rule)
        if "not_computed" in e:
            return {**out, "not_computed": f"{name}: {e['not_computed']}"}
        out[name] = {k: e[k] for k in ("estimate", "ln_test_se", "ln_combined_se", "dof")}
        out[name]["ci95"] = list(bounds(e["estimate"], e["ln_combined_se"], e["dof"]))
    den_e = out["denominator"]
    if den_e["estimate"] == 0:
        return {**out, "not_computed": "the denominator is 0"}
    f = out["numerator"]["estimate"] / den_e["estimate"]
    out["fraction"] = f
    if den_e["ci95"][0] <= 0 <= den_e["ci95"][1]:
        return {**out, "ci95": None,
                "note": "the denominator's 95 % interval holds 0, so the Fieller set is unbounded"}

    def inside(x):
        e = combined_error(N - x * D, w, small_sample_rule=small_sample_rule)
        return abs(e["estimate"]) <= _t975(e["dof"]) * e["ln_combined_se"]

    def edge(sign):
        a, step = f, 1e-3 * max(abs(f), 1.0)          # a inside, b outside
        b = f + sign * step
        while inside(b):
            if step > 1e12:
                raise SystemExit("FATAL: the Fieller set does not close although the "
                                 "denominator's interval excludes 0")
            a, step = b, 2 * step
            b = f + sign * step
        for _ in range(200):
            m = 0.5 * (a + b)
            if m in (a, b):
                break
            a, b = (m, b) if inside(m) else (a, m)
        return 0.5 * (a + b)

    return {**out, "ci95": [edge(-1), edge(+1)], "interval": "Fieller, 95 %"}


# --------------------------------------------------------- decision labels (A14)
LN_1P1 = math.log(1.1)
DEPENDS = "depends on the checkpoint"
DEPENDS_UNDER_10 = "depends on the checkpoint, under 10%"
DEPENDENT = (DEPENDS, DEPENDS_UNDER_10)


def checkpoint_label(lo: float, hi: float) -> str:
    """A14, the 95 % interval [lo, hi] of ln(result at the other checkpoint / result
    at the primary): 'depends on the checkpoint' when it excludes 0; 'robust' when
    it lies within +-ln 1.1; 'inconclusive' otherwise. An interval that excludes 0
    and lies within +-ln 1.1 meets both rules, which A14 does not order, and is
    labelled for both: 'depends on the checkpoint, under 10%'. It counts as
    dependent (DEPENDENT), so the dependent results are the intervals that exclude
    0, 5 % of them under the null. p1_label treats its overlap the same way."""
    dep, small = lo > 0 or hi < 0, -LN_1P1 < lo and hi < LN_1P1
    if dep:
        return DEPENDS_UNDER_10 if small else DEPENDS
    return "robust" if small else "inconclusive"


def p1_label(lo: float, hi: float) -> str:
    """A14 P1, the 95 % interval [lo, hi] of the merging cost ln(merged / split):
    'merging costs nothing' when the upper bound is below ln 1.1; 'merging costs'
    when the lower bound is above 0; 'inconclusive' otherwise. An interval inside
    (0, ln 1.1) meets both rules, which A14 does not order, and is labelled for
    both: 'merging costs, under 10%' (a cost, and below the 10 % margin), as
    checkpoint_label treats its overlap."""
    if lo > 0:
        return "merging costs, under 10%" if hi < LN_1P1 else "merging costs"
    return "merging costs nothing" if hi < LN_1P1 else "inconclusive"


def beats_label(lo95: float) -> str:
    """A14, random against semantic: 'beats' when the 95 % lower bound is above 0."""
    return "beats" if lo95 > 0 else "inconclusive"


def equal_label(lo90: float, hi90: float) -> str:
    """A14, random against semantic: 'equal' when the 90 % interval lies within
    +-ln 1.1."""
    return "equal" if -LN_1P1 < lo90 and hi90 < LN_1P1 else "inconclusive"


def threshold_label(lo: float, hi: float, threshold: float = 0.0) -> str:
    """'holds' when the interval lies above `threshold`, 'fails' when it lies below,
    'inconclusive' when it spans both outcomes."""
    if lo > threshold:
        return "holds"
    if hi < threshold:
        return "fails"
    return "inconclusive"


def equivalence_label(lo: float, hi: float, margin: float) -> str:
    """'holds' when the interval lies within +-margin, 'fails' when it lies wholly
    outside it, 'inconclusive' when it spans both; 'not evaluable' when margin <= 0."""
    if not margin > 0:
        return "not evaluable"
    if -margin < lo and hi < margin:
        return "holds"
    if lo > margin or hi < -margin:
        return "fails"
    return "inconclusive"


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
