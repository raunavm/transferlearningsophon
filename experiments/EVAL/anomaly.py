#!/usr/bin/env python3
"""Anomaly detection as a VOCABULARY ABLATION (docs/PRD_PLAN.md 3.3, 4.3).

The question is not "can this representation do AD" -- it is whether a coarser
pretraining vocabulary costs AD sensitivity, and whether the loss is in the
SCORE (which is defined by the vocabulary) or in the FEATURES (which are not).
So every configuration runs two families side by side:

  vocabulary-DEFINED   class_sum   -- sum of resonant node scores over the QCD
                                     sum, leave-one-node-out. Constructible only
                                     where the arm's head still has the nodes.
                                     At R16_Q1 it is not, and THAT IS THE RESULT.
  vocabulary-FREE      knn, mahalanobis, iad_hgb -- run on the frozen 128-d
                                     features, identical code for every arm, so
                                     they isolate the representation from the
                                     head.

DEFINITIONS, taken from the papers and not invented:

  SIC(t)   = eps_S(t) / sqrt(eps_B(t)).
  max SIC  only over thresholds whose background efficiency has relative
           statistical error < 20 % -- verbatim from arXiv:2604.20965: "we only
           take into account classifier thresholds for which the relative
           statistical error of the background efficiency is smaller than 20%".
           That is n_B(t) > 25, and without it max SIC is a fluctuation finder.
  sigma_min = S_min / sqrt(B), the initial significance at which discovery is
           still possible: the sigma for which max(S)(sigma_min) = sigma_t
           (arXiv:2604.20965 eq. 4). sigma_t = 5 here.
  regret   r_f = sigma_min,f / sigma_min,best (arXiv:2604.20965) -- how much
           worse a feature set is than the best one tested, on the same signal.
  ARGOS    = eps_SR / sqrt(eps_BT) - sqrt(eps_BT), "Above Random Gain Of SIC"
           (arXiv:2511.14832 eq. 1). It needs a BACKGROUND TEMPLATE, a sample
           following the background distribution in the signal region, and it
           never sees a truth label -- which is the entire point. It is the
           SIGNAL-BLIND selection rule: the operating point is chosen by ARGOS
           on (data, template) alone, then the SIC at that point is reported.
           Choosing the operating point by max SIC instead would be selecting on
           the answer.

Zero GPU: every input is a cached feature directory written by
experiments/EVAL/extract_features.py.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import hashlib
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

REPO = pathlib.Path(__file__).resolve().parents[2]
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"

SIGMA_T = 5.0          # the discovery threshold sigma_min is defined against
STAT_CUT = 0.20        # relative statistical error on eps_B; arXiv:2604.20965
MIN_BKG_PASS = int(np.ceil(1.0 / STAT_CUT ** 2))   # = 25
N_TRAININGS = 10       # per (arm, signal, N_sig); PRD_PLAN 4.3
KNN_K = 10
# ARGOS on pure background should sit near its random baseline; above this the
# selection rule is finding structure in background alone.
NULL_ARGOS_MAX = 0.5


def _probe():
    """Reuse probe.py's cache loader rather than a second copy of it."""
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_map() -> list[dict]:
    with MAP.open() as f:
        return list(csv.DictReader(f))


def cell_seed(arm: str, sig: str, n_sig: int, t: int) -> int:
    """A stable seed for one cell. blake2b, not hash(): hash() is salted per
    interpreter, so the same cell drew a different partition on every run."""
    key = f"{arm}|{sig}|{n_sig}|{t}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


def node_roles(rung: str) -> tuple[dict[int, str], set[int], set[int]]:
    """(native label -> node id), resonant node ids, QCD node ids, for one rung.

    A node is QCD only if EVERY native class in it is QCD. A node mixing QCD
    with resonant classes is neither, and is excluded from both sums -- putting
    it in either one would silently define the score to include the thing it is
    meant to discriminate against.
    """
    rows = read_map()
    if rung not in rows[0]:
        raise SystemExit(f"FATAL: rung {rung} not a column of {MAP}")
    node_of, members = {}, {}
    for r in rows:
        lab, node = int(r["jet_label"]), int(r[rung])
        node_of[lab] = node
        members.setdefault(node, []).append(r["class_name"])
    qcd, res = set(), set()
    for node, names in members.items():
        isq = [n.startswith("label_QCD_") for n in names]
        if all(isq):
            qcd.add(node)
        elif not any(isq):
            res.add(node)
    return node_of, res, qcd


def sic_curve(y: np.ndarray, s: np.ndarray):
    """SIC over thresholds, with the 20 % background-statistics cut applied."""
    order = np.argsort(-s)
    ys = y[order]
    n_s, n_b = int(ys.sum()), int((1 - ys).sum())
    if n_s == 0 or n_b == 0:
        return None
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    ok = fp > MIN_BKG_PASS             # STRICT: 1/sqrt(25) == 0.20 is not < 0.20
    if not ok.any():
        return None
    eps_s = tp[ok] / n_s
    eps_b = fp[ok] / n_b
    return eps_s, eps_b, eps_s / np.sqrt(eps_b)


def argos(s_data: np.ndarray, s_template: np.ndarray, n_points: int = 200):
    """ARGOS = eps_SR/sqrt(eps_BT) - sqrt(eps_BT), and the threshold maximising it.

    Signal-blind by construction: only the unlabelled signal-region sample and
    the background template enter.
    """
    lo, hi = np.percentile(np.concatenate([s_data, s_template]), [50.0, 99.9])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    best = None
    for t in np.linspace(lo, hi, n_points):
        n_bt = int((s_template >= t).sum())
        if n_bt <= MIN_BKG_PASS:
            continue
        e_sr = float((s_data >= t).mean())
        e_bt = n_bt / s_template.size
        a = e_sr / np.sqrt(e_bt) - np.sqrt(e_bt)
        if best is None or a > best[0]:
            best = (float(a), float(t))
    return best


def sigma_min(eps_s: np.ndarray, eps_b: np.ndarray, n_b_total: int) -> float:
    """Smallest S/sqrt(B) whose best achievable significance reaches sigma_t.

    max_t [ S eps_S(t) / sqrt(B eps_B(t)) ] = sigma_t, with sigma = S/sqrt(B),
    so sigma_min = sigma_t / max_t[ eps_S/sqrt(eps_B) ] = sigma_t / max SIC.
    """
    m = float(np.max(eps_s / np.sqrt(eps_b)))
    return float("inf") if m <= 0 else SIGMA_T / m


def chi2_sculpting(m_pass: np.ndarray, m_all: np.ndarray, bins: int = 20) -> float:
    """chi2/ndf between the m_SD shape before and after the cut.

    A score that sculpts the resonant variable manufactures a bump, so the
    number is reported for EVERY selection, not only suspicious ones.
    """
    if m_pass.size < bins or m_all.size < bins:
        return float("nan")
    edges = np.quantile(m_all, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return float("nan")
    h_p, _ = np.histogram(m_pass, bins=edges)
    h_a, _ = np.histogram(m_all, bins=edges)
    exp = h_a * (h_p.sum() / max(h_a.sum(), 1))
    keep = exp > 0
    if keep.sum() < 2:
        return float("nan")
    return float(np.sum((h_p[keep] - exp[keep]) ** 2 / exp[keep]) / (keep.sum() - 1))


def score_class_sum(logits, rung, sig_node):
    """Sum of resonant node scores over resonant+QCD, leaving the signal's node out.

    Returns None when the arm's vocabulary cannot express it -- which is the
    measurement at the coarse end, not an error.
    """
    _, res, qcd = node_roles(rung)
    res = res - {sig_node}
    if not res or not qcd:
        return None
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    r = p[:, sorted(res)].sum(axis=1)
    q = p[:, sorted(qcd)].sum(axis=1)
    return r / np.maximum(r + q, 1e-12)


def score_knn(X_data, X_template, k=KNN_K):
    """Negative distance to the k-th nearest TEMPLATE neighbour (jBOT-style)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X_template)
    d, _ = nn.kneighbors(X_data)
    return d[:, -1]


def score_mahalanobis(X_data, X_template):
    mu = X_template.mean(axis=0)
    cov = np.cov(X_template, rowvar=False) + 1e-6 * np.eye(X_template.shape[1])
    inv = np.linalg.pinv(cov)
    d = X_data - mu
    return np.einsum("ij,jk,ik->i", d, inv, d)


def score_iad(X_data, X_template, seed):
    """IAD: a classifier separating the signal region from the template.

    Weakly supervised -- it never sees a truth label, only the two samples.
    """
    X = np.vstack([X_data, X_template])
    y = np.concatenate([np.ones(len(X_data)), np.zeros(len(X_template))])
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    clf.fit(X, y)
    return clf.predict_proba(X_data)[:, 1]


# The signal suite: >= 6 JetClass-II classes spanning 2/3/4 prongs
# (docs/PRD_PLAN.md 4.3). Chosen for prong coverage and flavour coverage, and
# FIXED HERE rather than per run, so the suite cannot be selected on the answer.
SIGNAL_SUITE = ["label_X_bb", "label_X_qq",              # 2-prong, heavy + light
                "label_X_YY_bbb", "label_X_YY_qqq",      # 3-prong
                "label_X_YY_bbbb", "label_X_YY_qqqq"]    # 4-prong
N_SIG_SCAN = [0, 250, 500, 1000, 2000, 4000]   # 0 is the null: SIC must not exceed 1


def run_one(arm, rung, F, L, logits, obs, sig_lab, sig_node, n_sig, rng,
            n_bkg, n_template, seed):
    """One AD experiment: build the SR sample and template, score, measure."""
    qcd = np.array(sorted(_probe().qcd_indices()))
    is_q = np.isin(L, qcd)
    is_s = L == sig_lab
    q_idx = np.flatnonzero(is_q)
    s_idx = np.flatnonzero(is_s)
    need = n_bkg + n_template
    if q_idx.size < need or s_idx.size < n_sig:
        return None
    rng.shuffle(q_idx)
    data_q, tmpl_q = q_idx[:n_bkg], q_idx[n_bkg:need]
    data_s = rng.choice(s_idx, size=n_sig, replace=False) if n_sig else np.array([], int)
    d_idx = np.concatenate([data_q, data_s]).astype(int)
    y = np.concatenate([np.zeros(data_q.size), np.ones(data_s.size)])

    scores = {}
    cs = score_class_sum(logits[d_idx], rung, sig_node) if logits is not None else None
    if cs is not None:
        scores["class_sum"] = cs
    Xd, Xt = F[d_idx], F[tmpl_q]
    scores["knn"] = score_knn(Xd, Xt)
    scores["mahalanobis"] = score_mahalanobis(Xd, Xt)
    scores["iad_hgb"] = score_iad(Xd, Xt, seed)

    out = {}
    tmpl_score = {
        "class_sum": (score_class_sum(logits[tmpl_q], rung, sig_node)
                      if logits is not None else None),
        "knn": score_knn(Xt, Xt),
        "mahalanobis": score_mahalanobis(Xt, Xt),
        # IAD's score is defined by a classifier fitted to (data vs template),
        # so it has no meaning applied to the template alone and gets no ARGOS.
        "iad_hgb": None,
    }
    for name, s in scores.items():
        rec = {"n_sig": int(n_sig)}

        # SIGNAL-BLIND FIRST. ARGOS and the sculpting check need no truth label,
        # so they are computed for EVERY point including the N_sig = 0 null --
        # which is the whole reason the null is in the scan. Computing them
        # inside the SIC branch, as this did, made the null produce nothing and
        # the null guard unreachable.
        st = tmpl_score.get(name)
        a = argos(s, st) if st is not None else None
        thr = None
        if a is not None:
            rec["argos"], thr = a
            sel = s >= thr
            if "jet_sdmass" in obs:
                rec["chi2_sculpting_msd"] = chi2_sculpting(
                    obs["jet_sdmass"][d_idx][sel], obs["jet_sdmass"][d_idx])

        curve = sic_curve(y, s)
        if curve is None:
            rec["null" if n_sig == 0 else "skipped"] = (
                True if n_sig == 0
                else "no threshold passes the 20% background-stat cut")
            out[name] = rec
            continue

        eps_s, eps_b, sic = curve
        rec["max_sic"] = float(np.max(sic))
        # max SIC cannot exceed sqrt(n_B / MIN_BKG_PASS_STRICT): the 20 % cut
        # floors eps_B. The smallest count that PASSES is MIN_BKG_PASS + 1,
        # because the cut is strict (1/sqrt(25) == 0.20 is not < 0.20), so the
        # ceiling must use that same bound or a clipped score reads as unclipped.
        # A score AT the ceiling has been clipped, not measured, and several
        # scores landing on the same value is the signature. Reported so that is
        # never mistaken for agreement between methods.
        n_b_tot = int((1 - y).sum())
        rec["max_sic_ceiling"] = float(np.sqrt(n_b_tot / (MIN_BKG_PASS + 1)))
        rec["at_ceiling"] = bool(rec["max_sic"] >= 0.999 * rec["max_sic_ceiling"])
        j = int(np.argmin(np.abs(eps_s - 0.5)))
        rec["sic_at_eps_s_0p5"] = float(sic[j])
        rec["sigma_min"] = sigma_min(eps_s, eps_b, n_b_tot)
        # The SIC is reported AT the ARGOS-chosen point. Choosing by max SIC
        # would be selecting on the answer.
        if thr is not None:
            sel = s >= thr
            if sel.sum() and (1 - y)[sel].sum() > MIN_BKG_PASS:
                e_s = float(y[sel].sum() / max(y.sum(), 1))
                e_b = float((1 - y)[sel].sum() / max(n_b_tot, 1))
                rec["sic_at_argos_point"] = e_s / np.sqrt(e_b) if e_b > 0 else float("nan")
        out[name] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True,
                    help="arm=DIR ... ; the arm name must carry its rung, e.g. r16q1-s2=/path")
    ap.add_argument("--rungs", nargs="+", required=True,
                    help="arm=RUNG ... , the vocabulary each arm was pretrained on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bkg", type=int, default=200_000)
    ap.add_argument("--n-template", type=int, default=200_000)
    ap.add_argument("--trainings", type=int, default=N_TRAININGS)
    ap.add_argument("--signals", nargs="+", default=SIGNAL_SUITE)
    ap.add_argument("--n-sig", type=int, nargs="+", default=N_SIG_SCAN)
    a = ap.parse_args(argv)

    probe = _probe()
    rung_of = dict(s.split("=", 1) for s in a.rungs)
    arms = {}
    for s in a.features:
        name, d = s.split("=", 1)
        if name not in rung_of:
            raise SystemExit(f"FATAL: no --rungs entry for arm {name}")
        p_arm = pathlib.Path(d)
        arms[name] = probe.load_arm(p_arm)
        # probe.load_arm reads features/labels/observers but NOT logits, so the
        # vocabulary-defined score has to load them here -- and validate the row
        # count, or class_sum would be computed against a different sample.
        lg = p_arm / "logits.npy"
        if lg.exists():
            arr = np.load(lg)
            if arr.shape[0] != arms[name]["L"].shape[0]:
                raise SystemExit(
                    f"FATAL: {p_arm} has {arr.shape[0]} logit rows and "
                    f"{arms[name]['L'].shape[0]} labels")
            arms[name]["logits"] = arr
    align = probe.check_alignment(arms)

    rows = read_map()
    by_name = {r["class_name"]: int(r["jet_label"]) for r in rows}
    for s in a.signals:
        if s not in by_name:
            raise SystemExit(f"FATAL: signal {s} is not a class in {MAP}")

    results = {"row_alignment_sha256": align, "sigma_t": SIGMA_T,
               "stat_cut": STAT_CUT, "min_bkg_pass": MIN_BKG_PASS,
               "trainings": a.trainings, "n_bkg": a.n_bkg,
               "n_template": a.n_template, "arms": {}}

    for arm, d in sorted(arms.items()):
        rung = rung_of[arm]
        node_of, _, _ = node_roles(rung)
        F, L, obs = d["F"], d["L"], d["obs"]
        logits = d.get("logits")
        if logits is None:
            print(f"  {arm}: no logits.npy -- the vocabulary-defined score "
                  f"cannot be built, only the vocabulary-free ones")
        results["arms"][arm] = {"rung": rung, "signals": {}}
        for sig in a.signals:
            lab = by_name[sig]
            snode = node_of[lab]
            per_n = {}
            for n_sig in a.n_sig:
                reps = []
                for t in range(a.trainings):
                    # NOT hash(): Python salts it per interpreter (PYTHONHASHSEED
                    # is unset in the job spec), so the QCD data/template split
                    # and the signal subsample -- i.e. every number in the file
                    # -- differ run to run and no field records the draw.
                    seed = cell_seed(arm, sig, n_sig, t)
                    rng = np.random.default_rng(seed)
                    r = run_one(arm, rung, F, L, logits, obs, lab, snode,
                                n_sig, rng, a.n_bkg, a.n_template, seed=t)
                    if r:
                        r["rng_seed"] = int(seed)
                        reps.append(r)
                if not reps:
                    per_n[str(n_sig)] = {"skipped": "insufficient jets"}
                    continue
                seeds_used = [r["rng_seed"] for r in reps if "rng_seed" in r]
                agg = {}
                # only the score-family dicts; a scalar bookkeeping key (e.g.
                # rng_seed) is not a family and must not be aggregated as one
                for fam in {k for r in reps for k, v in r.items()
                            if isinstance(v, dict)}:
                    vals = [r[fam] for r in reps if fam in r and "max_sic" in r[fam]]
                    if not vals:
                        nulls = [r[fam] for r in reps if fam in r and r[fam].get("null")]
                        if nulls:
                            # the N_sig = 0 point: no SIC exists, but ARGOS and
                            # the sculpting check do, and they ARE the null test
                            agg[fam] = {"null": True}
                            for m in ("argos", "chi2_sculpting_msd"):
                                have = [v[m] for v in nulls if m in v]
                                if have:
                                    agg[fam][m] = float(np.median(have))
                            agg[fam]["n_trainings"] = len(nulls)
                        else:
                            agg[fam] = {"skipped": reps[0].get(fam, {}).get("skipped", "n/a")}
                        continue
                    agg[fam] = {m: float(np.median([v[m] for v in vals if m in v]))
                                for m in ("max_sic", "max_sic_ceiling",
                                          "sic_at_eps_s_0p5", "sigma_min",
                                          "argos", "sic_at_argos_point",
                                          "chi2_sculpting_msd")
                                if any(m in v for v in vals)}
                    # a boolean does not survive a median; if ANY training
                    # clipped, the cell is not a clean measurement
                    agg[fam]["at_ceiling"] = bool(
                        any(v.get("at_ceiling") for v in vals))
                    agg[fam]["n_trainings"] = len(vals)
                    agg[fam]["max_sic_iqr"] = float(
                        np.subtract(*np.percentile([v["max_sic"] for v in vals], [75, 25])))
                # WITHIN-ARM regret: best family for this arm on this cell.
                # This is NOT the number the vocabulary ablation wants -- see
                # the cross-arm pass after every arm is built. Kept because it
                # answers a different, real question (which score family to
                # use given a fixed vocabulary), under a name that says so.
                best = min((v.get("sigma_min", float("inf")) for v in agg.values()
                            if isinstance(v, dict)), default=float("inf"))
                for v in agg.values():
                    if isinstance(v, dict) and "sigma_min" in v and np.isfinite(best) and best > 0:
                        v["regret_within_arm"] = v["sigma_min"] / best
                agg["rng_seeds"] = seeds_used
                per_n[str(n_sig)] = agg
                line = "  ".join(
                    f"{k}:maxSIC={v['max_sic']:.2f}" for k, v in sorted(agg.items())
                    if isinstance(v, dict) and "max_sic" in v)
                print(f"  {arm:12s} {sig:18s} N_sig={n_sig:5d}  {line}", flush=True)
            results["arms"][arm]["signals"][sig] = per_n

    # CROSS-ARM REGRET -- the number the vocabulary ablation is about, and the
    # reason this pass exists at all. Normalising inside one arm (as the first
    # pass does) makes every arm's best family score exactly 1.000, so the
    # cross-arm table reads "no regret from coarsening the vocabulary" for
    # every arm no matter how much worse the coarse one is: precisely the null
    # under test, manufactured by the aggregation. The minimum is taken over
    # ARMS at fixed (signal, N_sig, family), so families are never compared
    # against each other here.
    cells = {}
    for arm, ad in results["arms"].items():
        for sig, per_n in ad["signals"].items():
            for n_sig, agg in per_n.items():
                if not isinstance(agg, dict):
                    continue
                for fam, v in agg.items():
                    if isinstance(v, dict) and "sigma_min" in v:
                        cells.setdefault((sig, n_sig, fam), []).append(v)
    for key, vs in cells.items():
        best = min(v["sigma_min"] for v in vs)
        for v in vs:
            v["regret"] = v["sigma_min"] / best if best > 0 else float("nan")
            v["regret_n_arms"] = len(vs)
    results["regret_normalisation"] = (
        "sigma_min / min over ARMS at fixed (signal, n_sig, family); "
        "regret_within_arm is the same ratio taken over families inside one arm")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anomaly_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/anomaly_results.json")

    # THE NULL IS A GUARD, NOT A ROW. With no signal there is no SIC, so the
    # null is read on ARGOS -- the quantity that picks the operating point. On
    # pure background ARGOS should sit near its random baseline; a large value
    # means the selection rule is manufacturing an excess out of background
    # alone, and every number downstream of it is suspect.
    bad = []
    for arm, ad in results["arms"].items():
        for sig, per_n in ad["signals"].items():
            z = per_n.get("0", {})
            for fam, v in z.items():
                if isinstance(v, dict) and v.get("argos", 0) > NULL_ARGOS_MAX:
                    bad.append(f"{arm}/{sig}/{fam} ARGOS {v['argos']:.3f} on pure "
                               f"background (> {NULL_ARGOS_MAX})")
    if bad:
        print("\nWARNING: the N_sig=0 null is not flat -- do not quote these:")
        for b in bad:
            print("   ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
