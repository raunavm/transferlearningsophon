#!/usr/bin/env python3
"""The revision's statistics (ledger section 35; external audit V4, V6, V7, A1), from committed data only; called by
mlst_v2_tables.main, which writes its outputs beside Tables 1-3.

  V4   table_differences.tex: the differences the main text states, Delta, sigma (SC-1: both sides' run SDs and the
       paired test-jet bootstrap; for the transfer heads SC-2's analytic term), Delta/sigma, runs per side; with the
       Welch-Satterthwaite check of each verdict.
  V6   the smallest difference between nested label sets that the top and quark/gluon rows resolve at 3 sigma, and the
       observed spread of the four nested sets.
  V7   the random-grouping pool (stats/rand_pool_overlap.json, experiments/STATS/rev35_jobs.py pool).
  A1   S(L), pre-specified in ledger 35 before computing: for each pretrained label set L, the sum of
       w_ij = P(pred j | true i) + P(pred i | true j) over the class pairs i < j that L puts in different groups, over
       the sum over all pairs; P from the frozen linear probe of the 188-class model (mean over its runs) on the
       class-balanced JetClass-II test set (stats/confusion_jc2bal_probe.json), the untrained network's as the
       secondary. Outcome: the label set's frozen 188-class probe accuracy. Spearman rho over the label sets with the
       exact permutation p-value; the unweighted share of separated pairs as the naive baseline.

    python3 experiments/FIGS/rev35_stats.py      # prints everything; --csv writes stats/a1_separation.csv
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import pathlib
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "experiments/FIGS/data/v2"
BOOT = DATA / "bootstrap"
sys.path.insert(0, str(REPO / "experiments/FIGS"))
import make_tables as MT  # noqa: E402

# ------------------------------------------------------------------ A1
MAPS = (("rung_label_maps.v1.csv", ("L188", "L162", "R42_Q1", "R16_Q1"), "nested"),
        ("rand_label_map.v2.csv", tuple(f"RAND2_p{k}" for k in range(1, 6)), "random grouping"),
        ("flavour_pair_map.v2.csv", ("FLAV_F0", "FLAV_F1", "FLAV_F1R"), "flavor-boundary grouping"))
RUNS = {"L188": "mtx-l188", "L162": "mtx-l162", "R42_Q1": "mtx-r42q1", "R16_Q1": "mtx-r16q1", "FLAV_F0": "mtx-flavf0",
        "FLAV_F1": "mtx-flavf1", "FLAV_F1R": "mtx-flavf1r", **{f"RAND2_p{k}": f"mtx-rand2p{k}" for k in range(1, 6)}}
NAME = {"L188": "188 classes", "L162": "162 classes", "R42_Q1": "43 classes", "R16_Q1": "17 classes",
        "FLAV_F0": "17 classes, no flavor boundary", "FLAV_F1": "17 classes, b-quark boundary",
        "FLAV_F1R": "17 classes, random flavor boundary", **{f"RAND2_p{k}": f"17 classes, random grouping {k}" for k in range(1, 6)}}


def groups() -> dict:
    """{label set: array over the 188 native classes (jet_label order) of its group}."""
    out = {}
    for f, cols, _ in MAPS:
        rows = list(csv.DictReader((REPO / "configs/labelmaps" / f).open()))
        if [int(r["jet_label"]) for r in rows] != list(range(188)):
            raise SystemExit(f"FATAL: {f} is not in jet_label order 0..187")
        for c in cols:
            out[c] = np.array([int(r[c]) for r in rows])
    return out


def separation(P: np.ndarray, g: np.ndarray) -> float:
    w = P + P.T
    iu = np.triu_indices(len(g), 1)
    return float(w[iu][g[iu[0]] != g[iu[1]]].sum() / w[iu].sum())


def naive(g: np.ndarray) -> float:
    iu = np.triu_indices(len(g), 1)
    return float(np.mean(g[iu[0]] != g[iu[1]]))


def accuracy(arm: str) -> list[float]:
    out = []
    for f in sorted((DATA / "linprobe/fits_v5").glob(f"{RUNS[arm]}-s*.json")):
        out.append(json.loads(f.read_text())["cells"]["best70"]["jc2bal"]["1000000"]["accuracy"])
    if not out:
        raise SystemExit(f"FATAL: no frozen JetClass-II probe for {arm}")
    return out


def _ranks(v) -> list[int]:
    """Twice the mid-ranks (integers, ties shared)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = i + j + 2          # 2 * mean of 1-based ranks i+1..j+1
        i = j + 1
    return r


def spearman_exact(x, y) -> tuple[float, float]:
    """(rho, exact two-sided permutation p): the share of all n! orderings of y whose |rho| reaches the observed |rho|,
    counted exactly by dynamic programming over the assigned ranks (ties as mid-ranks)."""
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    t_obs = sum(a * b for a, b in zip(rx, ry))
    rho = (t_obs - n * mx * my) / (sx * sy)
    dist = {0: {0: 1}}              # mask of used y-ranks -> {sum of products: count}
    for i in range(n):
        nxt = {}
        for mask, sums in dist.items():
            for j in range(n):
                if mask >> j & 1:
                    continue
                d = nxt.setdefault(mask | 1 << j, {})
                for s, c in sums.items():
                    k = s + rx[i] * ry[j]
                    d[k] = d.get(k, 0) + c
        dist = nxt
    (full,) = dist.values()
    tot = sum(full.values())
    dev = abs(t_obs - n * mx * my)
    hit = sum(c for s, c in full.items() if abs(s - n * mx * my) >= dev - 1e-9)
    return rho, hit / tot


def a1() -> dict | None:
    f = DATA / "stats/confusion_jc2bal_probe.json"
    if not f.exists():
        return None
    conf = json.loads(f.read_text())
    gs = groups()
    rows = []
    for _f, cols, fam in MAPS:
        for c in cols:
            acc = accuracy(c)
            rows.append({"label_set": c, "name": NAME[c], "family": fam,
                         "S": separation(np.array(conf["mean"]["l188"]), gs[c]),
                         "S_untrained": separation(np.array(conf["mean"]["init_bn"]), gs[c]),
                         "naive": naive(gs[c]), "accuracy": statistics.fmean(acc),
                         "accuracy_sd": statistics.stdev(acc) if len(acc) > 1 else None, "n_runs": len(acc)})
    acc = [r["accuracy"] for r in rows]
    out = {"rows": rows}
    for key in ("S", "S_untrained", "naive"):
        out[key] = dict(zip(("rho", "p"), spearman_exact([r[key] for r in rows], acc)))
        sub = [r for r in rows if r["label_set"] != "L188"]
        out[key + "_without_188"] = dict(zip(("rho", "p"), spearman_exact([r[key] for r in sub], [r["accuracy"] for r in sub])))
    return out


def write_csv(res: dict, dest) -> None:
    """dest: a path, or a text stream."""
    with (dest.open("w", newline="") if isinstance(dest, pathlib.Path) else contextlib.nullcontext(dest)) as fh:
        w = csv.writer(fh)
        w.writerow(["label_set", "name", "family", "S", "S_untrained", "naive_separated_pairs", "accuracy", "accuracy_sd",
                    "n_runs"])
        for r in res["rows"]:
            w.writerow([r["label_set"], r["name"], r["family"], f"{r['S']:.6f}", f"{r['S_untrained']:.6f}",
                        f"{r['naive']:.6f}", f"{r['accuracy']:.6f}",
                        "" if r["accuracy_sd"] is None else f"{r['accuracy_sd']:.6f}", r["n_runs"]])


# ------------------------------------------------------------------ V4: the differences the main text states
def _row(f: str, diff: str, met: str, flip: bool) -> dict:
    d = json.loads((BOOT / f).read_text())["diffs"][diff]
    e = d[met]
    sign = -1 if e["quantity"] == "one_minus_auc" else 1
    sign *= -1 if flip else 1
    return {"delta": sign * e["delta"], "sd_first": e["sd_b" if flip else "sd_a"], "sd_second": e["sd_a" if flip else "sd_b"],
            "boot_sd": e["boot_sd"], "sigma": e["sigma"], "n_first": e["n_b" if flip else "n_a"],
            "n_second": e["n_a" if flip else "n_b"], "source": f"bootstrap/{f} {diff} {met}", "kind": "SC-1"}


EPS_B = 0.01


def _xbx_net(sig: str, path: pathlib.Path, scratch: bool) -> tuple[float, float]:
    """(eps_S in percent, its analytic SD) of one network's transfer heads (mean over head seeds) or one from-scratch
    seed; the term of tests/test_xbx_claims_l3.py (mlst_results.xbx_side)."""
    doc = json.loads(path.read_text())
    recs = [doc] if scratch else doc["signals"][sig]["head"]

    def one(r):
        eb, s = np.array(r["curve"]["eps_b"]), np.array(r["curve"]["eps_s"])
        i = int(np.argmin(abs(eb - EPS_B)))
        slope = (s[i + 1] - s[i - 1]) / (eb[i + 1] - eb[i - 1])
        e = r["eps_s_at_star"]
        return e, e * (1 - e) / r["n_s"] + slope ** 2 * EPS_B * (1 - EPS_B) / r["n_b"]
    v = [one(r) for r in recs]
    return 100 * float(np.mean([e for e, _ in v])), 100 * math.sqrt(float(np.mean([t for _, t in v])))


def _xbx_row(sig: str, first: str, second: str) -> dict:
    def side(arm):
        if arm == "scratch":
            return [_xbx_net(sig, DATA / f"xbx_scratch/{sig}/s{k}.json", True) for k in (1, 2, 3)]
        return [_xbx_net(sig, f, False) for f in sorted((DATA / f"xbx/{sig}").glob(f"{arm}-s*.json"))]
    a, b = side(first), side(second)
    va, vb = [x for x, _ in a], [x for x, _ in b]
    sa, sb = statistics.stdev(va), statistics.stdev(vb)
    ta, tb = float(np.mean([t for _, t in a])), float(np.mean([t for _, t in b]))
    return {"delta": statistics.fmean(va) - statistics.fmean(vb), "sd_first": sa, "sd_second": sb,
            "boot_sd": math.hypot(ta, tb), "sigma": math.sqrt(sa * sa + sb * sb + ta * ta + tb * tb), "n_first": len(va),
            "n_second": len(vb), "source": f"xbx/{sig} transfer heads (SC-2 analytic term)", "kind": "SC-2"}


def _mass_row(group: str | None, diff: str | None, probe: str) -> dict:
    doc = json.loads((DATA / "head/mass_sd_boot_v5.json").read_text())
    if group:
        e = doc["groups"][group][probe]["all|inclusive"]
        return {"delta": e["delta"], "sd_first": statistics.stdev(e["runs"]), "sd_second": 0.0,
                "boot_sd": math.sqrt(max(e["sigma"] ** 2 - statistics.stdev(e["runs"]) ** 2, 0.0)), "sigma": e["sigma"],
                "n_first": len(e["runs"]), "n_second": 1, "source": f"head/mass_sd_boot_v5.json {group} {probe}",
                "kind": "SC-1 (soft-drop mass: no run spread)"}
    e = doc["diffs"][diff][probe]["all|inclusive"]["resolution"]
    return {"delta": e["delta"], "sd_first": statistics.stdev(e["a"]), "sd_second": statistics.stdev(e["b"]),
            "boot_sd": e.get("boot_sd"), "sigma": e["sigma"], "n_first": len(e["a"]), "n_second": len(e["b"]),
            "source": f"head/mass_sd_boot_v5.json {diff} {probe}", "kind": "SC-1"}


LP, FT = "probe_jc2bal_N1000000.json", "ft_jc2bal_N1000000.json"
V4_ROWS = (
    ("JetClass-II accuracy, frozen", "188 $-$ 162 classes", lambda: _row("l3_v5/" + LP, "162-188", "accuracy", True)),
    ("JetClass-II accuracy, frozen", "188 $-$ 43 classes", lambda: _row("l3_v5/" + LP, "43-188", "accuracy", True)),
    ("JetClass-II accuracy, frozen", "188 $-$ 17 classes", lambda: _row("l3_v5/" + LP, "17-188", "accuracy", True)),
    ("JetClass-II accuracy, frozen", "188 classes $-$ random grouping",
     lambda: _row("l3_v5/probe_jc2bal_N1000000.all.json", "rand-188", "accuracy", True)),
    ("JetClass accuracy, frozen", "188 $-$ 17 classes", lambda: _row("l3_v5/probe_jc1_N1000000.json", "17-188", "accuracy", True)),
    ("JetClass AUC, frozen", "188 $-$ 17 classes", lambda: _row("l3_v5/probe_jc1_N1000000.json", "17-188", "macro", True)),
    ("JetClass accuracy, frozen", "188 classes $-$ random grouping",
     lambda: _row("l3_bs3/probe_jc1_N1000000.parts.json", "rand-188", "accuracy", True)),
    ("JetClass accuracy, frozen", "random grouping $-$ 17 classes",
     lambda: _row("l3_bs3/probe_jc1_N1000000.parts.json", "rand-r16q1", "accuracy", False)),
    ("JetClass-II accuracy, fine-tuned", "188 $-$ 43 classes", lambda: _row("l3_fixed/" + FT, "43-188", "accuracy", True)),
    ("JetClass-II accuracy, fine-tuned", "188 $-$ 17 classes", lambda: _row("l3_fixed/" + FT, "17-188", "accuracy", True)),
    ("JetClass accuracy, fine-tuned", "188 $-$ 17 classes",
     lambda: _row("l3_fixed/ft_jc1_N1000000.json", "17-188", "accuracy", True)),
    ("17-group task, linear probe", "188 $-$ 17-class features",
     lambda: _row("l3_v5/probe_jc2bal_R16_Q1_N1000000.json", "r16q1-188", "accuracy", True)),
    ("17-group task, MLP probe", "188 $-$ 17-class features",
     lambda: _row("l3_bs3/mlp_jc2bal_R16_Q1_N1000000.json", "r16q1-188", "accuracy", True)),
    (r"$X\to b\bar b$, signal efficiency (\%)", "188 $-$ 17 classes", lambda: _xbx_row("xbb", "mtx-l188", "mtx-r16q1")),
    (r"$X\to bs$, signal efficiency (\%)", "188 $-$ 17 classes", lambda: _xbx_row("xbs", "mtx-l188", "mtx-r16q1")),
    (r"$X\to b\bar b$, signal efficiency (\%)", "17 classes $-$ from scratch", lambda: _xbx_row("xbb", "mtx-r16q1", "scratch")),
    (r"$X\to bs$, signal efficiency (\%)", "17 classes $-$ from scratch", lambda: _xbx_row("xbs", "mtx-r16q1", "scratch")),
    ("mass resolution", "regressed $-$ soft-drop, 162 + mass", lambda: _mass_row("162 classes + mass", None, "regressed")),
    ("mass resolution, MLP probe", "162 + mass $-$ 162 classes", lambda: _mass_row(None, "162 classes + mass - 162 classes", "mlp")),
    ("mass resolution, MLP probe", "17 + mass $-$ 17 classes", lambda: _mass_row(None, "17 classes + mass - 17 classes", "mlp")),
    ("JetClass-II accuracy, frozen", "162 + mass $-$ 162 classes", lambda: _row("l3_v5/" + LP, "162mass-162", "accuracy", False)),
    ("JetClass-II accuracy, frozen", "17 + mass $-$ 17 classes", lambda: _row("l3_v5/" + LP, "17mass-17", "accuracy", False)),
    ("JetClass-II accuracy, fine-tuned", "162 + mass $-$ 162 classes", lambda: _row("l3_fixed/" + FT, "162mass-162", "accuracy", False)),
    ("JetClass-II accuracy, fine-tuned", "17 + mass $-$ 17 classes", lambda: _row("l3_fixed/" + FT, "17mass-17", "accuracy", False)),
    # ledger 35, second pass (claims checker F6, F16): the further differences the text quotes
    ("JetClass accuracy, fine-tuned", "162 + mass $-$ 162 classes",
     lambda: _row("l3_fixed/ft_jc1_N1000000.json", "162mass-162", "accuracy", False)),
    ("JetClass accuracy, fine-tuned", "17 + mass $-$ 17 classes",
     lambda: _row("l3_fixed/ft_jc1_N1000000.json", "17mass-17", "accuracy", False)),
    ("top tagging accuracy, frozen", "random grouping $-$ 17 classes",
     lambda: _row("l3_bs3/go7_probe_top_N1200000.json", "rand-r16q1", "accuracy", False)),
    (r"$X\to b\bar b$ vs $X\to c\bar c$, AUC, linear probe", "188 $-$ 43 classes",
     lambda: _row("l3_v5/probe_jc2bal_pairs_bb_vs_cc.json", "r42q1-188", "auc", True)),
    (r"$X\to b\bar bqq$ vs $X\to c\bar cqq$, AUC, linear probe", "188 $-$ 43 classes",
     lambda: _row("l3_v5/probe_jc2bal_pairs_bbqq_vs_ccqq.json", "r42q1-188", "auc", True)),
    ("mass resolution, MLP probe", "17 $-$ 188 classes", lambda: _mass_row(None, "17 classes - 188 classes", "mlp")),
    ("top tagging Rej$_{30\\%}$, frozen", "162 + mass $-$ 162 classes",
     lambda: _row("l3/probe_top_N1200000.itop4.json", "162mass-162", "r30", False)),
    ("top tagging Rej$_{30\\%}$, frozen", "17 + mass $-$ 17 classes",
     lambda: _row("l3/probe_top_N1200000.itop4.json", "17mass-17", "r30", False)),
    ("top tagging Rej$_{30\\%}$, fine-tuned", "188 classes $-$ from scratch",
     lambda: _row("l3/ft_top_N1200000.scratch.json", "188-scratch", "r30", False)),
    ("top tagging accuracy, fine-tuned", "188 classes $-$ from scratch",
     lambda: _row("l3/ft_top_N1200000.scratch.json", "188-scratch", "accuracy", False)),
)


def _t_quantile(p_one_sided: float, nu: float) -> float:
    """Student t quantile by bisection on the regularized incomplete beta (no scipy in the build image)."""
    def cdf(t):
        x = nu / (nu + t * t)
        return 1 - 0.5 * _betainc(nu / 2, 0.5, x)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < 1 - p_one_sided else (lo, mid)
    return (lo + hi) / 2


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by its continued fraction (Numerical Recipes, betai/betacf)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)
    tiny = 1e-300
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        for aa in (m * (b - m) * x / ((a - 1 + 2 * m) * (a + 2 * m)),
                   -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 1 + 2 * m))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / (c if abs(c) > tiny else tiny)
            h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(lbeta) * h / a


P_3SIGMA = 0.00135          # one-sided tail of 3 standard deviations


def welch(r: dict) -> dict:
    """Welch-Satterthwaite on the difference of the two means: SE^2 = sd_a^2/n_a + sd_b^2/n_b + boot^2 (the bootstrap term
    with infinite degrees of freedom); significant when |t| exceeds the t quantile of a two-sided 3-sigma level."""
    va = (r["sd_first"] or 0) ** 2 / r["n_first"] if r["n_first"] > 1 else 0.0
    vb = (r["sd_second"] or 0) ** 2 / r["n_second"] if r["n_second"] > 1 else 0.0
    se2 = va + vb + (r["boot_sd"] or 0) ** 2
    den = (va * va / (r["n_first"] - 1) if r["n_first"] > 1 else 0) + (vb * vb / (r["n_second"] - 1) if r["n_second"] > 1 else 0)
    nu = se2 * se2 / den if den > 0 else math.inf
    t = r["delta"] / math.sqrt(se2)
    crit = 3.0 if math.isinf(nu) else _t_quantile(P_3SIGMA, nu)
    return {"se": math.sqrt(se2), "nu": nu, "t": t, "t_crit": crit, "significant": abs(t) > crit}


# ledger 35, second pass: rows the text quotes as Delta +- sigma; (macro suffix, scale to the text's unit)
TEXT_ROWS = {("JetClass accuracy, frozen", "188 $-$ 17 classes"): ("LpJcOneCoarse", 100),
             ("JetClass accuracy, fine-tuned", "188 $-$ 17 classes"): ("FtJcOneCoarse", 100),
             (r"$X\to b\bar b$, signal efficiency (\%)", "188 $-$ 17 classes"): ("XbbCoarse", 1),
             (r"$X\to bs$, signal efficiency (\%)", "188 $-$ 17 classes"): ("XbsCoarse", 1),
             (r"$X\to b\bar b$, signal efficiency (\%)", "17 classes $-$ from scratch"): ("XbbScratch", 1),
             (r"$X\to bs$, signal efficiency (\%)", "17 classes $-$ from scratch"): ("XbsScratch", 1),
             ("JetClass-II accuracy, frozen", "162 + mass $-$ 162 classes"): ("LpMassRes", 100),
             ("JetClass-II accuracy, fine-tuned", "162 + mass $-$ 162 classes"): ("FtMassRes", 100),
             ("JetClass-II accuracy, fine-tuned", "17 + mass $-$ 17 classes"): ("FtMassCoarse", 100),
             ("JetClass accuracy, fine-tuned", "17 + mass $-$ 17 classes"): ("FtJcOneMassCoarse", 100),
             ("top tagging Rej$_{30\\%}$, fine-tuned", "188 classes $-$ from scratch"): ("FtTopRejGain", 1),
             ("JetClass accuracy, frozen", "random grouping $-$ 17 classes"): ("LpJcOneRandCoarse", 100)}


def welch_dagger(r: dict) -> bool:
    """A row that passes |Delta/sigma| > 3 but not the Welch-Satterthwaite test: the text gives it as a size."""
    return abs(r["z"]) > 3 and not r["welch"]["significant"]


def v4() -> list[dict]:
    out = []
    for what, pair, f in V4_ROWS:
        r = {"what": what, "pair": pair, **f()}
        r["z"] = r["delta"] / r["sigma"]
        r["welch"] = welch(r)
        out.append(r)
    return out


def table_v4(rows: list[dict]) -> str:
    body = []
    last = None
    for r in rows:
        head = r["what"] if r["what"] != last else ""
        last = r["what"]
        d, s = MT.fmt_pm(r["delta"], r["sigma"]).split(r"\,$\pm$\,")
        neg = lambda v: v.replace("-", "$-$", 1) if v.startswith("-") else v  # noqa: E731
        dag = r"$^\dagger$" if welch_dagger(r) else ""
        body.append(f"{head} & {r['pair']} & {neg(d)} & {s} & {neg(f'{r[chr(122)]:.1f}')}{dag} & {r['n_first']}/{r['n_second']} \\\\")
    return "\n".join([
        "% written by experiments/FIGS/rev35_stats.py (ledger 35, audit V4)",
        r"\begin{table}[t]\centering",
        r"\caption{Differences between models quoted in the text: $\Delta$, its combined uncertainty $\sigma$ (both "
        r"sides' run standard deviations and the paired test-jet bootstrap; for the resonance benchmark the binomial "
        r"term of the signal and background jets in place of the bootstrap), $\Delta/\sigma$, and the runs on each side. "
        r"A difference is called resolved when $|\Delta/\sigma|>3$; rows marked $\dagger$ exceed this but do not pass a "
        r"Welch--Satterthwaite $t$ test on the means with two to five degrees of freedom, and the text does not call "
        r"them resolved.}\label{tab:differences}",
        # ledger 35: the two label columns wrap, so the table fits the text width (it was 84 pt too wide at 9 pt)
        r"\footnotesize\setlength{\tabcolsep}{4pt}",   # 32 rows: at \small the table ran into the page foot
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{4.3cm}ccrc@{}}",
        r"\toprule",
        r"Quantity & Difference & $\Delta$ & $\sigma$ & $\Delta/\sigma$ & Runs \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


# ------------------------------------------------------------------ V6
def v6() -> dict:
    nested = {"188": "l188", "162": "l162", "43": "r42q1", "17": "r16q1"}
    out = {}
    for task, f in (("top", "top_N1200000"), ("quark/gluon", "qg_N1600000")):
        for mode, pre in (("frozen", "go7_probe"), ("fine-tuned", "go7_ft")):
            d = json.loads((BOOT / f"l3_bs3/{pre}_{f}.json").read_text())
            for met in ("accuracy", "r30"):
                sig = [e[met]["sigma"] for k, e in d["diffs"].items() if all(x in ("l188", "188", "l162", "r42q1", "r16q1")
                                                                               for x in k.split("-", 1))]
                means = {n: statistics.fmean(d["models"][m][met]["value"] for m in d["models"] if m.startswith(f"mtx-{a}-"))
                         for n, a in nested.items()}
                ref = means["188"]
                out[f"{task}|{mode}|{met}"] = {"min_3sigma": 3 * min(sig), "max_3sigma": 3 * max(sig),
                                               "min_3sigma_share": 3 * min(sig) / ref, "max_3sigma_share": 3 * max(sig) / ref,
                                               "spread": max(means.values()) - min(means.values()), "means": means,
                                               "n_pairs": len(sig)}
    return out


# ------------------------------------------------------------------ V7
def v7() -> dict | None:
    f = DATA / "stats/rand_pool_overlap.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    sel = json.loads((REPO / "configs/labelmaps/rand_v2_selection.json").read_text())
    ov = {s: r["overlap"] for s, r in d["draws"].items()}
    rule3 = [str(s) for s in sel["rule3_seeds"]]
    acc = [str(s) for s in sel["accepted_seeds"]]
    # the QCD classes form one group in the 17-class set and in every draw (the resonant pool): their pairs are merged
    # by both always, so the overlap is also given over the resonant classes alone
    qcd = sum(1 for r in csv.DictReader((REPO / "configs/labelmaps/rung_label_maps.v1.csv").open())
              if r["class_name"].startswith("label_QCD"))
    qp = qcd * (qcd - 1) // 2
    res = {s: (r["merged_both"] - qp) / (r["merged_17"] - qp) for s, r in d["draws"].items()}
    q = lambda v: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v), "mean": statistics.fmean(v)}  # noqa: E731
    out = {"pool_draws": len(ov), "rule3_draws": len(rule3), "merged_17": next(iter(d["draws"].values()))["merged_17"],
           "qcd_pairs": qp, "samples_drawn": sel["samples_drawn"], "valid_subsets": len(sel["valid_subsets"])}
    for key, o in (("all", ov), ("resonant", res)):
        pool = list(o.values())
        out[key] = {"pool": q(pool), "rule3": q([o[s] for s in rule3]), "accepted": {s: o[s] for s in acc},
                    "accepted_percentile_in_pool": {s: sum(v <= o[s] for v in pool) / len(pool) for s in acc}}
    return out


def build() -> tuple[dict, str]:
    """({file: text} for mlst_v2_tables' OUT, LaTeX macros) of everything above."""
    rows = v4()
    files = {"table_differences.tex": table_v4(rows)}
    a = a1()
    if a:
        buf = io.StringIO()
        write_csv(a, buf)
        files["a1_separation.csv"] = buf.getvalue()
    mac = ["% written by experiments/FIGS/rev35_stats.py (ledger 35)"]
    for r in rows:                         # second pass: the differences the text quotes, Delta +- sigma
        key = TEXT_ROWS.get((r["what"], r["pair"]))
        if key:
            w, k = key
            mac.append(rf"\newcommand{{\RevD{w}}}{{{MT.fmt_pm(abs(k * r['delta']), k * r['sigma'])}}}")
    missing = set(TEXT_ROWS) - {(r["what"], r["pair"]) for r in rows}
    if missing:
        raise SystemExit(f"FATAL: rows the text quotes are missing from the differences table: {sorted(missing)}")
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    for key, v in v6().items():           # V6: the 3-sigma reach of the nested label sets on the benchmarks
        task, mode, met = key.split("|")
        w = ("Top" if task == "top" else "Qg") + ("Lp" if mode == "frozen" else "Ft") + ("Acc" if met == "accuracy" else "Rthirty")
        share = (lambda x: pct(x)) if met == "r30" else (lambda x: f"{100 * x:.2f}")
        mac += [rf"\newcommand{{\RevReach{w}Lo}}{{{share(v['min_3sigma_share'])}}}",
                rf"\newcommand{{\RevReach{w}Hi}}{{{share(v['max_3sigma_share'])}}}"]
        if met == "accuracy":
            mac.append(rf"\newcommand{{\RevSpread{w}Pts}}{{{100 * v['spread']:.2f}}}")
    p7 = v7()
    if p7:
        mac += [rf"\newcommand{{\RevPoolDraws}}{{{MT._fixed(p7['pool_draws'], 0)}}}",
                rf"\newcommand{{\RevPoolRuleThree}}{{{p7['rule3_draws']}}}",
                rf"\newcommand{{\RevPoolOverlapMedianPct}}{{{pct(p7['resonant']['pool']['median'])}}}",
                rf"\newcommand{{\RevPoolOverlapLoPct}}{{{pct(min(p7['resonant']['accepted'].values()))}}}",
                rf"\newcommand{{\RevPoolOverlapHiPct}}{{{pct(max(p7['resonant']['accepted'].values()))}}}",
                rf"\newcommand{{\RevPoolPctileLo}}{{{pct(min(p7['resonant']['accepted_percentile_in_pool'].values()))}}}",
                rf"\newcommand{{\RevPoolPctileHi}}{{{pct(max(p7['resonant']['accepted_percentile_in_pool'].values()))}}}"]
    if a:
        for key, w in (("S", "Sep"), ("S_untrained", "SepUntrained"), ("naive", "SepNaive")):
            mac.append(rf"\newcommand{{\Rev{w}Rho}}{{{a[key]['rho']:.2f}}}")
            mac.append(rf"\newcommand{{\Rev{w}P}}{{{MT.fmt_one(a[key]['p'], 4) if a[key]['p'] >= 1e-4 else '<0.0001'}}}")
    return files, "\n".join(mac) + "\n"


def main() -> int:
    for r in v4():
        w = r["welch"]
        print(f"{r['what'][:34]:34} {r['pair'][:32]:32} d={r['delta']:+.5g} s={r['sigma']:.3g} z={r['z']:+.2f} "
              f"n={r['n_first']}/{r['n_second']} | Welch t={w['t']:+.2f} nu={w['nu']:.1f} crit={w['t_crit']:.2f} "
              f"{'SIG' if w['significant'] else 'ns'} vs SC-1 {'SIG' if abs(r['z']) > 3 else 'ns'}")
    print(json.dumps(v6(), indent=1))
    print(json.dumps(v7(), indent=1))
    a = a1()
    if a:
        for r in sorted(a["rows"], key=lambda r: r["S"]):
            print(f"{r['name']:38} S={r['S']:.4f} S0={r['S_untrained']:.4f} naive={r['naive']:.4f} acc={r['accuracy']:.4f}")
        print({k: a[k] for k in a if k != "rows"})
        if "--csv" in sys.argv:
            write_csv(a, DATA / "stats/a1_separation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
