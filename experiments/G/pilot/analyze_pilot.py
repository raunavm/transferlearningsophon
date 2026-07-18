#!/usr/bin/env python3
"""G-pilot analyzer (PLAN v5.1 §7.7 a-h) -> reports/pilot_rebaseline.md.

Reads one or more run_pilot.sh output dirs and computes the G1 re-baselining
measurements. The §7.8 mu=0 counterfactual test and the Poisson GoF are pure
functions (unit-tested via --selftest, no ROOT needed); the ntuple-reading
parts run once pilot data + the ntupler extension exist.

§7.7 checklist -> where it is computed here:
  (a) NPU ~ Poisson(mu) GoF          npu_poisson_gof()        [needs NPU branch]
  (b) rerun reference-jet identity   refjet_identity()        [gen jets, mu50 vs mu50b]
  (c) truth-branch survival          branch_survival()        [needs extension]
  (d) loader readability             read via uproot below
  (e) jets/event acceptance a_c      acceptance()
  (f) s/event per mu                 timing.tsv
  (g) bytes/event (raw branches)     timing.tsv
  (h) mu=0 counterfactual test       mu0_counterfactual_test()  §7.8 criteria verbatim

Usage:
  python3 experiments/G/pilot/analyze_pilot.py --selftest
  python3 experiments/G/pilot/analyze_pilot.py --pilot-dir <dir> [<dir> ...] \
      --out reports/pilot_rebaseline.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_TRUTH_BRANCHES = [   # §7.1.3 ntupler extension — validated by (c)
    "event_npu",              # per-event number of pileup interactions
    "part_isPU",              # per-particle pileup-truth flag
    "rawpart_px", "rawpart_py", "rawpart_pz", "rawpart_energy",  # pre-PUPPI EFlow
    "part_puppiw",            # per-particle PUPPI weight (needed for §7.8)
]


def npu_poisson_gof(npu, mu, n_bins_min=5):
    """(a) chi2 GoF of the NPU sample vs Poisson(mu). Returns (chi2, ndf, p)."""
    from scipy import stats
    npu = np.asarray(npu, int)
    ks = np.arange(max(0, int(mu - 5 * np.sqrt(mu))), int(mu + 5 * np.sqrt(mu)) + 1)
    exp = stats.poisson.pmf(ks, mu) * len(npu)
    obs = np.array([(npu == k).sum() for k in ks], float)
    keep = exp >= 5                     # merge sparse tails implicitly by masking
    if keep.sum() < n_bins_min:
        return None
    chi2 = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum())
    ndf = int(keep.sum() - 1)
    return {"chi2": chi2, "ndf": ndf, "p": float(1 - stats.chi2.cdf(chi2, ndf))}


def refjet_hash(pts, etas, phis, ms, decimals=4):
    """(b) per-event hash of the sorted gen-jet four-vector table."""
    import hashlib
    tab = np.round(np.stack([pts, etas, phis, ms], 1), decimals)
    tab = tab[np.lexsort(tab.T[::-1])]
    return hashlib.sha256(tab.tobytes()).hexdigest()


def refjet_identity(events_a, events_b):
    """(b) fraction of events whose gen-jet hash matches between two reruns.
    events_*: iterables of (pts, etas, phis, ms) per event."""
    match = sum(refjet_hash(*a) == refjet_hash(*b) for a, b in zip(events_a, events_b))
    n = min(len(events_a), len(events_b))
    return {"n": n, "match": match, "identical": match == n}


def mu0_counterfactual_test(puppi_w_per_jet, msd_i, msd_ii, msd_iii=None):
    """(h) §7.8 verbatim: PASS iff >=99.9% of jets have max |w_PUPPI - 1| < 1e-3
    AND median |dmSD|/mSD < 1e-3 with 99th pct < 1e-2 (and (iii) ~ (i) when given).

    puppi_w_per_jet: list of per-jet arrays of PUPPI weights (config ii)
    msd_i / msd_ii / msd_iii: per-jet soft-drop mass under configs (i)/(ii)/(iii)
    """
    wmax = np.array([np.max(np.abs(np.asarray(w) - 1.0)) if len(w) else 0.0
                     for w in puppi_w_per_jet])
    frac_w = float(np.mean(wmax < 1e-3))
    rel = np.abs(np.asarray(msd_ii) - np.asarray(msd_i)) / np.clip(np.asarray(msd_i), 1e-9, None)
    med, p99 = float(np.median(rel)), float(np.percentile(rel, 99))
    out = {"frac_jets_wmax_lt_1e-3": frac_w, "dmsd_rel_median": med,
           "dmsd_rel_p99": p99,
           "pass": bool(frac_w >= 0.999 and med < 1e-3 and p99 < 1e-2)}
    if msd_iii is not None:
        rel3 = np.abs(np.asarray(msd_iii) - np.asarray(msd_i)) / np.clip(np.asarray(msd_i), 1e-9, None)
        out["iii_vs_i_median"] = float(np.median(rel3))
        out["iii_consistent"] = bool(np.median(rel3) < 1e-3)
        out["pass"] = out["pass"] and out["iii_consistent"]
    out["consequence"] = ("mu=0 arm = PUPPI config (pipeline-uniform)" if out["pass"]
                          else "mu=0 arm = raw constituents; (i)-(ii) delta carried "
                               "as explicit systematic on every 0->50 number and T2")
    return out


def branch_survival(ntuple_branches):
    """(c) which required truth branches exist in the pilot ntuples."""
    missing = [b for b in REQUIRED_TRUTH_BRANCHES if b not in ntuple_branches]
    return {"required": REQUIRED_TRUTH_BRANCHES, "missing": missing,
            "pass": not missing}


def _selftest():
    from scipy import stats
    rng = np.random.default_rng(0)
    ok = True
    # (a) Poisson sample passes; shifted sample fails
    g = npu_poisson_gof(rng.poisson(50, 20000), 50)
    b = npu_poisson_gof(rng.poisson(56, 20000), 50)
    print(f"  GoF poisson(50) vs 50: p={g['p']:.3f} (expect >0.01) | "
          f"poisson(56) vs 50: p={b['p']:.2e} (expect ~0)")
    ok &= g["p"] > 0.01 and b["p"] < 1e-6
    # (b) identity: same events -> identical; perturbed -> not
    evs = [(rng.uniform(200, 800, 3), rng.normal(0, 2, 3), rng.uniform(-3, 3, 3),
            rng.uniform(10, 200, 3)) for _ in range(50)]
    evs_p = [(p + 1e-3, e, f, m) for p, e, f, m in evs]
    r1, r2 = refjet_identity(evs, evs), refjet_identity(evs, evs_p)
    print(f"  identity self={r1['identical']} perturbed={r2['identical']} (expect True/False)")
    ok &= r1["identical"] and not r2["identical"]
    # (b) permutation invariance of the hash (jet order must not matter)
    p, e, f, m = evs[0]
    perm = np.array([2, 0, 1])
    ok &= refjet_hash(p, e, f, m) == refjet_hash(p[perm], e[perm], f[perm], m[perm])
    # (h) clean case passes; PUPPI-distorted case fails with the right verdict
    w_good = [np.ones(30) + rng.normal(0, 1e-5, 30) for _ in range(5000)]
    msd = rng.uniform(30, 300, 5000)
    t_pass = mu0_counterfactual_test(w_good, msd, msd * (1 + rng.normal(0, 1e-5, 5000)))
    w_bad = [np.clip(rng.beta(5, 1, 30), 0, 1) for _ in range(5000)]
    t_fail = mu0_counterfactual_test(w_bad, msd, msd * (1 + rng.normal(0, 0.02, 5000)))
    print(f"  §7.8 clean: pass={t_pass['pass']} | distorted: pass={t_fail['pass']} "
          f"(expect True/False)")
    ok &= t_pass["pass"] and not t_fail["pass"]
    # (c)
    ok &= not branch_survival(REQUIRED_TRUTH_BRANCHES + ["jet_pt"])["missing"]
    ok &= branch_survival(["jet_pt"])["missing"] == REQUIRED_TRUTH_BRANCHES
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", nargs="*", default=[])
    ap.add_argument("--out", default="reports/pilot_rebaseline.md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()

    # Data-driven path: aggregate timing.tsv (f, g) now; ntuple-based checks
    # (a)-(e), (h) activate once the ntupler extension lands in the pilot dirs.
    rows = []
    for d in map(Path, args.pilot_dir):
        for line in (d / "timing.tsv").read_text().splitlines():
            mu, nev, sec, byt = line.split("\t")
            rows.append({"dir": d.name, "mu": mu, "n_events": int(nev),
                         "s_per_event": float(sec) / int(nev),
                         "bytes_per_event": int(byt) / int(nev)})
    lines = ["# G-pilot re-baseline (§7.7) — PARTIAL (timing/bytes lanes)", "",
             "| batch | mu | s/event | bytes/event |", "|---|---|---|---|"]
    lines += [f"| {r['dir']} | {r['mu']} | {r['s_per_event']:.3f} | "
              f"{r['bytes_per_event']:,.0f} |" for r in rows]
    lines += ["", "Pending ntupler extension: NPU GoF (a), identity (b), "
              "truth survival (c), acceptance (e), §7.8 test (h).",
              "Production volumes freeze only on full acceptance (§7.7)."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
