#!/usr/bin/env python3
"""The real-data top fits again (fit_v5), from the bins fit_v3 fitted, with the tops that
fail each score's cut in the fail region.

WHY. fit_v4's model put signal in the pass region only. Tops in the fail region are part
of the fail counts, q absorbs them, and the pass background TF * q carries TF * F of them:
every yield came out low. These taggers keep a few per cent of the data tops (their yields
are 0.03-0.11 of the CMS reference's at the same data efficiency), so nearly every top
fails the cut. peak_fit._Model now takes the tops in each bin; the fewest there can be are
those the reference passes (peak_fit.tops_from_reference, EPS_REF = 1), and that bound is
the fit. Fewer reference-passed tops than tops in all -- eps_ref < 1 -- would raise every
yield further: the one-sided systematic, each fit again at its order and shape with the
tops scaled by 1 / eps_ref for eps_ref in EPS_SYST.

HOW, from fit_v3/bins.npz (the bins the cluster run fitted, export_fit_bins.py):
  1. the reference: the fit_v4 procedure (the reference has no fail-region term at
     EPS_REF = 1); it must reproduce fit_v4's reference within REPRODUCE_TOL, or nothing
     is written;
  2. the tops per bin from it; every score's bins must be the same cells;
  3. each score: peak_fit.fit_binned(float_shape=True, tops=...) from the reference's
     shape, its validation band (background only, so as fit_v4's) and band signal;
  4. the shape systematic (peak_fit.shape_variations, given the tops), the start check
     from the pooled shape, and the leak systematic;
  5. results.json (fit_v4's schema plus fail_tops and leak_systematic), histograms.npz,
     fit_quality.json, then refit_from_bins.write_analysis into analysis_v5.
Carried from fit_v4, which carried them from fit_v3 (they need the jets and not the fit):
auc_vs_cms_proxy, data_efficiency, the validation band's edges.

Usage (from the repository root):
    python3 experiments/AOJ/fit_v5.py --workers 15
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("peak_fit", HERE / "peak_fit.py")
RB = _load("refit_from_bins", HERE / "refit_from_bins.py")

DATA = pathlib.Path("experiments/FIGS/data/aoj_full_v1")
KEYS = RB.KEYS
PEAK = "top"
PUBLISHED = P.PUBLISHED
REPRODUCE_TOL = RB.REPRODUCE_TOL
EPS_SYST = (0.6, 0.4)


def _bins(z, name, part):
    return {k: z[f"{name}|{part}|{k}"] for k in KEYS}


def fit_one(job):
    """The procedure on one score, with the tops in the fail region; its validation."""
    name, b, bv, start, n_toys, tops = job
    fit, hist, _ = P.fit_binned(b, PEAK, *start, float_shape=True, tops=tops)
    fit["floated_mean"], fit["floated_width"] = fit["mean"], fit["width"]
    fit["validation"], hist_v = P.validation_binned(bv, PEAK, n_toys, shape=(fit["mean"], fit["width"]))
    return name, fit, dict(hist, **{f"validation_{k}": v for k, v in hist_v.items()})


def restart(job):
    """Order and shape from another starting shape, and the yield there."""
    name, b, start, tops = job
    window = P.PEAKS[PEAK]["window"]
    order, shape, _ = P._choose_shape_and_order(b, P._tf_norm(b, window), window, [start], tops)
    fit = P.fit_binned(b, PEAK, *shape, order=order, tops=tops)[0]
    return name, dict(start=list(start), tf_order=list(order), mean=shape[0], width=shape[1],
                      signal_yield=fit["signal_yield"])


def leak_systematic(job):
    """The yield at the fit's order and shape with the tops scaled by 1 / eps_ref."""
    name, b, fit, tops = job
    return name, {f"{e:g}": P.fit_binned(b, PEAK, fit["mean"], fit["width"], order=tuple(fit["tf_order"]),
                                         tops=tops / e)[0]["signal_yield"] for e in EPS_SYST}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bins", default=str(DATA / "fit_v3" / "bins.npz"))
    ap.add_argument("--previous", default=str(DATA / "fit_v4" / "results.json"), help="fit_v4's results")
    ap.add_argument("--out", default=str(DATA / "fit_v5"))
    ap.add_argument("--analysis-out", default=str(DATA / "analysis_v5"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--models", nargs="+", default=None, help="a subset (a test); the run fits every model")
    ap.add_argument("--toys", type=int, default=None, help="validation toys (default: fit_v4's); a test sets 0")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    for p in (out / "results.json", out / "histograms.npz", out / "fit_quality.json",
              pathlib.Path(a.analysis_out) / "aoj_top.json"):
        if p.exists():
            raise SystemExit(f"FATAL: {p} exists; a result is a record, give a new --out")
    v4 = json.loads(pathlib.Path(a.previous).read_text())
    z = np.load(a.bins)
    n_toys = v4["n_toys"] if a.toys is None else a.toys
    models = a.models or list(v4["models"])
    stored = {"reference": v4["reference"][PEAK], **{n: v4["models"][n][PEAK] for n in models}}

    # 1. the reference, as fit_v4 fitted it
    _, ref, ref_hist = fit_one(("reference", _bins(z, "reference", "main"), _bins(z, "reference", "validation"),
                                (None, None), n_toys, None))
    old = stored["reference"]
    shift = (ref["signal_yield"] - old["signal_yield"]) / old["signal_yield_err"]
    same_p = n_toys != v4["n_toys"] or ref["validation"]["toy_p"] == old["validation"]["toy_p"]
    if ref["tf_order"] != old["tf_order"] or abs(shift) > REPRODUCE_TOL or not same_p:
        raise SystemExit(f"FATAL: the reference does not reproduce fit_v4 (order {ref['tf_order']} vs "
                         f"{old['tf_order']}, yield shift {shift:.2e} of its error); nothing written")

    # 2. the tops in each bin
    b_ref = _bins(z, "reference", "main")
    tops = P.tops_from_reference(b_ref, ref)
    for n in models:
        if not P.same_cells(_bins(z, n, "main"), b_ref):
            raise SystemExit(f"FATAL: {n}'s bins are not the reference's cells")
    print(f"reference reproduced (yield shift {shift:.1e} of its error); tops in the fit bins: {tops.sum():.0f}",
          flush=True)

    # 3. every score with the tops
    start = (ref["mean"], ref["width"])
    jobs = [(n, _bins(z, n, "main"), _bins(z, n, "validation"), start, n_toys, tops) for n in models]
    new = [("reference", ref, ref_hist)] + RB._map(fit_one, jobs, a.workers)
    fits = {n: f for n, f, _ in new}
    hists = {f"{n}_{PEAK}_{k}": v for n, _, h in new for k, v in h.items()}
    ref["ok"] = bool(ref["z_wald"] >= P.PEAKS[PEAK]["reference_z"])
    for n, f in fits.items():
        f["data_efficiency"] = stored[n]["data_efficiency"]
        f["validation"]["band"] = stored[n]["validation"]["band"]
        if n != "reference":
            f["efficiency_relative_to_reference"] = f["signal_yield"] / ref["signal_yield"]
            f["auc_vs_cms_proxy"] = stored[n]["auc_vs_cms_proxy"]
            f["criteria"] = P.criteria(f, PEAK, n_toys)
            f["fit_v4_signal_yield"] = stored[n]["signal_yield"]

    # 4. the shape systematic, the start check, the leak systematic
    pool = [n for n in models if n != PUBLISHED]
    bins = {n: _bins(z, n, "main") for n in fits}
    variations = P.shape_variations(bins, fits, pool, ref["shape_start"], PEAK, {n: tops for n in models})
    for n, other in RB._map(restart, [(n, bins[n], variations["pooled"], tops if n != "reference" else None)
                                      for n in fits], a.workers):
        fits[n]["start_check"] = RB.one_answer(fits[n], other)
    start_dependent = [n for n, f in fits.items() if not f["start_check"]["one_answer"]]
    if start_dependent:
        raise SystemExit(f"FATAL: from the pooled shape {start_dependent} give another answer; nothing written")
    for n, ys in RB._map(leak_systematic, [(n, bins[n], fits[n], tops) for n in models], a.workers):
        fits[n]["leak_systematic"] = dict(eps_ref=list(EPS_SYST), signal_yield=ys,
                                          shift={e: y - fits[n]["signal_yield"] for e, y in ys.items()})

    pipeline_ok = ref["ok"]
    res = dict(reference={PEAK: ref}, models={n: {PEAK: fits[n]} for n in models},
               verdict={n: P.decide({PEAK: fits[n]}, v4["closure_hard_flags"], pipeline_ok)[0] for n in models},
               closure_hard_flags=v4["closure_hard_flags"], pipeline_ok=pipeline_ok, eff=v4["eff"],
               n_jets=v4["n_jets"], n_toys=n_toys, peaks=v4["peaks"], shape_variations={PEAK: variations},
               fail_tops=dict(source="the reference's fitted signal per bin (peak_fit.tops_from_reference)",
                              eps_ref=P.EPS_REF, total=float(tops.sum()),
                              systematic=f"one-sided: eps_ref in {list(EPS_SYST)} raises every yield by "
                                         "leak_systematic.shift"),
               refit=dict(bins=a.bins, bins_sha256=RB._sha(a.bins), previous=a.previous,
                          previous_sha256=RB._sha(a.previous), peak_fit_sha256=RB._sha(HERE / "peak_fit.py"),
                          script_sha256=RB._sha(__file__),
                          carried_from_fit_v4=["auc_vs_cms_proxy", "data_efficiency", "validation.band"],
                          reference_reproduces_fit_v4=True, reference_yield_shift_over_err=shift,
                          width_at_bound=[n for n, f in fits.items() if f["width_at_bound"]],
                          mean_at_bound=[n for n, f in fits.items() if f["mean_at_bound"]],
                          start_check=dict(starts="the reference's fitted shape, and the pooled shape",
                                           tol=RB.START_TOL, start_dependent=start_dependent),
                          no_profile_error=[n for n, f in fits.items() if not f["profile_error_ok"]]))
    quality = dict(
        n_fits=len(fits), all_converged=all(f["converged"] for f in fits.values()),
        max_edm=max(f["edm"] for f in fits.values()),
        n_no_profile_error=len(res["refit"]["no_profile_error"]),
        n_width_or_mean_at_bound=sum(f["width_at_bound"] or f["mean_at_bound"] for f in fits.values()),
        n_start_dependent=len(start_dependent), start_tol=RB.START_TOL,
        max_start_shape_shift_gev=max(f["start_check"]["shape_shift_gev"] for f in fits.values()),
        max_abs_start_yield_shift_over_err=max(abs(f["start_check"]["yield_shift_over_err"]) for f in fits.values()),
        tf_order={n: f["tf_order"] for n, f in fits.items()},
        yield_change_vs_fit_v4={n: fits[n]["signal_yield"] - stored[n]["signal_yield"] for n in models})
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "histograms.npz", **hists)
    (out / "fit_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"wrote {out / 'results.json'}, histograms.npz and fit_quality.json", flush=True)
    if a.models is None:
        summary = RB.write_analysis(out / "results.json", a.analysis_out)
        for level, s in summary["label_sets"].items():
            print(f"{level:>9s}  yield {s['signal_yield']['mean']:7.0f} +- {s['signal_yield']['sd']:5.0f}  "
                  f"(median stat. error {s['median_stat_err']:4.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
