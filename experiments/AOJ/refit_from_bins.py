#!/usr/bin/env python3
"""Refit the real-data top peak from the bins fit_v3 fitted, the signal shape floating (fit_v4).

WHY. fit_v3 fitted every score with one signal shape, the CMS ParticleNet reference's
Gaussian floated at the TF order (2, 1): 180.05 GeV mean, 17.32 GeV width. No score's
peak has that shape (peak_fit._choose_shape_and_order). peak_fit.py now floats the
mean and width in every fit, profiled at every TF order the F-test compares, and
profiles them in the yield's error; a label-set-blind shape systematic refits
everything with one pooled shape and with the old one.

HOW. The fits see the jets only through their (m_SD, pT) bins, and fit_v3/bins.npz
holds the bins the cluster run fitted (export_fit_bins.py checked them against the
run's histograms). From them:
  1. fit_v3's procedure is replayed and must reproduce fit_v3/results.json -- the
     reference's shape, every TF order (main and validation), every yield and error
     within REPRODUCE_TOL, every validation toy p-value and band signal -- or nothing
     is written. Across machines the minimiser stops a hair apart, so "reproduces"
     cannot mean bit-identical; it means well inside the fit's own precision;
  2. the new procedure, peak_fit.fit_binned(float_shape=True), each score starting
     from the reference's fitted shape; the validation band's background-only fit and
     toys (the signal shape does not enter them) and the band signal at the new shape;
  3. the shape systematic, peak_fit.shape_variations, pooled over the 30 pretrained
     models (the published checkpoint is a reference row, not in the pool);
  4. two checks of the fits: every order and shape again from the POOLED shape, which
     must give the same answer as from the reference's (START_TOL), or nothing is
     written; and each quoted profile error re-derived with another minimiser over
     the shape (crossing_check);
  5. fit_v4/results.json (fit_v3's schema plus the new fields), histograms.npz and
     fit_quality.json, then experiments/STATS/seed_level.py --real-data on it into
     analysis_v4/aoj_top.json -- the code that made analysis_v3 -- to which
     `per_label_set` is added: mean +- sd (n - 1) over the five seeds.
Carried from fit_v3 because they need the jets and do not involve the signal shape:
auc_vs_cms_proxy (scores inside the window) and data_efficiency (all jets).

Usage (from the repository root, so the paths it records are repository-relative):
    python3 experiments/AOJ/refit_from_bins.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import optimize

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("peak_fit", HERE / "peak_fit.py")

DATA = pathlib.Path("experiments/FIGS/data/aoj_full_v1")
KEYS = ("m_edges", "i", "j", "n_pass", "n_fail", "rho", "pt")
PEAK = "top"
PUBLISHED = P.PUBLISHED
REPRODUCE_TOL = 1e-3                # yield shift / error, and relative error change
# The band signal is a ~2.5 sigma bump fitted at order (4, 3) in the fail band, not a
# reported number; across machines its z moved by up to 1.03e-3 (the reference's).
BAND_Z_TOL = 1e-2
# "One answer" from two starting shapes: the same order, and the shape and yield equal
# well inside the shape polish's precision (Powell, xtol 0.01 GeV) and the error.
START_TOL = dict(shape_gev=0.1, yield_over_err=0.1)
LABEL_SETS = ("188", "162", "43", "17", "162+mass", "17+mass")


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _map(fn, jobs, workers):
    if workers == 1:
        return list(map(fn, jobs))
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


def old_fit(job):
    """fit_v3's procedure on one score: the shape fixed (the reference's, floated at
    START_ORDER), the order by F-test; and the diagnostic float at START_ORDER."""
    name, b, bv, shape, n_toys = job
    window = P.PEAKS[PEAK]["window"]
    floated = P._float_shape(b, P._tf_norm(b, window), P.START_ORDER, window)
    fit = P.fit_binned(b, PEAK, *(shape or floated))[0]
    fit["validation"] = P.validation_binned(bv, PEAK, n_toys, shape=(fit["mean"], fit["width"]))[0]
    return name, dict(fit, floated=list(floated))


def new_fit(job):
    """The new procedure on one score, as peak_fit.analyse runs it after the cut."""
    name, b, bv, start, n_toys = job
    fit, hist, _ = P.fit_binned(b, PEAK, *(start or (None, None)), float_shape=True)
    fit["floated_mean"], fit["floated_width"] = fit["mean"], fit["width"]
    fit["validation"], hist_v = P.validation_binned(bv, PEAK, n_toys, shape=(fit["mean"], fit["width"]))
    return name, fit, dict(hist, **{f"validation_{k}": v for k, v in hist_v.items()})


def restart(job):
    """The new procedure's order and shape on one score from another starting shape,
    and the yield there."""
    name, b, _, start, _ = job
    window = P.PEAKS[PEAK]["window"]
    order, shape, _ = P._choose_shape_and_order(b, P._tf_norm(b, window), window, [start])
    fit = P.fit_binned(b, PEAK, *shape, order=order)[0]
    return name, dict(start=list(start), tf_order=list(order), mean=shape[0], width=shape[1],
                      signal_yield=fit["signal_yield"])


def one_answer(fit, other):
    """Does `other` (restart) give `fit`'s answer, within START_TOL?"""
    d = dict(other, same_order=other["tf_order"] == fit["tf_order"],
             shape_shift_gev=max(abs(other["mean"] - fit["mean"]), abs(other["width"] - fit["width"])),
             yield_shift_over_err=(other["signal_yield"] - fit["signal_yield"]) / fit["signal_yield_err"])
    d["one_answer"] = bool(d["same_order"] and d["shape_shift_gev"] <= START_TOL["shape_gev"]
                           and abs(d["yield_shift_over_err"]) <= START_TOL["yield_over_err"])
    return d


def crossing_check(job):
    """Twice the rise of the loss, profiled over the TF and the shape, at the quoted
    crossings -- 1 if the quoted errors are right -- re-derived with another minimiser
    over the shape (Nelder-Mead from an offset start, peak_fit uses Powell from the
    fitted shape). Also how far the profile's minimum lies below the fit's (<= 0 if
    the fitted shape is the minimum). None without a profile error."""
    name, b, fit = job
    if not fit["profile_error_ok"]:
        return name, None
    order, window = tuple(fit["tf_order"]), P.PEAKS[PEAK]["window"]
    tf_norm = P._tf_norm(b, window)
    x = P._Model(b, order, tf_norm, fit["mean"], fit["width"]).fit()[0]

    def prof(y):
        at = lambda v: P._Model(b, order, tf_norm, v[0], v[1]).fit_at_yield(y, x)[1]
        return optimize.minimize(at, (fit["mean"] + 0.7, fit["width"] - 0.7), method="Nelder-Mead",
                                 options=dict(xatol=1e-4, fatol=1e-10)).fun
    y0 = fit["signal_yield"]
    f0 = prof(y0)
    return name, dict(twice_rise=[2 * (prof(y0 + s * e) - f0) for s, e in
                                  ((-1, fit["signal_yield_err_lo"]), (1, fit["signal_yield_err_hi"]))],
                      fit_minus_profile_minimum=fit["deviance"] / 2 - f0)


def reproduction(old, stored):
    """Per fit, the replayed fit_v3 procedure against fit_v3; `ok` only if it is fit_v3.
    Toy p-values are compared only where toys were run."""
    rows = {}
    for name, o in old.items():
        s, ov, sv = stored[name], old[name]["validation"], stored[name]["validation"]
        r = dict(same_order=o["tf_order"] == s["tf_order"],
                 same_validation_order=ov["tf_order"] == sv["tf_order"],
                 yield_shift_over_err=(o["signal_yield"] - s["signal_yield"]) / s["signal_yield_err"],
                 err_ratio_minus_1=o["signal_yield_err"] / s["signal_yield_err"] - 1,
                 band_signal_z_shift=ov["band_signal_z"] - sv["band_signal_z"],
                 floated_shape_shift_gev=max(abs(o["floated"][0] - s.get("floated_mean", s["mean"])),
                                             abs(o["floated"][1] - s.get("floated_width", s["width"]))),
                 toy_p=None if ov["toy_p"] is None else [ov["toy_p"], sv["toy_p"]])
        r["ok"] = bool(r["same_order"] and r["same_validation_order"]
                       and abs(r["yield_shift_over_err"]) <= REPRODUCE_TOL
                       and abs(r["err_ratio_minus_1"]) <= REPRODUCE_TOL
                       and abs(r["band_signal_z_shift"]) <= BAND_Z_TOL
                       and r["floated_shape_shift_gev"] <= 0.01            # Powell's xtol
                       and (r["toy_p"] is None or r["toy_p"][0] == r["toy_p"][1]))
        rows[name] = r
    return rows


def per_label_set(res, S):
    """Mean +- sd (n - 1) over the pretraining seeds of each label set -- how the paper
    reports every result, with no test -- of the yield under the fitted shape and under
    each shape variation, and of the fitted mean and width; the median per-model
    statistical error; the reference and the published checkpoint as single rows."""
    msd = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
    row = lambda f: dict(signal_yield=f["signal_yield"], signal_yield_err=f["signal_yield_err"],
                         mean_gev=f["mean"], width_gev=f["width"],
                         by_shape={k: v["signal_yield"] for k, v in f["shape_variations"].items()})
    groups = {}
    for name, per_peak in res["models"].items():
        m = S.AOJ_ARM_RE.match(name)
        if m:
            groups.setdefault(str(S.AOJ_CELL[m.group(1)]), []).append((int(m.group(2)), name, per_peak[PEAK]))
    out = {}
    for level in LABEL_SETS:
        g = sorted(groups[level])
        f = [x for _, _, x in g]
        out[level] = dict(models=[n for _, n, _ in g], signal_yields=[x["signal_yield"] for x in f],
                          signal_yield=msd([x["signal_yield"] for x in f]),
                          median_stat_err=float(np.median([x["signal_yield_err"] for x in f])),
                          mean_gev=msd([x["mean"] for x in f]), width_gev=msd([x["width"] for x in f]),
                          by_shape={k: msd([x["shape_variations"][k]["signal_yield"] for x in f])
                                    for k in f[0]["shape_variations"]})
    return dict(method="mean and standard deviation (n - 1) over the pretraining seeds; no test. "
                       "by_shape: the same fits with one fixed shape (peak_fit.shape_variations)",
                shapes=res["shape_variations"][PEAK], label_sets=out,
                reference=row(res["reference"][PEAK]),
                published_188_class_checkpoint=row(res["models"][PUBLISHED][PEAK]),
                script=str(pathlib.Path(__file__).resolve().relative_to(REPO)), script_sha256=_sha(__file__))


def write_analysis(results_path, out_dir):
    """seed_level.py --real-data exactly as analysis_v3 was made, then per_label_set."""
    S = _load("seed_level", REPO / "experiments/STATS/seed_level.py")
    if S.main(["--real-data", str(results_path), "--out", str(out_dir)]):
        raise SystemExit("FATAL: seed_level.py --real-data failed")
    path = pathlib.Path(out_dir) / "aoj_top.json"
    doc = json.loads(path.read_text())
    doc["per_label_set"] = per_label_set(json.loads(pathlib.Path(results_path).read_text()), S)
    path.write_text(json.dumps(doc, indent=2))
    return doc["per_label_set"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bins", default=str(DATA / "fit_v3" / "bins.npz"))
    ap.add_argument("--results", default=str(DATA / "fit_v3" / "results.json"), help="fit_v3's, to reproduce")
    ap.add_argument("--out", default=str(DATA / "fit_v4"))
    ap.add_argument("--analysis-out", default=str(DATA / "analysis_v4"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    for p in (out / "results.json", out / "histograms.npz", out / "fit_quality.json",
              pathlib.Path(a.analysis_out) / "aoj_top.json"):
        if p.exists():
            raise SystemExit(f"FATAL: {p} exists; a result is a record, give a new --out")
    v3 = json.loads(pathlib.Path(a.results).read_text())
    z = np.load(a.bins)
    n_toys = v3["n_toys"]
    stored = {"reference": v3["reference"][PEAK], **{n: m[PEAK] for n, m in v3["models"].items()}}
    models = list(v3["models"])
    job = lambda n, start: (n, *({k: z[f"{n}|{p}|{k}"] for k in KEYS} for p in ("main", "validation")), start, n_toys)

    # 1. fit_v3's procedure must give fit_v3
    ref_old = old_fit(job("reference", None))[1]
    old = dict([("reference", ref_old)] + _map(old_fit, [job(n, ref_old["floated"]) for n in models], a.workers))
    rows = reproduction(old, stored)
    bad = [n for n, r in rows.items() if not r["ok"]]
    if bad:
        raise SystemExit(f"FATAL: fit_v3's procedure replayed from {a.bins} does not reproduce "
                         f"{a.results} for {bad}; nothing written")
    print(f"fit_v3 reproduced: {len(rows)} fits, largest |yield shift| "
          f"{max(abs(r['yield_shift_over_err']) for r in rows.values()):.1e} of its error", flush=True)

    # 2. the new procedure
    _, ref, ref_hist = new_fit(job("reference", None))
    new = [("reference", ref, ref_hist)] + _map(new_fit, [job(n, (ref["mean"], ref["width"])) for n in models],
                                                  a.workers)
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

    # 3. the shape systematic
    pool = [n for n in models if n != PUBLISHED]
    variations = P.shape_variations({n: job(n, None)[1] for n in fits}, fits, pool, ref["shape_start"], PEAK)

    # 4. one answer from another start, and the profile errors re-derived
    for n, other in _map(restart, [job(n, variations["pooled"]) for n in fits], a.workers):
        fits[n]["start_check"] = one_answer(fits[n], other)
    start_dependent = [n for n, f in fits.items() if not f["start_check"]["one_answer"]]
    if start_dependent:
        raise SystemExit(f"FATAL: from the pooled shape {start_dependent} give another answer; nothing written")
    crossings = dict(_map(crossing_check, [(n, job(n, None)[1], f) for n, f in fits.items()], a.workers))
    rises = [r for c in crossings.values() if c for r in c["twice_rise"]]
    rise_range = [min(rises), max(rises)] if rises else None

    pipeline_ok = ref["ok"]
    res = dict(reference={PEAK: ref}, models={n: {PEAK: fits[n]} for n in models},
               verdict={n: P.decide({PEAK: fits[n]}, v3["closure_hard_flags"], pipeline_ok)[0] for n in models},
               closure_hard_flags=v3["closure_hard_flags"], pipeline_ok=pipeline_ok, eff=v3["eff"],
               n_jets=v3["n_jets"], n_toys=n_toys, peaks=v3["peaks"], shape_variations={PEAK: variations},
               refit=dict(bins=a.bins, bins_sha256=_sha(a.bins), fit_v3_results=a.results,
                          fit_v3_results_sha256=_sha(a.results), peak_fit_sha256=_sha(HERE / "peak_fit.py"),
                          script_sha256=_sha(__file__), carried_from_fit_v3=["auc_vs_cms_proxy", "data_efficiency"],
                          reproduces_fit_v3=True, reproduce_tol=REPRODUCE_TOL, band_z_tol=BAND_Z_TOL,
                          reproduction=rows,
                          width_at_bound=[n for n, f in fits.items() if f["width_at_bound"]],
                          mean_at_bound=[n for n, f in fits.items() if f["mean_at_bound"]],
                          start_check=dict(starts="the reference's fitted shape (the reference: its own shape "
                                                  "floated at START_ORDER), and the pooled shape",
                                           tol=START_TOL, start_dependent=start_dependent),
                          no_profile_error=[n for n, f in fits.items() if not f["profile_error_ok"]]))
    quality = dict(
        n_fits=len(fits), all_converged=all(f["converged"] for f in fits.values()),
        max_edm=max(f["edm"] for f in fits.values()),
        profile_error_check=dict(
            what="twice the rise of the loss profiled over TF and shape at each quoted crossing, "
                 "by Nelder-Mead over the shape (refit_from_bins.crossing_check); 1 if the error is right",
            twice_rise_range=rise_range,
            max_fit_minus_profile_minimum=max((c["fit_minus_profile_minimum"] for c in crossings.values() if c),
                                              default=None),
            per_fit=crossings),
        n_no_profile_error=len(res["refit"]["no_profile_error"]),
        n_width_or_mean_at_bound=sum(f["width_at_bound"] or f["mean_at_bound"] for f in fits.values()),
        n_start_dependent=len(start_dependent), start_tol=START_TOL,
        max_start_shape_shift_gev=max(f["start_check"]["shape_shift_gev"] for f in fits.values()),
        max_abs_start_yield_shift_over_err=max(abs(f["start_check"]["yield_shift_over_err"]) for f in fits.values()),
        tf_order={n: f["tf_order"] for n, f in fits.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "histograms.npz", **hists)
    (out / "fit_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"wrote {out / 'results.json'}, histograms.npz and fit_quality.json; width at a bound: "
          f"{res['refit']['width_at_bound'] or 'none'}; no profile error: {res['refit']['no_profile_error'] or 'none'}; "
          f"twice the rise at the quoted crossings: {rise_range}")

    # 5. the registered readout, and the numbers the paper reports
    summary = write_analysis(out / "results.json", a.analysis_out)
    for level, s in summary["label_sets"].items():
        print(f"{level:>9s}  yield {s['signal_yield']['mean']:7.0f} +- {s['signal_yield']['sd']:5.0f}  "
              f"(median stat. error {s['median_stat_err']:4.0f})  "
              + "  ".join(f"{k} {v['mean']:7.0f}" for k, v in s["by_shape"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
