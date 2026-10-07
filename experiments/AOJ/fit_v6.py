#!/usr/bin/env python3
"""The real-data top fits with ONE peak shape for the pretrained models (fit_v6), the tops
that fail each cut in the fail region as in fit_v5.

WHY (peak_fit.pooled_shape). Floated per score, the Gaussian's mean and width trade against
the transfer factor; on toys the procedure's pulls were far wider than 1 at the smaller
yields and their outliers came from the shape, while at a fixed shape the pulls had mean 0
and width about 1. The scores' own floated shapes are consistent with one shape.

HOW, from fit_v3/bins.npz and fit_v5/results.json:
  1. the reference: fit_v5's (its own floated shape: a different tagger, peak at 34 sigma);
     the tops in each bin from it (peak_fit.tops_from_reference);
  2. the pooled shape over the 30 pretrained models, each at its F-test order at that
     shape, given the tops (peak_fit.pooled_shape), started from fit_v5's pooled shape;
  3. every model, the published checkpoint included, at the pooled shape: order by F-test,
     given the tops; the validation band's background-only fit and toys are fit_v5's
     (they do not involve the signal), the band signal at the pooled shape;
  4. the shape systematic: the pooled mean and width each moved by the spread (SD) of the
     models' own floated values in fit_v5, at each fit's order -- one shift for every model
     together, so it moves the yields coherently; the yield at the model's own floated
     shape (fit_v5) beside it; the leak systematic as in fit_v5;
  5. results.json (fit_v5's schema plus pooled_shape), fit_quality.json, analysis_v6.

Usage (from the repository root):
    python3 experiments/AOJ/fit_v6.py --workers 31
The v2 grid's run (scripts/build_aoj_jobs.py --v2) passes --v2 and its own fit and bins: peak_fit.py's
fit makes fit_v5's fits on the jets (own floated shape, the tops in the fail region), so it stands in
for fit_v5; it has no fit_v4 to carry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("peak_fit", HERE / "peak_fit.py")
# peak_fit.pooled_shape hands its own functions to the process pool: they pickle by module
# name, so the module is registered under it (a spawned worker re-runs this line)
sys.modules.setdefault("peak_fit", P)
RB = _load("refit_from_bins", HERE / "refit_from_bins.py")

DATA = pathlib.Path("experiments/FIGS/data/aoj_full_v1")
KEYS = RB.KEYS
PEAK = "top"
PUBLISHED = P.PUBLISHED
EPS_SYST = (0.6, 0.4)
SHAPE_KEYS = ("pooled_mean_down", "pooled_mean_up", "pooled_width_down", "pooled_width_up")
# the v2 grid scores five checkpoints of every run; its pooled shape is built from the primary
# (A14) alone, so no run is counted once per checkpoint, and every checkpoint is fitted at it
V2_PRIMARY = "best70"


def _bins(z, name, part):
    return {k: z[f"{name}|{part}|{k}"] for k in KEYS}


def fit_one(job):
    """One model at the pooled shape, given the tops: the fit, its band signal, the shape
    and leak systematics."""
    name, b, bv, shape, delta, tops, val_order = job
    fit, hist, _ = P.fit_binned(b, PEAK, *shape, tops=tops)
    order = tuple(fit["tf_order"])
    at = lambda m, w, t=tops: P.fit_binned(b, PEAK, m, w, order=order, tops=t)[0]["signal_yield"]
    (m, w), (dm, dw) = shape, delta
    fit["shape_systematic"] = dict(zip(SHAPE_KEYS, (at(m - dm, w), at(m + dm, w), at(m, w - dw), at(m, w + dw))))
    fit["leak_systematic"] = dict(eps_ref=list(EPS_SYST), signal_yield={f"{e:g}": at(m, w, tops / e) for e in EPS_SYST})
    fit["leak_systematic"]["shift"] = {e: y - fit["signal_yield"] for e, y in fit["leak_systematic"]["signal_yield"].items()}
    band = P.fit_binned(bv, PEAK, *shape, order=tuple(val_order))[0]
    return name, fit, hist, band["z_wald"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bins", default=str(DATA / "fit_v3" / "bins.npz"))
    ap.add_argument("--previous", default=str(DATA / "fit_v5" / "results.json"), help="fit_v5's results")
    ap.add_argument("--out", default=str(DATA / "fit_v6"))
    ap.add_argument("--analysis-out", default=str(DATA / "analysis_v6"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--models", nargs="+", default=None, help="a subset (a test); the run fits every model")
    ap.add_argument("--v2", action="store_true",
                    help="the v2 grid's run (scripts/build_aoj_jobs.py --v2): --previous is its own peak_fit.py "
                         "fit, fit_v5's fits on the jets; the pooled shape is built from the V2_PRIMARY entries "
                         "alone; the readout is per grid arm at each checkpoint")
    ap.add_argument("--shape-from", default=None, metavar="RESULTS_JSON",
                    help="hold the pooled shape and its systematic step of this fit_v6 results.json instead of "
                         "deriving them (the v2 grid's tier 3 holds the freeze run's); its reference fit must be "
                         "this one's, i.e. the same jets")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    for p in (out / "results.json", out / "fit_quality.json", pathlib.Path(a.analysis_out) / "aoj_top.json"):
        if p.exists():
            raise SystemExit(f"FATAL: {p} exists; a result is a record, give a new --out")
    v5 = json.loads(pathlib.Path(a.previous).read_text())
    if not v5.get("fail_tops"):
        raise SystemExit("FATAL: --previous is not a fit with the tops in the fail region (fit_v5)")
    z = np.load(a.bins)
    models = a.models or list(v5["models"])
    pool = [n for n in models if n != PUBLISHED]
    if a.v2:
        pool = [n for n in pool if n.endswith(f"-{V2_PRIMARY}")]
    window = P.PEAKS[PEAK]["window"]

    # 1. the reference and the tops
    ref = v5["reference"][PEAK]
    # tops_from_reference refuses a refit that misses the stored yield by 1e-3 of its error;
    # across machines the refit lands ~1e-4 of the error apart (first launch, 2026-10-01:
    # a 1e-6 relative check here refused it), so that check is the one that holds
    tops = P.tops_from_reference(_bins(z, "reference", "main"), ref)
    bins = {n: _bins(z, n, "main") for n in models}

    # 2. the pooled shape
    own = {n: v5["models"][n][PEAK] for n in models}
    start = tuple(v5["shape_variations"][PEAK]["pooled"])
    held = json.loads(pathlib.Path(a.shape_from).read_text()) if a.shape_from else None
    if held and abs(ref["signal_yield"] - held["reference"][PEAK]["signal_yield"]) > \
            RB.REPRODUCE_TOL * held["reference"][PEAK]["signal_yield_err"]:
        raise SystemExit(f"FATAL: the reference fit gives {ref['signal_yield']:.1f} here and "
                         f"{held['reference'][PEAK]['signal_yield']:.1f} in {a.shape_from}: not the same jets")
    with ProcessPoolExecutor(a.workers) as ex:
        mapper = (lambda f, xs: list(ex.map(f, xs))) if a.workers > 1 else (lambda f, xs: list(map(f, xs)))
        if held:
            ps = held["pooled_shape"]
            shape, orders, trail = (ps["mean"], ps["width"]), {}, []
            spread = (ps["systematic_step"]["mean"], ps["systematic_step"]["width"])
        else:
            shape, orders, trail = P.pooled_shape(bins, pool, window, start, {n: tops for n in pool}, mapper)
            spread = (float(np.std([own[n]["mean"] for n in pool], ddof=1)),
                      float(np.std([own[n]["width"] for n in pool], ddof=1)))
        print(f"pooled shape {shape[0]:.3f} / {shape[1]:.3f} GeV "
              + (f"held from {a.shape_from}" if held else f"after {len(trail)} passes")
              + f"; systematic step {spread[0]:.2f} / {spread[1]:.2f} GeV", flush=True)

        # 3-4. every model at the pooled shape
        jobs = [(n, bins[n], _bins(z, n, "validation"), shape, spread, tops, own[n]["validation"]["tf_order"])
                for n in models]
        rows = mapper(fit_one, jobs)
    fits, hists = {}, {}
    for n, f, h, band_z in rows:
        f.update(floated_mean=f["mean"], floated_width=f["width"], shape_source="pooled")
        f["validation"] = dict(own[n]["validation"], band_signal_z=band_z)
        for k in ("data_efficiency", "auc_vs_cms_proxy"):
            f[k] = own[n][k]
        f["efficiency_relative_to_reference"] = f["signal_yield"] / ref["signal_yield"]
        f["criteria"] = P.criteria(f, PEAK, v5["n_toys"])
        f["own_shape_fit"] = {k: own[n][k] for k in ("mean", "width", "tf_order", "signal_yield", "signal_yield_err")}
        if "fit_v4_signal_yield" in own[n]:            # the first run's chain; a v2 run has no fit_v4
            f["fit_v4_signal_yield"] = own[n]["fit_v4_signal_yield"]
        f["shape_variations"] = dict(
            own_shape=dict(mean=own[n]["mean"], width=own[n]["width"], signal_yield=own[n]["signal_yield"],
                           signal_yield_err=own[n]["signal_yield_err"]),
            **{k: dict(signal_yield=y) for k, y in f["shape_systematic"].items()})
        if n in orders and tuple(f["tf_order"]) != tuple(orders[n]):
            raise SystemExit(f"FATAL: {n}'s order at the pooled shape is {f['tf_order']}, the pool's {orders[n]}")
        fits[n] = f
        hists.update({f"{n}_{PEAK}_{k}": v for k, v in h.items()})

    pooled = dict(mean=shape[0], width=shape[1], pool=pool, passes=trail,
                  systematic_step=dict(mean=spread[0], width=spread[1],
                                       what="SD over the pool of the models' own floated shapes "
                                            "(fit_v5); one shift for every model together"))
    if held:
        pooled = dict(held["pooled_shape"], held_from=a.shape_from, held_from_sha256=RB._sha(a.shape_from))
    res = dict(reference={PEAK: ref}, models={n: {PEAK: fits[n]} for n in models},
               verdict={n: P.decide({PEAK: fits[n]}, v5["closure_hard_flags"], v5["pipeline_ok"])[0] for n in models},
               closure_hard_flags=v5["closure_hard_flags"], pipeline_ok=v5["pipeline_ok"], eff=v5["eff"],
               n_jets=v5["n_jets"], n_toys=v5["n_toys"], peaks=v5["peaks"], fail_tops=v5["fail_tops"],
               pooled_shape=pooled, shape_variations={PEAK: dict(pooled=list(shape), pool=pooled["pool"])},
               refit=dict(bins=a.bins, bins_sha256=RB._sha(a.bins), previous=a.previous,
                          previous_sha256=RB._sha(a.previous), peak_fit_sha256=RB._sha(HERE / "peak_fit.py"),
                          script_sha256=RB._sha(__file__),
                          carried_from_fit_v5=["reference", "validation (but band_signal_z)", "data_efficiency",
                                               "auc_vs_cms_proxy"]))
    shifts = lambda k: [fits[n]["shape_systematic"][k] - fits[n]["signal_yield"] for n in pool]
    quality = dict(
        n_fits=len(fits), all_converged=all(f["converged"] for f in fits.values()),
        max_edm=max(f["edm"] for f in fits.values()), tf_order={n: f["tf_order"] for n, f in fits.items()},
        yield_change_vs_fit_v5={n: fits[n]["signal_yield"] - own[n]["signal_yield"] for n in models},
        shape_systematic_median_shift={k: float(np.median(shifts(k))) for k in SHAPE_KEYS})
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "histograms.npz", **hists)
    (out / "fit_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"wrote {out / 'results.json'}, histograms.npz and fit_quality.json", flush=True)
    if a.models is None:
        summaries = (RB.write_analysis_v2(out / "results.json", a.analysis_out) if a.v2 else
                     {"": RB.write_analysis(out / "results.json", a.analysis_out)})
        for tag, summary in summaries.items():
            for level, s in summary["label_sets"].items():
                print(f"{level:>9s}  yield {s['signal_yield']['mean']:7.0f} +- {s['signal_yield']['sd']:5.0f}  "
                      f"(median stat. error {s['median_stat_err']:4.0f})" + (f"  {tag}" if tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
