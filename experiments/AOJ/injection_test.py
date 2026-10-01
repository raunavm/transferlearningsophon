#!/usr/bin/env python3
"""Signal injection into the real-data top fit: is a known signal recovered without bias?

The checks of 2026-09-30 (realdata_checks.py, injection step) added a Gaussian to the
passing data of a signal-free pseudo-window (250-310 GeV) and fitted it back with the
shape floating: recovered at a median of 0.72 of its size, mean pull -1.7. The fits see
the jets only through their (m_SD, pT) bins, so everything here runs from bins.

  bins    from the merged jets of the first run (the scores the main fit saw): for every
          score, the pseudo-window bins at 1 % (the map with the pseudo window masked, as
          realdata_checks._injection_one builds them) and the top bins at the extra working
          points EXTRA_EFF. The top bins at 1 % are rebuilt too and must equal the committed
          fit_v3/bins.npz, or nothing is written: then these are the jets the fit saw.

Usage (in a job, after merge_shards.py over the first run's shards):
    python3 experiments/AOJ/injection_test.py bins --merged /scratch/merged \\
        --committed experiments/FIGS/data/aoj_full_v1/fit_v3/bins.npz --out OUT/injection_bins.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
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
RC = _load("realdata_checks", HERE / "realdata_checks.py")

KEYS = ("m_edges", "i", "j", "n_pass", "n_fail", "rho", "pt")
EFF = 0.01
EXTRA_EFF = (0.005, 0.02)
PSEUDO = RC.PSEUDO
TOP = P.PEAKS["top"]
BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pseudo_bins(z, mass, pt, eff=EFF):
    """The pseudo-window bins exactly as realdata_checks._injection_one makes them."""
    masked = P.MASKED + (PSEUDO["window"],)
    passed = P.passes(z, mass, pt, P.build_map(z, mass, pt, eff, masked=masked))
    return P._bins(mass, pt, passed, PSEUDO["fit_range"])


def top_bins(z, mass, pt, eff):
    """The top bins at data efficiency `eff`, as peak_fit.analyse makes them."""
    return P._bins(mass, pt, P.passes(z, mass, pt, P.build_map(z, mass, pt, eff)), TOP["fit_range"])


_W: dict = {}


def _init(merged):
    _W["data"] = RC.load_data(pathlib.Path(merged), kinds=("three_prong",))


def _export_one(name):
    d = _W["data"]
    z = P.logit(d["pn"]) if name == "reference" else d["scores"][name]["three_prong"].astype(float)
    m, pt = d["mass"], d["pt"]
    parts = {"main": top_bins(z, m, pt, EFF), "pseudo": pseudo_bins(z, m, pt)}
    parts.update({f"main_eff{e:g}": top_bins(z, m, pt, e) for e in EXTRA_EFF})
    return name, parts


def export(merged, committed, out, workers):
    c = np.load(committed)
    names = sorted({k.split("|")[0] for k in c.files})
    saved = {k: os.environ.get(k) for k in BLAS}
    os.environ.update({k: "1" for k in BLAS})
    try:
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(merged),)) as pool:
                rows = pool.map(_export_one, names, chunksize=1)
        else:
            _init(merged)
            rows = [_export_one(n) for n in names]
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    arrays = {}
    for name, parts in rows:
        for k in KEYS:
            if not np.array_equal(parts["main"][k], c[f"{name}|main|{k}"]):
                raise SystemExit(f"FATAL: {name} top bins at 1 % differ from {committed} ({k}); "
                                 "these are not the jets the fit saw; nothing written")
        for part, b in parts.items():
            if part != "main":
                arrays.update({f"{name}|{part}|{k}": b[k] for k in KEYS})
        print(f"{name:14s} pseudo {len(parts['pseudo']['n_pass'])} bins {parts['pseudo']['n_pass'].sum():.0f} pass; "
              + "; ".join(f"top at {e:g}: {parts[f'main_eff{e:g}']['n_pass'].sum():.0f} pass" for e in EXTRA_EFF)
              + "; top at 1 % = committed", flush=True)
    out = pathlib.Path(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"wrote {out}: {len(names)} scores x (pseudo, {', '.join(f'main_eff{e:g}' for e in EXTRA_EFF)})")


# ------------------------------------------------------------------ toys
DATA = pathlib.Path("experiments/FIGS/data/aoj_full_v1")
REGIONS = ("top", "band", "pseudo", *(f"top_eff{e:g}" for e in EXTRA_EFF))
MODES = ("bootstrap", "data", "leak")
VARIANTS = ("full", "float", "fixed", "fixed_ftest")


def region_bins(region, name, committed, extra):
    """The bins of `region` for score `name`: top and band (the validation band) at 1 % from
    the committed bins, the pseudo window and the other working points from `extra`."""
    src, part = dict(top=(committed, "main"), band=(committed, "validation"),
                     pseudo=(extra, "pseudo")).get(region, (extra, region.replace("top_", "main_")))
    return {k: src[f"{name}|{part}|{k}"] for k in KEYS}


def region_peak(region):
    return PSEUDO if region == "pseudo" else "top"


def _per_pt(fit):
    return np.array([max(fit["yield_per_pt_bin"].get(f"{P.PT_EDGES[j]:g}-{P.PT_EDGES[j + 1]:g}", 0.0), 0.0)
                     for j in range(len(P.PT_EDGES) - 1)])


def truth(region, name, b, top_fit, start):
    """The generator of one score's toys in one region: a background with no signal, and a
    signal template that sums to 1 over the region's bins.
      top       the main fit's own model (top_fit: its order and floated shape), signal removed
      top_eff*  the procedure (peak_fit.fit_binned, shape floating) on that working point's data
      band, pseudo  a background-only fit, order by F-test, as validation() makes it
    The template is a Gaussian of the top fit's shape (the pseudo window: centred in it, the
    top fit's width, as realdata_checks injects) and pT split; at another working point the
    shape and split fitted there."""
    window = P._peak_cfg(region_peak(region))["window"]
    if region == "top":
        model = P._Model(b, tuple(top_fit["tf_order"]), P._tf_norm(b, window), top_fit["mean"], top_fit["width"])
        x, _ = model.fit()
        fit = top_fit
    elif region.startswith("top_eff"):
        fit, _, (model, x) = P.fit_binned(b, "top", *start, float_shape=True)
    else:
        fit = None
        _, _, (model, x) = P.fit_binned(b, region_peak(region))
    t, _, q, _ = model.expect(x)
    src = fit if fit is not None else top_fit
    mean = 0.5 * sum(window) if region == "pseudo" else src["mean"]
    g = P._gauss_bins(b["m_edges"], mean, src["width"])[b["i"]] * _per_pt(src)[b["j"]]
    return dict(background=t * q, fail=q, template=g / g.sum(), order=tuple(model.args[0]),
                mean=float(mean), width=float(src["width"]))


def toy_bins(b, tr, size, mode, rng, eps=None):
    """One pseudo-experiment. bootstrap: Poisson pass and fail around the generator, plus
    the injected signal; data: the real counts plus a Poisson-fluctuated injected signal;
    leak: bootstrap, with the signal that FAILS the cut added to the fail region -- a
    tagger of signal efficiency eps leaves size * (1 - eps) / eps behind there."""
    s = size * tr["template"]
    if mode == "data":
        return dict(b, n_pass=b["n_pass"] + rng.poisson(s).astype(float))
    fail = tr["fail"] + (s * (1 - eps) / eps if mode == "leak" else 0.0)
    return dict(b, n_pass=rng.poisson(tr["background"] + s).astype(float), n_fail=rng.poisson(fail).astype(float))


def _hessian_yield(model, x):
    v = model.G.sum(axis=0)
    cov = model.covariance(x)[model.n_tf:, model.n_tf:]
    return float(v @ x[model.n_tf:]), float(np.sqrt(max(v @ cov @ v, 0.0)))


def fit_variant(variant, b, region, tr, start):
    """One fit of the toy `b`:
      full         the procedure: peak_fit.fit_binned, shape floating, order by F-test
      float        the shape floating at the generator's order, profile error
      fixed        the generator's shape and order
      fixed_ftest  the generator's shape, order by F-test"""
    peak = region_peak(region)
    window = P._peak_cfg(peak)["window"]
    if variant == "full":
        f = P.fit_binned(b, peak, *start, float_shape=True)[0]
        return dict(y=f["signal_yield"], err=f["signal_yield_err"], lo=f["signal_yield_err_lo"],
                    hi=f["signal_yield_err_hi"], mean=f["mean"], width=f["width"], order=f["tf_order"],
                    at_bound=bool(f["mean_at_bound"] or f["width_at_bound"]))
    if variant == "fixed_ftest":
        f = P.fit_binned(b, peak, tr["mean"], tr["width"])[0]
        return dict(y=f["signal_yield"], err=f["signal_yield_err"], order=f["tf_order"])
    tf_norm = P._tf_norm(b, window)
    shape = (tr["mean"], tr["width"])
    if variant == "float":
        shape = P._float_shape(b, tf_norm, tr["order"], window, [shape])
    model = P._Model(b, tr["order"], tf_norm, *shape)
    x, _ = model.fit()
    y, err = _hessian_yield(model, x)
    out = dict(y=y, err=err, mean=float(shape[0]), width=float(shape[1]), order=list(tr["order"]))
    if variant == "float":
        lo, hi = P._profile_yield_error(model, x, window, err) or (None, None)
        bound = lambda v, bd: bool(min(v - bd[0], bd[1] - v) < 0.05)
        out.update(lo=lo, hi=hi, at_bound=bound(shape[0], window) or bound(shape[1], P.WIDTH_BOUNDS))
    return out


_T: dict = {}


def _init_toys(committed, extra, fit_path, eps_path):
    _T.clear()
    _T.update(committed=np.load(committed), extra=np.load(extra) if extra else None,
              v4=json.loads(pathlib.Path(fit_path).read_text()), truths={},
              eps=json.loads(pathlib.Path(eps_path).read_text())["models"] if eps_path else {})


def _top_fit(name):
    v4 = _T["v4"]
    return v4["reference"]["top"] if name == "reference" else v4["models"][name]["top"]


def _start(region, name):
    """Where the shape search starts, as in the run: the reference's fitted shape (the
    reference itself: its own shape floated at START_ORDER, recorded as shape_start); the
    pseudo window: its centre at the top fit's width, as realdata_checks does."""
    if region == "pseudo":
        return 0.5 * sum(PSEUDO["window"]), _top_fit(name)["width"]
    ref = _T["v4"]["reference"]["top"]
    return tuple(ref["shape_start"]) if name == "reference" else (ref["mean"], ref["width"])


def task_key(t):
    return "|".join(map(str, (t["region"], t["mode"], t["name"], t["size"], t["toy"])))


def _run_task(t):
    region, name = t["region"], t["name"]
    b = region_bins(region, name, _T["committed"], _T["extra"])
    if (region, name) not in _T["truths"]:
        _T["truths"][(region, name)] = truth(region, name, b, _top_fit(name), _start(region, name))
    tr = _T["truths"][(region, name)]
    eps = None
    if t["mode"] == "leak":
        eps = _T["eps"][name]["three_prong"]["top_like_at_data_cut"]["eff"]
    seq = np.random.SeedSequence([t["seed"], REGIONS.index(region), MODES.index(t["mode"]),
                                  int(t["size"]), t["toy"] + 1, *map(ord, name)])
    toy = b if t["toy"] < 0 else toy_bins(b, tr, t["size"], t["mode"], np.random.default_rng(seq), eps)
    out = dict(t, key=task_key(t), truth_order=list(tr["order"]), truth_mean=tr["mean"], truth_width=tr["width"],
               eps=eps, fits={})
    for v in t["variants"]:
        out["fits"][v] = fit_variant(v, toy, region, tr, _start(region, name))
    return out


def tasks(regions, modes, names, sizes, n_toys, variants, seed, shard, n_shards):
    """Every (region, mode, score, size, toy) of the study, the k-th of n_shards of them.
    toy -1 with size 0 is the region's real data with nothing injected (mode data)."""
    out = []
    for region in regions:
        for mode in modes:
            for name in names:
                if mode == "data":
                    out.append(dict(region=region, mode=mode, name=name, size=0.0, toy=-1))
                for size in sizes:
                    out += [dict(region=region, mode=mode, name=name, size=float(size), toy=k) for k in range(n_toys)]
    for t in out:
        t.update(seed=seed, variants=list(variants))
    return out[shard::n_shards]


def run_toys(a):
    todo = tasks(a.regions, a.modes, a.names, a.sizes, a.toys, a.variants, a.seed, a.shard, a.n_shards)
    out = pathlib.Path(a.out)
    done = set()
    if out.exists():    # resumable: a restarted pod skips what is written
        done = {json.loads(ln)["key"] for ln in out.read_text().splitlines() if ln.strip()}
    todo = [t for t in todo if task_key(t) not in done]
    print(f"{len(todo)} tasks to run, {len(done)} already written", flush=True)
    init = (str(a.committed), str(a.extra) if a.extra else None, str(a.fit), str(a.eps) if a.eps else None)
    saved = {k: os.environ.get(k) for k in BLAS}
    os.environ.update({k: "1" for k in BLAS})
    try:
        with open(out, "a") as fh:
            if a.workers > 1:
                with multiprocessing.get_context("spawn").Pool(a.workers, initializer=_init_toys, initargs=init) as pool:
                    for r in pool.imap_unordered(_run_task, todo, chunksize=1):
                        fh.write(json.dumps(r) + "\n"); fh.flush()
            else:
                _init_toys(*init)
                for t in todo:
                    fh.write(json.dumps(_run_task(t)) + "\n"); fh.flush()
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bins")
    b.add_argument("--merged", required=True, type=pathlib.Path)
    b.add_argument("--committed", required=True, type=pathlib.Path)
    b.add_argument("--out", required=True, type=pathlib.Path)
    b.add_argument("--workers", type=int, default=1)
    t = sub.add_parser("toys")
    t.add_argument("--committed", default=DATA / "fit_v3/bins.npz", type=pathlib.Path)
    t.add_argument("--extra", default=None, type=pathlib.Path, help="injection_bins.npz (pseudo, other working points)")
    t.add_argument("--fit", default=DATA / "fit_v4/results.json", type=pathlib.Path)
    t.add_argument("--eps", default=None, type=pathlib.Path, help="model_vs_domain.json (mode leak)")
    t.add_argument("--regions", nargs="+", choices=REGIONS, default=["top"])
    t.add_argument("--modes", nargs="+", choices=MODES, default=["bootstrap"])
    t.add_argument("--names", nargs="+", required=True)
    t.add_argument("--sizes", nargs="+", type=float, default=[1000.0, 2000.0, 4000.0])
    t.add_argument("--toys", type=int, default=20)
    t.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    t.add_argument("--seed", type=int, default=20261001)
    t.add_argument("--shard", type=int, default=0)
    t.add_argument("--n-shards", type=int, default=1)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.cmd == "bins":
        export(a.merged, a.committed, a.out, a.workers)
    elif a.cmd == "toys":
        run_toys(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
