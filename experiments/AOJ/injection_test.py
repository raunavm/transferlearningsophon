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
FLOAT_KEYS, FLOAT_RTOL = ("rho", "pt"), 1e-9
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
    arrays, worst = {}, dict.fromkeys(FLOAT_KEYS, 0.0)
    for name, parts in rows:
        for k in KEYS:
            # the counts and bin indices exactly; the per-bin MEAN rho and pT to float
            # precision -- a weighted sum, it differed in the last bits from the committed
            # bins on the cluster (first launch, 2026-10-01) with every count identical
            new, old = parts["main"][k], c[f"{name}|main|{k}"]
            if k in FLOAT_KEYS and new.shape == old.shape:
                rel = float(np.max(np.abs(new - old) / np.abs(old)))
                worst[k] = max(worst[k], rel)
                same = rel <= FLOAT_RTOL
            else:
                same = np.array_equal(new, old)
            if not same:
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
    print(f"largest relative difference from the committed bins: {worst}", flush=True)
    print(f"wrote {out}: {len(names)} scores x (pseudo, {', '.join(f'main_eff{e:g}' for e in EXTRA_EFF)})")


# ------------------------------------------------------------------ toys
DATA = pathlib.Path("experiments/FIGS/data/aoj_full_v1")
REGIONS = ("top", "band", "pseudo", *(f"top_eff{e:g}" for e in EXTRA_EFF))
MODES = ("bootstrap", "data", "leak", "tops", "tops_half", "pooled", "ensemble")
# THE FIT WITH THE TOPS IN THE FAIL REGION (peak_fit._Model, fit_v5): toys whose fail region
# holds the tops failing the cut -- the reference's fitted signal per bin over EPS_TRUE --
# fitted given the reference's signal (EPS_REF = 1, the fewest tops there can be). tops: the
# fit's assumption holds; tops_half: twice as many tops as the reference passes.
EPS_TRUE = dict(tops=1.0, tops_half=0.5, pooled=1.0)
# THE POOLED SHAPE (peak_fit.pooled_shape, fit_v6): mode pooled draws the signal and the
# generator's background at the pooled shape (--pooled: fit_v6's results), the tops in the
# fail region as in mode tops; variant fixed_ftest is then fit_v6's procedure.
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


def truth(region, name, b, top_fit, start, tops=None, pooled=None):
    """The generator of one score's toys in one region: a background with no signal, and a
    signal template that sums to 1 over the region's bins.
      top       the main fit's own model (top_fit: its order and floated shape), signal removed
      top_eff*  the procedure (peak_fit.fit_binned, shape floating) on that working point's data
      band, pseudo  a background-only fit, order by F-test, as validation() makes it
    The template is a Gaussian of the top fit's shape (the pseudo window: centred in it, the
    top fit's width, as realdata_checks injects) and pT split; at another working point the
    shape and split fitted there."""
    window = P._peak_cfg(region_peak(region))["window"]
    if region == "top" and pooled is not None:
        order = P._choose_order(b, P._tf_norm(b, window), *pooled, tops)[0]
        model = P._Model(b, order, P._tf_norm(b, window), *pooled, tops)
        x, _ = model.fit()
        fit = dict(top_fit, mean=pooled[0], width=pooled[1])
    elif region == "top":
        model = P._Model(b, tuple(top_fit["tf_order"]), P._tf_norm(b, window), top_fit["mean"], top_fit["width"],
                         tops)
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


def toy_bins(b, tr, size, mode, rng, eps=None, tops=None):
    """One pseudo-experiment. bootstrap: Poisson pass and fail around the generator, plus
    the injected signal; data: the real counts plus a Poisson-fluctuated injected signal;
    leak: bootstrap, with the signal that FAILS the cut added to the fail region -- a
    tagger of signal efficiency eps leaves size * (1 - eps) / eps behind there; tops,
    tops_half: bootstrap with the tops in each bin, `tops`, failing the cut where the
    injected signal does not pass it."""
    s = size * tr["template"]
    if mode == "data":
        return dict(b, n_pass=b["n_pass"] + rng.poisson(s).astype(float))
    fail = tr["fail"] + (s * (1 - eps) / eps if mode == "leak" else 0.0)
    if mode in EPS_TRUE:
        fail = fail + np.maximum(tops - s, 0.0)
    return dict(b, n_pass=rng.poisson(tr["background"] + s).astype(float), n_fail=rng.poisson(fail).astype(float))


def _hessian_yield(model, x):
    v = model.G.sum(axis=0)
    cov = model.covariance(x)[model.n_tf:, model.n_tf:]
    return float(v @ x[model.n_tf:]), float(np.sqrt(max(v @ cov @ v, 0.0)))


def fit_variant(variant, b, region, tr, start, tops=None):
    """One fit of the toy `b`:
      full         the procedure: peak_fit.fit_binned, shape floating, order by F-test
      float        the shape floating at the generator's order, profile error
      fixed        the generator's shape and order
      fixed_ftest  the generator's shape, order by F-test"""
    peak = region_peak(region)
    window = P._peak_cfg(peak)["window"]
    if variant == "full":
        f = P.fit_binned(b, peak, *start, float_shape=True, tops=tops)[0]
        return dict(y=f["signal_yield"], err=f["signal_yield_err"], lo=f["signal_yield_err_lo"],
                    hi=f["signal_yield_err_hi"], mean=f["mean"], width=f["width"], order=f["tf_order"],
                    at_bound=bool(f["mean_at_bound"] or f["width_at_bound"]))
    if variant == "fixed_ftest":
        f = P.fit_binned(b, peak, tr["mean"], tr["width"], tops=tops)[0]
        return dict(y=f["signal_yield"], err=f["signal_yield_err"], order=f["tf_order"])
    tf_norm = P._tf_norm(b, window)
    shape = (tr["mean"], tr["width"])
    if variant == "float":
        shape = P._float_shape(b, tf_norm, tr["order"], window, [shape], tops)
    model = P._Model(b, tr["order"], tf_norm, *shape, tops)
    x, _ = model.fit()
    y, err = _hessian_yield(model, x)
    out = dict(y=y, err=err, mean=float(shape[0]), width=float(shape[1]), order=list(tr["order"]))
    if variant == "float":
        lo, hi = P._profile_yield_error(model, x, window, err) or (None, None)
        bound = lambda v, bd: bool(min(v - bd[0], bd[1] - v) < 0.05)
        out.update(lo=lo, hi=hi, at_bound=bound(shape[0], window) or bound(shape[1], P.WIDTH_BOUNDS))
    return out


_T: dict = {}


def _init_toys(committed, extra, fit_path, eps_path, pooled_path=None):
    _T.clear()
    _T.update(committed=np.load(committed), extra=np.load(extra) if extra else None,
              v4=json.loads(pathlib.Path(fit_path).read_text()), truths={},
              eps=json.loads(pathlib.Path(eps_path).read_text())["models"] if eps_path else {})
    if pooled_path:
        ps = json.loads(pathlib.Path(pooled_path).read_text())["pooled_shape"]
        _T["pooled"], _T["pool"] = (ps["mean"], ps["width"]), list(ps["pool"])


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


def _tops():
    """The tops per top-fit bin the fit is given: peak_fit.tops_from_reference on the
    reference's committed bins and its fit_v4 fit (fit_v5's reference is fit_v4's)."""
    if "tops" not in _T:
        b_ref = region_bins("top", "reference", _T["committed"], None)
        _T["tops"] = P.tops_from_reference(b_ref, _T["v4"]["reference"]["top"])
    return _T["tops"]


def _run_ensemble(t):
    """One pseudo-experiment of the whole fit_v6 procedure: a toy of every pretrained score
    (mode pooled's generator, one injected size for all), the pooled shape re-derived from
    the toys (peak_fit.pooled_shape, started at the true shape), and each toy fitted at it
    with its F-test order, given the tops. The shape is the toys' own, so its uncertainty is
    in the yields -- for scores whose toys are independent, unlike the data's passing jets."""
    pooled, tops = _T["pooled"], _tops()
    names = _T["pool"]
    toys = {}
    for n in names:
        b = region_bins("top", n, _T["committed"], None)
        key = ("top", n, True, pooled)
        if key not in _T["truths"]:
            _T["truths"][key] = truth("top", n, b, _top_fit(n), _start("top", n), tops, pooled)
        seq = np.random.SeedSequence([t["seed"], REGIONS.index("top"), MODES.index("ensemble"), int(t["size"]),
                                      t["toy"] + 1, *map(ord, n)])
        toys[n] = toy_bins(b, _T["truths"][key], t["size"], "pooled", np.random.default_rng(seq), None, tops)
    shape, orders, trail = P.pooled_shape(toys, names, P.PEAKS["top"]["window"], pooled, {n: tops for n in names})
    fits = {n: P.fit_binned(toys[n], "top", *shape, tops=tops)[0] for n in names}
    return dict(t, key=task_key(t), truth_mean=pooled[0], truth_width=pooled[1], shape=list(shape),
                n_passes=len(trail), per_score={n: dict(y=f["signal_yield"], err=f["signal_yield_err"],
                                                        order=f["tf_order"]) for n, f in fits.items()})


def _run_task(t):
    if t["mode"] == "ensemble":
        return _run_ensemble(t)
    region, name = t["region"], t["name"]
    b = region_bins(region, name, _T["committed"], _T["extra"])
    told = t["mode"] in EPS_TRUE
    if told and region != "top":
        raise SystemExit("FATAL: the tops are known in the top window only")
    tops = _tops() if told else None
    pooled = None
    if t["mode"] == "pooled":
        if "pooled" not in _T:
            raise SystemExit("FATAL: mode pooled needs --pooled (fit_v6's results)")
        pooled = _T["pooled"]
    key = (region, name, told, pooled)
    if key not in _T["truths"]:
        _T["truths"][key] = truth(region, name, b, _top_fit(name), _start(region, name), tops, pooled)
    tr = _T["truths"][key]
    eps = None
    if t["mode"] == "leak":
        eps = _T["eps"][name]["three_prong"]["top_like_at_data_cut"]["eff"]
    seq = np.random.SeedSequence([t["seed"], REGIONS.index(region), MODES.index(t["mode"]),
                                  int(t["size"]), t["toy"] + 1, *map(ord, name)])
    toy = b if t["toy"] < 0 else toy_bins(b, tr, t["size"], t["mode"], np.random.default_rng(seq), eps,
                                          tops / EPS_TRUE[t["mode"]] if told else None)
    out = dict(t, key=task_key(t), truth_order=list(tr["order"]), truth_mean=tr["mean"], truth_width=tr["width"],
               eps=eps, fits={})
    for v in t["variants"]:
        out["fits"][v] = fit_variant(v, toy, region, tr, _start(region, name), tops)
    return out


def tasks(regions, modes, names, sizes, n_toys, variants, seed, shard, n_shards):
    """Every (region, mode, score, size, toy) of the study, the k-th of n_shards of them.
    toy -1 with size 0 is the region's real data with nothing injected (mode data)."""
    out = []
    for region in regions:
        for mode in modes:
            if mode == "ensemble":          # one task per toy: every pooled score at once
                out += [dict(region=region, mode=mode, name="pool", size=float(size), toy=k)
                        for size in sizes for k in range(n_toys)]
                continue
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
    init = (str(a.committed), str(a.extra) if a.extra else None, str(a.fit), str(a.eps) if a.eps else None,
            str(a.pooled) if a.pooled else None)
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


# ------------------------------------------------------------------ readout
REPRODUCE_TOL = 1e-3        # yield shift / error: across machines the minimiser stops a hair apart
# ...except where the floated shape sits in a flat minimum: there Powell's end point moved by
# up to 1 GeV in mean between the cluster and a laptop (5 of 31 replays of the checks, 2026-10-01)
FLAT_TOL = 0.15


def _replay_one(job):
    """One score of reproduce_checks."""
    name, b, top, stored = job
    centre, width = 0.5 * sum(PSEUDO["window"]), top["width"]
    spur = P.fit_binned(b, PSEUDO, centre, width)[0]
    inj, cats = RC.inject_asimov(b, centre, width, top["yield_per_pt_bin"])
    y = sum(cats.values())
    got, _, (model, _) = P.fit_binned(inj, PSEUDO, centre, width, float_shape=True)
    order = tuple(got["tf_order"])
    # THE DECOMPOSITION, at the order the procedure chose:
    #   fixed  the injected shape held fixed: what the data's own background does to the yield
    #   data   nothing injected, the injected shape fixed: the data's spurious signal there
    #   smooth the procedure on the background-only fit's expectation plus the same Asimov
    #          signal: the estimator with no fluctuation at all
    fixed = P.fit_binned(inj, PSEUDO, centre, width, order=order)[0]["signal_yield"]
    spur_at = P.fit_binned(b, PSEUDO, centre, width, order=order)[0]["signal_yield"]
    _, _, (bm, bx) = P.fit_binned(b, PSEUDO)
    t, _, q, _ = bm.expect(bx)
    smooth_inj, _ = RC.inject_asimov(dict(b, n_pass=t * q, n_fail=q), centre, width, top["yield_per_pt_bin"])
    smooth = P.fit_binned(smooth_inj, PSEUDO, centre, width, float_shape=True)[0]
    s, i = stored["spurious"], stored["injected"]
    shift = dict(spurious=(spur["signal_yield"] - s["signal_yield"]) / s["signal_yield_err"],
                 fitted=(got["signal_yield"] - i["fitted"]) / i["fitted_err"],
                 injected=(y - i["signal_yield"]) / i["fitted_err"])
    worst = max(map(abs, shift.values()))
    return name, dict(injected=y, fitted=got["signal_yield"], fitted_err=got["signal_yield_err"],
                      fitted_mean=got["mean"], fitted_width=got["width"], injected_width=width, tf_order=list(order),
                      spurious=spur["signal_yield"], spurious_err=spur["signal_yield_err"],
                      shift_over_err=shift, ok=bool(worst <= REPRODUCE_TOL), within_flat_tol=bool(worst <= FLAT_TOL),
                      recovered=dict(procedure=got["signal_yield"] / y, injected_shape_fixed=fixed / y,
                                     data_spurious_at_that_order=spur_at / y,
                                     procedure_on_smooth_background=smooth["signal_yield"] / y,
                                     smooth_width=smooth["width"]))


def reproduce_checks(extra, fit, checks, workers=1):
    """The checks' pseudo-window test replayed from the exported bins, exactly as
    realdata_checks._injection_one runs it: the spurious signal at a fixed shape, and the
    Asimov injection fitted back with the shape floating; with the decomposition of what
    it recovers (_replay_one). Per score: the replay beside the stored numbers, ok when
    every yield is within REPRODUCE_TOL of its error, within_flat_tol within FLAT_TOL."""
    jobs = [(n, region_bins("pseudo", n, None, extra), fit["models"][n]["top"], st) for n, st in checks["models"].items()]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return dict(pool.map(_replay_one, jobs, chunksize=1))
    return dict(map(_replay_one, jobs))


def _pull(y, target, f):
    """(symmetric, asymmetric) pull: the asymmetric one divides by the profile error on the
    side of the truth (hi when the fit is low), the symmetric one by the quoted error."""
    sym = (y - target) / f["err"] if f["err"] else None
    side = f.get("hi") if y < target else f.get("lo")
    return sym, ((y - target) / side if side else sym)


def _stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return None
    return dict(n=int(len(v)), mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                sd=float(v.std(ddof=1)) if len(v) > 1 else None, median=float(np.median(v)))


def summarise(records):
    """Per (region, mode, size, variant): the recovered fraction fitted / injected and the
    pull, over every score and toy, and per score.
      bootstrap, leak  target = the injected size
      data             target = the injected size + what the same fit finds in the data with
                       nothing injected: in the top window that is the real peak, at the same
                       shape. In the band and the pseudo-window the no-injection fit of the
                       FIXED shape is used (a floating shape finds the largest bump anywhere)."""
    flat = []
    for r in records:       # an ensemble: one record per score, variant "ensemble"
        if r["mode"] != "ensemble":
            flat.append(r)
            continue
        flat += [dict(r, name=n, truth_order=f["order"], truth_width=r["truth_width"],
                      fits={"ensemble": dict(f, width=r["shape"][1], mean=r["shape"][0])})
                 for n, f in r["per_score"].items()]
    records = flat
    base = {(r["region"], r["name"], v): f["y"] for r in records if r["toy"] < 0 for v, f in r["fits"].items()}
    groups = {}
    for r in records:
        if r["toy"] < 0:
            continue
        for v, f in r["fits"].items():
            y0 = 0.0
            if r["mode"] == "data":
                y0 = base[(r["region"], r["name"], v if r["region"].startswith("top") else "fixed")]
            target = r["size"] + y0
            sym, asym = _pull(f["y"], target, f)
            g = groups.setdefault((r["region"], r["mode"], r["size"], v), dict(rows=[], per={}))
            row = dict(ratio=(f["y"] - y0) / r["size"], pull=sym, pull_asym=asym, at_bound=f.get("at_bound"),
                       ensemble=r["toy"] if r["mode"] == "ensemble" else None,
                       order_moved=f["order"] != r["truth_order"], width=f.get("width"), mean=f.get("mean"),
                       width_ratio=f["width"] / r["truth_width"] if f.get("width") else None)
            g["rows"].append(row)
            g["per"].setdefault(r["name"], []).append(row)
    out = []
    for (region, mode, size, v), g in sorted(groups.items()):
        rows = g["rows"]
        col = lambda k, rs=rows: [x[k] for x in rs]
        out.append(dict(
            region=region, mode=mode, size=size, variant=v, n=len(rows),
            ratio=_stats(col("ratio")), pull=_stats(col("pull")), pull_asym=_stats(col("pull_asym")),
            width_ratio=_stats(col("width_ratio")),
            at_bound=float(np.mean([bool(x) for x in col("at_bound")])) if any(x is not None for x in col("at_bound")) else None,
            order_moved=float(np.mean(col("order_moved"))),
            # the scores of one ensemble share its pooled shape: the error of a mean is over ensembles
            ensemble_se=(None if mode != "ensemble" else {
                k: float(np.std(m, ddof=1) / np.sqrt(len(m))) for k in ("ratio", "pull")
                for m in [[np.mean([x[k] for x in rows if x["ensemble"] == e]) for e in sorted({x["ensemble"] for x in rows})]]}),
            per_score={n: dict(ratio=_stats(col("ratio", rs)), pull=_stats(col("pull", rs))) for n, rs in sorted(g["per"].items())}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce")
    r.add_argument("--extra", required=True, type=pathlib.Path)
    r.add_argument("--fit", default=DATA / "fit_v4/results.json", type=pathlib.Path)
    r.add_argument("--checks", default=pathlib.Path("experiments/FIGS/data/aoj_checks_v1/map_injection_data.json"),
                   type=pathlib.Path)
    r.add_argument("--out", required=True, type=pathlib.Path)
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("--toys", nargs="+", required=True, type=pathlib.Path)
    s.add_argument("--out", required=True, type=pathlib.Path)
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
    t.add_argument("--pooled", default=None, type=pathlib.Path, help="fit_v6's results.json (mode pooled)")
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
    elif a.cmd == "reproduce":
        z = np.load(a.extra)
        rows = reproduce_checks({k: z[k] for k in z.files}, json.loads(a.fit.read_text()),
                                json.loads(a.checks.read_text()), a.workers)
        doc = dict(models=rows, all_ok=all(x["ok"] for x in rows.values()), tol=REPRODUCE_TOL,
                   all_within_flat_tol=all(x["within_flat_tol"] for x in rows.values()), flat_tol=FLAT_TOL,
                   n_exact=sum(x["ok"] for x in rows.values()),
                   max_shift_over_err=max(abs(v) for x in rows.values() for v in x["shift_over_err"].values()),
                   inputs={k: str(v) for k, v in dict(extra=a.extra, fit=a.fit, checks=a.checks).items()})
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2))
        print(f"replayed {len(rows)} scores; reproduces the checks: {doc['all_ok']}")
    elif a.cmd == "summary":
        records = [json.loads(ln) for p in a.toys for ln in p.read_text().splitlines() if ln.strip()]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(groups=summarise(records), n_records=len(records),
                                         toys=[str(p) for p in a.toys]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
