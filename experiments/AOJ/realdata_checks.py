#!/usr/bin/env python3
"""The checks behind the real-data section, from committed inputs, in one entry point.

The audit of 2026-09-29 (report B5, must-fix 9) found the section's claims either
unstated or untested at the working point. Each step below answers one of them and
writes one JSON into --out; every number the text quotes is a key there.

  selection   the chain from the dataset to the fitted jets, each count with the
              file and key it comes from, rho as coded, the pT categories.
  closure     the input closure pooled over ALL shards (closure.pool_shards), and
              the domain shift per input feature: median shift, IQR ratio, KS.
  per_run     every model's fitted yield with its fit status and the head defects.
  reference   the CMS ParticleNet top score at 1 % of the SIDEBAND jets (the working
              point every score is cut at) and at 1 % of ALL jets (matched data
              efficiency), each with its validation p-value under the pipeline's own
              criterion (peak_fit.MIN_VALIDATION_P on the toy p-value).
  sim_closure the map at 1 % on JetClass-II QCD: the SAME procedure (peak_fit.
              build_map, windows masked) built and cut on simulated QCD, where
              there is no signal. The background shape change it makes in the top
              window, with its binomial error, and the spurious signal the fit finds.
  injection   signal-free data: a pseudo-window at 250-310 GeV (PSEUDO), masked in
              the map exactly as the top window is. The background shape change
              the cut makes there, the spurious signal fitted there, and a known
              signal injected there and fitted back.
  domain      model vs domain: each model's efficiency on JetClass-II three-prong
              decays and QCD at ITS OWN DATA CUT (the map built on data, applied to
              simulation), and at a cut built on simulated QCD, beside its data yield.
  prong       the prong-only score (discriminants.SCORES["prong_only"]: three-prong
              over two- and four-prong decays, no QCD node) through the whole
              procedure -- map at 1 %, floated-shape fit, validation -- per model.
              It tests whether the prong count alone selects the top peak.

INPUTS, all written by committed code:
  --merged   merge_shards.py over the shards: jets.npz, scores_<model>.npz
  --shards   the shard directories (staging/*.stats.json, closure.json)
  --fit      the main fit's results.json (peak_fit.py; fit_v4 for the first run)
  --sim      sim_scores.py's directory: jets.npz, scores_<model>.npz/.json

Run the first run (v1) or any later one the same way:
    python3 experiments/AOJ/realdata_checks.py --merged M --shards S0 .. S9 \\
        --fit experiments/FIGS/data/aoj_full_v1/fit_v4/results.json --sim SIM --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import pathlib
import subprocess
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("peak_fit", HERE / "peak_fit.py")
D = _load("discriminants", HERE / "discriminants.py")
CL = _load("closure", HERE / "closure.py")
SA = _load("stage_aoj", REPO / "scripts" / "stage_aoj.py")

EFF = 0.01
TOP = P.PEAKS["top"]
# THE SIGNAL-FREE PSEUDO-PEAK. Above the top window there is no resonance in these
# data: the top peak (fitted width ~11 GeV at ~183 GeV) is 3 widths below 220 GeV. The
# window is 60 GeV wide in a 140 GeV range, as the top's is 80 in 195. Bins fully
# inside the rho window exist here from pT 750 GeV up (five of the eight categories).
PSEUDO = dict(name="pseudo_250_310", window=(250.0, 310.0), fit_range=(220.0, 360.0))
LABEL_SET = {"l188": "188", "l162": "162", "r42q1": "43", "r16q1": "17",
             "l162mass": "162+mass", "r16q1mass": "17+mass"}
# The four runs whose epoch-79 output layer is defective while their frozen features
# probe normally: audit 2026-09-29, section B4 (head top-1 accuracy and P(QCD) on
# 2M test jets, head_acc.py). Reported beside the yields, never used to drop a run.
HEAD_DEFECTS = {"l188-s5": "top-1 0.327 vs 0.463-0.490; P(QCD) on resonant jets 0.301",
                "l162-s5": "top-1 0.386 vs 0.501-0.545; P(QCD) on resonant jets 0.239",
                "r42q1-s5": "top-1 0.550 vs 0.605-0.671; never predicts QCD (P(QCD) 0.000)",
                "r16q1mass-s4": "top-1 0.479 vs 0.682-0.719; P(QCD) on resonant jets 0.231"}
QCD_LABELS = range(161, 188)
TOP_LIKE = ("label_X_YY_qqb", "label_X_YY_bcs")     # b + a W-like light pair
STEPS = ("reproduce", "selection", "closure", "per_run", "reference", "sim_closure", "injection", "domain",
         "prong")


def label_set(name: str) -> str:
    if name == P.PUBLISHED:
        return "published"
    return LABEL_SET[name.rsplit("-s", 1)[0]]


def pretrained(names):
    return [n for n in names if n != P.PUBLISHED]


# ---------------------------------------------------------------- provenance
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(inputs: dict) -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "experiments/AOJ"],
                           capture_output=True, text=True)
    return dict(code=head.stdout.strip() or None, code_dirty=bool(dirty.stdout.strip()),
                inputs={k: dict(path=str(v), sha256=_sha256(v)) for k, v in inputs.items()})


def _plain(x):
    """numpy scalars and arrays as JSON."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"{type(x).__name__} is not JSON serialisable")


def _where(fn) -> str:
    """file:line of a function, for the record of what the code does."""
    return f"{pathlib.Path(inspect.getsourcefile(fn)).relative_to(REPO)}:{inspect.getsourcelines(fn)[1]}"


# ---------------------------------------------------------------- loading
IDENTITY = ("run", "lumi", "event", "jet_sdmass", "aoj_jet_pt")


def load_data(merged: pathlib.Path, kinds=("three_prong", "prong_only"), first: pathlib.Path | None = None) -> dict:
    """The merged run, restricted to the rho window and pT range the fit uses.

    `first`: the merged run the MAIN FIT was made from, when `merged` is a rescore of
    it. Its three-prong scores are then the ones used -- exactly what the fit saw --
    and the rescore's are kept as three_prong_rescore; the jets must be identical.
    A rescore on another GPU model differs in the last bit of a few jets' float16
    scores (reproduce.json quantifies it), so "the same model" is not "the same number"."""
    j = np.load(merged / "jets.npz")
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    scores = {}
    for path in sorted(merged.glob("scores_*.npz")):
        s = np.load(path)
        scores[path.stem.removeprefix("scores_")] = {k: s[f"{k}_logodds"][ok] for k in kinds
                                                     if f"{k}_logodds" in s.files}
    if first is not None:
        jf = np.load(first / "jets.npz")
        bad = [k for k in IDENTITY if not np.array_equal(jf[k], j[k])]
        if bad:
            raise SystemExit(f"FATAL: {first} and {merged} hold different jets ({bad})")
        for name, sc in scores.items():
            path = first / f"scores_{name}.npz"
            if "three_prong" in sc and path.exists():
                sc["three_prong_rescore"] = sc["three_prong"]
                sc["three_prong"] = np.load(path)["three_prong_logodds"][ok]
    return dict(mass=mass[ok], pt=pt[ok], pn=np.asarray(j["aoj_pn_TvsQCD"])[ok], n_staged=int(len(ok)),
                n_window=int(ok.sum()), scores=scores)


def load_sim(sim: pathlib.Path) -> dict:
    """sim_scores.py's output, in the same acceptance as the data. A model whose
    jets_sha256 is not that of jets.npz was scored on other jets and is refused."""
    SS = _load("sim_scores", HERE / "sim_scores.py")
    j = dict(np.load(sim / "jets.npz"))
    digest = SS.jets_digest(j)
    mass, pt = j["jet_sdmass"].astype(float), j["jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    scores = {}
    for path in sorted(sim.glob("scores_*.npz")):
        name = path.stem.removeprefix("scores_")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["jets_sha256"] != digest:
            raise SystemExit(f"FATAL: {name} was scored on other jets than {sim / 'jets.npz'}")
        s = np.load(path)
        scores[name] = {k.removesuffix("_logodds"): s[k][ok] for k in s.files}
    label = j["label"][ok].astype(int)
    names = {int(r["jet_label"]): r["class_name"] for r in D._anomaly.read_map()}
    three = np.isin(label, sorted(D.native_classes(D.STRUCTURES["three_prong"])))
    has_b = np.array(["b" in names[x].removeprefix("label_X_YY_") for x in label])
    return dict(mass=mass[ok], pt=pt[ok], label=label, n_all=int(len(ok)), n_acc=int(ok.sum()),
                qcd=np.isin(label, QCD_LABELS), three=three,
                top_like=np.isin(label, [k for k, v in names.items() if v in TOP_LIKE]),
                three_b=three & has_b, three_nob=three & ~has_b, scores=scores)


# ---------------------------------------------------------------- helpers
def _bin_index(x, edges):
    """np.histogram's bin of each x (last edge inclusive); -1 outside."""
    i = np.searchsorted(edges, x, side="right") - 1
    i[x == edges[-1]] = len(edges) - 2
    return np.where((i >= 0) & (i < len(edges) - 1), i, -1)


def fit_acceptance(mass, pt, fit_range) -> np.ndarray:
    """The jets in the fit's (m_SD, pT) bins, those lying fully inside the rho window
    (peak_fit._bins), jet by jet."""
    m_edges = np.arange(fit_range[0], fit_range[1] + 1e-9, P.MASS_BIN)
    i, j = _bin_index(mass, m_edges), _bin_index(pt, P.PT_EDGES)
    m_lo, m_hi = m_edges[:-1, None], m_edges[1:, None]
    p_lo, p_hi = P.PT_EDGES[None, :-1], P.PT_EDGES[None, 1:]
    inside = (P.rho_of(m_lo, p_hi) > P.RHO_RANGE[0]) & (P.rho_of(m_hi, p_lo) < P.RHO_RANGE[1])
    ok = (i >= 0) & (j >= 0)
    out = np.zeros(len(mass), dtype=bool)
    out[ok] = inside[i[ok], j[ok]]
    return out


def binomial(k, n) -> dict:
    p = k / n if n else float("nan")
    return dict(k=int(k), n=int(n), eff=float(p), err=float(np.sqrt(p * (1 - p) / n)) if n else None)


def shape_change(passed, mass, acc, window) -> dict:
    """The fractional change the cut makes to the window-to-sideband ratio of the jets
    in `acc`: eff(window) / eff(sidebands) - 1. Zero: the passing jets have the mass
    shape of all jets. Error: binomial on both efficiencies, independent samples."""
    w = acc & P.in_windows(mass, [window])
    s = acc & ~w
    ew, es = binomial(passed[w].sum(), w.sum()), binomial(passed[s].sum(), s.sum())
    r = ew["eff"] / es["eff"]
    err = r * np.sqrt((1 - ew["eff"]) / max(ew["k"], 1) + (1 - es["eff"]) / max(es["k"], 1))
    return dict(delta=float(r - 1), err=float(err), window=ew, sidebands=es)


def eff_profile(passed, mass, acc, fit_range) -> dict:
    """Pass fraction per 5 GeV mass bin over the fit range, jets in `acc`."""
    edges = np.arange(fit_range[0], fit_range[1] + 1e-9, P.MASS_BIN)
    i = _bin_index(mass, edges)
    n = np.bincount(i[acc & (i >= 0)], minlength=len(edges) - 1)
    k = np.bincount(i[acc & (i >= 0) & passed], minlength=len(edges) - 1)
    return dict(m_edges=edges.tolist(), n=n.tolist(), n_pass=k.tolist())


def _fit_summary(fit: dict) -> dict:
    keys = ("signal_yield", "signal_yield_err", "signal_yield_err_lo", "signal_yield_err_hi", "z_wald",
            "s_over_sqrt_b", "s_in_window", "b_in_window", "mean", "width", "tf_order", "converged",
            "edm", "n_tf_at_floor", "profile_error_ok", "mean_at_bound", "width_at_bound",
            "delta_deviance_vs_background_only", "data_efficiency", "data_efficiency_sidebands",
            "data_efficiency_top_window", "yield_per_pt_bin")
    out = {k: fit[k] for k in keys if k in fit}
    if "validation" in fit:
        v = fit["validation"]
        out["validation"] = dict(toy_p=v.get("toy_p"), asymptotic_p=v.get("asymptotic_p"),
                                 band=v.get("band"), band_signal_z=v.get("band_signal_z"),
                                 tf_order=v.get("tf_order"),
                                 passes=(v.get("toy_p") or 0.0) > P.MIN_VALIDATION_P)
    return out


def _parallel(fn, items, workers):
    """fn over items in forked workers (the loaded arrays are shared, not copied)."""
    if workers <= 1:
        return [fn(x) for x in items]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, items)


def spread(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else None, min=float(v.min()), max=float(v.max()))


def correlation(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    return dict(n=int(len(x)), pearson_r=float(r), pearson_p=float(p), spearman_rho=float(rs),
                spearman_p=float(ps))


def by_label_set(rows: dict, key) -> dict:
    """mean and SD (n - 1) of rows[name][key] over the pretrained runs of each label set."""
    out = {}
    for name, row in rows.items():
        if name == P.PUBLISHED or row.get(key) is None:
            continue
        out.setdefault(label_set(name), []).append(row[key])
    return {ls: dict(n=len(v), mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                     min=float(np.min(v)), max=float(np.max(v))) for ls, v in out.items()}


# ================================================================ the steps
def step_reproduce(shards, first_run) -> dict:
    """The rescored shards against the first run's: the same jets, the same three-prong
    score bit for bit, the same closure. The main fit (fit results) was made from the
    first run's scores; the checks read the rescore's, so this is what licenses reading
    them together."""
    rows = {}
    for s, f in zip(shards, first_run):
        a, b = np.load(s / "jets.npz"), np.load(f / "jets.npz")
        same_jets = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in b.files)
        models = {}
        for path in sorted(f.glob("scores_*.npz")):
            new = s / path.name
            if not new.exists():
                models[path.stem.removeprefix("scores_")] = dict(identical=False, missing=True)
                continue
            x, y = np.load(new)["three_prong_logodds"], np.load(path)["three_prong_logodds"]
            d = np.abs(x.astype(float) - y.astype(float))
            ulp = np.spacing(np.abs(y)).astype(float)
            models[path.stem.removeprefix("scores_")] = dict(
                identical=bool(np.array_equal(x, y)), n_differ=int((d > 0).sum()), n=int(len(d)),
                max_abs_diff=float(d.max()), max_diff_float16_ulps=float((d / ulp).max()),
                n_over_one_ulp=int((d > 1.0001 * ulp).sum()))
        ca, cb = (json.loads((d / "closure.json").read_text()) for d in (s, f))
        strip = lambda rows_: [{k: v for k, v in r.items() if not k.startswith("quantiles_")
                                and k not in ("n_aoj", "n_reference")} for r in rows_]
        # the first run's closure predates the n_particles iqr_ratio/shift fields
        common = lambda r, o: {k: v for k, v in r.items() if k in o}
        same_closure = all(common(r, o) == common(o, r) for r, o in zip(strip(ca["rows"]), strip(cb["rows"]))) \
            and len(ca["rows"]) == len(cb["rows"])
        gpu = lambda d: dict(ln.split(" ", 1) for ln in (d / "gpu_per_model.txt").read_text().splitlines()
                             if " " in ln) if (d / "gpu_per_model.txt").exists() else {}
        rows[s.name] = dict(first_run=str(f), same_jets=bool(same_jets), same_closure=bool(same_closure),
                            three_prong=models, gpu_rescore=gpu(s), gpu_first_run=gpu(f))
    ok = all(r["same_jets"] and all(m["identical"] for m in r["three_prong"].values()) for r in rows.values())
    n_all = sum(m.get("n", 0) for r in rows.values() for m in r["three_prong"].values())
    return dict(shards=rows, all_identical=bool(ok),
                all_jets_identical=all(r["same_jets"] for r in rows.values()),
                all_closures_identical=all(r["same_closure"] for r in rows.values()),
                fraction_of_scores_differing=sum(m.get("n_differ", 0) for r in rows.values()
                                                 for m in r["three_prong"].values()) / max(n_all, 1),
                max_diff_float16_ulps=max(m.get("max_diff_float16_ulps", 0.0) for r in rows.values()
                                          for m in r["three_prong"].values()),
                n_models=len({m for r in rows.values() for m in r["three_prong"]}))


def _flips_one(name):
    d = _W["data"]
    z1, z2 = (d["scores"][name][k].astype(float) for k in ("three_prong", "three_prong_rescore"))
    p1 = P.passes(z1, d["mass"], d["pt"], P.build_map(z1, d["mass"], d["pt"], EFF))
    p2 = P.passes(z2, d["mass"], d["pt"], P.build_map(z2, d["mass"], d["pt"], EFF))
    return name, dict(n_pass_first=int(p1.sum()), n_pass_rescore=int(p2.sum()), n_flipped=int((p1 != p2).sum()))


def step_cut_flips(data, workers) -> dict:
    """At the 1 % cut, the jets whose pass/fail status differs between the first run's
    three-prong score and the rescore's: what the last-bit differences are worth."""
    _W.update(data=data)
    names = [n for n, s in sorted(data["scores"].items()) if "three_prong_rescore" in s]
    rows = dict(_parallel(_flips_one, names, workers))
    return dict(models=rows, max_flipped=max((r["n_flipped"] for r in rows.values()), default=0),
                max_flipped_fraction_of_pass=max((r["n_flipped"] / r["n_pass_first"] for r in rows.values()),
                                                 default=0.0))


def step_selection(shards, data, fit) -> dict:
    """Item 1: the selection chain."""
    stats_files = sorted(f for s in shards for f in (s / "staging").glob("*.stats.json"))
    st = [json.loads(f.read_text()) for f in stats_files]
    n_read = sum(s["n_jets_read"] for s in st)
    n_staged = sum(s["counters"]["n_jets"] for s in st)
    if n_staged != data["n_staged"]:
        raise SystemExit(f"FATAL: staging stats say {n_staged:,} jets, the merged run holds {data['n_staged']:,}")
    b = P._bins(data["mass"], data["pt"], np.ones(len(data["mass"]), bool), TOP["fit_range"])
    n_fit = int(b["n_pass"].sum() + b["n_fail"].sum())
    acc = fit_acceptance(data["mass"], data["pt"], TOP["fit_range"])
    if int(acc.sum()) != n_fit:
        raise SystemExit(f"FATAL: jet-level fit acceptance {acc.sum():,} != binned {n_fit:,}")
    per_model = {n: int(round(f["top"]["n_pass"] + f["top"]["n_fail"])) for n, f in fit["models"].items()}
    ref = int(round(fit["reference"]["top"]["n_pass"] + fit["reference"]["top"]["n_fail"]))
    j_pt = _bin_index(data["pt"][acc], P.PT_EDGES)
    return dict(
        # flat keys, read by experiments/FIGS/make_tables.py (its SLOTS)
        n_dataset=n_read, n_staged=n_staged, n_rho_window=data["n_window"], n_fit=n_fit,
        n_pt_bins=len(P.PT_EDGES) - 1, abs_eta_max=SA.SELECTION["abs_eta_max"],
        steps=[
            dict(stage="dataset", n=n_read, n_quoted="178M",
                 definition="AspenOpenJets: CMS 2016 JetHT Run G+H open data, AK8 PUPPI jets, pT > 300 GeV, "
                            "|eta| < 2.5 (arXiv:2412.10504, pp. 3-4)",
                 n_in_the_80_files=n_read,
                 source="sum over the 80 files of shard*/staging/<file>.stats.json:n_jets_read"),
            dict(stage="staged", n=n_staged,
                 definition=f"{SA.SELECTION['pt_min']:g} < pT < {SA.SELECTION['pt_max']:g} GeV, "
                            f"|eta| < {SA.SELECTION['abs_eta_max']:g}, "
                            f"{SA.SELECTION['sdmass_min']:g} < m_SD < {SA.SELECTION['sdmass_max']:g} GeV on the "
                            "stored jet: pT = FatJet_pt (JEC-corrected), m_SD = FatJet_msoftdrop",
                 code=_where(SA.select),
                 source="sum of shard*/staging/<file>.stats.json:counters.n_jets = merge_manifest.json:n_jets "
                        "= len(jets.npz)"),
            dict(stage="rho window", n=data["n_window"],
                 definition=f"{P.RHO_RANGE[0]} < rho < {P.RHO_RANGE[1]} and {P.PT_RANGE[0]:g} < pT < "
                            f"{P.PT_RANGE[1]:g} GeV; the map is built on these jets",
                 code=_where(P.rho_of), source=f"{fit.get('_path', 'results.json')}:n_jets",
                 n_in_results=int(fit["n_jets"])),
            dict(stage="fit", n=n_fit,
                 definition=f"(m_SD, pT) bins lying fully inside the rho window, {TOP['fit_range'][0]:g}-"
                            f"{TOP['fit_range'][1]:g} GeV in {P.MASS_BIN:g} GeV bins, {len(P.PT_EDGES) - 1} pT "
                            f"categories: {len(b['n_pass'])} bins",
                 code=_where(P._bins),
                 source="results.json:models.<m>.top.n_pass + n_fail (identical for every score)",
                 n_in_results_every_model=sorted(set(per_model.values())), n_in_results_reference=ref),
        ],
        rho=dict(definition="rho = 2 ln(m_SD / pT)", code=_where(P.rho_of),
                 m_sd="jet_sdmass = FatJet_msoftdrop (stored)", pt="aoj_jet_pt = FatJet_pt (JEC-corrected, stored)"),
        pt_categories_gev=P.PT_EDGES.tolist(),
        fit_jets_per_pt_category=np.bincount(j_pt, minlength=len(P.PT_EDGES) - 1).tolist(),
        mass_bin_gev=P.MASS_BIN, fit_range_gev=list(TOP["fit_range"]), n_fit_bins=int(len(b["n_pass"])),
        masked_windows_gev=[list(w) for w in P.MASKED], n_files=len(stats_files))


def step_closure(shards) -> dict:
    """Items 7 and 8: closure over every shard; domain shift per input feature."""
    closures = [json.loads((s / "closure.json").read_text()) for s in shards]
    pooled = CL.pool_shards(closures)
    shift = {}
    for feat, row in pooled["features"].items():
        if row["stat"] != "median|iqr":
            continue
        pick = lambda k: row.get("pooled", {}).get(k)
        shift[feat] = dict(
            iqr_ratio=pick("iqr_ratio") if "pooled" in row else row.get("iqr_ratio", {}).get("mean"),
            iqr_ratio_shard_sd=row.get("iqr_ratio", {}).get("sd"),
            median_ratio=pick("median_ratio") if "pooled" in row else row.get("ratio", {}).get("mean"),
            median_shift_in_reference_iqr=(pick("shift_in_ref_iqr") if "pooled" in row
                                           else row.get("shift_in_ref_iqr", {}).get("mean")),
            ks=pick("ks"), pooled_exactly="pooled" in row)
    return dict(pooled, domain_shift=shift,
                note="AspenOpenJets vs JetClass-II QCD, both through weaver's loader with the model's "
                     "transforms, jet pT 500-700 GeV, real (unpadded) particles; displacement features "
                     "compare |value| of tracks (non-zero). Per shard: the first ~50,000 jets of its files. "
                     "The reference is the same JetClass-II jets in every shard.")


def step_per_run(fit, domain=None) -> dict:
    """Item 6: every model's yield with its fit status."""
    rows = {}
    pooled = fit.get("shape_variations", {}).get("top", {})
    for name, f in sorted(fit["models"].items()):
        t = f["top"]
        row = dict(label_set=label_set(name), **_fit_summary(t))
        row["negative_pt_bins"] = {k: v for k, v in t.get("yield_per_pt_bin", {}).items() if v < 0}
        sv = t.get("shape_variations", {}).get("pooled")
        if sv:
            row["pooled_shape_yield"] = sv["signal_yield"]
            row["pooled_shape_yield_err"] = sv["signal_yield_err"]
        trail = t.get("f_test") or []
        row["lower_orders_shape_at_bound"] = [
            dict(order=e["order"], mean=e.get("mean"), width=e.get("width")) for e in trail
            if e.get("mean") is not None and (min(e["mean"] - TOP["window"][0], TOP["window"][1] - e["mean"]) < 0.05
                                              or min(e["width"] - P.WIDTH_BOUNDS[0], P.WIDTH_BOUNDS[1] - e["width"]) < 0.05)]
        row["head_defect"] = HEAD_DEFECTS.get(name)
        if domain and name in domain.get("models", {}):
            row["sim_signal_eff_at_data_cut"] = domain["models"][name]["three_prong"]["signal_at_data_cut"]["eff"]
        rows[name] = row
    lowest = {}
    for name, r in rows.items():
        if name != P.PUBLISHED:
            ls = r["label_set"]
            if ls not in lowest or r["signal_yield"] < rows[lowest[ls]]["signal_yield"]:
                lowest[ls] = name
    ybl = by_label_set(rows, "signal_yield")
    err = by_label_set(rows, "signal_yield_err")
    spread = {ls: dict(sd_over_run=v["sd"], median_stat_err=float(np.median(
                  [r["signal_yield_err"] for n, r in rows.items() if n != P.PUBLISHED and r["label_set"] == ls])),
                  mean_stat_err=err[ls]["mean"])
              for ls, v in ybl.items()}
    for v in spread.values():
        v["sd_over_median_stat_err"] = v["sd_over_run"] / v["median_stat_err"] if v["sd_over_run"] else None
    return dict(models=rows, lowest_yield_per_label_set=lowest,
                yield_by_label_set=ybl, run_spread_vs_stat_error=spread,
                pooled_shape=pooled.get("pooled"),
                head_defects_source="audit 2026-09-29, report section B4 (head_acc.py on 2M test jets)")


def _calibrate_overall(z, mass, pt, target, tol=2e-5, iters=40):
    """Sideband efficiency at which build_map's cut passes `target` of ALL jets."""
    lo, hi = target / 4, target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = P.passes(z, mass, pt, P.build_map(z, mass, pt, mid)).mean()
        if abs(got - target) < tol:
            return mid, float(got)
        lo, hi = (mid, hi) if got < target else (lo, mid)
    raise SystemExit(f"FATAL: no sideband efficiency gives {target} overall")


def _efficiencies(passed, mass):
    return dict(all=float(passed.mean()), sidebands=float(passed[~P.in_windows(mass, P.MASKED)].mean()),
                top_window=float(passed[P.in_windows(mass, [TOP["window"]])].mean()))


def step_reference(data, fit, n_toys) -> dict:
    """Item 3: the CMS reference at 1 % of the sidebands and at 1 % of all jets."""
    z, mass, pt = P.logit(data["pn"]), data["mass"], data["pt"]
    side = fit["reference"]["top"]
    at_side = _efficiencies(P.passes(z, mass, pt, P.build_map(z, mass, pt, EFF)), mass)
    eff_cal, got = _calibrate_overall(z, mass, pt, EFF)
    res, passed, _ = P.analyse(z, mass, pt, "top", eff_cal, n_toys)
    matched = dict(_fit_summary(res), sideband_efficiency_setting=eff_cal, efficiencies=_efficiencies(passed, mass),
                   z_ok=bool(res["z_wald"] >= TOP["reference_z"]))
    ratio = lambda y: {n: f["top"]["signal_yield"] / y for n, f in fit["models"].items()}
    return dict(
        at_sideband_1pct=dict(_fit_summary(side), efficiencies=at_side, source="the main fit's results.json:reference.top"),
        at_overall_1pct=matched,
        criterion=f"validation toy p > {P.MIN_VALIDATION_P} (peak_fit.MIN_VALIDATION_P), {n_toys} toys",
        model_over_reference_sideband=ratio(side["signal_yield"]),
        model_over_reference_overall=ratio(res["signal_yield"]))


# ---- per-model workers: they read the module-level _W, set before the fork
_W: dict = {}


def _sim_closure_one(name):
    s, fit = _W["sim"], _W["fit"]["models"][name]["top"]
    q = s["qcd"]
    z, m, pt = s["scores"][name]["three_prong"][q].astype(float), s["mass"][q], s["pt"][q]
    mp = P.build_map(z, m, pt, EFF)
    passed = P.passes(z, m, pt, mp)
    acc = fit_acceptance(m, pt, TOP["fit_range"])
    b = P._bins(m, pt, passed, TOP["fit_range"])
    spur, _, _ = P.fit_binned(b, "top", fit["mean"], fit["width"])
    frac = spur["signal_yield"] / spur["b_in_window"] if spur["b_in_window"] > 0 else None
    return name, dict(
        n_qcd=int(q.sum()), efficiencies=_efficiencies(passed, m),
        shape_change_top_window=shape_change(passed, m, acc, TOP["window"]),
        profile=eff_profile(passed, m, acc, TOP["fit_range"]),
        spurious=dict(signal_yield=spur["signal_yield"], signal_yield_err=spur["signal_yield_err"],
                      z=spur["z_wald"], b_in_window=spur["b_in_window"], over_background_in_window=frac,
                      tf_order=spur["tf_order"], shape=[fit["mean"], fit["width"]],
                      scaled_to_data=(frac * fit["b_in_window"] if frac is not None else None),
                      scaled_to_data_over_yield=(frac * fit["b_in_window"] / fit["signal_yield"]
                                                 if frac is not None else None)))


def step_sim_closure(sim, fit, workers) -> dict:
    """Item 2a: the map at 1 % on simulated QCD, where there is no signal."""
    _W.update(sim=sim, fit=fit)
    names = [n for n in sorted(fit["models"]) if n in sim["scores"]]
    rows = dict(_parallel(_sim_closure_one, names, workers))
    runs = pretrained(rows)
    summary = dict(
        shape_change=spread([rows[n]["shape_change_top_window"]["delta"] for n in runs]),
        shape_change_err=spread([rows[n]["shape_change_top_window"]["err"] for n in runs]),
        shape_change_pull=spread([rows[n]["shape_change_top_window"]["delta"] / rows[n]["shape_change_top_window"]["err"]
                                  for n in runs]),
        spurious_z=spread([rows[n]["spurious"]["z"] for n in runs]),
        spurious_over_background=spread([rows[n]["spurious"]["over_background_in_window"] for n in runs]),
        spurious_scaled_to_data_over_yield=spread([rows[n]["spurious"]["scaled_to_data_over_yield"] for n in runs]),
        note="the same simulated QCD jets serve every model, so the models' values are correlated; "
             "the spread is not an independent-sample error")
    return dict(models=rows, summary=summary,
                shape_change_by_label_set=by_label_set({n: dict(v=r["shape_change_top_window"]["delta"])
                                                        for n, r in rows.items()}, "v"),
                definition="shape change = eff(140-220 GeV) / eff(rest of 105-300 GeV) - 1 of the jets in the "
                           "fit's bins, after the 1 % cut the map (peak_fit.build_map, W and top windows masked) "
                           "builds on these same simulated QCD jets; spurious = the top fit's signal on them, the "
                           "shape fixed at that model's data fit, transfer-factor order by F-test; scaled_to_data = "
                           "spurious / background in the window x the data fit's background in the window",
                sample="JetClass-II test QCD (QCD_0350-0419) through the AspenOpenJets selection, sim jet pT, "
                       "same rho window")


def _injection_one(name):
    d, fit = _W["data"], _W["fit"]["models"][name]["top"]
    z, m, pt = d["scores"][name]["three_prong"].astype(float), d["mass"], d["pt"]
    masked = P.MASKED + (PSEUDO["window"],)
    passed = P.passes(z, m, pt, P.build_map(z, m, pt, EFF, masked=masked))
    acc = fit_acceptance(m, pt, PSEUDO["fit_range"])
    b = P._bins(m, pt, passed, PSEUDO["fit_range"])
    centre, width = 0.5 * sum(PSEUDO["window"]), fit["width"]
    spur, _, _ = P.fit_binned(b, PSEUDO, centre, width)
    # INJECTED: a Gaussian of the model's own fitted width at the pseudo-window's centre,
    # its yield and pT split those of the model's top fit in the pT categories present
    # here, added to the passing bins as expected counts (Asimov), then fitted back with
    # the shape floating, exactly as the top peak is fitted.
    inj, cats = inject_asimov(b, centre, width, fit["yield_per_pt_bin"])
    y_inj = float(sum(cats.values()))
    got, _, _ = P.fit_binned(inj, PSEUDO, centre, width, float_shape=True)
    return name, dict(
        efficiencies=dict(all=float(passed.mean()),
                          sidebands=float(passed[~P.in_windows(m, masked)].mean())),
        shape_change_pseudo_window=shape_change(passed, m, acc, PSEUDO["window"]),
        profile=eff_profile(passed, m, acc, PSEUDO["fit_range"]),
        spurious=dict(signal_yield=spur["signal_yield"], signal_yield_err=spur["signal_yield_err"],
                      z=spur["z_wald"], tf_order=spur["tf_order"], shape=[centre, width]),
        injected=dict(signal_yield=y_inj, per_pt_bin={f"{P.PT_EDGES[j]:g}-{P.PT_EDGES[j + 1]:g}": v for j, v in cats.items()},
                      fitted=got["signal_yield"], fitted_err=got["signal_yield_err"],
                      pull=(got["signal_yield"] - y_inj) / got["signal_yield_err"] if got["signal_yield_err"] else None,
                      fitted_mean=got["mean"], fitted_width=got["width"], tf_order=got["tf_order"],
                      profile_error_ok=got.get("profile_error_ok")))


def inject_asimov(b, centre, width, yield_per_pt_bin):
    """Bins `b` with a Gaussian's expected counts added to the passing jets. Its norm in
    each pT category present in `b` is that category's entry of yield_per_pt_bin
    (negative entries count as zero). Returns (bins, {category index: the signal
    ADDED TO THE BINS}) -- what a fit's signal_yield measures: at low pT the rho window
    cuts the upper mass bins, so part of the Gaussian lies outside the fitted bins."""
    per_pt = {k: max(v, 0.0) for k, v in yield_per_pt_bin.items()}
    norm = {int(j): per_pt.get(f"{P.PT_EDGES[j]:g}-{P.PT_EDGES[j + 1]:g}", 0.0) for j in np.unique(b["j"])}
    g = P._gauss_bins(b["m_edges"], centre, width)[b["i"]]
    added = g * np.array([norm[j] for j in b["j"]])
    return (dict(b, n_pass=b["n_pass"] + added),
            {j: float(added[b["j"] == j].sum()) for j in norm})


def step_injection(data, fit, workers) -> dict:
    """Item 2b: signal-free data at the working point."""
    _W.update(data=data, fit=fit)
    names = [n for n in sorted(fit["models"]) if n in data["scores"]]
    rows = dict(_parallel(_injection_one, names, workers))
    runs = pretrained(rows)
    summary = dict(
        shape_change=spread([rows[n]["shape_change_pseudo_window"]["delta"] for n in runs]),
        shape_change_err=spread([rows[n]["shape_change_pseudo_window"]["err"] for n in runs]),
        shape_change_pull=spread([rows[n]["shape_change_pseudo_window"]["delta"] / rows[n]["shape_change_pseudo_window"]["err"]
                                  for n in runs]),
        spurious_z=spread([rows[n]["spurious"]["z"] for n in runs]),
        injection_pull=spread([rows[n]["injected"]["pull"] for n in runs]),
        injected_yield=spread([rows[n]["injected"]["signal_yield"] for n in runs]),
        note="the same data jets serve every model, so the models' values are correlated")
    return dict(models=rows, summary=summary, pseudo=dict(PSEUDO, window=list(PSEUDO["window"]), fit_range=list(PSEUDO["fit_range"])),
                shape_change_by_label_set=by_label_set({n: dict(v=r["shape_change_pseudo_window"]["delta"])
                                                        for n, r in rows.items()}, "v"),
                definition="the map built at 1 % with the W, top AND pseudo windows masked, on the data; shape "
                           "change = eff(pseudo window) / eff(rest of its fit range) - 1 of the jets in the fit's "
                           "bins; spurious = the fitted signal there at a fixed shape (centre, the model's top "
                           "width); injected = expected counts of a Gaussian added to the passing bins, fitted "
                           "back with the shape floating (peak_fit.fit_binned, float_shape)")


def _domain_one(name):
    d, s, fit = _W["data"], _W["sim"], _W["fit"]["models"][name]["top"]
    acc_top = fit_acceptance(s["mass"], s["pt"], TOP["fit_range"]) & P.in_windows(s["mass"], [TOP["window"]])
    out = {}
    for kind in ("three_prong", "prong_only"):
        if kind not in d["scores"][name] or kind not in s["scores"][name]:
            continue
        zd = d["scores"][name][kind].astype(float)
        data_map = P.build_map(zd, d["mass"], d["pt"], EFF)
        zs = s["scores"][name][kind].astype(float)
        cut = P.passes(zs, s["mass"], s["pt"], data_map)
        q = s["qcd"]
        sim_map = P.build_map(zs[q], s["mass"][q], s["pt"][q], EFF)
        cut_sim = P.passes(zs, s["mass"], s["pt"], sim_map)
        sig = lambda sel, c: binomial(c[sel].sum(), sel.sum())
        out[kind] = dict(
            signal_at_data_cut=sig(s["three"] & acc_top, cut),
            top_like_at_data_cut=sig(s["top_like"] & acc_top, cut),
            with_b_at_data_cut=sig(s["three_b"] & acc_top, cut),
            without_b_at_data_cut=sig(s["three_nob"] & acc_top, cut),
            qcd_at_data_cut=sig(q, cut),
            qcd_top_window_at_data_cut=sig(q & acc_top, cut),
            signal_at_sim_cut=sig(s["three"] & acc_top, cut_sim),
            top_like_at_sim_cut=sig(s["top_like"] & acc_top, cut_sim),
            qcd_at_sim_cut=sig(q, cut_sim),
            # per native three-prong class: the flavour dependence at fixed prong count
            per_class_at_data_cut={_W["class_names"][c]: sig(s["three"] & acc_top & (s["label"] == c), cut)
                                   for c in np.unique(s["label"][s["three"] & acc_top])})
    return name, out


def step_domain(data, sim, fit, workers) -> dict:
    """Item 4: each model's efficiency on simulation at its own data cut, beside its
    data yield; and the flavour split of the three-prong efficiency."""
    _W.update(data=data, sim=sim, fit=fit,
              class_names={int(r["jet_label"]): r["class_name"] for r in D._anomaly.read_map()})
    names = [n for n in sorted(fit["models"]) if n in sim["scores"] and n in data["scores"]]
    rows = dict(_parallel(_domain_one, names, workers))
    for n, r in rows.items():
        t = fit["models"][n]["top"]
        r["data_yield"], r["data_yield_err"] = t["signal_yield"], t["signal_yield_err"]
        r["head_defect"] = HEAD_DEFECTS.get(n)
    runs = pretrained(names)
    if len(runs) < 3:
        return dict(models=rows, summary=None, note="fewer than three pretrained runs: no summary")
    y = np.array([rows[n]["data_yield"] for n in runs])
    sy = np.array([rows[n]["data_yield_err"] for n in runs])
    summary = {}
    for key in ("signal_at_data_cut", "signal_at_sim_cut", "top_like_at_data_cut", "qcd_at_data_cut"):
        e = np.array([rows[n]["three_prong"][key]["eff"] for n in runs])
        se = np.array([rows[n]["three_prong"][key]["err"] for n in runs])
        summary[key] = dict(correlation_with_yield=correlation(e, y),
                            by_label_set=by_label_set({n: dict(v=rows[n]["three_prong"][key]["eff"]) for n in runs}, "v"),
                            relative_spread=float(np.std(e, ddof=1) / np.mean(e)),
                            median_relative_error=float(np.median(se / e)))
    # tops implied by each run if data tops were tagged like simulated three-prong decays
    e = np.array([rows[n]["three_prong"]["signal_at_data_cut"]["eff"] for n in runs])
    se = np.array([rows[n]["three_prong"]["signal_at_data_cut"]["err"] for n in runs])
    r, sr = y / e, np.sqrt((sy / e) ** 2 + (y * se / e ** 2) ** 2)
    w = 1 / sr ** 2
    rbar = float((w * r).sum() / w.sum())
    chi2 = float((((r - rbar) / sr) ** 2).sum())
    yb = float((y / sy ** 2).sum() / (1 / sy ** 2).sum())
    chi2_y = float((((y - yb) / sy) ** 2).sum())
    summary["yield_over_sim_efficiency"] = dict(
        per_run={n: dict(value=float(a), err=float(b)) for n, a, b in zip(runs, r, sr)},
        weighted_mean=rbar, chi2=chi2, ndf=len(runs) - 1, p=float(stats.chi2.sf(chi2, len(runs) - 1)),
        yield_alone_chi2=chi2_y, yield_alone_p=float(stats.chi2.sf(chi2_y, len(runs) - 1)),
        reading="if every run's data yield were its simulated efficiency times one common number of "
                "tops, yield / efficiency would be constant (chi2/ndf ~ 1); the yields alone are the "
                "comparison: the smaller chi2 falls below yield_alone_chi2, the more of the run-to-run "
                "spread the simulated efficiency explains")
    return dict(models=rows, summary=summary,
                signal="JetClass-II test three-prong hadronic decays (native classes of 3P_HAD_3PARTON) "
                       "with 140 < m_SD < 220 GeV in the top fit's bins; top-like = X->YY->qqb and bcs; "
                       "with/without b by parton content",
                background="JetClass-II test QCD (native labels 161-187) in the rho window, all masses",
                data_cut="peak_fit.build_map at 1 % built on the DATA score of that model, applied to the "
                         "simulated jets' score (sim jet pT and m_SD)",
                sim_cut="the same procedure built on the simulated QCD jets")


def _prong_one(name):
    d, fit = _W["data"], _W["fit"]
    ref = fit["reference"]["top"]
    zp = d["scores"][name]["prong_only"].astype(float)
    zt = d["scores"][name]["three_prong"].astype(float)
    res, passed, _ = P.analyse(zp, d["mass"], d["pt"], "top", EFF, _W["n_toys"], shape=(ref["mean"], ref["width"]))
    three = P.passes(zt, d["mass"], d["pt"], P.build_map(zt, d["mass"], d["pt"], EFF))
    t = fit["models"][name]["top"]
    return name, dict(_fit_summary(res), criteria=P.criteria(dict(res, auc_vs_cms_proxy=1.0), "top", _W["n_toys"]),
                      three_prong_yield=t["signal_yield"], three_prong_yield_err=t["signal_yield_err"],
                      ratio_to_three_prong=res["signal_yield"] / t["signal_yield"],
                      overlap_jaccard=float((passed & three).sum() / (passed | three).sum()),
                      shared_fraction_of_prong_only=float((passed & three).sum() / passed.sum()))


def step_prong(data, fit, n_toys, workers) -> dict:
    """Item 5: the prong-only score through the whole procedure."""
    _W.update(data=data, fit=fit, n_toys=n_toys)
    names = [n for n in sorted(fit["models"]) if "prong_only" in data["scores"].get(n, {})]
    rows = dict(_parallel(_prong_one, names, workers))
    runs = pretrained(rows)
    return dict(models=rows,
                yield_by_label_set=by_label_set(rows, "signal_yield"),
                ratio_by_label_set=by_label_set(rows, "ratio_to_three_prong"),
                n_runs_peak_found=int(sum(rows[n]["z_wald"] >= 3 and rows[n]["criteria"]["peak_position"] for n in runs)),
                n_runs_validation_passes=int(sum(rows[n]["validation"]["passes"] for n in runs)),
                n_runs=len(runs),
                score="log[ sum P(three-prong hadronic) / sum P(two- and four-prong hadronic) ] "
                      "(discriminants.SCORES['prong_only']): no QCD node, so the resonance-vs-QCD "
                      "information cancels and only the prong count is left",
                procedure="peak_fit.analyse at 1 % data efficiency: the map (windows masked), the "
                          "floated-shape fit started from the reference's shape, the validation band; "
                          "the criteria's AUC term is not applicable and is set to pass",
                tests="whether the prong count alone, with the resonance-vs-QCD part of the score "
                      "removed and the mass decorrelated by the same map, selects the top peak, and "
                      "how many of the three-prong-vs-QCD score's tops it keeps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, type=pathlib.Path)
    ap.add_argument("--shards", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--fit", required=True, type=pathlib.Path, help="the main fit's results.json")
    ap.add_argument("--sim", type=pathlib.Path, default=None)
    ap.add_argument("--first-run-shards", nargs="+", type=pathlib.Path, default=None,
                    help="the shards the main fit was made from, when --shards is a rescore of them")
    ap.add_argument("--first-run-merged", type=pathlib.Path, default=None,
                    help="those shards merged: their three-prong scores are the ones every check uses")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--toys", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=None,
                    help="only these models in the per-model steps (a partial run; the text reads full runs)")
    a = ap.parse_args()

    fit = json.loads(a.fit.read_text())
    fit["_path"] = str(a.fit)
    if a.models:
        missing = set(a.models) - set(fit["models"])
        if missing:
            raise SystemExit(f"FATAL: {sorted(missing)} not in {a.fit}")
        fit["models"] = {n: fit["models"][n] for n in a.models}
        fit["_partial"] = sorted(a.models)
    data = load_data(a.merged, first=a.first_run_merged)
    sim = load_sim(a.sim) if a.sim and {"sim_closure", "domain"} & set(a.steps) else None
    a.out.mkdir(parents=True, exist_ok=True)
    inputs = {"fit": a.fit, "jets": a.merged / "jets.npz"}
    if sim is not None:
        inputs["sim_jets"] = a.sim / "jets.npz"

    def write(name, obj, extra=()):
        prov = dict(provenance({**inputs, **dict(extra)}), partial_models=fit.get("_partial"))
        (a.out / f"{name}.json").write_text(json.dumps(dict(obj, provenance=prov), indent=2, default=_plain))
        print(f"wrote {a.out / name}.json", flush=True)

    domain = None
    if "reproduce" in a.steps and a.first_run_shards:
        rep = step_reproduce(a.shards, a.first_run_shards)
        if a.first_run_merged:
            rep["cut"] = step_cut_flips(data, a.workers)
        write("reproduce", rep)
        if not rep["all_jets_identical"]:
            raise SystemExit("FATAL: the rescore staged other jets than the first run (reproduce.json)")
    if "selection" in a.steps:
        write("selection_chain", step_selection(a.shards, data, fit))
    if "closure" in a.steps:
        write("closure_all_shards", step_closure(a.shards),
              {f"closure_{s.name}": s / "closure.json" for s in a.shards})
    if "domain" in a.steps:
        domain = step_domain(data, sim, fit, a.workers)
        write("model_vs_domain", domain)
    if "per_run" in a.steps:
        write("per_run_yields", step_per_run(fit, domain))
    if "reference" in a.steps:
        write("reference", step_reference(data, fit, a.toys))
    if "sim_closure" in a.steps:
        write("map_closure_sim", step_sim_closure(sim, fit, a.workers))
    if "injection" in a.steps:
        write("map_injection_data", step_injection(data, fit, a.workers))
    if "prong" in a.steps:
        write("prong_test", step_prong(data, fit, a.toys, a.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
