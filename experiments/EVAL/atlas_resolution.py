#!/usr/bin/env python3
"""Jet-mass resolution with the ATLAS definition (PI decision P7, 2026-11-02).

THE DEFINITION. For a mass estimate m_est the response is r = m_est / m_gen, and

    resolution = 0.5 * (q84.1 - q15.9) / median        (ATL-PHYS-PUB-2024-015 Sec. 6, Fig. 8)

with q15.9 and q84.1 the 15.9 % and 84.1 % quantiles of r and the median response reported
beside it. The PI chose it over sigma_eff (half the smallest 68 % interval of ln r, the CMS /
GloParT form mass_resolution.py carries), which is no longer reported anywhere. Quantiles are
numpy's default ("linear": interpolation between the two nearest order statistics,
np.quantile(..., method="linear")), at exactly 0.159 and 0.841 as the definition's words,
and the median is np.median.

THE TARGET. m_gen is the generator-level soft-drop mass (genjet_sdmass) for resonant and QCD
jets alike. It is what the auxiliary mass regression was trained to reach (target
ln(genjet_sdmass / jet_sdmass), experiments/MTX/hybrid_mass.py, docs/DECISIONS.md D2 as
amended), what the frozen-feature probes regress (mass_resolution.py), and the m_gen of the
paper (outline Sec. 3.2 P5, 3.3 P7). The observers carry no resonance mass. A jet whose
generator match is missing has genjet_sdmass = 0.0 exactly (docs/GROUND_TRUTH.md) and is
dropped (mask m_gen > 0, and m_SD > 0), a validity cut and not a tail cut.

THE ESTIMATORS, all on the SAME jets: the test split of experiments/EVAL/mass_resolution.py
(its valid jets, its classes with >= 50 training jets, its seeded 60/20/20 split), so every
number here sits on the jets of the stored mass_v3.json and the four estimators are compared
on one sample.
  groomed    m_SD, the reconstructed soft-drop mass (no correction)
  regressed  m_SD exp(y), y the auxiliary mass output (column K of the output layer); only the
             models pretrained with the auxiliary mass regression
  ridge, mlp m_SD exp(y_pred + c), y_pred the frozen-feature probe's prediction of the target
             centred on its native class's training median c, refitted here exactly as
             mass_resolution.py fits them (its functions, features, split, centring and seeds),
             because mass_resolution.py kept no per-jet predictions (--save-residuals was not
             passed by scripts/build_head_jobs.py)
Every response is computed as exp of the log-ratio mass_resolution.py scores, so the overall
numbers check against mass_v3.json: the central 68 % half-width of ln r it stored
(sigma68_central) is recomputed here for every estimator and written beside the stored value.
The groomed and regressed values are the same arithmetic on the same jets and must agree to
1e-9, or the jets differ and the job stops (exit 42, no retry). The ridge and MLP values are
refits: the ridge agrees to ~1e-6, the MLP to ~1e-3 on another CPU model (latent_scale_probe.py
MLP_THREADS note), and both differences are recorded, not enforced.

THE BINS. Per group (resonant = native label < 161, qcd, and all jets) the inclusive values and
the values in bins [lo, hi) of the reconstructed jet pT and of the generator soft-drop mass,
with the edges mass_resolution.py's binned blocks (head/<model>_mass.json) use, so the panels
match the earlier binning. A bin of fewer than MIN_JETS jets carries its count and null values.

THE ERRORS. Every value with at least MIN_JETS jets carries the statistical SD of its median and
of its resolution (median_sd, resolution_sd) from a jet-level bootstrap: BOOT_RESAMPLES
resamples with replacement, each seeded with BOOT_SEED, so two estimators on the same jets
(same count) are resampled identically and their bars are paired. These are the per-bin bars
of the groomed mass in Figure F8a (one value per bin, no training runs) and the per-bin errors
of appendix A17 (docs/FIGURE_PLAN.md); the spread over training runs is the readers' own.

THE HISTOGRAM. Per estimator and group, the counts of r on HIST_EDGES with the jets below and
above them, on the same test jets, so F8a's response panel and its resolution panels sit on
one sample (head/curves_v2/*_response.json is filled from every matched jet, five times as
many, and holds ln r).

THE OUTPUT. <out-dir>/<model>.json (schema: the Phase B brief's interface, plus provenance
keys) and <out-dir>/<model>_perjet.npz, the per-jet values every number is computed from
(stream row, native label, jet pT, m_gen and each estimate); the summaries, bootstrap errors
and histograms are computed from exactly those values. An existing JSON is left as it is.

    python3 experiments/EVAL/atlas_resolution.py --model mtx-l162mass-s1 \\
        --observers /data/results/eval/v2_head/mtx-l162mass-s1/test/best70 \\
        --features /data/results/eval/v2_linprobe/mtx-l162mass-s1/jc2/test/best70 \\
        --head /data/results/eval/v2_head/mtx-l162mass-s1/test/best70 \\
        --mass-v3 /data/results/eval/v2_head/mtx-l162mass-s1/mass_v3.json \\
        --out-dir /data/results/eval/atlas_res
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.EVAL import mass_resolution as MR  # noqa: E402

QUANTILES = (0.159, 0.841)
MIN_JETS = 100
DEFINITION = ("half(q84.1 - q15.9)/median of r = m_est/m_gen, m_gen the generator-level soft-drop "
              "mass for resonant and QCD jets alike (ATL-PHYS-PUB-2024-015 Sec. 6); numpy linear "
              "quantiles at 0.159 and 0.841")
GROUPS = {"resonant": lambda lab: lab < MR.N_RESONANT, "qcd": lambda lab: lab >= MR.N_RESONANT,
          "all": lambda lab: np.ones(lab.shape, bool)}
CHECK_TOL = 1e-9        # groomed and regressed: the same arithmetic on the same jets as mass_v3.json
BOOT_RESAMPLES, BOOT_SEED = 200, 20261102
HIST_EDGES = np.linspace(0.0, 2.0, 101)     # r, bins of 0.02 (np.histogram: the last edge is closed)


def edges(bins) -> list:
    """The edges of contiguous [lo, hi) bins (mass_resolution.PT_BINS, MGEN_BINS)."""
    for (_a, hi), (lo, _b) in zip(bins, bins[1:]):
        if hi != lo:
            raise SystemExit(f"FATAL: bins {bins} are not contiguous")
    return [bins[0][0]] + [hi for _lo, hi in bins]


PT_EDGES = edges(MR.PT_BINS)          # reconstructed jet pT, GeV
MGEN_EDGES = edges(MR.MGEN_BINS)      # generator-level soft-drop mass, GeV


KEYS = ("median", "q159", "q841", "resolution", "n", "median_sd", "resolution_sd")


def bootstrap_sd(r: np.ndarray) -> tuple[float, float]:
    """The bootstrap SD (ddof 1) of the median and of the resolution of r: BOOT_RESAMPLES
    resamples of r.size jets with replacement, the generator seeded with BOOT_SEED."""
    rng = np.random.default_rng(BOOT_SEED)
    med, res = np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES)
    for b in range(BOOT_RESAMPLES):
        lo, m, hi = np.quantile(r[rng.integers(0, r.size, r.size)], (QUANTILES[0], 0.5, QUANTILES[1]))
        med[b], res[b] = m, 0.5 * (hi - lo) / m
    return float(med.std(ddof=1)), float(res.std(ddof=1))


def summary(r: np.ndarray) -> dict:
    """Median, the 15.9 / 84.1 % quantiles, the resolution and the bootstrap SDs of the median and
    the resolution of a response sample; null values (with the count) below MIN_JETS jets."""
    n = int(r.size)
    if n < MIN_JETS:
        return {k: n if k == "n" else None for k in KEYS}
    med = float(np.median(r))
    q159, q841 = (float(q) for q in np.quantile(r, QUANTILES))
    med_sd, res_sd = bootstrap_sd(r)
    return {"median": med, "q159": q159, "q841": q841, "resolution": 0.5 * (q841 - q159) / med, "n": n,
            "median_sd": med_sd, "resolution_sd": res_sd}


def binned(r: np.ndarray, x: np.ndarray, e: list) -> dict:
    """summary() in the bins [e[i], e[i+1]) of x, as lists; jets outside the edges are in no bin."""
    cells = [summary(r[(x >= lo) & (x < hi)]) for lo, hi in zip(e[:-1], e[1:])]
    return {"edges": list(e), **{k: [c[k] for c in cells] for k in KEYS}}


def histogram(r: np.ndarray) -> dict:
    """Counts of r on HIST_EDGES and the jets below and above them."""
    return {"edges": HIST_EDGES.tolist(), "counts": np.histogram(r, bins=HIST_EDGES)[0].tolist(),
            "underflow": int((r < HIST_EDGES[0]).sum()), "overflow": int((r > HIST_EDGES[-1]).sum())}


def response_block(r: np.ndarray, lab: np.ndarray, pt: np.ndarray, mgen: np.ndarray) -> dict:
    """One estimator's response per group: inclusive, against jet pT, against m_gen, histogram."""
    out = {}
    for g, sel in GROUPS.items():
        m = sel(lab)
        out[g] = {"inclusive": summary(r[m]), "vs_pt": binned(r[m], pt[m], PT_EDGES),
                  "vs_mgen": binned(r[m], mgen[m], MGEN_EDGES), "hist": histogram(r[m])}
    return out


def frozen_jets(obs: dict) -> dict:
    """mass_resolution.main's jets, split and centring, step for step: the valid jets, the
    classes with enough training jets, the split recomputed over them, and the target centred
    on its class's training median. `rows` are positions in the observers' arrays."""
    vidx = np.flatnonzero(obs["valid"])
    y_all, lab_all = obs["y"], obs["label188"]
    tr0, _va0, _te0 = MR.make_splits(vidx.size)
    _y_c, usable, _cinfo = MR.class_center(y_all[vidx], lab_all[vidx], tr0)
    keep = np.flatnonzero(usable)
    tr, va, te = MR.make_splits(keep.size)
    y_raw = y_all[vidx][keep]
    lab_use = lab_all[vidx][keep]
    med = MR.class_median(y_raw, lab_use, tr)[lab_use]
    if np.isnan(med).any():
        raise SystemExit("FATAL: a kept class has no jet in the training split")
    return {"vidx": vidx, "keep": keep, "rows": vidx[keep], "tr": tr, "va": va, "te": te,
            "y_raw": y_raw, "y_use": y_raw - med, "med": med, "label188": lab_use}


def fatal(msg: str) -> None:
    """A deterministic failure: exit 42, which the job's podFailurePolicy does not retry."""
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(42)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def stored_central(mass_v3: dict | None, arm: str) -> dict:
    """mass_v3.json's sigma68_central of each estimator this file recomputes (absent: none)."""
    if mass_v3 is None:
        return {}
    out = {"groomed": mass_v3["uncorrected"]["sigma68_central"]}
    if arm in mass_v3.get("heads", {}):
        out["regressed"] = mass_v3["heads"][arm]["sigma68_central"]
    if arm in mass_v3.get("arms", {}):
        out |= {p: mass_v3["arms"][arm][p]["sigma68_central"] for p in ("ridge", "mlp")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the output's name, e.g. mtx-l162mass-s1 or init-s1")
    ap.add_argument("--observers", required=True, type=pathlib.Path,
                    help="an extract_v2.py --keep-head-outputs checkpoint directory (head_outputs.npz)")
    ap.add_argument("--features", required=True, type=pathlib.Path,
                    help="the model's linear-probe cache of the same JetClass-II test jets")
    ap.add_argument("--arm", default="best70", help="the arm / head name in mass_v3.json (init for "
                                                    "the untrained trunks)")
    ap.add_argument("--head", type=pathlib.Path, default=None,
                    help="checkpoint directory with the auxiliary mass output (mass-regression models)")
    ap.add_argument("--mass-v3", type=pathlib.Path, default=None,
                    help="the model's mass_v3.json, checked against (recommended)")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    out_json = a.out_dir / f"{a.model}.json"
    out_npz = a.out_dir / f"{a.model}_perjet.npz"
    if out_json.exists():
        print(f"{out_json} exists; not overwritten")
        return 0
    obs = MR.load_observers(a.observers)
    J = frozen_jets(obs)
    te = J["te"]
    rows = J["rows"][te]
    o = {k: np.asarray(v)[rows].astype(np.float64) for k, v in obs["obs"].items()}
    gen, pt, lab = o["genjet_sdmass"], o["jet_pt"], J["label188"][te].astype(np.int64)
    print(f"{a.model}: {obs['n']:,} jets, {obs['n_valid']:,} matched, {te.size:,} test jets "
          f"(row alignment {obs['label188_sha256'][:16]})")

    # ln(m_est / m_gen) of each estimator, the expression mass_resolution.py scores
    lr, inputs, fits = {"groomed": -J["y_raw"][te]}, {}, {}
    if a.head is not None:
        m, prov = MR.load_head(a.arm, a.head, obs["n"], obs["label188_sha256"])
        lr["regressed"] = m[rows] - J["y_raw"][te]
        inputs["head"] = {"dir": str(a.head), **prov}
    v3 = json.loads(a.mass_v3.read_text()) if a.mass_v3 else None
    stored = stored_central(v3, a.arm)
    check = {}

    def record(est: str) -> None:
        got = MR.resolution(lr[est])["sigma68_central"]
        ref = stored.get(est)
        check[est] = {"stored": ref, "recomputed": got, "abs_diff": None if ref is None else abs(got - ref)}

    for est in list(lr):
        record(est)
        d = check[est]["abs_diff"]
        if d is not None and d > CHECK_TOL:
            fatal(f"{a.model} {est}: sigma68_central {check[est]['recomputed']:.9f} here, "
                  f"{check[est]['stored']:.9f} in {a.mass_v3}: not the jets mass_v3.json scored")
    if v3 is not None and a.head is not None and "regressed" not in stored:
        fatal(f"{a.mass_v3} has no head {a.arm!r} to check the mass output against")
    if v3 is not None and "ridge" not in stored:
        fatal(f"{a.mass_v3} has no arm {a.arm!r} to check the frozen-feature probes against")

    # the frozen-feature probes, refitted as mass_resolution.main fits them
    F = np.load(a.features / "features.npy")
    sel = MR.v2_prefix(a.features, obs["n"])
    if sel is not None:
        F = F[sel].astype(np.float32)
    if F.shape[0] != obs["n"]:
        fatal(f"{a.model}: {F.shape[0]} feature rows, the observers have {obs['n']}")
    inputs["features"] = {"dir": str(a.features),
                          **MR.check_alignment(a.arm, a.features, obs["label188_sha256"], obs["n"])}
    Fu = F[J["vidx"]][J["keep"]]
    del F
    tr, va, y = J["tr"], J["va"], J["y_use"]
    for p, fit in (("ridge", MR.fit_ridge), ("mlp", MR.fit_mlp)):
        pred, fits[p] = fit(Fu[tr], y[tr], Fu[va], y[va], Fu[te])
        lr[p] = pred - y[te]
        record(p)
        print(f"  {p}: sigma68_central {check[p]['recomputed']:.6f} (mass_v3.json {check[p]['stored']}); "
              f"|diff| {check[p]['abs_diff']}")

    # per-jet values, then the summaries computed from exactly those values
    resp = {est: np.exp(v) for est, v in lr.items()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, rows=rows.astype(np.int64), label188=lab, jet_pt=pt, genjet_sdmass=gen,
                        **{f"m_{est}": gen * r for est, r in resp.items()})
    doc = {"model": a.model, "definition": DEFINITION,
           "estimators": {est: response_block(r, lab, pt, gen) for est, r in resp.items()},
           "source": {"file": str(out_npz), "sha256": sha256(out_npz)},
           "jets": {"n_stream": obs["n"], "n_matched": obs["n_valid"], "n_test": int(te.size),
                    "split_seed": MR.SPLIT_SEED, "row_alignment_sha256": obs["label188_sha256"],
                    "test": "mass_resolution.py's test split of the matched jets of classes with >= "
                            f"{MR.MIN_TRAIN_PER_CLASS} training jets"},
           "groups": "resonant: native label < 161; qcd: >= 161; all: both",
           "bins": "[lo, hi) GeV; vs_pt in the reconstructed jet pT, vs_mgen in the generator-level "
                   f"soft-drop mass; fewer than {MIN_JETS} jets: null values, the count kept",
           "errors": f"median_sd, resolution_sd: SD (ddof 1) over {BOOT_RESAMPLES} jet-level bootstrap "
                     f"resamples, seed {BOOT_SEED} for every value (equal counts resample identically)",
           "hist": "counts of r = m_est/m_gen in [edges[i], edges[i+1]) (last bin closed), on the same "
                   "test jets as the summaries; underflow / overflow: jets outside the edges",
           "inputs": {"observers": {"file": str(a.observers / MR.HEAD_FILE),
                                    "sha256": sha256(a.observers / MR.HEAD_FILE)},
                      "mass_v3": None if a.mass_v3 is None else {"file": str(a.mass_v3),
                                                                 "sha256": sha256(a.mass_v3)},
                      **inputs},
           "check_against_mass_v3": {"statistic": "sigma68_central: half the central 68% range of "
                                                  "ln r (mass_resolution.resolution)", **check},
           "fits": fits, "cpu_model": MR.cpu_model(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out_json)           # atomic: an existing JSON means done to a retried pod
    for est, b in doc["estimators"].items():
        for g in ("resonant", "qcd"):
            c = b[g]["inclusive"]
            print(f"  {est:9s} {g:8s} median {c['median']} resolution {c['resolution']} n {c['n']}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
