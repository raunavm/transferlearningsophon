#!/usr/bin/env python3
"""S7: frozen-feature jet-mass regression, as a WITHIN-CLASS resolution.

The prediction (docs/PRESPEC_2026-09.md, S7): resolution improves with the mass
output at both granularities; the gain is larger at 17 classes; without the mass
output, finer labels give equal or better resolution.

THE FOUR CHOICES THIS SCRIPT MAKES ARE PRE-REGISTERED, NOT FREE. S7 was signed
with "paired differences" and no definition of resolution, which left four knobs
on a directional prediction. They were fixed in writing, by the PI, before any
number existed (DECISIONS_PENDING item 40; PRESPEC amendment 2026-09-20):

1. RESOLUTION = the "effective resolution": HALF THE SMALLEST INTERVAL
   CONTAINING 68 % OF THE AREA of the residual distribution. The PI's
   instruction was to use what the field uses if it applies, and the field does
   have a settled answer -- a 68 % interval half-width on the mass response,
   never a standard deviation and never a Gaussian fit. ATLAS states the
   rejection in print: the interquantile range "is used instead of a gaussian
   width or standard deviation to reduce the sensitivity to non-gaussian tails
   and asymmetric response shapes" (arXiv 2607.25893 p.8).

   WHICH 68 % INTERVAL is a real fork, because "sigma_eff" names two different
   statistics. CMS means the SMALLEST such interval (CMS-DP-2021-017 p.5);
   ATLAS and the phenomenology literature mean the CENTRAL one, (q84-q16)/2
   (arXiv 2607.25893 p.8; PELICAN arXiv 2307.16506 p.20). This study takes the
   CMS form, because it is this study's own lineage: the mass-regression recipe
   is inherited from GloParT, whose definition (CERN-THESIS-2024-281 p.172) is
   the CMS one, and whose author is the first author of the architecture used
   here. The central form is reported beside it so that the fork is visible and
   cannot be resolved after the fact, and the standard deviation beside both --
   the two disagree in DIRECTION, not merely in size, when one model is better
   in the core and worse in the tail, which is exactly what a model trained to
   regress mass might be. Only the effective resolution carries the prediction.

2. "CLASS" = the 188 native labels. The requirement that settles this is not
   granularity but ARM INDEPENDENCE: scoring each model on its own vocabulary
   would score them on different partitions, the paired difference would stop
   being one quantity, and the 17-class model would be credited for having less
   to subtract. Of the arm-independent partitions the 188 native labels are the
   finest, hence the most conservative, and `label188.npy` is already cached
   beside every feature file.

3. "FAMILY MEANS REMOVED" (docs/DOWNSTREAM_SUITE.md:217-220) IS SUBSUMED by 2
   and is NOT a second operation. Each family is a union of native classes, so
   centering within the 188 labels has already removed the family means.
   Doing both would subtract the same thing twice. Stated here because the
   sentence in that document reads like two steps and is not.

4. NO TRIMMING. The 68 % half-width is tail-robust by construction, so a trim
   would be a second, redundant tail rule and a second free knob. The tail is
   REPORTED instead of removed: `tail_fraction` is the share of test jets with
   |residual| > 1 in the log-ratio. Jets whose generator-level groomed mass is
   unmatched are a different thing entirely -- they are stored as a hard 0.0 and
   are dropped as invalid, which is a validity cut and not a choice.

WHY CENTER AT ALL. Knowing a jet's class already tells you roughly its mass. A
model that is merely a better CLASSIFIER would therefore look like a better mass
regressor, and since this study varies nothing but the label vocabulary that is
precisely the confound that would make the contrast uninterpretable. Centering
the TARGET within class -- not the residual -- is what removes it: after
centering, the class carries no information about the target, so whatever the
probe recovers is within-class mass information. Class means come from the
TRAINING split alone and are applied to all three, or the test set leaks into
its own centering.

ALIGNMENT IS ASSERTED, NOT ASSUMED. The generator-level groomed mass lives in
one place, written by extract_observers.py, which verified row alignment against
the twenty granularity caches and refuses to write on a mismatch. The ten
mass-output caches were NOT in that check -- they did not exist when it ran --
so for them alignment rested on having used the same file list and settings and
nothing verified it. This script requires every arm's `label188_sha256` to equal
the observers'. A silent row offset between the features and the target would be
invisible in every number below.
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

from experiments.EVAL.latent_scale_probe import (  # noqa: E402
    SPLIT_SEED, fit_mlp, fit_ridge, make_splits)

TAIL_AT = 1.0          # |residual| > 1 in the log-ratio: choice 4, reported not trimmed
MIN_TRAIN_PER_CLASS = 50   # below this a class mean is noise; see drop_small_classes
PROBES = ("ridge", "mlp")


def sigma_eff(res: np.ndarray, frac: float = 0.68) -> tuple[float, float, float]:
    """Half the SMALLEST interval containing `frac` of the area. Returns (half, lo, hi).

    This is CMS's "effective resolution", and it is the headline because it is
    this project's own lineage: CMS-DP-2021-017 p.5 defines it as "half of the
    minimum interval containing the mode and 68% of the area under the response
    distributions", and the GloParT thesis (CERN-THESIS-2024-281 p.172), by the
    first author of the architecture used here, repeats it as "the half of the
    smallest interval containing 68% of the area under the curve".

    For a unimodal distribution the minimum-width window contains the mode, so
    scanning windows is the same thing as the definition's words.
    """
    x = np.sort(np.asarray(res, dtype=np.float64))
    n = x.size
    k = int(np.ceil(frac * n))
    if k >= n:
        return float((x[-1] - x[0]) / 2.0), float(x[0]), float(x[-1])
    widths = x[k - 1:] - x[:n - k + 1]
    i = int(np.argmin(widths))
    return float(widths[i] / 2.0), float(x[i]), float(x[i + k - 1])


def resolution(res: np.ndarray) -> dict:
    """The pre-registered resolution of a residual array, and its companions.

    THE RESIDUAL IS ALREADY THE LOG OF THE MASS RESPONSE. The probe predicts
    log(genjet_sdmass / jet_sdmass), so `pred - y` is log(m_predicted /
    m_true): exponentiating it gives exactly the response ratio m_pred/m_target
    that every source below quantifies. That is why no further normalisation is
    applied -- the quantity is dimensionless by construction, and
    `fractional` converts it to the percentage the field quotes.

    `sigma_eff` is the headline (choice 1, CMS/GloParT lineage). The other two
    widths are NOT a menu: the name "sigma_eff" means the smallest 68 % interval
    to a CMS reader and the central one to an ATLAS reader, so both are reported
    and which is which is stated, and the standard deviation is reported beside
    them because it is what a reader outside the field expects and because it is
    the one that moves when the tail does. Only `sigma_eff` carries the
    prediction. `tail_fraction` is choice 4: the tail is shown, never trimmed.
    """
    res = np.asarray(res, dtype=np.float64)
    eff, lo, hi = sigma_eff(res)
    q16, q84 = np.percentile(res, [15.865, 84.135])
    return {"sigma_eff": eff, "sigma_eff_interval": [lo, hi],
            "sigma68_central": float((q84 - q16) / 2.0),
            "sd": float(res.std(ddof=1)),
            # The scale, which every source in the field insists travels with the
            # width: a narrower twin that is merely mis-scaled is not a better one.
            "median": float(np.median(res)),
            "mode_of_eff_interval": float((lo + hi) / 2.0),
            "fractional": float(np.expm1(eff)),
            "tail_fraction": float((np.abs(res) > TAIL_AT).mean()),
            "tail_at": TAIL_AT, "n": int(res.size)}


def class_center(y: np.ndarray, lab: np.ndarray, train_idx: np.ndarray) -> tuple:
    """Subtract the per-native-class mean of the target (choice 2).

    Means are fitted on the training rows only and applied everywhere. Classes
    with too few training rows have no usable mean, and a mean estimated from a
    handful of jets would inject noise that differs per class; those jets are
    dropped from ALL splits and counted, rather than being centered badly.
    """
    lab = np.asarray(lab)
    counts = np.bincount(lab[train_idx], minlength=188)
    keep_cls = counts >= MIN_TRAIN_PER_CLASS
    sums = np.bincount(lab[train_idx], weights=y[train_idx], minlength=188)
    means = np.divide(sums, np.maximum(counts, 1), where=True)
    means[~keep_cls] = np.nan
    usable = keep_cls[lab]
    return y - np.where(usable, means[lab], 0.0), usable, {
        "n_classes_present": int((counts > 0).sum()),
        "n_classes_used": int(keep_cls.sum()),
        "min_train_per_class": MIN_TRAIN_PER_CLASS,
        "n_jets_dropped_small_class": int((~usable).sum())}


def load_observers(path: pathlib.Path) -> dict:
    """genjet_sdmass, jet_sdmass and the native labels, with their digest."""
    obs = dict(np.load(path / "observers.npz"))
    for k in ("genjet_sdmass", "jet_sdmass"):
        if k not in obs:
            raise SystemExit(f"FATAL: {path}/observers.npz lacks {k!r}; present: {sorted(obs)}")
    lab = np.load(path / "label188.npy")
    man = json.loads((path / "observers_manifest.json").read_text())
    sha = man.get("label188_sha256")
    if not sha:
        raise SystemExit(f"FATAL: {path}/observers_manifest.json has no label188_sha256, "
                         f"so no arm's row alignment can be checked against it.")
    gen = obs["genjet_sdmass"].astype(np.float64)
    sd = obs["jet_sdmass"].astype(np.float64)
    # A hard 0.0 means the jet was never matched to a generator-level jet
    # (docs/GROUND_TRUTH.md). Mask on > 0. This is a validity cut, not a tail cut.
    valid = (gen > 0) & (sd > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.where(valid, gen, 1.0) / np.where(valid, sd, 1.0))
    return {"y": y, "valid": valid, "label188": lab, "label188_sha256": sha,
            "n": int(gen.size), "n_valid": int(valid.sum()), "manifest": man}


def check_alignment(arm: str, d: pathlib.Path, obs_sha: str, n: int) -> dict:
    """Refuse unless this arm was scored on the same jets, in the same order."""
    man_path = d / "extract_manifest.json"
    if not man_path.exists():
        raise SystemExit(f"FATAL: {arm}: no {man_path}")
    man = json.loads(man_path.read_text())
    sha = man.get("label188_sha256")
    if sha is None:
        raise SystemExit(
            f"FATAL: {arm}: {man_path} carries no label188_sha256, so its row alignment "
            f"with the generator-level mass cannot be checked. S7 regresses one cached "
            f"array against another; an unchecked offset is invisible in every number.")
    if sha != obs_sha:
        raise SystemExit(
            f"FATAL: {arm}: label188_sha256 {sha[:16]} does not match the observers' "
            f"{obs_sha[:16]}. The features and the mass target are not the same jets in "
            f"the same order. This is the check the observer job ran against the twenty "
            f"granularity caches and could not run against the mass-output ones, because "
            f"they did not exist yet.")
    if int(man.get("n_jets", n)) != n:
        raise SystemExit(f"FATAL: {arm}: {man.get('n_jets')} jets, observers have {n}")
    return {"label188_sha256": sha, "n_jets": n,
            "checkpoint_sha256": man.get("checkpoint_sha256")}


def probe_arm(F: np.ndarray, y: np.ndarray, tr, va, te) -> dict:
    """Both probes, always. D6: a linear result never stands on its own."""
    out = {}
    pred, meta = fit_ridge(F[tr], y[tr], F[va], y[va], F[te])
    out["ridge"] = {**resolution(pred - y[te]), **meta}
    pred, meta = fit_mlp(F[tr], y[tr], F[va], y[va], F[te])
    out["mlp"] = {**resolution(pred - y[te]), "fit": meta}
    # The spread of the centered target itself: the resolution a probe that
    # learned nothing would report. Without it a resolution is not interpretable.
    out["target"] = resolution(y[te])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", nargs="+", required=True, help="ARM=/path/to/features")
    ap.add_argument("--observers", required=True,
                    help="directory holding observers.npz, label188.npy and "
                         "observers_manifest.json (the test2m_observers cache)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out)
    obs = load_observers(pathlib.Path(a.observers))
    print(f"observers: {obs['n']:,} jets, {obs['n_valid']:,} with a matched "
          f"generator-level groomed mass ({obs['n_valid'] / obs['n']:.1%}); "
          f"row alignment {obs['label188_sha256'][:16]}")

    res = {"prediction": "S7: resolution improves with the mass output at both "
                         "granularities; larger gain at 17 classes; without the mass "
                         "output, finer labels give equal or better resolution",
           "target": "log(genjet_sdmass / jet_sdmass), the groomed log mass-ratio",
           "resolution_statistic": "sigma_eff = half the SMALLEST interval containing "
                                   "68% of the residual area (CMS-DP-2021-017 p.5; "
                                   "CERN-THESIS-2024-281 p.172). The central 68% "
                                   "half-width and the standard deviation are reported "
                                   "beside it and carry no prediction.",
           "residual": "log(m_predicted / m_true); exponentiating gives the mass "
                       "response ratio the field quantifies",
           "centering": "per-188-native-class mean of the target, fitted on the "
                        "training split only; subsumes family-mean removal",
           "tail": f"reported as the share with |residual| > {TAIL_AT}, never trimmed",
           "split_seed": SPLIT_SEED, "tail_at": TAIL_AT,
           "row_alignment_sha256": obs["label188_sha256"],
           "n_jets_total": obs["n"], "n_jets_valid": obs["n_valid"],
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "arms": {}}

    vidx = np.flatnonzero(obs["valid"])
    y_all, lab_all = obs["y"], obs["label188"]
    tr0, va0, te0 = make_splits(vidx.size)
    y_c, usable, cinfo = class_center(y_all[vidx], lab_all[vidx], tr0)
    # Dropping the small classes changes the split membership, so recompute the
    # split over what survives rather than filtering the old indices: every arm
    # sees the identical split either way, which is the property that matters.
    keep = np.flatnonzero(usable)
    tr, va, te = make_splits(keep.size)
    y_use = y_c[keep]
    lab_use = lab_all[vidx][keep]
    res["centering_detail"] = {**cinfo, "n_jets_used": int(keep.size),
                               "split": [int(tr.size), int(va.size), int(te.size)]}
    print(f"centering: {cinfo['n_classes_used']} of {cinfo['n_classes_present']} native "
          f"classes have >= {MIN_TRAIN_PER_CLASS} training jets; "
          f"{cinfo['n_jets_dropped_small_class']:,} jets dropped; "
          f"{keep.size:,} used, split {tr.size}/{va.size}/{te.size}")
    print(f"centered target spread: sigma_eff "
          f"{resolution(y_use[te])['sigma_eff']:.4f} (an uninformed probe's resolution)")

    for spec in a.features:
        if "=" not in spec:
            raise SystemExit(f"FATAL: --features wants ARM=path, got {spec!r}")
        name, path = spec.split("=", 1)
        d = pathlib.Path(path)
        F = np.load(d / "features.npy")
        if F.shape[0] != obs["n"]:
            raise SystemExit(f"FATAL: {name}: {F.shape[0]} feature rows, "
                             f"observers have {obs['n']}")
        prov = check_alignment(name, d, obs["label188_sha256"], obs["n"])
        Fu = F[vidx][keep]
        r = probe_arm(Fu, y_use, tr, va, te)
        r["provenance"] = prov
        res["arms"][name] = r
        print(f"\n=== {name} ===  {Fu.shape[0]:,} jets, {Fu.shape[1]}-d")
        for p in PROBES:
            v = r[p]
            print(f"  {p:6s} sigma_eff {v['sigma_eff']:.4f} ({v['fractional']:+.1%})  "
                  f"central68 {v['sigma68_central']:.4f}  sd {v['sd']:.4f}  "
                  f"median {v['median']:+.4f}  tail(|r|>{TAIL_AT:g}) {v['tail_fraction']:.4f}")

    res["n_classes_used"] = cinfo["n_classes_used"]
    res["labels_used_sha256"] = hashlib.sha256(lab_use.tobytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "mass_resolution.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/mass_resolution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
