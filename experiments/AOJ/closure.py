#!/usr/bin/env python3
"""Input-distribution closure: staged AspenOpenJets vs JetClass-II QCD, on the
17 tensors the trunk actually receives.

scripts/stage_aoj.py translates a real-detector format into a simulated one, and
some of its conventions could not be verified from the producer code (its
UNVERIFIED_* constants). A wrong one is SILENT: the model runs, scores come out,
a peak may even appear, and every number is quietly degraded. So, as
experiments/FT/loadcheck.py does for the benchmark sets, both samples are read
through weaver's own loader -- the config's transforms applied by the code that
applies them in inference, not re-implemented here -- and compared feature by
feature on real (unpadded) particles.

THE REFERENCE IS JETCLASS-II QCD THROUGH THE SAME CUTS
(configs/finetune/JetClassII_base_selAspenOpenJets.yaml), and both sides are
further restricted to one jet-pT slice. JetClass-II's pT spectrum is far flatter
than the data's, and d0err, deltaR and multiplicity all move with jet pT, so
without the slice this table would measure the spectra and not the conventions.

FLAGS
  hard  support         a feature's [p1, p99] ranges do not overlap
  hard  unit            median |d0|, d0err, |dz| or dzerr ratio within a factor 2
                        of 10 or 1/10: the cm -> mm conversion is wrong
                        (UNVERIFIED_CM_TO_MM)
  hard  d0_dphi_sign,   the sign asymmetry <sign(d0 * dphi)> (resp. dz * deta) of
        dz_deta_sign    displaced tracks is significant in BOTH samples and
                        OPPOSITE: the impact-parameter sign convention is
                        flipped (UNVERIFIED_D0_SIGN / UNVERIFIED_DZ_SIGN)
  soft  *_one_sided     that asymmetry is significant in one sample and absent in
                        the other (e.g. JetClass-II mirrors dz with the jet-eta
                        sign, as it does deta, and the staging does not)
  soft  species         a particle-type fraction is off by more than +-30 %
  soft  charged_no_track  more than 20 % of charged candidates carry no
                        displacement (PFNano stores none without track
                        details); in JetClass-II a zero means "neutral"
  soft  neutral_with_displacement  JetClass-II neutrals are NOT all at zero
                        displacement, which is what stage_aoj.py assumed
A hard flag makes the feasibility verdict NO-GO whatever the peaks look like.

POOLING THE SHARDS (2026-09-29, audit B5). The full run checks closure once per
shard, on that shard's files. A median or an IQR cannot be averaged across
shards, so every continuous row also carries each sample's quantile function
(QUANTILE_LEVELS) and its particle count; pool_shards() rebuilds each shard's
CDF from it, averages the CDFs weighted by count, and reads the pooled median,
IQR, 1st/99th percentiles and the Kolmogorov-Smirnov distance to the reference
off the result -- exact to the quantile grid. The reference is the same JetClass-
II jets in every shard; pool_shards() refuses shards whose reference differs.
Closures written before this (no quantiles) are summarised shard by shard.

Run:  python3 experiments/AOJ/closure.py --aoj /scratch/staged/*.parquet \
          --reference /jc2/jet_data/QCD_035{0..3}.parquet --out /data/results/aoj/feasibility_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
AOJ_CONFIG = REPO / "configs" / "finetune" / "AspenOpenJets.yaml"
REF_CONFIG = REPO / "configs" / "finetune" / "JetClassII_base_selAspenOpenJets.yaml"

SPECIES = ["part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
           "part_isElectron", "part_isMuon"]
DISPLACEMENT = ["part_d0", "part_d0err", "part_dz", "part_dzerr"]
SPECIES_TOLERANCE = 0.30
UNIT_BANDS = ((5.0, 20.0), (0.05, 0.2))
NO_TRACK_MAX = 0.20
DISPLACED = 0.1          # |tanh(d0 / mm)| above which a track counts as displaced
ASYM_SIGMA = 5.0
QUANTILE_LEVELS = np.linspace(0.0, 1.0, 1001)


def load_inputs(config, files, pt_window, max_jets):
    """(features (N, 17, P), mask (N, P), feature names) through weaver's loader."""
    import torch
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset
    dc = DataConfig.load(str(config), load_observers=True)
    ds = SimpleIterDataset({"_": list(files)}, str(config), for_training=False,
                           fetch_by_files=True, fetch_step=1, name="closure")
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, drop_last=False, num_workers=0)
    feats, masks, n = [], [], 0
    for X, _y, Z in dl:
        pt = np.asarray(Z["jet_pt"])
        keep = (pt > pt_window[0]) & (pt < pt_window[1])
        feats.append(X["pf_features"].numpy()[keep])
        masks.append(X["pf_mask"].numpy()[keep, 0].astype(bool))
        n += int(keep.sum())
        if n >= max_jets:
            break
    if not n:
        raise SystemExit(f"FATAL: no jet of {list(files)[:2]}... in pT window {pt_window}")
    return np.concatenate(feats), np.concatenate(masks), list(dc.input_dicts["pf_features"])


def _quantiles(x):
    p1, p25, p50, p75, p99 = np.percentile(x, [1, 25, 50, 75, 99])
    return dict(median=float(p50), iqr=float(p75 - p25), p01=float(p1), p99=float(p99))


def _qfunc(x):
    return [float(v) for v in np.quantile(x, QUANTILE_LEVELS)]


def _asymmetry(a, b):
    """<sign(a * b)> over displaced tracks, and its binomial error."""
    sel = np.abs(a) > DISPLACED
    n = int(sel.sum())
    if n < 100:
        return dict(n=n, asym=None, err=None)
    return dict(n=n, asym=float(np.mean(np.sign(a[sel] * b[sel]))), err=float(1 / np.sqrt(n)))


def compare(f_aoj, m_aoj, f_ref, m_ref, names):
    """Closure rows and flags from two (features, mask) pairs."""
    col = lambda f, m, k: f[:, names.index(k), :][m]
    rows, hard, soft = [], [], []

    for k in names:
        a, r = col(f_aoj, m_aoj, k), col(f_ref, m_ref, k)
        if k in DISPLACEMENT:                       # compare TRACKS; zeros are a separate row
            a, r = np.abs(a[a != 0]), np.abs(r[r != 0])
        if k in SPECIES or k == "part_charge":
            fa, fr = float(np.mean(a != 0)), float(np.mean(r != 0))
            row = dict(feature=k, stat="fraction_nonzero", aoj=fa, reference=fr,
                       ratio=fa / fr if fr else None)
            if k in SPECIES and (fr == 0 or abs(fa / fr - 1) > SPECIES_TOLERANCE):
                row["flag"] = "species"; soft.append(f"species:{k} {fa:.4f} vs {fr:.4f}")
            rows.append(row)
            continue
        qa, qr = _quantiles(a), _quantiles(r)
        row = dict(feature=k, stat="median|iqr", aoj=qa["median"], reference=qr["median"],
                   ratio=qa["median"] / qr["median"] if abs(qr["median"]) > 1e-6 else None,
                   shift_in_ref_iqr=(qa["median"] - qr["median"]) / qr["iqr"] if qr["iqr"] else None,
                   iqr_aoj=qa["iqr"], iqr_reference=qr["iqr"],
                   iqr_ratio=qa["iqr"] / qr["iqr"] if qr["iqr"] else None,
                   p01_aoj=qa["p01"], p99_aoj=qa["p99"], p01_reference=qr["p01"], p99_reference=qr["p99"],
                   n_aoj=int(a.size), n_reference=int(r.size), quantiles_aoj=_qfunc(a), quantiles_reference=_qfunc(r))
        if qa["p01"] > qr["p99"] or qr["p01"] > qa["p99"]:
            row["flag"] = "support"; hard.append(f"support:{k}")
        if k in DISPLACEMENT and row["ratio"] and any(lo < row["ratio"] < hi for lo, hi in UNIT_BANDS):
            row["flag"] = "unit"; hard.append(f"unit:{k} median ratio {row['ratio']:.2f}")
        rows.append(row)

    for name, a_k, b_k in (("d0_dphi_sign", "part_d0", "part_dphi"), ("dz_deta_sign", "part_dz", "part_deta")):
        sa, sr = (_asymmetry(col(f, m, a_k), col(f, m, b_k)) for f, m in ((f_aoj, m_aoj), (f_ref, m_ref)))
        row = dict(feature=name, stat="sign_asymmetry", aoj=sa["asym"], reference=sr["asym"],
                   n_aoj=sa["n"], n_reference=sr["n"])
        if sa["asym"] is not None and sr["asym"] is not None:
            za, zr = abs(sa["asym"]) / sa["err"], abs(sr["asym"]) / sr["err"]
            if sa["asym"] * sr["asym"] < 0 and min(za, zr) > ASYM_SIGMA:
                row["flag"] = name; hard.append(f"{name}: {sa['asym']:+.3f} vs {sr['asym']:+.3f}")
            elif max(za, zr) > ASYM_SIGMA and min(za, zr) < 2.0 and abs(sa["asym"] - sr["asym"]) > 0.05:
                # present in one sample, absent in the other. For dz x deta that is what
                # JetClass-II mirroring dz with the jet-eta sign (as it mirrors deta)
                # would look like: the correlation survives there and averages out here.
                row["flag"] = f"{name}_one_sided"
                soft.append(f"{name}_one_sided: {sa['asym']:+.3f} vs {sr['asym']:+.3f}")
        rows.append(row)

    def no_track(f, m):
        q = col(f, m, "part_charge") != 0
        return float(np.mean(col(f, m, "part_d0err")[q] == 0)) if q.any() else None
    na, nr = no_track(f_aoj, m_aoj), no_track(f_ref, m_ref)
    row = dict(feature="charged_no_track", stat="fraction", aoj=na, reference=nr)
    if na is not None and na > NO_TRACK_MAX:
        row["flag"] = "charged_no_track"; soft.append(f"charged_no_track {na:.3f} vs {nr}")
    rows.append(row)

    # stage_aoj.py zeroes displacement on every neutral because JetClass-II is
    # documented to; this row is where that premise is checked on the reference.
    def neutral_displaced(f, m):
        q0 = col(f, m, "part_charge") == 0
        return float(np.mean((col(f, m, "part_d0")[q0] != 0) | (col(f, m, "part_d0err")[q0] != 0)))
    da, dr = neutral_displaced(f_aoj, m_aoj), neutral_displaced(f_ref, m_ref)
    row = dict(feature="neutral_with_displacement", stat="fraction", aoj=da, reference=dr)
    if abs(da - dr) > 0.05:
        row["flag"] = "neutral_with_displacement"; soft.append(f"neutral_with_displacement {da:.3f} vs {dr:.3f}")
    rows.append(row)

    na_, nr_ = m_aoj.sum(axis=1), m_ref.sum(axis=1)
    ma, mr = _quantiles(na_), _quantiles(nr_)
    rows.append(dict(feature="n_particles", stat="median|iqr", aoj=ma["median"], reference=mr["median"],
                     ratio=ma["median"] / mr["median"], iqr_aoj=ma["iqr"], iqr_reference=mr["iqr"],
                     iqr_ratio=ma["iqr"] / mr["iqr"] if mr["iqr"] else None,
                     shift_in_ref_iqr=(ma["median"] - mr["median"]) / mr["iqr"] if mr["iqr"] else None,
                     n_aoj=int(na_.size), n_reference=int(nr_.size),
                     quantiles_aoj=_qfunc(na_), quantiles_reference=_qfunc(nr_)))
    return rows, hard, soft


def _cdf(q, x):
    """CDF at x of a sample given by its quantile function q on QUANTILE_LEVELS.
    Flat stretches (ties) take the upper level, as a CDF does."""
    q = np.asarray(q, float)
    return np.interp(x, q, QUANTILE_LEVELS, left=0.0, right=1.0)


def _pooled_quantile(qs, ns, levels):
    """Quantiles at `levels` of the count-weighted mixture of samples given by their
    quantile functions: the mixture CDF on the union of their knots, inverted."""
    grid = np.unique(np.concatenate([np.asarray(q, float) for q in qs]))
    cdf = sum(n * _cdf(q, grid) for q, n in zip(qs, ns)) / sum(ns)
    cdf = np.maximum.accumulate(cdf)
    return np.interp(levels, cdf, grid)


def ks_distance(q_a, q_b) -> float:
    """max |F_a - F_b| of two samples given by their quantile functions."""
    grid = np.unique(np.concatenate([np.asarray(q_a, float), np.asarray(q_b, float)]))
    return float(np.max(np.abs(_cdf(q_a, grid) - _cdf(q_b, grid))))


SUMMARY_FIELDS = ("aoj", "reference", "ratio", "iqr_ratio", "shift_in_ref_iqr")


def pool_shards(closures: list[dict]) -> dict:
    """One closure table from the per-shard closure.json of a run.

    Per feature: every field of SUMMARY_FIELDS shard by shard, with its mean, SD
    (n - 1) and range over the shards; and, where every shard carries quantile
    functions, the POOLED AOJ median, IQR, 1st/99th percentiles, the ratios to the
    reference and the KS distance. Flags are the union, tagged by shard index."""
    if not closures:
        raise SystemExit("FATAL: no closure to pool")
    by_feat = {}
    for i, c in enumerate(closures):
        for r in c["rows"]:
            by_feat.setdefault(r["feature"], []).append((i, r))
    out = {}
    for feat, rs in by_feat.items():
        if len(rs) != len(closures):
            raise SystemExit(f"FATAL: {feat} is in {len(rs)} of {len(closures)} shard closures")
        row = dict(feature=feat, stat=rs[0][1]["stat"], n_shards=len(rs))
        for k in SUMMARY_FIELDS:
            v = [r.get(k) for _, r in rs]
            if all(x is not None for x in v):
                v = np.array(v, float)
                row[k] = dict(per_shard=v.tolist(), mean=float(v.mean()),
                              sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                              min=float(v.min()), max=float(v.max()))
        if all("quantiles_aoj" in r for _, r in rs):
            refs = {tuple(r["quantiles_reference"]) for _, r in rs}
            if len(refs) != 1:
                raise SystemExit(f"FATAL: {feat}: the shards' reference samples differ; pooling "
                                 "assumes one reference")
            q_ref = rs[0][1]["quantiles_reference"]
            pooled = _pooled_quantile([r["quantiles_aoj"] for _, r in rs],
                                      [r["n_aoj"] for _, r in rs], np.array([0.01, 0.25, 0.5, 0.75, 0.99]))
            p01, p25, p50, p75, p99 = (float(v) for v in pooled)
            ref = _quantiles_from(q_ref)
            grid_q = _pooled_quantile([r["quantiles_aoj"] for _, r in rs],
                                      [r["n_aoj"] for _, r in rs], QUANTILE_LEVELS)
            row["pooled"] = dict(
                n_aoj=int(sum(r["n_aoj"] for _, r in rs)), n_reference=int(rs[0][1]["n_reference"]),
                median_aoj=p50, iqr_aoj=p75 - p25, p01_aoj=p01, p99_aoj=p99,
                median_reference=ref["median"], iqr_reference=ref["iqr"],
                p01_reference=ref["p01"], p99_reference=ref["p99"],
                median_ratio=p50 / ref["median"] if abs(ref["median"]) > 1e-6 else None,
                iqr_ratio=(p75 - p25) / ref["iqr"] if ref["iqr"] else None,
                shift_in_ref_iqr=(p50 - ref["median"]) / ref["iqr"] if ref["iqr"] else None,
                ks=ks_distance(grid_q, q_ref))
        out[feat] = row
    hard = [f"shard{i}: {h}" for i, c in enumerate(closures) for h in c["hard_flags"]]
    soft = [f"shard{i}: {s}" for i, c in enumerate(closures) for s in c["soft_flags"]]
    return dict(n_shards=len(closures), n_jets_aoj=[c["n_jets_aoj"] for c in closures],
                n_jets_reference=[c["n_jets_reference"] for c in closures],
                pt_window=closures[0]["pt_window"], hard_flags=hard, soft_flags=soft, features=out)


def _quantiles_from(q):
    """_quantiles() of a sample given by its quantile function."""
    p1, p25, p50, p75, p99 = np.interp([0.01, 0.25, 0.5, 0.75, 0.99], QUANTILE_LEVELS, np.asarray(q, float))
    return dict(median=float(p50), iqr=float(p75 - p25), p01=float(p1), p99=float(p99))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoj", nargs="+", required=True, help="staged AspenOpenJets parquet")
    ap.add_argument("--reference", nargs="+", required=True, help="JetClass-II QCD parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pt-window", nargs=2, type=float, default=[500.0, 700.0],
                    help="model-facing jet_pt slice applied to BOTH samples")
    ap.add_argument("--max-jets", type=int, default=100_000)
    a = ap.parse_args()

    f_aoj, m_aoj, names = load_inputs(AOJ_CONFIG, a.aoj, a.pt_window, a.max_jets)
    f_ref, m_ref, names_ref = load_inputs(REF_CONFIG, a.reference, a.pt_window, a.max_jets)
    if names != names_ref:
        raise SystemExit("FATAL: the two configs disagree on the pf_features slots")
    rows, hard, soft = compare(f_aoj, m_aoj, f_ref, m_ref, names)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r if not k.startswith("quantiles_")},
                    key=lambda k: (k not in ("feature", "stat", "flag"), k))
    with (out / "closure_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out / "closure.json").write_text(json.dumps(dict(
        n_jets_aoj=int(len(f_aoj)), n_jets_reference=int(len(f_ref)), pt_window=a.pt_window,
        hard_flags=hard, soft_flags=soft, rows=rows), indent=2))

    for r in rows:
        fmt = lambda v: "      n/a" if v is None else f"{v:9.4f}"
        print(f"  {r['feature']:24s} {r['stat']:17s} aoj {fmt(r['aoj'])}  ref {fmt(r['reference'])}  "
              f"ratio {fmt(r.get('ratio'))}  iqr ratio {fmt(r.get('iqr_ratio'))}  {r.get('flag', '')}")
    print(f"\nclosure: {len(f_aoj):,} AOJ vs {len(f_ref):,} JetClass-II QCD jets, "
          f"{len(hard)} hard / {len(soft)} soft flags")
    for f in hard + soft:
        print(f"  FLAG {f}")
    return 0        # flags are data for the verdict, not a crash: see peak_fit.py


if __name__ == "__main__":
    sys.exit(main())
