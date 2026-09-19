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
                   p01_aoj=qa["p01"], p99_aoj=qa["p99"], p01_reference=qr["p01"], p99_reference=qr["p99"])
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

    ma, mr = _quantiles(m_aoj.sum(axis=1)), _quantiles(m_ref.sum(axis=1))
    rows.append(dict(feature="n_particles", stat="median|iqr", aoj=ma["median"], reference=mr["median"],
                     ratio=ma["median"] / mr["median"], iqr_aoj=ma["iqr"], iqr_reference=mr["iqr"]))
    return rows, hard, soft


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
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("feature", "stat", "flag"), k))
    with (out / "closure_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
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
