#!/usr/bin/env python3
"""Convert the top-tagging and q/g benchmark sets into the branch schema the
pretrained trunk's data configs read.

WHY A CONVERTER AND NOT A DATA CONFIG
-------------------------------------
`configs/finetune/TopReference.yaml` and `EnergyFlowQG.yaml` zero-fill the
features those datasets lack (scripts/build_downstream_configs.py). What they
CANNOT do is invent the branches that do exist under different names and
conventions: the top set ships (E, px, py, pz) per constituent in a flat HDF5
table, and EnergyFlow ships (pt, y, phi, pdgid) in an npz. Both must become the
JetClass-II per-jet ragged parquet layout before weaver can read them.

THE deta/dphi CONVENTION IS MEASURED, NOT ASSUMED
-------------------------------------------------
`part_deta` and `part_dphi` are RAW branches in JetClass-I and JetClass-II -- no
config in this repo derives them -- so a converter has to reproduce whatever the
dataset authors did, and getting it wrong is silent: the model trains, the loss
falls, and every downstream number is quietly degraded.

Measured 2026-09-07 on 100,000 jets of /jc2/jet_data/QCD_0000.parquet, comparing
each candidate against the released branch (float32 agreement is ~5e-07):

    part_deta = (part_eta - jet_eta) * sign(jet_eta)     max|diff| 4.8e-07  <-- YES
    part_deta =  part_eta - jet_eta                      max|diff| 6.44     no
    part_dphi = wrap(part_phi - jet_phi)                 max|diff| 4.2e-07  <-- YES
    part_dphi = wrap(part_phi - jet_phi) * sign(jet_eta) max|diff| 2.18     no

The sign flip is applied to eta and NOT to phi. That asymmetry is not guessable
and is the single most dangerous thing in this file.

CONSEQUENCE FOR A PRE-CENTRED DATASET. The flip needs a meaningful jet_eta sign.
If a source ships constituents already centred on the jet axis, jet_eta ~ 0, the
sign is noise, and half the jets would be mirrored at random. This script
therefore MEASURES |jet_eta| after reconstruction and refuses to write a
sign-flipped file when the sample looks pre-centred, rather than producing one
silently.

VALIDATION. After conversion the script recomputes summary statistics for every
feature the dataset does share with JetClass-II and writes them beside the
reference values, so a convention error shows up as a shifted distribution
before any GPU time is spent.

Run:  python3 scripts/stage_downstream.py --dataset top --out /data/finetune/top
      python3 scripts/stage_downstream.py --dataset qg  --out /data/finetune/qg
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

# Zenodo records named in docs/PRD_PLAN.md §2.
SOURCES = {
    "top": dict(
        record="2603256",
        files={"train": "train.h5", "val": "val.h5", "test": "test.h5"},
        n_const=200,
    ),
    "qg": dict(
        record="3164691",
        # Verified against the Zenodo API 2026-09-07: the record holds 40 files
        # in two families of 20 x 100k jets. We take the PLAIN family
        # (QG_jets.npz, then QG_jets_1..19.npz, 2.13 GB) and not
        # QG_jets_withbc_*.npz. The plain sample excludes b- and c-initiated
        # jets, and it is the one the EnergyFlow paper and every transfer result
        # we are compared against (ParT, OmniLearn, 2607.23377) report on, so
        # taking the withbc variant would silently make our row incomparable.
        files={f"chunk{i}": f"QG_jets_{i}.npz" if i else "QG_jets.npz"
               for i in range(20)},
        n_const=None,
    ),
}
ZENODO = "https://zenodo.org/records/{record}/files/{fname}?download=1"

# Reference statistics for the features every dataset shares, measured on
# JetClass-II so a convention error is visible as a shift. Filled by
# --reference-stats; see the job spec.
SHARED = ["part_pt_scale_log", "part_e_scale_log", "part_logptrel",
          "part_logerel", "part_deltaR", "part_deta", "part_dphi"]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def kinematics(px, py, pz, e):
    """Per-jet axis and the two angular branches, in the MEASURED convention."""
    jpx, jpy = ak.sum(px, axis=1), ak.sum(py, axis=1)
    jpz, je = ak.sum(pz, axis=1), ak.sum(e, axis=1)
    jet_pt = np.hypot(jpx, jpy)
    jet_eta = np.arcsinh(jpz / jet_pt)
    jet_phi = np.arctan2(jpy, jpx)

    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    sign = np.where(ak.to_numpy(jet_eta) > 0, 1.0, -1.0)
    deta = (eta - jet_eta) * sign          # sign flip: measured, see docstring
    dphi = wrap(phi - jet_phi)             # NO sign flip: measured
    return dict(jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_energy=je,
                part_deta=deta, part_dphi=dphi)


def check_not_precentred(jet_eta, name):
    """Refuse to sign-flip a sample whose jets sit on eta = 0."""
    med = float(np.median(np.abs(ak.to_numpy(jet_eta))))
    if med < 0.05:
        raise SystemExit(
            f"stage_downstream: {name} has median |jet_eta| = {med:.4f}, i.e. its "
            "constituents look ALREADY CENTRED on the jet axis. part_deta's sign "
            "flip is keyed on sign(jet_eta) (measured convention, see the module "
            "docstring), so on a pre-centred sample it would mirror ~half the "
            "jets at random. Refusing to write. Resolve the convention against "
            "the source before staging.")
    return med


def fetch(record, fname, dest: pathlib.Path):
    if dest.exists():
        print(f"  have {dest.name}")
        return dest
    url = ZENODO.format(record=record, fname=fname)
    print(f"  fetching {fname} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  {dest.name}  {dest.stat().st_size/1e9:.2f} GB")
    return dest


def load_top(path: pathlib.Path, n_const: int):
    """Flat HDF5: E_0..E_199, PX_0.., PY_0.., PZ_0.., is_signal_new."""
    import pandas as pd
    # The reference files are pandas HDFStores keyed "table", but read the key
    # from the file rather than trusting that: a wrong key raises here, at
    # staging, instead of after the download in whatever runs next.
    with pd.HDFStore(path, mode="r") as store:
        keys = [k.lstrip("/") for k in store.keys()]
    key = "table" if "table" in keys else keys[0]
    if key != "table":
        print(f"  note: HDF5 key is {key!r}, not 'table' (keys: {keys})")
    df = pd.read_hdf(path, key=key)
    e = df[[f"E_{i}" for i in range(n_const)]].to_numpy(np.float32)
    px = df[[f"PX_{i}" for i in range(n_const)]].to_numpy(np.float32)
    py = df[[f"PY_{i}" for i in range(n_const)]].to_numpy(np.float32)
    pz = df[[f"PZ_{i}" for i in range(n_const)]].to_numpy(np.float32)
    label = df["is_signal_new"].to_numpy(np.int64)
    keep = e > 0                                   # the tables are zero-padded
    to = lambda a: ak.drop_none(ak.mask(ak.Array(a), keep))
    return to(px), to(py), to(pz), to(e), label


# EnergyFlow's "exp" PID scenario: the five species the detector distinguishes,
# mapped onto Sophon's flags. |pdgid| -> (isCH, isNH, isPhoton, isEle, isMu)
_PID = {211: 0, 321: 0, 2212: 0,          # charged hadrons
        130: 1, 2112: 1, 310: 1,          # neutral hadrons
        22: 2,                            # photons
        11: 3, 13: 4}                     # electrons, muons


def load_qg(path: pathlib.Path):
    """npz: X (n_jets, max_const, 4) = (pt, y, phi, pdgid); y = labels."""
    z = np.load(path)
    X, label = z["X"], z["y"].astype(np.int64)
    pt, rap, phi, pid = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    keep = pt > 0
    to = lambda a: ak.drop_none(ak.mask(ak.Array(a), keep))
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz, e = pt * np.sinh(rap), pt * np.cosh(rap)   # massless constituents
    apid = np.abs(pid).astype(np.int64)
    species = np.vectorize(lambda p: _PID.get(int(p), 1))(apid)   # unknown -> NH
    # PDG sign convention is NOT uniform across these five species. For 211
    # (pi+), 321 (K+) and 2212 (p) the positive id carries positive charge, so
    # sign(pdgid) is right. For the leptons it is inverted: PDG 11 is the
    # ELECTRON (charge -1) and PDG 13 the mu- (charge -1). Plain sign(pdgid)
    # therefore gave every electron and muon constituent the opposite charge.
    lepton = np.isin(apid, [11, 13])
    charged = np.isin(apid, [211, 321, 2212, 11, 13])
    charge = np.sign(pid) * np.where(lepton, -1.0, 1.0) * charged
    flags = {f"part_is{n}": to((species == i).astype(np.float32))
             for i, n in enumerate(["ChargedHadron", "NeutralHadron", "Photon",
                                    "Electron", "Muon"])}
    return (to(px), to(py), to(pz), to(e), label,
            dict(part_charge=to(charge.astype(np.float32)), **flags))


def summarise(arrs: dict) -> dict:
    out = {}
    for k, v in arrs.items():
        x = ak.to_numpy(ak.flatten(v)) if v.ndim > 1 else ak.to_numpy(v)
        x = x[np.isfinite(x)]
        out[k] = dict(mean=float(x.mean()), std=float(x.std()),
                      p01=float(np.percentile(x, 1)), p99=float(np.percentile(x, 99)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="where downloads land (default <out>/raw)")
    ap.add_argument("--limit-files", type=int, default=None)
    a = ap.parse_args()

    spec = SOURCES[a.dataset]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = pathlib.Path(a.raw or out / "raw")

    names = list(spec["files"])[:a.limit_files] if a.limit_files else list(spec["files"])
    stats, n_total = {}, 0
    for split in names:
        fname = spec["files"][split]
        src = fetch(spec["record"], fname, raw / fname)
        print(f"  converting {fname}", flush=True)
        if a.dataset == "top":
            px, py, pz, e, label = load_top(src, spec["n_const"])
            extra = {}
        else:
            px, py, pz, e, label, extra = load_qg(src)

        kin = kinematics(px, py, pz, e)
        med = check_not_precentred(kin["jet_eta"], f"{a.dataset}/{split}")
        rec = dict(part_px=px, part_py=py, part_pz=pz, part_energy=e,
                   part_deta=kin["part_deta"], part_dphi=kin["part_dphi"],
                   jet_pt=kin["jet_pt"], jet_eta=kin["jet_eta"],
                   jet_phi=kin["jet_phi"], jet_energy=kin["jet_energy"],
                   label=ak.Array(label), **extra)
        dest = out / f"{a.dataset}_{split}.parquet"
        ak.to_parquet(ak.Array({k: v for k, v in rec.items()}), dest)
        n = len(label); n_total += n
        print(f"  wrote {dest.name}  {n:,} jets  median|jet_eta| {med:.3f}  "
              f"{dest.stat().st_size/1e9:.2f} GB", flush=True)

        pt = np.hypot(px, py)
        stats[split] = dict(n_jets=n, median_abs_jet_eta=med, **summarise(dict(
            part_deta=kin["part_deta"], part_dphi=kin["part_dphi"],
            part_logptrel=np.log(pt / kin["jet_pt"]),
            part_logerel=np.log(e / kin["jet_energy"]),
            part_deltaR=np.hypot(kin["part_deta"], kin["part_dphi"]),
            jet_pt=kin["jet_pt"])))

    (out / "staging_stats.json").write_text(json.dumps(
        dict(dataset=a.dataset, zenodo_record=spec["record"], n_jets=n_total,
             convention=("part_deta = (part_eta - jet_eta) * sign(jet_eta); "
                         "part_dphi = wrap(part_phi - jet_phi). MEASURED on "
                         "JetClass-II QCD_0000.parquet, 100k jets, agreement 5e-07."),
             splits=stats), indent=2))
    print(f"\n{a.dataset}: {n_total:,} jets -> {out}")
    print(f"stats -> {out/'staging_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
