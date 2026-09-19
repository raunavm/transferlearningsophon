#!/usr/bin/env python3
"""Convert one AspenOpenJets HDF5 file into the JetClass-II ragged parquet schema.

AspenOpenJets (arXiv:2412.10504, DOI 10.25592/uhhfdm.16505, CC-BY 4.0) is CMS
2016 JetHT Open Data: real collisions, no truth labels. The trunks under test
saw only JetClass-II fast simulation, so this file is the ONLY place where a
real-detector convention is translated into the simulated one, and every
translation below is either verified against the producer code or is a loud,
named, UNVERIFIED constant that experiments/AOJ/closure.py has to confirm on
distributions before any score is believed.

WHAT WAS VERIFIED, AND WHERE  (producer = github.com/OzAmram/AOJProcessing at
commit d8b51c0, H5_maker.py sha256 449a4e98...; PFNano = cms-opendata-analyses/
PFNanoProducerTool at 67e7fe9; NanoAOD = cms-sw/cmssw CMSSW_10_6_30)

  PFCands columns      [px, py, pz, E, d0, d0Err, dz, dzErr, charge, pdgId,
                       puppiWeight]                         H5_maker.py:94-96
  4-momenta are NOT    H5_maker.py:94 builds the vector from PFCands.pt, which
  PUPPI-weighted       is NanoAOD CandVars `pt = Var("pt")` (common_cff.py:43,
                       53) -- the packed candidate's own pT -- and stores
                       puppiWeight() as a separate column (addPFCands_cff.py:
                       42). The weight is applied HERE, and zero-weight
                       candidates are dropped.
  no-track sentinel    d0, d0Err, dz, dzErr are ALL `-1` when the candidate has
                       no track details (addPFCands_cff.py:46-49:
                       "?hasTrackDetails()?dxy():-1"). That covers every neutral
                       AND soft charged candidates. In JetClass-II d0/dz are
                       exactly 0 for every neutral (docs/DOWNSTREAM_SUITE.md:80),
                       and its config clips part_d0err to [0, 1], so the trunk
                       saw 0 for anything non-positive: the sentinel becomes 0
                       in all four. A raw -1 would NOT be harmless: the value
                       slots are tanh(d0 / mm), so -1 cm would read as -1.0, a
                       track displaced by more than any real one.
  d0 sign              CMS: dxy = -dx*sin(phi) + dy*cos(phi) (CMSSW_10_6_30
                       PackedCandidate.cc:38). Upstream Delphes: d0 = (xd*py -
                       yd*px)/pT (modules/ParticlePropagator.cc:298), stored in
                       mm (:335). These are OPPOSITE, so d0 is negated here.
                       dz has the same sense in both (PackedCandidate.cc:43,
                       ParticlePropagator.cc:299).
  charge               stored directly (H5_maker.py:96), NOT re-derived from the
                       PDG sign; the two are cross-checked below.
  pT ordering          sort_pfcands() orders by UNWEIGHTED pT (H5_maker.py:66-
                       73), and truncation to 150 happens BEFORE that sort
                       (:103 then :104), so for a jet with more than 150
                       candidates the lost ones are not guaranteed to be the
                       softest. Re-sorted here by PUPPI-WEIGHTED pT, then cut to
                       the model's 128.
  jet_kinematics       [pt, eta, phi, msoftdrop]              H5_maker.py:85
                       pt is the JEC-CORRECTED FatJet_pt; the constituents sum
                       to the uncorrected jet (paper footnote 1). The model
                       normalises constituents by the jet they sum to, so the
                       model-facing jet_pt/eta/phi/energy are REBUILT from the
                       weighted constituents; the corrected values are kept as
                       aoj_jet_* and are what the selection and rho use.
  soft-drop mass       FatJet_msoftdrop = groomedMass('SoftDropPuppi'), NanoAOD
                       doc "Corrected soft drop mass with PUPPI" (jets_cff.py:
                       443): subjet JECs applied, no further W-mass correction.
  jet_tagging          [nConstituents, tau1..4, PN H4qvsQCD, HbbvsQCD, HccvsQCD,
                       QCD, TvsQCD, WvsQCD, ZvsQCD, PN mass]   H5_maker.py:86-88
                       -> top = col 9, W = col 10, bb = col 6. These are the
                       NON-mass-decorrelated ParticleNet scores (jets_cff.py:
                       466-469); the MD ones are not shipped.
  event_info           [run, lumi, event] int64                H5_maker.py:82

WHAT COULD NOT BE VERIFIED -- see the UNVERIFIED_* constants below. Each one is
checked by a named row of the closure table, not assumed.

THE deta/dphi CONVENTION is scripts/stage_downstream.py's MEASURED one and its
code is imported, not copied: part_deta = (eta_p - eta_jet) * sign(eta_jet),
part_dphi unflipped.

Run:  python3 scripts/stage_aoj.py --in /scratch/raw/RunG_batch0.h5 --out /scratch/staged
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys

import awkward as ak
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stage_downstream():
    """The measured deta/dphi convention lives in ONE place."""
    spec = importlib.util.spec_from_file_location(
        "stage_downstream", ROOT / "scripts" / "stage_downstream.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_sd = _stage_downstream()
kinematics, wrap = _sd.kinematics, _sd.wrap
check_not_precentred, summarise = _sd.check_not_precentred, _sd.summarise

# ---- VERIFIED layout (line references in the module docstring) ---------------
PF = dict(px=0, py=1, pz=2, e=3, d0=4, d0err=5, dz=6, dzerr=7, charge=8,
          pdgid=9, puppi=10)
KIN = dict(pt=0, eta=1, phi=2, sdmass=3)
TAG = dict(nconst=0, pn_hbb=6, pn_top=9, pn_w=10)
NO_TRACK_SENTINEL = -1.0
N_STORED, N_MODEL = 150, 128

# ---- UNVERIFIED conventions: LOUD, NAMED, and each checked by closure.py -----
# UNVERIFIED. pat::PackedCandidate::dxy()/dz() carry no unit in the producer
# code; centimetres is CMSSW's native length unit (training knowledge, not
# re-checked). JetClass-II is in millimetres (docs/GROUND_TRUTH.md section 4).
# A wrong factor shows up as a median |d0| ratio near 10 or 0.1 -- closure flag
# `d0_unit`.
UNVERIFIED_CM_TO_MM = 10.0
# HALF VERIFIED. CMS and UPSTREAM Delphes sign d0 oppositely (module docstring),
# hence -1. What is NOT verified is that the JetClass-II production -- its Delphes
# version, card, and any track-smearing module that rewrites D0 -- kept upstream's
# convention, nor whether it mirrors dz with the jet-eta sign the way it mirrors
# deta. The sign is invisible in a 1-d histogram (both are symmetric) but it
# fixes the sign of the d0 x dphi correlation of displaced tracks, which the trunk
# can see -- closure flags `d0_dphi_sign` and `dz_deta_sign`. If one fires, flip
# the constant; do not reason about it.
UNVERIFIED_D0_SIGN = -1.0
UNVERIFIED_DZ_SIGN = +1.0
# A CHOICE, not a fact. CMS forward-calorimeter candidates carry pdgId 1
# (hadronic) and 2 (electromagnetic); JetClass-II has no such species. They only
# occur at |eta| > 3, i.e. at the edge of jets near the |eta| < 2.4 cut. Their
# rate is reported, and the particle-type fractions are a closure row.
HF_HADRON_PDGID, HF_HADRON_AS = 1, "NeutralHadron"
HF_EM_PDGID, HF_EM_AS = 2, "Photon"

SPECIES = ["ChargedHadron", "NeutralHadron", "Photon", "Electron", "Muon"]
_PDG = {211: 0, 130: 1, 22: 2, 11: 3, 13: 4,
        HF_HADRON_PDGID: SPECIES.index(HF_HADRON_AS),
        HF_EM_PDGID: SPECIES.index(HF_EM_AS)}
# PDG: the positive id is the NEGATIVE lepton (11 = e-, 13 = mu-); 211 = pi+.
_PDG_CHARGE_SIGN = {211: +1.0, 11: -1.0, 13: -1.0}

SELECTION = dict(pt_min=500.0, pt_max=2500.0, abs_eta_max=2.4,
                 sdmass_min=20.0, sdmass_max=500.0)


def select(jet_kin: np.ndarray) -> np.ndarray:
    """The staging selection, on the STORED (JEC-corrected) jet kinematics."""
    pt, eta, msd = (jet_kin[:, KIN[k]] for k in ("pt", "eta", "sdmass"))
    s = SELECTION
    return ((pt > s["pt_min"]) & (pt < s["pt_max"]) & (np.abs(eta) < s["abs_eta_max"])
            & (msd > s["sdmass_min"]) & (msd < s["sdmass_max"]))


def convert(event_info, jet_kin, pfcands, jet_tag):
    """Selected AOJ arrays -> (record of awkward arrays, counters).

    Inputs are the four HDF5 datasets for the SAME jets, already selected.
    """
    c = {k: pfcands[:, :, i].astype(np.float64) for k, i in PF.items()}
    w = c["puppi"]
    real = c["e"] > 0                                   # the table is zero-padded
    keep = real & (w > 0)

    # PUPPI-weighted four-momenta, ordered by WEIGHTED pT, real candidates first.
    pxw, pyw, pzw, ew = (c[k] * w for k in ("px", "py", "pz", "e"))
    order = np.argsort(-np.where(keep, np.hypot(pxw, pyw), -1.0), axis=1, kind="stable")
    counts = keep.sum(axis=1)
    live = np.arange(pfcands.shape[1])[None, :] < counts[:, None]

    def ragged(a):
        return ak.unflatten(np.take_along_axis(a, order, axis=1)[live], counts)

    px, py, pz, e = ragged(pxw), ragged(pyw), ragged(pzw), ragged(ew)
    kin = kinematics(px, py, pz, e)                     # axis from ALL kept candidates

    # species and charge
    apid = np.abs(c["pdgid"]).astype(np.int64)
    species = np.full(apid.shape, -1, dtype=np.int64)
    for pid, idx in _PDG.items():
        species[apid == pid] = idx
    unknown = keep & (species < 0)
    # an id outside the PF vocabulary: charged -> hadron, neutral -> neutral hadron
    species[unknown] = np.where(c["charge"][unknown] != 0, 0, 1)
    expect = np.zeros_like(c["charge"])
    for pid, sgn in _PDG_CHARGE_SIGN.items():
        expect[apid == pid] = sgn * np.sign(c["pdgid"][apid == pid])
    charge_mismatch = keep & (expect != c["charge"]) & ~unknown

    # displacement: sentinel (and anything non-finite) -> JetClass-II's "none" = 0
    ip = np.stack([c[k] for k in ("d0", "d0err", "dz", "dzerr")])
    # ... and only CHARGED candidates carry any: JetClass-II has none on a neutral
    has_track = ((c["d0err"] != NO_TRACK_SENTINEL) & (c["dzerr"] != NO_TRACK_SENTINEL)
                 & (c["d0err"] > 0) & (c["dzerr"] > 0) & np.isfinite(ip).all(axis=0)
                 & (c["charge"] != 0))
    mm = lambda a, sign=1.0: np.where(has_track, a * UNVERIFIED_CM_TO_MM * sign, 0.0)

    part = dict(
        part_px=px, part_py=py, part_pz=pz, part_energy=e,
        part_deta=kin["part_deta"], part_dphi=kin["part_dphi"],
        part_d0val=ragged(mm(c["d0"], UNVERIFIED_D0_SIGN)), part_d0err=ragged(mm(c["d0err"])),
        part_dzval=ragged(mm(c["dz"], UNVERIFIED_DZ_SIGN)), part_dzerr=ragged(mm(c["dzerr"])),
        part_charge=ragged(c["charge"]),
        **{f"part_is{n}": ragged((species == i).astype(np.float64))
           for i, n in enumerate(SPECIES)})
    # pT-ordered, so the model's 128 are the hardest 128
    part = {k: ak.values_astype(v[:, :N_MODEL], np.float32) for k, v in part.items()}

    # ascontiguousarray: a column slice of the (n, 4) table is a strided view and
    # pyarrow refuses to write one ("ndarray is not contiguous")
    f32 = lambda a: ak.Array(np.ascontiguousarray(a, dtype=np.float32))
    i64 = lambda a: ak.Array(np.ascontiguousarray(a, dtype=np.int64))
    rec = dict(
        **part,
        jet_pt=f32(kin["jet_pt"]), jet_eta=f32(kin["jet_eta"]),
        jet_phi=f32(kin["jet_phi"]), jet_energy=f32(kin["jet_energy"]),
        jet_sdmass=f32(jet_kin[:, KIN["sdmass"]]),
        jet_nparticles=ak.Array(counts.astype(np.int32)),
        aoj_jet_pt=f32(jet_kin[:, KIN["pt"]]), aoj_jet_eta=f32(jet_kin[:, KIN["eta"]]),
        aoj_jet_phi=f32(jet_kin[:, KIN["phi"]]),
        aoj_pn_WvsQCD=f32(jet_tag[:, TAG["pn_w"]]), aoj_pn_TvsQCD=f32(jet_tag[:, TAG["pn_top"]]),
        aoj_pn_HbbvsQCD=f32(jet_tag[:, TAG["pn_hbb"]]),
        run=i64(event_info[:, 0]),
        lumi=i64(event_info[:, 1]),
        event=i64(event_info[:, 2]),
        label=ak.Array(np.zeros(len(jet_kin), dtype=np.int64)))   # dummy: real data

    n_keep = max(int(keep.sum()), 1)
    charged = keep & (c["charge"] != 0)
    counters = dict(
        n_jets=int(len(jet_kin)), n_candidates=int(keep.sum()),
        n_dropped_zero_puppi=int((real & ~keep).sum()),
        n_unknown_pdgid=int(unknown.sum()), n_charge_mismatch=int(charge_mismatch.sum()),
        n_hf=int((keep & np.isin(apid, [HF_HADRON_PDGID, HF_EM_PDGID])).sum()),
        n_charged=int(charged.sum()), n_charged_no_track=int((charged & ~has_track).sum()),
        n_neutral_with_track_zeroed=int((keep & (c["charge"] == 0) & (c["d0err"] > 0)).sum()),
        n_jets_over_stored=int((jet_tag[:, TAG["nconst"]] > N_STORED).sum()),
        n_jets_over_model=int((counts > N_MODEL).sum()))
    for k in ("n_unknown_pdgid", "n_charge_mismatch"):
        if counters[k] / n_keep > 1e-3:
            raise SystemExit(
                f"stage_aoj: {k} = {counters[k]:,} of {n_keep:,} candidates. The "
                "pdgId/charge columns are not what H5_maker.py:96 says they are; "
                "refusing to write.")
    return rec, counters


def stage_file(src: pathlib.Path, dest: pathlib.Path, chunk: int = 50_000,
               max_jets: int = 0) -> dict:
    import h5py
    recs, totals, n_read = [], {}, 0
    with h5py.File(src, "r") as f:
        shape = f["PFCands"].shape
        if shape[1:] != (N_STORED, len(PF)) or f["jet_tagging"].shape[1] != 13:
            raise SystemExit(f"stage_aoj: {src.name} has PFCands {shape}, jet_tagging "
                             f"{f['jet_tagging'].shape}; not the layout H5_maker.py writes")
        n = shape[0]
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            kin_ = f["jet_kinematics"][lo:hi]
            sel = select(kin_)
            n_read += hi - lo
            if sel.any():
                rec, cnt = convert(f["event_info"][lo:hi][sel], kin_[sel],
                                   f["PFCands"][lo:hi][sel], f["jet_tagging"][lo:hi][sel])
                recs.append(rec)
                for k, v in cnt.items():
                    totals[k] = totals.get(k, 0) + v
            print(f"  {src.name}: {hi:,}/{n:,} read, {totals.get('n_jets', 0):,} selected",
                  flush=True)
            if max_jets and totals.get("n_jets", 0) >= max_jets:
                break
    if not recs:
        raise SystemExit(f"stage_aoj: no jet in {src.name} passes {SELECTION}")
    rec = {k: ak.concatenate([r[k] for r in recs]) for k in recs[0]}
    med = check_not_precentred(rec["jet_eta"], src.name)
    ak.to_parquet(ak.Array(rec), dest)

    pt = np.hypot(rec["part_px"], rec["part_py"])
    stats = dict(
        source=src.name, n_jets_read=n_read, selection=SELECTION,
        median_abs_jet_eta=med, counters=totals,
        frac_charged_no_track=totals["n_charged_no_track"] / max(totals["n_charged"], 1),
        # constituents sum to the UNCORRECTED jet; this is the JEC, not a bug
        median_rebuilt_over_stored_pt=float(np.median(
            ak.to_numpy(rec["jet_pt"]) / ak.to_numpy(rec["aoj_jet_pt"]))),
        unverified=dict(cm_to_mm=UNVERIFIED_CM_TO_MM, d0_sign=UNVERIFIED_D0_SIGN,
                        dz_sign=UNVERIFIED_DZ_SIGN, hf_hadron_as=HF_HADRON_AS,
                        hf_em_as=HF_EM_AS),
        features=summarise(dict(
            part_deta=rec["part_deta"], part_dphi=rec["part_dphi"],
            part_logptrel=np.log(pt / rec["jet_pt"]),
            part_logerel=np.log(rec["part_energy"] / rec["jet_energy"]),
            part_deltaR=np.hypot(rec["part_deta"], rec["part_dphi"]),
            jet_pt=rec["jet_pt"], jet_sdmass=rec["jet_sdmass"])))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="one AspenOpenJets .h5")
    ap.add_argument("--out", required=True, help="directory for <stem>.parquet")
    ap.add_argument("--chunk", type=int, default=50_000,
                    help="jets per read; 50k x 150 x 11 float32 = 330 MB")
    ap.add_argument("--max-jets", type=int, default=0, help="0 = all selected jets")
    a = ap.parse_args()

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    if free < 2.0:          # one staged file is ~0.3 GB; CLAUDE.md storage rule
        raise SystemExit(f"stage_aoj: {free:.1f} GB free under {out}, need 2 GB")
    dest = out / f"{src.stem}.parquet"
    stats = stage_file(src, dest, a.chunk, a.max_jets)
    (out / f"{src.stem}.stats.json").write_text(json.dumps(stats, indent=2))
    c = stats["counters"]
    print(f"wrote {dest}  {c['n_jets']:,} jets of {stats['n_jets_read']:,} read  "
          f"{dest.stat().st_size/1e9:.2f} GB\n"
          f"  charged without track details {stats['frac_charged_no_track']:.3f}  "
          f"zero-PUPPI dropped {c['n_dropped_zero_puppi']:,}  HF {c['n_hf']:,}  "
          f"jets >150 stored {c['n_jets_over_stored']:,}  >128 kept {c['n_jets_over_model']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
