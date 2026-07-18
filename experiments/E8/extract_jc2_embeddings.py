#!/usr/bin/env python3
"""Extract 128-d embeddings on JetClass-II for the JC1->JC2 transfer panel (E8).

Three JC1-trained backbones, each fed EXACTLY its own training preprocessing,
over the SAME JetClass-II jets in the SAME order (so the downstream probe
comparison is paired jet-for-jet):

  backbone   wrapper                preprocessing (data config it was trained on)
  --------   -------                ---------------------------------------------
  part       ParTModel              ParT      (raw lorentz; data/JetClass ParT yaml)
  armp       SophonTransferModel    ParT      (raw lorentz; experiments/E1/data_arm_p.yaml)
  arms       SophonTransferModel    Sophon    (scaled lorentz; experiments/E1/data_arm_s.yaml)

Design: each JC2 file is read ONCE; every jet is forwarded through all backbones
in the same row order. ParT-preproc features feed {part, armp}; Sophon-preproc
features feed {arms}. Row i is the same physical jet across every backbone, so
paired bootstrap on the shared test set is valid.

The 17 pf_features (order fixed, matching data_arm_{p,s}.yaml) differ between the
two preprocessings in ONLY three places (verified against data_arm_s.yaml):
  feat 0 pt_log  : ParT  log(pt)         | Sophon log(pt * 500/jet_pt)      [std 1.7,0.7]
  feat 1 e_log   : ParT  log(E)          | Sophon log(E  * 500/jet_pt)      [std 2.0,0.7]
  pf_vectors     : ParT  [px,py,pz,E]    | Sophon [px,py,pz,E]*500/jet_pt
Everything else (logptrel, logerel are scale-invariant; deltaR, charge, PID,
d0/dz, deta/dphi) is byte-identical between the two.

Outputs, per (backbone, split): <out>/<backbone>_<split>_emb.npy  (N,128) float16
                                <out>/<split>_label188.npy         (N,)   int16   (shared)
                                <out>/<split>_jetpt.npy            (N,)   float32 (shared, for Kish/WP)
                                <out>/metadata.json
The label / jetpt arrays are written once per split (identical across backbones).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.part_wrapper import ParTModel            # noqa: E402
from src.models.sophon_wrapper import SophonTransferModel  # noqa: E402

MAX_PART = 128
EPS = 1e-20
# JC2 raw ntuple branches (names confirmed from the Phase-A parquet schema).
PART_KEYS = ["part_px", "part_py", "part_pz", "part_energy",
             "part_deta", "part_dphi", "part_d0val", "part_d0err",
             "part_dzval", "part_dzerr", "part_charge",
             "part_isChargedHadron", "part_isNeutralHadron",
             "part_isPhoton", "part_isElectron", "part_isMuon"]
SCALAR_KEYS = ["jet_pt", "jet_energy", "jet_label"]


def _norm(x, sub, mul, lo=-5.0, hi=5.0):
    """weaver standardization: (x - sub) * mul, clipped to [lo, hi] (default ±5)."""
    return np.clip((x - sub) * mul, lo, hi)


def compute_features(px, py, pz, energy, deta, dphi, d0val, d0err, dzval, dzerr,
                     charge, isCH, isNH, isPh, isE, isMu, jet_pt, jet_energy, *,
                     sophon):
    """The 17 pf_features in data_arm_{p,s}.yaml order + the pf_vectors.

    sophon=False -> ParT preprocessing (data_arm_p.yaml, official ParT).
    sophon=True  -> Sophon preprocessing (data_arm_s.yaml): pt/E logs and the
                    four-vectors are scaled by 500/jet_pt; logptrel/logerel stay
                    scale-invariant; all other features unchanged.

    Broadcasting: particle arrays are (..., P); jet_pt / jet_energy must
    broadcast against them ((N, 1) for batched (N, P) inputs, scalar for (P,)).
    Returns (feats (..., P, 17) float32, lorentz (..., P, 4) float32).
    """
    jet_pt_safe = np.maximum(np.asarray(jet_pt, np.float64), EPS)
    jet_e_safe = np.maximum(np.asarray(jet_energy, np.float64), EPS)
    scale = (500.0 / jet_pt_safe) if sophon else 1.0

    pt = np.sqrt(px ** 2 + py ** 2)                       # unscaled constituent pT
    pt_log = _norm(np.log(np.clip(pt * scale, EPS, None)), 1.7, 0.7)
    e_log = _norm(np.log(np.clip(energy * scale, EPS, None)), 2.0, 0.7)
    # scale-invariant (pt/jet_pt unchanged by the 500/jet_pt rescale) -> identical to ParT
    logptrel = _norm(np.log(np.clip(pt / jet_pt_safe, EPS, None)), -4.7, 0.7)
    logerel = _norm(np.log(np.clip(energy / jet_e_safe, EPS, None)), -4.7, 0.7)
    deltaR = _norm(np.sqrt(deta ** 2 + dphi ** 2), 0.2, 4.0)
    d0 = np.tanh(d0val)
    d0err_c = np.clip(d0err, 0.0, 1.0)
    dz = np.tanh(dzval)
    dzerr_c = np.clip(dzerr, 0.0, 1.0)

    feats = np.stack([pt_log, e_log, logptrel, logerel, deltaR,
                      charge, isCH, isNH, isPh, isE, isMu,
                      d0, d0err_c, dz, dzerr_c, deta, dphi], axis=-1).astype(np.float32)
    lorentz = np.stack([px, py, pz, energy], axis=-1)
    if sophon:
        lorentz = lorentz * np.asarray(scale)[..., None]
    lorentz = lorentz.astype(np.float32)
    return feats, lorentz


def _process_arrays(arr, want_sophon):
    """Vectorized core: pT-sort desc, truncate/pad to MAX_PART, both preprocessings.

    Padded slots are zeroed (identical to the mask-authoritative weaver layout;
    the model never reads them). Row order = input order (NOT shuffled), so it
    is identical for every backbone.
    """
    jet_pt = np.asarray(arr["jet_pt"], np.float32)
    jet_e = np.asarray(arr["jet_energy"], np.float32)
    label = np.asarray(arr["jet_label"], np.int16)

    pt_jag = np.sqrt(arr["part_px"] ** 2 + arr["part_py"] ** 2)
    order = ak.argsort(pt_jag, axis=1, ascending=False)   # leading-pT first

    def grid(name):
        x = ak.pad_none(arr[name][order], MAX_PART, axis=1, clip=True)
        return np.asarray(ak.fill_none(x, 0.0), np.float64)   # (n, MAX_PART)

    cols = {k: grid(k) for k in PART_KEYS}
    counts = np.minimum(np.asarray(ak.num(arr["part_px"], axis=1)), MAX_PART)
    masks = np.arange(MAX_PART)[None, :] < counts[:, None]   # (n, MAX_PART) bool
    mk = masks[:, :, None]

    kw = dict(px=cols["part_px"], py=cols["part_py"], pz=cols["part_pz"],
              energy=cols["part_energy"], deta=cols["part_deta"],
              dphi=cols["part_dphi"], d0val=cols["part_d0val"],
              d0err=cols["part_d0err"], dzval=cols["part_dzval"],
              dzerr=cols["part_dzerr"], charge=cols["part_charge"],
              isCH=cols["part_isChargedHadron"], isNH=cols["part_isNeutralHadron"],
              isPh=cols["part_isPhoton"], isE=cols["part_isElectron"],
              isMu=cols["part_isMuon"],
              jet_pt=jet_pt[:, None].astype(np.float64),
              jet_energy=jet_e[:, None].astype(np.float64))
    pf, plv = compute_features(**kw, sophon=False)
    pf *= mk
    plv *= mk
    sf = slv = None
    if want_sophon:
        sf, slv = compute_features(**kw, sophon=True)
        sf *= mk
        slv *= mk
    return pf, plv, sf, slv, masks, label, jet_pt


def process_file(fpath, want_sophon):
    """Read one JC2 parquet; return (part_feats, part_lv, sophon_feats, sophon_lv,
    masks, label188, jet_pt) with jets padded/truncated to MAX_PART, pT-sorted.

    The Sophon arrays are only built if want_sophon (skips work when no arm-S
    backbone is requested).
    """
    arr = ak.from_parquet(fpath, columns=PART_KEYS + SCALAR_KEYS)
    return _process_arrays(arr, want_sophon)


@torch.no_grad()
def embed(model, feats, lv, masks, device, bs=512):
    """Forward (permuted to B,C,N) in batches; return (N,128) float16 embeddings."""
    n = len(feats)
    out = np.zeros((n, 128), np.float16)
    for i in range(0, n, bs):
        s = slice(i, i + bs)
        f = torch.from_numpy(feats[s]).permute(0, 2, 1).to(device).float()
        v = torch.from_numpy(lv[s]).permute(0, 2, 1).to(device).float()
        mk = torch.from_numpy(masks[s]).unsqueeze(1).to(device)
        _, x_cls = model(f, v, mk)
        out[s] = x_cls.float().cpu().numpy().astype(np.float16)
    return out


def load_backbones(spec, device):
    """spec: list of (name, kind, checkpoint). kind in {part, sophon}."""
    models = {}
    for name, kind, ckpt in spec:
        if kind == "part":
            m = ParTModel(checkpoint_path=ckpt)
        else:
            m = SophonTransferModel(checkpoint_path=ckpt, num_classes=10,
                                    head_type="linear", export_embed=True)
        models[name] = m.to(device).eval()
    return models


def count_macs(model, device):
    """Best-effort forward MACs/jet via fvcore (E1 anchor ~326.6M). None if fvcore absent."""
    try:
        from fvcore.nn import FlopCountAnalysis
        f = torch.zeros(1, 17, MAX_PART, device=device)
        v = torch.zeros(1, 4, MAX_PART, device=device)
        mk = torch.ones(1, 1, MAX_PART, dtype=torch.bool, device=device)
        return int(FlopCountAnalysis(model, (f, v, mk)).total())
    except Exception as e:  # noqa
        print(f"  [flops] fvcore unavailable/failed: {type(e).__name__}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--files", nargs="+", required=True, help="JC2 parquet paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part-ckpt", default=str(REPO / "models/ParT/ParT_full.pt"))
    ap.add_argument("--armp-ckpts", nargs="*", default=[], help="Arm P checkpoints")
    ap.add_argument("--arms-ckpts", nargs="*", default=[], help="Arm S checkpoints")
    ap.add_argument("--max-per-file", type=int, default=0, help="0 = all jets in file")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} split={args.split} files={len(args.files)}")

    import re

    def bb_name(ckpt, prefix):
        # derive the seed from the checkpoint's run dir, e.g. .../arm_p_s3/net...pt
        m = re.search(r"s(\d+)$", Path(ckpt).parent.name)
        return f"{prefix}_s{m.group(1)}" if m else f"{prefix}_{Path(ckpt).parent.name}"

    spec = [("part", "part", args.part_ckpt)]
    for c in args.armp_ckpts:
        spec.append((bb_name(c, "armp"), "sophon", c))   # Sophon arch, ParT preprocessing
    for c in args.arms_ckpts:
        spec.append((bb_name(c, "arms"), "sophon", c))   # Sophon arch, Sophon preprocessing
    want_sophon = len(args.arms_ckpts) > 0
    models = load_backbones(spec, device)
    part_like = [n for n, k, _ in spec if k == "part" or n.startswith("armp")]
    sophon_like = [n for n, _, _ in spec if n.startswith("arms")]

    emb = {n: [] for n, _, _ in spec}
    labels, jetpts = [], []
    t0 = time.time()
    for j, fp in enumerate(args.files):
        pf, plv, sf, slv, masks, lab, jpt = process_file(fp, want_sophon)
        if args.max_per_file and len(lab) > args.max_per_file:
            k = args.max_per_file
            pf, plv, masks, lab, jpt = pf[:k], plv[:k], masks[:k], lab[:k], jpt[:k]
            if want_sophon:
                sf, slv = sf[:k], slv[:k]
        for name in part_like:
            emb[name].append(embed(models[name], pf, plv, masks, device, args.batch_size))
        for name in sophon_like:
            emb[name].append(embed(models[name], sf, slv, masks, device, args.batch_size))
        labels.append(lab)
        jetpts.append(jpt)
        print(f"  [{j + 1}/{len(args.files)}] {Path(fp).name}: {len(lab):,} jets "
              f"({time.time() - t0:.0f}s)")

    label188 = np.concatenate(labels)
    jetpt = np.concatenate(jetpts)
    np.save(out / f"{args.split}_label188.npy", label188)
    np.save(out / f"{args.split}_jetpt.npy", jetpt)
    for name, _, _ in spec:
        e = np.concatenate(emb[name])
        assert len(e) == len(label188), f"{name}: row misalignment"
        np.save(out / f"{name}_{args.split}_emb.npy", e)
        print(f"  saved {name}_{args.split}_emb.npy {e.shape}")

    macs = {n: count_macs(models[n], device) for n, _, _ in spec}
    total_infer_flops = sum(2 * m * len(label188) for m in macs.values() if m)  # 2*MAC = FLOP
    meta = {"split": args.split, "n_jets": int(len(label188)),
            "n_files": len(args.files), "files": [Path(f).name for f in args.files],
            "backbones": {n: {"kind": k, "ckpt": c, "macs_per_jet": macs[n]} for n, k, c in spec},
            "max_particles": MAX_PART, "batch_size": args.batch_size,
            "wall_s": round(time.time() - t0, 1),
            "total_inference_flops": total_infer_flops,
            "preprocessing": "part=data_arm_p.yaml, sophon=data_arm_s.yaml (scaled)",
            "note": "row order = native file order, identical across backbones (paired)"}
    (out / f"metadata_{args.split}.json").write_text(json.dumps(meta, indent=2))
    print(f"done {args.split}: {len(label188):,} jets in {meta['wall_s']}s")


if __name__ == "__main__":
    main()
