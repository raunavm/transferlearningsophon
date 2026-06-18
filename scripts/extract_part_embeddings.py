#!/usr/bin/env python3
"""Extract official ParT 128-d CLS embeddings + 10-class logits from JetClass ROOT files.

Uses the official ParT preprocessing from data/JetClass/JetClass_full.yaml:
  - pf_features:  17 per-particle features, no Sophon 500/jet_pt scaling
  - pf_vectors:   raw (px, py, pz, energy), no scaling

For each class file group, saves to --output-dir:
  {ClassName}_embeddings.npy    (N, 128)  float16
  {ClassName}_logits.npy        (N, 10)   float16
  {ClassName}_labels.npy        (N,)      int8

Plus metadata.json describing the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import uproot

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from src.models.part_wrapper import ParTModel

MAX_PART = 128
TREE_NAME = "tree"

PARTICLE_KEYS = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron",
    "part_isPhoton", "part_isElectron", "part_isMuon",
]
SCALAR_KEYS = ["jet_pt", "jet_energy"]
LABEL_KEYS = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]
ALL_KEYS = PARTICLE_KEYS + SCALAR_KEYS + LABEL_KEYS

# Filename prefix -> integer label, matching official JetClass label_keys order.
CLASS_TO_LABEL = {
    "ZJetsToNuNu": 0,  # label_QCD
    "HToBB":       1,  # label_Hbb
    "HToCC":       2,  # label_Hcc
    "HToGG":       3,  # label_Hgg
    "HToWW4Q":     4,  # label_H4q
    "HToWW2Q1L":   5,  # label_Hqql
    "ZToQQ":       6,  # label_Zqq
    "WToQQ":       7,  # label_Wqq
    "TTBar":       8,  # label_Tbqq
    "TTBarLep":    9,  # label_Tbl
}


def _norm(x: np.ndarray, subtract: float, multiply: float,
          clip_lo: float = -5.0, clip_hi: float = 5.0) -> np.ndarray:
    return np.clip((x - subtract) * multiply, clip_lo, clip_hi)


def compute_part_features(
    px, py, pz, energy,
    deta, dphi, d0val, d0err, dzval, dzerr,
    charge, isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
    jet_pt: float, jet_energy: float,
) -> np.ndarray:
    """The 17 official ParT input features (NO Sophon 500/jet_pt scaling).

    Matches data/JetClass/JetClass_full.yaml exactly.
    Returns (n_part, 17) float32.
    """
    eps = 1e-20
    jet_pt_safe = max(jet_pt, eps)
    jet_energy_safe = max(jet_energy, eps)

    pt = np.sqrt(px ** 2 + py ** 2)

    # 1-2: raw log pt / log energy with the official normalization
    pt_log = _norm(np.log(np.clip(pt, eps, None)),     subtract=1.7, multiply=0.7)
    e_log  = _norm(np.log(np.clip(energy, eps, None)), subtract=2.0, multiply=0.7)

    # 3-4: log relative pt / energy
    logptrel = _norm(np.log(np.clip(pt / jet_pt_safe, eps, None)),
                     subtract=-4.7, multiply=0.7)
    logerel  = _norm(np.log(np.clip(energy / jet_energy_safe, eps, None)),
                     subtract=-4.7, multiply=0.7)

    # 5: deltaR to jet axis
    deltaR = _norm(np.sqrt(deta ** 2 + dphi ** 2), subtract=0.2, multiply=4.0)

    # 6-11: charge + particle type one-hots
    # 12-15: impact parameters (tanh-squashed values; err clipped to [0,1])
    d0      = np.tanh(d0val)
    d0err_c = np.clip(d0err, 0.0, 1.0)
    dz      = np.tanh(dzval)
    dzerr_c = np.clip(dzerr, 0.0, 1.0)

    return np.stack([
        pt_log, e_log, logptrel, logerel, deltaR,
        charge,
        isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
        d0, d0err_c, dz, dzerr_c,
        deta, dphi,
    ], axis=1).astype(np.float32)


def process_file(fpath: str, max_particles: int = MAX_PART):
    """Read one ROOT file and produce (features, lorentz_vectors, masks, labels).

    Particles are sorted by descending pT; the leading ``max_particles`` are kept.
    pf_vectors are RAW (px, py, pz, energy) — no jet-pT scaling.
    Returns float32 / bool / int arrays.
    """
    with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
        arrays = tree.arrays(ALL_KEYS, library="np")

    n_jets = len(arrays["jet_pt"])
    feats = np.zeros((n_jets, max_particles, 17), dtype=np.float32)
    lv    = np.zeros((n_jets, max_particles, 4),  dtype=np.float32)
    masks = np.zeros((n_jets, max_particles),     dtype=bool)

    label_matrix = np.stack([arrays[k] for k in LABEL_KEYS], axis=1)
    labels = np.argmax(label_matrix, axis=1).astype(np.int8)

    for i in range(n_jets):
        px = arrays["part_px"][i]; py = arrays["part_py"][i]
        pz = arrays["part_pz"][i]; e  = arrays["part_energy"][i]
        n_part = len(px)
        pt = np.sqrt(px ** 2 + py ** 2)
        if n_part > max_particles:
            keep = np.argsort(pt)[::-1][:max_particles]
            n_part = max_particles
        else:
            keep = np.argsort(pt)[::-1]

        px = px[keep]; py = py[keep]; pz = pz[keep]; e = e[keep]
        jet_pt     = float(arrays["jet_pt"][i])
        jet_energy = float(arrays["jet_energy"][i])

        # RAW lorentz vectors — official ParT does not scale by jet pT.
        lv[i, :n_part] = np.stack([px, py, pz, e], axis=1).astype(np.float32)

        f = compute_part_features(
            px=px, py=py, pz=pz, energy=e,
            deta=arrays["part_deta"][i][keep],
            dphi=arrays["part_dphi"][i][keep],
            d0val=arrays["part_d0val"][i][keep],
            d0err=arrays["part_d0err"][i][keep],
            dzval=arrays["part_dzval"][i][keep],
            dzerr=arrays["part_dzerr"][i][keep],
            charge=arrays["part_charge"][i][keep],
            isChargedHadron=arrays["part_isChargedHadron"][i][keep],
            isNeutralHadron=arrays["part_isNeutralHadron"][i][keep],
            isPhoton=arrays["part_isPhoton"][i][keep],
            isElectron=arrays["part_isElectron"][i][keep],
            isMuon=arrays["part_isMuon"][i][keep],
            jet_pt=jet_pt, jet_energy=jet_energy,
        )
        feats[i, :n_part] = f
        masks[i, :n_part] = True
    return feats, lv, masks, labels


def discover_class_files(root_dir: Path) -> dict[str, list[Path]]:
    """Group ROOT files by class prefix from filenames like HToBB_001.root."""
    out: dict[str, list[Path]] = {}
    for f in sorted(root_dir.glob("*.root")):
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        cls = parts[0]
        if cls in CLASS_TO_LABEL:
            out.setdefault(cls, []).append(f)
    for cls in out:
        out[cls].sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return out


@torch.no_grad()
def forward_batches(model, feats, lv, masks, device, batch_size: int = 512):
    """Permute to (B, C, N), batch, forward, return (logits, embeddings)."""
    n = len(feats)
    out_logits = np.zeros((n, 10), dtype=np.float16)
    out_emb    = np.zeros((n, 128), dtype=np.float16)
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        f = torch.from_numpy(feats[sl]).permute(0, 2, 1).to(device).float()
        v = torch.from_numpy(lv[sl]).permute(0, 2, 1).to(device).float()
        m = torch.from_numpy(masks[sl]).unsqueeze(1).to(device)
        logits, x_cls = model(f, v, m)
        out_logits[sl] = logits.float().cpu().numpy().astype(np.float16)
        out_emb[sl]    = x_cls.float().cpu().numpy().astype(np.float16)
    return out_logits, out_emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/ParT/ParT_full.pt")
    p.add_argument("--root-dir", default="/data/JetClass/Pythia/test_20M")
    p.add_argument("--events-per-class", type=int, default=10000)
    p.add_argument("--output-dir", default="embeddings_part_full_test")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root_dir = Path(args.root_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ParTModel(checkpoint_path=args.checkpoint).to(device).eval()

    files_by_class = discover_class_files(root_dir)
    missing = [c for c in CLASS_TO_LABEL if c not in files_by_class]
    if missing:
        print(f"WARNING: no ROOT files for classes {missing} under {root_dir}")

    counts: dict[str, int] = {}
    t_start = time.time()
    for cls in sorted(files_by_class.keys()):
        emb_path = out / f"{cls}_embeddings.npy"
        if args.skip_existing and emb_path.exists():
            print(f"[skip] {cls} (exists)")
            counts[cls] = int(np.load(emb_path, mmap_mode="r").shape[0])
            continue

        target = args.events_per_class
        emb_chunks = []; logit_chunks = []; label_chunks = []
        for fpath in files_by_class[cls]:
            if target <= 0:
                break
            feats, lv, masks, labels = process_file(str(fpath), MAX_PART)
            if len(labels) > target:
                feats  = feats[:target]
                lv     = lv[:target]
                masks  = masks[:target]
                labels = labels[:target]
            logits, emb = forward_batches(model, feats, lv, masks, device,
                                          batch_size=args.batch_size)
            emb_chunks.append(emb)
            logit_chunks.append(logits)
            label_chunks.append(labels)
            target -= len(labels)
            print(f"  {cls}: {fpath.name} -> {len(labels):,} jets  "
                  f"(remaining target: {max(target, 0):,})")

        if not emb_chunks:
            print(f"  {cls}: no jets extracted")
            continue
        emb_all   = np.concatenate(emb_chunks, axis=0)
        logit_all = np.concatenate(logit_chunks, axis=0)
        label_all = np.concatenate(label_chunks, axis=0)
        np.save(out / f"{cls}_embeddings.npy", emb_all)
        np.save(out / f"{cls}_logits.npy",     logit_all)
        np.save(out / f"{cls}_labels.npy",     label_all)
        counts[cls] = int(len(label_all))
        print(f"  {cls}: saved {len(label_all):,} jets")

    metadata = {
        "checkpoint":       args.checkpoint,
        "checkpoint_url":   "https://github.com/jet-universe/particle_transformer/raw/main/models/ParT_full.pt",
        "root_dir":         str(root_dir),
        "events_per_class": args.events_per_class,
        "max_particles":    MAX_PART,
        "batch_size":       args.batch_size,
        "device":           str(device),
        "preprocessing":    {
            "source":      "data/JetClass/JetClass_full.yaml (official ParT)",
            "pf_features": [
                "pt_log (subtract=1.7, multiply=0.7, clip [-5,5])",
                "e_log (subtract=2.0, multiply=0.7, clip [-5,5])",
                "logptrel (subtract=-4.7, multiply=0.7, clip [-5,5])",
                "logerel (subtract=-4.7, multiply=0.7, clip [-5,5])",
                "deltaR (subtract=0.2, multiply=4.0, clip [-5,5])",
                "charge",
                "isChargedHadron", "isNeutralHadron",
                "isPhoton", "isElectron", "isMuon",
                "tanh(d0val)", "clip(d0err,0,1)",
                "tanh(dzval)", "clip(dzerr,0,1)",
                "deta", "dphi",
            ],
            "pf_vectors":  ["px", "py", "pz", "energy"],
            "lv_scaling":  "NONE (raw four-vectors; no Sophon 500/jet_pt scaling)",
            "particle_order": "descending pT; leading max_particles kept",
        },
        "class_map":         CLASS_TO_LABEL,
        "per_class_counts":  counts,
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(THIS_DIR.parent), stderr=subprocess.DEVNULL).decode().strip()
        metadata["git_commit"] = sha
    except Exception:
        pass

    with open(out / "metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"\nWrote {out / 'metadata.json'}")
    print(f"Per-class counts: {counts}")
    print(f"Total wall clock: {metadata['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
