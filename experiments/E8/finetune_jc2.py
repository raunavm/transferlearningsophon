#!/usr/bin/env python3
"""Phase-2 fine-tuning driver for the JC1->JC2 transfer study (E8).

Fine-tunes ONE backbone (part / armp / arms) to the JC2 g10 task across
{partial_ft, full_ft} x size-ladder x seeds, REUSING run_single from
train_finetune_sweep (early stopping, resume, metrics, best_model.pt weights,
loss/ROC/confusion plots). Adds fvcore FLOPs + a provenance JSON.

Features come from prep_jc2_features:
  part, armp  ->  --feat-dir features_part   --arch part / sophon
  arms        ->  --feat-dir features_sophon  --arch sophon
(part uses the official ParT arch; armp/arms use the Sophon arch — both via the
generalized create_model(arch=...).)

One job per backbone runs the whole {partial,full} x sizes x seeds grid; the FT
fleet is 3 such jobs in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import train_finetune_sweep as tfs  # noqa: E402
from train_finetune_sweep import (FeatureDataset, collate_fn,  # noqa: E402
                                  load_npy_features, run_single)

# g10 class names (index 9 = QCD, matching the g10sem map); cosmetic for plots/metrics.
G10_NAMES = ["X2P_QQ", "X2P_gg", "X2P_lep", "YY_4Phad", "YY_3Phad",
            "YY_had_l", "YY_had_tau", "YY_QQ_lv", "YY_QQ_tauhv", "QCD"]


def make_loader(feat_dir, batch_size, max_jets=None):
    f, lv, m, lab = load_npy_features(feat_dir, max_jets=max_jets)
    return DataLoader(FeatureDataset(f, lv, m, lab), batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True, collate_fn=collate_fn)


def macs_per_jet(model, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        f = torch.zeros(1, 17, 128, device=device)
        v = torch.zeros(1, 4, 128, device=device)
        mk = torch.ones(1, 1, 128, dtype=torch.bool, device=device)
        return int(FlopCountAnalysis(model, (f, v, mk)).total())
    except Exception as e:  # noqa
        print(f"[flops] fvcore unavailable: {type(e).__name__}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["part", "armp", "arms"])
    ap.add_argument("--feat-dir", required=True, help="features_part or features_sophon")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", required=True, choices=["part", "sophon"])
    ap.add_argument("--strategies", nargs="+", default=["partial_ft", "full_ft"])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[10_000, 30_000, 100_000, 300_000, 1_000_000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-jets", type=int, default=200_000)
    ap.add_argument("--frozen-layers", type=int, default=4)
    args = ap.parse_args()

    tfs.LABEL_NAMES = G10_NAMES  # run_single reads the module global for plots/metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} backbone={args.backbone} arch={args.arch}")

    val_loader = make_loader(str(Path(args.feat_dir) / "val"), args.batch_size, args.val_jets)
    test_loader = make_loader(str(Path(args.feat_dir) / "test"), args.batch_size)
    train_dir = str(Path(args.feat_dir) / "train")
    out_bb = Path(args.output_dir) / args.backbone

    runs = []
    for strat in args.strategies:
        for size in args.sizes:
            for seed in args.seeds:
                print(f"\n=== {args.backbone} {strat} size={size:,} seed={seed} ===")
                try:
                    res = run_single(
                        train_dir, val_loader, test_loader, strat, args.checkpoint,
                        size, seed, device, output_dir=str(out_bb),
                        frozen_layers=args.frozen_layers, epochs=args.epochs,
                        batch_size=args.batch_size, head_type="mlp",
                        checkpoint_every=5, arch=args.arch)  # mid-ckpt for eviction resume
                except Exception as e:  # noqa — one diverged/OOM run must not kill the fleet job
                    print(f"    [FAILED] {strat} size={size} seed={seed}: {type(e).__name__}: {e}")
                    runs.append({"strategy": strat, "size": size, "seed": seed, "error": str(e)})
                    continue
                runs.append({"strategy": strat, "size": size, "seed": seed,
                             "test_acc": res["test_acc"], "test_auc_macro": res["test_auc_macro"],
                             "trainable_params": res["trainable_params"],
                             "num_epochs": res["num_epochs"], "wall_s": res["wall_clock_seconds"]})

    # provenance + FLOPs (representative full-FT model)
    from src.models.sophon_wrapper import create_model
    mm = create_model("full_ft", args.checkpoint, num_classes=10,
                      head_type="mlp", arch=args.arch).to(device)
    mpj = macs_per_jet(mm, device)
    prov = {"experiment": "E8 Phase-2 fine-tuning", "backbone": args.backbone, "arch": args.arch,
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "image_digest": "sha256:db235b515a278198",
            "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip().split("\n")[0],
            "node": os.environ.get("NODE_NAME", "?"), "macs_per_jet_fwd": mpj,
            "flops_model": "train FLOPs/run ~= 3 * macs_per_jet_fwd * (size * num_epochs) "
                           "(fwd+bwd ~3x fwd); best_model.pt weights + results.json per run",
            "runs": runs}
    out_bb.mkdir(parents=True, exist_ok=True)
    (out_bb / "ft_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    print(f"\nwrote {out_bb}/ft_provenance.json + per-run {{results.json,best_model.pt,*.png}}")


if __name__ == "__main__":
    main()
