#!/usr/bin/env python3
"""Measure forward MACs/jet (fvcore) for every E8 backbone configuration.

The in-cluster extract/FT jobs could not count MACs (fvcore absent from the
image), so the reference numbers are measured here — MACs are an architecture
property, independent of weights/GPU, so a local CPU trace is exact. Inputs
mirror the jobs: (1,17,128) features, (1,4,128) vectors, full mask.

Writes experiments/E8/flops_reference.json. Total-FLOPs conventions:
  inference:  FLOPs = 2 x MACs x n_jets
  training:   FLOPs ~= 3 x 2 x MACs x jets_seen   (fwd+bwd ~ 3x fwd)
  (jets_seen per FT run = train_size x num_epochs from its results.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.part_wrapper import ParTModel                      # noqa: E402
from src.models.sophon_wrapper import SophonTransferModel          # noqa: E402

MAX_PART = 128


def macs(model):
    model.eval()
    f = torch.zeros(1, 17, MAX_PART)
    v = torch.zeros(1, 4, MAX_PART)
    mk = torch.ones(1, 1, MAX_PART, dtype=torch.bool)
    with torch.no_grad():
        return int(FlopCountAnalysis(model, (f, v, mk)).unsupported_ops_warnings(False)
                   .uncalled_modules_warnings(False).total())


def main():
    out = {
        "method": "fvcore FlopCountAnalysis, (1,17,128)+(1,4,128)+mask, eval mode, CPU",
        "e1_anchor_macs_sophon_arch": 326_638_704,
        "note": ("measured locally 2026-07-19 because fvcore is absent from the "
                 "compute image; MACs are architecture properties (weight- and "
                 "device-independent)"),
        "configs": {},
    }
    specs = {
        "part_arch_10c (official ParT, E8 extraction/FT)":
            lambda: ParTModel(checkpoint_path=None),
        "sophon_arch_linear10 (Arm P/S extraction)":
            lambda: SophonTransferModel(None, num_classes=10, head_type="linear"),
        "sophon_arch_mlp10 (Arm P/S fine-tune head)":
            lambda: SophonTransferModel(None, num_classes=10, head_type="mlp"),
    }
    for name, build in specs.items():
        m = macs(build())
        out["configs"][name] = {"macs_per_jet_fwd": m,
                                "flops_per_jet_2xmac": 2 * m}
        print(f"{name}: {m:,} MACs/jet")
    dest = REPO / "experiments/E8/flops_reference.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
