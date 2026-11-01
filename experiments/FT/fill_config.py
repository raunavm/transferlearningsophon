#!/usr/bin/env python3
"""A downstream data config with its zero-filled slots at a model's input-BatchNorm mean (2026-11-01).

The top and quark/gluon configs (configs/finetune/TopReference.yaml, EnergyFlowQG.yaml) read every
absent particle feature from `part_zero_<feature>: ak.zeros_like(part_energy)`. This writes the same
config with each of those set to the running mean of the checkpoint's input BatchNorm for that slot
(the mean fill of experiments/EVAL/input_variants.py, ntop/nqg), so fine-tuning sees what the mean-fill
probe sees. Every filled slot's standardization must be the identity on that value ([name, null], or
centre 0, scale 1 with the value inside the clip range), so the network input is the mean itself.

    python3 experiments/FT/fill_config.py --checkpoint ckpt.pt --src configs/finetune/TopReference.yaml \\
        --out /workspace/meanfill/TopReference_meanfill.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

import numpy as np

BN_MEAN = "mod.embed.input_bn.running_mean"


def pf_feature_entries(text: str) -> list:
    """The pf_features variable entries ([name, *params]) of a data config, in order."""
    m = re.search(r"^\s*pf_features:\s*$", text, re.M)
    if m is None:
        raise SystemExit("FATAL: no pf_features input in this config")
    block = text[m.end():].split("vars:", 1)[1]
    out = []
    for line in block.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if not s.startswith("- ["):
            break
        out.append([x.strip() for x in s[3:].rstrip("]").split(",")])
    return out


def fill(text: str, mean: np.ndarray) -> tuple:
    """(new config text, {variable: value}) with every part_zero_* slot at mean[slot]."""
    entries = pf_feature_entries(text)
    if len(entries) != mean.size:
        raise SystemExit(f"FATAL: {len(entries)} pf_features but a BatchNorm over {mean.size}")
    values = {}
    for i, (name, *par) in enumerate(entries):
        if not name.startswith("part_zero_"):
            continue
        v = float(mean[i])
        identity = par == ["null"] or (len(par) == 4 and float(par[0]) == 0 and float(par[1]) == 1
                                       and float(par[2]) <= v <= float(par[3]))
        if not identity:
            raise SystemExit(f"FATAL: {name} {par}: not the identity on {v}, the input would not be the mean")
        line = re.compile(rf"^(\s*{name}: ak\.zeros_like\(part_energy\))\s*$", re.M)
        if len(line.findall(text)) != 1:
            raise SystemExit(f"FATAL: {name} is not defined once as ak.zeros_like(part_energy)")
        text = line.sub(rf"\g<1> + {v!r}", text)
        values[name] = v
    if not values:
        raise SystemExit("FATAL: no zero-filled slot in this config")
    return text, values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--src", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    import torch
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    mean = sd[BN_MEAN].numpy().astype(np.float64)
    text, values = fill(a.src.read_text(), mean)
    sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    head = (f"# MEAN FILL of {a.src}: each zero-filled slot at the input-BatchNorm running mean of\n"
            f"# {a.checkpoint} (sha256 {sha}), written by experiments/FT/fill_config.py.\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(head + text)
    print(a.out, {k: round(v, 5) for k, v in values.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
