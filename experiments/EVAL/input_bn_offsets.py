#!/usr/bin/env python3
"""Where a zeroed input lies after the network's input normalization (table tab:inputs, appendix A8).

The top and quark/gluon configurations feed a missing input as a raw zero with no preprocessing, so
the first thing it meets is ParT's Embed.input_bn. In standard deviations of the pretraining
particles, a zero lands at (0 - running_mean) / sqrt(running_var + eps). This reads those buffers
from one checkpoint and names each channel by its position in the top configuration's pf_features
(part_zero_<x> named part_<x>, as mlst_results.table_inputs does).

    python3 experiments/EVAL/input_bn_offsets.py --checkpoint <net_epoch-N_state.pt> \\
        --out /data/results/eval/input_bn_offsets.json
"""
import argparse
import hashlib
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/finetune/TopReference.yaml"
KEY = "embed.input_bn."
EPS = 1e-5          # torch.nn.BatchNorm1d's default, which ParT's Embed uses


def names(config: pathlib.Path = CONFIG) -> list:
    import yaml
    c = yaml.safe_load(config.read_text())
    return [v[0].replace("part_zero_", "part_") for v in c["inputs"]["pf_features"]["vars"]]


def offsets(state: dict, inputs: list, eps: float = EPS) -> dict:
    """{input: (0 - mean) / sqrt(var + eps)} from the one input BatchNorm in a state dict."""
    mean = [k for k in state if k.endswith(KEY + "running_mean")]
    if len(mean) != 1:
        raise SystemExit(f"FATAL: {len(mean)} input BatchNorm layers in the checkpoint")
    m, v = (state[mean[0]].tolist(), state[mean[0].replace("running_mean", "running_var")].tolist())
    if len(m) != len(inputs):
        raise SystemExit(f"FATAL: {len(m)} input channels, {len(inputs)} names")
    return {n: -mu / math.sqrt(var + eps) for n, mu, var in zip(inputs, m, v)}


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="mtx-l188-s1")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    inputs = names()
    off = offsets(state, inputs)
    a.out.write_text(json.dumps({
        "model": a.model, "checkpoint": str(a.checkpoint),
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "config": str(CONFIG.relative_to(REPO)), "eps": EPS, "offsets": off}, indent=1) + "\n")
    print(json.dumps(off, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
