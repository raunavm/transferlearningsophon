#!/usr/bin/env python3
"""Compute logits.npy from an existing feature cache, without re-extracting.

WHY THIS EXISTS. anomaly.py's vocabulary-DEFINED score (class_sum) needs the
per-class head outputs, and no extraction in this project writes them:
extract_features.py gates the write on --save-logits, which is off by default
and is passed by no job spec. The score therefore could not be built at all,
and anomaly_results.json simply had no class_sum for any cell.

WHY IT DOES NOT NEED THE GPU OR THE DATA. extract_features.py takes its
features with a forward PRE-hook on `mod.fc`, so features.npy holds exactly
`fc`'s INPUT -- the 128-d class token after the final norm. weaver 0.4.17's
forward ends

    x_cls  = self.norm(cls_tokens).squeeze(0)
    output = self.fc(x_cls)

so the logits are `fc(features)` EXACTLY, not an approximation. The head is a
few hundred KB of weights and the arithmetic is one matmul per jet on CPU. That
makes this a ~minute of CPU per arm instead of a full re-extraction over 335
files, and -- more importantly -- it cannot disagree with the cached features,
because it is applied TO them.

The equivalence is asserted, not assumed: extract_features.py's own self-check
already verifies `fc(captured) == model(...)` to numerical tolerance before any
data is read, and this script re-checks the head's output width against the
arm's declared K before writing anything.

Usage:
    python3 experiments/EVAL/logits_from_features.py \
        --features /data/results/eval/mtx-r16q1-s2/features_e79 \
        --checkpoint /data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt \
        --num-classes 17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import torch


def head_from_checkpoint(ckpt: pathlib.Path, declared_k: int):
    """The `mod.fc` Sequential, rebuilt from the checkpoint alone.

    fc_params=[(512, 0.1)] gives fc.0 = Linear(128, 512) and fc.3 = Linear(512, K)
    with ReLU and Dropout between; dropout is inert in eval mode, so only the
    two Linears carry parameters and the layer INDICES are read off the state
    dict rather than assumed.
    """
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    lin = {}
    for k, v in state.items():
        m = re.search(r"(?:^|\.)fc\.(\d+)\.(weight|bias)$", k)
        if m and getattr(v, "ndim", 0) in (1, 2):
            lin.setdefault(int(m.group(1)), {})[m.group(2)] = v
    idx = sorted(i for i, d in lin.items() if "weight" in d and d["weight"].ndim == 2)
    if not idx:
        sys.exit(f"FATAL: no fc.<i>.weight in {ckpt}")

    layers, in_dim = [], None
    for i in idx:
        w = lin[i]["weight"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if in_dim is not None and in_f != in_dim:
            sys.exit(f"FATAL: fc.{i} expects {in_f} inputs, previous layer gives {in_dim}")
        layer = torch.nn.Linear(in_f, out_f)
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(lin[i]["bias"])
        layers.append(layer)
        in_dim = out_f

    k_ckpt = in_dim
    if k_ckpt != declared_k:
        sys.exit(f"FATAL: this head outputs {k_ckpt} classes, --num-classes says "
                 f"{declared_k}. The cache would be attributed to the wrong arm.")

    # ReLU between Linears -- NOT GELU (docs/GROUND_TRUTH.md), and dropout is
    # inert under eval() so it is simply omitted.
    mods = []
    for j, layer in enumerate(layers):
        mods.append(layer)
        if j < len(layers) - 1:
            mods.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*mods).eval()
    return net, int(layers[0].weight.shape[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="an existing feature cache dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    d = pathlib.Path(a.features)
    fpath, lpath = d / "features.npy", d / "logits.npy"
    if not fpath.exists():
        sys.exit(f"FATAL: no {fpath}")
    if lpath.exists() and not a.overwrite:
        sys.exit(f"FATAL: {lpath} already exists; pass --overwrite to replace it")

    ckpt = pathlib.Path(a.checkpoint)
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    # The cache must have been built from THIS checkpoint, or the logits are a
    # different model's head applied to these features and nothing would say so.
    mpath = d / "extract_manifest.json"
    if mpath.exists():
        prior = json.loads(mpath.read_text())
        was = prior.get("checkpoint_sha256") or prior.get("sha256")
        if was and was != ckpt_sha:
            sys.exit(f"FATAL: {d} was extracted from checkpoint {was[:16]}, but "
                     f"--checkpoint is {ckpt_sha[:16]}. Applying this head to "
                     f"those features would mix two models.")
    else:
        print(f"WARNING: {mpath} absent; cannot confirm the cache came from "
              f"this checkpoint.")

    net, in_dim = head_from_checkpoint(ckpt, a.num_classes)
    F = np.load(fpath, mmap_mode="r")
    if F.shape[1] != in_dim:
        sys.exit(f"FATAL: features are {F.shape[1]}-d, the head expects {in_dim}")

    out = np.empty((F.shape[0], a.num_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, F.shape[0], a.batch_size):
            x = torch.from_numpy(np.array(F[i:i + a.batch_size], dtype=np.float32))
            out[i:i + a.batch_size] = net(x).numpy()
    np.save(lpath, out)

    # Record HOW these logits were made. They are not from the extraction pass,
    # and a reader must be able to tell.
    side = {"source": "logits_from_features.py",
            "note": "fc(features.npy); features.npy is fc's input by construction",
            "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
            "num_classes": int(a.num_classes),
            "n_jets": int(out.shape[0]),
            "logits_sha256": hashlib.sha256(out.tobytes()).hexdigest()}
    (d / "logits_manifest.json").write_text(json.dumps(side, indent=2))
    print(f"wrote {lpath}  {out.shape[0]:,} x {out.shape[1]}")
    print(f"argmax agrees with a stored prediction only if the cache is this "
          f"arm's; checkpoint {ckpt_sha[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
