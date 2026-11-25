#!/usr/bin/env python3
"""Accuracy of the pretrained 188-class output layer itself, with no training (PI 2026-11-25, LEDGER §28 1.2), on the
JetClass-II test jets the linear probes use (the class-balanced test set, 903,922 jets), and the majority-class
accuracy on those jets. CPU only; nothing is fitted.

The stored features are the input of the classifier's output layer (extract_features.ClsTap: fc's input, x_cls), so
the output layer applied to them gives the pretrained network's own 188 logits: ParticleTransformer's fc, each hidden
block Linear -> ReLU (dropout off at evaluation), then the last Linear. The features were stored as float16, so a logit
can differ from the network's float32 one in its last digits; an argmax tie broken differently moves the accuracy by
at most the share of such jets, printed as "near_ties" (top-two logits closer than 1e-3).

    python3 experiments/EVAL/head_accuracy.py --model mtx-l188-s1=<feature dir> ... --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np


def fc_layers(state: dict) -> list:
    """[(weight, bias, relu after)] of the output layer: mod.fc.<i>.0.* hidden blocks, then mod.fc.<last>.*."""
    keys = {k[len("mod.fc."):]: v for k, v in state.items() if k.startswith("mod.fc.")}
    if not keys:
        raise SystemExit("FATAL: no mod.fc.* tensors: the checkpoint has no output layer")
    idx = sorted({int(k.split(".")[0]) for k in keys})
    out = []
    for i in idx:
        if f"{i}.0.weight" in keys:
            out.append((keys[f"{i}.0.weight"], keys[f"{i}.0.bias"], True))
        elif f"{i}.weight" in keys:
            out.append((keys[f"{i}.weight"], keys[f"{i}.bias"], False))
        else:
            raise SystemExit(f"FATAL: fc.{i} has neither a hidden block nor a final layer: {sorted(keys)[:6]}")
    if [r for *_, r in out] != [True] * (len(out) - 1) + [False]:
        raise SystemExit("FATAL: the output layer is not hidden blocks followed by one final layer")
    return [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32), r) for w, b, r in out]


def logits(x: np.ndarray, layers: list) -> np.ndarray:
    h = x.astype(np.float32)
    for w, b, relu in layers:
        h = h @ w.T + b
        if relu:
            h = np.maximum(h, 0.0)
    return h


def head_accuracy(feats: np.ndarray, labels: np.ndarray, layers: list, chunk: int = 50_000) -> dict:
    hit, ties = 0, 0
    for i in range(0, feats.shape[0], chunk):
        z = logits(feats[i:i + chunk], layers)
        top2 = np.partition(z, -2, axis=1)[:, -2:]
        ties += int((np.abs(top2[:, 1] - top2[:, 0]) < 1e-3).sum())
        hit += int((z.argmax(1) == labels[i:i + chunk]).sum())
    n = int(labels.size)
    return {"accuracy": hit / n, "n_jets": n, "near_ties": ties / n, "n_out": int(layers[-1][0].shape[0])}


def majority(labels: np.ndarray) -> dict:
    c = np.bincount(labels)
    return {"class": int(c.argmax()), "accuracy": float(c.max() / labels.size), "n_classes_present": int((c > 0).sum())}


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FEATURE_DIR[=CHECKPOINT]")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists")
    doc, ref = {"script": "experiments/EVAL/head_accuracy.py",
                "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "models": {}}, None
    for spec in a.model:
        name, fdir, *ck = spec.split("=")
        fdir = pathlib.Path(fdir)
        man = json.loads((fdir / "manifest.json").read_text())
        path = pathlib.Path(ck[0] if ck else man["checkpoint"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != man["checkpoint_sha256"]:
            raise SystemExit(f"FATAL: {path} sha256 {sha} is not the features' checkpoint {man['checkpoint_sha256']}")
        labels = np.load(fdir / "label188.npy").astype(np.int64)
        if ref is None:
            ref = (man["rows_sha256"], man["stream_label188_sha256"])
            doc["majority"] = majority(labels)
        elif (man["rows_sha256"], man["stream_label188_sha256"]) != ref:
            raise SystemExit(f"FATAL: {name} is not on the same test jets as the first model")
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state = {re.sub(r"^module\.", "", k): v for k, v in state.items()}
        if not any(k.startswith("mod.fc.") for k in state):         # a checkpoint saved without the "mod." wrapper
            state = {f"mod.{k}": v for k, v in state.items()}
        layers = fc_layers(state)
        if layers[-1][0].shape[0] != man["num_classes"] + man.get("num_reg", 0):
            raise SystemExit(f"FATAL: {name}: output layer has {layers[-1][0].shape[0]} outputs")
        layers[-1] = (layers[-1][0][:man["num_classes"]], layers[-1][1][:man["num_classes"]], False)
        feats = np.load(fdir / "features.npy", mmap_mode="r")
        r = head_accuracy(feats, labels, layers)
        doc["models"][name] = {**r, "features": str(fdir), "checkpoint": str(path), "checkpoint_sha256": sha}
        print(name, json.dumps(r), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc["majority"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
