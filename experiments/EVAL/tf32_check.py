#!/usr/bin/env python3
"""Does TF32 move the output layer? One checkpoint, the same jets, one GPU.

WHY. Rescoring the real-data shards on another GPU model moved a model's
three-prong log-odds by up to 0.10 for the same checkpoint on the same jets
(experiments/FIGS/data/aoj_checks_v1/reproduce.json: L4/L40 against V100),
attributed to TF32 [I]. The scoring path, extract_features.py, keeps torch's
defaults: TF32 off for matmul, ON for cuDNN convolutions (the pair embedding's
1x1 Conv1d). Ampere and later GPUs (L4, L40, 3090) run those in TF32; V100 and
2080 Ti cannot.

This scores one checkpoint on the same jets on one Ampere-or-later GPU with TF32
off, at torch's defaults, with TF32 on, off again (the run-to-run spread), and in
float32 on the CPU (no TF32 anywhere), and compares each with 'off'. TF32 is the
cause if 'defaults' or 'on' move the log-odds by an amount comparable to the
cross-GPU difference while 'off' agrees with the CPU to float32 precision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def set_tf32(matmul: bool, cudnn: bool) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def scores(logits: np.ndarray, rung: str) -> dict[str, np.ndarray]:
    """The two output-layer scores the analyses read: resonance vs QCD (the v2
    head scores) and three-prong vs QCD (the real-data discriminant)."""
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    disc = _load("discriminants", "experiments/AOJ/discriminants.py")
    z = np.asarray(logits, dtype=np.float64)
    return {"logits": z,
            "logodds_res_qcd": xv.head_score_columns(z.astype(np.float32), rung, {})[
                "logodds_res_qcd"].astype(np.float64),
            "three_prong_logodds": disc.contrast(z, rung, "three_prong")}


def compare(a: dict, b: dict) -> dict:
    """|a - b| per score: max, median, 99th and 99.9th percentiles; argmax flips."""
    out = {}
    for k in a:
        d = np.abs(a[k] - b[k]).ravel()
        out[k] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)),
                  "q99_abs": float(np.quantile(d, 0.99)), "q999_abs": float(np.quantile(d, 0.999))}
    out["argmax_flips"] = int((a["logits"].argmax(1) != b["logits"].argmax(1)).sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--num-reg", type=int, default=0)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--data-config", default=str(REPO / "configs/data/JetClassII_base.yaml"))
    ap.add_argument("--max-jets", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    import torch
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset
    defaults = (bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32))
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no GPU")
    cap = torch.cuda.get_device_capability(0)
    if cap < (8, 0):
        raise SystemExit(f"FATAL: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}) has no TF32")
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    dc = DataConfig.load(a.data_config, load_observers=False)
    model = ex.build_model(dc, a.num_classes + a.num_reg)
    prov = ex.load_trunk_or_die(model, a.checkpoint, a.num_classes, a.num_reg)
    model.eval()

    # the same number of jets from each file, so a two-prong, a three/four-prong
    # and a QCD file all contribute (the loader reads one whole file at a time)
    per_file = -(-a.max_jets // len(a.data_test))
    batches, labels = [], []
    for f in a.data_test:
        ds = SimpleIterDataset({"_": [f]}, a.data_config, for_training=False,
                               fetch_by_files=True, fetch_step=1, name="tf32_check")
        n = 0
        for X, y, _ in torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=1):
            take = min(len(y[dc.label_names[0]]), per_file - n)
            batches.append([X[k][:take] for k in dc.input_names])
            labels.append(y[dc.label_names[0]][:take].numpy().astype(np.int16))
            n += take
            if n >= per_file:
                break
    labels = np.concatenate(labels)

    def run(device: str, matmul: bool, cudnn: bool) -> dict:
        set_tf32(matmul, cudnn)
        model.to(device)
        with torch.no_grad():
            z = np.concatenate([model(*[x.to(device) for x in b]).float().cpu().numpy()
                                for b in batches])
        return scores(z[:, :a.num_classes], a.rung)

    settings = {"off": ("cuda", False, False), "defaults": ("cuda", *defaults),
                "on": ("cuda", True, True), "off_repeat": ("cuda", False, False),
                "cpu": ("cpu", False, False)}
    res = {name: run(*s) for name, s in settings.items()}
    comparisons = {f"{k}_vs_off": compare(res[k], res["off"]) for k in res if k != "off"}
    comparisons["on_vs_defaults"] = compare(res["on"], res["defaults"])

    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    out = {"gpu": torch.cuda.get_device_name(0), "capability": f"sm_{cap[0]}{cap[1]}",
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "torch_defaults": {"matmul_allow_tf32": defaults[0], "cudnn_allow_tf32": defaults[1]},
           "settings": {k: {"device": v[0], "matmul_allow_tf32": v[1], "cudnn_allow_tf32": v[2]}
                        for k, v in settings.items()},
           "n_jets": int(labels.size), "label188_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
           "rung": a.rung, "num_classes": a.num_classes, "num_reg": a.num_reg,
           "data_test_first": a.data_test[:3], "data_config": a.data_config,
           "checkpoint": str(a.checkpoint), "checkpoint_sha256": prov["sha256"],
           "code_commit": commit,
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "comparisons": comparisons}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), label188=labels,
                        **{f"{k}|{s}": v[s].astype(np.float32) for k, v in res.items()
                           for s in ("logodds_res_qcd", "three_prong_logodds")})
    print(json.dumps({k: {s: v[s]["max_abs"] for s in ("logodds_res_qcd", "three_prong_logodds")}
                      for k, v in comparisons.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
