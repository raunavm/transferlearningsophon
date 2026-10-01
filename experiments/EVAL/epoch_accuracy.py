#!/usr/bin/env python3
"""Classification accuracy of pretraining runs' checkpoints at fixed epochs, on
fixed test jets. For S8 (docs/PRESPEC_2026-09.md, clarification of 2026-09-27).

WHY NOT THE LOGGED VALIDATION ACCURACY. Each epoch validates on the next slice of
a validation reader that restarts at every resume, so a mass-output model and
its twin validate on different jets at most epochs (experiments/MTX/val_by_epoch.py
measures it: after epoch 5 most pairs differ). Their logged accuracies are then
not a paired comparison. Here every checkpoint of every run is scored on the
same jets: every --stride-th of the first --max-jets of the test list, the
2,000,000 jets every frozen-feature cache holds. Not a head slice: the list is
interleaved by file, so its first 100,000 jets come from one or two files.
--align-with names one of those caches and the job refuses to score anything
unless its labels, strided the same way, are the jets read here.

The test jets are read ONCE and held in memory; each checkpoint is loaded with
extract_features.load_trunk_or_die (the same width and completeness guards)
and scored with the class outputs only -- a mass-output model's last output is
its mass regression and is dropped, exactly as its training loop scores it.
The truth is the native label mapped to the run's vocabulary through
configs/labelmaps/rung_label_maps.v1.csv, which tests/test_arm_configs.py ties
to the arm configs the runs trained on.

Writes one small JSON: per epoch the accuracy, the number of jets and the
checkpoint's sha256, plus the sha256 of the native-label array, so the analysis
can refuse pairs that were not scored on the same jets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("extract_features", REPO / "experiments/EVAL/extract_features.py")
XF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XF)
MAPS = REPO / "configs/labelmaps/rung_label_maps.v1.csv"


def vocabulary_map(rung: str) -> np.ndarray:
    """native jet_label (0-187) -> class index in `rung`'s vocabulary."""
    with MAPS.open() as f:
        rows = list(csv.DictReader(f))
    m = np.full(len(rows), -1, dtype=np.int64)
    for r in rows:
        m[int(r["jet_label"])] = int(r[rung])
    if (m < 0).any():
        raise SystemExit(f"FATAL: {MAPS} leaves native labels unmapped for {rung}")
    return m


def accuracy(logits: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Top-1 accuracy over the K class outputs; any further columns are regression outputs."""
    return float((logits[:, :k].argmax(1) == truth).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, metavar="RUN_DIR:RUNG:K:NUM_REG",
                    help="RUNG is a column of rung_label_maps.v1.csv (e.g. L162); NUM_REG is 1 "
                         "for a mass-output model. All runs are scored on the same jets.")
    ap.add_argument("--epochs", type=int, nargs="+", required=True, help="weaver epoch numbers")
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--max-jets", type=int, required=True, help="length of the stream read")
    ap.add_argument("--stride", type=int, default=1, help="score every k-th jet of it")
    ap.add_argument("--align-with", type=pathlib.Path, default=None,
                    help="a frozen-feature cache of the same stream; its label188.npy must match")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; write a new file")
    tf32 = XF.strict_fp32()        # the model runs on the CPU here; recorded all the same
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset

    cfg = DataConfig.load(a.data_config, load_observers=False)
    kw = dict(for_training=False, fetch_by_files=True, fetch_step=1, name="epoch_accuracy")
    unknown = [k for k in kw if k not in inspect.signature(SimpleIterDataset.__init__).parameters]
    if unknown:
        raise SystemExit(f"FATAL: this weaver's SimpleIterDataset has no {unknown}")
    loader = torch.utils.data.DataLoader(SimpleIterDataset({"_": list(a.data_test)}, a.data_config, **kw),
                                         batch_size=a.batch_size, num_workers=a.num_workers)
    batches, native, n = [], [], 0
    for X, y, _ in loader:                  # keep rows n, n + stride, ... below max_jets
        idx = np.arange(n, n + len(y[cfg.label_names[0]]))
        sel = torch.from_numpy((idx % a.stride == 0) & (idx < a.max_jets))
        n += len(idx)
        if sel.any():
            batches.append([X[k][sel] for k in cfg.input_names])
            native.append(y[cfg.label_names[0]][sel].numpy().astype(np.int64))
        if n >= a.max_jets:
            break
    if n < a.max_jets:
        raise SystemExit(f"FATAL: the test list holds {n} jets, fewer than {a.max_jets}")
    native = np.concatenate(native)
    if a.align_with is not None:
        ref = np.load(a.align_with / "label188.npy")[:a.max_jets][::a.stride]
        if not np.array_equal(ref.astype(np.int64), native):
            raise SystemExit(f"FATAL: these are not the jets of {a.align_with}")
    keep = len(native)
    print(f"{keep:,} test jets held in memory", flush=True)

    out = {"n_jets": keep, "stream_jets": a.max_jets, "stride": a.stride, "tf32": tf32,
           "aligned_with": None if a.align_with is None else str(a.align_with),
           "native_label_sha256": hashlib.sha256(native.astype(np.int16).tobytes()).hexdigest(),
           "data_config_sha256": hashlib.sha256(pathlib.Path(a.data_config).read_bytes()).hexdigest(),
           "runs": {}}
    for spec in a.runs:
        run_dir, rung, k, num_reg = spec.split(":")
        run_dir, k, num_reg = pathlib.Path(run_dir), int(k), int(num_reg)
        truth = vocabulary_map(rung)[native]
        res = out["runs"][run_dir.name] = {"run_dir": str(run_dir), "rung": rung,
                                           "num_classes": k, "num_reg": num_reg, "epochs": {}}
        for e in a.epochs:
            ckpt = run_dir / f"net_epoch-{e}_state.pt"
            model = XF.build_model(cfg, k + num_reg)
            prov = XF.load_trunk_or_die(model, ckpt, k, num_reg)
            model.eval()
            t0, logits = time.time(), []
            with torch.no_grad():
                for inputs in batches:
                    logits.append(model(*inputs).float().numpy())
            res["epochs"][str(e)] = {"accuracy": accuracy(np.concatenate(logits), truth, k),
                                     "checkpoint_sha256": prov["sha256"]}
            print(f"{run_dir.name} epoch {e}: {time.time() - t0:.0f} s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
