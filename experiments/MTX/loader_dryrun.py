#!/usr/bin/env python3
"""Loader-only dry run of the v2 training stream (audit 2026-09-29, item 2).

Reads the training files through stream_v2.StreamDataset exactly as
pretrain_v2.py does, but with column projection (labels_only: jet_pt,
jet_sdmass and jet_label only; no particle column, no model, no GPU). Every
selection, weight and random draw sees the same inputs as in training, so the
rows drawn are the rows training would draw. Per epoch it records the QCD
share, the per-fetch shares, and the stream hashes after `--check-jets` jets
(to compare with a GPU run of the same seed) and after the whole epoch.

    python3 experiments/MTX/loader_dryrun.py --seed 1 --epochs 20 \\
        --samples-per-epoch 10240000 --data-config configs/arms/R16_Q1.yaml \\
        --data-train Res2P:... Res34P:... QCD:... --out /data/results/mtx_v2/loader_dryrun/x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples-per-epoch", type=int, default=10_240_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--data-split-num", type=int, default=200)
    ap.add_argument("--fetch-step", type=float, default=1.0)
    ap.add_argument("--data-fraction", type=float, default=1.0)
    ap.add_argument("--check-jets", type=int, default=200_000,
                    help="also hash the stream after this many jets (the smoke runs' epoch)")
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--data-train", nargs="+", required=True)
    ap.add_argument("--full-columns", action="store_true",
                    help="read and finalise every input column as training does (memory and "
                         "loader throughput); the default reads the label and weight columns only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import pretrain_v2 as pv
    import stream_v2 as sv
    from src.utils.reproducibility import derive_all

    seeds = derive_all(a.seed)
    files = pv.to_file_dict(a.data_train)
    ds = sv.StreamDataset(files, pv.sidecar(a.data_config), mode="train", batch_size=a.batch_size,
                          seed=seeds["data_sampling"], split_num=a.data_split_num,
                          fetch_step=a.fetch_step, labels_only=not a.full_columns,
                          data_fraction=a.data_fraction)
    steps = a.samples_per_epoch // a.batch_size
    check_steps = a.check_jets // a.batch_size
    mem = pv.MemMonitor()
    epochs = []
    for e in range(a.epochs):
        ds.set_epoch(e)
        rec = pv.StreamRecord()
        per_fetch = {}
        check = None
        t0 = time.time()
        it = iter(DataLoader(ds, batch_size=None, num_workers=a.num_workers,
                             multiprocessing_context="fork" if a.num_workers else None))
        for i in range(steps):
            _, _, Z = next(it)
            rec.update(Z)
            w = i % max(a.num_workers, 1)          # DataLoader takes workers round robin
            qcd = (Z["_jet_label"].numpy() >= sv.NATIVE_QCD_FIRST)
            for fid in np.unique(Z["_fetch"].numpy()):
                m = Z["_fetch"].numpy() == fid
                c = per_fetch.setdefault((w, int(fid)), [0, 0])
                c[0] += int(m.sum())
                c[1] += int(qcd[m].sum())
            if i + 1 == check_steps:
                check = rec.h.copy().hexdigest()
        del it
        files_sha = sv.plan_sha256(files, seeds["data_sampling"], e, a.num_workers,
                                   a.data_split_num, a.fetch_step, a.data_fraction)
        r = rec.record("dryrun", e, seeds["data_sampling"], seeds["dropout"], files_sha)
        n = r["n_jets"]
        q = int(rec.native[sv.NATIVE_QCD_FIRST:].sum())
        epochs.append({
            "epoch": e, "n_jets": n, "qcd_share": q / n, "seconds": round(time.time() - t0, 1),
            "sha256": r["sha256"], "files_sha256": files_sha,
            f"sha256_after_{check_steps * a.batch_size}": hashlib.sha256((files_sha + check).encode()).hexdigest()
            if check else None,
            "per_fetch": [[w, f, c[0], c[1]] for (w, f), c in sorted(per_fetch.items())],
            "max_fetch_id": max(f for _, f in per_fetch),
            "native_counts": rec.native.tolist(), "peak_anon_gb": mem.take()})
        print(f"epoch {e}: qcd share {q / n:.5f} over {n} jets, {len(per_fetch)} fetches, "
              f"{epochs[-1]['seconds']} s", flush=True)

    shares = np.array([x["qcd_share"] for x in epochs])
    p, n = shares.mean(), epochs[0]["n_jets"]
    fetch_sh = np.array([c[3] / c[2] for x in epochs for c in x["per_fetch"] if c[2] >= 10_000])
    fetch_n = np.array([c[2] for x in epochs for c in x["per_fetch"] if c[2] >= 10_000])
    summary = {
        "epochs": len(epochs), "jets_per_epoch": n, "mean_qcd_share": p,
        "sd_qcd_share": float(shares.std(ddof=1)) if len(shares) > 1 else None,
        "binomial_sd": math.sqrt(p * (1 - p) / n),
        "min": float(shares.min()), "max": float(shares.max()),
        "per_fetch": {"n_fetches_with_10k_jets": int(len(fetch_sh)),
                      "median_jets": float(np.median(fetch_n)) if len(fetch_n) else None,
                      "sd_share": float(fetch_sh.std(ddof=1)) if len(fetch_sh) > 1 else None,
                      "binomial_sd_at_median": math.sqrt(p * (1 - p) / np.median(fetch_n)) if len(fetch_n) else None,
                      "min": float(fetch_sh.min()) if len(fetch_sh) else None,
                      "max": float(fetch_sh.max()) if len(fetch_sh) else None},
        "v1_epoch_range_for_reference": [0.080, 0.175],
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "epochs": epochs}, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
