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


FAMILIES = ("two-prong", "three/four-prong", "QCD")
FAMILY_EDGES = (0, 15, 161)                      # native jet_label where each family starts


def family_counts(native) -> list:
    return [int(native[0:15].sum()), int(native[15:161].sum()), int(native[161:].sum())]


def _stats(shares, n, copy_factor) -> dict:
    """Mean and SD of a family share over epochs, against the binomial SD of n jets
    with weaver's repeated rows counted (variance x copy factor) and without."""
    import numpy as np
    shares = np.asarray(shares, float)
    p = float(shares.mean())
    b = math.sqrt(p * (1 - p) / n)
    return {"mean": p, "sd": float(shares.std(ddof=1)) if len(shares) > 1 else None,
            "binomial_sd": b, "binomial_sd_with_copies": b * math.sqrt(copy_factor),
            "sd_over_binomial_with_copies": float(shares.std(ddof=1)) / (b * math.sqrt(copy_factor))
            if len(shares) > 1 else None,
            "min": float(shares.min()), "max": float(shares.max())}


def summarise(epochs) -> dict:
    """Epoch-level and last-20% family shares, and the per-fetch family mix (all
    fetches, and those read in the last 20% of an epoch), each against binomial
    with copies: Pearson chi2 = sum over fetches and families of
    (count - n p)^2 / (n p c), two degrees of freedom per fetch."""
    import numpy as np
    c = float(np.mean([x["copy_factor"] for x in epochs]))
    out = {"copy_factor": c,
           "epoch": {f: _stats([x["family_shares"][j] for x in epochs], epochs[0]["n_jets"], c)
                     for j, f in enumerate(FAMILIES)},
           "last20": {f: _stats([x["last20"]["family_shares"][j] for x in epochs],
                                int(np.mean([x["last20"]["n_jets"] for x in epochs])), c)
                      for j, f in enumerate(FAMILIES)}}
    for key, sel in (("per_fetch", lambda r: True), ("per_fetch_last20", lambda r: r[6] == 1)):
        chi, dof, sizes = 0.0, 0, []
        for x in epochs:
            p = np.array(x["family_shares"])
            rows = np.array([r for r in x["per_fetch"] if sel(r) and r[2] >= 1000], float)
            if not len(rows):
                continue
            n = rows[:, 2:3]
            chi += float((((rows[:, 3:6] - n * p) ** 2) / (n * p * c)).sum())
            dof += 2 * len(rows)                 # three shares that sum to one
            sizes += rows[:, 2].tolist()
        out[key] = {"fetches": len(sizes), "median_jets": float(np.median(sizes)) if sizes else None,
                    "chi2_per_dof_with_copies": chi / dof if dof else None}
    return out


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
    last_from = int(0.8 * steps)                   # the epoch's last 20% of batches
    for e in range(a.epochs):
        ds.set_epoch(e)
        rec = pv.StreamRecord()
        per_fetch, last_fetches, rids = {}, set(), []
        check = None
        t0 = time.time()
        it = iter(DataLoader(ds, batch_size=None, num_workers=a.num_workers,
                             multiprocessing_context="fork" if a.num_workers else None))
        for i in range(steps):
            _, _, Z = next(it)
            rec.update(Z, last=i >= last_from)
            rids.append(Z["_rowid"].numpy())
            w = i % max(a.num_workers, 1)          # DataLoader takes workers round robin
            fam = np.searchsorted(FAMILY_EDGES, Z["_jet_label"].numpy(), side="right") - 1
            fetch = Z["_fetch"].numpy()
            for fid in np.unique(fetch):
                m = fetch == fid
                c = per_fetch.setdefault((w, int(fid)), [0, 0, 0, 0])
                c[0] += int(m.sum())
                for j in range(3):
                    c[1 + j] += int((fam[m] == j).sum())
                if i >= last_from:
                    last_fetches.add((w, int(fid)))
            if i + 1 == check_steps:
                check = rec.h.copy().hexdigest()
        del it
        files_sha = sv.plan_sha256(files, seeds["data_sampling"], e, a.num_workers,
                                   a.data_split_num, a.fetch_step, a.data_fraction)
        r = rec.record("dryrun", e, seeds["data_sampling"], seeds["dropout"], files_sha)
        n = r["n_jets"]
        _, copies = np.unique(np.concatenate(rids), return_counts=True)
        fam_all = family_counts(rec.native)
        fam_last = family_counts(rec.native_last)
        epochs.append({
            "epoch": e, "n_jets": n, "qcd_share": fam_all[2] / n, "seconds": round(time.time() - t0, 1),
            "sha256": r["sha256"], "files_sha256": files_sha,
            f"sha256_after_{check_steps * a.batch_size}": hashlib.sha256((files_sha + check).encode()).hexdigest()
            if check else None,
            "distinct_jets": int(len(copies)), "copy_factor": float((copies ** 2).sum() / copies.sum()),
            "family_shares": [x / n for x in fam_all],
            "last20": {"n_jets": int(sum(fam_last)), "family_shares": [x / max(sum(fam_last), 1) for x in fam_last],
                       "native_counts": rec.native_last.tolist()},
            "per_fetch": [[w, f, *c, int((w, f) in last_fetches)] for (w, f), c in sorted(per_fetch.items())],
            "max_fetch_id": max(f for _, f in per_fetch),
            "native_counts": rec.native.tolist(), "peak_anon_gb": mem.take()})
        print(f"epoch {e}: qcd share {fam_all[2] / n:.5f} over {n} jets (last 20%: "
              f"{epochs[-1]['last20']['family_shares'][2]:.5f}), {len(per_fetch)} fetches, "
              f"copy factor {epochs[-1]['copy_factor']:.3f}, {epochs[-1]['seconds']} s", flush=True)

    out_summary = summarise(epochs)
    summary = {"epochs": len(epochs), "jets_per_epoch": epochs[0]["n_jets"],
               "mean_qcd_share": out_summary["epoch"]["QCD"]["mean"],
               "sd_qcd_share": out_summary["epoch"]["QCD"]["sd"],
               "binomial_sd": out_summary["epoch"]["QCD"]["binomial_sd"],
               **out_summary, "v1_epoch_range_for_reference": [0.080, 0.175]}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "epochs": epochs}, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
