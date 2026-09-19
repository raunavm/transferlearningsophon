#!/usr/bin/env python3
"""Add an observer to feature caches that were extracted without it.

WHY THIS EXISTS
---------------
Every frozen-feature cache under /data/results/eval/<run>/features_e79 was
written by extract_features.py with its DEFAULT observers -- jet_pt, jet_sdmass,
jet_eta, jet_nparticles -- from configs/data/JetClassII_base.yaml, whose
`observers:` block does not list genjet_sdmass at all. The jet-mass analysis
needs the generator-level groomed mass for every jet of every model it compares.

The caches are row-aligned by construction (one file list, one order, no
shuffling), so the missing column is a property of the JETS, not of any model:
it is read ONCE here, with no network and no checkpoint, instead of
re-extracting every model for the sake of one float per jet.

WHY THE RESULT CAN BE TRUSTED TO LINE UP
----------------------------------------
It is checked, not assumed, and checked BEFORE anything is written. For every
--align-with cache: label188_sha256 in its manifest must equal the digest of the
labels read here, and every observer the two have in common (jet_pt, jet_sdmass,
...) must be BIT-IDENTICAL. The label digest alone would pass two orderings that
permute jets within a class; two million float32 jet_pt values would not.

The loader arguments below are extract_features.py's, deliberately unchanged --
for_training=False (no shuffle, no resampling), fetch_by_files, one worker --
and configs/data/JetClassII_massreg.yaml differs from the base config by the one
added observer, so selection and order are the same. If either ever drifts, the
check above fails the job; it cannot produce a misaligned file quietly.

genjet_sdmass is a hard 0.0f for an unmatched jet (docs/GROUND_TRUTH.md). It is
stored as read. MASK ON > 0 before using it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OBSERVERS = ["jet_pt", "jet_sdmass", "jet_eta", "jet_nparticles", "genjet_sdmass"]


def misalignments(L, obs: dict, ref_dir: pathlib.Path) -> list[str]:
    """Every way the rows read here disagree with the cache in `ref_dir`."""
    man = json.loads((ref_dir / "extract_manifest.json").read_text())
    bad = []
    sha = hashlib.sha256(L.tobytes()).hexdigest()
    if man.get("label188_sha256") != sha:
        bad.append(f"label188_sha256 {str(man.get('label188_sha256'))[:16]} != "
                   f"{sha[:16]} (cache n_jets={man.get('n_jets')}, "
                   f"stride={man.get('stride', 1)}; here n_jets={L.shape[0]})")
    if (ref_dir / "observers.npz").exists():
        z = np.load(ref_dir / "observers.npz")
        for k in sorted(set(z.files) & set(obs)):
            if not np.array_equal(z[k], obs[k]):
                bad.append(f"observer {k} is not bit-identical")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-config",
                    default=str(REPO / "configs/data/JetClassII_massreg.yaml"))
    ap.add_argument("--observers", nargs="+", default=list(OBSERVERS))
    ap.add_argument("--align-with", nargs="+", required=True, metavar="CACHE_DIR",
                    help="existing feature caches these rows must match")
    ap.add_argument("--max-jets", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--fetch-step", type=int, default=1)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if (out / "observers_manifest.json").exists():
        raise SystemExit(f"FATAL: {out} already holds a finished observer file. "
                         f"Use a new --out.")
    refs = [pathlib.Path(d) for d in args.align_with]
    absent = [str(d) for d in refs if not (d / "extract_manifest.json").exists()]
    if absent:
        raise SystemExit(f"FATAL: no extract_manifest.json in {absent}. Every "
                         f"cache named by --align-with must exist.")

    import torch
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset

    data_config = DataConfig.load(args.data_config, load_observers=True)
    ds = SimpleIterDataset({"_": list(args.data_test)}, args.data_config,
                           for_training=False, fetch_by_files=True,
                           fetch_step=args.fetch_step, name="extract")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, persistent_workers=False)

    labels, obs = [], {k: [] for k in args.observers}
    n, t0 = 0, time.time()
    for _, y, Z in loader:
        labels.append(y[data_config.label_names[0]].cpu().numpy().astype(np.int16))
        for k in args.observers:
            if k in Z:
                obs[k].append(np.asarray(Z[k]).astype(np.float32))
        n += labels[-1].shape[0]
        if n % (args.batch_size * 200) < args.batch_size:
            print(f"  {n:,} jets  {n/(time.time()-t0):.0f} jets/s", flush=True)
        if args.max_jets and n >= args.max_jets:
            break

    missing = [k for k, v in obs.items() if not v]
    if missing:
        print(f"FATAL: requested observer(s) {missing} never appeared in any "
              f"batch. {args.data_config} must list them under `observers:`.",
              file=sys.stderr)
        return 4
    cut = args.max_jets or None
    L = np.concatenate(labels)[:cut]
    obs = {k: np.concatenate(v)[:cut] for k, v in obs.items()}

    failed = False
    for d in refs:
        bad = misalignments(L, obs, d)
        print(f"  {'FAIL' if bad else 'ok  '} {d}")
        for b in bad:
            failed = True
            print(f"       {b}", file=sys.stderr)
    if failed:
        print("FATAL: the rows read here are not the rows of the caches above. "
              "Nothing written.", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "label188.npy", L)
    np.savez(out / "observers.npz", **obs)
    matched = int((obs["genjet_sdmass"] > 0).sum()) if "genjet_sdmass" in obs else None
    (out / "observers_manifest.json").write_text(json.dumps({
        "n_jets": int(L.shape[0]),
        "label188_sha256": hashlib.sha256(L.tobytes()).hexdigest(),
        "observers": sorted(obs),
        "genjet_sdmass_note": "0.0 means unmatched; mask on > 0",
        "n_genjet_sdmass_matched": matched,
        "data_config": args.data_config,
        "data_config_sha256": hashlib.sha256(
            pathlib.Path(args.data_config).read_bytes()).hexdigest(),
        "n_test_files": len(args.data_test),
        "aligned_with": [str(d) for d in refs],
    }, indent=2))
    print(f"\nwrote {L.shape[0]:,} rows x {sorted(obs)} to {out}; "
          f"bit-identical to {len(refs)} caches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
