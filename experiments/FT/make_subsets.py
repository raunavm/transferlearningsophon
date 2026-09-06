#!/usr/bin/env python3
"""Materialise FIXED fine-tuning subsets: nested N-jet training files, one per
(N, seed), plus one validation file, for the data-efficiency legs.

WHY MATERIALISE
---------------
weaver's `--data-fraction` / `--file-fraction` re-draw the subset EVERY EPOCH
(train.py: "randomly selected for each epoch"), so they define a sampling
rate, not a dataset. A data-efficiency curve needs a dataset of exactly N jets
that is the same on every epoch, for every arm and every fine-tuning seed. So
the subsets are written once and every fine-tune reads the same file.

NESTED
------
For one seed, the N=1e4 file is the first 1e4 rows of the N=1e5 file, which is
the first 1e5 rows of the N=1e6 file -- the standard construction for a
data-scaling curve: a larger budget strictly adds data.

jc2  JetClass-II parquet (the in-domain recovery leg). Files are drawn per
     family in proportion to the family's file count (train: Res2P 200,
     Res34P 860, QCD 280), the study's selection (200<pt<2500, 20<msd<500) is
     applied, and a FIXED FRACTION of each file's selected rows is kept, so the
     pool's family mix is the natural post-selection stream's. A fixed COUNT
     per file would over-weight QCD, whose selection efficiency is 51.8%
     against 86-91% for resonant jets (docs/GROUND_TRUTH.md). Rows are copied
     whole, so the schema is the release's own.
jc1  JetClass-I ROOT (the pileup-shift leg; 10 classes, train_100M). Balanced,
     N/10 per class, from `files_per_class` random files per class per seed,
     written as parquet -- weaver reads .parquet with ak.from_parquet and the
     branch names are unchanged by the round trip.

No reweighting is baked in: the fine-tuning configs under configs/finetune/
carry no `weights:` block, so a subset is used at its natural composition.

Outputs in --out:  train_N<size>_s<seed>.parquet, val.parquet, manifest.json,
and DONE (written last, so a consumer can wait on it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

JC1_CLASSES = ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
               "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu"]
PT_LO, PT_HI, MSD_LO, MSD_HI = 200.0, 2500.0, 20.0, 500.0
SIZES = [10_000, 100_000, 1_000_000]
SEEDS = [1, 2, 3]
VAL_SEED = 0
DATE_SALT = 20260905


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    """One generator per (seed, purpose); the salt keeps it distinct from every
    other RNG in the project (four training streams, probe splits)."""
    return np.random.default_rng([int(seed), {"train": 1, "val": 2}[purpose], DATE_SALT])


def family_of(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def group_by_family(files: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(files):
        out.setdefault(family_of(f), []).append(f)
    return out


def choose_files(families: dict[str, list[str]], n_files: int,
                 rng: np.random.Generator) -> list[str]:
    """`n_files` files, per family in proportion to the family's file count,
    at least one per family, uniformly within a family."""
    total = sum(len(v) for v in families.values())
    chosen: list[str] = []
    for fam in sorted(families):
        files = families[fam]
        k = min(len(files), max(1, int(round(n_files * len(files) / total))))
        idx = np.sort(rng.choice(len(files), size=k, replace=False))
        chosen += [files[i] for i in idx]
    return chosen


def select_jc2(table: pa.Table) -> pa.Table:
    pt, m = table.column("jet_pt"), table.column("jet_sdmass")
    keep = pc.and_(pc.and_(pc.greater(pt, PT_LO), pc.less(pt, PT_HI)),
                   pc.and_(pc.greater(m, MSD_LO), pc.less(m, MSD_HI)))
    return table.filter(keep)


def take_fraction(table: pa.Table, frac: float, rng: np.random.Generator) -> pa.Table:
    k = int(round(table.num_rows * frac))
    idx = np.sort(rng.choice(table.num_rows, size=k, replace=False))
    return table.take(pa.array(idx))


def nested_prefixes(table: pa.Table, sizes: list[int],
                    rng: np.random.Generator) -> dict[int, pa.Table]:
    """Shuffle once; every size is a prefix of the same shuffle."""
    if table.num_rows < max(sizes):
        raise SystemExit(f"FATAL: pool has {table.num_rows:,} rows, need "
                         f"{max(sizes):,}. Raise --n-files / --take-fraction.")
    shuffled = table.take(pa.array(rng.permutation(table.num_rows)))
    return {n: shuffled.slice(0, n) for n in sizes}


def build_jc2(train_files, val_files, out: pathlib.Path, sizes, seeds,
              n_files: int, take: float, val_size: int, n_val_files: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": "jc2", "sizes": sizes, "seeds": seeds, "n_files": n_files,
                "take_fraction": take, "selection": [PT_LO, PT_HI, MSD_LO, MSD_HI],
                "per_seed": {}, "val": {}}
    fams = group_by_family(train_files)
    for seed in seeds:
        rng = rng_for(seed, "train")
        files = choose_files(fams, n_files, rng)
        parts, per_family = [], {}
        for f in files:
            t = take_fraction(select_jc2(pq.read_table(f)), take, rng)
            per_family[family_of(f)] = per_family.get(family_of(f), 0) + t.num_rows
            parts.append(t)
            print(f"  seed {seed}: {os.path.basename(f)} -> {t.num_rows:,} rows", flush=True)
        pool = pa.concat_tables(parts)
        subs = nested_prefixes(pool, sizes, rng)
        for n, t in subs.items():
            pq.write_table(t, out / f"train_N{n}_s{seed}.parquet")
        manifest["per_seed"][str(seed)] = {"files": files, "n_files_used": len(files),
                                           "pool_rows": pool.num_rows,
                                           "pool_rows_per_family": per_family}
        print(f"seed {seed}: pool {pool.num_rows:,} rows {per_family}", flush=True)

    rng = rng_for(VAL_SEED, "val")
    vfiles = choose_files(group_by_family(val_files), n_val_files, rng)
    parts = [take_fraction(select_jc2(pq.read_table(f)), take, rng) for f in vfiles]
    vpool = pa.concat_tables(parts)
    val = nested_prefixes(vpool, [val_size], rng)[val_size]
    pq.write_table(val, out / "val.parquet")
    manifest["val"] = {"files": vfiles, "pool_rows": vpool.num_rows, "rows": val.num_rows}
    return manifest


def read_root(path: str):
    """All branches of the single TTree in a JetClass-I file, as awkward."""
    import uproot
    with uproot.open(path) as f:
        trees = sorted({k.split(";")[0] for k, v in f.classnames().items() if v == "TTree"})
        if len(trees) != 1:
            raise SystemExit(f"FATAL: {path} has {len(trees)} TTrees: {trees}")
        return f[trees[0]].arrays()


def build_jc1(train_dir: str, val_dir: str, out: pathlib.Path, sizes, seeds,
              files_per_class: int, val_per_class: int, classes=JC1_CLASSES) -> dict:
    import awkward as ak
    out.mkdir(parents=True, exist_ok=True)
    if max(sizes) % len(classes):
        raise SystemExit(f"FATAL: max size {max(sizes)} is not divisible by {len(classes)} classes")
    per = max(sizes) // len(classes)
    manifest = {"mode": "jc1", "sizes": sizes, "seeds": seeds, "classes": classes,
                "files_per_class": files_per_class, "per_class_rows": per,
                "per_seed": {}, "val": {}}

    def pick(cls_dir: str, cls: str, k: int, rng) -> list[str]:
        files = sorted(glob.glob(os.path.join(cls_dir, f"{cls}_*.root")))
        if len(files) < k:
            raise SystemExit(f"FATAL: {cls}: {len(files)} files in {cls_dir}, need {k}")
        idx = np.sort(rng.choice(len(files), size=k, replace=False))
        return [files[i] for i in idx]

    for seed in seeds:
        rng = rng_for(seed, "train")
        parts, used = [], {}
        for cls in classes:
            files = pick(train_dir, cls, files_per_class, rng)
            arr = ak.concatenate([read_root(f) for f in files])
            if len(arr) < per:
                raise SystemExit(f"FATAL: {cls}: {len(arr):,} jets in {files}, need {per:,}")
            idx = np.sort(rng.choice(len(arr), size=per, replace=False))
            parts.append(arr[idx])
            used[cls] = files
            print(f"  seed {seed}: {cls}: {len(arr):,} jets -> {per:,}", flush=True)
        pool = ak.concatenate(parts)
        pool = pool[rng.permutation(len(pool))]
        for n in sizes:
            ak.to_parquet(pool[:n], out / f"train_N{n}_s{seed}.parquet")
        manifest["per_seed"][str(seed)] = {"files": used, "pool_rows": len(pool)}

    rng = rng_for(VAL_SEED, "val")
    parts, used = [], {}
    for cls in classes:
        files = pick(val_dir, cls, 1, rng)
        arr = read_root(files[0])
        if len(arr) < val_per_class:
            raise SystemExit(f"FATAL: {cls}: {len(arr):,} val jets, need {val_per_class:,}")
        idx = np.sort(rng.choice(len(arr), size=val_per_class, replace=False))
        parts.append(arr[idx])
        used[cls] = files
    val = ak.concatenate(parts)
    val = val[rng.permutation(len(val))]
    ak.to_parquet(val, out / "val.parquet")
    manifest["val"] = {"files": used, "rows": len(val)}
    return manifest


def finish(out: pathlib.Path, manifest: dict) -> None:
    manifest["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["outputs"] = {p.name: pq.read_metadata(p).num_rows
                           for p in sorted(out.glob("*.parquet"))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "DONE").write_text(manifest["written_utc"] + "\n")
    print(json.dumps(manifest["outputs"], indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("jc2")
    a.add_argument("--train-files", nargs="+", required=True)
    a.add_argument("--val-files", nargs="+", required=True)
    a.add_argument("--n-files", type=int, default=60,
                   help="train files read per seed; ~82k selected jets each")
    a.add_argument("--take-fraction", type=float, default=0.30,
                   help="fraction of each file's SELECTED rows kept in the pool")
    a.add_argument("--val-size", type=int, default=200_000)
    a.add_argument("--n-val-files", type=int, default=12)
    b = sub.add_parser("jc1")
    b.add_argument("--train-dir", required=True)
    b.add_argument("--val-dir", required=True)
    b.add_argument("--files-per-class", type=int, default=2)
    b.add_argument("--val-per-class", type=int, default=20_000)
    for p in (a, b):
        p.add_argument("--out", required=True)
        p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
        p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args(argv)

    sizes = sorted(args.sizes)
    out = pathlib.Path(args.out)
    if (out / "DONE").exists():
        print(f"{out}/DONE exists; nothing to do")
        return 0
    if args.mode == "jc2":
        m = build_jc2(args.train_files, args.val_files, out, sizes, args.seeds,
                      args.n_files, args.take_fraction, args.val_size, args.n_val_files)
    else:
        m = build_jc1(args.train_dir, args.val_dir, out, sizes, args.seeds,
                      args.files_per_class, args.val_per_class)
    finish(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
