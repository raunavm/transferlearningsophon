#!/usr/bin/env python3
"""Class-balanced JetClass-II sets for the probes and fine-tuning (plan 3.4, PI 2026-10-30).

The launched JetClass-II subsets keep the natural class mix: the rarest class has 11 training jets
at 1e6, 28 classes are absent at 1e3, and the test prefix holds 27 jets of QCD bcs. JetClass itself
has equal numbers of jets per class. These sets do the same, drawn from the files no pretraining run
trained or validated on, with the pretraining selection (200 < pT < 2500 GeV, 20 < mSD < 500 GeV):

  test.parquet          from the fine-tuning test files (JC2_ROLES["ft_test"]): every class up to
                        TEST_PER_CLASS jets, all of a class's jets where it has fewer
  train_N<N>_s1.parquet from the held-out files (JC2_ROLES["ft_heldout"]), N = 1e3 ... 1e6: an
                        equal quota per class (N // 188, one more for N % 188 classes drawn at
                        random), all of a class's jets where it has fewer; nested (each class's
                        jets in one fixed random order, every size takes a prefix of it)
  val.parquet           from the held-out jets not in train_N1000000: up to VAL_PER_CLASS per class
  pairs_train.parquet,  for each probe pair (linear_probe_v2.PAIRS and QCD_PAIRS), up to
  pairs_test.parquet    PAIR_TRAIN_PER_SIDE / PAIR_TEST_PER_SIDE jets per side, drawn uniformly
                        from the side's jets (the natural mix within a side), from the held-out and
                        the test files; pairs_index.npz gives each pair's rows

A class short of its quota keeps all its jets; the census (census.json) records every class's
count in each role, and the manifest each output's class coverage and sha256. Every output row
carries jc2bal_src = file index * 10^7 + row in its source file, so any row can be traced back.
Rows are written in a random order (one permutation per output).

    python3 experiments/FT/stage_jc2_balanced.py --out /data/finetune/jc2_bal --workers 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
N_NATIVE = 188
SIZES = (1_000, 10_000, 100_000, 1_000_000)
TEST_PER_CLASS = 5_000
VAL_PER_CLASS = 109                   # 188 x 109 = 20,492 at most
PAIR_TRAIN_PER_SIDE = 100_000
PAIR_TEST_PER_SIDE = 200_000
SEED = 20261031
SRC_MULT = 10_000_000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("make_subsets", "experiments/FT/make_subsets.py")


def rng(purpose: str) -> np.random.Generator:
    return np.random.default_rng([SEED, {"test": 1, "train": 2, "val": 3, "pairs_train": 4, "pairs_test": 5,
                                         "extra": 6, "perm": 7}[purpose]])


def census_file(path: str) -> tuple:
    """(path, selected row indices, their native labels) of one source file."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["jet_label", "jet_pt", "jet_sdmass"])
    pt, m = t.column("jet_pt").to_numpy(), t.column("jet_sdmass").to_numpy()
    keep = (pt > MS.PT_LO) & (pt < MS.PT_HI) & (m > MS.MSD_LO) & (m < MS.MSD_HI)
    rows = np.flatnonzero(keep).astype(np.int64)
    return path, rows, t.column("jet_label").to_numpy()[rows].astype(np.int16)


def quotas(n: int, avail: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Equal jets per class: n // 188 each, one more for n % 188 classes drawn at random; a class
    with fewer jets keeps all of them (the shortfall is not moved to other classes)."""
    q = np.full(N_NATIVE, n // N_NATIVE, dtype=np.int64)
    q[r.permutation(N_NATIVE)[: n % N_NATIVE]] += 1
    return np.minimum(q, avail)


def per_class_order(src: np.ndarray, lab: np.ndarray, r: np.random.Generator) -> list:
    """Each class's source ids in one fixed random order."""
    return [src[lab == c][r.permutation(int((lab == c).sum()))] for c in range(N_NATIVE)]


def choose(order: list, q: np.ndarray) -> np.ndarray:
    return np.concatenate([order[c][: q[c]] for c in range(N_NATIVE)]) if q.sum() else np.zeros(0, np.int64)


def pair_rows(src: np.ndarray, lab: np.ndarray, groups: dict, cap: int, r: np.random.Generator) -> dict:
    """{pair: (signal ids, background ids)}, up to `cap` per side, uniform within a side."""
    out = {}
    for pair, (sig, bkg) in groups.items():
        sides = []
        for cls in (sig, bkg):
            ids = src[np.isin(lab, cls)]
            sides.append(np.sort(ids[r.permutation(ids.size)[:cap]]))
        out[pair] = tuple(sides)
    return out


def coverage(lab) -> dict:
    return MS.class_coverage(lab)


def write(out: pathlib.Path, name: str, ids: np.ndarray, files: list, perm_rng) -> dict:
    """Rows `ids` (file index * SRC_MULT + row) of the source files, in a random order, as one parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = np.asarray(ids, np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    fidx, rows = sorted_ids // SRC_MULT, sorted_ids % SRC_MULT
    parts = []
    for f in np.unique(fidx):
        t = pq.read_table(files[f])
        sel = rows[fidx == f]
        tt = t.take(pa.array(sel))
        parts.append(tt.append_column("jc2bal_src", pa.array(f * SRC_MULT + sel, pa.int64())))
    table = pa.concat_tables(parts, promote_options="default") if parts else None
    perm = perm_rng.permutation(table.num_rows)
    table = table.take(pa.array(perm))
    tmp = out / f"{name}.tmp"
    pq.write_table(table, tmp, row_group_size=50_000)
    os.replace(tmp, out / name)
    lab = table.column("jet_label").to_numpy()
    return {"rows": int(table.num_rows), "coverage": coverage(lab),
            "src_sha256": hashlib.sha256(table.column("jc2bal_src").to_numpy().tobytes()).hexdigest()}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def plan(census: dict, groups: dict) -> dict:
    """Every output's source ids, from the census alone (no data read): testable on its own."""
    out = {}
    te_src, te_lab = census["ft_test"]
    he_src, he_lab = census["ft_heldout"]
    te_avail = np.bincount(te_lab, minlength=N_NATIVE)
    he_avail = np.bincount(he_lab, minlength=N_NATIVE)
    out["test.parquet"] = choose(per_class_order(te_src, te_lab, rng("test")),
                                 np.minimum(TEST_PER_CLASS, te_avail))
    order = per_class_order(he_src, he_lab, rng("train"))
    big = quotas(max(SIZES), he_avail, rng("extra"))
    prev = None
    for n in SIZES:
        q = np.minimum(quotas(n, he_avail, rng("extra")), big)
        if prev is not None:
            q = np.maximum(q, prev)            # nested: a larger size never drops a jet
        out[f"train_N{n}_s1.parquet"] = choose(order, q)
        prev = q
    rest = [order[c][big[c]:] for c in range(N_NATIVE)]
    out["val.parquet"] = choose(rest, np.minimum(VAL_PER_CLASS, [r.size for r in rest]))
    tr_pairs = pair_rows(he_src, he_lab, groups, PAIR_TRAIN_PER_SIDE, rng("pairs_train"))
    te_pairs = pair_rows(te_src, te_lab, groups, PAIR_TEST_PER_SIDE, rng("pairs_test"))
    out["pairs_train.parquet"] = np.unique(np.concatenate([np.concatenate(v) for v in tr_pairs.values()]))
    out["pairs_test.parquet"] = np.unique(np.concatenate([np.concatenate(v) for v in te_pairs.values()]))
    out["_pairs"] = {"train": tr_pairs, "test": te_pairs}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default="/jc2/jet_data")
    a = ap.parse_args(argv)
    if (a.out / "DONE").exists():
        raise SystemExit(f"FATAL: {a.out} is complete; staged data is never rebuilt")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    lp = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    groups = lp.native_pair_groups(names)

    files = {role: MS.jc2_role_files(role, a.root) for role in ("ft_heldout", "ft_test")}
    all_files = files["ft_heldout"] + files["ft_test"]
    missing = [f for f in all_files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} source files missing, e.g. {missing[:3]}")
    with ProcessPoolExecutor(a.workers) as ex:
        res = {p: (r, l) for p, r, l in ex.map(census_file, all_files, chunksize=4)}
    census, counts = {}, {}
    for role, fl in files.items():
        src = np.concatenate([all_files.index(f) * SRC_MULT + res[f][0] for f in fl])
        lab = np.concatenate([res[f][1] for f in fl]).astype(np.int64)
        census[role] = (src, lab)
        counts[role] = np.bincount(lab, minlength=N_NATIVE).tolist()
    (a.out / "census.json").write_text(json.dumps(
        {"selection": [MS.PT_LO, MS.PT_HI, MS.MSD_LO, MS.MSD_HI], "files": files,
         "counts": counts, "class_names": [names[i] for i in range(N_NATIVE)],
         "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"census: {sum(len(v[1]) for v in census.values()):,} selected jets ({time.time() - t0:.0f} s)", flush=True)

    p = plan(census, groups)
    pairs = p.pop("_pairs")
    manifest = {"script": "experiments/FT/stage_jc2_balanced.py", "seed": SEED, "sizes": list(SIZES),
                "test_per_class": TEST_PER_CLASS, "val_per_class": VAL_PER_CLASS,
                "pair_train_per_side": PAIR_TRAIN_PER_SIDE, "pair_test_per_side": PAIR_TEST_PER_SIDE,
                "selection": [MS.PT_LO, MS.PT_HI, MS.MSD_LO, MS.MSD_HI], "outputs": {}}
    for name, ids in p.items():
        manifest["outputs"][name] = write(a.out, name, ids, all_files, rng("perm"))
        print(f"  {name}: {manifest['outputs'][name]['rows']:,} rows ({time.time() - t0:.0f} s)", flush=True)
    # each pair's rows, as indices into its parquet's (permuted) rows, via jc2bal_src
    import pyarrow.parquet as pq
    idx = {}
    for split in ("train", "test"):
        src = pq.read_table(a.out / f"pairs_{split}.parquet", columns=["jc2bal_src"]).column("jc2bal_src").to_numpy()
        pos = {int(s): i for i, s in enumerate(src)}
        for pair, (sig, bkg) in pairs[split].items():
            idx[f"{split}/{pair}/signal"] = np.array([pos[int(s)] for s in sig], np.int64)
            idx[f"{split}/{pair}/background"] = np.array([pos[int(s)] for s in bkg], np.int64)
    np.savez(a.out / "pairs_index.npz", **idx)
    manifest["pairs"] = {k: int(v.size) for k, v in idx.items()}
    manifest["sha256"] = {n: sha256(a.out / n) for n in list(p) + ["pairs_index.npz", "census.json"]}
    manifest["seconds"] = round(time.time() - t0, 1)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"DONE ({manifest['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
