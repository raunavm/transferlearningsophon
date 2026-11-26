#!/usr/bin/env python3
"""Two CPU computations for the revision (ledger section 35; audit A1 and V7), on the cluster.

  confusion   the 188 x 188 confusion of the frozen linear probe on the class-balanced JetClass-II test set
              (903,922 jets): P(pred j | true i) of each 188-class run (mtx-l188-s1..s3, best70) and of each untrained
              network with BatchNorm recomputed (init-s1..s3, init_bn), and their means over runs. The predictions are
              perclass_recall.probe_pred's (the saved probe re-scored in float64, its correct count asserted against
              the fit). Writes <out>/confusion_jc2bal_probe.json.
  pool        the random-grouping pool (configs/labelmaps/rand_v2_selection.json: seeds 100-6699, share-matched draws
              of build_rand_control.share_draw): for every draw, the pairs of the 188 classes it merges that the
              17-class set also merges, with rule 3 and the partition's sha256 (build_v2_arms._pool_draw's digest),
              the rule-3 seeds and the five accepted partitions checked against the record. Needs numba
              (scripts/fast_compositions.py); the five accepted draws are also redrawn by the pure-Python code and must
              agree. Writes <out>/rand_pool_overlap.json.

    python3 experiments/STATS/rev35_jobs.py confusion --out /data/results/stats/rev35
    python3 experiments/STATS/rev35_jobs.py pool --out /data/results/stats/rev35 --workers 4
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = {"l188": [("mtx-l188-s1", "best70"), ("mtx-l188-s2", "best70"), ("mtx-l188-s3", "best70")],
        "init_bn": [("init-s1", "init_bn"), ("init-s2", "init_bn"), ("init-s3", "init_bn")]}
K = 188


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def confusion(out: pathlib.Path) -> dict:
    pr = _load("perclass_recall", "experiments/EVAL/perclass_recall.py")
    doc = {"script": "experiments/STATS/rev35_jobs.py confusion", "test_set": "class-balanced JetClass-II test",
           "definition": "P[i][j] = share of true-class-i test jets the probe labels j (argmax)", "runs": {}, "mean": {}}
    y0 = None
    for fam, ms in RUNS.items():
        mats = []
        for m, ck in ms:
            pred, y, rec = pr.probe_pred("jc2", m, ck)
            if y0 is None:
                y0 = y
            elif not np.array_equal(y, y0):
                raise SystemExit(f"FATAL: {m} is scored on other jets")
            n = np.bincount(y, minlength=K)
            if (n == 0).any():
                raise SystemExit(f"FATAL: classes {np.flatnonzero(n == 0).tolist()} have no test jet")
            c = np.zeros((K, K), np.int64)
            np.add.at(c, (y, pred), 1)
            p = c / n[:, None]
            mats.append(p)
            doc["runs"][m + "/" + ck] = {"accuracy": float((pred == y).mean()), "accuracy_stored": rec["accuracy_stored"],
                                         "P": np.round(p, 10).tolist()}
            print(f"  {fam} {m} {ck}: accuracy {(pred == y).mean():.5f}", flush=True)
        doc["mean"][fam] = np.round(np.mean(mats, 0), 10).tolist()
    doc["class_counts"] = np.bincount(y0, minlength=K).tolist()
    (out / "confusion_jc2bal_probe.json").write_text(json.dumps(doc))
    return doc


# ---------------------------------------------------------------- the random-grouping pool
def _pool_init():
    sys.path.insert(0, str(ROOT / "scripts"))
    global BV
    BV = _load("build_v2_arms", "scripts/build_v2_arms.py")
    BV._fast_init()


def _overlap(assign: dict, tgt: dict) -> tuple[int, int, int]:
    """(pairs of the 188 classes merged by the 17-class set, by the draw, by both)."""
    c2 = lambda n: n * (n - 1) // 2  # noqa: E731
    ns = sorted(tgt)
    both = collections.Counter((tgt[n], assign[n]) for n in ns)
    return (sum(c2(v) for v in collections.Counter(tgt[n] for n in ns).values()),
            sum(c2(v) for v in collections.Counter(assign[n] for n in ns).values()),
            sum(c2(v) for v in both.values()))


def _draw(seed: int, pure: bool = False):
    rows = BV.read_map()
    units = BV.brc.exact_share_units()
    tgt = BV.column(rows, BV.TARGET)
    with contextlib.redirect_stdout(io.StringIO()):
        assign = BV.brc.share_draw(rows, units, BV.TARGET, seed, 0, BV.brc.POOLS["resonant"])
    ok, dev = BV.rule3(assign, BV.realised_counts(), tgt)
    digest = hashlib.sha256(json.dumps([assign[n] for n in sorted(assign)]).encode()).hexdigest()
    m17, mr, both = _overlap(assign, tgt)
    return seed, bool(ok), float(dev), digest, m17, mr, both, list(BV.merge_vector(assign, BV.names_of(rows)))


def pool(out: pathlib.Path, workers: int) -> dict:
    rec = json.loads((ROOT / "configs/labelmaps/rand_v2_selection.json").read_text())
    seeds = list(range(rec["pool"]["first_seed"], rec["pool"]["last_seed"] + 1))
    if len(seeds) != rec["pool"]["draws"]:
        raise SystemExit("FATAL: the pool's seed range does not match its draw count")
    with multiprocessing.Pool(workers, initializer=_pool_init) as p:
        res = sorted(p.imap_unordered(_draw, seeds, chunksize=8))
    by = {r[0]: r for r in res}
    rule3 = sorted(s for s, r in by.items() if r[1])
    if rule3 != rec["rule3_seeds"]:
        raise SystemExit(f"FATAL: rule-3 seeds {rule3} differ from the record's")
    bad = [s for s in rule3 if by[s][3] != rec["partition_sha256"][str(s)]]
    if bad:
        raise SystemExit(f"FATAL: partitions of seeds {bad} differ from the record's sha256")
    # the five accepted, drawn again by the unmodified pure-Python code (no numba)
    sys.path.insert(0, str(ROOT / "scripts"))
    global BV
    BV = _load("build_v2_arms_pure", "scripts/build_v2_arms.py")
    for s in rec["accepted_seeds"]:
        if _draw(s)[3] != by[s][3]:
            raise SystemExit(f"FATAL: seed {s}: the numba and pure-Python draws differ")
    doc = {"script": "experiments/STATS/rev35_jobs.py pool", "selection_record": "configs/labelmaps/rand_v2_selection.json",
           "overlap": "pairs of the 188 classes merged by both the draw and the 17-class set, over the pairs the 17-class "
                      "set merges",
           "pool_draws": len(res), "rule3_draws": len(rule3), "accepted_seeds": rec["accepted_seeds"],
           "draws": {str(s): {"rule3": r[1], "max_rel_dev": r[2], "sha256": r[3], "merged_17": r[4], "merged_draw": r[5],
                              "merged_both": r[6], "overlap": r[6] / r[4], "probe_pair_merges": r[7]}
                     for s, r in by.items()}}
    (out / "rand_pool_overlap.json").write_text(json.dumps(doc))
    ov = lambda ss: [doc["draws"][str(s)]["overlap"] for s in ss]  # noqa: E731
    for what, ss in (("pool", seeds), ("rule 3", rule3), ("accepted", rec["accepted_seeds"])):
        v = np.array(ov(ss))
        print(f"  {what}: {v.size} draws, overlap min {v.min():.4f} median {np.median(v):.4f} max {v.max():.4f}")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("confusion", "pool"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    confusion(a.out) if a.what == "confusion" else pool(a.out, a.workers)
    print(f"REV35 {a.what.upper()} COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
