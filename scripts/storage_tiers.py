#!/usr/bin/env python3
"""The approved /data deletion of 2026-11-23 (PI 2026-11-23T01:56:47Z, LEDGER ST-5; list docs/STORAGE_DELETION_1122.md):
tier 1 (quarantined data) and tier 2 (retired July-September work), in three steps run on the cluster:

    manifest  every file of both tiers: path, size, sha256, how it could be regenerated, whether its network is archived;
              one CSV per tier; refuses to overwrite one
    archive   the only copy of every trained network in the tiers, if intact (loads, every tensor finite), with its
              config and log, copied under OUT with a SHA256SUMS and intact.tsv, then one tar.gz to fetch to the laptop
    delete    exactly the manifest's rows whose file re-hashes to its manifest value, after checking the manifest's own
              sha256; empty folders left behind under the tier roots are removed; every action is logged

    python3 scripts/storage_tiers.py manifest --out /data/results/storage_1123
    python3 scripts/storage_tiers.py archive --out /data/results/storage_1123/archive
    python3 scripts/storage_tiers.py delete --manifest /data/results/storage_1123/tier1.csv --sha256 <hex>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import pathlib
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

R = pathlib.Path("/data/results")
# (root, which files go): "all" or a suffix; g1 keeps its logs, yaml, json and tfevents (the record of how the
# pretraining rate was chosen) and loses only its checkpoints
TIERS = {1: ((R / "eval/v2_linprobe_quarantine", "all"), (R / "mtx_v2_quarantine", "all")),
         2: ((R / "e8", "all"), (R / "e2", "all"), (R / "ft/lrprobe", "all"), (R / "ft/lrprobe2", "all"),
             (R / "ft/lrprobe3", "all"), (R / "g1", ".pt"), (R / "aoj", "all"), (R / "ft/smoke", "all"))}
# tier 3 (PI 2026-11-23, LEDGER ST-6): superseded render folders, keeping the latest of each kind and every render that
# RUNS.csv or STATE.md cites (selected 2026-11-23 07:15 UTC), and the unpacked copy of the network archive (archive.tgz
# stays); test-output folders stay
TIER3_RENDERS = ("a059066-gallery", "5fa878e-gallery", "c0ce260-gallery", "31c3393-gallery", "6032982-gallery", "2a8a898-gallery", 
                 "b64ce89-gallery", "7fff1d2-gallery", "09fd1fc-gallery", "30cd14b-gallery", "cac7955-gallery", "00b8362-gallery", 
                 "a826f1f-gallery", "64c079f-gallery", "087d030-gallery", "61bf8f7-gallery", "3700e1c-gallery", "2116899-gallery", 
                 "f57fa9b-gallery", "6489644-gallery", "eba15a8-gallery", "e97d22f-gallery", "af2b888-gallery", "dfb4839-gallery", 
                 "9e4baa0-gallery", "dc20f78-gallery", "1037b5d-gallery", "15c825d-gallery", "e5abffb-gallery", "fcfb35b-gallery", 
                 "bc95b1e-gallery", "12f5b47-gallery", "6022754-gallery", "e5b8e40-gallery", "13d5492-gallery", "e2d427c-gallery", 
                 "64b3ab4-gallery", "6d1c16f-gallery", "cc6cac7-gallery", "40520a2-gallery", "68bf1b3-gallery", "483c844-gallery", 
                 "aec91e7", "b4292c7", "f611940", "32b02ee", "bde0c92", "45ac82c", 
                 "f54dd65", "6b196e2", "4b96161", "1b3d09b", "7188348", "58ba8e2", 
                 "e88fab9", "0b1f7b0", "af2b888", "6022754", "12f5b47", "3018e0e", 
                 "3b31a99", "4a80870-preview", "b7608ef-preview", "561084a-bookpreview", "2894691-bookpreview", "6c83431-bookpreview", 
                 "ed58de6-bookpreview", "1b23711-bookpreview", "f3faeb5-bookpreview", "6b6a652-bookpreview", "7fcc1a1-bookpreview", "e8e96cb-releasepreview", 
                 )
TIERS[3] = tuple((R / "render" / d, "all") for d in TIER3_RENDERS) + ((R / "storage_1123/archive", "all"),)
REGEN = {"render/": "a render of its git commit", "storage_1123/archive": "archive.tgz (kept) and the laptop copy",
         "eval/v2_linprobe_quarantine": "re-extraction (superseded by eval/v2_linprobe)",
         "aoj": "re-extraction", "e8": "retraining or re-extraction"}
# a trained network: its weights file(s); the files beside it that hold its config and its log
NET_FILES = ("net_best_epoch_state.pt", "best_model.pt")
NET_PATTERNS = ("_mlp.pt",)
CONFIG = (".yaml", "recipe.json", "RECIPE", "run_manifest", "run_provenance.json", "stage0_meta.json", "best_epoch.json",
          "batch_stream_sha.txt", "results.json", "repo_commit.txt", "val_metrics.txt", "DONE")
LOGS = ("train.log", "attempts.log", "stdout.log", "training_history.csv", "NODE_FAULTS")
FIELDS = ("path", "size_bytes", "sha256", "regenerated_by", "network_archived")
THREADS = 4          # low: the running fine-tuning pods stream from this volume (review, 2026-11-23)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files(tier: int):
    for root, which in TIERS[tier]:
        for dp, _dn, fn in os.walk(root):
            for f in sorted(fn):
                if which == "all" or f.endswith(which):
                    yield pathlib.Path(dp) / f


def is_net(p: pathlib.Path) -> bool:
    return p.name in NET_FILES or p.name.endswith(NET_PATTERNS)


def regen(p: pathlib.Path) -> str:
    rel = str(p.relative_to(R))
    return next((v for k, v in REGEN.items() if rel.startswith(k)), "retraining only")


def manifest(out: pathlib.Path, only=None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for tier in [only] if only else TIERS:
        dest = out / f"tier{tier}.csv"
        if dest.exists():
            raise SystemExit(f"FATAL: {dest} exists; a manifest is never overwritten")
        ps = list(files(tier))
        with ThreadPoolExecutor(THREADS) as ex:
            shas = list(ex.map(sha256, ps))
        with open(dest, "w", newline="") as f:
            w = csv.writer(f); w.writerow(FIELDS)
            for p, h in zip(ps, shas):
                w.writerow((p, p.stat().st_size, h, regen(p), "yes" if is_net(p) else ""))
        print(f"tier {tier}: {len(ps)} files, {sum(p.stat().st_size for p in ps) / 2**30:.2f} GiB -> {dest} "
              f"sha256 {sha256(dest)}", flush=True)
    return 0


def intact(p: pathlib.Path) -> tuple[bool, str]:
    import torch
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                      # noqa: BLE001 - a file that does not load is not intact
        return False, f"does not load: {type(e).__name__}: {e}"[:200]
    tensors = []
    def walk(o):
        if isinstance(o, torch.Tensor):
            tensors.append(o)
        elif isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
        elif hasattr(o, "state_dict"):
            walk(o.state_dict())
    walk(obj)
    if not tensors:
        return False, "no tensors"
    bad = sum(1 for t in tensors if t.is_floating_point() and not bool(torch.isfinite(t).all()))
    n = sum(t.numel() for t in tensors)
    return bad == 0, f"{len(tensors)} tensors, {n} values, {bad} with non-finite values"


def archive(out: pathlib.Path) -> int:
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists")
    out.mkdir(parents=True)
    rows, copied = [], []
    for tier in TIERS:
        for p in files(tier):
            if not is_net(p):
                continue
            ok, why = intact(p)
            rows.append((tier, p, ok, why))
            if not ok:
                continue
            side = [q for q in p.parent.iterdir() if q.is_file() and q != p
                    and (q.name.endswith(CONFIG) or q.name.startswith(CONFIG) or q.name in LOGS)]
            for q in [p] + sorted(side):
                d = out / q.relative_to(R)
                if not d.exists():
                    d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(q, d); copied.append(d)
    with open(out / "intact.tsv", "w") as f:
        f.write("tier\tnetwork\tintact\tcheck\n")
        for tier, p, ok, why in rows:
            f.write(f"{tier}\t{p}\t{'yes' if ok else 'NO'}\t{why}\n")
    with open(out / "SHA256SUMS", "w") as f:
        for d in sorted(copied):
            f.write(f"{sha256(d)}  {d.relative_to(out)}\n")
    tgz = out.with_suffix(".tgz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"networks {len(rows)}, intact {sum(r[2] for r in rows)}, files copied {len(copied)}, "
          f"{sum(d.stat().st_size for d in copied) / 2**20:.1f} MiB; {tgz} {tgz.stat().st_size} bytes sha256 {sha256(tgz)}",
          flush=True)
    for tier, p, ok, why in rows:
        if not ok:
            print(f"NOT INTACT: {p}: {why}", flush=True)
    return 0


def delete(man: pathlib.Path, want: str) -> int:
    got = sha256(man)
    if got != want:
        raise SystemExit(f"FATAL: {man} sha256 {got} is not the reviewed {want}")
    log = man.with_suffix(".deleted.log")
    rows = list(csv.DictReader(open(man)))
    roots = {r for tier in TIERS for r, _ in TIERS[tier]}
    done = kept = freed = 0
    with open(log, "a") as L:
        for r in rows:
            p = pathlib.Path(r["path"])
            if not p.exists():
                L.write(f"absent\t{p}\n"); continue
            if not any(p.is_relative_to(x) for x in roots):
                L.write(f"KEPT outside the tiers\t{p}\n"); kept += 1; continue
            if sha256(p) != r["sha256"]:
                L.write(f"KEPT changed since the manifest\t{p}\n"); kept += 1; continue
            p.unlink(); done += 1; freed += int(r["size_bytes"]); L.write(f"deleted\t{p}\n")
        for x in sorted(roots, key=lambda q: -len(str(q))):
            for dp, dn, fn in sorted(os.walk(x, topdown=False), key=lambda t: -len(t[0])):
                if dp != str(x) and not os.listdir(dp):
                    os.rmdir(dp); L.write(f"rmdir\t{dp}\n")
    print(f"{man}: deleted {done} of {len(rows)} rows ({freed / 2**30:.2f} GiB), kept {kept}; log {log}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--out", type=pathlib.Path, required=True); m.add_argument("--tier", type=int)
    a = sub.add_parser("archive"); a.add_argument("--out", type=pathlib.Path, required=True)
    d = sub.add_parser("delete"); d.add_argument("--manifest", type=pathlib.Path, required=True); d.add_argument("--sha256", required=True)
    x = ap.parse_args(argv)
    return {"manifest": lambda: manifest(x.out, x.tier), "archive": lambda: archive(x.out),
            "delete": lambda: delete(x.manifest, x.sha256)}[x.cmd]()


if __name__ == "__main__":
    sys.exit(main())
