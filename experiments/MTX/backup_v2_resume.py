"""Verified backups of what a v2 run cannot regenerate once it has moved on.

A run keeps one resume file, its newest (pretrain_v2.prune), and writes each kept state
file once. Both are written by rename without fsync, and on 2026-10-07 a full storage
pool left two recipe.json files written that way as zeros (RUNS.csv
data-volume-create-denied-1007-outcome). A zero-filled newest resume file would leave a
run nothing to resume from but epoch 0; a zero-filled state of epochs 70-79 would leave
no weight average. This script, run every half hour by a CPU job, keeps for every run of
the grid, in <run>/backup/ (pretrain_v2 and the job script look only at the top level of
the run directory, so nothing there is read as the run's own):

  resume/net_epoch-N_resume.pt, net_epoch-N_state.pt   the newest complete epoch N and
      the one before it, each copied only after it loads, names its own epoch, holds
      the model, optimizer, scheduler, scaler, trimmer counters and best epoch, and its
      model equals the state file of N tensor for tensor;
  states/net_epoch-E_state.pt   every state file retention keeps for good (EARLY_KEEP,
      epochs 70-79, the best epoch), once each, after it loads;
  recipe.json   once, after it parses;
  manifest.json   the sha256 of every copy and of its source when copied.

Every copy is written to a temporary name, fsynced, renamed and read back against the
source's sha256. Nothing in the run's own directory is changed. A file that does not
load is not copied and is reported (exit 1), so the job's log names it. Restoring from a
backup is the PI's decision and is not done here.

    python3 experiments/MTX/backup_v2_resume.py [--root /data/results/mtx_v2] [--runs RUN ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import time

EARLY_KEEP = (0, 2, 4, 9, 19, 29, 39, 49, 55, 62, 69)   # pretrain_v2.EARLY_KEEP
WINDOW = range(70, 80)                                   # pretrain_v2.last_epochs(80)
RESUME_KEYS = {"epoch", "model", "optimizer", "scheduler", "scaler", "trimmer_counters", "best"}
N_RESUME_KEPT = 2


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(src: pathlib.Path, dst: pathlib.Path) -> str:
    """Copy src to dst through a temporary name with fsync; return the sha256 both share."""
    want = sha256(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    got = sha256(dst)
    if got != want:
        dst.unlink()
        raise OSError(f"{dst}: copy reads back as {got[:12]}, source {want[:12]}")
    return want


def latest_complete_epoch(run: pathlib.Path):
    """As pretrain_v2.latest_complete_epoch: the newest epoch with resume and state files."""
    best = None
    for p in run.glob("net_epoch-*_resume.pt"):
        e = int(p.name[len("net_epoch-"):-len("_resume.pt")])
        if (run / f"net_epoch-{e}_state.pt").exists() and (best is None or e > best):
            best = e
    return best


def check_resume(resume: pathlib.Path, state: pathlib.Path, epoch: int) -> str | None:
    """None if the resume file is whole and its model is the state file's, else why not."""
    import torch
    try:
        r = torch.load(resume, map_location="cpu", weights_only=False)
        s = torch.load(state, map_location="cpu", weights_only=False)
    except Exception as e:                       # a zero-filled or truncated file
        return f"does not load: {type(e).__name__}: {str(e)[:80]}"
    missing = RESUME_KEYS - set(r)
    if missing:
        return f"lacks {sorted(missing)}"
    if r["epoch"] != epoch:
        return f"names epoch {r['epoch']}, not {epoch}"
    if set(r["model"]) != set(s) or any(not torch.equal(r["model"][k], s[k]) for k in s):
        return "its model is not the state file of its epoch"
    return None


def backup_run(run: pathlib.Path) -> list[str]:
    """Bring run/backup up to date; return the problems found (empty when all is well)."""
    import torch
    problems = []
    bdir = run / "backup"
    man_path = bdir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"copies": {}}
    copies = man["copies"]

    def record(rel: str, src: pathlib.Path, digest: str):
        copies[rel] = {"sha256": digest, "source": src.name, "copied_utc":
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    rec = run / "recipe.json"
    if rec.exists() and "recipe.json" not in copies:
        try:
            json.loads(rec.read_text())
            record("recipe.json", rec, copy_verified(rec, bdir / "recipe.json"))
        except (ValueError, UnicodeDecodeError) as e:
            problems.append(f"{run.name}/recipe.json does not parse ({type(e).__name__})")

    n = latest_complete_epoch(run)
    if n is not None and f"resume/net_epoch-{n}_resume.pt" not in copies:
        resume, state = run / f"net_epoch-{n}_resume.pt", run / f"net_epoch-{n}_state.pt"
        why = check_resume(resume, state, n)
        if why:
            problems.append(f"{run.name}/{resume.name} {why}")
        else:
            for src in (state, resume):          # the resume file last, as the run writes them
                rel = f"resume/{src.name}"
                record(rel, src, copy_verified(src, bdir / rel))
            # keep the newest N_RESUME_KEPT backed-up epochs: older copies are superseded
            kept = sorted({int(re.search(r"net_epoch-(\d+)_", k).group(1)) for k in copies
                           if k.startswith("resume/")}, reverse=True)
            for old in kept[N_RESUME_KEPT:]:
                for kind in ("resume", "state"):
                    rel = f"resume/net_epoch-{old}_{kind}.pt"
                    (bdir / rel).unlink(missing_ok=True)
                    copies.pop(rel, None)

    best = json.loads((run / "best_epoch.json").read_text()).get("epoch") \
        if (run / "best_epoch.json").exists() else None
    for e in sorted(set(EARLY_KEEP) | set(WINDOW) | ({best} if best is not None else set())):
        src = run / f"net_epoch-{e}_state.pt"
        rel = f"states/{src.name}"
        if not src.exists() or rel in copies:
            continue
        if n is not None and e > n:              # not final yet: a later attempt may rewrite it
            continue
        try:
            torch.load(src, map_location="cpu", weights_only=False)
        except Exception as ex:
            problems.append(f"{run.name}/{src.name} does not load: {type(ex).__name__}")
            continue
        record(rel, src, copy_verified(src, bdir / rel))

    if copies:
        bdir.mkdir(parents=True, exist_ok=True)
        tmp = man_path.with_name("manifest.json.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, man_path)
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/data/results/mtx_v2")
    ap.add_argument("--runs", nargs="*", default=None, help="run directory names (default: every mtx-*)")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    runs = [root / r for r in a.runs] if a.runs else sorted(p for p in root.glob("mtx-*") if p.is_dir())
    problems = []
    for run in runs:
        problems += backup_run(run)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} backed up {len(runs)} runs; "
          f"{len(problems)} problems", flush=True)
    for p in problems:
        print("PROBLEM", p, flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
