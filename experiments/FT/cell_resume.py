#!/usr/bin/env python3
"""A fine-tuning cell resumes from its last completed epoch (the retry logic of
2026-10-01; "retries that survive a flaky cluster" in scripts/build_ft_jobs.py).

WHY. Until then an interrupted cell started again at epoch 0, and a cap of six
attempts of any kind stopped the job. On 2026-09-30 bench_v2/leg_qg/mpm-s1-v2/
N1600000/s2 (20 epochs of ~9 min) was interrupted in mid-epoch six times, five
of them on ry-gpu-10 while its GPU was failing, after 9, 5, 13, 4, 3 and 8
epochs. None of the six failed; 42 epochs were trained and none counted, and the
cap halted the job. The other halted job failed twice on that node's GPU after
all its training was done, and retrained from scratch in between.

What one attempt can leave in its cell, besides weaver's files:
    ATTEMPT_FAILED      the job's EXIT trap: a step failed (a real failure)
    ATTEMPT_FAILED.<n>  an earlier failure of a cell kept in place (TRAINED)
    NODE_FAULT.<time>   the trap: a step failed and the GPU no longer answered,
                        or the log holds a CUDA device fault -- the node's fault
    TRAINED             training finished and passed its checks; what is left
                        is the read-out
    ATTEMPTS            one line per attempt with the epoch it resumed from

    prepare --dir OUT [--max-failed F] [--max-node-faults G] [--max-stalled K]
        Run before weaver. Prints the epoch this attempt resumes from, -1 for
        "from the start". Exits 3, printing why, to halt the job when
          * the cell's failed attempts reach F (moved-aside ones included), or
          * its node faults reach G, or
          * its last K attempts completed no epoch between them;
        and then renames what it counted (a failed attempt directory, and OUT
        itself unless TRAINED, to OUT.halted.partial.<time>; markers of a TRAINED
        OUT and node faults into OUT/halted.<time>/; the ledger to
        ATTEMPTS.halted.<time>), so once the cause is fixed and the FAILED marker
        removed the cell is not halted again by the same evidence. Otherwise:
          * no OUT: -1.
          * a FAILED attempt: if TRAINED, the training stands -- the marker is
            renamed ATTEMPT_FAILED.<n> (still counted) and the cell resumes after
            its last epoch, so only the read-out is redone; if not, the attempt is
            moved aside to OUT.partial.<time> with its epoch checkpoints deleted
            (its logs and markers kept) and the cell starts again: a failed
            training is never resumed, its state may be what failed.
          * otherwise it was interrupted (an evicted or lost pod is SIGKILLed and
            runs no trap), or a node fault ended it: E is the last epoch weaver
            finished ("Epoch #E: Current validation metric") whose state and
            optimizer files are whole, and the cell is put back as it stood after
            epoch E: train.log cut after that line (the cut kept as
            train.log.cut.<n>), stdout.log kept as stdout.log.<n>, the checkpoints
            of later epochs and every output of the steps after training removed
            (they are rebuilt; extract_features.py refuses a cache from another
            checkpoint).

    best --dir OUT --epochs N
        Run after weaver. weaver 0.4.17 starts its best-validation tracking at 0
        in every process (train.py:820), so after --load-epoch its
        net_best_epoch_state.pt is the best of the last attempt only. The best
        epoch is recomputed as weaver chooses it -- the first strict maximum,
        starting from 0 -- over the full-precision metric of every epoch that
        experiments/FT/ft_weaver.py writes to train.log, and
        net_best_epoch_state.pt made a copy of that epoch's state. In a run that
        never resumed, weaver's own file must already be that copy, or the cell
        fails. best_epoch.json records the choice.

    lock --path OUT.lock --owner SHARD
        Claim a cell atomically: a temporary directory holding the owner is
        renamed onto the lock, which fails if the lock exists. Prints
        "acquired", "reentered" (a lock of this owner, left by an evicted pod) or
        the other owner ("unknown" when the lock records none).
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
import zipfile

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FINISHED = re.compile(r"Epoch #(\d+): Current validation metric: ")
EXACT = re.compile(r"Epoch #(\d+): exact validation metric (\S+)\s*$")
FROM = re.compile(r"from_epoch=(-?\d+)")
EPOCH_FILE = re.compile(r"net_epoch-(\d+)_(state|optimizer)\.pt")
RESUMED = "Resume training from epoch"
FAILED, LEDGER, TRAINED = "ATTEMPT_FAILED", "ATTEMPTS", "TRAINED"
REBUILT = ("pred.root", "predict.log", "net_last_epoch_state.pt", "net_best_epoch_state.pt",
           "best_epoch.json")
EXIT_HALT = 3


def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text(errors="replace").splitlines(keepends=True) if path.exists() else []


def whole(path: pathlib.Path) -> bool:
    """torch.save writes a zip archive; a file cut short by a lost node is not one."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def resume_point(cell: pathlib.Path) -> tuple[int, int]:
    """(E, number of train.log lines up to and including epoch E's), E = -1 if none.
    Epochs must run 0, 1, ..., E in the log, as they do in a log this module cut."""
    finished = [(int(m.group(1)), i) for i, ln in enumerate(_lines(cell / "train.log"))
                if (m := FINISHED.search(ANSI.sub("", ln)))]
    clean = []
    for k, (e, i) in enumerate(finished):
        if e != k:
            break
        clean.append((e, i))
    for e, i in reversed(clean):
        if whole(cell / f"net_epoch-{e}_state.pt") and whole(cell / f"net_epoch-{e}_optimizer.pt"):
            return e, i + 1
    return -1, 0


def stalled(starts: list[int], now: int) -> int:
    """How many of the latest attempts completed no epoch: attempt i started at
    starts[i] and the next one at starts[i + 1] (this one: `now`)."""
    n, nxt = 0, now
    for s in reversed(starts):
        if nxt > s:
            break
        n, nxt = n + 1, s
    return n


def _failures(d: pathlib.Path) -> list[pathlib.Path]:
    return sorted(d.glob(FAILED + "*")) if d.is_dir() else []


def _prune_epochs(d: pathlib.Path, after: int = -1) -> None:
    for p in d.glob("net_epoch-*_*.pt"):
        m = EPOCH_FILE.fullmatch(p.name)
        if m and int(m.group(1)) > after:
            p.unlink()


def _halt(reason: str) -> None:
    print(reason)
    sys.exit(EXIT_HALT)


def prepare(cell: pathlib.Path, max_failed: int = 2, max_node_faults: int = 4,
            max_stalled: int = 3) -> int:
    stamp = int(time.time())
    partials = sorted(p for p in cell.parent.glob(cell.name + ".partial.*") if p.is_dir())
    failed = [(p, f) for p in partials + [cell] for f in _failures(p)]
    if len(failed) >= max_failed:
        for i, p in enumerate(partials):
            if _failures(p):
                p.rename(cell.with_name(f"{cell.name}.halted.partial.{stamp}.{i}"))
        if _failures(cell) and not (cell / TRAINED).exists():
            cell.rename(cell.with_name(f"{cell.name}.halted.partial.{stamp}"))
        elif _failures(cell):             # trained: the training stands, the markers go
            (cell / f"halted.{stamp}").mkdir()
            for f in _failures(cell):
                f.rename(cell / f"halted.{stamp}" / f.name)
        _halt(f"{len(failed)} failed attempts; renamed so they are not counted again "
              f"({cell.name}.halted.partial.{stamp}.*, {cell.name}/halted.{stamp})")
    if not cell.exists():
        return -1
    faults = sorted(cell.glob("NODE_FAULT.*"))
    if len(faults) >= max_node_faults:
        nodes = sorted({ln.split("node=")[1].split()[0] for f in faults
                        for ln in _lines(f) if "node=" in ln})
        (cell / f"halted.{stamp}").mkdir(exist_ok=True)
        for f in faults:
            f.rename(cell / f"halted.{stamp}" / f.name)
        _halt(f"{len(faults)} node faults (nodes {', '.join(nodes)}); moved to {cell.name}/halted.{stamp}")
    starts = [int(m.group(1)) for ln in _lines(cell / LEDGER) if (m := FROM.search(ln))]
    k = len(starts)
    if (cell / FAILED).exists():
        if not (cell / TRAINED).exists():
            dest = cell.with_name(f"{cell.name}.partial.{stamp}")
            cell.rename(dest)
            _prune_epochs(dest)
            print(f"{cell}: the last attempt failed; moved to {dest.name} (epoch checkpoints "
                  f"deleted), starting again", file=sys.stderr)
            return -1
        (cell / FAILED).rename(cell / f"{FAILED}.{k}")
        print(f"{cell}: the last attempt failed after training; redoing the read-out", file=sys.stderr)
    e, cut = resume_point(cell)
    n = stalled(starts, e)
    if n >= max_stalled:
        (cell / LEDGER).rename(cell / f"{LEDGER}.halted.{stamp}")
        _halt(f"no epoch completed in the last {n} attempts (resumed at epoch {e}; limit "
              f"{max_stalled}); ledger kept as {LEDGER}.halted.{stamp}")
    log = cell / "train.log"
    lines = _lines(log)
    if lines[cut:]:
        (cell / f"train.log.cut.{k}").write_text("".join(lines[cut:]))
        log.write_text("".join(lines[:cut]))
    if (cell / "stdout.log").exists():
        (cell / "stdout.log").rename(cell / f"stdout.log.{k}")
    _prune_epochs(cell, e)
    for name in REBUILT:
        (cell / name).unlink(missing_ok=True)
    for d in cell.glob("features*"):
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    print(f"{cell}: attempt {k} ended; resuming after epoch {e}" if e >= 0 else
          f"{cell}: attempt {k} completed no epoch; starting again", file=sys.stderr)
    return e


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def best(cell: pathlib.Path, n_epochs: int) -> dict:
    text = (cell / "train.log").read_text(errors="replace")
    exact: dict[int, float] = {}
    for ln in text.splitlines():
        m = EXACT.search(ANSI.sub("", ln))
        if m:
            e = int(m.group(1))
            if e in exact:
                raise SystemExit(f"FATAL: {cell}/train.log holds epoch {e}'s validation twice")
            exact[e] = float(m.group(2))
    if sorted(exact) != list(range(n_epochs)):
        raise SystemExit(f"FATAL: {cell}/train.log has the exact validation metric of epochs "
                         f"{sorted(exact)}, not 0..{n_epochs - 1}")
    top, b = 0.0, None                    # weaver: best_valid_metric = 0, then strict >
    for e in range(n_epochs):
        if exact[e] > top:
            top, b = exact[e], e
    if b is None:
        raise SystemExit(f"FATAL: {cell}: no epoch's validation metric exceeds 0")
    src, dst = cell / f"net_epoch-{b}_state.pt", cell / "net_best_epoch_state.pt"
    want = sha256(src)
    have = sha256(dst) if dst.exists() else None
    resumed = RESUMED in text
    if have != want:
        if not resumed and have is not None:
            raise SystemExit(f"FATAL: {cell} never resumed, yet weaver's best-epoch file is not "
                             f"epoch {b}, the first maximum of the logged metric")
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        if sha256(dst) != want:
            raise SystemExit(f"FATAL: {dst} is not a copy of {src} after copying")
    rec = {"epoch": b, "metric": top, "n_epochs": n_epochs, "resumed": resumed,
           "restored": have != want, "sha256": want,
           "metrics": [exact[e] for e in range(n_epochs)]}
    (cell / "best_epoch.json").write_text(json.dumps(rec, indent=1))
    print(f"{cell}: best epoch {b} ({top!r}){', restored after resume' if have != want else ''}")
    return rec


def lock(path: pathlib.Path, owner: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{time.time_ns()}")   # skipped by the readers
        tmp.mkdir()
        (tmp / "owner").write_text(owner + "\n")
        try:
            os.rename(tmp, path)          # fails when the lock exists (it is never empty)
            return "acquired"
        except OSError:
            shutil.rmtree(tmp)
    try:
        held = (path / "owner").read_text().strip() or "unknown"
    except OSError:
        held = "unknown"
    return "reentered" if held == owner else held


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dir", required=True, type=pathlib.Path)
    p.add_argument("--max-failed", type=int, default=2)
    p.add_argument("--max-node-faults", type=int, default=4)
    p.add_argument("--max-stalled", type=int, default=3)
    b = sub.add_parser("best")
    b.add_argument("--dir", required=True, type=pathlib.Path)
    b.add_argument("--epochs", required=True, type=int)
    k = sub.add_parser("lock")
    k.add_argument("--path", required=True, type=pathlib.Path)
    k.add_argument("--owner", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(prepare(a.dir, a.max_failed, a.max_node_faults, a.max_stalled))
    elif a.cmd == "best":
        best(a.dir, a.epochs)
    else:
        print(lock(a.path, a.owner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
