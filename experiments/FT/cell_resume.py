#!/usr/bin/env python3
"""A fine-tuning cell resumes from its last completed epoch (the retry logic of
2026-10-01; "retries that survive a flaky cluster" in scripts/build_ft_jobs.py).

WHY. Until then an interrupted cell started again at epoch 0, and a cap of six
attempts of any kind stopped the job. On 2026-09-30 bench_v2/leg_qg/mpm-s1-v2/
N1600000/s2 (20 epochs of ~9 min) was interrupted in mid-epoch six times, five
of them on ry-gpu-10 while its GPU was failing, after 9, 5, 13, 4, 3 and 8
epochs. None of the six failed; 42 epochs were trained and none counted, and the
cap halted the job.

    prepare --dir OUT [--max-stalled K]
        Run before weaver. Prints the epoch this attempt resumes from, -1 for
        "from the start":
          * no OUT: -1.
          * OUT holds a FAILED attempt (ATTEMPT_FAILED, written by the job's
            EXIT trap): moved aside to OUT.partial.<unix time>, -1. A failed
            attempt is never resumed, since its state may be what failed (a NaN
            run, for one); the job counts it and stops at two.
          * otherwise OUT was interrupted (an evicted or lost pod is SIGKILLed
            and runs no trap). E is the last epoch weaver finished (its
            "Epoch #E: Current validation metric" line) whose state and
            optimizer files are whole, and the cell is put back as it stood
            after epoch E: train.log cut after that line (the cut kept as
            train.log.cut.<n>), stdout.log kept as stdout.log.<n>, the
            checkpoints of later epochs and every output of the steps after
            training removed (they are rebuilt, and extract_features.py refuses
            a cache from another checkpoint).
        Exits 3, printing why, when the last K attempts in OUT/ATTEMPTS (the
        ledger the job appends to, one line per attempt with its from_epoch)
        completed no epoch between them: a cell that makes no progress is
        stopped rather than retried for ever.

    best --dir OUT --epochs N
        Run after weaver. weaver 0.4.17 starts its best-validation tracking at 0
        in every process (train.py:820), so after --load-epoch its
        net_best_epoch_state.pt is the best of the last attempt only. The best
        epoch is recomputed exactly as weaver chooses it -- the first strict
        maximum, starting from 0 -- over the full-precision metric of every
        epoch that experiments/FT/ft_weaver.py writes to train.log, and
        net_best_epoch_state.pt is made a copy of that epoch's state. In a run
        that never resumed, weaver's own file must already be that copy, or the
        cell fails. best_epoch.json records the choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
RESUMED = "Resume training from epoch"
FAILED, LEDGER = "ATTEMPT_FAILED", "ATTEMPTS"
REBUILT = ("pred.root", "predict.log", "net_last_epoch_state.pt", "net_best_epoch_state.pt",
           "best_epoch.json")
EXIT_STALLED = 3


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


def prepare(cell: pathlib.Path, max_stalled: int) -> int:
    if not cell.exists():
        return -1
    if (cell / FAILED).exists():
        dest = cell.with_name(f"{cell.name}.partial.{int(time.time())}")
        cell.rename(dest)
        print(f"{cell}: the last attempt failed; moved to {dest.name}, starting again", file=sys.stderr)
        return -1
    starts = [int(m.group(1)) for ln in _lines(cell / LEDGER) if (m := FROM.search(ln))]
    e, cut = resume_point(cell)
    n = stalled(starts, e)
    if n >= max_stalled:
        print(f"no epoch completed in the last {n} attempts (resumed at epoch {e}; "
              f"limit {max_stalled}); see {cell / LEDGER}")
        sys.exit(EXIT_STALLED)
    k = len(starts)
    log = cell / "train.log"
    lines = _lines(log)
    if lines[cut:]:
        (cell / f"train.log.cut.{k}").write_text("".join(lines[cut:]))
        log.write_text("".join(lines[:cut]))
    if (cell / "stdout.log").exists():
        (cell / "stdout.log").rename(cell / f"stdout.log.{k}")
    for p in cell.glob("net_epoch-*_*.pt"):
        m = re.fullmatch(r"net_epoch-(\d+)_(state|optimizer)\.pt", p.name)
        if m and int(m.group(1)) > e:
            p.unlink()
    for name in REBUILT:
        (cell / name).unlink(missing_ok=True)
    for d in cell.glob("features*"):
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    print(f"{cell}: interrupted attempt {k}; resuming after epoch {e}" if e >= 0 else
          f"{cell}: interrupted attempt {k} completed no epoch; starting again", file=sys.stderr)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dir", required=True, type=pathlib.Path)
    p.add_argument("--max-stalled", type=int, default=3)
    b = sub.add_parser("best")
    b.add_argument("--dir", required=True, type=pathlib.Path)
    b.add_argument("--epochs", required=True, type=int)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(prepare(a.dir, a.max_stalled))
    else:
        best(a.dir, a.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
