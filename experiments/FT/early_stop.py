#!/usr/bin/env python3
"""Early stopping for a weaver fine-tuning run (plan item 3.1, 2026-10-31).

Below the full training set every fine-tuning and from-scratch run gets the same budget: 40 weaver
epochs of 500 steps (20,000 steps at batch 512), a validation pass on the fixed validation file after
each, and the checkpoint with the best validation metric. Training stops early once 10 consecutive
evaluations bring no improvement. weaver has no early stopping, so this watcher follows the run's log
and ends the trainer.

The rule is weaver's own: experiments/FT/ft_weaver.py logs every epoch's validation metric at full
precision ("Epoch #<e>: exact validation metric <v>"), and weaver keeps an epoch as best when its
metric is strictly greater than the best so far (train.py). The watcher applies that rule to the
exact values, so its best epoch is the one weaver copied to net_best_epoch_state.pt. It stops the
trainer only after weaver has logged the epoch's "Current validation metric" line, which follows the
best-state copy, and only when that epoch is at least `patience` epochs after the best one.

    python3 experiments/FT/early_stop.py watch --log RUN/train.log --pid PID --patience 10 \
        --max-epochs 40 --out CELL/early_stop.json
"""
import argparse
import json
import os
import re
import signal
import sys
import time

EXACT = re.compile(r"Epoch #(\d+): exact validation metric (\S+)")
CURRENT = re.compile(r"Epoch #(\d+): Current validation metric")


def scan(lines) -> tuple:
    """(metrics by epoch, epochs whose Current line is logged) from log lines."""
    metrics, current = {}, set()
    for line in lines:
        m = EXACT.search(line)
        if m:
            metrics[int(m.group(1))] = float(m.group(2))
        m = CURRENT.search(line)
        if m:
            current.add(int(m.group(1)))
    return metrics, current


def best_epoch(metrics: dict) -> tuple:
    """weaver's rule: an epoch is the new best if its metric is strictly greater."""
    best_e, best_v = None, -float("inf")
    for e in sorted(metrics):
        if metrics[e] > best_v:
            best_e, best_v = e, metrics[e]
    return best_e, best_v


def decide(metrics: dict, current: set, patience: int) -> tuple:
    """(stop?, last complete epoch, best epoch). An epoch is complete once its Current line is
    logged; stop when the last complete epoch is `patience` or more epochs after the best."""
    done = sorted(e for e in metrics if e in current)
    if not done:
        return False, None, None
    last = done[-1]
    be, _ = best_epoch({e: metrics[e] for e in done})
    return last - be >= patience, last, be


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def watch(a) -> int:
    stopped, last, be = False, None, None
    while True:
        lines = open(a.log).readlines() if os.path.exists(a.log) else []
        metrics, current = scan(lines)
        stop, last, be = decide(metrics, current, a.patience)
        if stop:
            os.kill(a.pid, signal.SIGTERM)
            for _ in range(120):
                if not alive(a.pid):
                    break
                time.sleep(1)
            if alive(a.pid):
                os.kill(a.pid, signal.SIGKILL)
            stopped = True
            break
        if not alive(a.pid):
            break
        time.sleep(a.poll)
    metrics, current = scan(open(a.log).readlines())
    done = {e: metrics[e] for e in metrics if e in current}
    be, bv = best_epoch(done)
    rec = {"rule": "best validation metric (weaver's, on ft_weaver.py's exact values); stop after "
                   f"{a.patience} epochs without improvement",
           "patience": a.patience, "max_epochs": a.max_epochs, "epochs_completed": len(done),
           "last_epoch": max(done) if done else None, "best_epoch": be, "best_metric": bv,
           "stopped_early": stopped, "metrics": {str(e): done[e] for e in sorted(done)}}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "metrics"}))
    if not done:
        return 1
    if not stopped and len(done) != a.max_epochs:
        print(f"FATAL: the trainer ended after {len(done)} of {a.max_epochs} epochs without being stopped")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--log", required=True)
    w.add_argument("--pid", required=True, type=int)
    w.add_argument("--patience", type=int, default=10)
    w.add_argument("--max-epochs", type=int, default=40)
    w.add_argument("--out", required=True)
    w.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args(argv)
    return watch(a)


if __name__ == "__main__":
    sys.exit(main())
