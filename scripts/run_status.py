#!/usr/bin/env python3
"""One-line status for every raunav training pod.

WHY A SCRIPT AND NOT A grep
---------------------------
Four things about weaver's output have each cost time already, and a one-off
grep gets them wrong in a way that looks like a real answer:

  1. tqdm writes progress with CARRIAGE RETURNS, so an entire epoch's progress
     is ONE line. Anchoring a match with `^` finds nothing and reads exactly
     like "the run has not started". A watcher written that way sat for 8 h
     reporting no verdict while the run it watched was 16,000 iterations in.

  2. weaver logs each validation line TWICE, and NOT adjacently -- the whole
     sequence repeats in blocks. `uniq` does not collapse it; keying by epoch
     number does.

  3. The validation lines carry ANSI colour codes, so a naive regex on
     "metric: ([0-9.]+)" can pick up escape bytes.

  4. `kubectl logs` serves only the CURRENT rotated chunk, so early epochs
     disappear from it while `${OUT}/train.log` on the PVC keeps everything.
     Anything reported from kubectl is a LOWER BOUND on epochs completed and is
     labelled as such rather than presented as the count.

Usage:  python3 scripts/run_status.py [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# tolerate the ANSI codes rather than assuming they are absent
EPOCH_RE = re.compile(
    r"Epoch #(\d+): Current validation metric: ([0-9.]+) \(best: ([0-9.]+)\)")
# no ^ anchor: tqdm progress shares a line with whatever preceded the \r
ITER_RE = re.compile(r"(\d+)it \[([0-9:]+),\s*([0-9.]+)it/s[^\]]*\]")
LR_RE = re.compile(r"lr=([0-9.e+-]+)")
ACC_RE = re.compile(r"AvgAcc=([0-9.]+)")


def sh(*a) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def pods() -> list[tuple[str, str, str]]:
    out = sh("kubectl", "get", "pods", "--no-headers")
    rows = []
    for ln in out.splitlines():
        f = ln.split()
        if len(f) >= 3 and "raunav" in f[0] and re.match(r"(mtx|g1)-", f[0]):
            rows.append((f[0], f[2], f[-1]))
    return sorted(rows)


def status(pod: str) -> dict:
    log = sh("kubectl", "logs", pod, "--tail=4000")
    d: dict = {}
    if not log:
        return d
    by_epoch = {int(e): float(b) for e, _, b in EPOCH_RE.findall(log)}
    if by_epoch:
        ks = sorted(by_epoch)
        d["epochs_seen"] = len(ks)
        d["last_epoch"] = ks[-1]
        d["best"] = max(by_epoch.values())
    it = ITER_RE.findall(log)
    if it:
        d["iter"], d["elapsed"], d["it_s"] = int(it[-1][0]), it[-1][1], float(it[-1][2])
    lr = LR_RE.findall(log)
    if lr:
        d["lr"] = lr[-1]
    acc = ACC_RE.findall(log)
    if acc:
        d["avg_acc"] = float(acc[-1])
    if re.search(r"\bnan\b", log, re.I):
        d["nan"] = True
    if "AUTO-RESUME" in log:
        m = re.search(r"AUTO-RESUME: ([^\n]+)", log)
        d["resume"] = m.group(1)[:60] if m else True
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = pods()
    if not rows:
        print("no raunav mtx-/g1- pods")
        return 0
    out = {}
    for pod, phase, age in rows:
        st = status(pod) if phase == "Running" else {}
        out[pod] = {"phase": phase, "age": age, **st}

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"{'pod':38s} {'phase':9s} {'age':7s} {'ep':>4s} {'best':>8s} "
          f"{'iter':>7s} {'it/s':>5s} {'lr':>9s} {'avgacc':>7s}")
    for pod, d in out.items():
        ep = f"{d['last_epoch']}+" if "last_epoch" in d else ""
        print(f"{pod[:38]:38s} {d['phase'][:9]:9s} {d['age'][:7]:7s} "
              f"{ep:>4s} {d.get('best', ''):>8} "
              f"{d.get('iter', ''):>7} {d.get('it_s', ''):>5} "
              f"{d.get('lr', ''):>9} {d.get('avg_acc', ''):>7}"
              + ("  NAN" if d.get("nan") else ""))
    print("\nep is a LOWER BOUND: kubectl serves only the current log chunk; "
          "${OUT}/train.log on the PVC is authoritative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
