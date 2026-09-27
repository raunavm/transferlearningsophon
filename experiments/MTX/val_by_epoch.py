#!/usr/bin/env python3
"""Per-epoch validation accuracy of pretraining runs, with what makes two runs comparable.

For S8 (docs/PRESPEC_2026-09.md, clarification of 2026-09-27): the mass-output
models against their twins at fixed epochs. Reads each run's train.log and
run_manifest*.json and prints one JSON document; writes nothing.

Per run:
  metric      {epoch: validation metric}, the LAST value logged for the epoch
              (a resume re-runs an epoch; the later pass is the one the run
              continued from)
  val_state   {epoch: [file_lists_sha256, passes_since_reader_start]} at the
              validation that produced that metric. The validation reader
              starts at launch and at every resume, logs its workers' file lists
              only then, and reads the next samples_per_epoch_val jets at every
              epoch, so two runs validate on the same jets at an epoch exactly
              when these two values agree.
  gpu         every GPU product any attempt of the run recorded
  resumes     every "Resume training from epoch N"

Run inside a pod that mounts /data:
    kubectl exec -i <pod> -- python3 - mtx-l162mass-s1 mtx-l162-s1b ... < experiments/MTX/val_by_epoch.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/data/results/mtx")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(r"Epoch #(\d+): Current validation metric: (-?[0-9.]+) \(best:")
RESTART = re.compile(r"Restarted DataIter (val_worker\d+), .*file_list:")
RESUME = re.compile(r"Resume training from epoch (\d+)")


def parse(log: pathlib.Path) -> dict:
    lines = [ANSI.sub("", s) for s in log.read_text(errors="replace").splitlines()]
    metric, state, resumes = {}, {}, []
    lists, sha, passes, i = {}, None, 0, 0
    while i < len(lines):
        s = lines[i]
        m = RESTART.search(s)
        if m:                                   # a JSON block follows, up to a line "}"
            j = i + 1
            while j < len(lines) and lines[j] != "}":
                j += 1
            lists[m.group(1)] = json.loads("\n".join(lines[i + 1:j + 1]))
            i = j + 1
            continue
        m = METRIC.search(s)
        if m:                                   # the restart lines come AFTER "validating"
            if lists:                           # the reader (re)started for this pass
                sha = hashlib.sha256(json.dumps(lists, sort_keys=True).encode()).hexdigest()
                passes, lists = 0, {}
            passes += 1
            e = int(m.group(1))
            metric[e] = float(m.group(2))
            state[e] = [sha, passes]
        m = RESUME.search(s)
        if m:
            resumes.append(int(m.group(1)))
        i += 1
    return {"metric": metric, "val_state": state, "resumes": resumes}


def gpus(run: pathlib.Path) -> list[str]:
    out = set()
    for p in sorted(run.glob("run_manifest*.json")):
        hw = json.loads(p.read_text()).get("hardware") or {}
        if hw.get("gpu_product_nodelabel"):
            out.add(hw["gpu_product_nodelabel"])
    return sorted(out)


def main(runs: list[str]) -> int:
    out = {}
    for r in runs:
        d = ROOT / r
        log = d / "train.log"
        out[r] = {**parse(log), "gpu": gpus(d),
                  "train_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
