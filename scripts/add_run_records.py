#!/usr/bin/env python3
"""Make each MTX training spec keep the records an evicted attempt loses.

Measured 2026-09-07 on the running wave: a taint eviction deletes the pod, and
with it everything that was only in the pod -- the tensorboard event files
(`./runs`, copied to the PVC only when training ENDS), the AUTO-RESUME line
saying which epoch the attempt resumed from, and the attempt's pod/node. The
launch-time manifest is rewritten by every attempt, so `launched_utc` and the
node name are the LAST attempt's. train.log survives (weaver appends), so
per-epoch curves and `Resume training from epoch N` lines are safe; the
per-iteration curves and the attempt provenance are not.

Three deltas, applied to the YAML only (no code change):

  1. `attempts.log` in the run dir: one line per attempt with UTC time, pod,
     node, and the resume decision. Appended, never rewritten.
  2. the previous attempt's run_manifest.json is kept as
     run_manifest.<utc>.prev.json before the new one is written.
  3. `./runs` is a symlink to `${OUT}/tb`, so tensorboard writes to the PVC
     continuously; the end-of-run `cp -r ./runs` is dropped (it would copy the
     symlink into its own target).

Plus the REPO_REF bump to the tag that carries the resume-safe scheduler
(src/utils/resume.py), which is the reason these jobs are re-created at all.

Idempotent; refuses specs it does not recognise; only touches specs named on
the command line.

Run:  python3 scripts/add_run_records.py --tag mtx-s1.11 experiments/MTX/k8s/job-mtx-*.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

MARK = "# ATTEMPT RECORDS (scripts/add_run_records.py)"

RECORDS = f"""          {MARK}
          ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)
          [ -f ${{OUT}}/run_manifest.json ] && cp ${{OUT}}/run_manifest.json ${{OUT}}/run_manifest.${{ATTEMPT}}.prev.json
          mkdir -p ${{OUT}}/tb
          [ -e ./runs ] || ln -s ${{OUT}}/tb ./runs
"""

ATTEMPT_LINE = """          echo "${ATTEMPT} pod=${POD_NAME} node=${NODE_NAME} resume=[${RESUME:-fresh}]" >> ${OUT}/attempts.log
"""


def patch(path: pathlib.Path, tag: str) -> str:
    text = path.read_text()
    if MARK in text:
        return "already patched"

    # 1+2+3: right after the run dir is created (first occurrence).
    anchor = "          mkdir -p ${OUT}\n"
    if text.count(anchor) < 1:
        return "FAILED: no 'mkdir -p ${OUT}' line"
    text = text.replace(anchor, anchor + RECORDS, 1)

    # attempt line: after the RECIPE stamp, when RESUME is known.
    stamp = "          printf '%s' \"${RECIPE}\" > ${OUT}/RECIPE\n"
    if text.count(stamp) != 1:
        return f"FAILED: expected 1 RECIPE stamp line, found {text.count(stamp)}"
    text = text.replace(stamp, stamp + ATTEMPT_LINE)

    # drop the end-of-run copy
    cp = "          cp -r ./runs ${OUT}/tb 2>/dev/null || true\n"
    if text.count(cp) != 1:
        return f"FAILED: expected 1 'cp -r ./runs' line, found {text.count(cp)}"
    text = text.replace(cp, "          # tensorboard already on the PVC: ./runs -> ${OUT}/tb\n")

    # repo pin
    n = len(re.findall(r'value: "mtx-s1\.\d+"', text))
    if n != 1:
        return f"FAILED: expected 1 REPO_REF pin, found {n}"
    text = re.sub(r'value: "mtx-s1\.\d+"', f'value: "{tag}"', text)

    d = yaml.safe_load(text)
    args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    for must in (MARK, "attempts.log", "ln -s ${OUT}/tb ./runs", ".prev.json"):
        if must not in args:
            return f"FAILED: patched spec is missing {must!r}"
    if "cp -r ./runs" in args:
        return "FAILED: cp -r ./runs still present"
    path.write_text(text)
    return f"patched (records + tb symlink, pin -> {tag})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    rc = 0
    for s in a.specs:
        p = pathlib.Path(s)
        r = patch(p, a.tag)
        print(f"{p.name:40s} {r}")
        if r.startswith("FAILED"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
