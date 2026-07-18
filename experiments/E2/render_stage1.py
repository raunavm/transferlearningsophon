#!/usr/bin/env python3
"""Render E2 Stage-1 confirmatory jobs from the Stage-0 LR selection (PLAN §9.3).

Reads stage0_lr_selection.json (produced by analyze_stage0.py) + the Stage-1
template, and emits one job YAML per (arm x seed) with __GRAN__/__SEED__/__LR__
filled. Seeds S1 = {101,102,103}. The key pair (k188, k10sem) is emitted FIRST
(§9.3 "scheduled first") — the launch order in the printed manifest reflects that.

Refuses to render any arm whose chosen_lr is null (Stage-0 not yet complete for it).

Usage:
  python3 experiments/E2/render_stage1.py \
      --selection /data/results/e2/stage0_lr_selection.json \
      --template experiments/E2/k8s/job-e2-stage1-TEMPLATE-raunav.yaml \
      --outdir /tmp/e2s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEEDS = [101, 102, 103]                      # §9.3 shared paired seeds
KEY_PAIR = ["g188", "g10sem"]                # §9.3 scheduled first
ARMS_ORDER = KEY_PAIR + ["g2", "g10rand", "g30"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--template", default="experiments/E2/k8s/job-e2-stage1-TEMPLATE-raunav.yaml")
    ap.add_argument("--outdir", default="/tmp/e2s1")
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text())
    template = Path(args.template).read_text()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arms = sel["arms"]
    missing = [a for a in ARMS_ORDER if arms.get(a, {}).get("chosen_lr") is None]
    if missing:
        print(f"REFUSING to render — no chosen LR for: {missing}\n"
              f"(run analyze_stage0.py to completion first)")
        sys.exit(1)

    launch_order = []
    for arm in ARMS_ORDER:
        lr = arms[arm]["chosen_lr"]
        tag = arms[arm]["chosen_tag"]
        for seed in SEEDS:
            job = (template
                   .replace("__GRAN__", arm)
                   .replace("__SEED__", str(seed))
                   .replace("__LR__", lr))
            fname = outdir / f"s1-{arm}-s{seed}.yaml"
            fname.write_text(job)
            launch_order.append((arm, seed, lr, tag, fname.name))

    print(f"rendered {len(launch_order)} Stage-1 jobs -> {outdir}\n")
    print("launch order (key pair k188/k10sem first, §9.3):")
    for arm, seed, lr, tag, fn in launch_order:
        print(f"  {fn:28s}  arm={arm:8s} seed={seed} lr={lr} ({tag})")
    print("\nCluster caps: <=5 concurrent, stagger 60-90s, L40-only (queue penalty).")
    print("15 runs x 76 GPU-h = 1,140 GPU-h (§9.3). Key pair (6 runs) first.")


if __name__ == "__main__":
    main()
