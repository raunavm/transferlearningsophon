#!/usr/bin/env python3
"""Emit the four-granularity frozen-probe and label-recovery jobs.

ONE JOB PER PRETRAINING-SEED INDEX, four models each (188-, 162-, 43-, 17-class).
That is the pairing the analysis uses: models sharing a seed index share the
data order and the initialisation streams, so every cross-granularity contrast
is a within-job contrast on identical test jets (probe.check_alignment gates
it). It also keeps each job at ~4 GB of features instead of one 20-model job.

The 162-class model at seed index 1 is `mtx-l162-s1b` (the 5e-4 repair of the
1e-3 `s1`, which is deliberately excluded everywhere).

Templates are derived from job-probe-physics-v4 and job-eval-labelrec-v3, which
ran. `bc_vs_rest` is NOT in the task list: it needs the windowed |V_cb| caches,
which exist for two granularities only.

TWO PROBE VERSIONS ARE EMITTED, AND v1 IS HISTORY.
v1 ran, its results are committed under experiments/FIGS/data/probe_ladder_v1/,
and its text is therefore frozen: this builder must keep reproducing it byte for
byte or the drift check turns a provenance record into a fiction. v2 re-runs the
same probes with the 70 % and 90 % operating points that
docs/PRESPEC_2026-09.md fixed once 50 % proved censored on bvc_resonant, and
differs from v1 in exactly four places -- job name, output directory, the added
`--eps-s` flag, and the pin. It writes to probe_ladder_v2, so it cannot touch
v1's results. The label-recovery jobs are unrelated to that defect and stay at
v1 and at their original pin.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "EVAL" / "k8s"

PIN = "mtx-s1.48"          # what v1 actually ran at; never bump it
PIN_V2 = "mtx-s1.51"       # the tag carrying --eps-s
EPS_S_V2 = [0.5, 0.7, 0.9]  # 50 % kept so v2 also reproduces v1's number
SEEDS = [1, 2, 3, 4, 5]
# (run-name stem, rung) in ladder order; the rung feeds label_recovery --own-rung
LADDER = [("mtx-l188", "L188"), ("mtx-l162", "L162"),
          ("mtx-r42q1", "R42_Q1"), ("mtx-r16q1", "R16_Q1")]
TASKS = ["bvc_resonant", "retained_topology", "bvc_qcd", "ee_vs_mm",
         "bvc_4prong", "visible_content"]


def run_name(stem: str, seed: int) -> str:
    return f"{stem}-s1b" if (stem == "mtx-l162" and seed == 1) else f"{stem}-s{seed}"


HEAD = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: probe
        image: gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          export PYTHONUNBUFFERED=1
          date -u +"start %Y-%m-%dT%H:%M:%SZ"; nproc; free -g | head -2

          ARMS=""; RUNGS=""
          for spec in {specs}; do
            a=${{spec%%:*}}; r=${{spec##*:}}
            d=/data/results/eval/${{a}}/features_e79
            for f in features.npy label188.npy extract_manifest.json; do
              [ -f "${{d}}/${{f}}" ] || {{ echo "FATAL: no ${{d}}/${{f}}"; exit 1; }}
            done
            ARMS="${{ARMS}} ${{a#mtx-}}=${{d}}"
            RUNGS="${{RUNGS}} ${{a#mtx-}}=${{r}}"
          done
          echo "arms:${{ARMS}}"
"""

PROBE = """
          OUT=/data/results/eval/probe_ladder_{ver}/s{seed}
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/probe.py \\
            --features ${{ARMS}} \\
            --out ${{OUT}} \\
            --tasks {tasks}{eps} \\
            --bootstrap 2000
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

LABELREC = """
          OUT=/data/results/eval/label_recovery_ladder_v1/s{seed}
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/label_recovery.py \\
            --features ${{ARMS}} \\
            --own-rung ${{RUNGS}} \\
            --out ${{OUT}} \\
            --n 200000
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

TAIL = """        volumeMounts:
        - { name: data, mountPath: /data }
        resources:
          requests: { memory: "32Gi", cpu: "8", ephemeral-storage: "10Gi" }
          limits:   { memory: "32Gi", cpu: "8", ephemeral-storage: "10Gi" }
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""


def build() -> dict[str, str]:
    out = {}
    tasks = " ".join(TASKS)
    for seed in SEEDS:
        specs = " ".join(f"{run_name(stem, seed)}:{rung}" for stem, rung in LADDER)
        for kind, body in (("probe-ladder", PROBE), ("labelrec-ladder", LABELREC)):
            name = f"{kind}-v1-s{seed}-raunav"
            text = (HEAD.format(name=name, pin=PIN, specs=specs)
                    + body.format(seed=seed, tasks=tasks, ver="v1", eps="") + TAIL)
            out[f"job-{name}.yaml"] = text
        # The re-run. Same four models, same tasks, same resources; the only
        # differences are the ones listed in the module docstring.
        name = f"probe-ladder-v2-s{seed}-raunav"
        eps = " \\\n            --eps-s " + " ".join(str(e) for e in EPS_S_V2)
        out[f"job-{name}.yaml"] = (
            HEAD.format(name=name, pin=PIN_V2, specs=specs)
            + PROBE.format(seed=seed, tasks=tasks, ver="v2", eps=eps) + TAIL)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a committed spec differs from the generated one")
    args = ap.parse_args()
    bad = 0
    for fname, text in build().items():
        p = K8S / fname
        if args.check:
            if not p.exists() or p.read_text() != text:
                print(f"DRIFT: {p.relative_to(ROOT)}")
                bad += 1
        else:
            p.write_text(text)
            print(f"wrote {p.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
