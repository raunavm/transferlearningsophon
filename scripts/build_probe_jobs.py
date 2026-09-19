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

Two further blocks answer predictions of their own rather than the ladder: the
random-label control (C4) and the granularity x mass-output 2x2 (C5). Each
writes to its own output directory and neither shares a cell key with the
ladder, so no analysis can mistake one for the other.
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

# THE SEMANTICS-MATCHED RANDOM-LABEL CONTROL (prediction C4). This is the answer
# to the objection that the whole study is a tautology -- that we merged two
# labels and then found the distinction they encoded got worse. A random
# partition matched in class-size structure merges just as many classes, but
# merges the WRONG ones, so if coarseness alone drove the effect the control
# would reproduce it and it does not have to.
#
# One job per draw, each holding the draw AND the 17-class model at the SAME
# seed index, because C4 is a within-seed contrast (random draw minus 17-class)
# and probe.check_alignment only gates arms that are inside one job. Three
# draws, not one: different random partitions merge different class pairs, so
# draw-to-draw variation is a different quantity from training-seed variation
# and cannot be estimated from a single draw.
#
# Only the two tasks C4 is defined on. The other four are not part of the
# prediction and would be four more chances to find something.
CONTROL_TASKS = ["bvc_4prong", "visible_content"]
CONTROL_DRAWS = [("mtx-rand-d1-s1b", "mtx-r16q1-s1", 1),
                 ("mtx-rand-d2-s2", "mtx-r16q1-s2", 2),
                 ("mtx-rand-d3-s3", "mtx-r16q1-s3", 3)]

# THE GRANULARITY x MASS-OUTPUT 2x2 (prediction C5, confirmatory). Four models
# per seed index: the 162- and 17-class models, each with and without the added
# jet-mass regression output. C5 is a difference-in-differences -- how much the
# mass output changes the b-versus-c probe at 162 classes, minus how much it
# changes it at 17 -- so all four cells of one seed must be scored on the same
# jets in the same order, which is what putting them in ONE job buys
# (probe.check_alignment only gates arms inside a single job).
#
# The pairing is valid on the seed axis: a mass run and its plain twin at index
# N are both launched with `--seed N`, and seed_weaver derives the four RNG
# sub-streams as sha256("seed-stream|v1|<seed>|<stream>") -- no arm, no class
# count, no run id -- so index N means the same four streams on both sides.
# Both carry the same required GPU-product pin, so invariant I7 holds within
# every pair.
#
# ALL SIX TASKS, not just C5's. The approved plan specifies "same probes" for
# the 2x2, and unlike the random-label control (whose extra tasks would be
# meaningless, because a random partition has no rung) every task here is a
# real measurement on a real vocabulary. Only bvc_resonant is confirmatory --
# docs/PRESPEC_2026-09.md fixed C5 on the b-versus-c probe before any of this
# existed. The other five are exploratory and the analysis labels them so; they
# carry no inferential claim and enter no multiplicity family.
#
# 162 and 17 only. There is no 188-class or 43-class model with a mass output:
# the 2x2 was pretrained at two granularities, which is what makes it a 2x2.
MASS_PIN = "mtx-s1.53"     # contains probe.py at c53e861, the tag v2 also ran
MASS_CELLS = [("mtx-l162", "L162"), ("mtx-l162mass", "L162_MASS"),
              ("mtx-r16q1", "R16_Q1"), ("mtx-r16q1mass", "R16_Q1_MASS")]


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

    # The random-label control, one job per draw.
    ctasks = " ".join(CONTROL_TASKS)
    ceps = " \\\n            --eps-s " + " ".join(str(e) for e in EPS_S_V2)
    for rand, ref, draw in CONTROL_DRAWS:
        name = f"probe-randcontrol-d{draw}-raunav"
        # The rung label is only consumed by label_recovery, which this job does
        # not run; the random arm HAS no rung, which is the point of it.
        cspecs = f"{rand}:RAND {ref}:R16_Q1"
        out[f"job-{name}.yaml"] = (
            HEAD.format(name=name, pin=PIN_V2, specs=cspecs)
            + PROBE.format(seed=f"d{draw}", tasks=ctasks, ver="randcontrol", eps=ceps)
            + TAIL)

    # The mass-output 2x2, one job per seed index. Same tasks and same operating
    # points as v2, so a mass cell and a ladder cell are the same measurement;
    # a SEPARATE output directory, so the four extra arms can never reach the
    # ladder analysis, whose loader would see two arms claiming level 162.
    meps = " \\\n            --eps-s " + " ".join(str(e) for e in EPS_S_V2)
    for seed in SEEDS:
        name = f"probe-mass2x2-s{seed}-raunav"
        mspecs = " ".join(f"{run_name(stem, seed)}:{arm}" for stem, arm in MASS_CELLS)
        out[f"job-{name}.yaml"] = (
            HEAD.format(name=name, pin=MASS_PIN, specs=mspecs)
            + PROBE.format(seed=seed, tasks=tasks, ver="mass2x2", eps=meps) + TAIL)
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
