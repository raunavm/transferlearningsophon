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
import re
import subprocess
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
# THE v2 PROBE TASKS: every task of probe.TASKS, on the v2 caches, which keep every
# task's rows (extract_v2.probe_feature_rules): the v1 ladder's six, the |V_cb|
# probe inside its window, and the two single-pair b-vs-c tasks (X->bc vs X->bq,
# X->bc vs X->cs, no window; probe.py, 2026-10-01). No v2 probe spec is emitted
# before the v2 caches exist; tests/test_probe_jobs.py keeps this list equal to
# probe.TASKS and covered by the extraction.
V2_TASKS = TASKS + ["bc_vs_rest", "bc_vs_bq", "bc_vs_cs"]

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

# S7, the frozen-feature jet-mass regression. SIX models per seed, not the 2x2's
# four: S7's third clause is about the granularity ladder WITHOUT the mass
# output ("without the mass output, finer labels give equal or better
# resolution"), so the two granularities that have no mass twin are needed too.
#
# What "resolution" means here was pre-registered before any of these numbers
# existed -- docs/PRESPEC_2026-09.md, amendment 2026-09-20, and
# DECISIONS_PENDING item 40. The readout refuses to run unless every model's
# cached-label digest matches the generator-level mass cache's, which is the one
# alignment the observer job could not check for the mass-output models.
MASSRES_PIN = "mtx-s1.54"      # the tag that first carries mass_resolution.py
MASSRES_OBS = "/data/results/eval/test2m_observers"
MASSRES_CELLS = [("mtx-l188", "L188"), ("mtx-l162", "L162"),
                 ("mtx-r42q1", "R42_Q1"), ("mtx-r16q1", "R16_Q1"),
                 ("mtx-l162mass", "L162_MASS"), ("mtx-r16q1mass", "R16_Q1_MASS")]


FEAT = "features_e79"           # the 2,000,000-jet caches every job above reads

# S10, the |V_cb| discriminant probe, 162 against 17 classes (docs/PRESPEC_2026-09.md,
# clarification of 2026-09-27). The task lives inside the published window, so
# it reads the windowed caches over the whole test split, not the 2 M ones.
# ONE job holding all ten models: every seed index is paired, and
# probe.check_alignment only gates the arms inside one job. Same pin as the 2x2,
# whose probe.py is the one every other probe in the paper ran.
VCB_FEAT = "features_vcbwindow_e79_full"
VCB_CELLS = [("mtx-l162", "L162"), ("mtx-r16q1", "R16_Q1")]
VCB_EPS_S = [0.6, 0.4]          # arXiv:2503.00118's working points, as the task pins

# THE MLP PROBE RE-FITTED TO A PLATEAU ("mlp2"). Every probe run above capped the
# MLP at 60 epochs, and fits were still improving when they hit it: 93 of the
# ladder's 360, 140 of the 2x2's 360, 22 of the random control's 36 and 1 of the
# |V_cb| probe's 30 -- mostly on the 17-class models, which is where the
# vocabularies are compared. probe.py now trains to a plateau (MLP_SCHEDULE) and
# --mlp-rerun-of re-fits only the MLP on the caches each run read, copying every
# linear number from that run's own output. One re-run per spec whose MLP the
# paper reads, derived from it by substitution: same models, caches and
# resources; only the name, the pin and the probe call move, and it writes to a
# new directory beside the old one, which it refuses to find already written.
MLP2_PIN = "mtx-s1.64"          # tagged after the commit: build with --pin-not-yet-tagged
MLP2_NEEDED = {"experiments/EVAL/probe.py": "--mlp-rerun-of"}
MLP2_SOURCES = ([f"probe-ladder-v2-s{s}-raunav" for s in SEEDS]
                + [f"probe-randcontrol-d{d}-raunav" for _, _, d in CONTROL_DRAWS]
                + [f"probe-mass2x2-s{s}-raunav" for s in SEEDS]
                + ["probe-vcbwindow-s10-raunav"])

MLP2 = """          SRC={src}/probe_results.json
          [ -f "${{SRC}}" ] || {{ echo "FATAL: no ${{SRC}}"; exit 1; }}
          OUT={out}
          [ ! -e "${{OUT}}/probe_results.json" ] || {{ echo "FATAL: ${{OUT}}/probe_results.json exists"; exit 1; }}
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/probe.py \\
            --features ${{ARMS}} \\
            --out ${{OUT}} \\
            --mlp-rerun-of ${{SRC}} \\
            --bootstrap 2000
"""


def mlp2_name(name: str) -> str:
    """probe-ladder-v2-s1-raunav -> probe-ladder-v2-mlp2-s1-raunav."""
    stem, last = name.removesuffix("-raunav").rsplit("-", 1)
    return f"{stem}-mlp2-{last}-raunav"


def mlp2_spec(text: str) -> str:
    """The re-run of one probe spec: its probe call swapped for the MLP-only one,
    reading that spec's own output, writing to <its directory>_mlp2."""
    name = re.search(r"^  name: (\S+)$", text, re.M).group(1)
    pin = re.search(r'--branch "([^"]+)"', text).group(1)
    call = re.search(r"^          OUT=(\S+)/(\S+)\n.*?^            --bootstrap 2000\n",
                     text, re.M | re.S)
    subs = [(f"  name: {name}\n", f"  name: {mlp2_name(name)}\n"),
            (f'--branch "{pin}"', f'--branch "{MLP2_PIN}"'),
            (call.group(0), MLP2.format(src=f"{call.group(1)}/{call.group(2)}",
                                        out=f"{call.group(1)}_mlp2/{call.group(2)}"))]
    for a, b in subs:
        if text.count(a) != 1:
            raise SystemExit(f"FATAL: {name} changed shape; cannot derive its re-run ({a[:40]!r})")
        text = text.replace(a, b)
    return text


def verify_pin(pin: str, not_yet_tagged: bool, flags: dict) -> None:
    """The pod clones the TAG, so the flags a spec passes must be in the tag's copy
    of the script (build_aoj_jobs.verify_pin). A tag made after the commit cannot
    exist while its specs are written; --pin-not-yet-tagged then reads the working
    tree instead and says so."""
    tagged = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify",
                             f"refs/tags/{pin}"], capture_output=True).returncode == 0
    if not tagged and not not_yet_tagged:
        raise SystemExit(f"FATAL: tag {pin} does not exist. Pass --pin-not-yet-tagged if it "
                         f"is about to be created, and create it BEFORE applying any spec.")
    for path, flag in flags.items():
        text = ((ROOT / path).read_text() if not tagged else subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{pin}:{path}"],
            capture_output=True, text=True).stdout)
        if flag not in text:
            raise SystemExit(f"FATAL: {path} at {pin if tagged else 'the working tree'} has no {flag}")
    if not tagged:
        print(f"WARNING: tag {pin} DOES NOT EXIST YET. Create it on a commit carrying "
              f"{sorted(flags)} before applying any spec that clones it.")


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
            d=/data/results/eval/${{a}}/{feat}
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

MASSRES = """
          OBS={obs}
          for f in observers.npz label188.npy observers_manifest.json; do
            [ -f "${{OBS}}/${{f}}" ] || {{ echo "FATAL: no ${{OBS}}/${{f}}"; exit 1; }}
          done
          OUT=/data/results/eval/mass_resolution/s{seed}
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/mass_resolution.py \\
            --features ${{ARMS}} \\
            --observers ${{OBS}} \\
            --out ${{OUT}}
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


# ======================================================= v1 errors (audit 2026-09-29)
# The committed probe, mass-probe and label-recovery numbers carry no error on a
# ratio of two models, because no per-jet output was kept. These jobs refit the
# same probes on the same features with the per-jet outputs saved
# (probe.py --save-scores, mass_resolution.py --save-residuals), run the new
# label-recovery learning curve, and bootstrap the fine-tuning cells; the
# ratios are then formed by experiments/STATS/paired_errors.py. Outputs go under
# /data/results/eval/v1err/, never over a committed result.
#
# THE RETRY POLICY OF COMMIT 3cb4d7a, IN ITS CPU FORM. An evicted pod (node
# drained, preempted) is not counted (DisruptionTarget -> Ignore); a pod killed
# by a signal -- OOM, a lost node -- is counted up to V1ERR_BACKOFF; a Python
# failure is deterministic and exits 42, which fails the Job at once instead of
# burning the retries. Every step skips work a previous attempt finished.
V1ERR_PIN = "mtx-s1.66"
# The label-recovery curve and the fine-tuning bootstrap had not been applied when
# label_recovery_curve.py (summary mode) and paired_errors.py (ratio side) changed
# after mtx-s1.66, so they take the next tag; the probe and mass reruns, applied
# from mtx-s1.66 and unchanged since, keep it (tests/test_spec_pins.py).
V1ERR_PIN2 = "mtx-s1.71"
V1ERR_BACKOFF = 6
# Batch B again, at the tag whose mass-resolution MLP fits at a pinned thread count
# (latent_scale_probe.MLP_THREADS). Unpinned, batch B's refit did not reproduce the
# committed MLP sigma_eff (up to 0.0021 in one model; every ridge value agreed to
# 6e-6). A new output directory: batch B's results stay as they are.
V1ERR_PIN_MASS2 = "mtx-s1.84"
V1ERR_MASS2_NEEDED = {"experiments/EVAL/latent_scale_probe.py": "MLP_THREADS",
                      "experiments/EVAL/mass_resolution.py": "--save-residuals"}
THREADS_OF = {False: 8, True: 1}   # BLAS threads: the CPUs a spec requests; 1 per worker in the pooled FT job
V1ERR_ROOT = "/data/results/eval/v1err"
V1ERR_PROBE_SOURCES = ([f"probe-ladder-v2-s{s}-raunav" for s in SEEDS]
                       + [f"probe-randcontrol-d{d}-raunav" for _, _, d in CONTROL_DRAWS]
                       + [f"probe-mass2x2-s{s}-raunav" for s in SEEDS]
                       + ["probe-vcbwindow-s10-raunav"])
V1ERR_MASSRES_SOURCES = [f"massres-s{s}-raunav" for s in SEEDS]
V1ERR_NEEDED = {"experiments/EVAL/probe.py": "--save-scores",
                "experiments/EVAL/mass_resolution.py": "--save-residuals",
                "experiments/EVAL/label_recovery_curve.py": "--mlp-rungs",
                "experiments/STATS/paired_errors.py": "ft-replicates",
                "src/stats/paired.py": "def paired_ratio"}
CURVE_SIZES = [14_000, 44_000, 140_000, 443_000, 0]     # 0 = the whole training pool
FT_LEGS = {"leg1": ("/data/results/ft/w2b/leg1", "/data/results/ft/w2b_leg1_metrics_v2/leg1_metrics.json"),
           "leg2": ("/data/results/ft/w2b/leg2", "/data/results/ft/w2b_leg2_metrics_v2/leg2_metrics.json")}

ROBUST_HEAD = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: {backoff}
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          # a signal (OOM, lost node) is retried; any other failure is deterministic
          halt () {{ rc=$?; [ $rc -ge 128 ] && exit $rc; echo "HALT: exit $rc, not retried"; exit 42; }}
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          export PYTHONUNBUFFERED=1
          # BLAS threads = the CPUs requested: nproc reports the node's cores
          export OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
          date -u +"start %Y-%m-%dT%H:%M:%SZ"; nproc; free -g | head -2
"""

ARMS_LOOP = """
          ARMS=""; RUNGS=""
          for spec in {specs}; do
            a=${{spec%%:*}}; r=${{spec##*:}}
            d=/data/results/eval/${{a}}/{feat}
            for f in features.npy label188.npy extract_manifest.json; do
              [ -f "${{d}}/${{f}}" ] || {{ echo "FATAL: no ${{d}}/${{f}}"; exit 42; }}
            done
            ARMS="${{ARMS}} ${{a#mtx-}}=${{d}}"
            RUNGS="${{RUNGS}} ${{a#mtx-}}=${{r}}"
          done
          echo "arms:${{ARMS}}"
"""

ROBUST_TAIL = """        volumeMounts:
        - {{ name: data, mountPath: /data }}
        resources:
          requests: {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "10Gi" }}
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

V1ERR_PROBE = """
          OUT={out}
          if [ -f "${{OUT}}/scores.npz" ] && [ -f "${{OUT}}/probe_results.json" ]; then
            echo "done by an earlier attempt"; exit 0
          fi
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/probe.py \\
            --features ${{ARMS}} \\
            --out ${{OUT}} \\
            --tasks {tasks} \\
            --eps-s {eps} \\
            --bootstrap 2000 \\
            --save-scores || halt
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

V1ERR_CURVE = """
          # the four models side by side, four threads each; resumable, since
          # label_recovery_curve.py keeps every finished cell and skips it
          pids=""
          for spec in {specs}; do
            a=${{spec%%:*}}; r=${{spec##*:}}; m=${{a#mtx-}}
            mkdir -p {out}/${{m}}
            OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
            python3 experiments/EVAL/label_recovery_curve.py \\
              --features ${{m}}=/data/results/eval/${{a}}/{feat} \\
              --own-rung ${{m}}=${{r}} \\
              --out {out}/${{m}} \\
              --sizes {sizes} \\
              --mlp-rungs L188 \\
              --threads 4 > {out}/${{m}}.log 2>&1 &
            pids="${{pids}} $!"
          done
          for p in ${{pids}}; do wait ${{p}} || halt; done
          tail -n 3 {out}/*.log
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

def v1err_name(name: str) -> str:
    """probe-ladder-v2-s1-raunav -> probe-ladder-v2-v1err-s1-raunav."""
    stem, last = name.removesuffix("-raunav").rsplit("-", 1)
    return f"{stem}-v1err-{last}-raunav"


def _field(text: str, pattern: str) -> str:
    m = re.findall(pattern, text, re.M)
    if len(m) != 1:
        raise SystemExit(f"FATAL: a source spec changed shape ({pattern!r} found {len(m)} times)")
    return m[0]


def v1err_probe_spec(text: str) -> str:
    """The rerun of one committed probe spec with its per-jet scores saved: the
    same models, features, tasks and working points, into v1err/."""
    name = _field(text, r"^  name: (\S+)$")
    specs = _field(text, r"^          for spec in (.+); do$")
    feat = _field(text, r"^            d=/data/results/eval/\$\{a\}/(\S+)$")
    out = _field(text, r"^          OUT=/data/results/eval/(\S+)$")
    tasks = _field(text, r"^            --tasks (.+) \\$")
    eps = _field(text, r"^            --eps-s (.+) \\$")
    return (ROBUST_HEAD.format(name=v1err_name(name), pin=V1ERR_PIN, backoff=V1ERR_BACKOFF, threads=THREADS_OF[v1err_name(name).startswith("paired-ft")])
            + ARMS_LOOP.format(specs=specs, feat=feat)
            + V1ERR_PROBE.format(out=f"{V1ERR_ROOT}/{out}", tasks=tasks, eps=eps)
            + ROBUST_TAIL.format(mem="32Gi", cpu="8"))


def v1err_curve_spec(seed: int) -> tuple[str, str]:
    """One job per seed index, its four models fitted side by side (16 CPUs):
    one pod slot instead of four on a cluster whose pod cap is the constraint."""
    specs = " ".join(f"{run_name(stem, seed)}:{rung}" for stem, rung in LADDER)
    name = f"labelrec-curve-v1err-s{seed}-raunav"
    return name, (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN2, backoff=V1ERR_BACKOFF, threads=4)
                  + ARMS_LOOP.format(specs=specs, feat=FEAT)
                  + V1ERR_CURVE.format(out=f"{V1ERR_ROOT}/label_recovery_curve", specs=specs,
                                       feat=FEAT, sizes=" ".join(map(str, CURVE_SIZES)))
                  + ROBUST_TAIL.format(mem="64Gi", cpu="16"))


# BATCHED, for a cluster whose pod cap is the constraint (2026-09-30: 29-30 raunav
# pods active against a cap of 25 for hours, so seven single-purpose pods would
# wait all night). Batch A runs the class count, the fine-tuning bootstrap and
# the five probe reruns not yet applied in ONE pod; batch B the five mass
# reruns side by side. Same scripts, same outputs, same skip-if-done checks as
# the single specs above, which are therefore not applied.
BATCH_A_PROBES = [f"probe-randcontrol-d{d}-raunav" for _, _, d in CONTROL_DRAWS] + [
    "probe-vcbwindow-s10-raunav", "probe-mass2x2-s5-raunav"]
RUN_PROBE = """
          run_probe () {{   # SPECS FEAT OUT TASKS EPS; background, so its exit is its status
            local ARMS=""
            for spec in $1; do
              a=${{spec%%:*}}; d=/data/results/eval/${{a}}/$2
              for f in features.npy label188.npy extract_manifest.json; do
                [ -f "${{d}}/${{f}}" ] || {{ echo "FATAL: no ${{d}}/${{f}}"; exit 42; }}
              done
              ARMS="${{ARMS}} ${{a#mtx-}}=${{d}}"
            done
            if [ -f "$3/scores.npz" ] && [ -f "$3/probe_results.json" ]; then exit 0; fi
            mkdir -p $3
            OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \\
            python3 experiments/EVAL/probe.py --features ${{ARMS}} --out $3 \\
              --tasks $4 --eps-s $5 --bootstrap 2000 --save-scores > $3.log 2>&1
          }}
"""
RUN_MASSRES = """
          run_massres () {{   # SPECS OUT; background
            local ARMS=""
            for spec in $1; do
              a=${{spec%%:*}}; d=/data/results/eval/${{a}}/{feat}
              for f in features.npy label188.npy extract_manifest.json; do
                [ -f "${{d}}/${{f}}" ] || {{ echo "FATAL: no ${{d}}/${{f}}"; exit 42; }}
              done
              ARMS="${{ARMS}} ${{a#mtx-}}=${{d}}"
            done
            if [ -f "$2/residuals.npz" ] && [ -f "$2/mass_resolution.json" ]; then exit 0; fi
            mkdir -p $2
            OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \\
            python3 experiments/EVAL/mass_resolution.py --features ${{ARMS}} \\
              --observers {obs} --out $2 --save-residuals > $2.log 2>&1
          }}
"""


def v1err_batch_a(base: dict[str, str]) -> tuple[str, str]:
    """Class count, then the fine-tuning bootstrap, then five probe reruns in parallel."""
    bx = _load_builder("build_extract_jobs")
    name = "v1err-batch-a-raunav"
    body = ["\n          pip install --no-cache-dir -q pyarrow",
            "          OUTC=/data/results/eval/v1err/class_counts/test_class_counts.json",
            "          if [ ! -f ${OUTC} ]; then",
            "            python3 experiments/EVAL/class_counts.py --data-test "
            + bx.interleaved_files() + " --out ${OUTC} || halt",
            "          fi"]
    L1, L2 = FT_LEGS["leg1"], FT_LEGS["leg2"]
    ft = (f"\n          OUT={V1ERR_ROOT}/ft\n"
          '          if ! { [ -f "${OUT}/replicates_ft.npz" ] && [ -f "${OUT}/replicates_ft.json" ]; }; then\n'
          "            mkdir -p ${OUT}\n"
          "            python3 experiments/STATS/paired_errors.py ft-replicates \\\n"
          f"              --leg1-root {L1[0]} --leg1-metrics {L1[1]} \\\n"
          f"              --leg2-root {L2[0]} --leg2-metrics {L2[1]} \\\n"
          "              --procs 14 --out ${OUT}/replicates_ft.npz || halt\n"
          "          fi\n")
    calls = []
    for src in BATCH_A_PROBES:
        s = base[f"job-{src}.yaml"]
        specs = _field(s, r"^          for spec in (.+); do$")
        feat = _field(s, r"^            d=/data/results/eval/\$\{a\}/(\S+)$")
        out = _field(s, r"^          OUT=/data/results/eval/(\S+)$")
        tasks = _field(s, r"^            --tasks (.+) \\$")
        eps = _field(s, r"^            --eps-s (.+) \\$")
        calls.append(f'          run_probe "{specs}" {feat} {V1ERR_ROOT}/{out} "{tasks}" "{eps}" &\n'
                     f'          P="$P $!"')
    text = (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN2, backoff=V1ERR_BACKOFF, threads=1)
            + "\n".join(body) + "\n" + ft + RUN_PROBE.format()
            + '          P=""\n' + "\n".join(calls) + "\n"
            + "          for p in ${P}; do wait ${p} || halt; done\n"
            + '          date -u +"end %Y-%m-%dT%H:%M:%SZ"\n')
    tail = ROBUST_TAIL.format(mem="64Gi", cpu="16")
    tail = tail.replace("        - { name: data, mountPath: /data }\n",
                        "        - { name: data, mountPath: /data }\n"
                        "        - { name: jc2,  mountPath: /jc2, readOnly: true }\n")
    tail = tail.replace("      volumes:\n", "      volumes:\n      - name: jc2\n"
                        "        persistentVolumeClaim:\n          claimName: tn-pvc-base-jetclass2\n"
                        "          readOnly: true\n")
    return name, text + tail


# BATCH A2 replaces batch A (applied 2026-09-30 04:18Z, deleted 04:57Z): its
# fine-tuning bootstrap ran ~40 min per cell at B = 1000 on its node (264 cells,
# ~12 h), kept nothing until the end, and held the five probe reruns behind it.
# A2 runs the bootstrap at B = 200 (the test-sample SD is then known to ~5 %,
# enough for an error bar), keeps every finished cell (--cache), and runs the
# probe reruns beside it. The class count batch A wrote is reused.
V1ERR_PIN3 = "mtx-s1.75"
FT_B = 200


def v1err_batch_a2(base: dict[str, str]) -> tuple[str, str]:
    """The fine-tuning bootstrap (resumable) and the five probe reruns, side by side."""
    name = "v1err-batch-a2-raunav"
    L1, L2 = FT_LEGS["leg1"], FT_LEGS["leg2"]
    ft = (f"\n          FT={V1ERR_ROOT}/ft_b{FT_B}\n"
          "          run_ft () {   # background\n"
          '            if [ -f "${FT}/replicates_ft.npz" ] && [ -f "${FT}/replicates_ft.json" ]; then exit 0; fi\n'
          "            mkdir -p ${FT}/cells\n"
          "            python3 experiments/STATS/paired_errors.py ft-replicates \\\n"
          f"              --leg1-root {L1[0]} --leg1-metrics {L1[1]} \\\n"
          f"              --leg2-root {L2[0]} --leg2-metrics {L2[1]} \\\n"
          f"              --b {FT_B} --procs 10 --cache ${{FT}}/cells \\\n"
          "              --out ${FT}/replicates_ft.npz > ${FT}.log 2>&1\n"
          "          }\n")
    calls = ['          run_ft &\n          P="$P $!"']
    for src in BATCH_A_PROBES:
        s = base[f"job-{src}.yaml"]
        specs = _field(s, r"^          for spec in (.+); do$")
        feat = _field(s, r"^            d=/data/results/eval/\$\{a\}/(\S+)$")
        out = _field(s, r"^          OUT=/data/results/eval/(\S+)$")
        tasks = _field(s, r"^            --tasks (.+) \\$")
        eps = _field(s, r"^            --eps-s (.+) \\$")
        calls.append(f'          run_probe "{specs}" {feat} {V1ERR_ROOT}/{out} "{tasks}" "{eps}" &\n'
                     f'          P="$P $!"')
    text = (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN3, backoff=V1ERR_BACKOFF, threads=1)
            + ft + RUN_PROBE.format() + '          P=""\n' + "\n".join(calls) + "\n"
            + "          for p in ${P}; do wait ${p} || halt; done\n"
            + '          date -u +"end %Y-%m-%dT%H:%M:%SZ"\n')
    return name, text + ROBUST_TAIL.format(mem="80Gi", cpu="24")


def v1err_batch_b(base: dict[str, str]) -> tuple[str, str]:
    """The five mass reruns side by side."""
    name = "v1err-batch-b-raunav"
    calls = []
    for src in V1ERR_MASSRES_SOURCES:
        s = base[f"job-{src}.yaml"]
        specs = _field(s, r"^          for spec in (.+); do$")
        out = _field(s, r"^          OUT=/data/results/eval/(\S+)$")
        calls.append(f'          run_massres "{specs}" {V1ERR_ROOT}/{out} &\n          P="$P $!"')
    text = (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN, backoff=V1ERR_BACKOFF, threads=1)
            + RUN_MASSRES.format(feat=FEAT, obs=MASSRES_OBS) + '          P=""\n'
            + "\n".join(calls) + "\n          for p in ${P}; do wait ${p} || halt; done\n"
            + '          date -u +"end %Y-%m-%dT%H:%M:%SZ"\n')
    return name, text + ROBUST_TAIL.format(mem="80Gi", cpu="16")


def v1err_batch_b2(base: dict[str, str]) -> tuple[str, str]:
    """Batch B with the MLP at a pinned thread count, into mass_resolution_pinned/."""
    _, text = v1err_batch_b(base)
    name = "v1err-batch-b2-raunav"
    subs = [("v1err-batch-b-raunav", name), (f'--branch "{V1ERR_PIN}"', f'--branch "{V1ERR_PIN_MASS2}"'),
            (f"{V1ERR_ROOT}/mass_resolution/", f"{V1ERR_ROOT}/mass_resolution_pinned/")]
    for old, new in subs:
        if old not in text:
            raise SystemExit(f"FATAL: batch B has no {old!r}")
        text = text.replace(old, new)
    return name, text


# B2's MLP must not depend on the pod either: seed 1's six models again at B2's tag
# on another node than B2's, into mass_resolution_pinned_check/; every sigma_eff
# must equal B2's.
MASS2_NODE = "cph-dgx-node6.humboldt.edu"
# mtx-s1.85, not B2's mtx-s1.84: mass_resolution.py gained only the v2-cache paths
# between the two (v2_prefix(), which returns None for these v1 caches), so the
# computation is B2's, and the spec is not pinned behind a script it runs.
MASS2_CHECK_PIN = "mtx-s1.85"


def _node_term(text: str, key: str, op: str, value: str) -> str:
    """`text` with one more required node-affinity term beside the region's."""
    anchor = '                values: ["us-west"]\n'
    if text.count(anchor) != 1:
        raise SystemExit("FATAL: no single region term to add a node term beside")
    return text.replace(anchor, anchor + f"              - key: {key}\n                operator: {op}\n"
                                         f'                values: ["{value}"]\n')


def _massres_seed1(name: str, pin: str, out: str, mem: str = "32Gi", cpu: str = "6") -> str:
    specs = _field(_BASE["job-massres-s1-raunav.yaml"], r"^          for spec in (.+); do$")
    return (ROBUST_HEAD.format(name=name, pin=pin, backoff=V1ERR_BACKOFF, threads=1)
            + RUN_MASSRES.format(feat=FEAT, obs=MASSRES_OBS) + '          P=""\n'
            + f'          run_massres "{specs}" {V1ERR_ROOT}/{out}/s1 &\n'
            + '          P="$P $!"\n          for p in ${P}; do wait ${p} || halt; done\n'
            + '          date -u +"end %Y-%m-%dT%H:%M:%SZ"\n' + ROBUST_TAIL.format(mem=mem, cpu=cpu))


_BASE: dict[str, str] = {}


def v1err_mass_pinned_check(base: dict[str, str]) -> tuple[str, str]:
    """Seed 1 of batch B2 again, away from B2's node."""
    _BASE.update(base)
    name = "v1err-mass-pinned-check-raunav"
    return name, _node_term(_massres_seed1(name, MASS2_CHECK_PIN, "mass_resolution_pinned_check"),
                            "kubernetes.io/hostname", "NotIn", MASS2_NODE)


# Batch B3: B2 with the CPU code path fixed as well (latent_scale_probe's
# CPU_REPRODUCIBLE_ENV), on Intel nodes, into mass_resolution_cpufixed/; its seed 1
# is repeated on an AMD node (v1err-mass-cpufixed-check) and must agree bit for bit.
# The pinned check showed a fixed thread count alone does not: Intel B2 and AMD
# check differed by up to 1.0e-3.
V1ERR_PIN_MASS3 = "mtx-s1.97"
VENDOR = "feature.node.kubernetes.io/cpu-model.vendor_id"


def v1err_batch_b3(base: dict[str, str]) -> tuple[str, str]:
    _, text = v1err_batch_b2(base)
    name = "v1err-batch-b3-raunav"
    for old, new in (("v1err-batch-b2-raunav", name),
                     (f'--branch "{V1ERR_PIN_MASS2}"', f'--branch "{V1ERR_PIN_MASS3}"'),
                     (f"{V1ERR_ROOT}/mass_resolution_pinned/", f"{V1ERR_ROOT}/mass_resolution_cpufixed/")):
        if old not in text:
            raise SystemExit(f"FATAL: batch B2 has no {old!r}")
        text = text.replace(old, new)
    return name, _node_term(text, VENDOR, "In", "Intel")


def v1err_mass_cpufixed_check(base: dict[str, str]) -> tuple[str, str]:
    _BASE.update(base)
    name = "v1err-mass-cpufixed-check-raunav"
    return name, _node_term(_massres_seed1(name, V1ERR_PIN_MASS3, "mass_resolution_cpufixed_check"),
                            VENDOR, "In", "AMD")


def _load_builder(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_v1err(base: dict[str, str]) -> dict[str, str]:
    """The v1-error specs. The probe reruns applied one per pod (the ladder and
    the 2x2 at seed indices 1-4) keep their single specs as their record; the
    reruns and jobs that never ran singly are emitted only inside batches A and
    B (the single specs for them were removed 2026-09-30, never applied)."""
    out = {}
    for name in V1ERR_PROBE_SOURCES:
        if name not in BATCH_A_PROBES:
            out[f"job-{v1err_name(name)}.yaml"] = v1err_probe_spec(base[f"job-{name}.yaml"])
    for seed in SEEDS:
        name, text = v1err_curve_spec(seed)
        out[f"job-{name}.yaml"] = text
    for fn in (v1err_batch_a, v1err_batch_a2, v1err_batch_b, v1err_batch_b2, v1err_mass_pinned_check,
               v1err_batch_b3, v1err_mass_cpufixed_check):
        name, text = fn(base)
        out[f"job-{name}.yaml"] = text
    bx = _load_builder("build_extract_jobs")
    return {f: bx.storage_guarded(f, t) for f, t in out.items()}


def build() -> dict[str, str]:
    out = {}
    tasks = " ".join(TASKS)
    for seed in SEEDS:
        specs = " ".join(f"{run_name(stem, seed)}:{rung}" for stem, rung in LADDER)
        for kind, body in (("probe-ladder", PROBE), ("labelrec-ladder", LABELREC)):
            name = f"{kind}-v1-s{seed}-raunav"
            text = (HEAD.format(name=name, feat=FEAT, pin=PIN, specs=specs)
                    + body.format(seed=seed, tasks=tasks, ver="v1", eps="") + TAIL)
            out[f"job-{name}.yaml"] = text
        # The re-run. Same four models, same tasks, same resources; the only
        # differences are the ones listed in the module docstring.
        name = f"probe-ladder-v2-s{seed}-raunav"
        eps = " \\\n            --eps-s " + " ".join(str(e) for e in EPS_S_V2)
        out[f"job-{name}.yaml"] = (
            HEAD.format(name=name, feat=FEAT, pin=PIN_V2, specs=specs)
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
            HEAD.format(name=name, feat=FEAT, pin=PIN_V2, specs=cspecs)
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
            HEAD.format(name=name, feat=FEAT, pin=MASS_PIN, specs=mspecs)
            + PROBE.format(seed=seed, tasks=tasks, ver="mass2x2", eps=meps) + TAIL)

    # S10, all ten windowed models in one job.
    name = "probe-vcbwindow-s10-raunav"
    vspecs = " ".join(f"{run_name(stem, seed)}:{arm}" for stem, arm in VCB_CELLS for seed in SEEDS)
    veps = " \\\n            --eps-s " + " ".join(str(e) for e in VCB_EPS_S)
    out[f"job-{name}.yaml"] = (
        HEAD.format(name=name, feat=VCB_FEAT, pin=MASS_PIN, specs=vspecs)
        + PROBE.format(seed="all", tasks="bc_vs_rest", ver="vcbwindow", eps=veps) + TAIL)

    # S7's mass regression, one job per seed index, six models each.
    for seed in SEEDS:
        name = f"massres-s{seed}-raunav"
        rspecs = " ".join(f"{run_name(stem, seed)}:{arm}" for stem, arm in MASSRES_CELLS)
        out[f"job-{name}.yaml"] = (
            HEAD.format(name=name, feat=FEAT, pin=MASSRES_PIN, specs=rspecs)
            + MASSRES.format(seed=seed, obs=MASSRES_OBS) + TAIL)

    # The MLP re-runs, one per probe spec above whose MLP the paper reads.
    for name in MLP2_SOURCES:
        out[f"job-{mlp2_name(name)}.yaml"] = mlp2_spec(out[f"job-{name}.yaml"])
    bx = _load_builder("build_extract_jobs")
    return {f: bx.storage_guarded(f, t) for f, t in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a committed spec differs from the generated one")
    ap.add_argument("--pin-not-yet-tagged", action="store_true",
                    help=f"check the working tree instead of {V1ERR_PIN}, which is "
                         f"tagged after the commit")
    args = ap.parse_args()
    verify_pin(MLP2_PIN, False, MLP2_NEEDED)
    verify_pin(V1ERR_PIN, args.pin_not_yet_tagged, V1ERR_NEEDED)
    verify_pin(V1ERR_PIN_MASS2, args.pin_not_yet_tagged, V1ERR_MASS2_NEEDED)
    verify_pin(V1ERR_PIN_MASS3, args.pin_not_yet_tagged,
               {"experiments/EVAL/latent_scale_probe.py": "CPU_REPRODUCIBLE_ENV",
                "experiments/EVAL/mass_resolution.py": "cpu_reproducible()"})
    bad = 0
    jobs = build()
    jobs.update(build_v1err(jobs))
    for fname, text in jobs.items():
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
