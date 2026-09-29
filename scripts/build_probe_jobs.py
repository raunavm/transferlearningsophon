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
V1ERR_BACKOFF = 6
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

V1ERR_MASSRES = """
          OBS={obs}
          OUT={out}
          if [ -f "${{OUT}}/residuals.npz" ] && [ -f "${{OUT}}/mass_resolution.json" ]; then
            echo "done by an earlier attempt"; exit 0
          fi
          mkdir -p ${{OUT}}
          python3 experiments/EVAL/mass_resolution.py \\
            --features ${{ARMS}} \\
            --observers ${{OBS}} \\
            --out ${{OUT}} \\
            --save-residuals || halt
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

V1ERR_CURVE = """
          OUT={out}
          mkdir -p ${{OUT}}
          # resumable: label_recovery_curve.py keeps every finished cell and skips it
          python3 experiments/EVAL/label_recovery_curve.py \\
            --features ${{ARMS}} \\
            --own-rung ${{RUNGS}} \\
            --out ${{OUT}} \\
            --sizes {sizes} \\
            --mlp-rungs L188 \\
            --threads {cpu} || halt
          date -u +"end %Y-%m-%dT%H:%M:%SZ"
"""

V1ERR_FT = """
          OUT={out}
          if [ -f "${{OUT}}/replicates_ft.npz" ] && [ -f "${{OUT}}/replicates_ft.json" ]; then
            echo "done by an earlier attempt"; exit 0
          fi
          mkdir -p ${{OUT}}
          python3 experiments/STATS/paired_errors.py ft-replicates \\
            --leg1-root {leg1_root} --leg1-metrics {leg1_metrics} \\
            --leg2-root {leg2_root} --leg2-metrics {leg2_metrics} \\
            --procs {procs} --out ${{OUT}}/replicates_ft.npz || halt
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


def v1err_massres_spec(text: str) -> str:
    name = _field(text, r"^  name: (\S+)$")
    specs = _field(text, r"^          for spec in (.+); do$")
    feat = _field(text, r"^            d=/data/results/eval/\$\{a\}/(\S+)$")
    out = _field(text, r"^          OUT=/data/results/eval/(\S+)$")
    obs = _field(text, r"^          OBS=(\S+)$")
    return (ROBUST_HEAD.format(name=v1err_name(name), pin=V1ERR_PIN, backoff=V1ERR_BACKOFF, threads=THREADS_OF[v1err_name(name).startswith("paired-ft")])
            + ARMS_LOOP.format(specs=specs, feat=feat)
            + V1ERR_MASSRES.format(obs=obs, out=f"{V1ERR_ROOT}/{out}")
            + ROBUST_TAIL.format(mem="32Gi", cpu="8"))


def v1err_curve_spec(stem: str, rung: str, seed: int) -> tuple[str, str]:
    """One model per job: the largest fit (1.4 M jets x 188 classes) is hours."""
    run = run_name(stem, seed)
    name = f"labelrec-curve-v1err-{run.removeprefix('mtx-')}-raunav"
    return name, (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN, backoff=V1ERR_BACKOFF, threads=THREADS_OF[name.startswith("paired-ft")])
                  + ARMS_LOOP.format(specs=f"{run}:{rung}", feat=FEAT)
                  + V1ERR_CURVE.format(out=f"{V1ERR_ROOT}/label_recovery_curve/{run.removeprefix('mtx-')}",
                                       sizes=" ".join(map(str, CURVE_SIZES)), cpu=8)
                  + ROBUST_TAIL.format(mem="32Gi", cpu="8"))


def v1err_ft_spec() -> tuple[str, str]:
    name = "paired-ft-v1err-raunav"
    return name, (ROBUST_HEAD.format(name=name, pin=V1ERR_PIN, backoff=V1ERR_BACKOFF, threads=THREADS_OF[name.startswith("paired-ft")])
                  + V1ERR_FT.format(out=f"{V1ERR_ROOT}/ft", procs=14,
                                    leg1_root=FT_LEGS["leg1"][0], leg1_metrics=FT_LEGS["leg1"][1],
                                    leg2_root=FT_LEGS["leg2"][0], leg2_metrics=FT_LEGS["leg2"][1])
                  + ROBUST_TAIL.format(mem="48Gi", cpu="16"))


def build_v1err(base: dict[str, str]) -> dict[str, str]:
    out = {}
    for name in V1ERR_PROBE_SOURCES:
        out[f"job-{v1err_name(name)}.yaml"] = v1err_probe_spec(base[f"job-{name}.yaml"])
    for name in V1ERR_MASSRES_SOURCES:
        out[f"job-{v1err_name(name)}.yaml"] = v1err_massres_spec(base[f"job-{name}.yaml"])
    for seed in SEEDS:
        for stem, rung in LADDER:
            name, text = v1err_curve_spec(stem, rung, seed)
            out[f"job-{name}.yaml"] = text
    name, text = v1err_ft_spec()
    out[f"job-{name}.yaml"] = text
    return out


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
    return out


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
