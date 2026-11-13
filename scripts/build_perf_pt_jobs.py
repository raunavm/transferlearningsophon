#!/usr/bin/env python3
"""The CPU job of performance against jet pT and the JetClass confusion matrices (figure round 3,
PI 2026-11-06; docs/figstyle/TAGGER_GAP.md): experiments/EVAL/perf_vs_pt.py for every model, one pod.

  --parts jc2  the JetClass-II linear probes at 1e6 (the 27 paper runs) -> OUT_ROOT/jc2/<model>.json
  --parts jc1  the long-budget JetClass cells at 1e6 -> OUT_ROOT/jc1/<model>.json and
               CONF_ROOT/<model>.json; refuses to start (exit 42, not retried) unless all 30 cells
               (build_ft_budget_jobs.long_cells("jc1"): every paper run and three seeds from scratch)
               have DONE
Round 5 (PI 2026-11-13, approved inference on stored outputs; CPU, no forward pass), into new folders:
  --parts jc2ft     JetClass-II fine-tuned at 1e6, the 27 paper runs and the three from-scratch seeds (the cells
                    the paper reports) -> OUT_V3/jc2_ft/<model>.json
  --parts jc1probe  JetClass linear probes at 1e6, the 27 paper runs and the random network (init-s1..3 at init,
                    initbn-s1..3 at init_bn) on the 20 test files the probe extraction read
                    (build_linprobe_jobs.LIST_TEST1) -> OUT_V3/jc1_probe/
  --parts jc1full   the from-scratch JetClass cells the paper reads (build_ft_budget_jobs.long_scratch_full_cells,
                    the full schedule; the files in perf_vs_pt_v2/jc1 come from superseded cells) ->
                    OUT_V3/jc1_fulldecay/scratch-s<S>.json and CONF_FULL/scratch-s<S>.json
A part writes only files that do not exist; /data above the storage line refuses the start (exit 42).
experiments/FIGS/pull_mlst_data.py pulls perf_vs_pt/<task>/ and confusion_jc1/.

    python3 scripts/build_perf_pt_jobs.py --tag mtx-s2.NN --parts jc2 [--pin-not-yet-tagged]
    python3 scripts/build_perf_pt_jobs.py --commit <pushed commit> --parts jc2ft jc1probe jc1full --storage-line 89
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import storage_line  # noqa: E402
REPO_URL = "https://github.com/raunavm/transferlearningsophon.git"
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "perf_pt"
OUT_ROOT, CONF_ROOT = "/data/results/eval/perf_vs_pt_v2", "/data/results/eval/confusion_jc1"   # v2: with accuracy per bin
# (figure round 4); the files of round 3 stay in perf_vs_pt/, and confusion_jc1/ is complete, so jc1 skips it
OUT_V3, CONF_FULL = "/data/results/eval/perf_vs_pt_v3", "/data/results/eval/confusion_jc1_fulldecay"
SCRIPT = "experiments/EVAL/perf_vs_pt.py"
NEEDED = {SCRIPT: "def confusion"}
NEEDED_V3 = {SCRIPT: "INIT_BN_MODELS = "}
PARTS = ("jc2", "jc1", "jc2ft", "jc1probe", "jc1full")
MEM, CPU = "16Gi", "4"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LAUNCH = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
BFB = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")


def long_jc1_cells() -> list[str]:
    """The 30 long-budget JetClass cells (the learning rates do not change the paths)."""
    return [c["out"] for c in BFB.long_cells("jc1", "1e-3", "1e-3")]


def full_jc1_cells() -> list[str]:
    """The three from-scratch JetClass cells at 1e6 with the full schedule, the ones the paper reads."""
    return [c["out"] for c in BFB.long_scratch_full_cells("jc1", "1e-3")]


def pushed_commit(ref: str) -> str:
    """The full sha of ref, which must be on origin/main: the pod clones it from GitHub."""
    sha = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    if not sha:
        raise SystemExit(f"FATAL: {ref} is not a commit here")
    remote = subprocess.run(["git", "branch", "-r", "--contains", sha], cwd=ROOT, capture_output=True, text=True).stdout
    if "origin/main" not in remote:
        raise SystemExit(f"FATAL: {sha[:7]} is not pushed to origin/main; the pod clones it from GitHub")
    return sha


def clone_line(ref: str, commit: bool) -> str:
    """The bash clone function of a pod: a tag shallowly, a commit by full clone and checkout."""
    if commit:
        return (f"clone () {{ mkdir -p /workspace && cd /workspace && rm -rf tls && git clone -q {REPO_URL} tls "
                f"&& git -C tls checkout -q {ref}; }}")
    return (f"clone () {{ mkdir -p /workspace && cd /workspace && rm -rf tls && git clone -q --depth 1 --branch {ref} "
            f"{REPO_URL} tls; }}")


def name(parts) -> str:
    return "perf-pt-" + "-".join(parts) + "-raunav"


def spec(tag: str, parts, commit: bool = False, line: int = storage_line.GENERAL) -> str:
    # the refusals come before any part runs: nothing starts while a cell it reads is unfinished
    guard, calls = [], []
    if "jc1" in parts:
        guard = ["for c in " + " ".join(long_jc1_cells())
                 + '; do [ -e "${c}/DONE" ] || { echo "FATAL: ${c} not DONE"; exit 42; }; done']
    if "jc1full" in parts:
        guard += ["for c in " + " ".join(full_jc1_cells())
                  + '; do [ -e "${c}/DONE" ] || { echo "FATAL: ${c} not DONE"; exit 42; }; done']
    if "jc2" in parts:
        calls += ["retry pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1",
                  f"python3 {SCRIPT} jc2 --out-dir {OUT_ROOT}/jc2"]
    if "jc1" in parts:
        calls += [f"python3 {SCRIPT} jc1 --out-dir {OUT_ROOT}/jc1 --confusion-out {CONF_ROOT}"]
    if "jc2ft" in parts:
        # every paper run needs one finished cell (perf_vs_pt.py refuses otherwise); from scratch all three
        calls += [f"python3 {SCRIPT} jc2ft --out-dir {OUT_V3}/jc2_ft --probe-dir {OUT_ROOT}/jc2"]
    if "jc1full" in parts:
        calls += [f"python3 {SCRIPT} jc1 --tree {BFB.LONG_FULL_ROOT}/leg2 --out-dir {OUT_V3}/jc1_fulldecay "
                  f"--confusion-out {CONF_FULL}"]
    if "jc1probe" in parts:
        # the 20 files of the probe extraction, listed as build_linprobe_jobs.LIST_TEST1 lists them; last,
        # since its row map is checked in the pod and a refusal stops the pod
        calls += ['TEST1=""',
                  f"for C in {' '.join(BFB.JC1_CLASSES)}; do for f in $(ls /data/JetClass/Pythia/test_20M/${{C}}_*.root | sort "
                  '| head -2); do TEST1="${TEST1} ${f}"; done; done',
                  '[ "$(echo ${TEST1} | wc -w)" -eq 20 ] || { echo "FATAL: expected 20 JetClass test files"; exit 42; }',
                  f"python3 {SCRIPT} jc1probe --out-dir {OUT_V3}/jc1_probe --test-files ${{TEST1}}"]
    script = "\n".join([
        "set -euo pipefail",
        "retry () { for T in 1 2 3 4 5; do \"$@\" && return 0; echo \"attempt $T failed: $1\"; sleep 30; done; return 1; }",
        clone_line(tag, commit),
        "retry clone || exit 1",
        "cd /workspace/tls",
        "git rev-parse HEAD",
        f"export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={CPU} OPENBLAS_NUM_THREADS={CPU} MKL_NUM_THREADS={CPU}",
        "USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)",
        f'[ "${{USED}}" -le {line} ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 42; }}',
        *guard,
        *calls,
        f"echo \"PERF PT {' '.join(parts)} COMPLETE\""])
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in LAUNCH.V2_BAD_NODES + LAUNCH.V2_LATER_BAD_NODES)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_perf_pt_jobs.py -- performance against jet pT, JetClass confusion matrices. CPU.
  name: {name(parts)}
  namespace: cms-ml
spec:
  backoffLimit: 1
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {LAUNCH.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "10Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{exclude}]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--tag", help="the pushed tag the pod clones")
    ref.add_argument("--commit", help="the pushed commit the pod clones and checks out")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=["jc2", "jc1"])
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    storage_line.add_argument(ap)
    a = ap.parse_args(argv)
    parts = [p for p in PARTS if p in a.parts]
    needed = NEEDED_V3 if set(parts) - {"jc2", "jc1"} else NEEDED
    pin = pushed_commit(a.commit) if a.commit else a.tag
    _load("build_extract_jobs", "scripts/build_extract_jobs.py").verify_pin(
        pin, list(needed), a.pin_not_yet_tagged and not a.commit, needed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{name(parts)}.yaml"
    p.write_text(spec(pin, parts, bool(a.commit), storage_line.check(a.storage_line)))
    print(f"{p.relative_to(ROOT)}: {' '.join(parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
