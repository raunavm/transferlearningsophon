#!/usr/bin/env python3
"""The CPU job of the ATLAS jet-mass resolution (PI decision P7, 2026-11-02):
experiments/EVAL/atlas_resolution.py on every model of the jet-mass table, one model per index
of an Indexed Job, into OUT_ROOT/<model>.json (+ <model>_perjet.npz), and OUT_ROOT/DONE once
every model's JSON exists. experiments/FIGS/pull_mlst_data.py pulls the JSONs to head/atlas_res/.

The models and inputs are scripts/build_head_jobs.py's, which wrote mass_v3.json for the same
jets (HEAD_ROOT/<model>/):
  the 16 runs it reads the output layer of (four label sets x 3, the 162- and 17-class models
  with the auxiliary mass regression x 2): observers and, for the mass-regression models, the
  mass output from HEAD_ROOT/<model>/test/best70/head_outputs.npz; frozen features from
  LP_ROOT/<model>/jc2/test/best70
  the untrained trunks init-s1..s3: the observers of OBS_MODEL's head outputs (the same jets;
  the label digest is checked) and their features at LP_ROOT/<model>/jc2/test/init
An index whose inputs are missing stops with exit 42 (not retried); a signal (OOM, lost node)
is retried, up to backoffLimitPerIndex.

Memory: per model the script holds the 2,000,512 x 128 float16 test features (0.51 GB) as
float32 (1.0 GB), the matched-jet copy (1.0 GB), the split copies and the scaled copies of the
ridge and the MLP (about 4 GB), and five observer columns (0.04 GB; head_outputs.npz's logits
are never loaded): about 7 GB at peak. MEM is what the untrained trunks' mass_resolution.py
jobs, which fit the same probes twice on the same inputs, completed in (build_head_jobs.INIT_MEM).

    python3 scripts/build_atlas_res_job.py --tag mtx-s2.NN [--pin-not-yet-tagged] [--parallelism 5]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "head"
NAME = "atlas-res-raunav"
OUT_ROOT = "/data/results/eval/atlas_res"
SCRIPT = "experiments/EVAL/atlas_resolution.py"
MEM, CPU = "32Gi", "8"
# What the tag must carry for what the job runs.
NEEDED = {SCRIPT: "BOOT_RESAMPLES, BOOT_SEED = 200, 20261102",
          "experiments/EVAL/mass_resolution.py": "OUT_JSON, OUT_RESIDUALS = \"mass_v3.json\""}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HB = _load("build_head_jobs", "scripts/build_head_jobs.py")
LAUNCH = _load("build_mtx_launch", "scripts/build_mtx_launch.py")


def models() -> list[tuple[str, str, str, str, str]]:
    """(model, observers dir, features dir, arm, mass-output dir or "") of every model."""
    out = [(m, f"{HB.HEAD_ROOT}/{m}/test/best70", f"{HB.LP.LP_ROOT}/{m}/jc2/test/best70", "best70",
            f"{HB.HEAD_ROOT}/{m}/test/best70" if reg else "") for m, _run, _rung, _k, reg in HB.models()]
    out += [(m, f"{HB.HEAD_ROOT}/{HB.OBS_MODEL}/test/best70", f"{HB.LP.LP_ROOT}/{m}/jc2/test/init", "init", "")
            for m in HB.init_models()]
    return out


def spec(tag: str, parallelism: int) -> str:
    ms = models()
    cases = "".join(f'  {i}) MODEL={m}; OBS={o}; FEAT={f}; ARM={arm}; HEAD="{h}";;\n'
                    for i, (m, o, f, arm, h) in enumerate(ms))
    script = f"""set -euo pipefail
# a signal (OOM, lost node) is retried; any other failure is deterministic
halt () {{ rc=$?; [ $rc -ge 128 ] && exit $rc; echo "HALT: exit $rc, not retried"; exit 42; }}
retry () {{ for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt $T failed: $1"; sleep 30; done; return 1; }}
clone () {{ mkdir -p /workspace && cd /workspace && rm -rf tls && git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git tls; }}
retry clone || exit 1
cd /workspace/tls
git rev-parse HEAD
export PYTHONUNBUFFERED=1
# BLAS threads = the CPUs requested: nproc reports the node's cores
export OMP_NUM_THREADS={CPU} OPENBLAS_NUM_THREADS={CPU} MKL_NUM_THREADS={CPU}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
OUT={OUT_ROOT}
MASSV3={HB.HEAD_ROOT}/${{MODEL}}/mass_v3.json
if [ -f ${{OUT}}/${{MODEL}}.json ]; then
  echo "${{MODEL}} done"
else
  for F in ${{OBS}}/head_outputs.npz ${{FEAT}}/manifest.json ${{MASSV3}}; do
    [ -f ${{F}} ] || {{ echo "FATAL: no ${{F}}"; exit 42; }}
  done
  USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
  [ "${{USED}}" -lt 85 ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 42; }}
  python3 {SCRIPT} --model ${{MODEL}} --observers ${{OBS}} --features ${{FEAT}} \\
    --arm ${{ARM}} ${{HEAD:+--head ${{HEAD}}}} --mass-v3 ${{MASSV3}} --out-dir ${{OUT}} || halt
fi
ALL=1
for M in {" ".join(m for m, *_ in ms)}; do [ -f ${{OUT}}/${{M}}.json ] || ALL=0; done
[ ${{ALL}} = 0 ] || touch ${{OUT}}/DONE
echo "ATLAS RES ${{MODEL}} COMPLETE"
"""
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in LAUNCH.V2_BAD_NODES + LAUNCH.V2_LATER_BAD_NODES)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_atlas_res_job.py -- ATLAS jet-mass resolution (PI decision P7). CPU.
  name: {NAME}
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(ms)}
  parallelism: {parallelism}
  backoffLimitPerIndex: 6
  podFailurePolicy:
    rules:
    - action: FailIndex
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
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
          requests: {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "20Gi" }}
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
    ap.add_argument("--tag", required=True, help="the pushed tag the pods clone")
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--parallelism", type=int, default=5)
    a = ap.parse_args(argv)
    HB.BX.verify_pin(a.tag, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{NAME}.yaml"
    p.write_text(spec(a.tag, a.parallelism))
    print(f"{p.relative_to(ROOT)}: {len(models())} models into {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
