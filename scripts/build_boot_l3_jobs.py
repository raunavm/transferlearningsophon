#!/usr/bin/env python3
"""The CPU job of the test-jet bootstraps behind decision L3 (PI 2026-11-13: a difference is claimed when
|delta| > 3 sigma, sigma^2 = the SD^2 of each side over its runs + the bootstrap SD^2 over test jets), on stored
outputs only, so that every main-text and mass-effect comparison of the label sets has a test-jet term: each label
set minus 188 classes (162 - 188 among them) and each mass-output model minus its label set (162 + mass - 162,
17 + mass - 17). One Indexed Job, one part per pod, every output new under OUT (an existing file is skipped, so a
retried pod does only what is missing):

  0 ft-jc2b    experiments/STATS/paired_bootstrap.py --task jc2b --headline: JetClass-II fine-tuned at 1e6, the cell
               the paper reports (the long budget where it is DONE, else the fixed budget; mlst_results.merge_budget)
  1 sculpt     experiments/EVAL/sculpting_boot.py on build_head_jobs.SCULPT_V3, with its 162 - 188 row and the "l3"
               block, replica 0 checked against the sculpting_v3 documents
  2 probe-jc   paired_bootstrap.py --headline on the v4 linear probes at 1e6: class-balanced JetClass-II and JetClass
  3 ft-jc1     paired_bootstrap.py --task jc1_all --headline: JetClass fine-tuned at 1e6 (long budget), every test jet
  4 jc1-acc    experiments/FT/jc1_acc_boot.py at 1e6 with the mass pairs and the "l3" block, replica 0 checked
               against budget_long_metrics.json
  5 ft-top     paired_bootstrap.py --task top --headline (R30): fine-tuned on the full top training set
  6 ft-qg      paired_bootstrap.py --task qg and qg_herwig --headline: fine-tuned on the full quark/gluon set
  7 probe-bin  paired_bootstrap.py --headline on the linear probes of top and quark/gluon (Pythia, Herwig) at the
               full training sets under the substitutions chosen on validation (build_gallery_data_job: best70.itop4,
               best70.nqg)

    python3 scripts/build_boot_l3_jobs.py --commit <pushed commit> --storage-line 89
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import storage_line  # noqa: E402
OUT_DIR = ROOT / "experiments" / "STATS" / "k8s"
OUT = "/data/results/bootstrap/l3"
NAME = "boot-l3-raunav"
FT = "/data/results/ft_v2"
PROBS, FILL_PROBS = "/data/results/eval/v2_linprobe_bal/fits_v4_probs", "/data/results/eval/v2_linprobe_bal/fits_v4_fill_probs"
# arm -> run indices (configs/arms/v2_paper_runs.txt)
ARMS = {"l188": (1, 2, 3), "l162": (1, 2, 3), "r42q1": (1, 2, 3), "r16q1": (1, 2, 3), "l162mass": (1, 2), "r16q1mass": (1, 2)}
# name -> (configuration, reference)
DIFFS = {"162-188": ("l162", "l188"), "43-188": ("r42q1", "l188"), "17-188": ("r16q1", "l188"),
         "162mass-162": ("l162mass", "l162"), "17mass-17": ("r16q1mass", "r16q1")}
REPLICAS = 200
MEM, CPU, PARALLEL, DEADLINE_S = "48Gi", "4", 4, 43200
NEEDED = {"experiments/STATS/paired_bootstrap.py": "def group_diff", "experiments/FT/jc1_acc_boot.py": "def l3",
          "experiments/EVAL/sculpting_boot.py": "def unpaired"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runs() -> list[str]:
    return [f"mtx-{a}-s{k}" for a, ks in ARMS.items() for k in ks]


def diff_args() -> str:
    side = lambda arm: ",".join(f"mtx-{arm}-s{k}" for k in ARMS[arm])  # noqa: E731
    return " ".join(f"--diff {n}={side(a)}/{side(b)}" for n, (a, b) in DIFFS.items())


def pb(task: str, models: str, out: str, extra: str = "") -> str:
    """One paired_bootstrap.py call into OUT/out, skipped when that file exists."""
    return (f"[ -e {OUT}/{out} ] || python3 experiments/STATS/paired_bootstrap.py --task {task} {models} --headline "
            f"{diff_args()}{extra} --b {REPLICAS} --out {OUT}/{out}")


def cells(tree: str, n: int) -> str:
    """--model mtx-<run>=<tree>/<run>/N<n>/s1 for every run."""
    return " ".join(f"--model {m}={tree}/{m.removeprefix('mtx-')}/N{n}/s1" for m in runs())


def probes(probs: str, ck: str, file: str) -> str:
    return " ".join(f"--model {m}={probs}/{m}/{ck}/{file}.npz" for m in runs())


def parts() -> list[tuple[str, list[str]]]:
    hb = _load("build_head_jobs", "scripts/build_head_jobs.py")
    jc2b = ["ARGS=()",
            "for R in " + " ".join(runs()) + "; do",
            f"  C={FT}/budget_long/leg_jc2b/${{R#mtx-}}/N1000000/s1",
            f"  [ -f ${{C}}/DONE ] || C={FT}/budget/leg_jc2b/${{R#mtx-}}/N1000000/s1",
            '  [ -f ${C}/DONE ] || { echo "FATAL: ${C} is not done"; exit 42; }',
            '  echo "${R}: ${C}"; ARGS+=(--model ${R}=${C})',
            "done",
            pb("jc2b", '"${ARGS[@]}"', "ft_jc2bal_N1000000.json")]
    sets = " ".join(f"{m}={arm}" for m, _r, arm, _k, _g in hb.flav_models())
    sculpt = [f"python3 experiments/EVAL/sculpting_boot.py --head-root {hb.HEAD_ROOT} --models {' '.join(hb.SCULPT_V3)} "
              f"--label-set {sets} --replicas {REPLICAS} --sculpting-dir {hb.SCULPT_V3_DIR} --out {OUT}/sculpting_boot_l3.json"]
    probe_jc = [pb("npz", probes(PROBS, "best70", f), f"probe_{f}.json") for f in ("jc2bal_N1000000", "jc1_N1000000")]
    jc1 = [pb("jc1_all", cells(f"{FT}/budget_long/leg2", 1_000_000), "ft_jc1_N1000000.json")]
    f = f"{OUT}/jc1_acc_boot_N1000000.json"       # written under .new and moved, so a pull never takes a part
    acc = [f"[ -e {f} ] || {{ python3 experiments/FT/jc1_acc_boot.py --root {FT}/budget_long --n 1000000 "
           f"--metrics {FT}/budget_long/budget_long_metrics.json --replicas {REPLICAS} --out {f}.new && mv -n {f}.new {f}; }}"]
    top = [pb("top", cells(f"{FT}/best70/leg_top", 1_200_000), "ft_top_N1200000.json")]
    qg = [pb(t, cells(f"{FT}/best70/leg_qg", 1_600_000), f"ft_{t}_N1600000.json") for t in ("qg", "qg_herwig")]
    probe_bin = [pb("npz", probes(FILL_PROBS, ck, f), f"probe_{f}.{ck.split('.')[1]}.json")
                 for ck, f in (("best70.itop4", "top_N1200000"), ("best70.nqg", "qg_N1600000"),
                               ("best70.nqg", "qg_herwig_N1600000"))]
    return [("ft-jc2b", jc2b), ("sculpt", sculpt), ("probe-jc", probe_jc), ("ft-jc1", jc1), ("jc1-acc", acc),
            ("ft-top", top), ("ft-qg", qg), ("probe-bin", probe_bin)]


def spec(commit: str, line: int = storage_line.GENERAL) -> str:
    launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    clone = _load("build_perf_pt_jobs", "scripts/build_perf_pt_jobs.py").clone_line(commit, True)
    ps = parts()
    case = []
    for i, (label, lines) in enumerate(ps):
        case += [f"  {i})  # {label}"] + [f"    {ln}" for ln in lines] + ["    ;;"]
    script = "\n".join([
        "set -euo pipefail",
        "retry () { for T in 1 2 3 4 5; do \"$@\" && return 0; echo \"attempt $T failed: $1\"; sleep 30; done; return 1; }",
        clone,
        "retry clone || exit 1",
        "cd /workspace/tls",
        "git rev-parse HEAD",
        f"export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={CPU} OPENBLAS_NUM_THREADS={CPU} MKL_NUM_THREADS={CPU}",
        "USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)",
        f'[ "${{USED}}" -le {line} ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 42; }}',
        f"mkdir -p {OUT}",
        "case ${JOB_COMPLETION_INDEX} in",
        *case,
        '  *) echo "FATAL: no part for index ${JOB_COMPLETION_INDEX}"; exit 42 ;;',
        "esac",
        'echo "BOOT L3 PART ${JOB_COMPLETION_INDEX} COMPLETE"'])
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_boot_l3_jobs.py -- test-jet bootstraps for decision L3. CPU only; writes new files only.
  name: {NAME}
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(ps)}
  parallelism: {PARALLEL}
  backoffLimit: 1
  activeDeadlineSeconds: {DEADLINE_S}
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {launch.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "5Gi" }}
          limits:   {{ memory: "{MEM}", cpu: "{CPU}", ephemeral-storage: "5Gi" }}
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
    ap.add_argument("--commit", required=True, help="the pushed commit the pods clone")
    storage_line.add_argument(ap)
    a = ap.parse_args(argv)
    commit = _load("build_perf_pt_jobs", "scripts/build_perf_pt_jobs.py").pushed_commit(a.commit)
    _load("build_extract_jobs", "scripts/build_extract_jobs.py").verify_pin(commit, list(NEEDED), False, NEEDED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{NAME}.yaml"
    p.write_text(spec(commit, storage_line.check(a.storage_line)))
    print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
