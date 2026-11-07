#!/usr/bin/env python3
"""The job of the UMAP figures (PI 2026-11-05; one layout for all four tasks, figure round 3): the
class token of the 188- and 17-class models on each task's test jets, frozen and after fine-tuning on
that task, with experiments/EVAL/extract_features.py (one loader worker; the frozen model reads the
jets with the fine-tuned model's data config, so the rows align), then
experiments/EVAL/umap_class_token.py, into /data/results/eval/<out-name>/. umap-learn and pyarrow are pinned
here (the image has neither). CPU only (2026-11-07): the job is inference, its GPU run held 3 GB and 1.7 cores,
and the us-west GPU queue kept it pending for hours; extract_features.py falls back to the CPU and records the
device in each manifest (the first JetClass extractions of l188-s1 ran on a V100 and are kept). On the CPU one
pod read 68 jets/s, so the job is Indexed: one extraction per pod, in parallel; the pod that finds every
extraction in place (an atomic mkdir decides between two) fits the UMAPs and writes DONE.

  jc1  one JetClass test file per class, every 10th jet; fine-tuned: budget_long/leg2/<m>/N1000000/s1 (K = 10)
  jc2  the class-balanced JetClass-II test set, every 2nd jet (2,458-2,519 of each single-class group, so 2,000
       per group still holds; extract_features.py runs every jet and keeps every STRIDE-th, so the stride
       shrinks the cache, not the compute); fine-tuned: the JetClass-II cell
       at 1e6 the paper reports, budget_long/leg_jc2b/<m>/N1000000/s1 where it is DONE, else
       budget/leg_jc2b/<m>/N1000000/s1, chosen in the pod (K = 188); its test config
  top  the top-tagging test set, every 20th jet; fine-tuned: best70/leg_top/<m>/N1200000/s1 (K = 2)
  qg   the quark/gluon (Pythia) test chunks, every 10th jet; fine-tuned: best70/leg_qg/<m>/N1600000/s1 (K = 2)

The frozen checkpoint is the one the frozen probes read; all four tasks' probe manifests must name it.
(umap_preview and umap_v1 were built by this file's earlier form, at mtx-s2.63 and mtx-s2.71.)

    python3 scripts/build_umap_jobs.py --tag mtx-s2.NN --out-name umap_v2
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "umap"
UMAP_PIN = "umap-learn==0.5.7"
CPUS, THREADS = 8, 7        # one core left for the loader worker and the shell
PARALLEL = 12                # with our GPU pods, under the pod cap of 25
PYARROW_PIN = "pyarrow==25.0.1"   # weaver reads the parquet test sets (JetClass-II, top, q/g) through it and the image
# has none: umap-v2 died on it at the first JetClass-II read, 2026-11-06
MODELS = (("l188-s1", 188), ("r16q1-s1", 17))
FT = "/data/results/ft_v2"
LP_OLD, LP_BAL = "/data/results/eval/v2_linprobe", "/data/results/eval/v2_linprobe_bal"
# task -> (data config, stride, test files, K, fine-tuned cell (None: chosen in the pod), frozen-probe manifest);
# configs and test files as the fine-tuning read-outs (build_ft_budget_jobs.TASKS, TEST_CFG; build_ft_jobs bench)
TASKS = {
    "jc1": ("configs/finetune/JetClassI_sophon_noweight.yaml", 10, "${JC1}", 10,
            FT + "/budget_long/leg2/{m}/N1000000/s1", LP_OLD + "/mtx-{m}/jc1/test/best70/manifest.json"),
    "jc2": ("configs/data/JetClassII_base.yaml", 2, "/data/finetune/jc2_bal/test.parquet", 188,
            None, LP_BAL + "/mtx-{m}/jc2bal/test/best70/manifest.json"),
    "top": ("configs/finetune/TopReference.yaml", 20, "/data/finetune/top/top_test.parquet", 2,
            FT + "/best70/leg_top/{m}/N1200000/s1", LP_OLD + "/mtx-{m}/top/test/best70/manifest.json"),
    "qg": ("configs/finetune/EnergyFlowQG.yaml", 10,
           "/data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet", 2,
           FT + "/best70/leg_qg/{m}/N1600000/s1", LP_OLD + "/mtx-{m}/qg/test/best70/manifest.json"),
}
JC2_ROOTS = (FT + "/budget_long", FT + "/budget")       # the long budget where it ran, else the standard one


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spec(tag: str, out_name: str) -> str:
    launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    classes = " ".join(_load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py").JC1_CLASSES)
    out = f"/data/results/eval/{out_name}"
    blocks, runs = [], []          # every checkpoint is resolved and checked before the first extraction
    for i, (m, k) in enumerate(MODELS):
        mans = " ".join(t[5].format(m=m) for t in TASKS.values())
        # the frozen checkpoint: the one every task's frozen probes read
        blocks += [f"CK{i}=$(python3 -c \"import json, sys; d = [json.load(open(p)) for p in sys.argv[1:]]; "
                   f"assert len({{x['checkpoint_sha256'] for x in d}}) == 1, 'the probes read different checkpoints'; "
                   f"print(d[0]['checkpoint'])\" {mans}) || exit 42",
                   f'echo "mtx-{m} frozen: ${{CK{i}}}"']
        for task, (cfg, stride, files, kt, cell, _) in TASKS.items():
            c = f"C_{task}_{i}"
            blocks.append((f"{c}={cell.format(m=m)}" if cell else
                           f'{c}=$(jc2_cell {m}) || {{ echo "FATAL: no JetClass-II cell of {m} is done"; exit 42; }}')
                          + f"; done_cell ${{{c}}}")
            runs += [f"x {task} frozen mtx-{m} ${{CK{i}}} {k} {cfg} {stride} {files}",
                     f"x {task} ft mtx-{m} ${{{c}}}/net_best_epoch_state.pt {kt} {cfg} {stride} {files}"]
    script = f"""set -euo pipefail
retry () {{ for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt $T failed: $1"; sleep 30; done; return 1; }}
clone () {{ mkdir -p /workspace && cd /workspace && rm -rf tls && git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git tls; }}
retry clone || exit 1
cd /workspace/tls
git rev-parse HEAD
retry pip install --no-cache-dir -q {UMAP_PIN} {PYARROW_PIN} || exit 1
python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
export OMP_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS}      # torch would otherwise size its pool to the host
[ ! -f {out}/DONE ] || {{ echo "{out} done"; exit 0; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
JC1=""; for C in {classes}; do JC1="${{JC1}} $(ls /data/JetClass/Pythia/test_20M/${{C}}_*.root | sort | head -1)"; done
[ "$(echo ${{JC1}} | wc -w)" -eq 10 ] || {{ echo "FATAL: JetClass test files"; exit 42; }}
# the JetClass-II cell at 1e6 the paper reports: the long budget where it is done, else the standard one
jc2_cell () {{ for R in {" ".join(JC2_ROOTS)}; do [ -f ${{R}}/leg_jc2b/$1/N1000000/s1/DONE ] && {{ echo ${{R}}/leg_jc2b/$1/N1000000/s1; return 0; }}; done; return 1; }}
done_cell () {{ [ -f $1/DONE ] && [ -f $1/net_best_epoch_state.pt ] || {{ echo "FATAL: $1 is not done"; exit 42; }}; echo "fine-tuned: $1"; }}
# x TASK KIND MODEL CHECKPOINT K CONFIG STRIDE FILES...: the class token on FILES, every STRIDE-th jet, one loader worker
x () {{
  local D={out}/features/$1/$3/$2 ARM=UMAP_$1_$3_$2 CKPT=$4 K=$5 CFG=$6 STRIDE=$7
  shift 7
  [ -f ${{D}}/extract_manifest.json ] && return 0
  python3 experiments/EVAL/extract_features.py --checkpoint ${{CKPT}} --num-classes ${{K}} --arm ${{ARM}} \\
    --data-config ${{CFG}} --data-test "$@" --observers jet_pt --stride ${{STRIDE}} --num-workers 1 --out ${{D}}
}}
{chr(10).join(blocks)}
case ${{JOB_COMPLETION_INDEX}} in
{chr(10).join(f"  {j}) {r} ;;" for j, r in enumerate(runs))}
  *) echo "FATAL: no extraction for index ${{JOB_COMPLETION_INDEX}}"; exit 42 ;;
esac
N=$(find {out}/features -mindepth 4 -maxdepth 4 -name extract_manifest.json | wc -l)
[ "${{N}}" -eq {len(runs)} ] || {{ echo "extraction ${{JOB_COMPLETION_INDEX}} in; ${{N}} of {len(runs)}"; exit 0; }}
mkdir {out}/fit.lock 2>/dev/null || {{ echo "another pod fits the UMAPs"; exit 0; }}
python3 experiments/EVAL/umap_class_token.py --feat-dir {out}/features --out-dir {out} || {{ rmdir {out}/fit.lock; exit 1; }}
touch {out}/DONE
echo "UMAP COMPLETE"
"""
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_umap_jobs.py -- the UMAP figures (PI 2026-11-05, round 3). Writes new files only.
  name: {out_name.replace("_", "-")}-raunav
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(runs)}
  parallelism: {PARALLEL}
  backoffLimit: 1
  activeDeadlineSeconds: 43200
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
          requests: {{ memory: "48Gi", cpu: "{CPUS}", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "48Gi", cpu: "{CPUS}", ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
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
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-name", required=True, help="the folder under /data/results/eval/")
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{a.out_name.replace('_', '-')}-raunav.yaml"
    p.write_text(spec(a.tag, a.out_name))
    print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
