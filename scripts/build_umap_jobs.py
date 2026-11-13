#!/usr/bin/env python3
"""The job of the UMAP figures (PI 2026-11-05; one layout for all four tasks, figure round 3): the
class token of the models of MODELS on each task's test jets, frozen and after fine-tuning on
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
(umap_preview and umap_v1 were built by this file's earlier form, at mtx-s2.63 and mtx-s2.71; umap_v2 with the
188- and 17-class models only, at mtx-s2.7x.)

Round 5 (decision L12, PI 2026-11-13): MODELS is pretraining run 1 of the four label sets and of the two mass-output
models (their frozen checkpoints read with --num-reg 1; a fine-tuned cell has a fresh head without the mass output).
--reuse <out-name> takes REUSABLE's extractions from that earlier output instead of repeating them: the pod that
fits links them into this output's feature tree (symbolic links; nothing of the earlier output is changed), and
umap_class_token.py checks that they hold the same jets as the new ones. The fit also embeds the X->bb / X->bs
evaluation jets from experiments/XBS/extract_xbx.py's stored class token (sets xbb and xbs, frozen only; no extraction).
The pod clones a pushed commit (--commit) and refuses to start above the storage line.

    python3 scripts/build_umap_jobs.py --commit <pushed commit> --out-name umap_v3 --reuse umap_v2 --storage-line 89
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import storage_line  # noqa: E402
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "umap"
UMAP_PIN = "umap-learn==0.5.7"
CPUS, THREADS = 8, 7        # one core left for the loader worker and the shell
PARALLEL = 12                # with our GPU pods, under the pod cap of 25
PYARROW_PIN = "pyarrow==25.0.1"   # weaver reads the parquet test sets (JetClass-II, top, q/g) through it and the image
# has none: umap-v2 died on it at the first JetClass-II read, 2026-11-06
# (run, classes, regression outputs of the pretraining head)
MODELS = (("l188-s1", 188, 0), ("l162-s1", 162, 0), ("r42q1-s1", 43, 0), ("r16q1-s1", 17, 0),
          ("l162mass-s1", 162, 1), ("r16q1mass-s1", 17, 1))
REUSABLE = ("l188-s1", "r16q1-s1")       # extracted by umap_v2 with these very configs, files and cells
XBX_ROOT, XBX_SPLITS = "/data/results/eval/xbx", ("xbb", "xbs", "qcd")   # umap_class_token.XBX_SPLITS
DEADLINE_S = 86400                       # 32 extractions, 12 at a time; the JetClass-II ones take hours each
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
NEEDED = {"experiments/EVAL/umap_class_token.py": "def run_xbx(a, names)",
          "experiments/EVAL/extract_features.py": "--num-reg"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spec(commit: str, out_name: str, reuse: str | None = None, line: int = storage_line.GENERAL) -> str:
    launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    classes = " ".join(_load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py").JC1_CLASSES)
    clone = _load("build_perf_pt_jobs", "scripts/build_perf_pt_jobs.py").clone_line(commit, True)
    out = f"/data/results/eval/{out_name}"
    old = f"/data/results/eval/{reuse}" if reuse else None
    reused = [m for m, _k, _r in MODELS if old and m in REUSABLE]
    blocks, runs = [], {t: [] for t in TASKS}   # every checkpoint is resolved and checked before the first extraction
    for i, (m, k, reg) in enumerate(MODELS):
        if m in reused:
            blocks.append(f"for D in {old}/features/{{{','.join(TASKS)}}}/mtx-{m}/{{frozen,ft}}; do "
                          '[ -f ${D}/extract_manifest.json ] || { echo "FATAL: no ${D}"; exit 42; }; done')
            continue
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
            runs[task] += [f"x {task} frozen mtx-{m} ${{CK{i}}} {k} {reg} {cfg} {stride} {files}",
                           f"x {task} ft mtx-{m} ${{{c}}}/net_best_epoch_state.pt {kt} 0 {cfg} {stride} {files}"]
    # the X->bx class token the fit reads, for every model
    blocks.append(f"for D in {XBX_ROOT}/mtx-{{{','.join(m for m, _k, _r in MODELS)}}}/{{{','.join(XBX_SPLITS)}}}/eval/best70; do "
                  '[ -f ${D}/manifest.json ] || { echo "FATAL: no ${D}"; exit 42; }; done')
    # the longest extractions first: JetClass-II (every jet of 903,922 through the network), JetClass, top, q/g
    runs = runs["jc2"] + runs["jc1"] + runs["top"] + runs["qg"]
    links = ([f"for D in {old}/features/{{{','.join(TASKS)}}}/mtx-{{{','.join(reused)}}}/{{frozen,ft}}; do "
              f'L={out}/features/${{D#{old}/features/}}; mkdir -p $(dirname ${{L}}); [ -e ${{L}} ] || ln -s ${{D}} ${{L}}; done']
             if reused else [])
    script = f"""set -euo pipefail
retry () {{ for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt $T failed: $1"; sleep 30; done; return 1; }}
{clone}
retry clone || exit 1
cd /workspace/tls
git rev-parse HEAD
retry pip install --no-cache-dir -q {UMAP_PIN} {PYARROW_PIN} || exit 1
python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
export OMP_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS}      # torch would otherwise size its pool to the host
[ ! -f {out}/DONE ] || {{ echo "{out} done"; exit 0; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le {line} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
JC1=""; for C in {classes}; do JC1="${{JC1}} $(ls /data/JetClass/Pythia/test_20M/${{C}}_*.root | sort | head -1)"; done
[ "$(echo ${{JC1}} | wc -w)" -eq 10 ] || {{ echo "FATAL: JetClass test files"; exit 42; }}
# the JetClass-II cell at 1e6 the paper reports: the long budget where it is done, else the standard one
jc2_cell () {{ for R in {" ".join(JC2_ROOTS)}; do [ -f ${{R}}/leg_jc2b/$1/N1000000/s1/DONE ] && {{ echo ${{R}}/leg_jc2b/$1/N1000000/s1; return 0; }}; done; return 1; }}
done_cell () {{ [ -f $1/DONE ] && [ -f $1/net_best_epoch_state.pt ] || {{ echo "FATAL: $1 is not done"; exit 42; }}; echo "fine-tuned: $1"; }}
# x TASK KIND MODEL CHECKPOINT K REG CONFIG STRIDE FILES...: the class token on FILES, every STRIDE-th jet, one loader
# worker; REG the checkpoint head's regression outputs after its K classes
x () {{
  local D={out}/features/$1/$3/$2 ARM=UMAP_$1_$3_$2 CKPT=$4 K=$5 REG=$6 CFG=$7 STRIDE=$8
  shift 8
  [ -f ${{D}}/extract_manifest.json ] && return 0
  python3 experiments/EVAL/extract_features.py --checkpoint ${{CKPT}} --num-classes ${{K}} --num-reg ${{REG}} --arm ${{ARM}} \\
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
{chr(10).join(links)}
python3 experiments/EVAL/umap_class_token.py --feat-dir {out}/features --out-dir {out} || {{ rmdir {out}/fit.lock; exit 1; }}
touch {out}/DONE
echo "UMAP COMPLETE"
"""
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_umap_jobs.py -- the UMAP figures (PI 2026-11-05; round 5, L12). Writes new files only.
  name: {out_name.replace("_", "-")}-raunav
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(runs)}
  parallelism: {PARALLEL}
  backoffLimit: 1
  activeDeadlineSeconds: {DEADLINE_S}
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
    ap.add_argument("--commit", required=True, help="the pushed commit the pods clone")
    ap.add_argument("--out-name", required=True, help="the folder under /data/results/eval/ (new)")
    ap.add_argument("--reuse", help="an earlier output whose REUSABLE extractions are linked in, not repeated")
    storage_line.add_argument(ap)
    a = ap.parse_args(argv)
    if a.reuse == a.out_name:
        raise SystemExit("FATAL: --reuse names the output itself")
    perf = _load("build_perf_pt_jobs", "scripts/build_perf_pt_jobs.py")
    commit = perf.pushed_commit(a.commit)
    _load("build_extract_jobs", "scripts/build_extract_jobs.py").verify_pin(commit, list(NEEDED), False, NEEDED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{a.out_name.replace('_', '-')}-raunav.yaml"
    p.write_text(spec(commit, a.out_name, a.reuse, storage_line.check(a.storage_line)))
    print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
