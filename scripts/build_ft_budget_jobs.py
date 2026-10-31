#!/usr/bin/env python3
"""Fine-tuning below the full training set with a fixed step budget (plan item 3.1, 2026-10-31).

The launched cells at 1e3 and 1e4 jets ran 20 epochs of 19 steps (380 optimizer steps) on top
and quark/gluon tagging and 50 x 19 = 950 on JetClass, far short of convergence at a backbone
learning rate of 1e-4; every pretraining study we compare with trains small-data fine-tuning to
convergence. Below the full set every run now gets the same budget, pretrained or from scratch:

  * 40 weaver epochs of 500 steps (--samples-per-epoch 256,000): at most 20,000 steps at batch 512;
  * a validation pass after each epoch over the whole fixed validation file (200,000 jets), so
    every evaluation sees the same jets;
  * the best-validation checkpoint (weaver's rule), with training stopped once 10 consecutive
    evaluations bring no improvement (experiments/FT/early_stop.py);
  * the training subset s1 for every run, so a from-scratch seed varies only the training seed.

Pretrained runs: the checkpoint's best70 epoch (ft_v2.py resolve), the head re-initialised, the
backbone at --lr-pre with the head at 50x capped at 5e-3, constant rate, weight decay 0.01.
From scratch: --lr-scratch with weaver's flat+decay, weight decay 0.01 (ParT's top recipe). One
fine-tuning per pretraining run (seed 1); three training seeds from scratch.

Kinds of job (each one Indexed Job, V100 only, one cell per index):
  lrsweep   top and quark/gluon at 1e4 jets: 188-s1 at backbone {1e-4, 3e-4, 1e-3} and from
            scratch (seed 1) at {3e-4, 1e-3, 3e-3}; picked on the validation metric
  bench     top and quark/gluon at 1e3, 1e4, 1e5 for every paper run and from scratch
  jc1       JetClass at 1e4 and 1e5 (10^3 dropped, PI 2026-10-31)

Output: /data/results/ft_v2/budget/<tree>/<init>/N<N>/s<S>/ (the sweep under .../lrsweep/),
with early_stop.json, ft_manifest.json, net_best_epoch_state.pt, net_last_epoch_state.pt and the
test outputs the read-outs take: features/logits.npy (top, quark/gluon, plus Herwig) or pred.root
(JetClass). Per-epoch states stay on the pod's local disk and are never written to /data.

    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.23 lrsweep
    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.24 --lr-pre 3e-4 --lr-scratch 1e-3 bench jc1
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "FT" / "k8s" / "budget"
BUDGET_ROOT = "/data/results/ft_v2/budget"
SWEEP_ROOT = "/data/results/ft_v2/lrsweep"
MTX_V2 = "/data/results/mtx_v2"
PAPER_RUNS = ROOT / "configs" / "arms" / "v2_paper_runs.txt"
GPU = "Tesla-V100-SXM2-32GB"
EPOCHS, STEPS, BATCH, PATIENCE, VAL_N = 40, 500, 512, 10, 200_000
HEAD_MULT, HEAD_CAP = 50.0, 5e-3
SWEEP_PRE, SWEEP_SCRATCH, SWEEP_N = ("1e-4", "3e-4", "1e-3"), ("3e-4", "1e-3", "3e-3"), 10_000
SCRATCH_SEEDS = (1, 2, 3)
HERWIG_TEST = ["/data/finetune/qg_herwig/qg_herwig_chunk0.parquet",
               "/data/finetune/qg_herwig/qg_herwig_chunk1.parquet"]
JC1_CLASSES = ("HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q", "TTBar", "TTBarLep", "WToQQ", "ZToQQ",
               "ZJetsToNuNu")
# task -> (subset directory, data config, classes, network config, tree, sizes, test, memory)
TASKS = {
    "top": ("/data/finetune/top_sub", "configs/finetune/TopReference.yaml", 2,
            "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_000, 10_000, 100_000),
            "/data/finetune/top/top_test.parquet", "48Gi"),
    "qg": ("/data/finetune/qg_v2_sub", "configs/finetune/EnergyFlowQG.yaml", 2,
           "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_qg", (1_000, 10_000, 100_000),
           "/data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet", "48Gi"),
    "jc1": ("/data/finetune/jc1", "configs/finetune/JetClassI_sophon_noweight.yaml", 10,
            "experiments/E1/ParT_sophon_arch_10c.py", "leg2", (10_000, 100_000), "JC1_TEST", "64Gi"),
}


def head_mult(lr: str) -> str:
    """The head's multiplier: 50x the backbone rate, capped at 5e-3 for the head."""
    return repr(min(HEAD_MULT, HEAD_CAP / float(lr)))


def paper_runs() -> list:
    return [l.strip()[len("mtx-"):] for l in PAPER_RUNS.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def cell(task: str, init: str, n: int, s: int, lr: str, root: str, sub: str = "") -> dict:
    """One cell. init '' is from scratch; sub names the cell's directory under its init."""
    tree = TASKS[task][4]
    name = init or "scratch"
    leaf = f"{name}{sub}"
    return {"task": task, "init": init, "run_dir": f"{MTX_V2}/mtx-{init}" if init else "", "n": n, "s": s,
            "lr": lr, "mult": head_mult(lr) if init else "", "out": f"{root}/{tree}/{leaf}/N{n}/s{s}"}


def lrsweep_cells() -> list:
    out = []
    for task in ("top", "qg"):
        out += [cell(task, "l188-s1", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_PRE]
        out += [cell(task, "", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_SCRATCH]
    return out


def bench_cells(lr_pre: str, lr_scratch: str, tasks=("top", "qg"), runs=None) -> list:
    runs = paper_runs() if runs is None else runs
    out = []
    for task in tasks:
        sizes = TASKS[task][5]
        out += [cell(task, r, n, 1, lr_pre, BUDGET_ROOT) for n in sizes for r in runs]
        out += [cell(task, "", n, s, lr_scratch, BUDGET_ROOT) for n in sizes for s in SCRATCH_SEEDS]
    return out


def _launch():
    spec = importlib.util.spec_from_file_location("build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CELL_SCRIPT = r"""
HALT=42
IDX=${JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}
case ${IDX} in
__CASES__  *) echo "FATAL: no cell for index ${IDX}"; exit ${HALT};;
esac
[ -f ${CELL}/DONE ] && { echo "skip ${CELL} (DONE)"; exit 0; }
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${USED}" -le 85 ] || { echo "FATAL: /data at ${USED}%"; exit ${HALT}; }
case ${TASK} in
__TASKS__  *) echo "FATAL: task ${TASK}"; exit ${HALT};;
esac
python3 experiments/FT/gpu_probe.py || exit 43
RUN=/workspace/run; rm -rf ${RUN}; mkdir -p ${RUN}
[ -d ${CELL} ] && mv ${CELL} ${CELL}.partial.$(date -u +%s)
mkdir -p ${CELL}
echo "$(date -u +%FT%TZ) pod=${POD_NAME} node=${NODE_NAME} ref=${REPO_REF}" >> ${CELL}/ATTEMPTS
if [ -n "${RUN_DIR}" ]; then
  python3 experiments/FT/ft_v2.py resolve --run-dir ${RUN_DIR} --rule best70 --link /workspace/ckpt/${INIT}.pt || exit ${HALT}
  cp /workspace/ckpt/${INIT}.pt.json ${CELL}/init_checkpoint.json
  LOAD=(--load-model-weights /workspace/ckpt/${INIT}.pt --exclude-model-weights "mod\.fc\..*")
  MULTARG=(--optimizer-option lr_mult "(r'mod\.fc\..*', ${MULT})")
  SCHED=(--lr-scheduler none); RECIPE="head_lr_mult=${MULT} lr_schedule=constant ckpt_rule=best70"
else
  LOAD=(); MULTARG=(); SCHED=(); RECIPE="head_lr_mult=1 lr_schedule=flat+decay recipe=part-from-scratch"
fi
python3 experiments/FT/smoke_checks.py manifest --out ${CELL}/ft_manifest.json task=${TASK} \
  init=${INIT:-scratch} checkpoint=${RUN_DIR} n_train=${N} ft_seed=${S} lr=${LR} ${RECIPE} weight_decay=0.01 \
  subset=${SUB}/train_N${N}_s1.parquet data_config=${CFG} num_classes=${K} batch_size=__BATCH__ \
  samples_per_epoch=$((__STEPS__ * __BATCH__)) steps_per_epoch=__STEPS__ max_epochs=__EPOCHS__ \
  samples_per_epoch_val=__VAL_N__ early_stop_patience=__PATIENCE__ val_rule=fixed-file budget=v3.1
python3 experiments/FT/ft_weaver.py --seed ${S} --lean-val-metrics \
  --data-train ${SUB}/train_N${N}_s1.parquet --data-val ${SUB}/val.parquet --data-config ${CFG} \
  --network-config ${NET} -o num_classes ${K} -o fc_params '[(512,0.1)]' \
  --use-amp --batch-size __BATCH__ --num-workers 2 --fetch-by-files --fetch-step 1 --optimizer ranger \
  ${SCHED[@]+"${SCHED[@]}"} --optimizer-option weight_decay 0.01 ${MULTARG[@]+"${MULTARG[@]}"} \
  --start-lr ${LR} --samples-per-epoch $((__STEPS__ * __BATCH__)) --samples-per-epoch-val __VAL_N__ \
  --num-epochs __EPOCHS__ ${LOAD[@]+"${LOAD[@]}"} --model-prefix ${RUN}/net --log ${RUN}/train.log \
  > ${RUN}/stdout.log 2>&1 &
PID=$!
ES=0
python3 experiments/FT/early_stop.py watch --log ${RUN}/train.log --pid ${PID} --patience __PATIENCE__ \
  --max-epochs __EPOCHS__ --out ${CELL}/early_stop.json || ES=$?
wait ${PID} || true
cp ${RUN}/train.log ${RUN}/stdout.log ${CELL}/
[ "${ES}" -eq 0 ] || { echo "FATAL: ${CELL}: training did not finish (early_stop rc=${ES})"; tail -30 ${RUN}/stdout.log; exit 1; }
grep -q "('steps_per_epoch', __STEPS__)" ${RUN}/train.log || { echo "FATAL: weaver did not run __STEPS__ steps per epoch"; exit 1; }
if grep -q "AvgLoss: nan" ${RUN}/train.log; then echo "FATAL: ${CELL} diverged (NaN training loss)"; exit 1; fi
if [ -n "${RUN_DIR}" ]; then
  python3 experiments/FT/smoke_checks.py load-log --log ${RUN}/stdout.log
  grep -q "Parameters with lr multiplied by" ${RUN}/stdout.log || { echo "FATAL: weaver did not apply the head lr multiplier"; exit 1; }
fi
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_epoch'])" ${CELL}/early_stop.json)
LAST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['last_epoch'])" ${CELL}/early_stop.json)
cmp -s ${RUN}/net_best_epoch_state.pt ${RUN}/net_epoch-${BEST}_state.pt || { echo "FATAL: weaver's best state is not epoch ${BEST}"; exit ${HALT}; }
cp ${RUN}/net_best_epoch_state.pt ${CELL}/net_best_epoch_state.pt
cp ${RUN}/net_epoch-${LAST}_state.pt ${CELL}/net_last_epoch_state.pt
case ${TASK} in
  jc1)
    weaver --predict --data-test ${TEST} --data-config ${CFG} --network-config ${NET} -o num_classes ${K} \
      -o fc_params '[(512,0.1)]' --predict-gpus 0 --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1 \
      --model-prefix ${CELL}/net_best_epoch_state.pt --predict-output ${CELL}/pred.root 2>&1 | tee ${CELL}/predict.log | tail -3
    [ -f ${CELL}/pred.root ] || { echo "FATAL: no pred.root in ${CELL}"; exit 1; }
    python3 experiments/FT/slim_outputs.py pred --file ${CELL}/pred.root;;
  *)
    python3 experiments/EVAL/extract_features.py --checkpoint ${CELL}/net_best_epoch_state.pt --num-classes ${K} \
      --arm FTB_${TASK}_${INIT:-scratch}_N${N}_s${S} --data-config ${CFG} --data-test ${TEST} \
      --observers jet_pt jet_energy --out ${CELL}/features --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
    if [ "${TASK}" = "qg" ]; then
      python3 experiments/EVAL/extract_features.py --checkpoint ${CELL}/net_best_epoch_state.pt --num-classes ${K} \
        --arm FTB_${TASK}_${INIT:-scratch}_N${N}_s${S}_herwig --data-config ${CFG} --data-test __HERWIG__ \
        --observers jet_pt jet_energy --out ${CELL}/features_herwig --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
      mv ${CELL}/features_herwig/logits.npy ${CELL}/features/logits_herwig.npy
      mv ${CELL}/features_herwig/label188.npy ${CELL}/features/label_herwig.npy
    fi;;
esac
touch ${CELL}/DONE
echo "FT BUDGET CELL ${IDX} DONE: ${CELL}"
"""


def task_cases() -> str:
    out = []
    for t, (sub, cfg, k, net, _tree, _sizes, test, _mem) in TASKS.items():
        if test == "JC1_TEST":
            test_block = ("TEST=\"\"; for C in " + " ".join(JC1_CLASSES) + "; do for f in $(ls /data/JetClass/Pythia/"
                          "test_20M/${C}_*.root | sort | head -2); do TEST=\"${TEST} ${f}\"; done; done; "
                          "[ \"$(echo ${TEST} | wc -w)\" -eq 20 ] || { echo \"FATAL: JetClass test files\"; exit ${HALT}; }")
        else:
            test_block = f'TEST="{test}"'
        out.append(f"  {t}) SUB={sub}; CFG={cfg}; K={k}; NET={net}; {test_block};;\n")
    return "".join(out)


def cell_cases(cells: list) -> str:
    return "".join(f"  {i}) TASK={c['task']}; INIT={c['init']}; RUN_DIR={c['run_dir']}; N={c['n']}; S={c['s']}; "
                   f"LR={c['lr']}; MULT={c['mult']}; CELL={c['out']};;\n" for i, c in enumerate(cells))


def spec(name: str, cells: list, tag: str, parallelism: int, memory: str) -> str:
    launch = _launch()
    script = (CELL_SCRIPT.replace("__CASES__", cell_cases(cells)).replace("__TASKS__", task_cases())
              .replace("__STEPS__", str(STEPS)).replace("__BATCH__", str(BATCH)).replace("__EPOCHS__", str(EPOCHS))
              .replace("__VAL_N__", str(VAL_N)).replace("__PATIENCE__", str(PATIENCE))
              .replace("__HERWIG__", " ".join(HERWIG_TEST)))
    script = ("set -euo pipefail\n"
              'git clone -q --depth 1 --branch "${REPO_REF}" https://github.com/raunavm/transferlearningsophon.git '
              "/workspace/transferlearningsophon\ncd /workspace/transferlearningsophon\n"
              "pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1\n" + script.lstrip("\n"))
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES)
    if len(name) > 63:
        raise SystemExit(f"{name}: too long")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_ft_budget_jobs.py -- fine-tuning with the fixed step budget (plan 3.1).
  name: {name}
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(cells)}
  parallelism: {parallelism}
  backoffLimitPerIndex: 3
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
        image: {launch.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        env:
        - name: REPO_REF
          value: "{tag}"
        - name: NODE_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: spec.nodeName }} }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{memory}", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{memory}", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
      tolerations:
      - {{ key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
              - key: nvidia.com/gpu.product
                operator: In
                values: ["{GPU}"]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr-pre", default=None)
    ap.add_argument("--lr-scratch", default=None)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("kinds", nargs="+", choices=("lrsweep", "bench", "jc1"))
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        if kind == "lrsweep":
            cells, mem = lrsweep_cells(), "48Gi"
        else:
            if not (a.lr_pre and a.lr_scratch):
                ap.error(f"{kind} needs --lr-pre and --lr-scratch (from the sweep)")
            cells = bench_cells(a.lr_pre, a.lr_scratch, ("top", "qg") if kind == "bench" else ("jc1",))
            mem = "48Gi" if kind == "bench" else "64Gi"
        name = f"ft-budget-{kind}-raunav"
        p = OUT_DIR / f"job-{name}.yaml"
        p.write_text(spec(name, cells, a.tag, a.parallelism, mem))
        print(p, len(cells), "cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
