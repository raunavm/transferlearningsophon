#!/usr/bin/env python3
"""Fine-tuning below the full training set with a fixed step budget (plan item 3.1, 2026-10-31).

The launched cells at 1e3 and 1e4 jets ran 20 epochs of 19 steps (380 optimizer steps) on top
and quark/gluon tagging and 50 x 19 = 950 on JetClass, far short of convergence at a backbone
learning rate of 1e-4; every pretraining study we compare with trains small-data fine-tuning to
convergence. Below the full set every run now gets the same budget, pretrained or from scratch:

  * at most 20,000 steps at batch 512: 40 weaver epochs of 500 steps (--samples-per-epoch 256,000)
    at 1e5 jets and above, 400 epochs of 50 steps (25,600) below. The first sweep (500-step epochs,
    2026-11-01) had its best validation at the first check in 11 of 12 cells: 500 steps are 25
    passes over 1e4 jets, so the check came after the best and could not resolve it (PI: check
    every 50 steps below 1e5);
  * a validation pass after each epoch over the whole fixed validation file (200,000 jets), so
    every evaluation sees the same jets;
  * the best-validation checkpoint (weaver's rule), with training stopped once 10 consecutive
    evaluations bring no improvement (experiments/FT/early_stop.py);
  * the training subset s1 for every run, so a from-scratch seed varies only the training seed.

Pretrained runs: the checkpoint's best70 epoch (ft_v2.py resolve), the head re-initialised, the
backbone at --lr-pre with the head at 50x capped at 5e-3, constant rate, weight decay 0.01.
From scratch: --lr-scratch with weaver's flat+decay, weight decay 0.01 (ParT's top recipe). One
fine-tuning per pretraining run (seed 1); three training seeds from scratch.

Kinds of job (each one Indexed Job, any 24 GB+ GPU (build_mtx_launch.V2_FT_GPU), one cell per index):
  lrsweep   top and quark/gluon at 1e4 jets: 188-s1 at backbone {1e-4, 3e-4, 1e-3} and from
            scratch (seed 1) at {3e-4, 1e-3, 3e-3}; picked on the validation metric
  bench     top and quark/gluon at 1e3, 1e4, 1e5 for every paper run and from scratch
  jc1       JetClass at 1e4 and 1e5 (10^3 dropped, PI 2026-10-31)
  long      10^6 jets with four times the budget (80,000 steps, 160 epochs of 500; PI 2026-11-05, reversing
            the figure plan's decision 18: the from-scratch runs were still improving at the limit). Under
            /data/results/ft_v2/budget_long/. JetClass-II: from scratch and the three pretrained runs that
            reached the limit; the others stopped early under the same rule and stand. JetClass: every run,
            since all had the old fixed 10 epochs with no stopping rule, and one recipe per figure. From
            scratch keeps flat+decay, whose decay starts at 70%, so patience counts from there
  fillcheck top at the full set from 188-s1, zero fill and mean fill (experiments/FT/fill_config.py),
            ParT's recipe: does the fill move fine-tuning? (PI 2026-11-01: stays zero-fill if R30
            moves by less than the seed spread)

Output: /data/results/ft_v2/budget/<tree>/<init>/N<N>/s<S>/ (the sweep under .../lrsweep/),
with early_stop.json, ft_manifest.json, net_best_epoch_state.pt, net_last_epoch_state.pt and the
test outputs the read-outs take: features/logits.npy (top, quark/gluon, plus Herwig) or pred.root
(JetClass). Per-epoch states stay on the pod's local disk and are never written to /data.

    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.23 lrsweep
    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.24 --lr-pre 3e-4 --lr-scratch 1e-3 bench jc1
    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.60 --lr-pre 1e-3 --lr-scratch 1e-3 long-jc2b long-jc1
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "FT" / "k8s" / "budget"
BUDGET_ROOT = "/data/results/ft_v2/budget"
FILLCHECK_ROOT = "/data/results/ft_v2/fillcheck"
LONG_ROOT = "/data/results/ft_v2/budget_long"
LONG_EPOCHS, LONG_N, LONG_JC2B = 160, 1_000_000, ("r16q1-s1", "r16q1mass-s1", "r16q1mass-s2")
SWEEP_ROOT = "/data/results/ft_v2/lrsweep_v2"     # the 500-step sweep stays in .../lrsweep (RUNS.csv)
MTX_V2 = "/data/results/mtx_v2"
PAPER_RUNS = ROOT / "configs" / "arms" / "v2_paper_runs.txt"
EPOCHS, STEPS, BATCH, PATIENCE, VAL_N = 40, 500, 512, 10, 200_000
SMALL_N, SMALL_STEPS = 100_000, 50       # below SMALL_N a check every 50 steps, same 20,000-step cap
HEAD_MULT, HEAD_CAP = 50.0, 5e-3
MEANFILL_TOP = "/workspace/meanfill/TopReference_meanfill.yaml"
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
    "topkin": ("/data/finetune/top_sub", "configs/finetune/TopReference_kin.yaml", 2,
               "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_000, 10_000, 100_000),
               "/data/finetune/top/top_test.parquet", "48Gi"),
    # top with the ten absent slots at the model's input-BatchNorm mean: the config is written in the
    # pod from the resolved checkpoint (experiments/FT/fill_config.py)
    "topmean": ("/data/finetune/top_sub", MEANFILL_TOP, 2,
                "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_200_000,),
                "/data/finetune/top/top_test.parquet", "48Gi"),
    "qg": ("/data/finetune/qg_v2_sub", "configs/finetune/EnergyFlowQG.yaml", 2,
           "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_qg", (1_000, 10_000, 100_000),
           "/data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet", "48Gi"),
    "jc1": ("/data/finetune/jc1", "configs/finetune/JetClassI_sophon_noweight.yaml", 10,
            "experiments/E1/ParT_sophon_arch_10c.py", "leg2", (10_000, 100_000), "JC1_TEST", "64Gi"),
    # the class-balanced JetClass-II sets (stage_jc2_balanced.py): trained on the 188 native classes,
    # scored on every jet of the balanced test set through the native-label config
    "jc2b": ("/data/finetune/jc2_bal", "configs/finetune/JetClassII_L188_noweight.yaml", 188,
             "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_jc2b", (10_000, 100_000, 1_000_000),
             "/data/finetune/jc2_bal/test.parquet", "64Gi"),
}
VAL_SIZE = {"jc2b": 20_480}        # the balanced validation file (188 x 109 at most); 200,000 elsewhere
TEST_CFG = {"jc2b": "configs/data/JetClassII_base.yaml"}
FULL = {"topkin": 1_200_000, "top": 1_200_000, "topmean": 1_200_000}   # the full training set: ParT's
# recipe, 20 epochs, no early stop (top's full set only in the fill check; bench stops at 1e5)
KIN_RUNS = ("l188kin-s1",)          # the 17-class kinematic-only run was stopped for good (2026-11-01)
FULL_MEM = "88Gi"


def head_mult(lr: str) -> str:
    """The head's multiplier: 50x the backbone rate, capped at 5e-3 for the head."""
    return repr(min(HEAD_MULT, HEAD_CAP / float(lr)))


def paper_runs() -> list:
    return [l.strip()[len("mtx-"):] for l in PAPER_RUNS.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def cell(task: str, init: str, n: int, s: int, lr: str, root: str, sub: str = "") -> dict:
    """One cell. init '' is from scratch; sub names the cell's directory under its init. Below the
    full set: 20,000 steps in epochs of 500 steps (50 below 1e5 jets), early stop after 10 epochs
    without improvement; at the full set (FULL): ParT's 20 epochs of the whole set, kept to the end."""
    tree = TASKS[task][4]
    name = init or "scratch"
    leaf = f"{name}{sub}"
    full = FULL.get(task) == n
    steps = SMALL_STEPS if n < SMALL_N else STEPS
    return {"task": task, "init": init, "run_dir": f"{MTX_V2}/mtx-{init}" if init else "", "n": n, "s": s,
            "lr": lr, "mult": head_mult(lr) if init else "", "out": f"{root}/{tree}/{leaf}/N{n}/s{s}",
            "epochs": 20 if full else EPOCHS * STEPS // steps, "spe": n if full else steps * BATCH,
            "patience": 1000 if full else PATIENCE, "val": VAL_SIZE.get(task, VAL_N)}


def lrsweep_cells() -> list:
    out = []
    for task in ("top", "qg"):
        out += [cell(task, "l188-s1", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_PRE]
        out += [cell(task, "", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_SCRATCH]
    return out


def bench_cells(lr_pre: str, lr_scratch: str, tasks=("top", "qg"), runs=None, lr_scratch_task=None) -> list:
    """lr_scratch_task: {task: rate} where a task's own sweep chose another from-scratch rate."""
    runs = paper_runs() if runs is None else runs
    out = []
    for task in tasks:
        sizes = TASKS[task][5]
        lr_s = (lr_scratch_task or {}).get(task, lr_scratch)
        out += [cell(task, r, n, 1, lr_pre, BUDGET_ROOT) for n in sizes for r in runs]
        out += [cell(task, "", n, s, lr_s, BUDGET_ROOT) for n in sizes for s in SCRATCH_SEEDS]
    return out


def jc2b_cells(lr_pre: str, lr_scratch: str) -> list:
    """The hierarchy and the mass-output runs (the Methods' JetClass-II scope) and from scratch."""
    runs = [r for r in paper_runs() if r.split("-")[0] in ("l188", "l162", "r42q1", "r16q1", "l162mass", "r16q1mass")]
    return bench_cells(lr_pre, lr_scratch, ("jc2b",), runs)


def long_cells(task: str, lr_pre: str, lr_scratch: str) -> list:
    inits = LONG_JC2B if task == "jc2b" else paper_runs()
    out = [{**cell(task, "", LONG_N, s, lr_scratch, LONG_ROOT), "patience_from": LONG_EPOCHS - max(1, int(LONG_EPOCHS * 0.3))}
           for s in SCRATCH_SEEDS]      # weaver's flat+decay: the last 30% of the epochs decay
    out += [cell(task, r, LONG_N, 1, lr_pre, LONG_ROOT) for r in inits]
    return [{**c, "epochs": LONG_EPOCHS} for c in out]


def long_ctrl_cells(lr_pre: str) -> list:
    """The five random partitions and the three controlled ones on JetClass-II at 1e6 with the long
    budget, so the JetClass-II fine-tuning figures carry the controls (PI, 2026-11-05)."""
    runs = [r for r in paper_runs() if r.startswith(("rand", "flav"))]
    return [{**cell("jc2b", r, LONG_N, 1, lr_pre, LONG_ROOT), "epochs": LONG_EPOCHS} for r in runs]


def kin_cells(lr_pre: str) -> list:
    """The kinematic-only runs on top tagging: below the full set at the swept rate, and the full set
    with ParT's fine-tuning recipe (1e-4, head x50)."""
    out = [cell("topkin", r, n, 1, lr_pre, BUDGET_ROOT) for n in TASKS["topkin"][5] for r in KIN_RUNS]
    out += [cell("topkin", r, FULL["topkin"], 1, "1e-4", BUDGET_ROOT) for r in KIN_RUNS]
    return out


def fillcheck_cells() -> list:
    """Top at the full set from 188-s1 with ParT's recipe (1e-4, head x50), zero and mean fill."""
    return [cell(task, "l188-s1", 1_200_000, 1, "1e-4", FILLCHECK_ROOT, sub)
            for task, sub in (("top", "_zero"), ("topmean", "_mean"))]


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
[ ! -f ${CELL}/DONE ] || { echo "${CELL} is done"; exit 0; }
[ -d ${CELL} ] && mv ${CELL} ${CELL}.partial.$(date -u +%s)
mkdir -p ${CELL}
echo "$(date -u +%FT%TZ) pod=${POD_NAME} node=${NODE_NAME} ref=${REPO_REF}" >> ${CELL}/ATTEMPTS
if [ -n "${RUN_DIR}" ]; then
  python3 experiments/FT/ft_v2.py resolve --run-dir ${RUN_DIR} --rule best70 --link /workspace/ckpt/${INIT}.pt || exit ${HALT}
  cp /workspace/ckpt/${INIT}.pt.json ${CELL}/init_checkpoint.json
  if [ "${TASK}" = topmean ]; then
    python3 experiments/FT/fill_config.py --checkpoint /workspace/ckpt/${INIT}.pt \
      --src configs/finetune/TopReference.yaml --out ${CFG} || exit ${HALT}
    cp ${CFG} ${CELL}/
  fi
  LOAD=(--load-model-weights /workspace/ckpt/${INIT}.pt --exclude-model-weights "mod\.fc\..*")
  MULTARG=(--optimizer-option lr_mult "(r'mod\.fc\..*', ${MULT})")
  SCHED=(--lr-scheduler none); RECIPE="head_lr_mult=${MULT} lr_schedule=constant ckpt_rule=best70"
else
  LOAD=(); MULTARG=(); SCHED=(); RECIPE="head_lr_mult=1 lr_schedule=flat+decay recipe=part-from-scratch"
fi
python3 experiments/FT/smoke_checks.py manifest --out ${CELL}/ft_manifest.json task=${TASK} \
  init=${INIT:-scratch} run_dir=${RUN_DIR} checkpoint=${RUN_DIR:+/workspace/ckpt/${INIT}.pt} n_train=${N} ft_seed=${S} lr=${LR} ${RECIPE} weight_decay=0.01 \
  subset=${SUB}/train_N${N}_s1.parquet data_config=${CFG} num_classes=${K} batch_size=__BATCH__ \
  samples_per_epoch=${SPE} steps_per_epoch=$((SPE / __BATCH__)) max_epochs=${EP} \
  samples_per_epoch_val=${VAL} early_stop_patience=${PAT} val_rule=fixed-file budget=v3.1
python3 experiments/FT/ft_weaver.py --seed ${S} --lean-val-metrics \
  --data-train ${SUB}/train_N${N}_s1.parquet --data-val ${SUB}/val.parquet --data-config ${CFG} \
  --network-config ${NET} -o num_classes ${K} -o fc_params '[(512,0.1)]' \
  --use-amp --batch-size __BATCH__ --num-workers 2 --fetch-by-files --fetch-step 1 --optimizer ranger \
  ${SCHED[@]+"${SCHED[@]}"} --optimizer-option weight_decay 0.01 ${MULTARG[@]+"${MULTARG[@]}"} \
  --start-lr ${LR} --samples-per-epoch ${SPE} --samples-per-epoch-val ${VAL} \
  --num-epochs ${EP} ${LOAD[@]+"${LOAD[@]}"} --model-prefix ${RUN}/net --log ${RUN}/train.log \
  > ${RUN}/stdout.log 2>&1 &
PID=$!
ES=0
python3 experiments/FT/early_stop.py watch --log ${RUN}/train.log --pid ${PID} --patience ${PAT} --patience-from ${PATFROM} \
  --max-epochs ${EP} --out ${CELL}/early_stop.json || ES=$?
wait ${PID} || true
cp ${RUN}/train.log ${RUN}/stdout.log ${CELL}/
[ "${ES}" -eq 0 ] || { echo "FATAL: ${CELL}: training did not finish (early_stop rc=${ES})"; tail -30 ${RUN}/stdout.log; exit 1; }
grep -q "('steps_per_epoch', $((SPE / __BATCH__)))" ${RUN}/train.log || { echo "FATAL: weaver did not run $((SPE / __BATCH__)) steps per epoch"; exit 1; }
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
      --arm FTB_${TASK}_${INIT:-scratch}_N${N}_s${S} --data-config ${TCFG} --data-test ${TEST} \
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
        out.append(f"  {t}) SUB={sub}; CFG={cfg}; TCFG={TEST_CFG.get(t, cfg)}; K={k}; NET={net}; {test_block};;\n")
    return "".join(out)


def cell_cases(cells: list) -> str:
    return "".join(f"  {i}) TASK={c['task']}; INIT={c['init']}; RUN_DIR={c['run_dir']}; N={c['n']}; S={c['s']}; "
                   f"LR={c['lr']}; MULT={c['mult']}; EP={c['epochs']}; SPE={c['spe']}; PAT={c['patience']}; "
                   f"VAL={c['val']}; PATFROM={c.get('patience_from', 0)}; CELL={c['out']};;\n" for i, c in enumerate(cells))


def spec(name: str, cells: list, tag: str, parallelism: int, memory: str, cpu: int = 4) -> str:
    launch = _launch()
    script = (CELL_SCRIPT.replace("__CASES__", cell_cases(cells)).replace("__TASKS__", task_cases())
              .replace("__BATCH__", str(BATCH)).replace("__HERWIG__", " ".join(HERWIG_TEST)))
    # a node's DNS can fail for minutes (cph-dgx-node4/5/8, 2026-11-03): six cells used up their three
    # attempts at the clone, in seconds, before reaching the training script
    script = ("set -euo pipefail\n"
              'retry () { for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt ${T} failed: $1"; sleep 60; done; return 1; }\n'
              'retry git clone -q --depth 1 --branch "${REPO_REF}" https://github.com/raunavm/transferlearningsophon.git '
              "/workspace/transferlearningsophon || exit 1\ncd /workspace/transferlearningsophon\n"
              "retry pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1\n" + script.lstrip("\n"))
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
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
          requests: {{ memory: "{memory}", cpu: "{cpu}", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{memory}", cpu: "{cpu}", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
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
                values: [{", ".join(f'"{g}"' for g in launch.V2_FT_GPU)}]
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
    ap.add_argument("--lr-scratch-qg", default=None, help="quark/gluon's own from-scratch rate (its sweep)")
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=4, help="CPUs per pod (3: fits CPU-starved nodes with a free GPU, PI 2026-11-06)")
    ap.add_argument("--name-suffix", default="", help="a second job of the same cells, e.g. cpu3")
    ap.add_argument("--mem", default="", help="memory per pod, overriding the rule below (64Gi: the long JetClass "
                    "cells peaked at 53 GiB over 22 pods, and 88Gi kept them off nodes with a free GPU, 2026-11-07)")
    ap.add_argument("--indices", default=None,
                    help="comma-separated indices of the full kind's job: a -retry job of those cells only")
    ap.add_argument("kinds", nargs="+", choices=("lrsweep", "bench", "jc1", "jc2b", "kin", "fillcheck", "long-jc2b", "long-jc1", "long-jc2b-ctrl"))
    a = ap.parse_args(argv)
    if a.indices and len(a.kinds) != 1:
        ap.error("--indices needs exactly one kind")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        if kind == "lrsweep":
            cells, mem = lrsweep_cells(), "48Gi"
        elif kind == "fillcheck":
            cells, mem = fillcheck_cells(), "48Gi"
        else:
            if not (a.lr_pre and a.lr_scratch):
                ap.error(f"{kind} needs --lr-pre and --lr-scratch (from the sweep)")
            per_task = {"qg": a.lr_scratch_qg} if a.lr_scratch_qg else None
            cells = {"bench": lambda: bench_cells(a.lr_pre, a.lr_scratch, ("top", "qg"), lr_scratch_task=per_task),
                     "jc1": lambda: bench_cells(a.lr_pre, a.lr_scratch, ("jc1",)),
                     "jc2b": lambda: jc2b_cells(a.lr_pre, a.lr_scratch),
                     "kin": lambda: kin_cells(a.lr_pre),
                     "long-jc2b": lambda: long_cells("jc2b", a.lr_pre, a.lr_scratch),
                     "long-jc1": lambda: long_cells("jc1", a.lr_pre, a.lr_scratch),
                     "long-jc2b-ctrl": lambda: long_ctrl_cells(a.lr_pre)}[kind]()
            mem = "48Gi" if kind in ("bench", "kin") else "64Gi"
        if any(c["n"] >= 1_000_000 for c in cells):
            # weaver holds a whole training file in memory: 48Gi OOM-killed the top fill check at
            # 1.2M jets four times (2026-11-02); the old full-size top fine-tunings ran at 88Gi
            mem = FULL_MEM
        mem = a.mem or mem
        name = f"ft-budget-{kind}-raunav"
        if a.indices:
            cells = [cells[int(i)] for i in a.indices.split(",")]
            name = f"ft-budget-{kind}-retry-raunav"
        if a.name_suffix:
            name = name.replace("-raunav", f"-{a.name_suffix}-raunav")
        p = OUT_DIR / f"job-{name}.yaml"
        p.write_text(spec(name, cells, a.tag, a.parallelism, mem, a.cpu))
        print(p, len(cells), "cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
