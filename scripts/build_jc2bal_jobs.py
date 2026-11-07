#!/usr/bin/env python3
"""Jobs of the class-balanced JetClass-II sets (plan 3.4, 2026-10-31).

  stage    CPU: experiments/FT/stage_jc2_balanced.py -> /data/finetune/jc2_bal (census, balanced test,
           nested balanced training sets, validation, the pair sets)
  ref      CPU: the imputation reference of the missing-feature control (input_variants.py ref) from
           the balanced 1e5 training set -> /data/finetune/jc2_bal/impute_ref.npz
  extract  GPU (any but V100), one Indexed Job, one model per index (the 27 paper runs and the three
           untrained trunks): extract_v2.py on every balanced split (configs/data/JetClassII_massreg.yaml,
           native labels, every row kept); at the 1e6 training set, validation and test also the input
           variants mtop and mqg, and for the 188- and 17-class runs itop and iqg (plan 3.2) ->
           /data/results/eval/v2_linprobe_bal/<model>/{jc2bal,jc2bal_pairs}/<split>/<checkpoint>[.<variant>]
  fit      GPU (any but V100), one model per index: experiments/EVAL/probe_bal.py, the v4 linear probes
           (balanced JetClass-II at every size, the pairs, the variants; JetClass, top and quark/gluon
           refit on the v4 grid; the 512-unit hidden layer for the 188-class runs) -> .../fits_v4
  mlp      CPU, one model per index: the nonlinear probe on the balanced JetClass-II task and pairs, for
           the hierarchy levels, the three controlled partitions and the untrained trunks (PI scope;
           JetClass, top and quark/gluon are linprobe-mlp4's) -> .../fits_v4_mlp
  ref2     CPU: the version-2 imputation reference, with the displacement pools of each particle species
           (input_variants.py, itop4/iqg4) -> /data/finetune/jc2_bal/impute_ref_v2.npz
  xfill    GPU, one model per index (all 30): the fills of the absent inputs (PI 2026-11-01), extract_v2.py
           with the zero fill of the downstream configs plus mean fill and four-draw imputation, on every
           top and quark/gluon split the downstream probes read (configs, files and sizes from
           build_linprobe_jobs.DATASETS) and on the balanced JetClass-II 1e6 / validation / test sets
           -> /data/results/eval/v2_linprobe_fill/<model>/{top,qg,jc2bal}/<split>/<tag>[.<variant>]
  fitfill  GPU, one model per index: probe_bal.py --fill (zero / mean / imputed side by side, and the
           re-extracted zero fill checked against the stored features) -> .../fits_v4_fill
  kin      GPU (any but V100), one kinematic-only model per index (plan 3.7): top tagging only, its
           splits extracted with configs/finetune/TopReference_kin.yaml into the downstream probes'
           tree, the v4 linear probe (probe_bal.py --old-only top) and the nonlinear probe at the
           largest training set -> .../fits_v4, .../fits_v4_mlp. Refuses a run without DONE.
  sib      CPU, one model per index (all 30), off the bad nodes: probe_bal.py --sibling-pairs, the
           X->ee vs X->mumu linear probe on the stored balanced features -> .../fits_v4_sib,
           .../fits_v4_sib_probs (new folders)
  sibstage CPU: experiments/FT/stage_jc2_siblings.py -> /data/finetune/jc2_sib (every test jet of X->ee and
           X->mumu, PI 2026-11-05: the balanced test set's 5,000 each stop the ROC at 250)
  sibx     GPU, one model per index (all 30): their features (dataset jc2sib) and probe_bal.py
           --sibling-pairs --sibling-test -> .../fits_v4_sib2, .../fits_v4_sib2_probs (new folders)
  coarse   GPU, one model per index (all 30): probe_bal.py --coarse, the balanced JetClass-II probe on the
           43- and 17-class levels and the partition with no flavor boundary (PI 2026-11-07)
           -> .../fits_v4_coarse, .../fits_v4_coarse_probs (new folders)
  qcdhf    CPU, one pod: experiments/EVAL/qcd_hf_rejection.py, the X->bb discriminant's rejection of each
           QCD heavy-flavor subclass from the stored probes -> .../fits_v4_qcdhf (new folder)
  ckptid   CPU, one pod: experiments/EVAL/checkpoint_identity.py, whether each paper run's bestval is its
           best70 file -> /data/results/eval/v2_ckpt_identity/identity.json
  sophon   GPU, one pod: the released Sophon model's features on the frozen probes' splits, probe_released.py
           and qcd_hf_rejection.py -> /data/results/eval/v2_linprobe_sophon (new folder)

    python3 scripts/build_jc2bal_jobs.py --tag mtx-s2.24 stage
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "FT" / "k8s" / "jc2bal"
BAL = "/data/finetune/jc2_bal"


def _launch():
    spec = importlib.util.spec_from_file_location("build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_spec(tag: str, name: str = "jc2bal-stage-raunav", out: str = BAL,
               stager: str = "experiments/FT/stage_jc2_balanced.py") -> tuple:
    launch = _launch()
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
rm -rf {out}.staging
python3 {stager} --out {out}.staging --workers 8
mv {out}.staging {out}
du -sh {out}; ls -la {out}
echo "JC2BAL STAGE COMPLETE"
"""
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    return name, f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_jc2bal_jobs.py -- class-balanced JetClass-II sets (plan 3.4). CPU.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 1
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
          requests: {{ memory: "96Gi", cpu: "8", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "96Gi", cpu: "8", ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: jc2, mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: jc2
        persistentVolumeClaim:
          claimName: tn-pvc-base-jetclass2
          readOnly: true
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""


LP_BAL = "/data/results/eval/v2_linprobe_bal"
PAPER_RUNS = ROOT / "configs" / "arms" / "v2_paper_runs.txt"
IMPUTE_ARMS = ("mtx-l188-", "mtx-r16q1-")          # imputed variants: 188- and 17-class runs only (PI)
CFG = "configs/data/JetClassII_massreg.yaml"
# Any GPU product (us-west, minus V2_BAD_NODES) for the extraction and probe jobs: the extraction left an
# L40 at 0% with one loader core at 100% (2026-11-01), so the GPU type does not set the speed; every
# product on the cluster (Pascal and newer) is supported by the image's torch 2.5.1. Manifests record it.
# pairs_test (2.1 M jets) read whole OOM-kills a 48Gi extraction pod; its three parts (split_spec) fit
PAIRS_TEST_PARTS = tuple(f"pairs_test_part{i}.parquet" for i in range(3))


# Start-up of the jobs that run on any GPU node. A node whose network or GPU is broken fails a pod in
# seconds and takes the next one: 70 attempts in 16 minutes burned indices 1-13 of jc2bal-x (2026-11-01).
# Clone and pip retry with pauses; a GPU that does not answer holds the pod two minutes before exit 43.
def _start(tag: str) -> str:
    return f"""retry () {{ for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt $T failed: $1"; sleep 30; done; return 1; }}
clone () {{ mkdir -p /workspace && cd /workspace && rm -rf tls && git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git tls; }}
retry clone || exit 1
cd /workspace/tls
retry pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1"""


GPU_CHECK = "python3 experiments/FT/gpu_probe.py || { sleep 120; exit 43; }"


def _probe_builder():
    spec = importlib.util.spec_from_file_location("build_linprobe_jobs", ROOT / "scripts" / "build_linprobe_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def models() -> list:
    """(model, run directory, rung, classes, regression outputs, checkpoints, variants): the paper runs
    and the untrained trunks (with their BatchNorm twin)."""
    paper = {l.strip() for l in PAPER_RUNS.read_text().splitlines() if l.strip() and not l.startswith("#")}
    out = []
    for model, run, rung, k, reg, ckpts in _probe_builder().models():
        if model.startswith("init-"):
            out.append((model, run, rung, k, reg, ("init", "init_bn"), ("mtop", "mqg")))
        elif model in paper:
            v = ("mtop", "mqg", "itop", "iqg") if model.startswith(IMPUTE_ARMS) and "mass" not in model \
                else ("mtop", "mqg")
            out.append((model, run, rung, k, reg, ("best70",), v))
    return out


def ref_spec(tag: str, out: str = "impute_ref.npz", name: str = "jc2bal-ref-raunav") -> tuple:
    launch = _launch()
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1
[ -f {BAL}/DONE ] || {{ echo "FATAL: {BAL} is not staged"; exit 42; }}
[ ! -f {BAL}/{out} ] || {{ echo "{BAL}/{out} exists"; exit 0; }}
python3 experiments/EVAL/input_variants.py ref --data {BAL}/train_N100000_s1.parquet \\
  --data-config configs/data/JetClassII_base.yaml --max-jets 100000 --out {BAL}/{out}
echo "JC2BAL REF COMPLETE"
"""
    return name, _cpu_job(name, script, launch, "32Gi", "4", "the imputation reference (plan 3.2)")


def extract_spec(tag: str, parallelism: int, only: tuple = (), name: str = "jc2bal-x-raunav") -> tuple:
    """only: the models (indices into models()) of this job, in order; all when empty."""
    launch = _launch()
    ms = models()
    ms = [ms[i] for i in only] if only else ms
    cases = "".join(f"  {i}) MODEL={m}; RUN_DIR={launch.V2_ROOT}/{run}; RUNG={rung}; K={k}; REG={reg}; "
                    f'CKPTS="{" ".join(c)}"; VARIANTS="{" ".join(v)}";;\n'
                    for i, (m, run, rung, k, reg, c, v) in enumerate(ms))
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {LP_BAL}/${{MODEL}}/DONE ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f {BAL}/DONE ] && [ -f {BAL}/impute_ref.npz ] && [ -f {BAL}/{PAIRS_TEST_PARTS[-1]} ] || {{ echo "FATAL: {BAL} not staged"; exit 42; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{GPU_CHECK}
# extract DATASET SPLIT "FILES" [variants]: one pass over the files, every checkpoint (and variant);
# a set whose every checkpoint directory has its manifest (written last) is done
extract () {{
  OUT={LP_BAL}/${{MODEL}}/$1/$2; FILES=$3; shift 3
  TAGS=""; for C in ${{CKPTS}}; do TAGS="${{TAGS}} ${{C}}"; [ "${{1:-}}" != --variants ] || for V in ${{VARIANTS}}; do TAGS="${{TAGS}} ${{C}}.${{V}}"; done; done
  ALL=1; for T in ${{TAGS}}; do [ -f ${{OUT}}/${{T}}/manifest.json ] || ALL=0; done
  [ ${{ALL}} = 0 ] || {{ echo "${{OUT}} done"; return 0; }}
  python3 experiments/EVAL/extract_v2.py --run-dir ${{RUN_DIR}} --rung ${{RUNG}} --num-classes ${{K}} \\
    --num-reg ${{REG}} --checkpoints ${{CKPTS}} --data-config {CFG} --max-jets 0 \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test ${{FILES}} "$@" --out ${{OUT}} || exit 1
}}
for N in 1000 10000 100000; do extract jc2bal train_N${{N}} {BAL}/train_N${{N}}_s1.parquet; done
for S in train_N1000000:train_N1000000_s1 val:val test:test; do
  extract jc2bal ${{S%%:*}} {BAL}/${{S##*:}}.parquet --variants ${{VARIANTS}} --impute-ref {BAL}/impute_ref.npz
done
extract jc2bal_pairs train {BAL}/pairs_train.parquet
extract jc2bal_pairs test "{" ".join(f"{BAL}/{p}" for p in PAIRS_TEST_PARTS)}"
touch {LP_BAL}/${{MODEL}}/DONE
echo "JC2BAL X ${{MODEL}} COMPLETE"
"""
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    return name, f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_jc2bal_jobs.py -- features on the class-balanced JetClass-II sets (plan 3.2, 3.4).
  name: {name}
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
        image: {launch.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        args:
        - |
{body}
        resources:
          requests: {{ memory: "48Gi", cpu: "2", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "48Gi", cpu: "2", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
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


MLP_SCOPE = ("mtx-l188-", "mtx-l162-", "mtx-r42q1-", "mtx-r16q1-s", "mtx-flavf", "init-")
HIDDEN = ("mtx-l188-",)


def _indexed(name: str, script_body: str, n: int, parallelism: int, gpu: bool, mem: str, cpu: str, what: str,
             exclude=None) -> str:
    """exclude: keep off V2_BAD_NODES + V2_LATER_BAD_NODES (default: GPU jobs only, as launched)."""
    launch = _launch()
    body = "\n".join(("          " + ln) if ln else "" for ln in script_body.splitlines())
    nodes = ", ".join(f'"{x}"' for x in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    gpu_req = ', nvidia.com/gpu: "1"' if gpu else ""
    gpu_aff = (f"""
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{nodes}]""" if (gpu if exclude is None else exclude) else "")
    tol = ("""      tolerations:
      - { key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }
""" if gpu else "")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_jc2bal_jobs.py -- {what}.
  name: {name}
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {n}
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
        image: {launch.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{mem}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
{tol}      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]{gpu_aff}
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


def fit_spec(tag: str, parallelism: int, mlp: bool, only: tuple = (), name: str = "") -> tuple:
    """only: the models (indices into the job's model list) of this job, in order; all when empty."""
    ms = [m for m in models() if not mlp or m[0].startswith(MLP_SCOPE)]
    ms = [ms[i] for i in only] if only else ms
    out = "fits_v4_mlp" if mlp else "fits_v4"
    cases = ""
    for i, (m, run, rung, k, reg, c, v) in enumerate(ms):
        extra = "--mlp --mlp-datasets jc2bal" if mlp else (
            f"--variants {' '.join(v)} --probs-out {LP_BAL}/fits_v4_probs"
            + (" --hidden" if m.startswith(HIDDEN) else ""))
        cases += f'  {i}) MODEL={m}; CKPTS="{" ".join(c)}"; EXTRA="{extra}";;\n'
    name = name or f"jc2bal-{'mlp' if mlp else 'fit'}-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {LP_BAL}/${{MODEL}}/DONE ] || {{ echo "FATAL: ${{MODEL}} is not extracted"; exit 42; }}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} ${{EXTRA}} --out {LP_BAL}/{out}
echo "JC2BAL {'MLP' if mlp else 'FIT'} ${{MODEL}} COMPLETE"
"""
    what = ("nonlinear probes on the balanced JetClass-II task and pairs (plan 3.3). CPU" if mlp
            else "v4 linear probes (plan 2.3, 3.2, 3.4, 3.5). GPU")
    return name, _indexed(name, script, len(ms), parallelism, not mlp, "64Gi" if mlp else "48Gi",
                          "8" if mlp else "4", what)


def sib_spec(tag: str, parallelism: int) -> tuple:
    """CPU, one model per index (all 30): probe_bal.py --sibling-pairs (X->ee vs X->mumu on the stored
    balanced features) -> new folders fits_v4_sib, fits_v4_sib_probs. Kept off the bad nodes."""
    ms = models()
    cases = "".join(f'  {i}) MODEL={m}; CKPTS="{" ".join(c)}";;\n' for i, (m, _r, _g, _k, _e, c, _v) in enumerate(ms))
    name = "jc2bal-sib-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {LP_BAL}/${{MODEL}}/DONE ] || {{ echo "FATAL: ${{MODEL}} is not extracted"; exit 42; }}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} --sibling-pairs \\
  --out {LP_BAL}/fits_v4_sib --probs-out {LP_BAL}/fits_v4_sib_probs
echo "JC2BAL SIB ${{MODEL}} COMPLETE"
"""
    return name, _indexed(name, script, len(ms), parallelism, False, "16Gi", "4",
                          "linear probe on X->ee vs X->mumu, the balanced sets (ROC redesign 2026-11-02). CPU",
                          exclude=True)


SIB = "/data/finetune/jc2_sib"      # every test jet of X->ee and X->mumu (stage_jc2_siblings.py)


def sibstage_spec(tag: str) -> tuple:
    return stage_spec(tag, "jc2sib-stage-raunav", SIB, "experiments/FT/stage_jc2_siblings.py")


def sibx_spec(tag: str, parallelism: int) -> tuple:
    """GPU, one model per index (all 30): the features of every test jet of X->ee and X->mumu, then
    probe_bal.py --sibling-pairs --sibling-test -> new folders fits_v4_sib2, fits_v4_sib2_probs."""
    launch = _launch()
    ms = models()
    cases = "".join(f"  {i}) MODEL={m}; RUN_DIR={launch.V2_ROOT}/{run}; RUNG={rung}; K={k}; REG={reg}; "
                    f'CKPTS="{" ".join(c)}";;\n' for i, (m, run, rung, k, reg, c, _v) in enumerate(ms))
    name = "jc2sib-x-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {LP_BAL}/fits_v4_sib2/${{MODEL}}.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f {SIB}/DONE ] && [ -f {LP_BAL}/${{MODEL}}/DONE ] || {{ echo "FATAL: {SIB} or ${{MODEL}}'s balanced features missing"; exit 42; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{GPU_CHECK}
OUT={LP_BAL}/${{MODEL}}/jc2sib/test
ALL=1; for C in ${{CKPTS}}; do [ -f ${{OUT}}/${{C}}/manifest.json ] || ALL=0; done
[ ${{ALL}} = 1 ] || python3 experiments/EVAL/extract_v2.py --run-dir ${{RUN_DIR}} --rung ${{RUNG}} --num-classes ${{K}} \\
  --num-reg ${{REG}} --checkpoints ${{CKPTS}} --data-config {CFG} --max-jets 0 \\
  --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
  --diag-stride 0 --no-pooled --data-test {SIB}/test.parquet --out ${{OUT}} || exit 1
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} --sibling-pairs --sibling-test \\
  --out {LP_BAL}/fits_v4_sib2 --probs-out {LP_BAL}/fits_v4_sib2_probs
echo "JC2SIB ${{MODEL}} COMPLETE"
"""
    return name, _indexed(name, script, len(ms), parallelism, True, "48Gi", "4",
                          "every test jet of X->ee and X->mumu: features and the linear probe (PI 2026-11-05)")


FILL_ROOT = "/data/results/eval/v2_linprobe_fill"           # experiments/EVAL/probe_bal.py FILL_ROOT
REF2 = f"{BAL}/impute_ref_v2.npz"
FILL_VARIANTS = {"top": "ntop itop4", "qg": "nqg iqg4", "jc2bal": "ntop nqg itop4 iqg4"}   # probe_bal.FILLS


def xfill_spec(tag: str, parallelism: int) -> tuple:
    """Every model's fills of the absent inputs: the downstream probes' top and quark/gluon splits
    (build_linprobe_jobs.DATASETS: same configs, files and sizes, so the zero fill reproduces the
    stored features) and the balanced JetClass-II sets the missing-feature table reads."""
    pb = _probe_builder()
    ms = models()
    cases = "".join(f"  {i}) MODEL={m}; RUN_DIR={_launch().V2_ROOT}/{run}; RUNG={rung}; K={k}; REG={reg}; "
                    f'CKPTS="{" ".join(c)}";;\n' for i, (m, run, rung, k, reg, c, _v) in enumerate(ms))
    calls = ""
    for ds in ("top", "qg"):
        sizes, cfg, extra = pb.DATASETS[ds]
        for n in sizes:
            calls += f'extract {ds} train_N{n} {cfg} "{FILL_VARIANTS[ds]}" "{pb.TRAIN_DIR[ds]}/train_N{n}_s1.parquet"\n'
        for split, (maxj, files) in extra.items():
            if maxj:
                raise SystemExit(f"FATAL: {ds}/{split} is read with --max-jets {maxj}")
            calls += f'extract {ds} {split} {cfg} "{FILL_VARIANTS[ds]}" "{files}"\n'
    for split, f in (("train_N1000000", "train_N1000000_s1"), ("val", "val"), ("test", "test")):
        calls += f'extract jc2bal {split} {CFG} "{FILL_VARIANTS["jc2bal"]}" "{BAL}/{f}.parquet"\n'
    name = "jc2bal-xfill-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {FILL_ROOT}/${{MODEL}}/DONE ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f {REF2} ] || {{ echo "FATAL: {REF2} missing (ref2 job)"; exit 42; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{GPU_CHECK}
# extract DATASET SPLIT CONFIG "VARIANTS" "FILES": one pass, every checkpoint and its fills; a set whose
# every directory has its manifest (written last) is done
extract () {{
  OUT={FILL_ROOT}/${{MODEL}}/$1/$2; CFGX=$3; VARS=$4; FILES=$5
  ALL=1; for C in ${{CKPTS}}; do for T in ${{C}} $(for V in ${{VARS}}; do echo ${{C}}.${{V}}; done); do
    [ -f ${{OUT}}/${{T}}/manifest.json ] || ALL=0; done; done
  [ ${{ALL}} = 0 ] || {{ echo "${{OUT}} done"; return 0; }}
  python3 experiments/EVAL/extract_v2.py --run-dir ${{RUN_DIR}} --rung ${{RUNG}} --num-classes ${{K}} \\
    --num-reg ${{REG}} --checkpoints ${{CKPTS}} --data-config ${{CFGX}} --max-jets 0 \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test ${{FILES}} --variants ${{VARS}} --impute-ref {REF2} \\
    --out ${{OUT}} || exit 1
}}
{calls}touch {FILL_ROOT}/${{MODEL}}/DONE
echo "JC2BAL XFILL ${{MODEL}} COMPLETE"
"""
    # weaver holds a whole file per fetch: the 1.6M-jet quark/gluon training file OOM-killed a 48Gi pod
    # (2026-11-01); the downstream extraction it reproduces ran at build_linprobe_jobs.X_MEM
    return name, _indexed(name, script, len(ms), parallelism, True, _probe_builder().X_MEM, "2",
                          "the fills of the absent inputs, every model (PI 2026-11-01). GPU")


def fitfill_spec(tag: str, parallelism: int) -> tuple:
    ms = models()
    cases = "".join(f'  {i}) MODEL={m}; CKPTS="{" ".join(c)}";;\n' for i, (m, _r, _g, _k, _e, c, _v) in enumerate(ms))
    name = "jc2bal-fitfill-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {FILL_ROOT}/${{MODEL}}/DONE ] || {{ echo "FATAL: ${{MODEL}}'s fills are not extracted"; exit 42; }}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} --fill \\
  --out {LP_BAL}/fits_v4_fill --probs-out {LP_BAL}/fits_v4_fill_probs
echo "JC2BAL FITFILL ${{MODEL}} COMPLETE"
"""
    return name, _indexed(name, script, len(ms), parallelism, True, "48Gi", "4",
                          "linear probes on the fills of the absent inputs (PI 2026-11-01). GPU")


HEAD_ROOT = "/data/results/eval/v2_head"
HEAD_MODELS = ("l188", "l162", "r42q1", "r16q1", "l162mass", "r16q1mass")


def masscurves_spec(tag: str) -> tuple:
    """CPU: the mass-response histograms and the sculpting curve (JS distance on a grid of QCD
    efficiencies) of every model with head outputs (the hierarchy and the mass-output runs)."""
    launch = _launch()
    name = "masscurves-raunav"
    paper = [l.strip() for l in PAPER_RUNS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    models = [m for m in paper if m[len("mtx-"):].split("-")[0] in HEAD_MODELS]
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
OUT={HEAD_ROOT}/curves_v2
mkdir -p ${{OUT}}
for M in {' '.join(models)}; do
  H={HEAD_ROOT}/${{M}}/test/best70
  [ -f ${{H}}/head_outputs.npz ] || {{ echo "FATAL: no head outputs for ${{M}}"; exit 42; }}
  [ -f ${{OUT}}/${{M}}_sculpting_v2.json ] || python3 experiments/EVAL/mass_sculpting.py --head ${{H}} --out ${{OUT}}/${{M}}_sculpting_v2.json
  [ -f ${{OUT}}/${{M}}_response.json ] || python3 experiments/EVAL/mass_response_hist.py --head ${{H}} --out ${{OUT}}/${{M}}_response.json
done
echo "MASSCURVES COMPLETE"
"""
    return name, _cpu_job(name, script, launch, "32Gi", "4", "mass-response histograms and sculpting curves")


def _cpu_job(name: str, script: str, launch, mem: str, cpu: str, what: str) -> str:
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_jc2bal_jobs.py -- {what}. CPU.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 1
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
          requests: {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "20Gi" }}
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
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      # weaver's DataLoader workers need more than the pod's default 64 MB of shared memory
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


KIN_MODELS = (("mtx-l188kin-s1", "L188", 188),)      # the 17-class kinematic-only run was stopped for good (2026-11-01)
TOP_SIZES = (1_000, 10_000, 100_000, 1_200_000)     # scripts/build_ft_jobs.py BENCH_SIZES["top"]
LP_OLD = "/data/results/eval/v2_linprobe"           # experiments/EVAL/probe_bal.py OLD_ROOT


def split_spec(tag: str) -> tuple:
    launch = _launch()
    name = "jc2bal-split-raunav"
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1
[ -f {BAL}/DONE ] || {{ echo "FATAL: {BAL} is not staged"; exit 42; }}
[ ! -f {BAL}/{PAIRS_TEST_PARTS[-1]} ] || {{ echo "{BAL}/{PAIRS_TEST_PARTS[-1]} exists"; exit 0; }}
python3 experiments/FT/split_parquet.py {BAL}/pairs_test.parquet --parts {len(PAIRS_TEST_PARTS)}
echo "JC2BAL SPLIT COMPLETE"
"""
    return name, _cpu_job(name, script, launch, "16Gi", "2", "pairs_test in three parts for the extraction")


def kin_spec(tag: str) -> tuple:
    launch = _launch()
    name = "jc2bal-kin-raunav"
    cases = "".join(f"  {i}) MODEL={m}; RUNG={rung}; K={k};;\n" for i, (m, rung, k) in enumerate(KIN_MODELS))
    top = "/data/finetune/top_sub"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
RUN_DIR={launch.V2_ROOT}/${{MODEL}}
[ -f ${{RUN_DIR}}/DONE ] || {{ echo "FATAL: ${{RUN_DIR}} has not finished"; exit 42; }}
[ ! -f {LP_BAL}/fits_v4_mlp/${{MODEL}}.json ] || {{ echo "${{MODEL}} done"; exit 0; }}
{GPU_CHECK}
extract () {{
  OUT={LP_OLD}/${{MODEL}}/top/$1; FILE=$2
  [ -f ${{OUT}}/best70/manifest.json ] && return 0
  python3 experiments/EVAL/extract_v2.py --run-dir ${{RUN_DIR}} --rung ${{RUNG}} --num-classes ${{K}} \\
    --num-reg 0 --checkpoints best70 --data-config configs/finetune/TopReference_kin.yaml --max-jets 0 \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test ${{FILE}} --out ${{OUT}} || exit 1
}}
for N in {" ".join(map(str, TOP_SIZES))}; do extract train_N${{N}} {top}/train_N${{N}}_s1.parquet; done
extract val {top}/val.parquet
extract test /data/finetune/top/top_test.parquet
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints best70 --old-only top \\
  --out {LP_BAL}/fits_v4 --probs-out {LP_BAL}/fits_v4_probs
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints best70 --mlp --mlp-datasets top \\
  --out {LP_BAL}/fits_v4_mlp
echo "JC2BAL KIN ${{MODEL}} COMPLETE"
"""
    return name, _indexed(name, script, len(KIN_MODELS), len(KIN_MODELS), True, "48Gi", "4",
                          "top-tagging probes of the kinematic-only models (plan 3.7)")


COARSE = ("R42_Q1", "R16_Q1", "FLAV_F0")       # experiments/EVAL/probe_bal.py COARSE_MAPS


def coarse_spec(tag: str, parallelism: int) -> tuple:
    """GPU, one model per index (all 30): probe_bal.py --coarse, the balanced JetClass-II probe on the
    43- and 17-class levels and the partition with no flavor boundary (PI 2026-11-07) -> new folders
    fits_v4_coarse, fits_v4_coarse_probs."""
    ms = models()
    cases = "".join(f'  {i}) MODEL={m}; CKPTS="{" ".join(c)}";;\n' for i, (m, _r, _g, _k, _e, c, _v) in enumerate(ms))
    name = "jc2bal-coarse-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no model for index ${{IDX}}"; exit 42;;
esac
[ -f {LP_BAL}/${{MODEL}}/DONE ] || {{ echo "FATAL: ${{MODEL}} is not extracted"; exit 42; }}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} --coarse {" ".join(COARSE)} \\
  --out {LP_BAL}/fits_v4_coarse --probs-out {LP_BAL}/fits_v4_coarse_probs
echo "JC2BAL COARSE ${{MODEL}} COMPLETE"
"""
    return name, _indexed(name, script, len(ms), parallelism, True, "48Gi", "4",
                          "linear probes on the coarse label sets, balanced JetClass-II (PI 2026-11-07). GPU")


def qcdhf_spec(tag: str) -> tuple:
    """CPU, one pod: experiments/EVAL/qcd_hf_rejection.py for every model from the stored probes
    (fits_v4_probs) and test features -> new folder fits_v4_qcdhf."""
    launch = _launch()
    loop = "".join(f"[ -f {LP_BAL}/fits_v4_qcdhf/{m}.json ] || python3 experiments/EVAL/qcd_hf_rejection.py "
                   f"--model {m} --checkpoints {' '.join(c)} --probs-root {LP_BAL}/fits_v4_probs "
                   f"--features-root {LP_BAL} --out {LP_BAL}/fits_v4_qcdhf\n"
                   for m, _r, _g, _k, _e, c, _v in models())
    name = "jc2bal-qcdhf-raunav"
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
{loop}echo "JC2BAL QCDHF COMPLETE"
"""
    return name, _cpu_job(name, script, launch, "24Gi", "4", "QCD heavy-flavor rejection by the X->bb discriminant")


CKPT_ID = "/data/results/eval/v2_ckpt_identity/identity.json"


def ckptid_spec(tag: str) -> tuple:
    """CPU, one pod: experiments/EVAL/checkpoint_identity.py, which paper runs' bestval is best70's file."""
    launch = _launch()
    name = "ckpt-identity-raunav"
    script = f"""set -euo pipefail
git clone -q --depth 1 --branch {tag} https://github.com/raunavm/transferlearningsophon.git /workspace/tls
cd /workspace/tls
python3 experiments/EVAL/checkpoint_identity.py --runs {PAPER_RUNS.relative_to(ROOT)} --root {launch.V2_ROOT} --out {CKPT_ID}
echo "CKPT IDENTITY COMPLETE"
"""
    return name, _cpu_job(name, script, launch, "4Gi", "1", "bestval against best70 for every paper run")


# The released Sophon model (arXiv:2405.12972; massreg-e0b-extract, RUNS.csv), pinned by its sha256
SOPHON_URL = "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
SOPHON_SHA256 = "cc7c33b522e796b5bbf0aa9bb5b01361c964f4ef3acebdd9682d7519c095b824"
SOPHON_ROOT = "/data/results/eval/v2_linprobe_sophon"
SOPHON_MODEL, SOPHON_TAG = "sophon-released", "released"


def sophon_spec(tag: str) -> tuple:
    """GPU, one pod: the released Sophon model's class-token features on every split our models' frozen
    probes read (balanced JetClass-II with its config; JetClass, top and quark/gluon with Herwig with the
    downstream configs and files of build_linprobe_jobs.DATASETS, so the same inputs are zero-filled),
    then probe_released.py (the fits_v4 probes and the coarse label sets) and qcd_hf_rejection.py."""
    pb = _probe_builder()
    ck = f"/workspace/sophon/{SOPHON_TAG}.pt"
    calls = "".join(f"extract jc2bal train_N{n} {CFG} 0 {BAL}/train_N{n}_s1.parquet\n" for n in (1000, 10000, 100000, 1000000))
    calls += f"extract jc2bal val {CFG} 0 {BAL}/val.parquet\nextract jc2bal test {CFG} 0 {BAL}/test.parquet\n"
    for ds in ("jc1", "top", "qg"):
        sizes, cfg, extra = pb.DATASETS[ds]
        calls += "".join(f"extract {ds} train_N{n} {cfg} 0 {pb.TRAIN_DIR[ds]}/train_N{n}_s1.parquet\n" for n in sizes)
        for split, (maxj, files) in extra.items():
            calls += f"extract {ds} {split} {cfg} {maxj} {files}\n"
    out = f"{SOPHON_ROOT}/{SOPHON_MODEL}"
    name = "sophon-probe-raunav"
    script = f"""set -euo pipefail
{_start(tag)}
[ -f {SOPHON_ROOT}/fits_v4_qcdhf/{SOPHON_MODEL}.json ] && {{ echo "{SOPHON_MODEL} done"; exit 0; }}
[ -f {BAL}/DONE ] || {{ echo "FATAL: {BAL} not staged"; exit 42; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{GPU_CHECK}
mkdir -p /workspace/sophon
retry python3 -c "import urllib.request; urllib.request.urlretrieve('{SOPHON_URL}', '{ck}')" || exit 1
GOT=$(sha256sum {ck} | cut -d' ' -f1)
[ "${{GOT}}" = "{SOPHON_SHA256}" ] || {{ echo "FATAL: {ck} sha256 ${{GOT}}"; exit 42; }}
TEST1=""
for C in {pb.BF.JC1_CLASSES}; do
  for f in $(ls /data/JetClass/Pythia/test_20M/${{C}}_*.root | sort | head -2); do TEST1="${{TEST1}} ${{f}}"; done
done
[ "$(echo ${{TEST1}} | wc -w)" -eq 20 ] || {{ echo "FATAL: expected 20 JetClass test files"; exit 42; }}
# extract DATASET SPLIT CONFIG MAX_JETS FILES...: the split's features; a split with its manifest is done
extract () {{
  OUT={out}/$1/$2; CFGX=$3; MAXJ=$4; shift 4
  [ -f ${{OUT}}/{SOPHON_TAG}/manifest.json ] && return 0
  python3 experiments/EVAL/extract_v2.py --run-dir /workspace/sophon --rung L188 --num-classes 188 \\
    --num-reg 0 --checkpoints {ck} --data-config ${{CFGX}} --max-jets ${{MAXJ}} \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test "$@" --out ${{OUT}} || exit 1
}}
{calls}python3 experiments/EVAL/probe_released.py --model {SOPHON_MODEL} --checkpoints {SOPHON_TAG} --root {SOPHON_ROOT} \\
  --coarse {" ".join(COARSE)} --out {SOPHON_ROOT}/fits_v4 --probs-out {SOPHON_ROOT}/fits_v4_probs
python3 experiments/EVAL/qcd_hf_rejection.py --model {SOPHON_MODEL} --checkpoints {SOPHON_TAG} \\
  --probs-root {SOPHON_ROOT}/fits_v4_probs --features-root {SOPHON_ROOT} --out {SOPHON_ROOT}/fits_v4_qcdhf
echo "SOPHON PROBE COMPLETE"
"""
    return name, _indexed(name, script, 1, 1, True, pb.X_MEM, "4",
                          "the released Sophon model as a reference row of the frozen probes (PI 2026-11-07). GPU")


def _indices(spec: str) -> tuple:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parallelism", type=int, default=5)
    ap.add_argument("--only", default="", help="extract, fit: model indices, e.g. 1-13, for a job of that subset")
    ap.add_argument("--name", default="jc2bal-x-raunav", help="extract, fit: the job name (with --only)")
    ap.add_argument("kinds", nargs="+", choices=("stage", "ref", "ref2", "split", "extract", "fit", "mlp",
                                                 "masscurves", "kin", "xfill", "fitfill", "sib", "sibstage", "sibx",
                                                 "coarse", "qcdhf", "ckptid", "sophon"))
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        name, text = {"stage": lambda: stage_spec(a.tag), "ref": lambda: ref_spec(a.tag),
                      "ref2": lambda: ref_spec(a.tag, "impute_ref_v2.npz", "jc2bal-ref2-raunav"),
                      "xfill": lambda: xfill_spec(a.tag, a.parallelism),
                      "fitfill": lambda: fitfill_spec(a.tag, a.parallelism), "split": lambda: split_spec(a.tag),
                      "sib": lambda: sib_spec(a.tag, a.parallelism),
                      "sibstage": lambda: sibstage_spec(a.tag), "sibx": lambda: sibx_spec(a.tag, a.parallelism),
                      "extract": lambda: extract_spec(a.tag, a.parallelism, _indices(a.only), a.name),
                      "fit": lambda: fit_spec(a.tag, a.parallelism, False, _indices(a.only),
                                              a.name if a.only else ""),
                      "mlp": lambda: fit_spec(a.tag, a.parallelism, True),
                      "masscurves": lambda: masscurves_spec(a.tag), "kin": lambda: kin_spec(a.tag),
                      "coarse": lambda: coarse_spec(a.tag, a.parallelism), "qcdhf": lambda: qcdhf_spec(a.tag),
                      "ckptid": lambda: ckptid_spec(a.tag), "sophon": lambda: sophon_spec(a.tag)}[kind]()
        p = OUT_DIR / f"job-{name}.yaml"
        p.write_text(text)
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
