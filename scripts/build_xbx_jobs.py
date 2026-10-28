"""Jobs for X->bb and X->bs (docs/PRESPEC_2026-09.md A5; docs/SOPHON_XBB_XBS_PROTOCOL.md;
experiments/XBS/).

  xbx-stage-jc2          CPU. JetClass-II QCD and the stand-in X->bb: the evaluation jets of the
                         test files and the matched pair drawn from the held-out files
                         (experiments/XBS/stage_xbx.py). Needs nothing from outside.
  xbx-stage-ihep         CPU, once the outside samples are in SAMPLE_DIRS: each is checked against
                         the config (the job stops if a branch is missing or of the wrong kind),
                         converted, and its pair drawn against the held-out QCD.
  xbx-x-<model>          GPU. experiments/XBS/extract_xbx.py (extract_v2.py, unchanged) over every
                         staged split of every signal; a split already extracted is skipped.
  xbx-fit-<model>        GPU. experiments/XBS/fit_xbx.py model.
  xbx-scratch-<sig>-s<k> GPU, V100 as every fine-tune of the paper. From scratch as Sophon App. B.3:
                         ParT on the matched pair, 50 epochs of the whole pair, batch 512, LR 5e-4,
                         ranger, weight decay 0.01, weaver's flat+decay; resumable (cell_resume.py);
                         scored by fit_xbx.py scratch. A5: X->bs only, three seeds.

Models: the 27 paper runs (configs/arms/v2_paper_runs.txt) from their primary checkpoint best70,
and the untrained networks init-s1..3 (the linear probe only).
Order: stage -> x -> fit per model; the scratch jobs after their pair is staged.

    python3 scripts/build_xbx_jobs.py [--signals xbb_jc2] [--pin-not-yet-tagged] [--check-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "XBS" / "k8s"
PIN = "mtx-s2.13"
# what the tag must carry for what the jobs run
NEEDED = {"experiments/XBS/extract_xbx.py": "FLAGS = ",
          "experiments/XBS/stage_xbx.py": "def pair(",
          "experiments/XBS/fit_xbx.py": "def scratch(",
          "experiments/XBS/xbx.py": "def matched_draw(",
          "configs/data/JetClassII_xbx.yaml": "truth_label: xbx_class",
          "tests/test_xbx_stage.py": "_outside_sample"}
CFG = "configs/data/JetClassII_xbx.yaml"
STAGE = "/data/xbx/v1"
EX_ROOT = "/data/results/eval/xbx"
FITS = f"{EX_ROOT}/fits"
SCRATCH_ROOT = "/data/results/xbx/scratch"
# where the PI's copy of the IHEP samples goes, one folder per sample as at the source
# (root://cceos.ihep.ac.cn:1094//eos/ihep/cms/store/user/coli/datasets/JetClassII/sm_ext/)
SAMPLE_DIRS = {"xbb": "/data/xbx/ihep/Xbb_ntuple", "xbs": "/data/xbx/ihep/Xbs_ntuple"}
# the stand-in until the samples come; then xbb and xbs (xbb_jc2 in xbb's place if the
# IHEP folder holds no X->bb sample)
SIGNALS = ("xbb_jc2",)
# off the grid's RTX 3090s and, while the last fine-tunes queue for them, the V100s (2026-10-28)
NOT_ON = ("NVIDIA-GeForce-RTX-3090", "Tesla-V100-SXM2-32GB")
SCRATCH_GPU = "Tesla-V100-SXM2-32GB"
SCRATCH_SIGNALS = ("xbs",)
SCRATCH_SEEDS = (1, 2, 3)
B3 = {"lr": "5e-4", "epochs": 50, "weight_decay": "0.01"}     # Sophon App. B.3
SELF_TESTS = ("tests/test_xbx.py", "tests/test_xbx_stage.py", "tests/test_xbx_fit.py")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BX = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
LPJ = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")


def models() -> list[tuple[str, str, str, int, int, str]]:
    """(model, run directory, arm, classes, regression outputs, checkpoint)."""
    paper = [ln.strip() for ln in (ROOT / "configs/arms/v2_paper_runs.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    grid = {run: (arm, k, reg) for run, arm, k, reg, _s in BX.v2_runs()}
    out = [(run, run, *grid[run], "best70") for run in paper]
    out += [(ref, run, "L188", 0, 0, "init") for ref, run in BX.v2_init_refs()]
    return out


def gpu_off(text: str) -> str:
    """linprobe's GPU rules (off the grid's product, its fault handling) and off NOT_ON."""
    text = LPJ.off_grid_gpu(text, extract=False)
    old = "values: [" + ", ".join(f'"{p}"' for p in LPJ.NOT_ON) + "]"
    assert text.count(old) == 1
    return text.replace(old, "values: [" + ", ".join(f'"{p}"' for p in NOT_ON) + "]")


def _need(*paths: str) -> str:
    return "".join(f'          [ -f {p} ] || {{ echo "FATAL: {p} absent: stage first"; exit 42; }}\n' for p in paths)


def x_spec(model, run, arm, k, reg, ckpt, signals) -> str:
    rung = BX.v2_rung(arm) if k else "none"
    body = (_need(*[f"{STAGE}/pairs/{s}/DONE" for s in signals], *[f"{STAGE}/{s}/eval/DONE" for s in signals],
                  f"{STAGE}/qcd/eval/DONE")
            + "          extract () {\n"
            f"            OUT={EX_ROOT}/{model}/$1; shift\n"
            f"            python3 experiments/XBS/extract_xbx.py --arm {arm} --run-dir {BX.V2_ROOT}/{run} --rung {rung} \\\n"
            f"              --num-classes {k} --num-reg {reg} --checkpoints {ckpt} --data-config {CFG} \\\n"
            f"              --observers jet_pt jet_eta jet_sdmass --no-pooled \\\n"
            "              --data-test \"$@\" --out ${OUT} || halt\n"
            "          }\n")
    for s in signals:
        body += (f"          extract {s}/train $(ls {STAGE}/pairs/{s}/train_*.parquet | sort)\n"
                 f"          extract {s}/val {STAGE}/pairs/{s}/val.parquet\n"
                 f"          extract {s}/eval $(ls {STAGE}/{s}/eval/*.parquet | sort)\n")
    body += (f"          extract qcd/eval $(ls {STAGE}/qcd/eval/*.parquet | sort)\n"
             f'          echo "every X->bx split of {model} extracted"\n')
    return gpu_off(BX.V1ERR_TEMPLATE.format(
        name=f"xbx-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=PIN, body=body,
        mem="48Gi", cpu="4", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def fit_spec(model, ckpt, signals) -> str:
    body = (f"          python3 experiments/XBS/fit_xbx.py model --root {EX_ROOT} --model {model} \\\n"
            f"            --signals {' '.join(signals)} --checkpoint {ckpt} --out {FITS} || halt\n")
    return gpu_off(BX.V1ERR_TEMPLATE.format(
        name=f"xbx-fit-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=PIN, body=body,
        mem="48Gi", cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def _cpu(name: str, body: str) -> str:
    return BX.V1ERR_TEMPLATE.format(name=name, image=BX.IMAGE, pin=PIN, body=body, mem="32Gi", cpu="4",
                                    gpu_req="", gpu_check="", node_exclude=BX.NODE_EXCLUDE)


def _unless_done(done: str, cmd: str) -> str:
    """A step whose output is complete is skipped: stage_xbx.py refuses to rebuild one."""
    return f"          [ -f {done} ] || python3 experiments/XBS/stage_xbx.py --root {STAGE} {cmd} || halt\n"


def stage_jc2_spec() -> str:
    # the X->bx tests in the image itself first: the laptop's uproot cannot write ROOT under
    # numpy 2, so the outside-sample test runs only here, and none may be skipped
    body = ("          pip install --no-cache-dir -q pytest || exit 1\n"
            f"          python3 -m pytest -q -p no:cacheprovider -rs {' '.join(SELF_TESTS)} 2>&1 | tee /tmp/xbx_tests.log || halt\n"
            '          if grep -q "skipped" /tmp/xbx_tests.log; then echo "FATAL: an X->bx test was skipped in the image"; exit 42; fi\n'
            + _unless_done(f"{STAGE}/qcd/eval/DONE", "jc2-eval --name qcd")
            + _unless_done(f"{STAGE}/xbb_jc2/eval/DONE", "jc2-eval --name xbb_jc2")
            + _unless_done(f"{STAGE}/pairs/xbb_jc2/DONE", "pair --signal xbb_jc2")
            + f"          du -sh {STAGE}/*\n")
    return _cpu("xbx-stage-jc2-raunav", body)


def stage_ihep_spec() -> str:
    body = f"          mkdir -p {STAGE}\n"
    for s, d in SAMPLE_DIRS.items():
        body += (f"          F_{s}=$(ls {d}/*.root 2>/dev/null | sort || true)\n"
                 f'          [ -n "${{F_{s}}}" ] || {{ echo "FATAL: no files in {d}"; exit 42; }}\n'
                 f"          python3 experiments/XBS/stage_xbx.py check --files ${{F_{s}}} --out {STAGE}/check_{s}.json || halt\n"
                 + _unless_done(f"{STAGE}/{s}/eval/DONE", f"convert --name {s} --files ${{F_{s}}}")
                 + _unless_done(f"{STAGE}/pairs/{s}/DONE", f"pair --signal {s}"))
    return _cpu("xbx-stage-ihep-raunav", body + f"          du -sh {STAGE}/*\n")


SCRATCH_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_xbx_jobs.py -- do not hand-edit. Regenerate.
  # X->{sig} FROM SCRATCH, seed {seed} (Sophon App. B.3; docs/PRESPEC_2026-09.md A5): ParT on the
  # matched pair {stage}/pairs/{sig}, scored on the evaluation jets by fit_xbx.py scratch.
  # RETRIES as the v2 fine-tunes: evictions are not counted, exit 42 fails the Job at once, an
  # interrupted cell resumes from its last completed epoch (experiments/FT/cell_resume.py).
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 20
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Count
      onExitCodes: {{ containerName: main, operator: In, values: [43] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {image}
        command: ["/bin/bash", "-c"]
        env:
        - name: NODE_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: spec.nodeName }} }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        args:
        - |
          set -euo pipefail
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          pip install --no-cache-dir -q pyarrow || exit 1
          export PYTHONUNBUFFERED=1
          HALT=42   # the pod failure policy fails the Job at once on this code
          NODE_FAULT=43   # ...and counts this one against the retries
          PAIR={stage}/pairs/{sig}
          OUT={out}
{need}          [ -f ${{OUT}}/DONE ] && {{ echo "${{OUT}} is DONE"; exit 0; }}
          space_ok () {{ local p=$(df --output=pcent /data | tail -1 | tr -dc 0-9); echo "/data ${{p}}% used"; [ "$p" -lt 85 ] || {{ echo "FATAL: /data at ${{p}}%: stop and ask the PI"; exit ${{HALT}}; }}; }}
          FAIL_MARK=${{OUT}}.FAILED
          [ -f ${{FAIL_MARK}} ] && {{ echo "FATAL: an earlier attempt halted the job: $(cat ${{FAIL_MARK}})"; exit ${{HALT}}; }}
          CELL=""
          on_exit () {{
            local rc=$?
            [ ${{rc}} -ne 0 ] && [ ${{rc}} -ne ${{HALT}} ] && [ ${{rc}} -ne ${{NODE_FAULT}} ] && [ -n "${{CELL}}" ] && [ -d "${{CELL}}" ] && [ ! -f "${{CELL}}/DONE" ] || return 0
            local why="rc=${{rc}} pod=${{POD_NAME}} node=${{NODE_NAME}} $(date -u +%FT%TZ)"
            if ! python3 experiments/FT/gpu_probe.py || grep -qsE "CUDA error: unknown error|Cannot access accelerator device|GPU is lost|CUDA-capable device\\(s\\) is/are busy or unavailable|uncorrectable ECC error" "${{CELL}}/stdout.log"; then
              echo "${{why}}" > "$(mktemp "${{CELL}}/NODE_FAULT.XXXXXX")"
              exit ${{NODE_FAULT}}
            fi
            echo "${{why}}" > "${{CELL}}/ATTEMPT_FAILED"
          }}
          trap on_exit EXIT
          python3 experiments/FT/gpu_probe.py || {{ echo "preflight: no working GPU on ${{NODE_NAME}}"; exit ${{NODE_FAULT}}; }}
          mkdir -p ${{OUT%/*}}
          HELD=$(python3 experiments/FT/cell_resume.py lock --path ${{OUT}}.lock --owner {name})
          case "${{HELD}}" in
            acquired|reentered) ;;
            *) echo "FATAL: ${{OUT}} is held by ${{HELD}}"; exit ${{HALT}};;
          esac
          space_ok
          E=$(python3 experiments/FT/cell_resume.py prepare --dir ${{OUT}} --max-failed 2 --max-node-faults 4 --max-stalled 3) || {{
            echo "${{OUT}}: ${{E}}" | tee ${{FAIL_MARK}}; exit ${{HALT}}; }}
          mkdir -p ${{OUT}}
          CELL=${{OUT}}
          RESUME=(); [ "${{E}}" -lt 0 ] || RESUME=(--load-epoch ${{E}})
          echo "$(date -u +%FT%TZ) pod=${{POD_NAME}} node=${{NODE_NAME}} from_epoch=${{E}}" >> ${{OUT}}/ATTEMPTS
          TRAIN=$(ls ${{PAIR}}/train_*.parquet | sort | tr '\\n' ' ')
          N=$(python3 -c "import json; print(json.load(open('${{PAIR}}/manifest.json'))['train_draw']['n_per_class'])")
          SPE=$((2 * N))
          [ "${{E}}" -ge 0 ] || python3 experiments/FT/smoke_checks.py manifest --out ${{OUT}}/ft_manifest.json leg=xbx_{sig} init=scratch checkpoint= n_train=${{SPE}} ft_seed={seed} lr={lr} head_lr_mult=1 weight_decay={wd} epochs={epochs} lr_schedule=flat+decay recipe=sophon-b3 data_config={cfg} num_classes=2 batch_size=512 samples_per_epoch=${{SPE}} steps_per_epoch=$((SPE/512)) val_rule=fixed-sample-whole-pass
          python3 experiments/FT/ft_weaver.py --seed {seed} ${{RESUME[@]+"${{RESUME[@]}}"}} --lean-val-metrics \\
            --data-train ${{TRAIN}} --data-val ${{PAIR}}/val.parquet --data-config {cfg} \\
            --network-config experiments/MTX/ParT_sophon_arch_mtx.py -o num_classes 2 -o fc_params '[(512,0.1)]' \\
            --use-amp --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1 --optimizer ranger \\
            --optimizer-option weight_decay {wd} --start-lr {lr} --num-epochs {epochs} \\
            --samples-per-epoch ${{SPE}} --steps-per-epoch-val -1 \\
            --model-prefix ${{OUT}}/net --log ${{OUT}}/train.log 2>&1 | tee ${{OUT}}/stdout.log
          grep -q "('steps_per_epoch', $((SPE/512)))" ${{OUT}}/train.log || {{
            echo "FATAL: ${{OUT}}: weaver did not run $((SPE/512)) steps per epoch"; exit 1; }}
          if grep -q "AvgLoss: nan" ${{OUT}}/train.log 2>/dev/null; then
            echo "FATAL: ${{OUT}} diverged (NaN training loss)"; exit 1; fi
          python3 experiments/FT/cell_resume.py best --dir ${{OUT}} --epochs {epochs}
          touch ${{OUT}}/TRAINED
          for P in sig qcd; do
            case ${{P}} in sig) D={stage}/{sig}/eval;; qcd) D={stage}/qcd/eval;; esac
            python3 experiments/EVAL/extract_features.py --checkpoint ${{OUT}}/net_best_epoch_state.pt \\
              --num-classes 2 --arm XBX_{sig}_scratch_s{seed}_${{P}} --data-config {cfg} \\
              --data-test $(ls ${{D}}/*.parquet | sort) --observers jet_sdmass --out ${{OUT}}/features_${{P}} \\
              --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
            rm -f ${{OUT}}/features_${{P}}/features.npy
          done
          python3 experiments/XBS/fit_xbx.py scratch --sig ${{OUT}}/features_sig --qcd ${{OUT}}/features_qcd \\
            --out ${{OUT}}/endpoints.json
          [ -f ${{OUT}}/net_best_epoch_state.pt ] || {{
            echo "FATAL: no net_best_epoch_state.pt in ${{OUT}}; refusing to prune"; exit 1; }}
          rm -f ${{OUT}}/net_epoch-*_state.pt ${{OUT}}/net_epoch-*_optimizer.pt
          touch ${{OUT}}/DONE
          rm -rf ${{OUT}}.lock
          CELL=""
          echo "XBX SCRATCH {sig} s{seed} COMPLETE"
        resources:
          requests: {{ memory: "88Gi", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "88Gi", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
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
                values: ["{gpu}"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{bad_nodes}]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


def scratch_spec(sig: str, seed: int) -> str:
    name = f"xbx-scratch-{sig.replace('_', '')}-s{seed}-raunav"
    return SCRATCH_TEMPLATE.format(
        sig=sig, seed=seed, name=name, image=BX.IMAGE, pin=PIN, stage=STAGE, cfg=CFG, gpu=SCRATCH_GPU,
        out=f"{SCRATCH_ROOT}/{sig}/s{seed}", lr=B3["lr"], wd=B3["weight_decay"], epochs=B3["epochs"],
        need=_need(f"{STAGE}/pairs/{sig}/DONE", f"{STAGE}/{sig}/eval/DONE", f"{STAGE}/qcd/eval/DONE"),
        bad_nodes=", ".join(f'"{n}"' for n in BX.V2_GPU_FAULT_NODES))


def build(signals=SIGNALS) -> dict[str, str]:
    specs = {"job-xbx-stage-jc2-raunav.yaml": stage_jc2_spec(),
             "job-xbx-stage-ihep-raunav.yaml": stage_ihep_spec()}
    for model, run, arm, k, reg, ckpt in models():
        stem = model.removeprefix("mtx-")
        specs[f"job-xbx-x-{stem}-raunav.yaml"] = x_spec(model, run, arm, k, reg, ckpt, signals)
        specs[f"job-xbx-fit-{stem}-raunav.yaml"] = fit_spec(model, ckpt, signals)
    for s in SCRATCH_SIGNALS:
        for seed in SCRATCH_SEEDS:
            specs[f"job-xbx-scratch-{s.replace('_', '')}-s{seed}-raunav.yaml"] = scratch_spec(s, seed)
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signals", nargs="+", default=list(SIGNALS))
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    BX.verify_pin(PIN, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    specs = build(tuple(a.signals))
    if not a.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (OUT_DIR / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(OUT_DIR.relative_to(ROOT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
