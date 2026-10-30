"""Jobs for X->bb and X->bs (docs/PRESPEC_2026-09.md A5; docs/SOPHON_XBB_XBS_PROTOCOL.md;
experiments/XBS/).

  xbx-stage-jc2          CPU. JetClass-II QCD and the stand-in X->bb: the evaluation jets of the
                         test files and the matched pair drawn from the held-out files
                         (experiments/XBS/stage_xbx.py). Needs nothing from outside.
  xbx-x-<model>          GPU. experiments/XBS/extract_xbx.py (extract_v2.py, unchanged) over every
                         staged split of every signal; a split already extracted is skipped.
  xbx-fit-<model>        GPU. experiments/XBS/fit_xbx.py model.
These ran on the stand-in at PIN, window 150-230 GeV, fits in FITS; they stay as they ran.
Superseded, never launched, kept only because their files are on disk: xbx-stage-ihep and
xbx-scratch-xbs-s<k> (PIN; they stop when one sample is missing and score in 150-230 GeV).

The real samples (A5; PRESPEC A15), at PIN_IHEP, mass window xbx.window_for(--mx) passed
explicitly (Sophon's 150-230 GeV at mX = 200, scaled by mX/200; 93.75-143.75 GeV at 125):
  xbx-ihep-stage         CPU, once the samples are in SAMPLE_DIRS: stages whichever is there. Each
                         is inspected (trees, branches, types, entries) and checked (stage_xbx.py
                         check: branches after BRANCH_MAP, no zero-fill in any file, A5's floor of
                         10,000 evaluation jets in the window); one that passes is converted and its
                         pair drawn against the held-out QCD. A sample that fails does not stop the
                         other; the job exits 42 at the end if any failed.
  xbx-ihep-<sig>-x-<model>
                         GPU, as xbx-x on one sample (the QCD evaluation jets, extracted for the
                         stand-in from the same staged files, are skipped). One job per sample, so
                         a sample that comes alone or late is extracted and fitted on its own.
  xbx-ihep-<sig>-fit-<model>
                         GPU, fit_xbx.py model into FITS_IHEP/<sig> (a fit never overwrites one).
  xbx-ihep-scratch-xbs-s<k>
                         GPU, V100 as every fine-tune of the paper. From scratch as Sophon App. B.3
                         in epochs, batch, LR and weight decay, but the Sophon network (our ParT with
                         Sophon's (512, 0.1) head) on scaled inputs, where Sophon's reference is ParT
                         on unscaled inputs: 50 epochs of the whole pair (50 x 2.5 M examples at
                         1.25 M jets per class), ranger, weaver's flat+decay; resumable
                         (cell_resume.py); scored by fit_xbx.py scratch with the examples it saw.
                         A5: X->bs only, three seeds.
  xbx-scratch-dryrun     the same chain for 2 epochs on the staged stand-in pair (xbb_jc2, window
                         150-230 GeV, 200,000 evaluation jets of each class), to prove it before the
                         samples arrive.

Models: the 27 paper runs (configs/arms/v2_paper_runs.txt) from their primary checkpoint best70,
and the untrained networks init-s1..3 (the linear probe only).
Order: stage -> x -> fit per model; the scratch jobs after their pair is staged.

    python3 scripts/build_xbx_jobs.py [--pin-not-yet-tagged] [--check-only]
(mX: change MX below once the samples' mass is confirmed; --mx writes specs that differ from
the default build, which tests/test_xbx_jobs.py holds equal to the files on disk.)
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "XBS" / "k8s"
PIN = "mtx-s2.17"     # the stand-in chain as it ran
PIN_IHEP = "mtx-s2.20"   # NOT YET TAGGED; mtx-s2.18 is the 188-class probe refit (4f92021), without this code
# mtx-s2.20: linear_probe_v2.py and probe.py changed after mtx-s2.19 (da6e0be); no IHEP spec had run
# mtx-s2.16: the staged files carried the source's awkward description without xbx_class, which
# awkward (so weaver) refuses; both dry-run extractions failed on their first file
# mtx-s2.15: a commit that held the config move without the code that reads the new path; never cloned
# mtx-s2.14: the data config sat in configs/data, which tests/test_pretrain_v2.py holds byte-identical
# to the commits pretraining may resume from; moved to configs/xbx, its content unchanged
# mtx-s2.13: the first staging attempt halted on its own ROOT test fixture (uproot cannot write
# option types); the code it tested was unchanged
# what the tag must carry for what the jobs run
NEEDED = {"experiments/XBS/extract_xbx.py": "FLAGS = ",
          "experiments/XBS/stage_xbx.py": "def pair(",
          "experiments/XBS/fit_xbx.py": "def scratch(",
          "experiments/XBS/xbx.py": "def matched_draw(",
          "configs/xbx/JetClassII_xbx.yaml": "truth_label: xbx_class",
          "tests/test_xbx_stage.py": "_outside_sample"}
# mtx-s2.19: the real samples: whichever is present staged, zero-fill (per file) and A5 floor
# enforced, a branch map and inspect; the window and mX explicit and consistent; fits to fits_ihep,
# never overwritten; Eq. 4 only where X->bb is a class of its own; the scratch fit records the
# examples seen; xbx.json written once
NEEDED_IHEP = {**NEEDED,
               "experiments/XBS/extract_xbx.py": "records another vocabulary",
               "experiments/XBS/stage_xbx.py": "flagged_charged_without_charge",
               "experiments/XBS/fit_xbx.py": "X.window_for(a.mx)",
               "experiments/XBS/xbx.py": "def window_for(",
               "tests/test_xbx_stage.py": "test_a_partly_zero_filled_file_is_refused"}
CFG = "configs/xbx/JetClassII_xbx.yaml"
STAGE = "/data/xbx/v2"   # v1 (2026-10-28) holds files awkward cannot read; kept, not reused
EX_ROOT = "/data/results/eval/xbx"
FITS = f"{EX_ROOT}/fits"
FITS_IHEP = f"{EX_ROOT}/fits_ihep"
SCRATCH_ROOT = "/data/results/xbx/scratch"
SCRATCH_ROOT_IHEP = "/data/results/xbx/scratch_ihep"
DRYRUN_ROOT = "/data/results/xbx/scratch_dryrun"
# where the PI's copy of the samples goes, one folder per sample as at the source (UAF
# /ceph/cms/store/user/woodson/{Xbb,Xbs}_ntuple, 30 ROOT files each)
SAMPLE_DIRS = {"xbb": "/data/xbx/ihep/Xbb_ntuple", "xbs": "/data/xbx/ihep/Xbs_ntuple"}
IHEP_SIGNALS = ("xbb", "xbs")
MX = 125.0     # the PI: "the SM (mX=125) Xbb and Xbs samples"; confirm on arrival (inspect, check's m_SD peak)
# the stand-in until the samples come; then xbb and xbs (xbb_jc2 in xbb's place if the
# IHEP folder holds no X->bb sample)
SIGNALS = ("xbb_jc2",)
# off the grid's RTX 3090s and, while the last fine-tunes queue for them, the V100s (2026-10-28)
NOT_ON = ("NVIDIA-GeForce-RTX-3090", "Tesla-V100-SXM2-32GB")
SCRATCH_GPU = "Tesla-V100-SXM2-32GB"
# the dry run proves the plumbing only, so it does not wait for a V100 behind the fine-tuning
# (2026-10-30: every V100 in use): any of these, which the fine-tunes do not use
DRYRUN_GPUS = ("NVIDIA-A10", "NVIDIA-L40", "NVIDIA-RTX-A6000", "NVIDIA-A100-SXM4-80GB", "NVIDIA-L40S")
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
X = _load("xbx", "experiments/XBS/xbx.py")


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


def x_spec(model, run, arm, k, reg, ckpt, signals, prefix="xbx", pin=PIN) -> str:
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
        name=f"{prefix}-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=pin, body=body,
        mem="48Gi", cpu="4", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def fit_spec(model, ckpt, signals, prefix="xbx", pin=PIN, out=FITS, window="") -> str:
    body = (f"          python3 experiments/XBS/fit_xbx.py model --root {EX_ROOT} --model {model} \\\n"
            f"            --signals {' '.join(signals)} --checkpoint {ckpt}{window} --out {out} || halt\n")
    return gpu_off(BX.V1ERR_TEMPLATE.format(
        name=f"{prefix}-fit-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=pin, body=body,
        mem="48Gi", cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def _cpu(name: str, body: str, pin=PIN) -> str:
    return BX.V1ERR_TEMPLATE.format(name=name, image=BX.IMAGE, pin=pin, body=body, mem="32Gi", cpu="4",
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
    """Superseded by ihep_stage_spec (kept: its file is on disk; never launched)."""
    body = f"          mkdir -p {STAGE}\n"
    for s, d in SAMPLE_DIRS.items():
        body += (f"          F_{s}=$(ls {d}/*.root 2>/dev/null | sort || true)\n"
                 f'          [ -n "${{F_{s}}}" ] || {{ echo "FATAL: no files in {d}"; exit 42; }}\n'
                 f"          python3 experiments/XBS/stage_xbx.py check --files ${{F_{s}}} --out {STAGE}/check_{s}.json || halt\n"
                 + _unless_done(f"{STAGE}/{s}/eval/DONE", f"convert --name {s} --files ${{F_{s}}}")
                 + _unless_done(f"{STAGE}/pairs/{s}/DONE", f"pair --signal {s}"))
    return _cpu("xbx-stage-ihep-raunav", body + f"          du -sh {STAGE}/*\n")


def ihep_stage_spec(window: str) -> str:
    """Whichever sample is present is inspected and checked; one that passes is converted and
    paired. A failed check does not stop the other sample; the job exits 42 at the end if any
    sample failed, or if none is present."""
    stage = lambda cmd: f"python3 experiments/XBS/stage_xbx.py --root {STAGE} {cmd} || halt"
    body = f"          mkdir -p {STAGE}\n          STAGED=\"\"; FAILED=\"\"\n"
    for s, d in SAMPLE_DIRS.items():
        body += (f"          F_{s}=$(ls {d}/*.root 2>/dev/null | sort || true)\n"
                 f"          if [ -z \"${{F_{s}}}\" ]; then\n"
                 f"            echo \"no files in {d}: {s} not staged\"\n"
                 f"          elif python3 experiments/XBS/stage_xbx.py inspect --file \"${{F_{s}%%[[:space:]]*}}\" && \\\n"
                 f"               python3 experiments/XBS/stage_xbx.py check {window} --files ${{F_{s}}} --out {STAGE}/check_{s}.json; then\n"
                 f"            [ -f {STAGE}/{s}/eval/DONE ] || {stage(f'convert --name {s} {window} --files ${{F_{s}}}')}\n"
                 f"            [ -f {STAGE}/pairs/{s}/DONE ] || {stage(f'pair --signal {s}')}\n"
                 f"            STAGED=\"${{STAGED}} {s}\"\n"
                 f"          else\n"
                 f"            echo \"{s} failed its inspection or check ({STAGE}/check_{s}.json): not staged\"; FAILED=\"${{FAILED}} {s}\"\n"
                 f"          fi\n")
    body += (f'          [ -n "${{STAGED}}${{FAILED}}" ] || {{ echo "FATAL: no sample in {" or ".join(SAMPLE_DIRS.values())}"; exit 42; }}\n'
             f'          echo "staged:${{STAGED}}"\n'
             f"          du -sh {STAGE}/* || true\n"
             f'          [ -z "${{FAILED}}" ] || {{ echo "FATAL: failed its check:${{FAILED}}"; exit 42; }}\n')
    return _cpu("xbx-ihep-stage-raunav", body, pin=PIN_IHEP)


SCRATCH_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_xbx_jobs.py -- do not hand-edit. Regenerate.
  # X->{sig} FROM SCRATCH, seed {seed} (Sophon App. B.3; docs/PRESPEC_2026-09.md A5): {what} on the
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
              --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits{extract_args}
            rm -f ${{OUT}}/features_${{P}}/features.npy
          done
          {fit_guard}python3 experiments/XBS/fit_xbx.py scratch --sig ${{OUT}}/features_sig --qcd ${{OUT}}/features_qcd \\
            --out ${{OUT}}/endpoints.json{fit_args}
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
                values: [{gpu}]
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


def scratch_spec(sig: str, seed: int, name=None, pin=PIN, out_root=SCRATCH_ROOT, epochs=B3["epochs"],
                 mx=None, extract_args="", gpus=(SCRATCH_GPU,)) -> str:
    """mx None: the superseded spec as it was written; else the window and mX are passed, the
    examples seen recorded, and an endpoints.json already written is not redone."""
    name = name or f"xbx-scratch-{sig.replace('_', '')}-s{seed}-raunav"
    new = mx is not None
    return SCRATCH_TEMPLATE.format(
        sig=sig, seed=seed, name=name, image=BX.IMAGE, pin=pin, stage=STAGE, cfg=CFG, gpu=", ".join(f'"{g}"' for g in gpus),
        out=f"{out_root}/{sig}/s{seed}", lr=B3["lr"], wd=B3["weight_decay"], epochs=epochs,
        need=_need(f"{STAGE}/pairs/{sig}/DONE", f"{STAGE}/{sig}/eval/DONE", f"{STAGE}/qcd/eval/DONE"),
        bad_nodes=", ".join(f'"{n}"' for n in BX.V2_GPU_FAULT_NODES),
        what="the Sophon network (scaled inputs)" if new else "ParT", extract_args=extract_args,
        fit_guard="[ -f ${OUT}/endpoints.json ] || " if new else "",
        fit_args=f" {window_arg(mx)} --mx {mx:g} --ft-manifest ${{OUT}}/ft_manifest.json" if new else "")


def window_arg(mx: float) -> str:
    """xbx.window_for(mx) as stage_xbx.py and fit_xbx.py take it."""
    lo, hi = X.window_for(mx)
    return f"--window {lo:g} {hi:g}"


def build(signals=SIGNALS, mx=MX) -> dict[str, str]:
    specs = {"job-xbx-stage-jc2-raunav.yaml": stage_jc2_spec(),
             "job-xbx-stage-ihep-raunav.yaml": stage_ihep_spec()}
    for model, run, arm, k, reg, ckpt in models():
        stem = model.removeprefix("mtx-")
        specs[f"job-xbx-x-{stem}-raunav.yaml"] = x_spec(model, run, arm, k, reg, ckpt, signals)
        specs[f"job-xbx-fit-{stem}-raunav.yaml"] = fit_spec(model, ckpt, signals)
    for s in SCRATCH_SIGNALS:
        for seed in SCRATCH_SEEDS:
            specs[f"job-xbx-scratch-{s.replace('_', '')}-s{seed}-raunav.yaml"] = scratch_spec(s, seed)
    # the real samples
    specs["job-xbx-ihep-stage-raunav.yaml"] = ihep_stage_spec(window_arg(mx))
    for sig in IHEP_SIGNALS:
        for model, run, arm, k, reg, ckpt in models():
            stem, pre = model.removeprefix("mtx-"), f"xbx-ihep-{sig}"
            specs[f"job-{pre}-x-{stem}-raunav.yaml"] = x_spec(model, run, arm, k, reg, ckpt, (sig,),
                                                              prefix=pre, pin=PIN_IHEP)
            specs[f"job-{pre}-fit-{stem}-raunav.yaml"] = fit_spec(model, ckpt, (sig,), prefix=pre, pin=PIN_IHEP,
                                                                  out=f"{FITS_IHEP}/{sig}",
                                                                  window=f" {window_arg(mx)} --mx {mx:g}")
    for s in SCRATCH_SIGNALS:
        for seed in SCRATCH_SEEDS:
            specs[f"job-xbx-ihep-scratch-{s}-s{seed}-raunav.yaml"] = scratch_spec(
                s, seed, name=f"xbx-ihep-scratch-{s}-s{seed}-raunav", pin=PIN_IHEP, out_root=SCRATCH_ROOT_IHEP, mx=mx)
    # the from-scratch chain proven on the stand-in: Sophon's own window (mX = 200)
    specs["job-xbx-scratch-dryrun-raunav.yaml"] = scratch_spec(
        "xbb_jc2", 1, name="xbx-scratch-dryrun-raunav", pin=PIN_IHEP, out_root=DRYRUN_ROOT, epochs=2,
        mx=X.SOPHON_MX, extract_args=" --max-jets 200000", gpus=DRYRUN_GPUS)
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signals", nargs="+", default=list(SIGNALS))
    ap.add_argument("--mx", type=float, default=MX, help="the samples' resonance mass, GeV")
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    BX.verify_pin(PIN, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    BX.verify_pin(PIN_IHEP, list(NEEDED_IHEP), a.pin_not_yet_tagged, NEEDED_IHEP)
    specs = build(tuple(a.signals), a.mx)
    if not a.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (OUT_DIR / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(OUT_DIR.relative_to(ROOT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
