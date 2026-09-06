#!/usr/bin/env python3
"""Emit experiments/FT/k8s/job-ft-*-raunav.yaml -- the fine-tuning legs of the
journal paper (DECISIONS_PENDING item 14), their subset builders, and the CPU
smoke that validates every code path on the image before a GPU is spent.

    job-ft-subsets-jc2-raunav   CPU   nested N-jet JetClass-II subsets, 3 seeds
    job-ft-subsets-jc1-raunav   CPU   nested N-jet JetClass-I subsets, 3 seeds
    job-ft-smoke-raunav         CPU   hybrid mass loop, both fine-tune paths,
                                      subset writers, checkpoint loads -- tiny
    job-ft-legs-raunav          GPU   leg 1 (in-domain recovery) then leg 2
                                      (JetClass-I, the pileup shift), every
                                      (init, N, seed), resumable per fine-tune

THE DESIGN THE LEGS IMPLEMENT (item 14, addendum 2)
---------------------------------------------------
inits    r16q1-s2 / s3 / s4 (coarse, three pretraining seeds), l162-s1b (fine),
         sophon-public (the released 188-class model, an UNCONTROLLED reference
         row: different recipe, hardware and seed), scratch
N        1e4, 1e5, 1e6 jets, nested, from experiments/FT/make_subsets.py
seeds    3 fine-tuning seeds per (init, N) -- the field's 2026 standard
         (2606.14870 §III.B: 3; 2606.19781: 5; 2607.23377: 5)
epochs   50 / 30 / 10 at N = 1e4 / 1e5 / 1e6 (5e5, 3e6, 1e7 examples); weaver
         keeps the best-validation epoch; every epoch checkpoint is kept
rate     1e-4 for a pretrained trunk, 5e-4 (the pretraining rate) from scratch
head     re-initialised: --exclude-model-weights 'mod\\.fc\\..*'
leg 1    R16_Q1 -> the 162-way vocabulary on JetClass-II; readout = frozen
         features on the paper's own 2,000,000-jet test subset, the SAME file
         list (hence the same jets, in the same order) as features_v2, taken
         verbatim from job-extract-mtx-r16q1-s2-raunav.yaml, plus the 162 logits
leg 2    the 10-class JetClass-I task at its Sophon preprocessing (E1 arm S's
         config, weights: block removed); readout = weaver --predict on the
         first two test_20M files of every class; the E1 arm S checkpoints are
         scored on the SAME subset once, as the N_max scratch reference
no reweighting on subsets: configs/finetune/ (see build_finetune_configs.py)

ONE GPU JOB, SEQUENTIAL. CLAUDE.md caps running pods at five and this wave
already holds a training pod; a single pod holding one GPU for the whole sweep
is what the cap allows, and every fine-tune is resumable (DONE marker; a
partial directory is moved aside, never overwritten) so an eviction costs one
fine-tune, not the sweep.

Run:  python3 scripts/build_ft_jobs.py [--pin TAG]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "FT" / "k8s"
EXTRACT_SPEC = ROOT / "experiments" / "EVAL" / "k8s" / "job-extract-mtx-r16q1-s2-raunav.yaml"
PIN = "mtx-s1.8"
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"
LAMBDA = "5.0"
# E0b's pinned sha256 of the released Sophon checkpoint (job-massreg-e0b-extract).
SOPHON_SHA256 = "cc7c33b522e796b5bbf0aa9bb5b01361c964f4ef3acebdd9682d7519c095b824"

JC1_CLASSES = "HToBB HToCC HToGG HToWW2Q1L HToWW4Q TTBar TTBarLep WToQQ ZToQQ ZJetsToNuNu"

# name : checkpoint : K of that checkpoint's head (0 = from scratch)
INITS = [
    ("r16q1-s2", "/data/results/mtx/mtx-r16q1-s2/net_best_epoch_state.pt", 17),
    ("r16q1-s3", "/data/results/mtx/mtx-r16q1-s3/net_best_epoch_state.pt", 17),
    ("r16q1-s4", "/data/results/mtx/mtx-r16q1-s4/net_best_epoch_state.pt", 17),
    ("l162-s1b", "/data/results/mtx/mtx-l162-s1b/net_best_epoch_state.pt", 162),
    ("sophon-public", "/workspace/sophon_public.pt", 188),
    ("scratch", "", 0),
]
SIZES = [10_000, 100_000, 1_000_000]
EPOCHS = {10_000: 50, 100_000: 30, 1_000_000: 10}
FT_SEEDS = [1, 2, 3]
LR_PRETRAINED, LR_SCRATCH = "1e-4", "5e-4"


def test2m_list() -> str:
    """The 335-file --data-test list of the paper's feature extraction, verbatim."""
    text = EXTRACT_SPEC.read_text()
    m = re.search(r"--data-test ((?:/jc2/jet_data/\S+\.parquet ?)+)", text)
    if not m:
        sys.exit(f"FATAL: no --data-test list in {EXTRACT_SPEC}")
    files = m.group(1).split()
    if len(files) != 335:
        sys.exit(f"FATAL: expected 335 test files in {EXTRACT_SPEC.name}, found {len(files)}")
    return " ".join(files)


PREAMBLE = """          set -euo pipefail
          # Clone the pinned TAG (not a branch): every job in this wave runs the
          # same code, and a job that queues for days cannot pick up a later
          # commit when it finally starts.
          git clone --depth 1 --branch "${REPO_REF}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          pip install --no-cache-dir -q pyarrow || exit 1
          export PYTHONUNBUFFERED=1
"""

SPLIT_GUARD = """          # Split guard, as in every training job: count files that EXIST.
          TRAIN_FILES=(/jc2/jet_data/Res2P_{0000..0199}.parquet /jc2/jet_data/Res34P_{0000..0859}.parquet /jc2/jet_data/QCD_{0000..0279}.parquet)
          VAL_FILES=(/jc2/jet_data/Res2P_{0200..0249}.parquet /jc2/jet_data/Res34P_{0860..1074}.parquet /jc2/jet_data/QCD_{0280..0349}.parquet)
          n_present () { local n=0; for f in "$@"; do [ -f "$f" ] && n=$((n+1)); done; echo $n; }
          [ "$(n_present "${TRAIN_FILES[@]}")" -eq 1340 ] || { echo "FATAL: train split incomplete on the PVC"; exit 1; }
          [ "$(n_present "${VAL_FILES[@]}")" -eq 335 ] || { echo "FATAL: val split incomplete on the PVC"; exit 1; }
"""

SPACE_GUARD = """          # CLAUDE.md: check free space before a write of this size.
          FREE_G=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
          echo "free on /data: ${FREE_G}G"
          [ "${FREE_G}" -ge 100 ] || { echo "FATAL: ${FREE_G}G free on /data, need 100G"; exit 1; }
"""

FETCH_SOPHON = """          # The released checkpoint is gitignored, so it is never in the clone, and
          # it is not on the PVC. Fetch it as E0b did and pin it to E0b's sha256.
          SOPHON=/workspace/sophon_public.pt
          curl -fsSL -o "${SOPHON}" https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt
          GOT=$(sha256sum "${SOPHON}" | cut -d' ' -f1)
          [ "${GOT}" = "__SOPHON_SHA256__" ] || { echo "FATAL: public checkpoint sha256 ${GOT} != E0b pin __SOPHON_SHA256__"; exit 1; }
"""

SUBSETS_JC2 = PREAMBLE + """
          OUT=/data/finetune/jc2
          [ -f ${OUT}/DONE ] && { echo "already built:"; head -40 ${OUT}/manifest.json; exit 0; }
""" + SPLIT_GUARD + SPACE_GUARD + """
          # 60 train files per seed (9 Res2P / 39 Res34P / 12 QCD, proportional),
          # 30% of each file's SELECTED rows -> a ~1.5M-row pool per seed, of which
          # the nested 1e4 / 1e5 / 1e6 subsets are prefixes of one shuffle.
          python3 experiments/FT/make_subsets.py jc2 \\
            --train-files "${TRAIN_FILES[@]}" --val-files "${VAL_FILES[@]}" \\
            --out ${OUT} --sizes 10000 100000 1000000 --seeds 1 2 3 \\
            --n-files 60 --take-fraction 0.30 --val-size 200000 --n-val-files 12
          ls -la ${OUT}; du -sh ${OUT}
"""

SUBSETS_JC1 = PREAMBLE + """
          OUT=/data/finetune/jc1
          [ -f ${OUT}/DONE ] && { echo "already built:"; head -40 ${OUT}/manifest.json; exit 0; }
          for C in __JC1_CLASSES__; do
            n=$(find /data/JetClass/Pythia/train_100M -maxdepth 1 -name "${C}_*.root" 2>/dev/null | wc -l || true)
            v=$(find /data/JetClass/Pythia/val_5M -maxdepth 1 -name "${C}_*.root" 2>/dev/null | wc -l || true)
            echo "${C}: ${n} train files, ${v} val files"
            [ "${n}" -ge 2 ] && [ "${v}" -ge 1 ] || { echo "FATAL: ${C}: JetClass-I files missing on the PVC"; exit 1; }
          done
""" + SPACE_GUARD + """
          # Balanced: 1e5 jets per class from 2 random 100k-jet files per class
          # per seed; val: 2e4 per class from one val_5M file per class.
          python3 experiments/FT/make_subsets.py jc1 \\
            --train-dir /data/JetClass/Pythia/train_100M --val-dir /data/JetClass/Pythia/val_5M \\
            --out ${OUT} --sizes 10000 100000 1000000 --seeds 1 2 3 \\
            --files-per-class 2 --val-per-class 20000
          ls -la ${OUT}; du -sh ${OUT}
"""

SMOKE = PREAMBLE + """
          S=/data/results/ft/smoke/$(date -u +%Y%m%dT%H%M%SZ)
          mkdir -p ${S}
          python3 -c "import weaver, torch, awkward, uproot; print('weaver', getattr(weaver, '__version__', '?'), 'torch', torch.__version__, 'awkward', awkward.__version__, 'uproot', uproot.__version__)"
          CKPT=/data/results/mtx/mtx-r16q1-s2/net_best_epoch_state.pt
          [ -f "${CKPT}" ] || { echo "FATAL: no ${CKPT}"; exit 1; }
          # --gpus "" on EVERY weaver call: it defaults to GPU 0 and model_setup does
          # model.to(device) before anything else (see job-mtx-makeweight-raunav).
          W="--batch-size 128 --num-workers 1 --fetch-by-files --fetch-step 1 --optimizer ranger"

          echo "===== [1/5] hybrid class+mass loop, R16_Q1_MASS, 2 tiny epochs on CPU ====="
          mkdir -p ${S}/hybrid
          python3 experiments/E1/seed_weaver.py --seed 1 --lean-val-metrics --mass-lambda __LAMBDA__ \\
            --data-train Res2P:/jc2/jet_data/Res2P_0000.parquet Res34P:/jc2/jet_data/Res34P_0000.parquet QCD:/jc2/jet_data/QCD_0000.parquet \\
            --data-val /jc2/jet_data/Res2P_0200.parquet /jc2/jet_data/Res34P_0860.parquet /jc2/jet_data/QCD_0280.parquet \\
            --data-config configs/arms/R16_Q1_MASS.yaml \\
            --network-config experiments/MTX/ParT_sophon_arch_mass.py -o num_classes 17 -o fc_params '[(512,0.1)]' \\
            --gpus "" ${W} --start-lr 5e-4 --samples-per-epoch 2048 --samples-per-epoch-val 1024 --num-epochs 2 \\
            --model-prefix ${S}/hybrid/net --log ${S}/hybrid/train.log 2>&1 | tee ${S}/hybrid/stdout.log
          grep -q "hybrid class+mass loop installed" ${S}/hybrid/stdout.log || { echo "FATAL: hybrid loop not installed"; exit 1; }
          grep -q "AvgLossReg" ${S}/hybrid/stdout.log || { echo "FATAL: no regression loss logged -- stock loop ran"; exit 1; }
          python3 experiments/FT/smoke_checks.py head-width --checkpoint ${S}/hybrid/net_best_epoch_state.pt --expect 18

          echo "===== [2/5] subset writers, tiny ====="
          python3 experiments/FT/make_subsets.py jc2 \\
            --train-files /jc2/jet_data/Res2P_0001.parquet /jc2/jet_data/Res34P_0001.parquet /jc2/jet_data/QCD_0001.parquet \\
            --val-files /jc2/jet_data/Res2P_0201.parquet /jc2/jet_data/Res34P_0861.parquet /jc2/jet_data/QCD_0281.parquet \\
            --out ${S}/sub2 --sizes 1000 4000 --seeds 1 --n-files 3 --take-fraction 0.05 --val-size 500 --n-val-files 3
          python3 experiments/FT/make_subsets.py jc1 \\
            --train-dir /data/JetClass/Pythia/train_100M --val-dir /data/JetClass/Pythia/val_5M \\
            --out ${S}/sub1 --sizes 1000 4000 --seeds 1 --files-per-class 1 --val-per-class 50

          echo "===== [3/5] leg-1 path: R16_Q1 trunk -> 162-way head, 2 tiny epochs, features ====="
          python3 experiments/EVAL/extract_features.py --checkpoint ${CKPT} --num-classes 17 --arm R16_Q1 \\
            --data-test /jc2/jet_data/Res2P_0250.parquet --out ${S}/selfcheck --self-check-only
          mkdir -p ${S}/leg1
          python3 experiments/E1/seed_weaver.py --seed 1 --lean-val-metrics \\
            --data-train ${S}/sub2/train_N4000_s1.parquet --data-val ${S}/sub2/val.parquet \\
            --data-config configs/finetune/JetClassII_L162_noweight.yaml \\
            --network-config experiments/MTX/ParT_sophon_arch_mtx.py -o num_classes 162 -o fc_params '[(512,0.1)]' \\
            --gpus "" ${W} --start-lr 1e-4 --samples-per-epoch 4000 --samples-per-epoch-val 500 --num-epochs 2 \\
            --load-model-weights ${CKPT} --exclude-model-weights 'mod\\.fc\\..*' \\
            --model-prefix ${S}/leg1/net --log ${S}/leg1/train.log 2>&1 | tee ${S}/leg1/stdout.log
          python3 experiments/FT/smoke_checks.py load-log --log ${S}/leg1/stdout.log
          python3 experiments/FT/smoke_checks.py head-width --checkpoint ${S}/leg1/net_best_epoch_state.pt --expect 162
          python3 experiments/EVAL/extract_features.py --checkpoint ${S}/leg1/net_best_epoch_state.pt --num-classes 162 --arm FT_SMOKE \\
            --data-config configs/data/JetClassII_base.yaml \\
            --data-test /jc2/jet_data/Res2P_0250.parquet /jc2/jet_data/Res34P_1075.parquet /jc2/jet_data/QCD_0350.parquet \\
            --out ${S}/leg1/features_v2 --batch-size 256 --num-workers 1 --fetch-step 1 --max-jets 3000 --save-logits
          python3 experiments/FT/smoke_checks.py features --dir ${S}/leg1/features_v2 --n 3000 --k 162

          echo "===== [4/5] leg-2 path: JetClass-I parquet subset, 2 tiny epochs, predict ====="
          mkdir -p ${S}/leg2
          python3 experiments/E1/seed_weaver.py --seed 1 --lean-val-metrics \\
            --data-train ${S}/sub1/train_N4000_s1.parquet --data-val ${S}/sub1/val.parquet \\
            --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
            --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
            --gpus "" ${W} --start-lr 1e-4 --samples-per-epoch 4000 --samples-per-epoch-val 500 --num-epochs 2 \\
            --load-model-weights ${CKPT} --exclude-model-weights 'mod\\.fc\\..*' \\
            --model-prefix ${S}/leg2/net --log ${S}/leg2/train.log 2>&1 | tee ${S}/leg2/stdout.log
          python3 experiments/FT/smoke_checks.py load-log --log ${S}/leg2/stdout.log
          T1=$(ls /data/JetClass/Pythia/test_20M/HToBB_*.root | sort | head -1)
          weaver --predict --gpus "" --data-test ${S}/sub1/val.parquet ${T1} \\
            --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
            --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
            --model-prefix ${S}/leg2/net --predict-output ${S}/leg2/pred.root \\
            --batch-size 256 --num-workers 1 --fetch-by-files --fetch-step 1 2>&1 | tail -5
          python3 -c "import uproot,sys; f=uproot.open('${S}/leg2/pred.root'); t=f[[k for k in f.keys() if not k.startswith('_')][0]]; n=t.num_entries; s=[b for b in t.keys() if b.startswith('score_')]; print('pred entries', n, 'score branches', len(s)); sys.exit(0 if len(s)==10 and n>500 else 1)"

          echo "===== [5/5] the released Sophon checkpoint downloads, pins and loads as an init ====="
""" + FETCH_SOPHON + """          python3 experiments/EVAL/extract_features.py --checkpoint ${SOPHON} --num-classes 188 --arm SOPHON_PUBLIC \\
            --data-test /jc2/jet_data/Res2P_0250.parquet --out ${S}/selfcheck2 --self-check-only
          echo "SMOKE PASS ${S}"
"""

LEGS = PREAMBLE + """
          SUB2=/data/finetune/jc2
          SUB1=/data/finetune/jc1
          ROOT_OUT=/data/results/ft
          mkdir -p ${ROOT_OUT}

          # The subsets come from two CPU jobs applied BEFORE this one, which is
          # applied only once both DONE files exist; so this wait is a guard, not
          # a plan. Bounded at 2 h, and it leaves a marker: with backoffLimit 50
          # a bare exit 1 would re-queue and hold a GPU for 2 h per retry, the
          # marker makes every retry fail at once.
          WAIT_MARK=${ROOT_OUT}/WAIT_TIMEOUT
          [ -f ${WAIT_MARK} ] && { echo "FATAL: an earlier attempt timed out: $(cat ${WAIT_MARK}). Fix the producer, then remove ${WAIT_MARK}"; exit 1; }
          wait_for () { local f=$1; local t=0; until [ -f "$f" ]; do [ "$t" -ge 7200 ] && { echo "$f absent after 2 h" | tee ${WAIT_MARK}; exit 1; }; sleep 60; t=$((t+60)); done; }
          wait_for ${SUB2}/DONE
          wait_for ${SUB1}/DONE
""" + SPACE_GUARD + """
          # Every fine-tune writes up to ~4 GB (weaver keeps state + optimizer per
          # epoch; leg 1 adds 2M x (128 + 162) float32 features + logits): ~250 GB
          # over the wave. CLAUDE.md: no write > 1 GB past 85% without the PI.
          space_ok () { local p=$(df --output=pcent /data | tail -1 | tr -dc 0-9); local g=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); echo "/data ${p}% used, ${g}G free"; [ "$p" -lt 85 ] && [ "$g" -ge 50 ] || { echo "FATAL: /data at ${p}% used, ${g}G free: stop and ask the PI"; exit 1; }; }

""" + FETCH_SOPHON + """

          # init:checkpoint:K -- K is the CHECKPOINT's head width, checked before
          # any fine-tune so a wrong path or layout fails here, not after a run.
          INITS="__INITS__"
          for spec in ${INITS}; do
            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}; k=${rest#*:}
            [ "${name}" = "scratch" ] && continue
            [ -f "${ckpt}" ] || { echo "FATAL: ${name}: no ${ckpt}"; exit 1; }
            python3 experiments/EVAL/extract_features.py --checkpoint ${ckpt} --num-classes ${k} --arm ${name} \\
              --data-test /jc2/jet_data/Res2P_0250.parquet --out ${ROOT_OUT}/selfcheck --self-check-only
          done

          epochs_for () { case $1 in 10000) echo __E1__;; 100000) echo __E2__;; 1000000) echo __E3__;; *) echo "FATAL: no epoch budget for N=$1" >&2; exit 1;; esac; }
          COMMON="--use-amp --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1 --optimizer ranger"

          # ---------------------------------------------------------------- leg 1
          # R16_Q1 (and every other init) -> the 162-way vocabulary. Readout:
          # frozen 128-d features + 162 logits on the paper's 2M-jet test subset.
          TEST2M="__TEST2M__"
          for spec in ${INITS}; do
            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}
            for N in __SIZES__; do
              for S in __FT_SEEDS__; do
                OUT=${ROOT_OUT}/leg1/${name}/N${N}/s${S}
                [ -f ${OUT}/DONE ] && { echo "skip ${OUT} (DONE)"; continue; }
                space_ok
                [ -d ${OUT} ] && mv ${OUT} ${OUT}.partial.$(date -u +%s)
                mkdir -p ${OUT}
                if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} --exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; else LOAD=""; LR=__LR_SCRATCH__; fi
                EP=$(epochs_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=1 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB2}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassII_L162_noweight.yaml num_classes=162
                python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \\
                  --data-train ${SUB2}/train_N${N}_s${S}.parquet --data-val ${SUB2}/val.parquet \\
                  --data-config configs/finetune/JetClassII_L162_noweight.yaml \\
                  --network-config experiments/MTX/ParT_sophon_arch_mtx.py -o num_classes 162 -o fc_params '[(512,0.1)]' \\
                  ${COMMON} --start-lr ${LR} --samples-per-epoch ${N} --samples-per-epoch-val 200000 --num-epochs ${EP} \\
                  ${LOAD} --model-prefix ${OUT}/net --log ${OUT}/train.log 2>&1 | tee ${OUT}/stdout.log
                [ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log --log ${OUT}/stdout.log
                python3 experiments/EVAL/extract_features.py --checkpoint ${OUT}/net_best_epoch_state.pt --num-classes 162 --arm FT1_${name}_N${N}_s${S} \\
                  --data-config configs/data/JetClassII_base.yaml --data-test ${TEST2M} \\
                  --out ${OUT}/features_v2 --batch-size 512 --num-workers 1 --fetch-step 1 --max-jets 2000000 --save-logits
                python3 experiments/FT/smoke_checks.py features --dir ${OUT}/features_v2 --n 2000000 --k 162
                touch ${OUT}/DONE
              done
            done
          done

          # ---------------------------------------------------------------- leg 2
          # JetClass-I 10-class at the Sophon preprocessing: the pileup-shift leg.
          TEST1=""
          for C in __JC1_CLASSES__; do
            for f in $(ls /data/JetClass/Pythia/test_20M/${C}_*.root | sort | head -2); do TEST1="${TEST1} ${f}"; done
          done
          echo "leg-2 test subset: $(echo ${TEST1} | wc -w) files"
          [ "$(echo ${TEST1} | wc -w)" -eq 20 ] || { echo "FATAL: expected 20 JetClass-I test files"; exit 1; }
          PRED="--data-config configs/finetune/JetClassI_sophon_noweight.yaml --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 --predict-gpus 0 --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1"

          # The N_max scratch reference (E1 arm S, three seeds) on the SAME subset.
          for S in 1 2 3; do
            OUT=${ROOT_OUT}/leg2/ref_e1arms-s${S}
            [ -f ${OUT}/DONE ] && continue
            CK=/data/results/e1/arm_s_s${S}/net_best_epoch_state.pt
            [ -f "${CK}" ] || { echo "FATAL: no ${CK}"; exit 1; }
            mkdir -p ${OUT}
            weaver --predict --data-test ${TEST1} ${PRED} -o fc_params '[(512,0.1)]' --model-prefix ${CK} --predict-output ${OUT}/pred.root 2>&1 | tail -3
            touch ${OUT}/DONE
          done

          for spec in ${INITS}; do
            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}
            for N in __SIZES__; do
              for S in __FT_SEEDS__; do
                OUT=${ROOT_OUT}/leg2/${name}/N${N}/s${S}
                [ -f ${OUT}/DONE ] && { echo "skip ${OUT} (DONE)"; continue; }
                space_ok
                [ -d ${OUT} ] && mv ${OUT} ${OUT}.partial.$(date -u +%s)
                mkdir -p ${OUT}
                if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} --exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; else LOAD=""; LR=__LR_SCRATCH__; fi
                EP=$(epochs_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=2 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB1}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassI_sophon_noweight.yaml num_classes=10
                python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \\
                  --data-train ${SUB1}/train_N${N}_s${S}.parquet --data-val ${SUB1}/val.parquet \\
                  --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
                  --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
                  ${COMMON} --start-lr ${LR} --samples-per-epoch ${N} --samples-per-epoch-val 200000 --num-epochs ${EP} \\
                  ${LOAD} --model-prefix ${OUT}/net --log ${OUT}/train.log 2>&1 | tee ${OUT}/stdout.log
                [ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log --log ${OUT}/stdout.log
                weaver --predict --data-test ${TEST1} ${PRED} -o fc_params '[(512,0.1)]' --model-prefix ${OUT}/net --predict-output ${OUT}/pred.root 2>&1 | tail -3
                [ -f ${OUT}/pred.root ] || { echo "FATAL: no pred.root in ${OUT}"; exit 1; }
                touch ${OUT}/DONE
              done
            done
          done
          echo "FT LEGS COMPLETE"
"""


def job(name: str, script: str, *, gpu: bool, cpu: str, memory: str, shm: str,
        backoff: int, pin: str, header: str) -> str:
    gpu_req = ', nvidia.com/gpu: "1"' if gpu else ""
    gpu_env = ('        - name: GPU_PRODUCT\n          value: "NVIDIA-GeForce-RTX-3090"\n'
               if gpu else "")
    gpu_sched = ""
    if gpu:
        gpu_sched = (
            "      tolerations:\n"
            '      - { key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }\n')
    product = ("              - key: nvidia.com/gpu.product\n"
               "                operator: In\n"
               '                values: ["NVIDIA-GeForce-RTX-3090"]\n' if gpu else "")
    body = "\n".join("          " + ln if ln and not ln.startswith("          ") else ln
                     for ln in script.splitlines())
    return f"""apiVersion: batch/v1
kind: Job
metadata:
{header}  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: {backoff}
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {IMAGE}
        command: ["/bin/bash", "-c"]
        env:
{gpu_env}        - name: NODE_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: spec.nodeName }} }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        - name: REPO_REF
          value: "{pin}"
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{memory}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{memory}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
{gpu_sched}      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
{product}      volumes:
      - name: jc2
        persistentVolumeClaim:
          claimName: tn-pvc-base-jetclass2
          readOnly: true
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "{shm}" }}
"""


def _fill(script: str, pin: str) -> str:
    inits = " ".join(f"{n}:{c}:{k}" for n, c, k in INITS)
    return (script
            .replace("__TEST2M__", test2m_list())
            .replace("__INITS__", inits)
            .replace("__SIZES__", " ".join(str(s) for s in SIZES))
            .replace("__FT_SEEDS__", " ".join(str(s) for s in FT_SEEDS))
            .replace("__E1__", str(EPOCHS[10_000]))
            .replace("__E2__", str(EPOCHS[100_000]))
            .replace("__E3__", str(EPOCHS[1_000_000]))
            .replace("__LR_PRE__", LR_PRETRAINED)
            .replace("__LR_SCRATCH__", LR_SCRATCH)
            .replace("__LAMBDA__", LAMBDA)
            .replace("__JC1_CLASSES__", JC1_CLASSES)
            .replace("__SOPHON_SHA256__", SOPHON_SHA256))


def build(pin: str) -> dict[str, str]:
    h = "  # GENERATED by scripts/build_ft_jobs.py -- do not hand-edit. Regenerate.\n  #\n"
    specs = {
        "job-ft-subsets-jc2-raunav.yaml": job(
            "ft-subsets-jc2-raunav", _fill(SUBSETS_JC2, pin), gpu=False, cpu="4",
            memory="48Gi", shm="4Gi", backoff=1, pin=pin,
            header=h + "  # Nested fine-tuning subsets from JetClass-II (leg 1). CPU. ~1.5M-row\n"
                       "  # pool per seed held in memory, hence 48Gi.\n"),
        "job-ft-subsets-jc1-raunav.yaml": job(
            "ft-subsets-jc1-raunav", _fill(SUBSETS_JC1, pin), gpu=False, cpu="4",
            memory="32Gi", shm="4Gi", backoff=1, pin=pin,
            header=h + "  # Nested fine-tuning subsets from JetClass-I (leg 2). CPU.\n"),
        "job-ft-smoke-raunav.yaml": job(
            "ft-smoke-raunav", _fill(SMOKE, pin), gpu=False, cpu="4", memory="32Gi",
            shm="4Gi", backoff=0, pin=pin,
            header=h + "  # CPU SMOKE, run and read BEFORE any GPU job in this wave starts:\n"
                       "  # the hybrid mass loop on the image's weaver, both fine-tune paths\n"
                       "  # (load + head exclusion + readout), the subset writers, and the\n"
                       "  # released checkpoint, all on a few thousand jets. Prints SMOKE PASS.\n"),
        "job-ft-legs-raunav.yaml": job(
            "ft-legs-raunav", _fill(LEGS, pin), gpu=True, cpu="4", memory="64Gi",
            shm="8Gi", backoff=50, pin=pin,
            header=h + "  # THE FINE-TUNING LEGS (DECISIONS_PENDING item 14): every (init, N,\n"
                       "  # seed) of leg 1 then leg 2 in ONE pod on ONE GPU, sequential and\n"
                       "  # resumable per fine-tune. backoffLimit 50 for the same reason the\n"
                       "  # training jobs carry it: the queue reaps Pending pods.\n"
                       "  # Memory: extraction's measured need is fetch_step 1 / 1 worker\n"
                       "  # (experiments/EVAL/extract_features.py); training on a 1e6-jet file\n"
                       "  # holds one file per worker. 64Gi covers both with margin.\n"
                       "  # Storage: ~250 GB on /data over the wave; space_ok guards every\n"
                       "  # fine-tune at CLAUDE.md's 85% line.\n"),
    }
    for name, text in specs.items():
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav"), name
        args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        assert 'git clone --depth 1 --branch "${REPO_REF}"' in args, name
        for tok in ("__TEST2M__", "__INITS__", "__SIZES__", "__E1__", "__LAMBDA__", "__SOPHON_SHA256__"):
            assert tok not in args, f"{name}: {tok} unfilled"
        assert "--checkpoint models/" not in args, f"{name}: the released checkpoint is not in a clone"
        if name in ("job-ft-smoke-raunav.yaml", "job-ft-legs-raunav.yaml"):
            assert 'curl -fsSL -o "${SOPHON}"' in args and 'sha256sum "${SOPHON}"' in args, name
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    specs = build(args.pin)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in specs.items():
        if not args.check_only:
            (OUT_DIR / name).write_text(text)
        print(f"{name:34s} {'checked' if args.check_only else 'written'} (pin {args.pin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
