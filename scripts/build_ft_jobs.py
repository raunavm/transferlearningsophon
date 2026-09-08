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
epochs   50 / 30 / 10 at N = 1e4 / 1e5 / 1e6. weaver floors steps at
         samples_per_epoch // batch_size and drops the last partial batch, so
         the budget is 486,400 / 2,995,200 / 9,999,360 optimizer examples, not
         5e5 / 3e6 / 1e7; batch_size and steps_per_epoch go in each manifest.
         Validation stays at 200k every epoch even at N=1e4, where that is ~20x
         the training compute (~1 GPU-day over the wave): best-epoch selection
         is noisiest exactly at the smallest N, which is the headline cell.
         weaver
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
PIN = "mtx-s1.21"
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"
LAMBDA = "5.0"
# E0b's pinned sha256 of the released Sophon checkpoint (job-massreg-e0b-extract).
SOPHON_SHA256 = "cc7c33b522e796b5bbf0aa9bb5b01361c964f4ef3acebdd9682d7519c095b824"

JC1_CLASSES = "HToBB HToCC HToGG HToWW2Q1L HToWW4Q TTBar TTBarLep WToQQ ZToQQ ZJetsToNuNu"

# name : checkpoint : K of that checkpoint's head (0 = from scratch)
INITS = [
    ("r16q1-s2", "/data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt", 17),
    ("r16q1-s3", "/data/results/mtx/mtx-r16q1-s3/net_epoch-79_state.pt", 17),
    ("r16q1-s4", "/data/results/mtx/mtx-r16q1-s4/net_epoch-79_state.pt", 17),
    ("l162-s1b", "/data/results/mtx/mtx-l162-s1b/net_epoch-79_state.pt", 162),
    ("sophon-public", "/workspace/sophon_public.pt", 188),
    ("scratch", "", 0),
]
SIZES = [10_000, 100_000, 1_000_000]
EPOCHS = {10_000: 50, 100_000: 30, 1_000_000: 10}
FT_SEEDS = [1, 2, 3]
LR_PRETRAINED, LR_SCRATCH = "1e-4", "5e-4"

# The published top / q-g recipe (docs/PRD_PLAN.md 4.1 `[V G]`). BENCH_HEAD_MULT
# is what makes the trunk/head pair 1e-4 / 5e-3: weaver sets the matched
# parameters to `start_lr * mult`, so writing 50 here fixes BOTH numbers from
# LR_PRETRAINED and they cannot drift apart. NMAX_REPS is the benchmark's
# convention of a median over 9 head re-initialisations, top only, at N_max
# only, pretrained only.
BENCH_EPOCHS = 20
BENCH_HEAD_MULT = 50
BENCH_SETS = ["top", "qg"]
NMAX_REPS = list(range(1, 10))
assert float(LR_PRETRAINED) * BENCH_HEAD_MULT == 5e-3, (
    "the published head rate is 5e-3; trunk x mult must equal it")
assert len(NMAX_REPS) == 9


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
          # 61 train files per seed (9 Res2P / 39 Res34P / 13 QCD): choose_files
          # rounds each family separately, so --n-files 60 reads 61 (and 12 val
          # files reads 13). manifest.json records the realised n_files_used.
          # 30% of each file's SELECTED rows -> a ~1.5M-row pool per seed, of which
          # the nested 1e4 / 1e5 / 1e6 subsets are prefixes of one shuffle.
          python3 experiments/FT/make_subsets.py jc2 \\
            --train-files "${TRAIN_FILES[@]}" --val-files "${VAL_FILES[@]}" \\
            --out ${OUT} --sizes 10000 100000 1000000 --seeds 1 2 3 \\
            --n-files 60 --take-fraction 0.30 --val-size 200000 --n-val-files 12
          ls -la ${OUT}; du -sh ${OUT}
"""

SUBSETS_BENCH = PREAMBLE + """
          # The two published benchmarks (legs 3 and 4), staged by
          # scripts/stage_downstream.py.
          #
          # q/g reads qg_v2, NOT qg. The first staging gave every electron and
          # muon constituent the opposite electric charge -- sign(pdgid) is
          # right for 211/321/2212 but PDG 11 and 13 are the NEGATIVE leptons
          # -- and part_charge is an input feature of
          # configs/finetune/EnergyFlowQG.yaml, so that was a defect in the
          # training data. The wrong copy is left in place rather than
          # overwritten; see DECISIONS_PENDING.
          src_for () { case $1 in
            top) echo "/data/finetune/top";;
            qg)  echo "/data/finetune/qg_v2";;
            *) echo "FATAL: no source for $1" >&2; exit 1;; esac; }
          for D in top qg; do
            S=$(src_for ${D})
            [ -d "${S}" ] || { echo "FATAL: ${S} not staged"; exit 1; }
          done
""" + SPACE_GUARD + """
          # top ships train/val/test. qg ships 20 chunks of 100,000 with no split;
          # make_subsets splits them 16/2/2, which is the 1.6M/200k/200k that
          # ParticleNet (1902.08570) calls the recommended splitting.
          #
          # The shuffle inside make_subsets is load-bearing here, not hygiene:
          # top_train stores its rows in blocks of 10 sharing a label, so a
          # contiguous window's class fraction is sqrt(10) times noisier than an
          # iid draw (MEASURED 3.01x at N=1e3, 3.12x at N=1e4), and the first
          # 1,000 rows of the file are 46.0% top. Cutting the small-N points off
          # the front would bias the headline cell of the scaling curve, with
          # nothing erroring. tests/test_bench_subsets.py binds this.
          sizes_for () { case $1 in
            top) echo "1000 10000 100000 1200000";;
            qg)  echo "1000 10000 100000 1600000";;
            *) echo "FATAL: no grid for $1" >&2; exit 1;; esac; }
          for D in top qg; do
            # docs/PRD_PLAN.md 4.1: top {1e3, 1e4, 1e5, 1.2e6} (2606.14870's grid),
            # q/g {1e3, 1e4, 1e5, 1.6e6}. N_max is each benchmark's OWN training
            # split -- 1,211,000 for top and 16 x 100,000 for q/g -- so the two
            # rows drop into the community tables rather than a size we invented.
            # N=1e3 is in the grid because every N-sweep paper reaches it and the
            # vocabulary effect is predicted largest there; it is also the cell the
            # block structure would have damaged most (SD 0.048 unshuffled).
            S=$(src_for ${D}); O=${S}_sub
            python3 experiments/FT/make_subsets.py bench --dataset ${D} \\
              --src ${S} --out ${O} \\
              --sizes $(sizes_for ${D}) --seeds 1 2 3 --val-size 200000
            ls -la ${O}; du -sh ${O}
          done
"""


# ---------------------------------------------------------------- legs 3 and 4
# THE TWO PUBLISHED BENCHMARKS. These are the rows that drop into the community
# tables, so the recipe is the benchmark's, not ours (docs/PRD_PLAN.md 4.1,
# `[V G]`): 20 epochs, trunk 1e-4 with the head at 50x via weaver's lr_mult,
# CONSTANT lr (--lr-scheduler none, not weaver's flat+decay default), weight
# decay 0.01, and median + spread over head re-initialisations.
#
# The 50x is not a free parameter: 1e-4 x 50 = 5e-3 is exactly the published
# 1e-4 trunk / 5e-3 head. weaver applies `args.start_lr * mult_factor` to every
# parameter matching the pattern, so the ratio is set once and cannot drift
# between the two numbers. Scratch takes a single 5e-4 and NO lr_mult -- there
# is no pretrained trunk to hold back, and 5e-4 is the scratch rate, not 5e-3.
#
# q/g reads qg_v2_sub. The first staging inverted the electric charge of every
# electron and muon constituent (PDG 11 and 13 are the NEGATIVE leptons) and
# part_charge is an input feature here.
LEGS_BENCH = PREAMBLE + """
          ROOT_OUT=/data/results/ft
          mkdir -p ${ROOT_OUT}
          src_for () { case $1 in
            top) echo "/data/finetune/top";;
            qg)  echo "/data/finetune/qg_v2";;
            *) echo "FATAL: no source for $1" >&2; exit 1;; esac; }
          cfg_for () { case $1 in
            top) echo "configs/finetune/TopReference.yaml";;
            qg)  echo "configs/finetune/EnergyFlowQG.yaml";;
            *) echo "FATAL: no config for $1" >&2; exit 1;; esac; }
          sizes_for () { case $1 in
            top) echo "1000 10000 100000 1200000";;
            qg)  echo "1000 10000 100000 1600000";;
            *) echo "FATAL: no grid for $1" >&2; exit 1;; esac; }
          nmax_for () { case $1 in
            top) echo 1200000;; qg) echo 1600000;;
            *) echo "FATAL: no N_max for $1" >&2; exit 1;; esac; }
          test_for () { case $1 in
            top) echo "$(src_for top)/top_test.parquet";;
            qg)  echo "$(src_for qg)/qg_chunk18.parquet $(src_for qg)/qg_chunk19.parquet";;
            *) echo "FATAL: no test files for $1" >&2; exit 1;; esac; }

          for D in __BENCH_SETS__; do
            S=$(src_for ${D}); O=${S}_sub
            [ -f ${O}/DONE ] || { echo "FATAL: ${O}/DONE absent -- run ft-subsets-bench first"; exit 1; }
            for f in $(test_for ${D}); do
              [ -f "${f}" ] || { echo "FATAL: ${D}: no test file ${f}"; exit 1; }
            done
            [ -f "$(cfg_for ${D})" ] || { echo "FATAL: ${D}: no $(cfg_for ${D})"; exit 1; }
          done
""" + SPACE_GUARD + """
          space_ok () { local p=$(df --output=pcent /data | tail -1 | tr -dc 0-9); local g=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); echo "/data ${p}% used, ${g}G free"; [ "$p" -lt 85 ] && [ "$g" -ge 50 ] || { echo "FATAL: /data at ${p}% used, ${g}G free: stop and ask the PI"; exit 1; }; }
          FAIL_MARK=${ROOT_OUT}/FAILED_BENCH
          [ -f ${FAIL_MARK} ] && { echo "FATAL: an earlier attempt failed twice: $(cat ${FAIL_MARK}). Fix it, then remove ${FAIL_MARK}"; exit 1; }
          attempt_ok () { local o=$1; local n=$(ls -d ${o}.partial.* 2>/dev/null | wc -l); [ "$n" -ge 2 ] && { echo "${o} failed ${n} times" | tee ${FAIL_MARK}; exit 1; }; return 0; }

""" + FETCH_SOPHON + """
          INITS="__INITS__"
          for spec in ${INITS}; do
            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}; k=${rest#*:}
            [ "${name}" = "scratch" ] && continue
            [ -f "${ckpt}" ] || { echo "FATAL: ${name}: no ${ckpt}"; exit 1; }
          done

          # The benchmark recipe, set in ONE place. --lr-scheduler none is the
          # constant LR the published recipe specifies; weaver's default is
          # flat+decay, which would silently anneal and make the row not
          # comparable to the community table.
          BENCH="--use-amp --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1 \
            --optimizer ranger --lr-scheduler none --num-epochs __BENCH_EPOCHS__ \
            --optimizer-option weight_decay 0.01"
          # An ARRAY, not a string. The value carries parentheses and single
          # quotes; as a plain string it is re-split on expansion and the
          # parens reach the shell as syntax.
          HEAD_MULT=(--optimizer-option lr_mult "(r'mod\\.fc\\..*', __HEAD_MULT__)")

          for D in __BENCH_SETS__; do
            SUB=$(src_for ${D})_sub; CFG=$(cfg_for ${D}); TEST=$(test_for ${D})
            NMAX=$(nmax_for ${D})
            for spec in ${INITS}; do
              name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}
              for N in $(sizes_for ${D}); do
                # 9 head re-inits at N_max, top only, pretrained arms only --
                # the benchmark's convention for the headline cell
                # (docs/PRD_PLAN.md 4.1). Everywhere else the three
                # fine-tuning seeds are the spread.
                REPS="__FT_SEEDS__"
                if [ "${D}" = "top" ] && [ "${N}" = "${NMAX}" ] && [ -n "${ckpt}" ]; then
                  REPS="__NMAX_REPS__"
                fi
                for S in ${REPS}; do
                  # S is the TRAINING seed (head init, data order, dropout).
                  # DSEED indexes the data subset and is a different thing.
                  # Tying them together made "9 head re-initialisations" demand
                  # 9 training subsets, but make_subsets writes 3 -- so reps
                  # 4..9 read train_N..._s{4..9}.parquet, which do not exist,
                  # and weaver dies in a worker with a message naming no file.
                  # Holding the data fixed is also what the published
                  # convention MEANS (docs/PRD_PLAN.md 4.1): the spread is over
                  # re-initialisations, not over resampled training sets.
                  DSEED=${S}
                  [ "${REPS}" = "__NMAX_REPS__" ] && DSEED=1
                  OUT=${ROOT_OUT}/leg_${D}/${name}/N${N}/s${S}
                  [ -f ${OUT}/DONE ] && { echo "skip ${OUT} (DONE)"; continue; }
                  space_ok
                  attempt_ok ${OUT}
                  [ -d ${OUT} ] && mv ${OUT} ${OUT}.partial.$(date -u +%s)
                  mkdir -p ${OUT}
                  if [ -n "${ckpt}" ]; then
                    LOAD=(--load-model-weights ${ckpt} --exclude-model-weights "mod\\.fc\\..*")
                    LR=__LR_PRE__; MULT=("${HEAD_MULT[@]}")
                  else
                    LOAD=(); LR=__LR_SCRATCH__; MULT=()
                  fi
                  python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json \
                    leg=${D} init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} \
                    lr=${LR} head_lr_mult=__HEAD_MULT__ epochs=__BENCH_EPOCHS__ lr_schedule=constant \
                    weight_decay=0.01 subset=${SUB}/train_N${N}_s${DSEED}.parquet \
                    data_config=${CFG} num_classes=2 batch_size=512 steps_per_epoch=$((N/512))
                  python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \
                    --data-train ${SUB}/train_N${N}_s${DSEED}.parquet --data-val ${SUB}/val.parquet \
                    --data-config ${CFG} \
                    --network-config experiments/MTX/ParT_sophon_arch_mtx.py -o num_classes 2 -o fc_params '[(512,0.1)]' \
                    ${BENCH} ${MULT[@]+"${MULT[@]}"} --start-lr ${LR} --samples-per-epoch ${N} --samples-per-epoch-val 200000 \
                    ${LOAD[@]+"${LOAD[@]}"} --model-prefix ${OUT}/net --log ${OUT}/train.log 2>&1 | tee ${OUT}/stdout.log
                  [ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log --log ${OUT}/stdout.log
                  # The 50x must appear in weaver's own log, or the row is not
                  # the published recipe and nothing else would say so.
                  if [ -n "${ckpt}" ]; then
                    grep -q "Parameters with lr multiplied by __HEAD_MULT__" ${OUT}/stdout.log || {
                      echo "FATAL: weaver did not apply the head lr multiplier; the trunk/head"
                      echo "       ratio is not the published 1e-4/5e-3."; exit 1; }
                  fi
                  python3 experiments/EVAL/extract_features.py --checkpoint ${OUT}/net_best_epoch_state.pt \
                    --num-classes 2 --arm FT_${D}_${name}_N${N}_s${S} \
                    --data-config ${CFG} --data-test ${TEST} --observers jet_pt jet_energy \
                    --out ${OUT}/features --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
                  touch ${OUT}/DONE
                done
              done
            done
          done
          echo "FT BENCH LEGS COMPLETE"
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
          CKPT=/data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt
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
          # weaver writes net_best_epoch_state.pt only when validation accuracy
          # beats its initial best of 0 (train.py:820,847); two tiny epochs at 162
          # classes can stay at exactly 0.0 (attempt 1 did). Fall back to the last
          # epoch here only: the production legs train 10-50 epochs and use the
          # best-epoch file unconditionally. Step [1/5] exercises the best-epoch path.
          BEST1=${S}/leg1/net_best_epoch_state.pt
          [ -f ${BEST1} ] || { echo "no best-epoch file (validation accuracy stayed at 0); using the last epoch"; BEST1=$(ls ${S}/leg1/net_epoch-*_state.pt | sort -V | tail -1); }
          python3 experiments/FT/smoke_checks.py head-width --checkpoint ${BEST1} --expect 162
          python3 experiments/EVAL/extract_features.py --checkpoint ${BEST1} --num-classes 162 --arm FT_SMOKE \\
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
          BEST2=${S}/leg2/net_best_epoch_state.pt
          [ -f ${BEST2} ] || { echo "no best-epoch file (validation accuracy stayed at 0); using the last epoch"; BEST2=$(ls ${S}/leg2/net_epoch-*_state.pt | sort -V | tail -1); }
          T1=$(ls /data/JetClass/Pythia/test_20M/HToBB_*.root | sort | head -1)
          weaver --predict --gpus "" --data-test ${S}/sub1/val.parquet ${T1} \\
            --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
            --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
            --model-prefix ${BEST2} --predict-output ${S}/leg2/pred.root \\
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

          # A fine-tune that fails deterministically (OOM, a failed check) would
          # otherwise be re-run by all 50 retries, each one moving the previous
          # attempt aside: 50 x hours of a held GPU and ~200 GB of .partial dirs.
          # Two attempts, then the marker stops every later pod at the top.
          FAIL_MARK=${ROOT_OUT}/FAILED
          [ -f ${FAIL_MARK} ] && { echo "FATAL: an earlier attempt failed twice: $(cat ${FAIL_MARK}). Fix it, then remove ${FAIL_MARK}"; exit 1; }
          attempt_ok () { local o=$1; local n=$(ls -d ${o}.partial.* 2>/dev/null | wc -l); [ "$n" -ge 2 ] && { echo "${o} failed ${n} times" | tee ${FAIL_MARK}; exit 1; }; return 0; }

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

          # Leg-2 preconditions, checked HERE rather than days later after leg 1.
          for S in 1 2 3; do
            [ -f /data/results/e1/arm_s_s${S}/net_best_epoch_state.pt ] || { echo "FATAL: no E1 arm S seed ${S} checkpoint"; exit 1; }
          done
          for C in __JC1_CLASSES__; do
            [ "$(find /data/JetClass/Pythia/test_20M -maxdepth 1 -name "${C}_*.root" | wc -l)" -ge 2 ] || { echo "FATAL: ${C}: fewer than 2 JetClass-I test files"; exit 1; }
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
                attempt_ok ${OUT}
                [ -d ${OUT} ] && mv ${OUT} ${OUT}.partial.$(date -u +%s)
                mkdir -p ${OUT}
                if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} --exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; else LOAD=""; LR=__LR_SCRATCH__; fi
                EP=$(epochs_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=1 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB2}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassII_L162_noweight.yaml num_classes=162 batch_size=512 steps_per_epoch=$((N/512))
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
            weaver --predict --data-test ${TEST1} ${PRED} -o fc_params '[(512,0.1)]' --model-prefix ${CK} --predict-output ${OUT}/pred.root 2>&1 | tee ${OUT}/predict.log | tail -3
            # weaver's save_root catches its own write errors and still exits 0.
            [ -f ${OUT}/pred.root ] || { echo "FATAL: no pred.root in ${OUT}"; exit 1; }
            touch ${OUT}/DONE
          done

          for spec in ${INITS}; do
            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}
            for N in __SIZES__; do
              for S in __FT_SEEDS__; do
                OUT=${ROOT_OUT}/leg2/${name}/N${N}/s${S}
                [ -f ${OUT}/DONE ] && { echo "skip ${OUT} (DONE)"; continue; }
                space_ok
                attempt_ok ${OUT}
                [ -d ${OUT} ] && mv ${OUT} ${OUT}.partial.$(date -u +%s)
                mkdir -p ${OUT}
                if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} --exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; else LOAD=""; LR=__LR_SCRATCH__; fi
                EP=$(epochs_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=2 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB1}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassI_sophon_noweight.yaml num_classes=10 batch_size=512 steps_per_epoch=$((N/512))
                python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \\
                  --data-train ${SUB1}/train_N${N}_s${S}.parquet --data-val ${SUB1}/val.parquet \\
                  --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
                  --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
                  ${COMMON} --start-lr ${LR} --samples-per-epoch ${N} --samples-per-epoch-val 200000 --num-epochs ${EP} \\
                  ${LOAD} --model-prefix ${OUT}/net --log ${OUT}/train.log 2>&1 | tee ${OUT}/stdout.log
                [ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log --log ${OUT}/stdout.log
                weaver --predict --data-test ${TEST1} ${PRED} -o fc_params '[(512,0.1)]' --model-prefix ${OUT}/net --predict-output ${OUT}/pred.root 2>&1 | tee ${OUT}/predict.log | tail -3
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
            .replace("__BENCH_EPOCHS__", str(BENCH_EPOCHS))
            .replace("__HEAD_MULT__", str(BENCH_HEAD_MULT))
            .replace("__BENCH_SETS__", " ".join(BENCH_SETS))
            .replace("__NMAX_REPS__", " ".join(str(r) for r in NMAX_REPS))
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
        "job-ft-subsets-bench-raunav.yaml": job(
            "ft-subsets-bench-raunav", _fill(SUBSETS_BENCH, pin), gpu=False, cpu="4",
            memory="64Gi", shm="4Gi", backoff=1, pin=pin,
            header=h + "  # Nested subsets for the two published benchmarks, top and q/g\n"
                       "  # (legs 3 and 4). CPU. q/g N_max is its whole 1.6M-jet training\n"
                       "  # split, held in memory and then COPIED by the shuffle, so the\n"
                       "  # peak is roughly twice the pool: 64Gi, not the 48Gi that sizes\n"
                       "  # the 1.2M-row JetClass-II job.\n"),
        "job-ft-legs-bench-raunav.yaml": job(
            "ft-legs-bench-raunav", _fill(LEGS_BENCH, pin), gpu=True, cpu="4",
            memory="88Gi", shm="8Gi", backoff=50, pin=pin,
            header=h + "  # LEGS 3 AND 4 -- the two PUBLISHED benchmarks (top tagging and\n"
                       "  # EnergyFlow quark/gluon). The recipe is the benchmark's, not ours\n"
                       "  # (docs/PRD_PLAN.md 4.1): 20 epochs, trunk 1e-4 with the head at\n"
                       "  # 50x = 5e-3, CONSTANT lr, weight decay 0.01. weaver's default\n"
                       "  # scheduler is flat+decay, which would anneal and quietly make the\n"
                       "  # row incomparable to the community table, so --lr-scheduler none\n"
                       "  # is explicit. Top additionally takes 9 head re-inits at N_max.\n"),
        "job-ft-smoke-raunav.yaml": job(
            "ft-smoke-raunav", _fill(SMOKE, pin), gpu=False, cpu="4", memory="32Gi",
            shm="4Gi", backoff=0, pin=pin,
            header=h + "  # CPU SMOKE, run and read BEFORE any GPU job in this wave starts:\n"
                       "  # the hybrid mass loop on the image's weaver, both fine-tune paths\n"
                       "  # (load + head exclusion + readout), the subset writers, and the\n"
                       "  # released checkpoint, all on a few thousand jets. Prints SMOKE PASS.\n"),
        "job-ft-legs-raunav.yaml": job(
            "ft-legs-raunav", _fill(LEGS, pin), gpu=True, cpu="4", memory="88Gi",
            shm="8Gi", backoff=50, pin=pin,
            header=h + "  # THE FINE-TUNING LEGS (DECISIONS_PENDING item 14): every (init, N,\n"
                       "  # seed) of leg 1 then leg 2 in ONE pod on ONE GPU, sequential and\n"
                       "  # resumable per fine-tune. backoffLimit 50 for the same reason the\n"
                       "  # training jobs carry it: the queue reaps Pending pods.\n"
                       "  # Memory 88Gi, as the MTX training arms: at N=1e6 the subset is\n"
                       "  # ONE file, so weaver caps the train loader at one worker and its\n"
                       "  # async prefetch holds a second copy of the same 1e6 jets. CLAUDE.md\n"
                       "  # section 8's measured band is 35-58 GB for ~1.6M jets in flight and\n"
                       "  # records 64Gi OOM-killing g1-r16q1-lr5e4; this is ~2.2M in flight.\n"
                       "  # 31 of 37 us-west 3090 nodes allocate >= 88Gi (measured 2026-09-05).\n"
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
    # A spec on disk can carry edits this generator does not know about: the node
    # exclusions scripts/exclude_node.py appends, a raised attempt_ok threshold, a
    # provenance note, a pin moved past the generator's default. Regenerating the
    # whole set silently reverts them -- it did, to job-ft-legs on 2026-09-07,
    # dropping REPO_REF from mtx-s1.11 to mtx-s1.10 while that job was running.
    # Name what to write when only one spec is meant to change.
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="write only these spec files (substring match on the name)")
    ap.add_argument("--repin", action="store_true",
                    help="allow moving REPO_REF on specs whose jobs may have run")
    args = ap.parse_args()
    specs = build(args.pin)
    if args.only:
        keep = {n: t for n, t in specs.items() if any(k in n for k in args.only)}
        missing = [k for k in args.only if not any(k in n for n in specs)]
        if missing:
            sys.exit(f"FATAL: --only matched nothing for {missing}; have {sorted(specs)}")
        specs = keep
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # REPIN GUARD. A bare run of this script regenerates EVERY spec, including
    # those of jobs that have already run or are running -- silently moving
    # their REPO_REF to the current PIN, reverting hand-edits applied after
    # generation, and (for the legs) swapping the checkpoints under 20 results
    # already on disk. The spec is part of the provenance record
    # (docs/RECORD.md), so rewriting it detaches a completed run from the code
    # that produced it. Regenerating one spec is what --only is for.
    if not args.check_only:
        blocked = []
        for name, text in specs.items():
            p = OUT_DIR / name
            if not p.exists():
                continue
            old = re.search(r'REPO_REF\n\s+value: "([^"]+)"', p.read_text())
            new = re.search(r'REPO_REF\n\s+value: "([^"]+)"', text)
            if old and new and old.group(1) != new.group(1):
                blocked.append(f"  {name}: {old.group(1)} -> {new.group(1)}")
        if blocked and not args.repin:
            sys.exit("FATAL: this would move the pin on specs whose jobs may "
                     "already have run:\n" + "\n".join(blocked) +
                     "\nRegenerate just the spec you mean with --only <name>, "
                     "or pass --repin if you really mean to move all of them.")
    for name, text in specs.items():
        if not args.check_only:
            (OUT_DIR / name).write_text(text)
        print(f"{name:34s} {'checked' if args.check_only else 'written'} (pin {args.pin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
