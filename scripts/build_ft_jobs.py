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

# name : checkpoint : K of that checkpoint's head (0 = no head to check: from
# scratch, or the self-supervised init) : the fine-tuning seeds THIS init runs.
#
# THE SEEDS TRAVEL WITH THE INIT since 2026-09-18. The approved design makes the
# PRETRAINING seed the unit of replication -- one fine-tuning seed from every
# pretrained checkpoint -- and keeps three fine-tuning seeds only where the
# fine-tuning spread is itself the measurement. These six are that sub-study
# (waves 1 and 2): their seeds ARE FT_SEEDS, asserted below, and waves 1 and 2
# still emit the single global `for S in 1 2 3`, so their specs do not move by
# a byte. The per-init seeds are read by wave 3 and the v2 benchmarks only.
INITS = [
    ("r16q1-s2", "/data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt", 17, [1, 2, 3]),
    ("r16q1-s3", "/data/results/mtx/mtx-r16q1-s3/net_epoch-79_state.pt", 17, [1, 2, 3]),
    ("r16q1-s4", "/data/results/mtx/mtx-r16q1-s4/net_epoch-79_state.pt", 17, [1, 2, 3]),
    ("l162-s1b", "/data/results/mtx/mtx-l162-s1b/net_epoch-79_state.pt", 162, [1, 2, 3]),
    ("sophon-public", "/workspace/sophon_public.pt", 188, [1, 2, 3]),
    ("scratch", "", 0, [1, 2, 3]),
]
# N = 1e3 ADDED 2026-09-12 (DECISIONS_PENDING item 25, option B). docs/PRD_PLAN
# 4.1 asks for it because every N-sweep paper reaches it and the vocabulary
# effect is predicted to be LARGEST there; top and q/g already run it.
SIZES = [1_000, 10_000, 100_000, 1_000_000]
EPOCHS = {1_000: 50, 10_000: 50, 100_000: 30, 1_000_000: 10}

# OPTIMIZER STEPS, HELD FIXED -- the whole of option B.
#
# `--samples-per-epoch` is what weaver turns into steps_per_epoch (train.py:1006,
# samples // batch_size). At N=1e3 the naive choice `--samples-per-epoch 1000`
# with 50 epochs gives 50 * (1000/512) ~ 100 optimizer steps, an ORDER OF
# MAGNITUDE fewer than the ~1,000 the N=1e4 cell gets. That cell would measure
# under-training, not the vocabulary effect, in the one place 4.1 predicts the
# effect is largest -- and it would read as "the coarse arm collapses at 1e3".
#
# So the TRAINING SET stays 1,000 jets -- that is the controlled variable -- and
# an "epoch" becomes ten passes over it. Same compute as the N=1e4 cell, same 50
# validation passes, and the best-validation-epoch checkpoint rule applied at
# identical granularity, which is what makes the two comparable at all.
#
# WEAVER SUPPORTS THIS AND IT WAS VERIFIED, NOT ASSUMED (2026-09-12). Setting
# --samples-per-epoch puts the loader in infinity_mode (train.py:286), and
# _SimpleIter._try_get_next calls restart() on exhaustion rather than stopping
# (dataset.py:309-312). restart() RE-SHUFFLES the file list (dataset.py:183-184)
# and _preprocess re-shuffles rows on every load, so the ten cycles are ten
# independently shuffled passes, NOT a replay of one ordering. That mattered:
# a replay would make the 1e3 cell's optimizer trajectory incomparable to every
# other cell's. Checked against the installed weaver 0.4.17 source, the same
# version the image carries.
SAMPLES_PER_EPOCH = {1_000: 10_000}   # else: N itself
FT_SEEDS = [1, 2, 3]
assert all(seeds == FT_SEEDS for *_, seeds in INITS), (
    "waves 1 and 2 loop over the global FT_SEEDS; an init whose own seeds differ "
    "would be recorded as running seeds it does not run")
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
          # NO BARE DONE SHORT-CIRCUIT. make_subsets.py compares the stored
          # manifest against the request being made -- sizes, seeds, n_files,
          # take_fraction -- and either refuses loudly or grows the grid. A
          # `[ -f DONE ] && exit 0` in front of that MASKS all of it: when item
          # 25 added N=1e3 to SIZES this job re-ran, printed "already built" and
          # exited 0 while the N=1e3 subsets were never written, which the legs
          # only discover days later as a missing parquet.
          [ -f ${OUT}/manifest.json ] && head -40 ${OUT}/manifest.json
""" + SPLIT_GUARD + SPACE_GUARD + """
          # 61 train files per seed (9 Res2P / 39 Res34P / 13 QCD): choose_files
          # rounds each family separately, so --n-files 60 reads 61 (and 12 val
          # files reads 13). manifest.json records the realised n_files_used.
          # 30% of each file's SELECTED rows -> a ~1.5M-row pool per seed, of which
          # the nested 1e4 / 1e5 / 1e6 subsets are prefixes of one shuffle.
          python3 experiments/FT/make_subsets.py jc2 \\
            --train-files "${TRAIN_FILES[@]}" --val-files "${VAL_FILES[@]}" \\
            --out ${OUT} --sizes __SIZES__ --seeds 1 2 3 \\
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
          # NO BARE DONE SHORT-CIRCUIT. make_subsets.py compares the stored
          # manifest against the request being made -- sizes, seeds, n_files,
          # take_fraction -- and either refuses loudly or grows the grid. A
          # `[ -f DONE ] && exit 0` in front of that MASKS all of it: when item
          # 25 added N=1e3 to SIZES this job re-ran, printed "already built" and
          # exited 0 while the N=1e3 subsets were never written, which the legs
          # only discover days later as a missing parquet.
          [ -f ${OUT}/manifest.json ] && head -40 ${OUT}/manifest.json
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
            --out ${OUT} --sizes __SIZES__ --seeds 1 2 3 \\
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

          epochs_for () { case $1 in 1000) echo __E0__;; 10000) echo __E1__;; 100000) echo __E2__;; 1000000) echo __E3__;; *) echo "FATAL: no epoch budget for N=$1" >&2; exit 1;; esac; }
          # samples-per-epoch DECOUPLED from the subset size; see SAMPLES_PER_EPOCH
          # in scripts/build_ft_jobs.py. Only N=1e3 differs, and it differs so the
          # optimizer-step count matches the N=1e4 cell.
          samples_for () { case $1 in 1000) echo __S0__;; *) echo $1;; esac; }
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
                SPE=$(samples_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=1 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB2}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassII_L162_noweight.yaml num_classes=162 batch_size=512 steps_per_epoch=$((N/512))
                python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \\
                  --data-train ${SUB2}/train_N${N}_s${S}.parquet --data-val ${SUB2}/val.parquet \\
                  --data-config configs/finetune/JetClassII_L162_noweight.yaml \\
                  --network-config experiments/MTX/ParT_sophon_arch_mtx.py -o num_classes 162 -o fc_params '[(512,0.1)]' \\
                  ${COMMON} --start-lr ${LR} --samples-per-epoch ${SPE} --samples-per-epoch-val 200000 --num-epochs ${EP} \\
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
                SPE=$(samples_for ${N})
                python3 experiments/FT/smoke_checks.py manifest --out ${OUT}/ft_manifest.json leg=2 init=${name} checkpoint=${ckpt} n_train=${N} ft_seed=${S} lr=${LR} epochs=${EP} subset=${SUB1}/train_N${N}_s${S}.parquet data_config=configs/finetune/JetClassI_sophon_noweight.yaml num_classes=10 batch_size=512 steps_per_epoch=$((N/512))
                python3 experiments/E1/seed_weaver.py --seed ${S} --lean-val-metrics \\
                  --data-train ${SUB1}/train_N${N}_s${S}.parquet --data-val ${SUB1}/val.parquet \\
                  --data-config configs/finetune/JetClassI_sophon_noweight.yaml \\
                  --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes 10 -o fc_params '[(512,0.1)]' \\
                  ${COMMON} --start-lr ${LR} --samples-per-epoch ${SPE} --samples-per-epoch-val 200000 --num-epochs ${EP} \\
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


# Nodes measured to accept a pod and then fail it. Kept here rather than in each
# spec so one list serves every GPU job this generator emits.
#   ry-gpu-03, k8s-chase-ci-07, nrp-fiona-001, hcc-chase-shor-*: device plugin
#     rejects at admission (scripts/exclude_node.py).
#   nautilus-ext-gpu01: CSI volume attachment is forbidden for that node
#     ("no relationship found between node ... and this object"), so every pod
#     fails to mount /data, is evicted, and reschedules straight back onto it.
#     Measured 2026-09-16: it took three wave-2 pods in ~90 min and put two
#     .partial dirs on one cell, one short of halting the whole wave.
BAD_NODES = ("ry-gpu-03.sdsc.optiputer.net", "nautilus-ext-gpu01.fullerton.edu",
             "hcc-chase-shor-c4705.unl.edu", "hcc-chase-shor-c4709.unl.edu",
             "k8s-chase-ci-07.calit2.optiputer.net", "nrp-fiona-001.sdmz.amnh.org")


def job(name: str, script: str, *, gpu: bool, cpu: str, memory: str, shm: str,
        backoff: int, pin: str, header: str, exclude_hosts: tuple = ()) -> str:
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
    # NODES THAT ADVERTISE FREE GPUs AND THEN REJECT THE POD. Kubernetes tells
    # the scheduler nothing about a broken node until an admin taints it, so the
    # spec has to say so -- the reason scripts/exclude_node.py exists (38 pods
    # lost to ry-gpu-03 in one wave). Opt-in and empty by default, so adding it
    # here changes no other spec's bytes.
    if gpu and exclude_hosts:
        product += ("              - key: kubernetes.io/hostname\n"
                    "                operator: NotIn\n"
                    "                values: [" +
                    ", ".join(f'"{h}"' for h in exclude_hosts) + "]\n")
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


# WAVE 2 REBUILDS THE SUBSETS UNDER NEW JOB NAMES, NOT BY REGENERATING WAVE 1'S.
#
# Item 33 leaves this open and names the precedent: WINDOW_PIN in
# scripts/build_extract_jobs.py, where a second pass over the same data was
# emitted as its own spec rather than by repinning the first. The reason is that
# job-ft-subsets-jc2-raunav ALREADY RAN, at mtx-s1.10, and its spec is part of
# the provenance record for the wave-1 subsets now on disk (docs/RECORD.md).
# Moving its pin detaches a completed run from the code that produced it, and
# `kubectl apply` on an existing completed Job does not re-run it anyway.
#
# THE BODY IS IDENTICAL TO WAVE 1'S AND THAT IS THE WHOLE DESIGN. SIZES now reads
# [1_000, 10_000, 100_000, 1_000_000], and make_subsets.py grows the grid
# DOWNWARD: it writes train_N1000_s*.parquet and leaves every existing subset
# byte-identical, because no builder consumes RNG as a function of `sizes`.
# So wave 2 needs a new NAME and a new PIN, not new logic -- and the subsets are
# DATA, shared by both waves, so they are grown in place at /data/finetune/jc2
# rather than copied to a w2 tree. Only the legs' RESULTS need their own root.
W2_SUBSETS = {
    "job-ft-subsets-jc2-w2-raunav.yaml": (
        "ft-subsets-jc2-w2-raunav", "SUBSETS_JC2", "48Gi",
        "JetClass-II (leg 1)"),
    "job-ft-subsets-jc1-w2-raunav.yaml": (
        "ft-subsets-jc1-w2-raunav", "SUBSETS_JC1", "32Gi",
        "JetClass-I (leg 2)"),
}


def legs_w2() -> str:
    """Wave 2 of the legs: LEGS at ParT's published recipe, in its own tree.

    DERIVED FROM LEGS BY ASSERTED SUBSTITUTION, not copied. A 200-line duplicate
    would drift -- wave 1 gains a guard, wave 2 does not, and nothing says so.
    Every replacement below states how many times it must match and raises if it
    does not, so a change to LEGS breaks this loudly instead of silently
    producing a wave-2 spec that is missing the recipe it exists to apply.

    WHAT CHANGES, AND WHY EACH ONE (item 32(b), item 33):

      output root   /data/results/ft -> /data/results/ft/w2b. Wave 1's 108
                    cells are complete and keyed by the SAME init names and
                    sizes, so sharing a root means every wave-2 cell hits
                    `[ -f DONE ]` and skips. The two waves are reported side by
                    side.

                    w2b, NOT w2, AND THE `b` IS LOad-BEARING (item 36). The
                    first wave-2 launch completed 4 cells under the old
                    protocol -- 200k validation, full 2e6 feature rows -- before
                    it was stopped for running the volume out of space. Those
                    four carry DONE markers, so relaunching into w2/ would SKIP
                    them and ship a table whose first 4 cells were selected on a
                    different validation set and cached a different row set from
                    the other 140. That is an I1 break, and it would not even
                    reach the table: leg1_metrics.py compares label188_sha256
                    across cells and a mismatch is fatal, so the analysis would
                    refuse the merge after the wave had been paid for. A new
                    root re-runs all 144 under one protocol and OVERWRITES
                    NOTHING -- the superseded cells stay under w2/ as the record
                    of what was stopped and why.
      weight decay  0.01 on every arm INCLUDING scratch. Item 33 option B was
                    declined precisely because a wave-2 table without scratch
                    has no internal reference row; scratch must therefore get
                    the same optimiser change as everything else.
      head rate     50x the trunk via weaver's lr_mult, pretrained arms ONLY.
                    1e-4 x 50 = 5e-3 is exactly ParT's published 1e-4 trunk /
                    5e-3 head. Scratch keeps its single 5e-4 and NO multiplier:
                    there is no pretrained trunk to hold back and 5e-4 is the
                    scratch rate, not 5e-3.

    WHAT DELIBERATELY DOES NOT CHANGE: the LR scheduler. LEGS_BENCH passes
    `--lr-scheduler none` because the published BENCHMARK recipe specifies a
    constant rate, but item 32(b) adopts only `lr_mult 50` and
    `weight_decay 0.01`, and the point of doing so is to put the legs on the
    SAME protocol as our own LR sweep -- which ran weaver's default flat+decay.
    Adding the constant schedule here would re-open the protocol gap in a new
    place while appearing to close it.
    """
    subs = [
        # (old, new, expected occurrences)
        ("          ROOT_OUT=/data/results/ft\n",
         "          ROOT_OUT=/data/results/ft/w2b\n", 1),
        ('          COMMON="--use-amp --batch-size 512 --num-workers 2 '
         '--fetch-by-files --fetch-step 1 --optimizer ranger"',
         '          COMMON="--use-amp --batch-size 512 --num-workers 2 '
         '--fetch-by-files --fetch-step 1 --optimizer ranger '
         '--optimizer-option weight_decay 0.01"\n'
         '          # An ARRAY, not a string: the value carries parentheses and\n'
         '          # single quotes, and as a plain string it is re-split on\n'
         '          # expansion so the parens reach the shell as syntax.\n'
         '          HEAD_MULT=(--optimizer-option lr_mult '
         '"(r\'mod\\.fc\\..*\', __HEAD_MULT__)")', 1),
        ('if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} '
         '--exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; '
         'else LOAD=""; LR=__LR_SCRATCH__; fi',
         'if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} '
         '--exclude-model-weights mod\\.fc\\..*"; LR=__LR_PRE__; '
         'MULT=("${HEAD_MULT[@]}"); '
         'else LOAD=""; LR=__LR_SCRATCH__; MULT=(); fi', 2),
        ("${COMMON} --start-lr ${LR} --samples-per-epoch ${SPE}",
         '${COMMON} ${MULT[@]+"${MULT[@]}"} --start-lr ${LR} '
         "--samples-per-epoch ${SPE}", 2),
        # THE MULTIPLIER MUST APPEAR IN WEAVER'S OWN LOG. If lr_mult silently
        # fails to match, every pretrained cell trains its fresh head at the
        # trunk rate -- which is wave 1's protocol, the exact thing wave 2
        # exists to replace -- and the output is indistinguishable from a
        # correct run. LEGS_BENCH already checks this; wave 2 must too.
        ('[ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log '
         '--log ${OUT}/stdout.log',
         '[ -z "${ckpt}" ] || python3 experiments/FT/smoke_checks.py load-log '
         '--log ${OUT}/stdout.log\n'
         '                if [ -n "${ckpt}" ]; then\n'
         '                  grep -q "Parameters with lr multiplied by '
         '__HEAD_MULT__" ${OUT}/stdout.log || {\n'
         '                    echo "FATAL: weaver did not apply the head lr '
         'multiplier; this cell"\n'
         '                    echo "       ran wave 1\'s protocol, not ParT\'s '
         'published recipe."; exit 1; }\n'
         '                fi', 2),
        ("lr=${LR} epochs=${EP} subset=",
         "lr=${LR} head_lr_mult=__HEAD_MULT__ weight_decay=0.01 wave=2 "
         "epochs=${EP} subset=", 2),
        # EVERY SUBSET THE GRID ASKS FOR MUST EXIST BEFORE ANY GPU IS SPENT.
        # The legs wait on ${SUB}/DONE, and DONE ALREADY EXISTS from wave 1 --
        # so the wait passes instantly and the first N=1e3 cell dies on a
        # missing parquet after the wave has been running for hours. That is
        # item 33's own recorded defect ("Wave 2 would have run for days and
        # then died on a missing train_N1000_s1.parquet"), and a comment saying
        # "run the rebuild first" does not prevent it. This does.
        ("          epochs_for () { case $1 in 1000) echo __E0__;;",
         "          # Checked HERE, not hours later, for the reason item 33 records.\n"
         "          for S in __FT_SEEDS__; do\n"
         "            for N in __SIZES__; do\n"
         "              for SUB in ${SUB2} ${SUB1}; do\n"
         "                f=${SUB}/train_N${N}_s${S}.parquet\n"
         "                [ -f ${f} ] || { echo \"FATAL: no ${f}.\"; \\\n"
         "                  echo \"Run job-ft-subsets-jc2-w2-raunav and \"\\\n"
         "                       \"job-ft-subsets-jc1-w2-raunav first: the DONE marker\"\\\n"
         "                       \"predates this grid, so waiting on it proves nothing.\"; \\\n"
         "                  exit 1; }\n"
         "              done\n"
         "            done\n"
         "          done\n\n"
         "          epochs_for () { case $1 in 1000) echo __E0__;;", 1),
        # ---- item 36: the wave could not fit on the volume ----------------
        # Measured, not projected: 3.5 GB/cell x 144 = 306 GB against 174 GB
        # free. The three changes below take it to ~41 GB. Each is a wave-2
        # substitution rather than an edit to LEGS because wave 1's 108 cells
        # are complete and must keep the spec they actually ran under.
        #
        # VALIDATION WAS 20x TO 200x THE TRAINING SET. --samples-per-epoch is
        # 1000 or 10000 here; validating on 200000 spent ~80 s of every ~84 s
        # epoch, which is both why the wave was projected at 9.4 days and why
        # NRP measured 38.3% GPU utilisation (policy floor is 40%). Best-epoch
        # selection is on aggregate accuracy, where 20000 jets give s.e.
        # sqrt(.28*.72/2e4) = 0.0032 against an observed epoch-to-epoch step of
        # ~0.010 -- 3x margin. 200000 resolved the same step to 0.001.
        ("--samples-per-epoch-val 200000", "--samples-per-epoch-val 20000", 2),
        # STRIDE, NOT A SMALLER --max-jets. leg1_metrics.py is the only reader
        # of these caches and its --auc-stride DEFAULTS TO 4 (line 142), so the
        # AUC is already computed on 2e6/4 rows; caching the other 3/4 writes
        # 119 GB nothing opens. It must be a stride and not a head slice for the
        # reason that file's own docstring gives (lines 32-36): the test list
        # interleaves Res2P / Res34P / QCD by file, so `[:n]` is biased toward
        # whichever files come first. Striding here caches exactly the rows the
        # default analysis already selects -- no metric changes, and the
        # full-sample accuracy only moves from s.e. 0.0003 to 0.0006.
        ("--max-jets 2000000 --save-logits",
         "--max-jets 2000000 --stride 4 --save-logits", 1),
        ("--dir ${OUT}/features_v2 --n 2000000 --k 162",
         "--dir ${OUT}/features_v2 --n 500000 --k 162", 1),
        # PER-EPOCH CHECKPOINTS: 132 GB across the wave, read by nothing.
        # weaver has NO option to stop writing them -- train.py:836-837 saves
        # state AND optimizer every epoch, and :851 produces
        # net_best_epoch_state.pt by shutil.copy2 of the per-epoch file -- so
        # the only place to drop them is after that copy exists. Both call
        # sites already finished with the cell here: leg 1 has extracted
        # features from net_best_epoch_state.pt and leg 2 has run --predict
        # (which resolves to _best_epoch_state.pt, train.py:878).
        #
        # THE GLOB CANNOT MATCH THE FILE WE KEEP: net_best_epoch_state.pt does
        # not match net_epoch-*. The guards either side are still there because
        # "cannot" is a claim about today's model_prefix, and the cost of being
        # wrong is the only artifact the analysis reads.
        ("                touch ${OUT}/DONE",
         "                [ -f ${OUT}/net_best_epoch_state.pt ] || {\n"
         "                  echo \"FATAL: no net_best_epoch_state.pt in ${OUT};\"\n"
         "                  echo \"       refusing to prune per-epoch checkpoints.\"; exit 1; }\n"
         "                rm -f ${OUT}/net_epoch-*_state.pt ${OUT}/net_epoch-*_optimizer.pt\n"
         "                [ -f ${OUT}/net_best_epoch_state.pt ] || {\n"
         "                  echo \"FATAL: the prune removed net_best_epoch_state.pt in ${OUT}.\"; exit 1; }\n"
         "                touch ${OUT}/DONE", 2),
        # ---- item 36 addendum: the val cut LOWERED utilisation -----------
        # Cutting validation 200k -> 20k halved the epoch but made the GPU
        # number WORSE, because validation WAS the GPU work. Measured on the
        # relaunch, one N=1e3 epoch is 46 s: 4.4 s training and 3.1 s validating
        # (both GPU) against ~38 s of fixed overhead that does not shrink --
        # ~15 s writing the per-epoch checkpoint to a contended CephFS PVC,
        # ~12 s in weaver's per-epoch metrics, ~11 s of teardown. nvidia-smi
        # sampled 3 of 12 seconds busy: ~21%, against 38.3% before and a 40%
        # policy floor.
        #
        # THE ORDER IS THE FIX, NOT THE RECIPE. Training time scales with
        # --samples-per-epoch, so utilisation is a property of N:
        #     N=1e3, 1e4   ~16%      N=1e5   ~55%      N=1e6   ~92%
        # With N as the OUTER loop, each init runs three consecutive N=1e3 cells
        # and then three consecutive N=1e4 cells -- a contiguous ~3.8 h block at
        # ~16%, and NRP measures over 3 h windows. Swapping the loops interleaves
        # every cheap cell with an expensive one: a seed block becomes
        # [1e3, 1e4, 1e5, 1e6] = 199 min averaging 55%, and the worst 3 h window
        # is ~51%.
        #
        # This reorders WHEN cells are computed and nothing else. Every cell is
        # keyed by its own (init, N, seed) output directory, subset file and
        # seed, the loop body reads only ${N} and ${S}, and the DONE skip makes
        # the wave resumable in any order -- so cells already finished under the
        # old order stay valid and are not redone.
        ("            for N in __SIZES__; do\n"
         "              for S in __FT_SEEDS__; do",
         "            for S in __FT_SEEDS__; do\n"
         "              for N in __SIZES__; do", 2),
        # ---- item 36 addendum 2: the 85% line now blocks a wave that fits ----
        # PI-approved 2026-09-16. The guard fires before EVERY cell, and wave 2
        # tripped it at 85% with 163 GB free while needing ~41 GB.
        #
        # THE THRESHOLD WAS CALIBRATED AGAINST A DIFFERENT WAVE. 85% was set when
        # wave 2 was going to write 306 GB into 174 GB of free space -- a wave
        # that genuinely could not fit, and the guard was right to be the thing
        # that caught it. Item 36 cut the wave to ~41 GB, so the same threshold
        # now refuses a wave that peaks near 89% with ~115 GB still free. The
        # premise changed; the number had not.
        #
        # THE SECOND CONDITION IS DELIBERATELY UNTOUCHED. `g >= 50` is the check
        # that actually protects the volume: it is an absolute floor in GB and it
        # does not care how large the disk is or what fraction other people's data
        # occupies. A percentage is a proxy for "will this fill up"; 50 GB free is
        # the thing itself. Raising the proxy while keeping the floor is why this
        # is safe, and if the floor is ever the binding one the wave still stops.
        #
        # WAVE 1 AND THE BENCH LEGS KEEP 85%. space_ok is defined separately in
        # LEGS and LEGS_BENCH, so this substitution reaches wave 2 alone.
        ('[ "$p" -lt 85 ] && [ "$g" -ge 50 ]',
         '[ "$p" -lt 92 ] && [ "$g" -ge 50 ]', 1),
        ('echo "FT LEGS COMPLETE"', 'echo "FT LEGS WAVE 2 COMPLETE"', 1),
    ]
    out = LEGS
    for old, new, n in subs:
        got = out.count(old)
        if got != n:
            raise SystemExit(
                f"FATAL: wave-2 derivation expected {n} occurrence(s) of "
                f"{old[:70]!r} in LEGS, found {got}. LEGS changed; re-check the "
                "substitution rather than loosening it -- a silent miss here "
                "ships a wave-2 spec without the recipe it exists to apply.")
        out = out.replace(old, new)
    return out



# ======================================================================= wave 3
# THE APPROVED DESIGN (project lead, 2026-09-18). Every downstream comparison is
# FOUR label granularities x FIVE pretraining seeds, the PRETRAINING seed is the
# unit of replication, and every pretrained checkpoint is fine-tuned with ONE
# fine-tuning seed (seed 1, training subset s1). The running three-seed wave 2
# stays as the fine-tuning-variance sub-study. Wave 3 is therefore wave 2's
# script -- same recipe, pruning, validation size, guard and loop order, derived
# by the same asserted substitution -- run over the checkpoints wave 2 does not
# cover, into wave 2's own tree, so leg1_metrics.py / leg2_metrics.py read ONE
# tree. Init names are disjoint from wave 2's (asserted), so no cell is shared.
#
# FIVE SHARDS, ONE LOCK PER CELL. One pod per init subset, balanced by expected
# GPU-hours, and every cell is claimed with an atomic `mkdir ${OUT}.lock` before
# anything is written, so two shards cannot collide even if an init list is
# ever wrong. The FAILED / WAIT_TIMEOUT halt markers are per shard: wave 2's
# marker must not stop a shard and a shard's must not stop wave 2's restarts.
#
# NOT EMITTED BY DEFAULT: the groups in INITS_LATER. Their checkpoints do not
# exist yet (two random-label draws are training, ~6.5 days; the masked-particle
# run is at epoch 40 of 80). `--later <group>` emits one when it can be launched.
PIN_W3 = "mtx-s1.52"          # every spec below runs code that first exists here
W3_ROOT = "/data/results/ft/w2b"
BENCH_V2_ROOT = "/data/results/ft/bench_v2"
N_SHARDS = 5


def _ckpt(run: str) -> str:
    return f"/data/results/mtx/mtx-{run}/net_epoch-79_state.pt"


INITS_W3 = (
    [(f"l188-s{s}", _ckpt(f"l188-s{s}"), 188, [1]) for s in range(1, 6)]
    # NEVER mtx-l162-s1: it trained at a different rate. s1b replaced it and
    # already runs in wave 2, as do r16q1-s2/s3/s4.
    + [(f"l162-s{s}", _ckpt(f"l162-s{s}"), 162, [1]) for s in range(2, 6)]
    + [(f"r42q1-s{s}", _ckpt(f"r42q1-s{s}"), 43, [1]) for s in range(1, 6)]
    + [(f"r16q1-s{s}", _ckpt(f"r16q1-s{s}"), 17, [1]) for s in (1, 5)]
    # K is the CHECKPOINT's head width, which is what the preflight self-check
    # verifies: the mass-output twins carry one regression output after their
    # classes (ParT_sophon_arch_mass.py), and --exclude-model-weights drops the
    # whole head anyway, so they fine-tune through the identical code path.
    + [(f"l162mass-s{s}", _ckpt(f"l162mass-s{s}"), 163, [1]) for s in range(1, 6)]
    + [(f"r16q1mass-s{s}", _ckpt(f"r16q1mass-s{s}"), 18, [1]) for s in range(1, 6)]
    + [("rand-d1-s1b", _ckpt("rand-d1-s1b"), 17, [1])]
)
# The self-supervised init is CONVERTED in the pod (experiments/FT/mpm_init.py:
# its keys are `trunk.mod.*` and weaver 0.4.17 has no prefix option) and the
# converted file is what the cell loads, so its checkpoint path is in
# /workspace, like the public checkpoint's. K = 0: it has no head to check.
# mpm-s1 keeps THREE fine-tuning seeds: its pre-registered validity check is
# "beats scratch by more than the fine-tuning-seed spread at 1e3 and 1e4", and
# scratch has three seeds in wave 2.
MPM_SOURCE = {f"mpm-s{s}": _ckpt(f"mpm-s{s}") for s in (1, 2, 3)}
INITS_LATER = {
    "rand-d2": [("rand-d2-s2", _ckpt("rand-d2-s2"), 17, [1])],
    "rand-d3": [("rand-d3-s3", _ckpt("rand-d3-s3"), 17, [1])],
    "mpm-s1": [("mpm-s1", "/workspace/mpm-s1_trunk.pt", 0, [1, 2, 3])],
    "mpm-s2": [("mpm-s2", "/workspace/mpm-s2_trunk.pt", 0, [1])],
    "mpm-s3": [("mpm-s3", "/workspace/mpm-s3_trunk.pt", 0, [1])],
}
# What smoke_checks.py load-log must see for a converted self-supervised init:
# these 39 tensors (class token, two class-attention blocks, final norm) and the
# head are the ONLY missing keys. Measured on the architecture, 2026-09-18.
MPM_FRESH = ("mod.cls_token", "mod.cls_blocks.", "mod.norm.")
MPM_N_FRESH = 39

_all_new = INITS_W3 + [i for g in INITS_LATER.values() for i in g]
_names = [n for n, *_ in INITS + _all_new]
assert len(_names) == len(set(_names)), "an init name is used twice"
assert not ({n for n, *_ in INITS} & {n for n, *_ in _all_new}), (
    "a wave-3 init reuses a wave-2 name; the two waves share one output tree")
for _n, _c, _k, _s in _all_new:
    assert "/mtx-l162-s1/" not in _c and "/mtx-rand-d1-s1/" not in _c, _n
    assert _s and sorted(_s) == _s, (_n, _s)
    assert (_k == 0) == _n.startswith("mpm-"), (_n, _k)

# ---------------------------------------------------------------- benchmarks v2
# Top tagging and quark/gluon for every launchable init: the six of waves 1/2
# (fine-tuning seed 1 only, except scratch, which keeps its three as the
# reference row's spread) plus INITS_W3. The head re-initialisation repeats are
# restricted to two inits x 5 repeats on top tagging at N_max; they hold the
# data subset at s1 and are recorded as s2..s5 of that init, which is what the
# old template did with its 9 (bench_metrics.py tells the two apart by
# train_subsets). Those two inits have ft_seeds == [1], asserted, so a repeat's
# directory can never collide with a real fine-tuning seed's.
BENCH_V2_REPS = [1, 2, 3, 4, 5]
BENCH_V2_REP_INITS = ["l162-s1b", "r16q1-s2"]
BENCH_SIZES = {"top": [1_000, 10_000, 100_000, 1_200_000],
               "qg": [1_000, 10_000, 100_000, 1_600_000]}
# THE N=1e3 CELL GETS THE LEGS' RULE. PI decision, 2026-09-18.
#
# The published recipe is 20 epochs over the training set, and weaver floors
# steps at samples_per_epoch // batch_size (train.py:1006), so a literal N=1e3
# cell trains for ONE step per epoch -- 20 optimiser steps in total, against
# ~380 at N=1e4. That cell would measure the optimiser, not the representation,
# in the place a pretraining effect is predicted to be LARGEST. Item 25 option B
# settled exactly this for the legs (SAMPLES_PER_EPOCH above, with the
# re-shuffling verified against weaver's infinity_mode), and the benchmarks must
# not differ from the legs on a knob that is not the variable under study.
#
# So the TRAINING SET stays 1,000 jets -- that is the controlled variable -- and
# an "epoch" becomes ten passes over it: 19 steps x 20 epochs = 380, matching the
# N=1e4 cell, with the same 20 validation passes and the same best-epoch rule
# applied at the same granularity. The manifest records the samples and the steps
# actually used, so a cell says which rule produced it.
BENCH_SAMPLES_PER_EPOCH: dict[int, int] = {1_000: 10_000}
assert BENCH_SAMPLES_PER_EPOCH[1_000] == SAMPLES_PER_EPOCH[1_000], (
    "the benchmarks and the legs must decouple N=1e3 the same way, or the two "
    "tables' smallest cells are not trained to a comparable step count")
INITS_BENCH_V2 = ([(n, c, k, s if n == "scratch" else [1]) for n, c, k, s in INITS]
                  + INITS_W3)
for _n, _c, _k, _s in INITS_BENCH_V2:
    assert _n not in BENCH_V2_REP_INITS or _s == [1], (_n, _s)
HERWIG_TEST = ["/data/finetune/qg_herwig/qg_herwig_chunk0.parquet",
               "/data/finetune/qg_herwig/qg_herwig_chunk1.parquet"]
# The published test-split sizes. Asserted in the pod BEFORE any GPU work, from
# the parquet metadata, so a truncated staging fails at the top and not after
# 288 fine-tunes have been read out on a sample the community table is not.
N_TEST = {"top": 404_000, "qg": 200_000, "herwig": 200_000}


def cells_legs(inits) -> list[tuple]:
    """(leg, init, N, seed) for every cell wave 3 runs from these inits."""
    return [(leg, n, N, s) for leg in ("leg1", "leg2")
            for n, _, _, seeds in inits for s in seeds for N in SIZES]


def cells_bench(inits) -> list[tuple]:
    """(leg_<set>, init, N, seed) for every cell benchmarks v2 run."""
    out = []
    for d in BENCH_SETS:
        for n, _, _, seeds in inits:
            out += [(f"leg_{d}", n, N, s) for s in seeds for N in BENCH_SIZES[d]]
            if d == "top" and n in BENCH_V2_REP_INITS:
                out += [(f"leg_{d}", n, BENCH_SIZES[d][-1], r)
                        for r in BENCH_V2_REPS if r not in seeds]
    return out


# EXPECTED GPU-HOURS PER CELL -- used to balance the shards and by --plan only.
#   legs    145/144 h: wave 2's own header (144 fine-tunes, ~145 GPU-h) under
#           the protocol these cells run.
#   top/qg  docs/PRD_PLAN.md 5 [V]: "top/q-g ParT recipe ~ 2.3-3 h per
#           checkpoint at N_max" -> 2.3 h top (1.2M jets), 3.0 h q/g (1.6M).
#           Smaller N: the same 20 epochs with training scaled by N over the
#           ~38 s/epoch fixed overhead wave 2 measured, plus the validation
#           pass (20k at N <= 1e4, 200k above): ~0.2 h at 1e3 and 1e4, ~0.5 h
#           at 1e5. +0.05 h per cell for the test read-out.
#           1e3 AND 1e4 CARRY THE SAME FIGURE, and BENCH_SAMPLES_PER_EPOCH is
#           what makes that exact rather than approximate: both now run 10,000
#           samples per epoch, so they differ only in the training SET. The
#           table is unchanged by that decision -- it already assumed the
#           equality, and the fixed per-epoch overhead dominates both.
COST_LEG_CELL_H = 145 / 144
COST_BENCH_CELL_H = {"top": {1_000: 0.25, 10_000: 0.25, 100_000: 0.55, 1_200_000: 2.35},
                     "qg": {1_000: 0.25, 10_000: 0.25, 100_000: 0.55, 1_600_000: 3.05}}


def cost_h(cell) -> float:
    leg, _, N, _ = cell
    return COST_LEG_CELL_H if leg in ("leg1", "leg2") else COST_BENCH_CELL_H[leg[4:]][N]


def shard(inits, cells_of, n=N_SHARDS) -> list[list]:
    """Split inits into n disjoint subsets: heaviest init first, onto the
    lightest shard. Equal costs deal round-robin, so no shard holds one
    granularity alone. Each subset keeps the inits' original order."""
    cost = {i[0]: sum(cost_h(c) for c in cells_of([i])) for i in inits}
    order = sorted(range(len(inits)), key=lambda k: (-cost[inits[k][0]], k))
    load, out = [0.0] * n, [[] for _ in range(n)]
    for k in order:
        j = min(range(n), key=lambda j: (load[j], j))
        out[j].append(k)
        load[j] += cost[inits[k][0]]
    return [[inits[k] for k in sorted(ix)] for ix in out]


def _derive(base: str, subs: list, what: str) -> str:
    """Asserted substitution, as legs_w2 does it: every replacement states how
    many times it must match and raises otherwise."""
    out = base
    for old, new, n in subs:
        got = out.count(old)
        if got != n:
            raise SystemExit(
                f"FATAL: {what} derivation expected {n} occurrence(s) of "
                f"{old[:70]!r}, found {got}. The base template changed; re-check "
                "the substitution rather than loosening it.")
        out = out.replace(old, new)
    return out


def _lock(indent: int) -> str:
    p = " " * indent
    return (
        f"{p}# ONE CELL, ONE POD. mkdir creates the directory or fails, atomically,\n"
        f"{p}# so two pods can never both believe they hold a cell, whatever the\n"
        f"{p}# init lists say. A lock left by an evicted pod of THIS job is\n"
        f"{p}# re-entered (its half-written cell is moved to .partial below, as\n"
        f"{p}# always); a lock held by any OTHER job is left alone and counted.\n"
        f"{p}mkdir -p ${{OUT%/*}}\n"
        f"{p}if ! mkdir ${{OUT}}.lock 2>/dev/null; then\n"
        f"{p}  owner=$(cat ${{OUT}}.lock/owner 2>/dev/null || echo unknown)\n"
        f"{p}  if [ \"${{owner}}\" != \"${{SHARD}}\" ]; then\n"
        f"{p}    echo \"LOCKED: ${{OUT}} is held by ${{owner}}, this is ${{SHARD}}; leaving it\"\n"
        f"{p}    NLOCKED=$((NLOCKED+1)); continue\n"
        f"{p}  fi\n"
        f"{p}  echo \"re-entering ${{OUT}}.lock, left by an earlier pod of ${{SHARD}}\"\n"
        f"{p}fi\n"
        f"{p}echo \"${{SHARD}}\" > ${{OUT}}.lock/owner\n")


def _prune(indent: int) -> str:
    """Wave 2's per-epoch checkpoint prune (item 36), at another indentation.
    tests/test_wave3_specs.py asserts it equals wave 2's line for line."""
    p = " " * indent
    return (
        f"{p}[ -f ${{OUT}}/net_best_epoch_state.pt ] || {{\n"
        f"{p}  echo \"FATAL: no net_best_epoch_state.pt in ${{OUT}};\"\n"
        f"{p}  echo \"       refusing to prune per-epoch checkpoints.\"; exit 1; }}\n"
        f"{p}rm -f ${{OUT}}/net_epoch-*_state.pt ${{OUT}}/net_epoch-*_optimizer.pt\n"
        f"{p}[ -f ${{OUT}}/net_best_epoch_state.pt ] || {{\n"
        f"{p}  echo \"FATAL: the prune removed net_best_epoch_state.pt in ${{OUT}}.\"; exit 1; }}\n")


def _mpm_convert(inits) -> str:
    """The in-pod conversion of every self-supervised init in `inits`."""
    out = ("          # THE SELF-SUPERVISED INIT IS CONVERTED, NOT LOADED RAW. Its keys are\n"
           "          # trunk.mod.* and weaver 0.4.17 has no prefix option, so offered raw it\n"
           "          # loads NOTHING and the trunk trains from random weights (measured;\n"
           "          # experiments/FT/mpm_init.py). The converter keeps the embedding, the\n"
           "          # pair embedding and the 8 particle-attention blocks (194 tensors) and\n"
           "          # refuses to write anything less; the class-attention blocks, class\n"
           "          # token and final norm were never trained by masked-particle modelling\n"
           "          # and start fresh, seeded by the fine-tuning seed like the head.\n")
    for n, c, k, _ in inits:
        if n.startswith("mpm-"):
            out += (f"          [ -f {MPM_SOURCE[n]} ] || {{ echo \"FATAL: no {MPM_SOURCE[n]}\"; exit 1; }}\n"
                    f"          python3 experiments/FT/mpm_init.py --src {MPM_SOURCE[n]} --out {c}\n")
    return out


_MPM_LOADLOG = ("load-log --log ${OUT}/stdout.log --fresh-prefix " + " ".join(MPM_FRESH)
                + f" --expect-fresh {MPM_N_FRESH}")


def legs_w3(inits, shard_name: str) -> str:
    """Wave 3 = wave 2's script over `inits`, by asserted substitution."""
    has_mpm = any(n.startswith("mpm-") for n, *_ in inits)
    assert not has_mpm or all(n.startswith("mpm-") for n, *_ in inits), (
        "a self-supervised init shares a spec only with other self-supervised "
        "inits: the load-log check below is substituted for the whole spec")
    seeds = sorted({s for *_, ss in inits for s in ss})
    subs = [
        ("          ROOT_OUT=/data/results/ft/w2b\n",
         "          ROOT_OUT=/data/results/ft/w2b\n"
         f"          SHARD={shard_name}\n"
         "          NLOCKED=0\n", 1),
        # halt markers per shard: shared ones would let one job stop another
        ("          WAIT_MARK=${ROOT_OUT}/WAIT_TIMEOUT\n",
         "          WAIT_MARK=${ROOT_OUT}/WAIT_TIMEOUT.${SHARD}\n", 1),
        ("          FAIL_MARK=${ROOT_OUT}/FAILED\n",
         "          # Per shard: wave 2 and the other shards write to this same tree,\n"
         "          # and one job's deterministic failure must not halt another's restarts.\n"
         "          FAIL_MARK=${ROOT_OUT}/FAILED.${SHARD}\n", 1),
        # no public checkpoint in any wave-3 init; the self-supervised groups
        # convert their init here instead
        (FETCH_SOPHON, _mpm_convert(inits) if has_mpm else "", 1),
        # name:ckpt:K:seeds -- K is the third field now, not the rest of the line
        ("ckpt=${rest%%:*}; k=${rest#*:}\n",
         "ckpt=${rest%%:*}; k=${rest#*:}; k=${k%%:*}\n", 1),
        # ...AND SO ITS PRECONDITION GOES WITH IT. Leaving the arm-S check behind
        # would hard-fail a shard on three checkpoints it never opens -- the
        # precondition inversion tests/test_spec_preconditions.py exists for
        # ("a spec hard-fails on an OPTIONAL input while never checking a
        # MANDATORY one"). The JetClass-I test-file check below STAYS: leg 2
        # does read those.
        ("          # Leg-2 preconditions, checked HERE rather than days later after leg 1.\n"
         "          for S in 1 2 3; do\n"
         "            [ -f /data/results/e1/arm_s_s${S}/net_best_epoch_state.pt ] || "
         "{ echo \"FATAL: no E1 arm S seed ${S} checkpoint\"; exit 1; }\n"
         "          done\n",
         "          # Leg-2 preconditions, checked HERE rather than days later after leg 1.\n"
         "          # The E1 arm-S checkpoints are NOT among them: wave 2 writes the\n"
         "          # leg2/ref_e1arms-s* rows and this shard does not, so refusing to\n"
         "          # start without a file it never opens would be a precondition on\n"
         "          # someone else's input.\n", 1),
        # THE E1 ARM-S REFERENCE IS WAVE 2's. It is keyed by seed, not init, so
        # five shards and wave 2 would race on the same pred.root; only wave 2
        # writes it and leg2_metrics.py reads it from the shared tree.
        ("          # The N_max scratch reference (E1 arm S, three seeds) on the SAME subset.\n"
         "          for S in 1 2 3; do\n"
         "            OUT=${ROOT_OUT}/leg2/ref_e1arms-s${S}\n"
         "            [ -f ${OUT}/DONE ] && continue\n"
         "            CK=/data/results/e1/arm_s_s${S}/net_best_epoch_state.pt\n"
         "            [ -f \"${CK}\" ] || { echo \"FATAL: no ${CK}\"; exit 1; }\n"
         "            mkdir -p ${OUT}\n"
         "            weaver --predict --data-test ${TEST1} ${PRED} -o fc_params '[(512,0.1)]' "
         "--model-prefix ${CK} --predict-output ${OUT}/pred.root 2>&1 | tee ${OUT}/predict.log | tail -3\n"
         "            # weaver's save_root catches its own write errors and still exits 0.\n"
         "            [ -f ${OUT}/pred.root ] || { echo \"FATAL: no pred.root in ${OUT}\"; exit 1; }\n"
         "            touch ${OUT}/DONE\n"
         "          done\n\n",
         "          # The E1 arm-S reference rows (leg2/ref_e1arms-s*) are written by wave 2\n"
         "          # alone; they are keyed by seed, not init, and five shards racing on one\n"
         "          # pred.root is exactly what the per-cell lock exists to prevent.\n\n", 1),
        # the subset precondition checks every seed any init in this shard runs
        ("          for S in __FT_SEEDS__; do\n"
         "            for N in __SIZES__; do\n"
         "              for SUB in ${SUB2} ${SUB1}; do\n",
         f"          for S in {' '.join(map(str, seeds))}; do\n"
         "            for N in __SIZES__; do\n"
         "              for SUB in ${SUB2} ${SUB1}; do\n", 1),
        # each init runs ITS OWN seeds
        ("            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}\n"
         "            for S in __FT_SEEDS__; do\n",
         "            name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}; seeds=${spec##*:}\n"
         "            for S in ${seeds//,/ }; do\n", 2),
        ("                [ -f ${OUT}/DONE ] && { echo \"skip ${OUT} (DONE)\"; continue; }\n",
         "                [ -f ${OUT}/DONE ] && { echo \"skip ${OUT} (DONE)\"; continue; }\n"
         + _lock(16), 2),
        ("                touch ${OUT}/DONE\n",
         "                touch ${OUT}/DONE\n"
         "                rm -rf ${OUT}.lock\n", 2),
        # THE 128-d FEATURES NOTHING READS. leg1_metrics.py is the only reader of
        # a leg-1 cache and its discover() gates on logits.npy + label188.npy and
        # opens exactly those two (leg1_metrics.py:84-92, 95-97); no probe, no
        # anomaly job and no phase-0 script is ever pointed at a leg tree
        # (job-ft-phase0a imports probe.py for log1m_auc / rejection_at only, and
        # the phase-0 readers glob pred.root). At 500,000 strided rows the matrix
        # is 256 MB per leg-1 cell, 27.6 GB across wave 3 -- against 199 GB free
        # and ~123 GB owed by waves 2 and 3 plus the benchmarks, i.e. a shard
        # halting mid-wave on space_ok. Deleted AFTER the smoke check above,
        # which loads all three arrays, in the same place and the same way the
        # v2 benchmarks do it. logits.npy, label188.npy, observers.npz and
        # extract_manifest.json stay: those are what the analysis opens.
        #
        # WAVE 2 IS UNTOUCHED, as with every other item-36 cut: its spec is the
        # provenance record of a running job and must stay byte-identical.
        ("                python3 experiments/FT/smoke_checks.py features "
         "--dir ${OUT}/features_v2 --n 500000 --k 162\n",
         "                python3 experiments/FT/smoke_checks.py features "
         "--dir ${OUT}/features_v2 --n 500000 --k 162\n"
         "                rm -f ${OUT}/features_v2/features.npy\n", 1),
        ("weight_decay=0.01 wave=2 ", "weight_decay=0.01 wave=3 ", 2),
        ('echo "FT LEGS WAVE 2 COMPLETE"',
         'echo "FT LEGS WAVE 3 ${SHARD} COMPLETE (${NLOCKED} cells left to another job)"', 1),
    ]
    if has_mpm:
        subs += [
            # no head to self-check; the converter already verified the trunk
            ("            [ -f \"${ckpt}\" ] || { echo \"FATAL: ${name}: no ${ckpt}\"; exit 1; }\n"
             "            python3 experiments/EVAL/extract_features.py --checkpoint ${ckpt} --num-classes ${k}",
             "            [ -f \"${ckpt}\" ] || { echo \"FATAL: ${name}: no ${ckpt}\"; exit 1; }\n"
             "            case ${name} in mpm-*) continue;; esac   # no head: verified by mpm_init.py\n"
             "            python3 experiments/EVAL/extract_features.py --checkpoint ${ckpt} --num-classes ${k}", 1),
            ("load-log --log ${OUT}/stdout.log", _MPM_LOADLOG, 2),
        ]
    return _derive(legs_w2(), subs, f"wave-3 {shard_name}")


def legs_bench_v2(inits, shard_name: str) -> str:
    """Benchmarks v2 = LEGS_BENCH over `inits`, by asserted substitution.

    What changes and why: wave 2's prune, validation size at N <= 1e4 and loop
    order (item 36, measured); per-init seeds and the restricted head re-init
    repeats (the approved design); features.npy deleted after its smoke check
    (bench_metrics.py reads logits.npy, label188.npy and DONE; nothing reads
    the 128-d features -- 207 MB per top cell); the Herwig read-out on the q/g
    cells; the per-cell lock and per-shard halt marker. The recipe constants,
    the 85% guard and the constant-LR scheduler are untouched.
    """
    has_mpm = any(n.startswith("mpm-") for n, *_ in inits)
    assert not has_mpm or all(n.startswith("mpm-") for n, *_ in inits)
    has_public = any(c == "/workspace/sophon_public.pt" for _, c, *_ in inits)
    spe = "".join(f"{n}) echo {v};; " for n, v in BENCH_SAMPLES_PER_EPOCH.items())
    subs = [
        ("          ROOT_OUT=/data/results/ft\n          mkdir -p ${ROOT_OUT}\n",
         f"          ROOT_OUT={BENCH_V2_ROOT}\n"
         f"          SHARD={shard_name}\n"
         "          NLOCKED=0\n"
         "          mkdir -p ${ROOT_OUT}\n", 1),
        ("            *) echo \"FATAL: no test files for $1\" >&2; exit 1;; esac; }\n",
         "            *) echo \"FATAL: no test files for $1\" >&2; exit 1;; esac; }\n"
         "          # N <= 1e4 validates on 20k jets, as wave 2 does (item 36): a 200k\n"
         "          # pass there is 20-200x the training set and was measured to hold the\n"
         "          # GPU at 16-21%. Best-epoch selection on aggregate accuracy resolves\n"
         "          # the observed ~0.01 epoch-to-epoch step at s.e. 0.003 with 20k jets.\n"
         "          val_for () { if [ \"$1\" -le 10000 ]; then echo 20000; else echo 200000; fi; }\n"
         "          # BENCH_SAMPLES_PER_EPOCH in scripts/build_ft_jobs.py; empty = N itself.\n"
         f"          samples_for () {{ case $1 in {spe}*) echo $1;; esac; }}\n"
         f"          HERWIG=\"{' '.join(HERWIG_TEST)}\"\n"
         "          nrows () { python3 -c \"import sys, pyarrow.parquet as pq; "
         "print(sum(pq.ParquetFile(f).metadata.num_rows for f in sys.argv[1:]))\" \"$@\"; }\n", 1),
        ("            [ -f \"$(cfg_for ${D})\" ] || { echo \"FATAL: ${D}: no $(cfg_for ${D})\"; exit 1; }\n"
         "          done\n",
         "            [ -f \"$(cfg_for ${D})\" ] || { echo \"FATAL: ${D}: no $(cfg_for ${D})\"; exit 1; }\n"
         "          done\n"
         "          for f in ${HERWIG}; do\n"
         "            [ -f \"${f}\" ] || { echo \"FATAL: no ${f} -- run ft-stage-qg-herwig-raunav first\"; exit 1; }\n"
         "          done\n"
         "          # The test sets must be the PUBLISHED splits, or the rows do not drop\n"
         "          # into the community table. Counted from parquet metadata, here, not\n"
         "          # discovered by a failed smoke check after the fine-tune is paid for.\n"
         "          NTEST_top=$(nrows $(test_for top)); NTEST_qg=$(nrows $(test_for qg)); NTEST_herwig=$(nrows ${HERWIG})\n"
         "          echo \"test jets: top ${NTEST_top}  qg ${NTEST_qg}  qg-herwig ${NTEST_herwig}\"\n"
         f"          [ \"${{NTEST_top}}\" -eq {N_TEST['top']} ] && [ \"${{NTEST_qg}}\" -eq {N_TEST['qg']} ] "
         f"&& [ \"${{NTEST_herwig}}\" -eq {N_TEST['herwig']} ] || {{\n"
         f"            echo \"FATAL: test sets are not the published {N_TEST['top']} / {N_TEST['qg']} / "
         f"{N_TEST['herwig']} jets\"; exit 1; }}\n"
         "          ntest_for () { case $1 in top) echo ${NTEST_top};; qg) echo ${NTEST_qg};; esac; }\n", 1),
        ("          FAIL_MARK=${ROOT_OUT}/FAILED_BENCH\n",
         "          FAIL_MARK=${ROOT_OUT}/FAILED_BENCH.${SHARD}\n", 1),
        (FETCH_SOPHON, FETCH_SOPHON if has_public else (_mpm_convert(inits) if has_mpm else ""), 1),
        # seeds travel with the init; repeats are named per init; seeds OUTSIDE
        # sizes so the cheap and expensive cells interleave (item 36 addendum)
        ("              name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}\n"
         "              for N in $(sizes_for ${D}); do\n",
         "              name=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}\n"
         "              seeds=${spec##*:}; seeds=${seeds//,/ }\n"
         "              # Head re-initialisations: top only, N_max only, and only the inits\n"
         "              # named here -- the benchmark's convention for the headline cell,\n"
         "              # restricted to two inits x 5 (the approved design). A repeat runs\n"
         "              # as seed S on the s1 subset; S=1 IS the ordinary seed-1 cell.\n"
         "              REPS=\"\"\n"
         "              if [ \"${D}\" = \"top\" ]; then\n"
         f"                case \" {' '.join(BENCH_V2_REP_INITS)} \" in *\" ${{name}} \"*) "
         f"REPS=\"{' '.join(map(str, BENCH_V2_REPS))}\";; esac\n"
         "              fi\n"
         "              for S in $(echo ${seeds} ${REPS} | tr ' ' '\\n' | sort -nu); do\n"
         "                for N in $(sizes_for ${D}); do\n", 1),
        ("                # 9 head re-inits at N_max, top only, pretrained arms only --\n"
         "                # the benchmark's convention for the headline cell\n"
         "                # (docs/PRD_PLAN.md 4.1). Everywhere else the three\n"
         "                # fine-tuning seeds are the spread.\n"
         "                REPS=\"__FT_SEEDS__\"\n"
         "                if [ \"${D}\" = \"top\" ] && [ \"${N}\" = \"${NMAX}\" ] && [ -n \"${ckpt}\" ]; then\n"
         "                  REPS=\"__NMAX_REPS__\"\n"
         "                fi\n"
         "                for S in ${REPS}; do\n", "", 1),
        ("                  DSEED=${S}\n"
         "                  [ \"${REPS}\" = \"__NMAX_REPS__\" ] && DSEED=1\n",
         "                  DSEED=${S}\n"
         "                  case \" ${seeds} \" in\n"
         "                    *\" ${S} \"*) ;;\n"
         "                    *) [ \"${N}\" = \"${NMAX}\" ] || continue\n"
         "                       DSEED=1;;\n"
         "                  esac\n", 1),
        ("                  [ -f ${OUT}/DONE ] && { echo \"skip ${OUT} (DONE)\"; continue; }\n",
         "                  [ -f ${OUT}/DONE ] && { echo \"skip ${OUT} (DONE)\"; continue; }\n"
         + _lock(18), 1),
        ("data_config=${CFG} num_classes=2 batch_size=512 steps_per_epoch=$((N/512))",
         "data_config=${CFG} num_classes=2 batch_size=512 "
         "samples_per_epoch=$(samples_for ${N}) steps_per_epoch=$(($(samples_for ${N})/512)) "
         "samples_per_epoch_val=$(val_for ${N}) wave=bench-v2", 1),
        ("--start-lr ${LR} --samples-per-epoch ${N} --samples-per-epoch-val 200000",
         "--start-lr ${LR} --samples-per-epoch $(samples_for ${N}) --samples-per-epoch-val $(val_for ${N})", 1),
        ("                  touch ${OUT}/DONE\n",
         "                  python3 experiments/FT/smoke_checks.py features --dir ${OUT}/features --n $(ntest_for ${D}) --k 2\n"
         "                  # 128 floats per test jet that nothing reads (bench_metrics.py opens\n"
         "                  # logits.npy, label188.npy and DONE): 207 MB per top cell, gone.\n"
         "                  rm -f ${OUT}/features/features.npy\n"
         "                  if [ \"${D}\" = \"qg\" ]; then\n"
         "                    # GENERATOR SHIFT: the same model on Herwig 7.1 jets it never saw\n"
         "                    # (Zenodo 3066475, staged by ft-stage-qg-herwig-raunav). Its own\n"
         "                    # --out, because extract_features refuses to write a cache that\n"
         "                    # holds another manifest; then the pair is moved beside the Pythia\n"
         "                    # one under the names bench_metrics.py --herwig reads.\n"
         "                    python3 experiments/EVAL/extract_features.py --checkpoint ${OUT}/net_best_epoch_state.pt "
         "--num-classes 2 --arm FT_${D}_${name}_N${N}_s${S}_herwig "
         "--data-config ${CFG} --data-test ${HERWIG} --observers jet_pt jet_energy "
         "--out ${OUT}/features_herwig --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits\n"
         "                    python3 experiments/FT/smoke_checks.py features --dir ${OUT}/features_herwig --n ${NTEST_herwig} --k 2\n"
         "                    rm -f ${OUT}/features_herwig/features.npy\n"
         "                    mv ${OUT}/features_herwig/logits.npy ${OUT}/features/logits_herwig.npy\n"
         "                    mv ${OUT}/features_herwig/label188.npy ${OUT}/features/label_herwig.npy\n"
         "                  fi\n"
         "                  # Per-epoch checkpoints (state + optimizer, 20 x ~26 MB), read by\n"
         "                  # nothing once the best-epoch copy exists -- wave 2's prune, item 36.\n"
         + _prune(18)
         + "                  touch ${OUT}/DONE\n"
         "                  rm -rf ${OUT}.lock\n", 1),
        ('echo "FT BENCH LEGS COMPLETE"',
         'echo "FT BENCH V2 ${SHARD} COMPLETE (${NLOCKED} cells left to another job)"', 1),
    ]
    if has_mpm:
        subs.append(("load-log --log ${OUT}/stdout.log", _MPM_LOADLOG, 1))
    return _derive(LEGS_BENCH, subs, f"bench-v2 {shard_name}")


STAGE_HERWIG = PREAMBLE + """
          # THE GENERATOR-SHIFT TEST SET: EnergyFlow quark/gluon showered by
          # Herwig 7.1 (Zenodo 3066475; 40 files verified against the API on
          # 2026-09-18, we take two of the 20 PLAIN ones, never *_withbc_*).
          # Only the two evaluated chunks are staged: 200,000 jets, the size of
          # the Pythia test split, so the two read-outs carry the same
          # statistical weight. Same converter, same measured deta/dphi
          # convention, same lepton charge-sign fix as qg_v2.
          OUT=/data/finetune/qg_herwig
          [ -e ${OUT} ] && { echo "FATAL: ${OUT} exists; staged data is never overwritten"; exit 1; }
          space_ok () { local p=$(df --output=pcent /data | tail -1 | tr -dc 0-9); local g=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); echo "/data ${p}% used, ${g}G free"; [ "$p" -lt 85 ] && [ "$g" -ge 50 ] || { echo "FATAL: /data at ${p}% used, ${g}G free: stop and ask the PI"; exit 1; }; }
          space_ok
          # Into a temporary directory, renamed only once complete: a retry can
          # never find a half-staged ${OUT} and refuse, nor overwrite one.
          TMP=${OUT}.staging.$(date -u +%s)
          python3 scripts/stage_downstream.py --dataset qg_herwig --out ${TMP} \\
            --raw /data/finetune/_raw --limit-files 2
          for i in 0 1; do
            [ -f ${TMP}/qg_herwig_chunk${i}.parquet ] || { echo "FATAL: chunk ${i} not written"; exit 1; }
          done
          mv ${TMP} ${OUT}
          ls -la ${OUT}; du -sh ${OUT}
          echo "STAGE QG HERWIG DONE"
"""


def _new_specs(pin: str, wave3: bool, bench_v2: bool, later: list | None) -> dict:
    """The 2026-09-18 specs. Every one pins PIN_W3 or later."""
    if _tag_index(pin) < _tag_index(PIN_W3):
        raise SystemExit(f"FATAL: these specs run code that first exists at {PIN_W3} "
                         f"(mpm_init.py, load-log --fresh-prefix, the Herwig source); "
                         f"{pin} predates it")
    h = "  # GENERATED by scripts/build_ft_jobs.py -- do not hand-edit. Regenerate.\n  #\n"
    gpu = dict(gpu=True, cpu="4", memory="88Gi", shm="8Gi", backoff=50, pin=pin,
               exclude_hosts=BAD_NODES)
    groups = []                      # (suffix, inits)
    if later:
        groups += [(g, INITS_LATER[g]) for g in later]
    specs = {}
    if wave3:
        shards = [] if later else [(chr(ord("a") + i), s)
                                   for i, s in enumerate(shard(INITS_W3, cells_legs))]
        for suffix, inits in shards + groups:
            name = f"ft-legs-w3-{suffix}-raunav"
            cells = cells_legs(inits)
            specs[f"job-{name}.yaml"] = job(
                name, _fill(legs_w3(inits, name), pin, inits=inits), **gpu,
                header=h + f"  # WAVE 3 of the fine-tuning legs, shard {suffix}: wave 2's script over the\n"
                           "  # checkpoints wave 2 does not cover, ONE fine-tuning seed per pretrained\n"
                           "  # checkpoint (the pretraining seed is the unit of replication), same tree.\n"
                           f"  # inits: {' '.join(n for n, *_ in inits)}\n"
                           f"  # {len(cells)} fine-tunes, ~{sum(cost_h(c) for c in cells):.0f} GPU-h. "
                           "Per-cell mkdir lock; per-shard halt markers.\n")
    if bench_v2:
        shards = [] if later else [(chr(ord("a") + i), s)
                                   for i, s in enumerate(shard(INITS_BENCH_V2, cells_bench))]
        for suffix, inits in shards + groups:
            name = f"ft-legs-bench-v2-{suffix}-raunav"
            cells = cells_bench(inits)
            specs[f"job-{name}.yaml"] = job(
                name, _fill(legs_bench_v2(inits, name), pin, inits=inits), **gpu,
                header=h + f"  # BENCHMARKS v2, shard {suffix}: top tagging and quark/gluon at the\n"
                           "  # published recipe (20 epochs, 1e-4 trunk / 5e-3 head, constant LR,\n"
                           "  # weight decay 0.01), one fine-tuning seed per pretrained checkpoint,\n"
                           "  # three for scratch; wave 2's prune, 20k validation at N <= 1e4 and\n"
                           "  # loop order; the Herwig read-out on every q/g cell. Supersedes\n"
                           "  # job-ft-legs-bench-raunav.yaml, which never ran.\n"
                           f"  # inits: {' '.join(n for n, *_ in inits)}\n"
                           f"  # {len(cells)} fine-tunes, ~{sum(cost_h(c) for c in cells):.0f} GPU-h.\n")
        if not later:
            specs["job-ft-stage-qg-herwig-raunav.yaml"] = job(
                "ft-stage-qg-herwig-raunav", _fill(STAGE_HERWIG, pin), gpu=False, cpu="4",
                memory="32Gi", shm="4Gi", backoff=1, pin=pin,
                header=h + "  # Stage the Herwig 7.1 quark/gluon test set (generator shift). CPU.\n"
                           "  # Two chunks of Zenodo 3066475: ~0.2 GB downloaded, ~0.4 GB written.\n"
                           "  # Run BEFORE any bench-v2 shard; they refuse to start without it.\n")
    return specs


def _tag_index(pin: str) -> int:
    m = re.fullmatch(r"mtx-s1\.(\d+)", pin)
    if not m:
        raise SystemExit(f"FATAL: {pin!r} is not an mtx-s1.<n> tag")
    return int(m.group(1))


def plan(later: list | None = None) -> str:
    """Cell counts and expected GPU-hours per shard, for the launch note."""
    lines = []
    for what, inits, cells_of in (("wave 3", INITS_W3, cells_legs),
                                  ("bench v2", INITS_BENCH_V2, cells_bench)):
        total = 0.0
        for i, s in enumerate(shard(inits, cells_of)):
            cells = cells_of(s)
            h = sum(cost_h(c) for c in cells)
            total += h
            lines.append(f"{what} shard {chr(ord('a') + i)}: {len(s):2d} inits "
                         f"{len(cells):3d} cells {h:6.1f} GPU-h  "
                         f"[{' '.join(n for n, *_ in s)}]")
        lines.append(f"{what} total: {len(cells_of(inits))} cells {total:.1f} GPU-h")
    for g in later or []:
        for what, cells_of in (("wave 3", cells_legs), ("bench v2", cells_bench)):
            cells = cells_of(INITS_LATER[g])
            lines.append(f"{what} later {g}: {len(cells)} cells "
                         f"{sum(cost_h(c) for c in cells):.1f} GPU-h")
    return "\n".join(lines)


def _fill(script: str, pin: str, inits=None) -> str:
    # Waves 1 and 2 emit `name:ckpt:K` and loop over the global FT_SEEDS; wave
    # 3 and the v2 benchmarks emit `name:ckpt:K:s1,s2,..` -- the seeds travel
    # with the init. The first form is what the launched specs carry.
    if inits is None:
        init_str = " ".join(f"{n}:{c}:{k}" for n, c, k, _ in INITS)
    else:
        init_str = " ".join(f"{n}:{c}:{k}:{','.join(map(str, ss))}" for n, c, k, ss in inits)
    return (script
            .replace("__TEST2M__", test2m_list())
            .replace("__INITS__", init_str)
            .replace("__SIZES__", " ".join(str(s) for s in SIZES))
            .replace("__FT_SEEDS__", " ".join(str(s) for s in FT_SEEDS))
            .replace("__E0__", str(EPOCHS[1_000]))
            .replace("__S0__", str(SAMPLES_PER_EPOCH[1_000]))
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


def build(pin: str, wave2: bool = False, wave3: bool = False, bench_v2: bool = False,
          later: list | None = None) -> dict[str, str]:
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
    if wave2:
        # Emit ONLY the wave-2 specs, so a --wave2 run can never write a wave-1
        # file even by accident.
        bodies = {"SUBSETS_JC2": SUBSETS_JC2, "SUBSETS_JC1": SUBSETS_JC1}
        specs = {
            fn: job(jobname, _fill(bodies[body], pin), gpu=False, cpu="4",
                    memory=mem, shm="4Gi", backoff=1, pin=pin,
                    header=h + f"  # WAVE 2 subset rebuild -- {what}. CPU.\n"
                               "  # Adds the N=1e3 point (item 25) to the grid already on disk.\n"
                               "  # make_subsets.py grows DOWNWARD: every existing subset is kept\n"
                               "  # byte-identical, so this does not invalidate wave 1.\n"
                               "  # New name, not a repin of job-ft-subsets-*-raunav: that spec is\n"
                               "  # the provenance record for the subsets already built.\n")
            for fn, (jobname, body, mem, what) in W2_SUBSETS.items()
        }
        specs["job-ft-legs-w2-raunav.yaml"] = job(
            "ft-legs-w2-raunav", _fill(legs_w2(), pin), gpu=True, cpu="4",
            memory="88Gi", shm="8Gi", backoff=50, pin=pin, exclude_hosts=BAD_NODES,
            header=h + "  # WAVE 2 of the fine-tuning legs (item 33, option A, PI-approved).\n"
                       "  # 6 inits x 4 sizes x 3 seeds x 2 legs = 144 fine-tunes, ~145 GPU-h.\n"
                       "  #\n"
                       "  # THREE THINGS DIFFER FROM WAVE 1, and all three are the point:\n"
                       "  #  (1) ParT's published recipe -- head at 50x the trunk via lr_mult,\n"
                       "  #      weight decay 0.01 -- so the legs, our own LR sweep and the\n"
                       "  #      paper we compare against are finally ONE protocol. Item 32\n"
                       "  #      found wave 1 passed neither, so at a nominal 1e-4 the sweep's\n"
                       "  #      fresh head ran at 5e-3 and the legs' at 1e-4, a factor of 50.\n"
                       "  #  (2) The LAST-epoch checkpoints (item 18). Wave 1 loaded\n"
                       "  #      net_best_epoch_state.pt, i.e. epochs 74/64/76/78 -- a pilot.\n"
                       "  #  (3) N=1e3 is in the grid (item 25). Run the w2 subset rebuild\n"
                       "  #      FIRST or the first N=1e3 cell dies on a missing parquet.\n"
                       "  #\n"
                       "  # OUTPUT ROOT IS /data/results/ft/w2. Wave 1's 108 cells are keyed by\n"
                       "  # the same init names and sizes, so a shared root would make every\n"
                       "  # wave-2 cell hit `[ -f DONE ]` and skip. Wave 1 is untouched and the\n"
                       "  # two are reported side by side as the curve.\n")
    if wave3 or bench_v2:
        # ONLY the new specs, so a --wave3 / --bench-v2 run can never rewrite a
        # launched wave-1 or wave-2 file.
        specs = _new_specs(pin, wave3, bench_v2, later)
        for name, text in specs.items():
            left = re.findall(r"__[A-Z0-9_]+__", text)
            assert not left, f"{name}: unfilled {sorted(set(left))}"
    for name, text in specs.items():
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav"), name
        args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        assert 'git clone --depth 1 --branch "${REPO_REF}"' in args, name
        for tok in ("__TEST2M__", "__INITS__", "__SIZES__", "__E0__", "__S0__", "__E1__", "__LAMBDA__", "__SOPHON_SHA256__"):
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
    ap.add_argument("--wave2", action="store_true",
                    help="emit ONLY the wave-2 subset rebuild specs, under new names")
    ap.add_argument("--repin", action="store_true",
                    help="allow moving REPO_REF on specs whose jobs may have run")
    ap.add_argument("--wave3", action="store_true",
                    help=f"emit ONLY the five wave-3 shards (pin {PIN_W3})")
    ap.add_argument("--bench-v2", action="store_true",
                    help=f"emit ONLY the five bench-v2 shards + the Herwig staging (pin {PIN_W3})")
    ap.add_argument("--later", nargs="+", choices=sorted(INITS_LATER), metavar="GROUP",
                    help="with --wave3/--bench-v2: emit the not-yet-launchable group(s) "
                         f"{sorted(INITS_LATER)} instead of the shards")
    ap.add_argument("--plan", action="store_true",
                    help="print cells and expected GPU-hours per shard, write nothing")
    args = ap.parse_args()
    if args.plan:
        print(plan(args.later))
        return 0
    if args.later and not (args.wave3 or args.bench_v2):
        sys.exit("FATAL: --later needs --wave3 and/or --bench-v2")
    if (args.wave3 or args.bench_v2) and args.pin == PIN:
        args.pin = PIN_W3
    specs = build(args.pin, wave2=args.wave2, wave3=args.wave3, bench_v2=args.bench_v2,
                  later=args.later)
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
