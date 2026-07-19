#!/bin/bash
# E2 pretraining launcher — the official Sophon recipe, granularity-parameterized.
# Derived from jet-universe/sophon@9dd6dd6 train_sophon.sh. Recipe kept verbatim:
# 80 epochs x 10.24M jets, batch 512, start-lr 5e-4, ranger (weaver default
# flat+decay schedule), AMP, fc_params [(512,0.1)], num-workers 5,
# fetch-step 1.0, identical train/val file lists.
# AMENDMENT 2026-07-19: the image's weaver 0.4.17 lacks --data-split-num
# (newer-weaver loading-schedule knob; dropped) and names --log-file as
# --log. Loading-schedule/log-path only — optimizer math untouched, applied
# identically to all five arms, so the E2 invariant holds. Re-hashed at G0.
#
# Deliberate deltas from upstream (each is an E2 design requirement):
#   1. --data-config experiments/E2/data/JetClassII_<GRAN>.yaml — differs from
#      the official config in the truth_label line ONLY, so every granularity
#      trains on the SAME sampled jet stream.
#   2. num_classes per granularity (g2=2, g10sem/g10rand=10, g30=30, g188=188).
#   3. Seeding via experiments/E2/seed_weaver.py (weaver 0.4.17 has no --seed).
#      SINGLE-GPU ONLY: seed_weaver wraps weaver.train:main in-process; this
#      launcher does not support DDP.
#   4. make_weight runs ONCE (g188 config); per-granularity auto configs are
#      cloned from its output via make_data_configs.py --inject-hists, so the
#      flat-(pt,mSD) reweight hists are bit-identical across granularities.
#      Training always uses --no-remake-weights and FAILS FAST if the matching
#      auto config is absent (else weaver would silently recompute weights,
#      a full pass over the dataset).
#   5. No --tensorboard: the upstream eval function's TB block hardcodes
#      188-class score indices (scores[:, 161:188]) and would crash merged heads.
#
# Usage (from the repo root):
#   ./experiments/E2/train_e2.sh make_weight                  # once, g188
#   GRAN=g188 SEED=1 ./experiments/E2/train_e2.sh train [extra weaver args]
# Env: DATADIR (default /data/JetClassII), OUTDIR (default /data/results/e2).
# Extra args go last, so e.g. `--start-lr 2e-4` overrides for the LR sweep.

set -e
[[ -z $DATADIR ]] && DATADIR='/data/JetClassII'
[[ -z $OUTDIR ]] && OUTDIR='/data/results/e2'

MODE=$1
[[ -z $MODE ]] && echo "Usage: $0 [make_weight|train]" && exit 1

case ${GRAN:-g188} in
    g2)     NUMCLS=2   ;;
    g10sem|g10rand) NUMCLS=10 ;;
    g30)    NUMCLS=30  ;;
    g188)   NUMCLS=188 ;;
    *) echo "GRAN must be one of g2 g10sem g10rand g30 g188"; exit 1 ;;
esac

# ---- upstream recipe, verbatim ----
epochs=80
samples_per_epoch=$((10000 * 1024))
samples_per_epoch_val=$((2500 * 1024))
dataconfig="experiments/E2/data/JetClassII_${GRAN:-g188}.yaml"
modelopts="experiments/E2/networks/example_ParticleTransformer_sophon.py --use-amp -o num_classes $NUMCLS -o fc_params [(512,0.1)]"
dataopts="--num-workers 2 --fetch-by-files --fetch-step 5"   # A3: num-workers 5->2. A4: fetch-by-files (stream 5 whole files/fetch) — weaver 0.4.17's default event-fraction mode retains ~ALL loaded jets (anon leak: OOM'd at ~5% of an epoch, 64Gi AND 80Gi). fetch-by-files restores the chunked streaming that --data-split-num gave upstream; loading-only, applied to all arms. Tested: bounded 35-58GB.
batchopts="--batch-size 512 --start-lr 5e-4"

# train: {Res2P: 0000-0199, Res34P: 0000-0859, QCD: 0000-0279}
# val:   {Res2P: 0200-0249, Res34P: 0860-1074, QCD: 0280-0349}
trainset_res2p=$(for i in $(seq -w 0000 0199); do echo -n "Res2P:${DATADIR}/Pythia/Res2P_$i.parquet "; done)
trainset_res34p=$(for i in $(seq -w 0000 0859); do echo -n "Res34P:${DATADIR}/Pythia/Res34P_$i.parquet "; done)
trainset_qcd=$(for i in $(seq -w 0000 0279); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)
valset_res2p=$(for i in $(seq -w 0200 0249); do echo -n "${DATADIR}/Pythia/Res2P_$i.parquet "; done)
valset_res34p=$(for i in $(seq -w 0860 1074); do echo -n "${DATADIR}/Pythia/Res34P_$i.parquet "; done)
valset_qcd=$(for i in $(seq -w 0280 0349); do echo -n "${DATADIR}/Pythia/QCD_$i.parquet "; done)
allset_res2p=$(for i in $(seq -w 0000 0249); do echo -n "Res2P:${DATADIR}/Pythia/Res2P_$i.parquet "; done)
allset_res34p=$(for i in $(seq -w 0000 1074); do echo -n "Res34P:${DATADIR}/Pythia/Res34P_$i.parquet "; done)
allset_qcd=$(for i in $(seq -w 0000 0349); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)

if [[ $MODE == "make_weight" ]]; then
    # one shared weight pass with the g188 config; weaver writes
    # experiments/E2/data/JetClassII_g188.<md5>.auto.yaml next to the config.
    # Afterwards run: make_data_configs.py --inject-hists <that file>
    # and persist all *.auto.yaml to the PVC cache.
    weaver --print \
        --data-train $allset_res2p $allset_res34p $allset_qcd \
        --data-config experiments/E2/data/JetClassII_g188.yaml \
        --network-config experiments/E2/networks/example_ParticleTransformer_sophon.py \
        --use-amp -o num_classes 188 -o fc_params "[(512,0.1)]" \
        $dataopts $batchopts \
        --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} \
        --num-epochs $epochs --optimizer ranger --gpus "" \
        --log "${OUTDIR}/make_weight.log" \
        "${@:2}"

elif [[ $MODE == "train" ]]; then
    [[ -z $SEED ]] && echo "SEED is required for training" && exit 1
    ls experiments/E2/data/JetClassII_${GRAN}.*.auto.yaml >/dev/null 2>&1 \
        || { echo "FATAL: no auto config for ${GRAN} — run make_weight + --inject-hists first"; exit 1; }
    RUN="${OUTDIR}/${GRAN}_s${SEED}"
    python3 experiments/E2/seed_weaver.py --seed $SEED --no-remake-weights \
        --data-train $trainset_res2p $trainset_res34p $trainset_qcd \
        --data-val $valset_res2p $valset_res34p $valset_qcd \
        --data-config $dataconfig --network-config $modelopts \
        --model-prefix "${RUN}/net" \
        $dataopts $batchopts \
        --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} \
        --num-epochs $epochs --optimizer ranger --gpus 0 \
        --log "${RUN}/train.log" \
        "${@:2}"

else
    echo "Usage: $0 [make_weight|train]"
    exit 1
fi
