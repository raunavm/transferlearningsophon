#!/bin/bash
# G-campaign PILOT driver (PLAN v5.1 §7.7) — 1,000 events per class-group per mu.
#
# Matched-event mechanism (§7.1): the hard scatter is generated ONCE per
# (process, batch) into events.hepmc (seeded; seed recorded), then Delphes runs
# on the SAME hepmc once per mu card. Event i in every mu output is the same
# hard scatter by construction. mu50 runs TWICE (rerun-identity check §7.7b).
#
# Modeled on jetclass2_generation/run.sh; requires the same environment
# (MG5, Delphes 3.5.0, LHAPDF, MinBias_100k.pileup) plus the mu-variant cards
# from experiments/G/make_mu_cards.py (card_manifest.json hashes are verified
# before anything runs) and the ntupler with the §7.1.3 extension (NPU,
# per-particle isPU, raw EFlow) once that lands.
#
# Usage:
#   PROC=qcd NEVENT=1000 SEED=2001 ./run_pilot.sh
# Env (required): MG5_PATH DELPHES_PATH GENCFG_PATH CARDS_DIR OUTPUT_PATH
#   GENCFG_PATH = jetclass2_generation/gen_configs ; PROC = a genpack dir name
set -euo pipefail

PROC=${PROC:?set PROC (gen_configs process dir)}
NEVENT=${NEVENT:-1000}
SEED=${SEED:?set SEED (hard-scatter seed; recorded)}
: "${MG5_PATH:?}" "${DELPHES_PATH:?}" "${GENCFG_PATH:?}" "${CARDS_DIR:?}" "${OUTPUT_PATH:?}"

MUS="50 50b 0 0_nopu 25 80 140"     # 50b = second mu50 run (rerun identity)
WORK=$OUTPUT_PATH/pilot_${PROC}_seed${SEED}
mkdir -p "$WORK"; cd "$WORK"

echo "=== verify card hashes against card_manifest.json ==="
python3 - "$CARDS_DIR" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); m = json.loads((d / "card_manifest.json").read_text())
for name, spec in m["cards"].items():
    h = hashlib.sha256((d / name).read_bytes()).hexdigest()
    assert h == spec["sha256"], f"card hash drift: {name}"
print(f"  {len(m['cards'])} cards verified; release sha {m['release_card_sha256'][:16]}…")
EOF

echo "=== hard scatter ONCE (seed $SEED) ==="
mkdir -p proc_base && cp -r "$GENCFG_PATH/$PROC/"* proc_base/
[ -f proc_base/run_gen.sh ] || cp "$GENCFG_PATH/run_gen_default.sh" proc_base/run_gen.sh
cd proc_base
# seed the generator: genpacks read iseed from run_card / pythia cmnd; record it
echo "$SEED" > ../hard_scatter_seed.txt
MG5_PATH=$MG5_PATH ./run_gen.sh "$NEVENT" "$SEED"
cd "$WORK"
cp proc_base/events.hepmc hard_scatter.hepmc
gzip -k hard_scatter.hepmc                      # archived with the UID index (§7.1)

echo "=== Delphes x mu (same hepmc every time) ==="
ln -sf "$DELPHES_PATH/MinBias_100k.pileup" .
: > timing.tsv
for MU in $MUS; do
  CARD=$CARDS_DIR/delphes_card_CMS_JetClassII_mu${MU%b}.tcl
  OUTROOT=delphes_mu${MU}.root
  rm -f "$OUTROOT"
  T0=$(date +%s.%N)
  "$DELPHES_PATH/DelphesHepMC2" "$CARD" "$OUTROOT" hard_scatter.hepmc
  T1=$(date +%s.%N)
  SZ=$(stat -c %s "$OUTROOT" 2>/dev/null || stat -f %z "$OUTROOT")
  printf "mu%s\t%s\t%.3f\t%s\n" "$MU" "$NEVENT" "$(echo "$T1 - $T0" | bc)" "$SZ" >> timing.tsv
  echo "  mu$MU done: $(echo "$T1 - $T0" | bc)s, $SZ bytes"
done

echo "=== ntupler per mu (extended makeNtuples: UID/split, NPU, pu-truth, raw EFlow) ==="
# DSID from the frozen registry (configs/g/dsid_map.yaml in the analysis repo);
# run_number := the generation SEED. Both are part of the immutable L5 UID.
DSID=${DSID:?set DSID (from configs/g/dsid_map.yaml for PROC=$PROC)}
NTUPLER_DIR=${NTUPLER_DIR:-$(dirname "$GENCFG_PATH")/delphes_analyzers}
if [ -f "$NTUPLER_DIR/makeNtuples.C" ]; then
  for MU in $MUS; do
    MUVAL=${MU%b}; MUVAL=${MUVAL#0_nopu}; [ -z "$MUVAL" ] && MUVAL=0   # 50b->50, 0_nopu->0
    ( cd "$NTUPLER_DIR" && root -b -q -l \
        "makeNtuples.C++(\"$WORK/delphes_mu${MU}.root\",\"$WORK/ntuple_mu${MU}.root\",\"JetPUPPIAK8\",\"GenJetAK8\",false,false,${DSID},${SEED},${MUVAL})" )
  done
else
  echo "  makeNtuples.C not found under $NTUPLER_DIR — SKIPPED; analyze_pilot"
  echo "  runs (b),(f),(g) from the Delphes files/timing and defers the rest."
fi

echo "=== pilot batch complete: $WORK ==="
echo "next: python3 experiments/G/pilot/analyze_pilot.py --pilot-dir $WORK"
