#!/bin/bash
# Every 20 min: the raw rate-search scan through a running raunav pod (up to three), evaluated locally
# (experiments/FT/rate_search.py); exits with the rows that need action. A failed read waits and retries (PI
# 2026-11-22: never exit on a missing read); every miss is logged in rate_watch.log.
D=$(dirname "$0"); REPO=/Users/raunavmendiratta/transferlearningsophon; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
for i in $(seq 1 216); do
  OK=0
  # guard v7: while STORAGE_HOLD stands (a storage failure, until its load-check passes) nothing reads the cells
  if [ -e $D/guard7_state/STORAGE_HOLD ]; then date -u +%FT%TZ; echo "WATCH: storage hold ($(cat $D/guard7_state/STORAGE_HOLD)): not reading"; sleep 1200; continue; fi
  # 2026-11-22: only pods on this project's volume (another project's raunav pods mount another /data: an empty scan)
  for P in $(kubectl get pods -n cms-ml -o jsonpath='{range .items[*]}{.metadata.name} {.status.phase} {.spec.volumes[*].persistentVolumeClaim.claimName}{"\n"}{end}' 2>/dev/null | awk '$1 ~ /raunav/ && $1 !~ /^guard-probe/ && $1 !~ /^mtx2-tests/ && $2=="Running" && / transfer-learning-vol( |$)/ {print $1}' | head -3); do
    kubectl exec -i -n cms-ml "$P" -- python3 - rate_search < $REPO/experiments/FIGS/pull_mlst_data.py > $D/rate_scan.tmp 2>/dev/null \
      && $PY -c "import json,sys; json.load(open(sys.argv[1]))" $D/rate_scan.tmp 2>/dev/null && { OK=1; mv $D/rate_scan.tmp $D/rate_scan.json; break; }
  done
  [ $OK = 1 ] || echo "$(date -u +%FT%TZ) no read; retry in 20 min" >> $D/rate_watch.log
  if [ $OK = 1 ]; then
    OUT=$($PY $D/rate_watch.py $D/rate_scan.json $D/rate_watch_state.json 2>&1) || { date -u +%FT%TZ; echo "$OUT"; exit 0; }
  fi
  sleep 1200
done
echo "72 h without a change"
