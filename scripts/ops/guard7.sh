#!/bin/bash
# guard v7 (PI 2026-11-22, after v6.2 suspended everything on one failed write test, in another project's pod: about
# 100 GPU-h of progress and every GPU we held):
#  - storage is judged by the probe job on transfer-learning-vol only. A failed write test in a running pod of a managed
#    job (on our volume), or a new failed pod with a storage error, starts a probe; a failure is CONFIRMED only when the
#    probe fails twice in a row, PROBE_GAP s apart. A probe that gives no answer is no evidence either way.
#  - on a confirmed failure nothing is suspended: every managed job's parallelism is held at its running pods (pending
#    pods are removed, running pods are left alone), re-applied every check, and STORAGE_HOLD is written (readers wait).
#  - recovered after CLEAN_N passing probes in a row: launching resumes, and every file under SCAN_ROOTS written from
#    the first failed probe less MARGIN s to the recovery is load-checked by a storage-scan job
#    (scripts/build_storage_scan_job.py at SCAN_TAG). STORAGE_HOLD is removed when the scan finds nothing; otherwise
#    the report is named in SCAN_FAILED, the hold stays, and the cells are rerun by hand.
#  - every storage event is appended to EVENTS (a watcher wakes on it).
#  - fills free slots under CAP in list order (live = the largest of the pods listed, the jobs' active pods and the pods
#    unsuspended jobs are owed, failed indexes not owed); alerts and exits on a failed pod of a managed job that is
#    neither a storage error nor a divergence, a failed job outside the search jobs, or an NRP refusal.
# Usage: guard7.sh <launch list: "job|spec|max parallelism" per line>
#   env (defaults): CAP=51 CHECK=180 PROBE_GAP=180 CLEAN_N=10 MARGIN=600 STATE_DIR=<scratchpad>/guard7_state
#                   SCAN_TAG=mtx-s3.39 SCAN_ROOTS=/data/results PROBE_NAME=guard-probe-raunav
#   STATE_DIR/probe_path, if present, names the file the probes write (the cluster test points it at a missing folder)
REPO=/Users/raunavmendiratta/transferlearningsophon; cd $REPO
D=$(cd "$(dirname "$0")" && pwd); LIST=$1; NS=cms-ml
CAP=${CAP:-51}; CHECK=${CHECK:-180}; PROBE_GAP=${PROBE_GAP:-180}; CLEAN_N=${CLEAN_N:-10}; MARGIN=${MARGIN:-600}
STATE_DIR=${STATE_DIR:-$D/guard7_state}; SCAN_TAG=${SCAN_TAG:-mtx-s3.39}; SCAN_ROOTS=${SCAN_ROOTS:-/data/results}
PROBE_NAME=${PROBE_NAME:-guard-probe-raunav}; STEP=4
mkdir -p $STATE_DIR; HOLD=$STATE_DIR/STORAGE_HOLD; EVENTS=$STATE_DIR/EVENTS; SCAN_FAILED=$STATE_DIR/SCAN_FAILED
ts () { date -u +%T; }
iso () { date -u -r $1 +%Y-%m-%dT%H:%M:%SZ; }
event () { echo "$(ts) $*"; echo "$(date -u +%FT%TZ) $*" >> $EVENTS; }
probe_path () { [ -s $STATE_DIR/probe_path ] && cat $STATE_DIR/probe_path || echo /data/results/.write_probe; }
managed () { cut -d'|' -f1 "$LIST" | grep . | sort -u; }
ours () { kubectl get pods -n $NS -o jsonpath='{range .items[*]}{.metadata.name} {.spec.volumes[*].persistentVolumeClaim.claimName}{"\n"}{end}' 2>/dev/null | awk '$1 ~ /raunav/ && / transfer-learning-vol( |$)/ {print $1}' | sort; }
of_managed () { local m; m=$(managed | sed 's/$/-[0-9]+-[a-z0-9]{5}$/' | sed 's/^/^/' | paste -sd'|' -); grep -E "$m"; }
failed_pods () { join <(kubectl get pods -n $NS --no-headers 2>/dev/null | awk '($3=="OOMKilled" || $3=="Error" || $3=="Failed" || $3 ~ /CrashLoop/ || $3 ~ /ContainerStatusUnknown/) {print $1}' | of_managed | sort) <(ours); }
running_pods () { join <(kubectl get pods -n $NS --no-headers 2>/dev/null | awk '$3=="Running" {print $1}' | of_managed | sort) <(ours); }
exec_test () {    # ok / fail / none: one write from up to 4 running managed pods on distinct nodes
  local nodes="" n=0 ok=0 out P nd w; w=$(probe_path)
  for P in $(running_pods); do
    nd=$(kubectl get pod -n $NS $P -o jsonpath='{.spec.nodeName}' 2>/dev/null); case " $nodes " in *" $nd "*) continue;; esac
    nodes="$nodes $nd"; n=$((n+1))
    out=$(kubectl exec -n $NS $P -- sh -c "date > $w && echo WRITE_OK || echo WRITE_FAIL" 2>/dev/null)
    case "$out" in *WRITE_OK*) ok=$((ok+1));; *WRITE_FAIL*) echo fail; return;; esac
    [ $n -ge 4 ] && break
  done
  [ $ok -gt 0 ] && echo ok || echo none
}
probe () {       # ok / fail / none: the probe job on transfer-learning-vol writes once and ends; deleted after
  local v=none i P ph w; w=$(probe_path)       # a subshell (R=...): the caller records LASTPROBE
  kubectl delete job -n $NS $PROBE_NAME --wait=true >/dev/null 2>/dev/null
  sed "s|name: guard-probe-raunav|name: $PROBE_NAME|; s|date > /data/results/.write_probe|date > $w|" $D/guard_probe_job.yaml \
    | kubectl create -f - >/dev/null 2>/dev/null || { echo none; return; }
  for i in $(seq 1 24); do
    sleep 10
    P=$(kubectl get pods -n $NS --no-headers 2>/dev/null | awk -v j="$PROBE_NAME" 'index($1, j "-") == 1 {print $1; exit}'); [ -z "$P" ] && continue
    ph=$(kubectl get pod -n $NS $P -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$ph" = Succeeded ] || [ "$ph" = Failed ]; then
      # PI 2026-11-24: a probe the laptop could not read (offline) is no probe, never a failed one. A log that cannot
      # be read counts as fail only when the API still answers for the pod (it never ran: no log); otherwise ask again
      if L=$(kubectl logs -n $NS $P 2>/dev/null); then echo "$L" | grep -q WRITE_OK && v=ok || v=fail
      elif kubectl get pod -n $NS $P >/dev/null 2>&1; then v=fail
      else continue; fi
      break
    fi
  done
  kubectl delete job -n $NS $PROBE_NAME --wait=false >/dev/null 2>/dev/null
  echo $v
}
live_count () {
  local a b c
  a=$(kubectl get pods -n $NS --no-headers 2>/dev/null | awk '$1 ~ /raunav/ && ($3=="Running" || $3=="Pending" || $3 ~ /Creat/ || $3 ~ /Init/)' | wc -l | tr -d ' ')
  b=$(kubectl get jobs -n $NS -o jsonpath='{range .items[*]}{.metadata.name} {.status.active}{"\n"}{end}' 2>/dev/null | awk '$1 ~ /raunav/ {s += $2} END {print s+0}')
  c=$(kubectl get jobs -n $NS -o jsonpath='{range .items[*]}{.metadata.name}|{.spec.parallelism}|{.spec.completions}|{.status.succeeded}|{.spec.suspend}|{.status.failedIndexes}|{.status.conditions[*].type}{"\n"}{end}' 2>/dev/null | awk -F'|' '$1 ~ /raunav/ && $7 !~ /Complete|Failed/ && $5 != "true" {p=$2+0; c=($3==""?p:$3+0); s=($4 ~ /^[0-9]+$/ ? $4+0 : 0); nf=0; if ($6 != "") {n=split($6, a, ","); for (k=1;k<=n;k++) {if (split(a[k], b, "-")==2) nf += b[2]-b[1]+1; else nf++}}; r=c-s-nf; if (r<0) r=0; t += (p<r?p:r)} END {print t+0}')
  [ -z "$a" ] && a=999; [ -z "$b" ] && b=999; [ -z "$c" ] && c=999
  m=$a; [ "$b" -gt "$m" ] && m=$b; [ "$c" -gt "$m" ] && m=$c; echo "$m"
}
freeze () {      # each managed job held at its running pods: no new pod starts, none running is touched
  local j cur run st
  for j in $(managed); do
    st=$(kubectl get job -n $NS $j --no-headers 2>/dev/null | awk '{print $2}'); [ -z "$st" ] || [ "$st" = Complete ] && continue
    cur=$(kubectl get job -n $NS $j -o jsonpath='{.spec.parallelism}' 2>/dev/null); [ -z "$cur" ] && continue
    run=$(kubectl get pods -n $NS --no-headers 2>/dev/null | awk -v j="$j" 'index($1, j "-") == 1 && $3=="Running"' | wc -l | tr -d ' ')
    if [ "$cur" -gt "$run" ]; then
      kubectl patch job -n $NS $j --type merge -p "{\"spec\":{\"parallelism\":$run}}" >/dev/null 2>&1 && echo "$(ts) frozen: $j parallelism $cur -> $run (its running pods)"
    fi
  done
}
MODE=normal; FAILS=0; CLEAN=0; T_FAIL=0; LASTPROBE=0; SCAN_JOB=""; SCAN_OUT=""
[ -e $HOLD ] && echo "$(ts) note: $HOLD is present from before (a failure not yet cleared): readers keep waiting"
BASE=$(failed_pods)
echo "$(ts) guard v7 start; CAP $CAP, check ${CHECK}s, probe gap ${PROBE_GAP}s, $CLEAN_N clean probes to recover; baseline $(echo "$BASE" | grep -c .) failed pods"
for i in $(seq 1 100000); do
  NOW=$(failed_pods); NEW=$(comm -13 <(echo "$BASE") <(echo "$NOW")); TRIGGER=""
  for p in $NEW; do
    LOG=$(kubectl logs -n $NS $p 2>/dev/null | tail -40)
    if echo "$LOG" | grep -q -E "Permission denied|No space left on device|Read-only file system|Input/output error|Stale file handle"; then
      TRIGGER="failed pod $p shows a storage error"
    elif echo "$LOG" | grep -q -E "diverged \((NaN|non-finite) training loss\)|diverged: not retried|diverged: not a result"; then
      echo "$(ts) note: $p diverged (NaN training loss), a rate-search outcome, not a fault"
    else
      echo "$(ts) ALERT non-storage failure: $p"; kubectl get pods -n $NS --no-headers | grep "^$p "; kubectl logs -n $NS $p 2>/dev/null | tail -5; exit 0
    fi
  done
  BASE=$(printf "%s\n%s\n" "$BASE" "$NOW" | grep . | sort -u)
  J=$(kubectl get jobs -n $NS --no-headers 2>/dev/null | awk '$2=="Failed" {print $1}' | grep -x -F -f <(managed) | grep -v -E "^ft-budget-fixed-sweep-[a-z0-9-]+-raunav$")
  [ -n "$J" ] && { echo "$(ts) ALERT failed job: $J"; exit 0; }
  NOWS=$(date -u +%s)
  case $MODE in
    normal)
      [ -z "$TRIGGER" ] && [ "$(exec_test)" = fail ] && TRIGGER="a write test in a running pod failed"
      if [ -n "$TRIGGER" ]; then
        LASTPROBE=$(date -u +%s); R=$(probe)
        if [ "$R" = fail ]; then MODE=suspect; FAILS=1; T_FAIL=$LASTPROBE; event "SUSPECT: $TRIGGER; probe 1 failed; probe 2 in ${PROBE_GAP}s"
        else echo "$(ts) $TRIGGER; the probe on transfer-learning-vol says $R: no action"; fi
      fi;;
    suspect)
      if [ $((NOWS - LASTPROBE)) -ge $PROBE_GAP ]; then
        LASTPROBE=$(date -u +%s); R=$(probe)
        if [ "$R" = fail ]; then MODE=frozen; FAILS=2; CLEAN=0; echo "$(iso $T_FAIL)" > $HOLD; event "CONFIRMED: probe 2 failed; launching stops, running pods left alone; $HOLD written"
        elif [ "$R" = ok ]; then MODE=normal; FAILS=0; event "NOT CONFIRMED: probe 2 passed; nothing changed"
        else echo "$(ts) probe 2 gave no answer; asked again next check"; fi
      fi;;
    frozen)
      LASTPROBE=$(date -u +%s); R=$(probe)
      if [ "$R" = ok ]; then CLEAN=$((CLEAN + 1)); elif [ "$R" = fail ]; then CLEAN=0; fi
      echo "$(ts) frozen: probe $R, $CLEAN of $CLEAN_N clean"
      if [ $CLEAN -ge $CLEAN_N ]; then
        SINCE=$(iso $((T_FAIL - MARGIN))); UNTIL=$(iso $NOWS)
        F=$(/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/build_storage_scan_job.py --tag $SCAN_TAG --since $SINCE --until $UNTIL --roots $SCAN_ROOTS)
        SCAN_JOB=$(awk '/^  name: /{print $2; exit}' $F); SCAN_OUT="$SINCE..$UNTIL"
        kubectl apply -n $NS -f $F >/dev/null && event "RECOVERED: $CLEAN_N clean probes; launching resumes; load-check of $SINCE .. $UNTIL: $SCAN_JOB" \
          || event "RECOVERED, but the load-check job did not apply ($F): hold kept"
        MODE=normal; FAILS=0; CLEAN=0
      fi;;
  esac
  if [ -n "$SCAN_JOB" ]; then
    SS=$(kubectl get job -n $NS $SCAN_JOB -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    case "$SS" in
      *Complete*)
        P=$(kubectl get pods -n $NS --no-headers 2>/dev/null | awk -v j="$SCAN_JOB" 'index($1, j "-") == 1 {print $1; exit}')
        NF=$(kubectl logs -n $NS $P 2>/dev/null | awk '/^SCAN FAILURES /{print $3}' | tail -1)
        if [ "$NF" = 0 ]; then rm -f $HOLD; event "LOAD-CHECK PASSED ($SCAN_OUT): $HOLD removed"
        else echo "$SCAN_JOB $SCAN_OUT failures=${NF:-?}" >> $SCAN_FAILED; event "LOAD-CHECK FOUND ${NF:-?} FAILURES ($SCAN_OUT): hold kept; rerun the cells (report.tsv of $SCAN_JOB)"; fi
        SCAN_JOB="";;
      *Failed*) event "LOAD-CHECK JOB FAILED ($SCAN_JOB): hold kept"; SCAN_JOB="";;
    esac
  fi
  if [ $MODE = frozen ]; then
    freeze
  else
    live=$(live_count); free=$((CAP - live))
    while IFS='|' read -r name spec maxp; do
      [ -z "$name" ] && continue; [ $free -ge 1 ] || break
      st=$(kubectl get job -n $NS $name --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = Complete ] && continue
      [ -z "$st" ] && continue                 # never re-create a job
      cur=$(kubectl get job -n $NS $name -o jsonpath='{.spec.parallelism}' 2>/dev/null)
      if [ -n "$cur" ] && [ "$cur" -lt "$maxp" ]; then
        add=$free; [ $add -gt $STEP ] && add=$STEP; np=$((cur + add)); [ $np -gt $maxp ] && np=$maxp
        out=$(kubectl patch job -n $NS $name --type merge -p "{\"spec\":{\"parallelism\":$np}}" 2>&1)
        echo "$out" | grep -qi "utiliz" && { echo "$(ts) ALERT NRP refused patch of $name: $out"; exit 0; }
        echo "$out" | grep -q "patched" && { echo "$(ts) $name parallelism $cur -> $np (live $live)"; free=$((free - (np - cur))); }
      fi
    done < "$LIST"
  fi
  sleep $CHECK
done
