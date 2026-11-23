#!/bin/bash
# stall watcher (2026-11-23, after a pod hung 24 h on node-2-3's broken mount unseen by the guard, which watches failures):
# every 30 min, every Running ft-budget raunav pod started over 60 min ago must have a training log written in the last
# 45 min; exits (waking the session) with the pods that do not. Read-only: kubectl get and exec stat only.
while true; do
  BAD=""
  for p in $(kubectl get pods -n cms-ml --no-headers 2>/dev/null | awk '$1 ~ /^ft-budget-.*raunav/ && $3=="Running" {print $1}'); do
    st=$(kubectl get pod -n cms-ml $p -o jsonpath='{.status.startTime}' 2>/dev/null); [ -z "$st" ] && continue
    age=$(( $(date -u +%s) - $(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$st" +%s) ))
    [ $age -lt 3600 ] && continue
    s=$(kubectl exec -n cms-ml $p -- sh -c 'f=$(ls -t /workspace/run/*.log 2>/dev/null | head -1); [ -n "$f" ] && echo $(( $(date +%s) - $(stat -c %Y $f) )) || echo nolog' 2>/dev/null)
    [ -z "$s" ] && continue                      # exec failed (pod ending): no evidence either way
    if [ "$s" = "nolog" ] || [ "$s" -gt 2700 ]; then BAD="$BAD\n$p running $((age/60)) min, log silent: $s s, node $(kubectl get pod -n cms-ml $p -o jsonpath='{.spec.nodeName}')"; fi
  done
  [ -n "$BAD" ] && { echo "$(date -u +%T) STALL:"; echo -e "$BAD"; exit 0; }
  sleep 1800
done
