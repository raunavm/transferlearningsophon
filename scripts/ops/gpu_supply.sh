#!/bin/bash
# GPUs (general nvidia.com/gpu and nvidia.com/a100) held by our pods (Running, mounting PVC transfer-learning-vol), now and averaged since a UTC time. Reads only.
# usage: gpu_supply.sh 2026-11-22T08:23:00Z
SINCE=$1; NOW=$(date -u +%s); T0=$(date -j -u -f %Y-%m-%dT%H:%M:%SZ "$SINCE" +%s); M=$(( (NOW - T0) / 60 ))
INNER='sum(max by (pod) (kube_pod_container_resource_requests{namespace="cms-ml",resource=~"RES"}) * on(pod) (max by (pod) (kube_pod_status_phase{namespace="cms-ml",phase="Running"}) == 1) * on(pod) max by (pod) (kube_pod_spec_volumes_persistentvolumeclaims_info{namespace="cms-ml",persistentvolumeclaim="transfer-learning-vol"}))'
q () { curl -s --max-time 120 https://prometheus.nrp-nautilus.io/api/v1/query --data-urlencode "query=$1" | python3 -c "import json,sys; r=json.load(sys.stdin)['data']['result']; print(r[0]['value'][1] if r else 'none')"; }
for R in "nvidia_com_gpu|nvidia_com_a100" nvidia_com_gpu nvidia_com_a100; do I=${INNER//RES/$R}
  # "or vector(0)": a sample with none of these pods held counts as zero (review 2026-11-22: the A100 average was
  # taken only over the samples when one was held)
  echo "$R: now $(q "$I"); avg since $SINCE $(q "avg_over_time((($I) or vector(0))[${M}m:5m])") over ${M} min"; done
