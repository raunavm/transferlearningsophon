#!/usr/bin/env bash
# Re-create the 24 non-complete pretraining jobs and the fine-tuning legs job on
# the specs patched 2026-09-07 (tag mtx-s1.11: resume-safe LR schedule; attempt
# records; tensorboard on the PVC). Written by Claude for the PI to run; it
# does nothing without --yes.
#
# What is lost: the three RUNNING jobs restart from their last complete epoch
# checkpoint (auto-resume, RECIPE unchanged), so at most one partial epoch each
# (<= ~75 min). Pending jobs lose nothing. ft-legs skips its 15 DONE legs.
#
# Order: Pending jobs first (no loss), the three running ones last so you can
# time each of them right after its next net_epoch-N_optimizer.pt lands
# (ls -lt /data/results/mtx/<run>/ from any pod that mounts /data).
#
# Preconditions, in this order, all by the PI:
#   git add -A src/utils/resume.py tests/test_resume_lr.py scripts/audit_run.py \
#       scripts/add_run_records.py experiments/E1/seed_weaver.py tests/test_ft_specs.py \
#       experiments/MTX/k8s experiments/FT/k8s/job-ft-legs-raunav.yaml experiments/RUNS.csv
#   git commit -m "keep the learning-rate schedule intact across a resume, and record every attempt"
#   git tag mtx-s1.11 && git push origin main --tags
# The pods clone the tag from GitHub; applying before the tag exists fails every job at clone.
set -euo pipefail
NS=cms-ml
K8S=$(cd "$(dirname "$0")" && pwd)
STAGGER=75

PENDING=(l162-s2 l162-s3 l162-s4 l162-s5 r42_q1-s2 r42_q1-s3 r42_q1-s4 r42_q1-s5
         l162_mass-s1 l162_mass-s2 l162_mass-s3 l162_mass-s4 l162_mass-s5
         r16_q1_mass-s2 r16_q1_mass-s3 r16_q1_mass-s5 l188-s1 l188-s2 l188-s3 l188-s4 l188-s5)
RUNNING=(r42_q1-s1 r16_q1_mass-s1 r16_q1_mass-s4)

jobname () { echo "mtx-$(echo "$1" | tr -d _)-raunav"; }

[ "${1:-}" = "--yes" ] || {
  echo "Would re-create, ${STAGGER}s apart:"
  for a in "${PENDING[@]}"; do echo "  pending  $(jobname $a)   <- ${K8S}/job-mtx-${a}-raunav.yaml"; done
  for a in "${RUNNING[@]}"; do echo "  RUNNING  $(jobname $a)   <- ${K8S}/job-mtx-${a}-raunav.yaml  (resumes from last complete epoch)"; done
  echo "  pending  ft-legs-raunav   <- ${K8S}/../../FT/k8s/job-ft-legs-raunav.yaml"
  echo "Run again with --yes. Verify first: git ls-remote --tags origin | grep mtx-s1.11"
  exit 0
}
git ls-remote --tags origin 2>/dev/null | grep -q 'refs/tags/mtx-s1.11$' || { echo "FATAL: tag mtx-s1.11 is not on origin; the pods would fail to clone"; exit 1; }

recreate () {  # jobname specpath
  kubectl delete job -n "$NS" "$1" --ignore-not-found
  for i in $(seq 1 60); do kubectl get pods -n "$NS" 2>/dev/null | grep -q "^$1-" || break; sleep 5; done
  kubectl apply -f "$2"
  sleep "$STAGGER"
}
for a in "${PENDING[@]}"; do recreate "$(jobname $a)" "${K8S}/job-mtx-${a}-raunav.yaml"; done
recreate ft-legs-raunav "${K8S}/../../FT/k8s/job-ft-legs-raunav.yaml"
for a in "${RUNNING[@]}"; do
  read -r -p "delete and re-create $(jobname $a) now (loses its partial epoch)? [y/N] " ans
  [ "$ans" = "y" ] && recreate "$(jobname $a)" "${K8S}/job-mtx-${a}-raunav.yaml" || echo "skipped $(jobname $a)"
done
kubectl get jobs -n "$NS" | grep -E 'mtx-|ft-legs'
