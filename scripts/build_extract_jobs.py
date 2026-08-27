#!/usr/bin/env python3
"""Emit experiments/EVAL/k8s/job-extract-<run>-raunav.yaml -- frozen-feature jobs.

WHY THESE CAN RUN WHEN NOTHING ELSE CAN
---------------------------------------
Feature extraction is a forward pass with no backward pass, so it does not need
a GPU to be correct -- only to be fast. That matters right now for a reason that
has nothing to do with the science: the us-west 3090 pool has not scheduled one
of our jobs in over three days, while CPU capacity is uncontended. A CPU-only
extraction job therefore runs TODAY against a checkpoint that already exists,
which is what makes it possible to validate the whole downstream chain --
extract -> probe -> paired contrast -- before the full-budget runs land, rather
than discovering it is broken on the day they finish.

Pass --gpu to build the fast version for when a GPU is actually obtainable.

WHY A SMOKE TEST AGAINST A G1 CHECKPOINT IS WORTH A JOB
-------------------------------------------------------
The G1 sweep checkpoints are 16-epoch models at 20% budget. Their probe numbers
are NOT a result and must never be reported as one -- the arms were trained at
different rates, one seed each, on a fifth of the budget. What they ARE is real
weights over real data, which is the only thing that exercises the parts a
synthetic test cannot: weaver's data loading, the observers, label alignment
across two independently-extracted arms, and the hook firing inside the actual
container image (whose weaver is 0.4.17, not the 0.4.16 installed locally).

Run:  python3 scripts/build_extract_jobs.py [--gpu] [--max-jets N]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s"
# mtx-s1.5 (was s1.4, was s1.3). s1.3 is at 4fcd165, which PREDATES
# experiments/EVAL/extract_features.py, so both extraction jobs cloned it,
# found no such file, and crash-looped. s1.4 is the first tag carrying the
# downstream code, and every file a TRAINING pod executes is byte-identical
# between s1.2 and s1.4, so nothing already running is affected.
PIN = "mtx-s1.6"
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"

# (run_id, arm, K, checkpoint dir). The G1 rows are the SMOKE TEST described
# above; the mtx rows are the real thing and only work once those runs finish.
RUNS = [
    ("g1-l162-lr1e3",   "L162",   162, "/data/results/g1/g1-l162-lr1e3"),
    ("g1-r16q1-lr5e4",  "R16_Q1",  17, "/data/results/g1/g1-r16q1-lr5e4"),
]

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  # FROZEN-FEATURE EXTRACTION -- {run_id} ({arm}, K={k})
  #
  # Reads the 128-d representation the classifier head sees, for the frozen
  # probes in docs/DOWNSTREAM_SUITE.md. NOT the same as the K-way scores that
  # scripts/build_eval_jobs.py caches: a probe fitted on scores measures the
  # head that this arm's own vocabulary trained, which is the variable under
  # study. See experiments/EVAL/extract_features.py.
  #
  # {device_note}
  name: extract-{name}-raunav
  namespace: cms-ml
spec:
  backoffLimit: 50
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: extract
        image: {image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          pip install --no-cache-dir -q pyarrow || exit 1

          CKPT={ckpt_dir}/net_best_epoch_state.pt
          # weaver writes net_best_epoch_state.pt from ITS OWN best-epoch rule.
          # If it is absent the run did not finish, and extracting from the last
          # epoch instead would silently probe a different model than the one
          # every other number refers to.
          [ -f "${{CKPT}}" ] || {{ echo "FATAL: no ${{CKPT}}. Run unfinished?"; ls -la {ckpt_dir} | head -20; exit 1; }}

          # features_v2, NOT features. The file list was interleaved by family
          # on 2026-08-27, so a re-extraction now reads DIFFERENT jets than the
          # ones under .../features -- which are what probe-bvc-v1 and the
          # recorded L162-vs-R16_Q1 numbers were computed on. extract_features.py
          # np.save()s unconditionally, so reusing the path would overwrite those
          # inputs with a different sample and silently invalidate a result
          # already in experiments/RUNS.csv. Different sample, different path.
          OUT=/data/results/eval/{run_id}/features_v2
          mkdir -p ${{OUT}}

          PYTHONUNBUFFERED=1 python3 experiments/EVAL/extract_features.py \\
            --checkpoint "${{CKPT}}" \\
            --num-classes {k} \\
            --arm {arm} \\
            --data-config configs/data/JetClassII_base.yaml \\
            --data-test {file_list} \\
            --out ${{OUT}} \\
            --batch-size 512 --num-workers 1 --fetch-step 1{max_jets}

          echo "=== manifest ==="
          cat ${{OUT}}/extract_manifest.json
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
        resources:
          requests: {{ memory: "{mem}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}"{gpu_req}, ephemeral-storage: "20Gi" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]{gpu_aff}
      volumes:
      - name: jc2
        persistentVolumeClaim:
          claimName: tn-pvc-base-jetclass2
          readOnly: true
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""

GPU_AFF = """
              - key: nvidia.com/gpu.product
                operator: In
                values: ["NVIDIA-GeForce-RTX-3090"]"""


# THE FILE ORDER IS LOAD-BEARING, and getting it wrong is silent.
#
# weaver sets shuffle=False when for_training=False (dataset.py:327, verified in
# the running image), so traversal is EXACTLY the order given on the command
# line, and --max-jets stops mid-list. The original list was
# Res2P_{0250..0299} Res34P_{1075..1289} QCD_{0350..0419} -- 335 files in strict
# family order. At ~82k selected jets per file, --max-jets 400000 stops inside
# the FIFTH file, so the 2026-08-27 extractions contain only Res2P: 15 unique
# native labels, range 0-14, and ZERO QCD jets. probe.py's bvc_qcd task (labels
# 169 vs 181) hit its rows.size < 1000 guard and returned {"skipped": true}.
# Nothing errored; the manifest looked fine.
#
# Interleaving by family fixes it: any prefix of the list is family-balanced, so
# --max-jets can be set for cost rather than coverage. Both arms must be
# extracted from the SAME list or label188_sha256 diverges and probe.py's
# alignment gate fails -- which is the one failure here that is not silent.
FAMILIES = [("Res2P", 250, 299), ("Res34P", 1075, 1289), ("QCD", 350, 419)]


def interleaved_files() -> str:
    """Round-robin across families, so every prefix spans all three."""
    per = [[f"/jc2/jet_data/{fam}_{i:04d}.parquet" for i in range(lo, hi + 1)]
           for fam, lo, hi in FAMILIES]
    out = []
    for i in range(max(len(x) for x in per)):
        for fam in per:
            if i < len(fam):
                out.append(fam[i])
    return " ".join(out)


def build(run_id, arm, k, ckpt_dir, gpu: bool, max_jets: int) -> tuple[str, str]:
    name = run_id.replace("_", "-").lower()
    text = TEMPLATE.format(
        run_id=run_id, arm=arm, k=k, ckpt_dir=ckpt_dir, image=IMAGE, pin=PIN,
        name=name + ("-gpu" if gpu else ""),
        device_note=("GPU build." if gpu else
                     "CPU-ONLY ON PURPOSE. Extraction is a forward pass, so a "
                     "GPU buys speed and\n  # not correctness -- and the GPU "
                     "queue has not scheduled anything in 3.5 days\n  # while "
                     "CPU is uncontended."),
        max_jets=(f" \\\n            --max-jets {max_jets}" if max_jets else ""),
        file_list=interleaved_files(),
        # 48Gi, not 32Gi. The loader dominates, not the model: 32Gi
        # produced 13 OOMKilled pods before a single jet was written. Training
        # measures 35-58 GB at fetch_step=5; extraction runs fetch_step=1 with
        # one worker, roughly a tenth of that file buffering, and 48Gi leaves
        # generous room over the estimate while staying far below the 76Gi that
        # has not scheduled in hours.
        mem="48Gi" if not gpu else "76Gi",
        cpu="8" if not gpu else "4",
        gpu_req=', nvidia.com/gpu: "1"' if gpu else "",
        gpu_aff=GPU_AFF if gpu else "")
    return f"job-extract-{name}{'-gpu' if gpu else ''}-raunav.yaml", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--max-jets", type=int, default=0)
    args = ap.parse_args()
    # The pod clones a TAG, not the working tree, so a script that exists here
    # can be absent there. That is exactly how the first attempt failed: the pin
    # predated the extractor and both jobs crash-looped on
    # "No such file or directory" after paying for a clone and a pip install.
    # Verified at BUILD time now, where it costs nothing.
    import subprocess
    needed = ["experiments/EVAL/extract_features.py",
              "configs/data/JetClassII_base.yaml",
              "experiments/MTX/ParT_sophon_arch_mtx.py",
              "experiments/E1/ParT_sophon_arch_10c.py"]
    for path in needed:
        r = subprocess.run(["git", "cat-file", "-e", f"{PIN}:{path}"],
                           cwd=ROOT, capture_output=True)
        if r.returncode != 0:
            sys.exit(f"FATAL: tag {PIN} does not contain {path}. The pod clones "
                     f"the TAG, so this job would fail after cloning. Tag a "
                     f"commit that has it, or fix PIN.")
    print(f"pin {PIN} verified to contain all {len(needed)} files the job runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, arm, k, ckpt in RUNS:
        fname, text = build(run_id, arm, k, ckpt, args.gpu, args.max_jets)
        d = yaml.safe_load(text)
        body = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        for must in (f"--num-classes {k}", f"--arm {arm}",
                     "configs/data/JetClassII_base.yaml",
                     "net_best_epoch_state.pt"):
            if must not in body:
                sys.exit(f"FATAL: {fname} missing {must!r}")
        res = d["spec"]["template"]["spec"]["containers"][0]["resources"]
        if args.gpu != ("nvidia.com/gpu" in res["limits"]):
            sys.exit(f"FATAL: {fname} gpu request does not match --gpu")
        (OUT_DIR / fname).write_text(text)
        print(f"  {fname}  arm={arm} K={k} "
              f"{'GPU' if args.gpu else 'CPU'} mem={res['limits']['memory']}"
              f"{f' max_jets={args.max_jets}' if args.max_jets else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
