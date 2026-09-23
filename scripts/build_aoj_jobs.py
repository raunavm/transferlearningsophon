#!/usr/bin/env python3
"""Emit the full real-data run: experiments/AOJ/k8s/job-aoj-full-s<i>-raunav.yaml
(ten shards) and job-aoj-full-fit-raunav.yaml (one merge-and-fit).

WHAT THIS RUNS AND WHY IN THIS SHAPE
------------------------------------
docs/PRESPEC_2026-09.md §6 fixes the real-data quantity: the top-quark signal
yield at 1 % DATA efficiency, from the simultaneous pass/fail fit of
experiments/AOJ/peak_fit.py, for every pretrained model, zero-shot. The
feasibility test (job-aoj-feasibility-v2cpu-raunav.yaml) ran it on 8 of the 80
files for two models and passed on the top channel; the W channel is withdrawn
(amendment 2026-09-19) and is neither scored nor fitted here.

  shard i (GPU)   RunG_batch{4i..4i+3} + RunH_batch{4i..4i+3}, downloaded and
                  staged in pod scratch, then EVERY model scored on them.
  fit (CPU)       the ten shards joined (merge_shards.py) and fitted ONCE, so the
                  decorrelation map is built from every jet.

SHARDED BY FILE, NOT BY MODEL, for two reasons. Staging 207 GB once instead of
31 times; and the models inside a shard run on one GPU over the same jets, so no
model-to-model difference comes from hardware -- the inference-time analogue of
invariant I7. A shard resumed on another node may change GPU part-way, so the
GPU is recorded PER MODEL (gpu_per_model.txt) rather than assumed.

NOT ON THE RTX 3090 POOL. The benchmark wave needs 3090s for its pairing
invariant and is Pending for them; inference has no such constraint, and us-west
has ~500 other GPUs (L4, V100, L40, 2080 Ti, 1080 Ti ...). The shards take any of
those, never a 3090.

RESUMABLE. Scores are written to the PVC model by model, and a restarted pod
skips every model whose scores are already there. discriminants.py then checks
the re-staged jets against the jets.npz the first attempt wrote, so a resumed
shard cannot silently mix two stagings. Staging is deterministic, so they match.

PERSISTED: jets.npz (~50 B/jet) and one float16 score per model per jet -- about
120-150 MB a shard, ~1.3 GB for the run. Raw HDF5, staged parquet, features and
logits live and die in scratch.

Run:  python3 scripts/build_aoj_jobs.py [--check-only] [--pin-not-yet-tagged]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
K8S = ROOT / "experiments" / "AOJ" / "k8s"
MTX_K8S = ROOT / "experiments" / "MTX" / "k8s"
FILES = ROOT / "configs" / "aoj" / "aspenopenjets_files.json"

PIN = "mtx-s1.55"
# The fit clones a later tag. mtx-s1.55's merge_shards.py required every
# (run, lumi, event) to be unique, but one event holds up to five jets, so the
# first fit job refused the real run 2,157,112 times (2026-09-23). The shards
# ran at mtx-s1.55 and their specs keep it: they are the record of what ran.
FIT_PIN = "mtx-s1.56"
FIT_NEEDED_FLAGS = {"experiments/AOJ/merge_shards.py": "occur in more than one shard",
                    "experiments/AOJ/peak_fit.py": "--peaks"}
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"
OUT_ROOT = "/data/results/aoj/full_v1"
N_SHARDS = 10
# Models scored at once on a shard's GPU. Measured on shard 0 (RTX 2080 Ti,
# 2026-09-23): ONE model runs at 788 jets/s with its single loader process at
# 100 % of a core and the GPU at 31 %. The loader must stay single-process for
# row alignment, so the parallelism is across models: three loaders, one GPU,
# ~3 GB of GPU memory each (fits the 11 GB of a 1080 Ti or 2080 Ti).
PARALLEL = 3
NEEDED_AT_PIN = [
    "experiments/AOJ/closure.py",
    "experiments/AOJ/discriminants.py",
    "experiments/AOJ/merge_shards.py",
    "experiments/AOJ/peak_fit.py",
    "experiments/EVAL/extract_features.py",
    "experiments/EVAL/anomaly.py",
    "scripts/build_usecase_survival.py",
    "scripts/build_aoj_config.py",
    "scripts/stage_aoj.py",
    "configs/finetune/AspenOpenJets.yaml",
    "configs/labelmaps/rung_label_maps.v1.csv",
]
# The flags this run passes that older tags do not have. Presence of the file at
# the pin is not enough -- that is the lesson of mtx-s1.48 (build_extract_jobs.py).
NEEDED_FLAGS = {"experiments/AOJ/discriminants.py": "--structures",
                "experiments/AOJ/peak_fit.py": "--peaks",
                "experiments/EVAL/extract_features.py": "--num-reg"}

# scripts/build_ft_jobs.py BAD_NODES + LOST_GPU_NODES, plus ry-gpu-10: "GPU is
# lost" (the same NVLink query failure as ry-gpu-01), measured 2026-09-23 when it
# refused all three attempts of the first shard-0 job at admission. It has also
# lost its gpu.product label, which is why the 3090 NotIn below did not keep the
# shard off it -- hence the Exists term as well.
BAD_NODES = ("ry-gpu-03.sdsc.optiputer.net", "nautilus-ext-gpu01.fullerton.edu",
             "hcc-chase-shor-c4705.unl.edu", "hcc-chase-shor-c4709.unl.edu",
             "k8s-chase-ci-07.calit2.optiputer.net", "nrp-fiona-001.sdmz.amnh.org",
             "ry-gpu-01.sdsc.optiputer.net", "ry-gpu-10.sdsc.optiputer.net",
             # k8s-haosu-15: shard 9's GPU failed mid-run after 19 models (CUDA abort in
             # extract_features, 2026-09-23 11:41Z), then both retries were refused at
             # admission with the same NVLink "GPU is lost" error.
             "k8s-haosu-15.sdsc.optiputer.net")

SOPHON_URL = "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
SOPHON_SHA256 = "cc7c33b522e796b5bbf0aa9bb5b01361c964f4ef3acebdd9682d7519c095b824"
REF_QCD = [f"/jc2/jet_data/QCD_{i:04d}.parquet" for i in range(350, 356)]
OBSERVERS = ("jet_pt jet_eta jet_sdmass aoj_jet_pt aoj_jet_eta "
             "aoj_pn_WvsQCD aoj_pn_TvsQCD aoj_pn_HbbvsQCD")


class Model(NamedTuple):
    name: str      # what the score is attributed to
    rung: str      # label-map column the head realises
    k: int         # class outputs
    num_reg: int   # regression outputs after them (1 for the mass-output models)
    spec: str      # training spec the head width is read from ("" for the public checkpoint)

    @property
    def arm(self) -> str:
        """Recorded in the extraction manifest; the mass models' names follow
        scripts/build_extract_jobs.py."""
        return "SOPHON_PUBLIC" if not self.spec else f"{self.rung}_MASS" if self.num_reg else self.rung

    @property
    def checkpoint(self) -> str:
        return (f"/data/results/mtx/mtx-{self.name}/net_epoch-79_state.pt"
                if self.spec else "/workspace/sophon_public.pt")


def _ladder(rung, k, seeds, stem):
    return [Model(f"{stem}-s{s}", rung, k, 0, f"job-mtx-{rung.lower()}-s{s}-raunav.yaml")
            for s in seeds]


# L162's five matrix seeds are s1b, s2..s5; mtx-l162-s1 trained at 1e-3 and is
# excluded everywhere (scripts/build_extract_jobs.py).
MODELS = [
    Model("sophon-public", "L188", 188, 0, ""),
    *_ladder("L188", 188, "12345", "l188"),
    *_ladder("L162", 162, ["1b", "2", "3", "4", "5"], "l162"),
    *_ladder("R42_Q1", 43, "12345", "r42q1"),
    *_ladder("R16_Q1", 17, "12345", "r16q1"),
    *[Model(f"l162mass-s{s}", "L162", 162, 1, f"job-mtx-l162_mass-s{s}-raunav.yaml") for s in "12345"],
    *[Model(f"r16q1mass-s{s}", "R16_Q1", 17, 1, f"job-mtx-r16_q1_mass-s{s}-raunav.yaml") for s in "12345"],
]


def verify_heads() -> None:
    """K from each model's own training spec, never assumed (R42_Q1 is 43)."""
    for m in MODELS:
        if not m.spec:
            continue
        text = (MTX_K8S / m.spec).read_text()
        ks = {int(x) for x in re.findall(r"(?:--num-classes|num_classes) (\d+)", text)}
        if ks != {m.k}:
            raise SystemExit(f"FATAL: {m.spec} says num_classes {ks}, this builder says {m.k}")
        if f"mtx-{m.name}" not in text:
            raise SystemExit(f"FATAL: {m.spec} does not train run mtx-{m.name}")


def shards() -> list[list[dict]]:
    files = json.loads(FILES.read_text())["files"]
    by_key = {f["key"]: f for f in files}
    if len(by_key) != 80:
        raise SystemExit(f"FATAL: {FILES} lists {len(by_key)} files, not 80")
    per = 80 // N_SHARDS // 2
    out = [[by_key[f"Run{r}_batch{b}.h5"] for r in "GH" for b in range(i * per, (i + 1) * per)]
           for i in range(N_SHARDS)]
    assert sorted(f["key"] for s in out for f in s) == sorted(by_key)
    return out


def verify_pin(pin: str, not_yet_tagged: bool, flags: dict | None = None) -> None:
    def read(path):
        if not_yet_tagged:
            p = ROOT / path
            return p.read_text() if p.exists() else None
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{pin}:{path}"],
                           capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    for path in NEEDED_AT_PIN:
        if read(path) is None:
            raise SystemExit(f"FATAL: {path} is not in {'the working tree' if not_yet_tagged else pin}")
    for path, flag in (NEEDED_FLAGS if flags is None else flags).items():
        if flag not in (read(path) or ""):
            raise SystemExit(f"FATAL: {path} at {pin} has no {flag}")


SHARD_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # FULL REAL-DATA RUN, SHARD {i} OF {n}. GENERATED by scripts/build_aoj_jobs.py --
  # edit the builder, not this file. Scores every pretrained model, zero-shot, on
  # {files_short} (CMS 2016 JetHT Open Data, AspenOpenJets, arXiv:2412.10504).
  # Top channel only; the W channel is withdrawn (PRESPEC amendment 2026-09-19).
  name: aoj-full-s{i}-raunav
  namespace: cms-ml
spec:
  # 2, not 1: the shard resumes model by model, so a retry after a lost node
  # repeats only the ~30 min of download and staging.
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: aoj
        image: {image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          OUT={out}/shard{i}
          RAW=/scratch/raw
          STAGED=/scratch/staged
          mkdir -p "${{OUT}}" "${{RAW}}" "${{STAGED}}"
          MODELS="{model_names}"

          USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
          FREE_G=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
          echo "PVC used: ${{USED}}%  free: ${{FREE_G}}G"
          [ "${{USED}}" -lt 95 ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 1; }}
          [ "${{FREE_G}}" -ge 5 ] || {{ echo "FATAL: ${{FREE_G}}G free on /data, need 5G"; exit 1; }}

          todo=""
          for m in ${{MODELS}}; do [ -f "${{OUT}}/scores_${{m}}.npz" ] || todo="${{todo}} ${{m}}"; done
          if [ -z "${{todo}}" ] && [ -f "${{OUT}}/closure.json" ]; then
            echo "shard {i} complete: every model scored"; touch "${{OUT}}/DONE"; exit 0
          fi
          echo "to score:${{todo}}"

          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          pip install --no-cache-dir -q pyarrow h5py || exit 1

          # ---- cheap preconditions BEFORE a ~20 GB download ---------------------
          for c in {checkpoints}; do [ -f "${{c}}" ] || {{ echo "FATAL: no ${{c}}"; exit 1; }}; done
          REF="{ref}"
          for f in ${{REF}}; do [ -f "${{f}}" ] || {{ echo "FATAL: no ${{f}}"; exit 1; }}; done
          curl -fsSL -o /workspace/sophon_public.pt {sophon_url}
          GOT=$(sha256sum /workspace/sophon_public.pt | cut -d' ' -f1)
          [ "${{GOT}}" = "{sophon_sha}" ] || {{ echo "FATAL: public checkpoint sha256 ${{GOT}}"; exit 1; }}
          CFG=configs/finetune/AspenOpenJets.yaml
          python3 scripts/build_aoj_config.py --check-only
          GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: ${{GPU}}"

          # ---- download + stage: two files in flight, raw deleted once staged ----
          BASE=https://www.fdr.uni-hamburg.de/record/16505/files
          fetch_and_stage () {{  # file md5
            curl -fsSL --retry 5 --retry-delay 20 -o "${{RAW}}/$1" "${{BASE}}/$1?download=1"
            echo "$2  ${{RAW}}/$1" | md5sum -c - || {{ echo "FATAL: md5 mismatch for $1"; return 1; }}
            PYTHONUNBUFFERED=1 python3 scripts/stage_aoj.py --in "${{RAW}}/$1" --out "${{STAGED}}"
            rm -f "${{RAW}}/$1"
          }}
          pair () {{
            fetch_and_stage "$1" "$2" & local a=$!
            fetch_and_stage "$3" "$4" & local b=$!
            wait ${{a}}; wait ${{b}}
          }}
{pairs}
          # never overwrite what an earlier attempt of this shard persisted
          mkdir -p "${{OUT}}/staging" && cp -n "${{STAGED}}"/*.stats.json "${{OUT}}/staging/"
          df -h /scratch | tail -1

          # THE ORDER OF THIS LIST IS LOAD-BEARING: extract_features.py reads it in
          # order and discriminants.py re-reads the same files to check row alignment.
          FILES="{staged}"

          [ -f "${{OUT}}/closure.json" ] || PYTHONUNBUFFERED=1 python3 experiments/AOJ/closure.py \
            --aoj ${{FILES}} --reference ${{REF}} --max-jets 50000 --out "${{OUT}}"

          # Every step is chained with &&, so score returns the status of the step
          # that failed even where set -e is suspended (a function run under ||).
          score () {{  # name checkpoint K num_reg arm rung
            [ -f "${{OUT}}/scores_$1.npz" ] && {{ echo "skip $1 (scored)"; return 0; }}
            PYTHONUNBUFFERED=1 python3 experiments/EVAL/extract_features.py \
              --checkpoint "$2" --num-classes "$3" --num-reg "$4" --arm "$5" \
              --data-config ${{CFG}} --observers {observers} --save-logits \
              --data-test ${{FILES}} --out "/scratch/extract/$1" \
              --batch-size 512 --num-workers 1 --fetch-step 1 \
            && python3 experiments/AOJ/discriminants.py --name "$1" --rung "$6" --structures three_prong \
              --extract-dir "/scratch/extract/$1" --staged ${{FILES}} --out "${{OUT}}" \
            && echo "$1 ${{GPU}}" >> "${{OUT}}/gpu_per_model.txt" \
            && rm -rf "/scratch/extract/$1"
          }}
          # One log per model, so {parallel} models at once do not interleave; on
          # failure its tail is printed and the shard stops (set -e on the wait).
          mkdir -p /scratch/logs
          scored () {{
            score "$@" > "/scratch/logs/$1.log" 2>&1 \
              || {{ echo "FATAL: $1 failed"; tail -40 "/scratch/logs/$1.log"; return 1; }}
            echo "$1: $(grep -E 'jets/s|^skip' /scratch/logs/$1.log | tail -1)"
          }}
          # The FIRST model runs alone: its discriminants.py call writes jets.npz,
          # which every later model is checked against, and two writers would race.
{scores}
          touch "${{OUT}}/DONE"
          ls -la "${{OUT}}"
        volumeMounts:
        - {{ name: jc2,     mountPath: /jc2, readOnly: true }}
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        - {{ name: dshm,    mountPath: /dev/shm }}
        resources:
          # 8 CPU: {parallel} loaders at 100 % of a core plus their main processes.
          # 48Gi: one model measured 6.6 GB loader + 1.5 GB main resident, times
          # {parallel}. Scratch: two raw files in flight (<= 10 GB), eight staged files,
          # and {parallel} models' logits and features (~1.5 GB each), deleted per model.
          requests: {{ memory: "48Gi", cpu: "8", nvidia.com/gpu: "1", ephemeral-storage: "48Gi" }}
          limits:   {{ memory: "48Gi", cpu: "8", nvidia.com/gpu: "1", ephemeral-storage: "48Gi" }}
      tolerations:
      - {{ key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
              # never the 3090 pool: the benchmark wave is Pending for it. NotIn
              # alone also matches a node whose GPU discovery failed and left no
              # product label (ry-gpu-10, 2026-09-23), so the label must exist.
              - key: nvidia.com/gpu.product
                operator: Exists
              - key: nvidia.com/gpu.product
                operator: NotIn
                values: ["NVIDIA-GeForce-RTX-3090"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{bad_nodes}]
      volumes:
      - name: jc2
        persistentVolumeClaim:
          claimName: tn-pvc-base-jetclass2
          readOnly: true
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: scratch
        emptyDir: {{ sizeLimit: "44Gi" }}
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""

FIT_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # FULL REAL-DATA RUN, THE FIT. GENERATED by scripts/build_aoj_jobs.py. Joins the
  # {n} shards (experiments/AOJ/merge_shards.py refuses a partial or inconsistent
  # set) and fits the top peak ONCE over all of them. Apply only after every
  # shard has written DONE; the precondition loop refuses otherwise.
  name: aoj-full-fit-raunav
  namespace: cms-ml
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: fit
        image: {image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          OUT={out}/fit
          [ ! -e "${{OUT}}/results.json" ] || {{ echo "FATAL: ${{OUT}}/results.json exists; use a new output directory"; exit 1; }}
          SHARDS=""
          for i in $(seq 0 {last}); do
            [ -f "{out}/shard${{i}}/DONE" ] || {{ echo "FATAL: shard ${{i}} has not finished"; exit 1; }}
            SHARDS="${{SHARDS}} {out}/shard${{i}}"
          done
          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          python3 experiments/AOJ/merge_shards.py --shards ${{SHARDS}} --out /scratch/merged
          SCORES=""
          for m in {model_names}; do SCORES="${{SCORES}} ${{m}}=/scratch/merged/scores_${{m}}.npz"; done
          mkdir -p "${{OUT}}"
          cp /scratch/merged/merge_manifest.json /scratch/merged/closure.json "${{OUT}}/"
          PYTHONUNBUFFERED=1 python3 experiments/AOJ/peak_fit.py --peaks top \
            --jets /scratch/merged/jets.npz --closure /scratch/merged/closure.json \
            --scores ${{SCORES}} --eff 0.01 --toys 200 --out "${{OUT}}"
          ls -la "${{OUT}}"
        volumeMounts:
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        resources:
          # ~12 M jets x (50 B + 31 float16 scores) held once in memory, then the fits.
          requests: {{ memory: "32Gi", cpu: "16", ephemeral-storage: "16Gi" }}
          limits:   {{ memory: "32Gi", cpu: "16", ephemeral-storage: "16Gi" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: scratch
        emptyDir: {{ sizeLimit: "12Gi" }}
"""


def render_shard(i: int, files: list[dict]) -> str:
    g = [f for f in files if f["key"].startswith("RunG")]
    h = [f for f in files if f["key"].startswith("RunH")]
    pairs = "\n".join(f"          pair {a['key']} {a['md5']} {b['key']} {b['md5']}" for a, b in zip(g, h))
    staged = " ".join(f"${{STAGED}}/{f['key'].removesuffix('.h5')}.parquet"
                      for a, b in zip(g, h) for f in (a, b))
    line = lambda m: f'scored {m.name} "{m.checkpoint}" {m.k} {m.num_reg} {m.arm} {m.rung}'
    first, rest = MODELS[0], MODELS[1:]
    scores = [f"          {line(first)}"]
    for start in range(0, len(rest), PARALLEL):
        group = rest[start:start + PARALLEL]
        scores += [f"          {line(m)} & p{k}=$!" for k, m in enumerate(group)]
        scores.append("          " + "; ".join(f"wait ${{p{k}}}" for k in range(len(group))))
    scores = "\n".join(scores)
    checkpoints = " ".join(m.checkpoint for m in MODELS if m.spec)
    return SHARD_TEMPLATE.format(
        i=i, n=N_SHARDS, image=IMAGE, pin=PIN, out=OUT_ROOT,
        files_short=f"{g[0]['key']}..{g[-1]['key']} and {h[0]['key']}..{h[-1]['key']}",
        model_names=" ".join(m.name for m in MODELS), checkpoints=checkpoints,
        ref=" ".join(REF_QCD), sophon_url=SOPHON_URL, sophon_sha=SOPHON_SHA256,
        pairs=pairs, staged=staged, observers=OBSERVERS, scores=scores, parallel=PARALLEL,
        bad_nodes=", ".join(f'"{b}"' for b in BAD_NODES))


def render_fit() -> str:
    return FIT_TEMPLATE.format(n=N_SHARDS, last=N_SHARDS - 1, image=IMAGE, pin=FIT_PIN, out=OUT_ROOT,
                               model_names=" ".join(m.name for m in MODELS))


def specs() -> dict[pathlib.Path, str]:
    out = {K8S / f"job-aoj-full-s{i}-raunav.yaml": render_shard(i, fs) for i, fs in enumerate(shards())}
    out[K8S / "job-aoj-full-fit-raunav.yaml"] = render_fit()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="verify, write nothing")
    ap.add_argument("--pin-not-yet-tagged", action="store_true",
                    help=f"check the working tree instead of {FIT_PIN}, which is tagged after the commit")
    a = ap.parse_args()
    verify_heads()
    verify_pin(PIN, False)                      # the shards already ran at it
    verify_pin(FIT_PIN, a.pin_not_yet_tagged, FIT_NEEDED_FLAGS)
    for path, text in specs().items():
        if a.check_only:
            print(f"ok   {path.relative_to(ROOT)}")
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
