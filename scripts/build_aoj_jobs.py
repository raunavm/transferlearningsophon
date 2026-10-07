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
      python3 scripts/build_aoj_jobs.py --v2 TIER [TIER ...] [--check-only] [--pin-not-yet-tagged]
"""
from __future__ import annotations

import argparse
import functools
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
# The fit-convergence check (2026-09-27): 21 of the 32 fits report L-BFGS-B
# success = false. It re-runs every fit from the same merged jets and reports
# whether each sits at its minimum; it writes a report and changes no result.
CHECK_PIN = "mtx-s1.59"
CHECK_NEEDED_FLAGS = {"experiments/AOJ/fit_convergence_check.py": "reproduces_stored_yield"}
# THE FIT AGAIN, WITH A MINIMISER THAT CONVERGES. The check at CHECK_PIN found 9 of the 32
# fits of the first run stopped at L-BFGS-B's iteration limit short of their minimum, and
# the transfer-factor order of every fit was chosen by F-tests run with the same
# minimiser. So the whole procedure is rerun -- order selection, fits, toys -- into a
# new directory, beside the first run, which stays as it is. Same shards, same merge.
FIT2_PIN = "mtx-s1.61"
FIT2_NEEDED_FLAGS = {"experiments/AOJ/peak_fit.py": "RESTART_TOL",
                     "experiments/AOJ/fit_convergence_check.py": "x, f = model.fit()"}
# THE BINNED COUNTS OF THE v2 FITS (2026-09-28). Its check found 16 of 32 fits with an
# estimated distance to the minimum above tolerance while no restart moved a yield by more
# than 0.09 sigma. The fits see the jets only through their (m_SD, pT) bins, so those are
# exported (checked against fit_v2/histograms.npz) and the fits examined from them.
BINS_PIN = "mtx-s1.62"
BINS_NEEDED_FLAGS = {"experiments/AOJ/export_fit_bins.py": "these are not the bins that run fitted"}
# THE FIT A THIRD TIME (v3, 2026-09-28). The bins showed the v2 fits short of their minimum
# (fit_v2_diagnostic/): the monomial basis left L-BFGS-B stalled and its finite-difference
# errors wrong. peak_fit.py now minimises and differentiates in an orthonormal basis of the
# same polynomial, and an order whose fit puts the transfer factor at zero is not admissible.
# The whole procedure reruns into fit_v3, then its own bins and diagnostic.
FIT3_PIN = "mtx-s1.63"
FIT3_NEEDED_FLAGS = {"experiments/AOJ/peak_fit.py": "n_at_floor",
                     "experiments/AOJ/fit_minimum_diagnostic.py": "def profile_error",
                     "experiments/AOJ/export_fit_bins.py": "these are not the bins that run fitted"}
# THE CHECKS OF THE REAL-DATA SECTION (2026-09-29, audit B5 / must-fix 9). Two GPU runs
# feed experiments/AOJ/realdata_checks.py, both scoring the SAME v1 checkpoints as the
# shards above:
#   rescore   the 80 files again, every model now also writing the prong-only score
#             (discriminants.SCORES) -- the test of "carried by prong structure" -- and
#             the closure now keeping quantile functions, so the ten shards pool exactly.
#             The three-prong score must come out bit-identical to the first run's.
#   sim       JetClass-II test files through the AspenOpenJets selection
#             (experiments/AOJ/sim_scores.py): QCD for the map's closure at 1 % and for
#             each model's background efficiency at its data cut, three-prong decays
#             for its signal efficiency there -- what separates model from domain.
# Both carry the pod failure policy of scripts/build_ft_jobs.py ("retries that survive
# a flaky cluster"): evictions are not counted, two failed attempts halt the job.
# Then one CPU job (render_checks) runs experiments/AOJ/realdata_checks.py over both.
# FOR THE v2 GRID the shards and the fits are built by --v2 (the v2 section at the end); its
# shards also write the prong-only score, so they serve the checks as both the first run and
# the rescore. The simulation and checks jobs are not built for it yet: they would change
# their output roots (SIM_ROOT, CHECKS_ROOT), their models and the main fit (MAIN_FIT2).
RESCORE_PIN = "mtx-s1.68"
RESCORE_NEEDED_FLAGS = {"experiments/AOJ/discriminants.py": "prong_only",
                        "experiments/AOJ/closure.py": "quantiles_aoj",
                        "experiments/AOJ/sim_scores.py": "scored on other jets"}
# The analysis job clones a later tag: realdata_checks.py is finished after the GPU
# runs were launched, and it reads only what they write.
# mtx-s1.73 had the forked worker pool that hung (realdata_checks._parallel); never launched.
CHECKS_PIN = "mtx-s1.74"
CHECKS_NEEDED_FLAGS = {"experiments/AOJ/realdata_checks.py": 'get_context("spawn")',
                       "experiments/AOJ/peak_fit.py": "data_efficiency_sidebands",
                       "experiments/FIGS/data/aoj_full_v1/fit_v4/results.json": "shape_variations"}
RESCORE_ROOT = "/data/results/aoj/full_v1_rescore"
SIM_ROOT = "/data/results/aoj/sim_v1"
SIM_CONFIG = "configs/finetune/JetClassII_base_selAspenOpenJets.yaml"
# JetClass-II TEST files only (scripts/build_extract_jobs.py FAMILIES: Res34P 1075-1289,
# QCD 350-419). All 70 QCD files: at 1 % the map's closure needs ~3,000 passing QCD jets
# in the top window for a 2 % error on the background shape, and a 2M-jet test cache
# holds 330. Eight Res34P files hold ~26,000 three-prong jets in the window.
SIM_FILES = ([f"/jc2/jet_data/Res34P_{i:04d}.parquet" for i in range(1075, 1083)]
             + [f"/jc2/jet_data/QCD_{i:04d}.parquet" for i in range(350, 420)])
N_SIM_JOBS = 6
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
# patternlab.calit2: every pod placed there ended in StartError, "failed to create
# containerd task: failed to create shim task: context canceled" (sim g0 and g3 and
# another agent's job, 2026-09-30 00:2xZ). Kept off the simulation jobs, which were
# re-created for it; the running rescore shards keep the list they were launched with.
SIM_BAD_NODES = BAD_NODES + ("patternlab.calit2.optiputer.net",)

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

    @property
    def run_id(self) -> str:
        return f"mtx-{self.name}"


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


def verify_heads(models: list | None = None) -> None:
    """K from each model's own training spec, never assumed (R42_Q1 is 43); for a v2 model
    also the mass output, from the spec's --mass-lambda."""
    for m in MODELS if models is None else models:
        if not m.spec:
            continue
        text = (MTX_K8S / m.spec).read_text()
        ks = {int(x) for x in re.findall(r"(?:--num-classes|num_classes) (\d+)", text)}
        if ks != {m.k}:
            raise SystemExit(f"FATAL: {m.spec} says num_classes {ks}, this builder says {m.k}")
        if m.run_id not in text:
            raise SystemExit(f"FATAL: {m.spec} does not train run {m.run_id}")
        if isinstance(m, V2Model) and ("--mass-lambda" in text) != bool(m.num_reg):
            raise SystemExit(f"FATAL: {m.spec} and this builder disagree on {m.run_id}'s mass output")


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


def render_shard(i: int, files: list[dict], models: list | None = None) -> str:
    models = MODELS if models is None else models
    g = [f for f in files if f["key"].startswith("RunG")]
    h = [f for f in files if f["key"].startswith("RunH")]
    pairs = "\n".join(f"          pair {a['key']} {a['md5']} {b['key']} {b['md5']}" for a, b in zip(g, h))
    staged = " ".join(f"${{STAGED}}/{f['key'].removesuffix('.h5')}.parquet"
                      for a, b in zip(g, h) for f in (a, b))
    line = lambda m: f'scored {m.name} "{m.checkpoint}" {m.k} {m.num_reg} {m.arm} {m.rung}'
    first, rest = models[0], models[1:]
    scores = [f"          {line(first)}"]
    for start in range(0, len(rest), PARALLEL):
        group = rest[start:start + PARALLEL]
        scores += [f"          {line(m)} & p{k}=$!" for k, m in enumerate(group)]
        scores.append("          " + "; ".join(f"wait ${{p{k}}}" for k in range(len(group))))
    scores = "\n".join(scores)
    checkpoints = " ".join(m.checkpoint for m in models if m.spec)
    return SHARD_TEMPLATE.format(
        i=i, n=N_SHARDS, image=IMAGE, pin=PIN, out=OUT_ROOT,
        files_short=f"{g[0]['key']}..{g[-1]['key']} and {h[0]['key']}..{h[-1]['key']}",
        model_names=" ".join(m.name for m in models), checkpoints=checkpoints,
        ref=" ".join(REF_QCD), sophon_url=SOPHON_URL, sophon_sha=SOPHON_SHA256,
        pairs=pairs, staged=staged, observers=OBSERVERS, scores=scores, parallel=PARALLEL,
        bad_nodes=", ".join(f'"{b}"' for b in BAD_NODES))


def render_fit() -> str:
    return FIT_TEMPLATE.format(n=N_SHARDS, last=N_SHARDS - 1, image=IMAGE, pin=FIT_PIN, out=OUT_ROOT,
                               model_names=" ".join(m.name for m in MODELS))


def render_fit_check() -> str:
    """The fit job with the fit replaced by experiments/AOJ/fit_convergence_check.py.
    Derived by substitution, each asserted, so the merge step is the one the fit ran."""
    t = render_fit()
    subs = [
        ("  # FULL REAL-DATA RUN, THE FIT. GENERATED by scripts/build_aoj_jobs.py. Joins the\n"
         f"  # {N_SHARDS} shards (experiments/AOJ/merge_shards.py refuses a partial or inconsistent\n"
         "  # set) and fits the top peak ONCE over all of them. Apply only after every\n"
         "  # shard has written DONE; the precondition loop refuses otherwise.\n",
         "  # FULL REAL-DATA RUN, FIT-CONVERGENCE CHECK. GENERATED by scripts/build_aoj_jobs.py.\n"
         "  # Re-merges the shards exactly as the fit did and re-runs every top fit from\n"
         "  # fit/results.json, reporting whether each sits at its minimum. Writes one report;\n"
         "  # changes no stored result.\n"),
        ("name: aoj-full-fit-raunav", "name: aoj-full-fitcheck-raunav"),
        (f"OUT={OUT_ROOT}/fit\n", f"OUT={OUT_ROOT}/fit_convergence_check\n"
                                  f"          [ -f {OUT_ROOT}/fit/results.json ] || {{ echo \"FATAL: no fit to check\"; exit 1; }}\n"),
        ('[ ! -e "${OUT}/results.json" ] || { echo "FATAL: ${OUT}/results.json exists',
         '[ ! -e "${OUT}/check.json" ] || { echo "FATAL: ${OUT}/check.json exists'),
        (f'--branch "{FIT_PIN}"', f'--branch "{CHECK_PIN}"'),
        ('          cp /scratch/merged/merge_manifest.json /scratch/merged/closure.json "${OUT}/"\n', ''),
        ("          PYTHONUNBUFFERED=1 python3 experiments/AOJ/peak_fit.py --peaks top \\\n"
         "            --jets /scratch/merged/jets.npz --closure /scratch/merged/closure.json \\\n"
         '            --scores ${SCORES} --eff 0.01 --toys 200 --out "${OUT}"\n',
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/fit_convergence_check.py --peak top \\\n"
         f"            --jets /scratch/merged/jets.npz --results {OUT_ROOT}/fit/results.json \\\n"
         '            --scores ${SCORES} --out "${OUT}/check.json"\n'),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the fit template changed; cannot derive the check from it ({a[:50]!r})")
        t = t.replace(a, b)
    return t


def render_fit_v2() -> str:
    """The fit job at FIT2_PIN into fit_v2, followed by the convergence check of its own
    results. Derived by substitution, each asserted, so everything else is the first run's."""
    t = render_fit()
    subs = [
        ("  # FULL REAL-DATA RUN, THE FIT. GENERATED by scripts/build_aoj_jobs.py. Joins the\n",
         "  # FULL REAL-DATA RUN, THE FIT AGAIN (v2), with the restarting minimiser of\n"
         "  # peak_fit.py at FIT2_PIN. GENERATED by scripts/build_aoj_jobs.py. Joins the\n"),
        ("name: aoj-full-fit-raunav", "name: aoj-full-fit-v2-raunav"),
        (f"OUT={OUT_ROOT}/fit\n", f"OUT={OUT_ROOT}/fit_v2\n"),
        (f'--branch "{FIT_PIN}"', f'--branch "{FIT2_PIN}"'),
        ('          ls -la "${OUT}"\n',
         '          ls -la "${OUT}"\n'
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/fit_convergence_check.py --peak top \\\n"
         '            --jets /scratch/merged/jets.npz --results "${OUT}/results.json" \\\n'
         '            --scores ${SCORES} --out "${OUT}/check.json"\n'),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the fit template changed; cannot derive v2 from it ({a[:50]!r})")
        t = t.replace(a, b)
    return t


def render_fit_bins() -> str:
    """The fit job with the fit replaced by experiments/AOJ/export_fit_bins.py on the v2
    results. Derived by substitution, each asserted, so the merge step is the one the fit ran."""
    t = render_fit()
    subs = [
        ("  # FULL REAL-DATA RUN, THE FIT. GENERATED by scripts/build_aoj_jobs.py. Joins the\n"
         f"  # {N_SHARDS} shards (experiments/AOJ/merge_shards.py refuses a partial or inconsistent\n"
         "  # set) and fits the top peak ONCE over all of them. Apply only after every\n"
         "  # shard has written DONE; the precondition loop refuses otherwise.\n",
         "  # FULL REAL-DATA RUN, THE BINNED COUNTS OF THE v2 FITS. GENERATED by\n"
         "  # scripts/build_aoj_jobs.py. Re-merges the shards exactly as the fit did and writes\n"
         "  # the (m_SD, pT) pass/fail bins of every top fit, checked against fit_v2's\n"
         "  # histograms. Fits nothing; changes no stored result.\n"),
        ("name: aoj-full-fit-raunav", "name: aoj-full-fitbins-raunav"),
        (f"OUT={OUT_ROOT}/fit\n", f"OUT={OUT_ROOT}/fit_v2_bins\n"
                                  f"          [ -f {OUT_ROOT}/fit_v2/results.json ] || {{ echo \"FATAL: no v2 fit\"; exit 1; }}\n"),
        ('[ ! -e "${OUT}/results.json" ] || { echo "FATAL: ${OUT}/results.json exists',
         '[ ! -e "${OUT}/bins.npz" ] || { echo "FATAL: ${OUT}/bins.npz exists'),
        (f'--branch "{FIT_PIN}"', f'--branch "{BINS_PIN}"'),
        ('          cp /scratch/merged/merge_manifest.json /scratch/merged/closure.json "${OUT}/"\n', ''),
        ("          PYTHONUNBUFFERED=1 python3 experiments/AOJ/peak_fit.py --peaks top \\\n"
         "            --jets /scratch/merged/jets.npz --closure /scratch/merged/closure.json \\\n"
         '            --scores ${SCORES} --eff 0.01 --toys 200 --out "${OUT}"\n',
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/export_fit_bins.py --peak top \\\n"
         f"            --jets /scratch/merged/jets.npz --results {OUT_ROOT}/fit_v2/results.json \\\n"
         f"            --histograms {OUT_ROOT}/fit_v2/histograms.npz \\\n"
         '            --scores ${SCORES} --out "${OUT}/bins.npz"\n'),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the fit template changed; cannot derive the export from it ({a[:50]!r})")
        t = t.replace(a, b)
    return t


def render_fit_v3() -> str:
    """The fit job at FIT3_PIN into fit_v3, then its convergence check, its bins (checked
    against its own histograms) and the minimum diagnostic on them. Derived by
    substitution, each asserted, so the merge and the fit are the first run's."""
    t = render_fit()
    subs = [
        ("  # FULL REAL-DATA RUN, THE FIT. GENERATED by scripts/build_aoj_jobs.py. Joins the\n",
         "  # FULL REAL-DATA RUN, THE FIT A THIRD TIME (v3), minimised in an orthonormal basis\n"
         "  # (peak_fit.py at FIT3_PIN). GENERATED by scripts/build_aoj_jobs.py. Joins the\n"),
        ("name: aoj-full-fit-raunav", "name: aoj-full-fit-v3-raunav"),
        (f"OUT={OUT_ROOT}/fit\n", f"OUT={OUT_ROOT}/fit_v3\n"),
        (f'--branch "{FIT_PIN}"', f'--branch "{FIT3_PIN}"'),
        ('          ls -la "${OUT}"\n',
         '          ls -la "${OUT}"\n'
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/fit_convergence_check.py --peak top \\\n"
         '            --jets /scratch/merged/jets.npz --results "${OUT}/results.json" \\\n'
         '            --scores ${SCORES} --out "${OUT}/check.json"\n'
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/export_fit_bins.py --peak top \\\n"
         '            --jets /scratch/merged/jets.npz --results "${OUT}/results.json" \\\n'
         '            --histograms "${OUT}/histograms.npz" --scores ${SCORES} --out "${OUT}/bins.npz"\n'
         "          PYTHONUNBUFFERED=1 python3 experiments/AOJ/fit_minimum_diagnostic.py \\\n"
         '            --bins "${OUT}/bins.npz" --results "${OUT}/results.json" \\\n'
         '            --out "${OUT}/diagnostic.json" --workers 15\n'),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the fit template changed; cannot derive v3 from it ({a[:50]!r})")
        t = t.replace(a, b)
    return t


def _ft():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_ft_jobs", ROOT / "scripts" / "build_ft_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _failure_accounting(out: str, halt: int) -> str:
    """Bash that halts the job (exit `halt`, which the pod failure policy turns into
    FailJob) after two FAILED attempts or eight in all. A failed attempt is recorded by
    the EXIT trap; an evicted pod is SIGKILLed, runs no trap and is not counted."""
    return (f'          mkdir -p "{out}"\n'
            f'          echo "$(date -u +%FT%TZ) $(hostname)" >> "{out}/ATTEMPTS"\n'
            # grep -c, not cat | wc -l: under pipefail a missing file fails the pipeline
            f'          NF=$(grep -c . "{out}/FAILED_ATTEMPTS" 2>/dev/null || true)\n'
            f'          NA=$(grep -c . "{out}/ATTEMPTS" 2>/dev/null || true)\n'
            f'          if [ "${{NF:-0}}" -ge 2 ] || [ "${{NA:-0}}" -gt 8 ]; then\n'
            f'            echo "FATAL: ${{NF}} failed attempts, ${{NA}} in all: fix the cause, then remove '
            f'{out}/FAILED_ATTEMPTS"; exit {halt}; fi\n'
            f"          trap 'rc=$?; if [ ${{rc}} -ne 0 ] && [ ${{rc}} -ne {halt} ]; then "
            f'echo "rc=${{rc}} $(date -u +%FT%TZ) $(hostname)" >> "{out}/FAILED_ATTEMPTS"; fi\' EXIT\n')


def _robust(text: str, backoff_old: str) -> str:
    """The pod failure policy and ROBUST_BACKOFF, container renamed `main` (the policy
    names it), by asserted substitution."""
    ft = _ft()
    subs = [(backoff_old, f"  backoffLimit: {ft.ROBUST_BACKOFF}\n{ft.POD_FAILURE_POLICY}"),
            ("      - name: aoj\n", "      - name: main\n")]
    for a, b in subs:
        if text.count(a) != 1:
            raise SystemExit(f"FATAL: template changed; cannot add the retry policy ({a[:40]!r})")
        text = text.replace(a, b)
    return text


# Rescore shards re-created off patternlab (SIM_BAD_NODES): shard 8 was placed there three
# times in a row, StartError each time (2026-09-30 01:xxZ). The others run on the list they
# were launched with, which their committed specs record.
# Its re-created pod then landed on a GTX 1080: three models at ~3 GB each (a batch-norm
# step asks for 1 GB more) do not fit 8 GB, and l188-s2 died of CUDA out-of-memory
# (2026-09-30 04:3xZ). The builder's own sizing (PARALLEL) assumed 11 GB, so a re-created
# shard also requires it, from the node's nvidia.com/gpu.memory label (MiB).
RESCORE_RECREATED = {8}
MIN_GPU_MEMORY_MIB = 10000


def render_rescore_shard(i: int, files: list[dict]) -> str:
    """Shard i of the first run, again, at RESCORE_PIN into RESCORE_ROOT, every model also
    writing the prong-only score, with the retry policy. Derived by substitution, each
    asserted, so staging, row alignment and scoring are the first run's line for line."""
    ft = _ft()
    t = render_shard(i, files)
    subs = [
        ("  # FULL REAL-DATA RUN, SHARD", "  # REAL-DATA CHECKS: THE FIRST RUN RESCORED (+ prong-only score), SHARD"),
        (f"name: aoj-full-s{i}-raunav", f"name: aoj-rescore-s{i}-raunav"),
        (f"          OUT={OUT_ROOT}/shard{i}\n",
         f"          OUT={RESCORE_ROOT}/shard{i}\n" + _failure_accounting("${OUT}", ft.EXIT_HALT)),
        (f'--branch "{PIN}"', f'--branch "{RESCORE_PIN}"'),
        ("--structures three_prong ", "--structures three_prong prong_only "),
    ]
    if i in RESCORE_RECREATED:
        subs.append(("values: [" + ", ".join(f'"{b}"' for b in BAD_NODES) + "]",
                     "values: [" + ", ".join(f'"{b}"' for b in SIM_BAD_NODES) + "]"))
        subs.append(("              - key: nvidia.com/gpu.product\n                operator: Exists\n",
                     "              - key: nvidia.com/gpu.product\n                operator: Exists\n"
                     "              - key: nvidia.com/gpu.memory\n                operator: Gt\n"
                     f'                values: ["{MIN_GPU_MEMORY_MIB}"]\n'))
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the shard template changed; cannot derive the rescore ({a[:50]!r})")
        t = t.replace(a, b)
    return _robust(t, "  # 2, not 1: the shard resumes model by model, so a retry after a lost node\n"
                      "  # repeats only the ~30 min of download and staging.\n  backoffLimit: 2\n")


# SHARD 8 IN THREE PARTS (2026-09-30 07:1xZ). Its third pod only found an 11 GB GPU after
# two hours Pending, and alone it needed ~7 h more, twice any other shard. So the shard's
# remaining models are split across RESCORE_PARTS jobs, each staging the same eight files
# and scoring a DISJOINT third of MODELS[1:] into the same directory (no two jobs ever
# score one model, so no file has two writers). jets.npz and closure.json were already
# written by the shard's first attempt and are required, not written; DONE is touched only
# when every one of the 31 models is scored, by whichever part finishes last.
RESCORE_PARTS = {8: 3}


def rescore_part_models(k: int, n_parts: int) -> list:
    return MODELS[1:][k::n_parts]


def render_rescore_part(i: int, files: list[dict], k: int, n_parts: int) -> str:
    ft = _ft()
    t = render_rescore_shard(i, files)
    part = rescore_part_models(k, n_parts)
    line = lambda m: f'scored {m.name} "{m.checkpoint}" {m.k} {m.num_reg} {m.arm} {m.rung}'
    block = []
    for start in range(0, len(part), PARALLEL):
        group = part[start:start + PARALLEL]
        block += [f"          {line(m)} & p{j}=$!" for j, m in enumerate(group)]
        block.append("          " + "; ".join(f"wait ${{p{j}}}" for j in range(len(group))))
    first = MODELS[0]
    old_block_start = f"          {line(first)}\n"
    s0 = t.index(old_block_start)
    s1 = t.index('          touch "${OUT}/DONE"\n          ls -la "${OUT}"\n')
    t = t[:s0] + "\n".join(block) + "\n" + t[s1:]
    all_names = " ".join(m.name for m in MODELS)
    subs = [
        ("  # REAL-DATA CHECKS: THE FIRST RUN RESCORED (+ prong-only score), SHARD",
         f"  # REAL-DATA CHECKS: SHARD {i}, PART {k} OF {n_parts} (RESCORE_PARTS). THE FIRST RUN RESCORED, SHARD"),
        (f"name: aoj-rescore-s{i}-raunav", f"name: aoj-rescore-s{i}-p{k}-raunav"),
        (_failure_accounting("${OUT}", ft.EXIT_HALT), _failure_accounting(f"${{OUT}}/attempts_p{k}", ft.EXIT_HALT)
         + '          [ -f "${OUT}/jets.npz" ] && [ -f "${OUT}/closure.json" ] || '
           '{ echo "FATAL: a part needs the jets.npz and closure.json of the shard\'s first attempt"; exit '
         + f"{ft.EXIT_HALT}; }}\n"
         + f'          ALL_MODELS="{all_names}"\n'
         + '          done_if_all () { for m in ${ALL_MODELS}; do [ -f "${OUT}/scores_${m}.npz" ] || return 0; done; '
           'touch "${OUT}/DONE"; }\n'),
        (f'          MODELS="{all_names}"\n', f'          MODELS="{" ".join(m.name for m in part)}"\n'),
        (f'echo "shard {i} complete: every model scored"; touch "${{OUT}}/DONE"; exit 0',
         f'echo "shard {i} part {k} complete"; done_if_all; exit 0'),
        ('          touch "${OUT}/DONE"\n          ls -la "${OUT}"\n', '          done_if_all\n          ls -la "${OUT}"\n'),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the rescore template changed; cannot derive the part ({a[:60]!r})")
        t = t.replace(a, b)
    return t


SIM_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # REAL-DATA CHECKS: THE SCORES ON SIMULATION, GROUP {g} OF {n}. GENERATED by
  # scripts/build_aoj_jobs.py. Scores {models_short} with the real-data scores
  # (experiments/AOJ/sim_scores.py) on {n_files} JetClass-II test files through the
  # AspenOpenJets selection. One loader worker and one file list for every model and
  # group, so every model's rows are the same jets; sim_scores.py refuses otherwise.
  name: aoj-sim-g{g}-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
          ACC="${{OUT}}/attempts/g{g}"
{accounting}          MODELS="{model_names}"
          todo=""
          for m in ${{MODELS}}; do [ -f "${{OUT}}/scores_${{m}}.npz" ] || todo="${{todo}} ${{m}}"; done
          if [ -z "${{todo}}" ]; then echo "group {g} complete"; touch "${{ACC}}/DONE"; exit 0; fi
          echo "to score:${{todo}}"

          USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
          FREE_G=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
          echo "PVC used: ${{USED}}%  free: ${{FREE_G}}G"
          [ "${{USED}}" -lt 85 ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 1; }}
          [ "${{FREE_G}}" -ge 5 ] || {{ echo "FATAL: ${{FREE_G}}G free on /data, need 5G"; exit 1; }}

          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          # THE IMAGE HAS NO pyarrow, and weaver reads parquet through it. Without it weaver
          # logs the ImportError, swallows it and fails later with "Zero entries loaded"
          # (every group of the first launch, 2026-09-29) -- so install, then import.
          pip install --no-cache-dir -q pyarrow || exit 1
          python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
          for c in {checkpoints}; do [ -f "${{c}}" ] || {{ echo "FATAL: no ${{c}}"; exit 1; }}; done
{sophon}          # THE ORDER OF THIS LIST IS LOAD-BEARING: every model reads it in this order.
          FILES="{files}"
          for f in ${{FILES}}; do [ -f "${{f}}" ] || {{ echo "FATAL: no ${{f}}"; exit 1; }}; done
          GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: ${{GPU}}"

          score () {{  # name checkpoint K num_reg arm rung
            [ -f "${{OUT}}/scores_$1.npz" ] && {{ echo "skip $1 (scored)"; return 0; }}
            PYTHONUNBUFFERED=1 python3 experiments/EVAL/extract_features.py \
              --checkpoint "$2" --num-classes "$3" --num-reg "$4" --arm "$5" \
              --data-config {config} --observers jet_pt jet_eta jet_sdmass --save-logits \
              --data-test ${{FILES}} --out "/scratch/extract/$1" \
              --batch-size 512 --num-workers 1 --fetch-step 1 \
            && python3 experiments/AOJ/sim_scores.py --name "$1" --rung "$6" \
              --extract-dir "/scratch/extract/$1" --out "${{OUT}}" \
            && echo "$1 ${{GPU}}" >> "${{OUT}}/gpu_per_model.txt" \
            && rm -rf "/scratch/extract/$1"
          }}
          mkdir -p /scratch/logs
          scored () {{
            score "$@" > "/scratch/logs/$1.log" 2>&1 \
              || {{ echo "FATAL: $1 failed"; tail -40 "/scratch/logs/$1.log"; return 1; }}
            echo "$1: $(grep -E 'jets/s|^skip' /scratch/logs/$1.log | tail -1)"
          }}
{scores}
          touch "${{ACC}}/DONE"
        volumeMounts:
        - {{ name: jc2,     mountPath: /jc2, readOnly: true }}
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        - {{ name: dshm,    mountPath: /dev/shm }}
        resources:
          # as the shards: {parallel} single-worker loaders at a core each. Each model holds
          # ~2.4 M jets of features and logits (~3 GB) before writing them.
          requests: {{ memory: "48Gi", cpu: "8", nvidia.com/gpu: "1", ephemeral-storage: "40Gi" }}
          limits:   {{ memory: "48Gi", cpu: "8", nvidia.com/gpu: "1", ephemeral-storage: "40Gi" }}
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
        emptyDir: {{ sizeLimit: "36Gi" }}
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


def sim_groups() -> list[list[Model]]:
    """MODELS dealt round-robin into N_SIM_JOBS groups; the published checkpoint is in
    group 0."""
    return [MODELS[g::N_SIM_JOBS] for g in range(N_SIM_JOBS)]


def render_sim(g: int, models: list[Model]) -> str:
    ft = _ft()
    line = lambda m: f'scored {m.name} "{m.checkpoint}" {m.k} {m.num_reg} {m.arm} {m.rung}'
    scores = []
    for start in range(0, len(models), PARALLEL):
        group = models[start:start + PARALLEL]
        scores += [f"          {line(m)} & p{k}=$!" for k, m in enumerate(group)]
        scores.append("          " + "; ".join(f"wait ${{p{k}}}" for k in range(len(group))))
    sophon = ""
    if any(not m.spec for m in models):
        sophon = (f"          curl -fsSL -o /workspace/sophon_public.pt {SOPHON_URL}\n"
                  "          GOT=$(sha256sum /workspace/sophon_public.pt | cut -d' ' -f1)\n"
                  f'          [ "${{GOT}}" = "{SOPHON_SHA256}" ] || {{ echo "FATAL: public checkpoint sha256 ${{GOT}}"; exit 1; }}\n')
    t = SIM_TEMPLATE.format(
        g=g, n=N_SIM_JOBS, image=IMAGE, pin=RESCORE_PIN, out=SIM_ROOT, n_files=len(SIM_FILES),
        models_short=", ".join(m.name for m in models),
        accounting=_failure_accounting("${ACC}", ft.EXIT_HALT),
        model_names=" ".join(m.name for m in models),
        checkpoints=" ".join(m.checkpoint for m in models if m.spec) or "",
        sophon=sophon, files=" ".join(SIM_FILES), config=SIM_CONFIG, scores="\n".join(scores),
        parallel=PARALLEL, bad_nodes=", ".join(f'"{b}"' for b in SIM_BAD_NODES))
    return _robust(t, "  backoffLimit: 2\n")


CHECKS_ROOT = "/data/results/aoj/checks_v1"
# the main fit the checks are read against: the first run's, refitted from its bins with
# the shape floating (experiments/AOJ/refit_from_bins.py); it exists only in the repository
MAIN_FIT = "experiments/FIGS/data/aoj_full_v1/fit_v4/results.json"

CHECKS_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # REAL-DATA CHECKS: THE ANALYSIS. GENERATED by scripts/build_aoj_jobs.py. Joins the
  # rescored shards (merge_shards.py) and runs experiments/AOJ/realdata_checks.py over
  # them, the simulation scores and the main fit: every check of the real-data
  # section, one JSON each, into {out}. Refuses to start before every rescore shard and
  # simulation group has finished.
  name: aoj-checks-v1-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
{accounting}          [ ! -e "${{OUT}}/prong_test.json" ] || {{ echo "done: ${{OUT}}/prong_test.json exists"; exit 0; }}
          SHARDS=""; FIRST=""
          for i in $(seq 0 {last}); do
            [ -f "{rescore}/shard${{i}}/DONE" ] || {{ echo "FATAL: rescore shard ${{i}} has not finished"; exit {halt}; }}
            SHARDS="${{SHARDS}} {rescore}/shard${{i}}"; FIRST="${{FIRST}} {first}/shard${{i}}"
          done
          for g in $(seq 0 {last_sim}); do
            [ -f "{sim}/attempts/g${{g}}/DONE" ] || {{ echo "FATAL: simulation group ${{g}} has not finished"; exit {halt}; }}
          done
          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          # the checks import the staging code for its selection, and its imports need
          # pyarrow, which the image lacks (the first launch died on it, 2026-09-30)
          pip install --no-cache-dir -q pyarrow || exit 1
          python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
          python3 experiments/AOJ/merge_shards.py --shards ${{SHARDS}} --out /scratch/merged
          # the first run, merged exactly as its fit merged it: its three-prong scores are
          # the ones the main fit saw, and every check reads those (realdata_checks.load_data)
          python3 experiments/AOJ/merge_shards.py --shards ${{FIRST}} --out /scratch/merged_first
          # ONE BLAS THREAD PER PROCESS. The parallelism is fifteen worker processes; the
          # image's OpenBLAS otherwise starts a thread per host CPU (256 here, 48 live per
          # worker), and the first run of the injection step spun for 2 h on 16 CPUs doing
          # what takes 11 s single-threaded (2026-09-30).
          export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
          PYTHONUNBUFFERED=1 python3 experiments/AOJ/realdata_checks.py \
            --merged /scratch/merged --shards ${{SHARDS}} --first-run-shards ${{FIRST}} \
            --first-run-merged /scratch/merged_first \
            --fit {main_fit} --sim {sim} --out "${{OUT}}" --workers 15 --toys 200
          cp /scratch/merged/merge_manifest.json "${{OUT}}/"
          ls -la "${{OUT}}"
        volumeMounts:
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        resources:
          # ~12.6 M jets x 31 models x two float16 scores, and the simulation, shared by
          # fifteen forked workers; each worker copies a model's scores to float64.
          requests: {{ memory: "48Gi", cpu: "16", ephemeral-storage: "16Gi" }}
          limits:   {{ memory: "48Gi", cpu: "16", ephemeral-storage: "16Gi" }}
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


def render_checks() -> str:
    ft = _ft()
    t = CHECKS_TEMPLATE.format(
        image=IMAGE, out=CHECKS_ROOT, accounting=_failure_accounting("${OUT}", ft.EXIT_HALT),
        last=N_SHARDS - 1, last_sim=N_SIM_JOBS - 1, rescore=RESCORE_ROOT, first=OUT_ROOT, sim=SIM_ROOT,
        halt=ft.EXIT_HALT, pin=CHECKS_PIN, main_fit=MAIN_FIT)
    return _robust(t, "  backoffLimit: 2\n")


# THE CHECKS' OUTPUT, READ BACK. The repository keeps a copy of what the checks job wrote
# (experiments/FIGS/data/aoj_checks_v1/). With no pod of ours left mounting the volume, a
# one-line job prints it: the JSON files as a base64 tar on its log, decoded locally. It
# writes nothing and mounts the volume read-only.
READ_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # REAL-DATA CHECKS: READ BACK {out}. GENERATED by scripts/build_aoj_jobs.py. Prints the
  # checks' JSON files as a base64 tar between markers on the log; writes nothing.
  name: aoj-read-checks-v1-raunav
  namespace: cms-ml
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          [ -f "{out}/prong_test.json" ] || {{ echo "FATAL: the checks have not finished"; exit 1; }}
          cd "{out}"
          echo "BEGIN-TAR"
          tar czf - *.json | base64 -w 0
          echo
          echo "END-TAR"
        volumeMounts:
        - {{ name: data, mountPath: /data, readOnly: true }}
        resources:
          requests: {{ memory: "1Gi", cpu: "1" }}
          limits:   {{ memory: "1Gi", cpu: "1" }}
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
          readOnly: true
"""


def render_read() -> str:
    return READ_TEMPLATE.format(image=IMAGE, out=CHECKS_ROOT)


# THE INJECTION TEST'S BINS (2026-10-01). The checks' injection into the pseudo-window
# recovered 0.72 of a known signal. Its cause is measured from bins: the pseudo-window
# bins and the top bins at two more working points, exported once from the first run's
# merged jets (experiments/AOJ/injection_test.py bins, which checks its 1 % top bins
# against the committed ones), printed on the log as a base64 tar for the repository.
INJECTION_ROOT = "/data/results/aoj/injection_v1"
# First launched at mtx-s1.79: it refused its own bins, every count equal to the committed
# ones but the per-bin mean rho not bit-equal (a weighted sum). Re-created at mtx-s1.81,
# which compares the means to FLOAT_RTOL, with a fresh attempts directory.
INJECTION_PIN = "mtx-s1.81"
INJECTION_NEEDED_FLAGS = {"experiments/AOJ/injection_test.py": "FLOAT_RTOL"}
COMMITTED_BINS = "experiments/FIGS/data/aoj_full_v1/fit_v3/bins.npz"

INJECTION_BINS_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # INJECTION TEST: THE BINS. GENERATED by scripts/build_aoj_jobs.py. Merges the first
  # run's shards exactly as the fit did and writes, per score, the pseudo-window bins at
  # 1 % and the top bins at the extra working points (experiments/AOJ/injection_test.py
  # bins), then prints them on the log as a base64 tar. Fits nothing.
  name: aoj-injection-bins-v1-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
{accounting}          SHARDS=""
          for i in $(seq 0 {last}); do
            [ -f "{first}/shard${{i}}/DONE" ] || {{ echo "FATAL: shard ${{i}} has not finished"; exit {halt}; }}
            SHARDS="${{SHARDS}} {first}/shard${{i}}"
          done
          if [ ! -e "${{OUT}}/injection_bins.npz" ]; then
            git clone --depth 1 --branch "{pin}" \
              https://github.com/raunavm/transferlearningsophon.git \
              /workspace/transferlearningsophon
            cd /workspace/transferlearningsophon
            git rev-parse HEAD
            # injection_test.py reads the pseudo window from the checks' code, which imports
            # the staging code, whose imports need pyarrow (absent from the image)
            pip install --no-cache-dir -q pyarrow || exit 1
            python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
            python3 experiments/AOJ/merge_shards.py --shards ${{SHARDS}} --out /scratch/merged
            export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
            PYTHONUNBUFFERED=1 python3 experiments/AOJ/injection_test.py bins --merged /scratch/merged \
              --committed {committed} --out /scratch/injection_bins.npz --workers 8
            cp /scratch/injection_bins.npz "${{OUT}}/injection_bins.npz"
          fi
          cd "${{OUT}}"
          sha256sum injection_bins.npz
          echo "BEGIN-TAR"
          tar czf - injection_bins.npz | base64 -w 0
          echo
          echo "END-TAR"
        volumeMounts:
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        resources:
          # ~12.6 M jets and 31 float16 scores merged once; eight spawned workers each hold
          # the ~9.4 M jets of the rho window and one model's score in float64.
          requests: {{ memory: "32Gi", cpu: "8", ephemeral-storage: "16Gi" }}
          limits:   {{ memory: "32Gi", cpu: "8", ephemeral-storage: "16Gi" }}
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


def render_injection_bins() -> str:
    ft = _ft()
    t = INJECTION_BINS_TEMPLATE.format(
        image=IMAGE, out=INJECTION_ROOT, accounting=_failure_accounting("${OUT}/attempts/bins", ft.EXIT_HALT),
        last=N_SHARDS - 1, first=OUT_ROOT, halt=ft.EXIT_HALT, pin=INJECTION_PIN, committed=COMMITTED_BINS)
    return _robust(t, "  backoffLimit: 2\n")


# THE INJECTION TEST'S TOYS: experiments/AOJ/injection_test.py toys, one CPU job per study,
# each worker one toy at a time; the output (JSON lines) is resumable, so an evicted pod
# picks up where it stopped, and printed on the log as a base64 tar at the end.
# top and band launched at mtx-s1.80; pseudo and wp, launched after the bins job's fix,
# clone mtx-s1.81 (the same toy code, plus the float comparison of the bins)
TOYS_PIN = "mtx-s1.80"
TOYS_PINS = {"top": TOYS_PIN, "band": TOYS_PIN, "pseudo": "mtx-s1.81", "wp": "mtx-s1.81", "tops": "mtx-s1.87",
             "pooled": "mtx-s1.92"}
TOYS_NEEDED_FLAGS = {"experiments/AOJ/injection_test.py": "def run_toys"}
TOYS_CPU = 32
SCORES = ["reference"] + [m.name for m in MODELS]
TOY_STUDIES = {
    # the top window at 1 %: toys around each fit's own background, the signal injected at
    # three sizes; the real data with a signal injected; the signal that fails the cut
    # left in the fail region (each model's simulated top-like efficiency at its data cut,
    # which the CMS reference does not have)
    "top": [("--regions top --modes bootstrap --toys 20", SCORES),
            ("--regions top --modes data --toys 10 --variants full fixed", SCORES),
            ("--regions top --modes leak --toys 20 --sizes 2000 --variants full fixed "
             "--eps experiments/FIGS/data/aoj_checks_v1/model_vs_domain.json", SCORES[1:])],
    # the checks' pseudo-window, at the sizes injected there and above
    "pseudo": [("--regions pseudo --modes bootstrap data --toys 20 --sizes 250 500 1000", SCORES)],
    # the validation band: the top window of real, signal-depleted data
    "band": [("--regions band --modes bootstrap data --toys 10", SCORES)],
    # two more working points
    "wp": [("--regions top_eff0.005 top_eff0.02 --modes bootstrap --toys 10 --variants full fixed", SCORES)],
    # THE FIXED PROCEDURE (fit_v5): toys with the tops failing the cut in the fail region,
    # fitted given the reference's signal per bin; tops_half holds twice that many (the
    # one-sided systematic). The reference is fitted with none (EPS_REF = 1), so not here.
    "tops": [("--regions top --modes tops tops_half --toys 20 --variants full fixed", SCORES[1:])],
    # THE POOLED SHAPE (fit_v6): each score's toys at the pooled shape fitted with fit_v6's
    # procedure for a given shape (fixed_ftest), and whole ensembles -- every pretrained
    # score at once, the pooled shape re-derived from the toys -- to put the shape's own
    # uncertainty into the yields.
    "pooled": [("--regions top --modes pooled ensemble --toys 40 --variants fixed_ftest fixed "
                "--pooled experiments/FIGS/data/aoj_full_v1/fit_v6/results.json", SCORES[1:])],
}
NEEDS_EXTRA = {"pseudo", "wp"}

INJECTION_TOYS_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # INJECTION TEST: TOYS, STUDY {study}. GENERATED by scripts/build_aoj_jobs.py. Runs
  # experiments/AOJ/injection_test.py toys from the committed bins{extra_note} into
  # {out}/toys_{study}_<k>.jsonl (resumable), then prints them on the log as a base64 tar.
  name: aoj-injection-toys-{study}-v1-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
{accounting}{precondition}          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          # injection_test.py reads the pseudo window from the checks' code, which imports
          # the staging code, whose imports need pyarrow (absent from the image)
          pip install --no-cache-dir -q pyarrow || exit 1
          python3 -c "import pyarrow" || {{ echo "FATAL: pyarrow does not import"; exit 1; }}
          export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
{runs}          cd "${{OUT}}"
          echo "BEGIN-TAR"
          tar czf - toys_{study}_*.jsonl | base64 -w 0
          echo
          echo "END-TAR"
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        resources:
          requests: {{ memory: "24Gi", cpu: "{cpu}" }}
          limits:   {{ memory: "24Gi", cpu: "{cpu}" }}
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
"""


def render_injection_toys(study: str) -> str:
    ft = _ft()
    extra = study in NEEDS_EXTRA
    pre = (f'          [ -f "${{OUT}}/injection_bins.npz" ] || {{ echo "FATAL: no injection bins"; exit {ft.EXIT_HALT}; }}\n'
           if extra else "")
    runs = "".join(
        f"          python3 experiments/AOJ/injection_test.py toys {args} \\\n"
        f"            --names {' '.join(names)} \\\n"
        + (f'            --extra "${{OUT}}/injection_bins.npz" \\\n' if extra else "")
        + f'            --workers {TOYS_CPU - 1} --out "${{OUT}}/toys_{study}_{k}.jsonl"\n'
        for k, (args, names) in enumerate(TOY_STUDIES[study]))
    t = INJECTION_TOYS_TEMPLATE.format(
        study=study, image=IMAGE, out=INJECTION_ROOT,
        accounting=_failure_accounting(f"${{OUT}}/attempts/toys_{study}", ft.EXIT_HALT),
        precondition=pre, pin=TOYS_PINS[study], runs=runs, cpu=TOYS_CPU,
        extra_note=" and the exported injection bins" if extra else "")
    return _robust(t, "  backoffLimit: 2\n")


def _storage_guard(path: str, need_g: int) -> str:
    """Bash that refuses to write to a volume 95 % full or with less than need_g GB free."""
    return (f'          USED=$(df --output=pcent {path} | tail -1 | tr -dc 0-9)\n'
            f'          FREE_G=$(df -BG --output=avail {path} | tail -1 | tr -dc 0-9)\n'
            f'          echo "PVC used: ${{USED}}%  free: ${{FREE_G}}G"\n'
            f'          [ "${{USED}}" -lt 95 ] || {{ echo "FATAL: {path} is ${{USED}}% full"; exit 1; }}\n'
            f'          [ "${{FREE_G}}" -ge {need_g} ] || {{ echo "FATAL: ${{FREE_G}}G free on {path}, need {need_g}G"; exit 1; }}\n')


# THE FITS AGAIN (v5, 2026-10-01), with the tops that fail each cut in the fail region
# (peak_fit._Model, tops_from_reference): experiments/AOJ/fit_v5.py from the committed bins,
# a CPU job; the output goes to the volume and, as a base64 tar on the log, to the repository.
FIT5_ROOT = "/data/results/aoj/fit_v5"
FIT5_PIN = "mtx-s1.86"
FIT5_NEEDED_FLAGS = {"experiments/AOJ/fit_v5.py": "leak_systematic", "experiments/AOJ/peak_fit.py": "def tops_from_reference"}
FIT5_CPU = 32

FIT5_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # THE REAL-DATA FITS, v5. GENERATED by scripts/build_aoj_jobs.py. Runs
  # experiments/AOJ/fit_v5.py from the committed bins (no jets): every score's top fit
  # with the tops that fail its cut in the fail region, validation toys, the shape and
  # leak systematics, and the per-vocabulary readout. Writes {out}; prints it on the log.
  name: aoj-fit-v5-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
{accounting}{storage}          if [ ! -e "${{OUT}}/results.json" ]; then
            git clone --depth 1 --branch "{pin}" \
              https://github.com/raunavm/transferlearningsophon.git \
              /workspace/transferlearningsophon
            cd /workspace/transferlearningsophon
            git rev-parse HEAD
            export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
            python3 experiments/AOJ/fit_v5.py --workers {workers} \
              --out /scratch/fit_v5 --analysis-out /scratch/fit_v5/analysis_v5
            cp -r /scratch/fit_v5/. "${{OUT}}/"
          fi
          cd "${{OUT}}"
          echo "BEGIN-TAR"
          tar czf - results.json histograms.npz fit_quality.json analysis_v5 | base64 -w 0
          echo
          echo "END-TAR"
        volumeMounts:
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        resources:
          requests: {{ memory: "24Gi", cpu: "{cpu}", ephemeral-storage: "4Gi" }}
          limits:   {{ memory: "24Gi", cpu: "{cpu}", ephemeral-storage: "4Gi" }}
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
        emptyDir: {{ sizeLimit: "4Gi" }}
"""


def render_fit_v5() -> str:
    ft = _ft()
    t = FIT5_TEMPLATE.format(image=IMAGE, out=FIT5_ROOT, accounting=_failure_accounting("${OUT}/attempts", ft.EXIT_HALT),
                             storage=_storage_guard("/data", 1), pin=FIT5_PIN, workers=FIT5_CPU - 1, cpu=FIT5_CPU)
    return _robust(t, "  backoffLimit: 2\n")


# ONE PEAK SHAPE (v6, 2026-10-01): experiments/AOJ/fit_v6.py from the committed bins and
# fit_v5, the fit_v5 job with the script and output changed.
FIT6_ROOT = "/data/results/aoj/fit_v6"
# first launched at mtx-s1.90: it refused the tops it rebuilt from the reference, 1e-6 apart
# from fit_v5's total; re-created at mtx-s1.91 with the reference refit's own tolerance
FIT6_PIN = "mtx-s1.91"
FIT6_NEEDED_FLAGS = {"experiments/AOJ/fit_v6.py": "pooled_shape", "experiments/AOJ/peak_fit.py": "def pooled_shape",
                     "experiments/FIGS/data/aoj_full_v1/fit_v5/results.json": "fail_tops"}


def render_fit_v6() -> str:
    t = render_fit_v5()
    subs = [("THE REAL-DATA FITS, v5.", "THE REAL-DATA FITS, v6: ONE PEAK SHAPE FOR THE PRETRAINED MODELS."),
            ("  # experiments/AOJ/fit_v5.py from the committed bins (no jets): every score's top fit\n"
             "  # with the tops that fail its cut in the fail region, validation toys, the shape and\n"
             "  # leak systematics, and the per-vocabulary readout.",
             "  # experiments/AOJ/fit_v6.py from the committed bins and fit_v5 (no jets): the pooled\n"
             "  # shape, every model's fit at it given the tops in the fail region, the shape and\n"
             "  # leak systematics, and the per-vocabulary readout."),
            ("name: aoj-fit-v5-raunav", "name: aoj-fit-v6-raunav"),
            (f"OUT={FIT5_ROOT}", f"OUT={FIT6_ROOT}"),
            (f'--branch "{FIT5_PIN}"', f'--branch "{FIT6_PIN}"'),
            ("python3 experiments/AOJ/fit_v5.py", "python3 experiments/AOJ/fit_v6.py"),
            ("--out /scratch/fit_v5 --analysis-out /scratch/fit_v5/analysis_v5", "--out /scratch/fit_v6 --analysis-out /scratch/fit_v6/analysis_v6"),
            ('cp -r /scratch/fit_v5/. "${OUT}/"', 'cp -r /scratch/fit_v6/. "${OUT}/"'),
            ("fit_quality.json analysis_v5 | base64", "fit_quality.json analysis_v6 | base64")]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the v5 fit template changed; cannot derive v6 ({a[:50]!r})")
        t = t.replace(a, b)
    return t


# THE CHECKS AGAIN (v2, 2026-10-01), against the main fit, with the corrections of the
# verification of checks_v1: the injection's recovery net of the data's own signal and the
# procedure's bias on toys, the leak scan, the prong-only test's power, the run spread as
# measured, the closure as a fraction with its error, the reference's validation as a count.
CHECKS2_ROOT = "/data/results/aoj/checks_v2"
# first launched at mtx-s1.89 against fit_v5 (floated shapes); stopped after 25 min when the
# pooled-shape fit (fit_v6) passed its toys, and re-created against fit_v6 at mtx-s1.93; that
# one died in the injection step's summary (no asymmetric pulls at a fixed shape, an empty
# spread), and the job was re-created at mtx-s1.96
CHECKS2_PIN = "mtx-s1.96"
CHECKS2_NEEDED_FLAGS = {"experiments/AOJ/realdata_checks.py": "def _floats",
                        "experiments/FIGS/data/aoj_full_v1/fit_v6/results.json": "pooled_shape"}
MAIN_FIT2 = "experiments/FIGS/data/aoj_full_v1/fit_v6/results.json"
# 19 spawned workers, each holding the jets of the rho window and one model's scores: the
# v1 job held 15 in 48Gi
CHECKS2_CPU = 20


def render_checks_v2() -> str:
    """The checks job against fit_v5 into checks_v2, by asserted substitution: its merge
    and steps are checks_v1's; more workers for the injection toys, and a storage guard."""
    ft = _ft()
    t = render_checks()
    subs = [
        ("  name: aoj-checks-v1-raunav\n", "  name: aoj-checks-v2-raunav\n"),
        (f"OUT={CHECKS_ROOT}\n", f"OUT={CHECKS2_ROOT}\n"),
        (f'--branch "{CHECKS_PIN}"', f'--branch "{CHECKS2_PIN}"'),
        (f"--fit {MAIN_FIT} ", f"--fit {MAIN_FIT2} "),
        ("--workers 15 --toys 200", f"--workers {CHECKS2_CPU - 1} --toys 200"),
        ('memory: "48Gi", cpu: "16"', f'memory: "64Gi", cpu: "{CHECKS2_CPU}"'),
        ('          SHARDS=""; FIRST=""\n', _storage_guard("/data", 1) + '          SHARDS=""; FIRST=""\n'),
        ('          ls -la "${OUT}"\n', '          ls -la "${OUT}"\n          cd "${OUT}"\n          echo "BEGIN-TAR"\n'
                                       '          tar czf - *.json | base64 -w 0\n          echo\n          echo "END-TAR"\n'),
    ]
    for a, b in subs:
        n = t.count(a)
        if n < 1:
            raise SystemExit(f"FATAL: the checks template changed; cannot derive v2 ({a[:40]!r})")
        t = t.replace(a, b)
    return t


def render_read_injection() -> str:
    """The injection toys' output so far, read back: the read-back job of the checks,
    pointed at the toys' JSON lines (complete lines only are read locally)."""
    t = render_read()
    subs = [(f"REAL-DATA CHECKS: READ BACK {CHECKS_ROOT}", f"INJECTION TOYS: READ BACK {INJECTION_ROOT}"),
            ("Prints the\n  # checks' JSON files", "Prints the\n  # toys' JSON lines"),
            ("name: aoj-read-checks-v1-raunav", "name: aoj-read-injection-v1-raunav"),
            (f'          [ -f "{CHECKS_ROOT}/prong_test.json" ] || {{ echo "FATAL: the checks have not finished"; exit 1; }}\n', ""),
            # the files grow while the toys run, and tar fails on a file that changes as it is
            # read (the second read-back, 2026-10-01): a snapshot in the pod's own disk first
            (f'cd "{CHECKS_ROOT}"', f'mkdir -p /tmp/snap\n          cp "{INJECTION_ROOT}"/toys_*.jsonl /tmp/snap/\n'
                                    '          cd /tmp/snap'),
            ("tar czf - *.json | base64 -w 0", "tar czf - toys_*.jsonl | base64 -w 0")]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the read-back template changed ({a[:40]!r})")
        t = t.replace(a, b)
    return t


# ============================================================ the v2 grid (PRESPEC A7-A14)
# THE SAME RUN FOR THE v2 GRID (configs/arms/v2_grid.json), one run per set of tiers, emitted
# when their runs have finished (--v2 TIER ...): V2_OUT_ROOT/t<tiers>/shard<i> are the first
# run's shards (render_shard) on v2_models(), with three changes. Each checkpoint is resolved
# in the pod by experiments/EVAL/extract_v2.py's own rule (experiments/AOJ/v2_checkpoints.py);
# the prong-only score is written beside the three-prong one (as the rescore), so the checks
# need no rescore; and the GPU is never a v2 grid product. Then one CPU job fits them all.
# Two runs: tiers 1 and 2 (the freeze, t12), then tier 3 (t3) at the freeze run's shape.
#
# CHECKPOINTS (decided 2026-10-07). PRESPEC A7-A14 name none for the real data. It is a frozen
# readout of each model's own output layer, so every checkpoint A14 reports for those: the
# primary (best70), the weight average (A8: every result), the global best (A14's sensitivity
# check of the frozen readouts), and the BatchNorm twins of best70 and the global best (A14's
# rule, fired 2026-10-03). A name whose file another name of the run already holds -- bestval
# at best70's epoch -- is not scored again: its scores are linked (v2_checkpoints.py), as the
# extraction links its directory. The pooled peak shape is built from best70 alone
# (fit_v6.V2_PRIMARY) and every checkpoint is fitted at it.
V2_CHECKPOINTS = ("best70", "wavg", "bestval", "best70_bn", "bestval_bn")
# THE FREEZE (A14 item 8): tiers 1 and 2 are fitted together and derive the pooled shape; any
# other tier set (tier 3) holds that run's shape, so its results change no frozen number.
V2_FREEZE = (1, 2)
# THE GPU IS NOT PINNED to the run index's product. Inference runs with TF32 off
# (extract_features.strict_fp32), where the GPU agrees with float32 on the CPU to 2e-4 in
# log-odds (experiments/FIGS/data/tf32_check), under the float16 spacing the scores are
# stored at for |log-odds| >= 0.5; and every model of a shard runs on the shard's one GPU, as
# in the first run. A pin would stage each shard once per product and queue for the GPUs the
# grid trains on. So: any GPU but those products, with the 11 GB PARALLEL is sized for.
# THE MODELS. discriminants.py defines the scores on a column of the contraction-tree label
# map, so the random partitions and flavour pairs (no column) are not scored; nor the
# self-supervised arms (no output layer), nor the leave-one-family-out arms (A13: anomaly).
V2_TREE_RUNGS = ("L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1")
V2_PIN = "mtx-s2.00"
V2_NEEDED_FLAGS = {"experiments/AOJ/v2_checkpoints.py": "resolve_checkpoints",
                   "experiments/EVAL/extract_v2.py": "def bn_twin",
                   "experiments/EVAL/extract_features.py": "def strict_fp32",
                   "experiments/AOJ/discriminants.py": "prong_only",
                   "experiments/AOJ/peak_fit.py": "def tops_from_reference",
                   "experiments/AOJ/export_fit_bins.py": "these are not the bins that run fitted",
                   "experiments/AOJ/fit_v6.py": "--v2",
                   "experiments/AOJ/refit_from_bins.py": "def write_analysis_v2"}
V2_OUT_ROOT = "/data/results/aoj/full_v2"
V2_K8S = K8S / "v2"
# the fit job: peak_fit.py and export_fit_bins.py run on one core, fit_v6.py on the rest
V2_FIT_CPU = 16


@functools.cache
def _launch():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class V2Model(NamedTuple):
    """One checkpoint of one v2 grid run, with the fields the templates read from a Model."""
    name: str      # <run directory less "mtx-">-<tag>, e.g. r16q1mass-s4-best70_bn
    arm: str       # the grid arm, recorded in the extraction manifest
    rung: str      # label-map column the head realises
    k: int
    num_reg: int
    run_id: str    # mtx-<arm slug>-s<run> (build_mtx_launch.v2_run_id)
    run_dir: str
    tag: str       # the checkpoint, as extract_v2.resolve_checkpoints names it
    spec: str      # its grid training spec, under experiments/MTX/k8s

    @property
    def checkpoint(self) -> str:
        """The link experiments/AOJ/v2_checkpoints.py makes in the pod to the resolved file."""
        return f"/workspace/ckpt/{self.name}.pt"


def v2_models(tiers) -> list:
    """The public checkpoint (the reference row of every run), then V2_CHECKPOINTS of every
    run of the arms of `tiers` the scores are defined on, in grid order."""
    launch = _launch()
    out = [MODELS[0]]
    for arm in launch.v2_arms():
        rung = arm["name"].removesuffix("_MASS_LM").removesuffix("_MASS")
        if int(arm["tier"]) not in tiers or rung not in V2_TREE_RUNGS or arm.get("parent"):
            continue
        for run in range(1, int(arm["runs"]) + 1):
            rid = launch.v2_run_id(arm["name"], run)
            out += [V2Model(f"{rid.removeprefix('mtx-')}-{tag}", arm["name"], rung, int(arm["num_classes"]),
                            int(arm["mass_lambda"] is not None), rid, f"{launch.V2_ROOT}/{rid}", tag,
                            f"v2/grid/job-{launch.v2_job_name(arm['name'], run)}.yaml")
                    for tag in V2_CHECKPOINTS]
    if len(out) == 1:
        raise SystemExit(f"FATAL: tiers {sorted(tiers)} hold no arm the scores are defined on")
    return out


def v2_label(tiers) -> str:
    """The run of tiers 1 and 2 is t12: its shards and fits are under V2_OUT_ROOT/t12."""
    return "".join(str(t) for t in sorted(set(tiers)))


def v2_shape_from(tiers) -> str | None:
    """None for the freeze run, which derives the pooled shape; for a run of tiers outside it,
    the freeze run's fit, whose shape it holds. A run of some but not all freeze tiers, or of
    them and more, would be a second freeze, and is refused."""
    if set(tiers) == set(V2_FREEZE):
        return None
    if set(tiers) & set(V2_FREEZE):
        raise SystemExit(f"FATAL: tiers {sorted(set(tiers))}: the freeze run is --v2 "
                         f"{' '.join(map(str, V2_FREEZE))}, and any other run holds its pooled shape")
    return f"{V2_OUT_ROOT}/t{v2_label(V2_FREEZE)}/fit_v6/results.json"


def render_v2_shard(i: int, files: list[dict], tiers) -> str:
    """Shard i of the v2 run of `tiers`: the first run's shard on v2_models(tiers), the
    checkpoints resolved in the pod before any download, with the prong-only score, the v2
    image and pyarrow, the retry policy, and a GPU off the grid's products. Derived by
    substitution, each asserted, so staging, row alignment and scoring are the first run's."""
    ft, launch = _ft(), _launch()
    models = v2_models(tiers)
    label = v2_label(tiers)
    resolve = ('          python3 experiments/AOJ/v2_checkpoints.py --links /workspace/ckpt '
               '--record "${OUT}/checkpoints.json" \\\n'
               + "".join(f"            {m.name}={m.run_dir}:{m.tag} \\\n" for m in models if isinstance(m, V2Model))
               + f"            || exit {ft.EXIT_HALT}\n")
    products = ", ".join(f'"{p}"' for p in sorted(set(launch.V2_GPU_BY_RUN.values())))
    exists = "              - key: nvidia.com/gpu.product\n                operator: Exists\n"
    skip = '[ -f "${OUT}/scores_$1.npz" ] && { echo "skip $1 (scored)"; return 0; }\n'
    done = '          touch "${OUT}/DONE"\n          ls -la "${OUT}"\n'
    t = render_shard(i, files, models)
    subs = [
        (skip, skip + '            [ -f "/workspace/ckpt/$1.same" ] && '
                      '{ echo "skip $1 (the file of $(cat /workspace/ckpt/$1.same))"; return 0; }\n'),
        (done, "          # a name whose file another name holds gets that name's scores (v2_checkpoints.py)\n"
               "          for f in /workspace/ckpt/*.same; do\n"
               '            [ -f "${f}" ] || continue\n'
               '            m=$(basename "${f}" .same); first=$(cat "${f}")\n'
               '            [ -f "${OUT}/scores_${first}.npz" ] || { echo "FATAL: ${first} is not scored"; exit 1; }\n'
               '            ln -sfn "scores_${first}.npz" "${OUT}/scores_${m}.npz"\n'
               '            ln -sfn "scores_${first}.json" "${OUT}/scores_${m}.json"\n'
               "          done\n" + done),
        ("  # FULL REAL-DATA RUN, SHARD", f"  # v2 GRID, TIERS {', '.join(label)}: FULL REAL-DATA RUN, SHARD"),
        (f"name: aoj-full-s{i}-raunav", f"name: aoj-v2-t{label}-s{i}-raunav"),
        (f"image: {IMAGE}", f"image: {launch.V2_IMAGE}"),
        (f"          OUT={OUT_ROOT}/shard{i}\n",
         f"          OUT={V2_OUT_ROOT}/t{label}/shard{i}\n" + _failure_accounting("${OUT}", ft.EXIT_HALT)),
        ('[ "${USED}" -lt 95 ]', '[ "${USED}" -lt 85 ]'),
        (f'--branch "{PIN}"', f'--branch "{V2_PIN}"'),
        ("pip install --no-cache-dir -q pyarrow h5py", f"pip install --no-cache-dir -q {launch.V2_PYARROW} h5py"),
        ("          for c in ", resolve + "          for c in "),
        ("--structures three_prong ", "--structures three_prong prong_only "),
        ("              # never the 3090 pool: the benchmark wave is Pending for it. NotIn\n",
         "              # never a v2 grid product (V2_GPU_BY_RUN): the grid waits for them. NotIn\n"),
        (exists, exists + "              - key: nvidia.com/gpu.memory\n                operator: Gt\n"
                          f'                values: ["{MIN_GPU_MEMORY_MIB}"]\n'),
        ('values: ["NVIDIA-GeForce-RTX-3090"]', f"values: [{products}]"),
        ("values: [" + ", ".join(f'"{b}"' for b in BAD_NODES) + "]",
         "values: [" + ", ".join(f'"{b}"' for b in dict.fromkeys(SIM_BAD_NODES + launch.V2_BAD_NODES)) + "]"),
    ]
    for a, b in subs:
        if t.count(a) != 1:
            raise SystemExit(f"FATAL: the shard template changed; cannot derive the v2 shard ({a[:50]!r})")
        t = t.replace(a, b)
    return _robust(t, "  # 2, not 1: the shard resumes model by model, so a retry after a lost node\n"
                      "  # repeats only the ~30 min of download and staging.\n  backoffLimit: 2\n")


V2_FIT_TEMPLATE = r"""apiVersion: batch/v1
kind: Job
metadata:
  # v2 GRID, TIERS {tiers}: THE REAL-DATA FITS. GENERATED by scripts/build_aoj_jobs.py --v2.
  # Joins the {n} shards (merge_shards.py refuses a partial or inconsistent set); fits every
  # score with its own floated shape and the tops failing its cut (peak_fit.py: fit_v5's
  # fits, on the jets) and exports the bins it fitted (export_fit_bins.py); then the main
  # fit at one pooled shape (fit_v6.py --v2), {shape_note},
  # and its per-arm readout at each checkpoint. Into {out}/fit, fit_v6 and analysis_v6, and on the log.
  name: aoj-v2-t{label}-fit-raunav
  namespace: cms-ml
spec:
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
          OUT={out}
{accounting}{storage}          SHARDS=""
          for i in $(seq 0 {last}); do
            [ -f "${{OUT}}/shard${{i}}/DONE" ] || {{ echo "FATAL: shard ${{i}} has not finished"; exit {halt}; }}
            SHARDS="${{SHARDS}} ${{OUT}}/shard${{i}}"
          done
{held}          git clone --depth 1 --branch "{pin}" \
            https://github.com/raunavm/transferlearningsophon.git \
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
          SCORES=""
          for m in {model_names}; do SCORES="${{SCORES}} ${{m}}=/scratch/merged/scores_${{m}}.npz"; done
          # A step's output reaches the volume only when the step is complete, its marker file
          # last, so a retry redoes only the steps that did not finish.
          if [ ! -e "${{OUT}}/fit/bins.npz" ]; then
            python3 experiments/AOJ/merge_shards.py --shards ${{SHARDS}} --out /scratch/merged
            python3 experiments/AOJ/peak_fit.py --peaks top \
              --jets /scratch/merged/jets.npz --closure /scratch/merged/closure.json \
              --scores ${{SCORES}} --shape-pool {pool_names} \
              --eff 0.01 --toys 200 --out /scratch/fit
            python3 experiments/AOJ/export_fit_bins.py --peak top --jets /scratch/merged/jets.npz \
              --results /scratch/fit/results.json --histograms /scratch/fit/histograms.npz \
              --scores ${{SCORES}} --out /scratch/fit/bins.npz
            mkdir -p "${{OUT}}/fit"
            cp /scratch/merged/merge_manifest.json /scratch/merged/closure.json /scratch/fit/results.json \
              /scratch/fit/histograms.npz "${{OUT}}/fit/"
            cp /scratch/fit/bins.npz "${{OUT}}/fit/bins.npz"
          fi
          if [ ! -e "${{OUT}}/analysis_v6/aoj_top.json" ]; then
            python3 experiments/AOJ/fit_v6.py --v2 --bins "${{OUT}}/fit/bins.npz" \
              --previous "${{OUT}}/fit/results.json" --out /scratch/fit_v6 \
              --analysis-out /scratch/analysis_v6 --workers {workers}{shape_from}
            mkdir -p "${{OUT}}/fit_v6" "${{OUT}}/analysis_v6"
            cp /scratch/fit_v6/* "${{OUT}}/fit_v6/"
            cp /scratch/analysis_v6/aoj_top.json "${{OUT}}/analysis_v6/aoj_top.json"
          fi
          cd "${{OUT}}"
          echo "BEGIN-TAR"
          tar czf - fit fit_v6 analysis_v6 | base64 -w 0
          echo
          echo "END-TAR"
        volumeMounts:
        - {{ name: data,    mountPath: /data }}
        - {{ name: scratch, mountPath: /scratch }}
        resources:
          # every score in float64 over the ~9.4 M jets of the fit region (~75 MB a model) in
          # peak_fit.py and again in export_fit_bins.py; fit_v6.py's workers hold bins only
          requests: {{ memory: "32Gi", cpu: "{cpu}", ephemeral-storage: "16Gi" }}
          limits:   {{ memory: "32Gi", cpu: "{cpu}", ephemeral-storage: "16Gi" }}
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
        emptyDir: {{ sizeLimit: "16Gi" }}
"""


def render_v2_fit(tiers) -> str:
    ft, launch = _ft(), _launch()
    label = v2_label(tiers)
    held = v2_shape_from(tiers)
    models = v2_models(tiers)
    t = V2_FIT_TEMPLATE.format(
        tiers=", ".join(label), label=label, n=N_SHARDS, out=f"{V2_OUT_ROOT}/t{label}", image=launch.V2_IMAGE,
        accounting=_failure_accounting("${OUT}/attempts/fit", ft.EXIT_HALT), storage=_storage_guard("/data", 1),
        last=N_SHARDS - 1, halt=ft.EXIT_HALT, pin=V2_PIN, model_names=" ".join(m.name for m in models),
        pool_names=" ".join(m.name for m in models if getattr(m, "tag", None) == "best70"),
        held=(f'          [ -f "{held}" ] || {{ echo "FATAL: no freeze fit {held}; its shape is held here"; '
              f"exit {ft.EXIT_HALT}; }}\n" if held else ""),
        shape_from=f" \\\n              --shape-from {held}" if held else "",
        shape_note=("the freeze run's, held from its fit_v6" if held else
                    "built from the best70 entries: the freeze's shape"),
        workers=V2_FIT_CPU - 1, cpu=V2_FIT_CPU)
    return _robust(t, "  backoffLimit: 2\n")


def v2_specs(tiers) -> dict[pathlib.Path, str]:
    v2_shape_from(tiers)
    label = v2_label(tiers)
    out = {V2_K8S / f"job-aoj-v2-t{label}-s{i}-raunav.yaml": render_v2_shard(i, fs, tiers)
           for i, fs in enumerate(shards())}
    out[V2_K8S / f"job-aoj-v2-t{label}-fit-raunav.yaml"] = render_v2_fit(tiers)
    return out


def specs() -> dict[pathlib.Path, str]:
    out = {K8S / f"job-aoj-full-s{i}-raunav.yaml": render_shard(i, fs) for i, fs in enumerate(shards())}
    out[K8S / "job-aoj-full-fit-raunav.yaml"] = render_fit()
    out[K8S / "job-aoj-full-fitcheck-raunav.yaml"] = render_fit_check()
    out[K8S / "job-aoj-full-fit-v2-raunav.yaml"] = render_fit_v2()
    out[K8S / "job-aoj-full-fitbins-raunav.yaml"] = render_fit_bins()
    out[K8S / "job-aoj-full-fit-v3-raunav.yaml"] = render_fit_v3()
    for i, fs in enumerate(shards()):
        out[K8S / f"job-aoj-rescore-s{i}-raunav.yaml"] = render_rescore_shard(i, fs)
        for k in range(RESCORE_PARTS.get(i, 0)):
            out[K8S / f"job-aoj-rescore-s{i}-p{k}-raunav.yaml"] = render_rescore_part(i, fs, k, RESCORE_PARTS[i])
    for g, ms in enumerate(sim_groups()):
        out[K8S / f"job-aoj-sim-g{g}-raunav.yaml"] = render_sim(g, ms)
    out[K8S / "job-aoj-checks-v1-raunav.yaml"] = render_checks()
    out[K8S / "job-aoj-read-checks-v1-raunav.yaml"] = render_read()
    out[K8S / "job-aoj-injection-bins-v1-raunav.yaml"] = render_injection_bins()
    for study in TOY_STUDIES:
        out[K8S / f"job-aoj-injection-toys-{study}-v1-raunav.yaml"] = render_injection_toys(study)
    out[K8S / "job-aoj-fit-v5-raunav.yaml"] = render_fit_v5()
    out[K8S / "job-aoj-checks-v2-raunav.yaml"] = render_checks_v2()
    out[K8S / "job-aoj-read-injection-v1-raunav.yaml"] = render_read_injection()
    out[K8S / "job-aoj-fit-v6-raunav.yaml"] = render_fit_v6()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="verify, write nothing")
    ap.add_argument("--pin-not-yet-tagged", action="store_true",
                    help=f"check the working tree instead of {CHECKS2_PIN} ({V2_PIN} with --v2), which is "
                         "tagged after the commit")
    ap.add_argument("--v2", nargs="+", type=int, metavar="TIER", default=None,
                    help="emit ONLY the v2 grid's real-data run of these tiers (configs/arms/v2_grid.json) into "
                         f"{V2_K8S.relative_to(ROOT)}, once their runs and BatchNorm twins exist")
    a = ap.parse_args()
    if a.v2:
        verify_heads(v2_models(a.v2))
        verify_pin(V2_PIN, a.pin_not_yet_tagged, V2_NEEDED_FLAGS)
        return write(v2_specs(a.v2), a.check_only)
    verify_heads()
    verify_pin(PIN, False)                      # the shards already ran at it
    verify_pin(FIT_PIN, False, FIT_NEEDED_FLAGS)
    verify_pin(CHECK_PIN, False, CHECK_NEEDED_FLAGS)
    verify_pin(FIT2_PIN, False, FIT2_NEEDED_FLAGS)
    verify_pin(BINS_PIN, False, BINS_NEEDED_FLAGS)
    verify_pin(FIT3_PIN, False, FIT3_NEEDED_FLAGS)
    verify_pin(RESCORE_PIN, False, RESCORE_NEEDED_FLAGS)
    verify_pin(CHECKS_PIN, False, CHECKS_NEEDED_FLAGS)
    for study, pin in sorted(TOYS_PINS.items()):
        if study not in ("tops", "pooled"):
            verify_pin(pin, False, TOYS_NEEDED_FLAGS)
    verify_pin(TOYS_PINS["tops"], False, {"experiments/AOJ/injection_test.py": "EPS_TRUE"})
    verify_pin(TOYS_PINS["pooled"], False, {"experiments/AOJ/injection_test.py": "def _run_ensemble",
                                                           "experiments/FIGS/data/aoj_full_v1/fit_v6/results.json": "pooled_shape"})
    verify_pin(INJECTION_PIN, False, INJECTION_NEEDED_FLAGS)
    verify_pin(FIT5_PIN, False, FIT5_NEEDED_FLAGS)
    verify_pin(FIT6_PIN, False, FIT6_NEEDED_FLAGS)
    verify_pin(CHECKS2_PIN, a.pin_not_yet_tagged, CHECKS2_NEEDED_FLAGS)
    return write(specs(), a.check_only)


def write(out: dict[pathlib.Path, str], check_only: bool) -> int:
    for path, text in out.items():
        if check_only:
            print(f"ok   {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
