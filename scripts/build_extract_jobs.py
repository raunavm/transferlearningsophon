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
# mtx-s1.7, so extraction and evaluation carry ONE provenance tag. Safe to
# move: `git diff mtx-s1.6 mtx-s1.7` over the four files this pod actually
# executes (extract_features.py, JetClassII_base.yaml, the two arch files)
# is EMPTY, so the pin change cannot alter what runs.
# BUMPED mtx-s1.23 -> mtx-s1.41 on 2026-09-15, when the ladder arms were
# added. tests/test_spec_pins.py caught that s1.23 predates 9710c97, which
# moved refuse_foreign_checkpoint() BEFORE the first np.save instead of
# after it -- at s1.23 the guard fires only once the arrays it protects
# have already been overwritten.
#
# THIS DOES NOT BREAK I1 AGAINST THE FIVE ARMS ALREADY EXTRACTED AT s1.23,
# and that was checked in the diff rather than assumed: 9710c97 touches
# only the guard and reuses a hash the old code already computed. No line
# in the feature or manifest computation changes, so the arrays this
# produces are identical to the ones on disk. The new pin is strictly
# safer, not different.
PIN = "mtx-s1.41"
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"

# (run_id, arm, K, checkpoint dir). The G1 rows are the SMOKE TEST described
# above; the mtx rows are the real thing and only work once those runs finish.
# The G1 sweep ran a 16-EPOCH budget and both rows finished 16/16
# (experiments/RUNS.csv). item 18's "the paper checkpoint is epoch 79" is a
# statement about the 80-epoch matrix and cannot apply to them: asking these
# two for net_epoch-79_state.pt requests a file that was never written, and the
# pod clones, pip-installs and exits 1 against backoffLimit 50.
BEST_EPOCH_RUNS = {"g1-l162-lr1e3", "g1-r16q1-lr5e4"}

RUNS = [
    ("g1-l162-lr1e3",   "L162",   162, "/data/results/g1/g1-l162-lr1e3"),
    ("g1-r16q1-lr5e4",  "R16_Q1",  17, "/data/results/g1/g1-r16q1-lr5e4"),
    # THE REAL THING. Full budget, 80 epochs, and -- unlike the pair above --
    # all five run at the SAME rate (5e-4), which is what the mtx-l162-s1b row
    # in experiments/RUNS.csv exists to establish. So vocabulary is the only
    # variable between L162 and R16_Q1 here, which the G1 smoke pair cannot say.
    ("mtx-l162-s1b",    "L162",   162, "/data/results/mtx/mtx-l162-s1b"),
    ("mtx-r16q1-s2",    "R16_Q1",  17, "/data/results/mtx/mtx-r16q1-s2"),
    ("mtx-r16q1-s3",    "R16_Q1",  17, "/data/results/mtx/mtx-r16q1-s3"),
    ("mtx-r16q1-s4",    "R16_Q1",  17, "/data/results/mtx/mtx-r16q1-s4"),
    ("mtx-r16q1-s5",    "R16_Q1",  17, "/data/results/mtx/mtx-r16q1-s5"),

    # THE REST OF THE LADDER, added 2026-09-15 once the 32-arm matrix finished.
    # Until now this list held one L162 seed and four R16_Q1 seeds, because that
    # is what had completed when it was written -- so TWO OF THE FOUR RUNGS IN D3
    # HAD NO FEATURES AT ALL and the headline granularity contrast rested on two
    # rungs, not four. Every arm below is verified at ckpts=80 / last_epoch=79 by
    # job-mtx-inventory-raunav.
    #
    # mtx-l162-s1 IS DELIBERATELY ABSENT AND MUST STAY ABSENT. It trained at
    # --start-lr 1e-3; every other arm in the matrix trains at 5e-4 (checked
    # across all 22 specs, not assumed). Including it would make the L162 row
    # differ from its siblings in RATE as well as seed, which is exactly the
    # confound I1 exists to prevent -- and it would look like a fifth seed.
    # L162's five matrix seeds are therefore s1b, s2, s3, s4, s5.
    ("mtx-l188-s1",     "L188",   188, "/data/results/mtx/mtx-l188-s1"),
    ("mtx-l188-s2",     "L188",   188, "/data/results/mtx/mtx-l188-s2"),
    ("mtx-l188-s3",     "L188",   188, "/data/results/mtx/mtx-l188-s3"),
    ("mtx-l188-s4",     "L188",   188, "/data/results/mtx/mtx-l188-s4"),
    ("mtx-l188-s5",     "L188",   188, "/data/results/mtx/mtx-l188-s5"),
    ("mtx-l162-s2",     "L162",   162, "/data/results/mtx/mtx-l162-s2"),
    ("mtx-l162-s3",     "L162",   162, "/data/results/mtx/mtx-l162-s3"),
    ("mtx-l162-s4",     "L162",   162, "/data/results/mtx/mtx-l162-s4"),
    ("mtx-l162-s5",     "L162",   162, "/data/results/mtx/mtx-l162-s5"),
    # K=43, not 42. The rung is named for the number of RESONANT groups; the
    # head also carries the QCD group, and the tree in docs/PLAN.md gives the
    # widths as 188, 162, 64, 43, 30, 17, 4, 2. Taken from each arm's own spec.
    ("mtx-r42q1-s1",    "R42_Q1",  43, "/data/results/mtx/mtx-r42q1-s1"),
    ("mtx-r42q1-s2",    "R42_Q1",  43, "/data/results/mtx/mtx-r42q1-s2"),
    ("mtx-r42q1-s3",    "R42_Q1",  43, "/data/results/mtx/mtx-r42q1-s3"),
    ("mtx-r42q1-s4",    "R42_Q1",  43, "/data/results/mtx/mtx-r42q1-s4"),
    ("mtx-r42q1-s5",    "R42_Q1",  43, "/data/results/mtx/mtx-r42q1-s5"),
    ("mtx-r16q1-s1",    "R16_Q1",  17, "/data/results/mtx/mtx-r16q1-s1"),
]

# THE RANDOM-LABEL CONTROL AND THE TEN MASS-OUTPUT MODELS, added 2026-09-18.
# Same 4-tuple as RUNS and the same template, file list, data config and jet
# cap, so their rows align with the twenty caches above -- but a SEPARATE LIST
# WITH ITS OWN PIN, for the reason WINDOW_PIN gives below: these need an
# extract_features.py that PIN's tag predates (--num-reg), and moving PIN would
# rewrite the tag on twenty specs whose jobs already ran.
#
# Run ids, K and arm names are read from experiments/MTX/k8s/job-mtx-*-raunav
# .yaml (RUN_ID=, -o num_classes, --arm), not from the file names: the spec
# files say `l162_mass`, the run directories say `l162mass`.
# mtx-s1.49, NOT mtx-s1.48. s1.48 was the intended name, but by the time these
# specs were written it already existed -- pushed, at 9fe3a1a, cloned by the ten
# probe-ladder / labelrec-ladder specs -- and 9fe3a1a has neither --num-reg nor
# extract_observers.py. A mass spec pinned there dies on "unrecognized
# arguments: --num-reg", fifty times. Moving a pushed tag other specs clone is
# not an option, so this wave takes the next name. verify_pin() now checks the
# FLAG is in the tagged extractor, not just that the file is: presence alone
# passed s1.48.
CONTROL_AND_MASS_PIN = "mtx-s1.49"
CONTROL_AND_MASS_RUNS = [
    ("mtx-rand-d1-s1b", "RAND_d1", 17, "/data/results/mtx/mtx-rand-d1-s1b"),
    *[(f"mtx-l162mass-s{s}", "L162_MASS", 162, f"/data/results/mtx/mtx-l162mass-s{s}")
      for s in range(1, 6)],
    *[(f"mtx-r16q1mass-s{s}", "R16_Q1_MASS", 17, f"/data/results/mtx/mtx-r16q1mass-s{s}")
      for s in range(1, 6)],
]
# NOT TRAINED YET. Buildable by naming them with --only, and never emitted
# otherwise: an un-launchable YAML on disk looks exactly like a launchable one,
# and with backoffLimit 50 applying it early is fifty clones that each stop at
# the checkpoint guard.
NOT_YET_TRAINED = [
    ("mtx-rand-d2-s2", "RAND_d2", 17, "/data/results/mtx/mtx-rand-d2-s2"),
    ("mtx-rand-d3-s3", "RAND_d3", 17, "/data/results/mtx/mtx-rand-d3-s3"),
]
# Regression outputs AFTER the K class outputs (ParT_sophon_arch_mass.py: one,
# the jet mass). The extractor is told --num-classes K --num-reg 1 rather than
# --num-classes K+1, so K stays the plain twin's and the manifest never counts
# the mass output as a class. It is recoverable ONLY as the last column of the
# raw head output, so these runs also pass --save-logits:
#   K=162: 2e6 x 163 x 4 B = 1.304 GB on top of the ~1.06 GB every run writes
#   K=17:  2e6 x  18 x 4 B = 0.144 GB
NUM_REG = {r[0]: 1 for r in CONTROL_AND_MASS_RUNS if r[1].endswith("_MASS")}

# Arms complete on the PVC that are NOT in RUNS, and why. Kept as data so a
# reader does not have to infer an omission from silence.
DELIBERATELY_EXCLUDED = {
    "mtx-l162-s1": "trained at --start-lr 1e-3; every matrix arm trains at 5e-4",
    "mtx-rand-d1-s1": "superseded by mtx-rand-d1-s1b (item 24); only 3 epochs",
    "mtx-r42q1-s1.lr2p5e-4.superseded-20260907": "superseded rate, 7 epochs",
    "mtx-mpm-s1": "masked-particle pretraining gives its two class-attention "
                  "blocks no gradient (experiments/MTX/ParT_sophon_arch_mpm.py:"
                  "8-11), so the class token this job would cache is a random "
                  "projection of the trunk; evaluated by fine-tuning only",
}

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

          # CLAUDE.md: check free space before a write of this size. Each arm
          # writes ~1.1 GB (2,000,000 jets x 128 float32, plus labels and
          # observers; {logits_clause}). The FT specs have carried a guard
          # like this since the legs; the extraction specs never did, and this
          # wave adds fifteen of them at once to a PVC measured at 82% on
          # 2026-09-15 -- roughly 34 GB of headroom before CLAUDE.md's line.
          # FAILS SAFE, verified rather than hoped: --output=pcent is GNU-only
          # (the image has it -- the FT specs' --output=avail guard runs here),
          # but if df ever lacked it USED would be empty, `[ "" -lt 85 ]` errors
          # under set -e, and the || branch exits 1. An unparseable check refuses
          # the write instead of waving it through.
          USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
          echo "PVC used: ${{USED}}%"
          [ "${{USED}}" -lt 85 ] || {{ echo "FATAL: /data is ${{USED}}% full, at or over the 85% line."; df -h /data; exit 1; }}

          CKPT={ckpt_dir}/{ckpt_file}
          # CHECKPOINT RULE (DECISIONS_PENDING item 18, decided 2026-09-07).
          # For the mtx arms this is net_epoch-79_state.pt -- the LAST epoch --
          # not weaver's net_best_epoch_state.pt. weaver's best marker is the
          # argmax over ~80 validation scores that each read a DIFFERENT slice
          # of the validation split: per-epoch SD is 0.047-0.075 with 4-11
          # epochs tied inside 0.02 of the maximum, so the epoch it lands on
          # (78, 74, 64, 76, 58 across the five R16_Q1 seeds) is slice luck.
          # Using it would make TRAINING DURATION an uncontrolled variable
          # across seeds. If the file is absent the run did not finish.
          [ -f "${{CKPT}}" ] || {{ echo "FATAL: no ${{CKPT}}. Run unfinished?"; ls -la {ckpt_dir} | head -20; exit 1; }}

          # features_v2, NOT features. The file list was interleaved by family
          # on 2026-08-27, so a re-extraction now reads DIFFERENT jets than the
          # ones under .../features -- which are what probe-bvc-v1 and the
          # recorded L162-vs-R16_Q1 numbers were computed on. extract_features.py
          # np.save()s unconditionally, so reusing the path would overwrite those
          # inputs with a different sample and silently invalidate a result
          # already in experiments/RUNS.csv. Different sample, different path.
          # The path encodes the CHECKPOINT, not just the sample. features_v2
          # was written from net_best_epoch_state.pt; item 18 then made epoch 79
          # the paper checkpoint, and both would have landed in the same
          # directory with nothing in the cache recording which model made it.
          # That is exactly how the zero-fill control came to compare epoch-79
          # masked features against a best-epoch baseline. Different checkpoint,
          # different path -- and extract_features.py now also refuses to write
          # into a directory whose manifest names a different checkpoint.
          OUT=/data/results/eval/{run_id}/{out_name}
          mkdir -p ${{OUT}}

          PYTHONUNBUFFERED=1 python3 experiments/EVAL/extract_features.py \\
            --checkpoint "${{CKPT}}" \\
            --num-classes {k} \\
            --arm {arm} \\
            --data-config {data_config} \\
            --data-test {file_list} \\
            --out ${{OUT}} \\
            --batch-size 512 --num-workers 1 --fetch-step 1{max_jets}{extra_flags}

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


# Selected test jets, docs/GROUND_TRUTH.md. Used only to convert a file count
# into a jet count for the balance guard below.
TEST_JETS_SELECTED = 27_448_839


def balanced_prefix_jets() -> int:
    """How many leading jets still span every family.

    Round-robin exhausts the SHORTEST family first, so only the first
    3 x min(files) entries carry all three. With Res2P 50, Res34P 215, QCD 70
    that is 150 of 335 files ~ 12.29 M jets; past ~210 files the list is Res34P
    alone. A --max-jets beyond this point is a CLASS-BIASED head slice that
    still looks like a uniform cut.
    """
    per = [hi - lo + 1 for _, lo, hi in FAMILIES]
    return int(min(per) * len(per) * TEST_JETS_SELECTED / sum(per))


def interleaved_files() -> str:
    """Round-robin across families.

    EVERY PREFIX SPANS ALL THREE ONLY WHILE THE SHORTEST FAMILY LASTS -- the
    original wording of this docstring said it held for every prefix, which is
    false and is exactly the assumption extract_features.py's `keep()` relies on
    when it head-slices to --max-jets before striding. See balanced_prefix_jets().
    """
    per = [[f"/jc2/jet_data/{fam}_{i:04d}.parquet" for i in range(lo, hi + 1)]
           for fam, lo, hi in FAMILIES]
    out = []
    for i in range(max(len(x) for x in per)):
        for fam in per:
            if i < len(fam):
                out.append(fam[i])
    return " ".join(out)


# THE |V_cb| WINDOW MODE (DECISIONS_PENDING item 26, option A; PI approved
# 2026-09-12 "do the most rigorous").
#
# bc_vs_rest SKIPPED on probe-physics-v2 with 270 signal jets in the test split
# against MIN_PER_CLASS = 1,000, because only 0.62 % of jets survive the
# published window. Extracting MORE jets unwindowed pays the model for 161 jets
# per 1 kept; putting the window in `selection:` makes weaver apply it at load,
# so the model runs on survivors only and the job turns I/O-bound.
#
# The SAME 335-file interleaved list is reused deliberately -- it already spans
# the whole test split (DECISIONS_PENDING records that the first QCD jet sits
# ~2.2e7 rows in), so `--max-jets` was the only thing capping the old
# extractions at 2 M. Reusing it also keeps row alignment with the existing
# caches derivable from one file list rather than two.
#
# --max-jets 400,000 WAS BELIEVED TO BE HEADROOM AND IT BOUND. This comment read
# "HEADROOM, NOT A TARGET: ~171,000 survivors are expected from 27.4 M streamed,
# so the cap cannot bind". All five arms stopped EXACTLY at 400,000 on
# 2026-09-14, so the stream was cut short and the split was never swept.
#
# The ~171,000 came from probe-physics-v2's 0.62 % survival rate, which was
# measured on a 2 M extraction drawn under the OLD strict-family-order list --
# i.e. on Res2P jets ALONE. Over the interleaved all-family stream the survival
# rate is far higher, so the cap was reached early. The cost is the only number
# the row turns on: label_X_bc came to 8,208 per arm, a test fifth of 1,641
# against probe.py's floor of 1,000 -- 1.64x, not the 3.76x this file sized for.
# bc_vs_rest RUNS rather than SKIPS, so nothing was lost, but the headroom the
# plan deliberately bought was halved. See ledger vcb-window-extraction-result.
# A SEPARATE PIN, deliberately. PIN is module-level and every extraction spec
# clones it, so moving PIN to pick up the new config would REWRITE the pins on
# specs whose jobs already ran -- the ledger row audit-2-anchors records what
# that costs: a spec still cloning a tag three commits before its own fix, and
# nothing saying so. Window specs are new files, so they can take a new tag
# without touching anybody else's provenance.
WINDOW_PIN = "mtx-s1.39"
WINDOW_CONFIG = "configs/data/JetClassII_vcbwindow.yaml"
WINDOW_OUT = "features_vcbwindow_e79"
WINDOW_MAX_JETS = 400_000

# THE SECOND PASS, sized off the MEASURED yield rather than a prediction.
# 8,208 label_X_bc per 400,000 in-window jets is 2.052 %, so reaching the
# plan's 3.76x floor needs ~916,000 survivors; 1,500,000 clears it with room
# and still bounds the job at ~2.8 h on the observed 149 jets/s. A SEPARATE
# OUTPUT DIRECTORY, not an overwrite: extract_features.py np.save()s
# unconditionally and the 400k caches are the provenance of a completed run.
WINDOW2_OUT = "features_vcbwindow_e79_full"
WINDOW2_MAX_JETS = 1_500_000
# Only the arms the physics-probe table is built on (probe-physics-v4).
WINDOW_RUNS = {"mtx-l162-s1b", "mtx-r16q1-s2", "mtx-r16q1-s3",
               "mtx-r16q1-s4", "mtx-r16q1-s5"}


def pin_for(run_id: str, window: bool = False) -> str:
    """The tag a run's spec clones. Per list, so adding a list never moves the
    tag under a spec that has already run."""
    if window:
        return WINDOW_PIN
    late = {r[0] for r in CONTROL_AND_MASS_RUNS + NOT_YET_TRAINED}
    return CONTROL_AND_MASS_PIN if run_id in late else PIN


def build(run_id, arm, k, ckpt_dir, gpu: bool, max_jets: int,
          ckpt_epoch: int | None = 79, window: bool = False,
          window_full: bool = False) -> tuple[str, str]:
    name = run_id.replace("_", "-").lower()
    num_reg = NUM_REG.get(run_id, 0)
    if num_reg and window:
        raise SystemExit(f"FATAL: {run_id} has a regression output, and "
                         f"{WINDOW_PIN}'s extract_features.py has no --num-reg.")
    # THE CAP MUST STAY INSIDE THE BALANCED PREFIX. extract_features.py keeps
    # a[:max_jets][::stride] -- a head slice first -- which is class-representative
    # only while the round-robin still spans every family. Window mode is exempt:
    # it applies the selection at load, so --max-jets counts SURVIVORS out of the
    # whole stream rather than leading rows of it.
    if max_jets and not window:
        cap = balanced_prefix_jets()
        if max_jets > cap:
            raise SystemExit(
                f"FATAL: --max-jets {max_jets:,} exceeds the balanced prefix of "
                f"{cap:,} jets for {run_id}. Past that point the interleaved "
                f"file list has exhausted the shortest family, so the head slice "
                f"is class-biased while still looking like a uniform cut. Raise "
                f"the family file ranges, or stride the full stream instead.")
    if window:
        name += "-vcbwindow-full" if window_full else "-vcbwindow"
    text = TEMPLATE.format(
        run_id=run_id, arm=arm, k=k, ckpt_dir=ckpt_dir, image=IMAGE,
        pin=pin_for(run_id, window),
        logits_clause=(
            "no --save-logits here" if not num_reg else
            f"PLUS logits.npy here, 2,000,000 x {k + num_reg} float32 = "
            f"{2_000_000 * (k + num_reg) * 4 / 1e9:.2f} GB, whose LAST column "
            f"is the mass output and not a class"),
        extra_flags=(f" \\\n            --num-reg {num_reg} --save-logits"
                     if num_reg else ""),
        ckpt_file=(f"net_epoch-{ckpt_epoch}_state.pt" if ckpt_epoch is not None
                   else "net_best_epoch_state.pt"),
        name=name + ("-gpu" if gpu else ""),
        device_note=("GPU build." if gpu else
                     "CPU-ONLY ON PURPOSE. Extraction is a forward pass, so a "
                     "GPU buys speed and\n  # not correctness -- and the GPU "
                     "queue has not scheduled anything in 3.5 days\n  # while "
                     "CPU is uncontended."),
        data_config=(WINDOW_CONFIG if window
                     else "configs/data/JetClassII_base.yaml"),
        out_name=((WINDOW2_OUT if window_full else WINDOW_OUT) if window else
                  (f"features_e{ckpt_epoch}" if ckpt_epoch is not None
                   else "features_v2")),
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


# THE OBSERVER THE TWENTY CACHES NEVER CARRIED.
#
# None of the specs above passes --observers, so every cache holds
# extract_features.py's default four (jet_pt, jet_sdmass, jet_eta,
# jet_nparticles), and JetClassII_base.yaml does not list genjet_sdmass at all.
# The jet-mass analysis needs it for every jet of every model. It is a property
# of the jets, and the caches are row-aligned, so ONE model-free pass over the
# same list supplies it for all of them: experiments/EVAL/extract_observers.py,
# which refuses to write unless its rows are bit-identical to the caches named
# here. ~44 MB (five float32 columns + int16 labels for 2,000,000 jets), under
# CLAUDE.md's 100 MB line, so no free-space guard.
OBSERVERS_OUT = "/data/results/eval/test2m_observers"
OBSERVERS_CONFIG = "configs/data/JetClassII_massreg.yaml"
OBSERVERS_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  # genjet_sdmass (AND THE FOUR DEFAULT OBSERVERS) FOR THE 2,000,000 TEST JETS
  # OF EVERY FEATURE CACHE -- no model, no checkpoint, CPU only.
  #
  # The twenty caches under /data/results/eval/<run>/features_e79 were written
  # without the generator-level groomed mass. They are row-aligned, so it is
  # read once here instead of re-extracting twenty models. The script checks,
  # BEFORE writing, that label188_sha256 equals every cache's and that jet_pt /
  # jet_sdmass / jet_eta / jet_nparticles are bit-identical to theirs; on any
  # difference it exits non-zero and writes nothing.
  #
  # genjet_sdmass is a hard 0.0 for an unmatched jet. MASK ON > 0.
  #
  # GENERATED by scripts/build_extract_jobs.py --observers-job.
  name: extract-observers-test2m-raunav
  namespace: cms-ml
spec:
  backoffLimit: 1
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

          OUT={out}
          PYTHONUNBUFFERED=1 python3 experiments/EVAL/extract_observers.py \\
            --data-config {data_config} \\
            --data-test {file_list} \\
            --align-with {align_with} \\
            --out ${{OUT}} \\
            --batch-size 512 --num-workers 1 --fetch-step 1 \\
            --max-jets {max_jets}

          echo "=== manifest ==="
          cat ${{OUT}}/observers_manifest.json
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
        resources:
          requests: {{ memory: "48Gi", cpu: "2", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "48Gi", cpu: "2", ephemeral-storage: "20Gi" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
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


def build_observers_job(max_jets: int) -> tuple[str, str]:
    if not max_jets or max_jets > balanced_prefix_jets():
        raise SystemExit("FATAL: the observers job needs the caches' own "
                         "--max-jets, inside the balanced prefix")
    caches = [f"/data/results/eval/{r[0]}/features_e79" for r in RUNS
              if r[0] not in BEST_EPOCH_RUNS]
    text = OBSERVERS_TEMPLATE.format(
        image=IMAGE, pin=CONTROL_AND_MASS_PIN, out=OBSERVERS_OUT,
        data_config=OBSERVERS_CONFIG, file_list=interleaved_files(),
        align_with=" ".join(caches), max_jets=max_jets)
    return "job-extract-observers-test2m-raunav.yaml", text


def verify_pin(pin: str, needed: list[str], allow_untagged: bool,
               must_say: dict[str, str] | None = None) -> None:
    """The pod clones a TAG, not the working tree, so a script that exists here
    can be absent there. That is exactly how the first attempt failed: the pin
    predated the extractor and both jobs crash-looped on "No such file or
    directory" after paying for a clone and a pip install. Verified at BUILD
    time, where it costs nothing.

    `must_say` maps a path to text the TAGGED copy must contain. A file being
    present says nothing about whether it is new enough to accept the flags the
    spec passes it."""
    import subprocess
    must_say = must_say or {}
    tagged = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{pin}"],
        cwd=ROOT, capture_output=True).returncode == 0
    if not tagged and allow_untagged:
        gone = [p for p in needed if not (ROOT / p).exists()]
        if gone:
            sys.exit(f"FATAL: {gone} not in the working tree")
        for path, text in must_say.items():
            if text not in (ROOT / path).read_text():
                sys.exit(f"FATAL: working-tree {path} does not contain {text!r}")
        print(f"WARNING: tag {pin} DOES NOT EXIST YET. Create it on a commit "
              f"containing all {len(needed)} files the job runs BEFORE "
              f"applying any spec that clones it.")
        return
    for path in needed:
        r = subprocess.run(["git", "cat-file", "-e", f"{pin}:{path}"],
                           cwd=ROOT, capture_output=True)
        if r.returncode != 0:
            sys.exit(f"FATAL: tag {pin} does not contain {path}. The pod clones "
                     f"the TAG, so this job would fail after cloning. Tag a "
                     f"commit that has it, or fix the pin.")
    for path, text in must_say.items():
        shown = subprocess.run(["git", "show", f"{pin}:{path}"], cwd=ROOT,
                               capture_output=True, text=True).stdout
        if text not in shown:
            sys.exit(f"FATAL: {path} at tag {pin} does not contain {text!r}, "
                     f"which the spec relies on. The tag predates it.")
    print(f"pin {pin} verified to contain all {len(needed)} files the job runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    # DEFAULT 2,000,000, not 0. With default=0 the flag is omitted entirely and
    # the extraction runs UNCAPPED over all 335 files -- days at the observed
    # 64 jets/s, and it has already happened once (RUNS.csv "extract-e79-recap"),
    # because regenerating the specs silently dropped a cap that had been added
    # by hand. Pass --max-jets 0 to mean "no cap" explicitly.
    ap.add_argument("--max-jets", type=int, default=2_000_000)
    # item 18: the paper checkpoint is the last epoch. Pass --ckpt-epoch '' to
    # restore the best-epoch file for a gate whose numbers are already published.
    ap.add_argument("--ckpt-epoch", type=int, default=79)
    # Build a subset. The concurrency cap in CLAUDE.md is 5 running jobs, and
    # RUNS is longer than that, so emitting all of them at once would either
    # breach the cap or leave un-launched YAML lying around that looks launched.
    ap.add_argument("--only", nargs="*", default=None, metavar="RUN_ID",
                    help="restrict to these run_ids (default: RUNS and "
                         "CONTROL_AND_MASS_RUNS; NOT_YET_TRAINED only by name)")
    # The specs for a wave are written, committed and THEN tagged, so the tag
    # they clone cannot exist while they are being written. Opt-in per call, so
    # a mistyped pin still fails the build everywhere else.
    ap.add_argument("--pin-not-yet-tagged", action="store_true",
                    help="allow a pin with no tag yet; the files are checked in "
                         "the working tree and the tag MUST be created on a "
                         "commit containing them before any kubectl apply")
    # The |V_cb| windowed extraction. Forces its own data config, its own output
    # directory and its own arm set, so it cannot overwrite an existing cache:
    # extract_features.py np.save()s unconditionally, and reusing a path is how
    # a published result would be silently replaced by a different sample.
    ap.add_argument("--window", action="store_true",
                    help="extract inside the arXiv:2503.00118 |V_cb| window "
                         "(DECISIONS_PENDING item 26 option A)")
    ap.add_argument("--window-full", action="store_true",
                    help="the SECOND windowed pass at the measured yield: "
                         "--max-jets 1.5M into a separate output directory, "
                         "because the 400k cap bound (ledger "
                         "vcb-window-extraction-result). Implies --window.")
    ap.add_argument("--observers-job", action="store_true",
                    help="emit ONLY the model-free job that adds genjet_sdmass "
                         "for the caches' 2,000,000 jets (see OBSERVERS_TEMPLATE)")
    args = ap.parse_args()

    if args.observers_job:
        verify_pin(CONTROL_AND_MASS_PIN,
                   ["experiments/EVAL/extract_observers.py", OBSERVERS_CONFIG],
                   args.pin_not_yet_tagged)
        fname, text = build_observers_job(args.max_jets)
        yaml.safe_load(text)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / fname).write_text(text)
        print(f"  {fname}  CPU, no model, max_jets={args.max_jets}")
        return 0

    if args.window_full:
        args.window = True
    runs = RUNS + CONTROL_AND_MASS_RUNS
    selectable = runs + NOT_YET_TRAINED
    if args.window:
        selectable = RUNS
        runs = [r for r in RUNS if r[0] in WINDOW_RUNS]
        if args.max_jets == 2_000_000:
            args.max_jets = WINDOW2_MAX_JETS if args.window_full else WINDOW_MAX_JETS
        print(f"window mode{' (FULL)' if args.window_full else ''}: "
              f"{len(runs)} arms, config {WINDOW_CONFIG}, "
              f"out {WINDOW2_OUT if args.window_full else WINDOW_OUT}, "
              f"max-jets {args.max_jets}")
    if args.only:
        known = {r[0] for r in selectable}
        unknown = set(args.only) - known
        if unknown:
            sys.exit(f"FATAL: unknown run_id(s) {sorted(unknown)}. "
                     f"Known: {sorted(known)}")
        runs = [r for r in selectable if r[0] in set(args.only)]
    needed = ["experiments/EVAL/extract_features.py",
              WINDOW_CONFIG if args.window else "configs/data/JetClassII_base.yaml",
              "experiments/MTX/ParT_sophon_arch_mtx.py",
              "experiments/E1/ParT_sophon_arch_10c.py"]
    for pin in sorted({pin_for(r[0], args.window) for r in runs}):
        uses_num_reg = any(r[0] in NUM_REG and pin_for(r[0], args.window) == pin
                           for r in runs)
        verify_pin(pin, needed, args.pin_not_yet_tagged,
                   {"experiments/EVAL/extract_features.py": '"--num-reg"'}
                   if uses_num_reg else None)

    # THE RANDOM-LABEL CONTROL MUST EXTRACT ON THE SAME DEVICE FOR EVERY DRAW.
    # Draw 1 ran on CPU (job-extract-mtx-rand-d1-s1b-raunav.yaml requests no
    # GPU), and prediction C4 compares the three draws with each other and with
    # the 17-class models, which also extracted on CPU. Running a later draw on
    # a GPU would put a device difference -- different kernels, and mixed
    # precision where the CPU path has none -- inside the one control that
    # answers the objection that this study is a tautology. Draws 2 and 3 are
    # still pretraining as of 2026-09-19; this exists so that whoever emits
    # their specs later cannot get it wrong by reaching for --gpu out of habit.
    if args.gpu:
        rand = sorted(r for r, _, _, _ in runs if r.startswith("mtx-rand-"))
        if rand:
            sys.exit("FATAL: --gpu with the random-label control draws "
                     f"{rand}. Draw 1 extracted on CPU and C4 compares the "
                     "draws with each other; a device difference between draws "
                     "would be a second uncontrolled variable in the control "
                     "itself. Emit them without --gpu.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, arm, k, ckpt in runs:
        epoch = None if run_id in BEST_EPOCH_RUNS else args.ckpt_epoch
        fname, text = build(run_id, arm, k, ckpt, args.gpu, args.max_jets,
                            epoch, window=args.window,
                            window_full=args.window_full)
        d = yaml.safe_load(text)
        body = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        for must in (f"--num-classes {k} ", f"--arm {arm} ",
                     f'--branch "{pin_for(run_id, args.window)}"',
                     *( (f"--num-reg {NUM_REG[run_id]} --save-logits",)
                        if run_id in NUM_REG else () ),
                     WINDOW_CONFIG if args.window else "configs/data/JetClassII_base.yaml",
                     *( (f"--max-jets {args.max_jets}",) if args.max_jets else () ),
                     f"net_epoch-{epoch}_state.pt"
                     if epoch is not None else "net_best_epoch_state.pt"):
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
