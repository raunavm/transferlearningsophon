#!/usr/bin/env python3
"""Bring the committed full-budget MTX specs up to launch state.

WHY THIS EXISTS RATHER THAN build_arm_jobs.py
---------------------------------------------
scripts/build_arm_jobs.py is marked STALE in its own docstring: the three
committed seed-1 YAMLs are AHEAD of its template, and regenerating from it
would revert five launch-blocking fixes. So the YAMLs stay the source of truth
and this script applies DELTAS to them, the same way scripts/add_autoresume.py
does. Nothing here regenerates a spec from scratch.

THE FOUR DELTAS, AND WHY EACH ONE
---------------------------------
1. PER-ARM LEARNING RATE.  The committed specs all carry `--start-lr 5e-4`,
   the frozen upstream Sophon rate, because they were written before G1 ran.
   G1 returned KILL -- the optimum MOVES with K -- so a single shared rate
   would confound granularity with tuning (invariant I1). docs/GATES.md's
   branch is explicit: arms are compared at their OWN optima. See RATES below
   for each arm's evidence.

2. REPO PIN mtx-s1.1 -> mtx-s1.2.  s1.2 is the commit that carries
   seed_weaver's --lean-val-metrics.

3. --lean-val-metrics.  THIS IS THE LOAD-BEARING ONE. The committed specs try
   to drop weaver's O(K^2) roc_auc_score_matrix at validation by passing
   `--network-config experiments/MTX/ParT_sophon_arch_mtx.py`, which defines a
   get_evaluate_fn hook. weaver 0.4.17 REMOVED that hook -- `get_train_fn` and
   `get_evaluate_fn` appear zero times in its source, and train.py:728 logs
   "Running in classification mode" unconditionally. The network config is
   still needed (it defines the MODEL) but its eval hook is dead code, so the
   pairwise-AUC matrix would run every epoch and nothing would say so.
   Measured cost at K=162: 26.7 min/epoch, i.e. 35.6 h wasted per 80-epoch
   run. seed_weaver's --lean-val-metrics monkeypatches
   weaver.utils.nn.tools.evaluate_classification before weaver_train.main()
   and hard-fails if that function has no eval_metrics parameter, so it cannot
   silently no-op the way the hook did.

4. AUTO-RESUME + backoffLimit 3, via scripts/add_autoresume.py. An 80-epoch
   run is ~5-6 days on the measured throughput; over that span a node eviction
   or a queue reap is close to certain, and backoffLimit 0 turns either into
   the loss of the whole budget. The recipe guard that ships with the snippet
   is what makes a retry safe now that the rate is arm-specific.

WHAT IS NOT BUILT HERE
----------------------
    R42_Q1  its rate is NOT BRACKETED. 5e-4 and 1e-3 both diverged to nan at
            iteration 2, so 2.5e-4 is the only point that trained -- a bound,
            not an optimum. Launching ~6 GPU-days against an unbracketed rate
            risks spending the arm's whole budget at a rate a cheap 16-epoch
            point could have shown was wrong. Emit it with --allow-unbracketed
            once scripts/build_lr_sweep.py's 1.25e-4 point has reported.
    L188    no arm config yet, and no reweighting sidecar.
    RAND42  must be rebuilt against R42_Q1 group sizes.
    MPM     different objective; gated on G0.

Run:  python3 scripts/build_mtx_launch.py [--allow-unbracketed] [--check-only]
      python3 scripts/build_mtx_launch.py --derive-draws 2:2 3:3
          the random control's further partition draws -- see derive_draw
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
K8S = ROOT / "experiments" / "MTX" / "k8s"
PIN = "mtx-s1.2"
OLD_LR = "5e-4"

# arm -> (K, rate, bracketed, evidence)
#
# "bracketed" means the sweep has a point on BOTH sides of the chosen rate that
# scored worse. An argmax at the edge of the grid is not an optimum, and this
# study compares arms AT their optima -- so an unbracketed rate is a rate we do
# not actually know. All figures are best-over-16-epochs validation accuracy at
# 20% budget; the measured seed spread is 0.0158, so gaps below that are ties.
RATES = {
    # NOT bracketed. This said True until 2026-08-22, on the reasoning that
    # 2e-3 diverging put a ceiling on the optimum. g1-r42q1-lr5e4-s2 then
    # cleared 16k iterations clean at a rate whose seed-1 run went nan at
    # iteration 2, so divergence is STOCHASTIC and bounds nothing. Every
    # divergence on record is a seed-1 run. mtx-l162-s1 is already training at
    # 1e-3 and stays; what this flag now blocks is queueing MORE seeds at a
    # rate g1-l162-lr14e4 has not yet confirmed -- turning one run at a
    # possibly-wrong rate into five.
    # RATE CHANGED 1e-3 -> 5e-4 on 2026-08-27, and this is the I1 repair.
    # mtx-l162-s1 launched at 1e-3 while every mtx-r16q1 seed runs at 5e-4, so
    # the headline pair varied vocabulary AND learning rate -- invisible to
    # tests/test_arm_configs.py, which only reads configs/arms/*.yaml, and those
    # carry no lr key. Re-deriving analyse_g1.py's own reversal test, exactly one
    # of three rate pairs is a KILL-grade reversal and it EXCLUDES 5e-4:
    #   2.5e-4 vs 5e-4  L162 +0.03049 clears  R16_Q1 +0.00360 TIE     no
    #   2.5e-4 vs 1e-3  L162 +0.03365 clears  R16_Q1 -0.01931 clears  REVERSAL
    #   5e-4   vs 1e-3  L162 +0.00316 TIE     R16_Q1 -0.02291 clears  no
    # 5e-4 is R16_Q1's outright argmax and ties L162's (0.55232 vs 0.55548, gap
    # 0.00316, a fifth of the margin), so a single rate DOES serve both arms --
    # docs/GATES.md G1's PASS condition. "Compare arms at their own optima"
    # appears in docs/GATES.md ZERO times; it was invented here and in
    # build_lr_sweep.py and back-attributed to the gate document.
    # Bracketed stays False: nothing above 1e-3 has trained and g1-l162-lr14e4
    # has never scheduled. That flag gates queueing MORE seeds, and it should.
    "L162": (162, "5e-4", False,
             "2.5e-4 0.52183 < 5e-4 0.55232 ~ 1e-3 0.55548. The 5e-4/1e-3 gap is "
             "0.00316, a fifth of the 0.0158 floor, so the two are TIED and 5e-4 "
             "is chosen because it is also R16_Q1's argmax, which makes the "
             "headline pair single-rate. weaver's flat+decay scales with "
             "num_epochs (train.py:509-521), so the 16-epoch sweep decays from "
             "epoch 12 and the 80-epoch run from epoch 56: the sweep gives a "
             "too-hot rate only 12 flat epochs to misbehave and is biased HIGH. "
             "Of two tied rates at 20% budget, the lower is safer at 100%."),
    # L188 differs from L162 only in the 27 QCD sub-labels: a pure loss change,
    # zero sampling change (configs/labelmaps/contraction_tree.v1.yaml). It is
    # the SAME head-size regime, so it takes L162's rate and L162's bracketing
    # status; sweeping it separately would put a second variable on the
    # 188-vs-162 step. Added 2026-09-07 when the PI asked for five controlled
    # seeds at the released vocabulary rather than the public checkpoint alone.
    "L188": (188, "5e-4", False,
             "the rate is L162's, by construction: 188 and 162 differ by a pure "
             "loss change on the background sub-labels and share the head-size "
             "regime, so a rate of its own would confound the finest step."),
    "R16_Q1": (17, "5e-4", True,
               "2.5e-4 0.77917 < 5e-4 0.78277 > 1e-3 0.75986 at seed 1, and "
               "2.5e-4 0.76598 < 5e-4 0.76697 at seed 2 -- same ordering at "
               "both seeds. Bracketed on both sides. The 1e-3 deficit "
               "(0.01931) is 1.22x the noise floor; the 2.5e-4 gap is inside "
               "it."),
    # RATE RE-CORRECTED 2.5e-4 -> 5e-4 on 2026-09-07 (PI go-ahead). The entry
    # below used to read "2.5e-4 is the ONLY point that trained", which this
    # file's own header already contradicts: g1-r42q1-lr5e4-s2 ran 16/16 clean
    # at 5e-4. The tie is also wider than it looked -- the 0.0158 floor came
    # from best-of-16 validation accuracy whose per-epoch SD measures 0.047-0.075
    # (scripts/checkpoint_selection_noise.py), so this sweep could not separate
    # rates within ~2x. With the rates indistinguishable, I1 decides: 5e-4 makes
    # the entire ladder single-rate.
    "R42_Q1": (43, "5e-4", False,
               "5e-4 0.70123 (seed 2, 16/16 clean) vs 2.5e-4 0.70721 (seed 1): "
               "a 0.006 gap, unresolvable against a per-epoch SD of 0.047-0.075, "
               "and confounded by seed. Both rates are trainable; the seed-1 nan "
               "at iteration 2 is stochastic and bounds nothing, as the RATES "
               "header note says. Chosen to match every other arm so the "
               "granularity contrast varies vocabulary ALONE (I1)."),
    # The mass-auxiliary twins (DECISIONS_PENDING item 14, addendum 2). A twin
    # MUST share its arm's rate: the 2x2 varies one output node and the loss
    # term that trains it, and a separately-swept rate would put a second
    # variable on the mass axis (I1). Not swept; the bracketing flag is the
    # twin's. scripts/build_mass_jobs.py asserts K and rate against the twin.
    "L162_MASS": (162, "5e-4", False,
                  "the rate is L162's, by construction -- the mass twin differs "
                  "from L162 in the head and the loss term only; a rate of its "
                  "own would confound the mass axis with tuning."),
    "R16_Q1_MASS": (17, "5e-4", True,
                    "the rate is R16_Q1's, by construction -- see L162_MASS; "
                    "R16_Q1's 5e-4 is bracketed on both sides, so the twin "
                    "inherits a bracketed rate."),
    # K = 0: the self-supervised arm builds no classification head at all.
    # Not swept, and deliberately so. The G1 sweep bracketed 5e-4 for a
    # CROSS-ENTROPY objective; MPM minimises L1 + CE over masked particles, so
    # that evidence does not transfer and this rate is inherited, not measured.
    # It is inherited anyway because the arm's job is to be a DENOMINATOR: it
    # must be matched to the supervised arms on unique jets, optimizer steps and
    # tuning budget, and giving the SSL arm a swept rate while the supervised
    # arms carry an inherited one would hand it an advantage the comparison then
    # could not separate from self-supervision. If it FAILS the pre-registered
    # bar (item 17: macro AUC >= 0.95 and Rej_bb >= 100), the documented tuning
    # budget is spent BEFORE the failure is reported, because an untuned SSL run
    # landing in the known collapse regime is evidence about our implementation
    # and not about self-supervision.
    "MPM": (0, "5e-4", False,
            "inherited from the supervised arms, not swept: matching the "
            "denominator to the arms on tuning budget matters more than "
            "optimising it, and G1's bracket was measured on a cross-entropy "
            "objective this arm does not use."),
}

SEEDS = [1]

# Seeds beyond 1 are DERIVED from the already-launch-ready seed-1 spec rather
# than rebuilt, so they inherit its rate, pin, lean-val flag and auto-resume by
# construction and cannot drift from it. Only four things carry the seed: the
# job name, RUN_ID (OUT derives from it), the two --seed flags (manifest writer
# and seed_weaver), and the tensorboard tag.
#
# ONLY FOR ARMS WHOSE RATE IS SETTLED. A spec's learning rate is fixed when it
# is written, so queueing seeds for an arm whose sweep is still open converts
# one run at a possibly-wrong rate into N of them.


def _autoresume():
    spec = importlib.util.spec_from_file_location(
        "add_autoresume", ROOT / "scripts" / "add_autoresume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch(path: pathlib.Path, arm: str, k: int, rate: str) -> str:
    """Transform the spec ATOMICALLY: build it in a temp file, verify the temp,
    and only then move it into place.

    An earlier version wrote the spec, then ran the auto-resume patch, then
    verified. When verification failed it left a HALF-PATCHED spec on disk that
    the next run could not re-patch (its `--start-lr 5e-4` had already been
    consumed), so a failed run poisoned the input for every run after it. A
    generator whose failure mode is a corrupted input is worse than one that
    simply refuses.
    """
    tmp = path.with_suffix(".yaml.tmp")
    try:
        text = path.read_text()

        # 1. per-arm learning rate. R16_Q1's optimum IS the old shared rate, so
        # for that arm alone the replacement is a deliberate no-op.
        n = text.count(f"--start-lr {OLD_LR}")
        if n != 1:
            return f"FAILED: expected 1 --start-lr {OLD_LR}, found {n}"
        if rate != OLD_LR:
            text = text.replace(f"--start-lr {OLD_LR}", f"--start-lr {rate}")

        # 2. repo pin
        if not re.search(r'value: "mtx-s1[^"]*"', text):
            return "FAILED: no repo pin found"
        text = re.sub(r'value: "mtx-s1[^"]*"', f'value: "{PIN}"', text)

        # 3. the lean validation metric
        call = "python3 experiments/E1/seed_weaver.py \\"
        if text.count(call) != 1:
            return f"FAILED: expected 1 seed_weaver call, found {text.count(call)}"
        if "--lean-val-metrics" not in text:
            text = text.replace(
                call, call + "\n            --lean-val-metrics \\")

        # record WHY this rate, in the spec the run clones.
        #
        # THE PHRASE "compared at their own optima" USED TO APPEAR HERE and has
        # been removed. It appears in docs/GATES.md ZERO times -- it was invented
        # in this file and in build_lr_sweep.py and back-attributed to the gate
        # document, as the RATES comment above records. Emitting it stamped the
        # invention into every spec a run clones, which is how a phrase with no
        # source became the documented justification for the headline pair.
        #
        # The rate is read from RATES[arm][1] rather than from `rate` so the
        # stated rate and the executed rate cannot drift: the five L162 specs
        # carried "LEARNING RATE 1e-3" against an executed 5e-4 for sixteen days
        # because they were derived by targeted substitution -- which updated
        # --start-lr but not this block -- rather than regenerated.
        assert rate == RATES[arm][1], (
            f"{arm}: emitter rate {rate} disagrees with RATES {RATES[arm][1]}")
        text = text.replace(
            "  # CORE-MATRIX GRANULARITY ARM",
            f"  # LEARNING RATE {RATES[arm][1]} -- MEASURED for this arm, and it\n"
            f"  # must equal the --start-lr below; tests/test_launch_specs.py\n"
            f"  # compares them with comments STRIPPED, so only this assertion\n"
            f"  # and a reader can catch a stale header.\n"
            f"  # Evidence: {RATES[arm][3]}\n"
            f"  # Applied by scripts/build_mtx_launch.py.\n"
            f"  #\n"
            f"  # CORE-MATRIX GRANULARITY ARM")

        tmp.write_text(text)

        # 4. auto-resume + recipe guard + backoffLimit
        r = _autoresume().patch(tmp)
        if r.startswith("FAILED"):
            return f"FAILED (autoresume): {r}"

        # --- verify by PARSING the emitted spec, not by trusting the edits ---
        d = yaml.safe_load(tmp.read_text())
        args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        code = "\n".join(ln for ln in args.splitlines()
                         if not ln.lstrip().startswith("#"))
        for must in (f"--start-lr {rate}", f"-o num_classes {k}",
                     f"configs/arms/{arm}.yaml", "--lean-val-metrics",
                     "--num-epochs 80", "${RESUME}",
                     f"RECIPE='lr={rate} epochs=80'",
                     "RECIPE stamp", "Refusing to resume"):
            if must not in code:
                return f"FAILED: emitted spec does not execute {must!r}"
        stray = [m for m in re.findall(r"--start-lr (\S+)", code) if m != rate]
        if stray:
            return f"FAILED: a second --start-lr survived: {stray}"
        if d["spec"]["backoffLimit"] != 3:
            return f"FAILED: backoffLimit {d['spec']['backoffLimit']}"
        dumped = yaml.dump(d)
        if PIN not in dumped:
            return f"FAILED: pin {PIN} not in spec"
        if "NVIDIA-GeForce-RTX-3090" not in dumped:
            return "FAILED: GPU pin missing (I7 requires one model across all arms)"

        tmp.replace(path)
        return f"launch-ready (lr={rate}, K={k}, pin {PIN}, backoffLimit 3)"
    finally:
        if tmp.exists():
            tmp.unlink()


def derive_seed(arm: str, base_seed: int, seed: int, base_tag: str | None = None) -> str:
    """Write job-mtx-<arm>-s<seed>-raunav.yaml from the seed-<base_seed> spec.

    base_tag names the SOURCE FILE when it is not simply s<base_seed> -- L162's
    launch-ready arm is `s1b`, the 5e-4 repair of the 1e-3 `s1`. The --seed VALUE
    inside it is still base_seed; only the filename and the name/RUN_ID/tensorboard
    strings carry the tag.
    """
    arm_lc = arm.lower()
    tag = base_tag if base_tag else f"{base_seed}"
    src = K8S / f"job-mtx-{arm_lc}-s{tag}-raunav.yaml"
    dst = K8S / f"job-mtx-{arm_lc}-s{seed}-raunav.yaml"
    if not src.exists():
        return f"FAILED: {src.name} not found"
    if dst.exists():
        return "exists, not regenerated"
    text = src.read_text()

    run_lc = arm.lower().replace("_", "")
    subs = [
        (f"name: mtx-{run_lc}-s{tag}-raunav", f"name: mtx-{run_lc}-s{seed}-raunav"),
        (f"RUN_ID=mtx-{run_lc}-s{tag}", f"RUN_ID=mtx-{run_lc}-s{seed}"),
        (f"--tensorboard mtx_{arm}_s{tag}", f"--tensorboard mtx_{arm}_s{seed}"),
    ]
    for old, new in subs:
        if text.count(old) != 1:
            return f"FAILED: expected 1 {old!r}, found {text.count(old)}"
        text = text.replace(old, new)
    n = text.count(f"--seed {base_seed}")
    if n != 2:
        return f"FAILED: expected 2 '--seed {base_seed}', found {n}"
    text = text.replace(f"--seed {base_seed}", f"--seed {seed}")

    d = yaml.safe_load(text)
    args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    code = "\n".join(ln for ln in args.splitlines() if not ln.lstrip().startswith("#"))
    if d["metadata"]["name"] != f"mtx-{run_lc}-s{seed}-raunav":
        return "FAILED: name not rewritten"
    if f"RUN_ID=mtx-{run_lc}-s{seed}" not in code:
        return "FAILED: RUN_ID not rewritten"
    if re.search(rf"--seed (?!{seed}\b)\d+", code):
        return f"FAILED: a --seed other than {seed} survived"
    if f"--start-lr {RATES[arm][1]}" not in code:
        return f"FAILED: rate is not {RATES[arm][1]}"
    if "${RESUME}" not in code or d["spec"]["backoffLimit"] not in (3, 50):
        return "FAILED: did not inherit auto-resume"
    dst.write_text(text)
    return f"derived from s{base_seed} (lr={RATES[arm][1]}, seed={seed})"


# ---------------------------------------------------------------------------
# THE RANDOM-LABEL CONTROL'S OTHER PARTITION DRAWS.
#
# derive_seed cannot express these. It holds the arm fixed and moves the seed;
# a second draw is a different ARM CONFIG (configs/arms/RAND_d<N>.yaml), so the
# config path, --arm and the sidecar name move with it. The source is the one
# control spec that has actually trained, and everything it does not name --
# pin, rate, budget, loader flags, memory, GPU model, backoffLimit, image, the
# manifest call, the resume guard -- is inherited byte for byte.
#
# NO BLANKET RENAME. experiments/RUNS.csv `spec-copy-hazard` records what a
# blanket L162 -> RAND_d1 rename did to a copied spec: it rewrote measured
# numbers. Every site below is named and counted, the leftover mentions of the
# source arm must all sit on comment lines, and none may survive.
RAND_TRAIN = "job-mtx-rand-d1-s1b-raunav.yaml"
RAND_MAKEWEIGHT = "job-mtx-makeweight-rand-raunav.yaml"

PAIRING = """\
  # DRAW {d} OF 3, SEED INDEX {s}. DERIVED from job-mtx-rand-d1-s1b-raunav.yaml
  # by scripts/build_mtx_launch.py --derive-draws -- do not hand-edit.
  #
  # WHY SEED INDEX {s} AND NOT 1 AGAIN. Each control run is paired with the
  # 17-class semantic model of the SAME seed index: draw 1 with mtx-r16q1-s1,
  # draw 2 with mtx-r16q1-s2, draw 3 with mtx-r16q1-s3. The four RNG streams
  # (trunk_init, head_init, data_sampling, dropout) derive from that index
  # exactly as for every other run, so within a pair the data order and the
  # initialisation are shared and the partition is the only difference. The
  # three paired differences then sample partition-draw and seed variation
  # TOGETHER. They do not separate the two; that would take several draws at
  # one seed.
  #
  # Differs from the draw-1 spec at EXACTLY these sites, and
  # tests/test_rand_draw_specs.py diffs the two files line by line: this
  # block, metadata.name, RUN_ID (hence OUT), both --seed values, --arm, the
  # arm config path (CFG, --data-config x2, the FATAL text), the sidecar name
  # (SIDECAR, SRC), --tensorboard, and the arm's name where a comment cites
  # it. Same pin -- configs/arms/RAND_d{d}.yaml is byte-identical at that tag.
  #
  # ITS SIDECAR IS ITS OWN: RAND_d{d}.<md5>.auto.yaml, written by
  # job-mtx-makeweight-rand-d{d}-raunav.yaml, which must COMPLETE before this
  # is applied. The histograms inside do not depend on the partition (I2);
  # the filename and the labels block it carries do.
  #
"""

MAKEWEIGHT_DERIVED = (
    "  # DERIVED from job-mtx-makeweight-raunav.yaml on 2026-08-22, changing four\n"
    "  # things: the name, the clone ref (arms-s1 no longer exists on the remote;\n"
    "  # mtx-s1.29 is the tag carrying the SHARE-matched configs/arms/RAND_d1.yaml), the arm\n"
    "  # list (RAND_d1 only -- the tree arms' sidecars exist, are in use by running\n"
    "  # jobs, and recomputing them risks loss for no gain), and the final check.\n",
    "  # DERIVED from job-mtx-makeweight-rand-raunav.yaml, the draw-1 pass, by\n"
    "  # scripts/build_mtx_launch.py --derive-draws -- do not hand-edit. Changed:\n"
    "  # the name, the clone ref ({pin}, the tag the control's training specs\n"
    "  # clone), the arm (RAND_d{d} only -- every other sidecar exists, and\n"
    "  # recomputing one risks loss for no gain), and the file the hash lands in.\n")

MAKEWEIGHT_PIN = (
    "          # Pinned to mtx-s1.29, the tag carrying the SHARE-matched control\n"
    "          # (DECISIONS_PENDING item 24, resolved 2026-09-08). The earlier\n"
    "          # mtx-s1.24 sidecar was built from the COUNT-matched RAND_d1.yaml and\n"
    "          # is keyed on md5 07e850fb; the arm is now 3e293063, so that sidecar\n"
    "          # is orphaned rather than wrong -- it is left in place, and this pass\n"
    "          # writes the one the training spec's guard will look for.\n",
    "          # Pinned to {pin}, read from the draw-1 training spec's REPO_REF.\n"
    "          # configs/arms/RAND_d{d}.yaml is byte-identical at that tag and in the\n"
    "          # tree this spec was generated from -- md5\n"
    "          # {md5} -- so this pass writes the name\n"
    "          # the training spec's guard will look for.\n")


def _substitute(text: str, subs, d: int, cites: int) -> str:
    """Apply each (old, new, expected count) in order, then rename the `cites`
    leftover mentions of the source arm, which must ALL be comments. Raises
    ValueError on any surprise, before anything is written."""
    for old, new, n in subs:
        if text.count(old) != n:
            raise ValueError(f"expected {n} {old!r}, found {text.count(old)}")
        text = text.replace(old, new)
    left = [ln for ln in text.splitlines() if "RAND_d1" in ln]
    live = [ln for ln in left if not ln.lstrip().startswith("#")]
    if live:
        raise ValueError(f"RAND_d1 survives on an executed line: {live[0].strip()}")
    if text.count("RAND_d1") != cites:
        raise ValueError(f"expected {cites} comment cites of RAND_d1, found "
                         f"{text.count('RAND_d1')}")
    return text.replace("RAND_d1", f"RAND_d{d}")


def _same_outside(a: dict, b: dict) -> bool:
    """True when two freshly parsed Jobs agree on everything but the name and
    script. Consumes its arguments."""
    def strip(j):
        j["metadata"].pop("name")
        j["spec"]["template"]["spec"]["containers"][0].pop("args")
        return j
    return strip(a) == strip(b)


def derive_draw(draw: int, seed: int) -> list[str]:
    """Write the training spec AND the sidecar spec for one further draw."""
    if draw == 1:
        return ["FAILED: draw 1 is the source spec"]
    arm = f"RAND_d{draw}"
    cfg = ROOT / "configs" / "arms" / f"{arm}.yaml"
    pair = K8S / f"job-mtx-r16_q1-s{seed}-raunav.yaml"
    for need in (cfg, pair):
        if not need.exists():
            return [f"FAILED: {need.name} not found"]

    src_train = (K8S / RAND_TRAIN).read_text()
    m = re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', src_train)
    if not m:
        return ["FAILED: no REPO_REF in the draw-1 spec"]
    pin = m.group(1)
    at_pin = subprocess.run(["git", "-C", str(ROOT), "show", f"{pin}:configs/arms/{arm}.yaml"],
                            capture_output=True)
    if at_pin.returncode != 0 or at_pin.stdout != cfg.read_bytes():
        return [f"FAILED: configs/arms/{arm}.yaml at {pin} is not the working "
                f"tree's; the pod would train (and name its sidecar after) a "
                f"different file. Repin deliberately, do not derive."]
    md5 = hashlib.md5(cfg.read_bytes()).hexdigest()

    run = f"mtx-rand-d{draw}-s{seed}"
    jobs = {
        # name: (source text, named sites, comment cites of the source arm)
        f"job-{run}-raunav.yaml": (src_train, [
            ("  # contraction tree.\n  #\n",
             "  # contraction tree.\n  #\n" + PAIRING.format(d=draw, s=seed), 1),
            ("RAND_d1 (K = 17), seed 1.", f"{arm} (K = 17), seed {seed}.", 1),
            ("name: mtx-rand-d1-s1b-raunav", f"name: {run}-raunav", 1),
            ("RUN_ID=mtx-rand-d1-s1b", f"RUN_ID={run}", 1),
            ("--tensorboard mtx_RAND_d1_s1b", f"--tensorboard mtx_{arm}_s{seed}", 1),
            ("--seed 1 \\", f"--seed {seed} \\", 2),
            ("--arm RAND_d1 \\", f"--arm {arm} \\", 1),
            ("configs/arms/RAND_d1.${MD5}.auto.yaml", f"configs/arms/{arm}.${{MD5}}.auto.yaml", 1),
            ("makeweight/RAND_d1.${MD5}.auto.yaml", f"makeweight/{arm}.${{MD5}}.auto.yaml", 1),
            ("configs/arms/RAND_d1.yaml", f"configs/arms/{arm}.yaml", 5),
        ], 4),
        f"job-mtx-makeweight-rand-d{draw}-raunav.yaml": ((K8S / RAND_MAKEWEIGHT).read_text(), [
            (MAKEWEIGHT_DERIVED[0], MAKEWEIGHT_DERIVED[1].format(pin=pin, d=draw), 1),
            (MAKEWEIGHT_PIN[0], MAKEWEIGHT_PIN[1].format(pin=pin, d=draw, md5=md5), 1),
            ("name: mtx-makeweight-rand2-raunav", f"name: mtx-makeweight-rand-d{draw}-raunav", 1),
            ("--branch mtx-s1.29 \\", f"--branch {pin} \\", 1),
            ("run_arm RAND_d1  17", f"run_arm {arm}  17", 1),
            ('ARM = "RAND_d1"', f'ARM = "{arm}"', 1),
            ("hist_hashes_RAND_d1.json", f"hist_hashes_{arm}.json", 1),
        ], 4),
    }
    out = []
    for name, (text, subs, cites) in jobs.items():
        dst = K8S / name
        if dst.exists():
            out.append(f"{name}   exists, not regenerated")
            continue
        try:
            new = _substitute(text, subs, draw, cites)
        except ValueError as e:
            out.append(f"{name}   FAILED: {e}")
            continue
        if not _same_outside(yaml.safe_load(text), yaml.safe_load(new)):
            out.append(f"{name}   FAILED: differs from its source outside the "
                       f"job name and the script")
            continue
        dst.write_text(new)
        out.append(f"{name}   derived ({arm}, pin {pin}, config md5 {md5[:8]})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unbracketed", action="store_true",
                    help="emit arms whose LR is not bracketed on both sides")
    ap.add_argument("--derive-draws", nargs="+", default=[], metavar="DRAW:SEED",
                    help="derive the random control's further partition draws "
                         "(training spec + reweighting-sidecar spec) from the "
                         "launched draw-1 specs, e.g. 2:2 3:3")
    ap.add_argument("--derive-seeds", nargs="+", default=[],
                    help="ARM:seed,seed,... derive extra seeds from that arm's "
                         "seed-1 spec. Refused for arms whose rate is not "
                         "bracketed, because the rate is baked in at write time.")
    args = ap.parse_args()

    if args.derive_draws:
        rc = 0
        for spec in args.derive_draws:
            draw, _, seed = spec.partition(":")
            for r in derive_draw(int(draw), int(seed)):
                print(r)
                rc |= "FAILED" in r
        return rc

    if args.derive_seeds:
        rc = 0
        for spec in args.derive_seeds:
            arm, _, seeds = spec.partition(":")
            if arm not in RATES:
                print(f"{arm}: FAILED: unknown arm")
                rc = 1
                continue
            if not RATES[arm][2] and not args.allow_unbracketed:
                print(f"{arm}: REFUSED -- rate not bracketed, so every derived "
                      f"seed would bake in a rate the sweep has not confirmed. "
                      f"{RATES[arm][3]}")
                rc = 1
                continue
            seeds, _, base_tag = seeds.partition(":")
            for sd in [int(x) for x in seeds.split(",") if x]:
                r = derive_seed(arm, 1, sd, base_tag or None)
                print(f"job-mtx-{arm.lower()}-s{sd}-raunav.yaml   {r}")
                rc |= r.startswith("FAILED")
        return rc

    rc = 0
    for arm, (k, rate, bracketed, why) in RATES.items():
        if not bracketed and not args.allow_unbracketed:
            print(f"{arm:8s} SKIPPED -- rate not bracketed. {why}")
            continue
        for seed in SEEDS:
            p = K8S / f"job-mtx-{arm.lower()}-s{seed}-raunav.yaml"
            if not p.exists():
                print(f"{arm:8s} FAILED: {p.relative_to(ROOT)} not found")
                rc = 1
                continue
            r = patch(p, arm, k, rate)
            print(f"{p.name:34s} {r}")
            if r.startswith("FAILED"):
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
