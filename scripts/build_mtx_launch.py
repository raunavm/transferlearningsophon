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
    # bar (docs/PRESPEC_2026-09.md:60-66, which REPLACED item 17's macro AUC
    # >= 0.95 / Rej_bb >= 100 because that bar assumed a frozen readout this
    # model has no trained class-attention blocks for: fine-tuning from it must
    # beat training from scratch at 1e3 and 1e4 jets by more than the
    # fine-tuning-seed spread, and at 1e6 jets land within 0.02 macro AUC of the
    # supervised models), the documented tuning
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


# ============================================================ v2 specs (audit 2026-09-29)
# v2 pretraining runs experiments/MTX/pretrain_v2.py: the Sophon loader
# (--fetch-step 1.0 --data-split-num 200, 5 workers), a fresh epoch-seeded stream
# every epoch, the fixed validation sample, full-state auto-resume and per-epoch
# metrics and stream records. One function emits the spec of any (arm, run index,
# GPU product); nothing here applies a spec.
V2_ROOT = "/data/results/mtx_v2"
V2_GRID = ROOT / "configs" / "arms" / "v2_grid.json"
V2_IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"
V2_GPU = "NVIDIA-GeForce-RTX-3090"
V2_RATE = "5e-4"                    # RATES: every arm of the ladder trains at 5e-4
V2_EPOCHS = 80
V2_SAMPLES = 10_240_000
V2_WINDOW = "--data-fraction 0.2"
V2_LOADER = "--num-workers 5 --fetch-step 1.0 --data-split-num 200 " + V2_WINDOW
# Held-out-family (LOFO) arms, PI 2026-10-01: the same 10,240,000 jets x 80 epochs, but each
# epoch reads a third of every file (each row at most once per three epochs), so the epoch fits
# inside its window with the excluded family's ~22% removed; an integer window count, no float
# drift (stream_v2.window_of). Every other arm keeps --data-fraction 0.2 unchanged.
V2_LOFO_WINDOWS = 3
# --data-fraction 0.2: an epoch reads a random fifth of EVERY file (stream_v2.cycle_of).
# A full pass over the training files yields ~52.8M jets (dry run, 52,800 per fetch x
# 200 splits x 5 workers), so a fifth, ~10.6M, covers the 10,240,000-jet epoch.
# Measured (RUNS.csv): the loader alone over a full 10,240,000-jet epoch, 5 workers, 8 CPUs,
# at mtx-s1.72 (--data-fraction 0.2) peaks at 18.8 GB anon and delivers 3,457 jets/s
# (mtx2-loader-memprobe-s172; 29.0 GB and 4,716 jets/s without the window, mtx-s1.69);
# training on an RTX 3090 is GPU-bound at 2,190-2,320 jets/s and the whole pod peaked at
# 28.3 GB in the smoke (mtx2-smoke-3090, larger fetches). 48Gi is 1.7x that peak.
# Best-validation metric, draft amendment A8: Sophon's rule, the reweighted accuracy. weaver
# 0.4.17 train.py:248 and hqucms/weaver-core@c97de3c train.py:289-291 build the training-time
# validation set with for_training=True (reweighted); c97de3c train.py:1022-1023 defaults
# --data-config-val to the training config and train_sophon.sh passes none.
V2_SELECT = "acc"
# Checkpoint retention (PI, 2026-10-01): epochs 70-79, the best epoch, the 70-79 weight
# average and the newest resume file (pretrain_v2.py prune, write_weight_average).
V2_KEEP = "window"
# GPU product per run index (PI, 2026-10-01): run k of every arm on one product, so a
# seed pair never mixes products (I7). Indices 4-5 may move to L40 only if the L40
# numerics check passes; the run directories keep the contract name mtx-<slug>-s<k>
# whatever the product.
V2_GPU_BY_RUN = {1: V2_GPU, 2: V2_GPU, 3: V2_GPU, 4: V2_GPU, 5: V2_GPU}
V2_CPU = "8"
V2_MEM = "48Gi"
V2_BACKOFF = 20                     # counted failures; evictions are ignored
EXIT_HALT = 42
V2_BAD_NODES = ("ry-gpu-01.sdsc.optiputer.net", "ry-gpu-03.sdsc.optiputer.net",
                "nautilus-ext-gpu01.fullerton.edu", "hcc-chase-shor-c4705.unl.edu",
                "hcc-chase-shor-c4709.unl.edu", "hcc-chase-shor-c4715.unl.edu",
                "k8s-chase-ci-07.calit2.optiputer.net", "nrp-fiona-001.sdmz.amnh.org")
TRAIN_GLOBS = ("Res2P:/jc2/jet_data/Res2P_{0000..0199}.parquet",
               "Res34P:/jc2/jet_data/Res34P_{0000..0859}.parquet",
               "QCD:/jc2/jet_data/QCD_{0000..0279}.parquet")
# The fixed validation sample (decided 2026-09-29): 25 validation-split files,
# every selected row, the same order every epoch. Fine-tuning v2 draws only from
# the other validation-split files (Res2P_0204-0249, Res34P_0876-1074, QCD_0285-0349).
VAL_GLOBS = ("/jc2/jet_data/Res2P_{0200..0203}.parquet",
             "/jc2/jet_data/Res34P_{0860..0875}.parquet",
             "/jc2/jet_data/QCD_{0280..0284}.parquet")
N_TRAIN_FILES, N_VAL_FILES = 1340, 25
ARCH = {"classification": "experiments/MTX/ParT_sophon_arch_mtx.py",
        "classification+mass": "experiments/MTX/ParT_sophon_arch_mass.py",
        "mpm": "experiments/MTX/ParT_sophon_arch_mpm.py"}
# Used only while configs/arms/v2_grid.json does not exist.
V2_DEFAULT_ARMS = [
    {"name": "L188", "config": "configs/arms/L188.yaml", "num_classes": 188, "mass_lambda": None,
     "runs": 5, "objective": "classification", "extra_selection": None},
    {"name": "L162", "config": "configs/arms/L162.yaml", "num_classes": 162, "mass_lambda": None,
     "runs": 5, "objective": "classification", "extra_selection": None},
    {"name": "R42_Q1", "config": "configs/arms/R42_Q1.yaml", "num_classes": 43, "mass_lambda": None,
     "runs": 5, "objective": "classification", "extra_selection": None},
    {"name": "R16_Q1", "config": "configs/arms/R16_Q1.yaml", "num_classes": 17, "mass_lambda": None,
     "runs": 5, "objective": "classification", "extra_selection": None},
    {"name": "L162_MASS", "config": "configs/arms/L162_MASS.yaml", "num_classes": 162, "mass_lambda": 5.0,
     "runs": 5, "objective": "classification+mass", "extra_selection": None},
    {"name": "R16_Q1_MASS", "config": "configs/arms/R16_Q1_MASS.yaml", "num_classes": 17, "mass_lambda": 5.0,
     "runs": 5, "objective": "classification+mass", "extra_selection": None},
]


def v2_arms() -> list:
    """The v2 grid (configs/arms/v2_grid.json, scripts/build_v2_arms.py), else the defaults."""
    if V2_GRID.exists():
        import json
        return json.loads(V2_GRID.read_text())["arms"]
    return V2_DEFAULT_ARMS


def v2_run_id(arm: str, run: int, gpu: str = V2_GPU) -> str:
    """The run directory under V2_ROOT: mtx-<arm, lower case, no underscores>-s<seed>,
    the name the fine-tuning jobs read (scratchpad ft_v2_checkpoint_contract.txt)."""
    slug = re.sub(r"[^a-z0-9]", "", arm.lower())
    tail = "" if gpu == V2_GPU else "-" + gpu_short(gpu)
    return f"mtx-{slug}-s{run}{tail}"


def v2_job_name(arm: str, run: int, gpu: str = V2_GPU) -> str:
    """mtx2-...: v1's jobs are called mtx-<slug>-s<seed>-raunav and some still exist."""
    return "mtx2-" + v2_run_id(arm, run, gpu)[len("mtx-"):] + "-raunav"


def gpu_short(gpu: str) -> str:
    return {"NVIDIA-GeForce-RTX-3090": "3090", "NVIDIA-L40": "l40", "NVIDIA-RTX-A6000": "a6000",
            "NVIDIA-A40": "a40"}.get(gpu) or re.sub(r"[^a-z0-9]", "", gpu.lower())[-12:]


def v2_spec(arm: dict, run: int, gpu: str = V2_GPU, *, tag: str, **kw) -> tuple:
    """(job name, spec text) for one v2 pretraining run.

    `arm` is one entry of configs/arms/v2_grid.json ("name", "config",
    "num_classes", "mass_lambda", "objective", "extra_selection"). The run index
    is the master seed, so run k of every arm draws the same stream.
    """
    run_id = kw.pop("run_id", None) or v2_run_id(arm["name"], run, gpu)
    job = kw.pop("job", None) or v2_job_name(arm["name"], run, gpu)
    cpu, mem = kw.pop("cpu", V2_CPU), kw.pop("mem", V2_MEM)
    script = v2_script(arm, run, run_id=run_id, **kw)
    title = f"v2 PRETRAINING -- {arm['name']}, run {run} (master seed {run}), {gpu}."
    return job, v2_job(job, [script], gpu, tag, title, cpu=cpu, mem=mem)


def v2_script(arm: dict, run: int, *, run_id: str, out_root: str = V2_ROOT,
              epochs: int = V2_EPOCHS, samples: int = V2_SAMPLES, deterministic: bool = False,
              kill_after_epoch: int | None = None) -> str:
    """The bash for one run, after the clone. It runs in a subshell of the pod
    script, so `exit` leaves this run only. kill_after_epoch (smoke only)
    SIGKILLs the trainer during the epoch after that one, then restarts it."""
    obj = arm["objective"]
    head = ""
    if obj != "mpm":
        head = f" -o num_classes {int(arm['num_classes'])} -o fc_params '[(512,0.1)]'"
    extra = ""
    if arm.get("mass_lambda") is not None:
        extra += f" --mass-lambda {float(arm['mass_lambda'])}"
    if obj == "mpm":
        extra += " --mpm --mpm-mask-rate 0.40"
    if arm.get("extra_selection"):
        if "'" in arm["extra_selection"]:
            raise ValueError("extra_selection must not contain a single quote")
        extra += f" --extra-selection '{arm['extra_selection']}'"
    if deterministic:
        extra += " --deterministic"
    k = 0 if obj == "mpm" else int(arm["num_classes"])
    cfg = arm["config"]
    window = f"--data-windows {V2_LOFO_WINDOWS}" if arm.get("extra_selection") else V2_WINDOW
    loader = V2_LOADER.replace(V2_WINDOW, window)
    train_cmd = (
        "python3 experiments/MTX/pretrain_v2.py --seed ${SEED} --out ${OUT}"
        f" --data-train {' '.join(TRAIN_GLOBS)} --data-val {' '.join(VAL_GLOBS)}"
        f" --data-config ${{CFG}} --network-config {ARCH[obj]}{head}"
        f" --use-amp --batch-size 512 --start-lr {V2_RATE} --num-epochs {epochs}"
        f" --samples-per-epoch {samples} {loader}{extra} --keep-checkpoints {V2_KEEP} --select-on {V2_SELECT}")
    if kill_after_epoch is None:
        run_block = f"PYTHONUNBUFFERED=1 {train_cmd} 2>&1 | tee -a ${{OUT}}/train.log\n"
    else:
        e = int(kill_after_epoch)
        run_block = (
            "if [ ! -f ${OUT}/KILLED ]; then\n"
            f"  PYTHONUNBUFFERED=1 {train_cmd} >> ${{OUT}}/train.log 2>&1 &\n"
            "  BG=$!\n"
            f"  until [ -f ${{OUT}}/net_epoch-{e}_resume.pt ] || ! kill -0 ${{BG}} 2>/dev/null; do sleep 5; done\n"
            "  sleep 45\n"
            "  kill -9 ${BG} || true      # the trainer only; its loader workers exit with it\n"
            "  wait ${BG} || true\n"
            f"  echo \"killed during epoch {e + 1} at $(date -u +%FT%TZ)\" | tee ${{OUT}}/KILLED\n"
            "fi\n"
            f"PYTHONUNBUFFERED=1 {train_cmd} 2>&1 | tee -a ${{OUT}}/train.log\n")
    return f"""(
set -euo pipefail
HALT={EXIT_HALT}
RUN_ID={run_id}
SEED={int(run)}
OUT={out_root}/${{RUN_ID}}
mkdir -p ${{OUT}}/attempts
if [ -f ${{OUT}}/DONE ]; then echo "${{RUN_ID}} is DONE"; exit 0; fi
USE=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
echo "/data at ${{USE}}%"
[ "${{USE}}" -le 85 ] || {{ echo "FATAL: /data at ${{USE}}%, above 85%"; exit ${{HALT}}; }}
# A FAILED attempt leaves attempts/failed-* naming the last complete epoch (the
# EXIT trap; an evicted pod is SIGKILLed, runs no trap and is not counted). Two
# failed attempts with no epoch completed in between stop the job.
# LAST = pretrain_v2.latest_complete_epoch: the newest resume file with its state file beside it.
LAST=-1
for f in ${{OUT}}/net_epoch-*_resume.pt; do
  [ -e "$f" ] || continue
  n=$(basename "$f" | sed 's/^net_epoch-\\([0-9]*\\)_resume\\.pt$/\\1/')
  if [ -f ${{OUT}}/net_epoch-${{n}}_state.pt ] && [ "$n" -gt "${{LAST}}" ]; then LAST=$n; fi
done
NF=$(ls ${{OUT}}/attempts | grep -c -- "-e${{LAST}}$" || true)
[ "${{NF}}" -lt 2 ] || {{ echo "FATAL: ${{NF}} failed attempts after epoch ${{LAST}}"; exit ${{HALT}}; }}
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)
trap 'rc=$?; if [ ${{rc}} -ne 0 ] && [ ${{rc}} -ne ${{HALT}} ]; then echo "rc=${{rc}} pod=${{POD_NAME}} node=${{NODE_NAME}}" > ${{OUT}}/attempts/failed-${{ATTEMPT}}-e${{LAST}}; fi' EXIT
echo "${{ATTEMPT}} pod=${{POD_NAME}} node=${{NODE_NAME}} gpu=${{GPU_PRODUCT}} after_epoch=${{LAST}} ref=${{REPO_REF}}" >> ${{OUT}}/attempts.log
TRAIN_FILES=({' '.join(g.split(':', 1)[1] for g in TRAIN_GLOBS)})
VAL_FILES=({' '.join(VAL_GLOBS)})
n_present () {{ local n=0; for f in "$@"; do [ -f "$f" ] && n=$((n+1)); done; echo $n; }}
N_TRAIN=$(n_present "${{TRAIN_FILES[@]}}"); N_VAL=$(n_present "${{VAL_FILES[@]}}")
[ "${{N_TRAIN}}" -eq {N_TRAIN_FILES} ] || {{ echo "FATAL: ${{N_TRAIN}} of {N_TRAIN_FILES} training files present"; exit 1; }}
[ "${{N_VAL}}" -eq {N_VAL_FILES} ] || {{ echo "FATAL: ${{N_VAL}} of {N_VAL_FILES} validation files present"; exit 1; }}
CFG={cfg}
MD5=$(md5sum ${{CFG}} | cut -d' ' -f1)
SIDECAR=${{CFG%.yaml}}.${{MD5}}.auto.yaml
SRC=/data/results/mtx/makeweight/$(basename ${{SIDECAR}})
[ -f "${{SRC}}" ] || {{ echo "FATAL: no reweighting sidecar ${{SRC}}: run the make_weight job for ${{CFG}}"; exit ${{HALT}}; }}
cp "${{SRC}}" "${{SIDECAR}}"
python3 - "${{CFG}}" "${{SIDECAR}}" <<'PY' || {{ echo "FATAL: ${{SIDECAR}} is not ${{CFG}} plus reweighting histograms"; exit ${{HALT}}; }}
import sys
sys.path.insert(0, "experiments/MTX")
import stream_v2 as sv
bad = sv.sidecar_mismatch(sys.argv[1], sys.argv[2])
print("sidecar check:", "differs in %s" % bad if bad else "its config plus reweight_hists")
sys.exit(1 if bad else 0)
PY
cp "${{SIDECAR}}" ${{OUT}}/
sha256sum "${{SIDECAR}}" > ${{OUT}}/reweight_sidecar.sha256
MANIFEST=${{OUT}}/run_manifest.json
[ -f ${{MANIFEST}} ] && MANIFEST=${{OUT}}/run_manifest.${{ATTEMPT}}.json
python3 scripts/write_run_manifest.py --driver pretrain_v2 --run-id ${{RUN_ID}} --arm {arm['name']} \\
  --num-classes {k} --seed ${{SEED}} --data-config ${{CFG}} --samples-per-epoch {samples} \\
  --num-epochs {epochs} --batch-size 512{f" --lambda-mass {float(arm['mass_lambda'])}" if arm.get('mass_lambda') is not None else ""}{" --mpm-mask-rate 0.40" if obj == "mpm" else ""} \\
  --num-workers 5 --data-split-num 200 --fetch-step 1.0 {window} --keep-checkpoints {V2_KEEP} \\
  --select-on {V2_SELECT}{f" --extra-selection '{arm['extra_selection']}'" if arm.get('extra_selection') else ""} \\
  --val-files "${{VAL_FILES[@]}}" --out ${{MANIFEST}}
{run_block})
"""


def v2_job(name: str, scripts: list, gpu, tag: str, title: str, *,
           cpu: str = V2_CPU, mem: str = V2_MEM, pre: str = "") -> str:
    """A Job that clones `tag` and runs each script in turn, with the retry policy
    of scripts/build_ft_jobs.py (commit 3cb4d7a): evictions ignored, exit 42 fails
    the Job at once, other failures counted up to V2_BACKOFF."""
    if len(name) > 63:
        raise ValueError(f"{name}: longer than a Kubernetes name allows")
    exclude = ", ".join(f'"{n}"' for n in V2_BAD_NODES)
    script = ("set -euo pipefail\n"
              'git clone --depth 1 --branch "${REPO_REF}" '
              "https://github.com/raunavm/transferlearningsophon.git /workspace/transferlearningsophon\n"
              "cd /workspace/transferlearningsophon\n"
              "pip install --no-cache-dir -q pyarrow || exit 1\n" + pre + "".join(scripts))
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # {title}
  # GENERATED by scripts/build_mtx_launch.py (v2_job) -- do not hand-edit.
  # experiments/MTX/pretrain_v2.py: Sophon loader, epoch-seeded streams, fixed
  # validation sample, full-state auto-resume, per-epoch metrics and stream records.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: {V2_BACKOFF}
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [{EXIT_HALT}] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {V2_IMAGE}
        command: ["/bin/bash", "-c"]
        env:
        - name: GPU_PRODUCT
          value: "{gpu if isinstance(gpu, str) else ','.join(gpu)}"
        - name: NODE_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: spec.nodeName }} }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        - name: REGION
          value: "us-west"
        - name: REPO_REF
          value: "{tag}"
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{mem}", cpu: "{cpu}", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
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
                operator: In
                values: [{", ".join(f'"{g}"' for g in ([gpu] if isinstance(gpu, str) else gpu))}]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{exclude}]
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


SMOKE_ROOT = V2_ROOT + "/smoke"
SMOKE_EPOCHS, SMOKE_SAMPLES = 3, 200_000        # the gate: at most 3 epochs x 200,000 jets
NUMERICS_GPUS = ("NVIDIA-L40", "NVIDIA-RTX-A6000", "NVIDIA-A40")


def _arm(name: str) -> dict:
    return next(a for a in v2_arms() if a["name"] == name)


def v2_smoke_specs(tag: str, deterministic: bool = False) -> dict:
    """The smoke and GPU-numerics jobs (3 epochs x 200,000 jets, R16_Q1 run 1):
    on one RTX 3090, in turn, A (uninterrupted), B (killed during epoch 2, then
    resumed), C (the 188-class output, same seed) and A2 (A again); and A once
    on each of NUMERICS_GPUS. Before training, the 3090 pod runs the v2 tests
    under the image's weaver 0.4.17."""
    sfx = "-det" if deterministic else ""
    kw = dict(out_root=SMOKE_ROOT, epochs=SMOKE_EPOCHS, samples=SMOKE_SAMPLES, deterministic=deterministic)
    a, c = _arm("R16_Q1"), _arm("L188")
    scripts = [v2_script(a, 1, run_id=f"smoke{sfx}-a", **kw),
               v2_script(a, 1, run_id=f"smoke{sfx}-b", kill_after_epoch=1, **kw),
               v2_script(c, 1, run_id=f"smoke{sfx}-c", **kw),
               v2_script(a, 1, run_id=f"smoke{sfx}-a2", **kw)]
    pre = (f"mkdir -p {SMOKE_ROOT}\n"
           "pip install --no-cache-dir -q pytest || exit 1\n"
           "python3 -m pytest tests/test_pretrain_v2.py tests/test_stream_ids.py -q -p no:cacheprovider "
           f"> {SMOKE_ROOT}/pytest_weaver0417{sfx}.txt 2>&1 || true\n"
           f"tail -3 {SMOKE_ROOT}/pytest_weaver0417{sfx}.txt\n") if not deterministic else f"mkdir -p {SMOKE_ROOT}\n"
    out = {}
    name = f"mtx2-smoke{sfx}-3090-raunav"
    out[f"job-{name}.yaml"] = v2_job(name, scripts, V2_GPU, tag,
                                     "v2 SMOKE on RTX 3090: A, B (kill + resume), C (188 outputs), A2.", pre=pre)
    for g in NUMERICS_GPUS:
        name = f"mtx2-smoke{sfx}-a-{gpu_short(g)}-raunav"
        out[f"job-{name}.yaml"] = v2_job(name, [v2_script(a, 1, run_id=f"smoke{sfx}-a-{gpu_short(g)}", **kw)],
                                         g, tag, f"v2 SMOKE, configuration A on {g} (GPU numerics check).",
                                         pre=f"mkdir -p {SMOKE_ROOT}\n")
    return out


# GPUs of 24 GB or more in us-west that the cu121 image supports (not Blackwell).
ANY_GPUS = ("NVIDIA-GeForce-RTX-3090", "NVIDIA-L40", "NVIDIA-RTX-A6000", "NVIDIA-A40", "NVIDIA-L4",
            "NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "NVIDIA-H100-80GB-HBM3",
            "Tesla-V100-SXM2-32GB", "NVIDIA-TITAN-RTX")


def v2_det_any_spec(tag: str) -> tuple:
    """The deterministic resume check on whichever ANY_GPUS product is free: A, B
    (SIGKILL during epoch 2, resumed) and A2 in one pod, so all three share one
    GPU. It asks only whether a resume repeats the run bit for bit once kernel
    choice is fixed, which does not depend on the product."""
    kw = dict(out_root=SMOKE_ROOT, epochs=SMOKE_EPOCHS, samples=SMOKE_SAMPLES, deterministic=True)
    a = _arm("R16_Q1")
    scripts = [v2_script(a, 1, run_id="smoke-detany-a", **kw),
               v2_script(a, 1, run_id="smoke-detany-b", kill_after_epoch=1, **kw),
               v2_script(a, 1, run_id="smoke-detany-a2", **kw)]
    name = "mtx2-smoke-detany-raunav"
    return name, v2_job(name, scripts, ANY_GPUS, tag,
                        "v2 SMOKE, deterministic resume check on any free 24 GB GPU: A, B (kill + resume), A2.",
                        pre=f"mkdir -p {SMOKE_ROOT}\n")


def v2_dryrun_spec(tag: str, full_columns: bool = False, label: str = "", arm: str = "R16_Q1",
                   epochs: int | None = None) -> tuple:
    """The CPU loader-only dry run: 20 epochs x 10,240,000 jets through the v2
    training stream of `arm` with column projection, or (full_columns) one epoch with
    every input column finalised, for memory and loader throughput. `label` names a
    repeat (job name and output file), e.g. the tag it checks. A held-out-family arm
    runs with its --extra-selection and --data-windows, as its training spec does."""
    a = _arm(arm)
    kind = "memprobe" if full_columns else "dryrun"
    name = f"mtx2-loader-{kind}{'-' + label if label else ''}-raunav"
    epochs = epochs or (1 if full_columns else 20)
    out = f"{V2_ROOT}/loader_dryrun/{kind}{'_' + label if label else ''}_seed1.json"
    window = V2_WINDOW
    if a.get("extra_selection"):
        if "'" in a["extra_selection"]:
            raise ValueError("extra_selection must not contain a single quote")
        window = f"--data-windows {V2_LOFO_WINDOWS} --extra-selection '{a['extra_selection']}'"
    cmd = (f"CFG={a['config']}\n"
           "MD5=$(md5sum ${CFG} | cut -d' ' -f1)\n"
           "SIDECAR=${CFG%.yaml}.${MD5}.auto.yaml\n"
           "cp /data/results/mtx/makeweight/$(basename ${SIDECAR}) ${SIDECAR}\n"
           "USE=$(df --output=pcent /data | tail -1 | tr -dc 0-9)\n"
           'echo "/data at ${USE}%"\n'
           '[ "${USE}" -le 85 ] || { echo "FATAL: /data at ${USE}%, above 85%"; exit 1; }\n'
           f"mkdir -p {V2_ROOT}/loader_dryrun\n"
           f"PYTHONUNBUFFERED=1 python3 experiments/MTX/loader_dryrun.py --seed 1 --epochs {epochs} "
           f"--samples-per-epoch {V2_SAMPLES} --num-workers 5 --data-split-num 200 --fetch-step 1.0 {window} "
           f"--data-config ${{CFG}} --data-train {' '.join(TRAIN_GLOBS)} "
           f"{'--full-columns ' if full_columns else ''}--out {out} 2>&1 | tee {out[:-5]}.log\n")
    exclude = ", ".join(f'"{n}"' for n in V2_BAD_NODES)
    body = "\n".join(("          " + ln) if ln else "" for ln in (
        "set -euo pipefail\n"
        'git clone --depth 1 --branch "${REPO_REF}" '
        "https://github.com/raunavm/transferlearningsophon.git /workspace/transferlearningsophon\n"
        "cd /workspace/transferlearningsophon\n"
        "pip install --no-cache-dir -q pyarrow || exit 1\n" + cmd).splitlines())
    mem = V2_MEM if full_columns else "16Gi"
    return name, f"""apiVersion: batch/v1
kind: Job
metadata:
  # v2 LOADER {'MEMORY PROBE' if full_columns else 'DRY RUN'} -- CPU only, no training (audit item 2).
  # GENERATED by scripts/build_mtx_launch.py v2_dryrun_spec() -- do not hand-edit.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 2
  podFailurePolicy:
    rules:
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {V2_IMAGE}
        command: ["/bin/bash", "-c"]
        env:
        - name: REPO_REF
          value: "{tag}"
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{mem}", cpu: "8", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "{mem}", cpu: "8", ephemeral-storage: "10Gi" }}
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{exclude}]
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


# ------------------------------------------------ reweighting sidecars for new configs
# Every config needs its own make_weight pass: the sidecar is keyed by the md5 of
# the config file. The histograms themselves key on NATIVE label categories, so
# they must equal every earlier arm's (sha256 below, hist_hashes*.json on /data,
# all 13 v1 sidecars). The job checks that and fails if any differs.
V1_SIDECAR_CONFIGS = {"configs/arms/L188.yaml", "configs/arms/L162.yaml", "configs/arms/R42_Q1.yaml",
                      "configs/arms/R16_Q1.yaml", "configs/arms/L162_MASS.yaml",
                      "configs/arms/R16_Q1_MASS.yaml"}
HIST_SHA256 = "546306f0ceb465ca095e8cb573ef3d8f4e754b026b3c6af4cf90b90c197b5a55"
MAKEWEIGHT_ROOT = "/data/results/mtx/makeweight"


def v2_new_configs() -> list:
    """(config, K) for every grid config without a v1 sidecar, in grid order."""
    out, seen = [], set(V1_SIDECAR_CONFIGS)
    for arm in v2_arms():
        if arm["config"] not in seen:
            seen.add(arm["config"])
            out.append((arm["config"], int(arm["num_classes"] or 188)))
    return out


def v2_makeweight_specs(tag: str, pods: int = 2, label: str = "") -> dict:
    """CPU jobs running weaver's make_weight (the v1 recipe: weaver --print over the
    1675 train+val files) for each new config, copying each sidecar out as soon as
    it exists, then checking its histograms against HIST_SHA256."""
    cfgs = v2_new_configs()
    chunks = [cfgs[i::pods] for i in range(pods)]
    exclude = ", ".join(f'"{n}"' for n in V2_BAD_NODES)
    out = {}
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        runs = "".join(f"run_cfg {c} {k}\n" for c, k in chunk)
        script = f"""set -euo pipefail
git clone --depth 1 --branch "${{REPO_REF}}" https://github.com/raunavm/transferlearningsophon.git /workspace/transferlearningsophon
cd /workspace/transferlearningsophon
pip install --no-cache-dir -q pyarrow || exit 1
OUT={MAKEWEIGHT_ROOT}
mkdir -p ${{OUT}}
USE=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USE}}" -le 85 ] || {{ echo "FATAL: /data at ${{USE}}%"; exit 1; }}
ALLSET=""
for i in $(seq -w 0000 0249); do ALLSET="$ALLSET Res2P:/jc2/jet_data/Res2P_$i.parquet"; done
for i in $(seq -w 0000 1074); do ALLSET="$ALLSET Res34P:/jc2/jet_data/Res34P_$i.parquet"; done
for i in $(seq -w 0000 0349); do ALLSET="$ALLSET QCD:/jc2/jet_data/QCD_$i.parquet"; done
N=$(echo $ALLSET | wc -w)
[ "$N" -eq 1675 ] || {{ echo "FATAL: expected 1675 train+val files, got $N"; exit 1; }}
run_cfg () {{
  CFG=$1; K=$2
  MD5=$(md5sum ${{CFG}} | cut -d' ' -f1)
  SIDECAR=${{CFG%.yaml}}.${{MD5}}.auto.yaml
  NAME=$(basename ${{SIDECAR}})
  if [ -f ${{OUT}}/${{NAME}} ]; then echo "=== ${{NAME}} exists"; cp ${{OUT}}/${{NAME}} ${{SIDECAR}}; else
    echo "=== ${{CFG}} K=${{K}} -> ${{NAME}}"
    PYTHONUNBUFFERED=1 weaver --print --gpus "" --data-train $ALLSET --data-config ${{CFG}} \
      --network-config experiments/E1/ParT_sophon_arch_10c.py -o num_classes ${{K}} -o fc_params '[(512,0.1)]' \
      --batch-size 512 --start-lr 5e-4 --num-workers 2 --fetch-by-files --fetch-step 5 \
      > ${{OUT}}/makeweight_$(basename ${{CFG%.yaml}}).log 2>&1
    [ -f "${{SIDECAR}}" ] || {{ echo "FATAL: no ${{SIDECAR}}"; exit 1; }}
    cp -v "${{SIDECAR}}" ${{OUT}}/
  fi
  python3 - "${{SIDECAR}}" "${{CFG}}" <<'PY'
import hashlib, json, sys, yaml
side, cfg = yaml.safe_load(open(sys.argv[1])), yaml.safe_load(open(sys.argv[2]))
h = hashlib.sha256(json.dumps(side["weights"]["reweight_hists"], sort_keys=True, default=str).encode()).hexdigest()
own = side.get("labels") == cfg.get("labels")
print(f"{{sys.argv[2]}}: reweight_hists sha256 {{h}} {{'MATCH' if h == '{HIST_SHA256}' else 'DIFFER'}}; labels block {{'own' if own else 'WRONG'}}")
json.dump({{sys.argv[2]: h}}, open("{MAKEWEIGHT_ROOT}/hist_hashes_" + sys.argv[1].split("/")[-1].split(".")[0] + "_v2.json", "w"), indent=2)
sys.exit(0 if (h == "{HIST_SHA256}" and own) else 1)
PY
}}
{runs}"""
        body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
        name = f"mtx2-makeweight{label}-{i + 1}-raunav"
        out[f"job-{name}.yaml"] = f"""apiVersion: batch/v1
kind: Job
metadata:
  # v2 REWEIGHTING SIDECARS (make_weight, CPU): {", ".join(c for c, _ in chunk)}.
  # GENERATED by scripts/build_mtx_launch.py v2_makeweight_specs() -- do not hand-edit.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 3
  podFailurePolicy:
    rules:
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {V2_IMAGE}
        command: ["/bin/bash", "-c"]
        env:
        - name: REPO_REF
          value: "{tag}"
        args:
        - |
{body}
        resources:
          requests: {{ memory: "48Gi", cpu: "4", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "48Gi", cpu: "4", ephemeral-storage: "10Gi" }}
        volumeMounts:
        - {{ name: jc2,  mountPath: /jc2, readOnly: true }}
        - {{ name: data, mountPath: /data }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{exclude}]
      volumes:
      - name: jc2
        persistentVolumeClaim:
          claimName: tn-pvc-base-jetclass2
          readOnly: true
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""
    return out


# ------------------------------------------------ pre-launch checks (coordinator, 2026-10-01)
def v2_expected_sidecars(tag: str) -> dict:
    """{config: sidecar file name} for every grid config: <name>.<md5>.auto.yaml with the
    md5 of the config as committed at `tag`, the file the pods clone. A working-tree
    config that differs from it is refused (the specs would describe another config)."""
    out = {}
    for arm in v2_arms():
        cfg = arm["config"]
        if cfg in out:
            continue
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{tag}:{cfg}"], capture_output=True,
                              check=True).stdout
        if (ROOT / cfg).read_bytes() != blob:
            raise SystemExit(f"{cfg}: the working tree differs from {tag}")
        out[cfg] = pathlib.Path(cfg).name.replace(".yaml", f".{hashlib.md5(blob).hexdigest()}.auto.yaml")
    return out


def v2_missing_sidecars(tag: str, listing) -> list:
    """Grid sidecars absent from `listing`, the file names in MAKEWEIGHT_ROOT."""
    have = set(listing)
    return sorted(f"{cfg}: {name}" for cfg, name in v2_expected_sidecars(tag).items() if name not in have)


# Storage under --keep-checkpoints window (pretrain_v2.prune and write_weight_average): at the
# end of a run, the state files of epochs 70-79, of the best epoch, net_best_epoch_state.pt and
# the weight average, plus one being written (14), two resume files (the newest and the one being
# written) and the records. Sizes of the 188-output model, the largest (2,305,136 state elements),
# written by pretrain_v2's own torch_save of its state and resume dicts (2026-10-01): 8.872 and
# 35.423 MiB, 44.3 MiB per epoch, so --keep-checkpoints all would hold 3.5 GiB per run. Records:
# init_trunk.pt (8.2 MiB) plus metrics, stream records and logs, bounded at 0.1 MiB per epoch.
V2_STATE_MIB = 8.872
V2_RESUME_MIB = 35.423
V2_RECORDS_MIB = 16.0


def v2_run_peak_gib() -> float:
    return (14 * V2_STATE_MIB + 2 * V2_RESUME_MIB + V2_RECORDS_MIB) / 1024


def v2_storage_problem(n_runs: int, headroom_gib: float):
    """None if n_runs runs at their peak fit the headroom below the jobs' 85% guard,
    else the reason."""
    need = n_runs * v2_run_peak_gib()
    if need > headroom_gib:
        return (f"{n_runs} runs x {v2_run_peak_gib():.3f} GiB = {need:.1f} GiB, above the "
                f"{headroom_gib:.1f} GiB to the 85% guard")
    return None


def v2_grid_specs(tag: str, tiers=None) -> dict:
    """{file name: spec} for every (arm, run) of the v2 grid, run k on V2_GPU_BY_RUN[k]."""
    out = {}
    for arm in v2_arms():
        if tiers is not None and arm.get("tier", 1) not in tiers:
            continue
        for run in range(1, int(arm["runs"]) + 1):
            name, spec = v2_spec(arm, run, V2_GPU_BY_RUN[run], tag=tag,
                                 run_id=v2_run_id(arm["name"], run), job=v2_job_name(arm["name"], run))
            out[f"job-{name}.yaml"] = spec
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
    ap.add_argument("--v2", metavar="DIR", default=None,
                    help="write the v2 pretraining specs (experiments/MTX/pretrain_v2.py) "
                         "for every arm and run of configs/arms/v2_grid.json into DIR; applies nothing")
    ap.add_argument("--tag", default=None, help="--v2: the repository tag the jobs clone")
    ap.add_argument("--sidecar-listing", default=None, metavar="FILE",
                    help="--v2: `ls /data/results/mtx/makeweight` read in a pod; every grid "
                         "config's sidecar (md5 at --tag) must be in it")
    ap.add_argument("--headroom-gib", type=float, default=None,
                    help="--v2: GiB free below 85%% of /data (df -B1 in a pod); the runs to start "
                         "must fit at their peak")
    ap.add_argument("--started", nargs="*", default=[], metavar="RUN_ID",
                    help="--v2: runs already started, not counted against the headroom")
    ap.add_argument("--v2-dryrun", metavar="DIR", default=None,
                    help="write the loader dry-run spec of --arm (its selection and window) into DIR")
    ap.add_argument("--arm", default="R16_Q1", help="--v2-dryrun: the grid arm")
    ap.add_argument("--epochs", type=int, default=None, help="--v2-dryrun: epochs")
    ap.add_argument("--label", default="", help="--v2-dryrun: names the job and output")
    ap.add_argument("--v2-makeweight", metavar="DIR", default=None,
                    help="write the make_weight specs for every grid config without a v1 sidecar "
                         "into DIR (a config whose sidecar exists is only re-checked); --label "
                         "names the jobs, since earlier make_weight jobs keep their names")
    ap.add_argument("--v2-smoke", metavar="DIR", default=None,
                    help="write the v2 smoke, GPU-numerics and loader dry-run specs into DIR")
    ap.add_argument("--deterministic", action="store_true", help="--v2-smoke: the -det variants")
    args = ap.parse_args()

    if args.v2_smoke:
        if not args.tag:
            ap.error("--v2-smoke needs --tag")
        d = pathlib.Path(args.v2_smoke)
        d.mkdir(parents=True, exist_ok=True)
        specs = v2_smoke_specs(args.tag, args.deterministic)
        if not args.deterministic:
            for full in (False, True):
                n, sp = v2_dryrun_spec(args.tag, full)
                specs[f"job-{n}.yaml"] = sp
            specs.update(v2_makeweight_specs(args.tag))
        for fn, spec in specs.items():
            (d / fn).write_text(spec)
            print(d / fn)
        return 0

    if args.v2_makeweight:
        if not args.tag:
            ap.error("--v2-makeweight needs --tag")
        d = pathlib.Path(args.v2_makeweight)
        d.mkdir(parents=True, exist_ok=True)
        for fn, spec in v2_makeweight_specs(args.tag, label=args.label).items():
            (d / fn).write_text(spec)
            print(d / fn)
        return 0

    if args.v2_dryrun:
        if not args.tag:
            ap.error("--v2-dryrun needs --tag")
        d = pathlib.Path(args.v2_dryrun)
        d.mkdir(parents=True, exist_ok=True)
        n, sp = v2_dryrun_spec(args.tag, False, args.label, arm=args.arm, epochs=args.epochs)
        (d / f"job-{n}.yaml").write_text(sp)
        print(d / f"job-{n}.yaml")
        return 0

    if args.v2:
        if not args.tag or args.sidecar_listing is None or args.headroom_gib is None:
            ap.error("--v2 needs --tag, --sidecar-listing and --headroom-gib")
        missing = v2_missing_sidecars(args.tag, pathlib.Path(args.sidecar_listing).read_text().split())
        if missing:
            print("REFUSED: no reweighting sidecar on /data for\n  " + "\n  ".join(missing))
            return 1
        specs = v2_grid_specs(args.tag)
        to_start = [fn for fn in specs if fn[len("job-mtx2-"):-len("-raunav.yaml")]
                    not in {r[len("mtx-"):] for r in args.started}]
        why = v2_storage_problem(len(to_start), args.headroom_gib)
        if why:
            print(f"REFUSED: {why}")
            return 1
        d = pathlib.Path(args.v2)
        d.mkdir(parents=True, exist_ok=True)
        for fn, spec in specs.items():
            (d / fn).write_text(spec)
            print(d / fn)
        return 0

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
