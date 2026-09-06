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
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
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
    "R16_Q1": (17, "5e-4", True,
               "2.5e-4 0.77917 < 5e-4 0.78277 > 1e-3 0.75986 at seed 1, and "
               "2.5e-4 0.76598 < 5e-4 0.76697 at seed 2 -- same ordering at "
               "both seeds. Bracketed on both sides. The 1e-3 deficit "
               "(0.01931) is 1.22x the noise floor; the 2.5e-4 gap is inside "
               "it."),
    "R42_Q1": (43, "2.5e-4", False,
               "2.5e-4 0.70721 is the ONLY point that trained. 5e-4 and 1e-3 "
               "both went nan at iteration 2 then hit a CUDA device-side "
               "assert. Nothing was run BELOW 2.5e-4, so this is an upper "
               "bound on the trainable range, not a located optimum."),
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

        # record WHY this rate, in the spec the run clones
        text = text.replace(
            "  # CORE-MATRIX GRANULARITY ARM",
            f"  # LEARNING RATE {rate} -- this arm's own optimum, not a shared rate.\n"
            f"  # G1 returned KILL (the optimum moves with K), so docs/GATES.md's\n"
            f"  # branch applies and arms are compared at their own optima.\n"
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


def derive_seed(arm: str, base_seed: int, seed: int) -> str:
    """Write job-mtx-<arm>-s<seed>-raunav.yaml from the seed-<base_seed> spec."""
    arm_lc = arm.lower()
    src = K8S / f"job-mtx-{arm_lc}-s{base_seed}-raunav.yaml"
    dst = K8S / f"job-mtx-{arm_lc}-s{seed}-raunav.yaml"
    if not src.exists():
        return f"FAILED: {src.name} not found"
    if dst.exists():
        return "exists, not regenerated"
    text = src.read_text()

    run_lc = arm.lower().replace("_", "")
    subs = [
        (f"name: mtx-{run_lc}-s{base_seed}-raunav", f"name: mtx-{run_lc}-s{seed}-raunav"),
        (f"RUN_ID=mtx-{run_lc}-s{base_seed}", f"RUN_ID=mtx-{run_lc}-s{seed}"),
        (f"--tensorboard mtx_{arm}_s{base_seed}", f"--tensorboard mtx_{arm}_s{seed}"),
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
    if "${RESUME}" not in code or d["spec"]["backoffLimit"] != 3:
        return "FAILED: did not inherit auto-resume"
    dst.write_text(text)
    return f"derived from s{base_seed} (lr={RATES[arm][1]}, seed={seed})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unbracketed", action="store_true",
                    help="emit arms whose LR is not bracketed on both sides")
    ap.add_argument("--derive-seeds", nargs="+", default=[],
                    help="ARM:seed,seed,... derive extra seeds from that arm's "
                         "seed-1 spec. Refused for arms whose rate is not "
                         "bracketed, because the rate is baked in at write time.")
    args = ap.parse_args()

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
            for sd in [int(x) for x in seeds.split(",") if x]:
                r = derive_seed(arm, 1, sd)
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
