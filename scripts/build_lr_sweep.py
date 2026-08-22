#!/usr/bin/env python3
"""Emit per-arm LR sweep specs -- the G1 KILL branch.

WHY THIS EXISTS
---------------
G1 measured L162 and R16_Q1 at 2.5e-4 / 5e-4 / 1e-3 and returned KILL: the
learning-rate optimum MOVES with K. The evidence is a resolved pairwise
reversal, which does not depend on either arm's argmax being pinned down:

    L162   prefers 1.0e-03 over 2.5e-04 by 0.03365   (both runs 16/16, FIXED)
    R16Q1  prefers 2.5e-04 over 1.0e-03 by 0.01931   (can only grow)

i.e. 1e-3 is L162's BEST rate and R16_Q1's WORST. No single rate serves both,
so `docs/GATES.md`'s branch applies: every arm needs its own sweep, and arms
are compared at their own optima rather than at a shared rate. Without this,
an arm difference in transfer could not be separated from a tuning difference
-- invariant I1, one controlled variable per contrast.

WHAT THIS BUILDS, AND WHY EACH PIECE
------------------------------------
1. R42_Q1 at the standard grid. The interior point of the ladder, "never cut"
   per docs/RUN_MATRIX.md. L162 wants >=1e-3 and R16_Q1 wants ~5e-4, so K=43
   is expected to land between them and the standard grid should bracket it.

2. L162 EXTENDED UPWARD (2e-3, 4e-3). Its G1 optimum sat at 1e-3, the top of
   the grid, so its optimum is not bracketed -- we know it is >=1e-3 and not
   where. A per-arm optimum that is actually a grid edge is not an optimum.

3. R16_Q1 SECOND SEED at its top two rates. G1 left this arm's argmax
   UNRESOLVED: 5e-4 and 2.5e-4 are 0.0036 apart, inside the 0.005 tie
   threshold. A second seed does double duty -- it resolves the argmax AND
   gives the first real estimate of run-to-run spread at 20% budget, which is
   what TIE_MARGIN currently only guesses at `[I]`.

NOT BUILT HERE, AND WHY
-----------------------
    L188     no arm config exists; it derives from the contraction tree, which
             is DECISIONS_PENDING item 1 and unsigned.
    RAND42   must be rebuilt against R42_Q1 group-size distributions first --
             the rand42_d{1,2,3} draws were built against the retired R15 block
             sizes and are archived, not reusable (docs/RUN_MATRIX.md).
    MPM      SSL, a different objective and head; its LR sweep is not this
             sweep, and the SSL pilot is gated on G0.

REPO PIN
--------
mtx-s1.2, which carries seed_weaver's --lean-val-metrics. weaver 0.4.17 removed
the get_train_fn/get_evaluate_fn hook, so ParT_sophon_arch_mtx.py could not
restrict eval_metrics and the O(K^2) roc_auc_score_matrix ran every epoch --
26.7 min/epoch measured at K=162. These runs pass the flag; the L162 points
save ~7.1 h each.

Run:  python3 scripts/build_lr_sweep.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments/G1/k8s/job-g1-l162-lr25e5-raunav.yaml"
OUT_DIR = ROOT / "experiments/G1/k8s"
# mtx-s1.3, not mtx-s1.2. L188's arm config does not exist in s1.2, so specs
# for that arm cannot resolve against it. s1.3 is a verified SUPERSET: every
# file a training pod executes -- seed_weaver.py, ParT_sophon_arch_mtx.py,
# write_run_manifest.py, and the L162/R42_Q1/R16_Q1 arm configs -- is
# byte-identical between the two tags, so moving the pin changes what is
# AVAILABLE to a run and nothing about what a run DOES. Specs already on disk
# are never regenerated (see main), so runs already launched keep their own pin.
PIN = "mtx-s1.3"

# (arm, K, rate, tag, seed). Tag convention matches build_g1_jobs.py: mantissa
# then exponent, minus sign dropped, so 2.5e-4 -> 25e5 and 2e-3 -> 2e3.
POINTS = [
    # 1. interior point, standard grid
    ("R42_Q1", 43, "2.5e-4", "25e5", 1),
    ("R42_Q1", 43, "5e-4", "5e4", 1),
    ("R42_Q1", 43, "1e-3", "1e3", 1),
    # 2. L162 extended upward -- its G1 optimum was at the 1e-3 grid edge
    ("L162", 162, "2e-3", "2e3", 1),
    ("L162", 162, "4e-3", "4e3", 1),
    # 3. R16_Q1 second seed at its unresolved top two
    ("R16_Q1", 17, "2.5e-4", "25e5", 2),
    ("R16_Q1", 17, "5e-4", "5e4", 2),
    # 4. R42_Q1 BRACKET (added 2026-08-22). Of the three points above, only
    #    2.5e-4 trained: 5e-4 and 1e-3 both went nan at iteration 2 and then hit
    #    a CUDA device-side assert. So 2.5e-4 is the top of the arm's TRAINABLE
    #    range, and nothing has been run below it -- the arm has no bracket at
    #    all, and "R42_Q1's optimum is 2.5e-4" is not a measurement, it is the
    #    only survivor. Two points fix that, and both are cheap next to the
    #    ~6 GPU-days an 80-epoch run at the wrong rate would waste:
    #
    #      1.25e-4  the missing lower side. If it scores worse, 2.5e-4 is
    #               bracketed (below by this, above by the divergence) and the
    #               arm can launch. If it scores BETTER, the optimum is lower
    #               still and the standard grid never contained it.
    #      5e-4 s2  is the divergence DETERMINISTIC? Both neighbouring arms
    #               train fine at 1e-3, so a K=43-specific failure at half that
    #               rate is odd. A second seed costs ~2 minutes if it reproduces
    #               (nan arrives at iteration 2) and settles whether the arm has
    #               a genuine stability ceiling or hit one bad initialisation.
    ("R42_Q1", 43, "1.25e-4", "125e6", 1),
    ("R42_Q1", 43, "5e-4", "5e4", 2),
    # 5. L188 (added 2026-08-22). The QCD-granularity arm, and the widest
    #    vocabulary on the ladder. Its grid is shifted UP relative to the
    #    standard one because L162 -- the nearest arm at K=162 -- put 2.5e-4
    #    far last (0.52183 against 0.55548 at 1e-3, a gap of 0.034 = 2.1x the
    #    noise floor). Spending a point there would buy a near-certain loser.
    #    1.4e-3 is the upper side: L162 diverged at 2e-3, so the trainable
    #    ceiling for a wide vocabulary sits between 1e-3 and 2e-3 and this
    #    probes inside that gap rather than repeating a known divergence.
    #
    #    ORDERING: these REQUIRE job-mtx-makeweight-l188-raunav to have run
    #    first. Each spec guards on a reweighting sidecar named for the arm
    #    config's md5, and L188 has no sidecar until that job writes one.
    ("L188", 188, "5e-4", "5e4", 1),
    ("L188", 188, "1e-3", "1e3", 1),
    ("L188", 188, "1.4e-3", "14e4", 1),
]


def build(arm: str, k: int, rate: str, tag: str, seed: int) -> tuple[str, str]:
    text = SRC.read_text()
    arm_lc = arm.lower().replace("_", "")
    run = f"g1-{arm_lc}-lr{tag}" + (f"-s{seed}" if seed != 1 else "")

    # identity
    text = text.replace("name: g1-l162-lr25e5-raunav", f"name: {run}-raunav")
    text = text.replace("RUN_ID=g1-l162-lr25e5", f"RUN_ID={run}")
    text = re.sub(r"--tensorboard \S+", f"--tensorboard {run}", text)

    # arm: config path, sidecar, num_classes, manifest --arm
    text = text.replace("configs/arms/L162.", f"configs/arms/{arm}.")
    text = text.replace("/data/results/mtx/makeweight/L162.",
                        f"/data/results/mtx/makeweight/{arm}.")
    text = text.replace("--arm L162", f"--arm {arm}")
    text = text.replace("--num-classes 162", f"--num-classes {k}")
    text = text.replace("-o num_classes 162", f"-o num_classes {k}")
    text = text.replace("configs/arms/L162.yaml.", f"configs/arms/{arm}.yaml.")

    # the swept variable
    n = text.count("--start-lr 2.5e-4")
    if n != 1:
        sys.exit(f"FATAL: {run}: expected 1 --start-lr, found {n}")
    text = text.replace("--start-lr 2.5e-4", f"--start-lr {rate}")

    # seed: appears twice (manifest writer and seed_weaver)
    if seed != 1:
        n = text.count("--seed 1")
        if n != 2:
            sys.exit(f"FATAL: {run}: expected 2 --seed, found {n}")
        text = text.replace("--seed 1", f"--seed {seed}")

    # repo pin + the lean validation metric
    text = re.sub(r'value: "mtx-s1[^"]*"', f'value: "{PIN}"', text)
    n = text.count("python3 experiments/E1/seed_weaver.py \\")
    if n != 1:
        sys.exit(f"FATAL: {run}: could not find the seed_weaver invocation")
    text = text.replace(
        "python3 experiments/E1/seed_weaver.py \\",
        "python3 experiments/E1/seed_weaver.py \\\n            "
        "--lean-val-metrics \\")

    text = text.replace(
        "  # G1 LEARNING-RATE SWEEP POINT:",
        f"  # PER-ARM LR SWEEP (G1 KILL branch): arm={arm} K={k} lr={rate} seed={seed}.\n"
        f"  # G1 showed the LR optimum MOVES with K, so a single shared rate would\n"
        f"  # confound granularity with tuning (I1). Generated by\n"
        f"  # scripts/build_lr_sweep.py from the G1 spec.\n"
        f"  #\n"
        f"  # ORIGINALLY --")
    return f"job-{run}-raunav.yaml", text


def main() -> int:
    if not SRC.exists():
        sys.exit(f"FATAL: {SRC} not found")
    written, skipped = [], []
    for arm, k, rate, tag, seed in POINTS:
        cfg = ROOT / f"configs/arms/{arm}.yaml"
        if not cfg.exists():
            sys.exit(f"FATAL: {cfg} does not exist; cannot sweep {arm}")
        fname, text = build(arm, k, rate, tag, seed)
        # A spec that already exists describes a run that already happened, and
        # may have been patched after generation (scripts/add_autoresume.py).
        # Regenerating it would silently revert that patch and rewrite the
        # record of what was actually launched.
        if (OUT_DIR / fname).exists():
            skipped.append(fname)
            continue
        d = yaml.safe_load(text)
        name = d["metadata"]["name"]
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", name):
            sys.exit(f"FATAL: {name} is not a valid RFC 1123 name")
        args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        for must in (f"--start-lr {rate}", f"-o num_classes {k}",
                     f"configs/arms/{arm}.yaml", "--lean-val-metrics",
                     "--num-epochs 16", f"--seed {seed}"):
            if must not in args:
                sys.exit(f"FATAL: {fname} missing {must!r}")
        # Check the EXECUTED script only. The spec carries prose explaining why
        # in-pod test eval is omitted, and that prose names L162 as the worst
        # case; matching against comments would fail on documentation.
        code = "\n".join(ln for ln in args.splitlines()
                         if not ln.lstrip().startswith("#"))
        if arm != "L162" and re.search(r"\bL162\b", code):
            sys.exit(f"FATAL: {fname} still executes a reference to L162")
        if arm != "L162" and re.search(r"num_classes 162|--num-classes 162", code):
            sys.exit(f"FATAL: {fname} still executes num_classes 162")
        if d["spec"]["backoffLimit"] != 0:
            sys.exit(f"FATAL: {fname} backoffLimit must stay 0")
        (OUT_DIR / fname).write_text(text)
        written.append((name, arm, k, rate, seed))
    if skipped:
        print(f"skipped {len(skipped)} spec(s) that already exist (not regenerated): "
              f"{', '.join(sorted(skipped))}")
    print(f"wrote {len(written)} sweep specs to {OUT_DIR.relative_to(ROOT)}/  (pin {PIN})")
    for name, arm, k, rate, seed in written:
        print(f"  {name:26s} arm={arm:7s} K={k:<4d} lr={rate:<7s} seed={seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
