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
    "L162": (162, "1e-3", True,
             "2.5e-4 0.52183 < 5e-4 0.55232 < 1e-3 0.55548, and 2e-3 diverged "
             "to nan at iteration 2. Bracketed below by 5e-4 and above by the "
             "divergence. The 1e-3 vs 5e-4 gap is 0.00316 = 0.2x the noise "
             "floor, so the two are tied and 1e-3 is the point estimate, not a "
             "resolved winner."),
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
}

SEEDS = [1]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unbracketed", action="store_true",
                    help="emit arms whose LR is not bracketed on both sides")
    args = ap.parse_args()

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
