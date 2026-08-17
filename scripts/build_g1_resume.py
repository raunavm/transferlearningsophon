#!/usr/bin/env python3
"""Emit resume specs for G1 points killed by a node failure.

WHY THIS EXISTS
---------------
2026-08-17 ~13:50Z: `ry-gpu-02.sdsc.optiputer.net` went NodeNotReady. The taint
manager evicted both G1 pods on it and, with `backoffLimit: 0`, both Jobs went
straight to Failed:

    g1-l162-lr25e5   died after epoch 9   (K=162, lr 2.5e-4)
    g1-r16q1-lr1e3   died after epoch 12  (K=17,  lr 1e-3)

`backoffLimit: 0` is what saved the work rather than what lost it. A retry does
NOT resume -- weaver only resumes when `--load-epoch` is passed -- so with
backoffLimit 1 Kubernetes would have silently restarted each run from epoch 0
and a 16-epoch run would quietly have become a 3-epoch one.

WHAT RESUMING PRESERVES, VERIFIED IN weaver 0.4.17 SOURCE
--------------------------------------------------------
`train.py:488-500` loads BOTH `_epoch-N_state.pt` and `_epoch-N_optimizer.pt`.
`train.py:519-521` passes `last_epoch=args.load_epoch` to the `flat+decay`
MultiStepLR, so the LR schedule continues at the right position instead of
restarting. That matters more here than anywhere else: G1 measures the learning
rate, so a schedule that silently restarted would corrupt the one thing the gate
is trying to read.

With `--num-epochs 16`, `num_decay_epochs = int(16*0.3) = 4`, so the milestones
are epochs 12-15. r16q1-lr1e3 died exactly at the decay onset; l162-lr25e5 died
while still flat.

WHAT RESUMING DOES *NOT* PRESERVE -- read before trusting a narrow verdict
--------------------------------------------------------------------------
`seed_weaver` re-seeds `data_sampling` at process start. On resume the sampler
restarts from that seed rather than continuing, so the remaining epochs see a
different data ORDER than an uninterrupted run would have. The data is drawn
from the same reweighted distribution either way, so this is a nuisance term,
not a bias -- but it perturbs final validation accuracy at roughly seed-noise
scale, and the margins seen mid-sweep were 0.003-0.006.

That is exactly what `analyse_g1.py`'s TIE_MARGIN guard is for. If the final
margins are comfortably above it, this perturbation cannot have decided the
verdict. If they are not, the verdict is unresolved regardless of the node
failure, and the fix -- a second seed at the top two rates -- is the same fix
the near-tie needs anyway.

WHY A NEW JOB NAME BUT THE SAME RUN_ID
--------------------------------------
The failed Job objects are left in place as evidence (docs/RECORD.md: crashed
runs belong in the ledger), so the new Jobs need new names. RUN_ID and OUT stay
identical so the checkpoints, train.log and manifest continue in place. Verified
the spec only does `mkdir -p ${OUT}` -- there is no rm, so relaunching cannot
destroy the checkpoints being resumed from.

Run:  python3 scripts/build_g1_resume.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "experiments/G1/k8s"

# point -> epoch to resume from (highest N with BOTH _state.pt and
# _optimizer.pt present on the PVC, confirmed equal to the last validated epoch)
RESUME = {
    "g1-l162-lr25e5": 9,
    "g1-r16q1-lr1e3": 12,
}
SUFFIX = "r2"


def build(point: str, epoch: int) -> tuple[str, str]:
    src = SRC_DIR / f"job-{point}-raunav.yaml"
    text = src.read_text()

    # new Job name only; RUN_ID and OUT deliberately unchanged
    text = text.replace(f"name: {point}-raunav", f"name: {point}-{SUFFIX}-raunav")

    # resume flag, next to the budget it interacts with
    n = text.count("--num-epochs 16 --optimizer ranger")
    if n != 1:
        sys.exit(f"FATAL: {point}: expected 1 weaver --num-epochs line, found {n}")
    text = text.replace(
        "--num-epochs 16 --optimizer ranger",
        f"--num-epochs 16 --load-epoch {epoch} --optimizer ranger")

    text = text.replace(
        "  # G1 LEARNING-RATE SWEEP POINT:",
        f"  # RESUMED from epoch {epoch} after ry-gpu-02 went NodeNotReady on\n"
        f"  # 2026-08-17. Same RUN_ID and OUT as the original, so checkpoints and\n"
        f"  # train.log continue in place. See scripts/build_g1_resume.py for what\n"
        f"  # resuming does and does not preserve.\n"
        f"  #\n"
        f"  # G1 LEARNING-RATE SWEEP POINT:")
    return f"job-{point}-{SUFFIX}-raunav.yaml", text


def main() -> int:
    written = []
    for point, epoch in RESUME.items():
        fname, text = build(point, epoch)
        d = yaml.safe_load(text)
        name = d["metadata"]["name"]
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", name):
            sys.exit(f"FATAL: {name} is not a valid RFC 1123 name")
        args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
        for must in (f"--load-epoch {epoch}", f"RUN_ID={point}",
                     "--num-epochs 16", "--model-prefix ${OUT}/net"):
            if must not in args:
                sys.exit(f"FATAL: {fname} missing {must!r}")
        if "rm -rf" in args or "rm -f" in args:
            sys.exit(f"FATAL: {fname} contains a remove; it would destroy the "
                     f"checkpoints being resumed from")
        if d["spec"]["backoffLimit"] != 0:
            sys.exit(f"FATAL: {fname} backoffLimit must stay 0; a retry would "
                     f"restart from epoch 0 without --load-epoch")
        (SRC_DIR / fname).write_text(text)
        written.append((name, epoch, 16 - epoch - 1))
    print(f"wrote {len(written)} resume specs to {SRC_DIR.relative_to(ROOT)}/")
    for name, epoch, remaining in written:
        print(f"  {name:34s} resume from epoch {epoch}, {remaining} epochs remaining")
    return 0


if __name__ == "__main__":
    sys.exit(main())
