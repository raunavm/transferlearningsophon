#!/usr/bin/env python3
"""Make a training job self-resuming, so a retry is always safe.

THE PROBLEM THIS SOLVES
-----------------------
Two different disruptions have now killed runs, and `backoffLimit: 0` handled
one well and the other badly:

  2026-08-17  ry-gpu-02 went NodeNotReady mid-training. backoffLimit 0 turned
              the eviction into a hard failure -- which was RIGHT, because a
              plain retry would have silently restarted from epoch 0 and we
              would not have noticed. 22 epochs were recovered by hand with
              --load-epoch.

  2026-08-19  Four R16_Q1 seed-2 attempts failed WITHOUT EVER RUNNING. Watched
              live, the pod sits Pending and unscheduled; nothing is ever
              written to the output directory; the job fails ~1-6 h in. The
              pods are reaped while queuing for a GPU and backoffLimit 0 makes
              that permanent. Here a retry would have been exactly right --
              there was nothing to lose.

Kubernetes cannot tell those apart: both surface as a failed pod. What makes
them differ is whether any work existed to lose. So instead of guessing with
backoffLimit, make a retry SAFE in both cases -- have the container resume
itself from whatever it finds on the PVC. Then:

    reaped while Pending  -> retry, nothing on disk, starts fresh   (correct)
    node dies mid-run     -> retry, resumes from last good epoch    (correct)
    NaN / OOM             -> retry, resumes, fails again, bounded   (acceptable)

WHY BOTH FILES ARE CHECKED
--------------------------
weaver writes `net_epoch-N_state.pt` and `net_epoch-N_optimizer.pt` separately,
so a crash between the two leaves a state file with no optimizer. Resuming from
that epoch would silently drop the optimizer state -- weaver only WARNS
("Optimizer state file NOT found!") and carries on with a fresh optimizer,
which corrupts a momentum-based run without failing. The detection therefore
requires BOTH files and falls back to the previous complete epoch.

Verified under POSIX sh (the container's shell, not zsh, which errors on an
unmatched glob rather than passing it through):

    empty dir            -> fresh start
    epochs 0-3 + half 4  -> --load-epoch 3
    state w/o optimizer  -> fresh start
    nonexistent dir      -> fresh start

Run:  python3 scripts/add_autoresume.py <job.yaml> [<job.yaml> ...]
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

BACKOFF = 3

SNIPPET = """          # AUTO-RESUME. See scripts/add_autoresume.py for why this exists.
          # Requires BOTH the state and optimizer file for an epoch: weaver
          # writes them separately, and resuming from a state file whose
          # optimizer is missing only WARNS and then trains on with a fresh
          # optimizer -- silent corruption of a momentum-based run.
          RESUME=""
          LAST_EP=-1
          for f in ${OUT}/net_epoch-*_state.pt; do
            [ -e "$f" ] || continue
            n=$(basename "$f" | sed 's/net_epoch-\\([0-9]*\\)_state\\.pt/\\1/')
            [ -f "${OUT}/net_epoch-${n}_optimizer.pt" ] || continue
            [ "$n" -gt "$LAST_EP" ] && LAST_EP=$n
          done
          if [ "$LAST_EP" -ge 0 ]; then
            RESUME="--load-epoch ${LAST_EP}"
            echo "AUTO-RESUME: found complete checkpoint for epoch ${LAST_EP}; resuming."
          else
            echo "AUTO-RESUME: no complete checkpoint; starting from scratch."
          fi

"""


def patch(path: pathlib.Path) -> str:
    text = path.read_text()
    if "AUTO-RESUME" in text:
        return "already patched"

    anchor = "          # seed_weaver derives four independent RNG streams"
    if anchor not in text:
        return "FAILED: could not find the seed_weaver comment anchor"
    text = text.replace(anchor, SNIPPET + anchor)

    # hand the flag to weaver, next to the budget it interacts with
    n = text.count("--num-epochs 16 --optimizer ranger")
    if n != 1:
        return f"FAILED: expected 1 weaver --num-epochs line, found {n}"
    text = text.replace("--num-epochs 16 --optimizer ranger",
                        "--num-epochs 16 ${RESUME} --optimizer ranger")

    # a retry is now safe, so let one happen
    n = text.count("  backoffLimit: 0")
    if n != 1:
        return f"FAILED: expected 1 backoffLimit line, found {n}"
    text = text.replace("  backoffLimit: 0", f"  backoffLimit: {BACKOFF}")

    d = yaml.safe_load(text)
    args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    for must in ("AUTO-RESUME", "${RESUME}", "net_epoch-${n}_optimizer.pt"):
        if must not in args:
            return f"FAILED: patched spec is missing {must!r}"
    if d["spec"]["backoffLimit"] != BACKOFF:
        return f"FAILED: backoffLimit is {d['spec']['backoffLimit']}"
    path.write_text(text)
    return f"patched (backoffLimit 0 -> {BACKOFF}, auto-resume added)"


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rc = 0
    for a in sys.argv[1:]:
        p = pathlib.Path(a)
        r = patch(p)
        print(f"{p.name:38s} {r}")
        if r.startswith("FAILED"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
