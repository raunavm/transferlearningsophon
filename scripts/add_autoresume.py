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

WHY THERE IS A RECIPE GUARD (added 2026-08-22)
----------------------------------------------
Auto-resume keys on the output directory, and the output directory is named for
the gate, arm and seed -- NOT for the learning rate. That was safe while one
rate served every arm. G1 returned KILL, so each arm now trains at its OWN
rate, and the rate attached to a given `<gate>-<arm>-<seed>` directory CHANGED.
A directory holding checkpoints from the old rate would be resumed onto by a
job running the new one, blending two learning rates inside a single run with
nothing in the logs saying so.

The guard stamps `${OUT}/RECIPE` with the rate and epoch budget this job was
generated for, and refuses to resume across a mismatch. Checkpoints with NO
stamp are also refused: they predate the guard, so their provenance is exactly
what the guard cannot verify. Both refusals are hard exits, not warnings --
a blended-rate run is unrecoverable after the fact and indistinguishable from
a clean one in the output.

Verified under POSIX sh (the container's shell, not zsh, which errors on an
unmatched glob rather than passing it through):

    empty dir                 -> fresh start
    epochs 0-3 + half 4       -> --load-epoch 3
    state w/o optimizer       -> fresh start
    nonexistent dir           -> fresh start
    checkpoints, no RECIPE    -> hard exit
    checkpoints, wrong RECIPE -> hard exit

Run:  python3 scripts/add_autoresume.py <job.yaml> [<job.yaml> ...]
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

# 50, not 3. Raised 2026-08-26 after MEASURING what the queue actually does:
# seven jobs sat Pending and were reaped by Nautilus three times each WITHOUT
# EVER RUNNING A STEP, arriving simultaneously at failed=3 against
# backoffLimit=3. One more reap would have killed all seven permanently, and
# they were saved only by live-patching the limit.
#
# The original 3 assumed retries were spent on real failures. They are not --
# they are spent on a queue that reaps pods for waiting, which is unbounded and
# has nothing to do with the run's health. The limit therefore has to outlast
# the queue, not the failure.
#
# The cost of a high limit is bounded by what a genuine crash-loop costs: a run
# that goes nan at iteration 2 burns ~3 min per attempt, so 50 attempts is about
# 2.5 h before the job gives up. That is an acceptable price for not losing an
# 8-day run to a scheduler.
BACKOFF = 50

# `{recipe}` is filled from the spec's OWN --start-lr and --num-epochs, so the
# stamp can never drift from what the job actually runs.
SNIPPET = """          # AUTO-RESUME + RECIPE GUARD. See scripts/add_autoresume.py for why.
          # Requires BOTH the state and optimizer file for an epoch: weaver
          # writes them separately, and resuming from a state file whose
          # optimizer is missing only WARNS and then trains on with a fresh
          # optimizer -- silent corruption of a momentum-based run.
          RECIPE='{recipe}'
          RESUME=""
          LAST_EP=-1
          for f in ${{OUT}}/net_epoch-*_state.pt; do
            [ -e "$f" ] || continue
            n=$(basename "$f" | sed 's/net_epoch-\\([0-9]*\\)_state\\.pt/\\1/')
            [ -f "${{OUT}}/net_epoch-${{n}}_optimizer.pt" ] || continue
            [ "$n" -gt "$LAST_EP" ] && LAST_EP=$n
          done
          if [ "$LAST_EP" -ge 0 ]; then
            # The output dir is named for gate/arm/seed, NOT for the learning
            # rate, and the per-arm rate changed when G1 returned KILL. Resuming
            # across a recipe change would blend two rates in one run silently.
            if [ ! -f "${{OUT}}/RECIPE" ]; then
              echo "FATAL: ${{OUT}} has checkpoints (through epoch ${{LAST_EP}}) but no"
              echo "RECIPE stamp, so the rate they were trained at is unknown."
              echo "Refusing to resume. Move that directory aside to start fresh."
              exit 1
            fi
            ON_DISK=$(cat "${{OUT}}/RECIPE")
            if [ "${{ON_DISK}}" != "${{RECIPE}}" ]; then
              echo "FATAL: ${{OUT}} holds output from a DIFFERENT recipe."
              echo "   on disk : ${{ON_DISK}}"
              echo "   this job: ${{RECIPE}}"
              echo "Refusing to resume. Move that directory aside to start fresh."
              exit 1
            fi
            RESUME="--load-epoch ${{LAST_EP}}"
            echo "AUTO-RESUME: complete checkpoint for epoch ${{LAST_EP}}, recipe matches; resuming."
          else
            echo "AUTO-RESUME: no complete checkpoint; starting from scratch."
          fi
          printf '%s' "${{RECIPE}}" > ${{OUT}}/RECIPE

"""

WEAVER_EPOCHS = re.compile(r"--num-epochs (\d+) --optimizer ranger")
WEAVER_LR = re.compile(r"--start-lr (\S+)")


def patch(path: pathlib.Path) -> str:
    text = path.read_text()
    if "AUTO-RESUME" in text:
        return "already patched"

    m_ep = WEAVER_EPOCHS.search(text)
    if not m_ep:
        return "FAILED: no '--num-epochs N --optimizer ranger' in the weaver call"
    epochs = m_ep.group(1)

    lrs = set(WEAVER_LR.findall(text))
    if len(lrs) != 1:
        return f"FAILED: expected exactly 1 distinct --start-lr, found {sorted(lrs)}"
    lr = lrs.pop()

    anchor = "          # seed_weaver derives four independent RNG streams"
    if anchor not in text:
        return "FAILED: could not find the seed_weaver comment anchor"
    recipe = f"lr={lr} epochs={epochs}"
    text = text.replace(anchor, SNIPPET.format(recipe=recipe) + anchor)

    # hand the flag to weaver, next to the budget it interacts with
    old = f"--num-epochs {epochs} --optimizer ranger"
    if text.count(old) != 1:
        return f"FAILED: expected 1 {old!r}, found {text.count(old)}"
    text = text.replace(old, f"--num-epochs {epochs} ${{RESUME}} --optimizer ranger")

    # a retry is now safe, so let one happen
    if text.count("  backoffLimit: 0") != 1:
        return f"FAILED: expected 1 backoffLimit line, found {text.count('  backoffLimit: 0')}"
    text = text.replace("  backoffLimit: 0", f"  backoffLimit: {BACKOFF}")

    d = yaml.safe_load(text)
    args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    for must in ("AUTO-RESUME", "${RESUME}", "net_epoch-${n}_optimizer.pt",
                 f"RECIPE='{recipe}'", 'RECIPE stamp', 'Refusing to resume'):
        if must not in args:
            return f"FAILED: patched spec is missing {must!r}"
    if d["spec"]["backoffLimit"] != BACKOFF:
        return f"FAILED: backoffLimit is {d['spec']['backoffLimit']}"
    path.write_text(text)
    return f"patched (backoffLimit 0 -> {BACKOFF}, auto-resume, recipe {recipe})"


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
