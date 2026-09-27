#!/usr/bin/env python3
"""List, and with --delete remove, fine-tuning attempts that a completed rerun
has replaced, plus the two quark/gluon staging folders built with the wrong
charge sign. Runs INSIDE a pod that mounts the project volume at /data.

An attempt directory is `<cell>.partial.<t>` or `<cell>.nodefail.<t>`: a cell
that died and was renamed so the cell could be rerun from scratch. It is
removed only if its cell now has DONE and the attempt itself does not, checked
again immediately before each removal -- the rule used for the six cleared on
2026-09-17 (ledger w2b-partials-cleared). The staging folders are removed only
while they still carry the DO_NOT_TRAIN_ON_THIS.txt marker. Nothing else is
touched.

Dry run (default) prints what would go; --delete removes it.
    kubectl exec -i <pod> -- python3 - [--delete] < scripts/clear_superseded_attempts.py
"""
import os
import re
import shutil
import sys

FT_ROOT = "/data/results/ft"
BAD_STAGING = ["/data/finetune/qg_WRONGCHARGE", "/data/finetune/qg_sub_WRONGCHARGE"]
ATTEMPT = re.compile(r"^(.*)\.(partial|nodefail)\.\d+$")


def size(p: str) -> int:
    return int(os.getxattr(p, "ceph.dir.rbytes"))


def superseded(path: str) -> bool:
    cell = ATTEMPT.match(path).group(1)
    return os.path.isfile(os.path.join(cell, "DONE")) and not os.path.isfile(os.path.join(path, "DONE"))


def main(delete: bool) -> int:
    attempts, kept = [], []
    for root, dirs, _ in os.walk(FT_ROOT):
        for d in [d for d in dirs if ATTEMPT.match(d)]:
            dirs.remove(d)
            p = os.path.join(root, d)
            (attempts if superseded(p) else kept).append(p)
    for p in kept:
        print(f"KEEP (its cell has no DONE, or it has its own): {p}")
    staging = [p for p in BAD_STAGING if os.path.isfile(os.path.join(p, "DO_NOT_TRAIN_ON_THIS.txt"))]
    total = sum(size(p) for p in attempts + staging)
    print(f"{len(attempts)} superseded attempts + {len(staging)} wrong-charge staging folders, "
          f"{total / 1e9:.2f} GB")
    if not delete:
        for p in attempts + staging:
            print(f"  would remove {size(p) / 1e9:6.2f} GB  {p}")
        print("dry run; pass --delete to remove")
        return 0
    freed = 0
    for p in attempts:
        if superseded(p):                  # re-checked right before removal
            b = size(p)
            shutil.rmtree(p)
            freed += b
    for p in staging:
        if os.path.isfile(os.path.join(p, "DO_NOT_TRAIN_ON_THIS.txt")):
            b = size(p)
            shutil.rmtree(p)
            freed += b
    print(f"removed {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main("--delete" in sys.argv[1:]))
