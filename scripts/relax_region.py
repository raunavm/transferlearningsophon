#!/usr/bin/env python3
"""Turn the hard us-west region pin into a soft preference.

WHY. The specs require `topology.kubernetes.io/region In [us-west]` because
weaver streams JetClass-II from the CephFS PVC at SDSC, and the spec comment
claims a distant pod "does not run slowly, it runs at near-zero GPU
utilisation". Two measurements on 2026-09-07 undercut that:

  * we are GPU-BOUND, not I/O-bound: 90-100% GPU utilisation, 14.8/24.6 GiB,
    306-348 W on every running pod;
  * training needs ~2,275 jets/s = ~5.5 MB/s at 2.4 kB/jet, while the PVC
    delivers >1,600 MB/s in region -- roughly 300x headroom.

and the scheduler says 271 of 527 nodes fail our affinity while 20 more have no
free GPU. A probe pinned to a 3090 OUTSIDE us-west scheduled immediately, twice,
while every in-region pod sat Pending.

The GPU-product pin STAYS. That one is invariant I7b (never compare arms trained
on different GPU models within a seed pair) and relaxing it would cost re-runs of
the finished R16_Q1 seeds. Region is not an invariant -- it is a performance
guess, and it is now measured.

Soft, not removed: us-west keeps weight 100, so nothing moves while SDSC has
capacity, and the queue drains elsewhere when it does not.

Run:  python3 scripts/relax_region.py experiments/MTX/k8s/job-mtx-*.yaml
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

REQ = re.compile(
    r"[ \t]*- key: topology\.kubernetes\.io/region\n"
    r"[ \t]*operator: In\n"
    r"[ \t]*values: \[\"us-west\"\]\n")

PREFERRED = """          preferredDuringSchedulingIgnoredDuringExecution:
          # Region was a REQUIRED term until 2026-09-07; see scripts/relax_region.py.
          # Kept as a strong preference so nothing moves while SDSC has capacity.
          - weight: 100
            preference:
              matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
"""


def patch(path: pathlib.Path) -> str:
    text = path.read_text()
    if "preferredDuringSchedulingIgnoredDuringExecution" in text:
        return "already relaxed"
    if not REQ.search(text):
        return "SKIP: no required us-west term"
    text = REQ.sub("", text, count=1)
    anchor = "        nodeAffinity:\n"
    if anchor not in text:
        return "FAILED: no nodeAffinity block"
    text = text.replace(anchor, anchor + PREFERRED, 1)

    d = yaml.safe_load(text)
    na = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]
    req = na["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    keys = {e["key"] for t in req for e in t["matchExpressions"]}
    if "topology.kubernetes.io/region" in keys:
        return "FAILED: region still required"
    if "nvidia.com/gpu.product" not in keys:
        return "FAILED: GPU-product pin lost -- that one is invariant I7b"
    if not na.get("preferredDuringSchedulingIgnoredDuringExecution"):
        return "FAILED: preference not added"
    path.write_text(text)
    return "region -> soft preference (GPU pin intact)"


def main() -> int:
    rc = 0
    for s in sys.argv[1:]:
        r = patch(pathlib.Path(s))
        print(f"{pathlib.Path(s).name:44s} {r}")
        rc |= r.startswith("FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
