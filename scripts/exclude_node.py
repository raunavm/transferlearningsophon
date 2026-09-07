#!/usr/bin/env python3
"""Add a hostname exclusion to a job spec's required nodeAffinity.

WHY (2026-09-07 03:46Z, measured): ry-gpu-03.sdsc.optiputer.net advertises free
RTX 3090s, so the scheduler binds our Pending pods to it, and its NVIDIA device
plugin then rejects every one at admission --

    Pod was rejected: Allocate failed due to device plugin GetPreferredAllocation
    rpc failed ... error getting NVLink for devices (5, 0): failed to get nvlink
    remote pci info

38 of our pods failed that way in one wave, one retry each against
backoffLimit 50, and the older jobs' failed counts of 7-9 are the earlier waves.
Nothing in Kubernetes tells the scheduler the node is broken until the admins
taint it, so the spec has to say so.

Inserts, once, directly after the GPU-product expression of the required
nodeSelectorTerm:

              - key: kubernetes.io/hostname
                operator: NotIn
                values: ["<node>"]

Idempotent per node (extends an existing NotIn list). Refuses a spec with no
GPU-product expression.

Run:  python3 scripts/exclude_node.py ry-gpu-03.sdsc.optiputer.net experiments/MTX/k8s/job-mtx-*.yaml
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ANCHOR = re.compile(r'^(\s*)- key: nvidia\.com/gpu\.product\n\s*operator: In\n\s*values: \[[^\]]*\]\n', re.M)
EXISTING = re.compile(r'^(\s*)- key: kubernetes\.io/hostname\n\s*operator: NotIn\n\s*values: \[([^\]]*)\]\n', re.M)


def patch(path: pathlib.Path, node: str) -> str:
    text = path.read_text()
    m = EXISTING.search(text)
    if m:
        have = [v.strip().strip('"') for v in m.group(2).split(",") if v.strip()]
        if node in have:
            return "already excluded"
        have.append(node)
        new = f'{m.group(1)}- key: kubernetes.io/hostname\n{m.group(1)}  operator: NotIn\n{m.group(1)}  values: [{", ".join(chr(34) + h + chr(34) for h in have)}]\n'
        text = text[:m.start()] + new + text[m.end():]
    else:
        a = ANCHOR.search(text)
        if not a:
            return "FAILED: no nvidia.com/gpu.product expression in a nodeSelectorTerm"
        ind = a.group(1)
        block = (f'{ind}# excluded 2026-09-07: device plugin rejects every pod (scripts/exclude_node.py)\n'
                 f'{ind}- key: kubernetes.io/hostname\n{ind}  operator: NotIn\n{ind}  values: ["{node}"]\n')
        text = text[:a.end()] + block + text[a.end():]
    d = yaml.safe_load(text)
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    ok = any(e["key"] == "kubernetes.io/hostname" and e["operator"] == "NotIn" and node in e["values"]
             for t in terms for e in t["matchExpressions"])
    if not ok:
        return "FAILED: exclusion not present after patch"
    path.write_text(text)
    return f"excluded {node}"


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    node, specs = sys.argv[1], sys.argv[2:]
    rc = 0
    for s in specs:
        r = patch(pathlib.Path(s), node)
        print(f"{pathlib.Path(s).name:40s} {r}")
        rc |= r.startswith("FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
