#!/usr/bin/env python3
"""Every fine-tuning run's recorded recipe, for the manuscript's settings table.

Walks the given result roots for ft_manifest.json -- written by each run at
launch (experiments/FT/smoke_checks.py manifest) -- and prints one JSON document
listing each run's recipe fields. Reads only; writes nothing. The table is then
built from what the runs recorded, not from what the job builder intended.

Run inside a pod that mounts /data:
    kubectl exec -i <pod> -- python3 - /data/results/ft/w2b /data/results/ft/bench_v2 \\
        < experiments/FT/collect_recipes.py
"""
from __future__ import annotations

import json
import pathlib
import sys

FIELDS = ("leg", "init", "n_train", "ft_seed", "lr", "head_lr_mult", "epochs", "lr_schedule",
          "weight_decay", "batch_size", "samples_per_epoch", "samples_per_epoch_val",
          "num_classes", "data_config", "wave", "gpu_product_pin", "repo_commit")


def main(roots: list[str]) -> int:
    rows = []
    for root in roots:
        for p in sorted(pathlib.Path(root).rglob("ft_manifest.json")):
            m = json.loads(p.read_text())
            rows.append({"path": str(p), **{k: m.get(k) for k in FIELDS}})
    print(json.dumps({"roots": roots, "n_runs": len(rows), "runs": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
