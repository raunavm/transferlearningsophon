#!/usr/bin/env python3
"""Seed wrapper for weaver 0.4.17 (which exposes no --seed).

Sets every RNG seed IN-PROCESS *before* weaver builds the model, so both weight
initialization and the DataLoader worker seeds become deterministic, then hands
off to weaver.train:main with the remaining args untouched.

Usage:  python seed_weaver.py --seed S  <all the normal weaver args>

Reproducibility (seed-only, per the E1 decision): a run is labeled and
reproducible on the SAME GPU model. Cross-GPU is statistically — not bit —
identical (different CUDA kernels/reduction orders); that is inherent to a
heterogeneous opportunistic cluster and is documented in RESULTS.md.
"""
import os
import random
import sys


def _pop_seed(argv):
    if "--seed" not in argv:
        raise SystemExit("seed_weaver: --seed S is required")
    i = argv.index("--seed")
    return int(argv[i + 1]), argv[:i] + argv[i + 2:]


seed, rest = _pop_seed(sys.argv[1:])
os.environ["PYTHONHASHSEED"] = str(seed)  # also set in the job env for full effect

import numpy as np  # noqa: E402
import torch  # noqa: E402

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"[seed_weaver] seed={seed} applied (python/numpy/torch/cuda); "
      f"handing off to weaver.train:main", flush=True)

from weaver.train import main  # noqa: E402

sys.argv = ["weaver"] + rest
main()
