#!/usr/bin/env python3
"""Seed wrapper for weaver 0.4.17 (which exposes no --seed).

Sets every RNG seed IN-PROCESS *before* weaver builds the model, turns on the
deterministic cuDNN policy (PLAN §3), and — critically for the §9.3 twin design
— PINS the DataLoader base seed so the per-worker RNG (hence file/event shuffle
AND the reweight keep/upsample draws) is IDENTICAL across arms at a shared run
seed.

Why the pin is required (audit 2026-07-19): weaver builds its DataLoaders with
`generator=None`, so PyTorch draws each loader's `base_seed` from the LIVE global
RNG at first-iter time — which is *after* the per-arm head (`Linear(512,
num_classes)`, built last) has consumed a num_classes-dependent number of draws.
Result: at a shared seed the arms silently get different `base_seed`s → different
worker numpy seeds → different jet order and even a different jet *multiset*
(reweight sampling desyncs). That violates the "same sampled jet stream"
invariant the whole paired comparison rests on. Injecting a run-seed generator
into every DataLoader makes `base_seed` a function of the run seed ONLY.

batch_stream_sha (§3): hash of the per-worker numpy seeds the run will use,
written next to the checkpoints. It is K-independent, so equality across arms at
a shared seed is the empirical pairing check, and equality across the §9.7
double-run is the stream-identity check.

Head init: the head derives deterministically from the shared run seed after the
(identical) backbone is built — reproducible per (K, seed), arm-specific by
shape. This meets §9.3's intent. The literal `sha256(K:seed)` head-seed is
SUPERSEDED: it changes no pairing property, and a post-hoc re-init would risk an
init-distribution mismatch (amendment 2026-07-19).

Reproducibility is seed-labeled on the SAME GPU model (L40, §3); cross-GPU is
statistical not bitwise, and §9.7 quantifies the residual.

Usage:  python seed_weaver.py --seed S  <all the normal weaver args>
"""
import hashlib
import os
import random
import sys
from pathlib import Path


def _pop_seed(argv):
    if "--seed" not in argv:
        raise SystemExit("seed_weaver: --seed S is required")
    i = argv.index("--seed")
    return int(argv[i + 1]), argv[:i] + argv[i + 2:]


def _get_opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


seed, rest = _pop_seed(sys.argv[1:])

# PYTHONHASHSEED + cuBLAS workspace must be set before the CUDA context inits.
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# PLAN §3 deterministic policy (bitwise determinism is NOT assumed; §9.7
# quantifies the residual). Applied identically to all arms.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as e:  # older torch: flag unavailable
    print(f"[seed_weaver] use_deterministic_algorithms unavailable: {e}", flush=True)

# --- §9.3 batch-stream pin ---------------------------------------------------
# Inject a run-seed generator into every DataLoader weaver builds with
# generator=None, so base_seed (=> per-worker numpy seeds => shuffles + reweight
# sampling) is a function of the run seed ONLY, never of num_classes.
_orig_dl_init = torch.utils.data.DataLoader.__init__


def _seeded_dl_init(self, *a, **kw):
    if kw.get("generator", None) is None:
        kw["generator"] = torch.Generator().manual_seed(seed)
    return _orig_dl_init(self, *a, **kw)


torch.utils.data.DataLoader.__init__ = _seeded_dl_init

# --- batch_stream_sha (§3) ---------------------------------------------------
# PyTorch draws base_seed as the first int64 from the loader's generator; weaver
# then seeds worker w with (base_seed + w) & 0xFFFFFFFF (dataset.py). Computed
# from the SAME generator we inject => equals what the run uses, K-independent.
_num_workers = int(_get_opt(rest, "--num-workers") or 0)
_base_seed = int(torch.empty((), dtype=torch.int64).random_(
    generator=torch.Generator().manual_seed(seed)).item())
_worker_seeds = [(_base_seed + w) & 0xFFFFFFFF for w in range(max(1, _num_workers))]
_batch_stream_sha = hashlib.sha256(
    ":".join(str(s) for s in _worker_seeds).encode()).hexdigest()

_model_prefix = _get_opt(rest, "--model-prefix")
if _model_prefix:
    run_dir = Path(_model_prefix).parent
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "batch_stream_sha.txt").write_text(
            f"batch_stream_sha={_batch_stream_sha}\n"
            f"run_seed={seed}\nbase_seed={_base_seed}\n"
            f"num_workers={_num_workers}\nworker_seeds={_worker_seeds}\n")
    except Exception as e:
        print(f"[seed_weaver] could not write batch_stream_sha: {e}", flush=True)

print(f"[seed_weaver] seed={seed} applied (python/numpy/torch/cuda); "
      f"cudnn.deterministic=True; DataLoader base_seed pinned to {_base_seed}; "
      f"batch_stream_sha={_batch_stream_sha[:16]}… ; handing off to weaver.train:main",
      flush=True)

from weaver.train import main  # noqa: E402

sys.argv = ["weaver"] + rest
main()
