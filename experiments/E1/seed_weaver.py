#!/usr/bin/env python3
"""Seed wrapper for weaver 0.4.17 (which exposes no --seed).

FOUR INDEPENDENT RNG STREAMS (invariant I7)
-------------------------------------------
The previous version set ONE global seed for python/numpy/torch/cuda. That is
not safe for this study. Model construction draws from the same generator as
data sampling and dropout, so an arm with a 188-way head and an arm with a
17-way head consume different numbers of values while building the head, and
every subsequent draw is offset. The two arms then see a different sampling
order and different dropout masks even though their data configs are
byte-identical — silently destroying the paired comparison the seed budget
assumes.

This wrapper derives four sub-seeds (trunk_init, head_init, data_sampling,
dropout) from the master seed via sha256 and re-seeds the global RNG at each
phase boundary, so head size cannot perturb the data order.

HONEST LIMITATION, read before trusting the guarantee
-----------------------------------------------------
weaver builds the model internally, so the head_init/dropout boundary can only
be hit by intercepting weaver's model-construction call. This wrapper attempts
that interception and VERIFIES it took effect. If the installed weaver does not
expose the expected symbol, the wrapper FAILS LOUDLY rather than continuing
with a partial guarantee — a silently-partial fix here is worse than no fix,
because the run would look seeded and not be.

Usage:  python seed_weaver.py --seed S [--allow-partial-streams] <weaver args>

Reproducibility: a run is reproducible on the SAME GPU model. Cross-GPU is
statistically — not bit — identical (different CUDA kernels and reduction
orders). Pin the GPU model in the job spec; see k8s/NODE_SELECTOR.md.
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _pop(argv: list[str], flag: str, has_value: bool = True):
    if flag not in argv:
        return None, argv
    i = argv.index(flag)
    if not has_value:
        return True, argv[:i] + argv[i + 1:]
    return argv[i + 1], argv[:i] + argv[i + 2:]


seed_raw, rest = _pop(sys.argv[1:], "--seed")
if seed_raw is None:
    raise SystemExit("seed_weaver: --seed S is required")
seed = int(seed_raw)
allow_partial, rest = _pop(rest, "--allow-partial-streams", has_value=False)

os.environ["PYTHONHASHSEED"] = str(seed)

import torch  # noqa: E402

from src.utils.reproducibility import (  # noqa: E402
    STREAMS, derive_all, seed_stream,
)

subs = derive_all(seed)
print(f"[seed_weaver] master seed={seed}", flush=True)
for name in STREAMS:
    print(f"[seed_weaver]   {name:<14} -> {subs[name]}", flush=True)

# Phase 1 - trunk init. weaver builds the model after this point.
seed_stream(seed, "trunk_init")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import weaver.train as weaver_train  # noqa: E402

# Phase 2/3 - re-seed at the model-construction boundary so head size cannot
# offset the data stream. We wrap weaver's model builder rather than guessing.
_HOOK_CANDIDATES = ("model_setup", "_model_setup", "build_model")
_hooked = None
for _name in _HOOK_CANDIDATES:
    _fn = getattr(weaver_train, _name, None)
    if callable(_fn):
        def _wrapped(*a, __fn=_fn, **kw):
            seed_stream(seed, "head_init")          # head construction
            out = __fn(*a, **kw)
            seed_stream(seed, "dropout")            # train-time dropout/augmentation
            print(f"[seed_weaver] re-seeded head_init/dropout around "
                  f"weaver.train.{__fn.__name__}", flush=True)
            return out
        setattr(weaver_train, _name, _wrapped)
        _hooked = _name
        break

if _hooked is None:
    msg = ("seed_weaver: could not find weaver's model-construction hook "
           f"(tried {_HOOK_CANDIDATES}). Head-size-independent seeding is NOT "
           "in effect, so arms with different K may diverge in sampling order "
           "and dropout masks (invariant I7). Refusing to run.\n"
           "  Fix: use the weaver dev branch, or pass --allow-partial-streams "
           "to proceed with data_sampling separated but head/dropout coupled "
           "(acceptable ONLY for single-arm smoke tests, never for a seed pair).")
    if not allow_partial:
        raise SystemExit(msg)
    print("[seed_weaver] WARNING (--allow-partial-streams):\n" + msg, flush=True)

# Phase 4 - data sampling. Seeded last so the sampler state is a function of
# (master_seed, "data_sampling") alone, independent of anything drawn above.
seed_stream(seed, "data_sampling")
print("[seed_weaver] handing off to weaver.train:main", flush=True)

sys.argv = ["weaver"] + rest
weaver_train.main()
