"""Reproducibility utilities — four independent RNG streams (invariant I7).

Why four streams and not one global seed
---------------------------------------
A single global seed feeds model init, data sampling and dropout from the SAME
generator. Two arms whose heads differ in size (K = 188 vs 162 vs 43 vs 17) draw
a *different number of values* during head construction, so every subsequent
draw is offset. The arms then diverge in sampling order and dropout masks even
though the data config is byte-identical — which silently destroys the paired
comparison the seed budget assumes, and is exactly the failure mode I7 names.

Splitting the master seed into four independently derived streams makes each
stream's state a function of (master_seed, stream_name) only, so head size
cannot perturb the data order.

Streams
    trunk_init      backbone weight initialization
    head_init       classifier / regression head initialization
    data_sampling   sampler order, shuffling, DataLoader workers
    dropout         dropout masks and any train-time augmentation

Derivation is sha256-based, so it is stable across Python versions, platforms
and processes — unlike `hash()`, which is randomized per process.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

STREAMS = ("trunk_init", "head_init", "data_sampling", "dropout")

_MASK64 = (1 << 64) - 1


def derive_seed(master_seed: int, stream: str) -> int:
    """Derive a stable 63-bit sub-seed for one named stream.

    Stable across Python versions, platforms and processes. Never use `hash()`
    here — it is salted per process unless PYTHONHASHSEED is fixed, and relying
    on that is how a "reproducible" run stops being reproducible.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    digest = hashlib.sha256(f"seed-stream|v1|{master_seed}|{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (_MASK64 >> 1)


def derive_all(master_seed: int) -> dict[str, int]:
    return {s: derive_seed(master_seed, s) for s in STREAMS}


def seed_stream(master_seed: int, stream: str) -> int:
    """Seed the global torch/numpy/random RNGs from ONE named stream.

    Call immediately before the phase that stream governs. Returns the sub-seed
    so callers can log it.
    """
    sub = derive_seed(master_seed, stream)
    random.seed(sub)
    np.random.seed(sub % (2 ** 32))
    torch.manual_seed(sub)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sub)
    return sub


def data_generator(master_seed: int) -> torch.Generator:
    """A dedicated torch.Generator for DataLoader shuffling.

    Passing this as `DataLoader(generator=...)` decouples sampling order from
    the global RNG entirely, which is the strongest form of the guarantee.
    """
    g = torch.Generator()
    g.manual_seed(derive_seed(master_seed, "data_sampling"))
    return g


def worker_init_fn(master_seed: int):
    """DataLoader worker_init_fn deriving each worker's seed from data_sampling."""
    base = derive_seed(master_seed, "data_sampling")

    def _init(worker_id: int) -> None:
        sub = (base + worker_id) & _MASK64
        random.seed(sub)
        np.random.seed(sub % (2 ** 32))
        torch.manual_seed(sub)

    return _init


def seed_everything(seed: int = 42, deterministic: bool = True) -> dict[str, int]:
    """Seed all four streams and configure deterministic kernels.

    Retained for call-site compatibility. Prefer `seed_stream` at each phase
    boundary when the training loop is under our control; this function seeds
    the global RNGs from `trunk_init` and returns every sub-seed for logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    subs = derive_all(seed)
    seed_stream(seed, "trunk_init")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return subs
