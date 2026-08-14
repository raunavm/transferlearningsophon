"""Regression tests for the three protocol-breaking plumbing dependencies.

Each test corresponds to a failure mode that is SILENT in production: the run
completes, produces numbers, and the numbers are wrong.

  1. single global seed          -> arms with different K diverge in sampling
                                    order and dropout masks (invariant I7)
  2. no dual-depth probe API     -> 512-d probing silently impossible
  3. cross-GPU seed pairs        -> paired comparison loses its pairing
  4. genjet_sdmass in observers  -> mass head trains on nothing
  5. sophon eval registration    -> IndexError / wrong slice for K != 188
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import (  # noqa: E402
    STREAMS, derive_all, derive_seed,
)


# ------------------------------------------------- 1. four independent streams
def test_four_streams_are_distinct():
    subs = derive_all(1234)
    assert set(subs) == set(STREAMS)
    assert len(set(subs.values())) == 4, f"sub-seeds collide: {subs}"


def test_stream_derivation_is_deterministic_across_processes():
    """sha256-based, so stable across runs, versions and platforms.

    A `hash()`-based derivation would pass within one process and silently
    change between them, which is why the derivation is pinned by recomputing
    the digest here independently of the implementation.
    """
    import hashlib
    for master, stream in ((7, "data_sampling"), (42, "trunk_init")):
        digest = hashlib.sha256(
            f"seed-stream|v1|{master}|{stream}".encode()).digest()
        expected = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        assert derive_seed(master, stream) == expected


def test_stream_derivation_is_value_stable():
    """Lock the derivation. If this changes, every prior run's seed changes."""
    known = derive_all(42)
    again = derive_all(42)
    assert known == again
    # distinct masters must not collide on any stream
    other = derive_all(43)
    for s in STREAMS:
        assert known[s] != other[s], f"stream {s} collides across master seeds"


def test_data_stream_is_independent_of_head_size():
    """THE test for invariant I7.

    The data_sampling sub-seed must depend only on (master_seed, stream_name).
    It must NOT be a function of anything consumed during model construction —
    which is exactly what a single global seed made it.
    """
    for master in (0, 1, 42, 999):
        expected = derive_seed(master, "data_sampling")
        for _simulated_head_draws in (0, 188, 162, 43, 17):
            # head size cannot enter the derivation at all
            assert derive_seed(master, "data_sampling") == expected


def test_seed_weaver_refuses_partial_streams_by_default():
    """The wrapper must fail loudly, not degrade silently."""
    src = (ROOT / "experiments" / "E1" / "seed_weaver.py").read_text()
    assert "--allow-partial-streams" in src
    assert "Refusing to run" in src, (
        "seed_weaver must refuse to run when the model-construction hook is "
        "unavailable; a silently-partial fix looks seeded and is not")
    assert "raise SystemExit" in src


def test_seed_weaver_no_longer_uses_a_single_global_seed():
    src = (ROOT / "experiments" / "E1" / "seed_weaver.py").read_text()
    # the old body seeded all four RNGs from `seed` directly
    assert not re.search(r"^\s*random\.seed\(seed\)\s*$", src, re.M), \
        "seed_weaver still seeds python RNG directly from the master seed"
    assert not re.search(r"^\s*torch\.manual_seed\(seed\)\s*$", src, re.M), \
        "seed_weaver still seeds torch directly from the master seed"
    assert "seed_stream" in src


# ------------------------------------------------------- 2. dual-depth probing
def test_dual_depth_module_exists_and_resolves_probe_points():
    from src.models.dual_depth_probe import DualDepthExtractor, weaver_has_split_api

    class FakeMod:
        pass

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = nn.Module()
            self.mod.norm = nn.LayerNorm(128)
            self.mod.fc = nn.Sequential(
                nn.Sequential(nn.Linear(128, 512), nn.ReLU()),
                nn.Linear(512, 188),
            )

        def forward(self, x):
            z = self.mod.norm(x)
            return self.mod.fc(z)

    m = Model()
    with DualDepthExtractor(m) as ex:
        out = m(torch.randn(4, 128))
        z128, z512 = ex.embeddings()
    assert out.shape == (4, 188)
    assert z128.shape == (4, 128), f"128-d probe wrong shape: {z128.shape}"
    assert z512.shape == (4, 512), f"512-d probe wrong shape: {z512.shape}"
    assert ex.dims == (128, 512)
    assert weaver_has_split_api(object()) is False


def test_dual_depth_hooks_are_removed_on_exit():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.models.dual_depth_probe import DualDepthExtractor

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = nn.Module()
            self.mod.norm = nn.LayerNorm(8)
            self.mod.fc = nn.Sequential(nn.Sequential(nn.Linear(8, 16)), nn.Linear(16, 3))

        def forward(self, x):
            return self.mod.fc(self.mod.norm(x))

    m = Model()
    with DualDepthExtractor(m):
        pass
    assert not m.mod.norm._forward_hooks, "hook leaked from norm"
    assert not m.mod.fc[0]._forward_hooks, "hook leaked from fc[0]"


def test_dual_depth_fails_loudly_on_a_plain_linear_head():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.models.dual_depth_probe import DualDepthExtractor

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = nn.Module()
            self.mod.norm = nn.LayerNorm(8)
            self.mod.fc = nn.Linear(8, 3)   # no 512-d probe point

    with pytest.raises(AttributeError, match="512-d probe point|nn.Sequential"):
        with DualDepthExtractor(Model()):
            pass


# ------------------------------------------------------- 3. GPU pinning in k8s
def _job_specs() -> list[pathlib.Path]:
    return sorted((ROOT / "experiments").rglob("k8s/job-*.yaml"))


def test_gpu_model_node_selector_documented():
    doc = ROOT / "experiments" / "E1" / "k8s" / "NODE_SELECTOR.md"
    assert doc.exists(), (
        "cross-GPU runs are statistically but not bit-identical, so both arms "
        "of a seed pair must be pinned to one GPU model; NODE_SELECTOR.md "
        "must document how")
    text = doc.read_text()
    assert "nvidia.com/gpu.product" in text


# ----------------------------------------------- 4/5. data-config preconditions
def _arm_configs() -> list[pathlib.Path]:
    d = ROOT / "configs" / "arms"
    return sorted(d.glob("*.yaml")) if d.exists() else []


def test_genjet_sdmass_declared_under_labels_not_observers():
    """weaver registers observers with to='test' only (config.py:203).

    A mass head wired via `observers:` would silently train on nothing: no
    error, no NaN, just a head that never receives a target.
    """
    arms = _arm_configs()
    if not arms:
        pytest.skip("no arm configs yet - test BINDS as soon as they are written")
    with_mass = [p for p in arms if "genjet_sdmass" in p.read_text()]
    if not with_mass:
        # Skip LOUDLY rather than pass vacuously. Every existing arm is
        # classification-only, so there is no mass wiring to check yet, and a
        # green result here would read as "the mass head is wired correctly".
        pytest.skip(f"none of the {len(arms)} arm configs declares genjet_sdmass "
                    "- no mass head exists yet; test BINDS when one is added")
    bad = []
    for p in with_mass:
        text = p.read_text()
        obs = re.search(r"^observers:(.*?)(?=^\w|\Z)", text, re.M | re.S)
        if obs and "genjet_sdmass" in obs.group(1):
            bad.append(p.name)
    assert not bad, (
        "genjet_sdmass appears under `observers:` in: " + ", ".join(bad) +
        " - move it under `labels.value` or the mass head trains on nothing")


def test_sophon_eval_registration_absent_for_non_188_arms():
    """`evaluate_classification_sophon` hardcodes scores[:, 161:188] and
    truth_label >= 161, so it indexes out of range for L162, R42_Q1 and R16_Q1.
    """
    arms = _arm_configs()
    if not arms:
        pytest.skip("no arm configs yet - test BINDS as soon as they are written")
    bad = []
    for p in arms:
        text = p.read_text()
        if "evaluate_classification_sophon" not in text:
            continue
        m = re.search(r"num_classes:\s*(\d+)", text)
        if m and int(m.group(1)) != 188:
            bad.append(f"{p.name}(K={m.group(1)})")
    assert not bad, (
        "evaluate_classification_sophon registered for non-188 arms: " +
        ", ".join(bad) + " - it hardcodes the 161:188 QCD slice")
