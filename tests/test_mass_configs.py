"""CI for the *_MASS arm configs: the mass target is what D2 (as amended by
DECISIONS_PENDING item 3) says it is, and a mass twin differs from its arm in
the two mass labels alone.

Run:  python3 -m pytest tests/test_mass_configs.py -v
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARM_DIR = ROOT / "configs" / "arms"


def _mass_pairs():
    pairs = []
    for p in sorted(ARM_DIR.glob("*_MASS.yaml")):
        twin = ARM_DIR / (p.stem[: -len("_MASS")] + ".yaml")
        if twin.exists():
            pairs.append((p, twin))
    if not pairs:
        pytest.skip("no *_MASS arm configs; run scripts/build_arm_configs.py")
    return pairs


def _label(text: str, name: str) -> str:
    m = re.search(rf"^\s+{name}:\s*(.*)$", text, re.M)
    assert m, f"no `{name}:` label"
    return m.group(1).strip()


def _eval(expr: str, **arrays):
    return eval(expr, {"__builtins__": {}}, {"np": np, **arrays})


@pytest.mark.parametrize("mass_path,twin_path", _mass_pairs())
def test_mass_target_is_the_groomed_log_ratio_masked_on_matching(mass_path, twin_path):
    text = mass_path.read_text()
    genjet = np.array([0.0, 50.0, 120.0, 0.0, 300.0])
    reco = np.array([60.0, 40.0, 100.0, 25.0, 310.0])
    target = _eval(_label(text, "mass_target"), genjet_sdmass=genjet, jet_sdmass=reco)
    valid = _eval(_label(text, "mass_valid"), genjet_sdmass=genjet, jet_sdmass=reco)
    assert valid.dtype == bool and valid.tolist() == [False, True, True, False, True]
    want = np.where(genjet > 0, np.log(np.maximum(genjet, 1e-6) / reco), 0.0)
    assert np.allclose(target, want)
    assert np.isfinite(target).all(), "an unmatched jet must never give -inf"
    assert target[0] == 0.0 and target[3] == 0.0


@pytest.mark.parametrize("mass_path,twin_path", _mass_pairs())
def test_mass_target_evaluates_on_awkward_arrays_too(mass_path, twin_path):
    """weaver evaluates label expressions on awkward columns, not numpy."""
    ak = pytest.importorskip("awkward")
    text = mass_path.read_text()
    genjet = ak.Array([0.0, 50.0, 120.0])
    reco = ak.Array([60.0, 40.0, 100.0])
    target = np.asarray(_eval(_label(text, "mass_target"), genjet_sdmass=genjet, jet_sdmass=reco))
    valid = np.asarray(_eval(_label(text, "mass_valid"), genjet_sdmass=genjet, jet_sdmass=reco))
    assert np.allclose(target, [0.0, np.log(50 / 40), np.log(1.2)])
    assert valid.tolist() == [False, True, True]


@pytest.mark.parametrize("mass_path,twin_path", _mass_pairs())
def test_mass_twin_differs_from_its_arm_in_the_two_labels_alone(mass_path, twin_path):
    mass, twin = mass_path.read_text(), twin_path.read_text()
    assert _label(mass, "truth_label") == _label(twin, "truth_label")
    hm = re.search(r"-o num_classes (\d+)", mass).group(1)
    ht = re.search(r"-o num_classes (\d+)", twin).group(1)
    assert hm == ht, "the mass arm is launched with the SAME K; the arch adds the node"
    # everything outside the labels block is byte-identical
    def outside_labels(t):
        a = re.search(r"^labels:", t, re.M).start()
        b = re.search(r"^observers:", t, re.M).start()
        return t[:a].split("\n", 8)[-1] + t[b:]   # drop the 8-line file header
    assert outside_labels(mass) == outside_labels(twin)
    # genjet_sdmass is a LABEL input, never an observer (tests/test_plumbing.py)
    obs = re.search(r"^observers:(.*?)^weights:", mass, re.M | re.S).group(1)
    assert "genjet_sdmass" not in obs
    for name in ("mass_target", "mass_valid"):
        assert name in mass and name not in twin
