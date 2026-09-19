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


def test_the_committed_configs_are_what_the_generator_produces_today():
    """Generator/artifact drift is invisible until someone regenerates, at which
    point a changed labels block changes the config md5, hence the reweighting
    sidecar's name, hence a required new make_weight pass. A mutation to
    MASS_LABEL_LINES (e.g. mass_valid always True) passed the whole suite
    because every other test reads only the committed YAML.
    """
    import csv as _csv
    import importlib.util
    import pathlib as _pathlib
    import sys as _sys

    root = _pathlib.Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "scripts"))
    import build_arm_configs as b

    maps = b.load_maps()
    rows = list(_csv.DictReader(b.MAPS.open()))
    names = {arm: {int(r[arm]): r[f"{arm}_name"] for r in rows} for arm in b.ARMS}
    base = b.BASE.read_text()
    todo = [(a, a, False) for a in b.ARMS] + [(f"{a}_MASS", a, True) for a in b.MASS_ARMS]
    for arm, src, mass in todo:
        want = (b.OUT_DIR / f"{arm}.yaml").read_text()
        got = b.build_one(base, arm, maps[src], names[src], mass=mass)
        assert got == want, f"configs/arms/{arm}.yaml is stale: regenerate it"


# ---------------------------------------------- the pairing C5 rests on

MASS_TWIN_SPECS = [
    (f"job-mtx-l162_mass-s{s}-raunav.yaml",
     f"job-mtx-l162-s{'1b' if s == 1 else s}-raunav.yaml", s) for s in range(1, 6)
] + [
    (f"job-mtx-r16_q1_mass-s{s}-raunav.yaml", f"job-mtx-r16_q1-s{s}-raunav.yaml", s)
    for s in range(1, 6)
]


def _seeds_in(spec: pathlib.Path) -> list[int]:
    return [int(l.split("--seed")[1].split()[0])
            for l in spec.read_text().splitlines() if "--seed" in l]


@pytest.mark.parametrize("mass_name,twin_name,seed", MASS_TWIN_SPECS)
def test_a_mass_run_and_its_plain_twin_carry_the_same_seed(mass_name, twin_name, seed):
    """C5 is a WITHIN-SEED difference-in-differences, so seed index N has to mean
    the same four RNG sub-streams on both sides of every pair. seed_weaver derives
    them as sha256("seed-stream|v1|<seed>|<stream>") -- no arm, no class count, no
    run id -- so identical `--seed` is exactly what makes the pairing valid.

    This held by construction and by audit, and nothing checked it:
    scripts/build_mass_jobs.py asserts the config, class count, rate, lambda,
    architecture, recipe, arm, name, wait guard and pin of each emitted spec, and
    does not assert the seed.
    """
    k8s = ROOT / "experiments" / "MTX" / "k8s"
    mass, twin = k8s / mass_name, k8s / twin_name
    assert mass.exists() and twin.exists(), (mass_name, twin_name)
    ms, ts = _seeds_in(mass), _seeds_in(twin)
    assert ms, f"{mass_name} passes no --seed at all"
    assert set(ms) == {seed}, f"{mass_name} carries {set(ms)}, not seed {seed}"
    assert set(ts) == {seed}, f"{twin_name} carries {set(ts)}, not seed {seed}"


def test_every_mass_run_has_exactly_one_plain_twin():
    """Ten pairs, and the 162-class seed 1 twin is the 5e-4 repair `s1b` -- NOT
    `mtx-l162-s1`, which trained at 1e-3 and is excluded from the study."""
    assert len(MASS_TWIN_SPECS) == 10
    twins = [t for _, t, _ in MASS_TWIN_SPECS]
    assert len(set(twins)) == 10
    assert "job-mtx-l162-s1b-raunav.yaml" in twins
    assert "job-mtx-l162-s1-raunav.yaml" not in twins
