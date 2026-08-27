"""CI tests for I1 AT THE LAUNCH LAYER -- the k8s job specs.

WHY THIS FILE EXISTS. tests/test_arm_configs.py asserts I1 ("arms differ in
NOTHING but the label vocabulary") by comparing sections of configs/arms/*.yaml.
That is a real check and it passes. It is also blind to the way I1 was actually
broken, because the learning rate is not in configs/arms/*.yaml at all -- it is
`--start-lr` in the k8s job spec. Measured 2026-08-27:

    job-mtx-l162-s1-raunav.yaml     --start-lr 1e-3
    job-mtx-r16_q1-s1..s5-raunav    --start-lr 5e-4

so the headline pair differs in learning rate as well as vocabulary, while
test_arms_differ_only_in_vocabulary stays green. A green invariant test that
structurally cannot see the violation is worse than no test, because it is read
as evidence the invariant holds.

The per-arm rate is not itself a defect -- docs/GATES.md's G1 KILL branch says
arms are compared at their own optima, which requires per-arm rates. The defect
is that the difference is INVISIBLE to CI and therefore unauditable. These tests
do not forbid a per-arm rate; they force every rate to be DECLARED in one place,
with evidence, and force the committed spec to match the declaration.

Run:  python3 -m pytest tests/test_launch_specs.py -v
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SPEC_DIR = ROOT / "experiments" / "MTX" / "k8s"

# job-mtx-<armslug>-s<N>[suffix]-raunav.yaml  ->  the armslug, lowercased with
# underscores, as the filenames actually spell it (r16_q1, not r16q1). The
# optional letter suffix carries a re-run of the same seed at a different rate
# (mtx-l162-s1b); it must be matched, or the one spec written specifically to
# repair a rate would be the one spec the rate tests never look at.
SPEC_RE = re.compile(r"^job-mtx-([a-z0-9_]+)-s(\d+[a-z]?)-raunav\.yaml$")
LR_RE = re.compile(r"--start-lr\s+([0-9.eE+-]+)")

# A spec may opt out of the rate assertions by carrying this marker with a
# reason. That is for a run ALREADY LAUNCHED at a rate the project has since
# moved off: its spec is a historical record of what is on the cluster, and
# rewriting it would make the repo disagree with the running pod. The marker
# does not hide the difference -- test_exceptions_are_explained prints every one
# -- it only distinguishes a declared exception from silent drift.
EXCEPT_RE = re.compile(r"^\s*#\s*RATE-EXCEPTION:\s*(\S.*)$", re.M)


def declared_rates() -> dict[str, str]:
    """The single place a per-arm rate may be declared, with its evidence."""
    from build_mtx_launch import RATES
    return {arm.lower(): rate for arm, (_k, rate, _br, _why) in RATES.items()}


def arm_specs() -> dict[str, list[pathlib.Path]]:
    """armslug -> its seed specs. Only per-arm training jobs; makeweight and
    any other utility job is not an arm and carries no vocabulary."""
    out: dict[str, list[pathlib.Path]] = {}
    if not SPEC_DIR.exists():
        return out
    for p in sorted(SPEC_DIR.glob("job-mtx-*.yaml")):
        m = SPEC_RE.match(p.name)
        if m:
            out.setdefault(m.group(1), []).append(p)
    return out


def start_lr(path: pathlib.Path) -> str:
    text = path.read_text()
    # Comments in these specs quote historical rates at length; only the live
    # command line counts.
    live = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    hits = LR_RE.findall(live)
    assert hits, f"{path.name} has no --start-lr; the rate is not recorded anywhere"
    assert len(set(hits)) == 1, f"{path.name} passes --start-lr more than once: {hits}"
    return hits[0]


def rate_exception(path: pathlib.Path) -> str | None:
    m = EXCEPT_RE.search(path.read_text())
    return m.group(1).strip() if m else None


@pytest.fixture(scope="module")
def specs():
    found = arm_specs()
    if not found:
        pytest.skip("no experiments/MTX/k8s/job-mtx-<arm>-s<N>-raunav.yaml specs")
    return found


def test_every_spec_states_its_learning_rate(specs):
    """A rate that is not written down cannot be audited after the run."""
    for arm, paths in specs.items():
        for p in paths:
            start_lr(p)


def test_seeds_of_one_arm_share_a_rate(specs):
    """I7: arms in a seed pair differ in exactly one variable. Two seeds of the
    SAME arm differ in the seed alone, so a rate split across seeds of one arm
    is never intentional -- it is drift between a hand-edited spec and its
    siblings."""
    offenders = []
    for arm, paths in sorted(specs.items()):
        rates = {p.name: start_lr(p) for p in paths if not rate_exception(p)}
        if len(set(rates.values())) > 1:
            offenders.append(f"{arm}: {rates}")
    assert not offenders, (
        "seeds of one arm disagree on --start-lr:\n  " + "\n  ".join(offenders))


def test_spec_rate_matches_the_declared_rate(specs):
    """The committed spec must match scripts/build_mtx_launch.py RATES.

    This is the one that fires today. RATES declares R42_Q1 at 2.5e-4 -- the
    only rate that ever trained for that arm, with 5e-4 and 1e-3 both going nan
    at iteration 2 -- while job-mtx-r42_q1-s1-raunav.yaml is written at 5e-4.
    The spec is unlaunched, so this is free to fix; once launched it would be
    eight days of compute at a rate the arm's own sweep contradicts."""
    declared = declared_rates()
    offenders = []
    for arm, paths in sorted(specs.items()):
        want = declared.get(arm)
        if want is None:
            offenders.append(f"{arm}: no entry in build_mtx_launch.RATES")
            continue
        for p in paths:
            if rate_exception(p):
                continue
            got = start_lr(p)
            if float(got) != float(want):
                offenders.append(f"{p.name}: spec {got} != declared {want}")
    assert not offenders, (
        "launch specs disagree with the declared per-arm rates:\n  " +
        "\n  ".join(offenders))


def test_exceptions_are_explained(specs):
    """Every RATE-EXCEPTION must carry a reason, and they are printed on every
    run so an exception cannot quietly become permanent."""
    for arm, paths in sorted(specs.items()):
        for p in paths:
            why = rate_exception(p)
            if why is None:
                continue
            assert len(why) > 20, (
                f"{p.name}: RATE-EXCEPTION needs a real reason, got {why!r}")
            print(f"RATE-EXCEPTION {p.name} (lr={start_lr(p)}, "
                  f"declared={declared_rates().get(arm)}): {why}")


def test_a_rate_split_across_arms_is_declared_not_accidental(specs):
    """I1 at the launch layer.

    Arms MAY carry different rates -- G1 returned KILL and docs/GATES.md
    compares arms at their own optima. What is not allowed is a rate difference
    that exists only in a job spec. Every distinct rate must trace to a RATES
    entry, so the confound is visible to anyone reading one file."""
    declared = declared_rates()
    used = {}
    for arm, paths in specs.items():
        live = [p for p in paths if not rate_exception(p)]
        if live:
            used[arm] = start_lr(live[0])
    if len(set(used.values())) == 1:
        return  # single shared rate: I1 holds at the launch layer trivially
    undeclared = {arm: r for arm, r in used.items()
                  if arm not in declared or float(declared[arm]) != float(r)}
    assert not undeclared, (
        "arms are trained at DIFFERENT rates and at least one is undeclared, so "
        "the headline contrast varies vocabulary AND learning rate with nothing "
        "recording it:\n  " + "\n  ".join(f"{a}: {r}" for a, r in undeclared.items()))
