"""Job preconditions must match the files the job's script actually opens.

THIS DEFECT CLASS HAS NOW APPEARED THREE TIMES and each instance was found by a
different audit rather than by CI:

  probe-physics-v4   checked observers.npz, never label188.npy or
                     extract_manifest.json          (found 2026-09-10)
  eval-labelrec-v3   a draft promoted inert prose about logits into a hard
                     `exit 1` under set -euo pipefail, for a script with zero
                     occurrences of the word                  (found 2026-09-10)
  eval-anomaly x4    same inversion as v4                     (found 2026-09-12)

The shape is always the same: a spec hard-fails on an OPTIONAL input while
never checking a MANDATORY one. Both halves cost something real -- the missing
check turns a clear precondition failure into a bare traceback after the clone
and any expensive setup, and the spurious check refuses a cache that would have
worked. A one-off fix does not stop a fourth instance; this does.

The contract, derived from experiments/EVAL/probe.py::load_arm:
    MANDATORY  features.npy, label188.npy, extract_manifest.json
    OPTIONAL   observers.npz   (opened under `if (d / "observers.npz").exists()`)
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "EVAL" / "k8s"

MANDATORY = ("features.npy", "label188.npy", "extract_manifest.json")
OPTIONAL = ("observers.npz",)

# Specs whose command line runs a script that goes through probe.load_arm.
FEATURE_CONSUMERS = ("anomaly.py", "probe.py", "label_recovery.py")


def _specs():
    return sorted(K8S.glob("job-*.yaml"))


def _reads_features(text):
    """Whether the spec RUNS a feature consumer -- comment lines excluded.

    A bare substring match over the whole file counts a filename MENTIONED in a
    comment as a feature cache being read. That is not hypothetical: the anomaly
    merge job consumes per-arm JSON and touches no cache, and it started failing
    this rule the moment its header comment explained what anomaly.py's scoring
    function does. The rule would then have been satisfied by prechecking three
    files the job never opens, which is worse than not checking -- it is a guard
    asserting a precondition that has nothing to do with the job.

    Dropping comment lines is the whole fix: an invocation is never `#`-prefixed,
    and this reclassifies exactly one spec in the directory (the merge job,
    True -> False) while every genuine consumer stays in."""
    live = "\n".join(l for l in text.splitlines()
                     if not l.lstrip().startswith("#"))
    return any(s in live for s in FEATURE_CONSUMERS)


def _windowed_tasks():
    """Task names carrying a `window`, read from probe.py rather than listed
    here, so adding a windowed task cannot silently invalidate this rule."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_for_specs", ROOT / "experiments/EVAL/probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    names = {k for k, v in m.TASKS.items() if v.get("window")}
    assert names, "no windowed task found in probe.TASKS; the rule below is dead"
    return names


def _runs_a_windowed_task(text):
    return any(name in text for name in _windowed_tasks())


def _hard_fails_on(text, fname):
    """Does the spec `exit 1` when fname is absent?

    Covers both idioms these specs use: a direct `[ -f ... ] || { ...; exit 1; }`
    naming the file, and a `for f in a b c; do [ -f ... ] || exit 1; done` loop.

    Loops are located by their header and scanned FORWARD by a fixed window
    rather than matched to a closing `done`. These specs nest loops (an outer
    one over arms, an inner one over filenames), and a non-greedy `(.*?)done`
    binds the outer header to the INNER `done`, consuming the inner header so
    findall never sees it -- which made an earlier version of this helper report
    that the already-fixed probe-physics-v4 had no label188.npy check.
    """
    if re.search(rf'\[ -f "\$\{{d\}}/{re.escape(fname)}" \]', text):
        return True
    for m in re.finditer(r"for \w+ in ([^;\n]+); do", text):
        if fname not in m.group(1).split():
            continue
        if "exit 1" in text[m.end():m.end() + 400]:
            return True
    return False


# Specs that ALREADY RAN with the inversion. They are provenance records of what
# was executed, and editing one would make the repository disagree with the pod
# that produced the result -- so they are frozen, not fixed. Each is superseded
# by a later spec in the same family which this file does enforce. The list may
# SHRINK (a spec is deleted) but an addition means a new spec shipped with the
# defect, which test_the_frozen_list_does_not_grow catches.
FROZEN = {
    "job-probe-bvc-raunav.yaml": "superseded by probe-bvc-v2, then probe-physics-v4",
    "job-probe-bvc-v2-raunav.yaml": "superseded by probe-physics-v4",
    "job-probe-physics-raunav.yaml": "v1; ran on mtx-s1.18, superseded by v4",
    "job-probe-physics-v2-raunav.yaml": "ran 2026-09-08, superseded by v4",
    "job-probe-physics-v3-raunav.yaml": "ran, superseded by v4",
    "job-eval-labelrec-raunav.yaml": "v1, superseded by v3",
    "job-eval-labelrec-v2-raunav.yaml": "ran, superseded by v3",
    "job-eval-fillcontrol-raunav.yaml": "ran; superseded by fillcontrol2",
    "job-eval-fillcontrol2-raunav.yaml": "ran 2026-09-08",
    "job-latent-scale-raunav.yaml": "ran",
    "job-latent-scale-sophon-raunav.yaml": "ran",
}


def test_the_contract_matches_probe_load_arm():
    """Pin the contract against its source, so this file cannot drift from the
    code it encodes."""
    src = (ROOT / "experiments/EVAL/probe.py").read_text()
    body = src[src.index("def load_arm"):][:1600]
    for f in MANDATORY:
        assert f'"{f}"' in body, f"{f} no longer opened by load_arm"
    assert 'if (d / "observers.npz").exists()' in body, (
        "observers.npz is no longer guarded by .exists(); if it became "
        "mandatory, move it to MANDATORY above")


@pytest.mark.parametrize("spec", _specs(), ids=lambda p: p.name)
def test_no_spec_hard_fails_on_an_optional_input(spec):
    text = spec.read_text()
    if not _reads_features(text):
        pytest.skip("does not consume a feature cache")
    if spec.name in FROZEN:
        pytest.skip(f"frozen provenance record: {FROZEN[spec.name]}")
    for f in OPTIONAL:
        if f == "observers.npz" and _runs_a_windowed_task(text):
            # DELIBERATE, not a defect. A windowed task (bc_vs_rest) needs
            # jet_pt / jet_sdmass / jet_eta to apply the published selection.
            # probe.py would SKIP it with a named message rather than compute
            # it over the whole spectrum, so a spec whose headline task is the
            # windowed one is right to refuse to start without them. This is
            # the distinction the blunter rule missed: the inversion is gating
            # on an input the job does not need, not gating at all.
            continue
        assert not _hard_fails_on(text, f), (
            f"{spec.name} refuses to start without {f}, which load_arm opens "
            f"only if it exists, and it requests no windowed task. A cache "
            f"lacking it would run fine.")


@pytest.mark.parametrize("spec", _specs(), ids=lambda p: p.name)
def test_every_spec_checks_every_mandatory_input(spec):
    text = spec.read_text()
    if not _reads_features(text):
        pytest.skip("does not consume a feature cache")
    if spec.name in FROZEN:
        pytest.skip(f"frozen provenance record: {FROZEN[spec.name]}")
    missing = [f for f in MANDATORY if not _hard_fails_on(text, f)]
    assert not missing, (
        f"{spec.name} does not precheck {missing}, which load_arm opens "
        f"unconditionally. Absent, the job clones, builds logits, and then dies "
        f"on a bare traceback.")


def test_this_file_is_not_vacuous():
    """Every test above skips on specs that do not read features. If the
    detection ever breaks, all of them skip and the suite still reads green."""
    consumers = [s for s in _specs() if _reads_features(s.read_text())]
    assert len(consumers) >= 4, (
        f"only {len(consumers)} specs detected as feature consumers; the "
        f"detector has probably broken and these tests are now no-ops")


def test_no_spec_gates_on_a_file_its_script_never_opens():
    """The eval-labelrec-v3 half of the defect class: a hard gate on an input
    the target script has no code path for."""
    lr = (ROOT / "experiments/EVAL/label_recovery.py").read_text()
    assert "logits" not in lr, (
        "label_recovery.py now mentions logits; the assertion below was "
        "written on the fact that it does not")
    for spec in _specs():
        text = spec.read_text()
        if "label_recovery.py" not in text or spec.name in FROZEN:
            continue
        assert not _hard_fails_on(text, "logits.npy"), (
            f"{spec.name} gates on logits.npy but runs label_recovery.py, "
            f"which never opens it")


def test_the_frozen_list_does_not_grow():
    """FROZEN records specs that already ran with the defect. A spec written
    from today on must pass outright; if one needs adding here, the defect
    shipped again and that is the finding."""
    assert len(FROZEN) <= 11, (
        "a spec was added to FROZEN, meaning a NEW spec shipped with the "
        "precondition inversion this file exists to stop")
    for name in FROZEN:
        assert (K8S / name).exists(), f"{name} is frozen but no longer exists"


def test_the_current_specs_are_actually_enforced():
    """Guards against the frozen list swallowing everything."""
    enforced = [s.name for s in _specs()
                if _reads_features(s.read_text()) and s.name not in FROZEN]
    assert len(enforced) >= 5, f"only {enforced} left enforced"
