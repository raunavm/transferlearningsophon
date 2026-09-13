"""D8's probe axes must be able to tell the random control from R16_Q1.

An audit (2026-09-12) found that every probe task defined before this file
existed uses a class pair that R16_Q1 and all three RAND draws group the SAME
way, so the paper's primary novelty leg was arithmetically incapable of
producing a result other than "only K matters". These tests pin the two axes
added to fix that, and pin the diagnosis, so neither can quietly regress.
"""
import csv
import importlib.util
import itertools
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAPS = ROOT / "configs" / "labelmaps"


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", ROOT / "experiments/EVAL/probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rand():
    with (MAPS / "rand_label_map.v1.csv").open() as f:
        return {int(r["jet_label"]): r for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def rung():
    with (MAPS / "rung_label_maps.v1.csv").open() as f:
        return {int(r["jet_label"]): r for r in csv.DictReader(f)}


def _groups(rand, a, b, col):
    return rand[a][col], rand[b][col]


# --------------------------------------------------------------- the diagnosis

@pytest.mark.parametrize("task,a,b", [
    ("bvc_resonant", 0, 1),
    ("ee_vs_mm", 10, 11),
    ("bvc_qcd", 169, 181),
])
def test_the_pre_existing_axes_cannot_separate_the_control_from_r16q1(rand, task, a, b):
    """The finding itself, pinned. These three merge identically under both
    maps, so on them the control is R16_Q1 by construction. This is not a
    defect to fix in the tasks -- they are correct as granularity probes -- it
    is the reason the two D8 axes below had to be added."""
    r16 = _groups(rand, a, b, "R16_Q1")
    rd1 = _groups(rand, a, b, "RAND_d1")
    assert (r16[0] == r16[1]) == (rd1[0] == rd1[1]), (
        f"{task}: if this ever differs the diagnosis has changed")


def test_res2p_and_qcd_carry_none_of_the_controls_power(rand):
    """Counted, not asserted: where the control can and cannot differ."""
    split = {"res2p": 0, "res34p": 0, "qcd": 0}
    merged = {"res2p": 0, "res34p": 0, "qcd": 0}
    for a, b in itertools.combinations(sorted(rand), 2):
        name = rand[a]["RAND_d1_name"]
        if rand[b]["RAND_d1_name"].split("_")[1] != name.split("_")[1]:
            continue
        block = name.split("_")[1]
        if block not in split:
            continue
        if rand[a]["R16_Q1"] != rand[b]["R16_Q1"]:
            continue
        if rand[a]["RAND_d1"] != rand[b]["RAND_d1"]:
            split[block] += 1
        else:
            merged[block] += 1

    assert split["res2p"] == 0, (
        "res2p is share-rigid, so a share-matched control reproduces R16_Q1 "
        "there exactly; a non-zero count means the control changed")
    assert split["qcd"] == 0, "the QCD block is copied by design"
    assert split["res34p"] > 500, (
        f"res34p is where ALL the control's power lives; got {split['res34p']}")


# ------------------------------------------------------------------- the axes

def test_both_d8_axes_are_registered():
    p = _probe()
    assert "bvc_4prong" in p.TASKS and "visible_content" in p.TASKS
    assert set(p.D8_TASKS) == {"bvc_4prong", "visible_content"}


def test_bvc_4prong_is_merged_by_r16q1_and_split_by_the_control(rand):
    p = _probe()
    a = p.TASKS["bvc_4prong"]["signal"][0]
    b = p.TASKS["bvc_4prong"]["background"][0]
    assert rand[a]["R16_Q1"] == rand[b]["R16_Q1"], "R16_Q1 must MERGE this pair"
    assert rand[a]["RAND_d1"] != rand[b]["RAND_d1"], "RAND_d1 must SPLIT it"


def test_visible_content_is_split_by_r16q1_and_merged_by_the_control(rand):
    p = _probe()
    a = p.TASKS["visible_content"]["signal"][0]
    b = p.TASKS["visible_content"]["background"][0]
    assert rand[a]["R16_Q1"] != rand[b]["R16_Q1"], "R16_Q1 must SPLIT this pair"
    assert rand[a]["RAND_d1"] == rand[b]["RAND_d1"], "RAND_d1 must MERGE it"


def test_the_two_axes_point_in_opposite_directions(rand):
    """A one-sided pair could not distinguish 'semantics matter' from 'this
    particular map happens to be better'. The test is only rigorous because
    each map is favoured on exactly one axis."""
    p = _probe()
    dirs = set()
    for name in ("bvc_4prong", "visible_content"):
        a = p.TASKS[name]["signal"][0]
        b = p.TASKS[name]["background"][0]
        dirs.add((rand[a]["R16_Q1"] == rand[b]["R16_Q1"],
                  rand[a]["RAND_d1"] == rand[b]["RAND_d1"]))
    assert dirs == {(True, False), (False, True)}


def test_the_two_axes_share_a_reference_class():
    """Controls for that class's own learnability across the two axes."""
    p = _probe()
    assert (p.TASKS["bvc_4prong"]["signal"]
            == p.TASKS["visible_content"]["signal"])


def test_the_fine_arm_splits_both_axes(rung):
    """L162 is the upper reference: it must retain both distinctions, else the
    axis measures something other than what the contraction removed."""
    p = _probe()
    for name in ("bvc_4prong", "visible_content"):
        a = p.TASKS[name]["signal"][0]
        b = p.TASKS[name]["background"][0]
        assert rung[a]["L162"] != rung[b]["L162"], f"{name} is merged at L162"


def test_bvc_4prong_collapses_exactly_where_the_headline_flavour_task_does():
    """It is the 4-prong analogue of bvc_resonant, so it must carry the same
    collapse signature -- same question, same depth, different sector."""
    p = _probe()
    assert (p.TASKS["bvc_4prong"]["collapsed_at"]
            == p.TASKS["bvc_resonant"]["collapsed_at"]
            == ["R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"])


# ------------------------------------------------------------- the statistics

# Counts in the committed 2,000,000-jet feature cache, measured read-only on the
# PVC 2026-09-12. The probe needs MIN_PER_CLASS in the 20 % test split, so the
# raw count must clear 5x that. Pinned because item 26 is a standing example of
# an axis committed without checking and found unmeasurable afterwards.
CACHE_COUNTS = {18: 15647, 34: 16078, 158: 12664}


def test_every_d8_class_clears_the_statistics_floor_with_margin():
    p = _probe()
    for name in ("bvc_4prong", "visible_content"):
        for lab in p.TASKS[name]["signal"] + p.TASKS[name]["background"]:
            n_test = CACHE_COUNTS[lab] * 0.2
            assert n_test >= 2 * p.MIN_PER_CLASS, (
                f"label {lab} gives {n_test:.0f} test jets against a floor of "
                f"{p.MIN_PER_CLASS}; a 2x margin is required because the split "
                f"fraction is fixed but the cache size is not")


def test_the_rejected_candidate_is_documented_as_too_tight():
    """bbbb-vs-cccc was the obvious first choice and clears the floor by 8 %.
    Pinned so nobody re-proposes it."""
    p = _probe()
    src = (ROOT / "experiments/EVAL/probe.py").read_text()
    assert "bbbb-vs-cccc was the first" in src
    assert p.TASKS["bvc_4prong"]["signal"] != [15]
