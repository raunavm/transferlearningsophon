"""CI tests for invariants I1, I5 and I6 -- the arm configs.

These three were specified in docs/INVARIANTS.md but never written. They bind
now that configs/arms/ exists.

I1  arms differ in NOTHING but the label vocabulary
I5  architecture is fixed across arms
I6  jet_label is never re-derived from auxiliary variables

The failure these guard against is not a crash. It is a silently confounded
comparison: any co-varying hyperparameter turns the granularity effect into a
contrast between two different experiments, and it is unrecoverable after the
fact because nothing in the output records that it happened.

Run:  python3 -m pytest tests/test_arm_configs.py -v
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ARM_DIR = ROOT / "configs" / "arms"
BASE = ROOT / "configs" / "data" / "JetClassII_base.yaml"
MAPS = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"

# Every top-level section of a weaver data config. `labels` is the ONLY one an
# arm may differ in; the rest must be byte-identical to the base.
SECTIONS = ["selection", "new_variables", "preprocess", "inputs",
            "labels", "observers", "weights"]
CONTROLLED = "labels"


def arm_configs() -> list[pathlib.Path]:
    return sorted(ARM_DIR.glob("*.yaml")) if ARM_DIR.exists() else []


def split_sections(text: str) -> dict[str, str]:
    """Top-level section name -> its raw text, comments and whitespace included."""
    hits = [(m.group(1), m.start()) for m in re.finditer(r"^([a-z_]+):", text, re.M)
            if m.group(1) in SECTIONS]
    out = {}
    for i, (name, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        out[name] = text[start:end]
    return out


@pytest.fixture(scope="module")
def arms():
    paths = arm_configs()
    if len(paths) < 2:
        pytest.skip(f"needs >=2 arm configs, found {len(paths)} - "
                    "run scripts/build_arm_configs.py")
    return {p.stem: p.read_text() for p in paths}


# ------------------------------------------------------------------------ I1
def test_arms_differ_only_in_vocabulary(arms):
    """I1. Every section except `labels` is byte-identical across arms."""
    per_section = {s: {a: split_sections(t).get(s) for a, t in arms.items()}
                   for s in SECTIONS}
    offenders = []
    for section, by_arm in per_section.items():
        if section == CONTROLLED:
            continue
        present = {a: v for a, v in by_arm.items() if v is not None}
        if len(set(present.values())) > 1:
            digests = {a: hashlib.sha256(v.encode()).hexdigest()[:12]
                       for a, v in present.items()}
            offenders.append(f"{section}: {digests}")
    assert not offenders, (
        "I1 VIOLATED - arms differ outside the label vocabulary:\n  " +
        "\n  ".join(offenders))


def test_labels_block_actually_does_differ(arms):
    """The converse of I1: if the labels blocks were identical the arms would
    not be different arms at all, and a green I1 would mean nothing."""
    blocks = {a: split_sections(t).get(CONTROLLED) for a, t in arms.items()}
    assert len(set(blocks.values())) == len(blocks), (
        "two or more arms share a labels block - they are the same arm")


def test_every_arm_matches_the_base_outside_labels(arms):
    """Arms must track the pinned base, not just each other.

    Three arms could agree with one another and all disagree with the base --
    e.g. if someone regenerated from an edited base. That is still a stream
    change, and I1-across-arms alone would not catch it.
    """
    if not BASE.exists():
        pytest.skip("pinned base config absent")
    base = split_sections(BASE.read_text())
    bad = []
    for arm, text in arms.items():
        got = split_sections(text)
        for section in SECTIONS:
            if section == CONTROLLED or section not in base:
                continue
            if got.get(section) != base[section]:
                bad.append(f"{arm}.{section}")
    assert not bad, "arms drifted from the pinned base in: " + ", ".join(bad)


# ------------------------------------------------------------------------ I2
def weights_sha(text: str) -> str | None:
    m = re.search(r"^weights:", text, re.M)
    return hashlib.sha256(text[m.start():].encode()).hexdigest() if m else None


def test_weights_block_is_byte_identical_and_matches_base(arms):
    """I2 again, from the arm side rather than the tree side.

    This is the invariant the whole study rests on: JetClass-II reweighting
    keys on NATIVE labels, so regrouping the vocabulary cannot move a jet
    between sampling strata -- PROVIDED this block is untouched.
    """
    digests = {a: weights_sha(t) for a, t in arms.items()}
    assert all(digests.values()), f"arm with no weights: block: {digests}"
    assert len(set(digests.values())) == 1, f"I2 VIOLATED: {digests}"
    if BASE.exists():
        assert set(digests.values()) == {weights_sha(BASE.read_text())}, (
            "arm weights blocks agree with each other but NOT with the base")


def test_class_weights_still_sum_to_the_ground_truth_value(arms):
    """Exact rational check on the reweighting weights: 30 categories, 1.73175.

    A single edited digit inside the weights block would change the stream
    while leaving the config structurally valid.
    """
    from fractions import Fraction
    for arm, text in arms.items():
        raw = re.search(r"class_weights:\s*\[(.*?)\]", text, re.S)
        assert raw, f"{arm}: no class_weights list"
        vals = [Fraction(x.strip()) for x in raw.group(1).replace("\n", " ").split(",")
                if x.strip()]
        assert len(vals) == 30, f"{arm}: {len(vals)} class weights, expected 30"
        assert sum(vals) == Fraction("1.73175"), (
            f"{arm}: class_weights sum to {sum(vals)}, expected 1.73175")


# ------------------------------------------------------------------------ I5
def test_architecture_is_not_pinned_inside_the_data_config(arms):
    """I5, as it applies here.

    Architecture reaches weaver through --network-config and `-o` options, not
    through the data config. An arm config that names a network or sets a
    width has smuggled an architecture difference into a vocabulary change,
    making the two inseparable.
    """
    forbidden = ["network_config", "num_layers", "embed_dim", "fc_params",
                 "num_heads", "block_params"]
    bad = [f"{arm}:{k}" for arm, text in arms.items() for k in forbidden
           if re.search(rf"^\s*{k}\s*:", text, re.M)]
    assert not bad, (
        "architecture pinned inside a data config: " + ", ".join(bad))


def test_num_classes_is_not_set_in_the_data_config(arms):
    """num_classes arrives via weaver's `-o num_classes K`.

    If it were also set here the two could disagree, and weaver would take one
    of them silently.
    """
    bad = [arm for arm, text in arms.items()
           if re.search(r"^\s*num_classes\s*:", text, re.M)]
    assert not bad, f"num_classes set inside data config for: {bad}"


# ------------------------------------------------------------------------ I6
def test_no_label_rederivation_from_auxiliary_variables(arms):
    """I6. jet_label is read, never reconstructed.

    aux_genpart_isQcdParton uses a 10 GeV parton threshold
    (FatJetMatching.h:591). Re-deriving would reconstruct a DIFFERENT
    partition, and the discrepancy is structurally undetectable from the
    released data -- there is no way to notice after the fact.
    """
    bad = []
    for arm, text in arms.items():
        body = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        for m in re.finditer(r"aux_genpart_\w+", body):
            bad.append(f"{arm}: {m.group(0)}")
    assert not bad, (
        "I6 VIOLATED - auxiliary generator variables used in an arm config: " +
        ", ".join(bad))


def test_truth_label_is_a_pure_function_of_jet_label(arms):
    """I6, positively stated: the label expression may read jet_label ONLY."""
    for arm, text in arms.items():
        expr = re.search(r"truth_label:\s*(.*)", text)
        assert expr, f"{arm}: no truth_label"
        idents = set(re.findall(r"[A-Za-z_]\w*", expr.group(1)))
        assert idents <= {"jet_label"}, (
            f"{arm}: truth_label reads {sorted(idents - {'jet_label'})}, "
            "must be a pure function of jet_label")


# --------------------------------------------------------------- correctness
def test_each_arm_expression_reproduces_the_committed_label_map(arms):
    """The check that actually matters: evaluate the emitted expression.

    However the boolean is written -- ranges, equalities, whatever collapsing
    the generator chose -- it must reproduce rung_label_maps.v1.csv exactly
    over jet_label = 0..187.
    """
    import csv
    import numpy as np
    rows = list(csv.DictReader(MAPS.open()))
    jet_label = np.arange(188)
    for arm, text in arms.items():
        # a mass twin (<ARM>_MASS) carries its arm's map under its arm's column
        col = arm[:-len("_MASS")] if arm.endswith("_MASS") else arm
        assert col in rows[0], f"no column {col!r} in the committed label map"
        expected = {int(r["jet_label"]): int(r[col]) for r in rows}
        expr = re.search(r"truth_label:\s*(.*)", text).group(1)
        got = eval(expr, {"__builtins__": {}}, {"jet_label": jet_label})
        got = {int(n): int(v) for n, v in enumerate(np.asarray(got))}
        mismatched = [n for n in expected if got[n] != expected[n]]
        assert not mismatched, (
            f"{arm}: expression disagrees with the label map on "
            f"{len(mismatched)} labels, first={mismatched[:5]}")
        assert set(got.values()) == set(range(len(set(expected.values())))), (
            f"{arm}: group ids are not dense from zero")


def test_arm_group_counts_are_the_expected_ones(arms):
    import numpy as np
    expected = {"L162": 162, "R42_Q1": 43, "R16_Q1": 17}
    for arm, text in arms.items():
        if arm not in expected:
            continue
        expr = re.search(r"truth_label:\s*(.*)", text).group(1)
        got = eval(expr, {"__builtins__": {}}, {"jet_label": np.arange(188)})
        n = len(set(int(v) for v in np.asarray(got)))
        assert n == expected[arm], f"{arm}: {n} groups, expected {expected[arm]}"


def test_documented_num_classes_matches_the_derived_count(arms):
    """The `-o num_classes K` line in the header is what a human copies into
    the launch command. If it disagrees with the map, the run is wrong."""
    import numpy as np
    for arm, text in arms.items():
        m = re.search(r"-o num_classes (\d+)", text)
        assert m, f"{arm}: header does not state the num_classes to launch with"
        expr = re.search(r"truth_label:\s*(.*)", text).group(1)
        got = eval(expr, {"__builtins__": {}}, {"jet_label": np.arange(188)})
        assert int(m.group(1)) == len(set(int(v) for v in np.asarray(got))), (
            f"{arm}: header says num_classes={m.group(1)}, map says otherwise")
