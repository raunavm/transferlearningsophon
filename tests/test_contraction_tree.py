"""CI tests for the locked contraction tree (invariants I1-I3).

The nine tests named in the locked plan. Tests 1-6 are structural and run
against the derived tree. Tests 7-8 guard the sampling stream and only bind
once arm configs exist - they FAIL if arm configs exist and disagree, and SKIP
(loudly) if none exist yet. Test 9 guards against seed-regeneration of label
arrays.

Run:  python3 -m pytest tests/test_contraction_tree.py -v
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_contraction_tree import (  # noqa: E402
    EXPECTED_NATIVE, EXPECTED_R16_SIZES, EXPECTED_RUNG_GROUPS, LADDER,
    build, build_rungs, check_chain_nesting, effective_shares, hf_counts,
    load_master, run_checks,
)

TREE_YAML = ROOT / "configs" / "labelmaps" / "contraction_tree.v1.yaml"
AUDIT_CSV = ROOT / "configs" / "labelmaps" / "stream_share_audit.v1.csv"
MAPS_CSV = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
REVIEW_CSVS = ROOT / "review" / "groupings" / "JetClass2_all_8_grouping_CSVs"


@pytest.fixture(scope="module")
def built():
    return build(load_master())


@pytest.fixture(scope="module")
def checks(built):
    return {name: (ok, detail) for name, ok, detail in run_checks(built)}


# ------------------------------------------------------- tests 1-6, structural
@pytest.mark.parametrize("check", [
    "every_native_label_exactly_once",
    "no_empty_groups",
    "consecutive_ids_from_zero",
    "qcd_always_QCD_ALL_below_L188",
    "compose_native_R42_R16_equals_native_R16",
    "R42_strictly_refines_R16",
])
def test_structural_check(checks, check):
    ok, detail = checks[check]
    assert ok, f"{check} failed: {detail}"


def test_group_counts_match_published(checks):
    """The derivation must reproduce independently published counts."""
    for name in ("r16_resonant_group_count", "r42_resonant_group_count",
                 "r16_group_size_vector_matches_published"):
        ok, detail = checks[name]
        assert ok, f"{name} failed: {detail}"


def test_r16_groups_are_unions_of_whole_reweight_groups(checks):
    """The property that makes R16 effective shares EXACT rather than bounded.

    If this ever fails, every effective_share_exact value in the audit CSV
    becomes an estimate and must be re-labelled.
    """
    ok, detail = checks["r16_groups_are_unions_of_whole_reweight_groups"]
    assert ok, f"R16 exactness property broken: {detail}"


def test_effective_shares_sum_to_exactly_one(built):
    """Exact rational arithmetic - not float-close, exactly 1."""
    assert sum(effective_shares(built).values()) == 1


def test_effective_shares_match_published_audit(built):
    """Regression lock on the published four-number audit."""
    published = {
        0: "0.2598528", 1: "0.0086618", 2: "0.0057745", 3: "0.0461960",
        4: "0.0355132", 5: "0.0017324", 6: "0.0072181", 7: "0.0086618",
        8: "0.0012993", 9: "0.2237621", 10: "0.0288725", 11: "0.1154901",
        12: "0.0288725", 13: "0.0461960", 14: "0.0086618", 15: "0.0288725",
        16: "0.1443626",
    }
    shares = effective_shares(built)
    for gid, exp in published.items():
        assert abs(float(shares[gid]) - float(exp)) < 5e-7, (
            f"group {gid}: computed {float(shares[gid]):.7f} != published {exp}")


# ----------------------------------------------- test 7, weights-block sha256
def _weights_block_sha256(path: pathlib.Path) -> str | None:
    """sha256 of the `weights:` block, from `weights:` to EOF."""
    text = path.read_text()
    m = re.search(r"^weights:", text, re.M)
    if not m:
        return None
    return hashlib.sha256(text[m.start():].encode()).hexdigest()


def _arm_configs() -> list[pathlib.Path]:
    return sorted((ROOT / "configs" / "arms").glob("*.yaml"))


def test_weights_block_sha256_identical_across_arms():
    """Invariant I2, mechanised.

    Regrouping the classifier vocabulary must leave the sampling stream
    bit-identical, which holds only if the `weights:` block is byte-identical
    in every arm config.
    """
    arms = _arm_configs()
    if len(arms) < 2:
        pytest.skip(f"needs >=2 arm configs, found {len(arms)} - "
                    "test BINDS as soon as arms are written")
    digests = {p.name: _weights_block_sha256(p) for p in arms}
    missing = [n for n, d in digests.items() if d is None]
    assert not missing, f"arm configs with no weights: block: {missing}"
    assert len(set(digests.values())) == 1, (
        "I2 VIOLATED - weights: block differs across arms:\n" +
        "\n".join(f"  {n}: {d[:16]}" for n, d in sorted(digests.items())))


# --------------------------------------------- test 8, stream-identity hash
def test_first_1e6_streamed_jet_ids_hash_identical_across_arms():
    """Invariant I3: the realised stream, not just the config, is identical."""
    hashes = sorted((ROOT / "artifacts" / "stream_hashes").glob("*.sha256")) \
        if (ROOT / "artifacts" / "stream_hashes").exists() else []
    if len(hashes) < 2:
        pytest.skip(f"needs >=2 recorded stream hashes, found {len(hashes)} - "
                    "produced by the G0 smoke run; test BINDS once they exist")
    vals = {p.name: p.read_text().strip().split()[0] for p in hashes}
    assert len(set(vals.values())) == 1, (
        "I3 VIOLATED - streamed jet-id hash differs across arms:\n" +
        "\n".join(f"  {n}: {v[:16]}" for n, v in sorted(vals.items())))


# ------------------------------------ test 9, materialized not seed-regenerated
def test_label_arrays_materialized_not_seed_regenerated():
    """Label maps must be frozen artifacts, never regenerated from a seed.

    A seed-regenerated map silently changes if the RNG, Python version or
    platform changes. The committed tree is the source of truth.
    """
    assert TREE_YAML.exists(), "contraction tree artifact missing"
    text = TREE_YAML.read_text()
    for forbidden in ("random.shuffle", "np.random", "random.seed", "rng.permutation"):
        assert forbidden not in text, (
            f"tree artifact references {forbidden} - label arrays must be "
            "materialized, never seed-regenerated")
    assert "natives:" in text, "tree does not expand native labels explicitly"
    n_native = len(re.findall(r"^\s+\d+: label_", text, re.M))
    assert n_native == EXPECTED_NATIVE, (
        f"tree expands {n_native} native labels, expected {EXPECTED_NATIVE}")


def test_audit_csv_exists_and_is_complete(built):
    import csv
    assert AUDIT_CSV.exists(), "stream_share_audit.v1.csv missing"
    rows = list(csv.DictReader(AUDIT_CSV.open()))
    for tag, expected in EXPECTED_RUNG_GROUPS.items():
        block = [r for r in rows if r["rung"] == tag]
        assert len(block) == expected, (
            f"expected {expected} {tag} rows, got {len(block)}")
        assert sum(int(r["n_native"]) for r in block) == EXPECTED_NATIVE, (
            f"{tag} rows do not account for all 188 natives")


def test_tree_yaml_actually_parses_as_yaml(built):
    """The frozen artifact must be machine-readable. It once was not.

    A blind audit found contraction_tree.v1.yaml unparseable -- 9 scanner
    errors, all in the free-text `rules:` block, where prose containing ': '
    and stray quotes was emitted as plain scalars. Every structural test
    passed anyway, because they all read the file with read_text() and never
    parsed it. A frozen reference document that no parser can load is not a
    reference document.
    """
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(TREE_YAML.read_text())
    assert isinstance(doc, dict), "tree artifact is not a YAML mapping"
    for key in ("schema_version", "provenance", "rules", "contraction_order",
                "rungs", "tree"):
        assert key in doc, f"tree artifact lost its {key!r} block"
    assert doc["contraction_order"] == LADDER
    assert set(doc["rungs"]) == set(LADDER)
    for tag in LADDER:
        assert doc["rungs"][tag]["n"] == EXPECTED_RUNG_GROUPS[tag], (
            f"{tag}: YAML claims n={doc['rungs'][tag]['n']}, "
            f"derivation gives {EXPECTED_RUNG_GROUPS[tag]}")


def test_tree_yaml_documents_the_tie_break_that_fixes_the_ids(built):
    """The ordering rule must be complete in the artifact, not just the code.

    The primary sort keys (prong class, then decay mode) leave ties, and the
    tie-break -- first native-label appearance -- is what actually determines
    the ids. An independent auditor given only the primary keys derived the
    same partition with different ids for 47 of 188 labels. The sign-off
    document has to state the rule that was actually used.
    """
    yaml = pytest.importorskip("yaml")
    order = yaml.safe_load(TREE_YAML.read_text())["rules"]["group_id_order"]
    low = order.lower()
    assert "tie-break" in low, "group_id_order does not mention the tie-break"
    assert "first native label appearance" in low, (
        "group_id_order does not name the tie-break actually used")


def test_audit_csv_never_puts_a_nominal_number_in_a_measured_column():
    """A group that cuts inside a reweighting group has NO exact share.

    Emitting a nominal value there is how this repository previously quoted
    upper bounds (up to 389.8% for QCD_ALL) as measured fact.
    """
    import csv
    for r in csv.DictReader(AUDIT_CSV.open()):
        if r["share_provenance"].startswith("MEASURED_REQUIRED"):
            assert r["effective_share_exact"] == "", (
                f"{r['rung']} {r['group_name']} carries a share it cannot know")
        for col in ("raw_stored_train", "selected_train", "unique_sampled_train"):
            assert r[col] == "", f"{col} is a G-0 measurement, not a derivation"


def test_r16_size_vector_is_the_published_one():
    assert sum(EXPECTED_R16_SIZES) == 161, "resonant sizes must total 161"
    assert len(EXPECTED_R16_SIZES) == 16


# ------------------------------------------------- the full eight-rung ladder
@pytest.fixture(scope="module")
def rungs(built):
    return build_rungs(built)


def test_ladder_order_is_frozen(rungs):
    assert list(rungs) == LADDER


@pytest.mark.parametrize("tag", LADDER)
def test_rung_is_wellformed(rungs, tag):
    """188 labels, contiguous ids from zero, bijective id <-> name."""
    mapping = rungs[tag]
    assert len(mapping) == EXPECTED_NATIVE
    ids = {v[0] for v in mapping.values()}
    assert ids == set(range(len(ids))), f"{tag}: ids not contiguous from zero"
    assert len(ids) == EXPECTED_RUNG_GROUPS[tag], (
        f"{tag}: {len(ids)} groups, expected {EXPECTED_RUNG_GROUPS[tag]}")
    fwd, rev = {}, {}
    for gid, nm in mapping.values():
        assert fwd.setdefault(gid, nm) == nm, f"{tag}: id {gid} has two names"
        assert rev.setdefault(nm, gid) == gid, f"{tag}: name {nm!r} has two ids"


def test_every_consecutive_pair_strictly_nests(rungs):
    """The ladder property. Without it no contrast on it is interpretable."""
    failures = [(n, d) for n, ok, d in check_chain_nesting(rungs) if not ok]
    assert not failures, "ladder nesting broken:\n" + "\n".join(
        f"  {n}: {d}" for n, d in failures)


def test_r63_counts_come_from_tokens_not_boolean_columns(built):
    """Regression guard on R63's entire reason for existing.

    has_b / has_c are BOOLEANS: label_X_bb has has_b == 1, not 2. Deriving R63
    from them would silently collapse multiplicity back into presence, leaving a
    rung that looks right, nests correctly, and measures nothing new.
    """
    rows = {r["class_name"]: r for r in built["rows"]}
    bb = rows["label_X_bb"]
    assert hf_counts(bb) == (2, 0), "n_b must count tokens, not read has_b"
    assert int(bb["has_b"]) == 1, "guard assumes has_b is boolean; schema changed"
    bc = rows["label_X_bc"]
    assert hf_counts(bc) == (1, 1)
    # bb and bc share a flavour tier at R42 but must separate at R63
    r = build_rungs(built)
    n_bb, n_bc = int(bb["jet_label"]), int(bc["jet_label"])
    assert r["R42_Q1"][n_bb][0] == r["R42_Q1"][n_bc][0], "bb/bc share the B tier"
    assert r["R63_Q1"][n_bb][0] != r["R63_Q1"][n_bc][0], (
        "R63 must separate 2b from 1b1c -- multiplicity is the axis it adds")


def test_r3_uses_visible_objects_so_a_neutrino_is_not_a_prong(built, rungs):
    """J6, resolved against arXiv:2405.12972 sec. 2.

    The JetClass-II authors: '...can also be 3 prongs if an object leaks out of
    the jet cone or if one of the objects is a neutrino.'
    """
    by_label = {int(r["jet_label"]): r for r in built["rows"]}
    moved = [n for n, r in by_label.items()
             if r["block"] != "qcd"
             and int(r["n_objects"]) != int(r["n_visible_objects"])]
    assert len(moved) == 30, f"expected 30 neutrino natives, got {len(moved)}"
    for n in moved:
        assert int(by_label[n]["n_objects"]) == 4
        assert rungs["R3_VIS"][n][1] == "3P_VIS", (
            f"{by_label[n]['class_name']} has a neutrino and must be 3-prong")


def test_r3_vis_groups_are_all_exact_share(built, rungs):
    """Every R3_VIS group is a union of whole reweighting groups.

    This is what lets the rung's shares be exact algebra rather than a
    uniformity assumption. It is not automatic -- it survives the J6 change.
    """
    rows = built["rows"]
    by_label = {int(r["jet_label"]): r for r in rows}
    rw_members: dict[str, set[int]] = {}
    for r in rows:
        rw_members.setdefault(r["reweight_group_name"], set()).add(int(r["jet_label"]))
    groups: dict[int, set[int]] = {}
    for n, (gid, _) in rungs["R3_VIS"].items():
        groups.setdefault(gid, set()).add(n)
    for gid, members in groups.items():
        for rw in {by_label[n]["reweight_group_name"] for n in members}:
            assert rw_members[rw] <= members, (
                f"R3_VIS group {gid} cuts inside reweighting group {rw}")


def test_rung_label_maps_materializes_every_rung(built, rungs):
    """I4 for the whole ladder, not just the four original rungs."""
    import csv
    assert MAPS_CSV.exists(), "rung_label_maps.v1.csv missing"
    rows = list(csv.DictReader(MAPS_CSV.open()))
    assert len(rows) == EXPECTED_NATIVE
    for tag in LADDER:
        assert tag in rows[0], f"{tag} column missing from the materialized map"
        assert f"{tag}_name" in rows[0]
    for r in rows:
        n = int(r["jet_label"])
        for tag in LADDER:
            gid, gname = rungs[tag][n]
            assert int(r[tag]) == gid and r[f"{tag}_name"] == gname, (
                f"{tag} disagrees with the derivation at label {n}")


@pytest.mark.parametrize("tag,ref", [
    ("L188", "L188"), ("L162", "L162"), ("R63_Q1", "R63_Q1"),
    ("R42_Q1", "R42_Q1"), ("R29_Q1", "R29_Q1"), ("R16_Q1", "R16_Q1"),
    ("R1_Q1", "R1_Q1"),
])
def test_matches_the_independently_supplied_reference_csvs(rungs, built, tag, ref):
    """Cross-check against the reviewer's own eight CSVs.

    These were produced independently of this generator. Agreement on ids AND
    names is a real test rather than a restatement. R3 is deliberately excluded:
    the reference R3_Q1.csv counts generated objects and the ladder now counts
    visible ones -- see test_r3_uses_visible_objects_so_a_neutrino_is_not_a_prong.
    """
    import csv
    path = REVIEW_CSVS / f"{ref}.csv"
    if not path.exists():
        pytest.skip(f"reference CSV {ref}.csv not present")
    label_of = {r["class_name"]: int(r["jet_label"]) for r in built["rows"]}
    reference = {r["Actual Sophon class name"]:
                 (int(r["Group number"]), r["Group class name"])
                 for r in csv.DictReader(path.open())}
    assert len(reference) == EXPECTED_NATIVE
    bad = [(cn, reference[cn], rungs[tag][label_of[cn]]) for cn in reference
           if rungs[tag][label_of[cn]] != reference[cn]]
    assert not bad, (
        f"{tag} disagrees with the reference on {len(bad)} labels; first: {bad[0]}")
