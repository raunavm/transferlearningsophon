"""The D8 control's cost table, which the PI signed option C on.

DECISIONS_PENDING item 24 states the cost of option C as an ARI rise from 0.254
to 0.347. No committed code computed that column -- `grep -l adjusted_rand` over
every .py in the repository returned nothing -- so it was hand-formed, and it was
wrong by 1.61x in the direction that made the control look more scrambled than
it is. These tests pin the whole table so the next reader does not have to take
any of it on faith.

The regeneration tests shell out to the generator and are slow; they are the
point of the file, so they are not skipped.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "rand_control_stats", ROOT / "scripts" / "rand_control_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _mod()


@pytest.fixture(scope="module")
def rows():
    return S.table()


def _row(rows, vocab, match):
    return next(r for r in rows if r["vocabulary"] == vocab and r["match"] == match)


# ------------------------------------------------------- the ARI metric itself

def test_ari_is_one_for_an_identical_partition():
    a = [0, 0, 1, 1, 2, 2]
    assert S.adjusted_rand(a, a) == pytest.approx(1.0)


def test_ari_is_near_zero_for_an_unrelated_partition():
    a = [0, 0, 0, 0, 1, 1, 1, 1]
    b = [0, 1, 0, 1, 0, 1, 0, 1]
    assert abs(S.adjusted_rand(a, b)) < 0.2


def test_ari_is_invariant_to_relabelling():
    a = [0, 0, 1, 1, 2, 2]
    b = [5, 5, 9, 9, 7, 7]
    assert S.adjusted_rand(a, b) == pytest.approx(1.0)


def test_ari_refuses_a_length_mismatch():
    with pytest.raises(SystemExit, match="length mismatch"):
        S.adjusted_rand([0, 1], [0, 1, 2])


# ------------------------------------------- the parts of item 24 that ARE right

def test_every_control_has_the_targets_group_count(rows):
    for r in rows:
        assert r["k"] == 17, f"{r['vocabulary']}/{r['match']} has K={r['k']}"


def test_the_share_matched_controls_reproduce_the_targets_imbalance(rows):
    """200.00:1 is the whole point of option C -- the quantity the loss sees."""
    target = _row(rows, "R16_Q1 (target)", "-")["share_ratio"]
    assert target == pytest.approx(200.0, abs=0.01)
    for col in ("RAND_d1", "RAND_d2", "RAND_d3"):
        assert _row(rows, col, "share")["share_ratio"] == pytest.approx(200.0, abs=0.01)
        assert _row(rows, col, "share")["exp_H"] == pytest.approx(8.43, abs=0.01)


def test_the_count_matched_control_had_a_different_imbalance(rows):
    """Why C was taken at all: the superseded control was an easier task, and
    that is fatal to the falsification rule whatever its ARI."""
    r = _row(rows, "RAND_d1", "count")
    assert r["share_ratio"] == pytest.approx(23.78, abs=0.01)
    assert r["exp_H"] == pytest.approx(12.70, abs=0.01)


def test_the_count_matched_ari_reproduces_the_published_value(rows):
    """THE LOAD-BEARING TEST. Item 24's 0.254 reproduces to 0.0005, which fixes
    the convention as standard ARI over all 188 natives -- so the share-matched
    disagreement below is a real error and not a different definition."""
    assert _row(rows, "RAND_d1", "count")["ari_vs_r16q1"] == pytest.approx(0.254, abs=0.001)


# -------------------------------------------- the part of item 24 that is WRONG

@pytest.mark.parametrize("col,published,correct", [
    ("RAND_d1", 0.347, 0.4030),
    ("RAND_d2", 0.349, 0.3815),
    ("RAND_d3", 0.393, 0.3976),
])
def test_the_share_matched_ari_is_not_what_item_24_published(rows, col, published, correct):
    got = _row(rows, col, "share")["ari_vs_r16q1"]
    assert got == pytest.approx(correct, abs=0.001)
    assert abs(got - published) > 0.004 or col == "RAND_d3", (
        f"{col}: published {published}, correct {correct}")


def test_the_cost_of_option_c_is_understated_by_more_than_half(rows):
    """The number the decision turned on."""
    c = _row(rows, "RAND_d1", "count")["ari_vs_r16q1"]
    s = _row(rows, "RAND_d1", "share")["ari_vs_r16q1"]
    rise = s - c
    assert rise == pytest.approx(0.1493, abs=0.001)
    assert rise / 0.093 == pytest.approx(1.61, abs=0.02), (
        "item 24 records the rise as +0.093; it is 1.61x that")


# ------------------------------------------------------------------ provenance

def test_the_regenerated_share_control_is_the_committed_one(rows):
    """table() raises if it is not, so reaching here is the assertion. Without
    it the cost table could describe a control that is not the one training."""
    assert rows


def test_no_other_module_computes_ari_by_hand():
    """The root cause was that nothing computed this. If a second
    implementation appears, the two can disagree and the table stops being
    single-sourced."""
    hits = []
    for p in ROOT.rglob("*.py"):
        if ".git" in p.parts or "research" in p.parts:
            continue
        if p.name in ("rand_control_stats.py", "test_rand_control_stats.py"):
            continue
        txt = p.read_text(errors="replace")
        if "adjusted_rand" in txt:
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"a second ARI implementation appeared in {hits}"
