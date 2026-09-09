"""The use-case survival table -- docs/PRD_PLAN.md 3.1(c).

The table's whole claim is that Sophon's Property 1 makes it EXACT. These tests
hold it to that: the criterion, the arithmetic, and the two numbers most likely
to be "corrected" by someone who has not read why they differ.
"""
import csv
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
SPEC = REPO / "configs" / "labelmaps" / "usecase_discriminants.v1.csv"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_usecase_survival", REPO / "scripts" / "build_usecase_survival.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows():
    with MAP.open() as f:
        return list(csv.DictReader(f))


def test_every_class_the_spec_names_exists_in_the_map():
    """A typo in a class name must fail loudly, not silently drop a node.

    Dropping a node from a denominator changes the discriminant into a different
    one and the table would still look plausible.
    """
    m = _mod()
    by_name, _ = m.read_map()
    m.read_spec(by_name)          # resolve() raises SystemExit on an unknown name
    with SPEC.open() as f:
        named = set()
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if r["classes"] != "@QCD":
                named.update(r["classes"].split(";"))
    assert named, "the spec names no classes at all"
    assert named <= set(by_name), sorted(named - set(by_name))


def test_the_criterion_is_group_constancy_not_set_membership():
    """Property 1 permits exactly the coefficient vectors constant on groups."""
    m = _mod()
    groups = {"g0": [0, 1], "g1": [2]}
    assert m.group_constant({0: 1.0, 1: 1.0, 2: 0.0}, groups)
    # 0 and 1 share a group but carry different weights -- unbuildable.
    assert not m.group_constant({0: 1.0, 1: 0.0, 2: 0.0}, groups)
    # A class absent from the vector is coefficient 0, not "ignore me".
    assert not m.group_constant({0: 1.0, 2: 0.0}, groups)


def test_survival_is_monotone_because_the_ladder_is_nested():
    """Once a discriminant dies it cannot come back.

    Each rung is a coarsening of the one above, so the set of group-constant
    vectors only shrinks. A non-monotone row means either a bug here or a
    non-nested rung map -- both are serious.
    """
    m = _mod()
    by_name, groups = m.read_map()
    specs, _, _ = m.read_spec(by_name)
    for d, vectors in specs.items():
        alive = [all(m.group_constant(v, groups[r]) for v in vectors.values())
                 for r in m.RUNGS]
        first_dead = next((i for i, a in enumerate(alive) if not a), len(alive))
        assert not any(alive[first_dead:]), f"{d} comes back to life: {alive}"


def test_per_qcd_subclass_work_is_possible_only_at_the_finest_rung():
    """L162 IS the QCD merge, so GN3X-style rejection dies at the first step.

    This is the concrete cost of the very first contraction, and it is the
    reason L188 is in the run matrix at all.
    """
    m = _mod()
    by_name, groups = m.read_map()
    specs, _, _ = m.read_spec(by_name)
    v = specs["gn3x_qcd_subclass"]
    alive = [r for r in m.RUNGS
             if all(m.group_constant(x, groups[r]) for x in v.values())]
    assert alive == ["L188"], alive

    qcd = {r["L162_name"] for r in _rows() if r["class_name"].startswith("label_QCD_")}
    assert len(qcd) == 1, f"L162 should hold all 27 QCD classes in one group, got {qcd}"


def test_d_bc_dies_one_rung_before_the_trainable_probe_collapses():
    """The documented divergence from probe.py. Neither number is wrong.

    D_bc's denominator names label_X_cs and not label_X_cq; R63_Q1 merges them,
    so the published RATIO cannot be built there. A trainable bc-vs-background
    PROBE is unaffected -- it just never sees label_X_cq -- and survives to
    R42_Q1, where label_X_bc finally joins label_X_bq.
    """
    m = _mod()
    by_name, groups = m.read_map()
    specs, _, _ = m.read_spec(by_name)
    alive = [r for r in m.RUNGS
             if all(m.group_constant(v, groups[r]) for v in specs["vcb_eq1"].values())]
    assert alive == ["L188", "L162"], alive

    sys.path.insert(0, str(REPO / "experiments" / "EVAL"))
    try:
        import probe
    finally:
        sys.path.pop(0)
    t = probe.TASKS["bc_vs_rest"]
    collapsed = probe.derive_collapsed_at(t["signal"], t["background"])
    assert collapsed[0] == "R42_Q1", collapsed

    # And the mechanism, so a future edit cannot keep the numbers while losing
    # the reason for them.
    at63 = {r["class_name"] for r in _rows()
            if r["R63_Q1_name"] == "2P_HAD_2PARTON|nb0_nc1"}
    assert at63 == {"label_X_cs", "label_X_cq"}, at63


def test_the_pigamma_row_is_deferred_rather_than_guessed():
    """arXiv:2606.09458 defines P1 over a 206-node head, not our 188.

    A guessed mapping would put a wrong row in a table whose value is exactness.
    If someone adds the row, they must also delete this test -- which is the
    point: it forces them to have the node table in hand.
    """
    m = _mod()
    by_name, _ = m.read_map()
    specs, _, _ = m.read_spec(by_name)
    assert "pigamma_p1" not in specs
    assert "DEFERRED ROW" in SPEC.read_text()
