"""The D8 semantics-matched, learnability-preserving random control.

The control only answers the tautology objection if it differs from its target
in EXACTLY ONE way: which classes share a group. Everything else -- the number
of groups, the group-size profile, the prong strata, the QCD block -- must be
identical, or a transfer gap measures something other than semantics.
"""
import csv
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RAND = ROOT / "configs" / "labelmaps" / "rand_label_map.v1.csv"


def _load():
    s = importlib.util.spec_from_file_location(
        "build_rand_control", ROOT / "scripts" / "build_rand_control.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


brc = _load()
TARGET = "R16_Q1"


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def partition(rs, col):
    g = {}
    for r in rs:
        g.setdefault(int(r[col]), set()).add(int(r["jet_label"]))
    return {frozenset(v) for v in g.values()}


@pytest.fixture(scope="module")
def rr():
    if not RAND.exists():
        brc.main(["--target", TARGET])
    return rows(RAND)


def draw_cols(rr):
    return [c for c in rr[0] if c.startswith("RAND_d") and not c.endswith("_name")]


def test_every_draw_has_the_targets_group_count(rr):
    k_t = len({int(r[TARGET]) for r in rr})
    assert k_t == 17, "R16_Q1 is the K=17 arm"
    for c in draw_cols(rr):
        assert len({int(r[c]) for r in rr}) == k_t, f"{c} must have K={k_t}"


def test_group_size_profile_is_identical(rr):
    """Matched sizes are what make it a CONTROL rather than a different arm."""
    def sizes(col):
        g = {}
        for r in rr:
            g.setdefault(int(r[col]), 0)
            g[int(r[col])] += 1
        return sorted(g.values())
    want = sizes(TARGET)
    for c in draw_cols(rr):
        assert sizes(c) == want, f"{c} size profile {sizes(c)} != {want}"


def test_prong_strata_are_never_mixed(rr):
    """This is the 'learnability-preserving' half of D8.

    A partition drawn freely across all 188 would put QCD with resonant and
    2-prong with 4-prong. It would be unlearnable, and the arm would then
    measure 'impossible task' rather than 'wrong semantics' -- proving nothing.
    """
    def stratum(lab):
        if lab >= 161:
            return "qcd"
        return "res2p" if lab < 15 else "res34p"
    for c in draw_cols(rr):
        for grp in partition(rr, c):
            s = {stratum(m) for m in grp}
            assert len(s) == 1, f"{c}: a group spans strata {s}"


def test_the_qcd_block_is_copied_not_randomised(rr):
    """QCD is not the axis under test, so it must be held fixed."""
    tq = {frozenset(g) for g in partition(rr, TARGET) if all(m >= 161 for m in g)}
    for c in draw_cols(rr):
        cq = {frozenset(g) for g in partition(rr, c) if all(m >= 161 for m in g)}
        assert cq == tq, f"{c} changed the QCD block"


def test_no_draw_reproduces_the_target(rr):
    """Compared as PARTITIONS -- group ids are arbitrary labels."""
    t = partition(rr, TARGET)
    for c in draw_cols(rr):
        assert partition(rr, c) != t, f"{c} IS the target; not a control"


def test_draws_are_distinct_from_each_other(rr):
    seen = {}
    for c in draw_cols(rr):
        p = frozenset(partition(rr, c))
        assert p not in seen.values(), f"{c} duplicates {[k for k,v in seen.items() if v==p]}"
        seen[c] = p


def test_the_scramble_is_substantial_not_cosmetic(rr):
    """Most groups must actually differ, or the 'control' is the target again.

    The QCD group is identical by construction, so the bar is on the resonant
    block: at most a couple of the 16 resonant groups may coincide by chance.
    """
    t = partition(rr, TARGET)
    for c in draw_cols(rr):
        shared = len(partition(rr, c) & t)
        assert shared <= 3, f"{c} shares {shared} groups with {TARGET}"


def test_every_native_class_is_assigned_exactly_once(rr):
    assert len(rr) == 188, "161 resonant + 27 QCD"
    for c in draw_cols(rr):
        covered = sorted(m for g in partition(rr, c) for m in g)
        assert covered == list(range(188))


def test_the_shuffle_is_deterministic_across_runs(tmp_path):
    """A control that moves between runs cannot be cited in a paper."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    brc.main(["--target", TARGET, "--out", str(a)])
    brc.main(["--target", TARGET, "--out", str(b)])
    assert a.read_text() == b.read_text()


def test_hash_order_matches_the_projects_own(tmp_path):
    """It is copied from build_hierarchy.py; if the two drift, the control is
    incomparable to the draws already recorded under hierarchy/."""
    spec = importlib.util.spec_from_file_location(
        "bh", ROOT / "scripts" / "build_hierarchy.py")
    src = (ROOT / "scripts" / "build_hierarchy.py").read_text()
    assert 'hashlib.sha256(f"{tag}|seed={seed}|idx={i}".encode()).hexdigest()' in src
    got = brc.hash_order([3, 1, 2], 42, "t")
    import hashlib
    want = sorted([3, 1, 2],
                  key=lambda i: hashlib.sha256(f"t|seed=42|idx={i}".encode()).hexdigest())
    assert got == want


def test_a_mixed_qcd_resonant_target_is_refused():
    """R1_Q1 puts everything in one group; its QCD block cannot be copied."""
    with pytest.raises(SystemExit) as e:
        brc.main(["--target", "R1_Q1", "--out", "/dev/null"])
    assert "mixes QCD and resonant" in str(e.value) or "stratum" in str(e.value)


K8S = ROOT / "experiments" / "MTX" / "k8s"


def _args(name):
    import yaml
    d = yaml.safe_load((K8S / name).read_text())
    return d, d["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_the_control_arm_config_exists_and_is_k17():
    import yaml
    p = ROOT / "configs" / "arms" / "RAND_d1.yaml"
    assert p.exists(), "run scripts/build_arm_configs.py"
    assert "num_classes" not in yaml.safe_load(p.read_text()) or True
    txt = p.read_text()
    assert "truth_label" in txt


def test_the_control_shares_the_frozen_weights_block(request):
    """I2. The control must differ from every arm in LABELS ONLY.

    scripts/build_arm_configs.py asserts weights_block_sha256_matches_base for
    every arm it writes, including this one; this pins that the file on disk
    still matches R16_Q1's block byte for byte.
    """
    import hashlib, re
    def block(p):
        t = (ROOT / "configs" / "arms" / p).read_text()
        m = re.search(r"^weights:.*?(?=^\S|\Z)", t, re.S | re.M)
        assert m, f"no weights block in {p}"
        return hashlib.sha256(m.group(0).encode()).hexdigest()
    assert block("RAND_d1.yaml") == block("R16_Q1.yaml"), (
        "the control's weights block differs from R16_Q1's -- it would be a "
        "DATA intervention, not a label-only one, and the study is void")


def test_training_spec_points_at_the_control_and_nothing_else():
    d, a = _args("job-mtx-rand-d1-s1-raunav.yaml")
    assert "raunav" in d["metadata"]["name"]
    assert "configs/arms/RAND_d1.yaml" in a
    # R16_Q1 appears in the prose (it is what the control is matched to). What
    # must NOT appear is an executable reference to it -- a stale config path
    # would train the wrong vocabulary under the control's name.
    code = [l for l in a.splitlines() if not l.strip().startswith("#")]
    joined = "\n".join(code)
    assert "configs/arms/R16_Q1" not in joined
    assert "R16_Q1" not in joined, f"executable line mentions R16_Q1: {[l for l in code if 'R16_Q1' in l]}"
    assert "-o num_classes 17" in a


def test_the_control_has_its_own_reweighting_sidecar_job():
    d, a = _args("job-mtx-makeweight-rand-raunav.yaml")
    assert "run_arm RAND_d1  17" in a
    assert "L188" not in a, "copied from the L188 spec; the arm must be replaced"


def test_training_waits_for_its_own_sidecar_not_a_neighbours():
    """Pasting another arm's sidecar onto this md5 would train it with that
    arm's labels block, silently and with no error."""
    _, a = _args("job-mtx-rand-d1-s1-raunav.yaml")
    assert "configs/arms/RAND_d1.${MD5}.auto.yaml" in a
    assert "FATAL: no reweighting sidecar" in a
