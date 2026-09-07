"""MPM masking, targets and rank tokens -- the parts that can be wrong silently.

These run without weaver (the Decoder and MPMNet need weaver's Block, which only
exists in the training image; experiments/MTX/k8s/job-mtx-mpm-smoke-raunav.yaml
exercises those there). What is tested here is everything that could produce a
plausible loss curve while training on the wrong thing:

  * masking a padded slot, which would make the model reconstruct padding;
  * masking every particle, which leaves the encoder attending over an empty set;
  * the type+charge -> 8-class map, where an off-by-one silently relabels muons;
  * the p_T rank, which must be computed among the DROPPED particles only -- if
    it leaked the rank within the whole jet the decoder would get the positional
    information MPMv2 explicitly says trivializes the task.
"""
import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_MPM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments", "MTX", "mpm.py")
_spec = importlib.util.spec_from_file_location("_mpm_under_test", _MPM)
mpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpm)

VARS = ["part_pt_scale_log", "part_e_scale_log", "part_logptrel", "part_logerel",
        "part_deltaR", "part_charge", "part_isChargedHadron", "part_isNeutralHadron",
        "part_isPhoton", "part_isElectron", "part_isMuon", "part_d0", "part_d0err",
        "part_dz", "part_dzerr", "part_deta", "part_dphi"]


def test_feature_groups_splits_the_arms_17_features():
    cont, typ, chg, pt = mpm.feature_groups(VARS)
    assert len(cont) == 11 and len(typ) == 5
    assert chg == VARS.index("part_charge")
    assert pt == VARS.index("part_pt_scale_log")
    # no feature is both continuous and categorical, and none is lost
    assert set(cont) | set(typ) | {chg} == set(range(len(VARS)))
    assert not set(cont) & (set(typ) | {chg})


def test_feature_groups_refuses_a_config_without_the_id_features():
    with pytest.raises(RuntimeError, match="missing"):
        mpm.feature_groups([v for v in VARS if v != "part_isMuon"])


def _jet(kind, charge):
    """One particle's 17 features, in the arm config's order."""
    x = [0.0] * len(VARS)
    x[VARS.index("part_charge")] = charge
    x[VARS.index(kind)] = 1.0
    return x


@pytest.mark.parametrize("kind,charge,expect", [
    ("part_isChargedHadron", +1, 0), ("part_isChargedHadron", -1, 1),
    ("part_isNeutralHadron", 0, 2), ("part_isPhoton", 0, 3),
    ("part_isElectron", +1, 4), ("part_isElectron", -1, 5),
    ("part_isMuon", +1, 6), ("part_isMuon", -1, 7),
])
def test_id_target_maps_all_eight_type_charge_classes(kind, charge, expect):
    x = torch.tensor([_jet(kind, charge)]).unsqueeze(-1)      # (1, C, 1)
    _, typ, chg, _ = mpm.feature_groups(VARS)
    assert mpm.id_target(x, typ, chg).item() == expect


def test_id_target_covers_exactly_the_declared_class_list():
    assert len(mpm.ID_CLASSES) == 8
    assert sorted({c for pair in mpm._TYPE_TO_CLASS for c in pair}) == list(range(8))


def _mask(counts, P):
    m = torch.zeros(len(counts), 1, P)
    for i, n in enumerate(counts):
        m[i, 0, :n] = 1
    return m


def test_draw_mask_never_drops_a_padded_slot():
    P, counts = 32, [30, 17, 3, 1]
    mask = _mask(counts, P)
    for _ in range(50):
        drop = mpm.draw_mask(mask, 0.4)
        assert not (drop & ~mask.squeeze(1).bool()).any(), "dropped a padded slot"


def test_draw_mask_always_leaves_at_least_one_particle_for_the_encoder():
    P = 16
    mask = _mask([1, 2, 5, 16], P)
    for rate in (0.4, 0.9, 1.0):
        for _ in range(30):
            drop = mpm.draw_mask(mask, rate)
            kept = (mask.squeeze(1).bool() & ~drop).sum(1)
            assert (kept >= 1).all(), f"rate {rate} left an empty set"


def test_draw_mask_drops_the_requested_fraction():
    P, counts = 128, [100, 50, 10]
    mask = _mask(counts, P)
    drop = mpm.draw_mask(mask, 0.4)
    assert drop.sum(1).tolist() == [40, 20, 4]


def test_draw_mask_is_not_deterministic_across_calls():
    mask = _mask([64], 64)
    a = mpm.draw_mask(mask, 0.4)
    assert any(not torch.equal(a, mpm.draw_mask(mask, 0.4)) for _ in range(10))


def test_drop_pt_rank_ranks_within_the_dropped_subset_only():
    # p_T descending across the jet; drop positions 1, 3, 4 only.
    pt = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    drop = torch.tensor([[False, True, False, True, True]])
    rank = mpm.drop_pt_rank(pt, drop, max_drop=8)
    # among {8.0, 6.0, 5.0} the ranks are 0, 1, 2 -- NOT the whole-jet ranks 1, 3, 4
    assert rank[0, 1].item() == 0
    assert rank[0, 3].item() == 1
    assert rank[0, 4].item() == 2


def test_drop_pt_rank_is_contiguous_from_zero():
    torch.manual_seed(0)
    pt = torch.randn(4, 40)
    drop = torch.rand(4, 40) < 0.4
    rank = mpm.drop_pt_rank(pt, drop, max_drop=128)
    for i in range(4):
        r = sorted(rank[i][drop[i]].tolist())
        assert r == list(range(len(r))), "ranks must be 0..M-1 with no gaps"


def test_drop_pt_rank_stays_inside_the_mask_token_bank():
    pt = torch.randn(2, 300)
    drop = torch.ones(2, 300, dtype=torch.bool)
    rank = mpm.drop_pt_rank(pt, drop, max_drop=128)
    assert int(rank.max()) == 127, "rank must clamp to the last mask token"


def test_mask_rate_default_is_the_papers_tuned_value_not_the_ablation_default():
    # 0.3 is 2409.12589's ablation rate; 0.4 is its final tuned config (Table 1).
    assert mpm.DEFAULT_MASK_RATE == 0.40
