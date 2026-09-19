"""experiments/AOJ/closure.py::compare -- the flags that stand between an
UNVERIFIED staging constant and a published score. Synthetic tensors only."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aoj_closure", ROOT / "experiments/AOJ/closure.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)

NAMES = ["part_pt_scale_log", "part_e_scale_log", "part_logptrel", "part_logerel", "part_deltaR",
         "part_charge", *cl.SPECIES, *cl.DISPLACEMENT, "part_deta", "part_dphi"]


def tensors(seed, n=1500, p=40, d0_scale=1.0, d0_sign=1.0, photon_frac=0.25, shift=0.0):
    """(features (n, 17, p), mask) shaped like weaver's pf_features after transforms."""
    rng = np.random.default_rng(seed)
    f = np.zeros((n, len(NAMES), p), dtype=np.float32)
    ix = NAMES.index
    for k in NAMES[:5]:
        f[:, ix(k)] = rng.normal(shift, 1.0, (n, p))
    species = rng.choice(5, (n, p), p=[0.6 - photon_frac / 2, 0.35 - photon_frac / 2, photon_frac, 0.03, 0.02])
    for i, k in enumerate(cl.SPECIES):
        f[:, ix(k)] = species == i
    charged = np.isin(species, [0, 3, 4])
    f[:, ix("part_charge")] = np.where(charged, rng.choice([-1, 1], (n, p)), 0)
    dphi, deta = rng.normal(0, 0.2, (n, p)), rng.normal(0, 0.2, (n, p))
    # displaced tracks: d0 correlated with -dphi (the CMS sign), dz with -deta
    d0 = rng.normal(0, 0.03, (n, p)) - 0.8 * dphi * (rng.random((n, p)) < 0.3)
    dz = rng.normal(0, 0.05, (n, p)) - 0.8 * deta * (rng.random((n, p)) < 0.3)
    f[:, ix("part_d0")] = np.tanh(d0 * d0_scale * d0_sign) * charged
    f[:, ix("part_dz")] = np.tanh(dz * d0_scale) * charged
    f[:, ix("part_d0err")] = np.clip(rng.uniform(0.01, 0.05, (n, p)) * d0_scale, 0, 1) * charged
    f[:, ix("part_dzerr")] = np.clip(rng.uniform(0.02, 0.08, (n, p)) * d0_scale, 0, 1) * charged
    f[:, ix("part_deta")], f[:, ix("part_dphi")] = deta, dphi
    mask = np.arange(p)[None, :] < rng.integers(15, p, n)[:, None]
    return f, mask


REF = tensors(0)


def flags(**kw):
    rows, hard, soft = cl.compare(*tensors(1, **kw), *REF, NAMES)
    return rows, hard, soft


def test_an_identically_distributed_sample_raises_no_flag():
    rows, hard, soft = flags()
    assert hard == [] and soft == []
    assert {r["feature"] for r in rows} >= set(NAMES) | {"d0_dphi_sign", "dz_deta_sign", "charged_no_track",
                                                         "neutral_with_displacement", "n_particles"}
    d0 = next(r for r in rows if r["feature"] == "part_d0")
    assert d0["ratio"] == pytest.approx(1.0, rel=0.1) and d0["iqr_ratio"] == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize("scale", [10.0, 0.1])
def test_a_factor_ten_in_the_displacement_units_is_a_hard_flag(scale):
    _, hard, _ = flags(d0_scale=scale)
    assert any(h.startswith("unit:part_d0 ") for h in hard) and any(h.startswith("unit:part_d0err") for h in hard)


def test_a_flipped_impact_parameter_sign_is_a_hard_flag_though_every_histogram_matches():
    rows, hard, _ = flags(d0_sign=-1.0)
    assert [h.split(":")[0] for h in hard] == ["d0_dphi_sign"]
    d0 = next(r for r in rows if r["feature"] == "part_d0")
    assert d0["ratio"] == pytest.approx(1.0, rel=0.1), "|d0| is blind to the sign; only the asymmetry sees it"


def test_a_species_fraction_off_by_more_than_30_percent_is_a_soft_flag():
    _, hard, soft = flags(photon_frac=0.40)
    assert hard == [] and any(s.startswith("species:part_isPhoton") for s in soft)


def test_disjoint_supports_are_a_hard_flag():
    _, hard, _ = flags(shift=8.0)
    assert {"support:part_logptrel", "support:part_deltaR"} <= set(hard)


def test_charged_candidates_without_displacement_are_reported_and_flagged():
    f, m = tensors(1)
    q = f[:, NAMES.index("part_charge")] != 0
    drop = q & (np.random.default_rng(2).random(q.shape) < 0.4)
    for k in cl.DISPLACEMENT:
        f[:, NAMES.index(k)][drop] = 0.0
    rows, hard, soft = cl.compare(f, m, *REF, NAMES)
    row = next(r for r in rows if r["feature"] == "charged_no_track")
    assert row["aoj"] == pytest.approx(0.4, abs=0.03) and row["reference"] == 0.0
    assert hard == [] and any(s.startswith("charged_no_track") for s in soft)


def test_a_correlation_present_in_one_sample_only_is_a_soft_flag():
    """What JetClass-II mirroring dz by the jet-eta sign would look like."""
    f, m = tensors(1)
    rng = np.random.default_rng(3)
    ix = NAMES.index("part_dz")
    f[:, ix] = f[:, ix] * rng.choice([-1.0, 1.0], (f.shape[0], 1))        # averages the asymmetry out
    _, hard, soft = cl.compare(f, m, *REF, NAMES)
    assert hard == [] and [s_.split(":")[0] for s_ in soft] == ["dz_deta_sign_one_sided"]
