"""scripts/stage_aoj.py on SYNTHETIC AspenOpenJets arrays -- no network, no GPU.

Every test pins one convention from the producer code (H5_maker.py line
references are in stage_aoj.py's docstring). A failure here means the staged
inputs are not what the trunk was trained on, which no downstream number would
reveal on its own.
"""
import importlib.util
import pathlib
import re

import awkward as ak
import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sa = _load("stage_aoj")
PDGS = np.array([211, -211, 130, 22, 11, -11, 13, -13, 1, 2])
CHARGE = {211: 1, -211: -1, 130: 0, 22: 0, 11: -1, -11: 1, 13: -1, -13: 1, 1: 0, 2: 0}


def synth(n_jets=40, n_const=(20, 60), seed=0, big_jet=None):
    """AOJ-layout arrays: zero-padded (n, 150, 11), UNWEIGHTED momenta, -1
    sentinels, candidates in RANDOM order so the re-sort is exercised."""
    rng = np.random.default_rng(seed)
    pf = np.zeros((n_jets, 150, 11), dtype=np.float32)
    jet_eta = rng.uniform(0.3, 2.0, n_jets) * rng.choice([-1, 1], n_jets)
    jet_phi = rng.uniform(-np.pi, np.pi, n_jets)
    for j in range(n_jets):
        k = big_jet if (big_jet and j == 0) else rng.integers(*n_const)
        pt = rng.exponential(15.0, k) + 0.5
        eta = jet_eta[j] + rng.normal(0, 0.2, k)
        phi = jet_phi[j] + rng.normal(0, 0.2, k)
        pdg = rng.choice(PDGS, k, p=[.3, .3, .1, .2, .02, .02, .02, .02, .01, .01])
        q = np.array([CHARGE[int(p)] for p in pdg], dtype=np.float32)
        track = (q != 0) & (pt > 1.0)                  # soft charged: no track details
        pf[j, :k, 0], pf[j, :k, 1] = pt * np.cos(phi), pt * np.sin(phi)
        pf[j, :k, 2], pf[j, :k, 3] = pt * np.sinh(eta), pt * np.cosh(eta)
        pf[j, :k, 4] = np.where(track, rng.normal(0, 0.005, k), -1)      # d0   [cm]
        pf[j, :k, 5] = np.where(track, rng.uniform(0.001, 0.01, k), -1)  # d0Err
        pf[j, :k, 6] = np.where(track, rng.normal(0, 0.01, k), -1)       # dz
        pf[j, :k, 7] = np.where(track, rng.uniform(0.002, 0.02, k), -1)  # dzErr
        pf[j, :k, 8], pf[j, :k, 9] = q, pdg
        pf[j, :k, 10] = rng.choice([0.0, 0.5, 1.0], k, p=[.15, .15, .7])
    kin = np.stack([np.full(n_jets, 600.0), jet_eta, jet_phi, np.full(n_jets, 80.0)],
                   axis=1).astype(np.float32)
    tag = np.tile(np.arange(13, dtype=np.float32) / 100, (n_jets, 1))
    tag[:, 0] = (pf[:, :, 3] > 0).sum(axis=1)
    ev = np.stack([np.full(n_jets, 280000), np.arange(n_jets), 5_000_000_000 + np.arange(n_jets)],
                  axis=1).astype(np.int64)
    return ev, kin, pf, tag


@pytest.fixture(scope="module")
def staged():
    ev, kin, pf, tag = synth()
    rec, cnt = sa.convert(ev, kin, pf, tag)
    return (ev, kin, pf, tag), rec, cnt


def test_puppi_weight_is_applied_and_zero_weight_candidates_are_dropped(staged):
    (_, _, pf, _), rec, cnt = staged
    w = pf[:, :, 10].astype(np.float64)
    keep = (pf[:, :, 3] > 0) & (w > 0)
    assert ak.to_numpy(ak.num(rec["part_px"])).tolist() == keep.sum(axis=1).tolist()
    assert cnt["n_dropped_zero_puppi"] == int(((pf[:, :, 3] > 0) & (w == 0)).sum()) > 0
    np.testing.assert_allclose(ak.to_numpy(ak.sum(rec["part_px"], axis=1)),
                               (pf[:, :, 0] * w * keep).sum(axis=1), rtol=1e-4, atol=1e-3)
    # jet_energy is REBUILT from the weighted constituents, not read from the file
    np.testing.assert_allclose(ak.to_numpy(rec["jet_energy"]),
                               (pf[:, :, 3] * w * keep).sum(axis=1), rtol=1e-5)


def test_candidates_are_ordered_by_weighted_pt(staged):
    _, rec, _ = staged
    pt = np.hypot(rec["part_px"], rec["part_py"])
    assert ak.all(pt[:, :-1] >= pt[:, 1:])


def test_truncation_keeps_the_hardest_128_but_the_jet_sums_all_of_them():
    ev, kin, pf, tag = synth(n_jets=3, big_jet=150, seed=3)
    pf[0, :, 10] = 1.0                                   # all 150 survive PUPPI
    rec, cnt = sa.convert(ev, kin, pf, tag)
    assert ak.num(rec["part_px"])[0] == 128 and rec["jet_nparticles"][0] == 150
    assert cnt["n_jets_over_model"] == 1
    pt_all = np.sort(np.hypot(pf[0, :, 0], pf[0, :, 1]))[::-1]
    np.testing.assert_allclose(ak.to_numpy(np.hypot(rec["part_px"], rec["part_py"])[0]),
                               pt_all[:128], rtol=1e-5)
    assert rec["jet_energy"][0] == pytest.approx(pf[0, :, 3].sum(), rel=1e-5)


def test_displacement_is_converted_cm_to_mm_and_the_sentinel_becomes_zero(staged):
    (_, _, pf, _), rec, cnt = staged
    assert sa.UNVERIFIED_CM_TO_MM == 10.0
    d0, d0e = ak.to_numpy(ak.flatten(rec["part_d0val"])), ak.to_numpy(ak.flatten(rec["part_d0err"]))
    dz, dze = ak.to_numpy(ak.flatten(rec["part_dzval"])), ak.to_numpy(ak.flatten(rec["part_dzerr"]))
    q = ak.to_numpy(ak.flatten(rec["part_charge"]))
    assert (d0e >= 0).all() and (dze >= 0).all(), "a -1 sentinel reached the output"
    none = d0e == 0
    assert (d0[none] == 0).all() and (dz[none] == 0).all() and (dze[none] == 0).all()
    assert none[q == 0].all(), "JetClass-II: neutrals carry exactly zero displacement"
    assert cnt["n_charged_no_track"] > 0 and none[q != 0].sum() == cnt["n_charged_no_track"]
    src = pf[(pf[:, :, 3] > 0) & (pf[:, :, 10] > 0) & (pf[:, :, 5] > 0)]
    np.testing.assert_allclose(np.sort(np.abs(d0[~none])), np.sort(np.abs(src[:, 4])) * 10, rtol=1e-5)
    np.testing.assert_allclose(np.sort(d0e[~none]), np.sort(src[:, 5]) * 10, rtol=1e-5)


def test_d0_takes_the_delphes_sign_and_dz_keeps_its_own():
    """CMS dxy = -dx sin(phi) + dy cos(phi); upstream Delphes d0 = (xd py - yd px)/pT.
    Opposite. Pinned so that flipping the constant is a decision, not an accident."""
    assert (sa.UNVERIFIED_D0_SIGN, sa.UNVERIFIED_DZ_SIGN) == (-1.0, +1.0)
    ev, kin, pf, tag = synth(n_jets=1, seed=7)
    pf[0] = 0
    for i, (pt, d0, dz) in enumerate([(30.0, 0.02, -0.03), (20.0, -0.01, 0.05), (10.0, 0.004, 0.001)]):
        pf[0, i] = [pt, 0.5, pt, 2 * pt, d0, 0.002, dz, 0.004, 1, 211, 1.0]
    pf[0, 3] = [5.0, 0.5, 5.0, 10.0, 0.03, 0.002, 0.03, 0.004, 0, 22, 1.0]   # a photon WITH "track" values
    rec, cnt = sa.convert(ev, kin, pf, tag)
    np.testing.assert_allclose(ak.to_numpy(rec["part_d0val"][0]), [-0.2, 0.1, -0.04, 0.0], rtol=1e-5)
    np.testing.assert_allclose(ak.to_numpy(rec["part_dzval"][0]), [-0.3, 0.5, 0.01, 0.0], rtol=1e-5)
    np.testing.assert_allclose(ak.to_numpy(rec["part_d0err"][0]), [0.02, 0.02, 0.02, 0.0], rtol=1e-5)
    assert cnt["n_neutral_with_track_zeroed"] == 1, "a neutral never carries displacement in JetClass-II"


def test_pdgid_maps_to_one_hot_species_with_forward_calorimeter_ids(staged):
    (_, _, pf, _), rec, cnt = staged
    flags = np.stack([ak.to_numpy(ak.flatten(rec[f"part_is{n}"])) for n in sa.SPECIES])
    assert (flags.sum(axis=0) == 1).all()
    assert cnt["n_hf"] > 0 and cnt["n_unknown_pdgid"] == 0 and cnt["n_charge_mismatch"] == 0
    # one jet, one candidate of each id, to read the mapping off directly
    ev, kin, one, tag = synth(n_jets=1, seed=1)
    one[0] = 0
    for i, pid in enumerate(PDGS):
        one[0, i, :4] = [10.0 + i, 1.0, 5.0, 20.0]
        one[0, i, 4:8], one[0, i, 8], one[0, i, 9], one[0, i, 10] = -1, CHARGE[int(pid)], pid, 1.0
    r, _ = sa.convert(ev, kin, one, tag)
    order = np.argsort(-np.hypot(one[0, :10, 0], one[0, :10, 1]))
    got = [sa.SPECIES[int(np.argmax([r[f"part_is{n}"][0][k] for n in sa.SPECIES]))] for k in range(10)]
    want = {211: "ChargedHadron", 130: "NeutralHadron", 22: "Photon", 11: "Electron",
            13: "Muon", 1: "NeutralHadron", 2: "Photon"}
    assert got == [want[abs(int(PDGS[i]))] for i in order]
    assert ak.to_numpy(r["part_charge"][0]).tolist() == [CHARGE[int(PDGS[i])] for i in order]


def test_a_misread_charge_column_is_refused():
    ev, kin, pf, tag = synth(seed=2)
    pf[:, :, 8] *= -1                                    # e.g. sign(pdgId) used for leptons and hadrons alike
    with pytest.raises(SystemExit, match="n_charge_mismatch"):
        sa.convert(ev, kin, pf, tag)


def test_deta_is_flipped_by_the_jet_eta_sign_and_dphi_is_not(staged):
    _, rec, _ = staged
    pt = np.hypot(rec["part_px"], rec["part_py"])
    eta, phi = np.arcsinh(rec["part_pz"] / pt), np.arctan2(rec["part_py"], rec["part_px"])
    sign = np.where(ak.to_numpy(rec["jet_eta"]) > 0, 1.0, -1.0)
    assert {-1.0, 1.0} == set(sign.tolist()), "fixture must contain both hemispheres"
    assert ak.max(abs(rec["part_deta"] - (eta - rec["jet_eta"]) * sign)) < 1e-4
    assert ak.max(abs(rec["part_dphi"] - sa.wrap(phi - rec["jet_phi"]))) < 1e-4


def test_selection_boundaries():
    kin = np.array([[600, 1.0, 0, 80], [499, 1.0, 0, 80], [2600, 1.0, 0, 80],
                    [600, -2.45, 0, 80], [600, 1.0, 0, 19], [600, 1.0, 0, 501],
                    [600, -2.39, 0, 499]], dtype=np.float32)
    assert sa.select(kin).tolist() == [True, False, False, False, False, False, True]


def test_observers_are_the_columns_the_producer_wrote(staged):
    (ev, kin, _, tag), rec, _ = staged
    np.testing.assert_array_equal(ak.to_numpy(rec["aoj_pn_WvsQCD"]), tag[:, 10])
    np.testing.assert_array_equal(ak.to_numpy(rec["aoj_pn_TvsQCD"]), tag[:, 9])
    np.testing.assert_array_equal(ak.to_numpy(rec["aoj_pn_HbbvsQCD"]), tag[:, 6])
    np.testing.assert_array_equal(ak.to_numpy(rec["jet_sdmass"]), kin[:, 3])
    np.testing.assert_array_equal(ak.to_numpy(rec["aoj_jet_pt"]), kin[:, 0])
    assert ak.to_numpy(rec["event"]).dtype == np.int64
    np.testing.assert_array_equal(ak.to_numpy(rec["event"]), ev[:, 2])   # > 2**32, exact


def test_hdf5_to_parquet_round_trip_and_config_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    ev, kin, pf, tag = synth(n_jets=30, seed=5)
    kin[:5, 0] = 300.0                                   # five jets fail pT > 500
    with h5py.File(tmp_path / "RunG_batch0.h5", "w") as f:
        for k, v in dict(event_info=ev, jet_kinematics=kin, PFCands=pf, jet_tagging=tag).items():
            f.create_dataset(k, data=v)
    stats = sa.stage_file(tmp_path / "RunG_batch0.h5", tmp_path / "out.parquet", chunk=7)
    got = ak.from_parquet(tmp_path / "out.parquet")
    assert len(got) == 25 == stats["counters"]["n_jets"] and stats["n_jets_read"] == 30
    np.testing.assert_array_equal(ak.to_numpy(got["event"]), ev[5:, 2])  # file order kept
    # ONE chunk is a different code path: no concatenate, so per-jet columns reach
    # pyarrow as strided views of the (n, 4) table unless made contiguous
    sa.stage_file(tmp_path / "RunG_batch0.h5", tmp_path / "one.parquet")
    one = ak.from_parquet(tmp_path / "one.parquet")
    assert ak.all(one["part_px"] == got["part_px"]) and ak.all(one["jet_sdmass"] == got["jet_sdmass"])

    # every raw branch the generated config reads must exist in the staged file
    cfg = yaml.safe_load((ROOT / "configs/finetune/AspenOpenJets.yaml").read_text())
    nv = cfg["new_variables"]
    exprs = [cfg["selection"], *map(str, nv.values())]
    used = {v[0] for blk in cfg["inputs"].values() for v in blk["vars"]} | set(cfg["observers"])
    used |= {t for e in exprs for t in re.findall(r"\b(?:part|jet|aoj|label)\w*|\brun\b|\blumi\b|\bevent\b", e)}
    missing = sorted(u for u in used if u not in nv and u not in got.fields)
    assert not missing, f"config reads {missing}, which stage_aoj.py does not write"


def test_aoj_configs_are_current_and_inference_only():
    bac = _load("build_aoj_config")
    ft = ROOT / "configs" / "finetune"
    for name, build in bac.TARGETS.items():
        assert (ft / name).read_text() == build(), f"{name} is stale; re-run build_aoj_config.py"
        assert "weights:" not in yaml.safe_load((ft / name).read_text())
    aoj = yaml.safe_load((ft / "AspenOpenJets.yaml").read_text())
    ref = yaml.safe_load((ft / "JetClassII_base_selAspenOpenJets.yaml").read_text())
    arm = yaml.safe_load((ROOT / "configs/arms/L162.yaml").read_text())
    assert aoj["inputs"] == ref["inputs"] == arm["inputs"], "nothing may be filled or re-standardized"
    assert not any(k.startswith(bac.bdc.ZERO_PREFIX) for k in aoj["new_variables"])
    assert aoj["labels"] == dict(type="simple", value=["label_data", "label_unused"])
    # the reference reads JetClass-II through the SAME cuts, on its own branch names
    assert ref["selection"] == aoj["selection"].replace("aoj_jet_", "jet_")
    for k in ("part_d0", "part_dz"):
        assert aoj["new_variables"][k] == ref["new_variables"][k]
