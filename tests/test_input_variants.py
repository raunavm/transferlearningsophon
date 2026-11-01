"""Input variants of the missing-feature control (experiments/EVAL/input_variants.py)."""
import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("input_variants", ROOT / "experiments/EVAL/input_variants.py")
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)

NAMES = ["part_pt_scale_log", "part_e_scale_log", "part_logptrel", "part_logerel", "part_deltaR",
         "part_charge", "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron",
         "part_isMuon", "part_d0", "part_d0err", "part_dz", "part_dzerr", "part_deta", "part_dphi"]
S = iv.slots(NAMES)


def _jets(n=64, p=40, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 17, p)).astype(np.float32)
    species = r.choice(5, size=(n, p), p=[0.6, 0.1, 0.25, 0.03, 0.02])
    for k, i in enumerate(S["flags"]):
        x[:, i, :] = species == k
    charged = np.isin(species, iv.CHARGED_SPECIES)
    x[:, S["charge"], :] = np.where(charged, r.choice([-1, 1], size=(n, p)), 0)
    for i in S["disp"]:
        x[:, i, :] = np.where(charged, r.uniform(-1, 1, size=(n, p)), 0)
    mask = np.ones((n, 1, p), np.float32)
    mask[:, :, p - 5:] = 0
    return x, mask


def _ref():
    x, m = _jets(200, seed=1)
    return iv.reference(x, m, S, np.random.default_rng(0))


def test_slots_follow_the_config_order():
    assert S["charge"] == 5 and S["flags"] == [6, 7, 8, 9, 10] and S["disp"] == [11, 12, 13, 14]
    assert sorted(S["top"]) == list(range(5, 15)) and S["qg"] == [11, 12, 13, 14]


def test_reference_frequencies_and_pools():
    ref = _ref()
    assert ref["species_p"].sum() == pytest.approx(1.0) and ref["species_p"][0] > 0.5
    assert (ref["pool_neutral"] == 0).all() and np.abs(ref["pool_charged"]).max() <= 1


@pytest.mark.parametrize("name,zeroed", [("mtop", list(range(5, 15))), ("mqg", [11, 12, 13, 14])])
def test_masks_zero_exactly_their_slots(name, zeroed):
    x, _ = _jets()
    y = iv.Variant(name, S)(torch.tensor(x), 0).numpy()
    assert (y[:, zeroed, :] == 0).all()
    kept = [i for i in range(17) if i not in zeroed]
    assert np.array_equal(y[:, kept, :], x[:, kept, :])


def test_itop_keeps_inputs_on_the_manifold_and_is_deterministic():
    x, _ = _jets()
    v = iv.Variant("itop", S, _ref())
    y = v(torch.tensor(x), 7).numpy()
    flags = y[:, S["flags"], :]
    assert ((flags == 0) | (flags == 1)).all() and (flags.sum(1) == 1).all()   # exactly one flag set
    species = flags.argmax(1)
    neutral = ~np.isin(species, iv.CHARGED_SPECIES)
    assert (y[:, S["charge"], :][neutral] == 0).all()
    assert (np.abs(y[:, S["charge"], :][~neutral]) == 1).all()
    assert (y[:, S["disp"], :].transpose(0, 2, 1)[neutral] == 0).all()
    kin = [0, 1, 2, 3, 4, 15, 16]
    assert np.array_equal(y[:, kin, :], x[:, kin, :])
    assert np.array_equal(y, v(torch.tensor(x), 7).numpy())                    # same counter, same draw
    assert not np.array_equal(y, v(torch.tensor(x), 8).numpy())


def test_iqg_redraws_displacement_only_for_the_kept_charge():
    x, _ = _jets()
    y = iv.Variant("iqg", S, _ref())(torch.tensor(x), 0).numpy()
    keep = [i for i in range(17) if i not in S["disp"]]
    assert np.array_equal(y[:, keep, :], x[:, keep, :])
    neutral = x[:, S["charge"], :] == 0
    assert (y[:, S["disp"], :].transpose(0, 2, 1)[neutral] == 0).all()
    assert not np.array_equal(y[:, S["disp"], :], x[:, S["disp"], :])


def test_wrapped_model_exposes_mod_and_counts_batches():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = torch.nn.Identity()

        def forward(self, a, f):
            return f

    w = iv.wrap(M(), iv.Variant("mtop", S), 1)
    x, _ = _jets(4)
    out = w(torch.zeros(1), torch.tensor(x))
    assert (out[:, 5:15, :] == 0).all() and w.counter == 1 and isinstance(w.mod, torch.nn.Identity)


def test_slots_from_the_real_data_configs():
    # weaver's input_dicts holds names, not [name, ...] lists; v[0] took each name's first letter (2026-10-31)
    DataConfig = pytest.importorskip("weaver.utils.data.config").DataConfig
    for cfg in ("configs/data/JetClassII_base.yaml", "configs/finetune/JetClassII_L188_noweight.yaml"):
        names = iv.pf_names(DataConfig.load(str(ROOT / cfg)))
        assert names == NAMES and iv.slots(names) == S


# ---- 2026-11-01: mean fill and four-draw imputation (PI: every model, zero / mean / imputed) ----

def _jets_by_species(n=48, p=30, seed=3):
    """Jets whose displacement encodes the species: charged species k carry 10k + u, neutral ones 0."""
    x, mask = _jets(n, p, seed)
    species = x[:, S["flags"], :].argmax(1)
    u = np.random.default_rng(seed).uniform(0, 1, size=(n, p))
    charged = np.isin(species, iv.CHARGED_SPECIES)
    for i in S["disp"]:
        x[:, i, :] = np.where(charged, 10 * species + u, 0)
    return x, mask, species


def _ref_by_species():
    x, m, _ = _jets_by_species(300, seed=4)
    return iv.reference(x, m, S, np.random.default_rng(0))


def test_reference_keeps_a_displacement_pool_per_species():
    ref = _ref_by_species()
    for k in range(5):
        pool = ref[f"pool_species{k}"]
        if k in iv.CHARGED_SPECIES:
            assert (np.floor(pool) == 10 * k).all()
        else:
            assert (pool == 0).all()


def test_mean_fill_reaches_the_network_at_the_batchnorm_bias():
    x, _ = _jets()
    bn = torch.nn.BatchNorm1d(17).eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.linspace(-0.4, 0.5, 17))
        bn.running_var.copy_(torch.linspace(0.1, 1.2, 17))
        bn.bias.copy_(torch.linspace(1.0, 2.0, 17))
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = torch.nn.Module()
            self.mod.embed = torch.nn.Module()
            self.mod.embed.input_bn = bn
    fill = iv.input_bn_mean(M())
    for name, sl in (("ntop", S["top"]), ("nqg", S["qg"])):
        y = iv.Variant(name, S, fill=fill)(torch.tensor(x), 0)
        out = bn(y).detach().numpy()
        for i in sl:
            assert np.allclose(out[:, i, :], float(bn.bias[i]), atol=1e-5)    # the centroid, exactly
        kept = [i for i in range(17) if i not in sl]
        assert np.array_equal(y.numpy()[:, kept, :], x[:, kept, :])


def test_iqg4_conditions_on_the_real_flags_and_keeps_flags_and_charge():
    x, _, species = _jets_by_species()
    ref = _ref_by_species()
    ys = [iv.Variant("iqg4", S, ref, draw=j)(torch.tensor(x), 5).numpy() for j in range(iv.DRAWS)]
    keep = [i for i in range(17) if i not in S["disp"]]
    charged = np.isin(species, iv.CHARGED_SPECIES)
    for y in ys:
        assert np.array_equal(y[:, keep, :], x[:, keep, :])                 # flags, charge, kinematics untouched
        d = y[:, S["disp"], :].transpose(0, 2, 1)                           # (n, p, 4)
        assert (np.floor(d[charged]) == 10 * species[charged][:, None]).all()   # the particle's own species
        assert (d[~charged] == 0).all()
    assert all(not np.array_equal(ys[0], y) for y in ys[1:])                # four different draws
    assert np.array_equal(ys[2], iv.Variant("iqg4", S, ref, draw=2)(torch.tensor(x), 5).numpy())


def test_itop4_draws_a_species_and_that_species_displacement():
    x, _, _ = _jets_by_species()
    y = iv.Variant("itop4", S, _ref_by_species(), draw=1)(torch.tensor(x), 0).numpy()
    flags = y[:, S["flags"], :]
    assert ((flags == 0) | (flags == 1)).all() and (flags.sum(1) == 1).all()
    sp = flags.argmax(1)
    charged = np.isin(sp, iv.CHARGED_SPECIES)
    assert (np.abs(y[:, S["charge"], :][charged]) == 1).all() and (y[:, S["charge"], :][~charged] == 0).all()
    d = y[:, S["disp"], :].transpose(0, 2, 1)
    assert (np.floor(d[charged]) == 10 * sp[charged][:, None]).all() and (d[~charged] == 0).all()


def test_new_variants_refuse_missing_inputs():
    x, _, _ = _jets_by_species()
    with pytest.raises(SystemExit):
        iv.Variant("ntop", S)                                                # no BatchNorm mean
    old = {k: v for k, v in _ref_by_species().items() if not k.startswith("pool_species")}
    with pytest.raises(SystemExit):
        iv.Variant("iqg4", S, old)                                           # a version-1 reference
    with pytest.raises(SystemExit):
        iv.Variant("iqg4", S, _ref_by_species(), draw=iv.DRAWS)
    x[:, S["flags"], 0] = 0                                                  # a particle with no flag
    with pytest.raises(SystemExit):
        iv.Variant("iqg4", S, _ref_by_species())(torch.tensor(x), 0)


def test_slots_read_the_downstream_zero_filled_names():
    DataConfig = pytest.importorskip("weaver.utils.data.config").DataConfig
    top = iv.slots(iv.pf_names(DataConfig.load(str(ROOT / "configs/finetune/TopReference.yaml"))))
    qg = iv.slots(iv.pf_names(DataConfig.load(str(ROOT / "configs/finetune/EnergyFlowQG.yaml"))))
    assert top == S and qg == S


def test_draws_are_averaged_into_one_variant():
    xs = importlib.util.spec_from_file_location("extract_v2", ROOT / "experiments/EVAL/extract_v2.py")
    ex = importlib.util.module_from_spec(xs)
    xs.loader.exec_module(ex)
    rows, lab = np.arange(5), np.zeros(5, np.int16)
    part = lambda v: {"features": np.full((5, 128), v, np.float16), "pooled": None, "rows": rows,
                      "label188": lab, "has_head": False, "head": {}, "head_rows": rows[:0],
                      "head_label188": lab[:0], "head_outputs": None}
    res = {"checkpoints": {"best70": part(9.0), **{f"best70.iqg4~{j}": part(j) for j in range(4)}}}
    ex.merge_draws(res)
    assert sorted(res["checkpoints"]) == ["best70", "best70.iqg4"]
    assert np.allclose(res["checkpoints"]["best70.iqg4"]["features"], 1.5) and not res["checkpoints"]["best70.iqg4"]["has_head"]
    bad = {"checkpoints": {f"t.itop4~{j}": {**part(j), "rows": rows + j} for j in range(2)}}
    with pytest.raises(SystemExit):
        ex.merge_draws(bad)
