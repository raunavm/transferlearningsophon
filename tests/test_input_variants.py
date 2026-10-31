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
