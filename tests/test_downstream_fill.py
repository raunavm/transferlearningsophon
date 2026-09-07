"""The zero-fill must be identical to the control that validates it.

`docs/PRD_PLAN.md` §4.5 / DECISIONS_PENDING item 16d approve zero-filling the
features the top and q/g benchmark sets do not carry, CONDITIONAL on a control
that masks the same features on JetClass-II and shows the arm ordering survives.
That control is evidence only if it masks exactly what the downstream config
fills. Nothing at runtime would notice a divergence: both configs still parse,
both still train, and the control would quietly measure a different intervention
than the one it exists to control for.

These tests re-derive the masked slots from the EMITTED YAML rather than from
the generator's table, so they fail if the generator is edited into
inconsistency, and they fail if either file is hand-edited afterwards.
"""
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent
FT = ROOT / "configs" / "finetune"
ARM = ROOT / "configs" / "arms" / "L162.yaml"
GEN = ROOT / "scripts" / "build_downstream_configs.py"

_spec = importlib.util.spec_from_file_location("_bdc", GEN)
bdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdc)


def _slots(path):
    """pf_features as a list of (varname, params) in file order."""
    d = yaml.safe_load(path.read_text())
    out = []
    for v in d["inputs"]["pf_features"]["vars"]:
        out.append((v[0], tuple(v[1:])) if isinstance(v, list) else (v, ()))
    return out


def _zero_indices(path):
    return {i for i, (name, _) in enumerate(_slots(path)) if name == bdc.ZERO_VAR}


@pytest.mark.parametrize("name", sorted(bdc.FILLS))
def test_control_masks_exactly_what_the_downstream_config_fills(name):
    down = _zero_indices(FT / f"{name}.yaml")
    ctrl = _zero_indices(FT / f"JetClassII_L162_mask{name}.yaml")
    assert down == ctrl, (
        f"{name}: downstream fills slots {sorted(down)} but its control masks "
        f"{sorted(ctrl)}. The control would not be measuring the fill.")
    assert len(down) == len(bdc.FILLS[name]["missing"])


@pytest.mark.parametrize("name", sorted(bdc.FILLS))
def test_control_differs_from_the_arm_only_in_the_masked_slots(name):
    arm, ctrl = _slots(ARM), _slots(FT / f"JetClassII_L162_mask{name}.yaml")
    assert len(arm) == len(ctrl) == 17
    masked = _zero_indices(FT / f"JetClassII_L162_mask{name}.yaml")
    for i, (a, c) in enumerate(zip(arm, ctrl)):
        if i in masked:
            assert c[0] == bdc.ZERO_VAR
            assert c[1] == a[1], f"slot {i}: standardization changed by masking"
        else:
            assert a == c, f"slot {i} differs from the arm but is not declared masked"


def test_generator_feature_table_matches_the_arm_config():
    """FEATURES is written out by hand; the arm config is ground truth."""
    arm = _slots(ARM)
    assert [n for n, _ in arm] == [n for n, _ in bdc.FEATURES]
    for (an, ap), (gn, gs) in zip(arm, bdc.FEATURES):
        got = tuple(None if x is None else x
                    for x in (yaml.safe_load(f"[{gs}]") if gs != "null" else [None]))
        assert ap == got, f"{an}: arm has {ap}, generator declares {got}"


@pytest.mark.parametrize("name", sorted(bdc.FILLS))
def test_every_filled_slot_maps_raw_zero_to_network_zero(name):
    """The fill is only a *zero*-fill if the standardization leaves it at zero.

    weaver computes clip((x - subtract) * multiply, clip_min, clip_max). A slot
    with a non-zero `subtract` would turn the fill into a constant offset the
    paper does not describe.
    """
    for i, (var, params) in enumerate(_slots(FT / f"{name}.yaml")):
        if var != bdc.ZERO_VAR:
            continue
        if not params or params[0] is None:
            continue                                   # no transform -> 0
        sub = params[0] or 0
        mul = params[1] if len(params) > 1 else 1
        lo = params[2] if len(params) > 2 else -5
        hi = params[3] if len(params) > 3 else 5
        val = min(max((0.0 - sub) * mul, lo), hi)
        assert val == 0.0, f"{name} slot {i}: raw 0 becomes {val}, not a zero-fill"


@pytest.mark.parametrize("name", sorted(bdc.FILLS))
def test_downstream_config_never_reads_a_branch_the_dataset_lacks(name):
    """A filled feature's SOURCE branch must not appear anywhere in the config.

    The top set has no `part_d0val`, so a leftover `part_d0: np.tanh(part_d0val)`
    in new_variables would crash the loader on the first batch -- after staging,
    after scheduling, on a GPU.
    """
    text = (FT / f"{name}.yaml").read_text()
    # Strip whole-line AND trailing comments: the generator annotates each filled
    # slot with the name of the feature it replaced, and those annotations are
    # documentation, not a read of the branch.
    body = "\n".join(re.split(r"\s+#", l)[0] for l in text.splitlines()
                     if not l.lstrip().startswith("#"))
    for var in bdc.FILLS[name]["missing"]:
        assert not re.search(rf"^\s*{re.escape(var)}\s*:", body, re.M), \
            f"{name}: defines {var} in new_variables although it is zero-filled"
        src = {"part_d0": "part_d0val", "part_dz": "part_dzval"}.get(var, var)
        assert src not in body, f"{name}: still references the absent branch {src}"


@pytest.mark.parametrize("name", sorted(bdc.FILLS))
def test_downstream_configs_carry_no_weights_block(name):
    for p in (FT / f"{name}.yaml", FT / f"JetClassII_L162_mask{name}.yaml"):
        assert yaml.safe_load(p.read_text()).get("weights") is None, \
            f"{p.name}: fine-tuning legs use the natural composition"


def test_committed_configs_are_what_the_generator_emits():
    r = subprocess.run([sys.executable, str(GEN), "--check-only"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
