"""The kinematic-only pretraining arms (scripts/build_kin_arms.py) and their launch specs."""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bka = _load("build_kin_arms", "scripts/build_kin_arms.py")


def test_configs_on_disk_are_current():
    assert bka.main(["--check"]) == 0


@pytest.mark.parametrize("arm", sorted(bka.PARENTS))
def test_only_pf_features_differ_from_parent(arm):
    parent, _ = bka.PARENTS[arm]
    child = (ROOT / "configs/arms/v2" / f"{arm}.yaml").read_text()
    par = (ROOT / "configs/arms" / f"{parent}.yaml").read_text()
    assert bka.block(child, "weights") == bka.block(par, "weights")   # I2, byte for byte
    assert bka.block(child, "labels") == bka.block(par, "labels")
    assert bka.block(child, "selection") == bka.block(par, "selection")
    removed = [l for l in par.splitlines(keepends=True) if l not in child.splitlines(keepends=True)]
    assert sorted(l for l in removed if not l.startswith("#")) == sorted(bka.DROP)


@pytest.mark.parametrize("arm", sorted(bka.PARENTS))
def test_seven_kinematic_features_in_parent_order(arm):
    child = (ROOT / "configs/arms/v2" / f"{arm}.yaml").read_text()
    parent, _ = bka.PARENTS[arm]
    par = (ROOT / "configs/arms" / f"{parent}.yaml").read_text()
    names = bka.pf_feature_names(child)
    assert names == list(bka.KEPT)
    full = bka.pf_feature_names(par)
    assert len(full) == 17 and [n for n in full if n in names] == names


def test_registry_matches_parents():
    reg = json.loads((ROOT / "configs/arms/v2_kin.json").read_text())
    assert {a["name"]: (a["parent"], a["num_classes"]) for a in reg["arms"]} == bka.PARENTS
    assert all(a["runs"] == 1 and a["mass_lambda"] is None for a in reg["arms"])


def test_derived_sidecar_changes_only_pf_features():
    dks = _load("derive_kin_sidecar", "experiments/MTX/derive_kin_sidecar.py")
    side = {"inputs": {"pf_features": {"length": 128, "vars": [["a", None], ["b", None], ["c", None]]},
                       "pf_vectors": {"vars": [["px", None]]}},
            "weights": {"reweight_hists": {"x": [[0.1, 0.2]]}}, "selection": "s"}
    child = {"inputs": {"pf_features": {"length": 128, "vars": [["a", None], ["c", None]]}}}
    out = dks.derive(side, child)
    assert out["inputs"]["pf_features"]["vars"] == [["a", None], ["c", None]]
    assert out["weights"] == side["weights"] and out["selection"] == "s"
    assert out["inputs"]["pf_vectors"] == side["inputs"]["pf_vectors"]
    assert side["inputs"]["pf_features"]["vars"][1] == ["b", None]       # parent untouched


def test_kin_specs():
    launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    specs = launch.v2_kin_specs("mtx-test")
    assert sorted(specs) == ["job-mtx2-l188kin-s1-raunav.yaml", "job-mtx2-r16q1kin-s1-raunav.yaml"]
    for fn, s in specs.items():
        arm = "L188_KIN" if "l188" in fn else "R16_Q1_KIN"
        k = 188 if arm == "L188_KIN" else 17
        assert "suspend: true" not in s
        assert f"--data-config ${{CFG}}" in s and f"CFG=configs/arms/v2/{arm}.yaml" in s
        assert f"-o num_classes {k} " in s and "--mass-lambda" not in s
        assert s.index("python3 experiments/MTX/derive_kin_sidecar.py") < s.index("python3 experiments/MTX/pretrain_v2.py")
        assert "RUN_ID=mtx-" + arm.lower().replace("_", "") + "-s1" in s
        assert "SEED=1" in s and "V100" not in s.split("nvidia.com/gpu.product", 1)[1].split("hostname")[0]
        assert '"NVIDIA-GeForce-RTX-3090"' in s and "raunav" in fn
