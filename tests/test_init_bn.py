"""The BatchNorm-recomputed twin of the untrained trunk (init_bn): written by
experiments/MTX/bn_twins_v2.py --init on the run's best70 twin's sample, resolved by
experiments/EVAL/extract_v2.py, and extracted for every split of the downstream linear
probes by scripts/build_extract_jobs.py --v2-init-bn.

The bn_twins part is a CPU run of pretrain_v2 on the synthetic files of
tests/test_pretrain_v2.py (tests/test_bn_twins_v2.py's fixtures); the rest needs no torch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "MTX"))
sys.path.insert(0, str(ROOT / "tests"))

HAVE_TORCH = all(importlib.util.find_spec(m) is not None for m in ("torch", "weaver"))
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch and weaver")
if HAVE_TORCH:
    import torch
    from test_pretrain_v2 import data  # noqa: F401  (a fixture of done)
    from test_bn_twins_v2 import BN_KEYS, done, run  # noqa: F401  (fixtures)
    import bn_twins_v2 as bt
    import pretrain_v2 as pv


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------ bn_twins_v2.py --init
def _best70_twin(run: pathlib.Path) -> dict:
    e = int(json.loads((run / "best_window_epoch.json").read_text())["epoch"])
    return json.loads(bt.twin_paths(run, e)[1].read_text())


@needs_torch
def test_the_init_twin_changes_only_batchnorm_on_best70s_sample(run):
    init_sha = _sha(run / bt.INIT_FILE)
    assert bt.main(["--out", str(run)]) == 0
    assert bt.main(["--out", str(run), "--init"]) == 0
    ref = _best70_twin(run)
    m = json.loads((run / bt.INIT_TWIN_JSON).read_text())
    assert m["bn_recompute"] == ref["bn_recompute"]                  # the same jets, rows, seeds
    assert m["inputs"] == {"init": init_sha} and _sha(run / bt.INIT_FILE) == init_sha
    assert m["sha256"] == pv.sha256_file(run / bt.INIT_TWIN)
    assert m["sample_of"]["rows_sha256"] == ref["bn_recompute"]["rows_sha256"]
    assert m["sample_of"]["device"] == ref["device"]
    assert m["code"]["driver"] == "experiments/MTX/bn_twins_v2.py"
    raw = torch.load(run / bt.INIT_TWIN)
    assert set(raw) == {"trunk"}                                     # init_trunk.pt's format
    old, new = torch.load(run / bt.INIT_FILE)["trunk"], raw["trunk"]
    assert old.keys() == new.keys()
    stats = [k for k in old if k.endswith(BN_KEYS)]
    assert stats and all(torch.equal(old[k], new[k]) for k in old if k not in stats)
    assert any(not torch.equal(old[k], new[k]) for k in stats if "running_" in k)
    # the input BatchNorm sees the jets before any weight: the same as best70's twin's
    twin70 = torch.load(bt.twin_paths(run, ref["epoch"])[0])
    for k in ("running_mean", "running_var"):
        torch.testing.assert_close(new[f"embed.input_bn.{k}"], twin70[f"mod.embed.input_bn.{k}"])


@needs_torch
def test_the_init_twin_needs_best70s_twin(run, capsys):
    assert bt.main(["--out", str(run), "--init"]) == pv.EXIT_HALT
    assert "lacks" in capsys.readouterr().out
    assert not list(run.glob("init_bn*"))


@needs_torch
def test_the_init_twin_is_never_overwritten(run, capsys):
    assert bt.main(["--out", str(run)]) == 0
    assert bt.main(["--out", str(run), "--init"]) == 0
    before = {p.name: p.stat().st_mtime_ns for p in run.glob("init_bn*")}
    assert sorted(before) == sorted([bt.INIT_TWIN, bt.INIT_TWIN_JSON])
    assert bt.main(["--out", str(run), "--init"]) == pv.EXIT_HALT
    assert "nothing is overwritten" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in run.glob("init_bn*")} == before


@needs_torch
def test_a_state_file_without_its_record_is_overwritten(run):
    assert bt.main(["--out", str(run)]) == 0
    (run / bt.INIT_TWIN).write_bytes(b"a write that died")             # no init_bn.json
    assert bt.main(["--out", str(run), "--init"]) == 0
    m = json.loads((run / bt.INIT_TWIN_JSON).read_text())
    assert m["sha256"] == pv.sha256_file(run / bt.INIT_TWIN)
    assert set(torch.load(run / bt.INIT_TWIN)) == {"trunk"}
    assert not list(run.glob("init_bn*.tmp"))


@needs_torch
def test_a_sample_that_is_not_best70s_twins_writes_nothing(run, monkeypatch, capsys):
    assert bt.main(["--out", str(run)]) == 0
    monkeypatch.setattr(pv, "BN_JETS", 128)
    assert bt.main(["--out", str(run), "--init"]) == pv.EXIT_HALT
    assert "is not best70's twin's" in capsys.readouterr().out
    assert not list(run.glob("init_bn*"))


# ------------------------------------------------------------ extract_v2.py init_bn
def _write_init_twin(d: pathlib.Path):
    """An untrained trunk and its twin as bn_twins_v2.py --init writes them."""
    (d / xv.INIT_FILE).write_text("init")
    (d / xv.INIT_BN_FILE).write_text("init twin")
    (d / xv.INIT_BN_JSON).write_text(json.dumps(
        {"inputs": {"init": _sha(d / xv.INIT_FILE)}, "sha256": _sha(d / xv.INIT_BN_FILE)}))


def test_init_bn_resolves_to_the_twin_and_has_no_output_layer(tmp_path):
    _write_init_twin(tmp_path)
    got = xv.resolve_checkpoints(tmp_path, ["init", "init_bn"])
    assert got == [("init", tmp_path / "init_trunk.pt"), ("init_bn", tmp_path / "init_bn_state.pt")]
    assert xv.aliases(got) == {"init": "init", "init_bn": "init_bn"}
    assert {"init", "init_bn"} <= set(xv.HEADLESS_TAGS)


def test_a_missing_or_mismatched_init_twin_is_refused(tmp_path):
    (tmp_path / xv.INIT_FILE).write_text("init")
    with pytest.raises(SystemExit, match="no BatchNorm twin of init_trunk.pt: run experiments/MTX/bn_twins_v2.py --init"):
        xv.resolve_checkpoints(tmp_path, ["init_bn"])
    _write_init_twin(tmp_path)
    (tmp_path / xv.INIT_FILE).write_text("another trunk")
    with pytest.raises(SystemExit, match="does not record init_bn_state.pt as the twin of init_trunk.pt"):
        xv.resolve_checkpoints(tmp_path, ["init_bn"])
    _write_init_twin(tmp_path)
    (tmp_path / xv.INIT_BN_FILE).write_text("another twin")
    with pytest.raises(SystemExit, match="does not record init_bn_state.pt"):
        xv.resolve_checkpoints(tmp_path, ["init_bn"])


# ------------------------------------------------------------ build_extract_jobs.py --v2-init-bn
def _bx():
    return _load("build_extract_jobs", "scripts/build_extract_jobs.py")


def _body(text: str) -> str:
    import yaml
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _rules(pod: dict) -> list:
    return [e for t in pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
            ["nodeSelectorTerms"] for e in t["matchExpressions"] if e["key"] != "kubernetes.io/hostname"]


def test_one_job_per_init_reference_twin_first_then_init_bn_on_init_s_splits():
    import yaml
    bx = _bx()
    bl = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")
    specs = bx.build_init_bn()
    refs = bx.v2_init_refs()
    assert sorted(specs) == sorted(f"job-init-bn-{r.removeprefix('init-')}-raunav.yaml" for r, _ in refs)
    for ref, run in refs:
        text = specs[f"job-init-bn-{ref.removeprefix('init-')}-raunav.yaml"]
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav") and d["spec"]["backoffLimit"] == 1
        pod = d["spec"]["template"]["spec"]
        assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
        assert {v["persistentVolumeClaim"]["claimName"] for v in pod["volumes"]
                if "persistentVolumeClaim" in v} == {"tn-pvc-base-jetclass2", "transfer-learning-vol"}
        launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
        assert pod["containers"][0]["image"] == launch.V2_IMAGE          # the grid's, by digest
        body = _body(text)
        assert f"pip install --no-cache-dir -q {launch.V2_PYARROW}\n" in body
        assert f'--branch "{bx.INIT_BN_PIN}"' in body and bx.INIT_BN_PIN == "mtx-s2.20"
        # the twin, written once, into the run the reference's init_trunk.pt is from
        twin = "python3 experiments/MTX/bn_twins_v2.py --out ${RUN_DIR} --init"
        assert f"RUN_DIR={bx.V2_ROOT}/{run}\n" in body and "if [ ! -f ${RUN_DIR}/init_bn.json ]" in body
        assert body.count(twin) == 1 and body.index(twin) < body.index("python3 experiments/EVAL/extract_v2.py")
        assert "sha256sum -c ${RUN_DIR}/reweight_sidecar.sha256" in body
        # then init_bn only, beside init, for every split linprobe-x-<ref> extracted at init
        assert "--checkpoints init_bn --data-config" in body and "--checkpoints init " not in body
        assert f"OUT={bl.LP_ROOT}/{ref}/$1/$2;" in body
        assert f"--run-dir {bx.V2_ROOT}/{run} --rung none --num-classes 0 --num-reg 0" in body
        x = (bl.OUT_DIR / f"job-linprobe-x-{ref}-raunav.yaml").read_text()
        calls = lambda b: [ln for ln in b.splitlines() if ln.startswith("extract ")]
        assert calls(body) == calls(_body(x)) and calls(body)
        # the same region and GPU rule as that extraction (its node exclusions are older)
        assert _rules(pod) == _rules(yaml.safe_load(x)["spec"]["template"]["spec"])
        assert {"key": "nvidia.com/gpu.product", "operator": "NotIn",
                "values": list(bl.NOT_ON)} in _rules(pod)


def test_the_storage_guard_comes_before_the_twin_writes():
    from test_storage_guard import guarded_before_first_write
    for name, text in _bx().build_init_bn().items():
        assert guarded_before_first_write(text) is True, name


def test_the_committed_specs_are_the_builders():
    bx = _bx()
    for name, text in bx.build_init_bn().items():
        assert (bx.INIT_BN_DIR / name).read_text() == text, name
    assert sorted(p.name for p in bx.INIT_BN_DIR.glob("*.yaml")) == sorted(bx.build_init_bn())
