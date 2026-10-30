"""The v2 zero-fill masking control jobs scripts/build_fill_control_v2.py emits: the masked
JetClass-II splits extracted as the linear probes' are, and the 188-class probe fitted on them."""
from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


FC = _load("build_fill_control_v2", "scripts/build_fill_control_v2.py")
SPECS = FC.build()


def _args(spec: str) -> str:
    return yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_one_job_per_hierarchy_run_and_untrained_trunk():
    ms = FC.models()
    assert [m for m, *_ in ms] == list(FC.HIERARCHY) + ["init-s1", "init-s2", "init-s3"]
    assert all(c == ("init" if m.startswith("init-") else "best70") for m, *_, c in ms)
    assert sorted(SPECS) == sorted(f"job-fillctl-{m.removeprefix('mtx-')}-raunav.yaml" for m, *_ in ms)
    for name, spec in SPECS.items():
        d = yaml.safe_load(spec)
        assert d["metadata"]["name"].endswith("-raunav") and name == f"job-{d['metadata']['name']}.yaml"
        a = _args(spec)
        assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0, name
        assert f'--branch "{FC.PIN}"' in a and '[ "${USED}" -lt 85 ]' in a
        assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
        ex = {e["key"]: e for e in d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]}
        assert ex["nvidia.com/gpu.product"]["values"] == list(FC.LP.NOT_ON)
        assert set(FC.BX.V2_GPU_FAULT_NODES) <= set(ex["kubernetes.io/hostname"]["values"])
        assert (FC.OUT_DIR / name).read_text() == spec, f"{name} not written as built"


def test_each_mask_extracts_the_three_splits_and_fits_the_188_class_probe():
    for model, run, rung, k, reg, ckpt in FC.models():
        a = _args(SPECS[f"job-fillctl-{model.removeprefix('mtx-')}-raunav.yaml"])
        # the linear-probe extraction's own function: every row kept, class token only
        assert f"OUT={FC.FC_ROOT}/${{MASK}}/{model}/$1/$2" in a
        assert f"--run-dir {FC.BX.V2_ROOT}/{run} --rung {rung} --num-classes {k} --num-reg {reg}" in a
        assert f"--checkpoints {ckpt} " in a and "--prefix-features 2000000000" in a and "--no-pooled" in a
        assert a.count("python3 experiments/EVAL/extract_v2.py") == 1
        blocks = re.split(r"^          MASK=", a, flags=re.M)[1:]
        assert [b.split("\n")[0] for b in blocks] == list(FC.MASKS)
        for m, b in zip(FC.MASKS, blocks):
            got = re.findall(r"^          extract (\w+) (\w+) (\S+) (\d+) (.+)$", b, re.M)
            ft = f"configs/finetune/JetClassII_L162_mask{m}.yaml"
            assert got == [("jc2", "train_N1000000", ft, "0", "/data/finetune/jc2_v2/train_N1000000_s1.parquet"),
                           ("jc2", "val", ft, "0", "/data/finetune/jc2_v2/val.parquet"),
                           ("jc2", "test", f"configs/finetune/JetClassII_base_mask{m}.yaml", "2000000",
                            FC.BF.test2m_list())]
            assert (f"linear_probe_v2.py --root {FC.FC_ROOT}/{m} --model {model} \\\n"
                    f"            --checkpoints {ckpt} --datasets jc2_188 --jc2-parquet /data/finetune/jc2_v2 \\\n"
                    f"            --out {FC.FC_ROOT}/{m}/fits || halt") in b


def test_the_test_split_is_the_unmasked_probes_test_split():
    maxj, cfg, files = FC.LP.DATASETS["jc2"][2]["test"]
    assert (maxj, files) == (2_000_000, FC.BF.test2m_list()) and cfg == "configs/data/JetClassII_base.yaml"


def test_the_mask_configs_are_current_and_carry_no_weights_block():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_downstream_configs.py"), "--check-only"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for path, text in FC.NEEDED.items():
        body = (ROOT / path).read_text()
        assert text in body, path
        if path.startswith("configs/"):
            assert not re.search(r"^weights:", body, re.M), path


def test_the_storage_plan():
    want = 15 * 2 * (1_000_000 + 20_480 + 2_000_512) * (128 * 2 + 2 + 8)
    assert FC.planned_bytes() == want
