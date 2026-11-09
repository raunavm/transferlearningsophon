"""A kind split over NRP's two GPU pools (PI 2026-11-09, "go ahead with what you believe is best"): the A100 part asks
for nvidia.com/a100 and names no gpu.product, the other part is the usual spec, and the two parts hold disjoint cells
that together are the whole kind, so no cell is ever queued in two jobs."""
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_ft_budget_jobs", ROOT / "scripts/build_ft_budget_jobs.py")
bfb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfb)
ARGS = ["--tag", "mtx-s2.87", "--lr-pre", "1e-3", "--lr-scratch", "1e-3", "--storage-line", "90", "--cpu", "3",
        "--rerun", "top:10000"]


def _cases(text):
    return {m[2] for m in re.finditer(r"^\s+(\d+)\) (TASK=.*;;)$", text, re.M)}


def _res(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["resources"]


def test_the_two_parts_split_the_cells_and_only_the_pool_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    n = len(bfb.pre_full_rerun_cells("1e-3", [("top", 10000)]))
    a100 = ",".join(str(i) for i in range(0, n, 2))
    gpu = ",".join(str(i) for i in range(1, n, 2))
    bfb.main(ARGS + ["--mem", "24Gi", "--parallelism", "6", "--gpu-pool", "a100", "--indices", a100,
                     "--part", "top4-a100", "pre-full-rerun"])
    bfb.main(ARGS + ["--mem", "24Gi", "--parallelism", "8", "--indices", gpu, "--part", "top4", "pre-full-rerun"])
    bfb.main(ARGS + ["--mem", "24Gi", "pre-full-rerun"])
    ta = (tmp_path / "job-ft-budget-pre-full-rerun-top4-a100-raunav.yaml").read_text()
    tg = (tmp_path / "job-ft-budget-pre-full-rerun-top4-raunav.yaml").read_text()
    whole = (tmp_path / "job-ft-budget-pre-full-rerun-raunav.yaml").read_text()
    assert _cases(ta).isdisjoint(_cases(tg)) and _cases(ta) | _cases(tg) == _cases(whole)
    assert _res(ta)["requests"]["nvidia.com/a100"] == _res(ta)["limits"]["nvidia.com/a100"] == "1"
    assert "nvidia.com/gpu" not in _res(ta)["requests"] and "gpu.product" not in ta
    assert _res(tg)["requests"]["nvidia.com/gpu"] == "1" and "gpu.product" in tg
    for t in (ta, tg):
        assert "us-west" in t and "ry-gpu-04.sdsc.optiputer.net" in t and _res(t)["requests"]["memory"] == "24Gi"
        assert 'value: "mtx-s2.87"' in t and '-le 90 ]' in t
    strip = lambda t: re.sub(r"^\s+\d+\) TASK=.*;;\n", "", t, flags=re.M)        # noqa: E731
    body = lambda t: yaml.safe_load(strip(t))["spec"]["template"]["spec"]["containers"][0]["args"][0]  # noqa: E731
    assert body(ta) == body(tg) == body(whole)
    with pytest.raises(SystemExit):
        bfb.spec("x-raunav", bfb.pre_full_cells("1e-3")[:1], "mtx-s2.87", 1, "24Gi", pool="h100")
