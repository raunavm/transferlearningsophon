"""experiments/EVAL/tf32_check.py and its job spec, without a GPU."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


tc = _load("tf32_check", "experiments/EVAL/tf32_check.py")


def test_set_tf32_sets_both_flags():
    import torch
    was = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        tc.set_tf32(True, False)
        assert torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        tc.set_tf32(False, True)
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        tc.set_tf32(*was)


def test_scores_are_the_analyses_own_and_compare_measures_the_shift():
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    disc = _load("discriminants", "experiments/AOJ/discriminants.py")
    z = np.random.default_rng(0).normal(size=(50, 17))
    s = tc.scores(z, "R16_Q1")
    assert np.allclose(s["three_prong_logodds"], disc.contrast(z, "R16_Q1", "three_prong"))
    assert np.allclose(s["logodds_res_qcd"], xv.head_score_columns(
        z.astype(np.float32), "R16_Q1", {})["logodds_res_qcd"], atol=1e-5)
    s2 = tc.scores(z + np.where(np.arange(17) == 16, 0.1, 0.0), "R16_Q1")   # one logit moved
    c = tc.compare(s2, s)
    assert c["logits"]["max_abs"] == np.float64(0.1).item() or abs(c["logits"]["max_abs"] - 0.1) < 1e-12
    assert c["three_prong_logodds"]["max_abs"] > 0 and c["argmax_flips"] >= 0
    assert tc.compare(s, s)["three_prong_logodds"]["max_abs"] == 0.0


def test_the_check_runs_on_a_tf32_gpu_with_the_gpu_fault_rule():
    import yaml
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    fname, text = bx.build_tf32_check()
    d = yaml.safe_load(text)
    assert d["metadata"]["name"] == "eval-tf32-check-raunav" and fname == "job-eval-tf32-check-raunav.yaml"
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    prod = [t for t in terms if t["key"] == "nvidia.com/gpu.product"][0]
    assert prod["operator"] == "In" and not any("V100" in p or "2080" in p or "1080" in p
                                                 for p in prod["values"])
    host = [t for t in terms if t["key"] == "kubernetes.io/hostname"][0]
    assert set(host["values"]) == set(bx.GPU_FAULT_NODES)
    assert bx.HALT_GPU in text and "gpu_ok ()" in text and f'--branch "{bx.TF32_PIN}"' in text
    assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
    assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"
