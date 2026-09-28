"""seed_level.py --s10: 17 - 162 in log(1 - AUC) per seed pair on bc_vs_rest, paired t,
confirmed only if the linear difference is positive with p < 0.05; GPU-mismatched
pairs dropped."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level", ROOT / "experiments/STATS/seed_level.py")
SL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SL)


def _cell(l1m):
    r = {e: {"rejection": 50.0, "rejection_is_bound": False} for e in ("0.60", "0.40")}
    return {"auc": 1 - np.exp(l1m), "log1m_auc": l1m, "log1m_auc_censored": False, "rejection_at": r}


def _write(tmp, gap=0.3, gpu=None):
    rng = np.random.default_rng(1)
    arms, log = {}, {}
    for s in range(1, 6):
        for stem, shift in (("l162", 0.0), ("r16q1", gap)):
            a = SL.S8_RERUN.get((stem, s), f"{stem}-s{s}")
            v = -3.0 + shift + rng.normal(0, 0.05)
            arms[a] = {"linear": _cell(v), "mlp": _cell(v - 0.1)}
            log["mtx-" + a] = {"gpu": (gpu or {}).get(a, ["NVIDIA-GeForce-RTX-3090"])}
    doc = {"row_alignment_sha256": "ab", "tasks": {"bc_vs_rest": {"n_signal_test": 2000,
                                                                  "n_background_test": 9000, "arms": arms}}}
    (tmp / "probe.json").write_text(json.dumps(doc))
    (tmp / "gpu.json").write_text(json.dumps(log))
    return arms


def _run(tmp):
    assert SL.main(["--s10", str(tmp / "probe.json"), "--s10-gpu-log", str(tmp / "gpu.json"),
                    "--out", str(tmp / "out")]) == 0
    return json.loads((tmp / "out/s10_vcb.json").read_text())["secondary"]["S10"]


def test_the_difference_is_17_minus_162_per_seed_and_a_clear_gap_confirms(tmp_path):
    arms = _write(tmp_path)
    r = _run(tmp_path)
    d = [arms[SL.S8_RERUN.get(("r16q1", s), f"r16q1-s{s}")]["linear"]["log1m_auc"]
         - arms[SL.S8_RERUN.get(("l162", s), f"l162-s{s}")]["linear"]["log1m_auc"] for s in range(1, 6)]
    L = r["probes"]["linear"]
    assert np.allclose(L["diff_17_minus_162"], d) and L["p"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)
    assert r["verdict"] == "confirmed" and r["seeds"] == [1, 2, 3, 4, 5]


def test_a_gap_in_the_wrong_direction_is_not_confirmed(tmp_path):
    _write(tmp_path, gap=-0.3)
    assert _run(tmp_path)["verdict"] == "not confirmed"


def test_a_pair_on_different_gpu_models_is_dropped(tmp_path):
    _write(tmp_path, gpu={"r16q1-s2": ["NVIDIA-A10"]})
    r = _run(tmp_path)
    assert r["seeds"] == [1, 3, 4, 5] and r["dropped_pairs"][0]["seed"] == 2
