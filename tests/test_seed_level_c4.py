"""C4 in experiments/STATS/seed_level.py: the random-label draws against the
17-class model of the same seed index, from probe.py-shaped files."""
import contextlib
import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level", REPO / "experiments/STATS/seed_level.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def probe_file(tmp_path, draw, seed, control, semantic, sha="ab" * 32):
    """control / semantic: {task: log(1-AUC)}; the stored contrast is semantic - control."""
    rand, sem = f"rand-d{draw}-s{seed}", f"r16q1-s{seed}"
    tasks = {}
    for t in S.CONTROL_TASKS:
        arms = {a: {k: {"log1m_auc": v[t], "log1m_auc_censored": False, "auc": 0.99} for k in S.PROBES}
                for a, v in ((rand, control), (sem, semantic))}
        d = semantic[t] - control[t]
        tasks[t] = {"arms": arms, "contrasts": {f"{k}:{sem}-{rand}": {"delta_log1m_auc": d,
                                                                        "ci95": [d - 0.1, d + 0.1]}
                                                for k in S.PROBES}}
    p = tmp_path / f"sd{draw}.json"
    p.write_text(json.dumps({"row_alignment_sha256": sha, "tasks": tasks}))
    return p


def test_the_difference_is_control_minus_semantic_and_the_interval_follows_it(tmp_path):
    ps = [probe_file(tmp_path, k, k, {"bvc_4prong": -4.5, "visible_content": -6.8},
                     {"bvc_4prong": -3.5, "visible_content": -6.0}) for k in (1, 2, 3)]
    d = S.load_random_control(ps)
    assert d["diffs"]["linear"]["bvc_4prong"] == [-1.0] * 3
    lo, hi = d["ci"]["linear"]["bvc_4prong"][0]
    assert lo < -1.0 < hi < 0
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main(["--random-control", *map(str, ps), "--out", str(tmp_path / "o")]) == 0
    r = json.loads((tmp_path / "o" / "c4_random_control.json").read_text())
    assert r["C4"]["linear"]["n_match"] == 2          # bvc draws 1, 2 match; visible 1, 3 do not
    assert all(x["control_better"] for x in r["section7_check"]["linear"])


def test_a_draw_paired_with_the_wrong_seed_is_refused(tmp_path):
    c, s = {"bvc_4prong": -4.5, "visible_content": -6.8}, {"bvc_4prong": -3.5, "visible_content": -6.0}
    ps = [probe_file(tmp_path, 1, 1, c, s), probe_file(tmp_path, 2, 3, c, s), probe_file(tmp_path, 3, 3, c, s)]
    with pytest.raises(SystemExit, match="pairs draw k"):
        S.load_random_control(ps)


def ladder_file(tmp_path, seed, fine, semantic):
    """A four-level probe file for one seed index: three finer models at `fine`
    (per task) and the 17-class model at `semantic`."""
    arms = {f"l188-s{seed}": fine, f"l162-s{seed}": fine, f"r42q1-s{seed}": fine,
            f"r16q1-s{seed}": semantic}
    tasks = {t: {"arms": {a: {k: {"log1m_auc": v[t]} for k in S.PROBES} for a, v in arms.items()}}
             for t in S.CONTROL_TASKS}
    p = tmp_path / f"s{seed}.json"
    p.write_text(json.dumps({"row_alignment_sha256": "ab" * 32, "tasks": tasks}))
    return p


def test_the_post_hoc_grouping_cost_is_against_the_three_finer_models_of_the_same_seed(tmp_path):
    c, s = {"bvc_4prong": -4.5, "visible_content": -6.8}, {"bvc_4prong": -3.5, "visible_content": -6.0}
    fine = {"bvc_4prong": -5.0, "visible_content": -7.0}
    ps = [probe_file(tmp_path, k, k, c, s) for k in (1, 2, 3)]
    ls = [ladder_file(tmp_path, k, fine, s) for k in (1, 2, 3)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main(["--random-control", *map(str, ps), "--c4-fine-ladder", *map(str, ls),
                       "--out", str(tmp_path / "o")]) == 0
    g = json.loads((tmp_path / "o" / "c4_random_control.json").read_text())["grouping_cost_post_hoc"]
    b = g["tasks"]["bvc_4prong"]["linear"]
    assert b["mean_control_minus_fine"] == pytest.approx(0.5)
    assert b["mean_semantic_minus_fine"] == pytest.approx(1.5)
    assert b["factor_semantic"] == pytest.approx(np.exp(1.5))
    assert "POST HOC" in g["reading"]


def test_the_grouping_cost_refuses_ladder_features_that_differ_from_the_draw_file(tmp_path):
    c, s = {"bvc_4prong": -4.5, "visible_content": -6.8}, {"bvc_4prong": -3.5, "visible_content": -6.0}
    ps = [probe_file(tmp_path, k, k, c, s) for k in (1, 2, 3)]
    other = {"bvc_4prong": -3.4, "visible_content": -6.0}
    ls = [ladder_file(tmp_path, k, {"bvc_4prong": -5.0, "visible_content": -7.0}, other)
          for k in (1, 2, 3)]
    rows = S.load_random_control(ps)["rows"]
    with pytest.raises(SystemExit, match="not the same features"):
        S.c4_grouping_cost(rows, ls)
