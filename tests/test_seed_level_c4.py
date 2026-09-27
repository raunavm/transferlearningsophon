"""C4 in experiments/STATS/seed_level.py: the random-label draws against the
17-class model of the same seed index, from probe.py-shaped files."""
import contextlib
import importlib.util
import io
import json
import pathlib

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
