"""experiments/FIGS/v2_rev35.py (2026-11-26 revision checks): the ROC point and its analytic term, the label-only
reference and the merge kinds of the per-class recall."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_rev35", ROOT / "experiments/FIGS/v2_rev35.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_eps_at_reads_the_stored_point_and_its_binomial_and_slope_terms():
    eb = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    rec = {"curve": {"eps_b": eb, "eps_s": [0.1, 0.3, 0.6, 0.9, 1.0]}, "n_s": 1000, "n_b": 100000}
    e, var = V.eps_at(rec, 0.01)
    slope = (0.9 - 0.3) / (1e-1 - 1e-3)
    assert e == 0.6 and var == pytest.approx(0.6 * 0.4 / 1000 + slope ** 2 * 0.01 * 0.99 / 100000)
    assert V.eps_at(rec, 0.001)[0] == 0.3
    with pytest.raises(SystemExit):
        V.eps_at(rec, 0.003)


def test_label_only_is_the_jets_share_over_their_groups_size(tmp_path):
    names = [r.split(",")[1] for r in (ROOT / "configs/labelmaps/rung_label_maps.v1.csv").read_text().splitlines()[1:]]
    one = {n.removeprefix("label_"): "g" for n in names}                 # a single group
    own = {n.removeprefix("label_"): n for n in names}                    # every class its own group
    counts = [5000] * len(names)
    for arm, m, want in (("L162", own, 100.0), ("R42_Q1", one, 100 / len(names))):
        out, _ = V.label_only(ROOT, {arm: m, "L162": own, "R42_Q1": one, "R16_Q1": one, "RAND2_p1": one}, counts)
        assert float(out[{"L162": "VLabelOnlyRes", "R42_Q1": "VLabelOnlyMid"}[arm]]) == pytest.approx(want, abs=0.05)
    out, _ = V.label_only(ROOT, {"L162": own, "R42_Q1": one, "R16_Q1": one, "RAND2_p1": one}, counts)
    assert out["VLabelOnlyRand"] == out["VLabelOnlyCoarse"]


def test_a_network_is_the_mean_of_its_head_trainings(tmp_path):
    import json
    eb = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    heads = [{"curve": {"eps_b": eb, "eps_s": [0.1, 0.3, s, 0.9, 1.0]}, "n_s": 1000, "n_b": 100000} for s in (0.5, 0.7)]
    (tmp_path / "xbx" / "xbb").mkdir(parents=True)
    (tmp_path / "xbx" / "xbb" / "m.json").write_text(json.dumps({"signals": {"xbb": {"head": heads}}}))
    e, var = V.head_side(tmp_path, "xbb", "m", 0.01)
    assert e == pytest.approx(0.6) and var == pytest.approx((V.eps_at(heads[0], 0.01)[1] + V.eps_at(heads[1], 0.01)[1]) / 2)
