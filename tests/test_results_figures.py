"""The results figures (fine-tuning, anomaly detection, mass, real data) are
drawn from the committed analysis files end to end, and the per-seed points a
figure draws reproduce the means the analysis stored."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_results_figures",
                                              REPO / "experiments/FIGS/make_results_figures.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_all_four_figures_are_written_from_the_committed_analyses(tmp_path):
    assert R.main(["--outdir", str(tmp_path)]) == 0
    for stem in ("finetune_curves", "anomaly_sensitivity", "mass_tradeoff", "realdata_top_yield"):
        for ext in ("pdf", "png"):
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 5000, f"{stem}.{ext}"


def test_the_two_by_two_points_are_the_ones_the_c5_test_used():
    """Five seeds per corner, and the with-minus-without gains they imply equal
    the gains stored beside C5 in the confirmatory analysis."""
    pts = R.mass2x2_points(R.INPUTS["mass2x2"])
    assert all(len(v) == 5 for v in pts.values())
    A = json.loads((REPO / "experiments/FIGS/data/probe_ladder_v2/analysis_family_of_four/"
                    "seed_level_results.json").read_text())
    gains = A["confirmatory"]["C5"]["confirmatory"]["probes"]["linear"]["gain_by_level"]
    for lv, g in (("162", "162+mass"), ("17", "17+mass")):
        assert np.mean(pts[g]) - np.mean(pts[lv]) == pytest.approx(gains[lv]["mean_diff"], abs=1e-9)
