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
    A = json.loads((REPO / "experiments/FIGS/data/probe_ladder_v2_mlp2/analysis/"
                    "seed_level_results.json").read_text())
    gains = A["confirmatory"]["C5"]["confirmatory"]["probes"]["linear"]["gain_by_level"]
    for lv, g in (("162", "162+mass"), ("17", "17+mass")):
        assert np.mean(pts[g]) - np.mean(pts[lv]) == pytest.approx(gains[lv]["mean_diff"], abs=1e-9)


def test_the_finetune_curves_read_the_cells_the_s3_s4_analysis_read():
    """Each column reads the metrics file the analysis names for that task, and
    its per-seed values average (in the analysis's natural log) to the stored
    level means, five pretraining seeds per point: same cells, same seed alias."""
    F = json.loads((REPO / "experiments/FIGS/data/finetune_s3_s4/analysis_v2/"
                    "s3_s4_finetune.json").read_text())
    for key, title in (("S4", "JetClass-II, 162 classes"), ("S3", "JetClass, 10 classes")):
        S = F["secondary"][key]
        paths = R.INPUTS["finetune"][title]
        assert [str(p.relative_to(REPO)) for p in paths] == [S["provenance"]["file"]["path"]]
        by = R.ft_by_seed(R.ft_cells(paths))
        for n, block in S["per_size"].items():
            for row in block["levels"]:
                v = by[row["level"]][R.n_of(n)]
                assert len(v) == row["n_seeds"] == 5
                assert np.mean(np.log(list(v.values()))) == pytest.approx(row["mean"], abs=1e-12)


def test_no_figure_carries_an_interval_or_a_test_result(tmp_path, monkeypatch):
    """The spread in a figure is the seed standard deviation; intervals, p-values
    and test verdicts belong to the tables."""
    import matplotlib.text
    texts, real = [], R.save

    def spy(fig, outdir, stem):
        texts.extend(t.get_text() for t in fig.findobj(matplotlib.text.Text))
        real(fig, outdir, stem)

    monkeypatch.setattr(R, "save", spy)
    assert R.main(["--outdir", str(tmp_path)]) == 0
    words = ("95%", "interval", "reject", "holm", "p =", "p=", r"\ln(1-")
    assert not [t for t in texts if any(w in t.lower() for w in words)]
