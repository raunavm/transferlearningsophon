"""seed_level.py --s8: the mass-output minus twin accuracy per seed index, paired t
per (label set, epoch) cell, Holm over the 12 cells, GPU-mismatched pairs dropped,
logged validation values shown only where a pair validated on the same jets."""
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

COMMON = dict(n_jets=100_000, stream_jets=2_000_000, stride=20, native_label_sha256="ab",
              data_config_sha256="cd")


def _write(tmp, delta=0.01, gpu=None, other_jets_seed=None):
    rng = np.random.default_rng(0)
    acc, log = {}, {}
    for s in range(1, 6):
        runs = {}
        for lset, (twin, mass, k) in SL.S8_SETS.items():
            base = {e: 0.4 + 0.002 * e + rng.normal(0, 0.003) for e in SL.S8_EPOCHS}
            for stem, reg, shift in ((twin, 0, 0.0), (mass, 1, delta)):
                run = SL._s8_run(stem, s)
                a = {e: base[e] + shift + rng.normal(0, 0.001) for e in SL.S8_EPOCHS}
                acc[run] = a
                runs[run] = {"num_classes": k, "num_reg": reg,
                             "epochs": {str(e): {"accuracy": v, "checkpoint_sha256": "x"} for e, v in a.items()}}
                log[run] = {"metric": {str(e): v + 0.05 for e, v in a.items()},
                            "val_state": {str(e): ["h", 1 if e == 0 else e + reg] for e in SL.S8_EPOCHS},
                            "gpu": (gpu or {}).get(run, ["NVIDIA-GeForce-RTX-3090"])}
        doc = dict(COMMON, runs=runs)
        if s == other_jets_seed:
            doc["stride"] = 10
        (tmp / f"s{s}.json").write_text(json.dumps(doc))
    (tmp / "val.json").write_text(json.dumps(log))
    return acc


def _run(tmp):
    return SL.main(["--s8",
                    *[str(tmp / f"s{s}.json") for s in range(1, 6)],
                    "--s8-validation-log", str(tmp / "val.json"), "--out", str(tmp / "out")])


def test_each_cell_is_the_paired_t_on_the_seed_differences_and_holm_runs_over_twelve(tmp_path):
    acc = _write(tmp_path)
    assert _run(tmp_path) == 0
    r = json.loads((tmp_path / "out/s8_mass_early_accuracy.json").read_text())["secondary"]["S8"]
    assert len(r["cells"]) == 12 and r["holm_table"] == "12 cells" and not r["dropped_pairs"]
    c = next(c for c in r["cells"] if c["label_set"] == "R16_Q1" and c["epoch"] == 10)
    d = [acc[SL._s8_run("r16q1mass", s)][9] - acc[SL._s8_run("r16q1", s)][9] for s in range(1, 6)]
    assert np.allclose(c["diffs"], d) and c["p"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)
    assert c["share_of_twin"] == pytest.approx(np.mean(d) / c["twin_mean"])
    assert c["holm_reject"] and c["early"]
    # only epoch 1 has the same validation state for both runs of a pair in this fixture
    assert all(len(c["logged_validation_same_jets"]) == (5 if c["epoch"] == 1 else 0) for c in r["cells"])


def test_a_pair_on_different_gpu_models_is_dropped(tmp_path):
    _write(tmp_path, gpu={"mtx-l162mass-s3": ["NVIDIA-A10"]})
    assert _run(tmp_path) == 0
    r = json.loads((tmp_path / "out/s8_mass_early_accuracy.json").read_text())["secondary"]["S8"]
    assert [(x["label_set"], x["seed"]) for x in r["dropped_pairs"]] == [("L162", 3)]
    assert all(c["seeds"] == ([1, 2, 4, 5] if c["label_set"] == "L162" else [1, 2, 3, 4, 5]) for c in r["cells"])


def test_files_scoring_other_jets_are_refused(tmp_path):
    _write(tmp_path, other_jets_seed=4)
    with pytest.raises(SystemExit, match="other jets"):
        _run(tmp_path)


def test_the_late_epoch_range_is_each_runs_spread_over_epochs_20_40_80(tmp_path):
    acc = _write(tmp_path)
    assert _run(tmp_path) == 0
    r = json.loads((tmp_path / "out/s8_mass_early_accuracy.json").read_text())["secondary"]["S8"]
    run = "mtx-r16q1mass-s4"
    v = [acc[run][e] for e in (19, 39, 79)]
    assert r["late_epoch_range"][run] == pytest.approx(max(v) - min(v))
    assert r["max_late_epoch_range"]["range"] == pytest.approx(max(r["late_epoch_range"].values()))


def test_the_scored_and_logged_accuracies_are_correlated_within_each_run(tmp_path):
    _write(tmp_path)          # the fixture logs the scored value + 0.05: perfectly correlated
    assert _run(tmp_path) == 0
    r = json.loads((tmp_path / "out/s8_mass_early_accuracy.json").read_text())["secondary"]["S8"]
    assert r["logged_agreement"]["within_run_correlation"] == pytest.approx(1.0)
    assert r["logged_agreement"]["n_points"] == 20 * 6
