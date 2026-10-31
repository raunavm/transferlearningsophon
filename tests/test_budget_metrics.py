"""Read-out of the fixed-budget fine-tuning cells (experiments/FT/budget_metrics.py)."""
import importlib.util
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_metrics", ROOT / "experiments/FT/budget_metrics.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_jc2b_cell_scores_every_jet_with_native_metrics(tmp_path):
    r = np.random.default_rng(0)
    y = np.repeat(np.arange(188), 30)
    z = r.normal(size=(y.size, 188)) + 4.0 * np.eye(188)[y]
    fd = tmp_path / "features"
    fd.mkdir()
    np.save(fd / "logits.npy", z.astype(np.float32))
    np.save(fd / "label188.npy", y.astype(np.int16))
    lp = bm._load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
    m = bm.jc2b_metrics(tmp_path, lp)
    assert m["n_jets"] == y.size and m["n_jets_auc"] == y.size and m["auc_stride"] == 1
    assert m["n_classes_present"] == 188 and m["macro_auc_ovr"] > 0.95
    assert lp.AUC_STRIDE == 4                     # restored after the read-out


def test_missing_cells_are_listed_not_filled(tmp_path):
    cell = tmp_path / "leg_top" / "l188-s1" / "N1000" / "s1"
    cell.mkdir(parents=True)
    out = tmp_path / "m.json"
    assert bm.main(["--root", str(tmp_path), "--out", str(out), "--trees", "leg_top"]) == 0
    doc = json.loads(out.read_text())
    assert doc["cells"] == {} and doc["missing"] == ["leg_top/l188-s1/N1000/s1"]
