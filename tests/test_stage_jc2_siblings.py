"""Every test jet of X->ee and X->mumu (experiments/FT/stage_jc2_siblings.py)."""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_jc2_siblings", ROOT / "experiments/FT/stage_jc2_siblings.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def test_the_labels_are_the_sibling_pairs_classes():
    names = {0: "label_X_bb", 10: "label_X_ee", 11: "label_X_mm", 161: "label_QCD_b"}
    assert ss.sibling_labels(names) == [10, 11]


def test_the_plan_takes_every_jet_of_the_labels_and_no_other():
    src = np.arange(1000, dtype=np.int64) * 7
    lab = np.random.default_rng(0).choice([0, 10, 11, 161], 1000)
    ids = ss.plan(src, lab, [10, 11])
    assert np.array_equal(ids, np.sort(src[np.isin(lab, [10, 11])]))
    assert ids.size == int(np.isin(lab, [10, 11]).sum())
