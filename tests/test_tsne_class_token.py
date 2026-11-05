import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sklearn")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tsne_class_token", ROOT / "experiments/EVAL/tsne_class_token.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_choose_gives_the_same_jets_from_the_same_labels_and_one_group_count():
    y = np.repeat([0, 1, 3, 161, 170], 50)
    rows, which = ts.choose(y, ts.SETS["two_prong"], 20, 1)
    again, _ = ts.choose(y, ts.SETS["two_prong"], 20, 1)
    assert np.array_equal(rows, again)
    assert np.bincount(which).tolist() == [20, 20, 20, 20]
    assert set(y[rows[which == 3]]) <= {161, 170}
    with pytest.raises(SystemExit):
        ts.choose(y, ts.SETS["two_prong"], 60, 1)
