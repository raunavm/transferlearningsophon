"""The selected pretraining jets (experiments/FT/count_jc2_train.py)."""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("count_jc2_train", ROOT / "experiments/FT/count_jc2_train.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def test_the_census_covers_every_pretraining_file_and_sums_the_selected_jets():
    files = ct.SB.MS.jc2_role_files("pretrain_train")
    assert len(files) == 200 + 860 + 280
    res = [("/jc2/jet_data/Res2P_0000.parquet", np.arange(3), np.array([0, 0, 5], np.int16)),
           ("/jc2/jet_data/QCD_0001.parquet", np.arange(2), np.array([170, 171], np.int16))]
    s = ct.summary(res)
    assert s == {"n_files": 2, "n_selected": 5, "per_family": {"QCD": 2, "Res2P": 3},
                 "per_label": {"0": 2, "5": 1, "170": 1, "171": 1}}


def test_the_census_function_pickles_for_the_worker_pool():
    import pickle
    assert pickle.loads(pickle.dumps(ct.SB.census_file)) is ct.SB.census_file
