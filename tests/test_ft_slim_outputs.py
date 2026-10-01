"""experiments/FT/slim_outputs.py: the v2 cells keep only what their readers open
(storage budget, 2026-10-01), and the readers give the same numbers from it.

  * leg 1: the logits of every --auc-stride-th row and every row's argmax give
    leg1_metrics.py's accuracy and macro AUC exactly as the full cache does;
  * leg 2: pred.root keeps its label_ and score_label_ branches bit for bit and
    leg2_metrics.py reads the same accuracy and AUC from it.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SLIM = _load("slim_outputs", "experiments/FT/slim_outputs.py")


def test_a_slim_leg1_cache_gives_the_full_caches_metrics(tmp_path):
    l1 = _load("leg1_metrics_slim", "experiments/FT/leg1_metrics.py")
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    eval_arm = _load("eval_arm", "experiments/EVAL/eval_arm.py")
    l162 = lr.rung_maps()["L162"]
    rng = np.random.default_rng(1)
    lab = rng.choice(sorted(l162), size=4003).astype(np.int16)   # not a multiple of the stride
    logits = (rng.normal(size=(lab.size, 162)) + 3 * np.eye(162)[[l162[int(x)] for x in lab]]).astype(np.float32)
    full, slim = tmp_path / "full", tmp_path / "slim"
    for d in (full, slim):
        d.mkdir()
        np.save(d / "label188.npy", lab)
        np.save(d / "logits.npy", logits)
    rec = SLIM.slim_leg1(slim, 4)
    assert not (slim / "logits.npy").exists() and rec["n_rows"] == lab.size
    a = l1.cell_metrics(full, l162, eval_arm, 4)
    b = l1.cell_metrics(slim, l162, eval_arm, 4)
    assert a == b
    with pytest.raises(SystemExit, match="every 4th row, not every 2th"):
        l1.cell_metrics(slim, l162, eval_arm, 2)
    sizes = {p.name: p.stat().st_size for p in slim.iterdir()}
    assert sizes["logits_auc.npy"] < (logits.nbytes / 4) + 1024 and sizes["argmax.npy"] < lab.size * 2 + 1024


def test_a_slim_pred_root_keeps_what_leg2_reads_bit_for_bit(tmp_path):
    uproot = pytest.importorskip("uproot")
    l2 = _load("leg2_metrics_slim", "experiments/FT/leg2_metrics.py")
    rng = np.random.default_rng(2)
    names = ["QCD", "Hbb", "Tbqq"]
    truth = rng.integers(0, 3, 5000)
    scores = rng.dirichlet(np.ones(3), 5000).astype(np.float32)
    branches = {**{f"label_{n}": truth == i for i, n in enumerate(names)},
                **{f"score_label_{n}": scores[:, i] for i, n in enumerate(names)},
                "_label_": truth.astype(np.int64), "jet_pt": rng.random(5000).astype(np.float32)}
    path = tmp_path / "pred.root"
    try:
        with uproot.recreate(path) as f:
            f["Events"] = branches
    except AttributeError as exc:     # uproot 5.1 cannot write under numpy 2 (np.VisibleDeprecationWarning)
        pytest.skip(f"this uproot cannot write ROOT files here: {exc}")
    before = l2.read_pred_root(path)
    size = path.stat().st_size
    kept = SLIM.slim_pred(path)
    assert sorted(kept) == sorted(k for k in branches if k.startswith(("label_", "score_label_")))
    after = l2.read_pred_root(path)
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and np.array_equal(before[2], after[2])
    assert path.stat().st_size < size and not list(tmp_path.glob("*.slim"))
