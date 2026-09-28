"""The frozen MLP probe trains to a plateau, and its MLP-only re-run.

The MLP used to stop at a hard 60 epochs, and 93 of the ladder's 360 fits were
still improving when they hit it -- mostly on the 17-class models, which is
where the label sets are compared. It now stops on a validation plateau after
three learning-rate cuts (probe.MLP_SCHEDULE), and --mlp-rerun-of re-fits only
the MLP of an earlier output, copying every linear number from it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "probe_mlp2", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _radial(n=12000, d=8, scale=1.2, seed=0):
    """Two centred Gaussians of different width: the optimal discriminant is
    |x|^2 and a linear probe cannot do better than chance, so reaching the
    optimum is the MLP's own doing. Five optimiser steps per epoch, like the
    smallest real task (visible_content, ~17 k training jets, batch 4096)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(np.int64)
    X = rng.normal(size=(2 * n, d)).astype(np.float32)
    X[y == 1] *= scale
    return X, y


def test_the_mlp_stops_on_a_plateau_and_reaches_the_optimum(probe):
    S = probe.MLP_SCHEDULE
    # the stopping rule outlasts three learning-rate cuts (1e-3 -> 1e-4 -> 1e-5)
    assert S["stop_patience"] == 3 * (S["lr_patience"] + 1)
    assert S["max_epochs"] >= 500
    X, y = _radial()
    tr, va, te = probe.make_splits(y.size)
    s, meta = probe.fit_mlp(X[tr], y[tr], X[va], y[va], X[te])
    for m in meta["seeds"]:
        assert m["converged"] is True, m["epochs_run"]
        assert m["epochs_run"] < S["max_epochs"]
        assert m["epochs_run"] == m["best_epoch"] + S["stop_patience"]
        # the curve is the audit trail: one entry per epoch, the kept state on it
        assert len(m["val_auc_curve"]) == m["epochs_run"]
        assert m["val_auc_curve"][m["best_epoch"] - 1] == pytest.approx(m["val_auc"], abs=1e-9)
    assert meta["all_converged"] is True
    oracle = roc_auc_score(y[te], (X[te] ** 2).sum(1))
    lin, _ = probe.fit_linear(X[tr], y[tr], X[va], y[va], X[te])
    assert roc_auc_score(y[te], lin) < 0.55
    assert roc_auc_score(y[te], s) > oracle - 0.01, (roc_auc_score(y[te], s), oracle)


def test_reaching_the_cap_is_recorded_as_not_converged(probe, monkeypatch):
    """The cap is a safety net. A fit that reaches it has not been shown to
    plateau and must say so, or D6's reading of the MLP rests on a fiction."""
    monkeypatch.setitem(probe.MLP_SCHEDULE, "max_epochs", 3)
    X, y = _radial(n=3000)
    tr, va, te = probe.make_splits(y.size)
    _, meta = probe.fit_mlp(X[tr], y[tr], X[va], y[va], X[te])
    assert all(m["converged"] is False and m["epochs_run"] == 3 for m in meta["seeds"])
    assert meta["all_converged"] is False


# --- the MLP-only re-run ------------------------------------------------------

def _arm(d: pathlib.Path, shift: float, tag: str) -> str:
    """bvc_resonant (native 0 vs 1), both classes above MIN_PER_CLASS in the test
    split. The two arms share labels (row-aligned) and differ in features."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(shift * 10))
    L = np.repeat([0, 1], 6000).astype(np.int16)
    X = rng.normal(size=(L.size, 8)).astype(np.float32)
    X[L == 0, 0] += shift
    np.save(d / "features.npy", X)
    np.save(d / "label188.npy", L)
    (d / "extract_manifest.json").write_text(json.dumps({"checkpoint_sha256": tag * 64}))
    return str(d)


def _main(probe, argv):
    old = sys.argv
    sys.argv = ["probe.py"] + argv
    try:
        return probe.main()
    finally:
        sys.argv = old


@pytest.fixture(scope="module")
def reruns(probe, tmp_path_factory):
    """A full run, its output with one linear number replaced by a sentinel (so
    a copy can be told from a re-fit), and the MLP-only re-run of that."""
    tmp = tmp_path_factory.mktemp("mlp2")
    feats = ["--features", f"A={_arm(tmp / 'A', 1.5, 'a')}", f"B={_arm(tmp / 'B', 2.0, 'b')}"]
    common = feats + ["--bootstrap", "20"]
    assert _main(probe, common + ["--out", str(tmp / "full"), "--tasks", "bvc_resonant",
                                  "--eps-s", "0.5", "0.9"]) == 0
    full = json.loads((tmp / "full" / "probe_results.json").read_text())
    src = json.loads(json.dumps(full))
    src["tasks"]["bvc_resonant"]["arms"]["A"]["linear"]["auc"] = 0.123
    src["tasks"]["bvc_resonant"]["contrasts"]["linear:A-B"]["ci95"] = [-9.0, 9.0]
    src.pop("mlp_training")           # as an output of the old code would lack it
    (tmp / "src").mkdir()
    src_path = tmp / "src" / "probe_results.json"
    src_path.write_text(json.dumps(src, indent=2))
    before = src_path.read_bytes()
    assert _main(probe, common + ["--out", str(tmp / "mlp2"), "--mlp-rerun-of", str(src_path)]) == 0
    return {"tmp": tmp, "common": common, "full": full, "src": src, "src_path": src_path,
            "src_bytes": before,
            "rerun": json.loads((tmp / "mlp2" / "probe_results.json").read_text())}


def test_the_rerun_copies_every_linear_number_and_leaves_the_source_alone(reruns):
    src, out = reruns["src"], reruns["rerun"]
    assert reruns["src_path"].read_bytes() == reruns["src_bytes"]
    t_src, t_out = src["tasks"]["bvc_resonant"], out["tasks"]["bvc_resonant"]
    for arm in ("A", "B"):
        assert t_out["arms"][arm]["linear"] == t_src["arms"][arm]["linear"], arm
    assert t_out["arms"]["A"]["linear"]["auc"] == 0.123, "linear was re-fitted, not copied"
    assert t_out["contrasts"]["linear:A-B"] == t_src["contrasts"]["linear:A-B"]
    assert out["mlp_rerun_of"] == {"path": str(reruns["src_path"]),
                                   "sha256": hashlib.sha256(reruns["src_bytes"]).hexdigest()}


def test_the_rerun_is_the_source_with_only_its_mlp_replaced(reruns):
    """Same schema, so every reader takes it by a change of path: every field
    but the MLP blocks, the MLP contrasts and the two provenance keys is the
    source's, byte for byte once parsed."""
    src, out = reruns["src"], json.loads(json.dumps(reruns["rerun"]))
    assert out.pop("mlp_training") == reruns["full"]["mlp_training"]
    out.pop("mlp_rerun_of")
    s, o = json.loads(json.dumps(src)), out
    for doc in (s, o):
        t = doc["tasks"]["bvc_resonant"]
        for arm in t["arms"].values():
            arm.pop("mlp")
        t["contrasts"] = {k: v for k, v in t["contrasts"].items() if k.startswith("linear:")}
    assert o == s
    # and the MLP blocks carry the full schema plus the convergence record
    m_src = src["tasks"]["bvc_resonant"]["arms"]["A"]["mlp"]
    m_out = reruns["rerun"]["tasks"]["bvc_resonant"]["arms"]["A"]["mlp"]
    assert set(m_out) == set(m_src)
    assert list(reruns["rerun"]["tasks"]["bvc_resonant"]["contrasts"]) == \
        list(src["tasks"]["bvc_resonant"]["contrasts"])
    for sd in m_out["selection"]["seeds"]:
        assert {"val_auc_curve", "best_epoch", "converged", "epochs_run"} <= set(sd)


def test_the_rerun_mlp_is_what_a_full_run_measures(reruns):
    """The re-fit is the same measurement a full run of this code makes -- same
    split, seeds and threads -- so the MLP numbers and their contrasts agree
    exactly with the unedited full run."""
    full, out = reruns["full"]["tasks"]["bvc_resonant"], reruns["rerun"]["tasks"]["bvc_resonant"]
    for arm in ("A", "B"):
        assert out["arms"][arm]["mlp"] == full["arms"][arm]["mlp"], arm
    assert out["contrasts"]["mlp:A-B"] == full["contrasts"]["mlp:A-B"]


@pytest.mark.parametrize("edit, argv, message", [
    (lambda d: d["tasks"]["bvc_resonant"].update(n_signal_test=1), [], "is not the task"),
    (lambda d: d.update(row_alignment_sha256="0" * 64), [], "row_alignment_sha256"),
    (lambda d: d["arm_checkpoints"].update(A="e" * 64), [], "arm_checkpoints"),
    (lambda d: None, ["--tasks", "bvc_resonant"], "takes its tasks"),
])
def test_the_rerun_refuses_a_source_it_cannot_copy_from(probe, reruns, edit, argv, message):
    """A linear number copied from a run on other jets or checkpoints would sit
    beside an MLP number it was never compared with, and nothing would error."""
    src = json.loads(json.dumps(reruns["src"]))
    edit(src)
    p = reruns["tmp"] / "bad" / "probe_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(src))
    with pytest.raises(SystemExit) as e:
        _main(probe, reruns["common"] + ["--out", str(reruns["tmp"] / "bad_out"),
                                         "--mlp-rerun-of", str(p)] + argv)
    assert message in str(e.value)


def test_the_rerun_refuses_to_overwrite_its_source(probe, reruns):
    with pytest.raises(SystemExit) as e:
        _main(probe, reruns["common"] + ["--out", str(reruns["src_path"].parent),
                                         "--mlp-rerun-of", str(reruns["src_path"])])
    assert "overwrite the source" in str(e.value)
    assert reruns["src_path"].read_bytes() == reruns["src_bytes"]
