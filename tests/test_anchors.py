"""The anchor check is only worth running if its ROC arithmetic is right.

`experiments/EVAL/anchors.py` exists to decide whether OUR evaluation code
reproduces arXiv:2503.00118 Table A1 before any arm-wise rejection is quoted.
That makes its own arithmetic load-bearing: a rejection computed at a threshold
fixed the wrong way round (on background rather than signal efficiency) would
still produce plausible numbers, still "agree" or "disagree" for reasons that
look physical, and would silently validate or reject the evaluation code for
the wrong reason.

These tests pin the published targets against transcription drift and check the
rejection arithmetic against cases whose answer is known by construction.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "anchors", ROOT / "experiments" / "EVAL" / "anchors.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


an = _load()


def test_published_anchors_match_table_a1():
    """arXiv:2503.00118 Table A1, Sophon column. Transcribed once; pinned here."""
    assert an.PUBLISHED == {
        ("label_X_bb", 0.60): 300.0,
        ("label_X_bb", 0.40): 810.0,
        ("label_X_cc", 0.60): 110.0,
        ("label_X_cc", 0.40): 320.0,
    }


def test_rejection_threshold_is_fixed_on_signal_efficiency():
    """eps_s of the SIGNAL must pass, whatever the background looks like."""
    d_sig = np.linspace(0.0, 1.0, 10_001)
    d_bkg = np.linspace(0.0, 1.0, 10_001)
    for eps_s in (0.60, 0.40):
        rej, thr, eps_b = an.rejection(d_sig, d_bkg, eps_s)
        assert (d_sig >= thr).mean() == pytest.approx(eps_s, abs=0.002)
        # identical distributions -> eps_B == eps_S -> rejection == 1/eps_S
        assert rej == pytest.approx(1.0 / eps_s, rel=0.01)


def test_rejection_is_not_computed_on_background_efficiency():
    """Guard against the inverted definition: skew the background so the two
    conventions give very different answers, and check we get the signal one."""
    d_sig = np.linspace(0.5, 1.0, 10_000)     # signal concentrated high
    d_bkg = np.linspace(0.0, 0.5, 10_000)     # background concentrated low
    rej, thr, eps_b = an.rejection(d_sig, d_bkg, 0.60)
    assert thr == pytest.approx(0.70, abs=0.01), "threshold must come from the signal"
    assert eps_b == 0.0 and rej == float("inf")


def test_perfect_separation_gives_infinite_rejection():
    rej, _, eps_b = an.rejection(np.ones(1000), np.zeros(1000), 0.60)
    assert eps_b == 0.0 and rej == float("inf")


def test_known_rejection_is_recovered_exactly():
    """Background 1 % above the signal's 60 % point -> rejection exactly 100."""
    d_sig = np.linspace(0.0, 1.0, 100_001)         # 60 % point at D = 0.4
    d_bkg = np.concatenate([np.zeros(99_000), np.ones(1_000)])
    rej, thr, eps_b = an.rejection(d_sig, d_bkg, 0.60)
    assert thr == pytest.approx(0.4, abs=1e-3)
    assert eps_b == pytest.approx(0.01, rel=1e-9)
    assert rej == pytest.approx(100.0, rel=1e-9)


def test_end_to_end_on_synthetic_cache(tmp_path):
    """Whole path: cached arrays -> selection -> discriminant -> rejection.

    Signal nodes are given a large logit for signal jets and QCD nodes a large
    logit for QCD jets, so the discriminant separates perfectly and every
    rejection is infinite -- which the report must express, not crash on.
    """
    n, K = 4_000, 188
    rng = np.random.default_rng(0)
    e1_qcd = an._e1_control().qcd_indices()
    label = np.zeros(n, dtype=np.int16)
    label[:1_000] = 0                       # X_bb
    label[1_000:2_000] = 1                  # X_cc
    label[2_000:] = e1_qcd[0]               # QCD
    logits = rng.normal(0, 0.1, size=(n, K)).astype(np.float32)
    logits[np.arange(1_000), 0] += 20.0
    logits[np.arange(1_000, 2_000), 1] += 20.0
    logits[2_000:, e1_qcd[0]] += 20.0
    np.save(tmp_path / "logits.npy", logits)
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz",
             jet_pt=np.full(n, 500.0), jet_sdmass=np.full(n, 100.0))
    out = tmp_path / "anchors.json"
    # Perfect separation cannot match 300/810/110/320, so a non-zero exit is
    # the CORRECT outcome here; the point is that it ran and reported.
    rc = an.main(["--features", str(tmp_path), "--out", str(out)])
    d = json.loads(out.read_text())
    assert rc == 1 and d["all_agree"] is False
    assert len(d["anchors"]) == 4
    assert d["n_selected"] == n
    for r in d["anchors"]:
        assert r["rejection"] == float("inf") or r["rejection"] > 1e6
        assert r["n_signal"] == 1_000 and r["n_qcd"] == 2_000


def test_jets_that_are_neither_signal_nor_qcd_are_excluded(tmp_path):
    """A third class must not inflate the background denominator."""
    n, K = 3_000, 188
    e1_qcd = an._e1_control().qcd_indices()
    label = np.zeros(n, dtype=np.int16)
    label[:1_000] = 0
    label[1_000:1_500] = 1                   # X_cc, so the cc pass has signal
    label[1_500:2_000] = e1_qcd[0]
    label[2_000:] = 3                        # label_X_qq: neither
    logits = np.zeros((n, K), dtype=np.float32)
    logits[np.arange(1_000), 0] = 20.0
    logits[1_000:1_500, 1] = 20.0
    logits[1_500:2_000, e1_qcd[0]] = 20.0
    logits[2_000:, 3] = 20.0
    np.save(tmp_path / "logits.npy", logits)
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz",
             jet_pt=np.full(n, 500.0), jet_sdmass=np.full(n, 100.0))
    out = tmp_path / "a.json"
    an.main(["--features", str(tmp_path), "--out", str(out)])
    bb = [r for r in json.loads(out.read_text())["anchors"] if r["signal"] == "label_X_bb"]
    assert all(r["n_qcd"] == 500 for r in bb), "X_qq jets leaked into the QCD denominator"


def test_a_class_with_no_signal_jets_fails_loudly(tmp_path):
    """Empty side -> a named SystemExit, not numpy's bare IndexError."""
    n, K = 200, 188
    e1_qcd = an._e1_control().qcd_indices()
    label = np.full(n, e1_qcd[0], dtype=np.int16)   # QCD only: no X_bb at all
    np.save(tmp_path / "logits.npy", np.zeros((n, K), dtype=np.float32))
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz",
             jet_pt=np.full(n, 500.0), jet_sdmass=np.full(n, 100.0))
    with pytest.raises(SystemExit) as e:
        an.main(["--features", str(tmp_path), "--out", str(tmp_path / "a.json")])
    assert "cannot be defined" in str(e.value)
