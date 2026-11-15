"""The resonance-tagging claims hold under L3 with both test-sample terms (audit 2026-11-15, P10).

The claim sentences (gallery_placement XBX and XBX_PART): 17 classes keep fewer X->bb and X->bs jets than 188 at 1% QCD
efficiency; each flavor-boundary partition keeps fewer than 188; no 17-class partition differs from 17 classes by more
than 3 sigma. sigma^2 = run SD^2 of each side + the test-sample variance of each side's signal efficiency: the signal
jets (binomial, eps_S(1-eps_S)/n_S) and the QCD jets through the cut (|d eps_S/d eps_B| sqrt(eps_B(1-eps_B)/n_B), the
slope from the stored ROC curve around eps_B = 1%). The test terms of the two sides are added, which is conservative
because both sides score the same jets. Reads the pulled classifier fits, experiments/FIGS/data/v2/xbx/<signal>/.
"""
import json
import math
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "experiments/FIGS/data/v2/xbx"
SIGNALS = ("xbb", "xbs")
EPS_B = 0.01


def _cell(signal: str, model: str):
    """(eps_S at eps_B = 1% averaged over the head trainings, its test-sample SD) of one network, or None."""
    f = DATA / signal / f"{model}.json"
    if not f.exists():
        return None
    heads = json.loads(f.read_text())["signals"][signal]["head"]
    es, var = [], []
    for h in heads:
        c = h["curve"]
        eb, s = np.array(c["eps_b"]), np.array(c["eps_s"])
        i = int(np.argmin(abs(eb - EPS_B)))
        slope = (s[i + 1] - s[i - 1]) / (eb[i + 1] - eb[i - 1])
        e, n_s, n_b = h["eps_s_at_star"], h["n_s"], h["n_b"]
        es.append(e)
        var.append(e * (1 - e) / n_s + (slope ** 2) * EPS_B * (1 - EPS_B) / n_b)
    return float(np.mean(es)), math.sqrt(float(np.mean(var)))


def _side(signal: str, arms):
    """(mean over runs, run SD or 0 for one run, test SD) of the runs of the given arms."""
    runs = [m for a in arms for m in sorted(p.stem for p in (DATA / signal).glob(f"mtx-{a}-s*.json"))]
    cells = [_cell(signal, m) for m in runs]
    if not cells or None in cells:
        pytest.skip(f"{signal}: no pulled fits for {arms}")
    v = np.array([c[0] for c in cells])
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, float(np.mean([c[1] for c in cells]))


def _z(signal, a, b):
    ma, sa, ta = _side(signal, a)
    mb, sb, tb = _side(signal, b)
    return (ma - mb) / math.sqrt(sa ** 2 + sb ** 2 + ta ** 2 + tb ** 2)


RAND = tuple(f"rand2p{k}" for k in range(1, 6))


@pytest.mark.parametrize("signal", SIGNALS)
def test_17_classes_keep_fewer_signal_jets_than_188(signal):
    assert _z(signal, ("r16q1",), ("l188",)) < -3


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("part", [("flavf0",), ("flavf1",), ("flavf1r",)])
def test_each_flavor_boundary_partition_keeps_fewer_than_188(signal, part):
    assert _z(signal, part, ("l188",)) < -3


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("part", [("flavf0",), ("flavf1",), ("flavf1r",), RAND])
def test_no_partition_differs_from_17_classes(signal, part):
    assert abs(_z(signal, part, ("r16q1",))) <= 3
