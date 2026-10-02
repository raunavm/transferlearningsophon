#!/usr/bin/env python3
"""How well each real-data top fit describes the jets that pass its cut below the top window.

WHY. A study of the same CMS open data with a score of the same kind (the sum of a
foundation model's multi-prong outputs over its QCD outputs) found that a non-smooth
feature in the left sideband skewed its top-peak fit (Mikuni and Nachman,
arXiv:2603.23593v2, p. 1 and Conclusion). Our top fit runs from the lower edge of
peak_fit.PEAKS["top"]["fit_range"] and so contains that region. If a model's cut picked up
such a feature, the pass-to-fail ratio and the peak would absorb it and its yield would
move. The background model is validated on a band around the median of the score, not at
the working point; this looks at the working point itself.

WHAT. For every model of a fit's histograms.npz (written by experiments/AOJ/fit_v6.py and
its predecessors: per model, pT-summed passing counts and the fitted passing background and
signal in each mass bin), the residual of the passing jets between the lower edge of the
fit range and the lower edge of the top window:

    z = (sum n_pass - sum mu) / sqrt(sum mu),   mu = background + signal,

the bins summed before the ratio is taken, and the largest single-bin Pearson pull there.
Nothing is refitted: the expectation is the one the fit returned. The edges must fall on
bin edges, or this refuses.

Reads committed files only.

Usage (from the repository root):
    python3 experiments/AOJ/sideband_residuals.py experiments/FIGS/data/aoj_full_v1/fit_v6/histograms.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PEAK = "top"
SUFFIXES = ("n_pass", "background", "signal", "m_edges")


def peak_fit():
    spec = importlib.util.spec_from_file_location("peak_fit_for_residuals", HERE / "peak_fit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sideband_edges(peak: str = PEAK) -> tuple[float, float]:
    """(lower edge of the fit range, lower edge of the peak window), from peak_fit.PEAKS."""
    cfg = peak_fit().PEAKS[peak]
    return float(cfg["fit_range"][0]), float(cfg["window"][0])


def models(hist) -> list[str]:
    """Model names in a histograms.npz, each with every array this needs."""
    names = sorted({k[: -len(f"_{PEAK}_n_pass")] for k in hist.files if k.endswith(f"_{PEAK}_n_pass")})
    for n in names:
        missing = [s for s in SUFFIXES if f"{n}_{PEAK}_{s}" not in hist.files]
        if missing:
            raise SystemExit(f"FATAL: {n} has no {missing} in the histograms")
    return names


def residual(n_pass, background, signal, edges, lo: float, hi: float) -> dict:
    """The passing-jet residual over [lo, hi) for one model's pT-summed histograms."""
    n_pass, mu = np.asarray(n_pass, float), np.asarray(background, float) + np.asarray(signal, float)
    edges = np.asarray(edges, float)
    if len(edges) != len(n_pass) + 1 or len(mu) != len(n_pass):
        raise SystemExit("FATAL: the histogram arrays do not have one more edge than bins")
    i0, i1 = np.flatnonzero(np.isclose(edges, lo)), np.flatnonzero(np.isclose(edges, hi))
    if len(i0) != 1 or len(i1) != 1 or i1[0] <= i0[0]:
        raise SystemExit(f"FATAL: [{lo}, {hi}) does not fall on the bin edges {edges.tolist()}")
    sl = slice(int(i0[0]), int(i1[0]))
    obs, exp = float(n_pass[sl].sum()), float(mu[sl].sum())
    pulls = (n_pass[sl] - mu[sl]) / np.sqrt(mu[sl])
    k = int(np.argmax(np.abs(pulls)))
    return {"z": (obs - exp) / math.sqrt(exp), "n_observed": obs, "n_expected": exp,
            "n_bins": int(sl.stop - sl.start), "max_abs_pull": float(abs(pulls[k])),
            "max_pull_bin_gev": [float(edges[sl.start + k]), float(edges[sl.start + k + 1])]}


def all_residuals(path: pathlib.Path, lo: float | None = None, hi: float | None = None) -> dict:
    """{model: residual} for every model of a histograms.npz, over [lo, hi) (default: the
    region between the top fit's lower edge and its window)."""
    if lo is None or hi is None:
        lo, hi = sideband_edges()
    hist = np.load(path)
    return {n: residual(hist[f"{n}_{PEAK}_n_pass"], hist[f"{n}_{PEAK}_background"],
                        hist[f"{n}_{PEAK}_signal"], hist[f"{n}_{PEAK}_m_edges"], lo, hi)
            for n in models(hist)}


def mean_pulls(path: pathlib.Path, names: list[str]) -> dict:
    """The Pearson pull of every bin of the fit, averaged over the models `names`: a
    fluctuation of the data that every model's cut keeps shows in many models at once, so a
    bin's pull in one model is read against the same bin's mean over the models and against
    the other bins. Returns the bin edges, the mean pulls and, for each bin, the rank of its
    |mean pull| among all bins (1 = largest)."""
    hist = np.load(path)
    edges = {tuple(hist[f"{n}_{PEAK}_m_edges"]) for n in names}
    if len(edges) != 1:
        raise SystemExit("FATAL: the models' histograms do not share their mass bins")
    pulls = np.array([(hist[f"{n}_{PEAK}_n_pass"] - hist[f"{n}_{PEAK}_background"] - hist[f"{n}_{PEAK}_signal"])
                      / np.sqrt(hist[f"{n}_{PEAK}_background"] + hist[f"{n}_{PEAK}_signal"]) for n in names])
    mean = pulls.mean(axis=0)
    order = np.argsort(-np.abs(mean))
    rank = np.empty(len(mean), int)
    rank[order] = np.arange(1, len(mean) + 1)
    return {"edges": list(map(float, edges.pop())), "mean_pull": mean.tolist(), "rank": rank.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("histograms", type=pathlib.Path)
    a = ap.parse_args(argv)
    lo, hi = sideband_edges()
    res = all_residuals(a.histograms, lo, hi)
    print(json.dumps({"range_gev": [lo, hi], "models": res,
                      "mean_pulls_all_models": mean_pulls(a.histograms, sorted(res))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
