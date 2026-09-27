#!/usr/bin/env python3
"""How much does unequal seed spread move the trend test? A committed simulation.

WHY. The trend test (src/stats/trend.py) permutes vocabulary labels within a
seed, so its null is exchangeability, not equal means. A vocabulary whose seeds
scatter much more than the others' breaks exchangeability with the means equal,
and the test can then reject more often than its nominal 5%. The 17-class models
scatter several times more than the others on the frozen b-versus-c probe (C1),
and the manuscript and docs/PRESPEC_2026-09.md (corrections of 2026-09-19) say
so; the size and the check of the observed statistic were quoted there from an
uncommitted calculation. This script is that calculation, committed.

WHAT. For every trend test in the committed analyses, under a Gaussian null
with EQUAL level means and the observed seed spreads:
  size   share of null draws in which the test, run as the analysis runs it,
         rejects at 5%;
  p_sim  share of null draws whose max-T statistic reaches the observed one:
         a p-value for "equal means" that does not assume equal spreads.
Three nulls, because five seeds estimate a spread poorly:
  cov          multivariate normal with the sample covariance of the
               level-centred values (unequal spreads AND the correlation that
               pairing by seed induces);
  indep        independent levels with the observed per-level SDs;
  indep_upper  as indep, with the noisiest level's SD at the upper 95%
               confidence limit of an SD estimated from n seeds,
               sqrt((n-1)/chi2_{0.025, n-1}) times the observed (2.87 at n = 5).

HOW THE DATA ARE OBTAINED. Not re-read by a second loader. Each analysis file
records the argv it was made with; seed_level.main is rerun on that argv into a
scratch directory with max_t_trend wrapped to record the exact (y, levels,
blocks) every trend test received. Every recorded test must reproduce a trend
result stored in the committed file (same statistic and p), and the count must
match, or the script stops: the tables simulated are then provably the ones
behind the published p-values.

Size is computed with the exact enumeration for the confirmatory tests (as the
analysis runs them) and with a 9,999-arrangement Monte-Carlo permutation test
for the others; C1 is also run with the Monte-Carlo test, so the two can be
compared. Nothing here changes a stored result.

Usage:
    python3 experiments/STATS/trend_size_sim.py --out experiments/FIGS/data/trend_size_sim
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.stats import trend as T                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("seed_level", REPO / "experiments/STATS/seed_level.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

DATA = REPO / "experiments/FIGS/data"
ANALYSES = [DATA / "probe_ladder_v2/analysis_family_of_four/seed_level_results.json",
            DATA / "finetune_s3_s4/analysis_v2/s3_s4_finetune.json",
            DATA / "anomaly_merged_v4/analysis_v2/anomaly_s5.json",
            DATA / "aoj_full_v1/analysis_labelled/aoj_top.json"]
ALPHA = 0.05
SEED = 20260927
N_SIZE = 10_000          # null draws per (test, null) for the size
N_STAT = 2_000_000       # null draws per (test, null) for p_sim
N_PERM_MC = 9_999        # Monte-Carlo arrangements for the non-confirmatory sizes
CHUNK = 200_000
NULLS = ("cov", "indep", "indep_upper")
_SLACK = 1e-9            # as trend._RTOL: the observed statistic counts as >= itself


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------ capture the tests

def stored_trends(node, path="") -> list[tuple[str, dict]]:
    """Every max-T trend result in a committed analysis file, with its JSON path."""
    out = []
    if isinstance(node, dict):
        if node.get("run") is True and {"stat", "p", "n_arrangements", "method"} <= node.keys():
            out.append((path, node))
        for k, v in node.items():
            out += stored_trends(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += stored_trends(v, f"{path}[{i}]")
    return out


def capture(analysis: pathlib.Path) -> list[dict]:
    """Rerun the analysis on its recorded argv and record every trend test's input."""
    doc = json.loads(analysis.read_text())
    argv = doc["provenance"]["argv"]
    argv = ast.literal_eval(argv) if isinstance(argv, str) else list(argv)
    calls, real = [], S.max_t_trend

    def spy(y, levels, blocks, **kw):
        r = real(y, levels, blocks, **kw)
        calls.append({"y": list(map(float, y)), "levels": list(levels),
                      "blocks": list(blocks), "kw": kw, "stat": r["stat"], "p": r["p"]})
        return r

    with tempfile.TemporaryDirectory() as tmp:
        i = argv.index("--out")
        argv = argv[:i + 1] + [str(pathlib.Path(tmp) / "out")] + argv[i + 2:]
        S.max_t_trend = spy
        cwd = os.getcwd()
        try:
            os.chdir(REPO)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = S.main(argv)
        finally:
            S.max_t_trend = real
            os.chdir(cwd)
    if rc:
        raise SystemExit(f"FATAL: rerunning {analysis} exited {rc}")

    stored = stored_trends(doc)
    if len(stored) != len(calls):
        raise SystemExit(f"FATAL: {analysis} stores {len(stored)} trend results but the "
                         f"rerun made {len(calls)} trend tests")
    unmatched = list(stored)
    for c in calls:
        hit = next((s for s in unmatched
                    if math.isclose(s[1]["stat"], c["stat"], rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(s[1]["p"], c["p"], rel_tol=1e-12, abs_tol=0.0)), None)
        if hit is None:
            raise SystemExit(f"FATAL: a rerun trend test (stat {c['stat']}, p {c['p']}) "
                             f"reproduces no result stored in {analysis}")
        unmatched.remove(hit)
        c["json_path"], c["task"], c["probe"] = hit[0], hit[1].get("task"), hit[1].get("probe")
    return calls


# ------------------------------------------------------------ the statistic

def table(call) -> tuple[np.ndarray, list]:
    """(blocks x levels) in the test's level order; only complete blocks, as the test uses."""
    order = list(call["kw"]["order"])
    ids = sorted(set(call["blocks"]), key=call["blocks"].index)
    Y = np.full((len(ids), len(order)), np.nan)
    for y, lv, b in zip(call["y"], call["levels"], call["blocks"]):
        Y[ids.index(b), order.index(lv)] = y
    return Y[np.isfinite(Y).all(1)], order


def max_t_stats(Y: np.ndarray, family: str, alternative: str) -> np.ndarray:
    """The max-T statistic of trend.max_t_trend for each (b x k) table in Y (..., b, k).

    Same formula as trend.max_t_trend's t_stats on block-centred rows; checked
    against it in tests/test_trend_size_sim.py."""
    b, k = Y.shape[-2:]
    rows = Y - Y.mean(-1, keepdims=True)
    n = np.full(k, float(b))
    m = rows.sum(-2) / n
    ss = (rows ** 2).sum((-2, -1))
    df = (b - 1) * (k - 1)
    s2 = np.maximum(ss - (n * m ** 2).sum(-1), 1e-30 * ss) / df
    c = T.contrast_matrix(k, family, n)
    scale = np.sqrt((c ** 2 / n).sum(1))
    t = (m @ c.T) / (np.sqrt(s2)[..., None] * scale)
    sign = {"increasing": 1.0, "decreasing": -1.0}.get(alternative)
    return (np.abs(t) if sign is None else sign * t).max(-1)


# ------------------------------------------------------------ the nulls

def null_cov(Y: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """Covariance of one block's k values under the named null, and what it rests on."""
    b = Y.shape[0]
    C = np.cov(Y, rowvar=False, ddof=1)
    sd = np.sqrt(np.diag(C))
    info = {"seed_sd": sd.tolist()}
    if kind == "cov":
        return C, info
    D = np.diag(sd ** 2)
    if kind == "indep":
        return D, info
    j = int(np.argmax(sd))
    f = math.sqrt((b - 1) / stats.chi2.ppf(0.025, b - 1))
    D[j, j] *= f ** 2
    return D, {**info, "inflated_level_index": j, "sd_factor": f}


def draw(C: np.ndarray, shape: tuple, rng) -> np.ndarray:
    """Gaussian draws with covariance C (positive semi-definite allowed)."""
    w, V = np.linalg.eigh(C)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    return rng.standard_normal(shape + (C.shape[0],)) @ L.T


def p_sim(C, b, family, alternative, observed, seed) -> dict:
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, N_STAT, CHUNK):
        m = min(CHUNK, N_STAT - start)
        st = max_t_stats(draw(C, (m, b), rng), family, alternative)
        hits += int((st >= observed - _SLACK * max(1.0, abs(observed))).sum())
    return {"n_ge": hits, "n_draws": N_STAT, "p_sim": (hits + 1) / (N_STAT + 1)}


def _size_chunk(args) -> int:
    C, b, order, kw, exact, seed, n = args
    rng = np.random.default_rng(seed)
    levels = list(order) * b
    blocks = np.repeat(np.arange(b), len(order))
    rej = 0
    for _ in range(n):
        y = draw(C, (b,), rng).ravel()
        if exact:
            r = T.max_t_trend(y, levels, blocks, alternative=kw["alternative"],
                              family=kw["family"], exact=True, order=order)
        else:
            r = T.max_t_trend(y, levels, blocks, alternative=kw["alternative"],
                              family=kw["family"], exact=False, n_perm=N_PERM_MC,
                              rng=np.random.default_rng(list(seed) + [_]), order=order)
        rej += r["p"] <= ALPHA
    return rej


def size(C, b, order, kw, exact, seed, pool, workers) -> dict:
    per = [N_SIZE // workers + (i < N_SIZE % workers) for i in range(workers)]
    jobs = [(C, b, order, kw, exact, list(seed) + [i], n) for i, n in enumerate(per) if n]
    rej = sum(pool.map(_size_chunk, jobs))
    lo, hi = stats.binomtest(rej, N_SIZE).proportion_ci(0.95, method="exact")
    return {"size": rej / N_SIZE, "size_ci95": [lo, hi], "n_rejections": rej,
            "n_draws": N_SIZE, "method": "exact" if exact else f"monte-carlo {N_PERM_MC}"}


# ------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analyses", nargs="+", type=pathlib.Path, default=ANALYSES)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    out_file = a.out / "trend_size_sim.json"
    if out_file.exists():
        raise SystemExit(f"FATAL: {out_file} exists; write a new directory")

    tests = []
    for path in a.analyses:
        for c in capture(path):
            tests.append({**c, "analysis": str(path.relative_to(REPO))})
    print(f"{len(tests)} trend tests captured, every one matched to a stored result", flush=True)

    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for ti, c in enumerate(tests):
            Y, order = table(c)
            b = Y.shape[0]
            kw = {"alternative": c["kw"]["alternative"], "family": c["kw"]["family"]}
            observed = float(max_t_stats(Y[None], kw["family"], kw["alternative"])[0])
            if not math.isclose(observed, c["stat"], rel_tol=1e-9):
                raise SystemExit(f"FATAL: the vectorised statistic {observed} differs from the "
                                 f"test's {c['stat']} for {c['json_path']}")
            confirmatory = c["json_path"].startswith("confirmatory")
            row = {"analysis": c["analysis"], "json_path": c["json_path"], "task": c["task"],
                   "probe": c["probe"], "levels": order, "n_blocks": b,
                   "stat": c["stat"], "p": c["p"], "rejects": c["p"] <= ALPHA,
                   "nulls": {}}
            for ni, kind in enumerate(NULLS):
                C, info = null_cov(Y, kind)
                seed = [SEED, ti, ni]
                r = {**info, **p_sim(C, b, kw["family"], kw["alternative"], observed, seed + [0]),
                     "size": size(C, b, order, kw, confirmatory, seed + [1], pool, a.workers)}
                if confirmatory:
                    r["size_monte_carlo"] = size(C, b, order, kw, False, seed + [2], pool, a.workers)
                row["nulls"][kind] = r
                print(f"{c['json_path'][:60]:60s} {kind:12s} size {r['size']['size']:.4f} "
                      f"p_sim {r['p_sim']:.2e} (observed p {c['p']:.2e})", flush=True)
            row["survives_every_null"] = all(v["p_sim"] <= ALPHA for v in row["nulls"].values())
            rows.append(row)

    rejected = [r for r in rows if r["rejects"]]
    out = {"provenance": {"script_sha256": sha256(__file__),
                          "trend_module_sha256": sha256(REPO / "src/stats/trend.py"),
                          "seed_level_sha256": sha256(REPO / "experiments/STATS/seed_level.py"),
                          "analyses": [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)}
                                       for p in a.analyses],
                          "seed": SEED, "alpha": ALPHA, "n_size_draws": N_SIZE,
                          "n_stat_draws": N_STAT, "n_perm_monte_carlo": N_PERM_MC,
                          "nulls": list(NULLS)},
           "tests": rows,
           "summary": {"n_tests": len(rows), "n_rejected": len(rejected),
                       "n_rejected_surviving_every_null": sum(r["survives_every_null"] for r in rejected),
                       "rejected_not_surviving": [r["json_path"] for r in rejected
                                                  if not r["survives_every_null"]],
                       "max_size": {k: max(r["nulls"][k]["size"]["size"] for r in rows) for k in NULLS}}}
    a.out.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
