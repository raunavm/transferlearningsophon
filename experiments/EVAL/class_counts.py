#!/usr/bin/env python3
"""Selected test-split jets per native class, and what the v2 extraction needs.

Reads four columns (jet_label, jet_pt, jet_sdmass, jet_eta) of every test file
with pyarrow, applies the data config's selection (200 < pT < 2500 and
20 < m_SD < 500 GeV, configs/data/JetClassII_base.yaml) and, for the windowed
task, probe.py's window. No model, no weaver.

`needs` then turns the counts into the audit's requirement (B3, must-fix 8):
for each probe task, the background jets its test split must hold for at least
MIN_PASS of them to pass the 90 % signal-efficiency cut, given the largest
rejection any v1 model reached there (1/eps_B = n_bkg / k). A task whose v1
models passed NO background jet at 90 % has no measured rejection; its need is
reported as unbounded and the full split is the most that can be done.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SELECTION = {"jet_pt": (200.0, 2500.0), "jet_sdmass": (20.0, 500.0)}
MIN_PASS = 100
EPS_S = "0.90"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def count_file(path: str, window: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(selected counts per native label, in-window counts, raw jets)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["jet_label", "jet_pt", "jet_sdmass", "jet_eta"])
    c = {k: t.column(k).to_numpy() for k in t.column_names}
    sel = np.ones(len(c["jet_label"]), bool)
    for k, (lo, hi) in SELECTION.items():
        sel &= (c[k] > lo) & (c[k] < hi)
    win = sel.copy()
    for k, (lo, hi) in window.items():
        win &= (c[k] > lo) & (c[k] < hi)
    lab = c["jet_label"].astype(np.int64)
    return (np.bincount(lab[sel], minlength=188), np.bincount(lab[win], minlength=188),
            int(sel.size))


def needs(counts: np.ndarray, window_counts: np.ndarray, probe_files: list[pathlib.Path],
          test_fraction: float) -> dict:
    """Per task: background jets in the whole split and in a test split of
    `test_fraction`, the largest v1 rejection at 90 %, and the passing count it
    implies. The largest rejection over every model and both probes is used,
    so the count holds for the best model, i.e. the fewest passing jets."""
    probe = _load("probe", "experiments/EVAL/probe.py")
    worst = {}
    for f in probe_files:
        for task, T in json.loads(pathlib.Path(f).read_text())["tasks"].items():
            if T.get("skipped"):
                continue
            for A in T["arms"].values():
                for kind in ("linear", "mlp"):
                    r = A[kind]["rejection_at"].get(EPS_S)
                    if r is None:
                        continue
                    w = worst.setdefault(task, {"max_rejection": 0.0, "unmeasured": 0})
                    if r["n_bkg_pass"] == 0:
                        w["unmeasured"] += 1
                    else:
                        w["max_rejection"] = max(w["max_rejection"], r["rejection"])
    out = {}
    for task, spec in probe.TASKS.items():
        cnt = window_counts if spec.get("window") else counts
        nb, ns = int(cnt[spec["background"]].sum()), int(cnt[spec["signal"]].sum())
        w = worst.get(task, {"max_rejection": None, "unmeasured": None})
        r = w["max_rejection"] or None
        nb_test = int(nb * test_fraction)
        out[task] = {"n_signal_split": ns, "n_background_split": nb,
                     "n_background_test": nb_test, "max_v1_rejection_at_90": r,
                     "v1_cells_with_no_passing_jet": w["unmeasured"],
                     "background_test_needed": None if r is None else int(np.ceil(MIN_PASS * r)),
                     "expected_pass_at_max_rejection": None if r is None else nb_test / r,
                     "meets_min_pass": None if r is None else bool(nb_test / r >= MIN_PASS)}
    return out


# Bytes per kept row, as extract_v2.write stores them.
FEATURE_ROW_BYTES = 128 * 2 + 8 + 2          # float16 features, int64 row, int16 label
HEAD_ROW_BYTES = 8 + 2 + 2 + 4 * 4 + 12 * 4  # rows, label, argmax, 4 floats, 12 class sums


def storage(counts: dict, prefix_counts: np.ndarray, feature_classes: list[int],
            head_classes: list[int], n_models: int, n_ckpt_features: int,
            n_ckpt_heads: int, diag: int = 20_000) -> dict:
    """Bytes the v2 extraction writes: features at n_ckpt_features checkpoints
    (probe classes over the split, plus the first 2,000,000 jets), head scores at
    n_ckpt_heads (the anomaly rows of the first 2,000,000 and a stride sample).
    Uncompressed; head_scores.npz is compressed, so its figure is an upper bound."""
    sel = np.asarray(counts["selected_per_class"])
    fc = sorted(set(feature_classes))
    n_feat = int(sel[fc].sum() + prefix_counts.sum() - prefix_counts[fc].sum())
    n_head = int(prefix_counts[sorted(set(head_classes))].sum() + diag)
    per_model = (n_ckpt_features * n_feat * FEATURE_ROW_BYTES
                 + n_ckpt_heads * n_head * HEAD_ROW_BYTES)
    return {"feature_rows_per_checkpoint": n_feat, "head_rows_per_checkpoint": n_head,
            "bytes_per_model": per_model, "bytes_total": per_model * n_models,
            "n_models": n_models, "checkpoints_with_features": n_ckpt_features,
            "checkpoints_with_heads": n_ckpt_heads}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    probe = _load("probe", "experiments/EVAL/probe.py")
    window = probe.TASKS["bc_vs_rest"]["window"]
    tot, totw, raw, per_file = np.zeros(188, np.int64), np.zeros(188, np.int64), 0, {}
    for i, f in enumerate(a.data_test):
        c, w, n = count_file(f, window)
        tot += c
        totw += w
        raw += n
        per_file[pathlib.Path(f).name] = int(c.sum())
        if i % 25 == 0:
            print(f"  {i + 1}/{len(a.data_test)} files, {int(tot.sum()):,} selected", flush=True)
    res = {"selection": SELECTION, "window": window, "n_files": len(a.data_test),
           "n_raw": raw, "n_selected": int(tot.sum()), "selected_per_class": tot.tolist(),
           "in_window_per_class": totw.tolist(), "selected_per_file": per_file,
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}: {res['n_selected']:,} selected of {raw:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
