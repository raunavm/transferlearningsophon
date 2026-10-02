#!/usr/bin/env python3
"""What the v2 extraction needs, from the measured test-split class counts.

Inputs: class_counts.py's test_class_counts.json (every selected test jet per
native class, all 335 files), the committed v1 probe results (the largest
rejection any model reached at 90 % signal efficiency, per task), and the
2,000,000-jet v1 label vector (the prefix whose rows the label-recovery and
anomaly analyses read). Output, per probe task: the background jets in the
split and in its test part at each candidate split, the passing jets expected
at the largest v1 rejection, whether that reaches MIN_PASS; and the storage the
extraction writes for the v2 grid (configs/arms/v2_grid.json) under the
checkpoint plans considered, against the volume's free space.

THE PLAN (amendment A14). Every run of the grid at best70, wavg and bestval (counted
as a checkpoint of its own for every run, the upper bound: where it is best70's epoch
it is extracted once), with float16 features, the pooled embedding and the observers
of the same rows, and head scores for a model with an output layer; the
self-supervised runs without head scores; and the untrained trunk of run indices 1-5
(scripts/build_extract_jobs.py v2_init_refs) at one checkpoint without head scores.
There is no smaller plan: A8 and A14 need every checkpoint's features.

THE PRETRAINING CHECKPOINTS COUNT TOO. The v2 runs' own checkpoints sit on the same
volume, so a plan fits only if the extraction AND every run's checkpoints stay
under the 85 % line (verification 2026-10-01: the model had left them out). The
per-run figure is --checkpoint-bytes-per-run, by default what a finished run keeps
(retained_bytes_per_run). scripts/build_extract_jobs.py refuses to emit a plan
this file says does not fit.

SO DOES THE FINE-TUNING: ONE BUDGET. The v2 fine-tuning specs write to the same
volume; each plan fitted alone and both together did not (verification
2026-10-02). bytes_total adds what every v2 fine-tuning spec leaves on /data
(scripts/build_ft_jobs.py v2_need), or --fine-tuning-bytes when the df above
already holds some of it; build_ft_jobs.py holds this plan's extraction_bytes back
from its own headroom in turn.

THE BOUND LEAVES OUT A14's BATCHNORM CONTINGENCY. If experiments/DIAG/head_bn_diag.py
finds that recomputing BatchNorm alone repairs at least half of the defective stored
v1 epochs, every reported v2 checkpoint gets a BatchNorm-recomputed twin: best70 and
bestval (the weight average has its statistics recomputed already). Extracting those
twins would add batchnorm_twins_bytes, which is not in bytes_total;
scripts/build_extract_jobs.py emits no plan until that readout says the rule does
not fire.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST_FRACTIONS = (0.2, 0.6, 0.7)            # v1's split, and two v2 candidates
PLAN = "features, pooled and heads at best70, wavg and bestval"
N_CHECKPOINTS = 3                           # best70, wavg, bestval: per run of the grid
POOLED_ROW_BYTES = 128 * 2                  # float16 pooled embedding per feature row
N_EPOCHS = 80
N_BN_TWINS = 2                              # best70, bestval: A14's BatchNorm contingency


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def retained_bytes_per_run() -> tuple[float, dict]:
    """(bytes, how) a finished v2 run keeps under --keep-checkpoints window
    (experiments/MTX/pretrain_v2.py prune and write_weight_average): the state files of
    window_epochs (EARLY_KEEP and the last ten epochs), of the best epoch when it is
    neither (counted always, the upper bound), net_best_epoch_state.pt and the weight
    average; the newest resume file; the records, init_trunk.pt included; and the
    gradient diagnostic's saved batch (pretrain_v2.DIAG_FILE). Sizes of the 188-output
    model, the largest (scripts/build_mtx_launch.py V2_STATE_MIB)."""
    pv = _load("pretrain_v2", "experiments/MTX/pretrain_v2.py")
    ml = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    n_states = len(pv.window_epochs(N_EPOCHS)) + 3
    mib = n_states * ml.V2_STATE_MIB + ml.V2_RESUME_MIB + ml.V2_RECORDS_MIB + ml.V2_DIAG_BATCH_MIB
    return mib * 2**20, {"state_files": n_states, "state_mib": ml.V2_STATE_MIB,
                         "early_keep": list(pv.EARLY_KEEP), "resume_files": 1,
                         "resume_mib": ml.V2_RESUME_MIB, "records_mib": ml.V2_RECORDS_MIB,
                         "diag_batch_mib": ml.V2_DIAG_BATCH_MIB}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--counts", required=True, type=pathlib.Path)
    ap.add_argument("--probe-files", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--prefix-labels", required=True, type=pathlib.Path)
    ap.add_argument("--free-bytes", type=float, required=True,
                    help="free space on /data now (df)")
    ap.add_argument("--size-bytes", type=float, required=True, help="volume size (df)")
    ap.add_argument("--checkpoint-bytes-per-run", type=float, default=None,
                    help="pretraining checkpoints each v2 run keeps on /data; default "
                         "retained_bytes_per_run()")
    ap.add_argument("--fine-tuning-bytes", type=float, default=None,
                    help="what the v2 fine-tuning will still leave on /data beyond the df above; "
                         "default every spec, scripts/build_ft_jobs.py v2_need()")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    cc = _load("class_counts", "experiments/EVAL/class_counts.py")
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    ft_how = {"source": "--fine-tuning-bytes"}
    if a.fine_tuning_bytes is None:
        need = _load("build_ft_jobs", "scripts/build_ft_jobs.py").v2_need()
        a.fine_tuning_bytes, ft_how = sum(need.values()), {"source": "v2_need", "specs": len(need)}
    counts = json.loads(a.counts.read_text())
    prefix = np.bincount(np.load(a.prefix_labels).astype(np.int64), minlength=188)
    tasks = {f: cc.needs(np.asarray(counts["selected_per_class"]),
                         np.asarray(counts["in_window_per_class"]), a.probe_files, f)
             for f in TEST_FRACTIONS}
    grid = {x["name"]: x for x in json.loads((REPO / "configs/arms/v2_grid.json").read_text())["arms"]}
    runs = bx.v2_runs()                                    # (run, arm, K, num_reg, seed)
    n_init = len(bx.v2_init_refs())
    how = {"source": "--checkpoint-bytes-per-run"}
    if a.checkpoint_bytes_per_run is None:
        a.checkpoint_bytes_per_run, how = retained_bytes_per_run()
        how = {"source": "retained_bytes_per_run", **how}
    ckpt_bytes = a.checkpoint_bytes_per_run * len(runs)  # the self-supervised runs too
    obs_row_bytes = 4 * len(xv.V2_OBSERVERS)               # float32 per kept observer
    qcd, signals = xv.anomaly_classes()
    # Feature rows as extract_v2 keeps them: classes an unwindowed task reads over
    # the whole split, windowed-only classes inside their window, and every row of
    # the 2,000,000-jet prefix. The windowed rows inside the prefix are counted
    # twice, so the figure is an upper bound (by < 0.1 M rows).
    anywhere, windowed = xv.probe_feature_rules()
    sel = np.asarray(counts["selected_per_class"])
    win = np.asarray(counts["in_window_per_class"])
    n_feat = int(sel[anywhere].sum() + sum(win[c].sum() for c, _ in windowed)
                 + prefix.sum() - prefix[anywhere].sum())
    n_head = int(prefix[sorted(set(qcd + list(signals.values())))].sum() + 20_000)
    feat_ckpt = n_feat * (cc.FEATURE_ROW_BYTES + POOLED_ROW_BYTES + obs_row_bytes)
    head_ckpt = n_head * cc.HEAD_ROW_BYTES
    storage = {}
    for label, keep in (("every run and the init references", lambda r: True),
                        ("tier-1 runs and the init references", lambda r: grid[r[1]]["tier"] == 1)):
        cls = sum(1 for r in runs if keep(r) and r[2])
        ssl = sum(1 for r in runs if keep(r) and not r[2])
        extraction = (cls * N_CHECKPOINTS * (feat_ckpt + head_ckpt)
                      + ssl * N_CHECKPOINTS * feat_ckpt + n_init * feat_ckpt)
        storage[f"{PLAN}; {label}"] = {
            "feature_rows_per_checkpoint": n_feat, "head_rows_per_checkpoint": n_head,
            "bytes_per_checkpoint_features": feat_ckpt, "bytes_per_checkpoint_heads": head_ckpt,
            "checkpoints_per_run": N_CHECKPOINTS, "n_classification_runs": cls,
            "n_self_supervised_runs": ssl, "n_init_references": n_init,
            "n_models": cls + ssl + n_init, "extraction_bytes": extraction,
            "pretraining_checkpoint_bytes": ckpt_bytes, "fine_tuning_bytes": a.fine_tuning_bytes,
            "bytes_total": extraction + ckpt_bytes + a.fine_tuning_bytes,
            "batchnorm_twins_bytes": N_BN_TWINS * (cls * (feat_ckpt + head_ckpt) + ssl * feat_ckpt)}
    used = a.size_bytes - a.free_bytes
    line85 = 0.85 * a.size_bytes
    for s in storage.values():
        s["fits_under_85pc"] = bool(used + s["bytes_total"] < line85)
        s["headroom_to_85pc_bytes"] = line85 - used
    res = {"inputs": {"counts": {"path": str(a.counts), "sha256": hashlib.sha256(
               a.counts.read_bytes()).hexdigest()},
                      "probe_files": [str(p) for p in a.probe_files],
                      "prefix_labels_sha256": hashlib.sha256(
                          np.load(a.prefix_labels).tobytes()).hexdigest()},
           "min_pass": cc.MIN_PASS, "eps_s": cc.EPS_S,
           "tasks_by_test_fraction": {str(f): t for f, t in tasks.items()},
           "n_runs_v2_grid": len(runs),
           "checkpoint_bytes_per_run": a.checkpoint_bytes_per_run,
           "checkpoint_bytes_per_run_from": how,
           "fine_tuning_bytes_from": ft_how,
           "observers": list(xv.V2_OBSERVERS), "pooled_row_bytes": POOLED_ROW_BYTES,
           "feature_rules": {"anywhere": anywhere, "windowed": windowed},
           "storage": storage,
           "volume": {"size_bytes": a.size_bytes, "free_bytes": a.free_bytes}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
