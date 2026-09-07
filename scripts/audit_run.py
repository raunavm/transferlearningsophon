#!/usr/bin/env python3
"""Audit a pretraining run directory against docs/RECORD.md, from what is on disk.

Reads train.log, RECIPE and the per-epoch optimizer checkpoints of one or more
`/data/results/mtx/<run_id>` directories and reports, per run:

  epochs       validation lines found (80 = complete)
  resumes      every `Resume training from epoch N` in train.log, and whether
               N fell in the double-decay window (src/utils/resume.py)
  lr           the learning rate stored in EVERY epoch's optimizer file
               against the flat+decay formula for the RECIPE's rate; any
               mismatch is flagged (this is what detects a double decay, or
               any other schedule anomaly, after the fact)
  best         argmax of the per-epoch validation metric, and whether
               net_best_epoch_state.pt is byte-identical to that epoch's state
               file (weaver resets its best-metric tracker on every resume, so
               the marker can be overwritten by a worse post-resume epoch)
  compute      per-epoch wall time (epoch start -> validation end), total
               attributable GPU-hours, training-only throughput in jets/s

Read-only by default. `--write` stores the result as run_audit.json in the run
directory (fills the compute fields the launch-time manifest leaves null).

Run inside a pod that mounts /data (torch is needed for the optimizer files):

    kubectl exec -i -n cms-ml <pod> -- python3 - /data/results/mtx/mtx-r16q1-s1 \\
        < scripts/audit_run.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys

TS = "%Y-%m-%d %H:%M:%S"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RESUME = re.compile(r"Resume training from epoch (\d+)")
TRAIN = re.compile(r"Epoch #(\d+) training")
VALID = re.compile(r"Epoch #(\d+) validating")
METRIC = re.compile(r"Epoch #(\d+): Current validation metric: ([0-9.]+) \(best: ([0-9.]+)\)")
TLOSS = re.compile(r"Train AvgLoss: ([0-9.]+),.*AvgAcc: ([0-9.]+)")
RECIPE = re.compile(r"lr=([0-9.e-]+) epochs=(\d+)")
# Measured in-image with fvcore at N=128 particles, K=10 (E1 JetClass-I):
# /data/results/e1/eval/flops.json. Verified: PARAMS_ANCHOR_K10 + 178*513 =
# 2,304,688 = the 188-class trainable count in docs/GROUND_TRUTH.md.
MACS_ANCHOR_K10 = 326_638_704
PARAMS_ANCHOR_K10 = 2_213_374
ARM_K = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17,
         "L162_MASS": 163, "R16_Q1_MASS": 18}  # mass twins carry one extra node


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_saved_lr(lr0: float, n_epochs: int, e: int) -> float:
    """lr in the optimizer file written after epoch e (scheduler already stepped)."""
    n_decay = max(1, int(n_epochs * 0.3))
    first = n_epochs - n_decay
    gamma = 0.01 ** (1.0 / n_decay)
    k = min(max(e + 1 - first + 1, 0), n_decay)
    return lr0 * gamma ** k


def parse_log(path: pathlib.Path) -> dict:
    start, vstart, vend, metric, tloss, resumes = {}, {}, {}, {}, {}, []
    cur = None
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\] \w+: (.*)$", line)
        if not m:
            continue
        t, msg = dt.datetime.strptime(m.group(1), TS), m.group(2)
        if (r := RESUME.match(msg)):
            resumes.append(int(r.group(1)))
        elif (r := TRAIN.match(msg)):
            cur = int(r.group(1))
            start[cur] = t
        elif (r := VALID.match(msg)):
            vstart[int(r.group(1))] = t
        elif (r := TLOSS.match(msg)) and cur is not None:
            tloss[cur] = (float(r.group(1)), float(r.group(2)))
        elif (r := METRIC.match(msg)):
            e = int(r.group(1))
            metric[e] = float(r.group(2))
            vend[e] = t
    return dict(start=start, vstart=vstart, vend=vend, metric=metric, tloss=tloss, resumes=resumes)


def audit(run: pathlib.Path, samples_per_epoch: int, batch_size: int = 512) -> dict:
    out = {"run_id": run.name, "problems": []}
    log = run / "train.log"
    if not log.exists():
        out["problems"].append("no train.log")
        return out
    L = parse_log(log)
    # K from the run id: mtx-<arm><maybe mass>-s<seed>
    stem = run.name.replace("mtx-", "").rsplit("-s", 1)[0]
    key = {"l188": "L188", "l162": "L162", "r42q1": "R42_Q1", "r16q1": "R16_Q1",
           "l162mass": "L162_MASS", "r16q1mass": "R16_Q1_MASS"}.get(stem)
    num_classes = ARM_K.get(key)
    out["arm"], out["num_classes"] = key, num_classes
    if key is None:
        out["problems"].append(f"run id {run.name!r} does not name a known arm: params/FLOPs not derived")
    recipe = (run / "RECIPE").read_text().strip() if (run / "RECIPE").exists() else ""
    rm = RECIPE.search(recipe)
    lr0, n_epochs = (float(rm.group(1)), int(rm.group(2))) if rm else (None, None)
    out.update(recipe=recipe, epochs_with_metric=len(L["metric"]), resumes=L["resumes"])

    # resumes: window check
    if lr0 is not None:
        first = n_epochs - max(1, int(n_epochs * 0.3))
        out["resumes_in_double_decay_window"] = [n for n in L["resumes"] if first - 1 <= n <= n_epochs - 2]

    # lr in every optimizer checkpoint vs formula
    lr_mismatch = []
    try:
        import torch
        for f in sorted(run.glob("net_epoch-*_optimizer.pt"),
                        key=lambda p: int(re.search(r"epoch-(\d+)_", p.name).group(1))):
            e = int(re.search(r"epoch-(\d+)_", f.name).group(1))
            g = torch.load(f, map_location="cpu", weights_only=False)["param_groups"][0]
            if lr0 is not None:
                exp = expected_saved_lr(lr0, n_epochs, e)
                if abs(g["lr"] - exp) > 1e-9 * max(exp, 1e-12):
                    lr_mismatch.append({"epoch": e, "stored": g["lr"], "expected": exp,
                                        "ratio": g["lr"] / exp})
        out["lr_checked_epochs"] = len(list(run.glob("net_epoch-*_optimizer.pt")))
    except ImportError:
        out["lr_checked_epochs"] = None
        out["problems"].append("torch not importable: optimizer lr not checked")
    out["lr_mismatch"] = lr_mismatch
    if lr_mismatch:
        out["problems"].append(f"{len(lr_mismatch)} optimizer files hold an off-schedule lr "
                               f"(first at epoch {lr_mismatch[0]['epoch']}, ratio {lr_mismatch[0]['ratio']:.4f})")

    # best marker
    if L["metric"]:
        argmax = max(L["metric"], key=lambda e: L["metric"][e])
        last = max(L["metric"])
        out["val_argmax_epoch"], out["val_argmax"] = argmax, L["metric"][argmax]
        out["val_last_epoch"], out["val_last"] = last, L["metric"][last]
        best, arg = run / "net_best_epoch_state.pt", run / f"net_epoch-{argmax}_state.pt"
        if best.exists() and arg.exists():
            ok = md5(best) == md5(arg)
            out["best_file_is_argmax"] = ok
            if not ok:
                out["problems"].append(f"net_best_epoch_state.pt is not epoch {argmax} (the validation argmax)")
        else:
            out["best_file_is_argmax"] = None
            out["problems"].append("best or argmax state file missing")

    # compute
    wall, train_only = [], []
    for e, t0 in L["start"].items():
        if e in L["vend"]:
            wall.append((L["vend"][e] - t0).total_seconds())
        if e in L["vstart"]:
            train_only.append((L["vstart"][e] - t0).total_seconds())
    if wall:
        jets_s = (samples_per_epoch / sorted(train_only)[len(train_only) // 2]) if train_only else None
        # ParT Table 4 convention: fvcore MACs, labelled FLOPs. Per-arm MACs are
        # the K=10 anchor plus the head, see docs/RECORD.md 2.1.
        k = num_classes
        macs = MACS_ANCHOR_K10 + (k - 10) * 512 if k else None
        seen = samples_per_epoch * n_epochs if n_epochs else None
        out["compute"] = {
            "epochs_timed": len(wall),
            "wall_hours_attributable": round(sum(wall) / 3600, 2),
            "epoch_wall_min_median": round(sorted(wall)[len(wall) // 2] / 60, 1),
            "epoch_wall_min_max": round(max(wall) / 60, 1),
            "train_jets_per_s_median": round(jets_s, 1) if jets_s else None,
            # PELICAN (arXiv:2307.16506) reports seconds per training batch.
            "sec_per_train_batch_median": round(batch_size / jets_s, 4) if jets_s else None,
            "batch_size": batch_size,
            "trainable_params": PARAMS_ANCHOR_K10 + (k - 10) * 513 if k else None,
            "macs_per_jet_fwd": macs,
            "flops_per_jet_fwd_2xmac": 2 * macs if macs else None,
            # Kaplan/Hoffmann convention: fwd+bwd ~ 3x fwd. The 3x is convention.
            "train_flops_total": 3 * 2 * macs * seen if (macs and seen) else None,
            "examples_seen_planned": seen,
            "kwh_estimate_at_250W": round(sum(wall) / 3600 * 0.250, 2),
            "flops_convention": "MACs, as in ParT Table 4 (PMLR 162:18281); FLOPs = 2 x MACs",
            "note": ("epoch start to validation end, summed over logged epochs; partial epochs "
                     "lost to evictions are NOT included, so this understates the GPU-hours the "
                     "run consumed. Params/MACs are derived from the measured K=10 anchor "
                     "(docs/RECORD.md 2.1), verified against the 188-class state dict. "
                     "250 W is the measured draw on an RTX 3090 at 90-100% utilisation, not the "
                     "350 W board limit."),
        }
    out["train_loss_by_epoch"] = {e: v[0] for e, v in sorted(L["tloss"].items())}
    out["val_metric_by_epoch"] = dict(sorted(L["metric"].items()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--samples-per-epoch", type=int, default=10_240_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--write", action="store_true", help="store run_audit.json in each run dir")
    ap.add_argument("--full", action="store_true", help="print per-epoch curves too")
    a = ap.parse_args()
    rc = 0
    for r in a.runs:
        run = pathlib.Path(r)
        res = audit(run, a.samples_per_epoch, a.batch_size)
        if not a.full:
            res = {k: v for k, v in res.items() if not k.endswith("_by_epoch")}
        print(json.dumps(res, indent=1))
        if res["problems"]:
            rc = 1
        if a.write:
            full = audit(run, a.samples_per_epoch, a.batch_size)
            (run / "run_audit.json").write_text(json.dumps(full, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
