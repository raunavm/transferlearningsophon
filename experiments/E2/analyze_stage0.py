#!/usr/bin/env python3
"""E2 Stage-0 LR selection (PLAN §9.2).

Selection rule (FROZEN in the estimand sheet): for each arm, choose the start-LR
whose Stage-0 run has the lower **final-checkpoint validation loss** — the loss
reported for the last training epoch (not best-val; the primary estimand is
fixed-budget, §9.6). Stage-0 runs are quarantined (stage:tune, P1) — this script
only picks the LR that Stage-1 will use; it touches no inferential sample.

Layout produced by the Stage-0 template:
  <root>/stage0_hi/<arm>_s<seed>/train.log     (hi = 5e-4)
  <root>/stage0_lo/<arm>_s<seed>/train.log     (lo = 2.5e-4)
  seed = 900 + arm_index, arms ordered [g2, g10sem, g10rand, g30, g188].

Output: <root>/stage0_lr_selection.json  (consumed by render_stage1.py).

The weaver val-loss line format is version-dependent; this parser tries several
patterns, REQUIRES a full epoch count (fails loud on a crashed/partial run), and
prints the parsed (epoch, val_loss) trace so it can be eyeballed against the
first real Stage-0 log before any Stage-1 job is rendered.

Usage:
  python3 experiments/E2/analyze_stage0.py --root /data/results/e2 [--expect-epochs 80]
  python3 experiments/E2/analyze_stage0.py --root /data/results/e2 --allow-partial   # debug only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ARMS = ["g2", "g10sem", "g10rand", "g30", "g188"]  # §9.1 order → seed 900+idx
LR = {"hi": "5e-4", "lo": "2.5e-4"}                 # §9.2 grid {η0, η0/2}

# Ordered val-loss patterns (most specific first). weaver logs an eval line once
# per validated epoch; we anchor epochs on "Epoch #N validating" and take the
# first loss-bearing line after each anchor.
_EPOCH_ANCHOR = re.compile(r"Epoch #(\d+) validating")
_VAL_LOSS_PATTERNS = [
    re.compile(r"(?:Eval|Valid|Validation)\s+AvgLoss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"),
    re.compile(r"[Cc]urrent validation loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"),
    re.compile(r"validation.*?loss[:= ]+\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.I),
]


def parse_val_curve(log: Path) -> list[tuple[int, float]]:
    """Return [(epoch, val_loss), ...] in epoch order. Empty if unparseable."""
    text = log.read_text(errors="replace").splitlines()
    curve: list[tuple[int, float]] = []
    cur_epoch = None
    for line in text:
        m = _EPOCH_ANCHOR.search(line)
        if m:
            cur_epoch = int(m.group(1))
            continue
        if cur_epoch is None:
            continue
        for pat in _VAL_LOSS_PATTERNS:
            mm = pat.search(line)
            if mm:
                curve.append((cur_epoch, float(mm.group(1))))
                cur_epoch = None  # one loss per validating block
                break
    # de-dup by epoch keeping the last reading, then sort
    by_epoch = dict(curve)
    return sorted(by_epoch.items())


def final_val_loss(run_dir: Path, expect_epochs: int, allow_partial: bool):
    log = run_dir / "train.log"
    if not log.exists():
        return None, f"no train.log at {run_dir}"
    curve = parse_val_curve(log)
    if not curve:
        return None, f"no validation-loss lines parsed in {log} (check weaver log format!)"
    n = len(curve)
    last_epoch, last_loss = curve[-1]
    # §9.2 wants the FINAL-checkpoint (last epoch) loss; guard against partial runs.
    if not allow_partial and n < expect_epochs:
        return None, (f"only {n} validated epochs (< {expect_epochs}); run incomplete "
                      f"or parser missed lines. last=({last_epoch},{last_loss}). "
                      f"trace: {curve}")
    return {"final_epoch": last_epoch, "final_val_loss": last_loss,
            "n_val_epochs": n, "curve": curve}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/results/e2",
                    help="dir holding stage0_hi/ and stage0_lo/")
    ap.add_argument("--expect-epochs", type=int, default=80)
    ap.add_argument("--allow-partial", action="store_true",
                    help="DEBUG ONLY: select from an incomplete run (never for the real pick)")
    ap.add_argument("--out", default=None, help="output json (default <root>/stage0_lr_selection.json)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "stage0_lr_selection.json"
    result = {"selection_metric": "final-checkpoint validation loss (PLAN §9.2)",
              "lr_grid": LR, "expect_epochs": args.expect_epochs,
              "allow_partial": args.allow_partial, "arms": {}}
    errors = []

    for idx, arm in enumerate(ARMS):
        seed = 900 + idx
        entry = {"seed": seed}
        losses = {}
        for tag in ("hi", "lo"):
            run_dir = root / f"stage0_{tag}" / f"{arm}_s{seed}"
            info, err = final_val_loss(run_dir, args.expect_epochs, args.allow_partial)
            print(f"[{arm} {tag} lr={LR[tag]}] ", end="")
            if err:
                print(f"UNAVAILABLE: {err}")
                errors.append(f"{arm}/{tag}: {err}")
                entry[f"{tag}_val_loss"] = None
            else:
                print(f"final(epoch {info['final_epoch']}) val_loss={info['final_val_loss']:.6f} "
                      f"[{info['n_val_epochs']} epochs]")
                entry[f"{tag}_val_loss"] = info["final_val_loss"]
                losses[tag] = info["final_val_loss"]
        if len(losses) == 2:
            chosen = min(losses, key=losses.get)
            entry["chosen_tag"] = chosen
            entry["chosen_lr"] = LR[chosen]
            entry["margin"] = round(abs(losses["hi"] - losses["lo"]), 6)
            print(f"  -> {arm}: CHOOSE {chosen} (lr={LR[chosen]}), "
                  f"margin={entry['margin']:.6f}")
        else:
            entry["chosen_tag"] = None
            entry["chosen_lr"] = None
            print(f"  -> {arm}: CANNOT SELECT (missing a run)")
        result["arms"][arm] = entry

    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    if errors:
        print(f"\n{len(errors)} arm/LR run(s) unavailable — NOT ready to render Stage-1:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    if any(a["chosen_lr"] is None for a in result["arms"].values()):
        sys.exit(1)
    print("\nAll 5 arms selected. Ready: render_stage1.py --selection", out)


if __name__ == "__main__":
    main()
