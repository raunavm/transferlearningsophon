#!/usr/bin/env python3
"""How far fine-tuning moved the trunk, per learning rate. Phase 0, part B.

THE QUESTION. The LR sweep (DECISIONS_PENDING item 22, settled by ft-lrprobe3)
found that top-tagging accuracy rises with the trunk rate and then PLATEAUS from
3e-3 to 3e-2 at ~0.930, with the two arms converging to within 0.00006 at the
top. Item 32 asks whether that plateau is simply THE REGIME WHERE PRETRAINING
HAS BEEN OVERWRITTEN -- if by 3e-3 the trunk has moved so far that nothing of
the pretrained solution survives, then every arm is effectively training from
scratch, which is exactly why they stop differing. That would make the plateau
uninformative about vocabulary and would mean no arm ordering may be read from
it, for a reason quite separate from the protocol mismatch item 32 already found.

WHAT THIS MEASURES. ||theta_ft - theta_pre|| over the TRUNK ONLY, against
||theta_pre||, per cell. The head is excluded and must be: weaver re-initialises
it (`--exclude-model-weights 'mod\\.fc\\..*'`) and its shape differs between the
pretraining vocabulary and the 2-class top task, so a "displacement" there is
not a displacement at all.

The scale-free quantity is the RELATIVE displacement ||d(theta)|| / ||theta_pre||.
Absolute norms are reported too, because a ratio alone cannot distinguish "the
trunk barely moved" from "the trunk was tiny to begin with".

WHAT IT CANNOT SETTLE. Parameter distance is necessary, not sufficient: a small
||d(theta)|| guarantees the function is close, but a large one does not prove it
is far, since networks have flat directions. A CKA measurement on the
representations is the sufficient version and is the remaining half of Phase 0.
So a HIGH relative displacement at the plateau is suggestive; a LOW one would be
decisive in the other direction, refuting the overwrite story outright.

Run:  python3 experiments/FT/phase0_dtheta.py --pre A.pt --post B.pt
      python3 experiments/FT/phase0_dtheta.py --grid /data/results/ft --out o.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

HEAD = re.compile(r"^mod\.fc\.")
# BatchNorm running statistics. THEY MUST BE COUNTED SEPARATELY, and finding out
# why is the main thing this script learned. Lumped in with the weights, the
# total relative displacement looks FLAT at ~0.022 from lr 3e-5 to 1e-3 -- a 33x
# range over which it should have scaled -- which reads like "the trunk is
# pinned". It is an artefact: BN buffers are updated by MOMENTUM on every forward
# pass, entirely independently of the learning rate, so they re-estimate on the
# new data distribution even at lr 3e-5. Measured, they move ~15% at EVERY rate
# and account for 99.4% of the apparent displacement at the bottom of the grid
# and 12% at the top. The weights -- the thing "overwriting pretraining" would
# have to mean -- move a clean, monotone 0.0018 -> 0.0825 across the same range.
BUFFER = re.compile(r"running_(mean|var)$")


def trunk_tensors(sd: dict) -> dict:
    """Float trunk tensors only.

    Three exclusions, each for its own reason: the head is re-initialised so its
    distance is meaningless; integer buffers (`num_batches_tracked`) are counters,
    not parameters, and would add a large integer to a float norm; and anything
    whose shape differs between the two checkpoints cannot be subtracted at all.
    """
    out = {}
    for k, v in sd.items():
        if HEAD.match(k):
            continue
        if not hasattr(v, "dtype") or not getattr(v.dtype, "is_floating_point", False):
            continue
        out[k] = v
    return out


def load(path: str) -> dict:
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("model", sd) if isinstance(sd, dict) else sd


def displacement(pre: dict, post: dict) -> dict:
    """||d(theta)||, ||theta_pre|| and their ratio over the shared trunk."""
    a, b = trunk_tensors(pre), trunk_tensors(post)
    shared = [k for k in a if k in b and tuple(a[k].shape) == tuple(b[k].shape)]
    if not shared:
        raise SystemExit("FATAL: no shared trunk tensors; are these the same architecture?")
    skipped = sorted(set(a) ^ set(b))
    part = {"weight": [0.0, 0.0], "buffer": [0.0, 0.0]}
    per_key = {}
    for k in shared:
        dk = float((b[k].float() - a[k].float()).pow(2).sum())
        pk = float(a[k].float().pow(2).sum())
        acc = part["buffer" if BUFFER.search(k) else "weight"]
        acc[0] += dk
        acc[1] += pk
        per_key[k] = {"d": math.sqrt(dk), "pre": math.sqrt(pk),
                      "rel": math.sqrt(dk / pk) if pk > 0 else float("nan")}

    def rel(d2, p2):
        return math.sqrt(d2 / p2) if p2 > 0 else float("nan")
    dw, pw = part["weight"]
    db, pb = part["buffer"]
    d2, p2 = dw + db, pw + pb
    return {"n_tensors": len(shared), "skipped": skipped,
            "d_theta": math.sqrt(d2), "theta_pre": math.sqrt(p2),
            "relative": rel(d2, p2),
            # the two that matter separately; see BUFFER above
            "relative_weight": rel(dw, pw),
            "relative_buffer": rel(db, pb),
            "buffer_share_of_d2": db / d2 if d2 > 0 else float("nan"),
            "per_tensor": per_key}


CELL = re.compile(r"^(?P<arm>[a-z0-9-]+)_lr(?P<lr>[0-9.e+-]+)$")


def grid(root: pathlib.Path, mtx: pathlib.Path) -> list[dict]:
    """Every lrprobe cell under `root`, against its own arm's epoch-79 init."""
    rows = []
    for d in sorted(root.glob("lrprobe*/*_lr*")):
        m = CELL.match(d.name)
        if not m or not (d / "net_best_epoch_state.pt").exists():
            continue
        init = mtx / f"mtx-{m['arm']}" / "net_epoch-79_state.pt"
        if not init.exists():
            print(f"skip {d.name}: no {init}")
            continue
        r = displacement(load(str(init)), load(str(d / "net_best_epoch_state.pt")))
        r.pop("per_tensor")
        rows.append({"probe": d.parent.name, "arm": m["arm"],
                     "lr": float(m["lr"]), **r})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre")
    ap.add_argument("--post")
    ap.add_argument("--grid")
    ap.add_argument("--mtx", default="/data/results/mtx")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    if a.grid:
        rows = grid(pathlib.Path(a.grid), pathlib.Path(a.mtx))
        rows.sort(key=lambda r: (r["arm"], r["lr"]))
        print(f"{'arm':<12} {'lr':>8} {'rel(total)':>11} {'rel(WEIGHT)':>12} "
              f"{'rel(bn buf)':>12} {'buf share':>10}")
        for r in rows:
            print(f"{r['arm']:<12} {r['lr']:>8.0e} {r['relative']:>11.4f} "
                  f"{r['relative_weight']:>12.4f} {r['relative_buffer']:>12.4f} "
                  f"{r['buffer_share_of_d2']*100:>9.1f}%")
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=2))
            print(f"wrote {a.out}")
        return 0

    if not (a.pre and a.post):
        raise SystemExit("need --grid, or both --pre and --post")
    r = displacement(load(a.pre), load(a.post))
    per = r.pop("per_tensor")
    print(json.dumps(r, indent=2))
    top = sorted(per.items(), key=lambda kv: -kv[1]["rel"])[:10]
    print("\nlargest relative movement:")
    for k, v in top:
        print(f"  {k:<52} {v['rel']:.4f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({**r, "per_tensor": per}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
