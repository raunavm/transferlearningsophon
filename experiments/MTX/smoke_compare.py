"""Compare v2 smoke runs (experiments/MTX/pretrain_v2.py output): stream hashes,
per-epoch training loss and validation metrics, the initial trunk and the last
common epoch's weights. Used for the GPU numerics check (audit 2026-09-29 item 6):
one pair is the test (a 3090 run against the same configuration on another GPU),
the others are references whose differences set the scale.

Run where /data is mounted:
  python3 experiments/MTX/smoke_compare.py /data/results/mtx_v2/smoke \\
      --test smoke-a:smoke-a-l40 --ref smoke-a:smoke-a2 --ref smoke-a:smoke-b ...

A ratio is the test pair's largest difference over epochs divided by the
reference pair's (None when the reference difference is zero).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

VAL_KEYS = ("loss", "acc", "head_top1_acc", "p_qcd_resonant", "p_qcd_qcd")


def _epochs(run: pathlib.Path, what: str) -> dict:
    return {int(p.stem.split("-")[1]): json.loads(p.read_text())
            for p in sorted((run / what).glob("epoch-*.json"))}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(a) if a else abs(a - b)


def compare(root, ref: str, other: str) -> dict:
    """Differences of run `other` from run `ref`, epoch by epoch."""
    root = pathlib.Path(root)
    am, bm = _epochs(root / ref, "metrics"), _epochs(root / other, "metrics")
    a_s, b_s = _epochs(root / ref, "stream"), _epochs(root / other, "stream")
    ep = sorted(set(am) & set(bm))
    if not ep:
        raise ValueError(f"{ref} and {other} share no epoch")
    keys = [k for k in VAL_KEYS if all(am[e]["val"].get(k) is not None and bm[e]["val"].get(k) is not None
                                       for e in ep)]
    r = {"ref": ref, "other": other, "epochs": ep,
         "devices": [am[ep[0]]["device"], bm[ep[0]]["device"]],
         "stream_equal": {e: e in a_s and e in b_s and a_s[e]["sha256"] == b_s[e]["sha256"] for e in ep},
         "train_loss": {e: [am[e]["train"]["loss"], bm[e]["train"]["loss"]] for e in ep},
         "train_loss_reldiff": {e: _rel(am[e]["train"]["loss"], bm[e]["train"]["loss"]) for e in ep},
         "val_reldiff": {e: {k: _rel(am[e]["val"][k], bm[e]["val"][k]) for k in keys} for e in ep}}
    r["val_reldiff_max"] = {e: max(r["val_reldiff"][e].values()) for e in ep}
    ia, ib = root / ref / "init_trunk.pt", root / other / "init_trunk.pt"
    if ia.exists() and ib.exists():
        ta = torch.load(ia, map_location="cpu")["trunk"]
        tb = torch.load(ib, map_location="cpu")["trunk"]
        r["init_trunk_bitwise_equal"] = ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta)
    last = ep[-1]
    sa = torch.load(root / ref / f"net_epoch-{last}_state.pt", map_location="cpu")
    sb = torch.load(root / other / f"net_epoch-{last}_state.pt", map_location="cpu")
    if sa.keys() != sb.keys() or any(sa[k].shape != sb[k].shape for k in sa):
        raise ValueError(f"{ref} and {other}: state dicts differ in keys or shapes")
    r["weights_epoch"] = last
    r["weights_max_abs_diff"] = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)
    return r


def _peak(r: dict) -> dict:
    return {"train_loss_reldiff": max(r["train_loss_reldiff"].values()),
            "val_reldiff_max": max(r["val_reldiff_max"].values()),
            "weights_max_abs_diff": r["weights_max_abs_diff"]}


def ratios(test: dict, ref: dict) -> dict:
    """Largest test difference over epochs / largest reference difference."""
    t, f = _peak(test), _peak(ref)
    return {k: (t[k] / f[k] if f[k] else None) for k in t}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--test", required=True, metavar="REF:OTHER")
    ap.add_argument("--ref", action="append", default=[], metavar="REF:OTHER")
    a = ap.parse_args(argv)
    test = compare(a.root, *a.test.split(":"))
    refs = {s: compare(a.root, *s.split(":")) for s in a.ref}
    print(json.dumps({"test": test, "test_peak": _peak(test),
                      "refs": {s: dict(r, peak=_peak(r)) for s, r in refs.items()},
                      "ratio_to_ref": {s: ratios(test, r) for s, r in refs.items()},
                      "stream_identical": all(test["stream_equal"].values())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
