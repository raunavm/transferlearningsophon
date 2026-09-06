#!/usr/bin/env python3
"""Assertions the fine-tuning jobs run in-pod, kept in one tested file rather
than as shell heredocs.

    head-width  --checkpoint C --expect W     the fc output layer is W wide
    load-log    --log FILE                    weaver's "Model initialized with
                                              weights" line lists NOTHING but
                                              head keys as missing/unexpected
    features    --dir D --n N --k K           extract_features wrote N rows of
                                              128-d features, N labels, and K
                                              logits
    pred        --file pred.root --k K --n N  weaver's predict wrote N entries
                                              with K score branches
    manifest    --out FILE key=value ...      write a fine-tune's manifest
                                              (git commit, weaver, GPU, args)

Every check exits non-zero with a sentence on failure; the job scripts run
under `set -e`, so a failed check stops the job at the point the evidence is.

THE load-log CHECK IS THE ONE THAT MATTERS. weaver loads --load-model-weights
with strict=False and only LOGS what did not match. A checkpoint from a
different key layout therefore leaves a randomly initialised trunk, trains it,
and reports a plausible accuracy. The log line is the only evidence, and this
turns it into a hard stop: anything missing or unexpected outside `mod.fc.`
(the head, which --exclude-model-weights drops on purpose) fails.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HEAD_PREFIX = "mod.fc."


def head_width(ckpt: str) -> int:
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    rows = []
    for k, v in sd.items():
        if k.startswith(HEAD_PREFIX) and k.endswith(".weight") and getattr(v, "ndim", 0) == 2:
            m = re.search(r"\.(\d+)\.weight$", k)
            rows.append((int(m.group(1)) if m else -1, int(v.shape[0])))
    if not rows:
        raise SystemExit(f"FAIL head-width: no {HEAD_PREFIX}*.weight in {ckpt}")
    return max(rows)[1]


def parse_load_log(text: str) -> tuple[list[str], list[str]]:
    """The two key lists from weaver's 'Model initialized with weights' block."""
    m = re.search(r"Model initialized with weights from .*?Missing: (\[.*?\]).*?Unexpected: (\[.*?\])",
                  text, re.S)
    if not m:
        raise SystemExit("FAIL load-log: no 'Model initialized with weights' block in the log "
                         "-- --load-model-weights did not run")
    import ast
    return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))


def check_load_log(path: str) -> None:
    missing, unexpected = parse_load_log(pathlib.Path(path).read_text(errors="replace"))
    bad_m = [k for k in missing if not k.startswith(HEAD_PREFIX)]
    bad_u = [k for k in unexpected if not k.startswith(HEAD_PREFIX)]
    if bad_m or bad_u:
        raise SystemExit(f"FAIL load-log: trunk keys did not load -- missing {bad_m[:5]} "
                         f"unexpected {bad_u[:5]} ({len(bad_m)} / {len(bad_u)} outside the head). "
                         f"The trunk would train from random init.")
    if not missing:
        raise SystemExit("FAIL load-log: nothing was excluded -- the head was loaded too, so "
                         "--exclude-model-weights did not match mod.fc.*")
    print(f"PASS load-log: trunk loaded; head re-initialised "
          f"({len(missing)} head tensors missing, {len(unexpected)} unexpected, all under {HEAD_PREFIX})")


def check_features(d: str, n: int, k: int) -> None:
    import numpy as np
    d = pathlib.Path(d)
    F = np.load(d / "features.npy")
    L = np.load(d / "label188.npy")
    G = np.load(d / "logits.npy")
    if F.shape != (n, 128) or L.shape != (n,) or G.shape != (n, k):
        raise SystemExit(f"FAIL features: shapes {F.shape} {L.shape} {G.shape}, want ({n},128) ({n},) ({n},{k})")
    if not np.isfinite(F).all() or not np.isfinite(G).all():
        raise SystemExit("FAIL features: non-finite values")
    print(f"PASS features: {n} jets, 128-d, {k} logits, finite")


def check_pred(path: str, k: int, n: int) -> None:
    import uproot
    with uproot.open(path) as f:
        tree = f[[x for x in f.keys() if not x.startswith("_")][0]]
        scores = [b for b in tree.keys() if b.startswith("score_")]
        if len(scores) != k:
            raise SystemExit(f"FAIL pred: {len(scores)} score_* branches, want {k}: {scores}")
        if tree.num_entries != n:
            raise SystemExit(f"FAIL pred: {tree.num_entries} entries, want {n}")
    print(f"PASS pred: {n} entries, {k} score branches")


def write_manifest(out: str, kv: list[str]) -> None:
    rec = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for item in kv:
        key, _, val = item.partition("=")
        rec[key] = val
    try:
        rec["repo_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True, timeout=10).stdout.strip()
    except Exception:
        rec["repo_commit"] = None
    try:
        import weaver
        rec["weaver_version"] = getattr(weaver, "__version__", None)
    except Exception:
        rec["weaver_version"] = None
    try:
        import torch
        rec["torch_version"] = torch.__version__
        rec["gpu_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        rec["torch_version"] = rec["gpu_device_name"] = None
    rec["node_name"] = os.environ.get("NODE_NAME")
    rec["gpu_product_pin"] = os.environ.get("GPU_PRODUCT")
    ck = rec.get("checkpoint")
    if ck and os.path.exists(ck):
        import hashlib
        rec["checkpoint_sha256"] = hashlib.sha256(pathlib.Path(ck).read_bytes()).hexdigest()
    sub = rec.get("subset")
    if sub and os.path.exists(sub):
        rec["subset_bytes"] = os.path.getsize(sub)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("head-width")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--expect", type=int, required=True)
    b = sub.add_parser("load-log")
    b.add_argument("--log", required=True)
    c = sub.add_parser("features")
    c.add_argument("--dir", required=True)
    c.add_argument("--n", type=int, required=True)
    c.add_argument("--k", type=int, required=True)
    d = sub.add_parser("pred")
    d.add_argument("--file", required=True)
    d.add_argument("--k", type=int, required=True)
    d.add_argument("--n", type=int, required=True)
    e = sub.add_parser("manifest")
    e.add_argument("--out", required=True)
    e.add_argument("kv", nargs="*")
    args = ap.parse_args(argv)

    if args.cmd == "head-width":
        w = head_width(args.checkpoint)
        if w != args.expect:
            raise SystemExit(f"FAIL head-width: {w}, want {args.expect} ({args.checkpoint})")
        print(f"PASS head-width: {w}")
    elif args.cmd == "load-log":
        check_load_log(args.log)
    elif args.cmd == "features":
        check_features(args.dir, args.n, args.k)
    elif args.cmd == "pred":
        check_pred(args.file, args.k, args.n)
    elif args.cmd == "manifest":
        write_manifest(args.out, args.kv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
