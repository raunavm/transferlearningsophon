#!/usr/bin/env python3
"""Run weaver's own loader over the staged benchmark sets and assert the tensor.

scripts/stage_downstream.py validated the CONVERSION -- jet counts, the measured
deta/dphi convention, the R=0.4 vs R=0.8 angular signature. This validates the
thing that actually reaches the model: what weaver produces when it reads
configs/finetune/*.yaml. Every check below has a silent failure mode, i.e. one
where training runs, the loss falls, and a number gets published anyway.

  1 LABELS ARE REAL. weaver's `type: simple` reduces the listed indicator
    branches with np.argmax (utils/data/config.py:105-110) and exposes the
    result as `_label_`. A one-element list argmaxes over a width-1 axis and
    hands EVERY jet class 0; the run would reach 100 % accuracy and the
    benchmark row would be meaningless. Asserts both classes are populated.
  2 THE FILL IS EXACTLY ZERO after standardization, not merely small. A
    non-zero subtract_by on a filled slot would silently make the fill a
    constant offset the paper does not describe.
  3 UNFILLED COLUMNS ARE FINITE. logptrel and logerel are logs of reconstructed
    ratios; one zero-energy constituent surviving the mask gives -inf.
  4 THE MASKING CONTROL DIFFERS FROM PLAIN JETCLASS-II IN EXACTLY THE FILLED
    COLUMNS. Loaded over the SAME jets and diffed column by column. This is the
    property the entire zero-fill control rests on, asserted on real tensors
    rather than on the YAML that tests/test_downstream_fill.py already checks.
  5 A PRETRAINED TRUNK LOADS INTO A 2-CLASS MODEL with the head excluded and a
    forward pass runs on a real batch.

Run:  python3 experiments/FT/loadcheck.py --out /data/results/ft/loadcheck/loadcheck.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = ROOT / "experiments" / "MTX" / "ParT_sophon_arch_mtx.py"
ZERO_PREFIX = "part_zero_"

CASES = [
    dict(name="TopReference", config="configs/finetune/TopReference.yaml",
         files="/data/finetune/top/top_test.parquet", n_filled=10),
    dict(name="EnergyFlowQG", config="configs/finetune/EnergyFlowQG.yaml",
         files="/data/finetune/qg/qg_chunk0.parquet", n_filled=4),
]
JC2 = sorted(glob.glob("/jc2/jet_data/QCD_028*.parquet"))[:1]
CONTROLS = [
    ("TopReference", "configs/finetune/JetClassII_L162_maskTopReference.yaml", 10),
    ("EnergyFlowQG", "configs/finetune/JetClassII_L162_maskEnergyFlowQG.yaml", 4),
]
PLAIN_JC2 = "configs/finetune/JetClassII_L162_noweight.yaml"


def load_arch():
    spec = importlib.util.spec_from_file_location("_arch", ARCH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def one_batch(config: str, files: list, batch: int = 512):
    """First batch, deterministically: one worker, file order, no resampling."""
    from weaver.utils.dataset import SimpleIterDataset
    from weaver.utils.data.config import DataConfig
    dc = DataConfig.load(config, load_observers=False)
    ds = SimpleIterDataset({"_": list(files)}, config, for_training=False,
                           fetch_by_files=True, fetch_step=1, name="loadcheck")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, drop_last=False,
                                     num_workers=0)
    X, y, _ = next(iter(dl))
    return dc, X, y


def filled_indices(dc):
    return [i for i, v in enumerate(dc.input_dicts["pf_features"]) if v.startswith(ZERO_PREFIX)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint",
                    default="/data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt")
    a = ap.parse_args()
    res, fail = {}, []

    for c in CASES:
        print(f"\n===== {c['name']} =====", flush=True)
        files = sorted(glob.glob(c["files"]))
        if not files:
            fail.append(f"{c['name']}: no files at {c['files']}"); continue
        dc, X, y = one_batch(c["config"], files)
        feats = X["pf_features"].numpy()          # (N, C, P)
        mask = X["pf_mask"].numpy().astype(bool)  # (N, 1, P)
        lab = y["_label_"].numpy()
        idx = filled_indices(dc)
        real = np.broadcast_to(mask, feats.shape)

        # 1 -- labels
        uniq, counts = np.unique(lab, return_counts=True)
        ok_lab = len(uniq) >= 2
        print(f"  labels: {dict(zip(uniq.tolist(), counts.tolist()))}  -> {'OK' if ok_lab else 'FAIL'}")
        if not ok_lab:
            fail.append(f"{c['name']}: only class {uniq.tolist()} in a {len(lab)}-jet batch")

        # 2 -- the fill
        ok_n = len(idx) == c["n_filled"]
        filled_max = float(np.abs(feats[:, idx, :]).max()) if idx else 0.0
        ok_zero = filled_max == 0.0
        print(f"  filled slots: {len(idx)} (expect {c['n_filled']})  "
              f"max|value| = {filled_max:.3e}  -> {'OK' if ok_n and ok_zero else 'FAIL'}")
        if not ok_n:
            fail.append(f"{c['name']}: {len(idx)} filled slots, expected {c['n_filled']}")
        if not ok_zero:
            fail.append(f"{c['name']}: filled slots reach {filled_max}, not a zero-fill")

        # 3 -- the rest is finite ON REAL PARTICLES (padding is free to be anything)
        keep = [i for i in range(feats.shape[1]) if i not in idx]
        vals = feats[:, keep, :][real[:, keep, :]]
        ok_fin = bool(np.isfinite(vals).all())
        print(f"  unfilled columns finite on real particles: {ok_fin}  "
              f"(range {vals.min():.3f} .. {vals.max():.3f})")
        if not ok_fin:
            fail.append(f"{c['name']}: non-finite values in unfilled columns")

        res[c["name"]] = dict(n_jets=int(len(lab)), labels=dict(
            zip(uniq.tolist(), counts.tolist())), filled_slots=idx,
            filled_max_abs=filled_max, unfilled_finite=ok_fin,
            unfilled_min=float(vals.min()), unfilled_max=float(vals.max()))

    # 4 -- the control, on the SAME JetClass-II jets
    print(f"\n===== masking control (JetClass-II) =====", flush=True)
    if not JC2:
        fail.append("no JetClass-II file found for the control")
    else:
        dcp, Xp, yp = one_batch(PLAIN_JC2, JC2)
        base = Xp["pf_features"].numpy()
        for name, cfg, n_exp in CONTROLS:
            dcm, Xm, ym = one_batch(cfg, JC2)
            got = Xm["pf_features"].numpy()
            if base.shape != got.shape:
                fail.append(f"control {name}: shape {got.shape} != plain {base.shape}"); continue
            same_jets = bool((yp["_label_"].numpy() == ym["_label_"].numpy()).all())
            idx = filled_indices(dcm)
            differs = sorted({int(i) for i in np.where(
                np.abs(base - got).max(axis=(0, 2)) > 0)[0]})
            ok = (differs == sorted(idx)) and len(idx) == n_exp and same_jets
            print(f"  {name}: masked {idx}  columns that actually differ {differs}  "
                  f"same jets {same_jets}  -> {'OK' if ok else 'FAIL'}")
            if not ok:
                fail.append(f"control {name}: declared {sorted(idx)} but tensors differ in {differs}"
                            + ("" if same_jets else "; and the two loads saw different jets"))
            res[f"control_{name}"] = dict(masked=idx, differs=differs,
                                          same_jets=same_jets, n_expected=n_exp)

    # 5 -- a pretrained trunk into a 2-class head
    print(f"\n===== checkpoint into a 2-class model =====", flush=True)
    ckpt = pathlib.Path(a.checkpoint)
    if not ckpt.exists():
        fail.append(f"checkpoint missing: {ckpt}")
    else:
        arch = load_arch()
        files = sorted(glob.glob(CASES[0]["files"]))
        dc, X, _y = one_batch(CASES[0]["config"], files, batch=32)
        model, _ = arch.get_model(dc, num_classes=2, fc_params=[(512, 0.1)])
        sd = torch.load(ckpt, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("mod.fc.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        ok_load = all(k.startswith("mod.fc.") for k in missing) and not unexpected
        model.eval()
        with torch.no_grad():
            out = model(*[X[n] for n in dc.input_names])
        ok_fwd = tuple(out.shape) == (X["pf_features"].shape[0], 2) and bool(torch.isfinite(out).all())
        print(f"  loaded {len(sd)} trunk tensors; missing outside the head: "
              f"{[k for k in missing if not k.startswith('mod.fc.')]}; unexpected: {list(unexpected)}")
        print(f"  forward: {tuple(out.shape)} finite={bool(torch.isfinite(out).all())} "
              f"-> {'OK' if ok_load and ok_fwd else 'FAIL'}")
        if not ok_load:
            fail.append(f"checkpoint: missing {missing}, unexpected {unexpected}")
        if not ok_fwd:
            fail.append(f"checkpoint: forward gave {tuple(out.shape)}")
        res["checkpoint"] = dict(path=str(ckpt), trunk_tensors=len(sd),
                                 loads_clean=ok_load, forward_ok=ok_fwd,
                                 out_shape=list(out.shape))

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res["failures"] = fail
    out.write_text(json.dumps(res, indent=2, default=str))
    print("\n" + ("ALL LOADCHECKS PASSED" if not fail else "FAILED:\n  " + "\n  ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
