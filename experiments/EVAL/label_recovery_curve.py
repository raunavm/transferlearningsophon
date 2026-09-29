#!/usr/bin/env python3
"""Label recovery as a learning curve: is 0.34 limited by data, by capacity, or by
the representation?

WHY THIS REPLACES label_recovery.py's PROTOCOL (audit 2026-09-29, B8). The v1
protocol fitted a logistic regression at C = 1 with no class weighting on
126,000 of the 2,000,000 cached jets -- about 670 per class on average and far
fewer for the rare ones -- and scored it by BALANCED accuracy. Two of the three
readings of "0.34 of the 188 native labels are recovered" were therefore never
excluded: that the probe had too few jets per class, and that an unweighted fit
spent its capacity on the common classes the metric weights least. The
converged 512-unit MLP beside it (0.3449 against 0.3433) excluded only the third,
capacity. Here:

  data       the probe is fitted on nested training sets of growing size, up to
             every jet not held out, so a curve still rising at the end says
             the number is data-limited and a flat one says it is not;
  weighting  the logistic regression is class-weighted (sklearn 'balanced'),
             matching the balanced accuracy it is scored by;
  capacity   a converged MLP (512 units, class-weighted loss, early stopping on
             the validation balanced accuracy) at the largest size, on the rungs
             given by --mlp-rungs.

The split is fixed and arm-independent (the caches are row-aligned; probe.
check_alignment gates it): test 20 %, validation 10 % (the MLP's early stopping;
the linear probe does not look at it), and the rest a training pool whose first
m jets, in one fixed random order, are the size-m training set. So each size's
set contains the smaller ones, and every arm sees the same jets at every size.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = pathlib.Path(__file__).resolve().parents[2]
SPLIT_SEED = 20260822
TEST_FRACTION, VAL_FRACTION = 0.2, 0.1
DEFAULT_SIZES = (14_000, 44_000, 140_000, 443_000, 0)   # 0 = the whole training pool
C = 1.0
LR_MAX_ITER = 1000
MLP = {"hidden": 512, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch": 4096,
       "max_epochs": 200, "lr_factor": 0.1, "lr_patience": 5, "stop_patience": 12,
       "min_gain": 1e-4, "seed": 0}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def split(n: int, sizes) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """(test, val, pool, sizes) with 0 in `sizes` meaning the whole pool."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    a, b = int(TEST_FRACTION * n), int((TEST_FRACTION + VAL_FRACTION) * n)
    te, va, pool = perm[:a], perm[a:b], perm[b:]
    got = sorted({pool.size if s in (0, None) else int(s) for s in sizes})
    if got[-1] > pool.size:
        raise SystemExit(f"FATAL: training size {got[-1]:,} exceeds the pool of {pool.size:,}")
    return te, va, pool, got


def fit_linear(Xtr, ytr, Xte, yte) -> dict:
    """Class-weighted multinomial logistic regression on standardised features."""
    sc = StandardScaler().fit(Xtr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.time()
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=LR_MAX_ITER)
        m.fit(sc.transform(Xtr), ytr)
        capped = any(issubclass(x.category, ConvergenceWarning) for x in w)
    pred = m.predict(sc.transform(Xte))
    return {"balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "accuracy": float((pred == yte).mean()),
            "n_iter": int(np.max(m.n_iter_)), "converged": not capped,
            "seconds": round(time.time() - t0, 1)}


def fit_mlp(Xtr, ytr, Xva, yva, Xte, yte, threads: int) -> dict:
    """512-unit MLP, class-weighted cross entropy, early stopping on validation
    balanced accuracy; `converged` means it stopped on the plateau, not the cap."""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(MLP["seed"])
    classes = np.unique(ytr)
    idx = {c: i for i, c in enumerate(classes)}
    enc = np.vectorize(idx.get)
    sc = StandardScaler().fit(Xtr)
    tr = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    va = torch.tensor(sc.transform(Xva), dtype=torch.float32)
    te = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(enc(ytr), dtype=torch.long)
    counts = np.bincount(enc(ytr), minlength=classes.size)
    wcls = torch.tensor(ytr.size / (classes.size * np.maximum(counts, 1)), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(tr.shape[1], MLP["hidden"]), torch.nn.ReLU(),
                              torch.nn.Dropout(MLP["dropout"]),
                              torch.nn.Linear(MLP["hidden"], classes.size))
    opt = torch.optim.AdamW(net.parameters(), lr=MLP["lr"], weight_decay=MLP["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=MLP["lr_factor"],
                                                       patience=MLP["lr_patience"])
    lossf = torch.nn.CrossEntropyLoss(weight=wcls)

    def predict(x):
        net.eval()
        with torch.no_grad():
            return classes[torch.cat([net(x[i:i + 65536]).argmax(1)
                                      for i in range(0, x.shape[0], 65536)]).numpy()]

    best, best_state, best_epoch, bad, curve, stopped = -1.0, None, 0, 0, [], False
    for epoch in range(MLP["max_epochs"]):
        net.train()
        perm = torch.randperm(tr.shape[0])
        for i in range(0, tr.shape[0], MLP["batch"]):
            j = perm[i:i + MLP["batch"]]
            opt.zero_grad()
            lossf(net(tr[j]), ytr_t[j]).backward()
            opt.step()
        v = float(balanced_accuracy_score(yva, predict(va)))
        curve.append(round(v, 6))
        sched.step(v)
        if v > best + MLP["min_gain"]:
            best, best_epoch, bad = v, epoch + 1, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP["stop_patience"]:
                stopped = True
                break
    net.load_state_dict(best_state)
    pred = predict(te)
    return {"balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "accuracy": float((pred == yte).mean()), "val_balanced_accuracy": best,
            "best_epoch": best_epoch, "epochs_run": len(curve), "converged": stopped,
            "val_curve": curve}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", nargs="+", required=True, help="arm=DIR ...")
    ap.add_argument("--own-rung", nargs="+", required=True, help="arm=RUNG ...")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES),
                    help="training sizes; 0 = every jet in the training pool")
    ap.add_argument("--rungs", nargs="+", default=None)
    ap.add_argument("--mlp-rungs", nargs="*", default=["L188"],
                    help="rungs that get the MLP capacity check at the largest size")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)

    lrm = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    rungs = a.rungs or list(lrm.RUNGS)
    own = dict(s.split("=", 1) for s in a.own_rung)
    arms = {}
    for s in a.features:
        name, d = s.split("=", 1)
        if name not in own:
            raise SystemExit(f"FATAL: no --own-rung for {name}")
        arms[name] = probe.load_arm(pathlib.Path(d))
    align = probe.check_alignment(arms)
    maps = lrm.rung_maps()
    n = next(iter(arms.values()))["L"].shape[0]
    te, va, pool, sizes = split(n, a.sizes)

    out_file = a.out / "label_recovery_curve.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else None
    if res and (res["row_alignment_sha256"] != align or res["sizes"] != sizes):
        raise SystemExit(f"FATAL: {out_file} was written for other jets or sizes")
    res = res or {"row_alignment_sha256": align, "n_jets": int(n), "split_seed": SPLIT_SEED,
                  "n_test": int(te.size), "n_val": int(va.size), "n_pool": int(pool.size),
                  "sizes": sizes, "linear": {"C": C, "class_weight": "balanced",
                                             "max_iter": LR_MAX_ITER, "solver": "lbfgs"},
                  "mlp": MLP, "mlp_rungs": a.mlp_rungs,
                  "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  "arms": {}}
    a.out.mkdir(parents=True, exist_ok=True)

    def flush():
        tmp = out_file.with_suffix(".partial")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(out_file)

    for arm, d in sorted(arms.items()):
        F = d["F"].astype(np.float32, copy=False)
        L = d["L"].astype(np.int64)
        ad = res["arms"].setdefault(arm, {"own_rung": own[arm], "rungs": {}})
        for rung in rungs:
            g = np.array([maps[rung][int(x)] for x in range(188)])[L]
            k = int(np.unique(g).size)
            if k < 2:
                ad["rungs"][rung] = {"skipped": "one group"}
                continue
            cell = ad["rungs"].setdefault(rung, {"n_groups": k, "chance": 1.0 / k, "curve": []})
            _, cnt = np.unique(g[te], return_counts=True)
            cell["chance_margin"] = lrm.chance_margin(k, cnt.tolist())
            done = {c["n_train"] for c in cell["curve"]}
            for m in sizes:
                if m in done:
                    continue
                tr = pool[:m]
                r = fit_linear(F[tr], g[tr], F[te], g[te])
                r["n_train"] = m
                r["min_train_per_group"] = int(np.bincount(g[tr], minlength=g.max() + 1)[
                    np.unique(g)].min())
                cell["curve"].append(r)
                cell["curve"].sort(key=lambda c: c["n_train"])
                flush()
                print(f"  {arm:12s} {rung:7s} K={k:3d} n={m:>9,} linear bal.acc "
                      f"{r['balanced_accuracy']:.4f} ({r['n_iter']} it, "
                      f"{'converged' if r['converged'] else 'CAPPED'}, {r['seconds']} s)", flush=True)
            if rung in a.mlp_rungs and "mlp" not in cell:
                tr = pool[:sizes[-1]]
                cell["mlp"] = fit_mlp(F[tr], g[tr], F[va], g[va], F[te], g[te], a.threads)
                cell["mlp"]["n_train"] = sizes[-1]
                flush()
                print(f"  {arm:12s} {rung:7s} MLP at n={sizes[-1]:,}: bal.acc "
                      f"{cell['mlp']['balanced_accuracy']:.4f} "
                      f"(linear {cell['curve'][-1]['balanced_accuracy']:.4f}; "
                      f"{cell['mlp']['epochs_run']} epochs, "
                      f"{'converged' if cell['mlp']['converged'] else 'CAPPED'})", flush=True)
    flush()
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
