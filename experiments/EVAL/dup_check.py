#!/usr/bin/env python3
"""Are near-identical results from different runs really different models? (plan item 2.1)

A mechanical scan of every multi-run cell (relative spread below a tenth of its column's median)
flagged a few cells whose runs agree far better than run-to-run variation. Agreement between two
pretraining runs at that level would mean a pipeline fault: the same features, checkpoint or scores
read for both. For each flagged pair this reads the saved per-jet outputs of both runs and reports
their sha256, the fraction of identical values, max |difference| and the Spearman correlation, and
recomputes each run's AUC from its scores (so the table value is re-derived, not re-read).

It also refits the nonlinear X->bc probe (bc_vs_bq_cs), whose three 188-class runs agree to
1.3e-6 in AUC: for 188-s1..s3 and 162-s1..s3, with the original MLP seeds (0, 1, 2; the stored
fit must reproduce) and with new seeds (3, 4, 5). If the AUC moves by much more than 1e-6 when
only the MLP seeds change, the three-run agreement is a coincidence of runs that differ; the
between-run spread is then measured with a paired bootstrap over test jets.

    python3 experiments/EVAL/dup_check.py --out /data/results/eval/dup_check/dup_check.json
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
LP = pathlib.Path("/data/results/eval/v2_linprobe")
PROBS = LP / "fits_v3_probs"
FT = pathlib.Path("/data/results/ft_v2/best70")
# (what, run a, run b, file under each run): the cells the scan flagged (2026-10-31)
PROBE_PAIRS = [
    ("F1R JetClass-II 1e6", "mtx-flavf1r-s1", "mtx-flavf1r-s2", "best70/jc2_188_N1000000.npz"),
    ("F1R QCD c vs light", "mtx-flavf1r-s1", "mtx-flavf1r-s2", "best70/jc2_188_pairs_qcd_c_vs_light_N1000000.npz"),
    ("17+mass Herwig 1e5", "mtx-r16q1mass-s1", "mtx-r16q1mass-s2", "best70/qg_herwig_N100000.npz"),
    ("17+mass Herwig full", "mtx-r16q1mass-s1", "mtx-r16q1mass-s2", "best70/qg_herwig_N1600000.npz"),
    ("43 QCD bb vs cc s1/s2", "mtx-r42q1-s1", "mtx-r42q1-s2", "best70/jc2_188_pairs_qcd_bb_vs_cc_N1000000.npz"),
    ("43 QCD bb vs cc s2/s3", "mtx-r42q1-s2", "mtx-r42q1-s3", "best70/jc2_188_pairs_qcd_bb_vs_cc_N1000000.npz"),
]
FT_PAIRS = [
    ("F1 q/g full FT", "leg_qg/flavf1-s1/N1600000/s1", "leg_qg/flavf1-s2/N1600000/s1"),
    ("17+mass q/g 1e4 FT", "leg_qg/r16q1mass-s1/N10000/s1", "leg_qg/r16q1mass-s2/N10000/s1"),
    ("F1R q/g full FT", "leg_qg/flavf1r-s1/N1600000/s1", "leg_qg/flavf1r-s2/N1600000/s1"),
]
MLP_MODELS = ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3", "mtx-l162-s1", "mtx-l162-s2", "mtx-l162-s3"]


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    n = min(a.size, 2_000_000)
    idx = np.random.default_rng(0).choice(a.size, n, replace=False) if a.size > n else slice(None)
    return {"sha_a": sha(a), "sha_b": sha(b), "n": int(a.size), "same_shape": a.shape == b.shape,
            "frac_identical": float(np.mean(a == b)), "max_abs_diff": float(np.max(np.abs(a - b))),
            "spearman": float(spearmanr(a[idx], b[idx]).correlation)}


def binary_auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def probe_pair(what, ra, rb, rel) -> dict:
    za, zb = np.load(PROBS / ra / rel), np.load(PROBS / rb / rel)
    key = "s" if "s" in za else "p"
    rec = {"what": what, "a": ra, "b": rb, "file": rel, "scores": key,
           "rows_equal": bool(np.array_equal(za["rows"], zb["rows"])),
           "labels_equal": bool(np.array_equal(za["y"], zb["y"])),
           "coef_equal": bool(np.array_equal(za["coef"], zb["coef"])),
           "C_a": float(za["C"]), "C_b": float(zb["C"]), **compare(za[key], zb[key])}
    if key == "s":
        rec["auc_a"], rec["auc_b"] = binary_auc(za["y"], za["s"]), binary_auc(zb["y"], zb["s"])
    return rec


def ft_pair(what, ca, cb) -> dict:
    out = {"what": what, "a": ca, "b": cb}
    la, lb = np.load(FT / ca / "features" / "logits.npy"), np.load(FT / cb / "features" / "logits.npy")
    ya = np.load(FT / ca / "features" / "label188.npy")
    out.update(compare(la, lb))
    ia = json.loads((FT / ca / "init_checkpoint.json").read_text())
    ib = json.loads((FT / cb / "init_checkpoint.json").read_text())
    out["init_sha_a"], out["init_sha_b"] = ia.get("sha256", "")[:16], ib.get("sha256", "")[:16]
    out["state_sha_a"] = hashlib.sha256((FT / ca / "net_best_epoch_state.pt").read_bytes()).hexdigest()[:16]
    out["state_sha_b"] = hashlib.sha256((FT / cb / "net_best_epoch_state.pt").read_bytes()).hexdigest()[:16]
    if la.ndim == 2 and la.shape[1] == 2:
        y = (ya == 1).astype(int) if set(np.unique(ya)) <= {0, 1} else ya
        sa = la[:, 1] - la[:, 0]
        sb = lb[:, 1] - lb[:, 0]
        out["auc_a"], out["auc_b"] = binary_auc(y, sa), binary_auc(y, sb)
    return out


def mlp_refit(lp, helpers, model: str, seeds) -> tuple:
    """The bc_vs_bq_cs nonlinear probe of `model` with MLP `seeds`, as mlp_cells fits it."""
    eval_arm, probe, l162, names = helpers
    md = LP / model
    xva, yva, _ = lp.load_dataset(md, "jc2_188", "val", "best70", l162, pathlib.Path(lp.JC2_PARQUET))
    xte, yte, _ = lp.load_dataset(md, "jc2_188", "test", "best70", l162, pathlib.Path(lp.JC2_PARQUET))
    n = max(lp.train_sizes(md, "jc2"))
    xtr, ytr, _ = lp.load_dataset(md, "jc2_188", f"train_N{n}", "best70", l162, pathlib.Path(lp.JC2_PARQUET))
    sig, bkg = lp.native_pair_groups(names)["bc_vs_bq_cs"]
    mtr, mva, mte = (np.isin(y, sig + bkg) for y in (ytr, yva, yte))
    b = lambda y: np.isin(y, sig).astype(np.int64)
    score, meta = probe.fit_mlp(xtr[mtr], b(ytr[mtr]), xva[mva], b(yva[mva]), xte[mte], seeds=seeds)
    return score, b(yte[mte]), np.flatnonzero(mte), meta


def paired_bootstrap(y, sa, sb, n_boot=200, seed=0) -> dict:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return {"delta": float(roc_auc_score(y, sa) - roc_auc_score(y, sb)), "sd": float(np.std(d, ddof=1)),
            "n_boot": n_boot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    t0 = time.time()
    doc = {"script": "experiments/EVAL/dup_check.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    doc["probe_pairs"] = [probe_pair(*p) for p in PROBE_PAIRS]
    doc["ft_pairs"] = [ft_pair(*p) for p in FT_PAIRS]
    for r in doc["probe_pairs"] + doc["ft_pairs"]:
        print(json.dumps(r), flush=True)

    lp = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    helpers = (_load("eval_arm", "experiments/EVAL/eval_arm.py"),
               _load("probe", "experiments/EVAL/probe.py"), lr.rung_maps()["L162"], names)
    stored = {m: json.loads((REPO / "experiments/FIGS/data/v2/linprobe/fits_v3_mlp2" / f"{m}.json").read_text())
              ["cells"]["best70"]["bc_vs_bq_cs"]["auc"] for m in MLP_MODELS}
    mlp, scores = {}, {}
    for m in MLP_MODELS:
        for tag, seeds in (("orig", (0, 1, 2)), ("new", (3, 4, 5))):
            s, y, rows, meta = mlp_refit(lp, helpers, m, seeds)
            scores[(m, tag)] = (s, y, rows)
            mlp[f"{m}/{tag}"] = {"auc": binary_auc(y, s), "seeds": list(seeds), "n_test": int(y.size),
                                 "rows_sha": sha(rows), "score_sha": sha(s),
                                 "converged": meta["all_converged"]}
            print(m, tag, json.dumps(mlp[f"{m}/{tag}"]), flush=True)
        mlp[f"{m}/stored"] = stored[m]
    pairs = {}
    for x, z in (("mtx-l188-s1", "mtx-l188-s2"), ("mtx-l188-s1", "mtx-l188-s3"), ("mtx-l188-s2", "mtx-l188-s3"),
                 ("mtx-l162-s1", "mtx-l162-s2")):
        for tag in ("orig", "new"):
            sa, y, ra = scores[(x, tag)]
            sb, yb, rb = scores[(z, tag)]
            assert np.array_equal(ra, rb) and np.array_equal(y, yb)
            pairs[f"{x}|{z}/{tag}"] = {**compare(sa, sb), **paired_bootstrap(y, sa, sb)}
            print(x, z, tag, json.dumps(pairs[f"{x}|{z}/{tag}"]), flush=True)
    doc["mlp_bc"] = mlp
    doc["mlp_bc_pairs"] = pairs
    doc["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{a.out} written ({doc['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
