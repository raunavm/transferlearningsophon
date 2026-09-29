#!/usr/bin/env python3
"""Output-layer results by the checkpoint rule: head diagnostics and the class-sum
anomaly scores at every checkpoint extract_v2.py scored.

WHY (audit 2026-09-29, B4 and must-fix 3). Every result read from the pretrained
OUTPUT LAYER -- the class-sum anomaly score, the real-data log-odds -- was read
from the epoch-79 head alone. Four of 30 such heads are defective (43-class run
5 almost never predicts QCD; its YY->bbbb class-sum AUC is 0.66 against 0.94 for
its siblings) while their frozen features probe normally. The rule is now:
primary = the best-validation checkpoint (v2), robustness = the mean over the
checkpoints of epochs 70-79. For v1, which kept every epoch but recorded no
fixed validation sample, the robustness form is computed; it is also the test
of the audit's explanation (heads that swing from epoch to epoch while the
probes stay flat).

Per model and checkpoint:
  head         on the stride sample (rows % diag_stride == 0 of the 2,000,000):
               top-1 accuracy at the model's own label set, mean P(QCD) on
               resonant jets, mean P(QCD) on QCD jets, median resonant-vs-QCD
               log-odds on QCD jets
  anomaly      when the extraction kept the anomaly rows: sigma_min and max SIC
               of class_sum and class_sum_matched, per signal and N_sig, with
               anomaly.py's own draws (run_one, cell_seed) -- so at epoch 79 the
               numbers must reproduce the committed ones, and do to REPRO_TOL
               (the logits are recomputed, not read, so the last digits move)
Across checkpoints: the mean of ln sigma_min over epochs 70-79 per cell.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
REPRO_TOL = 1e-3          # |d ln sigma_min| at epoch 79 against the committed run
ROBUST_EPOCHS = tuple(range(70, 80))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


an = _load("anomaly", "experiments/EVAL/anomaly.py")


def head_diagnostics(h: dict, rung: str, stride: int) -> dict:
    """The four numbers that expose a defective head, on the stride sample."""
    node_of, res, qcd = an.node_roles(rung)
    rows, lab = h["rows"], h["label188"].astype(np.int64)
    diag = rows % stride == 0
    lut = np.array([node_of[i] for i in range(188)])
    truth = lut[lab[diag]]
    is_q = np.isin(truth, sorted(qcd))
    pq = h["p_qcd"][diag]
    return {"n": int(diag.sum()),
            "top1_accuracy": float((h["argmax"][diag] == truth).mean()),
            "mean_p_qcd_resonant": float(pq[~is_q].mean()),
            "mean_p_qcd_qcd": float(pq[is_q].mean()),
            "median_logodds_res_qcd_on_qcd": float(np.median(h["logodds_res_qcd"][diag][is_q]))}


def anomaly_cells(arm: str, rung: str, L: np.ndarray, h: dict, signals, n_sigs,
                  trainings: int, n_bkg: int, n_template: int) -> dict | None:
    """sigma_min and max SIC of the two class sums, anomaly.py's draws exactly."""
    rows = h["rows"]
    qcd = an._probe().qcd_indices()
    need = np.flatnonzero(np.isin(L, qcd + [an_label(s) for s in signals]))
    if not np.isin(need, rows).all():
        return None                          # a diagnostics-only extraction
    pos = np.full(L.size, -1, dtype=np.int64)
    pos[rows] = np.arange(rows.size)
    node_of = an.node_roles(rung)[0]
    out = {}
    for sig in signals:
        lab = an_label(sig)
        pre = {fam: (lambda idx, c=h[f"{fam}|{sig}"]: c[pos[idx]])
               for fam in ("class_sum", "class_sum_matched")}
        per_n = {}
        for n_sig in n_sigs:
            reps = []
            for t in range(trainings):
                seed = an.cell_seed(arm, sig, n_sig, t)
                r = an.run_one(arm, rung, None, L, None, {}, lab, node_of[lab], n_sig,
                               np.random.default_rng(seed), n_bkg, n_template, seed=t,
                               families=("class_sum", "class_sum_matched"), precomputed=pre)
                if r:
                    r["rng_seed"] = int(seed)
                    reps.append(r)
            per_n[str(n_sig)] = (an.aggregate(reps, arm, sig, n_sig) if reps
                                 else {"skipped": "insufficient jets"})
        out[sig] = per_n
    return out


_BY_NAME = None


def an_label(name: str) -> int:
    global _BY_NAME
    if _BY_NAME is None:
        _BY_NAME = {r["class_name"]: int(r["jet_label"]) for r in an.read_map()}
    return _BY_NAME[name]


def committed_check(arm: str, cells: dict, committed: dict, rerun: dict | None) -> dict:
    """Epoch-79 cells against the committed anomaly run (class_sum) and its
    class-sum rerun (class_sum_matched): the largest |d ln| over every cell."""
    worst, n = 0.0, 0
    for sig, per_n in cells.items():
        for n_sig, c in per_n.items():
            for fam, src in (("class_sum", committed), ("class_sum_matched", rerun)):
                if src is None or fam not in c or "sigma_min" not in c[fam]:
                    continue
                ref = src["arms"][arm]["signals"][sig][n_sig]
                if ref.get("rng_seeds") != c.get("rng_seeds"):
                    raise SystemExit(f"FATAL: {arm}/{sig}/{n_sig}: the draws differ from "
                                     "the committed run's")
                for m in ("sigma_min", "max_sic"):
                    worst = max(worst, abs(math.log(c[fam][m]) - math.log(ref[fam][m])))
                    n += 1
    return {"cells_compared": n, "max_abs_dln": worst, "tolerance": REPRO_TOL,
            "reproduces": bool(worst <= REPRO_TOL)}


def one_checkpoint(job) -> tuple[str, str, dict]:
    """(arm, tag, result) for one extracted checkpoint; run in a worker process."""
    arm, rung, cdir, labels, committed, rerun, signals, n_sigs, trainings, n_bkg, n_template = job
    cdir = pathlib.Path(cdir)
    L = np.load(labels).astype(np.int64)
    man = json.loads((cdir / "manifest.json").read_text())
    if man["rung"] != rung:
        raise SystemExit(f"FATAL: {cdir} was extracted at {man['rung']}, not {rung}")
    h = dict(np.load(cdir / "head_scores.npz"))
    if not np.array_equal(h["label188"].astype(np.int64), L[h["rows"]]):
        raise SystemExit(f"FATAL: {cdir}: head rows do not carry the labels of {labels}")
    c = {"checkpoint_sha256": man["checkpoint_sha256"],
         "head": head_diagnostics(h, rung, int(man["diag_stride"]))}
    cells = anomaly_cells(arm, rung, L, h, signals, n_sigs, trainings, n_bkg, n_template)
    if cells is not None:
        c["anomaly"] = cells
        if cdir.name == "e079" and committed is not None:
            c["committed_check"] = committed_check(
                arm, cells, json.loads(pathlib.Path(committed).read_text()),
                json.loads(pathlib.Path(rerun).read_text()) if rerun else None)
    return arm, cdir.name, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True,
                    help="arm=RUNG=DIR, DIR holding extract_v2.py's checkpoint directories")
    ap.add_argument("--labels", required=True, type=pathlib.Path,
                    help="label188.npy of the 2,000,000-jet v1 caches (the anomaly draws index it)")
    ap.add_argument("--committed", type=pathlib.Path, default=None,
                    help="the committed anomaly_results.json (class_sum), for the epoch-79 check")
    ap.add_argument("--committed-rerun", type=pathlib.Path, default=None,
                    help="the committed class-sum rerun (class_sum_matched)")
    ap.add_argument("--signals", nargs="+", default=list(an.SIGNAL_SUITE))
    ap.add_argument("--n-sig", nargs="+", type=int, default=[2000, 4000])
    ap.add_argument("--trainings", type=int, default=an.N_TRAININGS)
    ap.add_argument("--n-bkg", type=int, default=200_000)
    ap.add_argument("--n-template", type=int, default=200_000)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; refusing to overwrite")

    res = {"labels_sha256": hashlib.sha256(np.load(a.labels).tobytes()).hexdigest(),
           "n_bkg": a.n_bkg, "n_template": a.n_template, "trainings": a.trainings,
           "sigma_min": "Asimov, arXiv:2604.20965 Eqs. 3-4, sigma_t = 5, B = n_bkg; thresholds "
                        f"with more than {an.MIN_BKG_PASS} background jets passing (n_B > "
                        f"{an.MIN_BKG_PASS}, anomaly.sic_curve)",
           "models": {}}
    jobs = []
    for spec in a.models:
        arm, rung, d = spec.split("=", 2)
        res["models"][arm] = {"rung": rung, "checkpoints": {}}
        for cdir in sorted(p for p in pathlib.Path(d).iterdir() if (p / "manifest.json").exists()):
            jobs.append((arm, rung, str(cdir), str(a.labels),
                         str(a.committed) if a.committed else None,
                         str(a.committed_rerun) if a.committed_rerun else None,
                         a.signals, a.n_sig, a.trainings, a.n_bkg, a.n_template))
    if a.procs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            done = list(ex.map(one_checkpoint, jobs))
    else:
        done = [one_checkpoint(j) for j in jobs]
    for arm, tag, c in done:
        res["models"][arm]["checkpoints"][tag] = c
        print(f"  {arm:14s} {tag}  head acc {c['head']['top1_accuracy']:.4f}  "
              f"P(QCD|res) {c['head']['mean_p_qcd_resonant']:.3f}"
              + ("  anomaly" if "anomaly" in c else ""), flush=True)
    for arm, m in res["models"].items():
        rob = [f"e{e:03d}" for e in ROBUST_EPOCHS]
        have = [t for t in rob if t in m["checkpoints"]]
        if have == rob:
            H = [m["checkpoints"][t]["head"] for t in rob]
            m["head_over_70_79"] = {k: {"mean": float(np.mean([x[k] for x in H])),
                                        "min": float(np.min([x[k] for x in H])),
                                        "max": float(np.max([x[k] for x in H]))}
                                    for k in H[0] if k != "n"}
            if all("anomaly" in m["checkpoints"][t] for t in rob):
                mean = {}
                for sig in a.signals:
                    for n_sig in map(str, a.n_sig):
                        for fam in ("class_sum", "class_sum_matched"):
                            v = [m["checkpoints"][t]["anomaly"][sig][n_sig].get(fam, {}).get("sigma_min")
                                 for t in rob]
                            if all(x and x > 0 for x in v):
                                mean.setdefault(fam, {}).setdefault(sig, {})[n_sig] = {
                                    "ln_sigma_min_mean": float(np.mean(np.log(v))),
                                    "ln_sigma_min_per_epoch": [float(math.log(x)) for x in v]}
                m["anomaly_mean_70_79"] = mean
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
