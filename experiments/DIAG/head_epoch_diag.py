#!/usr/bin/env python3
"""Output-layer ("head") diagnostic over the last ten pretraining epochs.

WHY. The 2026-09-29 audit (item 12) found four v1 runs whose epoch-79 output
layer is defective although their frozen features probe normally, and proposed
[I] that the loader (--fetch-by-files --fetch-step 5, two workers, files that
each hold one family) swings the class mix from epoch to epoch, so the last
epoch's head inherits the class mix of the last files read. This measures it.

WHAT. For each run and each checkpoint -- epochs 70-79, the best-validation
checkpoint, and the weight average of epochs 70-79 -- on ONE fixed sample of
test jets, the same for every checkpoint:
  head:  top-1 accuracy in the run's own vocabulary; P(QCD), the softmax mass on
         the vocabulary's QCD classes, averaged over resonant and over QCD jets;
         the resonance-vs-QCD log-odds logsumexp(z_res) - logsumexp(z_QCD),
         its median on QCD jets and its resonant-vs-QCD AUC;
  probe: the repository's linear probe (experiments/EVAL/probe.py: fit_linear,
         make_splits, log1m_auc, unchanged) on the frozen 128-d features, task
         bvc_resonant (label_X_bb vs label_X_cc).
The softmax averaged over epochs 70-79 ("ens") is scored as a head too.
`analyse` adds the QCD share of the training stream at each epoch, read from
the run's train.log (below), compares the epochs with a paired bootstrap
(paired_swing), flags defective heads under the rules stated at DEFECT_RULES,
and writes one JSON; `figure` draws it. `summarise` recomputes the defect flags
and counts and the class-mix result from an existing JSON, without the arrays.

THE SAMPLE. The first --max-jets jets of the test list, every --stride-th, read
once with configs/data/JetClassII_base.yaml in file order (the stream every
frozen-feature cache holds). --align-with names such a cache: its label188.npy,
strided the same way, must equal the labels read here. The files and rows are
recomputed from the parquet files with the config's own selection and
their jet_label sequence must equal the stream's, so "which jets" is a
statement checked against the data, not a description.

THE STREAM'S QCD SHARE. weaver prints, at the end of every training epoch,
"Train class distribution:" followed by one line [(class, count), ...] over
every jet trained on in that epoch (10,240,000). The share is the count on the
vocabulary's QCD classes over the total. A resumed epoch is printed again; the
LAST complete print for an epoch is the one whose checkpoint is on disk. No
finer record exists: at INFO level weaver logs only "Restarted DataIter" when a
worker exhausts its file list, not the files of each fetch, so the class mix of
the last fetches before a checkpoint cannot be reconstructed.

Usage (the job spec is experiments/DIAG/k8s/job-diag-head-epochs-raunav.yaml;
the analyse-only rerun is job-diag-head-epochs-analyse-v2-raunav.yaml):
    head_epoch_diag.py infer     --runs RUN_DIR:RUNG:K:NUM_REG ... --data-test ... --out DIR
    head_epoch_diag.py analyse   --out DIR --json FILE
    head_epoch_diag.py summarise --json FILE --out FILE2
    head_epoch_diag.py figure    --json FILE --outdir DIR
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TASK = "bvc_resonant"
N_BOOT = 200
PAIRED_BOOT = 1000          # 9x9 covariance of the epoch deviations: 200 replicates bias chi2 up ~5%
# The four runs the 2026-09-29 audit flagged for a defective epoch-79 head
# (B4: 2,000,000 test jets, against the same vocabulary's other runs).
AUDIT_FLAGGED = ("mtx-l188-s5", "mtx-l162-s5", "mtx-r42q1-s5", "mtx-r16q1mass-s4")


def is_epoch(tag: str) -> bool:
    return re.fullmatch(r"e\d+", tag) is not None


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("probe", "experiments/EVAL/probe.py")
EA = _load("epoch_accuracy", "experiments/EVAL/epoch_accuracy.py")
XF = EA.XF
QCD_NATIVE = np.array(P.qcd_indices())


def vocabulary(rung: str, k: int):
    """(native -> class map, QCD classes, resonant classes) of a vocabulary."""
    lut = EA.vocabulary_map(rung)
    qcd = sorted(set(lut[QCD_NATIVE].tolist()))
    res = sorted(set(lut[np.setdiff1d(np.arange(lut.size), QCD_NATIVE)].tolist()))
    if set(qcd) & set(res) or sorted(qcd + res) != list(range(k)):
        raise SystemExit(f"FATAL: {rung} does not split its {k} classes into QCD and resonant")
    return lut, qcd, res


def _lse(a: np.ndarray) -> np.ndarray:
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def per_jet(logp: np.ndarray, truth: np.ndarray, qcd: list, res: list, chunk: int = 100_000):
    """Per-jet (correct, P(QCD), resonance-vs-QCD log-odds) from log-probabilities.

    logp is log-softmax over the K classes, or the log of an average of softmaxes;
    anything proportional to it gives the same three numbers after normalisation.
    """
    out = [], [], []
    for i in range(0, len(truth), chunk):
        z = np.asarray(logp[i:i + chunk], dtype=np.float64)
        z = z - _lse(z)[:, None]
        lq, lr = _lse(z[:, qcd]), _lse(z[:, res])
        out[0].append(z.argmax(1) == truth[i:i + chunk])
        out[1].append(np.exp(lq).astype(np.float32))
        out[2].append((lr - lq).astype(np.float32))
    return tuple(np.concatenate(o) for o in out)


def head_summary(correct, p_qcd, log_odds, is_qcd) -> dict:
    from sklearn.metrics import roc_auc_score
    n, acc = correct.size, float(correct.mean())
    return {"n_jets": int(n), "top1_accuracy": acc,
            "top1_binomial_se": float(np.sqrt(acc * (1 - acc) / n)),
            "mean_p_qcd_resonant": float(p_qcd[~is_qcd].mean()),
            "mean_p_qcd_resonant_se": float(p_qcd[~is_qcd].std() / np.sqrt((~is_qcd).sum())),
            "mean_p_qcd_qcd": float(p_qcd[is_qcd].mean()),
            "median_log_odds_qcd": float(np.median(log_odds[is_qcd])),
            "res_vs_qcd_auc": float(roc_auc_score(~is_qcd, log_odds))}


def probe_summary(F: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[dict, np.ndarray]:
    """probe.py's linear probe on one task, plus a bootstrap SE of log(1-AUC);
    also the probe's scores on the test split, for the paired epoch comparison."""
    tr, va, te = P.make_splits(len(y))
    s, meta = P.fit_linear(F[tr], y[tr], F[va], y[va], F[te])
    l1m, censored, auc = P.log1m_auc(y[te], s)
    rng, yt, boot = np.random.default_rng(seed), y[te], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        boot.append(P.log1m_auc(yt[i], s[i])[0])
    return {"auc": auc, "log1m_auc": l1m, "censored": censored,
            "log1m_auc_boot_se": float(np.std(boot, ddof=1)), "C": meta["C"],
            "n_train": int(len(tr)), "n_test": int(len(te))}, s


def parse_train_log(text: str) -> tuple[dict, dict]:
    """{epoch: {class: count}} (last complete print wins), {epoch: logged val metric}."""
    dist, val, cur = {}, {}, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"Epoch #(\d+) training", line)
        if m:
            cur = int(m.group(1))
        elif "Train class distribution:" in line and cur is not None and i + 1 < len(lines):
            dist[cur] = dict(ast.literal_eval(lines[i + 1].strip()))
        m = re.search(r"Epoch #(\d+): Current validation metric: ([-0-9.eE]+)", line)
        if m:
            val[int(m.group(1))] = float(m.group(2))
    return dist, val


def qcd_share(counts: dict, qcd: list) -> float:
    tot = sum(counts.values())
    return sum(counts.get(c, 0) for c in qcd) / tot


def paired_swing(stat, n: int, n_boot: int = PAIRED_BOOT, seed: int = 0) -> dict:
    """Between-epoch spread of one metric against its sampling noise, when every
    epoch is scored on the SAME n jets. stat(idx) returns the metric of each of
    the K epochs on jets idx.

    Each bootstrap replicate draws the jets once and scores every epoch on them,
    so the noise the epochs share (which jets happen to be in the sample) cancels
    in each epoch's deviation from the epoch mean. Dividing by each epoch's own
    sampling error instead counts that shared noise as if the epochs had been
    scored on independent samples, and hides real differences between them.
    chi2 = d' S^+ d, with d the epochs' deviations from their mean and S the
    bootstrap covariance of those deviations (rank K-1, hence the pseudo-inverse);
    K-1 degrees of freedom. paired_se is the median over epochs of the bootstrap
    SD of an epoch's deviation from the epoch mean.
    """
    from scipy.stats import chi2
    v = np.asarray(stat(np.arange(n)), float)
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    dev = boot - boot.mean(1, keepdims=True)
    d = v - v.mean()
    c2 = float(d @ np.linalg.pinv(np.atleast_2d(np.cov(dev, rowvar=False)), hermitian=True) @ d)
    se = float(np.median(dev.std(0, ddof=1)))
    return {"sd": float(v.std(ddof=1)), "paired_se": se, "sd_over_paired_se": float(v.std(ddof=1) / se),
            "chi2": c2, "dof": int(v.size - 1), "p": float(chi2.sf(c2, v.size - 1)), "n_boot": n_boot}


def within_run_correlation(series: list[tuple], n_perm: int = 10_000, seed: int = 0) -> dict:
    """Pearson r of (x, y) pooled over runs after z-scoring each run's series;
    permutation p from shuffling y within each run."""
    def z(a):
        a = np.asarray(a, float)
        return (a - a.mean()) / a.std()
    xs = [z(x) for x, _ in series]
    ys = [z(y) for _, y in series]
    r = float(np.mean(np.concatenate(xs) * np.concatenate(ys)))
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(n_perm):
        rp = np.mean(np.concatenate([x * rng.permutation(y) for x, y in zip(xs, ys)]))
        hits += abs(rp) >= abs(r) - 1e-12
    return {"r": r, "n_points": int(sum(len(x) for x in xs)), "p_perm": float((hits + 1) / (n_perm + 1))}


def average_states(paths: list[pathlib.Path]) -> dict:
    """Arithmetic mean of floating tensors over checkpoints; integer buffers from the last."""
    import torch
    sts = []
    for p in paths:
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
        sts.append(raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw)
    if any(s.keys() != sts[0].keys() for s in sts):
        raise SystemExit("FATAL: checkpoints to average have different keys")
    return {k: (sum(s[k].double() for s in sts) / len(sts)).to(v.dtype)
            if v.is_floating_point() else sts[-1][k] for k, v in sts[0].items()}


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _save(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- infer (GPU)
def read_sample(a, cfg_path: str):
    """The fixed sample in memory, its native labels, and where its jets come from."""
    import torch
    import yaml
    import pyarrow.parquet as pq
    from weaver.utils.dataset import SimpleIterDataset
    from weaver.utils.data.config import DataConfig

    cfg = DataConfig.load(cfg_path, load_observers=False)
    ds = SimpleIterDataset({"_": list(a.data_test)}, cfg_path, for_training=False,
                           fetch_by_files=True, fetch_step=1, name="head_epoch_diag")
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
    batches, native, n = [], [], 0
    for X, y, _ in loader:
        lab = y[cfg.label_names[0]].numpy().astype(np.int64)
        idx = np.arange(n, n + lab.size)
        sel = (idx % a.stride == 0) & (idx < a.max_jets)
        n += lab.size
        if sel.any():
            t = torch.from_numpy(sel)
            batches.append([X[k][t] for k in cfg.input_names])
            native.append(lab[sel])
        if n >= a.max_jets:
            break
    if n < a.max_jets:
        raise SystemExit(f"FATAL: the test list holds {n} jets, fewer than {a.max_jets}")
    native = np.concatenate(native)
    ref = np.load(a.align_with / "label188.npy")[:a.max_jets][::a.stride].astype(np.int64)
    if not np.array_equal(ref, native):
        raise SystemExit(f"FATAL: these are not the jets of {a.align_with}")

    raw = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    if "test_time_selection" in raw:
        raise SystemExit("FATAL: the config has a test_time_selection; the row map assumes none")
    files, labels, start = [], [], 0
    for f in a.data_test:
        t = pq.read_table(f, columns=["jet_pt", "jet_sdmass", "jet_label"])
        cols = {c: t.column(c).to_numpy() for c in t.column_names}
        keep = eval(raw["selection"], {"np": np}, cols)
        rows = np.nonzero(keep)[0]
        take = rows[:a.max_jets - start]
        files.append({"file": f, "rows_in_file": int(keep.size), "selected": int(rows.size),
                      "stream_range": [start, start + int(take.size)],
                      "file_rows_used": [int(take[0]), int(take[-1])] if take.size else None})
        labels.append(cols["jet_label"][take].astype(np.int64))
        start += int(take.size)
        if start >= a.max_jets:
            break
    if not np.array_equal(np.concatenate(labels)[::a.stride], native):
        raise SystemExit("FATAL: the parquet row map does not reproduce the stream's labels")
    return cfg, batches, native, files


def run_model(model, tap, batches, device, k: int, rows: np.ndarray):
    """Class logits (all jets) and 128-d features (only `rows`) for one checkpoint."""
    import torch
    logits, feats, n = [], [], 0
    with torch.no_grad():
        for inputs in batches:
            out = model(*[x.to(device, non_blocking=True) for x in inputs])
            logits.append(out[:, :k].float().cpu().numpy())
            b = logits[-1].shape[0]
            r = rows[(rows >= n) & (rows < n + b)] - n
            feats.append(tap.buf[torch.from_numpy(r).to(device)].float().cpu().numpy())
            n += b
    return np.concatenate(logits), np.concatenate(feats)


def infer(a) -> int:
    import torch
    t_start = time.time()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device {device} ({gpu}), node {os.environ.get('NODE_NAME', '?')}", flush=True)
    cfg_path = str(REPO / "configs/data/JetClassII_base.yaml")
    cfg, batches, native, files = read_sample(a, cfg_path)
    task = P.TASKS[TASK]
    bvc = np.nonzero(np.isin(native, task["signal"] + task["background"]))[0]
    is_qcd = np.isin(native, QCD_NATIVE)
    sample = {"n_jets": int(native.size), "stream_jets": a.max_jets, "stride": a.stride,
              "aligned_with": str(a.align_with), "files": files,
              "native_label_sha256": hashlib.sha256(native.astype(np.int16).tobytes()).hexdigest(),
              "n_qcd": int(is_qcd.sum()), "n_bvc": int(bvc.size), "gpu": gpu,
              "node": os.environ.get("NODE_NAME", "?"), "torch": torch.__version__}
    (out / "sample.json").write_text(json.dumps(sample, indent=1))
    _save(out / "sample.npz", native=native, bvc_rows=bvc)
    print(f"{native.size:,} jets held ({is_qcd.sum():,} QCD, {bvc.size:,} {TASK}); "
          f"{len(files)} files; {time.time() - t_start:.0f} s", flush=True)

    for spec in a.runs:
        run_dir, rung, k, num_reg = spec.split(":")
        run_dir, k, num_reg = pathlib.Path(run_dir), int(k), int(num_reg)
        rd = out / run_dir.name
        if (rd / "DONE").exists():
            print(f"{run_dir.name}: DONE, skipped", flush=True)
            continue
        rd.mkdir(exist_ok=True)
        lut, qcd, res = vocabulary(rung, k)
        truth = lut[native]
        ckpts = {f"e{e}": run_dir / f"net_epoch-{e}_state.pt" for e in a.epochs}
        wavg = rd / f"wavg_e{a.epochs[0]}-{a.epochs[-1]}_state.pt"
        torch.save(average_states(list(ckpts.values())), wavg)
        best, best_epoch = run_dir / "net_best_epoch_state.pt", None
        if best.exists():
            b = sha256(best)
            best_epoch = next((e for e in range(a.max_epoch + 1)
                               if (run_dir / f"net_epoch-{e}_state.pt").exists()
                               and sha256(run_dir / f"net_epoch-{e}_state.pt") == b), None)
            if f"e{best_epoch}" not in ckpts:
                ckpts["best"] = best
        ckpts["wavg"] = wavg
        ens, meta = None, {"run_dir": str(run_dir), "rung": rung, "num_classes": k,
                           "num_reg": num_reg, "best_epoch": best_epoch, "checkpoints": {}}
        for tag, ck in ckpts.items():
            t0 = time.time()
            model = XF.build_model(cfg, k + num_reg)
            prov = XF.load_trunk_or_die(model, ck, k, num_reg)
            model.to(device).eval()
            tap = XF.ClsTap(model)
            XF.self_check(model, tap, cfg, device)
            logits, feats = run_model(model, tap, batches, device, k, bvc)
            tap.close()
            logp = torch.log_softmax(torch.from_numpy(logits).double(), 1).numpy()
            if is_epoch(tag):
                ens = np.exp(logp) if ens is None else ens + np.exp(logp)
            c, pq_, lo = per_jet(logp, truth, qcd, res)
            _save(rd / f"{tag}.npz", correct=c, p_qcd=pq_, log_odds=lo, feat_bvc=feats)
            meta["checkpoints"][tag] = {"path": str(ck), "sha256": prov["sha256"],
                                        "seconds": round(time.time() - t0, 1)}
            if tag == f"e{a.epochs[-1]}":        # reproduce the committed epoch-79 cache
                cache = pathlib.Path(a.cache_root) / run_dir.name / "features_e79"
                ref_l = np.load(cache / "logits.npy", mmap_mode="r")
                ref_f = np.load(cache / "features.npy", mmap_mode="r")
                rows = np.arange(native.size) * a.stride
                rl = np.asarray(ref_l[rows, :k], dtype=np.float32)
                meta["repro_features_e79"] = {
                    "max_abs_logit_diff": float(np.abs(rl - logits).max()),
                    "top1_agreement": float((rl.argmax(1) == logits.argmax(1)).mean()),
                    "max_abs_feature_diff_bvc": float(np.abs(
                        np.asarray(ref_f[rows[bvc]], dtype=np.float32) - feats).max())}
            print(f"{run_dir.name} {tag}: acc {c.mean():.4f}  {time.time() - t0:.0f} s", flush=True)
        c, pq_, lo = per_jet(np.log(ens / len(a.epochs)), truth, qcd, res)
        _save(rd / "ens.npz", correct=c, p_qcd=pq_, log_odds=lo)
        (rd / "DONE").write_text(json.dumps(meta, indent=1))
    print(f"infer finished in {time.time() - t_start:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- defective heads
# WHAT "DEFECTIVE" MEANS -- the primary rule was fixed 2026-10-01, before any
# count was computed with it. A checkpoint's output layer is defective when its
# top-1 accuracy on the sample is more than 10% below the median top-1 accuracy
# of the run's epochs 70-79.
#  * Top-1 accuracy is what the audit flagged the four heads on, and what the v2
#    checkpoint rule maximises.
#  * The reference is built from the epochs alone, never from averaged weights,
#    so the weight average and the best-validation checkpoint are judged against
#    the same reference as the epochs. A reference built from the weight average
#    would make "the weight average removes the defect" true by construction.
#  * 10% is twice the largest distance of a normal epoch-79 head from the middle
#    of its vocabulary's normal heads in the audit (2,000,000 jets; normal heads
#    0.463-0.490, 0.501-0.545, 0.605-0.671, 0.682-0.719: half-widths 2.8, 4.2,
#    5.2 and 2.6%). The binomial error on 500,000 jets is 0.1-0.2% of the
#    accuracy, so sampling noise cannot trigger the rule. It is a magnitude rule
#    because it has to be: every run's epochs spread over 52-152 binomial errors,
#    so a rule on significance alone would flag a large share of every run's epochs.
# THE MEDIAN IS NOT ALWAYS A SOUND HEAD. It is the level of a normal head only
# while at most four of the ten epochs are defective low. Two runs break that:
# in l188-s5 six epochs are low, and the median, 0.406, is the accuracy of
# epochs 71 and 78, whose heads never predict QCD; in r16q1mass-s4 five are,
# and the median, 0.632, lies between a never-QCD epoch (0.607) and the run's
# sound epochs at 0.66-0.70. So the counts are also given against two other
# references (review of 2026-10-01): the 75th percentile of epochs 70-79, sound
# while at most six epochs are low, and the most accurate of epochs 70-79, sound
# while any one is. From the median to the most accurate epoch the reference
# moves from the middle to the top of a run's sound epochs, so the rule gets
# stricter; the three bracket the choice.
# The other rules give the range of the counts: each reference at 5%, 10% and
# 15%, and a rule on the QCD-vs-resonance decision, the median resonance-vs-QCD
# log-odds on QCD jets more than 2.8 from the reference, either way, against the
# median and against the most accurate epoch's value (a percentile is no
# reference for a two-sided rule). The audit's defective heads erred both ways
# (one never predicts QCD, three over-predict it); 2.8 is twice the half-width
# of its normal 43-class heads (-1.5 to +1.3).
REFERENCES = {"median": "the median of the run's epochs 70-79",
              "q75": "the 75th percentile of the run's epochs 70-79",
              "most_accurate_epoch": "its value at the most accurate of the run's epochs 70-79"}
DEFECT_RULES = {   # name: (head metric, reference, direction, threshold, definition)
    **{f"top1_{p}pct_vs_{ref}": ("top1_accuracy", ref, "below", p / 100,
                                 f"top-1 accuracy more than {p}% below {REFERENCES[ref]}")
       for p in (10, 5, 15) for ref in ("median", "q75", "most_accurate_epoch")},
    **{f"qcd_log_odds_2p8_vs_{ref}": ("median_log_odds_qcd", ref, "either", 2.8,
                                      "median resonance-vs-QCD log-odds on QCD jets more than "
                                      f"2.8 from {REFERENCES[ref]}, either way")
       for ref in ("median", "most_accurate_epoch")},
}
PRIMARY_RULE = "top1_10pct_vs_median"


def is_defective(value: float, ref: float, direction: str, threshold: float) -> bool:
    if direction == "below":
        return value < (1 - threshold) * ref
    return abs(value - ref) > threshold


def reference_value(cks: dict, epochs: list, metric: str, how: str) -> float:
    """A run's reference for one head metric, from its epoch checkpoints only."""
    if how == "most_accurate_epoch":
        return float(cks[max(epochs, key=lambda t: cks[t]["head"]["top1_accuracy"])]["head"][metric])
    v = [cks[t]["head"][metric] for t in epochs]
    return float({"median": np.median, "q75": lambda x: np.percentile(x, 75)}[how](v))


def summarise(res: dict) -> dict:
    """From an analysis JSON's head summaries: every checkpoint's defect flag
    under each rule, the counts, their range over the rules, and the stream
    class-mix result."""
    runs = {}
    for run, r in res["runs"].items():
        cks = r["checkpoints"]
        ep = [t for t in cks if is_epoch(t)]
        ref = {name: reference_value(cks, ep, m, how) for name, (m, how, *_) in DEFECT_RULES.items()}
        runs[run] = {"reference": ref,
                     "best": r.get("best_is") or ("best" if "best" in cks else None),
                     "last_epoch": max(ep, key=lambda t: int(t[1:])),
                     "flags": {t: {name: is_defective(c["head"][m], ref[name], d, thr)
                                   for name, (m, _, d, thr, _) in DEFECT_RULES.items()}
                               for t, c in cks.items()}}

    def bad(run, tag, rule):
        return tag is None or runs[run]["flags"][tag][rule]   # no best checkpoint: not a fix

    counts = {}
    for rule in DEFECT_RULES:
        hit = [run for run, f in runs.items() if any(f["flags"][t][rule] for t in f["flags"] if is_epoch(t))]
        audit = [run for run in AUDIT_FLAGGED if run in runs]
        c = {"runs_with_a_defective_epoch": hit,
             "of_those_weight_average_sound": [x for x in hit if not bad(x, "wavg", rule)],
             "of_those_best_validation_sound": [x for x in hit if not bad(x, runs[x]["best"], rule)],
             "defective_at_last_epoch": [x for x in runs if bad(x, runs[x]["last_epoch"], rule)],
             "audit_flagged": {"runs": audit,
                               "defective_at_last_epoch": [x for x in audit if bad(x, runs[x]["last_epoch"], rule)],
                               "weight_average_sound": [x for x in audit if not bad(x, "wavg", rule)],
                               "best_validation_sound": [x for x in audit if not bad(x, runs[x]["best"], rule)]}}
        c["weight_average_defective"] = [x for x in runs if bad(x, "wavg", rule)]
        c["best_validation_defective"] = [x for x in runs if bad(x, runs[x]["best"], rule)]
        c["n"] = {"runs": len(runs), "runs_with_a_defective_epoch": len(hit),
                  "weight_average_fixes": len(c["of_those_weight_average_sound"]),
                  "best_validation_fixes": len(c["of_those_best_validation_sound"]),
                  "weight_average_defective": len(c["weight_average_defective"]),
                  "best_validation_defective": len(c["best_validation_defective"]),
                  "audit_flagged_weight_average_fixes": len(c["audit_flagged"]["weight_average_sound"]),
                  "audit_flagged_best_validation_fixes": len(c["audit_flagged"]["best_validation_sound"])}
        counts[rule] = c

    span = {}                          # each count's range over the rules, and which rules give the ends
    for k in counts[PRIMARY_RULE]["n"]:
        v = {rule: c["n"][k] for rule, c in counts.items()}
        lo, hi = min(v.values()), max(v.values())
        span[k] = {"min": lo, "max": hi, "rules_at_min": [x for x in v if v[x] == lo],
                   "rules_at_max": [x for x in v if v[x] == hi]}

    pw = res["pooled_within_run"]
    per_run = [r["epochs_70_79"] for r in res["runs"].values() if "pearson_share_vs_top1" in r["epochs_70_79"]]
    return {"defect": {"primary_rule": PRIMARY_RULE,
                       "rules": {name: rule[4] for name, rule in DEFECT_RULES.items()},
                       "runs": runs, "counts": counts, "range_over_rules": span},
            "class_mix": {
                "description": "whether the QCD share of an epoch's training stream tracks that "
                               "epoch's output layer: Pearson r over epochs 70-79, pooled over runs "
                               "after z-scoring each run's epochs; p from permuting epochs within runs",
                "share_vs_mean_p_qcd_resonant": pw["share_vs_p_qcd_resonant"],
                "share_vs_top1": pw["share_vs_top1"],
                "per_run_r_range_share_vs_mean_p_qcd_resonant":
                    [min(x["pearson_share_vs_p_qcd_resonant"] for x in per_run),
                     max(x["pearson_share_vs_p_qcd_resonant"] for x in per_run)],
                "per_run_r_range_share_vs_top1": [min(x["pearson_share_vs_top1"] for x in per_run),
                                                  max(x["pearson_share_vs_top1"] for x in per_run)]}}


# ---------------------------------------------------------------- analyse (CPU)
def analyse(a) -> int:
    out = a.out
    sample = json.loads((out / "sample.json").read_text())
    z = np.load(out / "sample.npz")
    native, bvc = z["native"], z["bvc_rows"]
    is_qcd = np.isin(native, QCD_NATIVE)
    y = np.isin(native[bvc], P.TASKS[TASK]["signal"]).astype(np.int64)
    res = {"sample": sample, "probe_task": {TASK: P.TASKS[TASK]["names"]},
           "definitions": {
               "top1_accuracy": "argmax of the K class outputs == native label mapped to the run's vocabulary",
               "mean_p_qcd": "softmax mass on the vocabulary's QCD classes, averaged over resonant / QCD jets",
               "log_odds": "logsumexp(z_resonant) - logsumexp(z_QCD) per jet",
               "ens": "softmax averaged over the epoch checkpoints, then scored",
               "wavg": "weights (and BatchNorm buffers) averaged over the epoch checkpoints",
               "stream_qcd_share": "QCD classes' share of weaver's per-epoch 'Train class distribution' in train.log",
               "swing_test": "spread of the epoch values against a paired bootstrap: each replicate resamples "
                             "the jets once and scores every epoch on them (all 500,000 jets for the head; the "
                             "probe's test split, fits held fixed, for log(1-AUC)); chi2 of the deviations from "
                             "the epoch mean with their bootstrap covariance, K-1 dof"},
           "runs": {}}
    te = P.make_splits(len(y))[2]
    corr_p, corr_a = [], []
    for rd in sorted(p for p in out.iterdir() if (p / "DONE").exists()):
        meta = json.loads((rd / "DONE").read_text())
        k, rung = meta["num_classes"], meta["rung"]
        _, qcd, _ = vocabulary(rung, k)
        dist, val = parse_train_log((pathlib.Path(meta["run_dir"]) / "train.log")
                                    .read_text(errors="replace"))
        r = {**{kk: meta[kk] for kk in ("rung", "num_classes", "num_reg", "best_epoch")},
             "repro_features_e79": meta.get("repro_features_e79"), "checkpoints": {}}
        tags = list(meta["checkpoints"]) + ["ens"]
        if meta["best_epoch"] is not None and "best" not in tags:
            r["best_is"] = f"e{meta['best_epoch']}"
        correct, p_qcd, scores = [], [], []          # per jet, epochs only: the paired bootstrap
        for tag in tags:
            d = np.load(rd / f"{tag}.npz")
            c, s = {"head": head_summary(d["correct"], d["p_qcd"], d["log_odds"], is_qcd)}, None
            if "feat_bvc" in d.files:
                c["probe"], s = probe_summary(d["feat_bvc"], y)
            if tag in meta["checkpoints"]:
                c["sha256"] = meta["checkpoints"][tag]["sha256"]
            if is_epoch(tag):
                e = int(tag[1:])
                c["stream_qcd_share"] = qcd_share(dist[e], qcd) if e in dist else None
                c["logged_val_metric"] = val.get(e)
                if s is None:
                    raise SystemExit(f"FATAL: {rd.name} {tag} has no probe features")
                correct.append(d["correct"])
                p_qcd.append(d["p_qcd"])
                scores.append(s)
            r["checkpoints"][tag] = c
        ep = [t for t in r["checkpoints"] if is_epoch(t)]
        H = [r["checkpoints"][t]["head"] for t in ep]
        C = np.stack(correct, 1).astype(np.float32)
        Q = np.stack(p_qcd, 1)
        S, yt = np.stack(scores, 1), y[te]
        sh = [r["checkpoints"][t]["stream_qcd_share"] for t in ep]
        r["epochs_70_79"] = {
            "head_top1": paired_swing(lambda i: C[i].mean(0), len(C)),
            "head_p_qcd_resonant": paired_swing(lambda i: Q[i][~is_qcd[i]].mean(0), len(Q)),
            "probe_log1m_auc": paired_swing(
                lambda i: [P.log1m_auc(yt[i], S[i, j])[0] for j in range(S.shape[1])], len(yt)),
            "mean_of_epoch_top1": float(np.mean([h["top1_accuracy"] for h in H]))}
        if None not in sh:
            pq_ = [h["mean_p_qcd_resonant"] for h in H]
            acc = [h["top1_accuracy"] for h in H]
            r["epochs_70_79"]["stream_qcd_share_range"] = [min(sh), max(sh)]
            r["epochs_70_79"]["pearson_share_vs_p_qcd_resonant"] = float(np.corrcoef(sh, pq_)[0, 1])
            r["epochs_70_79"]["pearson_share_vs_top1"] = float(np.corrcoef(sh, acc)[0, 1])
            corr_p.append((sh, pq_))
            corr_a.append((sh, acc))
        res["runs"][rd.name] = r
    res["pooled_within_run"] = {"share_vs_p_qcd_resonant": within_run_correlation(corr_p),
                                "share_vs_top1": within_run_correlation(corr_a)}
    res.update(summarise(res))
    a.json.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.json}")
    return 0


def summarise_json(a) -> int:
    """summarise() on an existing analysis JSON, written to a separate file: the
    defect counts and the class-mix result need only the head summaries, so they
    are recomputed from the committed JSON without the per-jet arrays."""
    if a.out.resolve() == a.json.resolve():
        raise SystemExit("FATAL: --out would overwrite the analysis JSON")
    doc = {"source": {"file": a.json.name, "sha256": sha256(a.json)},
           **summarise(json.loads(a.json.read_text()))}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


# ---------------------------------------------------------------- figure
def figure(a) -> int:
    sys.path.insert(0, str(REPO / "experiments/FIGS"))
    import style
    import matplotlib.pyplot as plt
    d = json.loads(a.json.read_text())
    defective = set(a.defective)
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(1, 3, figsize=(style.FIG_W_TWO_COLUMN, 2.5))
    for run, r in d["runs"].items():
        ep = sorted((int(t[1:]), c) for t, c in r["checkpoints"].items() if is_epoch(t))
        x = [e for e, _ in ep]
        kw = {"color": style.LEVEL_COLOURS[r["num_classes"]], "marker": style.LEVEL_MARKERS[r["num_classes"]],
              "markersize": 3, "linewidth": 1, "linestyle": "-" if run in defective else ":"}
        seed = re.search(r"-s(\d+)", run).group(1)       # outside the f-string: Python 3.10 (the image)
        tag = (f"{r['num_classes']} classes{' + mass' if r['num_reg'] else ''}, run {seed}"
               + (" (defective at 79)" if run in defective else ""))
        ax[0].plot(x, [c["head"]["top1_accuracy"] for _, c in ep], label=tag, **kw)
        ax[0].plot([x[-1] + 2], [r["checkpoints"]["wavg"]["head"]["top1_accuracy"]],
                   **{**kw, "linestyle": "none", "fillstyle": "none"})
        ax[1].plot(x, [c["probe"]["log1m_auc"] for _, c in ep], **kw)
        ax[2].plot([c["stream_qcd_share"] for _, c in ep],
                   [c["head"]["mean_p_qcd_resonant"] for _, c in ep], **{**kw, "linestyle": "none"})
    ticks = list(range(70, 80, 3))
    ax[0].set_xticks(ticks + [81], [str(t) for t in ticks] + ["avg"])
    ax[0].set(xlabel="epoch (avg: weights averaged over 70-79)", ylabel="top-1 accuracy",
              title="(a) output layer")
    ax[1].set(xlabel="epoch", ylabel=r"$\log(1-\mathrm{AUC})$", title=r"(b) linear probe, X$\to$bb vs X$\to$cc")
    ax[2].set(xlabel="QCD share of the epoch's training jets", ylabel="mean P(QCD) on resonant jets",
              title="(c) output layer vs stream")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout()
    a.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.outdir / f"head_epoch_diag.{ext}", dpi=style.DPI, bbox_inches="tight")
    plt.close(fig)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--runs", nargs="+", required=True, metavar="RUN_DIR:RUNG:K:NUM_REG")
    i.add_argument("--epochs", type=int, nargs="+", default=list(range(70, 80)))
    i.add_argument("--max-epoch", type=int, default=79, help="search 0..this for the best checkpoint's epoch")
    i.add_argument("--data-test", nargs="+", required=True)
    i.add_argument("--max-jets", type=int, required=True)
    i.add_argument("--stride", type=int, required=True)
    i.add_argument("--align-with", type=pathlib.Path, required=True)
    i.add_argument("--cache-root", default="/data/results/eval")
    i.add_argument("--batch-size", type=int, default=512)
    i.add_argument("--num-workers", type=int, default=1, help="1 keeps the file order")
    i.add_argument("--out", type=pathlib.Path, required=True)
    n = sub.add_parser("analyse")
    n.add_argument("--out", type=pathlib.Path, required=True)
    n.add_argument("--json", type=pathlib.Path, required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--json", type=pathlib.Path, required=True, help="an analysis JSON (read only)")
    s.add_argument("--out", type=pathlib.Path, required=True)
    f = sub.add_parser("figure")
    f.add_argument("--json", type=pathlib.Path, required=True)
    f.add_argument("--outdir", type=pathlib.Path, required=True)
    f.add_argument("--defective", nargs="+", default=list(AUDIT_FLAGGED))
    a = ap.parse_args(argv)
    return {"infer": infer, "analyse": analyse, "summarise": summarise_json, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
