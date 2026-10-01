#!/usr/bin/env python3
"""Extraction v2: one pass over the test split, any set of checkpoints, and only
the rows each analysis needs.

WHY (audit 2026-09-29, B3, B4 and must-fix 3, 8). Three defects of the v1
caches (extract_features.py, 2,000,000 jets from epoch 79):

  1. Too few test jets. The b vs c two-prong rejection at 90 % signal
     efficiency rested on 6-12 passing background jets per model out of 11,876.
     v2 scores every jet of every probe task's classes in the whole test split
     (--feature-classes probe), so the count is set by the split, and reports
     how many background jets are expected to pass (class_counts.py).
  2. One checkpoint, the last. Four of 30 epoch-79 output layers are defective
     while their features probe normally. v2 takes the checkpoint as a
     parameter -- the best-validation one (from the run's per-epoch record) and
     any of epochs 70-79 -- and runs them all in ONE pass over the data, so
     every checkpoint sees the same jets in the same order.
  3. Storage. v1 kept float32 features for every jet. v2 keeps float16 features
     only for the rows an analysis reads, and for the output layer keeps only
     the per-jet scores the analyses use (head_scores), never K-wide logits:

       features     float16, rows whose native label is in --feature-classes
                    (all of the split), plus every row of the first
                    --prefix-features jets (the label-recovery and
                    representation-anomaly set, identical to the v1 caches' rows)
       head scores  for the anomaly rows (QCD and the six signals) in the first
                    --head-prefix jets and a stride sample (--diag-stride) of
                    them: argmax, P(QCD), the resonant-vs-QCD log-odds, P(own
                    class), and the two class-sum anomaly scores per signal,
                    computed from float32 logits with anomaly.py's own function

Output, per checkpoint:  <out>/<tag>/{features.npy, rows.npy, label188.npy,
observers.npz, head_scores.npz, manifest.json}, tag = "bestval", "wavg" or "e079".
rows.npy indexes the test stream, so any two checkpoints, runs or vocabularies are
row-aligned by construction and checked by the stream's label sha256.
observers.npz holds, for the same rows, the kinematics a windowed probe task cuts
on (jet_pt, jet_eta, jet_sdmass) and the mass-regression truth (genjet_sdmass):
without them bc_vs_rest's window cannot be applied and mass_resolution.py has no
target (verification 2026-10-01). genjet_sdmass is an observer of
configs/data/JetClassII_massreg.yaml only, which is JetClassII_base.yaml plus that
one observer, so the default data config is that file.

The model is the same code extract_features.py uses (build_model,
load_trunk_or_die); only the row selection and the storage differ.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def probe_classes() -> list[int]:
    """Every native class any probe task reads (probe.TASKS), windowed ones included."""
    probe = _load("probe", "experiments/EVAL/probe.py")
    out = set()
    for spec in probe.TASKS.values():
        out |= set(spec["signal"]) | set(spec["background"])
    return sorted(out)


# X->bc, X->cs and X->bq over the WHOLE split, whatever probe.TASKS says today:
# probe.py is gaining two single-pair b-vs-c tasks, X->bc vs X->bq and X->bc vs
# X->cs, with no kinematic window (2026-10-01), and a cache made before they land
# must already hold their rows. bc_vs_rest alone would keep them only inside the
# |V_cb| window.
FULL_RANGE_CLASS_NAMES = ("label_X_bc", "label_X_cs", "label_X_bq")


def full_range_classes() -> list[int]:
    an = _load("anomaly", "experiments/EVAL/anomaly.py")
    by_name = {r["class_name"]: int(r["jet_label"]) for r in an.read_map()}
    return sorted(by_name[n] for n in FULL_RANGE_CLASS_NAMES)


def probe_feature_rules() -> tuple[list[int], list[tuple[list[int], dict]]]:
    """(classes kept everywhere, [(classes, window)] kept only inside a window).

    A class that only a WINDOWED task reads (bc_vs_rest's background: X->bqq and
    all 27 QCD classes) is needed only inside that task's window; keeping it over
    the whole split stored 3.6 M QCD rows per model for ~0.1 M in-window ones
    (class_counts.py, 2026-09-30). A class any unwindowed task reads, and every
    FULL_RANGE_CLASS_NAMES class, is kept everywhere."""
    probe = _load("probe", "experiments/EVAL/probe.py")
    anywhere, windowed = set(full_range_classes()), []
    for spec in probe.TASKS.values():
        cls = set(spec["signal"]) | set(spec["background"])
        if spec.get("window"):
            windowed.append((sorted(cls), dict(spec["window"])))
        else:
            anywhere |= cls
    windowed = [(sorted(set(c) - anywhere), w) for c, w in windowed]
    return sorted(anywhere), [(c, w) for c, w in windowed if c]


def anomaly_classes() -> tuple[list[int], dict[str, int]]:
    """(QCD native labels, {signal name: native label}) of anomaly.py's suite."""
    an = _load("anomaly", "experiments/EVAL/anomaly.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    by_name = {r["class_name"]: int(r["jet_label"]) for r in an.read_map()}
    return probe.qcd_indices(), {s: by_name[s] for s in an.SIGNAL_SUITE}


def best_epoch(run_dir: pathlib.Path) -> int:
    """The v2 primary checkpoint: the epoch with the highest validation selection
    metric on the fixed sample (experiments/MTX/pretrain_v2.py writes it per epoch
    to metrics/epoch-EEE.json; strict improvement, so ties go to the earlier
    epoch). Recomputed from the per-epoch files and checked against the run's
    own best_epoch.json, which a restart could otherwise leave stale."""
    recs = sorted((run_dir / "metrics").glob("epoch-*.json"))
    if not recs:
        raise SystemExit(f"FATAL: {run_dir}/metrics holds no per-epoch record")
    vals = {}
    for f in recs:
        r = json.loads(f.read_text())
        vals[int(r["epoch"])] = float(r["selection"]["value"])
    best = min(vals, key=lambda e: (-vals[e], e))
    rec = json.loads((run_dir / "best_epoch.json").read_text())
    if int(rec["epoch"]) != best:
        raise SystemExit(f"FATAL: {run_dir}/best_epoch.json says epoch {rec['epoch']}, the "
                         f"per-epoch records say {best}")
    return best


WAVG_FILE = "net_wavg70-79_state.pt"
# kept beside the feature rows: the |V_cb| window's cuts and the mass truth
V2_OBSERVERS = ("jet_pt", "jet_eta", "jet_sdmass", "genjet_sdmass")


def resolve_checkpoints(run_dir: pathlib.Path, spec: list[str]) -> list[tuple[str, pathlib.Path]]:
    """[(tag, path)] under the checkpoint rule (draft amendment A8):
      'bestval'  primary: the best epoch on the fixed validation sample, best_epoch()
      'wavg'     robustness: the weight average of epochs 70-79 that v2 pretraining
                 writes, net_wavg70-79_state.pt
      N, 'A-B'   single epochs, for diagnostics (v1 has no fixed validation sample
                 and no weight average of its own)."""
    out = []
    for s in spec:
        if s in ("bestval", "best"):
            out.append(("bestval", run_dir / f"net_epoch-{best_epoch(run_dir)}_state.pt"))
        elif s == "wavg":
            out.append(("wavg", run_dir / WAVG_FILE))
        elif "-" in s:
            a, b = (int(x) for x in s.split("-"))
            out += [(f"e{e:03d}", run_dir / f"net_epoch-{e}_state.pt") for e in range(a, b + 1)]
        else:
            out.append((f"e{int(s):03d}", run_dir / f"net_epoch-{int(s)}_state.pt"))
    missing = [str(p) for _, p in out if not p.exists()]
    if missing:
        raise SystemExit(f"FATAL: checkpoints missing: {missing[:3]}")
    return out


def head_score_columns(logits: np.ndarray, rung: str, signals: dict[str, int]) -> dict:
    """The per-jet output-layer scores the analyses read, from float32 logits.

    Class sums are anomaly.class_sum_without itself, so the anomaly analysis on
    these columns is the committed one on the committed definition."""
    an = _load("anomaly", "experiments/EVAL/anomaly.py")
    z = np.asarray(logits, dtype=np.float32)
    if rung == "none":
        # a vocabulary outside the contraction tree (random or flavour
        # partitions): no QCD node is defined, so no class sum either
        m = z.astype(np.float64).max(axis=1, keepdims=True)
        lse = np.log(np.exp(z.astype(np.float64) - m).sum(axis=1)) + m[:, 0]
        return {"argmax": z.argmax(axis=1).astype(np.int16), "logsumexp": lse.astype(np.float32)}
    node_of, res, qcd = an.node_roles(rung)
    zz = z.astype(np.float64)
    m = zz.max(axis=1, keepdims=True)
    lse = np.log(np.exp(zz - m).sum(axis=1)) + m[:, 0]
    lres = np.log(np.exp(zz[:, sorted(res)] - m).sum(axis=1)) + m[:, 0]
    lqcd = np.log(np.exp(zz[:, sorted(qcd)] - m).sum(axis=1)) + m[:, 0]
    out = {"argmax": z.argmax(axis=1).astype(np.int16),
           "p_qcd": np.exp(lqcd - lse).astype(np.float32),
           "logodds_res_qcd": (lres - lqcd).astype(np.float32),
           "logsumexp": lse.astype(np.float32)}
    for sig, lab in signals.items():
        out[f"class_sum|{sig}"] = an.class_sum_without(z, rung, {node_of[lab]}).astype(np.float32)
        out[f"class_sum_matched|{sig}"] = an.class_sum_without(
            z, rung, an.matched_nodes(rung, lab)).astype(np.float32)
    return out


class Selector:
    """Which stream rows are kept, as features and as head scores."""

    def __init__(self, feature_classes, prefix_features: int, head_classes, head_prefix: int,
                 diag_stride: int, windowed=()):
        self.fc = np.asarray(sorted(feature_classes), dtype=np.int64)
        self.windowed = [(np.asarray(c, dtype=np.int64), dict(w)) for c, w in windowed]
        self.prefix_features = int(prefix_features)
        self.hc = np.asarray(sorted(head_classes), dtype=np.int64)
        self.head_prefix = int(head_prefix)
        self.diag_stride = int(diag_stride)

    def feature_mask(self, rows: np.ndarray, labels: np.ndarray, obs: dict | None = None) -> np.ndarray:
        m = np.isin(labels, self.fc) | (rows < self.prefix_features)
        for cls, win in self.windowed:
            if obs is None or any(k not in obs for k in win):
                raise SystemExit(f"FATAL: a windowed feature rule needs observers {sorted(win)}")
            inside = np.isin(labels, cls)
            for k, (lo, hi) in win.items():
                v = np.asarray(obs[k])
                inside &= (v > lo) & (v < hi)
            m |= inside
        return m

    def head_mask(self, rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
        inside = rows < self.head_prefix
        diag = (rows % self.diag_stride == 0) if self.diag_stride else np.zeros_like(inside)
        return inside & (np.isin(labels, self.hc) | diag)


def run(batches, models: dict, selector: Selector, rung: str, k: int, signals: dict,
        tap_factory, to_inputs, features_at=None, observers=()) -> dict:
    """Stream `batches` (X, y) through every model; keep the selected rows.

    `features_at` names the checkpoints whose features are kept (default: all);
    the others keep head scores only, which is what bounds the storage of the
    robustness checkpoints. `observers` are kept for the feature rows (they are
    the stream's, so one copy serves every checkpoint). Separated from the loader
    so the selection and the storage are testable without weaver's data files."""
    features_at = set(models) if features_at is None else set(features_at)
    import torch
    keep = {t: {"feat": [], "frow": [], "flab": [], "hrow": [], "hlab": [], "logits": []}
            for t in models}
    labels_all, n0 = [], 0
    kept_obs = {o: [] for o in observers}
    taps = {t: tap_factory(m) for t, m in models.items()}
    with torch.no_grad():
        for item in batches:
            X, y = item[0], item[1]
            obs = item[2] if len(item) > 2 else None
            lab = np.asarray(y, dtype=np.int64)
            rows = np.arange(n0, n0 + lab.size)
            n0 += lab.size
            labels_all.append(lab.astype(np.int16))
            fm0 = selector.feature_mask(rows, lab, obs)
            hm = selector.head_mask(rows, lab)
            if features_at and kept_obs:
                missing = [o for o in kept_obs if obs is None or o not in obs]
                if missing:
                    raise SystemExit(f"FATAL: observers {missing} are not in the stream; "
                                     "the data config must list them")
                for o in kept_obs:
                    kept_obs[o].append(np.asarray(obs[o])[fm0].astype(np.float32))
            need = (fm0 if features_at else np.zeros_like(fm0)) | hm
            if not need.any():
                continue
            inputs = to_inputs(X, need)
            for t, model in models.items():
                fm = fm0 if t in features_at else np.zeros_like(fm0)
                if not (fm | hm).any():
                    continue
                out = model(*inputs)
                f = taps[t].buf.float().cpu().numpy()
                z = out.float().cpu().numpy()[:, :k]
                sub_f, sub_h = fm[need], hm[need]
                keep[t]["feat"].append(f[sub_f].astype(np.float16))
                keep[t]["frow"].append(rows[fm])
                keep[t]["flab"].append(lab[fm].astype(np.int16))
                keep[t]["hrow"].append(rows[hm])
                keep[t]["hlab"].append(lab[hm].astype(np.int16))
                keep[t]["logits"].append(z[sub_h].astype(np.float32))
    for tp in taps.values():
        tp.close()
    L = np.concatenate(labels_all) if labels_all else np.zeros(0, np.int16)
    res = {"n_stream": int(L.size), "label188_sha256": hashlib.sha256(L.tobytes()).hexdigest(),
           "labels": L, "checkpoints": {},
           "observers": {o: (np.concatenate(v) if v else np.zeros(0, np.float32))
                         for o, v in kept_obs.items()}}
    for t, kv in keep.items():
        cat = lambda xs, dt, w=None: (np.concatenate(xs) if xs else
                                      np.zeros((0,) if w is None else (0, w), dt))
        logits = cat(kv["logits"], np.float32, k)
        res["checkpoints"][t] = {
            "features": cat(kv["feat"], np.float16, 128),
            "rows": cat(kv["frow"], np.int64), "label188": cat(kv["flab"], np.int16),
            "head_rows": cat(kv["hrow"], np.int64), "head_label188": cat(kv["hlab"], np.int16),
            "head": head_score_columns(logits, rung, signals) if logits.shape[0] else {}}
    return res


def write(out: pathlib.Path, res: dict, meta: dict) -> dict:
    """One directory per checkpoint; the manifest is written last, so a directory
    with a manifest is complete."""
    sizes = {}
    for tag, c in res["checkpoints"].items():
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "features.npy", c["features"])
        np.save(d / "rows.npy", c["rows"])
        np.save(d / "label188.npy", c["label188"])
        np.savez_compressed(d / "head_scores.npz", rows=c["head_rows"],
                            label188=c["head_label188"], **c["head"])
        obs = {}
        if c["rows"].size and res.get("observers"):
            obs = res["observers"]
            bad = [o for o, v in obs.items() if v.shape[0] != c["rows"].size]
            if bad:
                raise SystemExit(f"FATAL: observers {bad} are not aligned with {tag}'s rows")
            np.savez(d / "observers.npz", **obs)
        man = {**meta, **meta["checkpoints"][tag], "tag": tag,
               "n_stream": res["n_stream"], "stream_label188_sha256": res["label188_sha256"],
               "n_feature_rows": int(c["rows"].size), "n_head_rows": int(c["head_rows"].size),
               "features_dtype": "float16",
               "rows_sha256": hashlib.sha256(c["rows"].tobytes()).hexdigest(),
               "head_rows_sha256": hashlib.sha256(c["head_rows"].tobytes()).hexdigest(),
               "observers": sorted(obs),
               "observers_sha256": (hashlib.sha256((d / "observers.npz").read_bytes()).hexdigest()
                                    if obs else None)}
        man.pop("checkpoints", None)
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        sizes[tag] = sum(p.stat().st_size for p in d.iterdir())
    return sizes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--rung", required=True,
                    help="the vocabulary the model was trained on, a column of the "
                         "contraction-tree map, or 'none' for a vocabulary outside it")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--num-reg", type=int, default=0)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="'bestval', 'wavg', an epoch, or a range 'A-B' (e.g. bestval wavg)")
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--data-config", default=str(REPO / "configs/data/JetClassII_massreg.yaml"),
                    help="JetClassII_massreg.yaml = JetClassII_base.yaml + genjet_sdmass")
    ap.add_argument("--observers", nargs="*", default=list(V2_OBSERVERS),
                    help="observers kept beside the feature rows (observers.npz)")
    ap.add_argument("--max-jets", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--feature-classes", nargs="*", default=["probe"],
                    help="native labels whose features are kept everywhere; 'probe' = "
                         "every probe task's classes; nothing = none")
    ap.add_argument("--features-at", nargs="*", default=None,
                    help="checkpoint tags whose features are kept (e.g. bestval wavg); "
                         "default every checkpoint; the rest keep head scores only")
    ap.add_argument("--prefix-features", type=int, default=0,
                    help="also keep features of every row among the first N jets")
    ap.add_argument("--head-prefix", type=int, default=2_000_000)
    ap.add_argument("--no-anomaly-rows", action="store_true",
                    help="head scores only for the stride sample, not the anomaly rows")
    ap.add_argument("--diag-stride", type=int, default=40)
    ap.add_argument("--align-with", type=pathlib.Path, default=None,
                    help="a v1 cache whose label188.npy must equal this stream's first rows")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    import torch
    from weaver.utils.dataset import SimpleIterDataset
    from weaver.utils.data.config import DataConfig
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    tf32 = ex.strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windowed = []
    if a.feature_classes == ["probe"]:
        fcls, windowed = probe_feature_rules()
    else:
        fcls = [int(x) for x in a.feature_classes]
    qcd, signals = anomaly_classes()
    hcls = [] if a.no_anomaly_rows else qcd + list(signals.values())
    sel = Selector(fcls, a.prefix_features, hcls, a.head_prefix, a.diag_stride, windowed)
    ckpts = resolve_checkpoints(a.run_dir, a.checkpoints)
    done = [t for t, _ in ckpts if (a.out / t / "manifest.json").exists()]
    if len(done) == len(ckpts):
        print("every checkpoint already extracted")
        return 0

    dc = DataConfig.load(a.data_config, load_observers=True)
    absent = [o for o in a.observers if o not in dc.observer_names]
    if absent:
        raise SystemExit(f"FATAL: {a.data_config} does not list the observers {absent}")
    models, meta = {}, {"run_dir": str(a.run_dir), "rung": a.rung, "num_classes": a.num_classes,
                        "num_reg": a.num_reg, "data_config": a.data_config,
                        "data_config_sha256": hashlib.sha256(
                            pathlib.Path(a.data_config).read_bytes()).hexdigest(),
                        "n_test_files": len(a.data_test), "max_jets": a.max_jets,
                        "feature_classes": fcls, "feature_windowed": windowed,
                        "prefix_features": a.prefix_features,
                        "head_classes": hcls, "head_prefix": a.head_prefix,
                        "diag_stride": a.diag_stride, "signals": signals,
                        "device": str(device), "device_name": ex.device_name(device),
                        "tf32": tf32, "checkpoints": {},
                        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    for tag, path in ckpts:
        m = ex.build_model(dc, a.num_classes + a.num_reg)
        prov = ex.load_trunk_or_die(m, path, a.num_classes, a.num_reg)
        models[tag] = m.to(device).eval()
        meta["checkpoints"][tag] = {"checkpoint": str(path), "checkpoint_sha256": prov["sha256"]}
    print(f"{len(models)} checkpoints on {device}: {list(models)}", flush=True)

    ds = SimpleIterDataset({"_": list(a.data_test)}, a.data_config, for_training=False,
                           fetch_by_files=True, fetch_step=1, name="extract_v2")
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, drop_last=False,
                                         num_workers=1, pin_memory=True, persistent_workers=False)
    label_name = dc.label_names[0]
    t0, seen = time.time(), [0]

    def batches():
        for X, y, Z in loader:
            yield X, y[label_name].cpu().numpy(), {k: np.asarray(v) for k, v in Z.items()}
            seen[0] += len(y[label_name])
            if seen[0] % (a.batch_size * 400) < a.batch_size:
                print(f"  {seen[0]:,} jets  {seen[0] / (time.time() - t0):.0f} jets/s", flush=True)
            if a.max_jets and seen[0] >= a.max_jets:
                return

    def to_inputs(X, need):
        idx = torch.from_numpy(np.flatnonzero(need))
        return [X[k][idx].to(device, non_blocking=True) for k in dc.input_names]

    res = run(batches(), models, sel, a.rung, a.num_classes, signals, ex.ClsTap, to_inputs,
              features_at=a.features_at, observers=a.observers)
    meta["features_at"] = sorted(a.features_at) if a.features_at is not None else sorted(models)
    if a.head_prefix and res["n_stream"] < a.head_prefix and not a.max_jets:
        raise SystemExit(f"FATAL: the stream ended at {res['n_stream']:,} jets, inside the "
                         f"head prefix of {a.head_prefix:,}")
    if a.align_with is not None:
        # a v1 cache (stride 1) is the first N rows of this same stream: the
        # anomaly draws and the probes index those rows, so they must agree
        v1 = np.load(a.align_with / "label188.npy")
        if not np.array_equal(res["labels"][:v1.size], v1):
            raise SystemExit(f"FATAL: this stream's first {v1.size:,} labels differ from "
                             f"{a.align_with}; not the same jets in the same order")
        meta["aligned_with"] = {"cache": str(a.align_with), "n": int(v1.size),
                                "label188_sha256": hashlib.sha256(v1.tobytes()).hexdigest()}
    sizes = write(a.out, res, meta)
    print(json.dumps({t: f"{s / 1e9:.3f} GB" for t, s in sizes.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
