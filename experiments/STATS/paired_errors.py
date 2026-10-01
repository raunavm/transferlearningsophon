#!/usr/bin/env python3
"""Paired errors on every ratio of two models' metrics the paper quotes.

Audit 2026-09-29 B3 and must-fix 5, 8 and 11(a). The method is in
src/stats/paired.py; this file only feeds it. Three steps:

  replicates   per model, the metric on the test sample and on B bootstrap
               resamplings of it (one resampling per b, shared by every model
               scored on the same jets), from the per-jet outputs:
                 probes        probe.py --save-scores       scores.npz
                 mass probes   mass_resolution.py --save-residuals  residuals.npz
                 fine-tuning   leg 1 logits.npy, leg 2 pred.root (fine-tuning seed s1)
                 anomaly (v2)  anomaly_heads.py's sigma_min, the value alone (B = 0)
  ratios       every comparison of a contrasts file (configs/analysis/), plus a
               Garwood interval on every rejection (per run, and pooled over runs
               with the caveat that pooling runs scored on the same jets
               understates the error). v2: every comparison also at the weight
               average over the best-validation checkpoint (A8).

Metrics are all "lower is better" -- 1 - AUC, eps_B at a fixed signal
efficiency, sigma_eff, 1 - macro AUC, sigma_min -- and every ratio is coarse over fine, so
a ratio above one means the coarser (or control) model is worse. A ratio of
eps_B is the inverse ratio of rejections.

MODELS. A replicate key is family|task|probe|model|metric. A v1 model is a run
name of configs/analysis/contrasts.v1.json; a v2 model is a run of
configs/arms/v2_grid.json at a checkpoint, '<run>@bestval' or '<run>@wavg', so
the two checkpoints of one run never collide. A v2 probe or mass-probe arm is
named through the checkpoint its features came from (the extract_v2.py
manifests, --extract-root), a v2 fine-tuning cell through the rule its
init_checkpoint.json records. A name that is neither a model nor a listed
baseline is fatal. The contrasts -- which arms are compared, how they are paired
-- are data in configs/analysis/contrasts.v{1,2}.json; this file only forms them.
An arm a contrast names must be a model arm, or listed under pending_arms there.

A replicate vector is comparable with another only if both were computed on the
same jets in the same order with the same B and seed; the key `jets` (sha256 of
the row indices and labels) is checked before any pair is formed.

Usage:
  paired_errors.py probe-replicates --probe-dirs DIR... [--extract-root D] --out R.npz
  paired_errors.py mass-replicates  --mass-dirs DIR...  [--extract-root D] --out R.npz
  paired_errors.py ft-replicates    --leg1-root D --leg2-root D [--checkpoint-rule R] --out R.npz
  paired_errors.py anomaly-replicates --anomaly F... --extract-root D --out R.npz
  paired_errors.py ratios --replicates R.npz... [--contrasts F] [--run-dirs-root D] --out ratios.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.stats import paired as P  # noqa: E402

B = 1000
SEED = P.SEED_DEFAULT
CONTRASTS = {"v1": REPO / "configs" / "analysis" / "contrasts.v1.json",
             "v2": REPO / "configs" / "analysis" / "contrasts.v2.json"}


def grid_models(grid: pathlib.Path) -> dict[str, tuple[str, int, str]]:
    """{model: (arm, run, run directory name)} for every run of the v2 grid. A model
    is named as scripts/build_ft_jobs.v2_runs names the run's fine-tuning init: the
    arm lower-cased without underscores, then -s<run>; the self-supervised arm's
    runs are mpm-v2-s<run>. Every run's directory is mtx-<arm slug>-s<run>."""
    out = {}
    for a in json.loads(grid.read_text())["arms"]:
        slug = a["name"].lower().replace("_", "")
        for k in range(1, a["runs"] + 1):
            out[f"mpm-v2-s{k}" if a["objective"] == "mpm" else f"{slug}-s{k}"] = \
                (a["name"], k, f"mtx-{slug}-s{k}")
    return out


def load_spec(path) -> dict:
    """A contrasts file, its model table resolved (v2: from the grid it names)."""
    path = pathlib.Path(path)
    spec = json.loads(path.read_text())
    m = spec["models"]
    spec["models"] = (grid_models(REPO / m["from_grid"]) if "from_grid" in m
                      else {k: (a, int(r), None) for k, (a, r) in m.items()})
    full = path.resolve()
    spec["path"] = str(full.relative_to(REPO)) if full.is_relative_to(REPO) else str(full)
    spec["sha256"] = _sha(path)
    # every arm a contrast names is a model arm, or is listed as pending (decided,
    # not yet in the grid): a misspelt arm would otherwise form no row, silently
    arms = {a for a, *_ in spec["models"].values()}
    pending = spec.setdefault("pending_arms", {})
    named = set()
    for c in spec["contrasts"]:
        named |= set(c.get("arms", [])) | set(c.get("weights", {}))
        named |= {a for pr in c.get("pairs", []) for a in pr}
        named |= {c[k] for k in ("arm", "proxy") if k in c}
    if set(pending) & arms:
        raise SystemExit(f"FATAL: {sorted(set(pending) & arms)} are model arms now; remove them "
                         f"from pending_arms in {spec['path']}")
    if named - arms - set(pending):
        raise SystemExit(f"FATAL: {spec['path']} compares {sorted(named - arms - set(pending))}, "
                         "which are neither model arms nor pending")
    return spec


def parse_model(name: str, spec: dict) -> tuple[str, int, str | None]:
    """(arm, run, checkpoint) of a replicate key's model: 'l162-s1b' (v1) or
    'l188-s1@bestval' (v2). An unknown model or checkpoint is fatal."""
    model, _, tag = name.partition("@")
    if model not in spec["models"]:
        raise SystemExit(f"FATAL: model {model!r} is not a model of {spec['path']}")
    ok = spec.get("checkpoints", [""])
    if (tag or "") not in ok:
        raise SystemExit(f"FATAL: model {name!r}: its checkpoint must be one of {ok} "
                         f"under {spec['path']}")
    arm, run, _ = spec["models"][model]
    return arm, run, tag or None


def extraction_index(root: pathlib.Path, spec: dict) -> dict[str, str]:
    """{checkpoint sha256: 'model@checkpoint'} from every extract_v2.py manifest
    under root (<root>/<run>/<checkpoint>/manifest.json): the model from the
    manifest's run_dir, the checkpoint from its tag. A v2 probe arm is named
    through the checkpoint its features came from, never through its label."""
    by_dir = {d: m for m, (_, _, d) in spec["models"].items()}
    out = {}
    for f in sorted(pathlib.Path(root).glob("*/*/manifest.json")):
        man = json.loads(f.read_text())
        run = pathlib.Path(man["run_dir"]).name
        if run not in by_dir:
            raise SystemExit(f"FATAL: {f}: {run} is not a run of {spec['path']}")
        name = f"{by_dir[run]}@{man['tag']}"
        if out.setdefault(man["checkpoint_sha256"], name) != name:
            raise SystemExit(f"FATAL: checkpoint {man['checkpoint_sha256'][:16]} is both "
                             f"{out[man['checkpoint_sha256']]} and {name}")
    if not out:
        raise SystemExit(f"FATAL: no extraction manifest under {root}")
    return out


def _named(index: dict | None, arm: str, sha: str | None, where) -> str:
    """The model name of a probe arm: its label for v1, its checkpoint's for v2."""
    if index is None:
        return arm
    if sha not in index:
        raise SystemExit(f"FATAL: {where}: arm {arm!r} was fitted on checkpoint "
                         f"{str(sha)[:16]}, which no extraction manifest holds")
    return index[sha]


def jets_key(rows: np.ndarray, labels: np.ndarray) -> str:
    h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def _sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------- probes
def reproduction(dirs: list[pathlib.Path], committed: dict) -> dict:
    """The refit against the committed result it repeats: largest |dAUC| per
    probe, over every task and model. Both are refitted with the committed seeds
    and settings, and are still not bit-reproducible across machines and BLAS
    thread counts. Measured over the five four-vocabulary refits (2026-09-30):
    |dAUC| <= 2.1e-4, which is <= 3 % of 1 - AUC for electron vs muon (the MLP
    near AUC = 1) and <= 0.5 % of 1 - AUC for every other task."""
    out = {}
    for d in dirs:
        ref = committed.get(str(d))
        if ref is None:
            continue
        A = json.loads((d / "probe_results.json").read_text())["tasks"]
        R = json.loads(pathlib.Path(ref).read_text())["tasks"]
        worst = {"linear": 0.0, "mlp": 0.0}
        for task, T in A.items():
            for arm, v in T.get("arms", {}).items():
                for kind in worst:
                    worst[kind] = max(worst[kind], abs(v[kind]["auc"] - R[task]["arms"][arm][kind]["auc"]))
        out[str(d)] = {"committed": str(ref), "committed_sha256": _sha(ref),
                       "max_abs_dauc": worst}
    return out


def probe_replicates(dirs: list[pathlib.Path], b: int = B, seed: int = SEED,
                     index: dict | None = None):
    """Replicates of 1 - AUC and eps_B at every working point, per task x probe x model.

    `index` (v2, extraction_index) names each arm by the checkpoint it was fitted
    on, so a run's bestval and wavg features are two models. A model scored in
    two jobs (the 162-class run 1 is in the ladder and in the 2x2) is kept once,
    from the first directory given, and the other copy's AUC is recorded as a
    reproducibility check."""
    vec, meta, dup = {}, {}, []
    inputs = []
    for d in dirs:
        J = json.loads((d / "probe_results.json").read_text())
        z = np.load(d / "scores.npz")
        inputs.append({"dir": str(d), "probe_results_sha256": _sha(d / "probe_results.json"),
                       "scores_sha256": _sha(d / "scores.npz")})
        if J.get("scores_npz_sha256") != _sha(d / "scores.npz"):
            raise SystemExit(f"FATAL: {d}/scores.npz is not the file its JSON records")
        for task, T in J["tasks"].items():
            if T.get("skipped"):
                continue
            y, rows = z[f"{task}|y"].astype(bool), z[f"{task}|rows"]
            jk = jets_key(rows, y)
            for arm, A in T["arms"].items():
                model = _named(index, arm, J.get("arm_checkpoints", {}).get(arm), d)
                for kind in ("linear", "mlp"):
                    s = z[f"{task}|{kind}|{arm}"]
                    base = f"probe|{task}|{kind}|{model}"
                    if f"{base}|1-auc" in vec:
                        dup.append({"model": base, "dir": str(d),
                                    "auc_here": A[kind]["auc"],
                                    "auc_kept": 1 - vec[f"{base}|1-auc"][0]})
                        continue
                    sc = P.AucScorer(y, s)
                    if abs(sc.auc() - A[kind]["auc"]) > 1e-12:
                        raise SystemExit(f"FATAL: {base}: AUC from scores {sc.auc()} "
                                         f"!= reported {A[kind]['auc']}")
                    vec[f"{base}|1-auc"] = P.replicates(sc, y.size, b, seed)
                    meta[f"{base}|1-auc"] = {"jets": jk, "n": int(y.size),
                                             "censored": bool(A[kind]["log1m_auc_censored"])}
                    for e in T["eps_s"]:
                        es = P.EpsBScorer(y, s, float(e))
                        key = f"{base}|eps_b@{float(e):.2f}"
                        vec[key] = P.replicates(es, y.size, b, seed)
                        nb = int((~y).sum())
                        meta[key] = {"jets": jk, "n": int(y.size), "n_bkg": nb,
                                     "k_pass": float(vec[key][0] * nb)}
                print(f"  {base}", flush=True)
    return vec, meta, {"inputs": inputs, "duplicates": dup}


# ------------------------------------------------------------ mass probes
def mass_replicates(dirs: list[pathlib.Path], b: int = B, seed: int = SEED,
                    index: dict | None = None):
    """Replicates of sigma_eff of the within-class mass residual, per probe x model
    (`index` as for probe_replicates)."""
    vec, meta, inputs = {}, {}, []
    for d in dirs:
        J = json.loads((d / "mass_resolution.json").read_text())
        z = np.load(d / "residuals.npz")
        inputs.append({"dir": str(d), "mass_resolution_sha256": _sha(d / "mass_resolution.json"),
                       "residuals_sha256": _sha(d / "residuals.npz")})
        rows, lab = z["rows"], z["label188"]
        jk = jets_key(rows, lab)
        for arm, A in J["arms"].items():
            model = _named(index, arm, A.get("provenance", {}).get("checkpoint_sha256"), d)
            for kind in ("ridge", "mlp"):
                res = z[f"{kind}|{arm}"].astype(np.float64)
                sc = P.SigmaEffScorer(res)
                if abs(sc() - A[kind]["sigma_eff"]) > 1e-6:
                    raise SystemExit(f"FATAL: {arm}/{kind}: sigma_eff from residuals "
                                     f"{sc()} != reported {A[kind]['sigma_eff']}")
                key = f"mass|resolution|{kind}|{model}|sigma_eff"
                if key in vec:
                    raise SystemExit(f"FATAL: {key} appears twice among the mass-probe inputs")
                vec[key] = P.replicates(sc, res.size, b, seed)
                meta[key] = {"jets": jk, "n": int(res.size)}
                print(f"  {key}", flush=True)
    return vec, meta, {"inputs": inputs}


# ------------------------------------------------------- anomaly detection
def anomaly_replicates(paths: list[pathlib.Path], index: dict):
    """sigma_min (lower = more sensitive) per model, checkpoint, signal, score and
    injected signal count, from experiments/EVAL/anomaly_heads.py outputs (the
    output-layer scores at every extracted checkpoint; A13's unseen-family
    signal is X->YY->bbbb). Key: anomaly|<signal>|<score>|<model>|sigma_min@<N_sig>.

    No resampling (B = 0): a run's value is the median over its detector
    trainings, on injection draws seeded by the model's own name
    (anomaly.cell_seed), so that noise is the run's own and sits in the spread
    over runs. Two models are compared only on one background pool and set of
    settings (`jets`). A value is a bound, not a measurement, when its max SIC is
    at the background-statistics ceiling (at_ceiling) or below the detection
    threshold (anomaly_summary.NOT_DETECTED_MAX_SIC: sigma_min then sits near
    sigma_t)."""
    nd = _load("anomaly_summary", "experiments/EVAL/anomaly_summary.py").NOT_DETECTED_MAX_SIC
    vec, meta, inputs = {}, {}, []
    for p in paths:
        J = json.loads(pathlib.Path(p).read_text())
        inputs.append({"path": str(p), "sha256": _sha(p)})
        jk = hashlib.sha256(json.dumps([J["labels_sha256"], J["n_bkg"], J["n_template"],
                                        J["trainings"]]).encode()).hexdigest()
        for arm, M in J["models"].items():
            for tag, c in M["checkpoints"].items():
                if "anomaly" not in c:
                    continue
                model = _named(index, arm, c.get("checkpoint_sha256"), p)
                if model.partition("@")[2] != tag:
                    raise SystemExit(f"FATAL: {p}: {arm}/{tag} is the checkpoint of {model}")
                for sig, per_n in c["anomaly"].items():
                    for n_sig, fams in per_n.items():
                        for fam, rec in fams.items():
                            if not isinstance(rec, dict) or "sigma_min" not in rec:
                                continue
                            key = f"anomaly|{sig}|{fam}|{model}|sigma_min@{n_sig}"
                            if key in vec:
                                raise SystemExit(f"FATAL: {key} appears twice among the inputs")
                            vec[key] = np.array([float(rec["sigma_min"])])
                            meta[key] = {"jets": jk, "n": int(J["n_bkg"]),
                                         "max_sic": rec.get("max_sic")}
                            why = ("max SIC at its background-statistics ceiling"
                                   if rec.get("at_ceiling") else
                                   f"max SIC below {nd}, not detected" if rec["max_sic"] < nd else None)
                            if why:
                                meta[key].update(censored=True, censor_note=(
                                    why + "; sigma_min is a bound and so is the ratio"))
    return vec, meta, {"inputs": inputs}


# ------------------------------------------------------------ fine-tuning
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ft_cell(job):
    """One fine-tuning cell's replicate vector of 1 - macro AUC, on the rows the
    committed readout used (leg1_metrics / leg2_metrics, stride 4). With a cache
    directory, a finished cell is written there at once and read back on a
    retry, so an evicted pod loses at most the cells in flight. `model` is the
    init's name, with its checkpoint rule for v2 (ft_jobs)."""
    leg, model, n, path, stride, b, seed, cache = job
    if cache is not None:
        c = pathlib.Path(cache) / f"{leg}|{n}|{model}"
        if (c.with_suffix(".json")).exists() and (c.with_suffix(".npy")).exists():
            return (f"ft|{leg}|{n}|{model}", np.load(c.with_suffix(".npy")),
                    json.loads(c.with_suffix(".json").read_text()))
    if leg == "leg1":
        lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
        l1 = _load("leg1_metrics", "experiments/FT/leg1_metrics.py")
        l162 = lr.rung_maps()["L162"]
        lab188 = np.load(path / "label188.npy")
        lut = np.full(188, -1, dtype=np.int64)
        for k, v in l162.items():
            lut[k] = v
        truth = lut[lab188]
        idx = np.arange(0, truth.size, stride)
        probs = l1.softmax(np.load(path / "logits.npy")[idx])
        y = truth[idx]
    else:
        l2 = _load("leg2_metrics", "experiments/FT/leg2_metrics.py")
        names, onehot, scores = l2.read_pred_root(path / "pred.root")
        truth = onehot.argmax(1)
        probs = scores.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        idx = np.arange(0, truth.size, stride)
        probs, y = probs[idx], truth[idx]
    sc = P.MacroAucScorer(y, probs)
    v = P.replicates(sc, y.size, b, seed)
    m = {"jets": jets_key(idx, y), "n": int(y.size), "source": str(path),
         "macro_auc": float(1 - v[0])}
    if cache is not None:
        c.parent.mkdir(parents=True, exist_ok=True)
        tmp = c.parent / (c.name + ".tmp.npy")
        np.save(tmp, v)
        tmp.replace(c.with_suffix(".npy"))
        c.with_suffix(".json").write_text(json.dumps(m))
    return (f"ft|{leg}|{n}|{model}", v, m)


def ft_jobs(leg1_root, leg2_root, cells_json: dict, spec: dict, rule: str | None = None,
            b: int = B, seed: int = SEED, stride: int = 4, cache=None):
    """(the cells to bootstrap, the baselines left out by name), at fine-tuning
    seed s1. v1 (rule None): a cell is its init's name. v2: every cell of the
    tree must record `rule` in its init_checkpoint.json, and is <init>@<rule>.
    An init that is neither a model nor a baseline of the contrasts file is
    fatal: it used to be skipped without a word."""
    jobs, baselines = [], set()
    for leg, root in (("leg1", leg1_root), ("leg2", leg2_root)):
        for init, per_n in cells_json[leg]["cells"].items():
            if init in spec["baselines"]:
                baselines.add(init)               # scratch, the public model: no pair
                continue
            if init not in spec["models"]:
                raise SystemExit(f"FATAL: fine-tuning init {init!r} is neither a model nor "
                                 f"a baseline of {spec['path']}")
            for n, per_seed in per_n.items():
                if "s1" not in per_seed:
                    continue
                cell = pathlib.Path(root) / init / n / "s1"
                model = init
                if rule is not None:
                    rec = cell / "init_checkpoint.json"
                    got = json.loads(rec.read_text()).get("rule") if rec.exists() else None
                    if got != rule:
                        raise SystemExit(f"FATAL: {rec} records checkpoint rule {got!r}; "
                                         f"this tree is {rule!r}")
                    model = f"{init}@{rule}"
                jobs.append((leg, model, n, cell / "features_v2" if leg == "leg1" else cell,
                             stride, b, seed, cache))
    return jobs, sorted(baselines)


def ft_replicates(leg1_root, leg2_root, cells_json: dict, b: int = B, seed: int = SEED,
                  procs: int = 8, stride: int = 4, cache=None, rule: str | None = None):
    """Every cell of the committed metrics files at fine-tuning seed s1 (ft_jobs)."""
    spec = load_spec(CONTRASTS["v1" if rule is None else "v2"])
    jobs, baselines = ft_jobs(leg1_root, leg2_root, cells_json, spec, rule, b, seed, stride, cache)
    vec, meta = {}, {}
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for key, v, m in ex.map(_ft_cell, jobs):
            leg, n, model = key.split("|")[1], key.split("|")[2], key.split("|")[3]
            ref = cells_json[leg]["cells"][model.partition("@")[0]][n]["s1"]["macro_auc_ovr"]
            if abs(m["macro_auc"] - ref) > 1e-9:
                raise SystemExit(f"FATAL: {key}: macro AUC {m['macro_auc']} does not "
                                 f"reproduce the committed {ref}")
            vec[f"{key}|1-macro_auc"], meta[f"{key}|1-macro_auc"] = v, m
            print(f"  {key}  1-macroAUC {v[0]:.6g}  test SD {v[1:].std(ddof=1):.3g}", flush=True)
    return vec, meta, {"cells_sha256": {k: v["sha256"] for k, v in cells_json.items()},
                       "checkpoint_rule": rule, "baselines_not_paired": baselines}


# ------------------------------------------------------------------ ratios
def _group(meta: dict):
    """{(family, task, kind, metric): {model: key}}."""
    out = {}
    for key in meta:
        fam, task, kind, model, metric = key.split("|")
        out.setdefault((fam, task, kind, metric), {})[model] = key
    return out


def between_checkpoints(spec: dict) -> str | None:
    """The tag of the A8 comparison, '<robustness>/<primary>' ('wavg/bestval'),
    when the contrasts file has a checkpoint contrast; else None."""
    c = next((c for c in spec["contrasts"] if c["kind"] == "checkpoint"), None)
    return None if c is None else f"{c['robustness']}/{c['primary']}"


class _Cell:
    """One family x task x probe x metric: its models by (arm, checkpoint) and run.

    A8 asks for every result at the robustness checkpoint beside the primary
    one. So, when the contrasts file has a checkpoint contrast, each run with both
    checkpoints also enters as '<run>@wavg/bestval', whose replicate vector is
    the ratio of the two (one run, one resampling of the test jets), and every
    contrast is formed at that tag too: the double ratio, the result at the
    weight average over the result at the best-validation checkpoint."""

    def __init__(self, group: tuple, by_model: dict, vec: dict, meta: dict, spec: dict, root):
        self.fam, self.task, self.kind, self.metric = group
        self.key, self.spec, self.root = by_model, spec, root
        self.vecs = {m: vec[k] for m, k in by_model.items()}
        self.metas = {m: meta[k] for m, k in by_model.items()}
        self.arms = {}
        for model in by_model:
            arm, run, tag = parse_model(model, spec)
            runs = self.arms.setdefault((arm, tag), {})
            if run in runs:
                raise SystemExit(f"FATAL: {'/'.join(group)}: {runs[run]} and {model} are both "
                                 f"run {run} of {arm}")
            runs[run] = model
        self.both = between_checkpoints(spec)
        if self.both is not None:
            q, p = self.both.split("/")
            for (arm, tag), runs in list(self.arms.items()):
                other = self.arms.get((arm, q), {}) if tag == p else {}
                for run in sorted(set(runs) & set(other)):
                    mp, mq = runs[run], other[run]
                    if self.metas[mp]["jets"] != self.metas[mq]["jets"]:
                        raise SystemExit(f"FATAL: {'/'.join(group)}: {mp} and {mq} were not "
                                         "scored on the same jets")
                    a, b = self.vecs[mq], self.vecs[mp]
                    m = f"{mp.partition('@')[0]}@{self.both}"
                    self.vecs[m] = np.divide(a, b, out=np.zeros_like(a, dtype=np.float64),
                                             where=(a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b))
                    cens = next((self.metas[x] for x in (mp, mq) if self.metas[x].get("censored")), None)
                    self.metas[m] = {"jets": self.metas[mp]["jets"], "censored": cens is not None}
                    if cens is not None and "censor_note" in cens:
                        self.metas[m]["censor_note"] = cens["censor_note"]
                    self.arms.setdefault((arm, self.both), {})[run] = m
        self.tags = sorted({t for _, t in self.arms}, key=lambda t: (t is not None, t or ""))

    def ck(self, tag):
        """The checkpoint(s) a stream check at `tag` covers."""
        return tuple(tag.split("/")) if self.both is not None and tag == self.both else tag

    def runs(self, arm, tag) -> dict:
        return self.arms.get((arm, tag), {})

    def v(self, model) -> np.ndarray:
        return self.vecs[model]

    def label(self, arm) -> str:
        return self.spec["labels"].get(arm, arm)

    def row(self, contrast: str, fine: str, coarse: str, tag, **kw) -> dict:
        return {"contrast": contrast, "family": self.fam, "task": self.task, "kind": self.kind,
                "metric": self.metric, "checkpoint": tag, "fine": fine, "coarse": coarse, **kw}

    def check(self, models, what) -> list:
        """Same jets for all; the models whose metric is zero or not finite
        somewhere (no log)."""
        if len({self.metas[m]["jets"] for m in models}) != 1:
            raise SystemExit(f"FATAL: {self.fam}/{self.task}/{self.kind}/{self.metric} {what}: "
                             "the models were not scored on the same jets")
        return [m for m in models if not np.all((self.v(m) > 0) & np.isfinite(self.v(m)))]

    def undefined(self, models) -> str:
        return (ANOMALY_UNDEFINED if self.fam == "anomaly" else ZERO).format(models)

    def dirs(self, models):
        """{model: run directory} when streams are checked (v2 with a root), else None."""
        if self.root is None:
            return None
        return {m: pathlib.Path(self.root) / self.spec["models"][m.partition("@")[0]][2]
                for m in models}

    def censor(self, row, models):
        censored = [m for m in models if self.metas[m].get("censored")]
        if censored:
            row["censored_models"] = censored
            row["note"] = self.metas[censored[0]].get("censor_note", AUC_FLOOR)


AUC_FLOOR = "an AUC of 1 is floored at one discordant pair; the ratio is a bound, not a value"
ZERO = ("{} pass no background jet in the test sample or in a resampling; the ratio "
        "of rejections is undefined there -- see the Poisson intervals")
ANOMALY_UNDEFINED = "{} have sigma_min 0 or infinite; the ratio is undefined"


def _pairs(cell: _Cell, c: dict) -> list[dict]:
    """Run k of the fine arm against run k of the coarse arm, at each checkpoint."""
    pairs = (c["pairs"] if c["kind"] == "pairs" else
             [(a, b) for i, a in enumerate(c["arms"]) for b in c["arms"][i + 1:]])
    rows = []
    for tag in cell.tags:
        for fa, ca in pairs:
            fr, cr = cell.runs(fa, tag), cell.runs(ca, tag)
            common = [r for r in cr if r in fr]
            if not common:
                continue
            fm, cm = [fr[r] for r in common], [cr[r] for r in common]
            row = cell.row(c["id"], cell.label(fa), cell.label(ca), tag, fine_arm=fa,
                           coarse_arm=ca, fine_models=fm, coarse_models=cm)
            zero = cell.check(fm + cm, f"{fa} vs {ca}")
            if zero:
                row["not_computed"] = cell.undefined(zero)
            else:
                row.update(P.paired_ratio({m: cell.v(m) for m in fm}, {m: cell.v(m) for m in cm},
                                          pairs=dict(zip(cm, fm)), run_dirs=cell.dirs(fm + cm),
                                          checkpoint=cell.ck(tag)))
            if c.get("fine_run_spread") and all(cell.v(m)[0] > 0 for m in fr.values()):
                pts = np.log([cell.v(m)[0] for m in fr.values()])
                row["fine_run_spread"] = {"arm": fa, "n_runs": len(pts),
                                          "ln_run_sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else None,
                                          "range": [float(np.exp(pts.min())), float(np.exp(pts.max()))]}
                row["coarse_values"] = [float(cell.v(m)[0]) for m in cm]
            cell.censor(row, fm + cm)
            rows.append(row)
    return rows


def _unpaired(cell: _Cell, c: dict) -> list[dict]:
    """Every run of the coarse arm against every run of the fine arm, unpaired:
    exp(mean ln m_coarse - mean ln m_fine), each arm with its own spread over
    runs (Welch): the runs that leave a family out need not vary as much as their
    parent's, and five runs against three do not let one spread stand for both."""
    rows = []
    for tag in cell.tags:
        for fa, ca in c["pairs"]:
            fm, cm = list(cell.runs(fa, tag).values()), list(cell.runs(ca, tag).values())
            if not fm or not cm:
                continue
            row = cell.row(c["id"], cell.label(fa), cell.label(ca), tag, fine_arm=fa,
                           coarse_arm=ca, fine_models=fm, coarse_models=cm,
                           stream_pairing="exempt: " + c["stream_exempt"])
            zero = cell.check(fm + cm, f"{fa} vs {ca}")
            if zero:
                row["not_computed"] = cell.undefined(zero)
            else:
                L = np.log([cell.v(m) for m in cm + fm])
                w = [1 / len(cm)] * len(cm) + [-1 / len(fm)] * len(fm)
                groups = [range(len(cm)), range(len(cm), len(cm) + len(fm))]
                row.update(n_fine_runs=len(fm), n_coarse_runs=len(cm),
                           **P.contrast(L, w, groups, separate=True))
            cell.censor(row, fm + cm)
            rows.append(row)
    return rows


def _linear(cell: _Cell, c: dict) -> list[dict]:
    """Per run k, sum over arms of weight x ln m (a ratio of ratios, or a ratio
    against a fraction of another), paired over the runs every arm has."""
    w = c["weights"]
    rows = []
    for tag in cell.tags:
        runs = [r for r in cell.runs(next(iter(w)), tag) if all(r in cell.runs(a, tag) for a in w)]
        if not runs:
            continue
        member = {r: {a: cell.runs(a, tag)[r] for a in w} for r in runs}
        fm = [member[r][a] for r in runs for a in w if w[a] < 0]
        cm = [member[r][a] for r in runs for a in w if w[a] > 0]
        row = cell.row(c["id"], c["fine"], c["coarse"], tag, weights=w,
                       fine_models=fm, coarse_models=cm)
        zero = cell.check(fm + cm, c["id"])
        if zero:
            row["not_computed"] = cell.undefined(zero)
        else:
            ln = {r: sum(x * np.log(cell.v(member[r][a])) for a, x in w.items()) for r in runs}
            d = cell.dirs(fm + cm)
            row.update(P.paired_log(ln, checkpoint=cell.ck(tag), run_dirs=None if d is None else
                                    {r: [d[m] for m in member[r].values()] for r in runs}))
        cell.censor(row, fm + cm)
        rows.append(row)
    return rows


def _checkpoint(cell: _Cell, c: dict) -> list[dict]:
    """A8: every model at the robustness checkpoint against itself at the primary
    one, paired by run (one run, one stream). Robust = the two agree within
    their combined error."""
    p, q = c["primary"], c["robustness"]
    rows = []
    for arm in dict.fromkeys(a for a, _ in cell.arms):
        fr, cr = cell.runs(arm, p), cell.runs(arm, q)
        common = [r for r in cr if r in fr]
        if not common:
            continue
        fm, cm = [fr[r] for r in common], [cr[r] for r in common]
        row = cell.row(c["id"], f"{cell.label(arm)} at {p}", f"{cell.label(arm)} at {q}",
                       cell.both, fine_arm=arm, coarse_arm=arm, fine_models=fm, coarse_models=cm)
        zero = cell.check(fm + cm, f"{arm} {q} vs {p}")
        if zero:
            row["not_computed"] = cell.undefined(zero)
        else:
            row.update(P.paired_ratio({m: cell.v(m) for m in fm}, {m: cell.v(m) for m in cm},
                                      pairs=dict(zip(cm, fm))))
            row["stream_pairing"] = "same run"
            row["robust_to_checkpoint"] = bool(abs(row["ln_ratio"]) <= row["ln_combined_se"])
        cell.censor(row, fm + cm)
        rows.append(row)
    return rows


def _draws(cell: _Cell, c: dict) -> list[dict]:
    """BETWEEN RANDOM DRAWS (must-fix 11a). One run per draw, so no run spread of
    their own. The difference of two single runs borrows the run variance of the
    proxy arm (17 classes, same head width), var a = max(s^2 - v_ind, 0) over its
    runs, once for each draw, and keeps its own full test variance:
    var = 2 var a + test SE^2. The interval is Student t at the
    Welch-Satterthwaite degrees of freedom, the proxy's spread carrying its n - 1."""
    rows = []
    for tag in cell.tags:
        draws, prox = cell.runs(c["arm"], tag), cell.runs(c["proxy"], tag)
        if len(draws) < 2 or len(prox) < 2:
            continue
        pv = np.array([cell.v(m) for m in prox.values()])
        sd = vind = var_a = float("nan")
        if np.all(pv > 0):
            e = P.combined_error(np.log(pv), np.full(len(pv), 1 / len(pv)))
            sd, vind = e["ln_spread_sd"], e["v_ind"]
            var_a = max(sd ** 2 - vind, 0.0)
        ds = sorted(draws)
        for i, di in enumerate(ds):
            for dj in ds[i + 1:]:
                mi, mj = draws[di], draws[dj]
                row = cell.row(c["id"], f"random draw {di}", f"random draw {dj}", tag,
                               fine_models=[mi], coarse_models=[mj])
                if cell.check([mi, mj], f"random draws {di} and {dj}"):
                    row["not_computed"] = "a draw passes no background jet"
                else:
                    row.update(P.paired_ratio({mi: cell.v(mi)}, {mj: cell.v(mj)}, pairs={mj: mi}))
                    var = 2 * var_a + row["ln_test_se"] ** 2
                    comb = float(np.sqrt(var))
                    dof = var ** 2 / ((2 * sd ** 2) ** 2 / (len(pv) - 1)) if sd > 0 else float("inf")
                    t = P._t975(dof) if np.isfinite(comb) else float("nan")
                    row.update({"run_sd_proxy_17": sd, "run_v_ind_proxy_17": vind,
                                "ln_combined_se_with_proxy": comb, "dof_with_proxy": dof,
                                "ci95_with_proxy": [float(np.exp(row["ln_ratio"] - t * comb)),
                                                    float(np.exp(row["ln_ratio"] + t * comb))]})
                rows.append(row)
    return rows


def partition_design(spec: dict, arms: list[str]) -> list[dict]:
    """A10: [{task, pairs, merged, split}] per probe task and merge pattern. Which
    probe pair each partition merges is read from probe_pairs.v2.json and must
    agree with the partition merge table (rand_v2_selection.json). Each pair is
    read on its own task, named in probe_pairs.v2.json's balance_pairs (since
    2026-10-01 X->bc vs X->bq and vs X->cs each have a single-pair task, and
    are also in the mixed |V_cb| task); a file without balance_pairs must hold
    each pair in one task only. Pairs of one task that every partition merges
    or splits alike cannot be told apart by any probe and form one entry."""
    pp = json.loads((REPO / spec["probe_pairs"]).read_text())
    sel = json.loads((REPO / spec["partition_merges"]).read_text())
    own = pp.get("balance_pairs")
    order = list(sel["pairs"])
    out = {}
    for name, (a, b) in sel["pairs"].items():
        sub = f"{a.removeprefix('label_')}|{b.removeprefix('label_')}"
        if own is not None:
            if name not in own or list(own[name]["classes"]) != [a, b]:
                raise SystemExit(f"FATAL: probe pair {name} ({a}, {b}) of the partition merge "
                                 "table is not a balance pair of probe_pairs.v2.json")
            tasks = [own[name]["task"]]
            if sub not in pp["probe_tasks"].get(tasks[0], {}).get("sub_pairs", []):
                raise SystemExit(f"FATAL: probe pair {name} ({sub}): its own task {tasks[0]!r} "
                                 "does not hold it")
        else:
            tasks = [t for t, T in pp["probe_tasks"].items() if sub in T["sub_pairs"]]
            if len(tasks) != 1:
                raise SystemExit(f"FATAL: probe pair {name} ({sub}) is in {len(tasks)} probe "
                                 "tasks and probe_pairs.v2.json names none as its own")
        pattern = []
        for arm in arms:
            st = pp["status"][arm][tasks[0]]
            merged = sub in st["merged"]
            if merged == (sub in st["split"]):
                raise SystemExit(f"FATAL: {arm} {sub}: neither merged nor split, or both")
            if bool(sel["merge_vectors"][str(pp["partition_seeds"][arm])][order.index(name)]) != merged:
                raise SystemExit(f"FATAL: {arm} {name}: probe_pairs.v2.json and the partition "
                                 "merge table disagree")
            pattern.append(merged)
        out.setdefault((tasks[0], tuple(pattern)), []).append(name)
    return [{"task": t, "pairs": names, "merged": [a for a, m in zip(arms, pat) if m],
             "split": [a for a, m in zip(arms, pat) if not m]} for (t, pat), names in out.items()]


def _partition_runs(cell: _Cell, arms, tag):
    """The runs every partition has at `tag`, less those whose streams differ
    across the partitions up to the checkpoint (A7): (runs, excluded or None)."""
    runs = [r for r in cell.runs(arms[0], tag) if all(r in cell.runs(a, tag) for a in arms)]
    if cell.root is None:
        return runs, None
    d = cell.dirs([cell.runs(a, tag)[r] for a in arms for r in runs])
    excluded = []
    for r in list(runs):
        bad = P.stream_check([d[cell.runs(a, tag)[r]] for a in arms], cell.ck(tag))
        if bad:
            excluded.append({"run": r, **bad})
            runs.remove(r)
    return runs, excluded


def _partition_split_vs_merged(cell: _Cell, c: dict) -> list[dict]:
    """A10 P1, per probe pair: exp(mean over merging partitions - mean over
    splitting ones) of the run-averaged ln metric, the partitions the units, the
    spread pooled within the two groups."""
    if cell.fam != "probe":
        return []
    rows = []
    for d in partition_design(cell.spec, c["arms"]):
        if d["task"] != cell.task:
            continue
        for tag in cell.tags:
            runs, excluded = _partition_runs(cell, c["arms"], tag)
            if not runs:
                continue
            units = d["merged"] + d["split"]
            fm = [cell.runs(a, tag)[r] for a in d["split"] for r in runs]
            cm = [cell.runs(a, tag)[r] for a in d["merged"] for r in runs]
            row = cell.row(c["id"], "partitions that split " + " and ".join(d["pairs"]),
                           "partitions that merge " + " and ".join(d["pairs"]), tag,
                           probe_pairs=d["pairs"], merged_partitions=d["merged"],
                           split_partitions=d["split"], runs=runs,
                           fine_models=fm, coarse_models=cm)
            if excluded is not None:
                row.update(stream_pairing="identical", excluded_runs=excluded)
            zero = cell.check(fm + cm, c["id"])
            if zero:
                row["not_computed"] = cell.undefined(zero)
            elif not d["merged"] or not d["split"]:
                row["not_computed"] = "every partition merges, or every one splits, this pair"
            else:
                L = np.array([np.mean([np.log(cell.v(cell.runs(a, tag)[r])) for r in runs], axis=0)
                              for a in units])
                nm, ns = len(d["merged"]), len(d["split"])
                row.update(P.contrast(L, [1 / nm] * nm + [-1 / ns] * ns,
                                      [range(nm), range(nm, nm + ns)]))
            cell.censor(row, fm + cm)
            rows.append(row)
    return rows


def _partition_joint(cells: dict, c: dict) -> list[dict]:
    """A10 P1, all probe pairs at once: the run-averaged ln metric of every task x
    partition on a task intercept, a partition effect shared by every task (how
    good a partition is overall), and one merge effect per task and merge
    pattern (P.fixed_effects). A merge effect is the cost of merging the pair
    with each partition's overall quality held fixed. Pairs that share a merge
    pattern in different tasks stay separate effects; in one task they are one."""
    arms = c["arms"]
    design = partition_design(next(iter(cells.values())).spec, arms)
    tasks = list(dict.fromkeys(d["task"] for d in design))
    by_km = {}
    for (fam, task, kind, metric), cell in cells.items():
        if fam == "probe" and task in tasks:
            by_km.setdefault((kind, metric), {})[task] = cell
    rows = []
    for (kind, metric), per_task in sorted(by_km.items()):
        ts = [t for t in tasks if t in per_task]
        effects = [d for d in design if d["task"] in ts]
        for tag in sorted({t for cell in per_task.values() for t in cell.tags},
                          key=lambda t: (t is not None, t or "")):
            first = per_task[ts[0]]
            runs, excluded = _partition_runs(first, arms, tag)
            runs = [r for r in runs if all(all(r in per_task[t].runs(a, tag) for a in arms)
                                           for t in ts)]
            if not runs:
                continue
            base = {"contrast": c["id"], "family": "probe", "kind": kind, "metric": metric,
                    "checkpoint": tag, "tasks_in_fit": ts, "runs": runs}
            if excluded is not None:
                base.update(stream_pairing="identical", excluded_runs=excluded)
            per = {t: [per_task[t].runs(a, tag)[r] for a in arms for r in runs] for t in ts}
            models = list(dict.fromkeys(m for t in ts for m in per[t]))
            censored = sorted({m for t in ts for m in per[t]
                               if per_task[t].metas[m].get("censored")})
            n, p = len(ts) * len(arms), len(ts) + len(arms) - 1 + len(effects)
            zero = sorted({m for t in ts for m in per_task[t].check(per[t], c["id"])})
            why = (first.undefined(zero) if zero else
                   f"{n} task x partition units for {p} coefficients" if n - p < 1 else None)
            X = np.zeros((n, p))
            for i, (t, a) in enumerate((t, a) for t in ts for a in arms):
                X[i, ts.index(t)] = 1.0
                if arms.index(a):
                    X[i, len(ts) + arms.index(a) - 1] = 1.0
                for j, d in enumerate(effects):
                    X[i, len(ts) + len(arms) - 1 + j] = float(d["task"] == t and a in d["merged"])
            if why is None and np.linalg.matrix_rank(X) < p:
                why = "the merge patterns are confounded with the partition effects"
            if why is None:
                L = np.array([np.mean([np.log(per_task[t].v(per_task[t].runs(a, tag)[r]))
                                       for r in runs], axis=0) for t in ts for a in arms])
                fit = P.fixed_effects(L, X, [t for t in ts for _ in arms],
                                      {f"{d['task']}|{'+'.join(d['pairs'])}": p - len(effects) + j
                                       for j, d in enumerate(effects)})
                why = fit.get("not_computed")
            for d in effects:
                row = {**base, "task": d["task"],
                       "fine": "partitions that split " + " and ".join(d["pairs"]),
                       "coarse": "partitions that merge " + " and ".join(d["pairs"]),
                       "probe_pairs": d["pairs"], "merged_partitions": d["merged"],
                       "split_partitions": d["split"], "models": models}
                if why is not None:
                    row["not_computed"] = why
                else:
                    row.update(resid_dof=fit["resid_dof"], dispersion=fit["dispersion"],
                               v_run_task=fit["v_run"][d["task"]],
                               task_dof=fit["stratum_dof"][d["task"]],
                               **fit["coefficients"][f"{d['task']}|{'+'.join(d['pairs'])}"])
                if censored:
                    row["censored_models"] = censored
                    row["note"] = ("an AUC of 1 is floored at one discordant pair; the "
                                   "effect is a bound, not a value")
                rows.append(row)
    return rows


KINDS = {"pairs": _pairs, "all_pairs": _pairs, "unpaired": _unpaired, "linear": _linear,
         "checkpoint": _checkpoint, "draws": _draws,
         "partition_split_vs_merged": _partition_split_vs_merged}


def _rejections(cell: _Cell) -> list[dict]:
    out = []
    for (arm, tag), runs in cell.arms.items():
        if tag is not None and tag == cell.both:
            continue
        per_run = []
        for run, model in sorted(runs.items()):
            m = cell.metas[model]
            per_run.append({"model": model, "run": run,
                            **P.rejection_interval(m["n_bkg"], m["k_pass"])})
        pooled = P.rejection_interval(sum(x["n_bkg"] for x in per_run),
                                      sum(x["n_bkg_pass"] for x in per_run))
        out.append({"family": cell.fam, "task": cell.task, "kind": cell.kind,
                    "eps_s": float(cell.metric.split("@")[1]), "level": cell.label(arm),
                    "arm": arm, "checkpoint": tag, "per_run": per_run, "pooled": pooled,
                    "pooled_caveat": "runs share the test jets, so pooling "
                                     "understates the error"})
    return out


def ratios(vec: dict, meta: dict, run_dirs_root: pathlib.Path | None = None,
           spec: dict | None = None) -> dict:
    """Every contrast of `spec` (default: contrasts.v2.json when the models carry a
    checkpoint, contrasts.v1.json when none does) in every family x task x probe
    x metric. With `run_dirs_root` (v2), run k of two arms is a pair only where
    their training streams agree up to the compared checkpoint; a run that does
    not is reported and left out (A7)."""
    groups = _group(meta)
    if spec is None:
        tagged = {"@" in m for g in groups.values() for m in g}
        if len(tagged) != 1:
            raise SystemExit("FATAL: some models carry a checkpoint and some do not")
        spec = load_spec(CONTRASTS["v2" if tagged.pop() else "v1"])
    if run_dirs_root is not None and not spec["stream_check"]:
        raise SystemExit(f"FATAL: the runs of {spec['path']} recorded no training stream; "
                         "--run-dirs-root does not apply")
    if spec["stream_check"] and run_dirs_root is None:
        print("WARNING: no --run-dirs-root; v2 pairs are formed without the stream check (A7)")
    cells = {g: _Cell(g, by_model, vec, meta, spec, run_dirs_root) for g, by_model in sorted(groups.items())}
    rows, rej, points = [], [], {}
    for g, cell in cells.items():
        points["|".join(g)] = {m: float(vec[k][0]) for m, k in cell.key.items()}
        for c in spec["contrasts"]:
            if c["kind"] != "partition_joint":
                rows += KINDS[c["kind"]](cell, c)
        if cell.metric.startswith("eps_b@"):
            rej += _rejections(cell)
    for c in spec["contrasts"]:
        if c["kind"] == "partition_joint":
            rows += _partition_joint(cells, c)
    both = between_checkpoints(spec)
    for r in rows:                  # A8: every result at the weight average over the primary
        if both is not None and r["checkpoint"] == both and "ln_combined_se" in r:
            r["robust_to_checkpoint"] = bool(abs(r["ln_ratio"]) <= r["ln_combined_se"])
    used = {m for r in rows for k in ("fine_models", "coarse_models", "models") for m in r.get(k, [])}
    unused = sorted({m for cell in cells.values() for m in cell.key} - used)
    if unused:
        print(f"WARNING: {len(unused)} models enter no contrast, e.g. {unused[:3]}")
    return {"ratios": rows, "rejections": rej, "point_values": points,
            "models_in_no_contrast": unused,
            "contrasts": {"path": spec["path"], "sha256": spec["sha256"],
                          "version": spec["version"], "pending_arms": spec["pending_arms"]},
            "audit_b3": bool(spec.get("audit_b3"))}


# ----------------------------------------------------- audit B3, the check
# The audit's table (2026-09-29 report, B3): paired ratio, run range, run SD of
# ln r, and h, the median over runs of the single-run test-sample 95 % half-width
# of the log difference (probe.py's within-job contrasts). Our test term is the
# error of the MEAN over runs of ln r on jets they share, so it is compared with
# h / 1.96 as the scale, not expected to equal it.
B3 = [  # (family, task, kind, metric, fine, coarse, ratio, lo, hi, run_sd, h)
    ("probe", "bvc_resonant", "linear", "1-auc", "43", "17", 3.66, 2.92, 4.36, 0.16, 0.145),
    ("probe", "bvc_resonant", "linear", "1-auc", "162", "43", 1.068, 1.04, 1.08, 0.015, 0.088),
    ("probe", "bvc_resonant", "linear", "1-auc", "188", "162", 1.02, 0.96, 1.07, 0.046, 0.084),
    ("probe", "bvc_qcd", "linear", "1-auc", "188", "162", 1.146, 1.11, 1.23, 0.041, 0.094),
    ("probe", "retained_topology", "linear", "1-auc", "162", "43", 1.239, 1.18, 1.30, 0.038, 0.113),
    ("probe", "retained_topology", "linear", "1-auc", "162", "17", 1.537, 1.38, 1.87, 0.116, 0.123),
    ("probe", "bvc_resonant", "mlp", "1-auc", "43", "17", 2.85, 2.45, 3.34, 0.12, 0.145),
    ("probe", "retained_topology", "mlp", "1-auc", "162", "17", 1.50, 1.36, 1.85, 0.12, 0.153),
    ("probe", "bc_vs_rest", "linear", "1-auc", "162", "17", 1.64, 1.51, 1.72, 0.05, None),
    ("probe", "bc_vs_rest", "mlp", "1-auc", "162", "17", 1.50, 1.41, 1.59, 0.04, None),
    ("probe", "bvc_resonant", "linear", "1-auc", "162", "162+mass", 1.110, 1.07, 1.15, 0.027, 0.085),
    ("probe", "bvc_resonant", "linear", "1-auc", "17", "17+mass", 2.99, 1.85, 4.79, 0.37, None),
    ("probe", "bvc_resonant", "mlp", "1-auc", "17", "17+mass", 1.95, 1.39, 2.69, None, None),
    ("ft", "leg1", "N10000", "1-macro_auc", "188", "17", 1.316, 1.30, 1.34, 0.014, None),
    ("ft", "leg1", "N1000000", "1-macro_auc", "188", "17", 1.144, 1.13, 1.16, 0.008, None),
    ("ft", "leg1", "N1000", "1-macro_auc", "188", "43", 0.839, 0.75, 0.90, 0.069, None),
    ("ft", "leg2", "N1000000", "1-macro_auc", "188", "43", 1.020, 1.012, 1.030, 0.007, None),
]


def b3_compare(res: dict) -> list[dict]:
    got = {(r["family"], r["task"], r["kind"], r["metric"], r["fine"], r["coarse"]): r
           for r in res["ratios"]}
    out = []
    for fam, task, kind, metric, fine, coarse, ratio, lo, hi, sd, h in B3:
        r = got.get((fam, task, kind, metric, fine, coarse))
        row = {"cell": f"{fam}/{task}/{kind}/{metric} {coarse}/{fine}",
               "audit": {"ratio": ratio, "run_range": [lo, hi], "ln_run_sd": sd, "h95": h}}
        if r is None or "ratio" not in r:
            row["ours"] = None if r is None else {"not_computed": r.get("not_computed")}
        else:
            row["ours"] = {k: r[k] for k in ("ratio", "run_range", "ln_run_sd", "ln_test_se",
                                             "ln_combined_se", "ci95", "z")}
            row["ratio_agrees_to_1pc"] = abs(r["ratio"] / ratio - 1) < 0.01
            row["ours"]["significance_run_plus_test"] = abs(r["ln_ratio"]) / r["ln_combined_se"]
        out.append(row)
    return out


# ------------------------------------------------------------------ io
def save(path: pathlib.Path, vec: dict, meta: dict, prov: dict, b: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in vec.items()})
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(
        {"b": b, "seed": seed, "npz_sha256": _sha(path), "script_sha256": _sha(__file__),
         "provenance": prov, "meta": meta}, indent=1))
    print(f"wrote {path} ({len(vec)} vectors)")


def load(paths) -> tuple[dict, dict, list]:
    vec, meta, prov = {}, {}, []
    for p in paths:
        p = pathlib.Path(p)
        side = json.loads(p.with_suffix(".json").read_text())
        if side["npz_sha256"] != _sha(p):
            raise SystemExit(f"FATAL: {p} is not the file {p.with_suffix('.json')} describes")
        z = np.load(p)
        clash = set(z.files) & set(vec)
        if clash:
            raise SystemExit(f"FATAL: {sorted(clash)[:3]} appear in two replicate files")
        vec.update({k: z[k] for k in z.files})
        meta.update(side["meta"])
        prov.append({"path": str(p), "sha256": side["npz_sha256"], "b": side["b"],
                     "seed": side["seed"], "provenance": side["provenance"]})
    if len({(x["b"], x["seed"]) for x in prov if x["b"]}) > 1:    # B = 0: no resampling
        raise SystemExit("FATAL: replicate files were made with different B or seed")
    return vec, meta, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe-replicates", "mass-replicates", "ft-replicates", "anomaly-replicates"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, type=pathlib.Path)
        if name == "anomaly-replicates":
            s.add_argument("--anomaly", nargs="+", required=True, type=pathlib.Path,
                           help="experiments/EVAL/anomaly_heads.py outputs")
            s.add_argument("--extract-root", type=pathlib.Path, required=True,
                           help="the extract_v2.py outputs that name each model by its checkpoint")
            continue
        s.add_argument("--b", type=int, default=B)
        s.add_argument("--seed", type=int, default=SEED)
        if name == "probe-replicates":
            s.add_argument("--probe-dirs", nargs="+", required=True, type=pathlib.Path)
            s.add_argument("--committed", nargs="*", default=[],
                           help="DIR=FILE: the committed probe_results each refit repeats")
        elif name == "mass-replicates":
            s.add_argument("--mass-dirs", nargs="+", required=True, type=pathlib.Path)
        if name != "ft-replicates":
            s.add_argument("--extract-root", type=pathlib.Path, default=None,
                           help="v2: the extract_v2.py outputs (<root>/<run>/<checkpoint>/"
                                "manifest.json) that name each arm by its checkpoint")
        else:
            s.add_argument("--leg1-root", required=True, type=pathlib.Path)
            s.add_argument("--leg2-root", required=True, type=pathlib.Path)
            s.add_argument("--leg1-metrics", required=True, type=pathlib.Path)
            s.add_argument("--leg2-metrics", required=True, type=pathlib.Path)
            s.add_argument("--procs", type=int, default=8)
            s.add_argument("--cache", type=pathlib.Path, default=None,
                           help="per-cell results, kept as they finish and reused on a retry")
            s.add_argument("--checkpoint-rule", choices=("bestval", "wavg"), default=None,
                           help="v2: the rule of the tree under the roots, which every "
                                "cell's init_checkpoint.json must record")
    s = sub.add_parser("ratios")
    s.add_argument("--replicates", nargs="+", required=True, type=pathlib.Path)
    s.add_argument("--contrasts", type=pathlib.Path, default=None,
                   help="the contrasts file (default: configs/analysis/contrasts.v2.json "
                        "when the models carry a checkpoint, contrasts.v1.json otherwise)")
    s.add_argument("--run-dirs-root", type=pathlib.Path, default=None,
                   help="v2: the pretraining runs' root; run k of two arms is a pair only "
                        "where their streams agree up to the compared checkpoint")
    s.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    index = None
    if getattr(a, "extract_root", None) is not None:
        index = extraction_index(a.extract_root, load_spec(CONTRASTS["v2"]))
    if a.cmd == "probe-replicates":
        vec, meta, prov = probe_replicates(a.probe_dirs, a.b, a.seed, index)
        prov["reproduction"] = reproduction(a.probe_dirs, dict(x.split("=", 1) for x in a.committed))
        save(a.out, vec, meta, prov, a.b, a.seed)
    elif a.cmd == "anomaly-replicates":
        vec, meta, prov = anomaly_replicates(a.anomaly, index)
        save(a.out, vec, meta, prov, 0, None)
    elif a.cmd == "mass-replicates":
        vec, meta, prov = mass_replicates(a.mass_dirs, a.b, a.seed, index)
        save(a.out, vec, meta, prov, a.b, a.seed)
    elif a.cmd == "ft-replicates":
        cj = {"leg1": json.loads(a.leg1_metrics.read_text()),
              "leg2": json.loads(a.leg2_metrics.read_text())}
        for k, p in (("leg1", a.leg1_metrics), ("leg2", a.leg2_metrics)):
            cj[k]["sha256"] = _sha(p)
        vec, meta, prov = ft_replicates(a.leg1_root, a.leg2_root, cj, a.b, a.seed, a.procs,
                                        cache=a.cache, rule=a.checkpoint_rule)
        save(a.out, vec, meta, prov, a.b, a.seed)
    else:
        if a.out.exists():
            raise SystemExit(f"FATAL: {a.out} exists; refusing to overwrite")
        vec, meta, prov = load(a.replicates)
        res = ratios(vec, meta, a.run_dirs_root,
                     None if a.contrasts is None else load_spec(a.contrasts))
        b3 = res.pop("audit_b3")
        v2 = res["contrasts"]["version"] == "v2"
        res = {"provenance": {"replicates": prov, "script_sha256": _sha(__file__),
                              "method_sha256": _sha(REPO / "src" / "stats" / "paired.py")},
               "method": {
                   "ratio": "exp(mean over paired runs of ln(coarse/fine)), coarse over fine; "
                            "above 1 = the coarser or control model is worse",
                   "metrics": "1-auc (probes), eps_b@X (background efficiency at signal "
                              "efficiency X; inverse ratio of rejections), sigma_eff (mass "
                              "probes), 1-macro_auc (fine-tuning)",
                   "run_error": "SD over runs of ln r_k / sqrt(n) (ln_run_se)",
                   "test_error": "SD over bootstrap resamplings of the test jets, one "
                                 "resampling shared by both models and every run, of the "
                                 "mean ln r_k (ln_test_se); across runs, the replicates' mean "
                                 "off-diagonal covariance is the test noise the runs share "
                                 "(v_shared, clipped at 0; v_shared_unclipped as measured) and "
                                 "the mean diagonal less it the noise of each run alone (v_ind)",
                   "combined": "ln_combined_se^2 = sum(w^2) v_run + ln_test_se^2 for the "
                               "contrast w.ln over units (runs; or the runs of the two models, "
                               "or partitions, in two groups): ln_test_se^2 the contrast's own "
                               "bootstrap variance, v_run = max(s^2 - v_ind, 0), s^2 the units' "
                               "spread about their group's mean, pooled, v_ind the part of it the "
                               "test noise explains, from the covariances WITHIN each group. For "
                               "a paired ratio over n runs: max(s^2, v_ind)/n + v_shared. Never "
                               "below ln_test_se; one run keeps its full test variance. Runs that "
                               "leave a family out against their parent's (unpaired, 5 against 3) "
                               "keep each group's own spread (Welch). The joint partition fit "
                               "splits its residual the same way, each probe task's residual "
                               "spread with its own degrees of freedom, and forms a coefficient's "
                               "run term from the tasks' spreads with nonnegative weights (one "
                               "task's run noise that reaches another's residuals through the "
                               "partition effects is not subtracted, which can only overstate the "
                               "error). 95% interval "
                               "with Student t at the Welch-Satterthwaite degrees of freedom, the "
                               "observed spread carrying its own (until 2026-10-01 the run and "
                               "test terms were added in quadrature, which counts each run's own "
                               "test noise twice; on the same day a two-group form that assumed "
                               "one test covariance shared by all units, too small, was replaced)",
                   "draws": "one run per draw: var = 2 max(s17^2 - v_ind17, 0) + test SE^2, "
                            "the run variance borrowed from the 17-class runs, Student t at "
                            "the Welch-Satterthwaite degrees of freedom (s17 carrying n17 - 1)",
                   "pairing": ("run k with run k only where the two runs' training streams "
                               "agree at every epoch up to the compared checkpoint (A7); a run "
                               "that does not is listed in excluded_pairs / excluded_runs; the "
                               "runs that leave a family out are compared with their parent "
                               "unpaired" + ("" if a.run_dirs_root is not None else
                                             "; NOT CHECKED in this file (no --run-dirs-root)")
                               if v2 else
                               "run k with run k (v1: shared initialisation, not a "
                               "shared stream; audit B1); random draw k with run k"),
                   "rejection_interval": "Garwood central 68.27% on the count of "
                                         "passing background jets"},
               **res}
        if v2:
            res["method"]["robust_to_checkpoint"] = (
                "A8: every contrast is also formed at checkpoint 'wavg/bestval', the result "
                "at the weight average over the result at the best-validation checkpoint, run "
                "by run on the same test resamplings (contrast 'checkpoint': one model); "
                "robust_to_checkpoint = |ln of that ratio| <= its combined error")
            res["method"]["anomaly"] = (
                "sigma_min, no resampling: a run's value is the median over its detector "
                "trainings on draws seeded by the model's name, so that noise is in the spread "
                "over runs; noise the models share through the common background pool is not "
                "measured")
        if b3:
            res["audit_b3_check"] = b3_compare(res)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}: {len(res['ratios'])} ratios, {len(res['rejections'])} rejection cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
