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
               understates the error). v2: every comparison also between the
               checkpoints, the weight average and the global best over the
               primary, each labelled by the A14 rule.

Metrics are all "lower is better" -- 1 - AUC, eps_B at a fixed signal
efficiency, sigma_eff, 1 - macro AUC, sigma_min -- and every ratio is coarse over fine, so
a ratio above one means the coarser (or control) model is worse. A ratio of
eps_B is the inverse ratio of rejections.

MODELS. A replicate key is family|task|probe|model|metric. A v1 model is a run
name of configs/analysis/contrasts.v1.json; a v2 model is a run of
configs/arms/v2_grid.json at a checkpoint (A14), '<run>@best70' (the primary),
'<run>@wavg' (robustness) or '<run>@bestval' (sensitivity), so the checkpoints
of one run never collide; the BatchNorm twins of the frozen readouts (A14,
fired 2026-10-03) are '<run>@best70_bn' and '<run>@bestval_bn'. A v2 probe or
mass-probe arm is named through the checkpoint its features came from (the
extract_v2.py manifests, --extract-root), a v2 fine-tuning cell through the rule
its init_checkpoint.json records. One checkpoint has two frozen readouts (A14):
the class token, the plain name, and the pooled embedding, '<run>@<checkpoint>:
pooled' (the readout each probe_results.json and mass_resolution.json records),
so the two never share a key although they share a checkpoint digest. The
untrained trunk of run index k (A14's frozen reference, extracted as init-s<k>)
is the reference model 'init-s<k>@init' of the contrasts file's `references`; it
enters no contrast and is kept apart. A name that is neither a model, a
reference nor a listed baseline is fatal. The contrasts -- which arms are
compared, how they are paired -- are data in configs/analysis/contrasts.v{1,2}
.json; this file only forms them. An arm a contrast names must be a model arm,
or listed under pending_arms there.

A replicate vector is comparable with another only if both were computed on the
same jets in the same order with the same B and seed; the key `jets` (sha256 of
the row indices and labels) is checked before any pair is formed.

A14 (v2 only; the v1 results do not change). Two runs: the error is never below
the observed spread, at one degree of freedom; one run pair: not computed.
Checkpoints: every result between the checkpoints is labelled 'depends on the
checkpoint', 'robust' or 'inconclusive' ('depends on the checkpoint, under 10%'
where both of the first two rules hold), and the dependent results are counted
against their 5 % null expectation, the two-run results also apart
(checkpoint_dependence). P1 per pair: Welch,
each side's run variance from the runs that replicate one partition, labelled;
the joint fit weights each task by its run-plus-test variance, leaves out a task
with a censored unit, and has a reading with one axis-dose term per axis.
Random against semantic: each partition and each merge-status group against
the 17- and 43-class runs 1-2, with the axis and pair accounts. P2 restated: one
verdict block (p2_verdict). A11: the fraction of the excess cost the matched
lambda removes, with a Fieller interval, and the realised loss and gradient
shares (a11_shares). A13 and the self-supervised model: Welch on runs 1-3.

Usage:
  paired_errors.py probe-replicates --probe-dirs DIR... [--extract-root D] --out R.npz
  paired_errors.py mass-replicates  --mass-dirs DIR...  [--extract-root D] --out R.npz
  paired_errors.py ft-replicates    --leg1-root D --leg2-root D [--checkpoint-rule R] --out R.npz
  paired_errors.py anomaly-replicates --anomaly F... --extract-root D --out R.npz
  paired_errors.py ratios --replicates R.npz... [--contrasts F] [--run-dirs-root D] --out ratios.json
  paired_errors.py a11-shares --run-dirs-root D --out loss_share.json
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
    arm lower-cased without underscores, then -s<run>; a self-supervised arm's runs
    are <slug>-v2-s<run> (MPM: mpm-v2-s<run>; MPM_LOFO4P: mpmlofo4p-v2-s<run>).
    Every run's directory is mtx-<arm slug>-s<run>. Two runs with one name are
    fatal: until 2026-10-02 every self-supervised arm was mpm-v2-s<run>, and the
    leave-one-family-out arm's runs silently replaced the self-supervised arm's."""
    out = {}
    for a in json.loads(grid.read_text())["arms"]:
        slug = a["name"].lower().replace("_", "")
        for k in range(1, a["runs"] + 1):
            name = f"{slug}-v2-s{k}" if a["objective"] == "mpm" else f"{slug}-s{k}"
            if name in out:
                raise SystemExit(f"FATAL: {name} names a run of {out[name][0]} and of {a['name']}")
            out[name] = (a["name"], k, f"mtx-{slug}-s{k}")
    return out


def load_spec(path) -> dict:
    """A contrasts file, its model table resolved (v2: from the grid it names)."""
    path = pathlib.Path(path)
    spec = json.loads(path.read_text())
    m = spec["models"]
    spec["models"] = (grid_models(REPO / m["from_grid"]) if "from_grid" in m
                      else {k: (a, int(r), None) for k, (a, r) in m.items()})
    if "from_grid" in m:            # the arms' records (the mass-output lambda, A11)
        spec["grid_arms"] = {a["name"]: a for a in
                             json.loads((REPO / m["from_grid"]).read_text())["arms"]}
    # reference models outside the grid (A14: the untrained trunk of run indices 1-5),
    # {model: (arm, run, None)}; each has the one tag its entry names and no run directory
    refs = spec.get("references", {})
    spec["reference_models"] = {r["model"].format(run=k): (arm, k, None)
                                for arm, r in refs.items() for k in range(1, r["runs"] + 1)}
    if set(spec["reference_models"]) & set(spec["models"]):
        raise SystemExit(f"FATAL: {sorted(set(spec['reference_models']) & set(spec['models']))} "
                         f"are both models and references in {path}")
    if spec.get("readouts", [CLASS_TOKEN])[0] != CLASS_TOKEN:
        raise SystemExit(f"FATAL: {path}: the first readout, the one names carry without a "
                         f"suffix, must be {CLASS_TOKEN!r}")
    ck = spec.get("checkpoints", [])
    if any(t not in ck or p not in ck for t, p in spec.get("twins", {}).items()) or \
            any(r["tag"] in ck for r in refs.values()):
        raise SystemExit(f"FATAL: {path}: a twin or its parent is not a checkpoint, or a "
                         "reference's tag is a model checkpoint")
    full = path.resolve()
    spec["path"] = str(full.relative_to(REPO)) if full.is_relative_to(REPO) else str(full)
    spec["sha256"] = _sha(path)
    # every arm a contrast names is a model arm, or is listed as pending (decided,
    # not yet in the grid): a misspelt arm would otherwise form no row, silently
    arms = {a for a, *_ in spec["models"].values()}
    pending = spec.setdefault("pending_arms", {})
    named = set()
    for c in spec["contrasts"]:
        named |= set(c.get("arms", [])) | set(c.get("weights", {})) | set(c.get("references", []))
        named |= set(c.get("numerator", {})) | set(c.get("denominator", {}))
        named |= set(c.get("roles", {}).values())
        named |= {a for pr in c.get("pairs", []) for a in pr}
        named |= {c[k] for k in ("arm", "proxy", "account_reference") if k in c}
    if set(pending) & arms:
        raise SystemExit(f"FATAL: {sorted(set(pending) & arms)} are model arms now; remove them "
                         f"from pending_arms in {spec['path']}")
    if named - arms - set(pending):
        raise SystemExit(f"FATAL: {spec['path']} compares {sorted(named - arms - set(pending))}, "
                         "which are neither model arms nor pending")
    return spec


def parse_model(name: str, spec: dict) -> tuple[str, int, str | None]:
    """(arm, run, tag) of a replicate key's model: 'l162-s1b' (v1), 'l188-s1@bestval'
    (v2), 'l188-s1@best70:pooled' (v2, the pooled-embedding readout) or 'init-s1@init'
    (a reference). The tag is the checkpoint, with ':<readout>' for every readout but
    the contrasts file's first. An unknown model, checkpoint or readout, or a reference
    at a tag other than its own, is fatal."""
    model, _, tag = name.partition("@")
    ck, colon, readout = tag.partition(":")
    refs = spec.get("reference_models", {})
    if model in refs:
        arm, run, _ = refs[model]
        ok = [spec["references"][arm]["tag"]]
    elif model in spec["models"]:
        arm, run, _ = spec["models"][model]
        ok = spec.get("checkpoints", [""])
    else:
        raise SystemExit(f"FATAL: model {model!r} is not a model of {spec['path']}")
    if ck not in ok:
        raise SystemExit(f"FATAL: model {name!r}: its checkpoint must be one of {ok} "
                         f"under {spec['path']}")
    others = spec.get("readouts", [])[1:]
    if colon and readout not in others:
        raise SystemExit(f"FATAL: model {name!r}: a readout suffix must be one of {others} "
                         f"under {spec['path']} (the first readout takes none)")
    return arm, run, tag or None


def base_checkpoint(tag: str, spec: dict) -> str:
    """The checkpoint whose training a tag's model carries, which the stream check (A7)
    reads: the readout dropped (one checkpoint, two readouts), and a BatchNorm twin read
    as its parent -- the same weights, whose statistics are recomputed on the epoch-80
    draw, seeded alike for every vocabulary of a run index (A7, A14)."""
    ck = tag.partition(":")[0]
    return spec.get("twins", {}).get(ck, ck)


def readouts_of(models) -> list[str]:
    """The readout suffixes (':pooled'; '' for the first readout) the models carry."""
    return sorted({":" + m.partition("@")[2].partition(":")[2]
                   if ":" in m.partition("@")[2] else "" for m in models})


def extraction_index(root: pathlib.Path, spec: dict) -> dict[str, list[str]]:
    """{checkpoint sha256: ['model@checkpoint', ...]} from every extract_v2.py
    manifest under root (<root>/<run>/<checkpoint>/manifest.json): the model from
    the manifest's run_dir, the checkpoints from its tags. Where a run's best70
    and bestval select one epoch, extract_v2.py extracts it once, links the
    second tag's directory to the first and lists both tags, so that checkpoint
    is a model under both names: '<run>@bestval' exists for every run, and its
    ratio to '<run>@best70' is then exactly 1. A v2 probe arm is named through
    the checkpoint its features came from, never through its label.

    A reference's checkpoint (the untrained trunk, tag init) is extracted from a
    run's init_trunk.pt into its own directory (<root>/init-s<k>/init) and is named
    by that directory, init-s<k>@init, which must be a reference model of the spec
    with the run's index: it is not that run's vocabulary at another checkpoint."""
    by_dir = {d: m for m, (_, _, d) in spec["models"].items()}
    ref_tags = {r["tag"] for r in spec.get("references", {}).values()}
    out, owner = {}, {}
    for f in sorted(pathlib.Path(root).glob("*/*/manifest.json")):
        man = json.loads(f.read_text())
        run = pathlib.Path(man["run_dir"]).name
        if run not in by_dir:
            raise SystemExit(f"FATAL: {f}: {run} is not a run of {spec['path']}")
        model, tags = by_dir[run], man.get("tags", [man["tag"]])
        if set(tags) & ref_tags:
            ref = f.parent.parent.name
            if (ref not in spec["reference_models"] or tags != [man["tag"]]
                    or spec["reference_models"][ref][1] != spec["models"][model][1]):
                raise SystemExit(f"FATAL: {f}: the {man['tag']} checkpoint of {run} sits in {ref}, "
                                 f"which is not that run index's reference of {spec['path']}")
            model = ref
        sha = man["checkpoint_sha256"]
        names = out.setdefault(sha, [])
        for tag in tags:
            name = f"{model}@{tag}"
            if names and names[0].partition("@")[0] != model:
                raise SystemExit(f"FATAL: checkpoint {sha[:16]} is both {names[0]} and {name}")
            if owner.setdefault(name, sha) != sha:
                raise SystemExit(f"FATAL: {name} is both checkpoint {owner[name][:16]} and {sha[:16]}")
            if name not in names:
                names.append(name)
    if not out:
        raise SystemExit(f"FATAL: no extraction manifest under {root}")
    return out


CLASS_TOKEN = "features"     # the readout every v1 result and a file that records none reads


def _with_readout(names: list[str], readout: str | None, where) -> list[str]:
    """The model names of an arm read through `readout` (A14): unchanged for the class
    token; '<name>:<readout>' otherwise, so the two readouts of one checkpoint, which
    share its digest, are two models. v1 models have the class token only."""
    if readout in (None, CLASS_TOKEN):
        return names
    if any("@" not in n for n in names):
        raise SystemExit(f"FATAL: {where}: readout {readout!r} of a v1 model; v1 has the class "
                         "token only")
    return [f"{n}:{readout}" for n in names]


def _names(index: dict | None, arm: str, sha: str | None, where) -> list[str]:
    """The model names of a probe arm: its label for v1; for v2, every name of
    the checkpoint it was fitted on (extraction_index)."""
    if index is None:
        return [arm]
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
    on, so a run's bestval and wavg features are two models; a checkpoint with
    two names (best70 and bestval one epoch) is one replicate vector under both.
    The file's `readout` (A14) suffixes the names (_with_readout): the class token
    and the pooled embedding of one checkpoint are two models.
    A model scored in two jobs (the 162-class run 1 is in the ladder and in the
    2x2), or a checkpoint probed under both its names, is kept once, from the
    first directory given, and the other copy's AUC is recorded as a
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
                models = _with_readout(_names(index, arm, J.get("arm_checkpoints", {}).get(arm), d),
                                       J.get("readout"), d)
                for kind in ("linear", "mlp"):
                    s = z[f"{task}|{kind}|{arm}"]
                    bases = [f"probe|{task}|{kind}|{m}" for m in models]
                    if f"{bases[0]}|1-auc" in vec:
                        dup.append({"model": bases[0], "dir": str(d),
                                    "auc_here": A[kind]["auc"],
                                    "auc_kept": 1 - vec[f"{bases[0]}|1-auc"][0]})
                        continue
                    sc = P.AucScorer(y, s)
                    if abs(sc.auc() - A[kind]["auc"]) > 1e-12:
                        raise SystemExit(f"FATAL: {bases[0]}: AUC from scores {sc.auc()} "
                                         f"!= reported {A[kind]['auc']}")
                    v = P.replicates(sc, y.size, b, seed)
                    m = {"jets": jk, "n": int(y.size), "censored": bool(A[kind]["log1m_auc_censored"])}
                    eps = {}
                    for e in T["eps_s"]:
                        es = P.EpsBScorer(y, s, float(e))
                        ve = P.replicates(es, y.size, b, seed)
                        nb = int((~y).sum())
                        eps[f"eps_b@{float(e):.2f}"] = (ve, {"jets": jk, "n": int(y.size), "n_bkg": nb,
                                                             "k_pass": float(ve[0] * nb)})
                    for base in bases:
                        vec[f"{base}|1-auc"], meta[f"{base}|1-auc"] = v, dict(m)
                        for metric, (ve, me) in eps.items():
                            vec[f"{base}|{metric}"], meta[f"{base}|{metric}"] = ve, dict(me)
                        print(f"  {base}", flush=True)
    return vec, meta, {"inputs": inputs, "duplicates": dup}


# ------------------------------------------------------------ mass probes
def mass_replicates(dirs: list[pathlib.Path], b: int = B, seed: int = SEED,
                    index: dict | None = None):
    """Replicates of sigma_eff of the within-class mass residual, per probe x model
    (`index` and the readout as for probe_replicates). A v2 checkpoint scored twice (best70 and
    bestval one epoch, each directory given as an arm) is kept once and the
    other copy recorded; a v1 model given twice is fatal."""
    vec, meta, inputs, dup = {}, {}, [], []
    for d in dirs:
        J = json.loads((d / "mass_resolution.json").read_text())
        z = np.load(d / "residuals.npz")
        inputs.append({"dir": str(d), "mass_resolution_sha256": _sha(d / "mass_resolution.json"),
                       "residuals_sha256": _sha(d / "residuals.npz")})
        rows, lab = z["rows"], z["label188"]
        jk = jets_key(rows, lab)
        for arm, A in J["arms"].items():
            models = _with_readout(_names(index, arm, A.get("provenance", {}).get("checkpoint_sha256"), d),
                                   J.get("readout"), d)
            for kind in ("ridge", "mlp"):
                res = z[f"{kind}|{arm}"].astype(np.float64)
                sc = P.SigmaEffScorer(res)
                if abs(sc() - A[kind]["sigma_eff"]) > 1e-6:
                    raise SystemExit(f"FATAL: {arm}/{kind}: sigma_eff from residuals "
                                     f"{sc()} != reported {A[kind]['sigma_eff']}")
                keys = [f"mass|resolution|{kind}|{m}|sigma_eff" for m in models]
                if keys[0] in vec:
                    if index is None:
                        raise SystemExit(f"FATAL: {keys[0]} appears twice among the mass-probe inputs")
                    dup.append({"model": keys[0], "dir": str(d), "arm": arm,
                                "sigma_eff_here": A[kind]["sigma_eff"],
                                "sigma_eff_kept": float(vec[keys[0]][0])})
                    continue
                v = P.replicates(sc, res.size, b, seed)
                for key in keys:
                    vec[key], meta[key] = v, {"jets": jk, "n": int(res.size)}
                    print(f"  {key}", flush=True)
    return vec, meta, {"inputs": inputs, **({"duplicates": dup} if dup else {})}


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
    sigma_t).

    anomaly_heads.py names a result by its checkpoint directory, so where best70
    and bestval are one epoch (bestval a link to best70, extract_v2.py) it
    reports both, with one checkpoint and the same values. Each is kept under
    its own name; a directory whose checkpoint has no such name is fatal, and a
    name of the checkpoint that no result gives takes the value of another."""
    nd = _load("anomaly_summary", "experiments/EVAL/anomaly_summary.py").NOT_DETECTED_MAX_SIC
    vec, meta, inputs, alias_of = {}, {}, [], {}
    for p in paths:
        J = json.loads(pathlib.Path(p).read_text())
        inputs.append({"path": str(p), "sha256": _sha(p)})
        jk = hashlib.sha256(json.dumps([J["labels_sha256"], J["n_bkg"], J["n_template"],
                                        J["trainings"]]).encode()).hexdigest()
        for arm, M in J["models"].items():
            for tag, c in M["checkpoints"].items():
                if "anomaly" not in c:
                    continue
                models = _names(index, arm, c.get("checkpoint_sha256"), p)
                model = f"{models[0].partition('@')[0]}@{tag}"
                if model not in models:
                    raise SystemExit(f"FATAL: {p}: {arm}/{tag} is the checkpoint of "
                                     f"{' and '.join(models)}")
                for sig, per_n in c["anomaly"].items():
                    for n_sig, fams in per_n.items():
                        for fam, rec in fams.items():
                            if not isinstance(rec, dict) or "sigma_min" not in rec:
                                continue
                            key = f"anomaly|{sig}|{fam}|{model}|sigma_min@{n_sig}"
                            if key in vec:
                                raise SystemExit(f"FATAL: {key} appears twice among the inputs")
                            alias_of[key] = [f"anomaly|{sig}|{fam}|{m}|sigma_min@{n_sig}"
                                             for m in models if m != model]
                            vec[key] = np.array([float(rec["sigma_min"])])
                            meta[key] = {"jets": jk, "n": int(J["n_bkg"]),
                                         "max_sic": rec.get("max_sic")}
                            why = ("max SIC at its background-statistics ceiling"
                                   if rec.get("at_ceiling") else
                                   f"max SIC below {nd}, not detected" if rec["max_sic"] < nd else None)
                            if why:
                                meta[key].update(censored=True, censor_note=(
                                    why + "; sigma_min is a bound and so is the ratio"))
    for key, others in alias_of.items():
        for k in others:
            if k not in vec:
                vec[k], meta[k] = vec[key], dict(meta[key])
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


def between_checkpoints(spec: dict, readouts=("",)) -> list[str]:
    """The tags of the comparisons between checkpoints, '<other>/<primary>': A14's
    robustness ('wavg/best70') and sensitivity ('bestval/best70') checks, when the
    contrasts file has a checkpoint contrast; else none. Each within every readout
    suffix of `readouts` ('' the class token; ':pooled' gives 'wavg:pooled/
    best70:pooled'): a checkpoint is compared with the primary through one readout."""
    c = next((c for c in spec["contrasts"] if c["kind"] == "checkpoint"), None)
    return [] if c is None else [f"{c[k]}{s}/{c['primary']}{s}" for s in readouts
                                 for k in ("robustness", "sensitivity") if k in c]


class _Cell:
    """One family x task x probe x metric: its models by (arm, checkpoint) and run.

    A8 and A14 ask for every result at the robustness checkpoint (the weight
    average) and the sensitivity one (the global best) beside the primary one.
    So, when the contrasts file has a checkpoint contrast, each run with both
    checkpoints of a comparison also enters as '<run>@wavg/best70' (and
    '<run>@bestval/best70'), whose replicate vector is the ratio of the two (one
    run, one resampling of the test jets), and every contrast is formed at that
    tag too: the double ratio, the result at the other checkpoint over the result
    at the primary."""

    def __init__(self, group: tuple, by_model: dict, vec: dict, meta: dict, spec: dict, root):
        self.fam, self.task, self.kind, self.metric = group
        self.key, self.spec, self.root = by_model, spec, root
        self.rule = bool(spec.get("small_sample_rule"))
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
        self.between = between_checkpoints(spec, readouts_of(by_model))
        for both in self.between:
            q, p = both.split("/")
            for (arm, tag), runs in list(self.arms.items()):
                other = self.arms.get((arm, q), {}) if tag == p else {}
                for run in sorted(set(runs) & set(other)):
                    mp, mq = runs[run], other[run]
                    if self.metas[mp]["jets"] != self.metas[mq]["jets"]:
                        raise SystemExit(f"FATAL: {'/'.join(group)}: {mp} and {mq} were not "
                                         "scored on the same jets")
                    a, b = self.vecs[mq], self.vecs[mp]
                    m = f"{mp.partition('@')[0]}@{both}"
                    self.vecs[m] = np.divide(a, b, out=np.zeros_like(a, dtype=np.float64),
                                             where=(a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b))
                    cens = next((self.metas[x] for x in (mp, mq) if self.metas[x].get("censored")), None)
                    self.metas[m] = {"jets": self.metas[mp]["jets"], "censored": cens is not None}
                    if cens is not None and "censor_note" in cens:
                        self.metas[m]["censor_note"] = cens["censor_note"]
                    self.arms.setdefault((arm, both), {})[run] = m
        self.tags = sorted({t for _, t in self.arms}, key=lambda t: (t is not None, t or ""))

    def ck(self, tag):
        """The checkpoint(s) a stream check at `tag` covers (base_checkpoint)."""
        if tag is None:
            return None
        if tag in self.between:
            return tuple(base_checkpoint(t, self.spec) for t in tag.split("/"))
        return base_checkpoint(tag, self.spec)

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
    """Run k of the fine arm against run k of the coarse arm, at each checkpoint.
    A contrast marked `descriptive` carries that reading on every row (A13: the
    ladder among the models that leave a family out, no null read)."""
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
            if c.get("descriptive"):
                row["reading"] = "descriptive: " + c["descriptive"]
            zero = cell.check(fm + cm, f"{fa} vs {ca}")
            if zero:
                row["not_computed"] = cell.undefined(zero)
            else:
                row.update(P.paired_ratio({m: cell.v(m) for m in fm}, {m: cell.v(m) for m in cm},
                                          pairs=dict(zip(cm, fm)), run_dirs=cell.dirs(fm + cm),
                                          checkpoint=cell.ck(tag), small_sample_rule=cell.rule))
            cell.censor(row, fm + cm)
            rows.append(row)
    return rows


def _unpaired(cell: _Cell, c: dict) -> list[dict]:
    """Every run of the coarse arm against every run of the fine arm, unpaired:
    exp(mean ln m_coarse - mean ln m_fine), each arm with its own spread over
    runs (Welch): the runs that leave a family out need not vary as much as their
    parent's, and five runs against three do not let one spread stand for both.
    `runs` keeps those run indices only (A14: runs 1-3, one GPU product on both
    sides); `families` keeps those families only (the self-supervised model is
    compared in fine-tuning)."""
    if c.get("families") and cell.fam not in c["families"]:
        return []
    keep = c.get("runs")
    rows = []
    for tag in cell.tags:
        for fa, ca in c["pairs"]:
            fm, cm = ([m for r, m in sorted(cell.runs(a, tag).items()) if keep is None or r in keep]
                      for a in (fa, ca))
            if not fm or not cm:
                continue
            row = cell.row(c["id"], cell.label(fa), cell.label(ca), tag, fine_arm=fa,
                           coarse_arm=ca, fine_models=fm, coarse_models=cm,
                           stream_pairing="exempt: " + c["stream_exempt"])
            if keep is not None:
                row["runs"] = keep
            zero = cell.check(fm + cm, f"{fa} vs {ca}")
            if zero:
                row["not_computed"] = cell.undefined(zero)
            else:
                L = np.log([cell.v(m) for m in cm + fm])
                w = [1 / len(cm)] * len(cm) + [-1 / len(fm)] * len(fm)
                groups = [range(len(cm)), range(len(cm), len(cm) + len(fm))]
                row.update(n_fine_runs=len(fm), n_coarse_runs=len(cm),
                           **P.contrast(L, w, groups, separate=True, small_sample_rule=cell.rule))
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
                                    {r: [d[m] for m in member[r].values()] for r in runs},
                                    small_sample_rule=cell.rule))
        cell.censor(row, fm + cm)
        rows.append(row)
    return rows


def _checkpoint(cell: _Cell, c: dict) -> list[dict]:
    """A8 and A14: every model at the robustness checkpoint (and at the
    sensitivity one) against itself at the primary one, paired by run (one run,
    one stream). The A14 label is added with every other result's (ratios)."""
    rows = []
    for both in cell.between:
        q, p = both.split("/")
        for arm in dict.fromkeys(a for a, _ in cell.arms):
            fr, cr = cell.runs(arm, p), cell.runs(arm, q)
            common = [r for r in cr if r in fr]
            if not common:
                continue
            fm, cm = [fr[r] for r in common], [cr[r] for r in common]
            row = cell.row(c["id"], f"{cell.label(arm)} at {p}", f"{cell.label(arm)} at {q}",
                           both, fine_arm=arm, coarse_arm=arm, fine_models=fm, coarse_models=cm)
            zero = cell.check(fm + cm, f"{arm} {q} vs {p}")
            if zero:
                row["not_computed"] = cell.undefined(zero)
            else:
                row.update(P.paired_ratio({m: cell.v(m) for m in fm}, {m: cell.v(m) for m in cm},
                                          pairs=dict(zip(cm, fm)), small_sample_rule=cell.rule))
                row["stream_pairing"] = "same run"
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


def axis_doses(spec: dict) -> dict:
    """configs/labelmaps/axis_doses.v2.json (A14): each probe pair's axis, each
    vocabulary's dose of every axis, and the cells of the axis account."""
    return json.loads((REPO / spec["axis_doses"]).read_text())


def pair_doses(doses: dict, pairs, arms) -> dict:
    """{pair: {axis, realised: {partition: dose}}}: each partition's realised dose
    of the pair's axis with the pair's own classes left out; an empty dose for a
    pair on no axis."""
    out = {}
    for name in pairs:
        axis = doses["probe_pairs"][name]["axis"]
        out[name] = {"axis": axis, "realised": {} if axis is None else
                     {a: doses["doses"][a][axis]["excluding"][name]["realised"] for a in arms}}
    return out


def _partition_split_vs_merged(cell: _Cell, c: dict) -> list[dict]:
    """A10 P1 as A14 reads it, per probe pair: exp(mean over merging partitions -
    mean over splitting ones) of the run-averaged ln metric. The units are the
    runs; each side keeps its own run variance, measured by the runs that
    replicate one partition (their spread about the partition's mean, pooled over
    the side's partitions), and the interval is Welch's. The partitions are taken
    as drawn: how good a partition is overall is not error here, and the joint
    fit's partition effect is what separates it from the merge. Labelled
    'merging costs nothing', 'merging costs', 'merging costs, under 10%' or
    'inconclusive' (P.p1_label), with each partition's axis dose beside it."""
    if cell.fam != "probe":
        return []
    doses = axis_doses(cell.spec) if "axis_doses" in cell.spec else None
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
            if doses is not None:
                row["axis_doses"] = pair_doses(doses, d["pairs"], c["arms"])
            if excluded is not None:
                row.update(stream_pairing="identical", excluded_runs=excluded)
            zero = cell.check(fm + cm, c["id"])
            if zero:
                row["not_computed"] = cell.undefined(zero)
            elif not d["merged"] or not d["split"]:
                row["not_computed"] = "every partition merges, or every one splits, this pair"
            else:
                R, nm, ns = len(runs), len(d["merged"]), len(d["split"])
                L = np.log([cell.v(cell.runs(a, tag)[r]) for a in units for r in runs])
                row.update(P.contrast(L, [1 / (nm * R)] * (nm * R) + [-1 / (ns * R)] * (ns * R),
                                      [range(i * R, (i + 1) * R) for i in range(nm + ns)],
                                      pools=[list(range(nm)), list(range(nm, nm + ns))],
                                      small_sample_rule=cell.rule))
                if "ln_combined_se" in row and tag not in cell.between:
                    row["p1_label"] = P.p1_label(*P.bounds(row["ln_ratio"], row["ln_combined_se"],
                                                           row["dof"]))
            cell.censor(row, fm + cm)
            rows.append(row)
    return rows


def _within_partition_run_var(cell: _Cell, arms, runs, tag) -> float | None:
    """A14: one task's run variance from the runs that replicate one partition:
    their spread about the partition's mean, pooled over the partitions, less
    the test noise each run has alone (combined_error's v_run with one group per
    partition). None without two runs of every partition."""
    if len(runs) < 2:
        return None
    R = len(runs)
    L = np.log([cell.v(cell.runs(a, tag)[r]) for a in arms for r in runs])
    return P.combined_error(L, np.full(len(L), 1 / len(L)),
                            [range(i * R, (i + 1) * R) for i in range(len(arms))])["v_run"]


def _partition_joint(cells: dict, c: dict) -> list[dict]:
    """A10 P1, all probe pairs at once, the secondary reading of A14: the
    run-averaged ln metric of every task x partition on a task intercept, a
    partition effect shared by every task (how good a partition is overall), and
    one merge effect per task and merge pattern (P.fixed_effects). A merge
    effect is the cost of merging the pair with each partition's overall quality
    held fixed. Pairs that share a merge pattern in different tasks stay separate
    effects; in one task they are one.

    A14: each task is weighted by 1 / (its run variance / runs + its test
    variance), the run variance from the runs that replicate one partition; a
    task with a censored unit is left out of the fit, and its row says so (the
    pair is reported alone, as a bound, by partition_split_vs_merged). With
    `dose_terms`, the fit also has one term per axis in (1 - the partition's
    realised dose of the probe pair's axis, the pair's own classes left out;
    axis_doses.v2.json), shared by the tasks on that axis: exp of it is the
    metric with the axis merged throughout over the metric with it kept
    throughout."""
    arms = c["arms"]
    spec = next(iter(cells.values())).spec
    design = partition_design(spec, arms)
    tasks = list(dict.fromkeys(d["task"] for d in design))
    doses = axis_doses(spec) if c.get("dose_terms") else None
    task_axis = {}                                    # {task: (axis, pair)}
    if doses is not None:
        for t in tasks:
            on = [(doses["probe_pairs"][n]["axis"], n) for d in design if d["task"] == t
                  for n in d["pairs"] if doses["probe_pairs"][n]["axis"] is not None]
            if len(on) > 1:
                raise SystemExit(f"FATAL: task {t} holds {len(on)} pairs on an axis; its dose "
                                 "term is not defined")
            if on:
                task_axis[t] = on[0]
    by_km = {}
    for (fam, task, kind, metric), cell in cells.items():
        if fam == "probe" and task in tasks:
            by_km.setdefault((kind, metric), {})[task] = cell
    rows = []
    for (kind, metric), per_task in sorted(by_km.items()):
        present = [t for t in tasks if t in per_task]
        for tag in sorted({t for cell in per_task.values() for t in cell.tags},
                          key=lambda t: (t is not None, t or "")):
            first = per_task[present[0]]
            runs, excluded = _partition_runs(first, arms, tag)
            runs = [r for r in runs if all(all(r in per_task[t].runs(a, tag) for a in arms)
                                           for t in present)]
            if not runs:
                continue
            per = {t: [per_task[t].runs(a, tag)[r] for a in arms for r in runs] for t in present}
            left_out = [t for t in present if any(per_task[t].metas[m].get("censored") for m in per[t])]
            ts = [t for t in present if t not in left_out]
            effects = [d for d in design if d["task"] in ts]
            axes = list(dict.fromkeys(task_axis[t][0] for t in ts if t in task_axis))
            base = {"contrast": c["id"], "family": "probe", "kind": kind, "metric": metric,
                    "checkpoint": tag, "tasks_in_fit": ts, "tasks_left_out": left_out, "runs": runs}
            if excluded is not None:
                base.update(stream_pairing="identical", excluded_runs=excluded)
            models = list(dict.fromkeys(m for t in ts for m in per[t]))
            n = len(ts) * len(arms)
            p = len(ts) + len(arms) - 1 + len(effects) + len(axes)
            zero = sorted({m for t in ts for m in per_task[t].check(per[t], c["id"])})
            v_run = ({} if zero or len(runs) < 2 else
                     {t: _within_partition_run_var(per_task[t], arms, runs, tag) for t in ts})
            why = (first.undefined(zero) if zero else
                   "no task is left once the tasks with a censored unit are left out" if not ts else
                   "one run per partition: no replicate to measure the run variance that weights "
                   "each task (A14)" if len(runs) < 2 else
                   f"{n} task x partition units for {p} coefficients" if n - p < 1 else None)
            X = np.zeros((n, p))
            for i, (t, a) in enumerate((t, a) for t in ts for a in arms):
                X[i, ts.index(t)] = 1.0
                if arms.index(a):
                    X[i, len(ts) + arms.index(a) - 1] = 1.0
                for j, d in enumerate(effects):
                    X[i, len(ts) + len(arms) - 1 + j] = float(d["task"] == t and a in d["merged"])
                if t in task_axis:
                    ax, pair = task_axis[t]
                    X[i, p - len(axes) + axes.index(ax)] = \
                        1.0 - doses["doses"][a][ax]["excluding"][pair]["realised"]
            if why is None and np.linalg.matrix_rank(X) < p:
                why = "the merge patterns are confounded with the partition effects"
            if why is None:
                L = np.array([np.mean([np.log(per_task[t].v(per_task[t].runs(a, tag)[r]))
                                       for r in runs], axis=0) for t in ts for a in arms])
                j0 = len(ts) + len(arms) - 1
                report = {f"{d['task']}|{'+'.join(d['pairs'])}": j0 + j for j, d in enumerate(effects)}
                report.update({f"dose|{ax}": p - len(axes) + k for k, ax in enumerate(axes)})
                fit = P.fixed_effects(L, X, [t for t in ts for _ in arms], report,
                                      run_var=[v_run[t] / len(runs) for t in ts for _ in arms])
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
                               v_run_within_partition=v_run[d["task"]],
                               task_dof=fit["stratum_dof"][d["task"]],
                               **fit["coefficients"][f"{d['task']}|{'+'.join(d['pairs'])}"])
                rows.append(row)
            for ax in axes:
                on = [t for t in ts if task_axis.get(t, (None,))[0] == ax]
                row = {**base, "task": None, "axis": ax, "tasks": on,
                       "fine": f"{ax} kept throughout (dose 1)",
                       "coarse": f"{ax} merged throughout (dose 0)", "models": models}
                if why is not None:
                    row["not_computed"] = why
                else:
                    row.update(resid_dof=fit["resid_dof"], dispersion=fit["dispersion"],
                               **fit["coefficients"][f"dose|{ax}"])
                rows.append(row)
            for d in design:
                if d["task"] in left_out:
                    cens = sorted(m for m in per[d["task"]] if per_task[d["task"]].metas[m].get("censored"))
                    rows.append({**base, "task": d["task"],
                                 "fine": "partitions that split " + " and ".join(d["pairs"]),
                                 "coarse": "partitions that merge " + " and ".join(d["pairs"]),
                                 "probe_pairs": d["pairs"], "models": per[d["task"]],
                                 "censored_models": cens,
                                 "not_computed": "left out of the joint fit (A14): a unit is "
                                                 "censored, so the pair is reported alone, as a "
                                                 "bound, by partition_split_vs_merged"})
    return rows


def _vs_reference(cell: _Cell, c: dict, tag, ref: str, members: list, runs: list, name: str,
                  **kw) -> dict:
    """ln m_ref,k - mean over `members` of ln m_k, paired over runs k."""
    per = {r: [cell.runs(ref, tag)[r]] + [cell.runs(a, tag)[r] for a in members] for r in runs}
    fm = [m for r in runs for m in per[r][1:]]
    cm = [per[r][0] for r in runs]
    row = cell.row(c["id"], name, cell.label(ref), tag, reference=ref, partitions=members,
                   fine_models=fm, coarse_models=cm, **kw)
    zero = cell.check(fm + cm, c["id"])
    if zero:
        row["not_computed"] = cell.undefined(zero)
    else:
        ln = {r: np.log(cell.v(ms[0])) - np.mean([np.log(cell.v(m)) for m in ms[1:]], axis=0)
              for r, ms in per.items()}
        d = cell.dirs(fm + cm)
        row.update(P.paired_log(ln, checkpoint=cell.ck(tag), small_sample_rule=cell.rule,
                                run_dirs=None if d is None else
                                {r: [d[m] for m in ms] for r, ms in per.items()}))
    cell.censor(row, fm + cm)
    return row


def _random_vs_semantic(cell: _Cell, c: dict) -> list[dict]:
    """A14, random against semantic: on each probe pair's own task, each random
    partition, the partitions that merge the pair and those that split it,
    against each reference (the 17- and 43-class models), paired over `runs`
    (1-2). The ratio is the reference over the random side: above 1 = the random
    partitions beat it. Against the account reference (17 classes), the cells of
    axis_doses.v2.json -- partitions that merge a pair the 17-class model also
    merges while keeping its axis elsewhere -- are read for the two registered
    accounts, at each checkpoint: axis account 'beats' when the 95 % lower bound
    is above 0, pair account 'equal' when the 90 % interval lies within
    +-ln 1.1, each 'inconclusive' otherwise. Each cell alone, and the cells of a
    pair together."""
    if cell.fam != "probe":
        return []
    arms = c["arms"]
    design = [d for d in partition_design(cell.spec, arms) if d["task"] == cell.task]
    if not design:
        return []
    doses = axis_doses(cell.spec)
    cells_of = {}
    for x in doses["cells"]:
        cells_of.setdefault(x["pair"], []).append(x["partition"])
    rows = []
    for tag in cell.tags:
        for ref in c["references"]:
            runs = [r for r in c["runs"] if r in cell.runs(ref, tag)
                    and all(r in cell.runs(a, tag) for a in arms)]
            if not runs:
                continue
            account = ref == c["account_reference"]
            for d in design:
                what = " and ".join(d["pairs"])
                for pair in d["pairs"]:
                    if set(cells_of.get(pair, [])) - set(d["merged"]):
                        raise SystemExit(f"FATAL: axis_doses.v2.json lists cells of {pair} that "
                                         "the partition merge table says do not merge it")
                groups = [(cell.label(a), [a], {"merges": a in d["merged"],
                                                "account_cells": [p for p in d["pairs"]
                                                                  if account and a in cells_of.get(p, [])]})
                          for a in arms]
                groups += [(f"partitions that {s} {what}", g, {"merges": s == "merge"})
                           for s, g in (("merge", d["merged"]), ("split", d["split"])) if g]
                groups += [(f"the cells of {p}: partitions that merge it and keep its axis elsewhere",
                            cells_of[p], {"merges": True, "account_cells": [p]})
                           for p in d["pairs"] if account and cells_of.get(p)]
                for name, members, kw in groups:
                    row = _vs_reference(cell, c, tag, ref, members, runs, name, probe_pairs=d["pairs"],
                                        axis_doses=pair_doses(doses, d["pairs"], members), **kw)
                    if not row.get("account_cells"):
                        row.pop("account_cells", None)
                    elif "ln_combined_se" in row and tag not in cell.between:
                        est, se, dof = row["ln_ratio"], row["ln_combined_se"], row["dof"]
                        row["axis_account"] = P.beats_label(P.bounds(est, se, dof)[0])
                        row["pair_account"] = P.equal_label(*P.bounds(est, se, dof, 0.90))
                    rows.append(row)
    return rows


def _fraction(cell: _Cell, c: dict) -> list[dict]:
    """A11 as A14 reads it: f = mean_k N_k / mean_k D_k over runs k, with N_k and
    D_k the sums of weight x ln m over the arms of `numerator` and
    `denominator`, paired by run, and its 95 % Fieller interval (P.fieller: the
    paper's run + test error and Student t, with the denominator's error inside
    the interval). At each checkpoint, not between them: a fraction is not a
    ratio, and its two checkpoint values are reported side by side."""
    nw, dw = c["numerator"], c["denominator"]
    arms = list(dict.fromkeys([*nw, *dw]))
    rows = []
    for tag in cell.tags:
        if tag in cell.between:
            continue
        runs = sorted(r for r in cell.runs(arms[0], tag) if all(r in cell.runs(a, tag) for a in arms))
        if not runs:
            continue
        member = {r: {a: cell.runs(a, tag)[r] for a in arms} for r in runs}
        models = [m for r in runs for m in member[r].values()]
        row = cell.row(c["id"], None, None, tag, estimand=c["estimand"], numerator=nw,
                       denominator=dw, models=models)
        zero = cell.check(models, c["id"])
        if zero:
            row["not_computed"] = cell.undefined(zero)
        else:
            d = cell.dirs(models)
            kept, excluded = P._stream_filter(runs, None if d is None else
                                              {r: [d[m] for m in member[r].values()] for r in runs},
                                              cell.ck(tag))
            if excluded is not None:
                row.update(stream_pairing="identical",
                           excluded_runs=[{"run": r, **bad} for r, bad in excluded])
            row["runs"] = kept
            if not kept:
                row["not_computed"] = "every run failed the stream check (A7)"
            else:
                def side(r, w):
                    return sum(x * np.log(cell.v(member[r][a])) for a, x in w.items())
                row.update(P.fieller([side(r, nw) for r in kept], [side(r, dw) for r in kept],
                                     small_sample_rule=cell.rule))
        cell.censor(row, models)
        rows.append(row)
    return rows


def _p2_clause(runs: list, per_run: list, what: str, rule: bool) -> dict:
    """One paired contrast of the P2 block, in ln(1 - AUC): the per-run values, the
    mean, its run + test error, and its 95 % and 90 % intervals."""
    e = P.paired_log(dict(zip(runs, per_run)), small_sample_rule=rule)
    out = {"contrast": what, "per_run": [float(v[0]) for v in per_run]}
    if "not_computed" in e:
        return {**out, "not_computed": e["not_computed"]}
    est, se, dof = e["ln_ratio"], e["ln_combined_se"], e["dof"]
    return {**out, "estimate": est, "ln_test_se": e["ln_test_se"], "ln_combined_se": se, "dof": dof,
            "ci95": list(P.bounds(est, se, dof)), "ci90": list(P.bounds(est, se, dof, 0.90))}


def _p2_verdict(cells: dict, c: dict) -> list[dict]:
    """A14, P2 restated, one block per probe and checkpoint with every number it
    used, in ln(1 - AUC) (lower is better), paired over the runs every role has
    on both tasks (the stream check over all five runs of an index). gap = R16_Q1
    - R42_Q1 on two-prong b vs c, run by run.
      (a) manipulation check, four-prong b vs c: F0 - F1, holds when its 95 % lower
          bound is above 0;
      (b) two-prong: (F0 - F1) - gap/2, holds when its 95 % lower bound is above 0;
      (c) two-prong: F0 - R16_Q1, holds when its 90 % interval lies within +-gap/4.
          The margin is a quarter of the point estimate of the gap over the same
          runs, taken as fixed: its own uncertainty is not carried into (c) or the
          withdrawal clause, and the gap's interval is recorded beside it;
      (d) alignment, two-prong: F1R - F1, holds when its 95 % lower bound is above 0.
    Each fails when its interval lies wholly on the other side, and is
    inconclusive when it spans both. Withdrawal: when (b) holds and the 95 %
    upper bound of F1R - F1 is below gap/4, the axis-specific rule is withdrawn;
    when its lower bound is above gap/4, it is not; an interval spanning gap/4
    is inconclusive, never a withdrawal."""
    spec = next(iter(cells.values())).spec
    own = json.loads((REPO / spec["probe_pairs"]).read_text())["balance_pairs"]
    two, four = own[c["two_prong_pair"]]["task"], own[c["four_prong_pair"]]["task"]
    ro = c["roles"]
    by_kind = {}
    for (fam, task, kind, metric), cell in cells.items():
        if fam == "probe" and metric == c["metric"] and task in (two, four):
            by_kind.setdefault(kind, {})[task] = cell
    out = []
    for kind, per in sorted(by_kind.items()):
        if two not in per or four not in per:
            continue
        T, F = per[two], per[four]
        for tag in T.tags:
            if tag in T.between:
                continue
            runs = sorted(r for r in T.runs(ro["F0"], tag) if all(r in T.runs(a, tag) for a in ro.values())
                          and all(r in F.runs(ro[x], tag) for x in ("F0", "F1")))
            if not runs:
                continue
            tm = {r: [T.runs(a, tag)[r] for a in ro.values()] for r in runs}
            fm = {r: [F.runs(ro[x], tag)[r] for x in ("F0", "F1")] for r in runs}
            block = {"contrast": c["id"], "kind": kind, "metric": c["metric"], "checkpoint": tag,
                     "two_prong_task": two, "four_prong_task": four, "roles": ro,
                     "units": "ln(1 - AUC), lower is better"}
            if T.root is not None:
                bad = {r: P.stream_check(list(T.dirs(tm[r]).values()), T.ck(tag)) for r in runs}
                block["excluded_runs"] = [{"run": r, **b} for r, b in bad.items() if b]
                runs = [r for r in runs if not bad[r]]
            block["runs"] = runs
            zero = T.check([m for r in runs for m in tm[r]], c["id"]) + \
                F.check([m for r in runs for m in fm[r]], c["id"])
            cens = [m for cl, ms in ((T, tm), (F, fm)) for r in runs for m in ms[r]
                    if cl.metas[m].get("censored")]
            if cens:
                block.update(censored_models=cens, note=AUC_FLOOR)
            if not runs or zero:
                block["not_computed"] = (T.undefined(zero) if zero else
                                         "every run failed the stream check (A7)")
                out.append(block)
                continue

            def lt(x, r):
                return np.log(T.v(T.runs(ro[x], tag)[r]))

            def lf(x, r):
                return np.log(F.v(F.runs(ro[x], tag)[r]))
            gap = _p2_clause(runs, [lt("R16", r) - lt("R42", r) for r in runs],
                             "R16_Q1 - R42_Q1, two-prong b vs c: the 43- to 17-class gap", T.rule)
            cl = {"a": _p2_clause(runs, [lf("F0", r) - lf("F1", r) for r in runs],
                                  "F0 - F1, four-prong b vs c (manipulation check)", T.rule),
                  "b": _p2_clause(runs, [lt("F0", r) - lt("F1", r) - (lt("R16", r) - lt("R42", r)) / 2
                                         for r in runs], "(F0 - F1) - gap/2, two-prong b vs c", T.rule),
                  "c": _p2_clause(runs, [lt("F0", r) - lt("R16", r) for r in runs],
                                  "F0 - R16_Q1, two-prong b vs c (equivalence)", T.rule),
                  "d": _p2_clause(runs, [lt("F1R", r) - lt("F1", r) for r in runs],
                                  "F1R - F1, two-prong b vs c (alignment)", T.rule)}
            block.update(gap=gap, clauses=cl)
            failed = [x for x in [gap, *cl.values()] if "not_computed" in x]
            if failed:
                block["not_computed"] = failed[0]["not_computed"]
                out.append(block)
                continue
            m = gap["estimate"] / 4
            block["margin"] = {"value": m, "is": "a quarter of the point estimate of the gap over the "
                               "same runs, taken as fixed; the gap's own 95 % interval is gap.ci95"}
            for k in ("a", "b", "d"):
                cl[k].update(threshold=0.0, interval="ci95", label=P.threshold_label(*cl[k]["ci95"]))
            cl["c"].update(margin=m, interval="ci90", label=P.equivalence_label(*cl["c"]["ci90"], m))
            if not m > 0:
                w = "not evaluable: the 43- to 17-class gap is not positive"
            elif cl["b"]["label"] != "holds":
                w = f"not reached: (b) is {cl['b']['label']}"
            else:
                w = {"holds": "not withdrawn", "fails": "withdrawn", "inconclusive": "inconclusive"}[
                    P.threshold_label(*cl["d"]["ci95"], m)]
            block["withdrawal"] = {"label": w, "reads": "(d)'s 95 % interval against gap/4, when (b) holds"}
            out.append(block)
    return out


KINDS = {"pairs": _pairs, "all_pairs": _pairs, "unpaired": _unpaired, "linear": _linear,
         "checkpoint": _checkpoint, "draws": _draws,
         "partition_split_vs_merged": _partition_split_vs_merged,
         "random_vs_semantic": _random_vs_semantic, "fraction": _fraction}
ACROSS = {"partition_joint": _partition_joint}   # contrasts that read several cells at once


def selected_epochs(spec: dict, root) -> dict:
    """A14: every finished run's selected epochs, by vocabulary: {arm: {run: {best70,
    bestval}}}, from best_window_epoch.json and best_epoch.json. A run without
    DONE is left out."""
    out = {}
    for arm, run, d in sorted(spec["models"].values()):
        rd = pathlib.Path(root) / d
        if not (rd / "DONE").exists():
            continue
        out.setdefault(arm, {})[run] = {
            ck: json.loads((rd / f).read_text())["epoch"] if (rd / f).exists() else None
            for ck, (f, _) in P.SELECTED_EPOCH.items()}
    return out


def mass_shares(spec: dict, root) -> dict:
    """A11 (A14): the realised loss and gradient shares of every finished run of a
    mass-output arm (the grid's mass_lambda), from its metrics/epoch-EEE.json. Per
    epoch, the loss ratio x = lambda L_reg / L_cls of the training means (A11's
    basis) and, where the epoch records grad_diag, the trunk-gradient norm ratio
    rho = |grad lambda L_reg| / |grad L_cls| on the fixed validation batch and the
    cosine of the two gradients. Each is averaged over the run's epochs; the
    shares are x / (1 + x) and rho / (1 + rho) of those averages; then mean and
    SD over runs per arm. Runs without DONE give nothing, so before any run has
    finished the result is empty."""
    out = {}
    for model, (arm, run, d) in sorted(spec["models"].items()):
        lam = spec.get("grid_arms", {}).get(arm, {}).get("mass_lambda")
        rd = pathlib.Path(root) / d
        if not lam or not (rd / "DONE").exists():
            continue
        x, rho, cos = [], [], []
        files = sorted((rd / "metrics").glob("epoch-*.json"))
        for f in files:
            rec = json.loads(f.read_text())
            x.append(lam * rec["train"]["loss_reg"] / rec["train"]["loss_cls"])
            g = rec.get("grad_diag")
            if g:
                rho.append(g["grad_norm"]["lambda_loss_reg"] / g["grad_norm"]["loss_cls"])
                cos.append(g["cosine"])
        r = {"model": model, "epochs": len(files)}
        if x:
            r.update(loss_ratio=float(np.mean(x)), loss_share=float(np.mean(x) / (1 + np.mean(x))))
        if rho:
            r.update(grad_epochs=len(rho), grad_norm_ratio=float(np.mean(rho)),
                     grad_share=float(np.mean(rho) / (1 + np.mean(rho))), grad_cosine=float(np.mean(cos)))
        out.setdefault(arm, {"lambda": lam, "runs": {}})["runs"][run] = r
    for a in out.values():
        for k in ("loss_ratio", "loss_share", "grad_norm_ratio", "grad_share", "grad_cosine"):
            v = [r[k] for r in a["runs"].values() if k in r]
            if v:
                a[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                        "n_runs": len(v)}
    return out


# A11's three mass-output arms, keyed as make_tables.py's SLOTS read loss_share.json
A11_SHARE_KEYS = {"L162_MASS": "162+mass", "R16_Q1_MASS": "17+mass", "R16_Q1_MASS_LM": "17+mass_matched"}


def a11_share_file(spec: dict, root) -> dict:
    """experiments/FIGS/data/v2/mass_lambda_matched/loss_share.json (A11, A14): per mass-output
    arm the per-run realised loss share x/(1+x), trunk-gradient share rho/(1+rho) and gradient
    cosine of mass_shares, runs in order. An arm with some runs finished and not all is
    fatal: the shares are reported over every run of it."""
    sh = mass_shares(spec, root)
    out = {"shares": {}, "grad_shares": {}, "grad_cosine": {}, "a11_shares": sh}
    for arm, key in A11_SHARE_KEYS.items():
        if arm not in sh:
            continue
        runs = sh[arm]["runs"]
        if sorted(runs) != list(range(1, spec["grid_arms"][arm]["runs"] + 1)):
            raise SystemExit(f"FATAL: {arm} has finished runs {sorted(runs)} of "
                             f"{spec['grid_arms'][arm]['runs']}")
        out["shares"][key] = [runs[k]["loss_share"] for k in sorted(runs)]
        if all("grad_share" in r for r in runs.values()):
            out["grad_shares"][key] = [runs[k]["grad_share"] for k in sorted(runs)]
            out["grad_cosine"][key] = [runs[k]["grad_cosine"] for k in sorted(runs)]
    return out


def checkpoint_dependence(rows: list, between: list) -> dict:
    """A14: per comparison between checkpoints, the number of dependent results
    (95 % interval excluding 0: 'depends on the checkpoint', or 'depends on the
    checkpoint, under 10%' when it also lies within +-ln 1.1, counted apart as
    well) against its 5 % null expectation, with the binomial tail P(X >= count)
    at N and 0.05. The results share models and test jets, so they are not
    independent and the tail is a reference, not a test. 'results' are the
    contrasts; 'models' the checkpoint contrast, one model's own metric.

    The 5 % is the nominal rate. A result left with two runs takes A14's floor
    (the error never below the observed spread) and Student t at 1 degree of
    freedom, and under the null excludes 0 far less often than 5 %
    (src/stats/tests/test_paired.py measures it), so 5 % of all results
    overstates what the null gives. Those results are also counted on their own
    (two_run_rule, no expectation) and the others with their own 5 % expectation
    and tail (other).

    A result whose two checkpoints are one file in every run (the global best is
    the run's selected epoch within 70-79, so bestval links to best70) has a ratio
    of exactly 1 with no error: it cannot depend on the checkpoint, so it is
    counted apart (identical_checkpoint) and left out of every count above."""
    from scipy.stats import binom

    def tally(sel):
        k, n = sum(r["checkpoint_label"] in P.DEPENDENT for r in sel), len(sel)
        return {"dependent": k, "n": n, "expected_under_null": 0.05 * n,
                "binomial_tail_p": float(binom.sf(k - 1, n, 0.05)) if n else None}

    def identical(r):
        return r.get("ln_ratio") == 0.0 and r.get("ln_combined_se") == 0.0

    def count(sel):
        same = [r for r in sel if identical(r)]
        sel = [r for r in sel if not identical(r)]
        two = [r for r in sel if r.get("two_run_rule")]
        return {**tally(sel),
                "dependent_under_10pc": sum(r["checkpoint_label"] == P.DEPENDS_UNDER_10 for r in sel),
                "robust": sum(r["checkpoint_label"] == "robust" for r in sel),
                "inconclusive": sum(r["checkpoint_label"] == "inconclusive" for r in sel),
                "censored": sum(bool(r.get("censored_models")) for r in sel),
                "identical_checkpoint": len(same),
                "two_run_rule": {"dependent": sum(r["checkpoint_label"] in P.DEPENDENT for r in two),
                                 "n": len(two)},
                "other": tally([r for r in sel if not r.get("two_run_rule")])}
    out = {}
    for tag in between:
        lab = [r for r in rows if r["checkpoint"] == tag and "checkpoint_label" in r]
        out[tag] = {"results": count([r for r in lab if r["contrast"] != "checkpoint"]),
                    "models": count([r for r in lab if r["contrast"] == "checkpoint"]),
                    "by_contrast": {cid: count([r for r in lab if r["contrast"] == cid])
                                    for cid in dict.fromkeys(r["contrast"] for r in lab)}}
    return out


def _rejections(cell: _Cell) -> list[dict]:
    out = []
    for (arm, tag), runs in cell.arms.items():
        if tag in cell.between:
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
    not is reported and left out (A7). v2 adds the A14 checkpoint label to every
    result between checkpoints and their count (checkpoint_dependence), the P2
    verdict blocks, and, from the run directories, each run's selected epochs
    and the A11 shares."""
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
            if c["kind"] in KINDS:
                rows += KINDS[c["kind"]](cell, c)
        if cell.metric.startswith("eps_b@"):
            rej += _rejections(cell)
    for c in spec["contrasts"]:
        if c["kind"] in ACROSS:
            rows += ACROSS[c["kind"]](cells, c)
    between = between_checkpoints(spec, readouts_of(m for g in groups.values() for m in g))
    for r in rows:                  # A14: every result at another checkpoint over the primary
        if r["checkpoint"] in between and "ln_combined_se" in r:
            r["checkpoint_label"] = P.checkpoint_label(*P.bounds(r["ln_ratio"], r["ln_combined_se"],
                                                                 r["dof"]))
    used = {m for r in rows for k in ("fine_models", "coarse_models", "models") for m in r.get(k, [])}
    unused = sorted({m for cell in cells.values() for m in cell.key} - used)
    if unused:
        print(f"WARNING: {len(unused)} models enter no contrast, e.g. {unused[:3]}")
    out = {"ratios": rows, "rejections": rej, "point_values": points,
           "models_in_no_contrast": unused,
           "contrasts": {"path": spec["path"], "sha256": spec["sha256"],
                         "version": spec["version"], "pending_arms": spec["pending_arms"]},
           "audit_b3": bool(spec.get("audit_b3"))}
    if between:
        out["checkpoint_dependence"] = checkpoint_dependence(rows, between)
    p2 = [c for c in spec["contrasts"] if c["kind"] == "p2_verdict"]
    if p2 and cells:
        out["p2_verdict"] = [b for c in p2 for b in _p2_verdict(cells, c)]
    if "grid_arms" in spec:
        out["selected_epochs"] = None if run_dirs_root is None else selected_epochs(spec, run_dirs_root)
        out["a11_shares"] = {} if run_dirs_root is None else mass_shares(spec, run_dirs_root)
    return out


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
            s.add_argument("--checkpoint-rule", choices=("best70", "wavg", "bestval"), default=None,
                           help="v2: the rule of the tree under the roots, which every "
                                "cell's init_checkpoint.json must record")
    s = sub.add_parser("a11-shares", help="loss_share.json of the mass-output runs (A11)")
    s.add_argument("--run-dirs-root", type=pathlib.Path, required=True)
    s.add_argument("--out", required=True, type=pathlib.Path)
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

    if a.cmd == "a11-shares":
        if a.out.exists():
            raise SystemExit(f"FATAL: {a.out} exists; refusing to overwrite")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({**a11_share_file(load_spec(CONTRASTS["v2"]), a.run_dirs_root),
                                     "provenance": {"run_dirs_root": str(a.run_dirs_root),
                                                    "script_sha256": _sha(__file__)}}, indent=1))
        print(f"wrote {a.out}")
        return 0
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
            res["method"]["checkpoint_label"] = (
                "A14: every contrast is also formed at checkpoints 'wavg/best70' (robustness) "
                "and 'bestval/best70' (sensitivity), the result at the weight average, or at "
                "the global best, over the result at the primary checkpoint, run by run on the "
                "same test resamplings (contrast 'checkpoint': one model). From its 95% "
                "interval: 'depends on the checkpoint' when it excludes 0, 'robust' when it "
                "lies within +-ln 1.1, 'depends on the checkpoint, under 10%' when both hold "
                "(A14 does not order them), else 'inconclusive'; checkpoint_dependence counts "
                "the results whose interval excludes 0 against 5% of them, and also apart "
                "the results with two runs (two_run_rule: their floor and 1 degree of "
                "freedom exclude 0 far less often than 5% under the null) and the others")
            res["method"]["small_samples"] = (
                "A14: with two runs the error is never below the observed spread and the "
                "interval takes Student t at 1 degree of freedom; a paired contrast left with "
                "one run pair is not computed")
            res["method"]["p1"] = (
                "A14: per pair, the runs of the merging and splitting partitions, each side "
                "with its own run variance from the runs that replicate one partition (Welch); "
                "p1_label 'merging costs nothing' (95% upper bound below ln 1.1), 'merging "
                "costs' (lower bound above 0), 'merging costs, under 10%' when both hold (A14 "
                "does not order them), else 'inconclusive'. The joint fit "
                "weights each task by 1 / (run variance / runs + test variance) and leaves out "
                "a task with a censored unit; partition_joint_dose adds one term per axis")
            res["method"]["fraction"] = (
                "A11: f = mean N / mean D over paired runs with a 95% Fieller interval, the "
                "error of mean(N - f D) by the combined error; no interval when the "
                "denominator's own 95% interval holds 0")
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
