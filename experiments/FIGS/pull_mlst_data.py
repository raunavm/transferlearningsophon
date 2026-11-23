"""The MLST paper's inputs from /data, as small JSONs for experiments/FIGS/data/v2/.

Runs in one of our pods (it reads /data) and writes a gzipped tar to stdout:

  linprobe/fits3000/<model>.json   the linear-probe fits, unchanged (scripts/build_linprobe_jobs.py)
  linprobe/mass/<model>.json       the jet-mass regressions, unchanged (mass_resolution.json)
  compute/<run>.json               per epoch of every finished run: the GPU it trained on, the
                                   seconds of its training and validation passes and, with a mass
                                   output, the mass term's share of the training loss, read from
                                   <run>/metrics/epoch-NNN.json
  ft_mlst/<tag>_<file>.json        the fine-tuning read-outs (scripts/build_ft_jobs.py, v2 read-outs):
                                   <tag> best70 (every paper run) or best70_done18 (the 18 runs
                                   finished before the split), <file> leg1_metrics, leg2_metrics,
                                   bench_metrics, bench_metrics_herwig; each must parse
  jc2_test/counts.json             per-class counts of the 2M JetClass-II test jets (all, and AUC rows)
  ft_mlst/manifests.json           {cell: its recipe, GPU and node} for every DONE fine-tuning cell of
                                   the paper runs (best70) and the V100 references (ref_v100), read
                                   from each cell's ft_manifest.json
  linprobe/fits_v3/<model>.json    the 2026-10-29 refit (JetClass-II at 188 classes), and
  linprobe/fits_v3_mlp/<model>.json  the nonlinear probes (scripts/build_linprobe_jobs.py --v3)
  linprobe/fits_v3_mlp2/<model>.json, fits_v3_mlp2_initbn/, fits_v3_initbn/  the nonlinear probes on
                                   every null task and the BatchNorm-recomputed random networks (--v4)
  ft_mlst/best70_leg188_leg1_metrics.json  JetClass-II fine-tuning at 188 classes with its reference
  ft_mlst/topcheck_bench_metrics.json      the top-tagging recipe check (100 epochs at 1e4)
  head/<model>_{mass,sculpting}.json  mass node, mass regressions and QCD mass sculpting
                                   (scripts/build_head_jobs.py)
  fillcontrol/<mask>/<model>.json  JetClass-II probes with the downstream data's missing
                                   features zeroed (scripts/build_fill_control_v2.py)
  curves/<name>.json, bootstrap/<family>/<task>_N<N>.json  per-class tables, ROC points,
                                   test-sample spreads
                                   (scripts/build_curve_jobs.py)
  xbx/<signal>/<model>.json        X->bb / X->bs on the real samples (scripts/build_xbx_jobs.py)
  linprobe/fits_v4/, fits_v4_mlp/  the v4 probes (experiments/EVAL/probe_bal.py, scripts/build_jc2bal_jobs.py),
                                   the kinematic-only model's top probe among them (jc2bal-kin)
  linprobe/fits_v4_fill/           the three substitutions of the missing inputs side by side
                                   (probe_bal.py --fill, scripts/build_jc2bal_jobs.py fitfill)
  linprobe/fits_v4_sib/            the X->ee vs X->mumu probe (probe_bal.py --sibling-pairs, jc2bal-sib)
  curves_v2/<family>/<model>.json  per-class ROC curves with the 20-background-jet cut
  tsne/<set>_<model>.json          t-SNE of the frozen class token (experiments/EVAL/tsne_class_token.py)
  umap/, umap_preview/<set>_<panel>.json  UMAP of the class token (experiments/EVAL/umap_class_token.py;
                                   the preview from the old 10-epoch JetClass cells)
  head/atlas_res_fine/<model>.json the ATLAS resolution in ten pT bins (experiments/EVAL/atlas_rebin.py)
  head/input_bn_offsets.json       where a zeroed input lies after the input BatchNorm (experiments/EVAL/input_bn_offsets.py)
  linprobe/fits_v4_sib2/<model>.json  X->ee vs X->mumu on every test jet of the pair
                                   (experiments/FT/roc_points.py; families whose DONE exists, no npz)
  head/atlas_res/<model>.json      ATLAS mass resolution and median response (/data/results/eval/atlas_res)
  xbx_scratch/<signal>/s<k>.json   X->bs from scratch, its finished cells' endpoints.json
                                   (scripts/build_xbx_jobs.py SCRATCH_ROOT_IHEP/<signal>/s<k>/)
  jc2bal/{census,manifest}.json    the class-balanced JetClass-II sets (experiments/FT/stage_jc2_balanced.py)
  head/curves_v2/<model>_{sculpting_v2,response}.json  sculpting curves and mass response histograms
  checks/test_disjoint.json        TEST2M shares no jet with the fine-tuning sets (experiments/EVAL/check_test_disjoint.py)
  budget/{budget,lrsweep}_metrics.json  the fixed-budget fine-tuning read-outs (experiments/FT/budget_metrics.py);
                                   budget_{,long_}fulldecay_metrics.json the from-scratch cells with the whole schedule
  budget/fixed_metrics.json        the one fine-tuning rule's 341 cells (PI 2026-11-15; job-budget-readout-fixed),
                                   fixed_metrics_partial.json its preview read-out (job-budget-readout-fixed-preview),
                                   best70_kin_metrics.json the kinematic-only network on the full top set
  budget/fixed_sweep.json          the one rule's rate sweep at 1e4 jets (fixed_sweep below, from each cell's files)
  head/mass_sd_boot.json           each mass estimator against the soft-drop mass, paired bootstrap (mass_sd_boot.py)
  stats/confusion_share.json       the confusion-share test of the 17-group label sets (confusion_share.py)
  budget/rate_search.json          every row's rate search and the reruns at the chosen rates, raw per cell
                                   (rate_search_scan below; the rules in experiments/FT/rate_search.py)
  budget/ft_attempts.json          every attempt of every fine-tuning cell under ft_v2: pod, start, end, GPU
                                   (ft_attempts below), for the compute table's fine-tuning GPU-hours
  budget/jc1_acc_boot_N<N>.json    the paired bootstrap of the JetClass accuracy differences (experiments/FT/jc1_acc_boot.py)
  partrepro/<variant>.json         ParT's top-tagging script reproduced (scripts/build_partrepro_jobs.py)
  dup_check/<file>.json            per-jet comparisons of near-identical results (experiments/EVAL/dup_check.py)
  perf_vs_pt/<task>/<model>.json   performance against jet pT, jc2 (linear probe) and jc1 (fine-tuning)
  confusion_jc1/<model>.json       the JetClass confusion matrices (experiments/EVAL/perf_vs_pt.py)
  curves_v2/pairs_ft/<model>.json  the fine-tuned networks on the class pairs (roc_points.py ft --family pairs_ft,
                                   from scratch included), with the other DONE families of curves_v2
  curves_v2/<family>_{v5,fixed}/   the reruns of the ROC families (scripts/build_gallery_data_job.py probe-v5,
                                   probe-v5-fill, jc1-fixed, ft-pairs-fixed; mlst_v4.ROC_RERUN), by the same rule:
                                   a job writes every DONE of its group after its last file, so a group comes whole
                                   (not /data/results/eval/v2_ft_head_fixed: per-jet outputs, too large to pull)
  head/sculpting_ft/<model>.json   QCD mass sculpting of the fine-tuned JetClass-II networks (experiments/EVAL/sculpting_ft.py),
                                   only once its job wrote DONE (after every model)
  head/sculpting_boot.json         the paired bootstrap of the heads' sculpting differences (experiments/EVAL/sculpting_boot.py)
                                   (not /data/results/eval/v2_ft_head: per-jet outputs, too large to pull)
  perf_vs_pt_fixed/{jc2_ft,jc1}/, confusion_jc1_fixed/, head/sculpting_ft_fixed/, umap_v4/  the rerun's read-outs
                                   of the one rule's fine-tuned cells at 1e6 (scripts/build_perf_pt_jobs.py jc2fixed
                                   jc1fixed, build_gallery_data_job.py sculpt-ft-fixed, build_umap_jobs.py umap_v4),
                                   each folder whole after its DONE
  perf_vs_pt_v5/{jc2,jc1_probe}/   the v5 linear probes per jet-pT bin (build_perf_pt_jobs.py jc2v5 jc1probev5), whole
                                   after its DONE and only with the v5 refits (V5_PASS)
  l3_rerun/<file>.json             the rerun's decision-L3 bootstraps (scripts/build_boot_l3_jobs.py --rerun), each file
                                   written whole; not under bootstrap/, whose folders are tab:boot's
  perclass_recall/{probe_<task>,ft_<task>_N<n>}.json  per-class recall of every network, its test-jet bootstrap and
                                   decision L3's differences (experiments/EVAL/perclass_recall.py), each written whole
  index.json                       {path: sha256} of every file above: the single source the
                                   macros counted over many files cite (mlst_results.py checks it)

Only finished runs (a DONE marker) are taken, so a run is never read half-trained.

    kubectl exec -i -n cms-ml <pod> -- python3 - < experiments/FIGS/pull_mlst_data.py \\
        | tar xzf - -C experiments/FIGS/data/v2
"""
import hashlib
import io
import json
import pathlib
import re
import sys
import tarfile
import time

LP = pathlib.Path("/data/results/eval/v2_linprobe")
FT = pathlib.Path("/data/results/ft_v2")
FT_TAGS = ("best70", "best70_done18")
FT_FILES = {"leg1": ("leg1_metrics",), "leg2": ("leg2_metrics",),
            "bench": ("bench_metrics", "bench_metrics_herwig")}
# (directory under FT, file in it, name pulled)
FT_EXTRA = (("best70_leg188_leg1_metrics", "leg1_metrics", "best70_leg188_leg1_metrics"),
            ("recipe_check/leg_top_e100_bench_metrics", "bench_metrics", "topcheck_bench_metrics"))
EVAL = pathlib.Path("/data/results/eval")
# the v5 refits (scripts/build_refit_v5_jobs.py): pulled whole once its checks job wrote PASS
V5_PASS = EVAL / "refit_v5_checks" / "PASS"
V5_PULL = ((EVAL / "v2_linprobe_bal" / "fits_v5", "linprobe/fits_v5"),
           (EVAL / "v2_linprobe_bal" / "fits_v5_fill", "linprobe/fits_v5_fill"),
           (EVAL / "v2_linprobe_bal" / "fits_v5_sib2", "linprobe/fits_v5_sib2"),
           (EVAL / "v2_linprobe_bal" / "fits_v5_coarse", "linprobe/fits_v5_coarse"),
           (EVAL / "v2_linprobe_bal" / "fits_v5_qcdhf", "linprobe/fits_v5_qcdhf"),
           (EVAL / "v2_linprobe_bal" / "fits_v5_mlp", "linprobe/fits_v5_mlp"),
           (EVAL / "v2_linprobe" / "fits_v5_nat", "linprobe/fits_v5_nat"),
           (EVAL / "v2_linprobe_sophon" / "fits_v5", "linprobe/sophon_v5"),
           (EVAL / "v2_linprobe_sophon" / "fits_v5_fill", "linprobe/sophon_v5_fill"))
# the truth-free mass refit (PI 2026-11-16/21): the only mass files the paper reads, pulled whole once each wrote DONE,
# not behind V5_PASS (the class-centred atlas_res is never read)
MASS_V3_PULL = ((EVAL / "atlas_res_v3", "head/atlas_res_v3"), (EVAL / "atlas_res_fine_v3", "head/atlas_res_fine_v3"))
V5_BOOT = ("probe_v5", "l3_v5")
# the rerun's read-outs (PI 2026-11-15/16): (folder, name pulled), each taken whole once its job wrote DONE; the v5
# probes' only with the v5 refits
RERUN_PULL = ((EVAL / "perf_vs_pt_fixed" / "jc2_ft", "perf_vs_pt_fixed/jc2_ft"),
              (EVAL / "perf_vs_pt_fixed" / "jc1", "perf_vs_pt_fixed/jc1"),
              (EVAL / "confusion_jc1_fixed", "confusion_jc1_fixed"),
              (EVAL / "v2_head" / "sculpting_ft_fixed", "head/sculpting_ft_fixed"),
              (EVAL / "umap_v4", "umap_v4"))
RERUN_PULL_V5 = ((EVAL / "perf_vs_pt_v5" / "jc2", "perf_vs_pt_v5/jc2"),
                 (EVAL / "perf_vs_pt_v5" / "jc1_probe", "perf_vs_pt_v5/jc1_probe"))
L3_RERUN = pathlib.Path("/data/results/bootstrap/l3_rerun")
PERCLASS_RECALL = EVAL / "perclass_recall"
MTX = pathlib.Path("/data/results/mtx_v2")
N_EPOCHS = 80
MANIFEST_KEYS = ("leg", "init", "n_train", "ft_seed", "lr", "head_lr_mult", "weight_decay", "epochs",
                 "lr_schedule", "recipe", "batch_size", "samples_per_epoch", "val_jets", "val_rule", "ckpt_rule",
                 "gpu_device_name", "node_name", "repo_commit", "checkpoint_sha256")


# Every model is scored on the same JetClass-II test jets (the fits record one test_label_sha256),
# so one model's cached labels give the test composition.
JC2_TEST_LABELS = EVAL / "v2_linprobe/mtx-l188-s1/jc2/test/best70/label188.npy"
JC2_TEST_N, JC2_AUC_STRIDE = 2_000_000, 4


def jc2_test_counts() -> dict:
    """Per-class counts of the JetClass-II test jets the probes score: all of them (accuracy, pairs)
    and every JC2_AUC_STRIDE-th (the macro AUC)."""
    import numpy as np
    y = np.load(JC2_TEST_LABELS)[:JC2_TEST_N].astype(np.int64)
    return {"source": str(JC2_TEST_LABELS), "n_jets": int(y.size), "auc_stride": JC2_AUC_STRIDE,
            "label_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
            "counts": np.bincount(y, minlength=188).tolist(),
            "counts_auc": np.bincount(y[::JC2_AUC_STRIDE], minlength=188).tolist()}


def manifests() -> dict:
    out = {}
    for pattern in ("best70/*/*/N*/s*", "ref_v100/*/*/*/N*/s*", "recipe_check/*/*/N*/s*"):
        for cell in sorted(FT.glob(pattern)):
            if cell.is_dir() and ".partial." not in cell.name and not cell.name.endswith(".lock") \
                    and (cell / "DONE").exists():
                m = json.loads((cell / "ft_manifest.json").read_text())
                out[str(cell.relative_to(FT))] = {k: m.get(k) for k in MANIFEST_KEYS}
    return out


FIXED_SWEEP, SWEEP_N = FT / "fixed_sweep", "N10000"


def fixed_sweep() -> dict:
    """{tree/<init>_lr<rate>: cell} of the one rule's rate sweep at 1e4 jets (build_ft_budget_jobs fixed_sweep_cells,
    -ext, -17): its best validation accuracy and check from early_stop.json, from the last attempt (s1.partial.<time>)
    where no attempt reached DONE (the cell was stopped), and its training log's divergence evidence: any NaN training
    loss, and the final training loss and accuracy."""
    out = {}
    for d in sorted(FIXED_SWEEP.glob(f"*/*_lr*/{SWEEP_N}")):
        cell, done = d / "s1", (d / "s1" / "DONE").exists()
        if not done:
            tries = [p for p in d.glob("s1.partial.*") if (p / "early_stop.json").exists()]
            if not tries:
                continue
            cell = max(tries, key=lambda p: int(p.name.rsplit(".", 1)[1]))
        es = json.loads((cell / "early_stop.json").read_text())
        train = re.findall(r"Train AvgLoss: ([^,\s]+), AvgAcc: ([^,\s]+)", (cell / "train.log").read_text())
        last = float(train[-1][0]) if train else None
        init, lr = d.parent.name.rsplit("_lr", 1)
        out[f"{d.parent.parent.name}/{d.parent.name}"] = {
            "tree": d.parent.parent.name, "init": init, "lr": lr, "n": int(SWEEP_N[1:]), "done": done,
            "dir": str(cell.relative_to(FT)), "best_metric": es["best_metric"], "best_epoch": es["best_epoch"],
            "epochs_completed": es["epochs_completed"],
            "last_metric": (es.get("metrics") or {}).get(str(es.get("last_epoch"))),
            "nan_loss": any(x == "nan" for x, _ in train),
            "final_train_loss": None if last is None or last != last else last,
            "final_train_acc": float(train[-1][1]) if train else None}
    return {"root": str(FIXED_SWEEP), "cells": out}


# PI 2026-11-21: the rate search of every row (build_ft_budget_jobs.fixed_search_plan) and the reruns at the chosen rates,
# read as raw facts per cell; experiments/FT/rate_search.py applies the rules (one divergence rule, the floor, the
# schedule check, the recorded rate). Also run alone by the watcher: python3 - rate_search < pull_mlst_data.py
RATE_SEARCH_GLOBS = ("fixed_sweep/*/*/N*/s*", "fixed_sweep_1e4/*/*/N*/s*", "fixed_rate/*/*/N*/s*",
                     "fixed/*/l188-s1/N*/s1", "fixed/*/r16q1-s1/N*/s1", "fixed/*/scratch/N*/s1")
NONFINITE = ("nan", "inf", "-inf")


def _attempt(d: pathlib.Path) -> dict:
    rec = {"dir": str(d.relative_to(FT))}
    log = d / "train.log"
    losses = re.findall(r"Train AvgLoss: ([^,\s]+)", log.read_text(errors="ignore")) if log.exists() else []
    bad = [i for i, x in enumerate(losses) if x.lower() in NONFINITE]
    rec["first_nonfinite_check"] = bad[0] if bad else None
    if (d / "ft_manifest.json").exists():
        m = json.loads((d / "ft_manifest.json").read_text())
        rec["manifest"] = {k: m.get(k) for k in ("lr", "head_lr_mult", "steps_per_epoch", "max_epochs", "init",
                                                 "n_train", "ft_seed", "lr_schedule")}
    if (d / "early_stop.json").exists():
        es = json.loads((d / "early_stop.json").read_text())
        rec.update({k: es.get(k) for k in ("best_metric", "best_epoch", "epochs_completed", "max_epochs")})
    return rec


def rate_search_scan() -> dict:
    """{cell directory under ft_v2: record} for every cell of RATE_SEARCH_GLOBS: DONE (a DONE that says the cell is not a
    result is not), its DIVERGED record, SUPERSEDED, and each attempt (the cell and its s1.partial.<time> siblings,
    oldest first) with its first non-finite training loss, manifest (rate, schedule) and best validation check."""
    out = {}
    for g in RATE_SEARCH_GLOBS:
        for d in sorted(FT.glob(g)):
            if ".partial." in d.name or d.name.endswith(".lock") or not d.is_dir():
                continue
            done = (d / "DONE").exists() and "not a result" not in (d / "DONE").read_text(errors="ignore")
            tries = sorted(d.parent.glob(d.name + ".partial.*"), key=lambda p: int(p.name.rsplit(".", 1)[1])) + [d]
            out[str(d.relative_to(FT))] = {
                "done": done, "superseded": (d / "SUPERSEDED").exists(),
                "diverged_record": json.loads((d / "DIVERGED").read_text()) if (d / "DIVERGED").exists() else None,
                "attempts": [_attempt(t) for t in tries]}
    return {"root": str(FT), "cells": out}


# PI 2026-11-22: the appendix counts every fine-tuning GPU-hour, the 15 Nov sweeps included. An interrupted attempt writes
# nothing when it stops; its folder is set aside (s1.partial.<time>) when the next attempt starts, which bounds its end.
# mlst_results takes a pod's own running time from the Prometheus snapshots (budget/ft_pod_hours_prom_*.json) where they
# hold its whole run.
def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def ft_attempts() -> dict:
    """{cell directory under ft_v2: [attempt, oldest first]} for every cell with an ATTEMPTS file or an ft_manifest.json
    (the first V100 references have no ATTEMPTS): each attempt's pod, node and start (its ATTEMPTS line, else its
    manifest's time), GPU, network and task (its ft_manifest.json), and end: DONE's time ("done"), the time its folder
    was set aside as the next attempt started ("next", an upper bound), or the newest file it wrote ("last write": a
    cell stopped without DONE, or still running)."""
    import os
    cells = set()
    for dp, _dn, fn in os.walk(FT):
        if "ATTEMPTS" in fn or "ft_manifest.json" in fn:
            d = pathlib.Path(dp)
            cells.add(d.parent / d.name.split(".partial.")[0])
    out = {}
    for d in sorted(cells):
        tries = sorted(d.parent.glob(d.name + ".partial.*"), key=lambda q: int(q.name.rsplit(".", 1)[1])) \
            + ([d] if d.is_dir() else [])
        recs = []
        for t in tries:
            lines = [w for ln in (t / "ATTEMPTS").read_text(errors="replace").splitlines()   # one zero-filled file
                     if (w := ln.split()) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", w[0])] \
                if (t / "ATTEMPTS").exists() else []
            kv = dict(x.split("=", 1) for x in (lines[-1][1:] if lines else []) if "=" in x)
            m = json.loads((t / "ft_manifest.json").read_text()) if (t / "ft_manifest.json").exists() else {}
            if t is not d:
                end, kind = int(t.name.rsplit(".", 1)[1]), "next"
            elif (d / "DONE").exists():
                end, kind = (d / "DONE").stat().st_mtime, "done"
            else:
                end, kind = max((q.stat().st_mtime for q in d.iterdir() if q.is_file()), default=d.stat().st_mtime), "last write"
            recs.append({"start": lines[-1][0] if lines else m.get("written_utc"), "end": _utc(end), "end_kind": kind,
                         "pod": kv.get("pod"), "node": kv.get("node"), "gpu": m.get("gpu_device_name"),
                         "init": m.get("init"), "task": m.get("task"), "n_train": m.get("n_train"), "lr": m.get("lr")})
        out[str(d.relative_to(FT))] = recs
    return {"root": str(FT), "cells": out}


def compute(run: pathlib.Path) -> dict:
    epochs = []
    for e in range(N_EPOCHS):
        d = json.loads((run / "metrics" / f"epoch-{e:03d}.json").read_text())
        if d["epoch"] != e or "device" not in d:
            raise SystemExit(f"FATAL: {run.name} epoch {e}: no device record")
        rec = {"epoch": e, "device": d["device"], "train_seconds": d["train"]["seconds"],
               "val_seconds": d["val"]["seconds"], "jets_per_s": d["train"]["jets_per_s"], "val_acc": d["val"]["acc"]}
        if "loss_reg" in d["train"]:
            # the mass term's share of the training loss: loss = loss_cls + lambda * loss_reg
            rec["loss_share_reg"] = (d["train"]["loss"] - d["train"]["loss_cls"]) / d["train"]["loss"]
        epochs.append(rec)
    return {"run": run.name, "epochs": epochs}


def main() -> int:
    buf = io.BytesIO()
    index = {}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            index[name] = hashlib.sha256(data).hexdigest()
        for f in sorted((LP / "fits3000").glob("*.json")):
            add(f"linprobe/fits3000/{f.name}", f.read_bytes())
        for f in sorted((LP / "mass").glob("*/mass_resolution.json")):
            add(f"linprobe/mass/{f.parent.name}.json", f.read_bytes())
        for tag in FT_TAGS:
            for leg, names in FT_FILES.items():
                for n in names:
                    f = FT / f"{tag}_{leg}_metrics" / f"{n}.json"
                    if f.exists():
                        b = f.read_bytes()
                        json.loads(b)                       # a file still being written stops the pull
                        add(f"ft_mlst/{tag}_{n}.json", b)
        for d, n, name in FT_EXTRA:
            f = FT / d / f"{n}.json"
            if f.exists():
                b = f.read_bytes()
                json.loads(b)
                add(f"ft_mlst/{name}.json", b)
        for f in sorted((LP / "fits_v3_mlp2" / "init_bn").glob("*.json")):
            add(f"linprobe/fits_v3_mlp2_initbn/{f.name}", f.read_bytes())
        for sub in ("fits_v3", "fits_v3_mlp", "fits_v3_mlp2", "fits_v3_initbn"):
            for f in sorted((LP / sub).glob("*.json")):
                add(f"linprobe/{sub}/{f.name}", f.read_bytes())
        for f in sorted((EVAL / "v2_head").glob("*/mass_v3.json")):
            add(f"head/{f.parent.name}_mass.json", f.read_bytes())
        for f in sorted((EVAL / "v2_head").glob("*/sculpting.json")):
            add(f"head/{f.parent.name}_sculpting.json", f.read_bytes())
        for f in sorted((EVAL / "v2_fillcontrol").glob("*/fits/*.json")):
            add(f"fillcontrol/{f.parent.parent.name}/{f.name}", f.read_bytes())
        # curves/<tree>.json; bootstrap/<family>/<task>_N<N>.json (scripts/build_curve_jobs.py)
        for d, name, pattern in ((FT / "curves", "curves", "*.json"),
                                 (pathlib.Path("/data/results/bootstrap"), "bootstrap", "*/*.json")):
            for f in sorted(d.glob(pattern)):
                if "." in f.parent.name:
                    continue        # a job's staging directory (<family>.staging.<n>): not finished
                if f.parent.name in V5_BOOT and not V5_PASS.exists():
                    continue        # the v5 bootstraps only with the v5 refits (below)
                if f.parent == L3_RERUN:
                    continue        # pulled to l3_rerun/ (below), outside tab:boot's folders
                b = f.read_bytes()
                json.loads(b)
                add(f"{name}/{f.relative_to(d)}", b)
        for f in sorted((EVAL / "xbx" / "fits_ihep").glob("*/*.json")):
            add(f"xbx/{f.parent.name}/{f.name}", f.read_bytes())
        # v4 (2026-10-31): each file must parse, so one still being written stops the pull
        bal = pathlib.Path("/data/finetune/jc2_bal")
        v4 = ((EVAL / "v2_linprobe_bal" / "fits_v4", "linprobe/fits_v4", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_mlp", "linprobe/fits_v4_mlp", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_fill", "linprobe/fits_v4_fill", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_sib", "linprobe/fits_v4_sib", "*.json"),
              (EVAL / "atlas_res", "head/atlas_res", "*.json"),
              (bal, "jc2bal", "census.json"), (bal, "jc2bal", "manifest.json"),
              (EVAL / "v2_head" / "curves_v2", "head/curves_v2", "*.json"),
              (FT / "budget", "budget", "budget_metrics.json"),
              (FT / "budget_long", "budget", "budget_long_metrics.json"),     # the long budget at 1e6 (2026-11-05)
              # from scratch with the whole flat+decay schedule (job-budget-readout-e, 2026-11-07)
              (FT / "budget_fulldecay", "budget", "budget_fulldecay_metrics.json"),
              (FT / "budget_long_fulldecay", "budget", "budget_long_fulldecay_metrics.json"),
              # pretrained cells run to their last epoch (equal-stopping test and its reruns, 2026-11-08/09)
              (FT / "budget_prefull", "budget", "budget_prefull_metrics.json"),
              # the JetClass accuracy differences against the run spread and a bootstrap of the test jets (PI 2026-11-09)
              (FT / "jc1_acc_boot", "budget", "jc1_acc_boot_N*.json"),
              # the one fine-tuning rule (PI 2026-11-15): its read-out, complete (the paper) or partial (a preview,
              # read only with MLST_FT_PREVIEW=1), and the kinematic-only network on the full top set
              (FT / "fixed", "budget", "fixed_metrics.json"),
              (FT / "fixed", "budget", "fixed_metrics_partial.json"),
              (FT / "best70_kin", "budget", "best70_kin_metrics.json"),
              # the TEST2M disjointness proof (experiments/EVAL/check_test_disjoint.py, 2026-11-09: all 30 fine-tuned heads)
              (pathlib.Path("/data/results/checks"), "checks", "test_disjoint.json"),
              (FT / "lrsweep_v2", "budget", "lrsweep_metrics.json"),     # 50-step checks (2026-11-01)
              (pathlib.Path("/data/results/partrepro"), "partrepro", "*/metrics.json"),
              (EVAL / "dup_check", "dup_check", "*.json"),
              (EVAL / "tsne_v1", "tsne", "*.json"),
              # the gallery decisions of 2026-11-05: finer pT bins, every X->ee / X->mumu test jet, UMAP
              (EVAL / "atlas_res_fine", "head/atlas_res_fine", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_sib2", "linprobe/fits_v4_sib2", "*.json"),
              (EVAL / "umap_preview", "umap_preview", "*.json"),
              (EVAL / "umap_v1", "umap", "*.json"),
              (EVAL / "umap_v2", "umap_v2", "*.json"),     # round 3: frozen and fine-tuned, four tasks
              (EVAL, "head", "input_bn_offsets.json"),
              # PI 2026-11-07: eps_S, R and R50/JSD50 for every model incl. the flavor controls; coarse
              # targets; QCD heavy-flavor subclasses; the released Sophon checkpoint; bestval = best70
              (EVAL / "v2_head" / "sculpting_v3", "head/sculpting_v3", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_coarse", "linprobe/fits_v4_coarse", "*.json"),
              (EVAL / "v2_linprobe_bal" / "fits_v4_qcdhf", "linprobe/fits_v4_qcdhf", "*.json"),
              (EVAL / "v2_linprobe_sophon" / "fits_v4", "linprobe/sophon", "*.json"),
              (EVAL / "v2_linprobe_sophon" / "fits_v4_qcdhf", "linprobe/sophon_qcdhf", "*.json"),
              (EVAL / "v2_ckpt_identity", "ckpt_identity", "identity.json"),
              (EVAL / "jc2_train", "jc2_train", "counts.json"),   # DesignNTrainSelected (experiments/FT/count_jc2_train.py)     # table tab:inputs (experiments/EVAL/input_bn_offsets.py)
              # round 3 (2026-11-06): performance against jet pT and the JetClass confusion matrices
              (EVAL / "perf_vs_pt_v2" / "jc2", "perf_vs_pt/jc2", "*.json"),   # v2: accuracy per bin (round 4)
              (EVAL / "perf_vs_pt_v2" / "jc1", "perf_vs_pt/jc1", "*.json"),
              (EVAL / "confusion_jc1", "confusion_jc1", "*.json"),
              # final figures (2026-11-13): fine-tuned JetClass-II and linear-probe JetClass per pT bin, from-scratch
              # JetClass on its full-schedule cells, and the UMAP grids per task (scripts/build_perf_pt_jobs.py, build_umap_jobs.py)
              (EVAL / "perf_vs_pt_v3" / "jc2_ft", "perf_vs_pt/jc2_ft", "*.json"),
              (EVAL / "perf_vs_pt_v3" / "jc1_probe", "perf_vs_pt/jc1_probe", "*.json"),
              (EVAL / "perf_vs_pt_v3" / "jc1_fulldecay", "perf_vs_pt/jc1_fulldecay", "*.json"),
              (EVAL / "confusion_jc1_fulldecay", "confusion_jc1_fulldecay", "*.json"),
              (EVAL / "umap_v3", "umap_v3", "*.json"),
              # plan 2026-11-08 step 2: the sculpting bootstrap (the fine-tuned sculpting below, once DONE)
              (EVAL / "v2_head", "head", "sculpting_boot.json"),
              # PI 2026-11-21: each mass estimator against the soft-drop mass, paired on the same jets (the mass claims),
              # and the pre-specified confusion-share test of the 17-group label sets
              (EVAL / "mass_sd_boot", "head", "mass_sd_boot_v3.json"),     # v3: after the one-CPU refit (PI 2026-11-21)
              # v4: v3 and tab:tests' mass differences (scripts/build_boot_tests_jobs.py, PI 2026-11-23 SP-12 step 4)
              (EVAL / "mass_sd_boot", "head", "mass_sd_boot_v4.json"),
              (EVAL / "confusion_share", "stats", "confusion_share.json"))
        # the v5 refits (PI 2026-11-14, scripts/build_refit_v5_jobs.py): every family at once, and only after the
        # refit-checks job passed (its PASS file), so a pull never takes part of them (mlst_v4.pick refuses a part)
        v4 += tuple((d, name, "*.json") for d, name in MASS_V3_PULL if (d / "DONE").exists())
        if V5_PASS.exists():
            v4 += tuple((d, name, "*.json") for d, name in V5_PULL)
            for s in ("xbb", "xbs"):
                v4 += ((EVAL / "xbx" / "fits_ihep_v5" / s, f"xbx_v5/{s}", "*.json"),)
        for d, name, pattern in v4:
            for f in sorted(d.glob(pattern)):
                b = f.read_bytes()
                try:
                    json.loads(b)
                except ValueError as e:     # a file still being written stops the pull; say which
                    raise SystemExit(f"FATAL: {f} is not valid JSON ({len(b)} bytes): {e}")
                rel = f"{f.parent.name}.json" if name == "partrepro" else f.name
                add(f"{name}/{rel}", b)
        for d, name in RERUN_PULL + (RERUN_PULL_V5 if V5_PASS.exists() else ()):
            if (d / "DONE").exists():
                for f in sorted(d.glob("*.json")):
                    b = f.read_bytes()
                    json.loads(b)
                    add(f"{name}/{f.name}", b)
        for f in sorted(L3_RERUN.glob("*.json")):
            b = f.read_bytes()
            json.loads(b)
            add(f"l3_rerun/{f.name}", b)
        # per-class recall (experiments/EVAL/perclass_recall.py, PI 2026-11-18): each file written whole (atomic)
        for f in sorted(PERCLASS_RECALL.glob("*.json")):
            b = f.read_bytes()
            json.loads(b)
            add(f"perclass_recall/{f.name}", b)
        # the fine-tuned sculpting (scripts/build_gallery_data_job.py ft-pairs-sculpt): whole or not at all
        sft = EVAL / "v2_head" / "sculpting_ft"
        if (sft / "DONE").exists():
            for f in sorted(sft.glob("*.json")):
                b = f.read_bytes()
                json.loads(b)
                add(f"head/sculpting_ft/{f.name}", b)
        # ROC curves (experiments/FT/roc_points.py): a family whose job finished it (DONE); each file
        # must parse. Not head/curves_v2 (the mass response), a separate tree.
        for fam in sorted(d for d in (FT / "curves_v2").glob("*") if (d / "DONE").exists()):
            for f in sorted(fam.glob("*.json")):
                b = f.read_bytes()
                json.loads(b)
                add(f"curves_v2/{fam.name}/{f.name}", b)
        # X->bs from scratch (scripts/build_xbx_jobs.py SCRATCH_ROOT_IHEP): finished cells only
        for f in sorted(pathlib.Path("/data/results/xbx/scratch_ihep").glob("*/s*/endpoints.json")):
            if (f.parent / "DONE").exists():
                b = f.read_bytes()
                json.loads(b)
                add(f"xbx_scratch/{f.parent.parent.name}/{f.parent.name}.json", b)
        add("budget/fixed_sweep.json", (json.dumps(fixed_sweep(), indent=1, sort_keys=True) + "\n").encode())
        add("budget/rate_search.json", (json.dumps(rate_search_scan(), indent=1, sort_keys=True) + "\n").encode())
        add("budget/ft_attempts.json", (json.dumps(ft_attempts(), indent=1, sort_keys=True) + "\n").encode())
        add("ft_mlst/manifests.json", (json.dumps(manifests(), indent=1) + "\n").encode())
        add("jc2_test/counts.json", (json.dumps(jc2_test_counts()) + "\n").encode())
        for run in sorted(MTX.glob("mtx-*")):
            if (run / "DONE").exists():
                add(f"compute/{run.name}.json", (json.dumps(compute(run), indent=1) + "\n").encode())
        add("index.json", (json.dumps(dict(sorted(index.items())), indent=1) + "\n").encode())
    sys.stdout.buffer.write(buf.getvalue())
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["rate_search"]:
        print(json.dumps(rate_search_scan(), sort_keys=True))
        sys.exit(0)
    if sys.argv[1:] == ["ft_attempts"]:
        print(json.dumps(ft_attempts(), sort_keys=True))
        sys.exit(0)
    sys.exit(main())
