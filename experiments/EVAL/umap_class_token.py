#!/usr/bin/env python3
"""UMAP of the class token for the appendix figures (PI 2026-11-05, in place of the t-SNE of
tsne_class_token.py; one layout for every task, figure round 3). Each figure has the 188- and 17-class
models as rows and, as columns, the frozen class token and the class token after fine-tuning on that
task. Every panel of a figure embeds the same jets, 2,000 of each group, chosen once from the labels, so
panels differ only in the features:

  jc1        JetClass, the ten classes; fine-tuned on 1e6 JetClass jets (the long-budget cells)
  two_prong  class-balanced JetClass-II test jets: X->bb, cc, qq and QCD (the 27 QCD classes as one group)
  leptons    the same jets' X->ee and X->mumu (tsne_class_token.SETS); both fine-tuned on 1e6 JetClass-II jets
  top, qg    top tagging, quark/gluon (Pythia); fine-tuned on the full training set

The features are extract_features.py outputs, written by scripts/build_umap_jobs.py at
<feat-dir>/<task>/<model>/{frozen,ft}/: each task's test jets read with its fine-tuned model's data
config and one loader worker. The four extractions of a task must agree in data config, label and
jet pT, or the script stops. One file per panel, <set>_<model>_{frozen,ft}.json, with the UMAP
settings, the rows' digest and the checkpoint.

    python3 experiments/EVAL/umap_class_token.py --feat-dir /data/results/eval/umap_v2/features \\
        --out-dir /data/results/eval/umap_v2
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO = pathlib.Path(__file__).resolve().parents[2]
MODELS = ("mtx-l188-s1", "mtx-r16q1-s1")
KINDS = ("frozen", "ft")
JC1 = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")   # JetClassI_sophon_noweight labels
N_PER_GROUP = 2000
UMAP = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 1, "standardized": True}


def _tsne():
    spec = importlib.util.spec_from_file_location("tsne_class_token", REPO / "experiments/EVAL/tsne_class_token.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _tsne()
# set -> (task, its groups: (labels drawn as one group, name)); top / quark-gluon: TopReference / EnergyFlowQG labels 0, 1
SETS = {"jc1": ("jc1", tuple(((k,), n) for k, n in enumerate(JC1))),
        "two_prong": ("jc2", T.SETS["two_prong"]), "leptons": ("jc2", T.SETS["leptons"]),
        "top": ("top", (((0,), "QCD"), ((1,), "top"))), "qg": ("qg", (((0,), "gluon"), ((1,), "quark")))}


def embed(x: np.ndarray) -> np.ndarray:
    import umap
    x = StandardScaler().fit_transform(np.asarray(x, np.float64))
    return umap.UMAP(n_neighbors=UMAP["n_neighbors"], min_dist=UMAP["min_dist"], metric=UMAP["metric"],
                     random_state=UMAP["random_state"]).fit_transform(x)


def aligned(dirs: list) -> np.ndarray:
    """The labels of extractions that must hold the same jets in the same order: equal labels and jet pT."""
    y = np.load(dirs[0] / "label188.npy")
    pt = np.load(dirs[0] / "observers.npz")["jet_pt"]
    for d in dirs[1:]:
        if not (np.array_equal(y, np.load(d / "label188.npy")) and np.array_equal(pt, np.load(d / "observers.npz")["jet_pt"])):
            raise SystemExit(f"FATAL: {d} holds other jets than {dirs[0]}")
    return y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True, type=pathlib.Path,
                    help="<task>/<model>/{frozen,ft}/ extract_features.py outputs on each task's test jets")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for task in dict.fromkeys(SETS[s][0] for s in a.sets):
        dirs = {(m, kind): a.feat_dir / task / m / kind for m in MODELS for kind in KINDS}
        man = {k: json.loads((d / "extract_manifest.json").read_text()) for k, d in dirs.items()}
        if len({v["data_config"] for v in man.values()}) != 1:
            raise SystemExit(f"FATAL: the {task} extractions read different data configs")
        y = aligned(list(dirs.values()))
        for name in [s for s in a.sets if SETS[s][0] == task]:
            groups = SETS[name][1]
            rows, which = T.choose(y, groups, N_PER_GROUP, UMAP["random_state"])
            for (m, kind), d in dirs.items():
                xy = embed(np.load(d / "features.npy", mmap_mode="r")[rows])
                path = a.out_dir / f"{name}_{m}_{kind}.json"
                path.write_text(json.dumps({
                    "set": name, "task": task, "model": m, "features": kind, "groups": [g for _, g in groups],
                    "group": which.tolist(), "x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(),
                    "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "n_per_group": N_PER_GROUP,
                    "method": "umap", "umap": UMAP, "checkpoint": man[m, kind]["checkpoint"],
                    "checkpoint_sha256": man[m, kind]["checkpoint_sha256"], "data_config": man[m, kind]["data_config"],
                    "stride": man[m, kind]["stride"], "source": str(d)}))
                print(path, man[m, kind]["checkpoint"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
