#!/usr/bin/env python3
"""UMAP of the class token for the appendix figures (PI 2026-11-05, in place of the t-SNE of
tsne_class_token.py). Every panel of a figure embeds the same jets, chosen once from the labels, so
panels differ only in the features:

  jc1   JetClass test jets, 1,000 of each of the ten classes; the 188- and 17-class models, frozen
        and after fine-tuning on 1e6 JetClass jets. Both sets of features come from
        extract_features.py run in this job on the same files with one loader worker; the rows must
        agree in label and jet pT, or the script stops.
  jc2   class-balanced JetClass-II test jets, frozen: X->bb, cc, qq and QCD; X->ee and X->mumu
        (tsne_class_token.SETS)
  top, qg  the frozen class token on the benchmark test jets, 2,000 of each class

    python3 experiments/EVAL/umap_class_token.py --jc1-dir /data/results/eval/umap_v1/jc1 \\
        --out-dir /data/results/eval/umap_v1
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO = pathlib.Path(__file__).resolve().parents[2]
BAL = pathlib.Path("/data/results/eval/v2_linprobe_bal")
OLD = pathlib.Path("/data/results/eval/v2_linprobe")
MODELS = ("mtx-l188-s1", "mtx-r16q1-s1")
JC1 = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")   # JetClassI_sophon_noweight labels
BENCH = {"top": ("QCD", "top"), "qg": ("gluon", "quark")}     # TopReference / EnergyFlowQG labels 0, 1
N_JC1, N_OTHER = 1000, 2000
N_NEIGHBORS, MIN_DIST, SEED = 15, 0.1, 1


def _tsne():
    spec = importlib.util.spec_from_file_location("tsne_class_token", REPO / "experiments/EVAL/tsne_class_token.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def embed(x: np.ndarray) -> np.ndarray:
    import umap
    x = StandardScaler().fit_transform(np.asarray(x, np.float64))
    return umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=SEED).fit_transform(x)


def aligned(dirs: list) -> np.ndarray:
    """The labels of extractions that must hold the same jets in the same order: equal labels and jet pT."""
    y = np.load(dirs[0] / "label188.npy")
    pt = np.load(dirs[0] / "observers.npz")["jet_pt"]
    for d in dirs[1:]:
        if not (np.array_equal(y, np.load(d / "label188.npy")) and np.array_equal(pt, np.load(d / "observers.npz")["jet_pt"])):
            raise SystemExit(f"FATAL: {d} holds other jets than {dirs[0]}")
    return y


def write(out: pathlib.Path, name: str, panel: str, groups: list, which: np.ndarray, rows: np.ndarray, xy, extra: dict):
    path = out / f"{name}_{panel}.json"
    path.write_text(json.dumps({
        "set": name, "panel": panel, "groups": groups, "group": which.tolist(),
        "x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(),
        "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "method": "umap",
        "n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST, "seed": SEED, **extra}))
    print(path, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--jc1-dir", type=pathlib.Path, default=None,
                    help="<model>/{frozen,ft}/ extract_features.py outputs on the same JetClass files")
    ap.add_argument("--sets", nargs="+", default=["jc1", "jc2", "top", "qg"])
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    T = _tsne()
    if "jc1" in a.sets:
        dirs = {(m, kind): a.jc1_dir / m / kind for m in MODELS for kind in ("frozen", "ft")}
        y = aligned(list(dirs.values()))
        rows, which = T.choose(y, tuple(((k,), n) for k, n in enumerate(JC1)), N_JC1, SEED)
        for (m, kind), d in dirs.items():
            f = np.load(d / "features.npy", mmap_mode="r")
            write(a.out_dir, "jc1", f"{m}_{kind}", list(JC1), which, rows, embed(f[rows]),
                  {"model": m, "features": kind, "source": str(d)})
    if "jc2" in a.sets:
        y = np.load(BAL / MODELS[0] / "jc2bal" / "test" / "best70" / "label188.npy")
        for m in MODELS[1:]:
            if not np.array_equal(y, np.load(BAL / m / "jc2bal" / "test" / "best70" / "label188.npy")):
                raise SystemExit(f"FATAL: {m} has other test jets")
        for name, groups in T.SETS.items():
            rows, which = T.choose(y, groups, T.N_PER_GROUP, SEED)
            for m in MODELS:
                f = np.load(BAL / m / "jc2bal" / "test" / "best70" / "features.npy", mmap_mode="r")
                write(a.out_dir, name, m, [g for _, g in groups], which, rows, embed(f[rows]), {"model": m})
    for ds in [s for s in a.sets if s in BENCH]:
        d = {m: OLD / m / ds / "test" / "best70" for m in MODELS}
        y = aligned(list(d.values())) if (d[MODELS[0]] / "observers.npz").exists() else np.load(d[MODELS[0]] / "label188.npy")
        rows, which = T.choose(y, (((0,), BENCH[ds][0]), ((1,), BENCH[ds][1])), N_OTHER, SEED)
        for m in MODELS:
            if not np.array_equal(y, np.load(d[m] / "label188.npy")):
                raise SystemExit(f"FATAL: {m} has other {ds} test jets")
            f = np.load(d[m] / "features.npy", mmap_mode="r")
            write(a.out_dir, ds, m, list(BENCH[ds]), which, rows, embed(f[rows]), {"model": m})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
