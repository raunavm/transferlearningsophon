#!/usr/bin/env python3
"""t-SNE of the frozen class token on the class-balanced JetClass-II test jets, for the appendix figure
(SuperJEPA Fig. 3, JetParticle-JEPA Fig. 2; PI 2026-11-05): the same jets for every model, chosen once from
the labels, so the panels differ only in the features.

    python3 experiments/EVAL/tsne_class_token.py --out-dir /data/results/eval/tsne_v1
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path("/data/results/eval/v2_linprobe_bal")
MODELS = ("mtx-l188-s1", "mtx-r16q1-s1")
QCD = tuple(range(161, 188))       # configs/labelmaps/rung_label_maps.v1.csv: the 27 QCD classes
# (jet labels drawn as one group, its name)
SETS = {"two_prong": (((0,), "bb"), ((1,), "cc"), ((3,), "qq"), (QCD, "QCD")),
        "leptons": (((10,), "ee"), ((11,), "mm"))}
N_PER_GROUP, PERPLEXITY, SEED = 2000, 30, 1


def choose(y: np.ndarray, groups: tuple, n: int, seed: int) -> tuple:
    """(rows, group index of each row): n jets of each group, the same for a given label array."""
    rng = np.random.default_rng(seed)
    rows, which = [], []
    for k, (labels, _) in enumerate(groups):
        pool = np.flatnonzero(np.isin(y, labels))
        if pool.size < n:
            raise SystemExit(f"FATAL: {pool.size} jets of {labels}, fewer than {n}")
        rows.append(np.sort(rng.choice(pool, n, replace=False)))
        which.append(np.full(n, k))
    return np.concatenate(rows), np.concatenate(which)


def embed(x: np.ndarray, seed: int) -> tuple:
    x = StandardScaler().fit_transform(np.asarray(x, np.float64))
    t = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", learning_rate="auto", random_state=seed)
    return t.fit_transform(x), float(t.kl_divergence_)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    labels = {m: np.load(ROOT / m / "jc2bal" / "test" / "best70" / "label188.npy") for m in MODELS}
    y = labels[MODELS[0]]
    for m, other in labels.items():
        if not np.array_equal(y, other):
            raise SystemExit(f"FATAL: {m} has other test jets")
    for name, groups in SETS.items():
        rows, which = choose(y, groups, N_PER_GROUP, SEED)
        digest = hashlib.sha256(rows.tobytes()).hexdigest()
        for m in MODELS:
            f = np.load(ROOT / m / "jc2bal" / "test" / "best70" / "features.npy", mmap_mode="r")
            xy, kl = embed(f[rows], SEED)
            path = a.out_dir / f"{name}_{m}.json"
            path.write_text(json.dumps({
                "model": m, "set": name, "groups": [g for _, g in groups], "group": which.tolist(),
                "x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(), "kl_divergence": kl,
                "rows_sha256": digest, "n_per_group": N_PER_GROUP, "perplexity": PERPLEXITY, "seed": SEED}))
            print(path, f"KL {kl:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
