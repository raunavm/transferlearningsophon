#!/usr/bin/env python3
"""The ATLAS jet-mass resolution in finer bins of jet pT (PI 2026-11-05: at least eight bins, ATLAS
Fig. 8's density), from the per-jet values atlas_resolution.py saved (<model>_perjet.npz): no refit,
the same jets, the same definition and bootstrap. Each <in-dir>/<model>.json is rewritten into
<out-dir>/<model>.json with its estimators recomputed on PT_EDGES; the inclusive values and the
generator-mass bins must reproduce the source to 1e-6 (same jets, same seeds; the response is
recovered as m_est / m_gen), or the script stops.

    python3 experiments/EVAL/atlas_rebin.py --in-dir /data/results/eval/atlas_res \\
        --out-dir /data/results/eval/atlas_res_fine
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.EVAL import atlas_resolution as AR  # noqa: E402

PT_EDGES = [200, 300, 400, 550, 700, 850, 1000, 1250, 1500, 2000, 2500]     # GeV, ten bins
TOL = 1e-6        # m_est / m_gen recovers the stored response to float rounding


def rebin(perjet: dict) -> dict:
    """{estimator: response_block} on PT_EDGES from one model's per-jet values."""
    gen, lab, pt = perjet["genjet_sdmass"], perjet["label188"], perjet["jet_pt"]
    AR.PT_EDGES = list(PT_EDGES)
    return {k[len("m_"):]: AR.response_block(perjet[k] / gen, lab, pt, gen) for k in perjet if k.startswith("m_")}


def same(a, b) -> bool:
    """Equal to TOL, with None where a bin holds fewer than MIN_JETS jets."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in a.in_dir.glob("*.json"))
    if not srcs:
        raise SystemExit(f"FATAL: no model JSON in {a.in_dir}")
    for src in srcs:
        out = a.out_dir / src.name
        if out.exists():
            print(f"{out} exists")
            continue
        doc = json.loads(src.read_text())
        npz = pathlib.Path(doc["source"]["file"])
        if hashlib.sha256(npz.read_bytes()).hexdigest() != doc["source"]["sha256"]:
            raise SystemExit(f"FATAL: {npz} is not the file {src} was computed from")
        est = rebin(dict(np.load(npz)))
        for e, groups in est.items():
            for g, block in groups.items():
                old = doc["estimators"][e][g]
                if not (same(block["inclusive"], old["inclusive"]) and same(block["vs_mgen"], old["vs_mgen"])):
                    raise SystemExit(f"FATAL: {src.name} {e} {g}: the inclusive or mass-binned values differ")
        doc["estimators"] = est
        doc["rebinned"] = {"from": str(src), "from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                           "pt_edges": PT_EDGES, "script": "experiments/EVAL/atlas_rebin.py",
                           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out)
        print(f"wrote {out}: {len(est)} estimators, pT bins {PT_EDGES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
