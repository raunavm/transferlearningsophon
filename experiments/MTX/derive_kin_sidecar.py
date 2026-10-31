#!/usr/bin/env python3
"""The reweighting sidecar of a kinematic-only arm, derived from its parent's (2026-10-31).

weaver's make_weight sidecar (<config>.<md5>.auto.yaml) is the config's options plus
weights.reweight_hists. The histograms depend on the selection, the weights block and the
training files, never on the input features, and a kinematic arm (scripts/build_kin_arms.py)
shares all three with its parent. Its sidecar is therefore the parent's with the parent's
pf_features replaced by its own: the same histograms, hence the same sampling and the same jet
stream (I2/I3), with no second make_weight pass that could differ in its last digits.

Checks before writing: the parent's sidecar is its config plus histograms
(stream_v2.sidecar_mismatch); the derived one is the child's config plus histograms; its
reweight_hists equal the parent's exactly. An existing sidecar is never overwritten with
different content. Writes <makeweight>/<child stem>.<md5>.auto.yaml and a JSON record beside it.

    python3 experiments/MTX/derive_kin_sidecar.py --arm configs/arms/v2/L188_KIN.yaml \\
        --parent configs/arms/L188.yaml --makeweight /data/results/mtx/makeweight
"""
import argparse
import copy
import hashlib
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sidecar_path(cfg: pathlib.Path, makeweight: pathlib.Path) -> pathlib.Path:
    return makeweight / f"{cfg.stem}.{md5(cfg)}.auto.yaml"


def derive(parent_side: dict, child_cfg: dict) -> dict:
    """The parent's sidecar options with the child's pf_features."""
    out = copy.deepcopy(parent_side)
    out["inputs"]["pf_features"]["vars"] = copy.deepcopy(child_cfg["inputs"]["pf_features"]["vars"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, type=pathlib.Path)
    ap.add_argument("--parent", required=True, type=pathlib.Path)
    ap.add_argument("--makeweight", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    import stream_v2 as sv

    p_side = sidecar_path(a.parent, a.makeweight)
    if not p_side.exists():
        raise SystemExit(f"FATAL: no parent sidecar {p_side}")
    bad = sv.sidecar_mismatch(str(a.parent), str(p_side))
    if bad:
        raise SystemExit(f"FATAL: {p_side} is not {a.parent} plus histograms: {bad}")
    parent_side = yaml.safe_load(p_side.read_text())
    child_cfg = yaml.safe_load(a.arm.read_text())
    text = yaml.safe_dump(derive(parent_side, child_cfg), sort_keys=False)

    out = sidecar_path(a.arm, a.makeweight)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text)
    bad = sv.sidecar_mismatch(str(a.arm), str(tmp))
    same = yaml.safe_load(tmp.read_text())["weights"]["reweight_hists"] == parent_side["weights"]["reweight_hists"]
    if bad or not same:
        tmp.unlink()
        raise SystemExit(f"FATAL: derived sidecar mismatch {bad}, histograms equal: {same}")
    if out.exists() and out.read_text() != text:
        tmp.unlink()
        raise SystemExit(f"FATAL: {out} exists with different content; not overwritten")
    tmp.replace(out)
    rec = {"script": "experiments/MTX/derive_kin_sidecar.py", "arm": str(a.arm), "parent": str(a.parent),
           "parent_sidecar": str(p_side), "parent_sidecar_sha256": hashlib.sha256(p_side.read_bytes()).hexdigest(),
           "sidecar": str(out), "sidecar_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "reweight_hists_equal_parent": True, "sidecar_mismatch": []}
    (a.makeweight / f"derived_{a.arm.stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
