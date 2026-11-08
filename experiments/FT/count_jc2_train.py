#!/usr/bin/env python3
"""The number of pretraining jets that pass the selection (methods, DesignNTrainSelected): every file of
the pretrain_train role (make_subsets.JC2_ROLES), with stage_jc2_balanced.census_file's selection, the
one every pretraining run applies.

    python3 experiments/FT/count_jc2_train.py --out /data/results/eval/jc2_train_counts.json --workers 8
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    if getattr(sys.modules.get(name), "__file__", None) == spec.origin:
        return sys.modules[name]    # one module object per process, whichever file loaded it first
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m           # the census workers unpickle census_file by module name
    spec.loader.exec_module(m)
    return m


SB = _load("stage_jc2_balanced", "experiments/FT/stage_jc2_balanced.py")


def summary(res: list) -> dict:
    """{n_files, n_selected, per_family, per_label} from census_file results."""
    fam, lab = collections.Counter(), collections.Counter()
    for path, rows, labels in res:
        fam[pathlib.Path(path).name.split("_")[0]] += int(rows.size)
        lab.update(int(x) for x in labels)
    return {"n_files": len(res), "n_selected": int(sum(fam.values())), "per_family": dict(sorted(fam.items())),
            "per_label": {str(k): v for k, v in sorted(lab.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default="/jc2/jet_data")
    a = ap.parse_args(argv)
    files = SB.MS.jc2_role_files("pretrain_train", a.root)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(SB.census_file, files, chunksize=4))
    out = {"script": "experiments/FT/count_jc2_train.py", "role": "pretrain_train",
           "selection": [SB.MS.PT_LO, SB.MS.PT_HI, SB.MS.MSD_LO, SB.MS.MSD_HI], **summary(res)}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(out["n_files"], "files,", out["n_selected"], "selected jets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
