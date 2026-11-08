#!/usr/bin/env python3
"""The TEST2M jets share no jet with the JetClass-II fine-tuning training or validation sets (PI 2026-11-08:
"show that these test jets share no jets with the fine-tuning training or validation sets, by file list or
event IDs"). Two independent proofs, both from the stored data:

  FILES  TEST2M is the first 2,000,000 jets of extract_v2.py's stream over the --data-test list of the head
         jobs (scripts/build_head_jobs.py, BF.test2m_list(): the 335 ft_test files of make_subsets.JC2_ROLES).
         The fine-tuning sets (stage_jc2_balanced.py) carry jc2bal_src = file index * 10^7 + row, the file index
         into census.json's ft_heldout + ft_test lists. Checked: the spec lists are one list, it is census
         ft_test exactly, census ft_heldout and ft_test share no file, every training and validation row's
         source file is an ft_heldout file, and (with the repo) the census lists are JC2_ROLES'.
  JETS   No event number survives into either set, so a jet is identified by the exact bits of (jet_label,
         jet_pt, jet_eta, jet_sdmass, genjet_pt, genjet_sdmass), float32 in the release and in head_outputs.npz.
         Overlaps of TEST2M with every training file and the validation file a 10^6..10^3 fine-tuning cell
         read; positive control: the balanced test set (test.parquet, also from the ft_test files) must overlap
         TEST2M, or the identity is broken and the check FAILs.

The cells: every <ft-root>/budget*/leg_jc2b/*/N*/s*/ft_manifest.json; each one's subset and val must be files of
--bal whose sha256, recomputed here, equals the cell's record, so the jets read are the jets it trained on.
Every fine-tuned head output (--ft-head-root/*/manifest.json) must hold TEST2M's 2,000,000 labels (sha256) and
come from one of those cells.

    python3 experiments/EVAL/check_test_disjoint.py --spec experiments/EVAL/k8s/v2/head/job-head-x-l188-s1-raunav.yaml \\
        experiments/EVAL/k8s/v2/head/job-ft-head-x-raunav.yaml --out /tmp/test_disjoint.json
Exit 0 and a final "PASS" line, or exit 1 and "FAIL: <reasons>".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST2M = pathlib.Path("/data/results/eval/v2_head/mtx-l188-s1/test/best70")   # sculpting_ft.REF
N_TEST2M = 2_000_000
SRC_MULT = 10_000_000                                                          # stage_jc2_balanced.SRC_MULT
FLOATS = ("jet_pt", "jet_eta", "jet_sdmass", "genjet_pt", "genjet_sdmass")


def jet_keys(label, cols: dict) -> np.ndarray:
    """One 24-byte key per jet: the label and the five float32 bit patterns."""
    b = [np.ascontiguousarray(cols[c], dtype=np.float32).view(np.uint32).astype(np.uint64) for c in FLOATS]
    k = np.empty((len(b[0]), 3), np.uint64)
    k[:, 0] = (np.asarray(label, np.uint64) << np.uint64(32)) | b[0]
    k[:, 1] = (b[1] << np.uint64(32)) | b[2]
    k[:, 2] = (b[3] << np.uint64(32)) | b[4]
    return k.view("V24").ravel()


def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Rows of `a` whose key is in `b`."""
    return int(np.isin(a, b).sum())


def data_test(spec_text: str) -> list[str]:
    m = re.search(r"--data-test ((?:\S+\.parquet ?)+)", spec_text)
    if not m:
        raise SystemExit("FATAL: no --data-test list in the spec")
    return m.group(1).split()


def file_checks(spec_lists: list, census_files: dict) -> dict:
    """The file-level proof, from the specs' --data-test lists and census.json's role lists (basenames)."""
    name = os.path.basename
    test = {name(f) for f in spec_lists[0]}
    held, ft_test = {name(f) for f in census_files["ft_heldout"]}, {name(f) for f in census_files["ft_test"]}
    return {"n_spec_files": len(spec_lists[0]), "specs_identical": all(s == spec_lists[0] for s in spec_lists),
            "spec_files_unique": len(test) == len(spec_lists[0]), "spec_equals_census_ft_test": test == ft_test,
            "census_roles_disjoint": not (held & ft_test), "spec_files_in_ft_heldout": len(test & held)}


def src_roles(src: np.ndarray, census_files: dict) -> dict:
    """Rows per census role of jc2bal_src ids (file index into ft_heldout + ft_test)."""
    fidx = np.asarray(src, np.int64) // SRC_MULT
    nh, nt = len(census_files["ft_heldout"]), len(census_files["ft_test"])
    return {"ft_heldout": int((fidx < nh).sum()), "ft_test": int(((fidx >= nh) & (fidx < nh + nt)).sum()),
            "outside": int(((fidx < 0) | (fidx >= nh + nt)).sum())}


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def read_set(path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["jet_label", "jc2bal_src", *FLOATS])
    return (jet_keys(t.column("jet_label").to_numpy(), {c: t.column(c).to_numpy() for c in FLOATS}),
            t.column("jc2bal_src").to_numpy())


def repo_roles_check(census_files: dict) -> dict | None:
    """census.json's lists against make_subsets.JC2_ROLES, when the repo is beside this script."""
    p = REPO / "experiments/FT/make_subsets.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("make_subsets", p)
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    return {r: census_files[r] == ms.jc2_role_files(r) for r in ("ft_heldout", "ft_test")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", nargs="+", required=True, type=pathlib.Path,
                    help="the head jobs' specs (their --data-test list is TEST2M's files)")
    ap.add_argument("--test2m", type=pathlib.Path, default=TEST2M)
    ap.add_argument("--bal", type=pathlib.Path, default=pathlib.Path("/data/finetune/jc2_bal"))
    ap.add_argument("--ft-root", type=pathlib.Path, default=pathlib.Path("/data/results/ft_v2"))
    ap.add_argument("--ft-head-root", type=pathlib.Path, default=pathlib.Path("/data/results/eval/v2_ft_head"))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    fail, res = [], {"script": "experiments/EVAL/check_test_disjoint.py",
                     "script_sha256": sha256(__file__), "key": ["jet_label", *FLOATS]}

    census = json.loads((a.bal / "census.json").read_text())["files"]
    res["files"] = file_checks([data_test(s.read_text()) for s in a.spec], census)
    res["files"]["specs"] = [str(s) for s in a.spec]
    res["files"]["census_matches_JC2_ROLES"] = repo_roles_check(census)
    f = res["files"]
    if not (f["specs_identical"] and f["spec_files_unique"] and f["spec_equals_census_ft_test"]
            and f["census_roles_disjoint"] and f["spec_files_in_ft_heldout"] == 0):
        fail.append("file lists")
    if f["census_matches_JC2_ROLES"] is not None and not all(f["census_matches_JC2_ROLES"].values()):
        fail.append("census lists differ from JC2_ROLES")

    man = json.loads((a.test2m / "manifest.json").read_text())
    z = np.load(a.test2m / "head_outputs.npz")
    t2 = jet_keys(z["label188"], {c: z[c] for c in FLOATS})
    lab_sha = hashlib.sha256(z["label188"].tobytes()).hexdigest()
    res["test2m"] = {"dir": str(a.test2m), "n": int(t2.size), "n_unique_keys": int(np.unique(t2).size),
                     "label188_sha256": lab_sha, "n_test_files": man["n_test_files"]}
    if t2.size != N_TEST2M or lab_sha != man["head_outputs"]["label188_sha256"] or man["n_test_files"] != f["n_spec_files"]:
        fail.append("TEST2M record")

    heads = sorted(glob.glob(str(a.ft_head_root / "*" / "manifest.json")))
    hm = {pathlib.Path(h).parent.name: json.loads(pathlib.Path(h).read_text()) for h in heads}
    res["ft_heads"] = {m: {"n": d["head_outputs"]["n"], "label188_sha256_matches": d["head_outputs"]["label188_sha256"] == lab_sha,
                           "n_test_files": d["n_test_files"], "cell": d["run_dir"]} for m, d in hm.items()}
    if not heads or not all(v["label188_sha256_matches"] and v["n"] == N_TEST2M for v in res["ft_heads"].values()):
        fail.append("fine-tuned head outputs not on TEST2M")

    bal_sha = json.loads((a.bal / "manifest.json").read_text())["sha256"]
    cells = sorted(p for p in glob.glob(str(a.ft_root / "budget*" / "leg_jc2b" / "*" / "N*" / "s*" / "ft_manifest.json"))
                   if ".partial" not in p)
    used, res["cells"] = {}, {}
    for c in cells:
        d = json.loads(pathlib.Path(c).read_text())
        rec = {"subset": d["subset"], "val": d["val"]}
        for kind, digest in (("subset", d.get("subset_sha256")), ("val", d.get("val_sha256"))):
            p = pathlib.Path(d[kind])
            if p.parent != a.bal:
                fail.append(f"{c}: {kind} {p} outside {a.bal}")
            used.setdefault(p.name, set()).add(digest)
        res["cells"][str(pathlib.Path(c).parent.relative_to(a.ft_root))] = rec
    res["n_cells"] = len(cells)
    stray = [m for m, v in res["ft_heads"].items()
             if os.path.relpath(v["cell"], a.ft_root) not in res["cells"]]
    if stray:
        fail.append(f"fine-tuned head outputs from cells not checked: {stray}")

    res["sets"] = {}
    for name in sorted(used) + ["test.parquet"]:
        p = a.bal / name
        digest = sha256(p)
        keys, src = read_set(p)
        r = {"rows": int(keys.size), "sha256": digest, "sha256_matches_staging": digest == bal_sha.get(name),
             "sha256_matches_cells": None if name == "test.parquet" else used[name] == {digest},
             "source_roles": src_roles(src, census), "overlap_with_test2m": overlap(keys, t2),
             "test2m_rows_in_set": overlap(t2, keys)}
        res["sets"][name] = r
        print(f"{name}: {r['rows']:,} jets, sources {r['source_roles']}, sha256 ok {r['sha256_matches_staging']}"
              f"/{r['sha256_matches_cells']}, overlap with TEST2M {r['overlap_with_test2m']:,} "
              f"(TEST2M rows {r['test2m_rows_in_set']:,})", flush=True)
        if name == "test.parquet":
            if r["overlap_with_test2m"] == 0 or r["source_roles"]["ft_heldout"]:
                fail.append("positive control: test.parquet does not overlap TEST2M")
        elif r["overlap_with_test2m"] or r["test2m_rows_in_set"] or r["source_roles"]["ft_test"] \
                or r["source_roles"]["outside"] or not (r["sha256_matches_staging"] and r["sha256_matches_cells"]):
            fail.append(name)
    res["verdict"] = "PASS" if not fail else "FAIL"
    res["fail"] = fail
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["files"]), flush=True)
    print(f"TEST2M {res['test2m']['n']:,} jets, {res['test2m']['n_unique_keys']:,} distinct keys; "
          f"{len(heads)} fine-tuned head outputs; {len(cells)} cells", flush=True)
    print("PASS" if not fail else f"FAIL: {fail}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
