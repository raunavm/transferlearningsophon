#!/usr/bin/env python3
"""Staging for X->bb and X->bs (docs/PRESPEC_2026-09.md A5, docs/SOPHON_XBB_XBS_PROTOCOL.md).

Every output is parquet holding the branches configs/xbx/JetClassII_xbx.yaml reads, in
JetClass-II's types, plus xbx_class (1 signal, 0 QCD). Under --root (/data/xbx/v1):

  qcd/eval/        JetClass-II QCD of the test files (make_subsets role ft_test), m_SD in
                   STAGE_WINDOW: the background every signal is evaluated against
  xbb_jc2/eval/    JetClass-II X->bb of the test files, in STAGE_WINDOW: the stand-in signal
                   that runs the chain end to end before the outside samples arrive, and the
                   X->bb sample if they hold none
  <sig>/pool/, <sig>/eval/
                   an outside sample: checked against the config, selected, and split per
                   jet 80 % / 20 % (A5)
  pairs/<sig>/train_<k>.parquet, val.parquet
                   A5's draw (experiments/XBS/xbx.py matched_draw): N_TRAIN signal and N_TRAIN
                   QCD jets with the same count in every (pT, m_SD) cell of the base config's
                   reweighting grid, the QCD from the held-out files (role ft_heldout), which
                   no pretraining run trained or validated on; N_VAL more of each, disjoint,
                   for the probe's choice of regularisation. Every shard mixes the classes in
                   a fixed shuffle, so every model reads the same jets in the same order.

No pretraining jet is staged: QCD comes from the held-out and test files only, and the
stand-in X->bb likewise (pool from ft_heldout, evaluation from ft_test).

    python3 experiments/XBS/stage_xbx.py check --files F...
    python3 experiments/XBS/stage_xbx.py jc2-eval --name qcd|xbb_jc2
    python3 experiments/XBS/stage_xbx.py convert --name xbs --files F...
    python3 experiments/XBS/stage_xbx.py pair --signal xbb_jc2|xbs|xbb

Each output directory ends with manifest.json (rows and sha256 of every file) and DONE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/xbx/JetClassII_xbx.yaml"
ROOT_DEFAULT = "/data/xbx/v1"
JC2 = "/jc2/jet_data"
# m_SD of the staged evaluation jets: holds Sophon's 150-230 GeV and a window about 125 GeV;
# experiments/XBS/fit_xbx.py applies the window itself
STAGE_WINDOW = (50.0, 300.0)
N_TRAIN = 1_250_000      # per class (A5)
N_VAL = 25_600           # per class: 50 batches of 512
SHARD_ROWS = 500_000     # weaver loads a whole file per fetch
EVAL_FRACTION = 0.2      # A5: a per-jet split, 80 % training pool, 20 % evaluation
QCD_LABELS = (161, 188)  # jet_label 161..187
XBB_LABEL = 0
SALT = 20261028
NAMESPACES = {"np", "ak", "math", "numpy", "awkward"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("make_subsets", "experiments/FT/make_subsets.py")
X = _load("xbx", "experiments/XBS/xbx.py")


def config_branches(cfg_path: pathlib.Path = CONFIG) -> list[str]:
    """The stored branches a data config reads: every name in its selection, new variables,
    inputs and observers that is not itself a new variable, a namespace or an attribute,
    without its label (xbx_class, which staging adds)."""
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    new = set(cfg["new_variables"])
    exprs = [cfg["selection"], *map(str, cfg["new_variables"].values()), *cfg["observers"]]
    exprs += [v[0] if isinstance(v, list) else v for g in cfg["inputs"].values() for v in g["vars"]]
    names = set()
    for e in exprs:
        names |= {m.group(1) for m in re.finditer(r"(?<![.\w])([A-Za-z_]\w*)", str(e))}
    return sorted(names - new - NAMESPACES)


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def read_table(path: str, columns: list[str]) -> pa.Table:
    """The columns of one parquet or ROOT file (its single TTree) as an arrow table."""
    if path.endswith(".parquet"):
        return pq.read_table(path, columns=columns)
    import awkward as ak
    import uproot
    with uproot.open(path) as f:
        trees = sorted({k.split(";")[0] for k, v in f.classnames().items() if v == "TTree"})
        if len(trees) != 1:
            raise SystemExit(f"FATAL: {path} has {len(trees)} TTrees: {trees}")
        return ak.to_arrow_table(f[trees[0]].arrays(columns), extensionarray=False)


def file_branches(path: str) -> dict[str, str]:
    """{branch: 'list' | 'scalar' | its type} of one parquet or ROOT file."""
    if path.endswith(".parquet"):
        sch = pq.read_schema(path)
        return {f.name: "list" if _is_list(f.type) else "scalar" if not pa.types.is_nested(f.type)
                else str(f.type) for f in sch}
    import uproot
    with uproot.open(path) as f:
        tree = next(f[k] for k, v in f.classnames().items() if v == "TTree")
        return {k: "list" if ("vector" in b.typename or "[]" in b.typename) else "scalar"
                for k, b in tree.items()}


def check(files: list[str], required: list[str] | None = None) -> dict:
    """Does every file carry every branch the config reads, part_* as per-particle lists and
    the rest as one value per jet? Also what the protocol's open questions need: entries,
    the share passing the study's selection, the m_SD peak, the branches the config does
    not read (event-level ones among them), and per part_* branch the share of particles
    at exactly zero (A5: no zero-fill)."""
    required = config_branches() if required is None else required
    report = {"files": len(files), "required": required, "problems": [], "per_file": {}}
    for f in files:
        br = file_branches(f)
        missing = [b for b in required if b not in br]
        kind = [b for b in required if b in br and br[b] != ("list" if b.startswith("part_") else "scalar")]
        if missing:
            report["problems"].append(f"{f}: missing {missing}")
        if kind:
            report["problems"].append(f"{f}: wrong kind {[(b, br[b]) for b in kind]}")
        report["per_file"][f] = {"extra_branches": sorted(set(br) - set(required))}
    if report["problems"]:
        return report
    t = read_table(files[0], required)
    sel = select(t)
    m = np.asarray(sel.column("jet_sdmass"))
    counts, edges = np.histogram(m, bins=np.arange(20, 505, 5))
    report["first_file"] = {
        "entries": t.num_rows, "selected": sel.num_rows,
        "msd_percentiles_5_50_95": np.percentile(m, [5, 50, 95]).tolist() if m.size else None,
        "msd_mode_bin": edges[int(counts.argmax())].tolist() if m.size else None,
        "in_150_230": int(((m > 150) & (m < 230)).sum()),
        "zero_share": {b: float(np.mean(np.asarray(pc.list_flatten(t.column(b))) == 0))
                       for b in required if b.startswith("part_")}}
    return report


def select(t: pa.Table) -> pa.Table:
    """The study's selection, 200 < pT < 2500 GeV and 20 < m_SD < 500 GeV (MS.select_jc2)."""
    return MS.select_jc2(t)


def window(t: pa.Table, lo: float, hi: float) -> pa.Table:
    m = t.column("jet_sdmass")
    return t.filter(pc.and_(pc.greater(m, lo), pc.less(m, hi)))


def with_class(t: pa.Table, cls: int, columns: list[str]) -> pa.Table:
    t = t.select(columns)
    return t.append_column("xbx_class", pa.array(np.full(t.num_rows, cls, np.int32)))


def conform(t: pa.Table, schema: pa.Schema) -> pa.Table:
    """t in the reference schema (JetClass-II's types); a cast that loses values fails."""
    return t.select(schema.names).cast(schema, safe=True)


def jc2_files(role: str, cls: str) -> list[str]:
    fam = {"qcd": "QCD", "xbb_jc2": "Res2P"}[cls]
    return [f for f in MS.jc2_role_files(role, JC2) if pathlib.Path(f).name.startswith(fam + "_")]


def class_mask(t: pa.Table, cls: str) -> pa.ChunkedArray:
    lab = t.column("jet_label")
    if cls == "qcd":
        return pc.and_(pc.greater_equal(lab, QCD_LABELS[0]), pc.less(lab, QCD_LABELS[1]))
    return pc.equal(lab, XBB_LABEL)


def jc2_eval(name: str, root: pathlib.Path) -> dict:
    out = root / name / "eval"
    if (out / "DONE").exists():
        raise SystemExit(f"FATAL: {out} is complete; staged data is never rebuilt")
    out.mkdir(parents=True, exist_ok=True)
    cols, n = config_branches(), 0
    files = jc2_files("ft_test", name)
    for f in files:
        t = select(read_table(f, cols + ["jet_label"]))
        t = window(t.filter(class_mask(t, name)), *STAGE_WINDOW)
        pq.write_table(with_class(t, int(name != "qcd"), cols), out / f"{pathlib.Path(f).stem}.parquet")
        n += t.num_rows
        print(f"  {pathlib.Path(f).name}: {t.num_rows:,}", flush=True)
    man = {"mode": "jc2-eval", "name": name, "role": "ft_test", "files": files, "rows": n,
           "stage_window": STAGE_WINDOW}
    MS.finish(out, man)
    return man


def convert(name: str, files: list[str], root: pathlib.Path) -> dict:
    """An outside sample: refused unless check() passes; then selected and split per jet,
    the split fixed by the file's name, so it never depends on which other files came."""
    rep = check(files)
    if rep["problems"]:
        raise SystemExit("FATAL: the sample does not carry what the config reads:\n  "
                         + "\n  ".join(rep["problems"]))
    pool, ev = root / name / "pool", root / name / "eval"
    # eval/ is finished last, so its DONE marks the whole conversion; a pool finished by a pod
    # that died before eval/ is redone, to the same bytes (the split is fixed per file)
    if (ev / "DONE").exists():
        raise SystemExit(f"FATAL: {ev} is complete; staged data is never rebuilt")
    for d in (pool, ev):
        (d / "DONE").unlink(missing_ok=True)
        d.mkdir(parents=True, exist_ok=True)
    cols = config_branches()
    ref = reference_schema(cols)
    counts = {}
    for f in sorted(files):
        t = select(read_table(f, cols))
        to_eval = np.random.default_rng([SALT, zlib.crc32(pathlib.Path(f).name.encode())]).random(t.num_rows) < EVAL_FRACTION
        stem = pathlib.Path(f).stem
        for d, m in ((pool, ~to_eval), (ev, to_eval)):
            pq.write_table(conform(with_class(t.filter(pa.array(m)), 1, cols), ref), d / f"{stem}.parquet")
        counts[f] = {"selected": t.num_rows, "eval": int(to_eval.sum())}
        print(f"  {pathlib.Path(f).name}: {t.num_rows:,} selected, {int(to_eval.sum()):,} to evaluation", flush=True)
    for d, part in ((pool, "pool"), (ev, "eval")):
        MS.finish(d, {"mode": "convert", "name": name, "part": part, "files": sorted(files),
                      "eval_fraction": EVAL_FRACTION, "salt": SALT, "per_file": counts, "check": rep})
    return counts


def reference_schema(cols: list[str]) -> pa.Schema:
    """JetClass-II's types for the staged columns, from a held-out QCD file."""
    t = with_class(read_table(jc2_files("ft_heldout", "qcd")[0], cols).slice(0, 1), 0, cols)
    return t.schema


def pool_index(cls: str, root: pathlib.Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(files, file index, row in file, pT, m_SD) of every jet a draw may take: JetClass-II
    held-out jets of the class, or an outside sample's pool."""
    if cls in ("qcd", "xbb_jc2"):
        files = jc2_files("ft_heldout", cls)
        cols = ["jet_pt", "jet_sdmass", "jet_label"]
    else:
        files = sorted(str(p) for p in (root / cls / "pool").glob("*.parquet"))
        if not (root / cls / "pool" / "DONE").exists():
            raise SystemExit(f"FATAL: {root / cls / 'pool'} is not complete: run convert first")
        cols = ["jet_pt", "jet_sdmass"]
    fi, ri, pt, msd = [], [], [], []
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=cols)
        p, m = np.asarray(t.column("jet_pt"), np.float64), np.asarray(t.column("jet_sdmass"), np.float64)
        keep = (p > MS.PT_LO) & (p < MS.PT_HI) & (m > MS.MSD_LO) & (m < MS.MSD_HI)
        if "jet_label" in cols:
            keep &= np.asarray(class_mask(t, cls))
        r = np.flatnonzero(keep)
        fi.append(np.full(r.size, i)); ri.append(r); pt.append(p[r]); msd.append(m[r])
    return files, np.concatenate(fi), np.concatenate(ri), np.concatenate(pt), np.concatenate(msd)


def gather(files: list[str], fi: np.ndarray, ri: np.ndarray, cls: int, cols: list[str],
           ref: pa.Schema) -> pa.Table:
    """The rows (fi, ri), in that order, which is file order (the draw's indices are sorted)."""
    parts = []
    for i in np.unique(fi):
        rows = ri[fi == i]
        parts.append(conform(with_class(pq.read_table(files[i], columns=cols).take(pa.array(rows)), cls, cols), ref))
    return pa.concat_tables(parts) if parts else ref.empty_table()


def pair(sig: str, root: pathlib.Path, n_train: int = N_TRAIN, n_val: int = N_VAL) -> dict:
    out = root / "pairs" / sig
    if (out / "DONE").exists():
        raise SystemExit(f"FATAL: {out} is complete; staged data is never rebuilt")
    out.mkdir(parents=True, exist_ok=True)
    cols = config_branches()
    ref = reference_schema(cols)
    S, B = pool_index(sig, root), pool_index("qcd", root)
    rng = np.random.default_rng([SALT, zlib.crc32(sig.encode())])
    # validation first, then training from what is left: disjoint by construction
    vs, vb, vinfo = X.matched_draw(S[3], S[4], B[3], B[4], n_val, rng)
    left_s, left_b = np.setdiff1d(np.arange(S[1].size), vs), np.setdiff1d(np.arange(B[1].size), vb)
    ts, tb, tinfo = X.matched_draw(S[3][left_s], S[4][left_s], B[3][left_b], B[4][left_b], n_train, rng)
    ts, tb = left_s[ts], left_b[tb]
    n = tinfo["n_per_class"]
    k = max(1, math.ceil(2 * n / SHARD_ROWS))
    shards = []
    for j, (cs, cb) in enumerate(zip(np.array_split(ts, k), np.array_split(tb, k))):
        t = pa.concat_tables([gather(S[0], S[1][cs], S[2][cs], 1, cols, ref),
                              gather(B[0], B[1][cb], B[2][cb], 0, cols, ref)])
        name = f"train_{j:02d}.parquet"
        pq.write_table(t.take(pa.array(rng.permutation(t.num_rows))), out / name)
        shards.append(name)
        print(f"  {name}: {t.num_rows:,} jets", flush=True)
    v = pa.concat_tables([gather(S[0], S[1][vs], S[2][vs], 1, cols, ref),
                          gather(B[0], B[1][vb], B[2][vb], 0, cols, ref)])
    pq.write_table(v.take(pa.array(rng.permutation(v.num_rows))), out / "val.parquet")
    man = {"mode": "pair", "signal": sig, "signal_files": S[0], "qcd_files": B[0],
           "qcd_role": "ft_heldout", "n_train_requested": n_train, "n_val_requested": n_val,
           "train_draw": tinfo, "val_draw": vinfo, "train_shards": shards,
           "head_samples": X.HEAD_SAMPLES, "repetition": X.HEAD_SAMPLES / (2 * n), "salt": SALT}
    MS.finish(out, man)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=ROOT_DEFAULT, type=pathlib.Path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--files", nargs="+", required=True)
    c.add_argument("--out", type=pathlib.Path, default=None)
    e = sub.add_parser("jc2-eval")
    e.add_argument("--name", choices=("qcd", "xbb_jc2"), required=True)
    v = sub.add_parser("convert")
    v.add_argument("--name", required=True)
    v.add_argument("--files", nargs="+", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--signal", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        rep = check(a.files)
        text = json.dumps(rep, indent=1)
        print(text)
        if a.out:
            a.out.write_text(text)
        return 1 if rep["problems"] else 0
    if a.cmd == "jc2-eval":
        jc2_eval(a.name, a.root)
    elif a.cmd == "convert":
        convert(a.name, a.files, a.root)
    else:
        pair(a.signal, a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
