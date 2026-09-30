#!/usr/bin/env python3
"""Bookkeeping for the v2 fine-tuning (audit 2026-09-29: item 4, must-fix 6, 7, 14).

    resolve  which pretrained checkpoint a v2 cell loads, by rule, in the pod,
             from the records experiments/MTX/pretrain_v2.py writes in the run:
               bestval  the best epoch on the fixed validation sample. The
                        argmax of selection.value over metrics/epoch-EEE.json
                        (the first maximum, as the driver's strict `>` keeps
                        it) must agree with best_epoch.json, and the epoch is
                        loaded by name as net_epoch-<e>_state.pt (the v2
                        specs keep every epoch, --keep-checkpoints all).
               wavg     the weight average of epochs 70-79, the robustness
                        check (audit 2026-09-29; the output-layer diagnostic,
                        experiments/FIGS/data/head_epoch_diag): WAVG_STATE,
                        with WAVG_JSON = {"inputs": {"70": <sha256 of
                        net_epoch-70_state.pt>, ..., "79": ...}, "sha256":
                        <sha256 of WAVG_STATE>}. Each input's sha256 is checked
                        against the epoch file in the run, and the average's
                        against the file itself.
             The run must be complete (its DONE).
             It links the file to --link and writes <link>.json with the rule,
             the epoch and the sha256, which each cell copies beside its result.
    hash     sha256, size and mtime of files, as the record a later job checks.
    verify   refuse to start unless every file matches that record.

Library use by the read-outs: weaver_args() reads what weaver actually ran from
its own argument dump in train.log (the v1 manifests wrote steps_per_epoch as
N/512, which is 1 at N = 1e3 against the 19 steps that ran), and
ref_cell_problem() checks that a reused cell trained on a recorded subset.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ARG = re.compile(r"^\s*- \('(\w+)', (.*)\)\s*$")
WAVG_STATE, WAVG_JSON, WAVG_EPOCHS = "net_wavg70-79_state.pt", "net_wavg70-79.json", range(70, 80)
RUN_ARGS = ("batch_size", "num_epochs", "samples_per_epoch", "samples_per_epoch_val",
            "steps_per_epoch", "steps_per_epoch_val", "start_lr", "lr_scheduler", "optimizer")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_selection(run_dir: pathlib.Path) -> dict[int, float]:
    """selection.value per epoch from metrics/epoch-EEE.json (higher is better)."""
    out = {}
    for p in sorted((run_dir / "metrics").glob("epoch-*.json")):
        d = json.loads(p.read_text())
        out[int(d["epoch"])] = float(d["selection"]["value"])
    return out


def resolve(run_dir: pathlib.Path, rule: str, n_epochs: int = 80) -> dict:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"FATAL: {run_dir} has no DONE; the run is not complete")
    rec = {"rule": rule, "run_dir": str(run_dir)}
    if rule == "bestval":
        sel = epoch_selection(run_dir)
        missing = sorted(set(range(n_epochs)) - set(sel))
        if missing:
            raise SystemExit(f"FATAL: {run_dir}/metrics has no record of epochs {missing[:10]}"
                             f"{' ...' if len(missing) > 10 else ''}")
        best = max(sel[e] for e in range(n_epochs))
        epoch = min(e for e in range(n_epochs) if sel[e] == best)
        stated = json.loads((run_dir / "best_epoch.json").read_text())
        if int(stated["epoch"]) != epoch:
            raise SystemExit(f"FATAL: {run_dir}/best_epoch.json says epoch {stated['epoch']}, "
                             f"the per-epoch records say {epoch}")
        rec.update(metric=stated.get("metric"), value=best)
    elif rule == "wavg":
        return resolve_wavg(run_dir, rec)
    else:
        raise SystemExit(f"FATAL: unknown checkpoint rule {rule!r} (bestval or wavg)")
    path = run_dir / f"net_epoch-{epoch}_state.pt"
    if not path.exists():
        raise SystemExit(f"FATAL: rule {rule} selects epoch {epoch}, and {path} is not kept")
    rec.update(epoch=epoch, path=str(path), sha256=sha256_file(path))
    return rec


def resolve_wavg(run_dir: pathlib.Path, rec: dict) -> dict:
    """The weight average of epochs 70-79 and proof of what was averaged."""
    path, meta = run_dir / WAVG_STATE, run_dir / WAVG_JSON
    for f in (path, meta):
        if not f.exists():
            raise SystemExit(f"FATAL: {f} absent; the weight average was not written")
    m = json.loads(meta.read_text())
    inputs = {int(e): sha for e, sha in m["inputs"].items()}
    if sorted(inputs) != list(WAVG_EPOCHS):
        raise SystemExit(f"FATAL: {meta} averages epochs {sorted(inputs)}, not "
                         f"{WAVG_EPOCHS.start}-{WAVG_EPOCHS.stop - 1}")
    for e, sha in sorted(inputs.items()):
        f = run_dir / f"net_epoch-{e}_state.pt"
        if not f.exists() or sha256_file(f) != sha:
            raise SystemExit(f"FATAL: {meta}: input epoch {e} is not {f} as it is on disk")
    got = sha256_file(path)
    if m["sha256"] != got:
        raise SystemExit(f"FATAL: {path} is not the file {meta} records")
    rec.update(epoch=f"wavg{WAVG_EPOCHS.start}-{WAVG_EPOCHS.stop - 1}", path=str(path),
               sha256=got, inputs={str(e): s for e, s in sorted(inputs.items())})
    return rec


def weaver_args(train_log: pathlib.Path) -> dict:
    """The run's own values of RUN_ARGS, from weaver's argument dump (the first one)."""
    out = {}
    for line in train_log.read_text(errors="replace").splitlines():
        m = ARG.match(ANSI.sub("", line.split("INFO: ", 1)[-1]))
        if m and m.group(1) in RUN_ARGS and m.group(1) not in out:
            try:
                out[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                out[m.group(1)] = m.group(2)
    return out


def load_table(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text())["files"]


def ref_cell_problem(cell: pathlib.Path, table: dict) -> str | None:
    """None when a reused cell's training subset is a file on record with the size
    the cell recorded when it ran; otherwise what is wrong."""
    man = cell / "ft_manifest.json"
    if not man.exists():
        return f"{cell}: no ft_manifest.json"
    m = json.loads(man.read_text())
    sub = m.get("subset")
    if sub not in table:
        return f"{cell}: training subset {sub} is not in the sha256 record"
    if str(m.get("subset_bytes")) != str(table[sub]["bytes"]):
        return (f"{cell}: trained on {m.get('subset_bytes')} bytes of {sub}, "
                f"the record holds {table[sub]['bytes']}")
    return None


def cmd_hash(files: list[str], out: pathlib.Path) -> int:
    rec = {}
    for f in sorted(files):
        st = os.stat(f)
        rec[f] = {"sha256": sha256_file(f), "bytes": st.st_size,
                  "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ")}
        print(rec[f]["sha256"], rec[f]["bytes"], rec[f]["mtime_utc"], f, flush=True)
    out.write_text(json.dumps({"written_utc": dt.datetime.now(dt.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"), "files": rec}, indent=1))
    return 0


def cmd_verify(table_path: pathlib.Path, files: list[str]) -> int:
    table = load_table(table_path)
    bad = []
    for f in files:
        if f not in table:
            bad.append(f"{f}: not in {table_path}")
        elif not os.path.exists(f):
            bad.append(f"{f}: absent")
        elif sha256_file(f) != table[f]["sha256"]:
            bad.append(f"{f}: sha256 differs from the record")
    for b in bad:
        print("FATAL:", b, flush=True)
    if not bad:
        print(f"verified {len(files)} files against {table_path}", flush=True)
    return 1 if bad else 0


def cmd_resolve(run_dir: pathlib.Path, rule: str, link: pathlib.Path, n_epochs: int) -> int:
    rec = resolve(run_dir, rule, n_epochs)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rec["path"])
    pathlib.Path(f"{link}.json").write_text(json.dumps(rec, indent=1))
    print(f"{run_dir.name}: rule {rule} -> epoch {rec['epoch']} ({rec['path']})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--run-dir", required=True, type=pathlib.Path)
    r.add_argument("--rule", required=True)
    r.add_argument("--link", required=True, type=pathlib.Path)
    r.add_argument("--n-epochs", type=int, default=80)
    h = sub.add_parser("hash")
    h.add_argument("--out", required=True, type=pathlib.Path)
    h.add_argument("--files", nargs="+", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--table", required=True, type=pathlib.Path)
    v.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "resolve":
        return cmd_resolve(a.run_dir, a.rule, a.link, a.n_epochs)
    if a.cmd == "hash":
        return cmd_hash(a.files, a.out)
    return cmd_verify(a.table, a.files)


if __name__ == "__main__":
    sys.exit(main())
