#!/usr/bin/env python3
"""Bookkeeping for the v2 fine-tuning (audit 2026-09-29: item 4, must-fix 6, 7, 14).

    resolve  which pretrained checkpoint a v2 cell loads, by rule, in the pod,
             from the records experiments/MTX/pretrain_v2.py writes in the run:
               best70   the primary (amendment A14): the first maximum of
                        selection.value within epochs 70-79 (self-supervised:
                        minus the validation loss, so its first minimum), from
                        metrics/epoch-070..079.json, which must agree with
                        best_window_epoch.json; loaded by name as
                        net_epoch-<e>_state.pt.
               bestval  the global best epoch on the fixed validation sample,
                        A14's sensitivity check. The
                        argmax of selection.value over metrics/epoch-EEE.json
                        (the first maximum, as the driver's strict `>` keeps
                        it) must agree with best_epoch.json, and the epoch is
                        loaded by name as net_epoch-<e>_state.pt (the v2
                        specs run --keep-checkpoints window, which keeps the
                        best epoch's and epochs 70-79's state files).
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


def window_best(run_dir: pathlib.Path, n_epochs: int = 80) -> dict:
    """The primary checkpoint of amendment A14: the first maximum of selection.value
    within the last ten epochs (70-79 of 80), recomputed from metrics/epoch-EEE.json and
    checked against the run's own best_window_epoch.json, which a restart could leave
    stale. Higher is better, so for the self-supervised run (value = minus the validation
    loss) it is the first minimum of the loss."""
    window = range(max(0, n_epochs - len(WAVG_EPOCHS)), n_epochs)
    sel = epoch_selection(run_dir)
    missing = [e for e in window if e not in sel]
    if missing:
        raise SystemExit(f"FATAL: {run_dir}/metrics has no record of epochs {missing}")
    top = max(sel[e] for e in window)
    epoch = min(e for e in window if sel[e] == top)
    path = run_dir / "best_window_epoch.json"
    if not path.exists():
        raise SystemExit(f"FATAL: {path} absent; the run did not record its window checkpoint")
    stated = json.loads(path.read_text())
    if stated.get("window") != [window[0], window[-1]]:
        raise SystemExit(f"FATAL: {path} is over epochs {stated.get('window')}, not "
                         f"{window[0]}-{window[-1]}")
    if int(stated["epoch"]) != epoch:
        raise SystemExit(f"FATAL: {path} says epoch {stated['epoch']}, the per-epoch records "
                         f"say {epoch}")
    return {"epoch": epoch, "metric": stated.get("metric"), "value": top,
            "window": [window[0], window[-1]]}


def resolve(run_dir: pathlib.Path, rule: str, n_epochs: int = 80) -> dict:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"FATAL: {run_dir} has no DONE; the run is not complete")
    rec = {"rule": rule, "run_dir": str(run_dir)}
    if rule == "best70":
        w = window_best(run_dir, n_epochs)
        epoch = w.pop("epoch")
        rec.update(w)
    elif rule == "bestval":
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
        raise SystemExit(f"FATAL: unknown checkpoint rule {rule!r} (best70, bestval or wavg)")
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


def expected_cells(spec: str, leg: str) -> set[str]:
    """The cells the generator emitted for one leg, "init/N<N>/s<S>": spec is
    FILE:PREFIX, FILE written by scripts/build_ft_jobs.py --v2
    (experiments/FT/data/ft_v2_expected_cells.json), key PREFIX/leg."""
    path, _, prefix = spec.rpartition(":")
    cells = json.loads(pathlib.Path(path).read_text())
    key = f"{prefix}/{leg}"
    if key not in cells:
        raise SystemExit(f"FATAL: {path} lists no cells for {key}; it has {sorted(cells)}")
    return set(cells[key])


def require_cells(found: set[str], spec: str | None, leg: str) -> None:
    """Fail when a cell the generator emitted for this leg was not read: a cell a
    job left to another job's lock, or never ran, would otherwise just be absent."""
    if not spec:
        return
    missing = sorted(expected_cells(spec, leg) - found)
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} expected {leg} cells were not read, e.g. {missing[:5]}")


def load_table(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text())["files"]


def ref_cell_problem(cell: pathlib.Path, table: dict) -> str | None:
    """None when a reused cell trained on files the sha256 record holds as they were
    when it ran; otherwise what is wrong. Its training subset must be on record with
    the size the cell recorded, and the subset and its directory's val.parquet must
    be on record with a modification time no later than the cell's manifest: a file
    rewritten after the cell ran is not the file it read, whatever its size. A cell
    whose manifest records the files' sha256 (smoke_checks.py manifest, 2026-10-01)
    must match the record exactly."""
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
    ran = m.get("written_utc")
    if not ran:
        return f"{cell}: ft_manifest.json records no written_utc; when it ran is unknown"
    val = str(pathlib.PurePosixPath(sub).parent / "val.parquet")
    for f, key in ((sub, "subset_sha256"), (val, "val_sha256")):
        if f not in table:
            return f"{cell}: {f} is not in the sha256 record"
        if table[f]["mtime_utc"] > ran:
            return (f"{cell}: {f} was modified at {table[f]['mtime_utc']}, after the cell ran "
                    f"({ran})")
        if m.get(key) and m[key] != table[f]["sha256"]:
            return f"{cell}: {f} has sha256 {m[key][:12]} in the cell's manifest, not the record's"
    return None


def cmd_hash(files: list[str], out: pathlib.Path, as_dir: tuple[str, str] | None = None) -> int:
    """The record of `files`, written whole or not at all (a temporary file renamed
    onto `out`). as_dir = (SRC, DST) records a file under SRC by its path under DST:
    the staging job hashes its build before moving it into place."""
    rec = {}
    for f in sorted(files):
        st = os.stat(f)
        key = _recorded_as(f, as_dir)
        rec[key] = {"sha256": sha256_file(f), "bytes": st.st_size,
                    "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")}
        print(rec[key]["sha256"], rec[key]["bytes"], rec[key]["mtime_utc"], key, flush=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"written_utc": dt.datetime.now(dt.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"), "files": rec}, indent=1))
    tmp.replace(out)
    return 0


def _recorded_as(f: str, as_dir: tuple[str, str] | None) -> str:
    if as_dir and (f == as_dir[0] or f.startswith(as_dir[0].rstrip("/") + "/")):
        return as_dir[1].rstrip("/") + f[len(as_dir[0].rstrip("/")):]
    return f


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
    h.add_argument("--as-dir", nargs=2, metavar=("SRC", "DST"),
                   help="record files under SRC by their paths under DST")
    v = sub.add_parser("verify")
    v.add_argument("--table", required=True, type=pathlib.Path)
    v.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "resolve":
        return cmd_resolve(a.run_dir, a.rule, a.link, a.n_epochs)
    if a.cmd == "hash":
        return cmd_hash(a.files, a.out, tuple(a.as_dir) if a.as_dir else None)
    return cmd_verify(a.table, a.files)


if __name__ == "__main__":
    sys.exit(main())
