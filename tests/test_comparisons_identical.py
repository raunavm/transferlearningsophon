"""Compared networks are trained and evaluated identically (PI 2026-11-14/15). A mismatch is rerun, never described.

Reads the record of every fine-tuning cell the paper reads (mlst_results.load_ft) and fails when two cells of one
comparison (task x training size) differ in any setting of the audit: steps, check interval, schedule, rates, batch,
optimizer settings, validation set (by hash), training set (by hash), data config, network, checkpoint rule, the
training code (by the git blob of each file at the cell's commit, never the commit id), weaver and torch versions. The
pretrained networks must agree among themselves; from scratch only in its learning rates and schedule (ParT's recipe);
the kinematic-only run only in its data config. Below the full top and quark/gluon sets every cell follows the one rule
(fixed steps per training size, no patience) and no best checkpoint sits at the first check or in the last 10% of the
schedule. Fails until the reruns of 2026-11-15 are in; the build fails with it.
"""
import functools
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments/FIGS")]
import mlst_results as M  # noqa: E402

DATA = ROOT / "experiments/FIGS/data/v2"
TRAINING_CODE = ("experiments/FT/ft_weaver.py", "experiments/E1/seed_weaver.py", "experiments/FT/early_stop.py",
                 "experiments/FT/fill_config.py")
KIN = "l188kin-s1"
SCRATCH_MAY_DIFFER = {"lr", "head_lr_mult", "lr_schedule", "patience_from"}     # ParT's from-scratch recipe
KIN_MAY_DIFFER = {"data_config", "code:data_config"}                            # kinematic inputs, by design
# The one accepted change of training code within a comparison (PI 2026-11-21), by both git blobs and nothing else:
# early_stop.py before (mtx-s3.10 to mtx-s3.17) and after the divergence rule (mtx-s3.18, commit 5e49098a), which adds
# only the stop at the first non-finite training loss (a run that trains is stopped, checked and scored the same way).
# Rows mix the two, e.g. the JetClass-II 1e6 rerun at mtx-s3.18 beside its from-scratch cells at mtx-s3.13.
ACCEPTED_CODE_CHANGES = {
    "experiments/FT/early_stop.py": ({"f087e30f1ae576dcdc2f2588f8bd2a612df8fb7c", "e2b74d430bfabce7a464eb6e05872d73b1d303db"},
                                     "the divergence rule: stop at the first non-finite training loss (PI 2026-11-21)"),
}
FIXED_STEPS = {1_000: 20_000, 10_000: 3_000, 100_000: 20_000, 1_000_000: 80_000}
FIXED_CHECK = {1_000: 50, 10_000: 25, 100_000: 50, 1_000_000: 500}


@functools.lru_cache(maxsize=None)
def _blob(commit: str, path: str) -> str:
    r = subprocess.run(["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"absent at {commit[:7]}"


def _accepted(path: str, blob: str) -> str:
    """A blob of one accepted change (ACCEPTED_CODE_CHANGES) reads as that change, so its two versions compare equal;
    any other blob is itself."""
    blobs, reason = ACCEPTED_CODE_CHANGES.get(path, (set(), ""))
    return f"accepted change: {reason}" if blob in blobs else blob


def _settings(rec: dict) -> dict:
    """Every compared setting of one cell record, the code as git blobs at its commit."""
    r, s, e = rec.get("recipe") or {}, rec.get("setup") or {}, rec.get("early_stop") or {}
    out = {k: r.get(k) for k in ("lr", "head_lr_mult", "lr_schedule", "weight_decay", "max_epochs", "steps_per_epoch",
                                 "early_stop_patience")}
    out.update({k: s.get(k) for k in ("batch_size", "samples_per_epoch", "samples_per_epoch_val", "val_rule",
                                      "val_sha256", "subset_sha256", "data_config", "network", "num_classes",
                                      "ckpt_rule", "weaver_version", "torch_version")})
    out.update({"patience": e.get("patience"), "patience_from": e.get("patience_from")})
    commit = s.get("repo_commit")
    files = TRAINING_CODE + tuple(f for f in (s.get("network"),) if f)
    out.update({f"code:{f}": _accepted(f, _blob(commit, f)) if commit else None for f in files})
    out["code:data_config"] = _blob(commit, s["data_config"]) if commit and s.get("data_config") else None
    return out


def _differences(group: dict, allowed=frozenset()) -> dict:
    """{setting: {value: [cells]}} for every setting with more than one value in group {cell: settings}."""
    keys = set().union(*(s.keys() for s in group.values()))
    out = {}
    for k in sorted(keys - set(allowed)):
        vals = {}
        for cell, s in group.items():
            vals.setdefault(json.dumps(s.get(k), sort_keys=True), []).append(cell)
        if len(vals) > 1:
            out[k] = vals
    return out


def _below_full():
    """{(task, n): {init: settings}} of every cell below the full sets the paper reads."""
    tag, per, absent = M.load_ft(DATA)
    out = {}
    for task, cells in per.items():
        for init, by_n in cells.items():
            for n, recs in by_n.items():
                if task in M.FULL_N and n >= M.FULL_N[task]:
                    continue
                for i, rec in enumerate(recs):
                    out.setdefault((task, n), {})[f"{init}#{i + 1}"] = (rec, _settings(rec))
    return out


def test_every_compared_fine_tuning_below_the_full_sets_is_trained_identically():
    paper = {r.removeprefix("mtx-") for rs in M._paper(ROOT).values() for r in rs if r.startswith("mtx-")}
    bad = {}
    for (task, n), cells in sorted(_below_full().items()):
        pre = {c: s for c, (_, s) in cells.items() if c.split("#")[0] in paper}
        scratch = {c: s for c, (_, s) in cells.items() if c.split("#")[0] == M.SCRATCH}
        kin = {c: s for c, (_, s) in cells.items() if c.split("#")[0] == KIN}
        ref = next(iter(pre.values()), {})
        checks = {"pretrained": _differences(pre),
                  "from scratch": _differences(scratch, SCRATCH_MAY_DIFFER),
                  "from scratch vs pretrained": _differences({**{c: s for c, s in scratch.items()}, "pretrained": ref},
                                                             SCRATCH_MAY_DIFFER),
                  "kinematic-only vs pretrained": _differences({**kin, "pretrained": ref}, KIN_MAY_DIFFER) if kin else {}}
        for what, d in checks.items():
            if d:
                bad[f"{task} N={n} {what}"] = d
    assert not bad, json.dumps(bad, indent=1)[:20000]


def test_below_the_full_sets_every_cell_follows_the_one_rule_and_no_best_checkpoint_sits_at_an_edge():
    """Fixed steps and check interval per training size (25 at 1e4, 50 at 1e3 and 1e5, 500 at 1e6), no patience; best checkpoint neither at the first check
    nor in the last 10% of the schedule (PI 2026-11-15)."""
    bad = []
    for (task, n), cells in sorted(_below_full().items()):
        for cell, (rec, s) in cells.items():
            e = rec.get("early_stop") or {}
            ep, best = e.get("max_epochs"), e.get("best_epoch")
            spe = int(s["steps_per_epoch"]) if s.get("steps_per_epoch") else None
            if spe != FIXED_CHECK.get(n) or ep is None or ep * spe != FIXED_STEPS.get(n):
                bad.append(f"{task} N={n} {cell}: {ep} checks x {spe} steps, the rule is {FIXED_STEPS.get(n)} steps "
                           f"in checks of {FIXED_CHECK.get(n)}")
            if e.get("patience") is None or e["patience"] < ep or e.get("epochs_completed") != ep:
                bad.append(f"{task} N={n} {cell}: patience {e.get('patience')}, ran {e.get('epochs_completed')} of {ep}")
            if best is None or best < 1 or best >= 0.9 * ep:
                bad.append(f"{task} N={n} {cell}: best checkpoint at check {best} of {ep}")
    assert not bad, "\n".join(bad[:300])


def test_the_full_top_and_quark_gluon_sets_are_trained_identically():
    """ParT's recipe for every network at the full sets (ft_mlst/manifests.json; the kinematic-only run's own config)."""
    man = json.loads((DATA / "ft_mlst" / "manifests.json").read_text())
    keys = ("lr", "head_lr_mult", "weight_decay", "epochs", "lr_schedule", "batch_size", "samples_per_epoch",
            "val_jets", "val_rule", "ckpt_rule")
    bad = {}
    for leg, n in (("leg_top", 1_200_000), ("leg_qg", 1_600_000)):
        group = {}
        for k, m in man.items():
            parts = k.split("/")
            if parts[1] == leg and parts[3] == f"N{n}" and parts[2] != "scratch-v2":
                s = {key: m.get(key) for key in keys}
                s.update({f"code:{f}": _blob(m["repo_commit"], f) for f in TRAINING_CODE[:2]})
                group[parts[2]] = s
        d = _differences(group)
        if d:
            bad[leg] = d
    assert not bad, json.dumps(bad, indent=1)[:20000]


def test_the_accepted_code_change_is_exactly_the_divergence_rule_and_nothing_else():
    """Both blobs exist at their tags, differ, and are the only versions between mtx-s3.10 and mtx-s3.18; a third
    version of the file would compare as itself and fail the identity test."""
    blobs, reason = ACCEPTED_CODE_CHANGES["experiments/FT/early_stop.py"]
    assert {_blob("mtx-s3.13", "experiments/FT/early_stop.py"), _blob("mtx-s3.18", "experiments/FT/early_stop.py")} == blobs
    assert _accepted("experiments/FT/early_stop.py", "0" * 40) == "0" * 40
    assert _accepted("experiments/FT/ft_weaver.py", next(iter(blobs))) == next(iter(blobs))
    assert set(ACCEPTED_CODE_CHANGES) == {"experiments/FT/early_stop.py"} and "non-finite" in reason


def test_the_edge_rule_binds_where_the_tables_read_the_cells():
    """mlst_results.edge_failures (PI 2026-11-20): the best check first or in the last 10% fails, the Herwig re-scoring
    of a quark/gluon cell is judged by its training record."""
    cells = {("jc2", "a", 1_000_000): [{"early_stop": {"best_epoch": 0, "max_epochs": 160}}],
             ("top", "a", 1_000): [{"early_stop": {"best_epoch": 360, "max_epochs": 400}}],
             ("qg", "a", 1_000): [{"early_stop": {"best_epoch": 1, "max_epochs": 400}}],
             ("qg_herwig", "a", 1_000): [{}]}
    assert [c[:3] for c in M.edge_failures(cells)] == [("jc2", "a", 1_000_000), ("top", "a", 1_000)]
