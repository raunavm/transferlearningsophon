"""A job spec must not be launched at a tag that predates a fix it depends on.

THIS DEFECT HAS LANDED THREE TIMES and cost a full rerun each time:
  - job-eval-anchors-raunav.yaml ran at mtx-s1.15, three commits before the
    unweighted-comparison fix, and "applying the committed spec reproduced the
    pre-fix number" (experiments/RUNS.csv, audit-2-anchors, 2026-09-08).
  - job-eval-labelrec-raunav.yaml ran at mtx-s1.31, one commit before 759bbad
    made the linear and MLP probes fit at equal training size.
  - job-probe-physics-raunav.yaml ran once at mtx-s1.18, was then repinned to
    mtx-s1.20 -- still twelve tags stale, missing both the MIN_PER_CLASS-on-window
    fix and the MLP convergence record -- and that pin was caught before a second
    run only because someone looked.

Nothing errors when this happens. The job clones, runs, and writes a plausible
number computed by old code.

WHAT THIS DOES NOT DO: bump historical pins. A spec that has already run is a
provenance record of what actually ran, and rewriting its tag would falsify the
ledger. So the rule binds only specs with NO run record -- the ones whose next
launch is still ahead of them. A run record is a ledger row naming the spec
whose status says the job was created from it (LAUNCHED); a row whose status is
exactly `retired` takes a spec that never ran out of service. Each exemption is
a skip carrying its row, never a silent pass. A row written when a spec was only
queued, built, held, deleted before it ran or superseded exempts nothing, and
neither does the launch of another spec whose name extends this one's
(_exemption).

SCOPE: every spec under experiments/*/k8s, subdirectories included (the v2
pretraining grid lives in experiments/MTX/k8s/v2/grid). The tag is a literal
`--branch` or, for `--branch "${REPO_REF}"`, the REPO_REF value the spec itself
sets: REPO_REF is a literal env entry in the spec, so a stale one clones stale
code exactly as a stale literal does (ledger row ft-subsets-bench-stale-tag). A
dependency is any repository .py file the spec names on a code line (a script
it runs, with or without interpreter flags, the --network-config model, a file
an inline snippet loads), a configs/ file (data config, label map, grid) it
names, and every repository module those load, followed recursively through
imports and loads by path in any directory (_resolve_loads): pretrain_v2.py
imports stream_v2.py, and the grid's --network-config wrapper loads
experiments/E1/ParT_sophon_arch_10c.py, so a fix to either alone must still
reach the grid. It also includes every tracked file other than code that those
spec-named files, or the modules they load directly, name by a string literal
(_data_files): extract_v2.py's default --data-config and the label map probe.py
reads are read at fixed paths, and a change to either is a change to the run.
Paths built at run time from shell variables or command-line arguments cannot
be resolved here and are skipped.
"""
import ast
import csv
import functools
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
LEDGER = REPO / "experiments" / "RUNS.csv"
SPECS = "experiments/*/k8s/**/*.yaml"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _code_lines(text: str) -> list[str]:
    """The spec's lines minus comments.

    Spec headers cite their generator (`GENERATED from scripts/build_arm_jobs.py`)
    and explain `--branch`; those citations are not part of the run.
    """
    return [ln.strip() for ln in text.splitlines()
            if not ln.strip().startswith("#")]


def _executed_scripts(text: str) -> set[str]:
    """Scripts on an actual `python3 [flags] ...` line, never ones named in a comment."""
    found = set()
    for line in _code_lines(text):
        found.update(re.findall(
            r"python3(?: +-\w+)* +((?:experiments|scripts)/[\w/]+\.py)", line))
    return found


def _module_files(base: pathlib.Path, dotted: str) -> list[pathlib.Path]:
    """What `import <dotted>` runs when found under `base`: every package
    __init__.py on the way down and the module itself. Given `pkg.name` for
    `from pkg import name`, it also yields pkg/name.py, which exists only when
    `name` is a submodule (`from src.stats import paired`)."""
    parts = [p for p in dotted.split(".") if p]
    out = []
    for k in range(1, len(parts) + 1):
        p = base.joinpath(*parts[:k])
        out += [p.with_suffix(".py"), p / "__init__.py"]
    return out


def _is_str(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _path_literals(tree: ast.AST) -> list[list[str]]:
    """Each path a module spells out where it is used as a path, as candidate
    spellings, longest first.

    Used as a path means: an operand of `/` (`REPO / "experiments" / "MTX" /
    "mpm.py"`, whose literal components are joined), a call argument
    (`os.path.join(d, "E1", "ParT_sophon_arch_10c.py")`, likewise joined with
    the literal arguments before it; `_load("probe", "experiments/EVAL/probe.py")`),
    or a value bound to a plain name (`ARCH = ...`). A literal stored into a
    record -- `manifest["driver"] = "experiments/MTX/pretrain_v2.py"`, a dict
    value, a docstring, part of a longer message -- is not a load and is not
    returned. `f"{name}.py"` stands for every one-word string literal of the
    module (`_load("discriminants")` with `HERE / f"{name}.py"`)."""
    parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    words = [n.value for n in ast.walk(tree) if _is_str(n) and re.fullmatch(r"\w+", n.value)]
    out = []
    for node in ast.walk(tree):
        if _is_str(node):
            names = [node.value]
        elif (isinstance(node, ast.JoinedStr) and len(node.values) == 2
              and isinstance(node.values[0], ast.FormattedValue)
              and _is_str(node.values[1]) and node.values[1].value == ".py"):
            names = [w + ".py" for w in words]
        else:
            continue
        up, before = parent.get(node), []
        if isinstance(up, ast.BinOp) and isinstance(up.op, ast.Div) and up.right is node:
            outer = parent.get(up)
            if (isinstance(outer, ast.BinOp) and isinstance(outer.op, ast.Div)
                    and outer.left is up):
                continue                # a middle component: the last one spells the path
            left = up.left
            while (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div)
                   and _is_str(left.right)):
                before.insert(0, left.right.value)
                left = left.left
        elif isinstance(up, ast.Call) and any(a is node for a in up.args):
            i = next(i for i, a in enumerate(up.args) if a is node)
            for a in reversed(up.args[:i]):
                if not _is_str(a):
                    break
                before.insert(0, a.value)
        elif not (isinstance(up, ast.keyword)
                  or isinstance(up, ast.Assign) and up.value is node
                  and all(isinstance(t, ast.Name) for t in up.targets)
                  or isinstance(up, ast.AnnAssign) and up.value is node
                  and isinstance(up.target, ast.Name)):
            continue
        for name in names:
            out.append([s for k in range(len(before) + 1)
                        if re.fullmatch(r"\w[\w.-]*(?:/[\w.-]+)*",
                                        s := "/".join(before[k:] + [name]))])
    return out


def _lookup(spelling: str, here: pathlib.Path) -> pathlib.Path | None:
    """The repository path a module in `here` means by `spelling`: one with a
    directory is looked up from `here` and each parent up to the repository root
    (`REPO / "configs/..."`, `dirname(dirname(__file__)), "E1", ...`); a bare
    file name only beside the module."""
    roots = [here, *here.parents][:len(here.relative_to(REPO).parts) + 1]
    return next((r / spelling for r in (roots if "/" in spelling else [here])
                 if (r / spelling).exists()), None)


@functools.cache
def _tracked() -> frozenset[str]:
    return frozenset(_git("ls-files").splitlines())


@functools.cache
def _data_files(path: str) -> frozenset[str]:
    """Tracked files other than code that the repository module `path` names."""
    file = REPO / path
    return _resolve_data(file.read_text(), file.parent)


def _resolve_data(src: str, here: pathlib.Path) -> frozenset[str]:
    """Tracked repository files, other than .py, that Python source `src` in
    `here` names by a string literal: a whole path (`default=str(REPO /
    "configs/data/JetClassII_massreg.yaml")`), a `/` chain or join arguments
    (`REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"`), or a path
    literal anywhere else -- a dict value, a function default. A file read at a
    fixed path is read by every run, so any literal that is wholly a path counts;
    prose that mentions one ("see configs/x.yaml") is not wholly a path. A bare
    file name counts only in a load position and beside the module, as for code
    (_lookup). A path the module only writes counts too (build_usecase_survival.py
    names its output, so the AOJ specs that load it for read_map() depend on
    usecase_survival.v1.json): that errs toward a needless repin, never a miss."""
    tree = ast.parse(src)
    spellings = _path_literals(tree) + [
        [n.value] for n in ast.walk(tree)
        if _is_str(n) and re.fullmatch(r"\w[\w.-]*(?:/[\w.-]+)+", n.value)]
    found = set()
    for candidates in spellings:
        for s in candidates:
            hit = _lookup(s, here)
            if hit is not None:
                rel = os.path.normpath(hit.relative_to(REPO))
                if hit.is_file() and rel in _tracked() and not rel.endswith(".py"):
                    found.add(rel)
                break
    return frozenset(found)


@functools.cache
def _loads(path: str) -> frozenset[str]:
    """Repository .py files that the repository file `path` loads."""
    file = REPO / path
    return _resolve_loads(file.read_text(), file.parent) - {path}


def _resolve_loads(src: str, here: pathlib.Path) -> frozenset[str]:
    """Repository .py files that Python source `src`, sitting in `here`, loads.

    Imports: `src.`/`experiments.`/`scripts.` modules from the repository root,
    with each package's __init__.py and `from pkg import submodule`; relative
    imports; a bare `import x` from the file's own directory (a script's
    directory is on sys.path, so pretrain_v2.py's `import stream_v2` is
    experiments/MTX/stream_v2.py) or from any repository directory the file
    names as a path (head_epoch_diag.py puts experiments/FIGS on sys.path and
    imports style).
    Files loaded by path (spec_from_file_location, runpy): every path literal
    in a load position, see _path_literals. A spelling with a directory is
    looked up from the file's own directory and each parent up to the
    repository root (`dirname(dirname(__file__)), "E1", ...` from
    experiments/MTX is experiments/E1/...); a bare file name only beside the
    file.
    Paths assembled at run time from variables -- a shell variable, an
    argument read from the command line -- cannot be resolved here."""
    tree = ast.parse(src)
    found, dirs = [], [here]
    for spellings in _path_literals(tree):
        for s in spellings:
            hit = _lookup(s, here)
            if hit is not None:
                (dirs if hit.is_dir() else found).append(hit)
                break
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods, level = [a.name for a in node.names], 0
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] + [f"{node.module or ''}.{a.name}" for a in node.names]
            level = node.level
        else:
            continue
        for mod in mods:
            if level:
                bases = [here.parents[level - 2] if level > 1 else here]
            elif mod.split(".")[0] in ("src", "experiments", "scripts"):
                bases = [REPO]
            else:
                bases = dirs
            for base in bases:
                found += _module_files(base, mod)
    return frozenset(str(p.relative_to(REPO)) for p in found
                     if p.suffix == ".py" and p.is_file())


def _dependencies(text: str) -> set[str]:
    """Repository files the run reads: every repository .py file the spec names
    on a code line (scripts it runs, with or without interpreter flags, the
    --network-config model, files an inline `python3 -c` snippet loads by path or
    a `grep` checks), every configs/ file and tracked data file it names (an
    input such as `--fit experiments/FIGS/data/.../results.json`), modules an
    inline snippet imports from a directory it puts on sys.path, and every
    repository module any of those loads, recursively (see _loads). Plus the
    tracked data and config files that each of those spec-named files, and each
    module it loads directly (one level), names by a literal (see _data_files)."""
    found = set()
    code = "\n".join(_code_lines(text))
    found.update(re.findall(r"\b((?:experiments|scripts|src)/[\w/.-]+\.py)\b", code))
    found.update(re.findall(r"\b(configs/[\w/.-]+\.(?:yaml|json|csv))\b", code))
    found.update(p for p in re.findall(r"\b((?:experiments|configs)/[\w/.-]+\.\w+)\b", code)
                 if p in _tracked())
    for d in re.findall(r"""sys\.path\.insert\( *0, *["']([\w/]+)["'] *\)""", code):
        for mod in re.findall(r"\b(?:from|import) +([\w.]+)", code):
            found.update(str(p.relative_to(REPO)) for p in _module_files(REPO / d, mod)
                         if p.is_file())
    named = [d for d in found if d.endswith(".py") and (REPO / d).is_file()]
    for script in named:
        for module in {script} | _loads(script):
            found |= _data_files(module)
    todo = list(named)
    while todo:
        new = _loads(todo.pop()) - found
        found |= new
        todo += new
    return found


def _tag(text: str) -> str | None:
    """The tag the job clones, or None when the spec pins none."""
    m = re.search(r'--branch +\\?"?([^"\s\\]+)', "\n".join(_code_lines(text)))
    if not m:
        return None
    if m.group(1) in ("${REPO_REF}", "$REPO_REF"):
        ref = re.search(r'name: *REPO_REF *\n(?:[ \t]*#.*\n)*[ \t]*value: *"?([^"\s]+)',
                        text)
        return ref.group(1) if ref else None
    return m.group(1)


@functools.cache
def _tag_exists(tag: str) -> bool:
    return _git("rev-parse", "--verify", f"{tag}^{{commit}}") != ""


@functools.cache
def _last_change(path: str) -> str:
    return _git("rev-list", "-1", "HEAD", "--", path)


@functools.cache
def _tag_contains(tag: str, commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, tag],
        capture_output=True).returncode == 0


# Ledger statuses that record a job created from the spec, so its pin is history
# (`complete-but-undersized` is `complete`). Left out, because they do not say
# the spec ran: queued, built, ready, pending, blocked, held, decided, note --
# and superseded and deleted, which the ledger uses both for runs and for specs
# replaced or deleted before they ran (mtx-r16q1-s1 "superseded ... NEVER RAN").
LAUNCHED = frozenset({"launched", "relaunched", "running", "complete", "done", "crashed",
                      "failed", "killed", "evicted", "interrupted", "partial", "recovered"})
RETIRED = "retired"


@functools.cache
def _ledger() -> tuple[tuple[str, str, str, str], ...]:
    """(spec stem, status, run_id, reason) for every ledger row, once per stem it
    names in the run_id and manifest_path COLUMNS.

    Deliberately not a substring search of the whole file. The `reason` column is
    prose and routinely names the specs a row is about -- the row recording this
    very guard names four of them -- so matching free text would exempt exactly
    the specs the guard exists to check.
    """
    out = []
    with LEDGER.open() as f:
        for row in csv.DictReader(f):
            rid = (row.get("run_id") or "").strip()
            man = (row.get("manifest_path") or "").strip()
            status = (row.get("status") or "").strip().split("-but-")[0]
            stems = {rid} - {""}
            if man:
                stems.add(pathlib.Path(man).name
                          .removeprefix("job-").removesuffix("-raunav.yaml"))
            out += [(s, status, rid, (row.get("reason") or "").strip()) for s in stems]
    return tuple(out)


def _launched_stems() -> frozenset[str]:
    """Spec stems with a run record: a row whose status is in LAUNCHED."""
    return frozenset(s for s, status, _, _ in _ledger() if status in LAUNCHED)


def _names(spec: pathlib.Path, text: str) -> set[str]:
    """The names the ledger can key a spec by. The ledger keys a run by its job
    name, which can differ from the file name (job-mtx-r16_q1-s1-raunav.yaml
    runs job mtx-r16q1-s1-raunav), so both are the spec's identity."""
    names = {spec.name.removeprefix("job-").removesuffix("-raunav.yaml")}
    job = re.search(r"^metadata:\n(?:[ \t]+.*\n)*?[ \t]+name: *(\S+)", text, re.M)
    if job:
        names.add(job.group(1).removesuffix("-raunav"))
    return names


@functools.cache
def _spec_names() -> frozenset[str]:
    """Every name any spec in the repository goes by."""
    return frozenset().union(*(_names(p, p.read_text()) for p in REPO.glob(SPECS)))


def _exemption(stems: set[str]) -> str | None:
    """Why the ledger takes the spec known by `stems` out of this check, or None.

    A run record means the pin is history, and it must be the spec's own row:
    its manifest_path is the spec, or its run_id is the spec's name, or the
    spec's name plus a version or seed number (probe-bvc-raunav.yaml ran as
    "probe-bvc-v1"). A versioned run_id that is the name of another spec
    belongs to that spec (probe-bvc-v2 is its own file), and a longer name is
    not a version: ft-legs-bench-v3-last-a and extract-mtx-l162-s4-vcbwindow-full
    are other specs, whose launches say nothing about ft-legs-bench or
    extract-mtx-l162-s4. Retirement names the spec exactly, so retiring one spec
    never exempts another."""
    version = re.compile("(?:" + "|".join(map(re.escape, sorted(stems))) + r")-[vs]\d+")
    ran = sorted({(rid, status) for s, status, rid, _ in _ledger() if status in LAUNCHED
                  and (s in stems or (s == rid and s not in _spec_names()
                                      and version.fullmatch(s)))})
    if ran:
        return f"launched: ledger row {ran[0][0]} ({ran[0][1]}) records a run; its pin is history"
    retired = sorted({(rid, why) for s, status, rid, why in _ledger()
                      if status == RETIRED and s in stems})
    if retired:
        return f"retired: ledger row {retired[0][0]}: {retired[0][1]}"
    return None


def _behind(text: str, tag: str) -> list[str]:
    """The spec's dependencies last changed in a commit that `tag` lacks."""
    behind = []
    for dep in sorted(_dependencies(text)):
        last = _last_change(dep) if (REPO / dep).exists() else ""
        if last and not _tag_contains(tag, last):
            behind.append(f"{dep} (last changed in {last[:8]})")
    return behind


@pytest.mark.parametrize("spec", sorted(REPO.glob(SPECS)),
                         ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_unlaunched_spec_is_pinned_behind_a_script_it_runs(spec):
    """One case per spec: a new stale pin is a new failing case, never one more
    line inside a failure that is already red for other specs."""
    text = spec.read_text()
    tag = _tag(text)
    if tag is None or not _tag_exists(tag):
        return                          # clones no fixed tag, or a tag not in this clone

    why = _exemption(_names(spec, text))
    if why:
        pytest.skip(why)

    behind = _behind(text, tag)
    assert not behind, (
        f"{spec.relative_to(REPO)} has no run record and clones {tag}, which "
        "predates changes to code or configs it uses:\n  " + "\n  ".join(behind)
        + "\n\nRepin to a tag containing the fix, or, if it already ran, add a run "
          "record whose run_id is its job name or whose manifest_path is this spec, "
          f"with a status in {sorted(LAUNCHED)}; if it will never run, a row whose "
          "status is 'retired'.")


def test_the_rule_would_have_caught_the_labelrec_defect():
    """The guard is only worth having if it fires on the known bad case.

    job-eval-labelrec-raunav.yaml @mtx-s1.31 executing label_recovery.py is the
    real defect found on 2026-09-08. Rebuild that pair here and check the
    ancestry test rejects it -- so a future refactor cannot quietly turn the
    check into one that always passes.
    """
    # The equal-size fix, named literally. Reading "the last commit that touched
    # label_recovery.py" instead made this fixture dissolve the moment that file
    # was edited again for any unrelated reason -- which happened on 2026-09-19
    # -- and the test then failed while the guard it checks was working
    # perfectly. A regression fixture has to name the commit it reproduces.
    last = "759bbad33cbfe108a4a1b1587ee31febea607af3"
    assert _git("cat-file", "-t", last) == "commit", "the fix commit is gone"
    stale = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", last, "mtx-s1.31"],
        capture_output=True)
    assert stale.returncode != 0, (
        "mtx-s1.31 now contains the latest label_recovery.py, so this fixture "
        "no longer reproduces the defect it is pinning")

    fixed = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", last, "mtx-s1.32"],
        capture_output=True)
    assert fixed.returncode == 0, "mtx-s1.32 should carry the equal-size fix"


def test_the_rule_would_have_caught_the_repo_ref_defect():
    """Ledger row ft-subsets-bench-stale-tag, rebuilt from the spec as committed.

    job-ft-subsets-bench-raunav.yaml at b1df2d4 set REPO_REF=mtx-s1.11 and ran
    `make_subsets.py bench`, a mode that b1df2d4 itself added. A rule that reads
    only literal --branch tags skips every REPO_REF spec and cannot see this.
    """
    text = _git("show", "b1df2d4:experiments/FT/k8s/job-ft-subsets-bench-raunav.yaml")
    assert _tag(text) == "mtx-s1.11"
    assert "experiments/FT/make_subsets.py" in _dependencies(text)
    fix = "b1df2d4ee4cf1afa4a5aac91d25b2163070aa2e5"
    assert not _tag_contains("mtx-s1.11", fix)
    assert _tag_contains("mtx-s1.12", fix)


def test_the_rule_reaches_the_v2_grid_and_every_dependency_kind():
    """The v2 grid sits below k8s/ and pins through REPO_REF; it reads a model
    file and an arm config besides the script, and none of that may be missed."""
    grid = sorted(REPO.glob("experiments/MTX/k8s/v2/grid/*.yaml"))
    assert grid and set(grid) <= set(REPO.glob(SPECS))
    for spec in grid:
        text = spec.read_text()
        assert _tag(text) not in (None, "${REPO_REF}"), spec.name
        deps = _dependencies(text)
        assert "experiments/MTX/pretrain_v2.py" in deps, spec.name
        assert "experiments/MTX/stream_v2.py" in deps, spec.name    # imported by it
        assert "src/utils/reproducibility.py" in deps, spec.name
        assert any(d.startswith("experiments/MTX/ParT_") for d in deps), spec.name
        # The --network-config file is a thin wrapper; the model it re-exports
        # is loaded from another directory and must be a dependency too.
        assert "experiments/E1/ParT_sophon_arch_10c.py" in deps, spec.name
        assert any(d.startswith("configs/arms/") for d in deps), spec.name
    assert _executed_scripts("python3 -u experiments/X/y.py --a 1") == {"experiments/X/y.py"}
    assert not _dependencies("# python3 scripts/b.py --network-config experiments/M/n.py "
                             "configs/arms/A.yaml")
    # Inline snippets, as job-mtx-mpm-smoke and job-massreg-e0b-extract write them.
    inline = _dependencies(
        'mpm = load("mpm", "experiments/MTX/mpm.py")\n'
        'grep -q "def diverged" experiments/FT/bench_metrics.py\n'
        'FILES=$(python3 -c "import sys; sys.path.insert(0,\'scripts\'); '
        'from build_extract_jobs import interleaved_files")\n')
    assert {"experiments/MTX/mpm.py", "experiments/FT/bench_metrics.py",
            "scripts/build_extract_jobs.py"} <= inline


_EVERY_LOAD_FORM = '''
import os, sys, importlib.util, pathlib
from src.stats import paired
from . import stream_v2
import stream_ids
REPO = pathlib.Path(__file__).resolve().parents[2]
_E1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "E1", "ParT_sophon_arch_10c.py")
ANCHORS = REPO / "experiments" / "EVAL" / "anchors.py"
P = _load("probe", "experiments/EVAL/probe.py")
sys.path.insert(0, str(REPO / "experiments/FIGS"))
import style
def _load(name):
    return HERE / f"{name}.py"
HM = _load("hybrid_mass")
manifest = {}
manifest["driver"] = "experiments/MTX/pretrain_v2.py"
NOTE = {"seed": "experiments/FT/ft_v2.py"}
"""Docstrings name files too: experiments/FT/make_subsets.py."""
'''


def test_the_closure_follows_every_load_form_the_code_uses():
    """Each way this repository's code loads another repository file, written as
    the code writes it, from a module sitting in experiments/MTX. A form the
    closure drops is a fix that reaches no spec: ParT_sophon_arch_mtx.py loads
    the E1 model through os.path.join, seed_weaver.py loads mpm.py through
    `REPO / "experiments" / "MTX" / ...`, paired_errors.py imports
    `from src.stats import paired`, sim_scores.py loads `HERE / f"{name}.py"`."""
    got = _resolve_loads(_EVERY_LOAD_FORM, REPO / "experiments" / "MTX")
    assert got >= {
        "src/__init__.py", "src/stats/__init__.py", "src/stats/paired.py",
        "experiments/MTX/stream_v2.py", "experiments/MTX/stream_ids.py",
        "experiments/E1/ParT_sophon_arch_10c.py", "experiments/EVAL/anchors.py",
        "experiments/EVAL/probe.py", "experiments/FIGS/style.py",
        "experiments/MTX/hybrid_mass.py"}, sorted(got)
    # Records are not loads: a manifest field, a dict value, a docstring.
    assert not got & {"experiments/MTX/pretrain_v2.py", "experiments/FT/ft_v2.py",
                      "experiments/FT/make_subsets.py"}, sorted(got)
    # Recursion reaches what a package's __init__ imports relatively.
    assert "src/stats/bootstrap.py" in _loads("src/stats/__init__.py")


def test_configs_read_at_a_fixed_path_are_dependencies():
    """The two the 2026-10-01 audit found missing: extract_v2.py reads its data
    config from an argparse default, and probe.py, which it imports, reads the
    collapse rungs from the label map."""
    extract = _dependencies("python3 experiments/EVAL/extract_v2.py --out /data/x\n")
    assert "configs/data/JetClassII_massreg.yaml" in extract
    assert "configs/labelmaps/rung_label_maps.v1.csv" in extract     # via probe.py
    probe = _dependencies("python3 -u experiments/EVAL/probe.py --out /data/x\n")
    assert "configs/labelmaps/rung_label_maps.v1.csv" in probe
    # A tracked data file the spec itself passes in.
    assert "experiments/FT/data/ft_v2_subsets_sha256.json" in _dependencies(
        "python3 x --sha experiments/FT/data/ft_v2_subsets_sha256.json\n")


def test_a_changed_config_flags_the_spec_that_reads_it(tmp_path, monkeypatch):
    """Mutation test on a throwaway repository. The spec runs run.py, which
    takes its data config from an argparse default and imports helper.py, which
    reads a label map at a fixed path. Pin the spec, change one file per commit,
    and ask the rule itself (_behind) which changes make the pin stale."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                       check=True, capture_output=True)

    files = {
        "configs/data/base.yaml": "a: 1\n",
        "configs/labelmaps/maps.csv": "native,group\n",
        "configs/labelmaps/notes.csv": "x\n",
        "README": "x\n",
        "experiments/X/helper.py": (
            "import pathlib\nREPO = pathlib.Path(__file__).resolve().parents[2]\n"
            "def maps():\n    return (REPO / 'configs' / 'labelmaps' / 'maps.csv').read_text()\n"),
        "experiments/X/run.py": (
            "import argparse, pathlib\nimport helper\n"
            "REPO = pathlib.Path(__file__).resolve().parents[2]\n"
            "ap = argparse.ArgumentParser()\n"
            "ap.add_argument('--data-config', default=str(REPO / 'configs/data/base.yaml'))\n"
            "print('see configs/labelmaps/notes.csv for the map')\n"),
    }
    for path, body in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(body)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    git("tag", "t1")
    spec = ("args:\n- |\n  git clone --depth 1 --branch t1 x /w\n"
            "  python3 experiments/X/run.py --out /data/x\n")

    caches = (_loads, _data_files, _tracked, _last_change, _tag_contains, _tag_exists)
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)

    def stale_after_changing(path, tag):
        (tmp_path / path).write_text((tmp_path / path).read_text() + "changed\n")
        git("commit", "-q", "-am", f"change {path}")
        for c in caches:
            c.cache_clear()
        return [b.split()[0] for b in _behind(spec, tag)]

    for c in caches:
        c.cache_clear()
    try:
        deps = _dependencies(spec)
        assert {"experiments/X/run.py", "experiments/X/helper.py", "configs/data/base.yaml",
                "configs/labelmaps/maps.csv"} <= deps, sorted(deps)
        assert _tag(spec) == "t1" and _behind(spec, "t1") == []
        assert stale_after_changing("README", "t1") == []
        assert stale_after_changing("configs/labelmaps/notes.csv", "t1") == []   # prose only
        assert stale_after_changing("configs/data/base.yaml", "t1") == ["configs/data/base.yaml"]
        git("tag", "t2")
        # Read by a module run.py imports, one level down.
        assert stale_after_changing("configs/labelmaps/maps.csv", "t2") == [
            "configs/labelmaps/maps.csv"]
    finally:
        for c in caches:
            c.cache_clear()


def test_only_a_launch_or_a_retirement_exempts_a_spec(tmp_path, monkeypatch):
    """A row recording a spec that was only queued, or superseded before it ran,
    must leave the spec under the check; a retirement exempts only the spec it
    names."""
    ledger = tmp_path / "RUNS.csv"
    ledger.write_text(
        "run_id,launched_utc,status,reason,manifest_path\n"
        "# a comment line\n"
        "specs-written,2026-01-01T00:00:00Z,queued,written,experiments/X/k8s/job-a-raunav.yaml\n"
        "b,2026-01-01T00:00:00Z,complete-but-undersized,ran,\n"
        "c-v2,2026-01-01T00:00:00Z,crashed,ran,\n"
        "d,2026-10-01T00:00:00Z,retired,never launched,experiments/X/k8s/job-d-raunav.yaml\n"
        "e-s2,2026-10-01T00:00:00Z,retired,never launched,\n"
        "f,2026-01-01T00:00:00Z,superseded,never ran,\n"
        # Launches of OTHER specs whose names extend g, h and j.
        "g-v2,2026-01-01T00:00:00Z,complete,ran,\n"
        "h-vcbwindow-full,2026-01-01T00:00:00Z,launched,ran,\n"
        "wave,2026-01-01T00:00:00Z,launched,ran,experiments/X/k8s/job-j-v3-last-a-raunav.yaml\n"
        # A versioned run_id with more after the version is not a version of i.
        "i-v2-result,2026-01-01T00:00:00Z,complete,ran,\n"
        # A manifest_path names one spec file exactly; k-v2 is not k.
        "wave2,2026-01-01T00:00:00Z,launched,ran,experiments/X/k8s/job-k-v2-raunav.yaml\n")
    monkeypatch.setattr(sys.modules[__name__], "LEDGER", ledger)
    monkeypatch.setattr(sys.modules[__name__], "_spec_names", lambda: frozenset(
        {"a", "b", "c", "d", "e", "f", "g", "g-v2", "h", "h-vcbwindow-full", "i",
         "j", "j-v3-last-a", "k"}))
    _ledger.cache_clear()
    try:
        assert _exemption({"a"}) is None
        assert _exemption({"b"}).startswith("launched: ledger row b (complete)")
        assert _exemption({"c"}).startswith("launched: ledger row c-v2 (crashed)")
        assert _exemption({"d"}) == "retired: ledger row d: never launched"
        assert _exemption({"e"}) is None                # retiring e-s2 does not retire e
        assert _exemption({"f"}) is None
        assert "f" not in _launched_stems() and "b" in _launched_stems()
        # A sibling spec's launch is that spec's record, never this one's.
        assert _exemption({"g"}) is None
        assert _exemption({"g-v2"}).startswith("launched: ledger row g-v2 ")
        assert _exemption({"h"}) is None
        assert _exemption({"j"}) is None
        assert _exemption({"j-v3-last-a"}).startswith("launched: ledger row wave ")
        assert _exemption({"i"}) is None
        assert _exemption({"k"}) is None
    finally:
        _ledger.cache_clear()


def test_a_sibling_specs_launch_does_not_exempt_a_spec():
    """The real cases. job-ft-legs-bench-raunav.yaml was never applied (kubectl:
    NotFound, 2026-09-18) and was once exempted by bench-v3-lastepoch, a row
    whose manifest_path is job-ft-legs-bench-v3-last-a-raunav.yaml; since
    2026-10-01 its own row retires it, and that row, not the sibling's, is what
    exempts it. The three
    extraction specs were exempted by their -vcbwindow-full siblings until their
    own launch rows were written. probe-bvc ran as probe-bvc-v1, which is no
    spec's name, so that row is its own."""
    def why(path):
        spec = REPO / "experiments" / path
        return _exemption(_names(spec, spec.read_text()))
    assert why("FT/k8s/job-ft-legs-bench-raunav.yaml").startswith(
        "retired: ledger row ft-legs-bench: never launched")
    for arm in ("l162-s4", "l162-s5", "r16q1-s1"):
        assert why(f"EVAL/k8s/job-extract-mtx-{arm}-raunav.yaml").startswith(
            f"launched: ledger row extract-mtx-{arm} (launched)")
    assert why("EVAL/k8s/job-probe-bvc-raunav.yaml").startswith(
        "launched: ledger row probe-bvc-v1 (complete)")


def test_an_exemption_is_a_skip_that_names_its_row():
    """Both exemptions show in the test output with the row behind them; neither
    is a silent pass. The three first-grid sweep specs never ran and were
    retired on 2026-10-01; bench-v2 shard a ran."""
    k8s = REPO / "experiments"
    for spec in ("job-g1-l162-lr4e3-raunav.yaml", "job-g1-l188-lr14e4-raunav.yaml",
                 "job-g1-r42q1-lr1e3-s2-raunav.yaml"):
        with pytest.raises(pytest.skip.Exception,
                           match=r"^retired: ledger row g1-.*never launched"):
            test_no_unlaunched_spec_is_pinned_behind_a_script_it_runs(k8s / "G1" / "k8s" / spec)
    with pytest.raises(pytest.skip.Exception, match=r"^launched: ledger row ft-legs-bench-v2-a "):
        test_no_unlaunched_spec_is_pinned_behind_a_script_it_runs(
            k8s / "FT" / "k8s" / "job-ft-legs-bench-v2-a-raunav.yaml")


def test_the_rerun_spec_is_pinned_at_the_fix_and_writes_somewhere_new():
    """The labelrec rerun must not reproduce the defect or overwrite the record."""
    spec = REPO / "experiments" / "EVAL" / "k8s" / "job-eval-labelrec-v2-raunav.yaml"
    text = spec.read_text()
    assert '--branch "mtx-s1.33"' in text
    assert "OUT=/data/results/eval/label_recovery_v2" in text
    assert "OUT=/data/results/eval/label_recovery\n" not in text


def test_the_physics_probe_rerun_does_not_repeat_or_overwrite_the_first_one():
    """probe-physics-raunav Completed at mtx-s1.18 and its results are on the PVC.

    The v1 spec must stay unappliable (its job name is taken by a Complete job,
    and reapplying would overwrite OUT), and the v2 spec must differ in all three
    of name, pin and output -- the same three the labelrec rerun changed.
    """
    k8s = REPO / "experiments" / "EVAL" / "k8s"
    v1 = (k8s / "job-probe-physics-raunav.yaml").read_text()
    v2 = (k8s / "job-probe-physics-v2-raunav.yaml").read_text()

    assert "SUPERSEDED" in v1 and "DO NOT APPLY" in v1
    assert "mtx-s1.18" in v1, "v1 must record the tag the job ACTUALLY ran at"

    assert "name: probe-physics-v2-raunav" in v2
    assert '--branch "mtx-s1.33"' in v2
    assert "OUT=/data/results/eval/probe_physics_v2" in v2
    assert "OUT=/data/results/eval/probe_physics\n" not in v2


def test_both_reruns_are_pinned_at_the_same_tag():
    """One tag for both reruns, so a single ref explains every new number."""
    k8s = REPO / "experiments" / "EVAL" / "k8s"
    pins = {f.name: re.search(r'--branch "([^"]+)"', f.read_text()).group(1)
            for f in (k8s / "job-probe-physics-v2-raunav.yaml",
                      k8s / "job-eval-labelrec-v2-raunav.yaml")}
    assert set(pins.values()) == {"mtx-s1.33"}, pins
