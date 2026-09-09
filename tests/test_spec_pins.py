"""A job spec must not be launched at a tag that predates a fix it depends on.

THIS DEFECT HAS LANDED THREE TIMES and cost a full rerun each time:
  - job-eval-anchors-raunav.yaml ran at mtx-s1.15, three commits before the
    unweighted-comparison fix, and "applying the committed spec reproduced the
    pre-fix number" (experiments/RUNS.csv, audit-2-anchors, 2026-09-08).
  - job-eval-labelrec-raunav.yaml ran at mtx-s1.31, one commit before 759bbad
    made the linear and MLP probes fit at equal training size.
  - job-probe-physics-raunav.yaml sat at mtx-s1.20 -- twelve tags stale, missing
    both the MIN_PER_CLASS-on-window fix and the MLP convergence record -- and
    was caught before its first run only because someone looked.

Nothing errors when this happens. The job clones, runs, and writes a plausible
number computed by old code.

WHAT THIS DOES NOT DO: bump historical pins. A spec that has already run is a
provenance record of what actually ran, and rewriting its tag would falsify the
ledger. So the rule binds only specs with NO run record -- the ones whose next
launch is still ahead of them.
"""
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
LEDGER = REPO / "experiments" / "RUNS.csv"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _executed_scripts(text: str) -> set[str]:
    """Scripts on an actual `python3 ...` line, never ones named in a comment.

    Spec headers cite their generator (`GENERATED from scripts/build_arm_jobs.py`)
    and those citations are not dependencies of the run.
    """
    found = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        found.update(re.findall(r"python3 +((?:experiments|scripts)/[\w/]+\.py)",
                                stripped))
    return found


def test_no_unlaunched_spec_is_pinned_behind_a_script_it_runs():
    ledger = LEDGER.read_text()
    behind = []
    for spec in sorted(REPO.glob("experiments/*/k8s/*.yaml")):
        text = spec.read_text()
        m = re.search(r'--branch "([^"$]+)"', text)     # literal tags only;
        if not m:                                       # "${REPO_REF}" is resolved
            continue                                    # at launch, not here
        tag = m.group(1)
        if _git("rev-parse", "--verify", f"{tag}^{{commit}}") == "":
            continue                                    # tag not in this clone

        # A run record means the pin is history. Match on the spec's stem
        # because a run_id often versions the spec name (probe-bvc-raunav.yaml
        # -> "probe-bvc-v1"), and the ledger cites both.
        stem = spec.name.removeprefix("job-").removesuffix("-raunav.yaml")
        if stem in ledger:
            continue

        for script in sorted(_executed_scripts(text)):
            if not (REPO / script).exists():
                continue
            last = _git("rev-list", "-1", "HEAD", "--", script)
            if not last:
                continue
            ancestor = subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor", last, tag],
                capture_output=True)
            if ancestor.returncode != 0:
                behind.append(f"{spec.name} @{tag} runs {script}, which has "
                              f"changed since that tag")
    assert not behind, (
        "unlaunched specs pinned behind code they execute:\n  "
        + "\n  ".join(behind)
        + "\n\nRepin to a tag containing the fix, or add a run record if it "
          "already ran.")


def test_the_rule_would_have_caught_the_labelrec_defect():
    """The guard is only worth having if it fires on the known bad case.

    job-eval-labelrec-raunav.yaml @mtx-s1.31 executing label_recovery.py is the
    real defect found on 2026-09-08. Rebuild that pair here and check the
    ancestry test rejects it -- so a future refactor cannot quietly turn the
    check into one that always passes.
    """
    last = _git("rev-list", "-1", "HEAD", "--", "experiments/EVAL/label_recovery.py")
    assert last, "label_recovery.py has no history"
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


def test_the_rerun_spec_is_pinned_at_the_fix_and_writes_somewhere_new():
    """The labelrec rerun must not reproduce the defect or overwrite the record."""
    spec = REPO / "experiments" / "EVAL" / "k8s" / "job-eval-labelrec-v2-raunav.yaml"
    text = spec.read_text()
    assert '--branch "mtx-s1.32"' in text
    assert "OUT=/data/results/eval/label_recovery_v2" in text
    assert "OUT=/data/results/eval/label_recovery\n" not in text
