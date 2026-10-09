#!/usr/bin/env python3
"""Emit experiments/EVAL/k8s/job-eval-anomaly-<model>-raunav.yaml, one per model.

WHY A BUILDER AND NOT TWELVE HAND-EDITS
---------------------------------------
The approved plan (2026-09-18) puts the weakly-supervised resonance-search study
at FOUR label granularities x FIVE pretraining seeds, with the seed as the unit
of inference. Four models are merged (anomaly_merged_v3) and four are in flight,
so twelve specs have to be written. Each is ~50 h of an 8-CPU pod, and the only
things that may legitimately differ between them are the model's identity and
the pinned tag -- everything else (the 72 h deadline measured off
eval-anomaly-r16q1-s3's own log timestamps, the precondition loop that matches
probe.load_arm, the region pin, the 32Gi/8-CPU request, --n-bkg / --n-template /
--trainings) is the thing the study holds fixed. Twelve hand-copies is twelve
chances to move one of those by accident, and nothing would error: the job would
run to completion and write a number computed under a different configuration,
which anomaly_merge.py's shared-key check would then refuse WITHOUT saying which
of the twelve was wrong.

So the committed in-flight spec is the template and every difference is an
ASSERTED substitution: each replacement declares how many occurrences it expects
and the build dies if the count is not exact, the same discipline
scripts/build_mtx_launch.py's derive_seed/derive_draw and
scripts/build_extract_jobs.py's verify_pin already use. render() then re-diffs
its own output against the template line by line and refuses anything that moved
outside ALLOWED_ANCHORS. tests/test_anomaly_jobs.py runs that same diff against
the COMMITTED files, so a later hand-edit to one spec is caught too.

THE HEAD WIDTH IS READ, NEVER ASSUMED
-------------------------------------
K comes out of each model's own training spec under experiments/MTX/k8s/ --
both `--num-classes` (what seed_weaver is told) and `-o num_classes` (what the
network is built with), which must agree -- and is then cross-checked against
the number of distinct nodes the committed label map gives for that column.
The trap is R42_Q1: the level is named for its 42 RESONANT groups and the head
also carries the QCD group, so K is 43, not 42. 188 / 162 / 17 are as named.
A wrong K here is not a crash: logits_from_features.py would refuse the
checkpoint (it compares the head's output width against --num-classes), which is
a clean failure -- but only after the clone, and only for this one wave.

RE-RUNS ARE VERSIONED, NEVER OVERWRITTEN
----------------------------------------
One of the thirteen is a RE-RUN of a model that was already scored: the
162-class seed-1 model's 2026-09-09 artifact holds five of six signals. A model
carries `version="v2"`, which moves the JOB NAME and the OUTPUT DIRECTORY to
`...-v2` and leaves the MODEL name alone. Both halves matter. Versioning the
directory is what keeps the earlier 30-cell artifact in place -- nothing in
/data/results is ever overwritten or deleted, and that file is the only record
of what merges v1..v3 actually read. NOT versioning the model name is what stops
the re-run reading as a sixth 162-class seed in the merged table, which is
precisely the miscount the seed-balanced regret exists to prevent.

v2 (--v2, specs under experiments/EVAL/k8s/v2/anomaly/)
-------------------------------------------------------
Every run of configs/arms/v2_grid.json that anomaly_summary.V2_ANOMALY_ARMS
selects (the vocabulary ladder, its leave-one-family-out arms, the self-supervised
arms; not the mass-output, random-partition or flavour arms), at every checkpoint
extract_v2.py extracts (build_extract_jobs.V2_CHECKPOINTS: best70 primary, wavg,
bestval, best70_bn, bestval_bn), each read two ways, the class-token features and
the pooled embedding (amendment A14; a self-supervised run has only the pooled
one); and the untrained trunk of run indices 1-3 (init-s1..3, tag init, both
readouts) as a reference row in every merge. Model name = run without "mtx-"
(l188lofo4p-s2), v1's names and so v1's resampling draws.

  eval-anomaly-v2-<model>-raunav     one per model, 8 CPUs: anomaly.py, Mahalanobis
      and kNN, on the uniform 2,000,000-jet prefix of each checkpoint's cache:
        /data/results/eval/v2/anomaly/<run>/<tag>/<readout>/anomaly_results.json
      (<run> = mtx-<model>, or init-s<k> for the references)
      A tag extract_v2.py linked to another (the same checkpoint file) is linked
      here too, not rescored. A unit already complete (its file carries
      null_unmeasured, written last) is skipped, so a retry loses one unit.
  eval-anomaly-v2-heads-<set>-raunav one per tier set: anomaly_heads.py, the output
      ratio (class_sum, class_sum_matched) of every model with an output layer:
        /data/results/eval/v2/anomaly_heads_<set>/anomaly_heads.json
  eval-anomaly-v2-merge-<set>-raunav one per tier set: anomaly_merge.py per (tag,
      readout), once every unit of the set is complete:
        /data/results/eval/v2/anomaly_merged_<set>/<tag>/<readout>/anomaly_results.json
  <set> = t12 (tiers 1-2, the analysis freeze of A14) or t123 (every tier).
  The table step: anomaly_summary.py --grid configs/arms/v2_grid.json --anomaly
  <merged>/<tag>/<readout>/anomaly_results.json [--heads <heads>] --out <dir>.

Run:  python3 scripts/build_anomaly_jobs.py [--check-only] [--only RUN_ID ...]
      python3 scripts/build_anomaly_jobs.py --pin-not-yet-tagged
      python3 scripts/build_anomaly_jobs.py --class-sum-rerun --pin-not-yet-tagged
      python3 scripts/build_anomaly_jobs.py --v2 --pin-not-yet-tagged [--check-only]
"""
from __future__ import annotations

import argparse
import difflib
import functools
import json
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
K8S = ROOT / "experiments" / "EVAL" / "k8s"
MTX_K8S = ROOT / "experiments" / "MTX" / "k8s"
LABEL_MAP = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"

TEMPLATE_SPEC = K8S / "job-eval-anomaly-l162-s2-raunav.yaml"

# mtx-s1.51. The four in-flight arms clone mtx-s1.47; this wave is written,
# committed and THEN tagged, so the tag cannot exist while the specs are being
# written -- hence --pin-not-yet-tagged, which checks the WORKING TREE instead
# and says so loudly. Moving an already-pushed tag is never an option: the four
# running pods cloned s1.47 and their provenance points at it.
PIN = "mtx-s1.51"
TEMPLATE_PIN = "mtx-s1.47"

# Everything the pod executes after the clone. A path that exists in the working
# tree can be absent from the tag, and the job would then pay for a clone and die
# on "No such file or directory" -- the failure build_extract_jobs.verify_pin was
# written for.
NEEDED_AT_PIN = [
    "experiments/EVAL/anomaly.py",
    "experiments/EVAL/logits_from_features.py",
    "experiments/EVAL/probe.py",
    "configs/labelmaps/rung_label_maps.v1.csv",
]


class Model(NamedTuple):
    rung: str          # the label-set name anomaly.py resolves against the tree
    seed: str          # "1".."5", or "1b" for the L162 repair run
    k: int             # head width, verified against the training spec
    version: str = ""  # "v2" when this is a RE-RUN of a model already scored

    @property
    def run(self) -> str:
        """The run directory on the PVC: mtx-r42q1-s1, not mtx-r42_q1-s1."""
        return f"mtx-{self.rung.lower().replace('_', '')}-s{self.seed}"

    @property
    def label(self) -> str:
        """The MODEL's name, which the score is attributed to.

        DELIBERATELY NOT VERSIONED. A re-run scores the same checkpoint over the
        same jets; versioning this would put `l162-s1b-v2` in the merged
        artifact beside l162-s2..s5 and read as a sixth 162-class seed, which
        is exactly the miscount the seed-balanced regret exists to prevent. The
        JOB and the OUTPUT DIRECTORY carry the version (see `tag`); the model
        does not.
        """
        return self.run.removeprefix("mtx-")

    @property
    def tag(self) -> str:
        """What names the job and the output directory: `l162-s1b-v2`.

        Versioned so a re-run neither takes a Complete job's name nor writes
        over the earlier artifact -- nothing in /data/results is ever
        overwritten or deleted.
        """
        return self.label if not self.version else f"{self.label}-{self.version}"

    @property
    def training_spec(self) -> pathlib.Path:
        """The spec files keep the underscores the run directories drop."""
        return MTX_K8S / f"job-mtx-{self.rung.lower()}-s{self.seed}-raunav.yaml"

    @property
    def job_spec(self) -> pathlib.Path:
        return K8S / f"job-eval-anomaly-{self.tag}-raunav.yaml"


# The template's own identity. Every substitution below replaces one of these.
TEMPLATE_MODEL = Model("L162", "2", 162)

# THE THIRTEEN. Usable coverage before this wave is SEVEN runs: 17-class seeds
# 2/3/4 merged into anomaly_merged_v3, and 162-class seeds 2/3/4 plus 17-class
# seed 5 in flight since 2026-09-17. The two granularities the approved plan
# adds have NO coverage at all, which is why they lead the launch order.
MODELS = [
    # Seeds 1, 2 AND 3 are listed at the bottom, versioned, because all three
    # first attempts are dead. They are REPLACED there, not joined -- see those
    # entries. 1 and 2 lost their nodes to the same taint eviction; 3 ran out
    # of its active deadline after losing time to that same eviction wave.
    *[Model("L188", str(s), 188) for s in range(4, 6)],
    *[Model("R42_Q1", str(s), 43) for s in range(1, 6)],
    Model("L162", "5", 162),
    Model("R16_Q1", "1", 17),
    # THE RE-RUN (decided 2026-09-19). The 162-class seed-1 model WAS scored, on
    # 2026-09-09, and that artifact at /data/results/eval/anomaly holds FIVE of
    # six signals: label_X_YY_qqqq is absent because the job died at ~25 h and it
    # is the last signal in the suite. Measured, not inferred -- anomaly_merged_v3
    # records arms_missing_signals {'l162-s1b': ['label_X_YY_qqqq']} and
    # signals_with_one_rung ['label_X_YY_qqqq'] -- so it contributes 30 cells
    # where every other model contributes 36, and the v4 merge refuses that.
    #
    # WHY THE FULL ~50 h RE-RUN AND NOT THE ~8.3 h SINGLE-SIGNAL TOP-UP. The
    # top-up would need two payloads unioned into one model's record, and
    # anomaly_merge.merge refuses a duplicated model BY DESIGN -- "keeping one
    # silently hides which run the published number came from". That refusal
    # guards the thing this study is most exposed to, and working around it to
    # save ~40 pod-hours out of ~650 is a bad trade.
    #
    # Its logits already exist (1,296,000,128 B), so this one skips the build.
    Model("L162", "1b", 162, version="v2"),
    # THE SECOND RE-RUN (2026-09-21). This entry REPLACES the unversioned
    # 188-class seed-1 model in the comprehension above; it is not an addition
    # beside it. Adding it beside was my first attempt and three tests caught
    # it: `(rung, label)` must stay unique across MODELS or a granularity gains
    # a phantom seed, which is the exact miscount the versioning scheme exists
    # to prevent. There is still exactly ONE model scoring the 188-class seed-1
    # checkpoint -- it just writes to a versioned directory now.
    #
    # eval-anomaly-l188-s1-raunav is FAILED,
    # not slow: `BackoffLimitExceeded`, failed=3 against backoffLimit=2, no
    # active pod. It lost its node twice to `TaintManagerEviction` and the
    # third pod went Unknown at ~8 h, which exhausted a budget sized for one
    # retry. Its artifact is PARTIAL -- no `null_unmeasured` key -- so it
    # contributes nothing to a merge, and there is no resume: the ~50 h starts
    # over. Versioned for the same two reasons as the 162-class re-run above:
    # the directory moves so the partial artifact stays on disk as the record
    # of what the failed attempts read, and the MODEL name does not, so this
    # cannot read as a sixth 188-class seed.
    Model("L188", "1", 188, version="v2"),
    # THE THIRD RE-RUN (2026-09-21, same cause, found in the same sweep).
    # eval-anomaly-l188-s2-raunav is also FAILED -- BackoffLimitExceeded,
    # failed=3 against backoffLimit=2 -- having lost its node to the same
    # TaintManagerEviction wave as seed 1 and then had its replacement go
    # Unknown at ~18 h. Its artifact is likewise partial. Replaces the
    # unversioned seed-2 entry above, for the reasons on the seed-1 entry.
    #
    # THE RETRY BUDGET IS THE COMMON CAUSE AND IT IS NOT FIXED HERE.
    # backoffLimit 2 was sized for one bad node, and this wave lost several at
    # once. Raising it is NOT obviously right: this scoring has no resume, so
    # every retry restarts ~50 h, and a larger budget spends more pod-hours
    # rather than fewer. Left alone deliberately, and recorded so the next
    # eviction wave is read as the same cause rather than a new mystery.
    Model("L188", "2", 188, version="v2"),
    # THE FOURTH RE-RUN (2026-09-22). eval-anomaly-l188-s3-raunav is FAILED
    # with DeadlineExceeded, NOT BackoffLimitExceeded: it started 2026-09-19
    # 08:10 and activeDeadlineSeconds (259200, 72 h) expired before its ~50 h
    # scoring finished, because the deadline counts across every retry and the
    # 2026-09-20 eviction wave cost it a restart. Its artifact has no
    # `null_unmeasured` key, so it is partial. Replaces the unversioned seed-3
    # entry above for the reasons on the seed-1 entry. The 72 h deadline is
    # left as it is: a clean ~50 h run fits inside it with a restart's margin,
    # and a longer deadline would only let a stuck pod burn longer.
    Model("L188", "3", 188, version="v2"),
]

# The only lines a generated spec may differ from the template on. Each entry is
# (anchor text that must appear on the template's line, what it is for). The
# order is the order they appear in the file, and render() checks BOTH that no
# other line moved and that each of these did.
#
# THE HEADER PIN LINE IS IN THIS SET, added 2026-09-19. It names the tag and the
# clone line names the tag, and a spec whose header disagrees with its own clone
# documents a run that never happened -- which is how this template came to cite
# a tag two moves behind the one it clones. Substituting it is what stops twelve
# derived specs inheriting that. It is the only COMMENT in the set, and it is
# here because it is the pinned tag, not because comments are negotiable.
ALLOWED_ANCHORS = [
    ("  # PIN mtx-", "the pinned tag, named in the header"),
    ("  name: eval-anomaly-", "the job name"),
    ('          git clone --depth 1 --branch "', "the pinned tag, cloned"),
    ("          a=mtx-", "the three shell variables naming the model"),
    ("          OUT=/data/results/eval/anomaly_", "the output directory"),
    ("            --features ", "the model name passed to --features"),
    ("            --rungs ", "the model name passed to --rungs"),
]


# THE STAGING THE PI APPROVED, 2026-09-19: three seeds at EVERY granularity
# first, then continue to five. Waves A and B are also the first six jobs of the
# five-seed plan, so taking the early read costs no re-ordering and no rerun --
# only one extra merge, which is minutes of JSON arithmetic.
#
# WHY THE TWO NEW GRANULARITIES LEAD. They have ZERO coverage, and a level with
# no seeds blocks the four-level comparison outright; a level at four of five
# seeds only widens an interval. After wave B every level has three seeds and
# the merge's equal-seed assertion passes on a 3/3/3/3 subset.
#
# WHY THE RE-RUN SITS IN C AND NOT EARLIER. It closes the 162-class level from
# four usable seeds to five. Nothing before wave C needs it, because the 3-seed
# interim merge reads 162-class seeds 2/3/4, which are already in flight.
LAUNCH_WAVES = [
    ("A", ["mtx-l188-s1", "mtx-r42q1-s1"]),
    ("B", ["mtx-l188-s2", "mtx-l188-s3", "mtx-r42q1-s2", "mtx-r42q1-s3"]),
    ("C", ["mtx-l162-s1b", "mtx-l162-s5", "mtx-r16q1-s1",
           "mtx-l188-s4", "mtx-r42q1-s4"]),
    ("D", ["mtx-l188-s5", "mtx-r42q1-s5"]),
]


def head_width_from_training_spec(m: Model) -> int:
    """K read off the model's OWN training spec, from both places it is stated.

    seed_weaver is told `--num-classes K` and the network is built with
    `-o num_classes K`. They are separate arguments and a spec could carry two
    different values, so both are collected and required to agree; a spec that
    disagrees with itself is a finding, not something to pick a side on.
    Comment lines are dropped: every mtx spec's header PROSE names the argument
    it varies, and matching that would read a number out of documentation.
    """
    if not m.training_spec.exists():
        raise SystemExit(f"FATAL: {m.training_spec} not found, so {m.run}'s head "
                         f"width cannot be verified. It must not be guessed: a "
                         f"wrong K builds logits from the wrong head.")
    live = "\n".join(ln for ln in m.training_spec.read_text().splitlines()
                     if not ln.lstrip().startswith("#"))
    flag = set(re.findall(r"--num-classes (\d+)", live))
    opt = set(re.findall(r"-o num_classes (\d+)", live))
    if len(flag) != 1 or flag != opt:
        raise SystemExit(
            f"FATAL: {m.training_spec.name} states its head width as "
            f"--num-classes {sorted(flag)} and -o num_classes {sorted(opt)}. "
            f"Those must be one number.")
    return int(flag.pop())


def head_width_from_label_map(rung: str) -> int:
    """K cross-checked against the committed tree: one head output per node.

    An independent source for the same integer. The training spec says what the
    head was BUILT with; this says what the vocabulary actually contains, and
    anomaly.node_roles indexes the logits by node id, so a mismatch would score
    the wrong columns rather than crash.
    """
    import csv
    with LABEL_MAP.open() as f:
        rows = list(csv.DictReader(f))
    if rung not in rows[0]:
        raise SystemExit(f"FATAL: {rung} is not a column of {LABEL_MAP.name}")
    return len({int(r[rung]) for r in rows})


def verify_head_widths(models=MODELS) -> dict[str, int]:
    """Every declared K against both sources. Returns {run: K}."""
    out = {}
    for m in models:
        spec_k = head_width_from_training_spec(m)
        map_k = head_width_from_label_map(m.rung)
        if not (m.k == spec_k == map_k):
            raise SystemExit(
                f"FATAL: {m.run} declares K={m.k}, its training spec says "
                f"{spec_k}, the committed label map gives {map_k}.")
        out[m.run] = m.k
    return out


def verify_pin(pin: str, allow_untagged: bool, needed=NEEDED_AT_PIN) -> None:
    """The pod clones a TAG, so check the TAG's tree, not the working tree."""
    tagged = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{pin}"],
        cwd=ROOT, capture_output=True).returncode == 0
    if not tagged:
        if not allow_untagged:
            sys.exit(f"FATAL: tag {pin} does not exist. Pass "
                     f"--pin-not-yet-tagged if it is about to be created on a "
                     f"commit carrying {len(needed)} files, and create it "
                     f"BEFORE applying any spec that clones it.")
        gone = [p for p in needed if not (ROOT / p).exists()]
        if gone:
            sys.exit(f"FATAL: {gone} not in the working tree either")
        print(f"WARNING: tag {pin} DOES NOT EXIST YET. All {len(needed)} "
              f"files are in the working tree; create the tag on a commit that "
              f"has them before applying anything.")
        return
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", pin, "--", *needed],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    missing = [p for p in needed if p not in listed]
    if missing:
        sys.exit(f"FATAL: tag {pin} does not contain {missing}. The pod clones "
                 f"the TAG, so the job would die after paying for the clone.")
    print(f"pin {pin} verified to contain all {len(needed)} files the "
          f"job runs")


def _substitute(text: str, subs) -> str:
    """Apply (old, new, expected count) in order. Raises before anything moves."""
    for old, new, n in subs:
        got = text.count(old)
        if got != n:
            raise SystemExit(
                f"FATAL: expected {n} occurrence(s) of {old!r} in the template, "
                f"found {got}. The template changed under this builder; re-read "
                f"it rather than loosening the count.")
        text = text.replace(old, new)
    return text


def changed_lines(template: str, generated: str) -> list[tuple[int, str, str]]:
    """(1-based template line no, template line, generated line) for every line
    that moved. Requires equal line counts -- a substitution that adds or drops
    a line is not one of the permitted differences."""
    a, b = template.splitlines(), generated.splitlines()
    if len(a) != len(b):
        raise SystemExit(f"FATAL: template has {len(a)} lines, generated has "
                         f"{len(b)}. Only in-place line rewrites are permitted.")
    return [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]


def render(m: Model, pin: str = PIN) -> str:
    """The spec text for one model, template-derived and self-checked."""
    t = TEMPLATE_MODEL
    text = TEMPLATE_SPEC.read_text()
    # WHOLE LINES, leading indent and trailing newline included. A bare token
    # like "l162-s2" occurs five times and would be rewritten everywhere at
    # once, including anywhere a future comment happens to mention it; a full
    # line can only match its own site.
    subs = [
        (f"  # PIN {TEMPLATE_PIN}.\n", f"  # PIN {pin}.\n", 1),
        (f"  name: eval-anomaly-{t.tag}-raunav\n",
         f"  name: eval-anomaly-{m.tag}-raunav\n", 1),
        (f'--branch "{TEMPLATE_PIN}"', f'--branch "{pin}"', 1),
        (f"          a={t.run}; r={t.rung}; k={t.k}\n",
         f"          a={m.run}; r={m.rung}; k={m.k}\n", 1),
        (f"          OUT=/data/results/eval/anomaly_{t.tag}\n",
         f"          OUT=/data/results/eval/anomaly_{m.tag}\n", 1),
        # The MODEL name, not the tag: a re-run scores the same checkpoint and
        # must be attributed to the same model, or it reads as an extra seed.
        (f"            --features {t.label}=${{d}} \\\n",
         f"            --features {m.label}=${{d}} \\\n", 1),
        (f"            --rungs {t.label}=${{r}} \\\n",
         f"            --rungs {m.label}=${{r}} \\\n", 1),
    ]
    out = _substitute(text, subs)

    # RE-DERIVE THE DIFF FROM THE RESULT rather than trusting that six
    # substitutions produced six changes. They are applied to overlapping text
    # and a template edit could make one of them land twice on one line.
    moved = changed_lines(text, out)
    if len(moved) != len(ALLOWED_ANCHORS):
        raise SystemExit(
            f"FATAL: {m.label} differs from the template on {len(moved)} lines, "
            f"expected {len(ALLOWED_ANCHORS)}:\n  "
            + "\n  ".join(f"{n}: {x.strip()!r} -> {y.strip()!r}"
                          for n, x, y in moved))
    for (lineno, was, _now), (anchor, what) in zip(moved, ALLOWED_ANCHORS):
        if not was.startswith(anchor):
            raise SystemExit(
                f"FATAL: {m.label} line {lineno} was expected to be {what} "
                f"(starting {anchor!r}) but is {was.strip()!r}")

    # The template model's identity must not survive anywhere the shell runs.
    # Its COMMENTS legitimately name other models (the 2026-09-09 162-class run,
    # eval-anomaly-r16q1-s3's timing measurement) and those are history.
    live = [ln for ln in out.splitlines() if not ln.lstrip().startswith("#")]
    for token in (t.label, t.run):
        hit = [ln for ln in live if token in ln]
        if hit:
            raise SystemExit(f"FATAL: {token!r} survives on an executed line of "
                             f"{m.tag}: {hit[0].strip()}")
    # The old pin is held to a STRICTER rule than the model name: it must be
    # gone from the comments too. A header citing one tag while the clone line
    # names another is the defect the header substitution was added to close,
    # and leaving the old tag anywhere in the file would reopen it.
    if TEMPLATE_PIN in out:
        raise SystemExit(f"FATAL: {TEMPLATE_PIN!r} survives somewhere in "
                         f"{m.tag}; it must be gone from the header as well as "
                         f"the clone line.")
    return out


def build(m: Model, pin: str, check_only: bool) -> str:
    text = render(m, pin)
    if check_only:
        if not m.job_spec.exists():
            return "MISSING"
        return "ok" if m.job_spec.read_text() == text else "DIFFERS"
    existed = m.job_spec.exists()
    if existed and m.job_spec.read_text() == text:
        return "unchanged"
    m.job_spec.write_text(text)
    return "rewritten" if existed else "written"


# ------------------------------------------------ THE CLASS-SUM RERUN (2026-09-28)
#
# class_sum leaves out only the signal's own output node: 1 native class at 188
# and 162 classes, 3-12 at 43, 10-29 at 17, so comparing it across label sets
# mixes the label-set effect with a change of estimator. anomaly.py's
# class_sum_matched leaves out the signal's whole 17-class group at every label
# set. It needs the head's logits and nothing else, so these twenty specs score
# ONLY that family (--families): no kNN, Mahalanobis or IAD.
#
# Each spec is the four-family template with the model and pin substituted as in
# render(), the lines in CS_ALLOWED_ANCHORS changed, and its header and deadline
# comments replaced (they describe the four-family run). The model name is NOT
# versioned (Model.label), so anomaly.cell_seed draws the committed resamplings;
# anomaly_summary.py refuses the merged rerun unless every cell's rng_seeds match
# and, at 17 classes, where the two estimators coincide, the committed class_sum
# is reproduced. The version "cs" moves only the job name and the output
# directory, as "v2" does for the re-runs above.
#
# mtx-s1.64: the first tag after mtx-s1.63 that carries class_sum_matched. It
# does not exist when these are written -- --pin-not-yet-tagged, as above.
CS_PIN = "mtx-s1.64"
CS_FAMILIES = "class_sum_matched"
CS_MODELS = [Model(rung, seed, k, version="cs")
             for rung, k, seeds in (("L188", 188, "12345"), ("L162", 162, ["1b", *"2345"]),
                                    ("R42_Q1", 43, "12345"), ("R16_Q1", 17, "12345"))
             for seed in seeds]
CS_MERGE = K8S / "job-eval-anomaly-cs-merge-raunav.yaml"
CS_MERGE_OUT = "/data/results/eval/anomaly_cs_merged_v1"
MERGE_V4 = K8S / "job-eval-anomaly-merge-v4-raunav.yaml"
MERGE_V4_PIN = "mtx-s1.51"
CS_NEEDED_AT_PIN = [*NEEDED_AT_PIN, "experiments/EVAL/anomaly_merge.py"]
# COST, MEASURED (2026-09-28, Apple M4 Pro, production sizes: 200,000 + 200,000
# QCD, 2,000 injected, 188 outputs). class_sum_matched alone: 37.6 s per
# resampling with a signal, almost all of it the sigma_min bisection, and 0.7 s
# per null resampling. 29 (signal, N_sig) cells carry a signal (X->YY->bbb is
# skipped at 4,000): 29 x 10 x 37.6 s = 3.0 h per model locally. The same
# machine takes 226 s (signal) / 50 s (null) for the four-family resampling,
# 197 s averaged over the grid, against 550 s on the pods (91.6 min per cell,
# measured off eval-anomaly-r16q1-s3's log), so a pod is ~2.8x slower:
# ~8.5 h per model, ~170 core-hours for the twenty (340 CPU-hours requested at
# 2 CPUs), against ~50 h x 8 CPUs per model for the four-family run.
# 36 h covers two attempts (backoffLimit 1) with a 2x margin on the rate.
CS_DEADLINE = 129600

# Executed lines a rerun spec may differ from the template on. Unlike
# ALLOWED_ANCHORS this is a set a changed line must fall in, not an exact
# sequence, because the template's own model (162-class seed 2) is one of the
# twenty and its identity lines therefore do not move.
CS_ALLOWED_ANCHORS = [
    ("  name: eval-anomaly-", "the job name"),
    ("  backoffLimit: ", "one retry"),
    ("  activeDeadlineSeconds: ", "the deadline, sized to class-sum scoring"),
    ('          git clone --depth 1 --branch "', "the pinned tag"),
    ("          a=mtx-", "the model"),
    ("          OUT=/data/results/eval/anomaly_", "the output directory"),
    ("            --features ", "the model name"),
    ("            --rungs ", "the model name"),
    ("            --out ${OUT}", "the families scored"),
    ("          requests: ", "2 CPUs, 16Gi: single-threaded scoring"),
    ("          limits:   ", "2 CPUs, 16Gi: single-threaded scoring"),
]
CS_MERGE_ANCHORS = [
    ("  name: eval-anomaly-", "the job name"),
    ("  backoffLimit: ", "one retry"),
    ('          git clone --depth 1 --branch "', "the pinned tag"),
    ("          for d in /data/results/eval/anomaly_", "the inputs"),
    ("                   /data/results/eval/anomaly_", "the inputs"),
    ("          OUT=/data/results/eval/anomaly_", "the output directory"),
]

CS_HEADER = """\
  # CLASS-SUM RERUN, ONE MODEL PER JOB (2026-09-28): ONLY class_sum_matched.
  #
  # WHY. The committed class_sum leaves out the signal's own output node, which
  # is 1 native class at 188 and 162 classes but 3-12 at 43 and 10-29 at 17, so
  # comparing it across label sets mixes the label-set effect with a change of
  # estimator. class_sum_matched leaves out the signal's whole 17-class group
  # at every label set; the tree is nested, so at every finer set that group is
  # an exact union of output nodes and the SAME native classes are removed.
  #
  # HELD FIXED. The feature cache and its logits (built from the features only
  # if absent, never overwritten), --n-bkg / --n-template / --trainings, the
  # signals and injections. --features names the UNVERSIONED model, so the
  # committed resamplings are drawn exactly; anomaly_summary.py refuses the
  # merged rerun unless every cell's rng_seeds match and, at 17 classes where
  # the two estimators coincide, the committed class_sum is reproduced.
  #
  # NOT RUN. kNN, Mahalanobis, IAD. IAD is dropped from the paper: its
  # sigma_min is not the definition of arXiv:2604.20965 Sec. V.4.
  #
  # PIN {pin}. Not tagged when this was written: create the tag on the commit
  # carrying class_sum_matched BEFORE applying this spec.
  #
  # CPU-ONLY, ZERO GPU, us-west (the caches are on the PVC at SDSC). 2 CPUs,
  # not 8: class-sum scoring is single-threaded Python. Generated by
  # scripts/build_anomaly_jobs.py --class-sum-rerun; do not hand-edit.
"""
CS_DEADLINE_COMMENT = """\
  # 36 h. ~8.5 h per model expected (measured locally and scaled by the pods'
  # measured rate; scripts/build_anomaly_jobs.py, CS_DEADLINE). The deadline
  # counts across the one retry and is TERMINAL: two attempts, 2x margin.
"""
CS_MERGE_HEADER = """\
  # MERGE THE CLASS-SUM RERUN: twenty models, only class_sum_matched (2026-09-28).
  #
  # The same merge, preflight and post-checks as eval-anomaly-merge-v4-raunav,
  # reading the twenty eval-anomaly-*-cs-raunav outputs instead. Written to a
  # NEW directory; anomaly_merged_v4 is not touched. The committed class_sum is
  # superseded only when anomaly_summary.py accepts this artifact beside it
  # (same draws; the committed class_sum reproduced at 17 classes).
  #
  # PIN {pin}, the tag the twenty per-model specs clone. Not tagged when this
  # was written. Generated by scripts/build_anomaly_jobs.py --class-sum-rerun.
  #
  # CPU-ONLY, ZERO GPU, minutes: this is JSON arithmetic, not a rerun.
"""


def _swap_comments(text: str, after: str, before: str, new: str) -> str:
    """Replace the comment lines between the unique lines `after` and `before`."""
    for s in (after, before):
        if text.count(s) != 1:
            raise SystemExit(f"FATAL: expected one {s!r} in the template, "
                             f"found {text.count(s)}")
    i, j = text.index(after) + len(after), text.index(before)
    if any(not ln.lstrip().startswith("#") for ln in text[i:j].splitlines()):
        raise SystemExit(f"FATAL: non-comment lines between {after!r} and "
                         f"{before!r}; only comments may be replaced")
    return text[:i] + new + text[j:]


def _live(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _check_live_diff(template: str, out: str, anchors, what: str) -> None:
    """Every executed line that moved must be one the anchors permit."""
    for n, was, now in changed_lines(_live(template), _live(out)):
        if not any(was.startswith(a) and now.startswith(a) for a, _ in anchors):
            raise SystemExit(f"FATAL: {what} executed line {n} moved outside the "
                             f"permitted set: {was.strip()!r} -> {now.strip()!r}")


def render_cs(m: Model, pin: str = CS_PIN) -> str:
    """The class-sum-only rerun spec for one model, template-derived and checked."""
    t = TEMPLATE_MODEL
    text = TEMPLATE_SPEC.read_text()
    out = _substitute(text, [
        (f"  name: eval-anomaly-{t.tag}-raunav\n",
         f"  name: eval-anomaly-{m.tag}-raunav\n", 1),
        ("  backoffLimit: 2\n", "  backoffLimit: 1\n", 1),
        ("  activeDeadlineSeconds: 259200\n",
         f"  activeDeadlineSeconds: {CS_DEADLINE}\n", 1),
        (f'--branch "{TEMPLATE_PIN}"', f'--branch "{pin}"', 1),
        (f"          a={t.run}; r={t.rung}; k={t.k}\n",
         f"          a={m.run}; r={m.rung}; k={m.k}\n", 1),
        (f"          OUT=/data/results/eval/anomaly_{t.tag}\n",
         f"          OUT=/data/results/eval/anomaly_{m.tag}\n", 1),
        (f"            --features {t.label}=${{d}} \\\n",
         f"            --features {m.label}=${{d}} \\\n", 1),
        (f"            --rungs {t.label}=${{r}} \\\n",
         f"            --rungs {m.label}=${{r}} \\\n", 1),
        ("            --out ${OUT} \\\n",
         f"            --out ${{OUT}} --families {CS_FAMILIES} \\\n", 1),
        ('memory: "32Gi", cpu: "8"', 'memory: "16Gi", cpu: "2"', 2),
    ])
    out = _swap_comments(out, "metadata:\n", "  name: eval-anomaly-",
                         CS_HEADER.format(pin=pin))
    out = _swap_comments(out, "  backoffLimit: 1\n", "  activeDeadlineSeconds: ",
                         CS_DEADLINE_COMMENT)
    _check_live_diff(text, out, CS_ALLOWED_ANCHORS, m.tag)
    if TEMPLATE_PIN in out:
        raise SystemExit(f"FATAL: {TEMPLATE_PIN!r} survives in {m.tag}")
    if m.label != t.label and t.label in _live(out):
        raise SystemExit(f"FATAL: {t.label!r} survives on an executed line of {m.tag}")
    return out


def render_cs_merge(pin: str = CS_PIN) -> str:
    """The v4 merge spec pointed at the twenty rerun outputs."""
    text = MERGE_V4.read_text()
    old = re.findall(r"^\s+(?:for d in )?/data/results/eval/anomaly_([\w.-]+)(?: \\|; do)$",
                     text, re.M)
    by_label = {m.label: m for m in CS_MODELS}
    if sorted(o.removesuffix("-v2") for o in old) != sorted(by_label):
        raise SystemExit(f"FATAL: the v4 merge reads {old}, not the twenty models")
    subs = [("  name: eval-anomaly-merge-v4-raunav\n",
             "  name: eval-anomaly-cs-merge-raunav\n", 1),
            ("  backoffLimit: 2\n", "  backoffLimit: 1\n", 1),
            (f'--branch "{MERGE_V4_PIN}"', f'--branch "{pin}"', 1),
            ("          OUT=/data/results/eval/anomaly_merged_v4\n",
             f"          OUT={CS_MERGE_OUT}\n", 1)]
    for o in old:
        new = by_label[o.removesuffix("-v2")].tag
        for end in (" \\\n", "; do\n"):
            if f"/data/results/eval/anomaly_{o}{end}" in text:
                subs.append((f"/data/results/eval/anomaly_{o}{end}",
                             f"/data/results/eval/anomaly_{new}{end}", 1))
    out = _substitute(text, subs)
    out = _swap_comments(out, "metadata:\n", "  name: eval-anomaly-",
                         CS_MERGE_HEADER.format(pin=pin))
    _check_live_diff(text, out, CS_MERGE_ANCHORS, CS_MERGE.name)
    if MERGE_V4_PIN in out:
        raise SystemExit(f"FATAL: {MERGE_V4_PIN!r} survives in {CS_MERGE.name}")
    return out


def _write(path: pathlib.Path, text: str, check_only: bool) -> str:
    if check_only:
        return "MISSING" if not path.exists() else (
            "ok" if path.read_text() == text else "DIFFERS")
    existed = path.exists()
    if existed and path.read_text() == text:
        return "unchanged"
    path.write_text(text)
    return "rewritten" if existed else "written"


def main_cs(a) -> int:
    pin = a.pin or CS_PIN
    widths = verify_head_widths(CS_MODELS)
    print(f"head widths verified for {len(widths)} models")
    verify_pin(pin, a.pin_not_yet_tagged, CS_NEEDED_AT_PIN)
    for m in CS_MODELS:
        print(f"  {m.job_spec.name:47s} "
              f"{_write(m.job_spec, render_cs(m, pin), a.check_only)}")
    print(f"  {CS_MERGE.name:47s} "
          f"{_write(CS_MERGE, render_cs_merge(pin), a.check_only)}")
    print(f"\nlaunch the {len(CS_MODELS)} per-model jobs (~8.5 h each on 2 CPUs), "
          f"then the merge; then anomaly_summary.py --class-sum-rerun")
    return 0


# ====================================== v1 anomaly by the checkpoint rule (v1err)
# (audit 2026-09-29, B4 and must-fix 3). Reads the per-jet output-layer scores
# extract_v2.py wrote for epochs 70-79 (scripts/build_extract_jobs.py --v1err):
# a model's anomaly directory when its GPU extraction finished, its diagnostics
# directory otherwise (head numbers only). experiments/EVAL/anomaly_heads.py
# redraws exactly the committed resamplings, so epoch 79 is checked against the
# committed anomaly run and its class-sum rerun (the /data copies are the files
# under experiments/FIGS/data, sha256 checked 2026-09-29).
V1ERR_PIN = "mtx-s1.75"      # anomaly_heads.py changed after mtx-s1.66 and mtx-s1.71
V1ERR_HEADS = "/data/results/eval/v1err/heads"
V1ERR_LADDER = [f"mtx-{a}-s{s}" if not (a == "l162" and s == 1) else "mtx-l162-s1b"
                for a in ("l188", "l162", "r42q1", "r16q1") for s in range(1, 6)]
V1ERR_MASS = [f"mtx-{a}mass-s{s}" for a in ("l162", "r16q1") for s in range(1, 6)]
V1ERR_RUNG = {"l188": "L188", "l162": "L162", "r42q1": "R42_Q1", "r16q1": "R16_Q1",
              "l162mass": "L162", "r16q1mass": "R16_Q1"}

V1ERR_SPEC = """apiVersion: batch/v1
kind: Job
metadata:
  name: anomaly-heads-v1err-raunav
  namespace: cms-ml
spec:
  backoffLimit: 6
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          halt () {{ rc=$?; [ $rc -ge 128 ] && exit $rc; echo "HALT: exit $rc, not retried"; exit 42; }}
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
          OUT=/data/results/eval/v1err/anomaly_heads/anomaly_heads.json
          [ -f ${{OUT}} ] && {{ echo "done by an earlier attempt"; exit 0; }}
          MODELS=""
          for spec in {specs}; do
            run=${{spec%%:*}}; rung=${{spec##*:}}
            d={heads}/anomaly/${{run}}
            ls ${{d}}/e0{{70..79}}/manifest.json >/dev/null 2>&1 || d={heads}/diag/${{run}}
            ls ${{d}}/e0{{70..79}}/manifest.json >/dev/null 2>&1 || {{ echo "FATAL: no heads for ${{run}}"; exit 42; }}
            MODELS="${{MODELS}} ${{run#mtx-}}=${{rung}}=${{d}}"
          done
          echo "models:${{MODELS}}"
          mkdir -p $(dirname ${{OUT}})
          python3 experiments/EVAL/anomaly_heads.py \\
            --models ${{MODELS}} \\
            --labels /data/results/eval/mtx-l188-s1/features_e79/label188.npy \\
            --committed /data/results/eval/anomaly_merged_v4/anomaly_results.json \\
            --committed-rerun /data/results/eval/anomaly_cs_merged_v1/anomaly_results.json \\
            --n-sig 2000 4000 --procs 15 \\
            --out ${{OUT}} || halt
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        resources:
          requests: {{ memory: "64Gi", cpu: "16", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "64Gi", cpu: "16", ephemeral-storage: "10Gi" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""


def v1err_spec() -> str:
    import importlib.util
    specs = " ".join(f"{r}:{V1ERR_RUNG[r.removeprefix('mtx-').rsplit('-s', 1)[0]]}"
                     for r in V1ERR_LADDER + V1ERR_MASS)
    s = importlib.util.spec_from_file_location("build_extract_jobs", ROOT / "scripts" / "build_extract_jobs.py")
    bx = importlib.util.module_from_spec(s)
    s.loader.exec_module(bx)
    return bx.storage_guarded("job-anomaly-heads-v1err-raunav.yaml",
                              V1ERR_SPEC.format(pin=V1ERR_PIN, specs=specs, heads=V1ERR_HEADS))


# =============================================================== v2 (2026-10-07)
# Layout and jobs: the module docstring. What PRESPEC fixes and this follows:
# the paper reports v2 (A7-A13); best70 is primary (A14); "every result" is also
# computed from wavg (A8); bestval is the frozen readouts' sensitivity check and
# best70_bn / bestval_bn sit beside them (A14, rule fired 2026-10-03); the pooled
# embedding is given to every model in the anomaly section (A14 design 5).
# Mahalanobis and kNN only (iad_hgb is dropped from the paper; the output ratio
# comes from the stored head scores, anomaly_heads.py), and N_sig 2000 (primary)
# and 4000 (reference), the injections anomaly_summary.py and anomaly_heads.py
# read; the N_sig = 0 ARGOS null is not rerun (+55 % cost, not reported).
#
# COST, MEASURED (2026-10-07, Apple M4 Pro, 8 threads, production sizes: 200,000
# + 200,000 QCD, 2,000 injected, 128-d Gaussian features, anomaly.run_one): kNN
# 42.3 s + Mahalanobis 8.9 s per resampling, 49 s for the null. One (run, tag,
# readout) unit is 6 signals x 2 injections x 10 resamplings, X->YY->bbb having
# too few jets at 4000: 110 x 51 s = 1.6 h here, 4.4 h at the 2.8x pod/laptop
# ratio measured for v1 (CS_DEADLINE above) -- an upper estimate, since the v1
# pods did not cap OpenMP at the CPUs requested and these do. The v2 prefix is
# the v1 cache's 2,000,000 jets and the draws are the same sizes, so the cost per
# resampling is v1's without IAD and the class sums: v1's ~50 h per model was
# 6 signals x 6 injections x 10 resamplings x four families.
# Per job (one run): 10 units (5 tags x 2 readouts; 6 where bestval is best70's
# epoch), 16-44 h; self-supervised 5 units, 8-22 h; an untrained trunk 2 units,
# 3-9 h. The study (V2_ANOMALY_ARMS: 42 runs with a head, 6 self-supervised, 5
# references): 42 x 10 + 6 x 5 + 5 x 2 = 460 units at most, 740-2,000 pod-hours
# at 8 CPUs (5,900-16,200 CPU-hours).
# Sharded by RUN: 53 specs rather than 460, and every unit resumable, so an
# eviction (ignored by the retry policy) or a recreated job costs at most one
# unit. The retry policy is V1ERR_SPEC's, which sets no deadline.
V2_PIN = "mtx-s2.00"
V2_NEEDED_AT_PIN = ["experiments/EVAL/anomaly.py", "experiments/EVAL/probe.py",
                    "experiments/EVAL/anomaly_merge.py", "experiments/EVAL/anomaly_heads.py",
                    "configs/labelmaps/rung_label_maps.v1.csv"]
V2_K8S = K8S / "v2" / "anomaly"
V2_ANOMALY = "/data/results/eval/v2/anomaly"
V2_TIER_SETS = (("t12", 2), ("t123", 3))


def _script_module(name: str, rel: str):
    import importlib.util
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@functools.cache
def _bx():
    return _script_module("build_extract_jobs", "scripts/build_extract_jobs.py")


@functools.cache
def _summary():
    return _script_module("anomaly_summary", "experiments/EVAL/anomaly_summary.py")


class V2Model(NamedTuple):
    model: str        # anomaly.py's arm name: the run without "mtx-" (v1's names, v1's draws)
    run: str          # its directory under build_extract_jobs.V2_OUT
    rung: str
    readouts: tuple
    head: bool        # an output layer on the tree: the output ratio is scored
    tier: int
    tags: tuple


def v2_models() -> list[V2Model]:
    """Every model the v2 anomaly study scores: the grid's runs of the arms
    anomaly_summary.v2_ladder selects (V2_ANOMALY_ARMS), so the jobs and the
    summary's cells are one list, and the untrained-trunk references (tag init)."""
    bx = _bx()
    S = _summary()
    cells = S.v2_ladder(bx.V2_GRID)[0]
    tier = {a["name"]: a["tier"] for a in json.loads(bx.V2_GRID.read_text())["arms"]}
    out = [V2Model(run.removeprefix("mtx-"), run, bx.v2_rung(arm),
                   ("features", "pooled") if k else ("pooled",),
                   bool(k) and bx.v2_rung(arm) != "none", tier[arm], tuple(bx.V2_CHECKPOINTS))
           for run, arm, k, _reg, _s in bx.v2_runs() if arm in cells]
    out += [V2Model(ref, ref, "none", ("features", "pooled"), False, tier[bx.V2_INIT_ARM],
                    (S.V2_INIT,)) for ref, _run in bx.v2_init_refs()]
    return out


V2_HEAD = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 6
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          halt () {{ rc=$?; [ $rc -ge 128 ] && exit $rc; echo "HALT: exit $rc, not retried"; exit 42; }}
          git clone --depth 1 --branch "{pin}" \\
            https://github.com/raunavm/transferlearningsophon.git \\
            /workspace/transferlearningsophon
          cd /workspace/transferlearningsophon
          git rev-parse HEAD
          export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
"""
V2_TAIL = """        volumeMounts:
        - {{ name: data, mountPath: /data }}
        resources:
          requests: {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "10Gi" }}
          limits:   {{ memory: "{mem}", cpu: "{cpu}", ephemeral-storage: "10Gi" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""
# The files anomaly.load_cache opens in a v2 cache; observers.npz is optional there.
V2_RUN_BODY = """          a={model}; r={rung}; src={src}/{run}; dst={dst}/{run}
          for tag in {tags}; do
            d=${{src}}/${{tag}}
            for f in label188.npy rows.npy manifest.json {files}; do
              [ -f "${{d}}/${{f}}" ] || {{ echo "FATAL: no ${{d}}/${{f}}"; exit 42; }}
            done
          done
          mkdir -p ${{dst}}
          for tag in {tags}; do
            d=${{src}}/${{tag}}
            if [ -L "${{d}}" ]; then ln -sfn "$(readlink ${{d}})" ${{dst}}/${{tag}}; continue; fi
            for ro in {readouts}; do
              OUT=${{dst}}/${{tag}}/${{ro}}
              grep -qs '"null_unmeasured"' ${{OUT}}/anomaly_results.json && {{ echo "done: ${{OUT}}"; continue; }}
              python3 experiments/EVAL/anomaly.py \\
                --features ${{a}}=${{d}} --rungs ${{a}}=${{r}} --readout ${{ro}} \\
                --families knn mahalanobis --n-sig 2000 4000 \\
                --n-bkg 200000 --n-template 200000 --trainings 10 \\
                --out ${{OUT}} || halt
            done
          done
          echo "=== done: ${{dst}} ==="
"""
V2_HEADS_BODY = """          OUT={out}
          [ -f ${{OUT}} ] && {{ echo "done by an earlier attempt"; exit 0; }}
          MODELS=""
          for spec in {specs}; do
            run=${{spec%%:*}}; rung=${{spec##*:}}
            for tag in {tags}; do
              [ -f "{src}/${{run}}/${{tag}}/head_scores.npz" ] || {{ echo "FATAL: no heads for ${{run}} at ${{tag}}"; exit 42; }}
            done
            MODELS="${{MODELS}} ${{run#mtx-}}=${{rung}}={src}/${{run}}"
          done
          mkdir -p $(dirname ${{OUT}})
          python3 experiments/EVAL/anomaly_heads.py \\
            --models ${{MODELS}} \\
            --labels {labels} \\
            --n-sig 2000 4000 --procs 15 \\
            --out ${{OUT}} || halt
"""
# Every merge (one tag, one readout) also carries the untrained-trunk references,
# scored once at tag init: the reference row beside each checkpoint's table.
V2_MERGE_BODY = """          units () {{ for r in $1; do echo {dst}/${{r}}/$2/$3; done; for r in {refs}; do echo {dst}/${{r}}/$3; done; }}
          for tag in {tags}; do
            for ro in features pooled; do
              [ ${{ro}} = features ] && runs="{feat}" || runs="{pool}"
              for u in $(units "${{runs}}" ${{tag}} ${{ro}}); do
                grep -qs '"null_unmeasured"' ${{u}}/anomaly_results.json || {{ echo "FATAL: ${{u}} absent or partial"; exit 42; }}
              done
            done
          done
          for tag in {tags}; do
            for ro in features pooled; do
              [ ${{ro}} = features ] && runs="{feat}" || runs="{pool}"
              OUT={out}/${{tag}}/${{ro}}
              [ -f ${{OUT}}/anomaly_results.json ] && {{ echo "done: ${{OUT}}"; continue; }}
              python3 experiments/EVAL/anomaly_merge.py --inputs $(units "${{runs}}" ${{tag}} ${{ro}}) --out ${{OUT}} || halt
            done
          done
"""


def _v2_spec(name: str, body: str, pin: str, mem: str, cpu: str, threads: int) -> str:
    """BLAS/OpenMP threads = the CPUs each process may use: nproc reports the node's."""
    text = (V2_HEAD.format(name=name, pin=pin, threads=threads) + body
            + V2_TAIL.format(mem=mem, cpu=cpu))
    return _bx().storage_guarded(f"job-{name}.yaml", text)


def render_v2_run(m: V2Model, pin: str = V2_PIN) -> str:
    """One model's anomaly.py units: each of its tags on each of its readouts."""
    body = V2_RUN_BODY.format(model=m.model, run=m.run, rung=m.rung, src=_bx().V2_OUT,
                              dst=V2_ANOMALY, tags=" ".join(m.tags), readouts=" ".join(m.readouts),
                              files=" ".join(f"{r}.npy" for r in m.readouts))
    return _v2_spec(f"eval-anomaly-v2-{m.model}-raunav", body, pin, "32Gi", "8", 8)


def render_v2_heads(tset: str, max_tier: int, pin: str = V2_PIN) -> str:
    """The output ratio of every model with an output layer on the tree, tiers <= max_tier."""
    ms = [m for m in v2_models() if m.head and m.tier <= max_tier]
    body = V2_HEADS_BODY.format(
        out=f"{_bx().V2_OUT}/anomaly_heads_{tset}/anomaly_heads.json", src=_bx().V2_OUT,
        specs=" ".join(f"{m.run}:{m.rung}" for m in ms), tags=" ".join(_bx().V2_CHECKPOINTS),
        labels=f"{_bx().V2_OUT}/{ms[0].run}/best70")
    return _v2_spec(f"eval-anomaly-v2-heads-{tset}-raunav", body, pin, "64Gi", "16", 1)


def render_v2_merge(tset: str, max_tier: int, pin: str = V2_PIN) -> str:
    """One merge per (tag, readout) over every model of tiers <= max_tier."""
    ms = [m for m in v2_models() if m.tier <= max_tier]
    runs = [m for m in ms if m.tags == tuple(_bx().V2_CHECKPOINTS)]
    refs = [m for m in ms if m not in runs]
    assert all(m.readouts == ("features", "pooled") for m in refs), refs
    body = V2_MERGE_BODY.format(
        tags=" ".join(_bx().V2_CHECKPOINTS), dst=V2_ANOMALY,
        out=f"{_bx().V2_OUT}/anomaly_merged_{tset}",
        refs=" ".join(f"{m.run}/{m.tags[0]}" for m in refs),
        feat=" ".join(m.run for m in runs if "features" in m.readouts),
        pool=" ".join(m.run for m in runs))
    return _v2_spec(f"eval-anomaly-v2-merge-{tset}-raunav", body, pin, "8Gi", "2", 1)


def v2_specs(pin: str = V2_PIN) -> dict[pathlib.Path, str]:
    out = {V2_K8S / f"job-eval-anomaly-v2-{m.model}-raunav.yaml": render_v2_run(m, pin)
           for m in v2_models()}
    for tset, max_tier in V2_TIER_SETS:
        out[V2_K8S / f"job-eval-anomaly-v2-heads-{tset}-raunav.yaml"] = render_v2_heads(tset, max_tier, pin)
        out[V2_K8S / f"job-eval-anomaly-v2-merge-{tset}-raunav.yaml"] = render_v2_merge(tset, max_tier, pin)
    return out


def main_v2(a) -> int:
    pin = a.pin or V2_PIN
    verify_pin(pin, a.pin_not_yet_tagged, V2_NEEDED_AT_PIN)
    if not a.check_only:
        V2_K8S.mkdir(parents=True, exist_ok=True)
    specs = v2_specs(pin)
    done = {p: _write(p, t, a.check_only) for p, t in specs.items()}
    stale = sorted(set(V2_K8S.glob("*.yaml")) - set(specs))
    for p, s in [*done.items(), *((p, "NOT RENDERED (stale)") for p in stale)]:
        print(f"  {p.name:52s} {s}")
    ms = v2_models()
    units = sum(len(m.tags) * len(m.readouts) for m in ms)
    print(f"\n{len(ms)} per-model jobs, {units} units at most (8 CPUs, 1.6-4.4 h per unit); "
          f"then, per tier set, the heads job (any time after extraction) and the merge")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="compare against the committed files, write nothing")
    ap.add_argument("--class-sum-rerun", action="store_true",
                    help="emit the class-sum-only rerun specs and their merge")
    ap.add_argument("--only", nargs="*", default=None, metavar="RUN_ID",
                    help="restrict to these run ids (default: all of MODELS)")
    ap.add_argument("--pin", default=None, help=f"default {PIN}, or {CS_PIN} "
                    "with --class-sum-rerun")
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--v1err", action="store_true",
                    help="emit ONLY the v1 anomaly-by-checkpoint-rule job (audit 2026-09-29)")
    ap.add_argument("--v2", action="store_true",
                    help=f"emit the v2 specs under {V2_K8S.relative_to(ROOT)} (default pin {V2_PIN})")
    a = ap.parse_args(argv)
    if a.v2:
        return main_v2(a)
    if a.v1err:
        out = ROOT / "experiments" / "EVAL" / "k8s" / "job-anomaly-heads-v1err-raunav.yaml"
        out.write_text(v1err_spec())
        print(f"wrote {out}")
        return 0
    if a.class_sum_rerun:
        return main_cs(a)
    a.pin = a.pin or PIN

    widths = verify_head_widths()
    print("head widths, read from each model's own training spec and "
          "cross-checked against the committed label map:")
    for rung in dict.fromkeys(m.rung for m in MODELS):
        ks = {k for r, k in widths.items() if r.startswith(
            f"mtx-{rung.lower().replace('_', '')}-")}
        print(f"  {rung:8s} K={ks.pop()}")

    verify_pin(a.pin, a.pin_not_yet_tagged)

    # KEYED ON `tag`, NOT ON `run`. Two models can share a run directory -- a
    # re-run scores the SAME checkpoint, which is why `run` is deliberately not
    # versioned -- so `{m.run: m}` silently dropped one of them and made
    # `--only` unable to name the re-run at all. `tag` is the versioned name and
    # is unique by construction; the assert says so rather than trusting it.
    known = {m.tag: m for m in MODELS}
    assert len(known) == len(MODELS), "two models share a tag: " + str(
        sorted({m.tag for m in MODELS if [x.tag for x in MODELS].count(m.tag) > 1}))
    chosen = MODELS
    if a.only is not None:
        unknown = [r for r in a.only if r not in known]
        if unknown:
            sys.exit(f"FATAL: unknown run ids {sorted(unknown)}. "
                     f"Known: {sorted(known)}")
        chosen = [known[r] for r in a.only]

    done = {m.tag: build(m, a.pin, a.check_only) for m in chosen}
    for wave, runs in LAUNCH_WAVES:
        print(f"\nwave {wave}:")
        # A wave names run directories, and a re-run adds a second model under
        # the same one, so this resolves run -> every model that scores it.
        for run in runs:
            for m in (x for x in MODELS if x.run == run and x.tag in done):
                print(f"  {m.job_spec.name:47s} {done[m.tag]}")
    print("\nmerge when every wave is Complete: "
          "job-eval-anomaly-merge-v4-raunav.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
