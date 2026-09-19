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

Run:  python3 scripts/build_anomaly_jobs.py [--check-only] [--only RUN_ID ...]
      python3 scripts/build_anomaly_jobs.py --pin-not-yet-tagged
"""
from __future__ import annotations

import argparse
import difflib
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
    *[Model("L188", str(s), 188) for s in range(1, 6)],
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


def verify_pin(pin: str, allow_untagged: bool) -> None:
    """The pod clones a TAG, so check the TAG's tree, not the working tree."""
    tagged = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{pin}"],
        cwd=ROOT, capture_output=True).returncode == 0
    if not tagged:
        if not allow_untagged:
            sys.exit(f"FATAL: tag {pin} does not exist. Pass "
                     f"--pin-not-yet-tagged if it is about to be created on a "
                     f"commit carrying {len(NEEDED_AT_PIN)} files, and create it "
                     f"BEFORE applying any spec that clones it.")
        gone = [p for p in NEEDED_AT_PIN if not (ROOT / p).exists()]
        if gone:
            sys.exit(f"FATAL: {gone} not in the working tree either")
        print(f"WARNING: tag {pin} DOES NOT EXIST YET. All {len(NEEDED_AT_PIN)} "
              f"files are in the working tree; create the tag on a commit that "
              f"has them before applying anything.")
        return
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", pin, "--", *NEEDED_AT_PIN],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    missing = [p for p in NEEDED_AT_PIN if p not in listed]
    if missing:
        sys.exit(f"FATAL: tag {pin} does not contain {missing}. The pod clones "
                 f"the TAG, so the job would die after paying for the clone.")
    print(f"pin {pin} verified to contain all {len(NEEDED_AT_PIN)} files the "
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="compare against the committed files, write nothing")
    ap.add_argument("--only", nargs="*", default=None, metavar="RUN_ID",
                    help="restrict to these run ids (default: all of MODELS)")
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    a = ap.parse_args(argv)

    widths = verify_head_widths()
    print("head widths, read from each model's own training spec and "
          "cross-checked against the committed label map:")
    for rung in dict.fromkeys(m.rung for m in MODELS):
        ks = {k for r, k in widths.items() if r.startswith(
            f"mtx-{rung.lower().replace('_', '')}-")}
        print(f"  {rung:8s} K={ks.pop()}")

    verify_pin(a.pin, a.pin_not_yet_tagged)

    known = {m.run: m for m in MODELS}
    chosen = MODELS
    if a.only is not None:
        unknown = [r for r in a.only if r not in known]
        if unknown:
            sys.exit(f"FATAL: unknown run ids {unknown}")
        chosen = [known[r] for r in a.only]

    done = {m.run: build(m, a.pin, a.check_only) for m in chosen}
    for wave, runs in LAUNCH_WAVES:
        print(f"\nwave {wave}:")
        for run in runs:
            if run in done:
                print(f"  {known[run].job_spec.name:47s} {done[run]}")
    print("\nmerge when every wave is Complete: "
          "job-eval-anomaly-merge-v4-raunav.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
