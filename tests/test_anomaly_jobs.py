"""The anomaly wave's specs must be the template plus the model's identity.

WHAT THIS FILE IS PROTECTING. Each spec is ~50 h of an 8-CPU pod, and the study
holds everything except the model fixed: the 72 h deadline measured off
eval-anomaly-r16q1-s3's own log timestamps, the precondition loop that matches
probe.load_arm, the region pin, the 32Gi/8-CPU request, --n-bkg / --n-template /
--trainings. A spec that quietly differs in one of those still runs to
completion and writes a plausible number; anomaly_merge's shared-key check would
then refuse the whole merge without saying which spec was wrong, after the
compute had been spent.

So the rule is a LINE-BY-LINE DIFF against the committed template, and only the
model's identity and the pinned tag may move. That also catches the case the
builder cannot: someone hand-editing one of the twelve files afterwards.
"""
import csv
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "EVAL" / "k8s"
MERGE_V4 = K8S / "job-eval-anomaly-merge-v4-raunav.yaml"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod("build_anomaly_jobs", "scripts/build_anomaly_jobs.py")

# The models whose anomaly runs are USABLE as v4 inputs without this wave.
# Written out rather than derived, because the point of the coverage test below
# is to compare the builder's list against the plan, and deriving both from one
# source would make the comparison vacuous.
#
# l162-s1b is DELIBERATELY ABSENT. It was scored on 2026-09-09, but that
# artifact holds five of six signals, so it is not usable and the wave re-runs
# it -- which is why it appears in MODELS instead.
ALREADY = {
    "r16q1-s2": "R16_Q1", "r16q1-s3": "R16_Q1",
    "r16q1-s4": "R16_Q1",                       # merged as anomaly_merged_v3
    "l162-s2": "L162", "l162-s3": "L162", "l162-s4": "L162",
    "r16q1-s5": "R16_Q1",                       # in flight since 2026-09-17
}
# The approved plan (2026-09-18): four granularities x five pretraining seeds,
# the seed being the unit of inference.
PLANNED_LEVELS = {"L188": 5, "L162": 5, "R42_Q1": 5, "R16_Q1": 5}


# --------------------------------------------------------------- the template

def test_the_template_is_an_in_flight_spec_and_still_carries_its_anchors():
    """Every substitution site, counted. If the template is edited so that one
    of these moves or repeats, the builder must fail loudly rather than rewrite
    the wrong line -- and this says so before the builder is even run."""
    text = B.TEMPLATE_SPEC.read_text()
    t = B.TEMPLATE_MODEL
    assert t.label == "l162-s2" and t.run == "mtx-l162-s2" and t.k == 162
    assert t.tag == t.label, "the template is a first run, not a re-run"
    for frag, n in [(f"  # PIN {B.TEMPLATE_PIN}.\n", 1),
                    (f"  name: eval-anomaly-{t.tag}-raunav\n", 1),
                    (f'--branch "{B.TEMPLATE_PIN}"', 1),
                    (f"          a={t.run}; r={t.rung}; k={t.k}\n", 1),
                    (f"          OUT=/data/results/eval/anomaly_{t.tag}\n", 1),
                    (f"            --features {t.label}=${{d}} \\\n", 1),
                    (f"            --rungs {t.label}=${{r}} \\\n", 1)]:
        assert text.count(frag) == n, f"{frag!r} occurs {text.count(frag)} times"
    assert text.count(t.label) == 5, (
        "the template's own model name must occur exactly on its five live "
        "sites; a sixth occurrence means a comment now names it and a "
        "token-level rewrite would hit it")
    assert text.count(B.TEMPLATE_PIN) == 2, (
        "the pin must appear exactly twice -- the header and the clone line -- "
        "and both are substituted")


def test_the_template_header_no_longer_contradicts_its_own_clone_line():
    """THE DEFECT THE HEADER ANCHOR WAS ADDED FOR. Until 2026-09-19 the header
    cited a tag two moves behind the one the job actually clones, so it
    documented a run that never happened."""
    text = B.TEMPLATE_SPEC.read_text()
    header = re.search(r"^  # PIN (\S+?)\.$", text, re.M).group(1)
    cloned = re.search(r'--branch "([^"]+)"', text).group(1)
    assert header == cloned == B.TEMPLATE_PIN, (header, cloned)
    assert "mtx-s1.36" not in text, "the stale tag is still cited"


def test_the_template_no_longer_claims_the_162_class_seed_1_model_is_not_rerun():
    """It says so of the 2026-09-12 wave, which was true then. The wave this
    template now seeds DOES re-run it, and a spec inheriting the old sentence
    would contradict the job standing next to it."""
    text = B.TEMPLATE_SPEC.read_text()
    assert "is NOT re-run" not in text
    assert "eval-anomaly-l162-s1b-v2-raunav" in text
    assert "overwritten or deleted" in text


# ------------------------------------------------- the committed specs vs it

@pytest.mark.parametrize("m", B.MODELS, ids=lambda m: m.tag)
def test_the_committed_spec_is_exactly_what_the_builder_renders(m):
    assert m.job_spec.exists(), f"{m.job_spec.name} was never written"
    assert m.job_spec.read_text() == B.render(m), (
        f"{m.job_spec.name} has been hand-edited away from the builder. Either "
        f"re-run scripts/build_anomaly_jobs.py or move the change into the "
        f"template so all twelve get it.")


@pytest.mark.parametrize("m", B.MODELS, ids=lambda m: m.tag)
def test_the_spec_differs_from_the_template_only_on_the_allowed_lines(m):
    """The whole contract, checked against the FILE ON DISK rather than the
    builder's return value, so a later hand-edit is caught too."""
    moved = B.changed_lines(B.TEMPLATE_SPEC.read_text(), m.job_spec.read_text())
    got = [(n, was.strip()) for n, was, _ in moved]
    assert len(moved) == len(B.ALLOWED_ANCHORS), (
        f"{m.job_spec.name} differs from the template on {len(moved)} lines, "
        f"only {len(B.ALLOWED_ANCHORS)} are permitted: {got}")
    for (lineno, was, now), (anchor, what) in zip(moved, B.ALLOWED_ANCHORS):
        assert was.startswith(anchor), (
            f"{m.job_spec.name} line {lineno} should be {what} but is {was!r}")
        assert now.startswith(anchor), f"line {lineno} lost its anchor: {now!r}"


@pytest.mark.parametrize("m", B.MODELS, ids=lambda m: m.tag)
def test_the_template_model_does_not_survive_on_an_executed_line(m):
    text = m.job_spec.read_text()
    live = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    for token in (B.TEMPLATE_MODEL.label, B.TEMPLATE_MODEL.run):
        assert not [ln for ln in live if token in ln], (
            f"{m.job_spec.name} still runs against {token!r}")
    assert any(f"a={m.run}; r={m.rung}; k={m.k}" in ln for ln in live)
    assert any(f'--branch "{B.PIN}"' in ln for ln in live)
    # Stricter for the pin: gone from the COMMENTS too, or the header would
    # again name a tag the job does not clone.
    assert B.TEMPLATE_PIN not in text
    assert f"  # PIN {B.PIN}." in text


@pytest.mark.parametrize("m", B.MODELS, ids=lambda m: m.tag)
def test_a_rerun_versions_its_job_and_its_output_but_not_its_model(m):
    """Both halves are load-bearing. A versioned OUTPUT is what keeps the older
    artifact -- nothing in /data/results is ever overwritten or deleted. An
    UNVERSIONED model name is what stops the re-run reading as an extra seed at
    its granularity, which is the miscount the seed-balanced regret exists for."""
    text = m.job_spec.read_text()
    assert f"          OUT=/data/results/eval/anomaly_{m.tag}\n" in text
    assert f"  name: eval-anomaly-{m.tag}-raunav\n" in text
    assert f"--features {m.label}=" in text and f"--rungs {m.label}=" in text
    if m.version:
        assert m.tag.endswith(f"-{m.version}") and m.tag != m.label
        assert f"anomaly_{m.label}\n" not in text, (
            "a re-run must not write the unversioned directory")


# ------------------------------------------------------------- the head width

def test_the_head_width_is_read_from_each_models_own_training_spec():
    """Not assumed, and not taken from the level's NAME. The spec states it
    twice -- once to seed_weaver, once to the network -- and both must agree."""
    for m in B.MODELS:
        assert B.head_width_from_training_spec(m) == m.k, (
            f"{m.run} declares K={m.k}; {m.training_spec.name} disagrees")


def test_the_43_class_level_is_43_and_not_42():
    """THE ONE THAT IS NOT WHAT THE NAME SAYS. The level is named for its 42
    RESONANT groups and the head also carries the QCD group. K=42 would make
    logits_from_features refuse the checkpoint -- a clean failure, but only
    after the clone, and only for this wave."""
    r42 = [m for m in B.MODELS if m.rung == "R42_Q1"]
    assert r42 and {m.k for m in r42} == {43}
    named = int(re.search(r"R(\d+)_Q1", "R42_Q1").group(1))
    assert named == 42 and r42[0].k == named + 1, (
        "the extra output is the QCD group; if that ever stops being true the "
        "arithmetic below it changes too")


def test_the_head_widths_agree_with_the_committed_label_map():
    """A second, independent source for the same integer. The training spec says
    what the head was BUILT with; the tree says what the vocabulary CONTAINS,
    and the score indexes the logits by node id -- a mismatch would sum the
    wrong columns rather than crash."""
    with (ROOT / "configs/labelmaps/rung_label_maps.v1.csv").open() as f:
        rows = list(csv.DictReader(f))
    for level, k in {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}.items():
        assert len({int(r[level]) for r in rows}) == k
    assert B.verify_head_widths() == {m.run: m.k for m in B.MODELS}


# ------------------------------------------------------------------ coverage

def test_the_wave_closes_exactly_the_gap_the_plan_leaves():
    """Thirteen specs, and thirteen is not a target -- it is what four
    granularities at five seeds leaves once the seven USABLE existing runs are
    subtracted. Every model appears exactly once, so no granularity gains a
    phantom seed from the re-run."""
    have = {(lvl, lbl) for lbl, lvl in ALREADY.items()}
    new = {(m.rung, m.label) for m in B.MODELS}
    assert len(new) == len(B.MODELS), "a model is listed twice"
    assert not (have & new), f"already covered: {sorted(have & new)}"
    total = have | new
    per_level = {lvl: sum(1 for l, _ in total if l == lvl)
                 for lvl in PLANNED_LEVELS}
    assert per_level == PLANNED_LEVELS, per_level
    assert len(total) == 20 and len(B.MODELS) == 13


def test_the_launch_waves_cover_every_model_exactly_once():
    """The staging is data, so the builder can print it and this can check it.
    A model missing from the waves is one that never gets launched."""
    ordered = [r for _w, runs in B.LAUNCH_WAVES for r in runs]
    assert len(ordered) == len(set(ordered)) == len(B.MODELS)
    assert set(ordered) == {m.run for m in B.MODELS}
    # Waves A and B must be the two granularities with NO coverage, or the
    # three-seed interim read is not available at the end of B.
    first_six = ordered[:6]
    assert {m.rung for m in B.MODELS if m.run in first_six} == {"L188", "R42_Q1"}
    after_b = {lvl: sum(1 for l, _ in
                        ({(lv, lb) for lb, lv in ALREADY.items()} |
                         {(m.rung, m.label) for m in B.MODELS
                          if m.run in first_six}) if l == lvl)
               for lvl in PLANNED_LEVELS}
    assert all(n >= 3 for n in after_b.values()), (
        f"after wave B not every granularity has three seeds: {after_b}")


def test_no_two_specs_share_a_job_name_or_an_output_directory():
    names, outs = set(), set()
    for m in B.MODELS:
        text = m.job_spec.read_text()
        name = re.search(r"^  name: (\S+)$", text, re.M).group(1)
        out = re.search(r"^          OUT=(\S+)$", text, re.M).group(1)
        assert name not in names and out not in outs
        assert "raunav" in name, "every job name must carry raunav"
        names.add(name)
        outs.add(out)


def test_no_spec_writes_over_a_directory_another_spec_already_owns():
    """Overwriting a results directory is RED, and the failure mode is silent:
    the job runs, writes, and the earlier model's table is gone."""
    mine = {re.search(r"^          OUT=(\S+)$", m.job_spec.read_text(), re.M)
            .group(1) for m in B.MODELS}
    theirs = set()
    for spec in K8S.glob("job-*.yaml"):
        if spec.name in {m.job_spec.name for m in B.MODELS}:
            continue
        theirs.update(re.findall(r"^          OUT=(\S+)$", spec.read_text(), re.M))
    assert not (mine & theirs), f"output collision: {sorted(mine & theirs)}"
    assert "/data/results/eval/anomaly" not in mine, (
        "that directory holds the 2026-09-09 l162-s1b run")


# ----------------------------------------------------------------- the guards

def test_an_unexpected_occurrence_count_is_fatal_rather_than_a_partial_rewrite():
    with pytest.raises(SystemExit, match="expected 3 occurrence"):
        B._substitute("a b a", [("a", "z", 3)])
    assert B._substitute("a b a", [("a", "z", 2)]) == "z b z"


def test_a_pin_with_no_tag_is_refused_unless_it_is_declared_not_yet_tagged():
    """The pod clones a TAG. A pin that names nothing would clone-fail fifty
    times, and a pin that names a tag PREDATING the code would be worse: it
    would run and write a number computed by old code."""
    with pytest.raises(SystemExit, match="does not exist"):
        B.verify_pin("mtx-s0.0-does-not-exist", allow_untagged=False)
    B.verify_pin("mtx-s0.0-does-not-exist", allow_untagged=True)


def test_the_logits_are_built_into_the_feature_cache_on_the_volume():
    """A DISK FACT, not a style point. The spec points the logit builder at
    ${d}, which is on the PVC, and the builder writes logits.npy INSIDE its
    --features directory -- so the file persists on the volume rather than
    living in the pod. At 2,000,000 jets that is ~1.5 GB per 188-class model,
    and this wave adds twelve of them."""
    src = (ROOT / "experiments/EVAL/logits_from_features.py").read_text()
    assert 'fpath, lpath = d / "features.npy", d / "logits.npy"' in src, (
        "the logit file's location moved; the volume budget below depends on it")
    for m in B.MODELS:
        text = m.job_spec.read_text()
        assert "d=/data/results/eval/${a}/features_e79" in text
        assert "--features ${d} --num-classes ${k}" in text


def test_the_logit_build_is_additive_and_never_rebuilds_or_overwrites():
    """TWO INDEPENDENT GUARDS, which is why this is worth a test. The spec
    skips the build when the file is already there, and if that guard were ever
    removed the builder itself still refuses to replace an existing file -- no
    spec passes --overwrite. So a re-run costs nothing and cannot damage a cache
    another model's scores were computed from."""
    src = (ROOT / "experiments/EVAL/logits_from_features.py").read_text()
    assert "if lpath.exists() and not a.overwrite:" in src
    assert "already exists; pass --overwrite to replace it" in src
    for m in B.MODELS:
        text = m.job_spec.read_text()
        assert 'if [ ! -f "${d}/logits.npy" ]; then' in text, (
            f"{m.job_spec.name} would rebuild logits that already exist")
        assert "--overwrite" not in text, (
            f"{m.job_spec.name} would replace an existing logit file")


# -------------------------------------------------------------- the v4 merge

def test_the_merge_reads_every_model_and_writes_somewhere_new():
    text = MERGE_V4.read_text()
    want = {f"/data/results/eval/anomaly_{lbl}" for lbl in ALREADY} | {
        f"/data/results/eval/anomaly_{m.tag}" for m in B.MODELS}
    # The loop header carries the first path and the last line ends `; do`, so
    # both endings are matched explicitly rather than with an optional tail --
    # an optional tail would also match the OUT= line and quietly pass.
    got = set(re.findall(
        r"^\s+(?:for d in )?(/data/results/eval/anomaly[\w.-]*)(?: \\|; do)$",
        text, re.M))
    assert got == want, f"missing {sorted(want - got)}, extra {sorted(got - want)}"
    assert len(want) == 20
    assert "OUT=/data/results/eval/anomaly_merged_v4" in text
    for old in ("anomaly_merged_v3", "anomaly_merged_v2"):
        assert f"OUT=/data/results/eval/{old}" not in text, (
            "v1..v3 are the only record of the table at their own tags")
    assert 'refusing to overwrite' in text
    # The 30-cell 2026-09-09 artifact stays on disk and is simply not read.
    # Listing it BESIDE the re-run would be the duplicated model merge() refuses.
    assert "/data/results/eval/anomaly \\" not in text
    assert "/data/results/eval/anomaly_l162-s1b-v2" in text
    assert "not deleted and not overwritten" in text.lower()


def test_the_merge_asserts_all_four_things_before_it_writes():
    text = MERGE_V4.read_text()
    body = text[:text.index("OUT=/data/results/eval/anomaly_merged_v4")]
    for marker, what in [
            ("mg.merge(payloads)", "shared configuration keys / duplicated model"),
            ("do not contribute the same", "identical (signal, N_sig) cells"),
            ("mg.backfill_classes_removed", "classes_removed against the tree"),
            ("seeds per granularity", "equal seed counts per granularity")]:
        assert marker in body, f"the preflight does not check {what}"
    assert "preflight passed" in body
    assert "signals_with_unequal_seeds_per_rung" in text, (
        "the flag v3 raised must be printed either way")


def test_the_merge_does_not_read_as_a_feature_cache_consumer():
    """It merges per-model JSON and opens no cache, so prechecking features.npy
    would be a guard asserting a precondition unrelated to the job -- the
    inversion tests/test_spec_preconditions.py exists to stop. The trap is a
    header comment naming the scoring module, so the module is reached through
    the merge module's own loader instead."""
    live = "\n".join(l for l in MERGE_V4.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    for name in ("anomaly.py", "probe.py", "label_recovery.py"):
        assert name not in live, f"{name} on an executed line"
    assert "mg._anomaly()" in live


def test_the_merge_and_the_wave_clone_one_tag():
    pins = {re.search(r'--branch "([^"]+)"', p.read_text()).group(1)
            for p in [MERGE_V4] + [m.job_spec for m in B.MODELS]}
    assert pins == {B.PIN}, pins
