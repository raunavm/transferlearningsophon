"""The wave-2 subset rebuild, emitted under new names rather than by repinning.

DECISIONS_PENDING item 33 leaves two things open, and this covers the second:
"Rebuilding the subsets to add N=1e3 re-runs two CPU jobs whose specs currently
pin mtx-s1.10. Regenerating them moves the pin on jobs that already ran; the
WINDOW_PIN precedent in scripts/build_extract_jobs.py says emit them under new
names instead."

The failure this guards against is quiet: a bare regeneration rewrites
job-ft-subsets-jc2-raunav.yaml, which is the provenance record for the subsets
ALREADY on disk (docs/RECORD.md), and detaches a completed run from the code
that produced it. Nothing errors, and `kubectl apply` on a completed Job does
not re-run it either -- so the rebuild silently does not happen.
"""
import importlib.util
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

W1 = {"job-ft-subsets-jc2-raunav.yaml", "job-ft-subsets-jc1-raunav.yaml",
      "job-ft-subsets-bench-raunav.yaml", "job-ft-legs-bench-raunav.yaml",
      "job-ft-smoke-raunav.yaml", "job-ft-legs-raunav.yaml"}
SUBSETS = {"job-ft-subsets-jc2-w2-raunav.yaml", "job-ft-subsets-jc1-w2-raunav.yaml"}
W2 = SUBSETS | {"job-ft-legs-w2-raunav.yaml"}


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_ft_jobs", ROOT / "scripts" / "build_ft_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod()


@pytest.fixture(scope="module")
def w2():
    return B.build("mtx-s1.41", wave2=True)


@pytest.fixture(scope="module")
def w1():
    return B.build("mtx-s1.41")


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _live(text):
    """The args with comments stripped. These specs DISCUSS the defect they
    avoid -- the header quotes the literal string `already built` -- so a bare
    scan of the body matches the warning about the bug and reports the bug.
    scripts/add_autoresume.py strips comments for exactly this reason."""
    return "\n".join(ln for ln in _args(text).splitlines()
                      if not ln.strip().startswith("#"))


# ------------------------------------------------- the two sets stay disjoint

def test_wave2_emits_the_rebuilds_and_the_legs_and_nothing_else(w2):
    assert set(w2) == W2


def test_wave2_can_never_write_a_wave_one_filename(w2):
    """THE LOAD-BEARING TEST. If a wave-1 name ever appears here, running
    --wave2 rewrites the provenance record item 33 is protecting."""
    assert not (set(w2) & W1)


def test_a_bare_build_emits_no_wave_two_spec(w1):
    assert set(w1) == W1
    assert not (set(w1) & W2)


def test_every_wave2_job_name_carries_raunav(w2):
    for name, text in w2.items():
        assert yaml.safe_load(text)["metadata"]["name"].endswith("-raunav"), name


# ------------------------------------------------------------ what they ask for

def test_both_rebuilds_request_all_four_sizes(w2):
    """N=1e3 is the point of the wave. Item 25 added it to SIZES and the
    builders were left asking for three sizes, so it was never written."""
    for name in SUBSETS:
        text = w2[name]
        assert "--sizes 1000 10000 100000 1000000" in _live(text), name


def test_the_rebuilds_carry_no_bare_done_short_circuit(w2):
    """The exact defect item 33 records: `[ -f DONE ] && exit 0` in front of
    make_subsets.py's careful manifest comparison produced a confident green
    while the N=1e3 subsets were never written."""
    for name in SUBSETS:
        live = _live(w2[name])
        assert "already built" not in live, name
        assert "exit 0" not in live.split("make_subsets")[0], name
        assert "${OUT}/manifest.json" in live, name


def test_the_rebuilds_grow_the_shared_data_dirs_in_place(w2):
    """The subsets are DATA, used by both waves, and growing downward keeps
    every existing file byte-identical. Only the legs' RESULTS need a w2 tree;
    copying the subsets would double ~100 GB for no gain and would break the
    nesting property the design rests on."""
    assert "OUT=/data/finetune/jc2" in _live(w2["job-ft-subsets-jc2-w2-raunav.yaml"])
    assert "OUT=/data/finetune/jc1" in _live(w2["job-ft-subsets-jc1-w2-raunav.yaml"])


def test_the_rebuilds_take_no_gpu(w2):
    """Staging jobs held a GPU once by inheriting a template. These are CPU."""
    for name in SUBSETS:
        res = yaml.safe_load(w2[name])["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "nvidia.com/gpu" not in res["limits"], name


def test_the_rebuilds_pin_the_tag_they_were_generated_for(w2):
    for name, text in w2.items():
        env = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["env"]
        assert {"name": "REPO_REF", "value": "mtx-s1.41"} in env, name


# ------------------------------------------------------------------ mechanics

def test_generation_is_idempotent(w2):
    """A second run must not produce a spec that differs from the one on disk,
    or the file and the generator disagree about what ran."""
    assert B.build("mtx-s1.41", wave2=True) == w2


def test_the_specs_on_disk_match_the_generator(w2):
    """These are committed; a hand-edit here would be silently reverted by the
    next regeneration, which is how the legs lost a pin on 2026-09-07."""
    for name, text in w2.items():
        p = ROOT / "experiments" / "FT" / "k8s" / name
        assert p.exists(), f"{name} not written"
        assert p.read_text() == text, f"{name} on disk differs from the generator"


# ------------------------------------------------------------ the wave-2 legs

LEGS_W2 = "job-ft-legs-w2-raunav.yaml"


def test_wave2_emits_the_legs_spec(w2):
    assert LEGS_W2 in w2


def test_the_legs_write_to_their_own_root(w2):
    """THE LOAD-BEARING TEST for item 33's first open sub-item. Wave 1's 108
    cells are keyed by the same init names and sizes, so a shared root makes
    every wave-2 cell hit `[ -f DONE ]` and skip -- the wave would appear to
    run, finish in minutes, and produce nothing."""
    live = _live(w2[LEGS_W2])
    assert "ROOT_OUT=/data/results/ft/w2" in live
    assert "ROOT_OUT=/data/results/ft\n" not in live


def test_the_legs_load_the_last_epoch_not_the_best_epoch(w2):
    """Item 18 fixed the paper's pretraining checkpoint to the last epoch. Wave
    1 loaded net_best_epoch_state.pt -- epochs 74/64/76/78 -- and is a pilot."""
    live = _live(w2[LEGS_W2])
    assert "net_epoch-79_state.pt" in live
    assert "mtx-r16q1-s2/net_best_epoch_state.pt" not in live


def test_the_legs_run_all_four_sizes(w2):
    assert "for N in 1000 10000 100000 1000000" in _live(w2[LEGS_W2])


# ------------------------------------------------- ParT's recipe, item 32(b)

def test_weight_decay_reaches_every_arm_including_scratch(w2):
    """Item 33 declined option B because a wave-2 table without scratch has no
    internal reference row. Putting the decay in COMMON is what makes scratch
    get the same optimiser change as everything else."""
    live = _live(w2[LEGS_W2])
    common = next(ln for ln in live.splitlines() if ln.strip().startswith("COMMON="))
    assert "--optimizer-option weight_decay 0.01" in common


def test_the_head_multiplier_is_emitted_exactly_as_the_working_bench_spec(w2):
    """I GOT THIS WRONG ONCE. The replacement over-escaped and emitted
    `mod\\\\.fc\\\\..*`, which as a Python regex matches a literal backslash and
    would have matched NO parameter -- so lr_mult would silently not apply and
    every pretrained cell would have run wave 1's protocol. Pinned against the
    bench spec, which is known to work, so the two cannot drift."""
    bench = (ROOT / "experiments" / "FT" / "k8s"
             / "job-ft-legs-bench-raunav.yaml").read_text()
    line = next(ln.strip() for ln in bench.splitlines()
                if ln.strip().startswith("HEAD_MULT=("))
    assert line in _live(w2[LEGS_W2]), (
        "the wave-2 head multiplier does not match the bench spec's working form")
    assert "mod\\\\.fc" not in _live(w2[LEGS_W2]), "over-escaped: matches nothing"


def test_scratch_gets_no_multiplier_and_pretrained_does(w2):
    """5e-4 is the scratch rate, not 5e-3: there is no pretrained trunk to hold
    back, so multiplying scratch's head by 50 would just be a different run."""
    live = _live(w2[LEGS_W2])
    for ln in live.splitlines():
        if "LOAD=\"--load-model-weights" in ln:
            assert 'MULT=("${HEAD_MULT[@]}")' in ln and "MULT=()" in ln
            assert "LR=1e-4" in ln and "LR=5e-4" in ln


def test_the_multiplier_is_verified_in_weavers_own_log_on_both_legs(w2):
    """If lr_mult silently fails to match, the cell trains its fresh head at the
    trunk rate -- wave 1's protocol, the exact thing wave 2 replaces -- and the
    output is indistinguishable from a correct run."""
    assert _live(w2[LEGS_W2]).count(
        'grep -q "Parameters with lr multiplied by 50"') == 2


def test_the_manifest_records_the_recipe_not_just_the_rate(w2):
    """docs/RECORD.md: the spec is provenance. `lr=1e-4` alone does not say the
    head ran at 5e-3, and wave 1's manifests say exactly that same `lr=1e-4`."""
    live = _live(w2[LEGS_W2])
    assert live.count("head_lr_mult=50 weight_decay=0.01 wave=2") == 2


def test_the_scheduler_is_deliberately_not_changed(w2):
    """LEGS_BENCH passes --lr-scheduler none because the published BENCHMARK
    recipe wants a constant rate. Item 32(b) adopts only lr_mult and weight
    decay, and the point is to match OUR LR sweep, which ran weaver's default
    flat+decay. Adding the constant schedule would re-open the protocol gap in
    a new place while looking like it closed it."""
    assert "--lr-scheduler none" not in _live(w2[LEGS_W2])


# --------------------------------------- the derivation refuses to drift quietly

def test_the_derivation_raises_if_wave_one_changes(monkeypatch):
    """The whole reason wave 2 is DERIVED rather than copied. If LEGS gains a
    guard and the substitution stops matching, this must fail loudly instead of
    emitting a wave-2 spec silently missing the recipe."""
    monkeypatch.setattr(B, "LEGS", B.LEGS.replace(
        "ROOT_OUT=/data/results/ft\n", "ROOT_OUT=/data/results/ft_renamed\n"))
    with pytest.raises(SystemExit, match="wave-2 derivation expected"):
        B.legs_w2()


def test_the_derivation_reports_what_it_expected_and_what_it_found(monkeypatch):
    monkeypatch.setattr(B, "LEGS", B.LEGS.replace('echo "FT LEGS COMPLETE"', ""))
    with pytest.raises(SystemExit, match="found 0"):
        B.legs_w2()


# ------------------------------------- the subset precondition, run under bash

def _precondition(text):
    a = _args(text)
    return a[a.index("# Checked HERE, not hours later"):a.index("epochs_for () {")]


def test_the_legs_check_every_subset_before_spending_a_gpu(w2):
    """A comment saying 'run the rebuild first' does not prevent anything. The
    legs wait on ${SUB}/DONE and that marker ALREADY EXISTS from wave 1, so the
    wait passes instantly and the first N=1e3 cell dies on a missing parquet
    hours in -- item 33's own recorded defect."""
    g = _precondition(w2[LEGS_W2])
    assert "for N in 1000 10000 100000 1000000" in g
    assert "for S in 1 2 3" in g
    assert "${SUB2} ${SUB1}" in g


@pytest.mark.parametrize("missing,expect_rc", [
    (None, 0),                                  # every subset present
    ("jc2/train_N1000_s2.parquet", 1),          # the real wave-2 failure mode
    ("jc1/train_N1000000_s3.parquet", 1),       # a wave-1 subset gone missing
])
def test_the_precondition_behaves_under_a_real_shell(w2, tmp_path, missing, expect_rc):
    """THE TEST THAT ACTUALLY CATCHES A QUOTING BUG. Everything above inspects
    strings; this runs the emitted block. It carries backslash continuations
    inside a brace group, which is exactly the shape that silently becomes a
    no-op when it is wrong -- and a precondition that never fires is worse than
    none, because it reads as checked."""
    import subprocess
    for d in ("jc2", "jc1"):
        (tmp_path / d).mkdir()
        for s in (1, 2, 3):
            for n in (1000, 10000, 100000, 1000000):
                (tmp_path / d / f"train_N{n}_s{s}.parquet").touch()
    if missing:
        (tmp_path / missing).unlink()
    script = (f"set -euo pipefail\nSUB2={tmp_path}/jc2\nSUB1={tmp_path}/jc1\n"
              + _precondition(w2[LEGS_W2]))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == expect_rc, r.stdout + r.stderr
    if missing:
        assert missing.split("/")[-1] in r.stdout, (
            "the refusal must NAME the missing file; 'a subset is missing' "
            "costs a cluster round trip to act on")
