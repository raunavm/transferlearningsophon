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
W2 = {"job-ft-subsets-jc2-w2-raunav.yaml", "job-ft-subsets-jc1-w2-raunav.yaml"}


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

def test_wave2_emits_only_the_two_rebuild_specs(w2):
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
    for name, text in w2.items():
        assert "--sizes 1000 10000 100000 1000000" in _live(text), name


def test_the_rebuilds_carry_no_bare_done_short_circuit(w2):
    """The exact defect item 33 records: `[ -f DONE ] && exit 0` in front of
    make_subsets.py's careful manifest comparison produced a confident green
    while the N=1e3 subsets were never written."""
    for name, text in w2.items():
        live = _live(text)
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
    for name, text in w2.items():
        res = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["resources"]
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
