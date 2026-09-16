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
LEGS_W2 = "job-ft-legs-w2-raunav.yaml"
W2 = SUBSETS | {LEGS_W2}

# Each wave-2 spec carries the pin it was generated for, and they are NOT the
# same tag. See the w2 fixture for why.
PINS = {LEGS_W2: "mtx-s1.45",
        "job-ft-subsets-jc2-w2-raunav.yaml": "mtx-s1.41",
        "job-ft-subsets-jc1-w2-raunav.yaml": "mtx-s1.41"}


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_ft_jobs", ROOT / "scripts" / "build_ft_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod()


@pytest.fixture(scope="module")
def w2():
    """Built at the pin each spec ACTUALLY carries, not at one shared pin.

    The two subset rebuilds ran and completed at mtx-s1.41; their pin is the
    provenance record for the parquet files now on disk, and moving it would
    detach a finished run from the code that produced it (the WINDOW_PIN
    precedent item 33 cites). The legs spec was deliberately re-specced and
    relaunched at mtx-s1.42 under item 36, because it needs
    `extract_features.py --stride`, which does not exist in mtx-s1.41."""
    return {name: B.build(pin, wave2=True)[name] for name, pin in PINS.items()}


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
        assert {"name": "REPO_REF", "value": PINS[name]} in env, name


def test_the_legs_pin_is_new_enough_to_have_the_stride_flag():
    """mtx-s1.42 is not cosmetic. The legs call `extract_features.py --stride 4`,
    which item 36 added; at mtx-s1.41 that is an unrecognised argument and every
    leg-1 cell dies AFTER its fine-tune has been paid for."""
    assert PINS["job-ft-legs-w2-raunav.yaml"] != "mtx-s1.41", (
        "the legs must not pin a tag that predates --stride")


# ------------------------------------------------------------------ mechanics

def test_generation_is_idempotent(w2):
    """A second run must not produce a spec that differs from the one on disk,
    or the file and the generator disagree about what ran."""
    assert {n: B.build(pin, wave2=True)[n] for n, pin in PINS.items()} == w2


def test_the_specs_on_disk_match_the_generator(w2):
    """These are committed; a hand-edit here would be silently reverted by the
    next regeneration, which is how the legs lost a pin on 2026-09-07."""
    for name, text in w2.items():
        p = ROOT / "experiments" / "FT" / "k8s" / name
        assert p.exists(), f"{name} not written"
        assert p.read_text() == text, f"{name} on disk differs from the generator"


# ------------------------------------------------------------ the wave-2 legs


def test_wave2_emits_the_legs_spec(w2):
    assert LEGS_W2 in w2


def test_the_legs_write_to_their_own_root(w2):
    """THE LOAD-BEARING TEST for item 33's first open sub-item. Wave 1's 108
    cells are keyed by the same init names and sizes, so a shared root makes
    every wave-2 cell hit `[ -f DONE ]` and skip -- the wave would appear to
    run, finish in minutes, and produce nothing."""
    live = _live(w2[LEGS_W2])
    assert "ROOT_OUT=/data/results/ft/w2b" in live, (
        "wave 2 relaunches into w2b, not w2: the first launch left 4 cells with "
        "DONE markers under the OLD protocol (200k val, full 2e6 rows), and "
        "reusing that root would skip them into a table whose first 4 cells "
        "disagree with the other 140 -- which leg1_metrics.py would then refuse "
        "on the label188_sha256 check, after the whole wave had been paid for.")
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


# ------------------------------------------------------------- item 36: space
#
# Wave 2 as first launched needed 306 GB against 174 GB free on a SHARED volume,
# and would have hit 100% in 3.3 days at cell ~54 of 144 -- taking MPM and the
# extraction wave down with it. These guard the three changes that fixed it.
# They are written against the LIVE body, not the comments, because the spec
# now explains the defect it avoids and a bare scan matches the explanation.

def test_wave2_validates_on_20k_not_200k(w2):
    """200k val against a 1k-10k train set is what produced BOTH the 9.4-day
    projection and NRP's 38.3% utilisation warning (policy floor 40%)."""
    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    assert "--samples-per-epoch-val 200000" not in body, (
        "wave 2 is validating on 200,000 jets per epoch against a training set "
        "of 1,000-10,000. That is ~80 s of every ~84 s epoch, and it is why the "
        "GPU sat at 38.3% -- below NRP's 40% policy floor -- and why the wave "
        "was projected at 9.4 days.")
    assert body.count("--samples-per-epoch-val 20000") == 2, (
        "both legs must make the cut, or leg 2 keeps the utilisation problem")


def test_wave1_still_validates_on_200k(w1):
    """The cut is a wave-2 SUBSTITUTION, not an edit to LEGS. Wave 1's 108 cells
    are complete and must keep the spec they actually ran under -- rewriting it
    would detach a finished run from the code that produced it."""
    body = _args(w1["job-ft-legs-raunav.yaml"])
    assert "--samples-per-epoch-val 200000" in body, (
        "the item-36 cut leaked into wave 1, whose cells are already finished")


def test_wave2_features_are_strided_not_head_sliced(w2):
    """A head slice is a BIASED sample, not a smaller one. The test list
    interleaves Res2P / Res34P / QCD by file (leg1_metrics.py docstring, lines
    32-36), so `--max-jets 500000` would over-represent whichever files come
    first. The stride keeps the class mix."""
    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    assert "--max-jets 2000000 --stride 4 --save-logits" in body, (
        "wave 2 must READ 2e6 jets and KEEP every 4th, not read the first "
        "500,000 -- the two differ in class mix, not just in size")


def test_extraction_stride_equals_what_leg1_metrics_actually_reads():
    """THE COUPLING THAT MAKES THE CUT SAFE, AND THE ONE THAT CAN SILENTLY ROT.

    Caching a stride-4 subsample is lossless ONLY because leg1_metrics.py
    already computes its AUC on `np.arange(0, n, 4)`. If that default ever
    moves -- to 2, say -- the cache silently stops containing the rows the
    analysis asks for, every cell still loads, and the published AUC is
    computed on a different sample than intended. Nothing else in the repo ties
    these two numbers together, so this test is the tie."""
    import re
    src = (ROOT / "experiments" / "FT" / "leg1_metrics.py").read_text()
    m = re.search(r'"--auc-stride",\s*type=int,\s*default=(\d+)', src)
    assert m, "leg1_metrics.py no longer declares --auc-stride the same way"
    reader_stride = int(m.group(1))

    body = _live(B.build(PINS[LEGS_W2], wave2=True)[LEGS_W2])
    m2 = re.search(r"--max-jets 2000000 --stride (\d+)", body)
    assert m2, "wave 2 no longer strides its feature extraction"
    cache_stride = int(m2.group(1))

    assert cache_stride == reader_stride, (
        f"wave 2 caches every {cache_stride}th jet but leg1_metrics.py reads "
        f"every {reader_stride}th by default. The cache no longer holds the "
        f"rows the analysis selects, and nothing will error -- the AUC will "
        f"just be computed on a different subsample than the one intended.")


def test_the_smoke_check_expects_the_strided_row_count(w2):
    """2e6 jets read at stride 4 is 500,000 rows on disk. A smoke check still
    asserting 2,000,000 fails every cell after the fine-tune has been paid for."""
    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    assert "--dir ${OUT}/features_v2 --n 500000 --k 162" in body
    assert "--n 2000000 --k 162" not in body


def test_per_epoch_checkpoints_are_pruned_in_both_legs(w2):
    """132 GB across the wave, read by nothing. weaver cannot be told to stop
    writing them (train.py:836-837), so they go after the best-epoch copy."""
    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    assert body.count(
        "rm -f ${OUT}/net_epoch-*_state.pt ${OUT}/net_epoch-*_optimizer.pt") == 2


def test_the_prune_cannot_match_the_file_the_analysis_reads(w2):
    """net_best_epoch_state.pt is the ONLY checkpoint anything downstream opens,
    and the prune runs in the same directory. The glob must be incapable of
    matching it -- and the guard either side must survive, because "incapable"
    is a claim about today's --model-prefix."""
    import fnmatch
    for pat in ("net_epoch-*_state.pt", "net_epoch-*_optimizer.pt"):
        assert not fnmatch.fnmatch("net_best_epoch_state.pt", pat), (
            f"the prune glob {pat!r} matches net_best_epoch_state.pt")
        assert fnmatch.fnmatch("net_epoch-7_state.pt", pat) or "optimizer" in pat

    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    # guarded before AND after: refuse to prune without the copy, and refuse to
    # continue if the prune took it anyway
    assert body.count(
        "[ -f ${OUT}/net_best_epoch_state.pt ] || {") == 4, (
        "each of the two prunes needs a precondition and a postcondition")


def test_the_prune_runs_after_the_cell_is_finished_with_the_checkpoints(w2):
    """Leg 1 extracts features from net_best_epoch_state.pt and leg 2 runs
    --predict (which resolves to _best_epoch_state.pt, weaver train.py:878).
    Pruning before either would delete the input to the step that follows."""
    body = _live(w2["job-ft-legs-w2-raunav.yaml"])
    prune = "rm -f ${OUT}/net_epoch-*_state.pt"
    for marker in ("--out ${OUT}/features_v2", "--predict-output ${OUT}/pred.root"):
        assert marker in body
        assert body.index(marker) < body.index(prune, body.index(marker)), (
            f"the prune precedes {marker!r}; it would remove the checkpoint "
            f"that step reads")


def test_striding_the_cache_selects_the_rows_the_reader_would_have():
    """THE ARITHMETIC THE "no metric changes" CLAIM RESTS ON.

    Caching every 4th row and then reading all of it must give the same jets as
    caching everything and reading every 4th. If it did not, item 36 would be
    trading disk for a silently different sample -- and the AUC would move for a
    reason no one could see from the artifact."""
    import numpy as np
    n, stride = 2_000_000, 4
    full = np.arange(n)

    cached = full[:n][::stride]                 # what extract_features now writes
    read_from_full = full[np.arange(0, n, stride)]   # what leg1_metrics selects today

    assert cached.shape == read_from_full.shape == (n // stride,)
    assert np.array_equal(cached, read_from_full)

    # and a head slice of the same SIZE is a different set of jets entirely --
    # which is the whole reason this is a stride
    head = full[: n // stride]
    assert not np.array_equal(head, cached)


def test_cheap_and_expensive_cells_interleave(w2):
    """UTILISATION IS A PROPERTY OF N, SO ORDER DECIDES WHETHER NRP SEES A FLOOR.

    Training time scales with --samples-per-epoch against a ~38 s fixed overhead
    per epoch that does not, so an N=1e3 cell runs the GPU at ~16% and an N=1e6
    cell at ~92%. With N as the outer loop each init runs three N=1e3 cells then
    three N=1e4 cells back to back -- ~3.8 contiguous hours at ~16%, and NRP
    averages over 3 h windows. Seeds outer / N inner interleaves them and no 3 h
    window falls below ~51%.

    This is scheduling, not protocol: cells are keyed by their own directory,
    subset and seed, so the order they run in cannot reach the numbers."""
    # indentation-agnostic: YAML dedents the block scalar on parse, so the
    # literal leading whitespace here is not the file's
    lines = [ln.strip() for ln in _live(w2[LEGS_W2]).splitlines()]
    seeds, sizes = "for S in 1 2 3; do", "for N in 1000 10000 100000 1000000; do"

    pairs = [(a, b) for a, b in zip(lines, lines[1:])]
    # THREE, not two: the subset precondition block (item 33) iterates the same
    # two loops before any GPU is spent, and it is already seeds-outer. Asserting
    # 3 rather than >=2 means a leg that loses its loops fails here instead of
    # being covered by the precondition's pair.
    assert pairs.count((seeds, sizes)) == 3, (
        "both legs must iterate seeds OUTSIDE sizes; with sizes outside, each "
        "init spends ~3.8 unbroken hours of GPU time at ~16% utilisation")
    assert pairs.count((sizes, seeds)) == 0, (
        "a size loop still wraps a seed loop -- that is the ~16% block")


def test_wave2_space_guard_is_raised_but_the_floor_is_not(w2):
    """PI-approved 2026-09-16. 85% was calibrated when wave 2 would write 306 GB
    into 174 GB free -- a wave that could not fit, which the guard correctly
    caught. Item 36 cut it to ~41 GB, so the same line then refused a wave
    peaking near 89% with ~115 GB still free.

    THE ABSOLUTE FLOOR MUST SURVIVE. `g >= 50` is what actually protects the
    volume: a percentage is a proxy for "will this fill up", 50 GB free is the
    thing itself, and it holds regardless of disk size or what share of it
    belongs to other people. Raising the proxy is only safe while the floor
    stands, so this asserts the floor as hard as it asserts the threshold."""
    body = _live(w2[LEGS_W2])
    assert '[ "$p" -lt 92 ] && [ "$g" -ge 50 ]' in body, (
        "wave 2 must check BOTH the raised percentage AND the unchanged 50 GB "
        "floor -- dropping the floor would remove the only absolute protection")
    assert '-lt 85' not in body


def test_wave1_and_bench_keep_the_original_space_guard(w1):
    """The raise is a wave-2 substitution. space_ok is defined separately in
    LEGS and LEGS_BENCH, and wave 1's 108 cells are finished -- rewriting their
    spec would detach a completed run from the code that produced it."""
    for name in ("job-ft-legs-raunav.yaml", "job-ft-legs-bench-raunav.yaml"):
        body = _args(w1[name])
        assert '[ "$p" -lt 85 ]' in body, f"{name} lost its 85% guard"
        assert '-lt 92' not in body, f"the wave-2 raise leaked into {name}"


def test_wave2_legs_exclude_the_known_bad_nodes(w2):
    """A NODE CAN ADVERTISE FREE GPUs AND STILL FAIL EVERY POD, and Kubernetes
    tells the scheduler nothing about it until an admin taints it.

    Measured 2026-09-16: nautilus-ext-gpu01.fullerton.edu cannot attach the CSI
    volume ("no relationship found between node ... and this object"), so each
    pod fails to mount /data, is evicted, and reschedules straight back onto the
    same node. It consumed three wave-2 pods in ~90 minutes and left TWO
    .partial dirs on one cell -- one short of tripping attempt_ok and halting the
    entire 144-cell wave. Every other GPU spec in the repo already carries this
    list; the legs were the one that did not."""
    import yaml
    spec = yaml.safe_load(w2[LEGS_W2])
    terms = (spec["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]
             ["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"])
    excl = [e for t in terms for e in t["matchExpressions"]
            if e["key"] == "kubernetes.io/hostname"]
    assert excl, "the wave-2 legs have no hostname exclusion at all"
    assert excl[0]["operator"] == "NotIn"
    assert "nautilus-ext-gpu01.fullerton.edu" in excl[0]["values"], (
        "the node that took three pods in 90 minutes is not excluded")
    assert "ry-gpu-03.sdsc.optiputer.net" in excl[0]["values"], (
        "the node scripts/exclude_node.py was written for is not excluded")


def test_the_exclusion_is_opt_in_and_reaches_nothing_else(w1, w2):
    """exclude_hosts defaults to empty, so adding the list to the legs cannot
    silently rewrite the specs of runs that already finished."""
    import yaml
    for name, text in w1.items():
        spec = yaml.safe_load(text)
        aff = spec["spec"]["template"]["spec"].get("affinity")
        if not aff:
            continue
        terms = (aff["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
                 ["nodeSelectorTerms"])
        keys = [e["key"] for t in terms for e in t["matchExpressions"]]
        assert "kubernetes.io/hostname" not in keys, (
            f"the wave-2 node exclusion leaked into {name}, whose run is finished")
