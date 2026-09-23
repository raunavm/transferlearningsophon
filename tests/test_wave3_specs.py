"""Wave 3 and the v2 benchmarks (2026-09-18): the specs that fine-tune from every
pretrained checkpoint, sharded five ways into wave 2's tree.

Three classes of thing are pinned here:

  * CONTENT -- what each shard asks for, in the style of test_wave2_specs.py:
    wave 2's recipe, prune, validation size, guard and loop order reach every
    wave-3 shard unchanged; the v2 benchmarks keep the published recipe and the
    85% guard; the not-yet-launchable groups are not emitted by default.
  * COVERAGE -- the union of the shards is exactly the intended cell set, no
    cell twice, and wave 3 shares no init name with the running wave 2, whose
    output root it writes into.
  * THE SHELL -- every emitted script is RUN under bash with python3, weaver,
    git, df and curl stubbed, into a temporary tree. String tests cannot see a
    loop that enumerates the wrong seeds, a lock that is never released, or a
    prune that deletes the file the read-out needs; this does. The set of DONE
    cells the script leaves behind must equal the Python-side cell set.
"""
import importlib.util
import os
import pathlib
import re
import stat
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_ft_jobs_w3", ROOT / "scripts" / "build_ft_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod()
W3 = [f"job-ft-legs-w3-{c}-raunav.yaml" for c in "abcde"]
BV2 = [f"job-ft-legs-bench-v2-{c}-raunav.yaml" for c in "abcde"]
STAGE = "job-ft-stage-qg-herwig-raunav.yaml"
NEW = set(W3) | set(BV2) | {STAGE}


@pytest.fixture(scope="module")
def new():
    return B.build(B.PIN_W3, wave3=True, bench_v2=True)


@pytest.fixture(scope="module")
def w2():
    return B.build("mtx-s1.45", wave2=True)["job-ft-legs-w2-raunav.yaml"]


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _live(text):
    return "\n".join(ln for ln in _args(text).splitlines()
                     if not ln.strip().startswith("#"))


def _inits(text):
    m = re.search(r'^\s*INITS="([^"]*)"', _args(text), re.M)
    return [tuple(s.split(":")) for s in m.group(1).split()]


# ------------------------------------------------------------------ emission

def test_the_new_flags_emit_exactly_the_eleven_new_specs_and_nothing_launched(new):
    assert set(new) == NEW
    assert not (set(new) & {"job-ft-legs-w2-raunav.yaml", "job-ft-legs-raunav.yaml",
                            "job-ft-legs-bench-raunav.yaml"})


def test_a_bare_build_and_a_wave2_build_emit_no_new_spec():
    assert not (set(B.build("mtx-s1.41")) & NEW)
    assert not (set(B.build("mtx-s1.45", wave2=True)) & NEW)


def test_every_new_spec_pins_the_new_tag_and_carries_raunav(new):
    for name, text in new.items():
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav"), name
        env = d["spec"]["template"]["spec"]["containers"][0]["env"]
        assert {"name": "REPO_REF", "value": B.PIN_W3} in env, name
    # Read from the builder, not written out here. These specs were repinned
    # 51 -> 52 when the tag they named was published without their code in it,
    # and a literal would have turned a correct repin into a test failure while
    # a spec pinning a tag that does not carry its code would still pass.
    assert B.PIN_W3.startswith("mtx-s1.")


def test_the_pin_guard_refuses_a_tag_that_predates_the_code_these_run():
    with pytest.raises(SystemExit, match=f"first exists at {B.PIN_W3}"):
        B.build("mtx-s1.50", wave3=True)


def test_the_specs_on_disk_match_the_generator(new):
    for name, text in new.items():
        p = K8S / name
        assert p.exists(), f"{name} not written"
        assert p.read_text() == text, f"{name} on disk differs from the generator"


def test_regenerating_the_launched_specs_still_reproduces_them_byte_for_byte():
    """INITS grew a fourth field. The launched specs must not move by a byte."""
    for name, pin, w2 in (("job-ft-legs-w2-raunav.yaml", "mtx-s1.45", True),
                          ("job-ft-subsets-jc2-w2-raunav.yaml", "mtx-s1.41", True),
                          ("job-ft-subsets-jc1-w2-raunav.yaml", "mtx-s1.41", True),
                          ("job-ft-subsets-bench-raunav.yaml", "mtx-s1.21", False)):
        assert B.build(pin, wave2=w2)[name] == (K8S / name).read_text(), name


def test_the_not_yet_launchable_groups_are_not_emitted_by_default(new):
    for text in new.values():
        names = {n for n, *_ in _inits(text)} if 'INITS="' in _args(text) else set()
        assert not any(n.startswith(("mpm-", "rand-d2", "rand-d3")) for n in names)
    later = B.build(B.PIN_W3, wave3=True, bench_v2=True, later=["mpm-s1", "rand-d2"])
    assert set(later) == {"job-ft-legs-w3-mpm-s1-raunav.yaml", "job-ft-legs-w3-rand-d2-raunav.yaml",
                          "job-ft-legs-bench-v2-mpm-s1-raunav.yaml",
                          "job-ft-legs-bench-v2-rand-d2-raunav.yaml"}
    # A spec on disk is allowed only for a group recorded as launched, whose
    # pretraining job is Complete. Everything else must still be absent.
    launched = {n for n in later if any(f"-{g}-raunav" in n for g in B.LAUNCHED_LATER)}
    assert not any((K8S / n).exists() for n in set(later) - launched), \
        "a later group was written to disk before it was launched"
    for n in launched:
        if (K8S / n).exists():
            assert (K8S / n).read_text() == later[n], f"{n} drifted from the builder"


def test_gpu_specs_are_sized_and_scheduled_like_wave_two(new, w2):
    ref = yaml.safe_load(w2)["spec"]
    for name in W3 + BV2:
        d = yaml.safe_load(new[name])["spec"]
        assert d["backoffLimit"] == ref["backoffLimit"] == 50
        c, rc = d["template"]["spec"]["containers"][0], ref["template"]["spec"]["containers"][0]
        assert c["resources"] == rc["resources"], name
        assert d["template"]["spec"]["affinity"] == ref["template"]["spec"]["affinity"], name
        assert {"name": "GPU_PRODUCT", "value": "NVIDIA-GeForce-RTX-3090"} in c["env"]
    s = yaml.safe_load(new[STAGE])["spec"]
    assert "nvidia.com/gpu" not in s["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert s["backoffLimit"] <= 1


# ------------------------------------------------------------------- coverage

def test_wave3_shares_no_init_name_with_the_running_wave_two(new):
    w2_names = {n for n, *_ in B.INITS}
    for name in W3:
        assert not ({n for n, *_ in _inits(new[name])} & w2_names), name
    assert "ROOT_OUT=/data/results/ft/w2b" in _live(new[W3[0]])


def test_the_shards_partition_the_intended_inits_and_cells_exactly(new):
    for names, inits, cells_of in ((W3, B.INITS_W3, B.cells_legs),
                                   (BV2, B.INITS_BENCH_V2, B.cells_bench)):
        seen = []
        for n in names:
            seen += [i[0] for i in _inits(new[n])]
        assert sorted(seen) == sorted(i[0] for i in inits), "shards do not partition the inits"
        assert len(seen) == len(set(seen)), "an init is in two shards"
        want = cells_of(inits)
        assert len(want) == len(set(want)), "the intended cell set names a cell twice"
        got = [c for s in B.shard(inits, cells_of) for c in cells_of(s)]
        assert sorted(got) == sorted(want)


def test_the_wave3_cell_set_is_the_approved_design():
    names = [n for n, *_ in B.INITS_W3]
    assert names == ([f"l188-s{s}" for s in range(1, 6)] + [f"l162-s{s}" for s in range(2, 6)]
                     + [f"r42q1-s{s}" for s in range(1, 6)] + ["r16q1-s1", "r16q1-s5"]
                     + [f"l162mass-s{s}" for s in range(1, 6)]
                     + [f"r16q1mass-s{s}" for s in range(1, 6)] + ["rand-d1-s1b"])
    assert all(s == [1] for *_, s in B.INITS_W3), "one fine-tuning seed per checkpoint"
    assert len(B.cells_legs(B.INITS_W3)) == 27 * 4 * 2 == 216
    assert not any("/mtx-l162-s1/" in c for _, c, *_ in B.INITS_W3)


def test_the_bench_v2_cell_set_is_the_approved_design():
    seeds = {n: s for n, _, _, s in B.INITS_BENCH_V2}
    assert seeds["scratch"] == [1, 2, 3]
    assert all(s == [1] for n, s in seeds.items() if n != "scratch")
    cells = B.cells_bench(B.INITS_BENCH_V2)
    reps = [c for c in cells if c[3] > 1 and c[1] != "scratch"]
    assert {c[1] for c in reps} == {"l162-s1b", "r16q1-s2"}
    assert all(c[0] == "leg_top" and c[2] == 1_200_000 for c in reps)
    assert len(reps) == 2 * 4
    assert len(cells) == (32 + 3) * 8 + 8 == 288


def test_mpm_s1_keeps_three_seeds_and_the_other_later_inits_one():
    assert B.INITS_LATER["mpm-s1"][0][3] == [1, 2, 3]
    for g in ("mpm-s2", "mpm-s3", "rand-d2", "rand-d3"):
        assert all(s == [1] for *_, s in B.INITS_LATER[g])


# ------------------------------------------------- wave 3 keeps wave 2's protocol

def test_wave3_keeps_wave_twos_recipe_prune_validation_guard_and_order(new, w2):
    ref = _live(w2)
    for name in W3:
        live = _live(new[name])
        for must in ("--optimizer-option weight_decay 0.01", "--samples-per-epoch-val 20000",
                     "--max-jets 2000000 --stride 4 --save-logits",
                     '[ "$p" -lt 92 ] && [ "$g" -ge 50 ]'):
            assert live.count(must) == ref.count(must), (name, must)
        assert all(c.endswith("net_epoch-79_state.pt") for _, c, *_ in _inits(new[name])), (
            "wave 3 loads the LAST epoch, as wave 2 does (item 18)")
        assert live.count("rm -f ${OUT}/net_epoch-*_state.pt ${OUT}/net_epoch-*_optimizer.pt") == 2
        assert live.count('grep -q "Parameters with lr multiplied by 50"') == 2
        assert live.count("head_lr_mult=50 weight_decay=0.01 wave=3") == 2
        assert "wave=2" not in live and "--lr-scheduler none" not in live
        lines = [ln.strip() for ln in live.splitlines()]
        pairs = list(zip(lines, lines[1:]))
        assert pairs.count(("for S in ${seeds//,/ }; do",
                            "for N in 1000 10000 100000 1000000; do")) == 2, "seeds outer, sizes inner"


def test_wave3_deletes_the_leg1_feature_matrix_that_nothing_reads(new, w2):
    """256 MB per leg-1 cell, 27.6 GB over the wave, against 199 GB free and
    ~123 GB owed. leg1_metrics.py's discover() gates on logits.npy +
    label188.npy and opens exactly those two, so the matrix is dead weight --
    and a shard halting mid-wave on space_ok is the failure this avoids.

    ORDER IS THE WHOLE SAFETY ARGUMENT: the smoke check above loads all three
    arrays, so the delete must follow it, and it must not touch what the
    analysis opens."""
    src = (ROOT / "experiments/FT/leg1_metrics.py").read_text()
    disc = src[src.index("def discover"):src.index("def cell_metrics")]
    assert '"logits.npy"' in disc and '"label188.npy"' in disc
    assert "np.load(fd / \"features.npy\")" not in src, (
        "leg1_metrics.py now loads features.npy; do not delete it")
    for name in W3:
        live = _live(new[name])
        chk = live.index("smoke_checks.py features --dir ${OUT}/features_v2 --n 500000 --k 162")
        rm = live.index("rm -f ${OUT}/features_v2/features.npy")
        assert live.count("rm -f ${OUT}/features_v2/features.npy") == 1, "leg 1 only"
        assert chk < rm < live.index("touch ${OUT}/DONE", rm)
        for keep in ("--save-logits", "features_v2"):
            assert keep in live
        for never in ("rm -f ${OUT}/features_v2/logits.npy",
                      "rm -f ${OUT}/features_v2/label188.npy",
                      "rm -rf ${OUT}/features_v2"):
            assert never not in live


def test_the_running_wave_two_keeps_its_feature_matrix(w2):
    """The cut is a wave-3 substitution. Wave 2 is running and its spec is the
    provenance record of the cells already on disk."""
    assert "rm -f ${OUT}/features_v2/features.npy" not in _live(w2)
    assert "--dir ${OUT}/features_v2 --n 500000 --k 162" in _live(w2)


def test_wave3_does_not_write_the_reference_rows_or_fetch_the_public_checkpoint(new):
    for name in W3:
        live = _live(new[name])
        assert "ref_e1arms" not in live, "the E1 arm-S reference is wave 2's to write"
        assert "curl" not in live and "sophon_public" not in live


def test_wave3_does_not_gate_on_the_reference_checkpoints_it_never_opens(new, w2):
    """A precondition on someone else's input is the inversion
    tests/test_spec_preconditions.py exists for: wave 3 dropped the
    leg2/ref_e1arms-s* rows, so the arm-S check must go with them -- while the
    JetClass-I test files, which leg 2 DOES read, stay checked."""
    for name in W3:
        live = _live(new[name])
        assert "arm_s_s${S}" not in live and "no E1 arm S seed" not in live
        assert "/data/results/e1/" not in live
        assert "fewer than 2 JetClass-I test files" in live
        assert live.index("fewer than 2 JetClass-I test files") < live.index("seed_weaver.py")
    ref = _live(w2)
    assert "arm_s_s${S}" in ref and "ref_e1arms" in ref, (
        "wave 2 writes the reference rows and must keep checking for them")


def test_wave3_halt_markers_are_per_shard_so_no_job_can_stop_another(new):
    for name in W3:
        live = _live(new[name])
        assert "FAIL_MARK=${ROOT_OUT}/FAILED.${SHARD}" in live
        assert "WAIT_MARK=${ROOT_OUT}/WAIT_TIMEOUT.${SHARD}" in live
        assert "FAIL_MARK=${ROOT_OUT}/FAILED\n" not in live
        assert f"SHARD={yaml.safe_load(new[name])['metadata']['name']}" in live


def test_every_cell_is_locked_before_anything_is_written_and_unlocked_after_done(new):
    for name in W3 + BV2:
        live = _live(new[name])
        n = 2 if name in W3 else 1
        assert live.count("if ! mkdir ${OUT}.lock 2>/dev/null; then") == n
        assert live.count("rm -rf ${OUT}.lock") == n
        for i in range(n):
            lock = live.index("mkdir ${OUT}.lock", [0, live.index("mkdir ${OUT}.lock") + 1][i])
            assert lock < live.index("space_ok\n", lock) < live.index("mkdir -p ${OUT}\n", lock)
            done = live.index("touch ${OUT}/DONE", lock)
            assert live.index("rm -rf ${OUT}.lock", lock) > done


# ------------------------------------------------------------- benchmarks v2

def test_bench_v2_keeps_the_published_recipe_and_the_85_percent_guard(new):
    old = (K8S / "job-ft-legs-bench-raunav.yaml").read_text()
    head = next(ln.strip() for ln in old.splitlines() if ln.strip().startswith("HEAD_MULT=("))
    for name in BV2:
        live = _live(new[name])
        assert head in live and "--lr-scheduler none" in live
        assert f"--num-epochs {B.BENCH_EPOCHS}" in live and B.BENCH_EPOCHS == 20
        assert "--optimizer-option weight_decay 0.01" in live
        assert '[ "$p" -lt 85 ] && [ "$g" -ge 50 ]' in live and "-lt 92" not in live
        assert "ROOT_OUT=/data/results/ft/bench_v2" in live
        assert 'top) echo "1000 10000 100000 1200000"' in live
        assert 'qg)  echo "1000 10000 100000 1600000"' in live
    assert float(B.LR_PRETRAINED) * B.BENCH_HEAD_MULT == 5e-3


def test_bench_v2_validates_on_20k_at_small_n_and_records_what_it_used(new):
    for name in BV2:
        live = _live(new[name])
        assert 'val_for () { if [ "$1" -le 10000 ]; then echo 20000; else echo 200000; fi; }' in live
        assert "--samples-per-epoch $(samples_for ${N}) --samples-per-epoch-val $(val_for ${N})" in live
        assert "--samples-per-epoch-val 200000" not in live
        assert ("samples_per_epoch=$(samples_for ${N}) steps_per_epoch=$(($(samples_for ${N})/512)) "
                "samples_per_epoch_val=$(val_for ${N}) wave=bench-v2") in live


def test_the_smallest_bench_cell_gets_the_same_step_count_rule_as_the_legs(new):
    """PI decision 2026-09-18. 20 epochs over 1,000 jets is ONE optimiser step
    per epoch -- 20 in total against ~380 at N=1e4 -- so the cell where a
    pretraining effect is predicted largest would measure the optimiser. Item 25
    settled this for the legs; the benchmarks must not differ from them on a knob
    that is not the variable under study.

    THE TRAINING SET IS STILL 1,000 JETS. Only samples-per-epoch is decoupled,
    and only at 1e3 -- any other size decoupled from its subset breaks the
    controlled variable, exactly as the legs' own assertion says."""
    assert B.BENCH_SAMPLES_PER_EPOCH == {1_000: 10_000}
    assert B.BENCH_SAMPLES_PER_EPOCH[1_000] == B.SAMPLES_PER_EPOCH[1_000], (
        "the two tables' smallest cells must be decoupled the same way")
    assert set(B.BENCH_SAMPLES_PER_EPOCH) == {1_000}
    for name in BV2:
        live = _live(new[name])
        assert "samples_for () { case $1 in 1000) echo 10000;; *) echo $1;; esac; }" in live
        # 10,000 // 512 = 19 steps x 20 epochs = 380, the N=1e4 cell's count
        assert B.BENCH_SAMPLES_PER_EPOCH[1_000] // 512 * B.BENCH_EPOCHS == 380
        # the training SUBSET is still the 1,000-jet one
        assert "--data-train ${SUB}/train_N${N}_s${DSEED}.parquet" in live


def test_bench_v2_prune_is_wave_twos_line_for_line(new, w2):
    ref = [ln.strip() for ln in _live(w2).splitlines()]
    i = ref.index("rm -f ${OUT}/net_epoch-*_state.pt ${OUT}/net_epoch-*_optimizer.pt")
    block = ref[i - 3:i + 3]
    for name in BV2:
        got = [ln.strip() for ln in _live(new[name]).splitlines()]
        j = got.index("rm -f ${OUT}/net_epoch-*_state.pt ${OUT}/net_epoch-*_optimizer.pt")
        assert got[j - 3:j + 3] == block, name
        readout = next(k for k, ln in enumerate(got) if "--out ${OUT}/features --batch-size 512" in ln)
        assert readout < j < got.index("touch ${OUT}/DONE"), "prune must follow the read-out"


def test_bench_v2_deletes_features_only_after_its_smoke_check_and_keeps_what_metrics_read(new):
    src = (ROOT / "experiments/FT/bench_metrics.py").read_text()
    for f in ("logits.npy", "label188.npy", "DONE", "ft_manifest.json"):
        assert f'"{f}"' in src
    assert "features.npy" not in src, "bench_metrics.py now reads features.npy; do not delete it"
    for name in BV2:
        live = _live(new[name])
        chk = live.index("smoke_checks.py features --dir ${OUT}/features --n $(ntest_for ${D}) --k 2")
        assert chk < live.index("rm -f ${OUT}/features/features.npy") < live.index("touch ${OUT}/DONE")
        assert "rm -f ${OUT}/features/logits.npy" not in live


def test_bench_v2_asserts_the_published_test_splits_before_any_gpu_work(new):
    for name in BV2:
        live = _live(new[name])
        assert '[ "${NTEST_top}" -eq 404000 ] && [ "${NTEST_qg}" -eq 200000 ] && [ "${NTEST_herwig}" -eq 200000 ]' in live
        assert live.index('-eq 404000') < live.index("seed_weaver.py")


def test_bench_v2_scores_every_qg_cell_on_herwig_into_the_named_files(new):
    for name in BV2:
        live = _live(new[name])
        assert 'if [ "${D}" = "qg" ]; then' in live
        assert "--data-test ${HERWIG} --observers jet_pt jet_energy --out ${OUT}/features_herwig" in live
        assert "smoke_checks.py features --dir ${OUT}/features_herwig --n ${NTEST_herwig} --k 2" in live
        assert "mv ${OUT}/features_herwig/logits.npy ${OUT}/features/logits_herwig.npy" in live
        assert "mv ${OUT}/features_herwig/label188.npy ${OUT}/features/label_herwig.npy" in live
        assert "run ft-stage-qg-herwig-raunav first" in live
    assert B.HERWIG_TEST == ["/data/finetune/qg_herwig/qg_herwig_chunk0.parquet",
                             "/data/finetune/qg_herwig/qg_herwig_chunk1.parquet"]


def test_every_requested_observer_is_declared_by_its_data_config(new):
    cfgdir = ROOT / "configs" / "finetune"
    for name in BV2:
        for line in _live(new[name]).splitlines():
            if "extract_features.py" not in line:
                continue
            assert "--observers" in line
            got = line.split("--observers")[1].split("--")[0].split()
            for cfg in ("TopReference.yaml", "EnergyFlowQG.yaml"):
                declared = yaml.safe_load((cfgdir / cfg).read_text())["observers"]
                assert set(got) <= set(declared), (cfg, got)


def test_bench_v2_reps_are_two_inits_times_five_on_top_and_scratch_keeps_three_seeds(new):
    assert B.BENCH_V2_REPS == [1, 2, 3, 4, 5] and B.BENCH_V2_REP_INITS == ["l162-s1b", "r16q1-s2"]
    for name in BV2:
        live = _live(new[name])
        assert 'case " l162-s1b r16q1-s2 " in *" ${name} "*) REPS="1 2 3 4 5";; esac' in live
        assert 'if [ "${D}" = "top" ]; then' in live
        for spec in _inits(new[name]):
            assert spec[3] == ("1,2,3" if spec[0] == "scratch" else "1"), spec


def test_only_the_bench_shard_holding_the_public_checkpoint_fetches_it(new):
    for name in BV2:
        has = any(n == "sophon-public" for n, *_ in _inits(new[name]))
        assert ('curl -fsSL -o "${SOPHON}"' in _live(new[name])) == has, name


def test_the_old_bench_spec_is_untouched_and_superseded(new):
    assert (K8S / "job-ft-legs-bench-raunav.yaml").exists()
    for name in BV2:
        assert "Supersedes" in new[name] and "job-ft-legs-bench-raunav.yaml" in new[name]


def test_the_herwig_staging_takes_two_plain_chunks_and_never_overwrites(new):
    live = _live(new[STAGE])
    assert "--dataset qg_herwig --out ${TMP}" in live and "--limit-files 2" in live
    assert '[ -e ${OUT} ] && { echo "FATAL: ${OUT} exists' in live
    assert '[ "$p" -lt 85 ] && [ "$g" -ge 50 ]' in live
    assert "mv ${TMP} ${OUT}" in live and "OUT=/data/finetune/qg_herwig" in live
    sd = importlib.util.spec_from_file_location("sd", ROOT / "scripts/stage_downstream.py")
    m = importlib.util.module_from_spec(sd); sd.loader.exec_module(m)
    src = m.SOURCES["qg_herwig"]
    assert src["record"] == "3066475"
    assert list(src["files"].values())[:2] == ["QG_jets_herwig_0.npz", "QG_jets_herwig_1.npz"]
    assert all("withbc" not in f for f in src["files"].values())


# ------------------------------------------------------------ the derivation

def test_the_derivations_raise_if_their_base_changes(monkeypatch):
    monkeypatch.setattr(B, "LEGS_BENCH", B.LEGS_BENCH.replace("FAILED_BENCH", "FAILED_X"))
    with pytest.raises(SystemExit, match="bench-v2 .* derivation expected"):
        B.legs_bench_v2(B.INITS_BENCH_V2[:2], "x")
    monkeypatch.setattr(B, "LEGS", B.LEGS.replace("ref_e1arms-s${S}", "ref_x"))
    with pytest.raises(SystemExit, match="wave-3 .* derivation expected"):
        B.legs_w3(B.INITS_W3[:2], "x")


# ------------------------------------------------------------------ the shell
#
# The emitted script is RUN. python3 / weaver / git / pip / curl / sha256sum /
# df are stubs on PATH that create the files the real programs would and log
# every call; /data, /jc2 and /workspace are redirected into tmp_path.

def _stub(bindir: pathlib.Path, name: str, body: str):
    p = bindir / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _shell_env(tmp_path: pathlib.Path, inits) -> tuple[dict, pathlib.Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    _stub(bindir, "git", 'case "$1" in clone) mkdir -p "${@: -1}";; *) echo deadbeef;; esac\n')
    _stub(bindir, "pip", "exit 0\n")
    _stub(bindir, "curl", 'while [ $# -gt 0 ]; do [ "$1" = -o ] && touch "$2"; shift; done\n')
    _stub(bindir, "sha256sum", f'echo "{B.SOPHON_SHA256}  $1"\n')
    _stub(bindir, "df", 'case "$*" in *pcent*) printf "Use%%\\n 60%%\\n";; *) printf "Avail\\n500G\\n";; esac\n')
    _stub(bindir, "weaver", f'echo "WEAVER $*" >> {calls}\n'
                            'while [ $# -gt 0 ]; do [ "$1" = --predict-output ] && touch "$2"; shift; done\n')
    _stub(bindir, "python3", f'echo "PY $*" >> {calls}\n' + r'''
next_after () { local key=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$key" ] && { echo "$2"; return; }; shift; done; }
case "$*" in
  *seed_weaver.py*)
    P=$(next_after --model-prefix "$@")
    touch ${P}_epoch-0_state.pt ${P}_epoch-0_optimizer.pt ${P}_best_epoch_state.pt
    echo "Parameters with lr multiplied by 50";;
  *extract_features.py*)
    case "$*" in *--self-check-only*) exit 0;; esac
    D=$(next_after --out "$@"); mkdir -p "$D"
    touch "$D/features.npy" "$D/logits.npy" "$D/label188.npy" "$D/extract_manifest.json";;
  *mpm_init.py*) touch "$(next_after --out "$@")";;
  "-c "*)
    n=0; for a in "$@"; do case "$a" in *top_test*) n=$((n+404000));; *.parquet) n=$((n+100000));; esac; done; echo $n;;
esac
exit 0
''')
    data, jc2, ws = tmp_path / "data", tmp_path / "jc2", tmp_path / "workspace"
    for d in ("finetune/jc2", "finetune/jc1"):
        for s in (1, 2, 3):
            for n in (1000, 10000, 100000, 1000000):
                (data / d / f"train_N{n}_s{s}.parquet").parent.mkdir(parents=True, exist_ok=True)
                (data / d / f"train_N{n}_s{s}.parquet").touch()
        (data / d / "val.parquet").touch(); (data / d / "DONE").touch()
    for s in (1, 2, 3):
        (data / "results/e1" / f"arm_s_s{s}").mkdir(parents=True)
        (data / "results/e1" / f"arm_s_s{s}" / "net_best_epoch_state.pt").touch()
    (data / "JetClass/Pythia/test_20M").mkdir(parents=True)
    for c in B.JC1_CLASSES.split():
        for i in (0, 1):
            (data / "JetClass/Pythia/test_20M" / f"{c}_{i}.root").touch()
    (jc2 / "jet_data").mkdir(parents=True)
    for d, sizes in (("top", B.BENCH_SIZES["top"]), ("qg_v2", B.BENCH_SIZES["qg"])):
        sub = data / "finetune" / f"{d}_sub"
        sub.mkdir(parents=True)
        (sub / "DONE").touch(); (sub / "val.parquet").touch()
        for s in (1, 2, 3):
            for n in sizes:
                (sub / f"train_N{n}_s{s}.parquet").touch()
    (data / "finetune/top").mkdir(); (data / "finetune/top/top_test.parquet").touch()
    (data / "finetune/qg_v2").mkdir()
    for i in (18, 19):
        (data / "finetune/qg_v2" / f"qg_chunk{i}.parquet").touch()
    (data / "finetune/qg_herwig").mkdir()
    for i in (0, 1):
        (data / "finetune/qg_herwig" / f"qg_herwig_chunk{i}.parquet").touch()
    for n, c, *_ in inits:
        if c.startswith("/data/"):
            p = data / c[len("/data/"):]
            p.parent.mkdir(parents=True, exist_ok=True); p.touch()
        if n.startswith("mpm-"):
            p = data / B.MPM_SOURCE[n][len("/data/"):]
            p.parent.mkdir(parents=True, exist_ok=True); p.touch()
    ws.mkdir()
    cfg = ws / "transferlearningsophon/configs/finetune"
    cfg.mkdir(parents=True)
    for f in ("TopReference.yaml", "EnergyFlowQG.yaml"):
        (cfg / f).touch()
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", REPO_REF="test",
               NODE_NAME="n", POD_NAME="p")
    return env, calls


def _redirect(text: str, stubs: pathlib.Path, tree: pathlib.Path) -> str:
    """/data, /jc2/jet_data and /workspace into tmp. /data first: the data
    tree itself holds a `finetune/jc2`, which a bare /jc2 rewrite would hit."""
    return (_args(text).replace("/workspace", str(stubs / "workspace"))
            .replace("/data", str(tree / "data"))
            .replace("/jc2/jet_data", str(tree / "jc2/jet_data"))
            .replace("sleep 60", "sleep 0"))


def _run(text: str, tmp_path: pathlib.Path, inits):
    env, calls = _shell_env(tmp_path, inits)
    r = subprocess.run(["bash", "-c", _redirect(text, tmp_path, tmp_path)], capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=600)
    return r, calls.read_text() if calls.exists() else ""


def _done_cells(root: pathlib.Path) -> set:
    out = set()
    for done in root.rglob("DONE"):
        rel = done.relative_to(root).parts[:-1]
        if len(rel) == 4:
            out.add((rel[0], rel[1], int(rel[2][1:]), int(rel[3][1:])))
    return out


@pytest.mark.parametrize("name", W3 + BV2)
def test_each_shard_runs_under_bash_and_leaves_exactly_its_cell_set(new, tmp_path, name):
    inits = [i for i in (B.INITS_W3 if name in W3 else B.INITS_BENCH_V2)
             if i[0] in {x[0] for x in _inits(new[name])}]
    r, calls = _run(new[name], tmp_path, inits)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    root = tmp_path / "data/results/ft" / ("w2b" if name in W3 else "bench_v2")
    want = set((B.cells_legs if name in W3 else B.cells_bench)(inits))
    assert _done_cells(root) == want
    assert "COMPLETE (0 cells left to another job)" in r.stdout
    assert not list(root.rglob("*.lock")), "a lock survived its cell"
    assert not list(root.rglob("net_epoch-*")), "per-epoch checkpoints were not pruned"
    assert len(list(root.rglob("net_best_epoch_state.pt"))) == len(want)
    if name in BV2:
        assert not list(root.rglob("features.npy")), "features.npy survived"
        qg = [c for c in want if c[0] == "leg_qg"]
        assert len(list(root.rglob("logits_herwig.npy"))) == len(qg)
        assert len(list(root.rglob("label_herwig.npy"))) == len(qg)
        assert len(list(root.glob("leg_top/*/N*/s*/features/logits.npy"))) == len(want) - len(qg)
        # a head re-initialisation trains on the s1 subset; a seed on its own
        for line in calls.splitlines():
            if "seed_weaver.py" not in line:
                continue
            seed = int(re.search(r"--seed (\d+)", line).group(1))
            sub = re.search(r"train_N(\d+)_s(\d+)\.parquet", line)
            init = re.search(r"leg_(top|qg)/([^/]+)/N", line).group(2)
            expect = seed if init == "scratch" or seed == 1 else 1
            assert int(sub.group(2)) == expect, line
        # the smallest cell runs 10,000 samples per epoch on the 1,000-jet subset
        for line in calls.splitlines():
            if "seed_weaver.py" in line and "/N1000/" in line:
                assert "--samples-per-epoch 10000" in line, line
                assert "train_N1000_s" in line, line
    else:
        n_leg1 = len([c for c in want if c[0] == "leg1"])
        assert not list(root.rglob("features.npy")), "the leg-1 feature matrix survived"
        assert len(list(root.glob("leg1/*/N*/s*/features_v2/logits.npy"))) == n_leg1
        assert len(list(root.glob("leg1/*/N*/s*/features_v2/label188.npy"))) == n_leg1
        for line in calls.splitlines():
            if "seed_weaver.py" in line:
                assert "--seed 1 " in line and "_s1.parquet" in line
                assert "--samples-per-epoch-val 20000" in line
        assert "--fresh-prefix" not in calls


def test_two_shards_into_one_tree_are_disjoint_and_their_union_is_the_plan(new, tmp_path):
    a, b = W3[0], W3[1]
    ia = [i for i in B.INITS_W3 if i[0] in {x[0] for x in _inits(new[a])}]
    ib = [i for i in B.INITS_W3 if i[0] in {x[0] for x in _inits(new[b])}]
    ra, _ = _run(new[a], tmp_path, ia + ib)
    assert ra.returncode == 0, ra.stdout[-2000:] + ra.stderr[-2000:]
    root = tmp_path / "data/results/ft/w2b"
    after_a = _done_cells(root)
    # run b into the SAME tree: new stubs, the data tree shard a left behind
    (tmp_path / "b").mkdir()
    env, _ = _shell_env(tmp_path / "b", ia + ib)
    rb = subprocess.run(["bash", "-c", _redirect(new[b], tmp_path / "b", tmp_path)],
                        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=600)
    assert rb.returncode == 0, rb.stdout[-2000:] + rb.stderr[-2000:]
    after_b = _done_cells(root)
    assert after_a == set(B.cells_legs(ia))
    assert after_b - after_a == set(B.cells_legs(ib))
    assert not (set(B.cells_legs(ia)) & set(B.cells_legs(ib)))


def test_a_cell_locked_by_another_job_is_left_alone_and_a_stale_own_lock_is_re_entered(new, tmp_path):
    name = W3[0]
    inits = [i for i in B.INITS_W3 if i[0] in {x[0] for x in _inits(new[name])}]
    root = tmp_path / "data/results/ft/w2b"
    foreign = root / "leg1" / inits[0][0] / "N1000" / "s1"
    foreign.parent.mkdir(parents=True)
    (foreign.parent / "s1.lock").mkdir()
    (foreign.parent / "s1.lock" / "owner").write_text("ft-legs-w2-raunav\n")
    own = root / "leg2" / inits[1][0] / "N10000" / "s1"
    own.mkdir(parents=True)
    (own / "net_epoch-3_state.pt").touch()              # a half-written cell
    (own.parent / "s1.lock").mkdir()
    (own.parent / "s1.lock" / "owner").write_text(f"ft-legs-w3-a-raunav\n")
    r, _ = _run(new[name], tmp_path, inits)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "LOCKED:" in r.stdout and "COMPLETE (1 cells left to another job)" in r.stdout
    assert not (foreign / "DONE").exists() and (foreign.parent / "s1.lock").is_dir()
    assert (own / "DONE").exists() and not (own.parent / "s1.lock").exists()
    assert list(own.parent.glob("s1.partial.*")), "the half-written attempt was not moved aside"
    assert _done_cells(root) == set(B.cells_legs(inits)) - {("leg1", inits[0][0], 1000, 1)}


def test_the_later_mpm_group_converts_its_init_and_checks_the_fresh_tensors(tmp_path):
    later = B.build(B.PIN_W3, wave3=True, bench_v2=True, later=["mpm-s1"])
    for name, cells_of in (("job-ft-legs-w3-mpm-s1-raunav.yaml", B.cells_legs),
                           ("job-ft-legs-bench-v2-mpm-s1-raunav.yaml", B.cells_bench)):
        live = _live(later[name])
        assert "mpm_init.py --src /data/results/mtx/mtx-mpm-s1/net_epoch-79_state.pt --out /workspace/mpm-s1_trunk.pt" in live
        assert "--fresh-prefix mod.cls_token mod.cls_blocks. mod.norm. --expect-fresh 39" in live
        assert "load-log --log ${OUT}/stdout.log\n" not in live
        r, calls = _run(later[name], tmp_path / name[:-5], B.INITS_LATER["mpm-s1"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        root = tmp_path / name[:-5] / "data/results/ft" / ("w2b" if "w3" in name else "bench_v2")
        assert _done_cells(root) == set(cells_of(B.INITS_LATER["mpm-s1"]))
        assert "mpm_init.py" in calls
        seeds = {int(re.search(r"--seed (\d+)", l).group(1))
                 for l in calls.splitlines() if "seed_weaver.py" in l}
        assert seeds == {1, 2, 3}
