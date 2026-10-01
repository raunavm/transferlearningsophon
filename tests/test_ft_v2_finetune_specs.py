"""The v2 fine-tuning specs (scripts/build_ft_jobs.py, audit 2026-09-29): every v2
run fine-tuned once per checkpoint rule on every leg, resolved from its own run,
on subsets verified against the sha256 record; JetClass-II on the held-out draw
with a fixed validation sample; manifests that record the steps weaver runs; and
each spec run under bash with stubs.
"""
import importlib.util
import json
import pathlib
import re
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_jobs_v2ft", "scripts/build_ft_jobs.py")


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


# ================================================================ fine-tuning
# The v2 fine-tuning specs need the sha256 record the staging job wrote
# (V2_SHA_TABLE, committed from /data/finetune/jc2_v2) and configs/arms/v2_grid.json.

T = _load("wave3_harness_v2", "tests/test_wave3_specs.py")
SCRATCH = "job-ft-v2-legs-scratch-raunav.yaml"


# the specs' content, regardless of whether they fit on /data today
@pytest.fixture(scope="module")
def v2():
    return B.build(B.PIN_V2, v2=True, headroom_gb=1e6)


def _live(text):
    return "\n".join(l for l in _args(text).splitlines() if not l.lstrip().startswith("#"))


def _inits(text):
    s = re.search(r'INITS="([^"]*)"', _args(text)).group(1).split()
    return [(x.split(":")[0], x.split(":")[1], int(x.split(":")[2]), [1]) for x in s]


def test_the_v2_specs_on_disk_are_the_generators(v2):
    on_disk = {p.name for p in K8S.glob("job-ft-v2-*.yaml")}
    assert SCRATCH in on_disk and on_disk <= set(v2)
    for name in on_disk:
        assert (K8S / name).read_text() == v2[name], name


# Measured 2026-10-01 (job-ft-inspect-retries-3-raunav): 1,099,511,627,776 B, 876,915,720,192 used.
HEADROOM_GB = (0.85 * 1099511627776 - 876915720192) / 1e9


def test_the_storage_budget_admits_the_scratch_reference_and_refuses_the_rest(v2, capsys):
    every = sum(B.spec_bytes(B.cells_legs(_inits(t)) if "-legs-" in n and n != SCRATCH else
                             B.cells_bench_v2ckpt(_inits(t)) if "-bench-" in n else
                             [c for c in B.cells_legs(B.INITS_LATER["scratch-v2"]) if c[0] == "leg1"])
                for n, t in v2.items())
    assert every > HEADROOM_GB * 1e9                      # the whole v2 set does not fit today
    with pytest.raises(SystemExit, match="Making room is the PI's call"):
        B.build(B.PIN_V2, v2=True, headroom_gb=HEADROOM_GB)
    got = B.build(B.PIN_V2, v2=True, headroom_gb=HEADROOM_GB, only=["ft-v2-legs-scratch"])
    assert set(got) == {SCRATCH} and got[SCRATCH] == v2[SCRATCH]
    assert "all 65: " in capsys.readouterr().out
    # 12 leg-1 cells and one cell's 50 epochs of checkpoints
    assert B.spec_bytes([c for c in B.cells_legs(B.INITS_LATER["scratch-v2"]) if c[0] == "leg1"]) == \
        12 * B.V2_CELL_BYTES["leg1"] + 50 * B.EPOCH_PAIR_BYTES


def test_the_expected_cell_list_on_disk_is_the_generators_and_covers_every_spec(v2):
    on_disk = json.loads((ROOT / B.V2_EXPECTED).read_text())
    assert on_disk == B.v2_expected_cells()
    for name, t in v2.items():
        if name == SCRATCH:
            continue
        rule = re.search(r"-(bestval|wavg)-", name).group(1)
        cells = B.cells_legs(_inits(t)) if "-legs-" in name else B.cells_bench_v2ckpt(_inits(t))
        for leg, n, N, s in cells:
            assert f"{n}/N{N}/s{s}" in on_disk[f"{rule}/{leg}"]
    assert len(on_disk["scratch/leg1"]) == 12


def test_every_v2_run_is_fine_tuned_once_per_rule_on_every_leg(v2):
    runs = {n for n, *_ in B.v2_runs()}
    for rule in B.V2_RULES:
        for kind in ("legs", "bench"):
            got = [n for name, t in v2.items() if name.startswith(f"job-ft-v2-{kind}-{rule}-")
                   for n, *_ in _inits(t)]
            assert sorted(got) == sorted(runs), (rule, kind)
    for name, t in v2.items():
        names = [n for n, *_ in _inits(t)]
        assert all(n.startswith("mpm-v2-") for n in names) or not any(n.startswith("mpm-") for n in names)


def test_every_v2_gpu_job_is_mine_pinned_to_the_3090_with_the_retry_policy(v2):
    for name, t in v2.items():
        assert B.resumable(yaml.safe_load(t)["metadata"]["name"]) and B.RETRY_NOTE_RESUME in t
        assert "cell_resume.py prepare" in t and "seed_weaver.py" not in _live(t)
        assert set(B.GPU_LOST_LATER) <= T._excluded(t)
        d = yaml.safe_load(t)
        assert d["metadata"]["name"].endswith("-raunav")
        assert d["spec"]["backoffLimit"] == B.ROBUST_BACKOFF and "podFailurePolicy" in d["spec"]
        assert 'values: ["NVIDIA-GeForce-RTX-3090"]' in t and 'values: ["us-west"]' in t
        env = {e["name"]: e.get("value") for e in d["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["REPO_REF"] == B.PIN_V2 and B._tag_index(B.PIN_V2) >= B._tag_index(B.PIN_RESUME)
        assert '[ "$p" -lt 85 ] && [ "$g" -ge 50 ]' in t and "-lt 92" not in t


def test_each_pretrained_init_is_resolved_by_its_rule_from_its_own_run(v2):
    run = {n: d for n, d, *_ in B.v2_runs()}
    for name, t in v2.items():
        if name == SCRATCH:
            assert "ft_v2.py resolve" not in t
            continue
        rule = re.match(r"job-ft-v2-\w+-(bestval|wavg)-", name).group(1)
        live = _live(t)
        for n, *_ in _inits(t):
            link = f"/workspace/ckpt/{n}.src.pt" if n.startswith("mpm-") else f"/workspace/ckpt/{n}.pt"
            line = f"ft_v2.py resolve --run-dir {run[n]} --rule {rule} --link {link} || exit ${{HALT}}"
            assert line in live, (name, n)
            assert live.index(line) < live.index("INITS=")
        assert live.count("cp /workspace/ckpt/${name}.pt.json ${OUT}/init_checkpoint.json") == (
            2 if "-legs-" in name else 1)


def test_jetclass2_reads_the_held_out_draw_and_validates_on_the_whole_fixed_sample(v2):
    for name, t in v2.items():
        if "-legs-" not in name:
            continue
        head = _args(t).split(B._LEG2_MARK.strip())[0]
        leg1 = "\n".join(l for l in head.splitlines() if not l.lstrip().startswith("#"))
        assert "SUB2=/data/finetune/jc2_v2" in leg1 and "/data/finetune/jc2\n" not in leg1
        assert "--data-train ${SUB2}/train_N${N}_s1.parquet --data-val ${SUB2}/val.parquet" in leg1
        assert "--steps-per-epoch-val -1" in leg1 and "--samples-per-epoch-val" not in leg1
        assert f'grep -c "Processed {B.V2_VAL_JETS} entries"' in leg1


def test_manifests_record_the_steps_weaver_runs_and_each_cell_checks_them(v2):
    for name, t in v2.items():
        live = _live(t)
        assert "$((N/512))" not in live
        n = 1 if "-bench-" in name or name == SCRATCH else 2
        assert live.count("('steps_per_epoch', ") == n
        if "-legs-" in name:
            assert live.count("samples_per_epoch=${SPE} steps_per_epoch=$((SPE/512))") == n
        else:
            assert "steps_per_epoch=$(($(samples_for ${N})/512))" in live


def test_every_file_a_job_reads_is_verified_against_the_record_first(v2):
    table = json.loads((ROOT / B.V2_SHA_TABLE).read_text())["files"]
    for name, t in v2.items():
        live = _live(t)
        files = re.search(r"ft_v2.py verify --table (\S+) \\\n\s*--files (.*?) \|\| exit", live, re.S)
        assert files.group(1) == B.V2_SHA_TABLE
        listed = files.group(2).split()
        assert set(listed) <= set(table), set(listed) - set(table)
        jc2 = {f"{B.V2_SUBSETS}/train_N{n}_s1.parquet" for n in B.SIZES} | {f"{B.V2_SUBSETS}/val.parquet"}
        if "-legs-" in name:
            assert jc2 <= set(listed)
            assert ("/data/finetune/jc1/val.parquet" in listed) == (name != SCRATCH)
        else:
            assert set(B.HERWIG_TEST) <= set(listed) and not jc2 & set(listed)


def test_the_benchmarks_score_the_full_set_at_its_last_epoch_and_run_no_repeats(v2):
    for name, t in v2.items():
        if "-bench-" not in name:
            continue
        live = _live(t)
        assert 'REPS="1 2 3 4 5"' not in live
        assert 'if [ "${N}" = "${NMAX}" ]; then' in live
        assert f"net_epoch-{B.BENCH_V3_LAST_EPOCH}_state.pt" in live and "features_last" in live


def test_the_scratch_reference_is_jetclass2_only_at_parts_from_scratch_recipe(v2):
    live = _live(v2[SCRATCH])
    for gone in ("SUB1", "jc1", "test_20M", "leg2", "e1/arm_s"):
        assert gone not in live, gone
    assert "ROOT_OUT=/data/results/ft_v2/scratch" in live
    assert 'LR=1e-3; MULT=()' in live and "recipe=part-from-scratch" in live
    assert re.search(r'INITS="scratch-v2::0:1,2,3"', live)


# ------------------------------------------------------------------ the shell

def _v2_run(text, tmp_path, inits, bad_steps=False, no_val=False):
    """Wave 3's shell harness -- its ft_weaver stub writes weaver's argument dump,
    the epochs and the 'Processed' lines -- with a resolve stub that links a file."""
    env, calls = T._shell_env(tmp_path, [])       # v2 checkpoints come from the resolve stub
    py = tmp_path / "bin" / "python3"
    old = "  *ft_weaver.py*)\n"
    assert py.read_text().count(old) == 1
    py.write_text(py.read_text().replace(old, (
        '  *ft_v2.py\\ resolve*)\n'
        '    K=$(next_after --link "$@"); mkdir -p "$(dirname "$K")"; touch "$K"; echo "{}" > "$K.json";;\n'
        + old)))
    data = tmp_path / "data"
    sub = data / "finetune/jc2_v2"
    sub.mkdir(parents=True)
    for f in [f"train_N{n}_s1.parquet" for n in B.SIZES] + ["val.parquet", "DONE"]:
        (sub / f).touch()
    if bad_steps:
        env["BAD_STEPS"] = "1"
    if no_val:
        env["NO_VAL"] = "1"
    r = subprocess.run(["bash", "-c", T._redirect(text, tmp_path, tmp_path)], capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=900)
    return r, calls.read_text() if calls.exists() else ""


def _first(v2, prefix):
    return next(n for n in sorted(v2) if n.startswith(prefix))


@pytest.mark.parametrize("pick", ["scratch", "legs", "bench", "legs-mpm", "bench-mpm"])
def test_a_v2_spec_runs_under_bash_and_leaves_its_cells(v2, tmp_path, pick):
    name = {"scratch": SCRATCH,
            "legs": _first(v2, "job-ft-v2-legs-bestval-t1"),
            "bench": _first(v2, "job-ft-v2-bench-wavg-t1"),
            "legs-mpm": "job-ft-v2-legs-bestval-t2mpm-raunav.yaml",
            "bench-mpm": "job-ft-v2-bench-wavg-t2mpm-raunav.yaml"}[pick]
    inits = _inits(v2[name])
    r, calls = _v2_run(v2[name], tmp_path, inits)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rule = "scratch" if name == SCRATCH else re.search(r"-(bestval|wavg)-", name).group(1)
    root = tmp_path / "data/results/ft_v2" / rule
    if name == SCRATCH:
        want = {("leg1", "scratch-v2", n, s) for n in B.SIZES for s in (1, 2, 3)}
    elif "-legs-" in name:
        want = set(B.cells_legs(inits))
    else:
        want = set(B.cells_bench_v2ckpt(inits))
    assert T._done_cells(root) == want
    for line in calls.splitlines():
        if "ft_weaver.py" in line and "/leg1/" in line:
            assert "jc2_v2/train_N" in line and "_s1.parquet" in line, line
    if name != SCRATCH:
        n_ic = len(list(root.rglob("init_checkpoint.json")))
        assert n_ic == len(want)
    if "-bench-" in name:
        nmax = [c for c in want if c[2] == B.BENCH_SIZES[c[0][4:]][-1]]
        assert len(list(root.rglob("net_last_epoch_state.pt"))) == len(nmax)
    # every cell ran the resumable logic: one attempt, its best epoch recorded
    assert len(list(root.rglob("best_epoch.json"))) == len(want)
    assert len(list(root.rglob("ATTEMPTS"))) == len(want) and "--load-epoch" not in calls


@pytest.mark.parametrize("bad", ["bad_steps", "no_val"])
def test_a_cell_whose_log_disagrees_with_its_manifest_fails(v2, tmp_path, bad):
    r, _ = _v2_run(v2[SCRATCH], tmp_path, _inits(v2[SCRATCH]), **{bad: True})
    assert r.returncode != 0
    assert ("steps per epoch" if bad == "bad_steps" else "fixed sample") in r.stdout
