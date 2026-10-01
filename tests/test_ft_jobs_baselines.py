"""The two fine-tuning baselines at their own recipes (scripts/build_ft_jobs.py,
BASELINE RECIPES), emitted as the later groups scratch-v2 and mpm-s1-v2, and
applied to mpm-s2 / mpm-s3 when their pretraining is Complete.

Pinned here:
  * the from-scratch cells carry ParT's from-scratch recipe for their dataset,
    read off the commands the emitted scripts actually run under bash;
  * the self-supervised multiplier matches EXACTLY the parameters pretraining
    never trained (class token, class-attention blocks, final norm) plus the
    head, on the real fine-tuning models, and nothing the trunk delivers;
  * every supervised and launched spec is byte-identical to what is on disk,
    and the new specs differ from the cells they supersede only at the
    declared sites.
"""
import ast
import collections
import importlib.util
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


# The shell harness (stubbed python3 / weaver / df, /data redirected) is wave 3's.
T = _load("wave3_harness", "tests/test_wave3_specs.py")
B = T.B
MI = _load("mpm_init_for_baselines", "experiments/FT/mpm_init.py")

LEGS = {g: f"job-ft-legs-baseline-{g}-raunav.yaml" for g in ("scratch-v2", "mpm-s1-v2")}
BENCH = {g: f"job-ft-legs-bench-baseline-{g}-raunav.yaml" for g in ("scratch-v2", "mpm-s1-v2")}

# ParT FROM SCRATCH, model "ParT" in jet-universe/particle_transformer @ 2925bdb:
# train_JetClass.sh (--start-lr 1e-3, no weight decay), train_TopLandscape.sh and
# train_QuarkGluon.sh (lr="1e-3", --optimizer-option weight_decay 0.01); all three
# --optimizer ranger, --batch-size 512, --use-amp, no --lr-scheduler (weaver's
# default flat+decay).
PART_FROM_SCRATCH = {"jetclass": ("1e-3", None), "top": ("1e-3", "0.01"), "qg": ("1e-3", "0.01")}


@pytest.fixture(scope="module")
def refs():
    return B.build(B.PIN_REFS, wave3=True, bench_v2=True, later=["scratch-v2", "mpm-s1-v2"])


def _removed_added(old: str, new: str):
    a, b = collections.Counter(old.splitlines()), collections.Counter(new.splitlines())
    return list((a - b).elements()), list((b - a).elements())


def _head_mult_value(text: str) -> tuple:
    """The lr_mult value exactly as weaver receives it: bash expands the array."""
    line = next(ln.strip() for ln in T._live(text).splitlines() if ln.strip().startswith("HEAD_MULT=("))
    out = subprocess.run(["bash", "-c", line + '; printf "%s\\n" "${HEAD_MULT[@]}"'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[:2] == ["--optimizer-option", "lr_mult"] and len(out) == 3, out
    return ast.literal_eval(out[2])


def _run(text, tmp_path, inits, nmult=B.MPM_N_MULT):
    """Wave 3's shell run, with a weaver stub that lists `nmult` multiplied names."""
    env, calls = T._shell_env(tmp_path, inits)
    py = tmp_path / "bin" / "python3"
    old = 'echo "Parameters with lr multiplied by 50";;'
    assert old in py.read_text()
    py.write_text(py.read_text().replace(old, (
        'echo "[t] INFO: Parameters with lr multiplied by 50:"; '
        'for i in $(seq 1 ${NMULT_STUB}); do echo " - mod.p${i}"; done; echo "[t] INFO: next";;')))
    env["NMULT_STUB"] = str(nmult)
    r = subprocess.run(["bash", "-c", T._redirect(text, tmp_path, tmp_path)], capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=600)
    return r, calls.read_text() if calls.exists() else ""


def _opt(line: str, flag: str):
    m = re.search(rf"{flag} (\S+)", line)
    return m.group(1) if m else None


# ------------------------------------------------------------------ the groups

def test_the_baseline_groups_are_the_new_names_and_the_unlaunched_self_supervised_inits():
    assert B.REFS_GROUPS == {"scratch-v2", "mpm-s1-v2", "mpm-s2", "mpm-s3"}
    assert "mpm-s1" not in B.REFS_GROUPS and B.MPM_HEAD_ONLY == {"mpm-s1"}
    assert B.INITS_LATER["scratch-v2"] == [("scratch-v2", "", 0, [1, 2, 3])]
    (n, c, k, s), = B.INITS_LATER["mpm-s1-v2"]
    assert (k, s) == (0, [1, 2, 3]) and B.MPM_SOURCE[n] == B.MPM_SOURCE["mpm-s1"]
    # the recipes the fine-tuning of PRETRAINED models uses do not move
    assert (B.LR_PRETRAINED, B.BENCH_HEAD_MULT, B.LR_SCRATCH) == ("1e-4", 50, "5e-4")
    assert B.LR_SCRATCH_PART == "1e-3"


def test_bench_v3_holds_no_scratch_cell_so_nothing_there_is_superseded():
    assert not [n for n, *_ in B.INITS_BENCH_V3 if not B._GRANULARITY.match(n)]
    for f in sorted(K8S.glob("job-ft-legs-bench-v3-last-*-raunav.yaml")):
        assert not any(s[0].startswith("scratch") for s in T._inits(f.read_text())), f.name


# --------------------------------------------------------------- from scratch

def test_the_scratch_legs_differ_from_the_old_scratch_cells_only_in_parts_recipe():
    """Same epochs, samples per epoch, validation, checkpoint rule, prune, lock:
    everything but the rate and the weight decay is wave 3's script."""
    old = B.legs_w3([("scratch", "", 0, [1, 2, 3])], "S")
    new = B.legs_w3(B.INITS_LATER["scratch-v2"], "S")
    removed, added = _removed_added(old, new)
    allowed = ('[ "${name}" = "scratch" ] && continue', "--optimizer-option weight_decay 0.01\"",
               "LR=__LR_SCRATCH__; MULT=()", "head_lr_mult=__HEAD_MULT__ weight_decay=0.01 wave=3")
    assert len(removed) == 6 and all(any(a in ln for a in allowed) for ln in removed), removed
    assert "\n".join(added).count("LR=1e-3; MULT=()") == 2
    assert "--lr-scheduler" not in new and "--optimizer-option weight_decay" not in new


def test_the_scratch_benchmarks_differ_from_the_old_scratch_cells_only_in_parts_recipe():
    old = B.legs_bench_v2([("scratch", "", 0, [1, 2, 3])], "S")
    new = B.legs_bench_v2(B.INITS_LATER["scratch-v2"], "S")
    removed, _ = _removed_added(old, new)
    allowed = ('[ "${name}" = "scratch" ] && continue', "# The benchmark recipe",
               "# constant LR the published", "# flat+decay, which would", "# comparable to the community",
               "--lr-scheduler none", "LR=__LR_SCRATCH__; MULT=()", "lr_schedule=constant")
    assert len(removed) == 8 and all(any(a in ln for a in allowed) for ln in removed), removed
    assert "--lr-scheduler" not in T._live(B.build(B.PIN_REFS, bench_v2=True, later=["scratch-v2"])[
        BENCH["scratch-v2"]])


@pytest.mark.parametrize("which", ["legs", "bench"])
def test_every_scratch_cell_runs_parts_from_scratch_recipe_for_its_dataset(refs, tmp_path, which):
    name = (LEGS if which == "legs" else BENCH)["scratch-v2"]
    inits = B.INITS_LATER["scratch-v2"]
    r, calls = _run(refs[name], tmp_path, inits)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    root = tmp_path / "data/results/ft" / ("w2b" if which == "legs" else "bench_v2")
    want = set((B.cells_legs if which == "legs" else B.cells_bench)(inits))
    assert T._done_cells(root) == want and len(want) == 24
    runs = [ln for ln in calls.splitlines() if "seed_weaver.py" in ln]
    assert len(runs) == len(want)
    for ln in runs:
        leg = re.search(r"/(leg1|leg2|leg_top|leg_qg)/", ln).group(1)
        lr, wd = PART_FROM_SCRATCH["jetclass" if leg in ("leg1", "leg2") else leg[4:]]
        assert _opt(ln, "--start-lr") == lr, ln
        assert _opt(ln, "--optimizer-option weight_decay") == wd, ln
        assert "--lr-scheduler" not in ln and "lr_mult" not in ln and "--load-model-weights" not in ln
        assert _opt(ln, "--batch-size") == "512" and _opt(ln, "--optimizer") == "ranger" and "--use-amp" in ln
    for ln in (ln for ln in calls.splitlines() if "smoke_checks.py manifest" in ln):
        assert "head_lr_mult=1 " in ln and "recipe=part-from-scratch" in ln
        assert ("lr_schedule=flat+decay" in ln) == (which == "bench")


# ------------------------------------------------------------ self-supervised

@pytest.fixture(scope="module")
def ft_models():
    """The fine-tuning models every self-supervised cell builds: leg 1 (162 classes),
    leg 2 (the E1 10-class architecture), top and q/g (2 classes)."""
    pytest.importorskip("torch")
    pytest.importorskip("weaver")
    from weaver.utils.data.config import DataConfig
    mtx = _load("mtx_arch_baselines", "experiments/MTX/ParT_sophon_arch_mtx.py")
    e1 = _load("e1_arch_baselines", "experiments/E1/ParT_sophon_arch_10c.py")
    out = {}
    for key, arch, cfg, k in (("leg1", mtx, "JetClassII_L162_noweight.yaml", 162),
                              ("leg2", e1, "JetClassI_sophon_noweight.yaml", 10),
                              ("top", mtx, "TopReference.yaml", 2),
                              ("qg", mtx, "EnergyFlowQG.yaml", 2)):
        dc = DataConfig.load(str(ROOT / "configs/finetune" / cfg), load_observers=False)
        out[key] = arch.get_model(dc, num_classes=k, fc_params=[(512, 0.1)])[0]
    return out


@pytest.mark.parametrize("which", ["legs", "bench"])
def test_the_multiplier_matches_exactly_what_pretraining_never_trained(refs, ft_models, which):
    """weaver 0.4.17 optim(): every parameter whose name re.match-es the pattern
    takes start_lr x mult. The set must be the head plus every tensor the
    converted init does not supply (mpm_init.KEPT is all the trunk pretraining
    trained), and must equal load-log's --fresh-prefix set plus the head."""
    pattern, mult = _head_mult_value(refs[(LEGS if which == "legs" else BENCH)["mpm-s1-v2"]])
    assert mult == B.BENCH_HEAD_MULT
    for key, model in ft_models.items():
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        matched = {n for n in names if re.match(pattern, n)}
        untrained = {n for n in names if n.startswith(MI.FRESH)}
        head = {n for n in names if n.startswith("mod.fc.")}
        assert len(untrained) == MI.N_FRESH == B.MPM_N_FRESH == 39, key
        assert matched == untrained | head, (key, sorted(matched ^ (untrained | head)))
        assert matched == {n for n in names if not n.startswith(MI.KEPT)}, key
        assert len(matched) == B.MPM_N_MULT == 43, key
    assert B.MPM_FRESH == MI.FRESH


@pytest.mark.parametrize("which", ["legs", "bench"])
def test_the_self_supervised_cells_differ_from_mpm_s1_only_in_the_multiplier(which):
    fn = B.legs_w3 if which == "legs" else B.legs_bench_v2
    old, new = fn(B.INITS_LATER["mpm-s1"], "S"), fn(B.INITS_LATER["mpm-s1-v2"], "S")
    removed, added = _removed_added(old, new)
    n = 1 if which == "bench" else 2
    assert len(removed) == 1 + n and all("HEAD_MULT=(" in ln or "head_lr_mult=" in ln for ln in removed)
    added = "\n".join(added)
    assert f"(r'{B.MPM_LR_MULT}', __HEAD_MULT__)" in added
    assert added.count(f'[ "${{NMULT}}" -eq {B.MPM_N_MULT} ]') == n
    assert added.count("lr_mult_prefixes=mod.fc.,mod.cls_token,mod.cls_blocks.,mod.norm.") == n


@pytest.mark.parametrize("which", ["legs", "bench"])
def test_self_supervised_cells_run_and_stop_if_weaver_multiplies_the_head_alone(refs, tmp_path, which):
    name = (LEGS if which == "legs" else BENCH)["mpm-s1-v2"]
    inits = B.INITS_LATER["mpm-s1-v2"]
    r, calls = _run(refs[name], tmp_path / "ok", inits)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    root = tmp_path / "ok/data/results/ft" / ("w2b" if which == "legs" else "bench_v2")
    assert T._done_cells(root) == set((B.cells_legs if which == "legs" else B.cells_bench)(inits))
    assert "mpm_init.py --src /data/results/mtx/mtx-mpm-s1/net_epoch-79_state.pt" in calls.replace(
        str(tmp_path / "ok"), "")
    runs = [ln for ln in calls.splitlines() if "seed_weaver.py" in ln]
    assert runs and all(_opt(ln, "--start-lr") == B.LR_PRETRAINED for ln in runs)
    assert all(f"(r'{B.MPM_LR_MULT}', 50)" in ln for ln in runs)
    assert all(("--lr-scheduler none" in ln) == (which == "bench") for ln in runs)
    # the defect: weaver matched only the four head tensors -- the cell must stop
    r, _ = _run(refs[name], tmp_path / "bad", inits, nmult=4)
    assert r.returncode != 0 and "gave the head rate to 4 parameters" in r.stdout
    assert not T._done_cells(tmp_path / "bad/data/results/ft" / ("w2b" if which == "legs" else "bench_v2"))


def test_mpm_s2_and_s3_get_the_fix_and_stay_off_disk_until_their_pretraining_is_complete():
    later = B.build(B.PIN_RESUME, wave3=True, bench_v2=True, later=["mpm-s2", "mpm-s3"])
    assert len(later) == 4
    for name, text in later.items():
        assert f"(r'{B.MPM_LR_MULT}', 50)" in text and "-eq 43 ]" in text, name
        assert yaml.safe_load(text)["spec"]["backoffLimit"] == B.ROBUST_BACKOFF
        assert not (K8S / name).exists(), f"{name} written before its pretraining job is Complete"
    assert not ({"mpm-s2", "mpm-s3"} & B.LAUNCHED_LATER)


# ------------------------------------------------- nothing else moves; the specs

def test_every_launched_later_spec_is_unchanged_and_keeps_the_head_only_multiplier():
    launched = B.build(B.PIN_W3, wave3=True, bench_v2=True, later=["mpm-s1", "rand-d2", "rand-d3"])
    for name, text in launched.items():
        assert (K8S / name).read_text() == text, f"{name} moved"
        assert B.MPM_LR_MULT not in text and "NMULT" not in text and "recipe=part" not in text


def test_the_new_specs_on_disk_are_the_generators_pinned_at_the_untagged_tag(refs):
    assert set(refs) == set(LEGS.values()) | set(BENCH.values())
    # outside the two globs make_tables.py reads the paper's recipe table from
    assert not [n for n in refs if n.startswith(("job-ft-legs-w2", "job-ft-legs-w3", "job-ft-legs-bench-v2-"))]
    assert B.PIN_REFS == "mtx-s1.64"
    for name, text in refs.items():
        assert (K8S / name).read_text() == text, f"{name} is stale; regenerate"
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav")
        # one retry on the job that ran to completion under it; the retry
        # policy (tests/test_ft_retry_policy.py) on the rest
        assert d["spec"]["backoffLimit"] == (B.ROBUST_BACKOFF if B.robust(d["metadata"]["name"]) else 1)
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert {"name": "REPO_REF", "value": B.PIN_REFS} in c["env"]
        assert {"name": "GPU_PRODUCT", "value": "NVIDIA-GeForce-RTX-3090"} in c["env"]
        assert set(B.BAD_NODES + B.LOST_GPU_NODES) <= T._excluded(text)
        assert "BASELINE RECIPES" in text


def test_the_pin_guards():
    with pytest.raises(SystemExit, match="predates"):
        B.build(B.PIN_W3, wave3=True, later=["scratch-v2"])
    with pytest.raises(SystemExit, match="does not exist"):
        B.verify_pin("mtx-s1.9999", False)
    B.verify_pin("mtx-s1.9999", True)               # declared: the working tree carries it all
