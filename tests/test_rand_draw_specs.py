"""The random control's second and third partition draws, at the launch layer.

The control has three draws and each trains once: draw 1 at seed index 1
(mtx-rand-d1-s1b, already trained), draw 2 at seed index 2, draw 3 at seed
index 3, each paired with the 17-class semantic model of the same seed index.
A pair is only interpretable if its two runs differ in the partition and in
nothing else, and the three control runs are only comparable with each other if
they differ in the partition and the seed and in nothing else. Neither is
visible from configs/arms/*.yaml: the rate, budget, loader flags, memory, GPU
model, pin and resume guard all live in the job spec.

So the new specs are DERIVED from the spec that trained, and this file diffs
them against it line by line. A blanket arm rename is the known way to get this
wrong (experiments/RUNS.csv, `spec-copy-hazard`), which is why the allowed edit
sites are written out here rather than read back from the builder.
"""
import hashlib
import importlib.util
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
K8S = ROOT / "experiments" / "MTX" / "k8s"
BASE_TRAIN = "job-mtx-rand-d1-s1b-raunav.yaml"
BASE_MAKEWEIGHT = "job-mtx-makeweight-rand-raunav.yaml"
DRAWS = [(2, 2), (3, 3)]                    # (partition draw, seed index)


def _builder():
    s = importlib.util.spec_from_file_location(
        "build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


bml = _builder()


def _train(d, s):
    return K8S / f"job-mtx-rand-d{d}-s{s}-raunav.yaml"


def _makeweight(d):
    return K8S / f"job-mtx-makeweight-rand-d{d}-raunav.yaml"


def _is_comment(line):
    return line.lstrip().startswith("#")


def _script(path):
    d = yaml.safe_load(path.read_text())
    return d, d["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _executed(path):
    return "\n".join(l for l in _script(path)[1].splitlines() if not _is_comment(l))


def _changed_lines(old, new):
    a, b = old.splitlines(), new.splitlines()
    assert len(a) == len(b), f"line counts differ: {len(a)} vs {len(b)}"
    return [(x, y) for x, y in zip(a, b) if x != y]


# ------------------------------------------------------------- training specs
@pytest.mark.parametrize("d,s", DRAWS)
def test_training_spec_differs_from_draw_1_only_at_the_enumerated_sites(d, s):
    old = (K8S / BASE_TRAIN).read_text()
    new = _train(d, s).read_text()

    # 1. the one inserted block: comments only, inside the metadata header
    block = bml.PAIRING.format(d=d, s=s)
    assert new.count(block) == 1
    assert all(_is_comment(l) for l in block.splitlines())
    assert new.index(block) < new.index("\n  name: ")
    assert f"mtx-r16q1-s{s}" in block, "the pairing rationale must name its pair"
    changed = _changed_lines(old, new.replace(block, ""))

    # 2. executed lines: exactly these twelve, each an exact rewrite
    arm, run = f"RAND_d{d}", f"mtx-rand-d{d}-s{s}"
    cfg = "configs/arms/RAND_d1.yaml"
    fatal = next(l for l in old.splitlines() if l.lstrip().startswith('[ -f "${SRC}" ]'))
    want = {
        "  name: mtx-rand-d1-s1b-raunav": f"  name: {run}-raunav",
        "          RUN_ID=mtx-rand-d1-s1b": f"          RUN_ID={run}",
        f"          CFG={cfg}": f"          CFG=configs/arms/{arm}.yaml",
        "          SIDECAR=configs/arms/RAND_d1.${MD5}.auto.yaml":
            f"          SIDECAR=configs/arms/{arm}.${{MD5}}.auto.yaml",
        "          SRC=/data/results/mtx/makeweight/RAND_d1.${MD5}.auto.yaml":
            f"          SRC=/data/results/mtx/makeweight/{arm}.${{MD5}}.auto.yaml",
        fatal: fatal.replace(cfg, f"configs/arms/{arm}.yaml"),
        "            --arm RAND_d1 \\": f"            --arm {arm} \\",
        "            --seed 1 \\": f"            --seed {s} \\",
        f"            --data-config {cfg} \\": f"            --data-config configs/arms/{arm}.yaml \\",
        "            --tensorboard mtx_RAND_d1_s1b": f"            --tensorboard mtx_{arm}_s{s}",
    }
    live = [(x, y) for x, y in changed if not _is_comment(x)]
    assert all(not _is_comment(y) for _, y in live)
    assert len(live) == 12, [x.strip() for x, _ in live]      # --seed, --data-config x2
    stray = [(x, y) for x, y in live if want.get(x) != y]
    assert not stray, f"executed lines changed outside the declared sites: {stray}"
    assert {x for x, _ in live} == set(want)

    # 3. comment lines: only the arm's name (and the seed, in the title line)
    notes = [(x, y) for x, y in changed if _is_comment(x)]
    assert all(_is_comment(y) for _, y in notes)
    assert len(notes) == 6, [x.strip() for x, _ in notes]
    for x, y in notes:
        assert x.replace("RAND_d1", arm).replace(", seed 1.", f", seed {s}.") == y, (x, y)


@pytest.mark.parametrize("d,s", DRAWS)
def test_everything_outside_the_script_is_identical_to_draw_1(d, s):
    """Pin, GPU model, memory, backoffLimit, image, affinity, exclusions,
    volumes: compared as parsed YAML, so nothing can hide in formatting."""
    old, _ = _script(K8S / BASE_TRAIN)
    new, _ = _script(_train(d, s))
    assert new["metadata"]["name"] == f"mtx-rand-d{d}-s{s}-raunav"
    for j in (old, new):
        j["metadata"].pop("name")
        j["spec"]["template"]["spec"]["containers"][0].pop("args")
    assert old == new

    # and the load-bearing ones by name, so a change to BOTH is still noticed
    c = new["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["GPU_PRODUCT"] == "NVIDIA-GeForce-RTX-3090"
    terms = new["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert {"key": "nvidia.com/gpu.product", "operator": "In",
            "values": ["NVIDIA-GeForce-RTX-3090"]} in terms
    assert c["resources"]["limits"]["memory"] == "76Gi"
    assert new["spec"]["backoffLimit"] == 50


@pytest.mark.parametrize("d,s", DRAWS)
def test_the_recipe_the_run_executes(d, s):
    code = _executed(_train(d, s))
    for must in ("--start-lr 5e-4", "--num-epochs 80 ${RESUME}", "--batch-size 512",
                 "--samples-per-epoch 10240000", "--fetch-by-files --fetch-step 5",
                 "--optimizer ranger", "--use-amp", "-o num_classes 17",
                 "--num-classes 17", "--no-remake-weights", "--lean-val-metrics",
                 'RECIPE="lr=5e-4 epochs=80 arm=${MD5}"'):
        assert must in code, f"draw {d}: does not execute {must!r}"
    assert re.findall(r"--seed (\d+)", code) == [str(s), str(s)], (
        "the manifest writer and seed_weaver must receive the same seed index")
    assert code.index("scripts/write_run_manifest.py") < code.index("seed_weaver.py"), (
        "the manifest (GPU product, weights-block sha256, four sub-seeds, start "
        "time) is written before training so a run that dies still has one")
    assert "R16_Q1" not in code and "RAND_d1" not in _train(d, s).read_text()


@pytest.mark.parametrize("d,s", DRAWS)
def test_the_paired_semantic_run_shares_the_seed_and_the_recipe(d, s):
    """What makes (control draw, semantic model) a PAIR. Memory and the pin
    differ between them and are recorded, not asserted: neither reaches the
    numerics, and the semantic runs are finished."""
    ctrl = _executed(_train(d, s))
    sem = _executed(K8S / f"job-mtx-r16_q1-s{s}-raunav.yaml")
    assert re.findall(r"--seed (\d+)", sem) == [str(s), str(s)]

    def weaver_flags(code):
        call = re.search(r"seed_weaver\.py.*?--tensorboard \S+", code, re.S).group(0)
        return re.sub(r"configs/arms/\w+\.yaml|--tensorboard \S+", "", call).split()
    assert "--fetch-by-files" in weaver_flags(ctrl)
    assert weaver_flags(ctrl) == weaver_flags(sem)
    gpu = 'values: ["NVIDIA-GeForce-RTX-3090"]'
    assert gpu in _train(d, s).read_text()
    assert gpu in (K8S / f"job-mtx-r16_q1-s{s}-raunav.yaml").read_text()


def test_each_control_run_writes_to_its_own_directory():
    specs = [K8S / BASE_TRAIN, K8S / "job-mtx-rand-d1-s1-raunav.yaml"] + [
        _train(d, s) for d, s in DRAWS]
    ids = [re.search(r"RUN_ID=(\S+)", _script(p)[1]).group(1) for p in specs]
    assert len(set(ids)) == len(ids), ids


# ------------------------------------------------------------- sidecar specs
@pytest.mark.parametrize("d,s", DRAWS)
def test_sidecar_spec_differs_from_draw_1s_only_at_the_enumerated_sites(d, s):
    arm = f"RAND_d{d}"
    old = (K8S / BASE_MAKEWEIGHT).read_text()
    new = _makeweight(d).read_text()
    pin = re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', _train(d, s).read_text()).group(1)

    # the two comment paragraphs that described draw 1's history are replaced
    # whole; take both out of both files and compare the rest line by line
    md5 = hashlib.md5((ROOT / "configs" / "arms" / f"{arm}.yaml").read_bytes()).hexdigest()
    for was, now in (bml.MAKEWEIGHT_DERIVED, bml.MAKEWEIGHT_PIN):
        now = now.format(pin=pin, d=d, md5=md5)
        assert old.count(was) == 1 and new.count(now) == 1
        assert all(_is_comment(l) for l in now.splitlines())
        old, new = old.replace(was, ""), new.replace(now, "")

    changed = _changed_lines(old, new)
    want = {
        "  name: mtx-makeweight-rand2-raunav": f"  name: mtx-makeweight-rand-d{d}-raunav",
        "          git clone --depth 1 --branch mtx-s1.29 \\":
            f"          git clone --depth 1 --branch {pin} \\",
        "          run_arm RAND_d1  17": f"          run_arm {arm}  17",
        '          ARM = "RAND_d1"': f'          ARM = "{arm}"',
        '                    open("/data/results/mtx/makeweight/hist_hashes_RAND_d1.json", "w"),':
            f'                    open("/data/results/mtx/makeweight/hist_hashes_{arm}.json", "w"),',
    }
    live = [(x, y) for x, y in changed if not _is_comment(x)]
    assert dict(live) == want
    for x, y in changed:
        if _is_comment(x):
            assert _is_comment(y) and x.replace("RAND_d1", arm) == y, (x, y)
    assert "RAND_d1" not in _makeweight(d).read_text()


@pytest.mark.parametrize("d,s", DRAWS)
def test_training_waits_for_the_sidecar_this_job_writes(d, s):
    """Pasting a neighbour's sidecar onto this md5 would train the neighbour's
    labels block, silently. Each draw names its own, from the same directory
    and the same clone ref the sidecar job uses."""
    arm = f"RAND_d{d}"
    job, mw = _script(_makeweight(d))
    train = _executed(_train(d, s))
    assert "raunav" in job["metadata"]["name"] and job["spec"]["backoffLimit"] == 1
    assert "OUT=/data/results/mtx/makeweight\n" in mw
    assert f"SRC=/data/results/mtx/makeweight/{arm}.${{MD5}}.auto.yaml" in train
    assert "FATAL: no reweighting sidecar" in train
    assert '[ "$N" -eq 1675 ]' in mw, "train + val only; the test split stays out"
    pin = re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', _train(d, s).read_text()).group(1)
    assert f"--branch {pin} " in mw, "sidecar job and training job must clone one ref"


# -------------------------------------------------------------------- the pin
@pytest.mark.parametrize("d,s", DRAWS)
def test_everything_the_run_needs_exists_unchanged_at_the_pinned_tag(d, s):
    """The pin is draw 1's, so all three control runs execute one code state.
    That is only launchable if the tag already carries this draw's config --
    byte-identical, because its md5 names the sidecar."""
    pin = re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', _train(d, s).read_text()).group(1)
    git = ["git", "-C", str(ROOT)]
    if subprocess.run(git + ["rev-parse", "--verify", f"{pin}^{{commit}}"],
                      capture_output=True).returncode != 0:
        pytest.skip(f"tag {pin} is not in this clone")
    at_tag = set(subprocess.run(git + ["ls-tree", "-r", pin, "--name-only"],
                                capture_output=True, text=True).stdout.splitlines())
    needed = [f"configs/arms/RAND_d{d}.yaml",
              "experiments/E1/seed_weaver.py", "scripts/write_run_manifest.py",
              "experiments/MTX/ParT_sophon_arch_mtx.py",
              "experiments/E1/ParT_sophon_arch_10c.py",        # the sidecar pass
              "src/utils/reproducibility.py", "src/utils/resume.py"]
    assert not [f for f in needed if f not in at_tag]
    cfg = f"configs/arms/RAND_d{d}.yaml"
    blob = subprocess.run(git + ["show", f"{pin}:{cfg}"], capture_output=True).stdout
    assert blob == (ROOT / cfg).read_bytes(), (
        f"{cfg} at {pin} is not the file the tests in this tree checked")


# ---------------------------------------------------------------- the builder
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name in (BASE_TRAIN, BASE_MAKEWEIGHT,
                 "job-mtx-r16_q1-s2-raunav.yaml", "job-mtx-r16_q1-s3-raunav.yaml"):
        shutil.copy(K8S / name, tmp_path / name)
    monkeypatch.setattr(bml, "K8S", tmp_path)
    return tmp_path


@pytest.mark.parametrize("d,s", DRAWS)
def test_the_committed_specs_are_the_builders_output(sandbox, d, s):
    out = bml.derive_draw(d, s)
    assert not [r for r in out if "FAILED" in r], out
    for p in (_train(d, s), _makeweight(d)):
        assert (sandbox / p.name).read_text() == p.read_text(), (
            f"{p.name} was hand-edited; regenerate with "
            f"scripts/build_mtx_launch.py --derive-draws {d}:{s}")


def test_the_builder_refuses_rather_than_guesses(sandbox):
    assert "FAILED" in bml.derive_draw(1, 1)[0], "draw 1 is the source"
    assert "FAILED" in bml.derive_draw(4, 4)[0], "there is no fourth draw config"
    assert "FAILED" in bml.derive_draw(2, 9)[0], "no semantic run to pair with"

    # a source spec that gained an executed mention of the arm: the count of a
    # named site is now wrong, and nothing may be written for that spec
    src = sandbox / BASE_TRAIN
    src.write_text(src.read_text().replace(
        "          mkdir -p ${OUT}\n",
        "          mkdir -p ${OUT}\n          echo configs/arms/RAND_d1.yaml\n", 1))
    out = bml.derive_draw(2, 2)
    assert "FAILED" in out[0] and not (sandbox / _train(2, 2).name).exists()
