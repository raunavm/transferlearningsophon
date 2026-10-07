"""scripts/build_aoj_jobs.py --v2: the real-data run of the v2 grid is the first run's
shards and the current fit procedure on the v2 checkpoints -- every v2 model whose output
layer the scores are defined on, each checkpoint resolved in the pod by extract_v2's rule
before any download, a file two checkpoints share scored once -- in two runs: tiers 1 and 2
(the freeze), which derive the pooled shape, and tier 3, which holds it."""
import difflib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("scripts/build_aoj_jobs.py", "build_aoj_jobs")
FT = _load("scripts/build_ft_jobs.py", "build_ft_jobs")
LAUNCH = B._launch()
GRID = json.loads((REPO / "configs/arms/v2_grid.json").read_text())["arms"]
T12, T3 = B.v2_specs([1, 2]), B.v2_specs([3])
SHARDS = [T12[B.V2_K8S / f"job-aoj-v2-t12-s{i}-raunav.yaml"] for i in range(B.N_SHARDS)]
FIT = T12[B.V2_K8S / "job-aoj-v2-t12-fit-raunav.yaml"]
FIT3 = T3[B.V2_K8S / "job-aoj-v2-t3-fit-raunav.yaml"]
INJ = T12[B.V2_K8S / "job-aoj-v2-t12-injection-raunav.yaml"]
FREEZE_FIT = f"{B.V2_OUT_ROOT}/t12/fit_v6/results.json"


def _script(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _terms(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]


def test_each_tier_scores_every_checkpoint_of_every_tree_vocabulary_run_and_nothing_else():
    want = {1: {"L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS"},
            2: {"R16_Q1_MASS_LM"}, 3: {"R63_Q1", "R29_Q1"}}
    by_name = {a["name"]: a for a in GRID}
    for tier, arms in want.items():
        models = B.v2_models([tier])
        assert models[0] == B.MODELS[0], "the public checkpoint is every run's reference row and runs first"
        v2 = models[1:]
        assert {m.arm for m in v2} == arms and len({m.name for m in models}) == len(models)
        assert len(v2) == sum(by_name[a]["runs"] for a in arms) * len(B.V2_CHECKPOINTS)
        for m in v2:
            a = by_name[m.arm]
            assert (m.k, m.num_reg) == (a["num_classes"], int(a["mass_lambda"] is not None))
            assert m.run_dir == f"{LAUNCH.V2_ROOT}/{m.run_id}" and m.tag in B.V2_CHECKPOINTS
            assert m.name == f"{m.run_id.removeprefix('mtx-')}-{m.tag}"
            assert m.checkpoint == f"/workspace/ckpt/{m.name}.pt"
    assert B.V2_CHECKPOINTS == ("best70", "wavg", "bestval", "best70_bn", "bestval_bn")
    assert [len(B.v2_models(t)) for t in ([1, 2], [3])] == [176, 51]
    # no random partition, flavour pair, self-supervised or leave-one-family-out arm, in any tier
    every = {m.arm for m in B.v2_models([1, 2, 3])[1:]}
    assert not any(a.startswith(("RAND", "FLAV", "MPM")) or a.endswith("LOFO4P") for a in every)
    with pytest.raises(SystemExit, match="hold no arm"):
        B.v2_models([4])


def test_the_freeze_is_tiers_one_and_two_and_every_other_run_holds_its_shape():
    assert B.v2_shape_from([2, 1]) is None
    assert B.v2_shape_from([3]) == FREEZE_FIT
    for tiers in ([1], [2], [1, 3], [1, 2, 3]):
        with pytest.raises(SystemExit, match="the freeze run is --v2 1 2"):
            B.v2_specs(tiers)


def test_the_scores_are_defined_on_every_vocabulary_the_run_scores():
    D = _load("experiments/AOJ/discriminants.py", "discriminants")
    for m in B.v2_models([1, 2, 3])[1:]:
        assert D.n_outputs(m.rung) == m.k
        for score in ("three_prong", "prong_only"):
            num, den = (D.nodes(m.rung, s) for s in D.SCORES[score])
            assert num and den and not set(num) & set(den)


def test_each_head_and_mass_output_is_read_from_its_own_grid_training_spec():
    B.verify_heads(B.v2_models([1, 2, 3]))
    m = B.v2_models([1])[1]
    with pytest.raises(SystemExit, match="says num_classes"):
        B.verify_heads([m._replace(k=m.k + 1)])
    with pytest.raises(SystemExit, match="mass output"):
        B.verify_heads([m._replace(num_reg=1)])
    with pytest.raises(SystemExit, match="does not train run"):
        B.verify_heads([m._replace(run_id="mtx-l188-s9")])


@pytest.mark.parametrize("path", sorted(T12) + sorted(T3), ids=lambda p: p.name)
def test_every_v2_spec_is_valid_yaml_and_bash_and_carries_the_retry_policy(path):
    text = {**T12, **T3}[path]
    doc = yaml.safe_load(text)
    assert doc["metadata"]["name"].endswith("-raunav") and doc["metadata"]["name"] == path.name[4:-5]
    r = subprocess.run(["bash", "-n"], input=_script(text), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    spec = doc["spec"]
    assert spec["backoffLimit"] == FT.ROBUST_BACKOFF
    assert spec["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [FT.EXIT_HALT]
    assert [c["name"] for c in spec["template"]["spec"]["containers"]] == ["main"]
    assert spec["template"]["spec"]["containers"][0]["image"] == LAUNCH.V2_IMAGE
    assert f'--branch "{B.V2_PIN}"' in text


def test_a_v2_shard_is_the_first_run_shard_with_only_the_listed_changes():
    """Staging, row alignment and scoring are render_shard's own, line for line."""
    models = B.v2_models([1, 2])
    disc = '&& python3 experiments/AOJ/discriminants.py --name "$1" --rung "$6" --structures three_prong'
    full = '|| { echo "FATAL: /data is ${USED}% full"; exit 1; }'
    for i, fs in enumerate(B.shards()):
        one = [ln.strip() for ln in _script(B.render_shard(i, fs, models)).splitlines()]
        two = [ln.strip() for ln in _script(SHARDS[i]).splitlines()]
        diff = [ln for ln in difflib.ndiff(one, two) if ln[:2] in ("- ", "+ ")]
        assert [ln[2:] for ln in diff if ln[0] == "-"] == [
            f"OUT={B.OUT_ROOT}/shard{i}", f'[ "${{USED}}" -lt 95 ] {full}', f'git clone --depth 1 --branch "{B.PIN}" \\',
            "pip install --no-cache-dir -q pyarrow h5py || exit 1", f"{disc} \\"]
        assert [ln[2:] for ln in diff if ln[0] == "+"] == [
            f"OUT={B.V2_OUT_ROOT}/t12/shard{i}",
            *[ln.strip() for ln in B._failure_accounting("${OUT}", FT.EXIT_HALT).splitlines()],
            f'[ "${{USED}}" -lt 85 ] {full}', f'git clone --depth 1 --branch "{B.V2_PIN}" \\',
            f"pip install --no-cache-dir -q {LAUNCH.V2_PYARROW} h5py || exit 1",
            'python3 experiments/AOJ/v2_checkpoints.py --links /workspace/ckpt --record "${OUT}/checkpoints.json" \\',
            *[f"{m.name}={m.run_dir}:{m.tag} \\" for m in models[1:]], f"|| exit {FT.EXIT_HALT}",
            '[ -f "/workspace/ckpt/$1.same" ] && { echo "skip $1 (the file of $(cat /workspace/ckpt/$1.same))"; '
            'return 0; }',
            f"{disc} prong_only \\",
            "# a name whose file another name holds gets that name's scores (v2_checkpoints.py)",
            "for f in /workspace/ckpt/*.same; do", '[ -f "${f}" ] || continue',
            'm=$(basename "${f}" .same); first=$(cat "${f}")',
            '[ -f "${OUT}/scores_${first}.npz" ] || { echo "FATAL: ${first} is not scored"; exit 1; }',
            'ln -sfn "scores_${first}.npz" "${OUT}/scores_${m}.npz"',
            'ln -sfn "scores_${first}.json" "${OUT}/scores_${m}.json"', "done"]


def test_the_shard_resolves_every_checkpoint_before_the_download_and_scores_every_model():
    s = _script(SHARDS[0])
    resolve = s.index("python3 experiments/AOJ/v2_checkpoints.py")
    assert s.index("pip install") < resolve < s.index("for c in /workspace/ckpt/") < s.index("fetch_and_stage ()")
    models = B.v2_models([1, 2])
    v2 = [m for m in models if isinstance(m, B.V2Model)]
    args = re.findall(r"^\s+(\S+)=(\S+):(\w+) \\$", s, re.M)
    assert args == [(m.name, m.run_dir, m.tag) for m in v2]
    lines = re.findall(r'^\s*scored (\S+) "(\S+)" (\d+) (\d) (\S+) (\S+)(?: & p\d=\$!)?$', s, re.M)
    assert [ln[0] for ln in lines] == [m.name for m in models]
    for (name, ckpt, k, reg, arm, rung), m in zip(lines, models):
        assert (ckpt, int(k), int(reg), arm, rung) == (m.checkpoint, m.k, m.num_reg, m.arm, m.rung)
    assert "--structures three_prong prong_only" in s and "two_prong " not in s
    assert f"OUT={B.V2_OUT_ROOT}/t12/shard0" in s and '"${OUT}/checkpoints.json"' in s


def _run_scoring(tmp_path, same):
    """The shard's scoring block and the same-file links under bash, with a fake python3;
    `same` maps a name to the name holding its file, as v2_checkpoints.py writes it."""
    s = _script(SHARDS[0])
    block = s[s.index("# Every step is chained"):s.rindex('touch "${OUT}/DONE"')]
    block = block.replace("/scratch/", f"{tmp_path}/scratch/").replace("/workspace/ckpt", f"{tmp_path}/ckpt")
    (tmp_path / "ckpt").mkdir()
    for name, first in same.items():
        (tmp_path / "ckpt" / f"{name}.same").write_text(first)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(f"""#!/bin/bash
args="$*"
if [[ "$args" == *extract_features.py* ]]; then
  echo "$args" | sed -E 's/.*--out [^ ]*extract\\/([^ ]+).*/\\1/' >> {tmp_path}/extracted.txt; exit 0
fi
if [[ "$args" == *discriminants.py* ]]; then
  name=$(sed -E 's/.*--name ([^ ]+).*/\\1/' <<< "$args"); echo "$name" > "$OUT/scores_$name.npz"
  echo "$name" > "$OUT/scores_$name.json"; exit 0
fi
exit 9
""")
    (bindir / "python3").chmod(0o755)
    harness = (f"set -euo pipefail\nexport OUT={tmp_path}/out\nFILES=f\nCFG=c\nGPU=fake\n"
               f"mkdir -p $OUT\n{block}touch \"$OUT/DONE\"\n")
    return subprocess.run(["bash", "-c", harness], capture_output=True, text=True,
                          env={"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)})


def test_a_file_two_checkpoints_share_is_scored_once_and_both_names_read_its_scores(tmp_path):
    same = {"l188-s1-bestval": "l188-s1-best70", "l188-s1-bestval_bn": "l188-s1-best70_bn"}
    r = _run_scoring(tmp_path, same)
    assert r.returncode == 0, r.stdout + r.stderr
    extracted = (tmp_path / "extracted.txt").read_text().split()
    assert sorted(extracted) == sorted(m.name for m in B.v2_models([1, 2]) if m.name not in same)
    out = tmp_path / "out"
    for name, first in same.items():
        for ext in ("npz", "json"):
            link = out / f"scores_{name}.{ext}"
            assert link.is_symlink() and link.read_text() == f"{first}\n"
    assert {p.stem.removeprefix("scores_") for p in out.glob("scores_*.npz")} == {m.name for m in B.v2_models([1, 2])}
    assert (out / "DONE").exists()


def test_a_shard_with_no_shared_file_scores_every_name(tmp_path):
    r = _run_scoring(tmp_path, {})
    assert r.returncode == 0, r.stdout + r.stderr
    assert len((tmp_path / "extracted.txt").read_text().split()) == len(B.v2_models([1, 2]))


def test_shards_take_no_v2_grid_product_and_the_gpu_memory_parallel_scoring_is_sized_for():
    for t in SHARDS:
        terms = _terms(t)
        gpu = [x for x in terms if x["key"] == "nvidia.com/gpu.product"]
        assert {g["operator"] for g in gpu} == {"Exists", "NotIn"}
        assert set(next(g for g in gpu if g["operator"] == "NotIn")["values"]) == set(LAUNCH.V2_GPU_BY_RUN.values())
        assert {"key": "nvidia.com/gpu.memory", "operator": "Gt", "values": [str(B.MIN_GPU_MEMORY_MIB)]} in terms
        hosts = next(x for x in terms if x["key"] == "kubernetes.io/hostname")
        assert hosts["operator"] == "NotIn" and set(hosts["values"]) >= set(B.SIM_BAD_NODES) | set(LAUNCH.V2_BAD_NODES)
        res = yaml.safe_load(t)["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        assert res["nvidia.com/gpu"] == "1"


def _run_preconditions(tmp_path, fail):
    """The shard's resolver call and the checkpoint loop under bash, with a fake python3
    that links every model (or fails)."""
    s = _script(SHARDS[0])
    block = s[s.index("python3 experiments/AOJ/v2_checkpoints.py"):s.index("REF=")]
    block = block.replace("/workspace/ckpt", f"{tmp_path}/ckpt")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "target.pt").write_text("x")
    fake = bindir / "python3"
    fake.write_text(f"""#!/bin/bash
[ "{int(fail)}" = 1 ] && {{ echo "FATAL: no BatchNorm twin"; exit 1; }}
for a in "$@"; do case "$a" in *=*:*) ln -s {tmp_path}/target.pt "{tmp_path}/ckpt/${{a%%=*}}.pt";; esac; done
""")
    fake.chmod(0o755)
    harness = f"set -euo pipefail\nOUT={tmp_path}/out\n{block}echo PRECONDITIONS-PASSED\n"
    return subprocess.run(["bash", "-c", harness], capture_output=True, text=True,
                          env={"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)})


def test_a_resolved_shard_passes_its_preconditions_and_an_unresolved_one_halts(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ok = _run_preconditions(tmp_path / "a", fail=False)
    assert ok.returncode == 0 and "PRECONDITIONS-PASSED" in ok.stdout, ok.stdout + ok.stderr
    assert len(list((tmp_path / "a" / "ckpt").iterdir())) == len(B.v2_models([1, 2])) - 1
    bad = _run_preconditions(tmp_path / "b", fail=True)
    assert bad.returncode == FT.EXIT_HALT and "PRECONDITIONS-PASSED" not in bad.stdout, \
        "a checkpoint that does not resolve halts the job before the download; it is not retried"


def test_the_freeze_fit_waits_for_every_shard_and_builds_the_shape_from_best70_alone():
    s = _script(FIT)
    assert f"seq 0 {B.N_SHARDS - 1}" in s and 'shard${i}/DONE' in s
    assert s.count(f"exit {FT.EXIT_HALT}; }}") == 1, "an unfinished shard halts; it is not retried"
    steps = ["merge_shards.py --shards ${SHARDS}", "peak_fit.py --peaks top", "export_fit_bins.py --peak top",
             "fit_v6.py --v2"]
    assert [s.index(x) for x in steps] == sorted(s.index(x) for x in steps)
    assert s.index("export OMP_NUM_THREADS=1") < s.index("python3 experiments/AOJ/merge_shards.py")
    assert s.index("df --output=pcent /data") < s.index("git clone")
    assert "--eff 0.01 --toys 200" in s and f"--workers {B.V2_FIT_CPU - 1}" in s
    models = B.v2_models([1, 2])
    names = re.search(r"for m in ([^;]+); do SCORES", s).group(1).split()
    assert names == [m.name for m in models]
    pool = re.search(r"--shape-pool ([^\\]+)\\", s).group(1).split()
    assert pool == [m.name for m in models if getattr(m, "tag", None) == "best70"] and len(pool) == 35
    assert "--shape-from" not in s, "the freeze run derives the pooled shape"
    assert f"OUT={B.V2_OUT_ROOT}/t12" in s and "nvidia.com/gpu" not in FIT
    assert "BEGIN-TAR" in s and "tar czf - fit fit_v6 analysis_v6" in s


def test_the_tier_three_fit_holds_the_freeze_runs_shape_and_waits_for_it():
    s = _script(FIT3)
    hold = f'[ -f "{FREEZE_FIT}" ] || {{ echo "FATAL: no freeze fit {FREEZE_FIT}; its shape is held here"; exit {FT.EXIT_HALT}; }}'
    assert hold in s and s.index(hold) < s.index("git clone")
    assert f"--shape-from {FREEZE_FIT}" in s and s.index("fit_v6.py --v2") < s.index("--shape-from")
    assert f"OUT={B.V2_OUT_ROOT}/t3" in s and f"{B.V2_OUT_ROOT}/t12" not in s.replace(FREEZE_FIT, "")
    names = re.search(r"for m in ([^;]+); do SCORES", s).group(1).split()
    assert names == [m.name for m in B.v2_models([3])]


def _run_fit(tmp_path, out):
    """The freeze fit job's script under bash with fake git and python3 that write what each
    step writes; returns the steps python3 was asked to run."""
    s = _script(FIT).replace("/scratch/", f"{tmp_path}/scratch/").replace("/workspace/", f"{tmp_path}/ws/")
    s = s.replace(f"OUT={B.V2_OUT_ROOT}/t12", f"OUT={out}")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "steps.log"
    (bindir / "git").write_text(f"#!/bin/bash\n[ \"$1\" = clone ] && mkdir -p {tmp_path}/ws/transferlearningsophon\nexit 0\n")
    (bindir / "df").write_text("#!/bin/bash\necho x; echo 10%\n")
    (bindir / "base64").write_text("#!/bin/bash\ncat > /dev/null\n")     # GNU -w 0; not the BSD one here
    (bindir / "python3").write_text(f"""#!/bin/bash
echo "$1" >> {log}
out=$(sed -E 's/.*--out ([^ ]+).*/\\1/' <<< "$*")
case "$1" in
  *merge_shards.py) mkdir -p "$out"; touch "$out/merge_manifest.json" "$out/closure.json";;
  *peak_fit.py) mkdir -p "$out"; touch "$out/results.json" "$out/histograms.npz";;
  *export_fit_bins.py) touch "$out";;
  *fit_v6.py) mkdir -p "$out"; touch "$out/results.json"; an=$(sed -E 's/.*--analysis-out ([^ ]+).*/\\1/' <<< "$*")
              mkdir -p "$an"; touch "$an/aoj_top.json";;
esac
""")
    for f in bindir.iterdir():
        f.chmod(0o755)
    r = subprocess.run(["bash", "-c", s], capture_output=True, text=True,
                       env={"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    steps = [pathlib.Path(x).name for x in log.read_text().split()] if log.exists() else []
    if log.exists():
        log.unlink()
    return r, steps


def test_a_retried_fit_redoes_only_the_steps_that_did_not_finish(tmp_path):
    out = tmp_path / "t12"
    for i in range(B.N_SHARDS):
        (out / f"shard{i}").mkdir(parents=True)
    r, _ = _run_fit(tmp_path, out)
    assert r.returncode == FT.EXIT_HALT and "shard 0 has not finished" in r.stdout
    for i in range(B.N_SHARDS):
        (out / f"shard{i}" / "DONE").touch()
    r, steps = _run_fit(tmp_path, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert steps == ["merge_shards.py", "peak_fit.py", "export_fit_bins.py", "fit_v6.py"]
    assert {p.name for p in (out / "fit").iterdir()} == {"merge_manifest.json", "closure.json", "results.json",
                                                         "histograms.npz", "bins.npz"}
    assert (out / "fit_v6" / "results.json").exists() and (out / "analysis_v6" / "aoj_top.json").exists()
    (out / "analysis_v6" / "aoj_top.json").unlink()
    assert _run_fit(tmp_path, out)[1] == ["fit_v6.py"], "the fits on the jets finished; only fit_v6 reruns"
    assert _run_fit(tmp_path, out)[1] == [], "everything finished: the log copy only"


def test_the_v2_pin_carries_every_flag_the_v2_run_passes():
    B.verify_pin(B.V2_PIN, not_yet_tagged=True, flags=B.V2_NEEDED_FLAGS)
    s = _script(SHARDS[0]) + _script(FIT) + _script(FIT3)
    for script in set(re.findall(r"python3 (experiments/\S+\.py)", s)):
        assert (REPO / script).exists()
    assert B.V2_NEEDED_FLAGS["experiments/AOJ/fit_v6.py"] in _script(FIT3)


def test_the_builder_emits_the_v2_runs_and_leaves_the_first_run_alone():
    for tiers, specs in (("1 2", T12), ("3", T3)):
        r = subprocess.run([sys.executable, str(REPO / "scripts/build_aoj_jobs.py"), "--v2", *tiers.split(),
                            "--check-only", "--pin-not-yet-tagged"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        listed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("ok ")]
        assert sorted(listed) == sorted(str(p.relative_to(REPO)) for p in specs)
        assert len(listed) == B.N_SHARDS + (2 if tiers == "1 2" else 1), "shards, the fit, the freeze's injection"
        assert all(p.startswith("experiments/AOJ/k8s/v2/") for p in listed)
    r = subprocess.run([sys.executable, str(REPO / "scripts/build_aoj_jobs.py"), "--v2", "1", "--check-only",
                        "--pin-not-yet-tagged"], capture_output=True, text=True)
    assert r.returncode != 0 and "the freeze run is --v2 1 2" in r.stderr
    for path, text in B.specs().items():
        assert path.read_text() == text, f"{path.name}: the first run's spec changed"


# ---- the injection test of the freeze run ----
def test_the_injection_runs_the_first_runs_two_studies_on_the_freeze_fit_for_best70_alone():
    s = _script(INJ)
    root = f"{B.V2_OUT_ROOT}/t12"
    calls = re.findall(r"injection_test.py toys (.+?) \\\n\s+--committed (\S+) --fit (\S+) \\\n"
                       r"(?:\s+--pooled (\S+) \\\n)?\s+--names ([^\\]+)\\\n\s+--workers (\d+) --out (\S+)", s)
    assert [c[0] for c in calls] == list(B.V2_INJECTION_STUDIES.values())
    v1 = {study: args for study, ((args, _),) in ((k, v[:1]) for k, v in B.TOY_STUDIES.items())}
    assert B.V2_INJECTION_STUDIES == {"top": v1["top"] + " --variants float", "pooled": v1["pooled"][0:v1["pooled"].index(" --pooled")]}
    best = [m.name for m in B.v2_models([1, 2]) if getattr(m, "tag", None) == "best70"]
    assert len(best) == 35
    for (args, committed, fit, pooled, names, workers, out), study in zip(calls, B.V2_INJECTION_STUDIES):
        assert (committed, fit) == (f"{root}/fit/bins.npz", f"{root}/fit/results.json")
        assert pooled == (f"{root}/fit_v6/results.json" if study == "pooled" else "")
        assert names.split() == (["reference"] if study == "top" else []) + ["sophon-public", *best]
        assert out == f'"${{OUT}}/toys_{study}_0.jsonl"' and int(workers) == B.TOYS_CPU - 1
    summary = 'injection_test.py summary --toys "${OUT}/toys_top_0.jsonl" "${OUT}/toys_pooled_0.jsonl" --out "${OUT}/summary.json"'
    assert summary in s and s.index(summary) > s.index("toys_pooled_0.jsonl")
    wait = f'[ -f "{root}/analysis_v6/aoj_top.json" ] || {{ echo "FATAL: the freeze fit has not finished"; exit {FT.EXIT_HALT}; }}'
    assert s.index("df --output=pcent /data") < s.index(wait) < s.index("git clone")
    assert s.index("export OMP_NUM_THREADS=1") < s.index("injection_test.py toys")
    assert f"pip install --no-cache-dir -q {LAUNCH.V2_PYARROW}" in s and "nvidia.com/gpu" not in INJ
    assert f"OUT={root}/injection" in s and "tar czf - summary.json toys_*.jsonl" in s
    assert not any("injection" in p.name for p in T3), "the injection test validates the freeze run's fit only"


def test_the_first_runs_injection_specs_regenerate_byte_identical():
    k8s = B.K8S
    want = {k8s / "job-aoj-injection-bins-v1-raunav.yaml": B.render_injection_bins(),
            k8s / "job-aoj-read-injection-v1-raunav.yaml": B.render_read_injection(),
            **{k8s / f"job-aoj-injection-toys-{s}-v1-raunav.yaml": B.render_injection_toys(s) for s in B.TOY_STUDIES}}
    assert len(want) == 2 + len(B.TOY_STUDIES)
    for path, text in want.items():
        assert path.read_text() == text, path.name
