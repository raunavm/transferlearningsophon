"""Legs 3 and 4 -- the two PUBLISHED benchmarks.

These rows drop into community tables, so the recipe is the benchmark's and
every departure from it is a silent incomparability rather than an error.
docs/PRD_PLAN.md 4.1 `[V G]`: 20 epochs, trunk 1e-4 with the head at 5e-3,
constant LR, weight decay 0.01, median + spread over head re-initialisations.
"""
import ast
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments" / "FT" / "k8s" / "job-ft-legs-bench-raunav.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_ft_jobs", ROOT / "scripts" / "build_ft_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfj = _load()


def _args():
    d = yaml.safe_load(SPEC.read_text())
    return d["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_head_rate_is_the_published_one_and_cannot_drift():
    """5e-3 is not written down: it is trunk x mult, so one edit moves both."""
    assert float(bfj.LR_PRETRAINED) * bfj.BENCH_HEAD_MULT == 5e-3
    assert bfj.LR_SCRATCH == "5e-4", "scratch is 5e-4, NOT the 5e-3 head rate"


def test_lr_mult_reaches_weaver_as_one_argument_and_selects_only_the_head():
    """The value carries parens and quotes; bad quoting re-splits it."""
    a = _args()
    m = re.search(r'HEAD_MULT=\(--optimizer-option lr_mult "([^"]+)"\)', a)
    assert m, "lr_mult must be built as a bash ARRAY element, quoted"
    pattern, mult = ast.literal_eval(m.group(1))
    assert mult == bfj.BENCH_HEAD_MULT
    head = ["mod.fc.0.weight", "mod.fc.0.bias", "mod.fc.2.weight"]
    trunk = ["mod.embed.embed.0.weight", "mod.blocks.0.attn.in_proj_weight",
             "mod.cls_blocks.1.attn.out_proj.weight", "mod.norm.weight",
             "mod.trimmed_fc.weight"]
    for n in head:
        assert re.match(pattern, n), f"{n} must take the head rate"
    for n in trunk:
        assert not re.match(pattern, n), f"{n} must stay at the trunk rate"


def test_the_multiplier_is_verified_in_weavers_own_log():
    """If weaver silently ignored the option the row would not be the recipe."""
    a = _args()
    assert f"Parameters with lr multiplied by {bfj.BENCH_HEAD_MULT}" in a
    assert "FATAL: weaver did not apply the head lr multiplier" in a


def test_lr_is_constant_not_weavers_annealing_default():
    a = _args()
    assert "--lr-scheduler none" in a, (
        "weaver's default is flat+decay, which would anneal and make the row "
        "incomparable to the published table")
    # ...and no scheduler is actually SET to an annealing one. Checked on the
    # executable lines only: the comment above the recipe names flat+decay in
    # order to say why it is not used, and a bare substring search on the whole
    # script would fail on that prose.
    code = [l for l in a.splitlines() if not l.strip().startswith("#")]
    for sched in ("flat+decay", "flat+linear", "flat+cos", "one-cycle", "steps"):
        assert f"--lr-scheduler {sched}" not in "\n".join(code)


def test_weight_decay_and_epochs_match_the_published_recipe():
    a = _args()
    assert "--optimizer-option weight_decay 0.01" in a
    assert f"--num-epochs {bfj.BENCH_EPOCHS}" in a
    assert bfj.BENCH_EPOCHS == 20


def test_per_dataset_grids_are_each_benchmarks_own_training_split():
    a = _args()
    assert 'top) echo "1000 10000 100000 1200000"' in a
    assert 'qg)  echo "1000 10000 100000 1600000"' in a


def test_nine_head_reinits_at_nmax_top_only_pretrained_only():
    a = _args()
    assert len(bfj.NMAX_REPS) == 9
    assert " ".join(str(r) for r in bfj.NMAX_REPS) in a
    guard = re.search(r'if \[ "\$\{D\}" = "top" \].*?fi', a, re.S)
    assert guard, "the re-init branch must be guarded"
    g = guard.group(0)
    assert '"${N}" = "${NMAX}"' in g, "N_max only"
    assert '-n "${ckpt}"' in g, "pretrained only -- scratch has no head to re-init"


def test_qg_reads_the_corrected_staging():
    a = _args()
    assert 'qg)  echo "/data/finetune/qg_v2"' in a
    assert "/data/finetune/qg_sub" not in a


def test_binary_heads_and_the_benchmark_configs():
    a = _args()
    assert "-o num_classes 2" in a
    assert "configs/finetune/TopReference.yaml" in a
    assert "configs/finetune/EnergyFlowQG.yaml" in a
    for c in ("TopReference", "EnergyFlowQG"):
        assert (ROOT / "configs" / "finetune" / f"{c}.yaml").exists()


def test_spec_is_mine_and_will_not_retry_forever_into_a_wall():
    d = yaml.safe_load(SPEC.read_text())
    assert "raunav" in d["metadata"]["name"]
    a = _args()
    assert "FAILED_BENCH" in a and "attempt_ok" in a
    assert "space_ok" in a


# ---------------------------------------------------------------------------
# Two defects that made 174/174 bench cells unreachable, and that a green test
# suite did not see: the readout asked for observers the bench data configs do
# not declare, and "9 head re-initialisations" was wired to 9 DATA SUBSETS of
# which the builder writes 3.
# ---------------------------------------------------------------------------

K8S = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "FT" / "k8s"
CFGDIR = pathlib.Path(__file__).resolve().parents[1] / "configs" / "finetune"


def _bench_script():
    d = yaml.safe_load((K8S / "job-ft-legs-bench-raunav.yaml").read_text())
    c = d["spec"]["template"]["spec"]["containers"][0]
    return (c.get("args") or c["command"])[-1]


def test_every_requested_observer_is_declared_by_its_data_config():
    """extract_features hard-fails (exit 4) on an observer the config does not
    declare -- correctly, and AFTER the fine-tune has been paid for."""
    script = _bench_script()
    for line in script.splitlines():
        if "extract_features.py" not in line:
            continue
        assert "--observers" in line, \
            "bench readout must name its observers; the 4-name default is absent " \
            "from TopReference.yaml and EnergyFlowQG.yaml"
    for cfg in ("TopReference.yaml", "EnergyFlowQG.yaml"):
        declared = yaml.safe_load((CFGDIR / cfg).read_text())["observers"]
        assert set(declared) >= {"jet_pt", "jet_energy"}
        for line in script.splitlines():
            if "--observers" in line:
                got = line.split("--observers")[1].split("--")[0].split()
                assert set(got) <= set(declared), (cfg, got, declared)


def test_nmax_reps_do_not_demand_subsets_the_builder_never_writes():
    """The bench subset builder writes seeds 1 2 3. Nine reps indexed to the
    data subset would read train_N..._s{4..9}.parquet, and weaver dies in a
    worker with a message naming no file."""
    sub = yaml.safe_load((K8S / "job-ft-subsets-bench-raunav.yaml").read_text())
    c = sub["spec"]["template"]["spec"]["containers"][0]
    subscript = (c.get("args") or c["command"])[-1]
    seeds = set(subscript.split("--seeds")[1].split("--")[0].split())
    script = _bench_script()
    assert "DSEED=1" in script, "the N_max reps must hold the data subset fixed"
    assert "train_N${N}_s${DSEED}.parquet" in script
    assert "train_N${N}_s${S}.parquet" not in script, \
        "the data subset must not be indexed by the training seed"
    assert seeds == {"1", "2", "3"}
    assert "1" in seeds, "DSEED=1 must be a subset the builder writes"
