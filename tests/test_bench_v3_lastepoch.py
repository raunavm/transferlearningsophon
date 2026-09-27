"""Bench v3: C2 and C3's 40 cells rerun so the registered checkpoint rule
(PRESPEC 2.8, the last epoch) can be computed at all. The rerun must BE bench v2
in everything but the declared sites, or the comparison of the two checkpoint
rules would also compare two recipes."""
import difflib
import importlib.util
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
K8S = ROOT / "experiments" / "FT" / "k8s"
spec = importlib.util.spec_from_file_location("build_ft_jobs", ROOT / "scripts" / "build_ft_jobs.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def _changed(inits):
    v2 = B.legs_bench_v2(inits, "S").splitlines()
    v3 = B.legs_bench_v3_last(inits, "S").splitlines()
    return [ln for ln in difflib.unified_diff(v2, v3, lineterm="", n=0)
            if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]


def test_exactly_the_twenty_granularity_models_at_seed_one_and_full_training_sets():
    names = sorted(n for n, *_ in B.INITS_BENCH_V3)
    assert len(names) == 20 and "l162-s1b" in names and "l162-s1" not in names
    assert all(s == [1] for *_, s in B.INITS_BENCH_V3)
    cells = B.cells_bench_v3(B.INITS_BENCH_V3)
    assert len(cells) == 40 and {(c[0], c[2]) for c in cells} == {("leg_top", 1_200_000),
                                                                   ("leg_qg", 1_600_000)}


def test_the_rerun_is_bench_v2_except_at_the_declared_sites():
    removed = [ln for ln in _changed(B.INITS_BENCH_V3[:4]) if ln.startswith("-")]
    allowed = ("ROOT_OUT=", 'top) echo "1000 10000 100000 1200000"', 'qg)  echo "1000 10000 100000 1600000"',
               "FAILED_BENCH.", 'REPS="1 2 3 4 5"', "wave=bench-v2", "FT BENCH V2")
    stray = [ln for ln in removed if not any(a in ln for a in allowed)]
    assert not stray, stray
    added = "\n".join(ln for ln in _changed(B.INITS_BENCH_V3[:4]) if ln.startswith("+"))
    assert "--lr-scheduler" not in added and "--start-lr" not in added and "seed_weaver" not in added


def test_the_last_epoch_is_scored_and_kept_and_the_best_epoch_still_is():
    s = B.legs_bench_v3_last(B.INITS_BENCH_V3[:4], "S")
    assert B.BENCH_V3_LAST_EPOCH == B.BENCH_EPOCHS - 1 == 19
    assert "--checkpoint ${OUT}/net_epoch-19_state.pt" in s and "--out ${OUT}/features_last " in s
    assert "--checkpoint ${OUT}/net_best_epoch_state.pt" in s and "--out ${OUT}/features " in s
    # the last epoch is scored BEFORE the prune deletes net_epoch-*, and kept past it
    assert s.index("features_last") < s.index("rm -f ${OUT}/net_epoch-*_state.pt")
    assert "cp ${OUT}/net_epoch-19_state.pt ${OUT}/net_last_epoch_state.pt" in s
    assert f"ROOT_OUT={B.BENCH_V3_ROOT}" in s and B.BENCH_V2_ROOT not in s


def test_the_committed_specs_are_what_the_generator_emits():
    specs = B.build("mtx-s1.52", bench_v3=True)
    assert len(specs) == 5
    for name, text in specs.items():
        assert (K8S / name).read_text() == text, f"{name} is stale; regenerate"
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav") and d["spec"]["backoffLimit"] == 50
