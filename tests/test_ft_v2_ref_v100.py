"""The V100 reruns of the from-scratch references (scripts/build_ft_jobs.py V2_REF_SRC, PI
2026-10-27): each is the spec that ran, but for its name, ROOT_OUT and GPU (and JetClass's
without its leg 1); under the bash harness it leaves exactly the reference cells, by the same
training and prediction commands as the spec it reruns; and the final read-outs read them."""
import collections
import importlib.util
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments/FT/k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_jobs_ref", "scripts/build_ft_jobs.py")
W = _load("wave3_harness_ref", "tests/test_wave3_specs.py")
V = _load("v2ft_harness_ref", "tests/test_ft_v2_finetune_specs.py")
SPECS = B.v2_ref_v100_specs()
WANT = {"jc2": {"leg1": 12}, "jc1": {"leg2": 12}, "bench": {"leg_top": 12, "leg_qg": 12}}


def _name(text):
    return re.search(r"^metadata:\n(?:[ \t]+.*\n)*?[ \t]+name: *(\S+)", text, re.M).group(1)


def test_the_reruns_on_disk_are_the_generators():
    assert set(SPECS) == {f"job-ft-v2-ref-v100-{p}-raunav.yaml" for p in B.V2_REF_SRC}
    for name, text in SPECS.items():
        assert (K8S / name).read_text() == text, name


@pytest.mark.parametrize("part", sorted(B.V2_REF_SRC))
def test_a_rerun_differs_from_the_spec_that_ran_only_in_name_root_gpu(part):
    import difflib
    src, root = B.V2_REF_SRC[part]
    a, b = (K8S / src).read_text(), SPECS[f"job-ft-v2-ref-v100-{part}-raunav.yaml"]
    old, new = _name(a), _name(b)
    for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if l.startswith(("+++", "---", "@@")):
            continue
        assert (old in l or new in l or "ROOT_OUT=" in l or "RTX-3090" in l or "V100" in l
                or "FROM-SCRATCH REFERENCE ON V100" in l or "ROOT_OUT (" in l
                or (part == "jc1" and "for N in" in l)), l
    assert f'value: "{B.V2_REF_GPU}"' in b and "RTX-3090" not in b
    assert re.search(r'REPO_REF\n\s+value: "([^"]+)"', a).group(1) == re.search(r'REPO_REF\n\s+value: "([^"]+)"', b).group(1)
    assert f"ROOT_OUT={B.V2_REF_ROOT}/{part}\n" in b


def _run(text, part, tmp):
    if part == "jc2":
        r, calls = V._v2_run(text, tmp, V._inits(text))
    else:
        env, cl = W._shell_env(tmp, [])
        r = subprocess.run(["bash", "-c", W._redirect(text, tmp, tmp)], capture_output=True, text=True,
                           env=env, cwd=tmp, timeout=900)
        calls = cl.read_text() if cl.exists() else ""
    done = [p.parent for p in tmp.rglob("DONE") if "results" in str(p)]
    return r, done, calls.replace(str(tmp), "T")


@pytest.mark.parametrize("part", sorted(B.V2_REF_SRC))
def test_a_rerun_leaves_the_reference_cells_by_the_commands_that_made_them(part, tmp_path):
    src, root = B.V2_REF_SRC[part]
    new_text, old_text = SPECS[f"job-ft-v2-ref-v100-{part}-raunav.yaml"], (K8S / src).read_text()
    r, done, calls = _run(new_text, part, tmp_path / "new")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    base = tmp_path / "new" / B.V2_REF_ROOT.lstrip("/") / part
    got = collections.Counter(str(d.relative_to(base)).split("/")[0] for d in done)
    assert dict(got) == WANT[part]
    assert all(d.relative_to(base).parts[1] == B.SCRATCH_REF for d in done)
    r0, _, calls0 = _run(old_text, part, tmp_path / "old")
    assert r0.returncode == 0
    norm = lambda c, name, root_: sorted(l.replace(name, "NAME").replace(root_, "ROOT") for l in c.splitlines()
                                         if "weaver" in l and "/leg1/" not in l or ("weaver" in l and part != "jc1"))
    assert norm(calls, _name(new_text), f"{B.V2_REF_ROOT}/{part}") == norm(calls0, _name(old_text), root)
    assert len(norm(calls, _name(new_text), f"{B.V2_REF_ROOT}/{part}")) == sum(WANT[part].values())


def test_the_final_readouts_read_the_v100_references_and_the_trial_the_ones_it_read():
    for leg, part, sub in (("leg1", "jc2", "leg1"), ("leg2", "jc1", "leg2")):
        assert B._v2_refs(leg, True) == [(f"{B.V2_REF_ROOT}/{part}/{sub}/{B.SCRATCH_REF}", B.SIZES)]
    assert B._v2_refs("bench", True) == [(f"{B.V2_REF_ROOT}/bench/leg_{d}/{B.SCRATCH_REF}", B.BENCH_SIZES[d])
                                         for d in B.BENCH_SETS]
    for leg in B.V2_READOUT_LEGS:
        final = B.v2_readout("best70", leg, B.PIN_V2_READOUT)[1]
        trial = (K8S / f"job-ft-v2-best70-done18-{leg}-metrics-raunav.yaml").read_text()
        assert B.V2_REF_ROOT in final and B.V2_REF_ROOT not in trial
