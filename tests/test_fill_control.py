"""The zero-fill control must mask on the extraction side, not the fine-tune side.

DECISIONS_PENDING item 16(d) approves zero-filling top/q-g ONLY alongside a
control that masks the same slots on JetClass-II and shows the arm ordering
survives. Two things about that control fail silently if got wrong:

1. Extracting through configs/finetune/JetClassII_L162_mask*.yaml instead of the
   base mask would hand the probes 162-GROUP labels under the same array name as
   the native 188-way jet_label they key on. Every probe still runs, still
   returns an AUC, and measures a different task.
2. The 335 test files are INTERLEAVED across families (Res2P_0250, Res34P_1075,
   QCD_0350, ...), so the list cannot be rebuilt by brace expansion. A
   reconstructed list would read different jets than the baseline it is
   subtracted from.
"""
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments" / "EVAL" / "k8s" / "job-eval-fillcontrol-raunav.yaml"


def _gen():
    spec = importlib.util.spec_from_file_location(
        "build_fill_control", ROOT / "scripts" / "build_fill_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfc = _gen()


def _args():
    if not SPEC.exists():
        pytest.skip("fill-control spec not generated")
    return yaml.safe_load(SPEC.read_text())["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_extraction_uses_the_base_mask_not_the_l162_mask():
    a = _args()
    for mask in bfc.MASKS:
        assert f"configs/finetune/JetClassII_base_mask{mask}.yaml" in a
        assert f"configs/finetune/JetClassII_L162_mask{mask}.yaml" not in a, (
            "extracting through the L162 control would give the probes "
            "162-group labels under the native-label array name")


def test_the_test_file_list_is_copied_from_the_committed_extraction_spec():
    a = _args()
    committed = bfc.test_list().split()
    assert len(committed) == 335
    m = re.search(r'TEST="([^"]+)"', a)
    assert m, "no TEST list in the emitted script"
    assert m.group(1).split() == committed


def test_the_file_list_is_not_reconstructible_by_brace_expansion():
    """Guards the reason test_list() reads the spec instead of generating."""
    files = [f.split("/")[-1] for f in bfc.test_list().split()]
    naive = ([f"Res2P_{i:04d}.parquet" for i in range(200, 250)]
             + [f"Res34P_{i:04d}.parquet" for i in range(860, 1075)]
             + [f"QCD_{i:04d}.parquet" for i in range(280, 350)])
    assert files != naive, "if this ever matches, the interleaving assumption changed"
    assert files[0].startswith("Res2P") and files[1].startswith("Res34P"), \
        "the list should interleave families, which is why 400k rows hold all 188 classes"


def test_every_arm_has_a_checkpoint_and_a_baseline_declared():
    a = _args()
    for arm, ckpt, k in bfc.ARMS:
        assert ckpt in a and f"--num-classes {k}" in a
        assert f"/data/results/eval/mtx-{arm}/features_v2" in a


def test_masked_and_unmasked_use_the_same_jet_count():
    a = _args()
    assert f"N={bfc.N_JETS}" in a
    assert a.count("--max-jets ${N}") == len(bfc.ARMS) * len(bfc.MASKS)
    assert a.count("--n ${N}") == len(bfc.ARMS)


def test_probes_run_on_unmasked_and_on_every_mask():
    a = _args()
    for tag in ["unmasked"] + bfc.MASKS:
        assert f"--out ${{ROOT_OUT}}/probe_{tag}" in a


def test_spec_is_mine_gpu_and_avoids_the_broken_nodes():
    d = yaml.safe_load(SPEC.read_text())
    assert "raunav" in d["metadata"]["name"]
    lim = d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert lim["nvidia.com/gpu"] == "1"
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    ex = [v for e in terms if e["key"] == "kubernetes.io/hostname" for v in e["values"]]
    assert any("fullerton" in v for v in ex)


def test_committed_spec_is_what_the_generator_emits():
    assert SPEC.read_text() == bfc.build(bfc.PIN), \
        "re-run scripts/build_fill_control.py"
