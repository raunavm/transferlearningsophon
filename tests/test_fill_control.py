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
        # The unmasked leg is EXTRACTED at the same checkpoint as the masked
        # legs, not truncated out of features_v2 (which came from
        # net_best_epoch_state.pt). Two legs from different checkpoints make
        # the control measure a checkpoint change as well as the mask.
        assert f"--arm {arm}_unmasked" in a
        assert "features_v2" not in a, (
            "features_v2 is a best-epoch cache; the control must not mix it "
            "with epoch-79 masked features")


def test_masked_and_unmasked_use_the_same_jet_count():
    a = _args()
    assert f"N={bfc.N_JETS}" in a
    # one unmasked leg + one per mask, per arm
    assert a.count("--max-jets ${N}") == len(bfc.ARMS) * (len(bfc.MASKS) + 1)
    # (the truncation leg is gone: every leg is now an extraction
    # capped by --max-jets, counted above)


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


def test_the_legs_row_alignment_is_actually_checked():
    """This file used to CLAIM a check that never ran.

    probe.py refuses arms whose label188 sha256 differs, but only among the
    arms passed to ONE invocation -- and the control runs unmasked and each
    mask as separate invocations, so nothing ever compared a masked leg
    against its own baseline. The emitted script must diff the legs itself.
    """
    a = _args()
    assert "row alignment" in a
    assert "label188.npy" in a
    # the comparison must run BEFORE the first probe, or a mismatch is only
    # discovered after the numbers have already been produced
    assert a.index("row alignment") < a.index("probe.py")
    for arm, _, _ in bfc.ARMS:
        assert arm in a.split("row alignment")[1].split("probe.py")[0]


def test_unmasked_leg_has_no_resume_short_circuit():
    """${U} is the same path the superseded launch filled by truncating
    features_v2 (best epoch). A `[ -f ... ] ||` guard there silently reuses the
    stale leg and reports masking-effect + checkpoint-change as the fill
    penalty, exit 0."""
    y = (pathlib.Path(__file__).resolve().parents[1] / "experiments" / "EVAL"
         / "k8s" / "job-eval-fillcontrol-raunav.yaml").read_text()
    for line in y.splitlines():
        if "--arm" in line and "_unmasked" in line:
            continue
        assert not ("[ -f ${U}/label188.npy ]" in line), \
            "the unmasked leg must be re-extracted, not resumed"
    cleared = [l for l in y.splitlines() if l.strip() == "rm -rf ${U}"]
    assert len(cleared) == 4, "each arm's unmasked leg must be cleared"


def test_legs_are_diffed_on_the_checkpoint_not_only_the_labels():
    """label188 is a property of the data: the same jets at two different
    checkpoints hash identically, so the label diff cannot see a checkpoint
    confound."""
    y = (pathlib.Path(__file__).resolve().parents[1] / "experiments" / "EVAL"
         / "k8s" / "job-eval-fillcontrol-raunav.yaml").read_text()
    assert "checkpoint_sha256" in y
    assert "extracted at a DIFFERENT" in y
