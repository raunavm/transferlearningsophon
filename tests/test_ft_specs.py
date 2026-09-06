"""CI for the fine-tuning wave's job specs (scripts/build_ft_jobs.py,
scripts/build_mass_jobs.py): the things a referee, or a later reader of the
ledger, would need to be true and that a hand edit could silently break.

Run:  python3 -m pytest tests/test_ft_specs.py -v
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FT = ROOT / "experiments" / "FT" / "k8s"
MTX = ROOT / "experiments" / "MTX" / "k8s"
EXTRACT = ROOT / "experiments" / "EVAL" / "k8s" / "job-extract-mtx-r16q1-s2-raunav.yaml"


def _spec(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    d = yaml.safe_load(path.read_text())
    c = d["spec"]["template"]["spec"]["containers"][0]
    code = "\n".join(ln for ln in c["args"][0].splitlines() if not ln.lstrip().startswith("#"))
    return d, c, code


ALL = ["job-ft-subsets-jc2-raunav.yaml", "job-ft-subsets-jc1-raunav.yaml",
       "job-ft-smoke-raunav.yaml", "job-ft-legs-raunav.yaml"]


@pytest.mark.parametrize("name", ALL)
def test_every_ft_job_is_mine_pinned_and_in_region(name):
    d, c, code = _spec(FT / name)
    assert "raunav" in d["metadata"]["name"]
    assert 'git clone --depth 1 --branch "${REPO_REF}"' in code
    pin = [e for e in c["env"] if e["name"] == "REPO_REF"][0]["value"]
    assert re.fullmatch(r"mtx-s\d+\.\d+", pin), pin
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert any(t["key"] == "topology.kubernetes.io/region" and t["values"] == ["us-west"] for t in terms)
    assert "set -euo pipefail" in code


def test_the_legs_job_is_sized_off_the_measured_band_not_below_it():
    """CLAUDE.md section 8: size memory off the measured 35-58 GB band. 64Gi is
    6 GB above its top and OOM-killed a training job; at N=1e6 the subset is one
    file, so weaver's single train worker double-buffers 1e6 jets."""
    _, c, _ = _spec(FT / "job-ft-legs-raunav.yaml")
    assert c["resources"]["requests"]["memory"] == "88Gi"
    assert c["resources"]["limits"]["memory"] == "88Gi"


def test_only_the_legs_job_asks_for_a_gpu_and_it_is_pinned():
    for name in ALL:
        d, c, code = _spec(FT / name)
        req = c["resources"]["requests"]
        if name == "job-ft-legs-raunav.yaml":
            assert req.get("nvidia.com/gpu") == "1"
            terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
            assert any(t["key"] == "nvidia.com/gpu.product" and t["values"] == ["NVIDIA-GeForce-RTX-3090"]
                       for t in terms), "I7b: one GPU model for the whole sweep"
            assert d["spec"]["backoffLimit"] == 50
        else:
            assert "nvidia.com/gpu" not in req, f"{name} must be CPU-only"
            assert d["spec"]["backoffLimit"] <= 1


def test_all_wave_jobs_share_one_pin():
    pins = set()
    for p in list(FT.glob("job-ft-*-raunav.yaml")) + [
            MTX / "job-mtx-l162_mass-s1-raunav.yaml", MTX / "job-mtx-r16_q1_mass-s1-raunav.yaml",
            MTX / "job-mtx-r42_q1-s1-raunav.yaml", MTX / "job-mtx-makeweight-mass-raunav.yaml"]:
        assert p.exists(), f"{p.name} is not generated: the wave is incomplete"
        d, c, code = _spec(p)                       # executed lines only, not comments
        m = re.search(r'--branch (?:"\$\{REPO_REF\}"|(\S+))', code)
        assert m, f"{p.name}: no git clone --branch in the executed code"
        pin = m.group(1) if m.group(1) else [e for e in c["env"] if e["name"] == "REPO_REF"][0]["value"]
        pins.add(pin)
    assert len(pins) == 1, f"the wave clones different tags: {pins}"


def test_leg1_readout_uses_the_papers_exact_test_jets():
    """Paired contrast: the fine-tuned models' features are on the SAME 2M jets,
    in the SAME order, as features_v2 (probe-bvc-v2)."""
    _, _, code = _spec(FT / "job-ft-legs-raunav.yaml")
    want = re.search(r"--data-test ((?:/jc2/jet_data/\S+\.parquet ?)+)", EXTRACT.read_text()).group(1).split()
    got = re.search(r'TEST2M="([^"]+)"', code).group(1).split()
    assert got == want and len(got) == 335
    assert "--max-jets 2000000" in code and "--save-logits" in code
    assert "smoke_checks.py features --dir ${OUT}/features_v2 --n 2000000 --k 162" in code


def test_legs_design_matches_item_14():
    from build_ft_jobs import EPOCHS, FT_SEEDS, INITS, LR_PRETRAINED, LR_SCRATCH, SIZES
    _, _, code = _spec(FT / "job-ft-legs-raunav.yaml")
    for name, ckpt, k in INITS:
        assert f"{name}:{ckpt}:{k}" in code
    assert {n for n, _, _ in INITS} == {"r16q1-s2", "r16q1-s3", "r16q1-s4", "l162-s1b", "sophon-public", "scratch"}
    assert SIZES == [10_000, 100_000, 1_000_000] and FT_SEEDS == [1, 2, 3]
    assert EPOCHS == {10_000: 50, 100_000: 30, 1_000_000: 10}
    # The WHOLE branch, once per leg: pretrained gets the load, the head
    # exclusion and LR_PRETRAINED; scratch gets no load and LR_SCRATCH. Asserting
    # only that both rates appear somewhere lets a swap pass.
    load_line = ('if [ -n "${ckpt}" ]; then LOAD="--load-model-weights ${ckpt} '
                 '--exclude-model-weights mod\\.fc\\..*"; LR=%s; else LOAD=""; LR=%s; fi'
                 % (LR_PRETRAINED, LR_SCRATCH))
    assert code.count(load_line) == 2, "one identical load/rate branch per leg"
    assert code.count("--exclude-model-weights mod\\.fc\\..*") == 2
    # both legs, both configs, no reweighting
    assert "configs/finetune/JetClassII_L162_noweight.yaml" in code
    assert "configs/finetune/JetClassI_sophon_noweight.yaml" in code
    assert "--no-remake-weights" not in code
    # resumable, never overwriting a partial fine-tune
    assert "[ -f ${OUT}/DONE ]" in code and ".partial." in code
    # the load is verified from weaver's own log after every pretrained fine-tune
    assert "smoke_checks.py load-log" in code
    # leg 2 scores the E1 arm S reference on the same test subset
    assert "/data/results/e1/arm_s_s${S}/net_best_epoch_state.pt" in code
    assert "--predict-gpus 0" in code
    # the released checkpoint is gitignored: fetched in-pod as E0b did, pinned
    assert 'curl -fsSL -o "${SOPHON}" https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt' in code
    assert 'sha256sum "${SOPHON}"' in code and "--checkpoint models/" not in code
    assert "sophon-public:/workspace/sophon_public.pt:188" in code
    # waits on the subset builders, bounded, with a marker so retries fail fast
    assert "wait_for ${SUB2}/DONE" in code and "wait_for ${SUB1}/DONE" in code
    assert "-ge 7200" in code and "86400" not in code and "WAIT_MARK" in code
    # a fine-tune that fails deterministically must not be retried 50 times
    assert "FAIL_MARK" in code and len(re.findall(r"^\s*attempt_ok \$\{OUT\}$", code, re.M)) == 2
    # weaver's save_root swallows write errors and exits 0, so DONE needs the file
    assert code.count('[ -f ${OUT}/pred.root ] || { echo "FATAL: no pred.root in ${OUT}"; exit 1; }') == 2
    assert code.count("tee ${OUT}/predict.log") == 2
    # leg-2 preconditions are checked before leg 1 runs for days, not after
    assert code.index("no E1 arm S seed") < code.index("leg-2 test subset")
    # the manifest records the epoch budget in optimizer steps, not just N
    assert code.count("batch_size=512 steps_per_epoch=$((N/512))") == 2
    # storage: the 100G guard at start and the 85% guard before every fine-tune
    assert "need 100G" in code and len(re.findall(r"^\s*space_ok$", code, re.M)) == 2


def test_smoke_exercises_every_path_on_cpu():
    _, _, code = _spec(FT / "job-ft-smoke-raunav.yaml")
    assert code.count('--gpus ""') >= 4, "CPU: weaver defaults to GPU 0 on every call"
    assert "--use-amp" not in code
    for must in ("--mass-lambda 5.0", "configs/arms/R16_Q1_MASS.yaml", "ParT_sophon_arch_mass.py",
                 "head-width --checkpoint ${S}/hybrid/net_best_epoch_state.pt --expect 18",
                 "make_subsets.py jc2", "make_subsets.py jc1",
                 "--exclude-model-weights 'mod\\.fc\\..*'", "smoke_checks.py load-log",
                 "--expect 162", "features --dir ${S}/leg1/features_v2 --n 3000 --k 162",
                 "weaver --predict", "--num-classes 188 --arm SOPHON_PUBLIC", "SMOKE PASS",
                 'curl -fsSL -o "${SOPHON}"', "--checkpoint ${SOPHON}",
                 "BEST1=${S}/leg1/net_best_epoch_state.pt", "--checkpoint ${BEST1} --expect 162",
                 "--model-prefix ${BEST2} --predict-output"):
        assert must in code, must


def test_makeweight_mass_asserts_i2_across_the_mass_axis():
    d, c, code = _spec(MTX / "job-mtx-makeweight-mass-raunav.yaml")
    assert "nvidia.com/gpu" not in c["resources"]["requests"]
    assert "run_arm L162_MASS    162" in code and "run_arm R16_Q1_MASS  17" in code
    assert 'ref["L162"]' in code and "hist_hashes_mass.json" in code
    assert "-eq 1675" in code, "train + val only; the test split stays out of the histograms"
    # the twins gate on the sidecar's final name: it appears only after the assertions pass
    assert 'cp -v "${SIDECAR}" "${OUT}/$(basename ${SIDECAR}).pending"' in code
    assert code.index("hist_hashes_mass.json") < code.index('mv -v "${p}" "${p%.pending}"')
    assert "set -euo pipefail" in code


def test_jc1_subsets_file_count_survives_pipefail():
    """`ls glob | wc -l` under pipefail aborts on an unmatched glob before the
    FATAL line it guards; find returns 0 matches instead."""
    _, _, code = _spec(FT / "job-ft-subsets-jc1-raunav.yaml")
    assert 'find /data/JetClass/Pythia/train_100M -maxdepth 1 -name "${C}_*.root"' in code
    assert "ls /data/JetClass/Pythia/train_100M" not in code
    assert "JetClass-I files missing on the PVC" in code


def test_mass_specs_differ_from_the_template_only_at_the_declared_sites():
    tmpl = (MTX / "job-mtx-r16_q1-s1-raunav.yaml")
    mass = MTX / "job-mtx-r16_q1_mass-s1-raunav.yaml"
    if not mass.exists():
        pytest.skip("mass spec not generated")
    _, _, a = _spec(tmpl)
    _, _, b = _spec(mass)
    import difflib
    changed = [ln for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
               if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    allowed = ("RUN_ID=", "CFG=", "SIDECAR=", "SRC=", "--arm ", "--num-classes", "--data-config",
               "--lambda-mass", "--out ${OUT}/run_manifest.json", "RECIPE=", "--lean-val-metrics",
               "--network-config", "-o num_classes", "--tensorboard", "until [", "WAITED", "sleep 60",
               "done", "[ \"${WAITED}\"", "[ -f \"${SRC}\" ]", "WAIT_MARK", 'mkdir -p "${OUT}"')
    stray = [ln for ln in changed if not any(tok in ln for tok in allowed)]
    assert not stray, "the mass spec changed lines outside the declared edit sites:\n" + "\n".join(stray)
    assert "--mass-lambda 5.0" in b and "ParT_sophon_arch_mass.py" in b and "-o num_classes 17" in b
