"""Frozen-feature extraction for the random-label control and the ten models
pretrained with a jet-mass regression output.

Three things here fail SILENTLY if they are wrong, so each is constructed
rather than asserted on the happy path:

  the head layout   a mass-output head is K + 1 wide. Declared as K + 1 classes
                    it loads cleanly and the manifest then counts the mass
                    output as a class; every softmax downstream is wrong.
  the pin           a spec pinned at a tag whose extract_features.py predates
                    --num-reg clones, pip-installs and dies on argparse, fifty
                    times. File PRESENCE at the tag does not catch it -- that
                    check passed mtx-s1.48 on 2026-09-18.
  row alignment     genjet_sdmass is read once, model-free, for caches written
                    without it. If those rows are not the caches' rows, every
                    mass residual is computed against another jet's truth.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
K8S = REPO / "experiments" / "EVAL" / "k8s"
MTX_K8S = REPO / "experiments" / "MTX" / "k8s"
BASE_CFG = REPO / "configs" / "data" / "JetClassII_base.yaml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def b():
    return _load("build_extract_jobs", "scripts/build_extract_jobs.py")


def _command(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


# ---------------------------------------------------------------------------
# The builder and the specs on disk
# ---------------------------------------------------------------------------

def test_run_ids_k_and_arm_match_the_training_specs(b):
    """Read from the specs that TRAINED the models, not from their file names:
    the files say `l162_mass`, the run directories say `l162mass`."""
    trained = {}
    for spec in MTX_K8S.glob("job-mtx-*-raunav.yaml"):
        text = _command(spec.read_text())
        rid = re.search(r"RUN_ID=(\S+)", text)
        k = re.search(r"-o num_classes (\d+)", text)
        arm = re.search(r"--arm (\S+)", text)
        if rid and k and arm:
            trained[rid.group(1)] = (arm.group(1), int(k.group(1)),
                                     "ParT_sophon_arch_mass.py" in text)
    for run_id, arm, k, ckpt in b.CONTROL_AND_MASS_RUNS + b.NOT_YET_TRAINED:
        assert run_id in trained, f"no training spec writes {run_id}"
        t_arm, t_k, t_mass = trained[run_id]
        assert (arm, k) == (t_arm, t_k), run_id
        assert ckpt == f"/data/results/mtx/{run_id}"
        assert (run_id in b.NUM_REG) == t_mass, (
            f"{run_id}: --num-reg must follow the architecture it trained with")
    # three random draws and ten mass-output models, however many have trained yet
    assert len(b.CONTROL_AND_MASS_RUNS) + len(b.NOT_YET_TRAINED) == 13 and len(b.NUM_REG) == 10


def test_the_committed_specs_are_what_the_generator_emits(b):
    for run_id, arm, k, ckpt in b.CONTROL_AND_MASS_RUNS:
        fname, text = b.build(run_id, arm, k, ckpt, gpu=False, max_jets=2_000_000)
        assert (K8S / fname).read_text() == text, f"{fname} is stale; regenerate"
    fname, text = b.build_observers_job(2_000_000)
    assert (K8S / fname).read_text() == text, f"{fname} is stale; regenerate"


def test_new_specs_differ_from_a_launched_one_only_where_they_must(b):
    """Same file list, order, cap, data config and space guard as the twenty
    caches they will be compared with -- so the rows align."""
    ref = _command((K8S / "job-extract-mtx-l188-s1-raunav.yaml").read_text())
    ref_list = re.search(r"--data-test (.*?) \\\n", ref).group(1)
    assert len(ref_list.split()) == 335
    for run_id, *_ in b.CONTROL_AND_MASS_RUNS:
        text = _command((K8S / f"job-extract-{run_id}-raunav.yaml").read_text())
        assert re.search(r"--data-test (.*?) \\\n", text).group(1) == ref_list
        assert "--max-jets 2000000" in text
        assert "--data-config configs/data/JetClassII_base.yaml" in text
        assert '[ "${USED}" -lt 85 ]' in text, "the 85% guard must not be weakened"
        assert f'--branch "{b.CONTROL_AND_MASS_PIN}"' in text
        assert f"OUT=/data/results/eval/{run_id}/features_e79" in text
        assert ("--num-reg 1 --save-logits" in text) == (run_id in b.NUM_REG)
        assert "--observers" not in text, "default observers, as the twenty have"


def test_no_launched_spec_was_repinned(b):
    """PIN is per list. The twenty launched specs keep the tag they ran at."""
    assert b.CONTROL_AND_MASS_PIN != b.PIN
    for run_id, *_ in b.RUNS:
        assert b.pin_for(run_id) == b.PIN
        text = (K8S / f"job-extract-{run_id}-raunav.yaml").read_text()
        assert b.CONTROL_AND_MASS_PIN not in text


def test_untrained_models_are_buildable_but_not_emitted(b):
    for run_id, arm, k, ckpt in b.NOT_YET_TRAINED:
        fname, text = b.build(run_id, arm, k, ckpt, gpu=False, max_jets=2_000_000)
        assert f"--arm {arm} " in text and "--num-reg" not in text
        assert not (K8S / fname).exists(), (
            f"{fname} is on disk but {run_id} has not been trained")
        assert run_id not in {r[0] for r in b.RUNS + b.CONTROL_AND_MASS_RUNS}


def test_exclusions_are_recorded_with_their_reasons(b):
    ex = b.DELIBERATELY_EXCLUDED
    assert "class-attention" in ex["mtx-mpm-s1"] and "fine-tuning" in ex["mtx-mpm-s1"]
    assert "mtx-l162-s1" in ex and "mtx-rand-d1-s1" in ex
    assert "mtx-rand-d1-s1b" not in ex
    listed = {r[0] for r in b.RUNS + b.CONTROL_AND_MASS_RUNS + b.NOT_YET_TRAINED}
    assert not listed & set(ex)


def test_a_tag_that_predates_num_reg_is_refused(b):
    """mtx-s1.48 CONTAINS extract_features.py, so the presence check passes it;
    its copy has no --num-reg. This is the pin these specs were nearly given."""
    needs = {"experiments/EVAL/extract_features.py": '"--num-reg"'}
    with pytest.raises(SystemExit, match="mtx-s1.48"):
        b.verify_pin("mtx-s1.48", list(needs), False, needs)
    assert b.CONTROL_AND_MASS_PIN != "mtx-s1.48"


def test_the_observers_job_is_checked_against_exactly_the_twenty_caches(b):
    text = _command((K8S / "job-extract-observers-test2m-raunav.yaml").read_text())
    caches = re.search(r"--align-with (.*?) \\\n", text).group(1).split()
    want = [f"/data/results/eval/{r[0]}/features_e79" for r in b.RUNS
            if r[0] not in b.BEST_EPOCH_RUNS]
    assert caches == want and len(caches) == 20
    assert re.search(r"--data-test (.*?) \\\n", text).group(1) == b.interleaved_files()
    assert "--max-jets 2000000" in text
    assert "configs/data/JetClassII_massreg.yaml" in text


def test_the_observers_config_differs_from_base_by_one_observer():
    """Same selection, inputs and labels => same rows. `weights:` untouched."""
    base = BASE_CFG.read_text().splitlines()
    mreg = [l for l in (REPO / "configs/data/JetClassII_massreg.yaml")
            .read_text().splitlines() if not l.startswith("#")]
    assert [l for l in mreg if l not in base] == ["   - genjet_sdmass"]
    assert [l for l in base if l not in mreg] == []


# ---------------------------------------------------------------------------
# Loading a mass-output checkpoint, on CPU
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("weaver")


@pytest.fixture(scope="module")
def ctx():
    from weaver.utils.data.config import DataConfig
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    return ex, DataConfig.load(str(BASE_CFG), load_observers=True)


@pytest.fixture(scope="module")
def mass17(ctx, tmp_path_factory):
    """A randomly initialised model built by the MASS architecture file."""
    _, dc = ctx
    arch = _load("arch_mass", "experiments/MTX/ParT_sophon_arch_mass.py")
    model, _ = arch.get_model(dc, num_classes=17, fc_params=[(512, 0.1)],
                              allow_without_hybrid=True)
    p = tmp_path_factory.mktemp("mass") / "mass17.pt"
    torch.save(model.state_dict(), p)
    return model.eval(), p


def test_a_mass_output_checkpoint_loads_and_reproduces_its_own_output(ctx, mass17):
    ex, dc = ctx
    trained, ckpt = mass17
    model = ex.build_model(dc, 17 + 1)
    assert list(model.state_dict()) == list(trained.state_dict()), (
        "the mass architecture's state_dict keys are not the plain one's")
    prov = ex.load_trunk_or_die(model, ckpt, 17, num_reg=1)
    assert prov["checkpoint_num_classes"] == 17
    assert prov["head_missing"] == 0 and prov["head_unexpected"] == 0
    assert prov["trunk_tensors_loaded"] > 200

    model.eval()
    tap = ex.ClsTap(model)
    args = ex.synthetic_batch(dc, 4, torch.device("cpu"))
    with torch.no_grad():
        out, want = model(*args), trained(*args)
    assert out.shape == (4, 18)
    assert torch.allclose(out, want), "loaded weights are not the checkpoint's"
    assert tap.buf.shape == (4, ex.EMBED_DIM)
    with torch.no_grad():
        assert torch.allclose(tap.fc(tap.buf), out, atol=1e-4, rtol=1e-3)
    tap.close()


def test_a_mass_checkpoint_declared_as_its_plain_twin_is_refused(ctx, mass17):
    ex, dc = ctx
    with pytest.raises(SystemExit) as e:
        ex.load_trunk_or_die(ex.build_model(dc, 17), mass17[1], 17)
    assert e.value.code == 3


def test_a_plain_checkpoint_declared_with_a_mass_output_is_refused(ctx, tmp_path):
    ex, dc = ctx
    p = tmp_path / "plain17.pt"
    torch.save(ex.build_model(dc, 17).state_dict(), p)
    with pytest.raises(SystemExit) as e:
        ex.load_trunk_or_die(ex.build_model(dc, 18), p, 17, num_reg=1)
    assert e.value.code == 3


# ---------------------------------------------------------------------------
# End to end through weaver's loader, on a synthetic JetClass-II file
# ---------------------------------------------------------------------------

def _jc2_file(path, n, seed):
    ak = pytest.importorskip("awkward")
    rng = np.random.default_rng(seed)
    npart = rng.integers(5, 40, n)

    def part(lo=-1.0, hi=1.0):
        flat = rng.uniform(lo, hi, int(npart.sum())).astype(np.float32)
        return ak.unflatten(flat, npart)

    def flag():
        return ak.values_astype(part(0, 1) > 0.5, np.float32)

    px, py, pz = part(-50, 50), part(-50, 50), part(-50, 50)
    pt = rng.uniform(100, 1200, n).astype(np.float32)      # some fail jet_pt > 200
    sd = rng.uniform(5, 400, n).astype(np.float32)         # some fail jet_sdmass > 20
    gen = np.where(rng.random(n) < 0.2, 0.0,               # unmatched: hard 0.0
                   sd * rng.uniform(0.8, 1.2, n)).astype(np.float32)
    ak.to_parquet(ak.Array({
        "part_px": px, "part_py": py, "part_pz": pz,
        "part_energy": ak.values_astype(np.sqrt(px**2 + py**2 + pz**2) + 1.0, np.float32),
        "part_deta": part(), "part_dphi": part(), "part_d0val": part(),
        "part_dzval": part(), "part_d0err": part(0, 1), "part_dzerr": part(0, 1),
        "part_charge": part(), "part_isChargedHadron": flag(),
        "part_isNeutralHadron": flag(), "part_isPhoton": flag(),
        "part_isElectron": flag(), "part_isMuon": flag(),
        "jet_pt": pt, "jet_eta": rng.uniform(-2, 2, n).astype(np.float32),
        "jet_phi": rng.uniform(-3, 3, n).astype(np.float32),
        "jet_energy": (pt * 1.5).astype(np.float32),
        "jet_nparticles": npart.astype(np.float32),
        "jet_sdmass": sd, "genjet_sdmass": gen,
        **{f"jet_tau{i}": rng.random(n).astype(np.float32) for i in (1, 2, 3, 4)},
        "jet_label": rng.integers(0, 188, n).astype(np.int32)}), path)


def _main(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    return mod.main()


def test_mass_extraction_and_observer_alignment_end_to_end(ctx, mass17, tmp_path,
                                                           monkeypatch):
    ex, _ = ctx
    ob = _load("extract_observers", "experiments/EVAL/extract_observers.py")
    files = []
    for i in range(3):
        files.append(str(tmp_path / f"f{i}.parquet"))
        _jc2_file(files[-1], 300, seed=i)
    small = ["--batch-size", "64", "--num-workers", "0", "--max-jets", "500"]

    cache = tmp_path / "cache"
    assert _main(ex, ["--checkpoint", str(mass17[1]), "--num-classes", "17",
                      "--num-reg", "1", "--save-logits", "--arm", "R16_Q1_MASS",
                      "--data-test", *files, "--out", str(cache), *small],
                 monkeypatch) == 0
    man = json.loads((cache / "extract_manifest.json").read_text())
    assert (man["num_classes"], man["num_reg"], man["n_jets"]) == (17, 1, 500)
    assert man["logit_columns"]["class_logits"] == [0, 17]
    assert man["logit_columns"]["regression"] == [17, 18]
    assert np.load(cache / "logits.npy").shape == (500, 18)
    assert "genjet_sdmass" not in man["observers"], (
        "the base config carries no genjet_sdmass -- the reason the job exists")

    # the model-free pass reads the SAME rows through the massreg config
    out = tmp_path / "obs"
    assert _main(ob, ["--data-test", *files, "--align-with", str(cache),
                      "--out", str(out), *small], monkeypatch) == 0
    oman = json.loads((out / "observers_manifest.json").read_text())
    assert oman["label188_sha256"] == man["label188_sha256"]
    got, had = np.load(out / "observers.npz"), np.load(cache / "observers.npz")
    for k in had.files:
        assert np.array_equal(got[k], had[k]), k
    gen = got["genjet_sdmass"]
    assert gen.shape == (500,) and 0 < int((gen == 0).sum()) < 500
    assert oman["n_genjet_sdmass_matched"] == int((gen > 0).sum())

    # a different file order is different rows: refused, and nothing written
    bad = tmp_path / "obs_bad"
    assert _main(ob, ["--data-test", *files[::-1], "--align-with", str(cache),
                      "--out", str(bad), *small], monkeypatch) == 2
    assert not bad.exists()

    # and a finished observer file is never overwritten
    with pytest.raises(SystemExit, match="already holds"):
        _main(ob, ["--data-test", *files, "--align-with", str(cache),
                   "--out", str(out), *small], monkeypatch)


def test_the_random_control_draws_refuse_to_be_built_for_a_gpu(b, monkeypatch, capsys):
    """Draw 1 extracted on CPU, and C4 compares the three draws with each other
    and with the 17-class models, which also extracted on CPU. A later draw on a
    GPU would differ in kernels and in mixed precision, putting a second
    uncontrolled variable inside the one control that answers the tautology
    objection. Draws 2 and 3 were still pretraining when this was written, so
    the refusal has to live in the builder, not in someone's memory."""
    import pytest
    monkeypatch.setattr(sys, "argv",
                        ["build_extract_jobs.py", "--gpu", "--only", "mtx-rand-d2-s2"])
    with pytest.raises(SystemExit) as e:
        b.main()
    msg = str(e.value)
    assert "mtx-rand-d2-s2" in msg and "without --gpu" in msg


def test_a_non_control_run_may_still_be_built_for_a_gpu(b):
    """The refusal must be specific to the control. Everything else keeps the
    option, or the guard would just be a broken flag.

    Checks the emitted TEXT rather than calling main(), because main() writes
    into experiments/EVAL/k8s/ and a test that litters the repository with a
    spec nobody asked for is how an unreviewed job spec ends up being launched."""
    fname, text = b.build("mtx-l162mass-s1", "L162_MASS", 162,
                          "/data/results/mtx/mtx-l162mass-s1", True, 2_000_000, 79)
    assert "nvidia.com/gpu" in text and "-gpu-raunav.yaml" in fname
