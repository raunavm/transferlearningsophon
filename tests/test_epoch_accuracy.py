"""S8's checkpoint scorer: every checkpoint on the same jets, class outputs only."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs" / "data" / "JetClassII_base.yaml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EA = _load("epoch_accuracy", "experiments/EVAL/epoch_accuracy.py")


def test_accuracy_ignores_the_regression_column():
    logits = np.array([[0.1, 0.9, 50.0], [0.8, 0.2, 50.0]])   # column 2 is a mass output
    assert EA.accuracy(logits, np.array([1, 0]), 2) == 1.0


def test_the_vocabulary_map_is_the_committed_one():
    m = EA.vocabulary_map("R16_Q1")
    assert m.shape == (188,) and m.max() == 16 and m[0] == 0 and m[14] == 1
    assert (m[161:] == 16).all()                              # all 27 QCD classes
    assert (EA.vocabulary_map("L188") == np.arange(188)).all()


def test_scores_equal_those_from_the_extraction_logits_of_the_same_jets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("weaver")
    mc = _load("mass_control_tests", "tests/test_extract_mass_control.py")
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    from weaver.utils.data.config import DataConfig
    dc = DataConfig.load(str(BASE_CFG), load_observers=True)
    arch = _load("arch_mass", "experiments/MTX/ParT_sophon_arch_mass.py")

    run = tmp_path / "run"
    run.mkdir()
    for e, seed in ((0, 1), (4, 2)):                          # two different random models
        torch.manual_seed(seed)
        model, _ = arch.get_model(dc, num_classes=17, fc_params=[(512, 0.1)],
                                  allow_without_hybrid=True)
        torch.save(model.state_dict(), run / f"net_epoch-{e}_state.pt")
    files = []
    for i in range(3):
        files.append(str(tmp_path / f"f{i}.parquet"))
        mc._jc2_file(files[-1], 300, seed=i)

    out = tmp_path / "acc.json"
    plain = tmp_path / "plain"
    plain.mkdir()
    torch.manual_seed(3)
    torch.save(ex.build_model(dc, 17).state_dict(), plain / "net_epoch-0_state.pt")
    torch.save(ex.build_model(dc, 17).state_dict(), plain / "net_epoch-4_state.pt")
    assert EA.main(["--runs", f"{run}:R16_Q1:17:1", f"{plain}:R16_Q1:17:0",
                    "--epochs", "0", "4", "--data-config", str(BASE_CFG),
                    "--data-test", *files, "--max-jets", "500", "--stride", "3",
                    "--batch-size", "64", "--num-workers", "0", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["n_jets"] == 167                               # rows 0, 3, ..., 498
    got = doc["runs"]["run"]
    assert set(doc["runs"]) == {"run", "plain"}

    vmap = EA.vocabulary_map("R16_Q1")
    for e in (0, 4):
        cache = tmp_path / f"cache{e}"
        monkeypatch.setattr(sys, "argv", ["x", "--checkpoint", str(run / f"net_epoch-{e}_state.pt"),
                                          "--num-classes", "17", "--num-reg", "1", "--save-logits",
                                          "--arm", "R16_Q1_MASS", "--data-test", *files,
                                          "--out", str(cache), "--batch-size", "64",
                                          "--num-workers", "0", "--max-jets", "500"])
        assert ex.main() == 0
        man = json.loads((cache / "extract_manifest.json").read_text())
        lab = np.load(cache / "label188.npy")[::3]
        assert man["n_jets"] == 500
        want = float((np.load(cache / "logits.npy")[::3, :17].argmax(1) == vmap[lab]).mean())
        assert got["epochs"][str(e)]["accuracy"] == pytest.approx(want, abs=1e-12)
        pl = tmp_path / f"plain_cache{e}"
        monkeypatch.setattr(sys, "argv", ["x", "--checkpoint", str(plain / f"net_epoch-{e}_state.pt"),
                                          "--num-classes", "17", "--save-logits", "--arm", "R16_Q1",
                                          "--data-test", *files, "--out", str(pl), "--batch-size",
                                          "64", "--num-workers", "0", "--max-jets", "500"])
        assert ex.main() == 0
        want = float((np.load(pl / "logits.npy")[::3].argmax(1) == vmap[lab]).mean())
        assert doc["runs"]["plain"]["epochs"][str(e)]["accuracy"] == pytest.approx(want, abs=1e-12)

    # --align-with: the cache's own jets pass; a cache of other jets is refused
    base = ["--runs", f"{run}:R16_Q1:17:1", "--epochs", "0", "--data-config", str(BASE_CFG), "--data-test", *files,
            "--max-jets", "500", "--stride", "3", "--batch-size", "64", "--num-workers", "0"]
    assert EA.main(base + ["--align-with", str(tmp_path / "cache0"),
                           "--out", str(tmp_path / "ok.json")]) == 0
    other = tmp_path / "other"
    other.mkdir()
    np.save(other / "label188.npy", np.load(tmp_path / "cache0" / "label188.npy")[::-1])
    with pytest.raises(SystemExit, match="not the jets"):
        EA.main(base + ["--align-with", str(other), "--out", str(tmp_path / "bad.json")])

    with pytest.raises(SystemExit, match="exists"):
        EA.main(["--runs", f"{run}:R16_Q1:17:1",
                 "--epochs", "0", "--data-config", str(BASE_CFG), "--data-test", *files,
                 "--max-jets", "500", "--out", str(out)])


def test_the_five_s8_jobs_hold_each_seed_index_s_four_models_on_the_cache_jets():
    yaml = pytest.importorskip("yaml")
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    for seed in range(1, 6):
        fname, text = bx.build_epoch_accuracy(seed)
        d = yaml.safe_load(text)
        assert d["metadata"]["name"] == f"s8-epoch-accuracy-s{seed}-raunav"
        assert fname == f"job-{d['metadata']['name']}.yaml" and d["spec"]["backoffLimit"] == 1
        twin = "mtx-l162-s1b" if seed == 1 else f"mtx-l162-s{seed}"   # never the 1e-3 run
        assert (f"--runs /data/results/mtx/{twin}:L162:162:0 "
                f"/data/results/mtx/mtx-l162mass-s{seed}:L162:162:1 "
                f"/data/results/mtx/mtx-r16q1-s{seed}:R16_Q1:17:0 "
                f"/data/results/mtx/mtx-r16q1mass-s{seed}:R16_Q1:17:1 \\\n") in text
        assert "--epochs 0 4 9 19 39 79 \\\n" in text
        assert "--max-jets 2000000 --stride 20 \\\n" in text
        assert f"--align-with /data/results/eval/mtx-r16q1-s{seed}/features_e79 \\\n" in text
        assert f'--branch "{bx.EPOCH_ACC_PIN}"' in text
        assert bx.interleaved_files() in text                  # the caches' own file list
