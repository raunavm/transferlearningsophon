"""The output-layer diagnostic over epochs 70-79 (experiments/DIAG/head_epoch_diag.py),
on synthetic inputs: metric definitions, the train.log parser, the noise and
correlation tests, and the whole infer -> analyse -> figure chain on CPU."""
import importlib.util
import json
import pathlib
import shutil
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs" / "data" / "JetClassII_base.yaml"
SPEC = REPO / "experiments/DIAG/k8s/job-diag-head-epochs-raunav.yaml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("head_epoch_diag", "experiments/DIAG/head_epoch_diag.py")


def test_vocabularies_split_into_qcd_and_resonant_classes():
    lut, qcd, res = H.vocabulary("R42_Q1", 43)
    assert qcd == [42] and res == list(range(42)) and lut[170] == 42
    _, qcd, _ = H.vocabulary("L188", 188)
    assert qcd == list(range(161, 188))
    with pytest.raises(SystemExit, match="does not split"):
        H.vocabulary("R16_Q1", 18)


def test_per_jet_metrics_match_a_hand_computation():
    z = np.log(np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2]]))  # class 2 is QCD
    correct, p_qcd, lo = H.per_jet(z + 7.0, np.array([0, 2, 1]), qcd=[2], res=[0, 1], chunk=2)
    assert correct.tolist() == [True, True, False]
    assert p_qcd == pytest.approx([0.2, 0.8, 0.2], abs=1e-6)
    assert lo == pytest.approx(np.log([0.8 / 0.2, 0.2 / 0.8, 0.8 / 0.2]), abs=1e-5)
    s = H.head_summary(correct, p_qcd, lo, is_qcd=np.array([False, True, False]))
    assert s["top1_accuracy"] == pytest.approx(2 / 3)
    assert s["mean_p_qcd_resonant"] == pytest.approx(0.2) and s["mean_p_qcd_qcd"] == pytest.approx(0.8)
    assert s["median_log_odds_qcd"] == pytest.approx(np.log(0.25), abs=1e-5)
    assert s["res_vs_qcd_auc"] == 1.0


LOG = """[2026-09-12 22:23:31,379] INFO: Epoch #70 training
[x] INFO: Train class distribution:
    [(0, 90), (1, 5), (16, 5)]
[x] INFO: \x1b[1mEpoch #70: Current validation metric: 0.61 (best: 0.62)\x1b[0m
[x] INFO: Epoch #71 training
[x] INFO: Restarted DataIter train_worker0, load_range=(0.0, 1.0), file_list:
[x] INFO: Epoch #71 training
[x] INFO: Train class distribution:
    [(0, 70), (16, 30)]
[x] INFO: \x1b[1mEpoch #71: Current validation metric: 0.55 (best: 0.62)\x1b[0m
[x] INFO: Epoch #71 training
[x] INFO: Train class distribution:
    [(0, 80), (16, 20)]
"""


def test_the_log_parser_keeps_the_last_complete_print_of_each_epoch():
    dist, val = H.parse_train_log(LOG)
    assert H.qcd_share(dist[70], [16]) == pytest.approx(0.05)
    assert H.qcd_share(dist[71], [16]) == pytest.approx(0.20)      # the resumed print wins
    assert val == {70: 0.61, 71: 0.55}


def test_swing_test_and_pooled_correlation():
    flat = H.swing_test([0.5] * 10, [0.001] * 10)
    assert flat["chi2"] == 0 and flat["p"] == 1.0
    big = H.swing_test([0.5, 0.52] * 5, [0.001] * 10)
    assert big["sd_over_se"] > 10 and big["p"] < 1e-12 and big["dof"] == 9
    rng = np.random.default_rng(1)
    x = [rng.random(10) for _ in range(4)]
    same = H.within_run_correlation([(a, 3 * a + 1) for a in x], n_perm=500)
    assert same["r"] == pytest.approx(1.0) and same["p_perm"] < 0.01 and same["n_points"] == 40
    noise = H.within_run_correlation([(a, rng.random(10)) for a in x], n_perm=500)
    assert abs(noise["r"]) < 0.5 and noise["p_perm"] > 0.01


def test_weight_average_is_the_mean_of_floats_and_the_last_integer_buffer(tmp_path):
    torch = pytest.importorskip("torch")
    for i, v in enumerate((1.0, 3.0)):
        torch.save({"model_state_dict": {"w": torch.full((2,), v), "n": torch.tensor(i)}},
                   tmp_path / f"{i}.pt")
    avg = H.average_states([tmp_path / "0.pt", tmp_path / "1.pt"])
    assert torch.equal(avg["w"], torch.full((2,), 2.0)) and int(avg["n"]) == 1


def test_probe_uses_the_repository_split_and_separates_what_is_separable():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good = H.probe_summary(rng.normal(size=(3000, 8)) + 3 * y[:, None], y)
    bad = H.probe_summary(rng.normal(size=(3000, 8)), y)
    assert good["auc"] > 0.99 and abs(bad["auc"] - 0.5) < 0.06
    assert good["n_test"] == 600 and bad["log1m_auc_boot_se"] > 0


def test_infer_analyse_figure_end_to_end_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("weaver")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    mc = _load("mass_control_tests", "tests/test_extract_mass_control.py")
    from weaver.utils.data.config import DataConfig
    dc = DataConfig.load(str(BASE_CFG), load_observers=True)

    files = []
    for i in range(3):                      # labels drawn so the b-vs-c task has rows
        f = tmp_path / f"f{i}.parquet"
        mc._jc2_file(str(f), 600, seed=i)
        t = pq.read_table(f)
        lab = np.random.default_rng(i).choice([0, 1, 20, 170, 181], t.num_rows).astype(np.int32)
        pq.write_table(t.set_column(t.column_names.index("jet_label"), "jet_label", pa.array(lab)), f)
        files.append(str(f))

    run = tmp_path / "mtx" / "mtx-r16q1-s9"
    run.mkdir(parents=True)
    for e in (50, 70, 71):
        torch.manual_seed(e)
        torch.save(H.XF.build_model(dc, 17).state_dict(), run / f"net_epoch-{e}_state.pt")
    shutil.copy(run / "net_epoch-50_state.pt", run / "net_best_epoch_state.pt")
    (run / "train.log").write_text(
        "".join(f"Epoch #{e} training\nTrain class distribution: \n    [(0, {100 - q}), (16, {q})]\n"
                for e, q in ((70, 8), (71, 17))))

    cache = tmp_path / "eval" / run.name / "features_e79"   # the committed cache of the last epoch
    monkeypatch.setattr(sys, "argv", ["x", "--checkpoint", str(run / "net_epoch-71_state.pt"),
                                      "--num-classes", "17", "--save-logits", "--arm", "R16_Q1",
                                      "--data-test", *files, "--out", str(cache), "--batch-size", "64",
                                      "--num-workers", "0", "--max-jets", "1500"])
    assert H.XF.main() == 0

    out = tmp_path / "diag"
    argv = ["infer", "--runs", f"{run}:R16_Q1:17:0", "--epochs", "70", "71",
            "--data-test", *files, "--max-jets", "1500", "--stride", "3", "--align-with", str(cache),
            "--cache-root", str(tmp_path / "eval"), "--batch-size", "64", "--num-workers", "0",
            "--out", str(out)]
    assert H.main(argv) == 0
    sample = json.loads((out / "sample.json").read_text())
    assert sample["n_jets"] == 500 and sample["files"][0]["stream_range"][0] == 0
    assert sample["files"][-1]["stream_range"][1] == 1500
    meta = json.loads((out / run.name / "DONE").read_text())
    assert meta["best_epoch"] == 50 and set(meta["checkpoints"]) == {"e70", "e71", "best", "wavg"}
    rep = meta["repro_features_e79"]
    assert rep["top1_agreement"] == 1.0 and rep["max_abs_logit_diff"] < 1e-4
    assert rep["max_abs_feature_diff_bvc"] < 1e-4
    assert H.main(argv) == 0                                    # a finished run is skipped

    doc_path = tmp_path / "diag.json"
    assert H.main(["analyse", "--out", str(out), "--json", str(doc_path)]) == 0
    r = json.loads(doc_path.read_text())["runs"][run.name]
    assert set(r["checkpoints"]) == {"e70", "e71", "best", "wavg", "ens"}
    assert [r["checkpoints"][t]["stream_qcd_share"] for t in ("e70", "e71")] == [0.08, 0.17]
    assert "probe" not in r["checkpoints"]["ens"] and "probe" in r["checkpoints"]["wavg"]
    lab = np.load(cache / "label188.npy")[::3]
    want = float((np.load(cache / "logits.npy")[::3].argmax(1) == H.EA.vocabulary_map("R16_Q1")[lab]).mean())
    assert r["checkpoints"]["e71"]["head"]["top1_accuracy"] == pytest.approx(want, abs=1e-12)

    assert H.main(["figure", "--json", str(doc_path), "--outdir", str(tmp_path / "fig"),
                   "--defective", run.name]) == 0
    assert (tmp_path / "fig" / "head_epoch_diag.png").stat().st_size > 0


def test_the_job_reads_the_caches_jets_under_the_retry_policy():
    yaml = pytest.importorskip("yaml")
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    text = SPEC.read_text()
    d = yaml.safe_load(text)
    assert "raunav" in d["metadata"]["name"] and d["spec"]["backoffLimit"] == 20
    rules = d["spec"]["podFailurePolicy"]["rules"]
    assert rules[0] == {"action": "FailJob",
                        "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}}
    assert rules[1] == {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}
    assert d["spec"]["template"]["spec"]["containers"][0]["name"] == "main"
    assert bx.interleaved_files() in text
    assert "--max-jets 2000000 --stride 4" in text
    for run in ("l188-s5", "l162-s5", "r42q1-s5", "r16q1mass-s4",
                "l188-s1", "l162-s1b", "r42q1-s1", "r16q1mass-s1"):
        assert f"/data/results/mtx/mtx-{run}:" in text
