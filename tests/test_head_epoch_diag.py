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


def test_paired_swing_cancels_the_test_noise_the_epochs_share():
    """Ten epochs scored on the same jets: which jets a head gets right is mostly
    common to all epochs (u), plus 0.5% of jets each epoch gets wrong on its own.
    The paired bootstrap must see through the common part, which the per-epoch
    binomial error (the old swing_test) counts as if the epochs were independent."""
    from scipy.stats import chi2
    rng = np.random.default_rng(3)
    n, k = 20_000, 10
    u = rng.random(n)
    own = rng.random((n, k)) < 0.005

    def heads(shift):
        return ((u[:, None] < 0.5 + np.asarray(shift)[None, :]) ^ own).astype(np.float32)

    same, moved = heads(np.zeros(k)), heads(np.tile([0.003, -0.003], 5))
    null = H.paired_swing(lambda i: same[i].mean(0), n, n_boot=400)
    alt = H.paired_swing(lambda i: moved[i].mean(0), n, n_boot=400)
    binomial_se = np.sqrt(0.25 / n)
    assert null["dof"] == 9 and null["paired_se"] < binomial_se / 3
    assert null["p"] > 1e-3                                  # no epoch effect: chi2 ~ chi2(9)
    assert alt["p"] < 1e-9 and alt["sd_over_paired_se"] > 5  # a 0.3% swing is seen ...
    v = moved.mean(0)                                        # ... which the unpaired chi2 misses
    assert (((v - v.mean()) / binomial_se) ** 2).sum() < chi2.isf(0.01, 9)

    # the probe's case: every epoch's scores share a large per-jet component
    m = 3000
    y = rng.integers(0, 2, m)
    s_null = 2 * y[:, None] + 3 * rng.normal(size=(m, 1)) + 0.3 * rng.normal(size=(m, k))
    s_alt = s_null + 0.1 * np.tile([1, -1], 5)[None, :] * y[:, None]

    def log1m(s):
        return lambda i: [H.P.log1m_auc(y[i], s[i, j])[0] for j in range(k)]

    pn, pa = H.paired_swing(log1m(s_null), m, n_boot=300), H.paired_swing(log1m(s_alt), m, n_boot=300)
    rb = np.random.default_rng(0)                 # one epoch's own bootstrap SE, as probe_summary's
    marginal = np.std([H.P.log1m_auc(y[i], s_null[i, 0])[0]
                       for i in (rb.integers(0, m, m) for _ in range(200))], ddof=1)
    assert pn["p"] > 1e-3 and pa["p"] < 1e-6 and pn["paired_se"] < marginal / 3
    w = np.array(log1m(s_alt)(np.arange(m)))
    assert (((w - w.mean()) / marginal) ** 2).sum() < chi2.isf(0.01, 9)


def test_pooled_within_run_correlation():
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
    good, s = H.probe_summary(rng.normal(size=(3000, 8)) + 3 * y[:, None], y)
    bad, _ = H.probe_summary(rng.normal(size=(3000, 8)), y)
    assert good["auc"] > 0.99 and abs(bad["auc"] - 0.5) < 0.06
    assert good["n_test"] == 600 == s.size and bad["log1m_auc_boot_se"] > 0
    assert H.P.log1m_auc(y[H.P.make_splits(3000)[2]], s)[0] == good["log1m_auc"]   # scores of the test split


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
    doc = json.loads(doc_path.read_text())
    sw = doc["runs"][run.name]["epochs_70_79"]
    assert all(sw[m]["dof"] == 1 and sw[m]["n_boot"] == H.PAIRED_BOOT
               for m in ("head_top1", "head_p_qcd_resonant", "probe_log1m_auc"))
    assert set(doc["defect"]["runs"][run.name]["flags"]) == {"e70", "e71", "best", "wavg", "ens"}
    assert doc["defect"]["primary_rule"] in doc["defect"]["counts"]
    assert doc["class_mix"]["share_vs_top1"] == doc["pooled_within_run"]["share_vs_top1"]
    r = doc["runs"][run.name]
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


def _run(acc, lo=None, best=None, wavg=0.5, best_is=None):
    """A run record as analyse writes it: epochs 70-79 with the given head values."""
    lo = lo or [0.0] * 10
    cks = {f"e{70 + i}": {"head": {"top1_accuracy": a, "median_log_odds_qcd": o}} for i, (a, o) in enumerate(zip(acc, lo))}
    cks["wavg"] = {"head": {"top1_accuracy": wavg, "median_log_odds_qcd": 0.0}}
    if best is not None:
        cks["best"] = {"head": {"top1_accuracy": best, "median_log_odds_qcd": 0.0}}
    r = {"checkpoints": cks, "epochs_70_79": {"pearson_share_vs_p_qcd_resonant": 0.1, "pearson_share_vs_top1": -0.2}}
    if best_is:
        r["best_is"] = best_is
    return r


def test_defect_flags_and_counts_follow_the_stated_rule():
    assert H.PRIMARY_RULE == "top1_10pct_vs_median" and not H.is_defective(0.45, 0.5, "below", 0.10)  # "more than"
    pooled = {"share_vs_p_qcd_resonant": {"r": 0.07, "p_perm": 0.56, "n_points": 30},
              "share_vs_top1": {"r": -0.04, "p_perm": 0.77, "n_points": 30}}
    res = {"pooled_within_run": pooled, "runs": {
        # audit-flagged name; epoch 79 defective in accuracy and in the QCD log-odds; best is epoch 75
        "mtx-l188-s5": _run([0.5] * 9 + [0.4], lo=[0.0] * 9 + [5.0], wavg=0.52, best_is="e75"),
        # no defective epoch; its best-validation checkpoint (epoch 60) is defective
        "mtx-x-s2": _run([0.6] * 10, best=0.5, wavg=0.6),
        # a defective epoch 72 that neither the weight average nor ...
        "mtx-x-s3": _run([0.5, 0.5, 0.3] + [0.5] * 7, best=0.5, wavg=0.4),
        # six of ten epochs low: the median IS the defective level, so only the
        # 75th-percentile and most-accurate-epoch references see them
        "mtx-x-s4": _run([0.3] * 6 + [0.5] * 4, best=0.47, wavg=0.5)}}
    s = H.summarise(res)
    P = H.PRIMARY_RULE
    f = s["defect"]["runs"]
    assert f["mtx-l188-s5"]["reference"][P] == 0.5 and f["mtx-l188-s5"]["best"] == "e75"
    assert [t for t, x in f["mtx-l188-s5"]["flags"].items() if x[P]] == ["e79"]
    assert f["mtx-x-s2"]["flags"]["best"][P] and not f["mtx-x-s2"]["flags"]["wavg"][P]
    ref4 = f["mtx-x-s4"]["reference"]
    assert (ref4[P], ref4["top1_10pct_vs_q75"], ref4["top1_10pct_vs_most_accurate_epoch"]) == (0.3, 0.5, 0.5)
    assert not any(x[P] for x in f["mtx-x-s4"]["flags"].values())
    assert [t for t, x in f["mtx-x-s4"]["flags"].items() if x["top1_10pct_vs_q75"]] == [f"e{e}" for e in range(70, 76)]
    c = s["defect"]["counts"][P]
    assert c["runs_with_a_defective_epoch"] == ["mtx-l188-s5", "mtx-x-s3"]
    assert c["of_those_weight_average_sound"] == ["mtx-l188-s5"]                 # s3's average is 0.4 < 0.45
    assert c["of_those_best_validation_sound"] == ["mtx-l188-s5", "mtx-x-s3"]
    assert c["defective_at_last_epoch"] == ["mtx-l188-s5"]
    assert c["weight_average_defective"] == ["mtx-x-s3"] and c["best_validation_defective"] == ["mtx-x-s2"]
    assert c["audit_flagged"] == {"runs": ["mtx-l188-s5"], "defective_at_last_epoch": ["mtx-l188-s5"],
                                  "weight_average_sound": ["mtx-l188-s5"], "best_validation_sound": ["mtx-l188-s5"]}
    assert c["n"] == {"runs": 4, "runs_with_a_defective_epoch": 2, "weight_average_fixes": 1,
                      "best_validation_fixes": 2, "weight_average_defective": 1, "best_validation_defective": 1,
                      "audit_flagged_weight_average_fixes": 1, "audit_flagged_best_validation_fixes": 1}
    q = s["defect"]["counts"]["top1_10pct_vs_q75"]
    assert q["runs_with_a_defective_epoch"] == ["mtx-l188-s5", "mtx-x-s3", "mtx-x-s4"]
    assert q["of_those_best_validation_sound"] == ["mtx-l188-s5", "mtx-x-s3", "mtx-x-s4"]   # 0.47 >= 0.45
    strict = s["defect"]["counts"]["top1_5pct_vs_most_accurate_epoch"]                      # 0.47 < 0.475
    assert strict["of_those_best_validation_sound"] == ["mtx-l188-s5", "mtx-x-s3"]
    assert s["defect"]["counts"]["qcd_log_odds_2p8_vs_median"]["runs_with_a_defective_epoch"] == ["mtx-l188-s5"]
    assert s["defect"]["counts"]["top1_15pct_vs_median"]["runs_with_a_defective_epoch"] == ["mtx-l188-s5", "mtx-x-s3"]
    assert s["defect"]["rules"]["top1_5pct_vs_q75"] == \
        "top-1 accuracy more than 5% below the 75th percentile of the run's epochs 70-79"
    span = s["defect"]["range_over_rules"]["runs_with_a_defective_epoch"]
    assert span == {"min": 1, "max": 3,
                    "rules_at_min": ["qcd_log_odds_2p8_vs_median", "qcd_log_odds_2p8_vs_most_accurate_epoch"],
                    "rules_at_max": [f"top1_{p}pct_vs_{r}" for p in (10, 5, 15) for r in ("q75", "most_accurate_epoch")]}
    assert s["class_mix"]["share_vs_mean_p_qcd_resonant"] == pooled["share_vs_p_qcd_resonant"]
    assert s["class_mix"]["per_run_r_range_share_vs_top1"] == [-0.2, -0.2]


def test_the_committed_defect_summary_is_what_summarise_writes(tmp_path):
    d = REPO / "experiments/FIGS/data/head_epoch_diag"
    assert H.main(["summarise", "--json", str(d / "head_epoch_diag.json"), "--out", str(tmp_path / "s.json")]) == 0
    assert (tmp_path / "s.json").read_text() == (d / "head_epoch_defects.json").read_text()
    src = tmp_path / "a.json"
    src.write_text("{}")
    with pytest.raises(SystemExit, match="overwrite"):
        H.main(["summarise", "--json", str(src), "--out", str(src)])


def test_the_script_compiles_under_the_images_python_310():
    """The job image runs Python 3.10.12. ast's feature_version=(3, 10) does not
    catch a PEP 701 f-string: the backslash inside a replacement field that
    86d96f5 added parsed under 3.13 with feature_version=(3, 10) and failed in the
    pod. So the script is also compiled by a real 3.10 interpreter when one is
    installed, and the test says it skipped when none is."""
    import ast
    import subprocess
    path = REPO / "experiments/DIAG/head_epoch_diag.py"
    ast.parse(path.read_text(), feature_version=(3, 10))
    py310 = shutil.which("python3.10")
    if py310 is None:
        pytest.skip("no python3.10 here: only the ast check ran")
    r = subprocess.run([py310, "-c", "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')",
                        str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_analyse_rerun_reads_the_arrays_and_writes_somewhere_new():
    import re
    import subprocess
    yaml = pytest.importorskip("yaml")
    new = yaml.safe_load((REPO / "experiments/DIAG/k8s/job-diag-head-epochs-analyse-v2-raunav.yaml").read_text())
    old = yaml.safe_load(SPEC.read_text())
    assert "raunav" in new["metadata"]["name"] and new["metadata"]["name"] != old["metadata"]["name"]
    for k in ("backoffLimit", "podFailurePolicy"):
        assert new["spec"][k] == old["spec"][k]
    pod = new["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["name"] == "main" and "gpu" not in json.dumps(c["resources"])
    terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert {"key": "topology.kubernetes.io/region", "operator": "In", "values": ["us-west"]} in terms[0]["matchExpressions"]
    # the tag it ran at (ledger row diag-head-epochs-analyse-v2, commit 9b66e5f)
    assert {"name": "REPO_REF", "value": "mtx-s1.94"} in c["env"]
    sh = c["args"][0]
    assert subprocess.run(["bash", "-n"], input=sh, text=True).returncode == 0
    # the placeholder and a full /data stop the job before it clones or writes anything
    assert sh.index('[ "${REPO_REF}" != "TAG_PENDING" ] || {') < sh.index("git clone")
    assert sh.index('[ "${USED}" -lt 85 ] || {') < sh.index("mkdir -p ${OUT}")
    assert "USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)" in sh
    assert 'git clone --depth 1 --branch "${REPO_REF}"' in sh
    # only the analyse step, on the existing arrays, into a new path
    assert re.findall(r"head_epoch_diag\.py (\w+)", sh) == ["analyse"]
    assert "ARRAYS=/data/results/eval/head_epoch_diag\n" in sh and "--out ${ARRAYS}" in sh
    out, js = re.search(r"OUT=(\S+)", sh).group(1), re.search(r"JSON=(\S+)", sh).group(1)
    assert out != "/data/results/eval/head_epoch_diag" and js == "${OUT}/head_epoch_diag_v2.json"
    assert '[ -e ${JSON} ] && {' in sh                       # its own output is never overwritten
