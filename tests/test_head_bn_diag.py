"""BatchNorm statistics recomputed on the v1 diagnostic's checkpoints
(experiments/DIAG/head_bn_diag.py): the recompute changes BatchNorm buffers and
nothing else, every checkpoint sees the same BatchNorm sample and random draws,
the analysis flags by rules that can be checked by hand, the job spec, and the
whole infer -> analyse chain on CPU."""
import importlib.util
import json
import pathlib
import re
import shlex
import shutil
import subprocess

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = REPO / "experiments/DIAG/k8s/job-diag-head-bn-raunav.yaml"
V1_SPEC = REPO / "experiments/DIAG/k8s/job-diag-head-epochs-raunav.yaml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pytest.importorskip("torch")     # head_epoch_diag, which the script and its analysis load, imports torch
B = _load("head_bn_diag", "experiments/DIAG/head_bn_diag.py")
P = B.HD.PRIMARY_RULE


# ------------------------------------------------------------------ the recompute
def _tiny():
    """BatchNorm layers upstream of a dropout, and a random draw in train mode, as in
    ParT's embeddings, dropout and trimmer. Records what it sees in train mode."""
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn0 = torch.nn.BatchNorm1d(3)
            self.lin = torch.nn.Linear(3, 4)
            self.bn1 = torch.nn.BatchNorm1d(4)
            self.drop = torch.nn.Dropout(0.5)
            self.head = torch.nn.Linear(4, 2)
            self.seen = []

        def forward(self, x):
            if self.training:
                self.seen.append((x.clone(), torch.rand(1).item()))
            return self.head(self.drop(self.bn1(self.lin(self.bn0(x)))))
    return Tiny()


def _sample(n_batches=4, size=16, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_batches * size, 3, generator=g) * 2 + 1
    batches = [({"x": x[i:i + size]}, {}, {"_rowid": torch.arange(i, i + size)})
               for i in range(0, len(x), size)]
    return {"batches": batches, "n_jets": len(x), "input_names": ["x"]}, x


def test_the_recompute_changes_batchnorm_buffers_and_nothing_else():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    model = _tiny()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["bn0.running_mean"].fill_(5.0)                  # stale statistics, as a stored epoch carries
    state["bn1.running_var"].fill_(9.0)
    sample, x = _sample()
    after, check = B.recompute_state(model, state, sample, torch.device("cpu"), False, torch_seed=7)
    params = [n for n, _ in model.named_parameters()]
    assert all(torch.equal(after[k], state[k]) for k in params)             # bitwise
    changed = {k for k in after if not torch.equal(after[k], state[k])}
    assert changed == {f"bn{i}.{b}" for i in (0, 1) for b in ("running_mean", "running_var", "num_batches_tracked")}
    assert check["ok"] and check["parameters_bitwise_equal"] and check["only_batchnorm_buffers_changed"]
    assert check["batchnorm_layers"] == check["batchnorm_layers_recomputed"] == 2
    assert check["running_stats_changed"] == 4
    # a cumulative average over equal batches: the mean of the whole sample, from scratch
    torch.testing.assert_close(after["bn0.running_mean"], x.mean(0))
    assert int(after["bn0.num_batches_tracked"]) == 4
    assert model.bn0.momentum == 0.1 and not model.training                 # momentum restored, eval mode
    # the check fails on anything but a BatchNorm buffer
    bad = {**after, "lin.weight": after["lin.weight"] + 1}
    assert not B.bn_buffer_check(model, state, bad)["ok"]
    assert not B.bn_buffer_check(model, state, {k: v for k, v in after.items() if k != "head.bias"})["ok"]


def test_every_checkpoint_sees_the_same_jets_and_random_draws():
    torch = pytest.importorskip("torch")
    sample, _ = _sample()
    seen = []
    for seed in (1, 2):                                   # two different checkpoints
        torch.manual_seed(seed)
        m = _tiny()
        state = {k: v.clone() for k, v in m.state_dict().items()}
        m.seen.clear()
        torch.manual_seed(100 + seed)                     # whatever the generators held before
        B.recompute_state(m, state, sample, torch.device("cpu"), False, torch_seed=7)
        seen.append(m.seen)
    a, b = seen
    assert len(a) == len(b) == 4
    assert all(torch.equal(x1, x2) and r1 == r2 for (x1, r1), (x2, r2) in zip(a, b))
    m = _tiny()
    B.recompute_state(m, {k: v.clone() for k, v in m.state_dict().items()}, sample, torch.device("cpu"),
                      False, torch_seed=8)
    assert [r for _, r in m.seen] != [r for _, r in a]   # the seed is what fixes the draws


def test_the_trimmer_is_put_past_its_warm_up_in_either_weaver():
    torch = pytest.importorskip("torch")

    class SequenceTrimmer(torch.nn.Module):               # weaver 0.4.17: a plain int, warm-up 5
        def __init__(self):
            super().__init__()
            self._counter = 0
    old = torch.nn.Sequential(SequenceTrimmer())
    B.past_trimmer_warmup(old)
    assert old[0]._counter == 5
    pytest.importorskip("weaver")
    from weaver.nn.model.ParticleTransformer import SequenceTrimmer as Dev
    new = Dev(enabled=True)
    if not torch.is_tensor(getattr(new, "_counter", None)):
        pytest.skip("this weaver's trimmer counter is not a tensor")
    B.past_trimmer_warmup(new)
    assert int(new._counter) == new.warmup_steps


def test_the_bn_sample_takes_exactly_n_jets_out_of_the_stream():
    torch = pytest.importorskip("torch")
    stream = [({"x": torch.arange(4.0)[:, None] + 10 * i, "unused": torch.zeros(4)}, {"y": torch.zeros(4)},
               {"_rowid": torch.arange(4) + 4 * i, "_jet_label": torch.full((4,), 160 + i)}) for i in range(5)]
    batches, rid, lab = B.take_jets(iter(stream), 10, ["x"])
    assert [len(z["_rowid"]) for _, _, z in batches] == [4, 4, 2] and rid.tolist() == list(range(10))
    assert lab.tolist() == [160] * 4 + [161] * 4 + [162] * 2 and set(batches[0][0]) == {"x"}
    with pytest.raises(SystemExit, match="ended after 20 jets"):
        B.take_jets(iter(stream), 21, ["x"])


# ------------------------------------------------------------------ the analysis
N = 1000


def _jets():
    native = np.full(N, 20)
    native[:300] = np.tile([0, 1], 150)                   # the b-vs-c probe's jets
    native[600:] = 170                                    # QCD
    return native


def _arrays(acc, native, rng):
    bvc = np.nonzero(np.isin(native, [0, 1]))[0]
    y = (native[bvc] == 0)
    return {"correct": np.arange(N) < round(acc * N), "p_qcd": np.zeros(N, np.float32),
            "log_odds": np.zeros(N, np.float32),
            "feat_bvc": (rng.normal(size=(bvc.size, 4)) + 2 * y[:, None]).astype(np.float32)}


def _write_run(out, name, accs, best_epoch, rng):
    rd = out / name
    rd.mkdir()
    native = _jets()
    for tag, acc in accs.items():
        np.savez(rd / f"{tag}.npz", **_arrays(acc, native, rng))
    meta = {"run_dir": f"/data/results/mtx/{name}", "rung": "R16_Q1", "num_classes": 17, "num_reg": 0,
            "best_epoch": best_epoch, "gpu": "cpu", "averages_agree": True,
            "bn_checks": {t: {"ok": True} for t in accs if t.endswith("_bn")},
            "checkpoints": {t: {"path": f"{t}.pt", "sha256": f"sha-{t}"} for t in accs}}
    (rd / "DONE").write_text(json.dumps(meta))


def test_the_analysis_flags_by_hand_computable_rules(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    out = tmp_path / "out"
    out.mkdir()
    native = _jets()
    (out / "sample.json").write_text(json.dumps({"n_jets": N}))
    (out / "bn_sample.json").write_text(json.dumps({"rows_sha256": "rows"}))
    np.savez(out / "sample.npz", native=native, bvc_rows=np.nonzero(np.isin(native, [0, 1]))[0])
    # Run a: epoch 79 is defective as stored (0.40 against the stored median 0.60). The
    # recompute takes it to 0.56: sound at 10% and 15% (thresholds 0.54, 0.51), still
    # defective at 5% (0.57). It takes the sound epoch 70 to 0.565, defective at 5% only.
    # The other recomputed epochs are at 0.70: a reference that took them in would sit
    # above 0.60 and judge the repair against a moved target.
    a = {**{f"e{e}": 0.6 for e in range(70, 79)}, "e79": 0.4,
         **{f"e{e}_bn": 0.7 for e in range(71, 79)}, "e70_bn": 0.565, "e79_bn": 0.56,
         "wavg_buf": 0.61, "wavg_bn": 0.62}
    # Run b: nothing defective as stored; its v2-style average (0.30) and its best-validation
    # checkpoint after the recompute (0.42) are, against 0.50 (threshold 0.45).
    b = {**{f"e{e}": 0.5 for e in range(70, 80)}, **{f"e{e}_bn": 0.5 for e in range(70, 80)},
         "wavg_buf": 0.5, "wavg_bn": 0.3, "best": 0.5, "best_bn": 0.42}
    _write_run(out, "mtx-a-s1", a, 75, rng)
    _write_run(out, "mtx-b-s2", b, 40, rng)
    # The committed v1 diagnostic, for run a only. Its e78 file differs from the one
    # scored here; its wavg file always differs (another file name in the archive).
    ref = tmp_path / "committed.json"
    ref.write_text(json.dumps({"runs": {"mtx-a-s1": {"checkpoints": {
        **{f"e{e}": {"head": {"top1_accuracy": 0.6}, "sha256": f"sha-e{e}"} for e in range(70, 78)},
        "e78": {"head": {"top1_accuracy": 0.6}, "sha256": "sha-other"},
        "e79": {"head": {"top1_accuracy": 0.403}, "sha256": "sha-e79"},
        "wavg": {"head": {"top1_accuracy": 0.612}, "sha256": "another"}}}}}))
    monkeypatch.setenv("REPO_REF", "test-ref")
    js = tmp_path / "bn.json"
    assert B.main(["analyse", "--out", str(out), "--json", str(js), "--reference", str(ref)]) == 0
    d = json.loads(js.read_text())
    assert d["code"]["repo_ref"] == "test-ref" and d["primary_rule"] == P and set(d["rules"]) == set(B.HD.DEFECT_RULES)
    assert {"sample.json", "sample.npz", "bn_sample.json", "reference", "mtx-a-s1/DONE"} <= set(d["inputs"])

    ra, rb = d["runs"]["mtx-a-s1"], d["runs"]["mtx-b-s2"]
    assert ra["reference"][P] == pytest.approx(0.6) and ra["best_is"] == "e75" and rb["best_is"] == "best"
    assert ra["checkpoints"]["e79_bn"]["head"]["top1_accuracy"] == pytest.approx(0.56)
    assert ra["checkpoints"]["e70"]["sha256"] == "sha-e70" and "auc" in ra["checkpoints"]["wavg_bn"]["probe"]
    assert set(ra["flags"]) == set(a) and set(ra["flags"]["e70"]) == set(B.HD.DEFECT_RULES)
    assert ra["bn_repair"][P] == {"defective_as_stored": ["e79"], "repaired_by_bn": ["e79"], "not_repaired_by_bn": [],
                                  "sound_made_defective_by_bn": [], "wavg_buf_defective": False,
                                  "wavg_bn_defective": False, "best_defective": False, "best_bn_defective": False}
    five = ra["bn_repair"]["top1_5pct_vs_median"]
    assert five["repaired_by_bn"] == [] and five["not_repaired_by_bn"] == ["e79"]
    assert five["sound_made_defective_by_bn"] == ["e70"]
    assert ra["bn_repair"]["top1_15pct_vs_median"]["repaired_by_bn"] == ["e79"]
    assert all(not v["defective_as_stored"] for k, v in ra["bn_repair"].items() if k.startswith("qcd_log_odds"))
    assert ra["bn_minus_stored_top1"]["e79"] == pytest.approx(0.16)
    w = ra["wavg_bn_minus_wavg_buf_top1"]
    assert w["diff"] == pytest.approx(0.01) and w["jets_that_differ"] == 10 and 0.002 < w["boot_se"] < 0.0045
    rep = ra["reproducibility"]
    assert rep["max_abs_top1_diff"] == pytest.approx(0.003) and rep["abs_top1_diff"]["wavg_buf"] == pytest.approx(0.002)
    assert rep["sha256_differs"] == ["e78"]                 # wavg_buf's sha256 is not compared
    assert rb["reproducibility"] is None and rb["bn_repair"][P]["best_bn_defective"]

    c = d["counts"][P]
    assert c["runs"] == 2 and c["runs_with_a_defective_stored_epoch"] == ["mtx-a-s1"]
    assert (c["defective_stored_epochs"], c["repaired_by_bn"], c["sound_epochs_made_defective_by_bn"]) == (1, 1, 0)
    assert c["runs_with_every_defective_epoch_repaired"] == ["mtx-a-s1"]
    assert c["wavg_buf_defective"] == [] and c["wavg_bn_defective"] == ["mtx-b-s2"]
    assert c["best_defective"] == [] and c["best_bn_defective"] == ["mtx-b-s2"]
    c5 = d["counts"]["top1_5pct_vs_median"]
    assert (c5["repaired_by_bn"], c5["sound_epochs_made_defective_by_bn"]) == (0, 1)
    assert c5["runs_with_a_sound_epoch_made_defective"] == ["mtx-a-s1"] and c5["runs_with_every_defective_epoch_repaired"] == []
    assert d["reproducibility"]["runs_compared"] == 1 and d["reproducibility"]["gpus"] == ["cpu"]
    assert d["reproducibility"]["sha256_differs"] == {"mtx-a-s1": ["e78"]}


# ------------------------------------------------------------------ the job spec
def _python_args(sh, script):
    """{flag: [values]} of the `python3 <script> infer` command in a spec's shell."""
    line = next(ln for ln in sh.replace("\\\n", " ").splitlines() if f"{script} infer" in ln)
    toks, args, flag = shlex.split(line.split(" || ")[0]), {}, None
    for t in toks[toks.index("infer") + 1:]:
        if t.startswith("--"):
            flag = t
            args[flag] = []
        else:
            args[flag].append(t)
    return args


def test_the_job_spec_is_the_v1_diagnostics_with_the_batchnorm_step():
    yaml = pytest.importorskip("yaml")
    new, old = yaml.safe_load(SPEC.read_text()), yaml.safe_load(V1_SPEC.read_text())
    assert "raunav" in new["metadata"]["name"] and new["metadata"]["name"] != old["metadata"]["name"]
    assert new["spec"]["backoffLimit"] == old["spec"]["backoffLimit"]
    # v1's policy plus the v2 fine-tuning specs' node-fault rule: exit 43 is retried
    # without counting as one of the two failed attempts
    count43 = {"action": "Count", "onExitCodes": {"containerName": "main", "operator": "In", "values": [43]}}
    rules = list(old["spec"]["podFailurePolicy"]["rules"])
    assert new["spec"]["podFailurePolicy"]["rules"] == rules[:1] + [count43] + rules[1:]
    pn, po = new["spec"]["template"]["spec"], old["spec"]["template"]["spec"]
    for k in ("restartPolicy", "tolerations", "volumes"):
        assert pn[k] == po[k]
    # v1's node rules with one GPU product, the L40 the committed v1 diagnostic ran on:
    # a retry after an eviction cannot land on another product and halt the job
    want = yaml.safe_load(V1_SPEC.read_text())["spec"]["template"]["spec"]["affinity"]
    terms = want["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    prod = next(e for e in terms[0]["matchExpressions"] if e["key"] == "nvidia.com/gpu.product")
    assert "NVIDIA-L40" in prod["values"]
    prod["values"] = ["NVIDIA-L40"]
    assert pn["affinity"] == want
    committed = json.loads((REPO / "experiments/FIGS/data/head_epoch_diag/head_epoch_diag_v2.json").read_text())
    assert committed["sample"]["gpu"] == "NVIDIA L40"
    cn, co = pn["containers"][0], po["containers"][0]
    for k in ("name", "image", "command", "volumeMounts"):
        assert cn[k] == co[k]
    assert cn["resources"]["limits"]["nvidia.com/gpu"] == co["resources"]["limits"]["nvidia.com/gpu"]
    assert {"name": "REPO_REF", "value": "mtx-s1.98"} in cn["env"]
    sh = cn["args"][0]
    assert subprocess.run(["bash", "-n"], input=sh, text=True).returncode == 0
    assert 'git clone --depth 1 --branch "${REPO_REF}"' in sh
    # a full /data stops the job before it writes anything
    assert "USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)" in sh
    assert sh.index('[ "${USED}" -le 85 ] || {') < sh.index("mkdir -p ${OUT}")
    assert "OUT=/data/results/eval/head_epoch_diag_bn\n" in sh
    assert re.findall(r"head_bn_diag\.py (\w+)", sh) == ["infer", "analyse"]
    # the v1 diagnostic's runs, epochs and test jets
    na, oa = _python_args(sh, "head_bn_diag.py"), _python_args(co["args"][0], "head_epoch_diag.py")
    for flag in ("--runs", "--epochs", "--data-test", "--max-jets", "--stride", "--align-with",
                 "--batch-size", "--num-workers"):
        assert na[flag] == oa[flag], flag
    # the BatchNorm jets come from the files every one of the eight runs trained on
    v1 = ("l188-s5", "l162-s5", "r42_q1-s5", "r16_q1_mass-s4", "l188-s1", "l162-s1b", "r42_q1-s1", "r16_q1_mass-s1")
    for run in v1:
        text = (REPO / f"experiments/MTX/k8s/job-mtx-{run}-raunav.yaml").read_text()
        train = re.search(r"--data-train ((?:\S+:\S+ )+)", text).group(1).split()
        assert na["--bn-train"] == train, run
    assert na["--bn-config"] == ["${CFG}"] and "CFG=configs/arms/L188.yaml\n" in sh
    # the derived states go to the pod's local disk, not the shared volume; no CPU fallback
    assert len(na["--state-dir"]) == 1 and not na["--state-dir"][0].startswith("/data")
    assert "--allow-cpu" not in na
    # the JSON and its sha256 end up in the log
    assert sh.index("sha256sum ${JSON}") > sh.index("head_bn_diag.py analyse")
    assert 'cat ${JSON}' in sh


def test_the_script_compiles_under_the_images_python_310():
    import ast
    path = REPO / "experiments/DIAG/head_bn_diag.py"
    ast.parse(path.read_text(), feature_version=(3, 10))
    py310 = shutil.which("python3.10")
    if py310 is None:
        pytest.skip("no python3.10 here: only the ast check ran")
    r = subprocess.run([py310, "-c", "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')",
                        str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------ infer -> analyse on CPU
def test_infer_refuses_to_start_without_a_gpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "diag"
    argv = ["infer", "--runs", "r:R16_Q1:17:0", "--data-test", "f.parquet", "--max-jets", "1", "--stride", "1",
            "--align-with", "cache", "--bn-config", "c.yaml", "--bn-train", "QCD:f.parquet",
            "--state-dir", str(tmp_path / "states"), "--out", str(out)]
    assert B.main(argv) == B.EXIT_NO_GPU != B.PV.EXIT_HALT
    assert not out.exists() and not (tmp_path / "states").exists()


def test_infer_and_analyse_end_to_end_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("weaver")
    pa = pytest.importorskip("pyarrow")
    yaml = pytest.importorskip("yaml")
    import sys
    import pyarrow.parquet as pq
    from weaver.utils.data.config import DataConfig, _md5
    mc = _load("mass_control_tests", "tests/test_extract_mass_control.py")
    dc = DataConfig.load(str(REPO / "configs/data/JetClassII_base.yaml"), load_observers=True)

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
        torch.save(B.HD.XF.build_model(dc, 17).state_dict(), run / f"net_epoch-{e}_state.pt")
    shutil.copy(run / "net_epoch-50_state.pt", run / "net_best_epoch_state.pt")

    cache = tmp_path / "eval" / run.name / "features_e79"
    monkeypatch.setattr(sys, "argv", ["x", "--checkpoint", str(run / "net_epoch-71_state.pt"),
                                      "--num-classes", "17", "--save-logits", "--arm", "R16_Q1",
                                      "--data-test", *files, "--out", str(cache), "--batch-size", "64",
                                      "--num-workers", "0", "--max-jets", "1500"])
    assert B.HD.XF.main() == 0

    cfg = tmp_path / "L188.yaml"            # the arm config, with a sidecar as make_weight writes it
    src = (REPO / "configs/arms/L188.yaml").read_text()
    cfg.write_text(src)
    opts = yaml.safe_load(src)
    w = opts["weights"]
    nx, ny = len(w["reweight_vars"]["jet_pt"]) - 1, len(w["reweight_vars"]["jet_sdmass"]) - 1
    w["reweight_hists"] = {c: np.full((nx, ny), 0.3 if "QCD" in c else 0.6).tolist() for c in w["reweight_classes"]}
    (tmp_path / f"L188.{_md5(str(cfg))}.auto.yaml").write_text(yaml.safe_dump(opts, sort_keys=False))
    monkeypatch.setitem(B.BN_STREAM, "batch_size", 64)   # small enough for 600-row files
    monkeypatch.setitem(B.BN_STREAM, "split_num", 2)

    out = tmp_path / "diag"
    argv = ["infer", "--runs", f"{run}:R16_Q1:17:0", "--epochs", "70", "71",
            "--data-test", *files, "--max-jets", "1500", "--stride", "3", "--align-with", str(cache),
            "--batch-size", "64", "--num-workers", "0", "--bn-config", str(cfg),
            "--bn-train", f"Res2P:{files[0]}", f"Res34P:{files[1]}", f"QCD:{files[2]}",
            "--bn-jets", "200", "--bn-workers", "0", "--reference", str(tmp_path / "absent.json"),
            "--state-dir", str(tmp_path / "states"), "--allow-cpu", "--out", str(out)]
    assert B.main(argv) == 0
    bn = json.loads((out / "bn_sample.json").read_text())
    assert bn["n_jets"] == 200 and bn["seed_data"] == B.BN_SEED and bn["amp"] is False
    assert set(bn["files"]) <= {pathlib.Path(f).name for f in files} and bn["files_per_family"]["QCD"] == 1
    assert bn["torch_seed"] == B.PV.epoch_seed(B.BN_SEED, "bn-recompute", 0)
    sample = json.loads((out / "sample.json").read_text())
    assert sample["n_jets"] == 500 and sample["matches_committed_sample"] is None
    meta = json.loads((out / run.name / "DONE").read_text())
    assert list(meta["checkpoints"]) == ["e70", "e71", "e70_bn", "e71_bn", "wavg_buf", "wavg_bn", "best", "best_bn"]
    assert meta["best_epoch"] == 50 and meta["averages_agree"] and meta["gpu"] == "cpu"
    assert set(meta["bn_checks"]) == {"e70_bn", "e71_bn", "best_bn", "wavg_bn"}
    assert all(c["ok"] and c["batchnorm_layers"] == 6 and c["running_stats_changed"] == 12
               for c in meta["bn_checks"].values())
    for tag, ck in meta["checkpoints"].items():
        assert ck["sha256"] == B.HD.sha256(pathlib.Path(ck["path"]))
    s70 = torch.load(run / "net_epoch-70_state.pt", weights_only=True)
    assert not list(out.rglob("*_state.pt"))                     # the derived states stay out of --out
    s70bn = torch.load(tmp_path / "states" / run.name / "e70_bn_state.pt", weights_only=True)
    differ = {k for k in s70 if not torch.equal(s70[k], s70bn[k])}
    assert differ and all(re.search(r"(running_mean|running_var|num_batches_tracked)$", k) for k in differ)
    # the stored epoch is scored exactly as head_epoch_diag scored it
    lab = np.load(cache / "label188.npy")[::3]
    want = float((np.load(cache / "logits.npy")[::3].argmax(1) == B.HD.EA.vocabulary_map("R16_Q1")[lab]).mean())
    assert np.load(out / run.name / "e71.npz")["correct"].mean() == pytest.approx(want, abs=1e-12)

    assert B.main(argv) == 0                                     # a finished run is skipped, same BatchNorm rows
    js = tmp_path / "bn.json"
    assert B.main(["analyse", "--out", str(out), "--json", str(js), "--reference", str(tmp_path / "absent.json")]) == 0
    d = json.loads(js.read_text())
    r = d["runs"][run.name]
    assert set(r["flags"]) == set(meta["checkpoints"]) and r["best_is"] == "best" and r["reproducibility"] is None
    assert set(d["counts"]) == set(B.HD.DEFECT_RULES)

    # the test jets must be the committed sample's
    (tmp_path / "other.json").write_text(json.dumps({"sample": {"native_label_sha256": "0" * 64}}))
    bad = [x if x != str(tmp_path / "absent.json") else str(tmp_path / "other.json") for x in argv]
    with pytest.raises(SystemExit, match="not the jets"):
        B.main([x if x != str(out) else str(tmp_path / "diag2") for x in bad])
    # a retry on another GPU product after a run has finished stops the job
    done = out / run.name / "DONE"
    done.write_text(json.dumps({**meta, "gpu": "another GPU"}))
    with pytest.raises(SystemExit) as e:
        B.main(argv)
    assert e.value.code == B.PV.EXIT_HALT
