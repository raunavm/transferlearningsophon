"""tab:tests' bootstraps (PI 2026-11-23, LEDGER SP-12 step 4): paired_bootstrap.py's class groups, step, second task,
headline-only and cache, mass_sd_boot.py's added differences, and scripts/build_boot_tests_jobs.py's spec."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PB = _load("paired_bootstrap_t", "experiments/STATS/paired_bootstrap.py")


def _npz(path, y, **kw):
    np.savez(path, rows=np.arange(y.size), y=y.astype(np.int16), **kw)
    return f"{path.stem}={path}"


def _probs(y, k, shift, seed):
    z = np.random.default_rng(seed).normal(size=(y.size, k))
    z[np.arange(y.size), y] += shift
    return (np.exp(z) / np.exp(z).sum(1, keepdims=True)).astype(np.float32)


def test_native_groups_are_the_mean_auc_over_the_resonant_and_the_qcd_classes(tmp_path):
    y = np.random.default_rng(1).choice([0, 5, 161, 187], 3000)
    p = _probs(y, 188, 1.0, 2)
    out = tmp_path / "o.json"
    PB.main(["--model", _npz(tmp_path / "a.npz", y, p=p), "--native-groups", "--b", "8", "--out", str(out)])
    m = json.loads(out.read_text())["models"]["a"]
    q = p.astype(np.float64)
    auc = {k: roc_auc_score(y == k, q[:, k]) for k in (0, 5, 161, 187)}
    assert m["macro_resonant"]["one_minus_auc"] == pytest.approx(1 - np.mean([auc[0], auc[5]]), abs=1e-12)
    assert m["macro_qcd"]["one_minus_auc"] == pytest.approx(1 - np.mean([auc[161], auc[187]]), abs=1e-12)
    assert m["macro"]["one_minus_auc"] == pytest.approx(1 - np.mean(list(auc.values())), abs=1e-12)


def _three(tmp_path, n=2000):
    y = np.random.default_rng(3).integers(0, 4, n)
    args = []
    for g, shift in (("a", 0.6), ("b", 1.0), ("c", 1.6)):
        args += [x for k in (1, 2) for x in ("--model", _npz(tmp_path / f"{g}{k}.npz", y, p=_probs(y, 4, shift, 10 * k + ord(g))))]
    return y, args


def test_a_step_is_two_differences_and_its_sigma_the_coefficient_weighted_run_sds(tmp_path):
    _, args = _three(tmp_path)
    out = tmp_path / "o.json"
    PB.main(args + ["--headline", "--step", "s=a1,a2/b1,b2/c1,c2", "--diff", "ab=a1,a2/b1,b2", "--diff", "bc=b1,b2/c1,c2",
                    "--b", "30", "--out", str(out)])
    d = json.loads(out.read_text())
    for k in ("macro", "accuracy"):
        s, ab, bc = d["diffs"]["s"][k], d["diffs"]["ab"][k], d["diffs"]["bc"][k]
        assert s["delta"] == pytest.approx(ab["delta"] - bc["delta"], abs=1e-12)
        sa, sb, sc = s["sd_runs"]["a"], s["sd_runs"]["b"], s["sd_runs"]["c"]
        assert (sa, sb, sc) == pytest.approx((ab["sd_a"], ab["sd_b"], bc["sd_b"]))
        assert s["sd_a"] ** 2 + s["sd_b"] ** 2 == pytest.approx(sa ** 2 + 4 * sb ** 2 + sc ** 2)
        assert s["sigma"] == pytest.approx(np.sqrt(s["sd_a"] ** 2 + s["sd_b"] ** 2 + s["boot_sd"] ** 2))
        assert s["quantity"] == ("one_minus_auc" if k == "macro" else k) and s["boot_sd"] > 0
    assert (d["diffs"]["s"]["a"], d["diffs"]["s"]["c"]) == (["a1", "a2"], ["c1", "c2"])


def test_a_cached_run_is_the_direct_run_and_a_changed_one_is_refused(tmp_path):
    _, args = _three(tmp_path)
    flags = ["--headline", "--diff", "ab=a1,a2/b1,b2", "--step", "s=a1,a2/b1,b2/c1,c2", "--b", "20"]
    PB.main(args + flags + ["--out", str(tmp_path / "direct.json")])
    cache = tmp_path / "cache"
    PB.main(args[:4] + ["--headline", "--b", "20", "--cache", str(cache), "--cache-only"])
    PB.main(args[4:] + ["--headline", "--b", "20", "--cache", str(cache), "--cache-only"])
    assert len(list(cache.glob("*.npz"))) == 6
    PB.main(args + flags + ["--cache", str(cache), "--out", str(tmp_path / "cached.json")])
    a, b = (json.loads((tmp_path / f).read_text()) for f in ("direct.json", "cached.json"))
    assert a["models"] == b["models"] and a["diffs"] == b["diffs"]
    with pytest.raises(SystemExit, match="was made from"):
        PB.main(args + ["--headline", "--b", "21", "--cache", str(cache), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="--cache-only"):
        PB.main(args + ["--cache-only"])


def _qg_cell(tmp_path, name, y, s, yh, sh):
    fd = tmp_path / name / "features"
    fd.mkdir(parents=True)
    for f, lab, sc in (("logits", "label188", s), ("logits_herwig", "label_herwig", sh)):
        np.save(fd / f"{f}.npy", np.c_[np.zeros(y.size if f == "logits" else yh.size), sc].astype(np.float32))
        np.save(fd / f"{lab}.npy", (y if f == "logits" else yh).astype(np.int16))
    return f"{name}={fd.parent}"


def test_the_second_task_ratio_is_taken_per_resample(tmp_path, monkeypatch):
    monkeypatch.setattr(PB._ft_curves().BM, "check_label_order", lambda ds: None)
    rng = np.random.default_rng(4)
    y, yh = (rng.random(3000) < 0.5).astype(int), (rng.random(2500) < 0.5).astype(int)
    cells = [_qg_cell(tmp_path, f"m{k}", y, y * (1 + 0.2 * k) + rng.normal(size=y.size),
                      yh, yh * (0.6 + 0.1 * k) + rng.normal(size=yh.size)) for k in (1, 2, 3, 4)]
    out = tmp_path / "o.json"
    PB.main([x for c in cells for x in ("--model", c)] + ["--task", "qg", "--second-task", "qg_herwig", "--headline",
                                                           "--diff", "d=m1,m2/m3,m4", "--diff", "hp=m1@second,m2@second/m1,m2",
                                                           "--b", "25", "--out", str(out)])
    d = json.loads(out.read_text())
    for k in ("r30", "r50", "accuracy"):
        m = d["models"]["m1"]
        sh = PB.headline(yh, PB.load(tmp_path / "m1", "qg_herwig")[2], PB.counts(yh, 1, 0))[k][0]
        assert m[f"{k}_ratio"]["value"] == pytest.approx(sh / m[k]["value"], rel=1e-12)
        e = d["diffs"]["d"][f"{k}_ratio"]
        assert e["quantity"] == f"{k}_ratio" and e["boot_sd"] > 0
        hp, va, vb = d["diffs"]["hp"][k], [d["models"][m][f"{k}_second"]["value"] for m in ("m1", "m2")], \
            [d["models"][m][k]["value"] for m in ("m1", "m2")]
        assert hp["delta"] == pytest.approx(np.mean(va) - np.mean(vb)) and hp["quantity"] == k
        assert (hp["sd_a"], hp["sd_b"]) == pytest.approx((np.std(va, ddof=1), np.std(vb, ddof=1))) and hp["boot_sd"] > 0
        # two test samples: unpaired, each side's mean's bootstrap variance added
        assert hp["resampling"] == "unpaired" and hp["boot_sd"] == pytest.approx(np.hypot(hp["boot_sd_a"], hp["boot_sd_b"]))
        assert hp["sigma"] == pytest.approx(np.sqrt(hp["sd_a"] ** 2 + hp["sd_b"] ** 2 + hp["boot_sd"] ** 2))
        assert "resampling" not in d["diffs"]["d"][k]
    assert d["second_task"] == "qg_herwig" and d["second_seed"] == d["seed"] + 1 and d["second_n_jets"] == yh.size
    with pytest.raises(SystemExit, match="not models"):
        PB.main([x for c in cells for x in ("--model", c)] + ["--task", "qg", "--headline", "--diff", "x=m1@second/m1",
                                                               "--out", str(tmp_path / "x.json")])


def test_headline_only_scores_no_auc(tmp_path):
    y = np.random.default_rng(5).integers(0, 4, 1000)
    out = tmp_path / "o.json"
    PB.main(["--model", _npz(tmp_path / "a.npz", y, p=_probs(y, 4, 1.0, 6)), "--headline-only", "--b", "5",
             "--out", str(out)])
    assert set(json.loads(out.read_text())["models"]["a"]) == {"input", "accuracy"}


def test_mass_sd_boot_keeps_its_two_pairs_and_adds_the_mass_tables_three():
    MS = _load("mass_sd_boot_t", "experiments/EVAL/mass_sd_boot.py")
    assert MS.DIFFS[:2] == (("162 classes + mass", "162 classes"), ("17 classes + mass", "17 classes"))
    assert set(MS.DIFFS[2:5]) == {("17 classes", "188 classes"), ("188 classes", "random network"),
                                  ("17 classes + mass", "162 classes + mass")}
    # review of 5c40d69: every pretrained group against the random network
    assert {a for a, b in MS.DIFFS if b == "random network"} == set(MS.GROUPS) - {"random network"}
    assert all(a in MS.GROUPS and b in MS.GROUPS for a, b in MS.DIFFS)
    assert MS.CROSS == ((("162 classes + mass", "regressed"), ("162 classes", "mlp")),)


def test_the_spec_is_ours_pinned_and_writes_new_files_only():
    b = _load("build_boot_tests_jobs_t", "scripts/build_boot_tests_jobs.py")
    for text, parts in ((b.main_spec("mtx-s9.99", "8Gi"), 5), (b.smoke_spec("mtx-s9.99", "16Gi"), 1)):
        assert f"completions: {parts}\n" in text and "ConfigMap" not in text and "kind: Secret" not in text
        assert text.count("-raunav\n") == 1 and "--branch mtx-s9.99 " in text and '-le 90 ]' in text
        assert all(f'"{n}"' in text for n in b.R5.CPU_NODES[b.CPU_MODEL]) and "nvidia.com/gpu" not in text
        assert f'[ "${{CPU}}" = "{b.CPU_MODEL}" ]' in text
    text = b.main_spec("mtx-s9.99", "8Gi")
    assert "parallelism: 4\n" in text and text.count("--cache-only") == 3
    existing = {"probe_jc2bal_N1000000.json", "probe_jc1_N1000000.json", "probe_top_N1200000.itop4.all.json",
                "probe_qg_N1600000.nqg.all.json", "ft_top_N1200000.all.json", "ft_qg_N1600000.all.json"}
    outs = [*b.probe_inputs(), *b.ft_inputs(), b.JC2_OUT]
    assert len(outs) == 16 and not {o.rsplit("/", 1)[1] for o in outs} & existing
    assert all(f"[ -e {o} ] ||" in text for o in outs) and f"mv -n {b.MASS_V4.replace('.json', '.new.json')}" in text
    for arm in b.JC2_ARMS:
        assert all(f"--model {m}=" in b.jc2_file() for m in b.models(arm))
    assert sorted(m for g in b.jc2_groups() for m in g) == sorted(b.jc2_inputs()) and len(b.jc2_inputs()) == 23
    h = b.herwig_spec("mtx-s9.99", "4Gi")
    assert "completions: 1\n" in h and f"[ -e {b.HERWIG_OUT} ] ||" in h and h.count("@second") == 6 and "-raunav\n" in h


def test_the_review_specs_write_new_files_from_the_listed_inputs():
    b = _load("build_boot_tests_jobs_r", "scripts/build_boot_tests_jobs.py")
    s = b.review_specs("mtx-s9.99", "6Gi")
    assert set(s) == {"boot-review-a-raunav", "boot-review-b-raunav"}
    for name, text in s.items():
        assert "completions: 1\n" in text and f"{name}\n" in text and "ConfigMap" not in text
        assert f'[ "${{CPU}}" = "{b.CPU_MODEL}" ]' in text
    a, ft = s["boot-review-a-raunav"], s["boot-review-b-raunav"]
    assert all(f"[ -e {o} ] ||" in a for o in [*b.REVIEW_PAIRS.values(), b.REVIEW_KIN_LP]) and f"mv -n " in a
    assert f"[ -e {b.REVIEW_FT} ] ||" in ft and "--task top" in ft and ft.count(".lpzero=") == 6
    assert ft.count("-188lpzero=") == ft.count("-initbnlpzero=") == len(b.R5.BOOTALL_ARMS)
    assert "--diff kin-188=mtx-l188kin-s1/" in ft and "best70_kin/leg_top/l188kin-s1/N1200000/s1" in ft
    assert "completions: 1\n" in b.review_smoke_spec("mtx-s9.99", "8Gi")
