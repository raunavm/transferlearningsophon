"""The results deck's figure script runs against the paper's current code (mlst_results.py,
mlst_figures.py): on a synthetic data tree it draws every figure and writes deck_numbers.json
with the paper's selections, and with no data pulled it skips every figure, removes any stale one, and still exits 0."""
import importlib.util
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ("probes_n_jc", "probes_n_bench", "pairs", "mass_resolution", "ft_n_jc", "ft_n_bench", "herwig")
SIZES = (1000, 100000)
GPU = "NVIDIA GeForce RTX 3090"


def _deck():
    spec = importlib.util.spec_from_file_location("deck_figures", REPO / "experiments/FIGS/deck_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(p: pathlib.Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def _tree(D, M, data: pathlib.Path) -> dict:
    """Linear probes, mass regression, fine-tuning read-outs, manifests and compute records for
    the slides' runs (hierarchy, mass regression, random initialization), values drawn at random."""
    rng = np.random.default_rng(1)
    runs = M._paper(REPO)
    models = [m for arm in D.LEVELS + M.MASS + (M.INIT,) for m in runs[arm]]
    auc = lambda: float(0.9 + 0.09 * rng.random())  # noqa: E731
    for m in models:
        cks = ("init",) if m.startswith("init-") else (M.PRIMARY, "best70_bn")
        cells = {}
        for ck in cks:
            jc2 = {"converged": True, **{k: auc() for k in ("macro_auc_ovr", "macro_auc_resonant",
                                                             "macro_auc_qcd", "macro_auc_trained")}}
            cells[ck] = {"jc2_188": {str(n): dict(jc2) for n in SIZES},
                         "jc1": {str(n): {"converged": True, "macro_auc_ovr": auc()} for n in SIZES},
                         **{ds: {str(n): {"converged": True, "auc": auc()} for n in SIZES}
                            for ds in ("top", "qg", "qg_herwig")},
                         M.PAIR_DS: {M.PAIR_N: {k: {"converged": True, "auc": auc()} for k, *_ in M.PAIRS}}}
        _write(data / M.FITS / f"{m}.json", {"cells": cells})
        ck = "init" if m.startswith("init-") else M.PRIMARY
        _write(data / "head" / f"{m}_mass.json",
               {"arms": {ck: {"mlp": {"sigma_eff": float(0.1 + 0.05 * rng.random())},
                              "target": {"sigma_eff": 0.2}}}})
    # fine-tuning: one seed per pretrained run, three from scratch; from-scratch seed 3's R30 is
    # only a bound (no background jet passes), so the deck must leave it out
    inits = [m[len("mtx-"):] for m in models if not m.startswith("init-")]
    rec = lambda bound=False: {"macro_auc_ovr": auc(), "macro_auc_qcd": auc(), "auc": auc(),  # noqa: E731
                               "r50": 50.0, "r30": 1e6 if bound else 120.0, "r50_n_bkg_pass": 400,
                               "r30_n_bkg_pass": 0 if bound else 150, "r50_is_bound": False, "r30_is_bound": bound}
    cells = lambda: {**{i: {f"N{n}": {"s1": rec()} for n in SIZES} for i in inits},  # noqa: E731
                     M.SCRATCH: {f"N{n}": {f"s{s}": rec(s == 3) for s in (1, 2, 3)} for n in SIZES}}
    for _, f, ds, _, _ in M.FT_TASKS:
        p = data / "ft_mlst" / f"{M.FT_TAGS[0]}_{f}.json"
        doc = json.loads(p.read_text()) if p.exists() else {"cells": {}}
        if ds is None:
            doc["cells"] = doc["cells"] or cells()
        else:
            doc["cells"].setdefault(ds, cells())
        _write(p, doc)
    # manifests as pull_mlst_data writes them: the paper's legs at both sizes, and a decoy cell of
    # a leg the paper does not read (leg1, all JetClass-II classes merged), which must not count
    man = {f"{M.PRIMARY}/{leg}/{i}/N{n}/s1": {"init": i, "gpu_device_name": GPU}
           for leg in D.LEGS for i in inits for n in SIZES}
    man.update({f"{M.PRIMARY}/leg1/{i}/N1000/s1": {"init": i, "gpu_device_name": "decoy"} for i in inits})
    _write(data / "ft_mlst/manifests.json", man)
    for m in models:
        if not m.startswith("init-"):
            _write(data / "compute" / f"{m}.json",
                   {"run": m, "epochs": [{"device": GPU, "train_seconds": 3600.0, "val_seconds": 360.0}]})
    return runs


def test_every_figure_is_drawn_from_the_papers_selections(tmp_path, monkeypatch, capsys):
    D = _deck()
    M = D.MF._mlst()
    data, out = tmp_path / "data", tmp_path / "deck"
    runs = _tree(D, M, data)
    monkeypatch.setattr(D, "DATA", data)
    assert D.main(["--outdir", str(out)]) == 0
    assert "skipped" not in capsys.readouterr().out
    for name in FIGURES:
        assert (out / f"{name}.png").stat().st_size > 0
    nums = json.loads((out / "deck_numbers.json").read_text())
    assert nums["fine_tuning_readout"] == M.FT_TAGS[0]
    fits = M._load(data / M.FITS)
    # linear probe, JetClass-II at its 188 classes over all classes: the paper table's value
    n = max(SIZES)
    want = np.mean([1 - fits[m]["cells"][M.PRIMARY]["jc2_188"][str(n)]["macro_auc_ovr"] for m in runs["L188"]])
    got = next(p for p in nums["probes_n_jc"]["jc2_188"]["L188"] if p["n"] == n)
    assert got["oma"] == pytest.approx(want) and got["runs"] == len(runs["L188"])
    # fine-tuning on JetClass-II: the 188-class read-out (leg188), the hierarchy and from scratch
    tag, per, _ = M.load_ft(data)
    want = np.mean([1 - M._ft(per, "jc2", m, n, "macro_auc_ovr") for m in runs["R16_Q1"]])
    got = next(p for p in nums["ft_n_jc"]["jc2"]["values"]["R16_Q1"] if p["n"] == n)
    assert got["oma"] == pytest.approx(want)
    assert set(nums["ft_n_jc"]["jc2"]["values"]) == set(D.LEVELS) | {M.SCRATCH}
    # the mass regression: two runs print both values, as the paper's tables do
    assert nums["mass_resolution"]["rows"]["L162_MASS"]["n"] == 2
    assert " ± " not in nums["mass_resolution"]["rows"]["L162_MASS"]["text"]
    assert set(nums["batchnorm_ratio"]) == {f"{ds}/{met}" for ds, met, _ in M.TASKS}
    assert set(nums["probe_over_init"]) == {"top", "qg", "qg_herwig"}
    rej = nums["ft_rejection"]
    assert rej["fewest_background_jets"] == 150
    assert rej["top_r30"][M.SCRATCH]["n"] == 2 and rej["top_r50"][M.SCRATCH]["n"] == 3
    assert len(rej["only_a_bound"]) == 3 and all(" r30 " in b for b in rej["only_a_bound"])
    pre = [m for a in D.LEVELS + M.MASS for m in runs[a]]
    per_run = len(D.LEGS) * len(SIZES)
    assert nums["ft_status"]["done"] == per_run * len(pre)
    assert nums["ft_status"]["expected"] == 16 * len(pre)
    assert set(nums["ft_status"]["incomplete"].values()) == {per_run}
    assert nums["ft_status"]["gpus"] == [GPU]
    assert nums["compute"]["all"]["runs"] == nums["ft_status"]["runs"]
    # no random partitions on the slides
    assert not any(r.startswith(("RAND", "FLAV")) for r in nums["probes_n_jc"]["jc2_188"])


def test_with_no_data_pulled_every_figure_is_skipped(tmp_path, monkeypatch, capsys):
    D = _deck()
    out = tmp_path / "deck"
    monkeypatch.setattr(D, "DATA", tmp_path / "empty")
    out.mkdir()
    for name in FIGURES:  # figures from an earlier pull must not survive as if current
        (out / f"{name}.png").write_bytes(b"stale")
    assert D.main(["--outdir", str(out)]) == 0
    printed = capsys.readouterr().out
    for name in FIGURES:
        assert f"{name}: skipped (no data yet)" in printed
        assert not (out / f"{name}.png").exists()
    assert json.loads((out / "deck_numbers.json").read_text()) == {"fine_tuning_readout": None}
