"""experiments/EVAL/sculpting_ft.py: mass sculpting of the fine-tuned JetClass-II cells on the natural-composition
TEST2M jets of the pretraining heads' sculpting, in sculpting_v3's schema (plan 2026-11-08, step 2)."""
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SF = _load("sculpting_ft", "experiments/EVAL/sculpting_ft.py")
MS, RP = SF.MS, SF.RP


def _jets(n=6_000, seed=3):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 188, n).astype(np.int16)
    msd = rng.uniform(20, 500, n).astype(np.float32)
    z = rng.normal(size=(n, 188))
    z[:, 161:] += (msd[:, None] / 250.0) * (lab[:, None] >= 161)     # the QCD score follows the mass
    return lab, msd, z.astype(np.float16)


def _head(d: pathlib.Path, lab, msd, z, ckpt_sha, rung="L188", cell="c"):
    d.mkdir(parents=True)
    np.savez(d / MS.HEAD_FILE, rows=np.arange(lab.size), label188=lab, logits=z, jet_sdmass=msd,
             jet_pt=np.full(lab.size, 300, np.float32))
    sha = hashlib.sha256((d / MS.HEAD_FILE).read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({
        "rung": rung, "num_classes": z.shape[1], "num_reg": 0, "run_dir": str(cell), "tag": "net_best_epoch_state",
        "checkpoint_sha256": ckpt_sha, "head_outputs": {"n": int(lab.size), "sha256": sha}}))


def _setup(tmp_path, tree="budget"):
    lab, msd, z = _jets()
    cell = tmp_path / tree / "leg_jc2b" / "l188-s1" / "N1000000" / "s1"
    cell.mkdir(parents=True)
    (cell / RP.FT_CKPT).write_bytes(b"fine-tuned state")
    (cell / "DONE").touch()
    _head(tmp_path / "heads" / "mtx-l188-s1", lab, msd, z, hashlib.sha256(b"fine-tuned state").hexdigest(), cell=cell)
    _head(tmp_path / "ref", lab, msd, z[:, :17], "pretrained", rung="R16_Q1")
    args = ["--tree", str(cell.parents[2]), "--inits", "l188-s1", "--head-root", str(tmp_path / "heads"),
            "--ref", str(tmp_path / "ref"), "--out-dir", str(tmp_path / "out")]
    return lab, msd, z, cell, args


def test_the_document_is_mass_sculptings_on_the_natural_jets_with_the_cell_beside_it(tmp_path):
    lab, msd, z, cell, args = _setup(tmp_path, "budget_long")
    assert SF.main(args) == 0
    doc = json.loads((tmp_path / "out" / "mtx-l188-s1.json").read_text())
    ref = MS.head_doc(tmp_path / "heads" / "mtx-l188-s1")
    for k, v in ref.items():
        if k != "script_sha256":
            assert doc[k] == v, k
    assert doc["mass_sculpting_sha256"] == ref["script_sha256"] and doc["script"] == "experiments/EVAL/sculpting_ft.py"
    assert doc["qcd_nodes"] == list(range(161, 188)) and doc["label_set"] == "L188"
    assert (doc["model"], doc["cell"], doc["source_tree"], doc["method"]) == ("mtx-l188-s1", str(cell), "budget_long",
                                                                            "fine-tuning")
    q = lab >= 161
    s = MS.score(z, range(161, 188))
    want = MS.sculpting(s[q], msd[q].astype(np.float64), s[(lab < 161)])
    assert doc["working_points"] == json.loads(json.dumps(want["working_points"]))
    assert doc["sig_eff_50"]["jsd"] == want["sig_eff_50"]["jsd"] and doc["n_qcd"] == int(q.sum())
    before = (tmp_path / "out" / "mtx-l188-s1.json").read_bytes()
    SF.main(args)                                                     # exists: skipped, never overwritten
    assert (tmp_path / "out" / "mtx-l188-s1.json").read_bytes() == before


def test_outputs_of_another_checkpoint_or_other_jets_are_refused(tmp_path):
    lab, msd, z, cell, args = _setup(tmp_path)
    (cell / RP.FT_CKPT).write_bytes(b"another state")
    RP.sha256.cache_clear()
    with pytest.raises(SystemExit, match="is not the head output of"):
        SF.main(args)
    (cell / RP.FT_CKPT).write_bytes(b"fine-tuned state")
    RP.sha256.cache_clear()
    (tmp_path / "ref" / MS.HEAD_FILE).unlink()
    np.savez(tmp_path / "ref" / MS.HEAD_FILE, label188=lab, jet_sdmass=msd[::-1].copy())
    with pytest.raises(SystemExit, match="not the jets of"):
        SF.main(args)
    assert not (tmp_path / "out").exists()


def test_the_job_scores_every_cell_of_the_head_job_into_one_directory():
    GD = _load("build_gallery_data_job_sf", "scripts/build_gallery_data_job.py")
    HB = _load("build_head_jobs_sf", "scripts/build_head_jobs.py")
    lines = [ln.strip() for ln in GD.calls_ft_pairs_sculpt().splitlines()]
    calls = [ln for ln in lines if "sculpting_ft.py" in ln]
    got = []
    for ln in calls:
        a = ln.split()
        tree, inits = a[a.index("--tree") + 1], a[a.index("--inits") + 1:a.index("--head-root")]
        assert a[a.index("--head-root") + 1] == HB.FT_HEAD_ROOT and a[-2:] == ["--out-dir", GD.SCULPT_FT]
        got += [f"{tree}/{i}" for i in inits]
    assert sorted(got) == sorted({c.rsplit("/", 2)[0] for _m, c in HB.ft_head_cells()})
    assert lines.index(calls[0]) > lines.index(f"echo done > {GD.ROC}/pairs_ft/DONE")
    # DONE after the last call, so the pull never takes a part of the directory
    assert lines[-1] == f"echo done > {GD.SCULPT_FT}/DONE" and lines.index(calls[-1]) == len(lines) - 2
    pull = (ROOT / "experiments/FIGS/pull_mlst_data.py").read_text()
    assert 'if (sft / "DONE").exists():' in pull and pull.count('"sculpting_ft"') == 1
