"""experiments/EVAL/check_test_disjoint.py on synthetic inputs: the jet key, the overlap count, the file-list and
source-role checks, and a whole run on a toy layout (one overlapping set must FAIL, the control must PASS)."""
import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("check_test_disjoint", ROOT / "experiments/EVAL/check_test_disjoint.py")
CT = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CT)


def _cols(n, seed):
    r = np.random.default_rng(seed)
    return r.integers(0, 188, n), {c: r.normal(300, 50, n).astype(np.float32) for c in CT.FLOATS}


def test_key_is_exact_and_overlap_counts_rows():
    lab, cols = _cols(1000, 1)
    k = CT.jet_keys(lab, cols)
    assert np.unique(k).size == 1000
    lab2, cols2 = _cols(500, 2)
    b = CT.jet_keys(np.r_[lab2, lab[:7]], {c: np.r_[cols2[c], cols[c][:7]] for c in CT.FLOATS})
    assert CT.overlap(k, b) == 7 and CT.overlap(b, k) == 7
    # one bit of one column, or the label, makes a different jet; float64 input is the same float32 jet
    c1 = {c: v.copy() for c, v in cols.items()}
    c1["genjet_sdmass"][0] = np.nextafter(c1["genjet_sdmass"][0], np.float32(np.inf))
    assert CT.overlap(CT.jet_keys(lab, c1), k) == 999
    assert CT.overlap(CT.jet_keys((lab + 1) % 188, cols), k) < 20
    assert CT.overlap(CT.jet_keys(lab, {c: v.astype(np.float64) for c, v in cols.items()}), k) == 1000


def test_file_checks_and_source_roles():
    census = {"ft_heldout": ["/jc2/jet_data/QCD_0285.parquet", "/jc2/jet_data/QCD_0286.parquet"],
              "ft_test": ["/jc2/jet_data/QCD_0350.parquet", "/jc2/jet_data/QCD_0351.parquet"]}
    good = ["/jc2/jet_data/QCD_0351.parquet", "/jc2/jet_data/QCD_0350.parquet"]
    f = CT.file_checks([good, list(good)], census)
    assert f["specs_identical"] and f["spec_equals_census_ft_test"] and f["census_roles_disjoint"]
    assert f["spec_files_in_ft_heldout"] == 0
    bad = CT.file_checks([good + ["/jc2/jet_data/QCD_0286.parquet"], good], census)
    assert not bad["specs_identical"] and not bad["spec_equals_census_ft_test"] and bad["spec_files_in_ft_heldout"] == 1
    src = np.array([0 * CT.SRC_MULT + 5, 1 * CT.SRC_MULT, 2 * CT.SRC_MULT + 9, 4 * CT.SRC_MULT])
    assert CT.src_roles(src, census) == {"ft_heldout": 2, "ft_test": 1, "outside": 1}
    assert CT.data_test("x --data-test /a/QCD_0350.parquet /a/QCD_0351.parquet \\\n --out y") == \
        ["/a/QCD_0350.parquet", "/a/QCD_0351.parquet"]


def _write(path, lab, cols, src):
    pq.write_table(pa.table({"jet_label": pa.array(lab, pa.int32()), **{c: pa.array(v) for c, v in cols.items()},
                             "jc2bal_src": pa.array(src, pa.int64())}), path)


def test_census_against_jc2_roles():
    ms = CT.REPO / "experiments/FT/make_subsets.py"
    spec = importlib.util.spec_from_file_location("ms_for_disjoint", ms)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    census = {r: m.jc2_role_files(r) for r in ("ft_heldout", "ft_test")}
    assert CT.repo_roles_check(census) == {"ft_heldout": True, "ft_test": True}
    assert CT.repo_roles_check({**census, "ft_test": census["ft_test"][:-1]})["ft_test"] is False


def test_whole_run(tmp_path, monkeypatch):
    monkeypatch.setattr(CT, "REPO", tmp_path)          # no make_subsets.py: the JC2_ROLES comparison is skipped
    monkeypatch.setattr(CT, "N_TEST2M", 300)
    held = [f"/jc2/jet_data/QCD_{i:04d}.parquet" for i in (285, 286)]
    ftest = [f"/jc2/jet_data/QCD_{i:04d}.parquet" for i in (350, 351)]
    bal, head, ftr = tmp_path / "bal", tmp_path / "head", tmp_path / "ft"
    for d in (bal, head):
        d.mkdir()
    (bal / "census.json").write_text(json.dumps({"files": {"ft_heldout": held, "ft_test": ftest}}))
    spec = tmp_path / "spec.yaml"
    spec.write_text("--data-test " + " ".join(ftest) + " \\\n --out x\n")
    lab, cols = _cols(300, 3)
    np.savez(head / "head_outputs.npz", label188=lab.astype(np.int16), **cols)
    tlab, tcols = _cols(200, 4)
    _write(bal / "train_N100_s1.parquet", tlab[:100], {c: v[:100] for c, v in tcols.items()}, np.arange(100))
    _write(bal / "val.parquet", tlab[100:], {c: v[100:] for c, v in tcols.items()}, CT.SRC_MULT + np.arange(100))
    _write(bal / "test.parquet", lab[:50], {c: v[:50] for c, v in cols.items()}, 2 * CT.SRC_MULT + np.arange(50))
    names = ("train_N100_s1.parquet", "val.parquet", "test.parquet")
    (bal / "manifest.json").write_text(json.dumps({"sha256": {n: CT.sha256(bal / n) for n in names}}))
    cell = ftr / "budget" / "leg_jc2b" / "l188-s1" / "N100" / "s1"
    cell.mkdir(parents=True)
    (cell / "ft_manifest.json").write_text(json.dumps({
        "subset": str(bal / names[0]), "val": str(bal / names[1]),
        "subset_sha256": CT.sha256(bal / names[0]), "val_sha256": CT.sha256(bal / names[1])}))
    hd = tmp_path / "fthead" / "mtx-l188-s1"
    hd.mkdir(parents=True)
    sha = __import__("hashlib").sha256(lab.astype(np.int16).tobytes()).hexdigest()
    (head / "manifest.json").write_text(json.dumps({"n_test_files": 2, "head_outputs": {"label188_sha256": sha}}))
    (hd / "manifest.json").write_text(json.dumps({"n_test_files": 2, "run_dir": str(cell),
                                                  "head_outputs": {"n": 300, "label188_sha256": sha}}))
    args = ["--spec", str(spec), "--test2m", str(head), "--bal", str(bal), "--ft-root", str(ftr),
            "--ft-head-root", str(tmp_path / "fthead"), "--out", str(tmp_path / "out.json")]
    assert CT.main(args) == 0
    r = json.loads((tmp_path / "out.json").read_text())
    assert r["verdict"] == "PASS" and r["sets"]["test.parquet"]["overlap_with_test2m"] == 50
    assert r["sets"]["val.parquet"]["overlap_with_test2m"] == 0 and r["n_cells"] == 1
    # a head output from a cell the check did not read fails
    stray = tmp_path / "fthead" / "mtx-other"
    stray.mkdir()
    (stray / "manifest.json").write_text(json.dumps({"n_test_files": 2, "run_dir": str(ftr / "elsewhere" / "s1"),
                                                     "head_outputs": {"n": 300, "label188_sha256": sha}}))
    assert CT.main(args) == 1 and "mtx-other" in json.loads((tmp_path / "out.json").read_text())["fail"][0]
    (stray / "manifest.json").unlink()
    stray.rmdir()
    # a validation set holding three TEST2M jets fails
    _write(bal / "val.parquet", np.r_[tlab[100:], lab[:3]], {c: np.r_[v[100:], cols[c][:3]] for c, v in tcols.items()},
           CT.SRC_MULT + np.arange(103))
    sv = CT.sha256(bal / "val.parquet")
    (bal / "manifest.json").write_text(json.dumps({"sha256": {**{n: CT.sha256(bal / n) for n in names}}}))
    (cell / "ft_manifest.json").write_text(json.dumps({
        "subset": str(bal / names[0]), "val": str(bal / names[1]), "subset_sha256": CT.sha256(bal / names[0]),
        "val_sha256": sv}))
    assert CT.main(args) == 1
    r = json.loads((tmp_path / "out.json").read_text())
    assert r["verdict"] == "FAIL" and r["sets"]["val.parquet"]["overlap_with_test2m"] == 3 and r["fail"] == ["val.parquet"]
