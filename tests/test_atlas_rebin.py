"""The finer pT bins of the ATLAS resolution (experiments/EVAL/atlas_rebin.py)."""
import importlib.util
import json
import hashlib
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("atlas_rebin", ROOT / "experiments/EVAL/atlas_rebin.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
AR = rb.AR


def _perjet(n=6000, seed=0):
    r = np.random.default_rng(seed)
    gen = r.uniform(20, 500, n)
    resp = {"groomed": r.lognormal(0, 0.2, n), "mlp": r.lognormal(0, 0.1, n)}
    return {"rows": np.arange(n), "label188": r.integers(0, 188, n), "jet_pt": r.uniform(200, 2500, n),
            "genjet_sdmass": gen, **{f"m_{k}": gen * v for k, v in resp.items()}}, resp


def test_at_least_eight_pt_bins_over_the_selection():
    assert len(rb.PT_EDGES) - 1 >= 8 and rb.PT_EDGES[0] == 200 and rb.PT_EDGES[-1] == 2500
    assert all(a < b for a, b in zip(rb.PT_EDGES, rb.PT_EDGES[1:]))


def test_the_rebinned_file_keeps_the_inclusive_and_mass_bins_and_refuses_other_jets(tmp_path):
    pj, resp = _perjet()
    npz = tmp_path / "m_perjet.npz"
    np.savez_compressed(npz, **pj)
    AR.PT_EDGES = [200, 400, 700, 1000, 2500]
    est = {k: AR.response_block(v, pj["label188"], pj["jet_pt"], pj["genjet_sdmass"]) for k, v in resp.items()}
    src = tmp_path / "in"
    src.mkdir()
    (src / "m.json").write_text(json.dumps({"model": "m", "estimators": est,
                                            "source": {"file": str(npz), "sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}}))
    assert rb.main(["--in-dir", str(src), "--out-dir", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out" / "m.json").read_text())
    for k in resp:
        assert doc["estimators"][k]["all"]["vs_pt"]["edges"] == rb.PT_EDGES
        assert rb.same(doc["estimators"][k]["qcd"]["inclusive"], est[k]["qcd"]["inclusive"])
        assert sum(doc["estimators"][k]["all"]["vs_pt"]["n"]) == pj["jet_pt"].size
    assert doc["rebinned"]["pt_edges"] == rb.PT_EDGES
    # a per-jet file that is not the source's stops the script
    np.savez_compressed(npz, **_perjet(seed=1)[0])
    with pytest.raises(SystemExit):
        rb.main(["--in-dir", str(src), "--out-dir", str(tmp_path / "out2")])


def test_the_rebin_job_is_one_cpu_pod_into_a_new_folder():
    import yaml
    spec = importlib.util.spec_from_file_location("build_atlas_res_job", ROOT / "scripts/build_atlas_res_job.py")
    BA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BA)
    d = yaml.safe_load(BA.rebin_spec("t"))
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert d["metadata"]["name"] == "atlas-rebin-raunav" and d["spec"]["backoffLimit"] == 1
    assert "completionMode" not in d["spec"] and "nvidia.com/gpu" not in c["resources"]["limits"]
    assert "--in-dir /data/results/eval/atlas_res --out-dir /data/results/eval/atlas_res_fine" in c["args"][0]
