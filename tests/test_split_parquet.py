"""experiments/FT/split_parquet.py and the balanced-set extraction that reads its parts."""
import importlib.util
import pathlib
import subprocess

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
yaml = pytest.importorskip("yaml")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


sp = _load("split_parquet", "experiments/FT/split_parquet.py")


def test_parts_hold_every_row_in_order_and_the_source_is_kept(tmp_path):
    src = tmp_path / "pairs_test.parquet"
    rng = np.random.default_rng(0)
    t = pa.table({"jc2bal_src": rng.permutation(1_000).astype(np.int64), "x": rng.normal(size=1_000)})
    pq.write_table(t, src, row_group_size=70)                    # 15 row groups
    out = sp.split(src, 3)
    assert [p.name for p in out] == [f"pairs_test_part{i}.parquet" for i in range(3)]
    assert src.exists() and not list(tmp_path.glob("*.tmp"))
    back = pa.concat_tables([pq.read_table(p) for p in out])
    assert back.equals(t)
    assert all(pq.ParquetFile(p).metadata.num_rows < 1_000 for p in out)


def test_extraction_reads_the_parts_in_order_and_skips_finished_sets():
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    script = yaml.safe_load(bj.extract_spec("t", 5)[1])["spec"]["template"]["spec"]["containers"][0]["args"][0]
    parts = " ".join(f"{bj.BAL}/pairs_test_part{i}.parquet" for i in range(3))
    assert f'extract jc2bal_pairs test "{parts}"' in script
    assert "pairs_test.parquet" not in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    # the skip logic, run on its own: variant sets need every checkpoint.variant manifest
    fn = script[script.index("extract () {"):script.index("for N in")]
    probe = "python3 () { echo RUN; }\n" + fn + """
CKPTS="best70"; VARIANTS="mtop mqg"; MODEL=m
mkdir -p $D/m/a/s/best70 $D/m/a/s/best70.mtop; touch $D/m/a/s/best70/manifest.json $D/m/a/s/best70.mtop/manifest.json
extract a s f --variants x
extract a s f
"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        probe = probe.replace(bj.LP_BAL, d).replace("$D", d)
        r = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0].startswith("RUN") and r.stdout.splitlines()[1].endswith("done")


def test_split_job_writes_the_parts_the_extraction_requires():
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    script = yaml.safe_load(bj.split_spec("t")[1])["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert f"split_parquet.py {bj.BAL}/pairs_test.parquet --parts {len(bj.PAIRS_TEST_PARTS)}" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_extraction_and_probe_jobs_take_any_gpu_in_us_west_minus_bad_nodes():
    # the extraction is loader-bound (an L40 at 0%, 2026-11-01), so no GPU product is required
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    for _, text in (bj.extract_spec("t", 5), bj.fit_spec("t", 5, False), bj.kin_spec("t")):
        pod = yaml.safe_load(text)["spec"]["template"]["spec"]
        exprs = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        keys = {e["key"]: e for e in exprs}
        assert "nvidia.com/gpu.product" not in keys and keys["topology.kubernetes.io/region"]["values"] == ["us-west"]
        assert keys["kubernetes.io/hostname"]["operator"] == "NotIn" and keys["kubernetes.io/hostname"]["values"]
        assert pod["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "1"
