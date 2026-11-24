"""scripts/build_sophon_jobs.py (PI 2026-11-23, LEDGER SH-13): the released Sophon checkpoint's pair and X->bx inference
and transfer-head fits run as our networks' did, with its pinned checkpoint, in one GPU pod."""
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_sophon_jobs", REPO / "scripts/build_sophon_jobs.py")
BS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BS)


def test_the_full_pod_extracts_pairs_and_xbx_then_fits_both_signals():
    text = BS.x_spec("mtx-s9.99", "24Gi")
    assert "name: bs3-sophon-x-raunav" in text and 'nvidia.com/gpu: "1"' in text and "--branch mtx-s9.99 " in text
    assert BS.JB.SOPHON_SHA256 in text and BS.JB.SOPHON_URL in text
    assert f"pairs {BS.PAIRS_OUT}/train 0 {BS.JB.BAL}/pairs_train.parquet" in text
    assert all(f"{BS.JB.BAL}/{p}" in text for p in BS.JB.PAIRS_TEST_PARTS)
    for s in ("xbb", "xbs"):
        assert f"xbx {s}/train $(ls {BS.XB.STAGE}/pairs/{s}/train_*.parquet | sort)" in text
        assert f"--signals {s} --checkpoint released --window 150 230 --mx 200 --out {BS.XB.FITS_IHEP}/{s}" in text
        assert f"--linear-only --grid v5 --out {BS.R5.XBX_V5}/{s}" in text
    assert "xbx qcd/eval" in text and "--arm L188" in text and "--data-config configs/xbx/JetClassII_xbx.yaml" in text
    assert text.index("xbx qcd/eval") < text.index("fit_xbx.py")     # the fits read the checkpoint the pod holds
    assert '-le 88 ]' in text and "ConfigMap" not in text and "CronJob" not in text


def test_the_smoke_extracts_a_slice_into_its_own_folder():
    text = BS.smoke_spec("mtx-s9.99", "24Gi")
    assert "name: bs3-sophon-x-smoke-raunav" in text and f"{BS.SMOKE}/jc2bal_pairs_test {BS.SMOKE_JETS} " in text
    assert "fit_xbx" not in text and "extract_xbx" not in text and BS.JB.SOPHON_SHA256 in text


def test_the_two_parts_split_the_full_pod():
    x, pr = BS.x_spec("mtx-s9.99", "16Gi", "xbx"), BS.x_spec("mtx-s9.99", "16Gi", "pairs")
    assert "name: bs3-sophon-xbx-raunav" in x and "fit_xbx.py" in x and "jc2bal_pairs" not in x
    assert "name: bs3-sophon-pairs-raunav" in pr and "fit_xbx" not in pr and "extract_xbx" not in pr
    assert "declare -f pairs > /tmp/sophon_fns.sh" in pr and pr.count("PEAK_RSS_MIB") == 2
    assert f"pairs {BS.PAIRS_OUT}/train 0 {BS.JB.BAL}/pairs_train.parquet" in pr
    assert all(f"{BS.JB.BAL}/{p}" in pr for p in BS.JB.PAIRS_TEST_PARTS)
    for t in (x, pr):
        assert BS.JB.SOPHON_SHA256 in t and 'nvidia.com/gpu: "1"' in t and "nvidia.com/a100" not in t


def test_the_pair_pod_is_kept_near_the_volume():
    pr, x = BS.x_spec("mtx-s9.99", "64Gi", "pairs"), BS.x_spec("mtx-s9.99", "16Gi", "xbx")
    assert '"ucm-fiona01.ucmerced.edu"' in pr and pr.count("operator: In") >= 2 and "operator: NotIn" in pr
    assert '"ucm-fiona01.ucmerced.edu"' not in x
    assert not set(BS.NEAR) & {"node-2-3.sdsc.optiputer.net", "k8s-haosu-10.sdsc.optiputer.net", "ry-gpu-01.sdsc.optiputer.net"}


def test_the_cpu_parts_run_on_their_familys_cpu_model_with_its_threads():
    want = {"mass": "AMD EPYC 7742 64-Core Processor", "pairfit": "AMD EPYC 7742 64-Core Processor",
            "recall": "AMD EPYC 7742 64-Core Processor", "mlp": "AMD EPYC 7402P 24-Core Processor",
            "mlp2": "AMD EPYC 7402P 24-Core Processor"}
    for part, cpu in want.items():
        text = BS.cpu_spec("mtx-s9.99", part, BS.CPU_MEM[part])
        assert f"name: bs3-sophon-{part}-raunav" in text and f'[ "${{CPU}}" = "{cpu}" ]' in text
        assert "nvidia.com/gpu" not in text and "probe_sophon.py" in text and "-le 88 ]" in text
    mass = BS.cpu_spec("mtx-s9.99", "mass", "8Gi")
    assert "probe_sophon.py massview" in mass and "atlas_resolution.py --model sophon-released" in mass
    assert "--grid v2 --centre global --mlp-max-epochs 200" in mass and "--mass-v3" not in mass and "--head" not in mass
    assert mass.index("massview") < mass.index("atlas_resolution.py")
    for part in ("mlp2", "pairfit"):
        assert all(f in BS.cpu_spec("mtx-s9.99", part, "8Gi") for f in BS.PAIRS_DONE)
    assert "--datasets jc1 top qg" in BS.cpu_spec("mtx-s9.99", "mlp", "7Gi")
