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
