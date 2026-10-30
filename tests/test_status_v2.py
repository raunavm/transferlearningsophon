"""experiments/MTX/status_v2.py and its CronJob status-raunav (scripts/build_curve_jobs.py)."""
import importlib.util
import json
import os
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load("status_v2", "experiments/MTX/status_v2.py")


def _done(tree, rel, t):
    p = tree / rel / "DONE"
    p.parent.mkdir(parents=True)
    p.touch()
    os.utime(p, (t, t))


def test_cells_are_counted_by_their_done_and_interrupted_attempts_are_not(tmp_path):
    tree = tmp_path / "leg2"
    _done(tree, "l188-s1/N1000/s1", 1_700_000_000)
    _done(tree, "l188-s1/N10000/s1", 1_700_000_100)
    _done(tree, "l188-s1/N100000/s1.partial.5", 1_800_000_000)
    (tree / "l188-s1/N1000000/s1").mkdir(parents=True)           # running: no DONE
    n, newest = S.count(tree, S.CELL)
    assert n == 2 and newest == 1_700_000_100


def test_status_reports_counts_against_expected_and_a_missing_directory_as_zero(tmp_path):
    fits = tmp_path / "fits_v3"
    fits.mkdir()
    for m in ("mtx-l188-s1", "init-s1"):
        (fits / f"{m}.json").write_text("{}")
    (fits / "mtx-l162-s1.json.tmp").write_text("{")             # being written
    d = S.status({"probe": (str(fits), "*.json", 2), "later": (str(tmp_path / "nope"), S.CELL, 12)},
                 data=str(tmp_path))
    e = d["entries"]
    assert e["probe"]["done"] == 2 and e["probe"]["complete"] and e["probe"]["newest"].endswith("Z")
    assert e["later"] == {"done": 0, "expected": 12, "complete": False, "newest": None,
                          "path": f"{tmp_path / 'nope'}/{S.CELL}"}
    assert 0 <= d["data_used_pct"] <= 100


def test_the_status_file_is_written_whole(tmp_path, monkeypatch):
    real = S.status
    monkeypatch.setattr(S, "status", lambda: real({"x": (str(tmp_path / "none"), "*.json", 1)}, str(tmp_path)))
    out = tmp_path / "status" / "status.json"
    S.main(["--out", str(out)])
    S.main(["--out", str(out)])                                  # every half hour, replaced
    assert json.loads(out.read_text())["entries"]["x"]["done"] == 0
    assert [p.name for p in out.parent.iterdir()] == ["status.json"]


def test_every_expected_total():
    e = S.ENTRIES
    assert all(e[f"ft best70/{leg}"][2] == 27 * 4 for leg in ("leg1", "leg2", "leg_top", "leg_qg"))
    assert e["ft best70/leg1_188"][2] == 16 * 4 and e["ft ref_v100/jc2/leg1_188"][2] == 3 * 4
    assert e["probe fits_v3"][2] == e["xbx fits_ihep"][2] == 30


BC = _load("build_curve_jobs_status", "scripts/build_curve_jobs.py")


def test_the_cronjob_clones_the_tag_runs_the_status_and_nothing_else():
    text = BC.status_spec(BC.PIN)
    assert (BC.STATUS_DIR / "cronjob-status-raunav.yaml").read_text() == text
    d = yaml.safe_load(text)
    assert d["kind"] == "CronJob" and d["metadata"]["name"] == "status-raunav"
    assert d["spec"]["schedule"] == "*/30 * * * *" and d["spec"]["concurrencyPolicy"] == "Forbid"
    pod = d["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert {e["name"]: e["value"] for e in c["env"]} == {"REPO_REF": BC.PIN}
    assert "git clone --depth 1 --branch \"${REPO_REF}\"" in c["args"][0]
    assert "python3 experiments/MTX/status_v2.py --out /data/results/status/status.json" in c["args"][0]
    assert "nvidia.com/gpu" not in c["resources"]["limits"]
    low = text.lower()
    assert "kubectl" not in low and "configmap" not in low and "secret" not in low
    assert "serviceaccount" not in low
    backup = yaml.safe_load(BC.BF._mtx_launch().v2_backup_spec(BC.PIN)[1])
    assert c["image"] == backup["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["image"]
    assert pod["volumes"] == backup["spec"]["jobTemplate"]["spec"]["template"]["spec"]["volumes"]
