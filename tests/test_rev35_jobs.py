"""experiments/STATS/rev35_jobs.py and its job (ledger 35: audit A1, V7)."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("rev35_jobs", "experiments/STATS/rev35_jobs.py")
BJ = _load("build_mlp_scores_jobs", "scripts/build_mlp_scores_jobs.py")


def test_overlap_counts_pairs_merged_by_both():
    tgt = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}          # merges (0,1) (0,2) (1,2) (3,4)
    draw = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}         # merges (0,1) (2,3)
    assert R._overlap(draw, tgt) == (4, 2, 1)


def test_the_confusion_runs_are_the_188_class_and_untrained_probes():
    assert [m for m, _ in R.RUNS["l188"]] == ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"]
    assert R.RUNS["init_bn"] == [("init-s1", "init_bn"), ("init-s2", "init_bn"), ("init-s3", "init_bn")]


def test_the_rev35_job_has_two_cpu_pods():
    rows = BJ.rev35_rows()
    assert len(rows) == 2 and "rev35_jobs.py confusion" in rows[0] and "rev35_jobs.py pool" in rows[1]
    assert "numba==" in rows[1]
    text = BJ.rev35_spec("mtx-s9.99", "8Gi")
    assert "name: bs3-rev35-raunav" in text and "nvidia.com/gpu" not in text
