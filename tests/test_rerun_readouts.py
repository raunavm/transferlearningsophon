"""The rerun's read-outs behind the appendix figures (PI 2026-11-15/16): the jobs that read the one rule's fine-tuned
cells at 1e6 (ft_v2/fixed) or the v5 linear probes refuse to start (exit 42) until every input they read is finished,
write only new folders and read no superseded tree; the random partitions pooled in the sculpting bootstrap, and the
17-class partitions in the mass and sculpting tables."""
import importlib.util
import json
import math
import pathlib
import subprocess

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PT = _load("build_perf_pt_jobs_rr", "scripts/build_perf_pt_jobs.py")
BL = _load("build_boot_l3_jobs_rr", "scripts/build_boot_l3_jobs.py")
GD = _load("build_gallery_data_job_rr", "scripts/build_gallery_data_job.py")
SB = _load("sculpting_boot_rr", "experiments/EVAL/sculpting_boot.py")
FIXED = "/data/results/ft_v2/fixed"
# the trees and probe fits the rerun replaced: none of them may be read by a rerun job
SUPERSEDED = ("budget_long", "/budget/", "_fulldecay", "budget_prefull", "/best70/leg_jc2b", "/best70/leg2", "fits_v4")


def _script(text: str) -> str:
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _guards(script: str, marker: str = "exit 42") -> list:
    """The refusal lines of a script that loop over inputs or test one file."""
    return [ln.strip() for ln in script.splitlines()
            if marker in ln and ln.strip().startswith(("for c in", "for C in", "for f in", "[ -f /data", "[ -e /data"))]


def _run(lines: list, tmp: pathlib.Path) -> subprocess.CompletedProcess:
    """The lines under bash -euo pipefail with /data/results moved to tmp/results."""
    body = "\n".join(ln.replace("/data/results", f"{tmp}/results") for ln in lines)
    return subprocess.run(["bash", "-c", "set -euo pipefail\n" + body + "\necho PASSED"], capture_output=True, text=True)


def _touch(tmp: pathlib.Path, path: str) -> None:
    """An empty file at path (under /data/results) moved to tmp/results, as _run moves the guards."""
    f = pathlib.Path(str(tmp) + path.removeprefix("/data"))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("")


# ------------------------------------------------------------------ performance against jet pT (PT)

def test_the_one_rule_perf_pt_job_reads_all_60_cells_into_new_folders():
    s = _script(PT.spec("TAG_TBD", ["jc2fixed", "jc1fixed"], line=90))
    cells = PT.fixed_cells("jc2b") + PT.fixed_cells("jc1")
    assert len(cells) == len(set(cells)) == 60 and all(c.startswith(f"{FIXED}/leg") for c in cells)
    assert sum("/scratch/N1000000/s" in c for c in cells) == 6
    guards = _guards(s)
    assert len(guards) == 2 and all(c in " ".join(guards) for c in cells)
    assert s.index(guards[-1]) < s.index("perf_vs_pt.py")              # nothing runs before every cell is DONE
    assert (f"perf_vs_pt.py jc2ft --out-dir {PT.OUT_FIXED}/jc2_ft --trees {FIXED}/leg_jc2b --scratch-tree "
            f"{FIXED}/leg_jc2b --probe-dir {PT.OUT_V5}/jc2") in s
    assert f"perf_vs_pt.py jc1 --tree {FIXED}/leg2 --out-dir {PT.OUT_FIXED}/jc1 --confusion-out {PT.CONF_FIXED}" in s
    assert s.index("perf_vs_pt.py jc2ft") < s.index(f"touch {PT.OUT_FIXED}/jc2_ft/DONE")
    assert s.index("perf_vs_pt.py jc1 ") < s.index(f"touch {PT.OUT_FIXED}/jc1/DONE {PT.CONF_FIXED}/DONE")
    assert not any(t in s for t in SUPERSEDED) and "configmap" not in s.lower()
    assert len({PT.OUT_FIXED, PT.OUT_V5, PT.OUT_ROOT, PT.OUT_V3}) == 4 and PT.CONF_FIXED not in (PT.CONF_ROOT, PT.CONF_FULL)
    d = yaml.safe_load(PT.spec("TAG_TBD", ["jc2fixed", "jc1fixed"], line=90))
    assert d["metadata"]["name"] == "perf-pt-jc2fixed-jc1fixed-raunav" and d["spec"]["backoffLimit"] == 1
    assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
    assert d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] == PT.MEM_RERUN


def test_the_one_rule_perf_pt_job_refuses_while_one_cell_is_unfinished(tmp_path):
    guards = _guards(_script(PT.spec("TAG_TBD", ["jc2fixed", "jc1fixed"], line=90)))
    cells = PT.fixed_cells("jc2b") + PT.fixed_cells("jc1")
    for c in cells[:-1]:
        _touch(tmp_path, f"{c}/DONE")
    r = _run(guards, tmp_path)
    assert r.returncode == 42 and "not DONE" in r.stdout and cells[-1].removeprefix("/data") in r.stdout
    assert "PASSED" not in r.stdout
    _touch(tmp_path, f"{cells[-1]}/DONE")
    assert _run(guards, tmp_path).returncode == 0


def test_the_v5_probe_perf_pt_job_waits_for_every_v5_fit(tmp_path):
    s = _script(PT.spec("TAG_TBD", ["jc2v5", "jc1probev5"], line=90))
    fits = [f"{PT.FITS_V5}/{m}.json" for m in PT.v5_models(True)]
    assert len(fits) == 30 and f"{PT.FITS_V5}/init-s3.json" in fits
    assert f"perf_vs_pt.py jc2 --out-dir {PT.OUT_V5}/jc2 --probs-root {PT.PROBS_V5}" in s
    assert f"perf_vs_pt.py jc1probe --out-dir {PT.OUT_V5}/jc1_probe --probs-root {PT.PROBS_V5} --test-files" in s
    assert not any(t in s for t in SUPERSEDED)
    guards = _guards(s)
    assert len(guards) == 1 and s.index(guards[0]) < s.index("perf_vs_pt.py")
    for f in fits[1:]:
        _touch(tmp_path, f)
    r = _run(guards, tmp_path)
    assert r.returncode == 42 and "the v5 fit is not done" in r.stdout
    _touch(tmp_path, fits[0])
    assert _run(guards, tmp_path).returncode == 0


def test_the_rerun_parts_are_never_mixed_with_the_older_ones():
    with pytest.raises(SystemExit, match="go in a job of their own"):
        PT.main(["--tag", "TAG_TBD", "--pin-not-yet-tagged", "--parts", "jc2", "jc2fixed"])


# ------------------------------------------------------------------ the decision-L3 bootstraps

def test_the_fine_tuned_l3_parts_read_the_one_rule_cells_and_refuse_until_they_and_the_read_out_exist(tmp_path):
    ps = dict(BL.ft_rerun())
    assert list(ps) == ["ft-jc2b", "ft-jc1", "jc1-acc"]
    for label, lines in ps.items():
        body = "\n".join(lines)
        assert not any(t in body for t in SUPERSEDED), label
        assert all(w.startswith(BL.OUT_RERUN + "/") for w in body.split() if w.startswith("/data/results/bootstrap/"))
        assert f"{FIXED}/" in body and lines[0].startswith("for C in")
    assert "--task jc2b " in ps["ft-jc2b"][1] and "--task jc1_all " in ps["ft-jc1"][1]
    assert f"jc1_acc_boot.py --root {FIXED} --n 1000000 --metrics {BL.FIXED_METRICS}" in ps["jc1-acc"][2]
    guard = ps["jc1-acc"][:2]
    cells = [f"{FIXED}/leg2/{m.removeprefix('mtx-')}/N1000000/s1" for m in BL.runs()]
    for c in cells[1:]:
        _touch(tmp_path, f"{c}/DONE")
    r = _run(guard, tmp_path)
    assert r.returncode == 42 and "is not done" in r.stdout
    _touch(tmp_path, f"{cells[0]}/DONE")
    r = _run(guard, tmp_path)
    assert r.returncode == 42 and "read out ft_v2/fixed first" in r.stdout
    _touch(tmp_path, BL.FIXED_METRICS)
    assert _run(guard, tmp_path).returncode == 0
    d = yaml.safe_load(BL.spec("ab" * 20, 90, BL.ft_rerun(), "boot-l3-ft-rerun-raunav", BL.MEM_RERUN, BL.OUT_RERUN))
    assert d["metadata"]["name"] == "boot-l3-ft-rerun-raunav" and d["spec"]["completions"] == 3
    assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
    assert d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] == BL.MEM_RERUN
    assert f"mkdir -p {BL.OUT_RERUN}" in _script(yaml.safe_dump(d)) and BL.OUT_RERUN != BL.OUT


def test_the_sculpting_l3_rerun_adds_the_random_partitions_with_their_label_sets():
    HB = _load("build_head_jobs_rr", "scripts/build_head_jobs.py")
    (line,) = BL.sculpt_rerun()
    args = line.split()
    models = args[args.index("--models") + 1:args.index("--label-set")]
    sets = dict(x.split("=") for x in args[args.index("--label-set") + 1:args.index("--replicas")])
    assert models == list(HB.SCULPT_V3) + list(HB.RAND_RUNS) and len(models) == 27
    assert {m: sets[m] for m in HB.RAND_RUNS} == {f"mtx-rand2p{k}-s{s}": f"RAND2_p{k}" for k, s in
                                                  zip(range(1, 6), (1, 2, 1, 2, 2))}
    assert set(sets) == set(HB.FLAV_RUNS) | set(HB.RAND_RUNS)
    assert args[-2:] == ["--out", f"{BL.OUT_RERUN}/sculpting_boot_l3.json"]
    prefix = SB.POOLED["rand-188"][0]
    assert [m for m in models if m.removeprefix("mtx-").startswith(prefix)] == list(HB.RAND_RUNS)


# ------------------------------------------------------------------ the fine-tuned sculpting

def test_the_one_rule_sculpting_refuses_until_every_cell_and_head_output_exists(tmp_path):
    HB = _load("build_head_jobs_rr2", "scripts/build_head_jobs.py")
    fn, name, out = GD.MODES["sculpt-ft-fixed"]
    lines = [ln.strip() for ln in fn().splitlines()]
    assert name == "gallery-data-sculpt-ft-fixed-raunav" and out == GD.SCULPT_FT_FIXED and "sculpt-ft-fixed" in GD.RERUNS
    assert not any(t in " ".join(lines) for t in SUPERSEDED)
    assert lines[1].startswith(f"python3 experiments/EVAL/sculpting_ft.py --tree {FIXED}/leg_jc2b --n 1000000 --inits ")
    assert lines[1].split("--inits ")[1].split(" --head-root")[0].split()[-1] == "scratch"
    assert f"--head-root {HB.FT_HEAD_FIXED_ROOT} --out-dir {GD.SCULPT_FT_FIXED}" in lines[1]
    assert lines[2] == f"echo done > {GD.SCULPT_FT_FIXED}/DONE"
    cells = HB.ft_head_fixed_cells()
    assert len(cells) == 30
    need = [f"{c}/DONE" for _m, c in cells] + [f"{HB.FT_HEAD_FIXED_ROOT}/{m}/{f}" for m, _c in cells
                                               for f in ("manifest.json", "head_outputs.npz")]
    for f in need[:-1]:
        _touch(tmp_path, f)
    r = _run(lines[:1], tmp_path)
    assert r.returncode == 42 and "head_outputs.npz" in r.stdout
    _touch(tmp_path, need[-1])
    assert _run(lines[:1], tmp_path).returncode == 0


# ------------------------------------------------------------------ the random partitions in the sculpting bootstrap

def _stored(runs, msd, lab):
    q, r = (lab >= 161) & (lab < 188), lab < 161
    out = {}
    for m, run in runs.items():
        ref = SB.MS.sculpting(run["s"][q], msd[q], run["s"][r])
        out[m] = {"eps_b=0.01": ref["working_points"]["0.01"]["jsd"], "eps_b=0.1": ref["working_points"]["0.1"]["jsd"],
                  "eps_s=0.5": ref["sig_eff_50"]["jsd"]}
    return out


def _pooled_doc():
    rng = np.random.default_rng(7)
    lab = np.r_[rng.integers(161, 188, 8000), rng.integers(0, 161, 8000)]
    msd = rng.uniform(20, 500, lab.size)
    rand = ["mtx-rand2p1-s1", "mtx-rand2p2-s2", "mtx-rand2p3-s1", "mtx-rand2p4-s2", "mtx-rand2p5-s2"]
    runs = {f"mtx-l188-s{k}": {"s": rng.uniform(size=lab.size) + 0.1 * msd / 500} for k in (1, 2, 3)}
    runs |= {m: {"s": rng.uniform(size=lab.size) + 0.5 * msd / 500} for m in rand}
    nominal, _, boot = SB.bootstrap(runs, msd, lab, 30, 1, _stored(runs, msd, lab))
    return list(runs), rand, nominal, boot


def test_the_random_partitions_are_one_pooled_side_against_188_classes():
    models, rand, nominal, boot = _pooled_doc()
    d = SB.pooled_diffs(models, nominal, boot)
    assert list(d) == ["rand-188"] and d["rand-188"]["pooled"] and d["rand-188"]["pairs"] == []
    assert not any(wp in d["rand-188"] for wp, *_ in SB.WPS)            # decision L3's block only
    u = d["rand-188"]["l3"]["eps_b=0.1"]["distance"]
    a, b = (np.sqrt([nominal[m]["eps_b=0.1"] for m in r]) for r in (rand, models[:3]))
    assert u["runs_config"] == rand and u["runs_reference"] == models[:3]
    assert u["delta"] == pytest.approx(a.mean() - b.mean()) and u["sd_config"] == pytest.approx(np.std(a, ddof=1))
    assert u["sigma"] == pytest.approx(math.sqrt(u["sd_config"] ** 2 + u["sd_reference"] ** 2 + u["boot_sd"] ** 2))
    assert u["delta"] > 0 and u["pass"]
    assert SB.pooled_diffs(models[:3], nominal, boot) == {}             # a call without them has no such row


def test_tab_sculptboot_prints_the_pooled_random_partitions_under_the_partitions():
    M = _load("mlst_results_rr", "experiments/FIGS/mlst_results.py")
    models, rand, nominal, boot = _pooled_doc()
    doc = json.loads(json.dumps({"differences": SB.pooled_diffs(models, nominal, boot)}))
    runs = {"L188": models[:3], **{f"RAND2_p{k}": [m] for k, m in enumerate(rand, 1)}}
    lines = M.table_sculpt_boot(doc, runs).splitlines()
    rows = [ln for ln in lines if ln.startswith(r"\quad " + M.LABEL[M.RAND_POOLED] + " $-$ " + M.LABEL["L188"])]
    assert len(rows) == len(M.SCULPT_BOOT_WPS)
    assert lines.index(rows[0]) - 1 == lines.index(M._full_row(6, M.PARTITIONS) + "*")


# ------------------------------------------------------------------ the 17-class partitions in tab:massdetail

def test_tab_massdetail_holds_the_partitions_probes_between_the_label_sets_and_the_random_network():
    M = _load("mlst_results_rr2", "experiments/FIGS/mlst_results.py")
    g = lambda r, m: {"inclusive": {"resolution": r, "median": m}}       # noqa: E731
    def doc(probe: bool, head: bool = False):
        est = {"groomed": {k: g(0.25, 0.93) for k in ("all", "resonant", "qcd")}}
        if probe:
            est |= {e: {k: g(0.3, 1.1) for k in ("all", "resonant", "qcd")} for e in ("mlp", "ridge")}
        if head:
            est["regressed"] = {k: g(0.12, 1.0) for k in ("all", "resonant", "qcd")}
        return {"estimators": est}
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"], "L162_MASS": ["mtx-l162mass-s1", "mtx-l162mass-s2"],
            "FLAV_F0": ["mtx-flavf0-s1", "mtx-flavf0-s2"], "RAND2_p1": ["mtx-rand2p1-s1"], "RAND2_p2": ["mtx-rand2p2-s2"],
            M.INIT: ["init-s1", "init-s2"]}
    atlas = {m: doc(True, a == "L162_MASS") for a in ("L188", "L162_MASS", M.INIT) for m in runs[a]}
    assert not M.mass_has_parts(atlas, runs)
    plain = M.table_mass_detail(atlas, runs, M.TAB)
    assert M.PARTITIONS not in plain
    atlas |= {m: doc(True) for a in ("FLAV_F0", "RAND2_p1", "RAND2_p2") for m in runs[a]}
    assert M.mass_has_parts(atlas, runs)
    lines = M.table_mass_detail(atlas, runs, M.TAB).split("%SPLIT%")[0].splitlines()
    for heading in (M.HEADING["mlp"], M.RIDGE):
        k = next(i for i, ln in enumerate(lines) if heading in ln)
        block = lines[k + 1:k + 1 + len(M.SETS) + 1 + len(M.MASS_PARTS) + 1]
        labels = [ln.split(" & ")[0] for ln in block]
        assert labels[:len(M.SETS)] == [M.LABEL[a] for a in M.SETS]
        assert M.PARTITIONS in labels[len(M.SETS)]
        assert labels[len(M.SETS) + 1:-1] == [r"\quad " + M.LABEL[a] for a in M.MASS_PARTS]
        assert labels[-1] == M.LABEL[M.INIT]
    head = next(i for i, ln in enumerate(lines) if M.HEADING["head"] in ln)
    assert M.PARTITIONS not in lines[head + 1] and M.PARTITIONS not in lines[head + 2]
