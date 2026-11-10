"""The storage line as a builder parameter (PI 2026-11-09): 85% stays the default everywhere; 90% only for the
finishing work (the remaining equal-stopping cells and any rerun they trigger, the read-outs, the test suite, the
gallery and paper renders); nothing above 90%."""
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments/FT/k8s/budget"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sl = _load("storage_line", "scripts/storage_line.py")
bfb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
btj = _load("build_test_job", "scripts/build_test_job.py")
bgj = _load("build_gallery_job", "scripts/build_gallery_job.py")
brj = _load("build_render_job", "scripts/build_render_job.py")
bgd = _load("build_gallery_data_job", "scripts/build_gallery_data_job.py")
LINE = re.compile(r'\[ "\$\{USED\}" -le (\d+) \]')


def _lines(text: str) -> set:
    return {int(x) for x in LINE.findall(text)}


def test_the_lines_are_the_pis():
    assert (sl.GENERAL, sl.FINISHING, sl.MAX) == (85, 90, 90)
    assert sl.check(85) == 85 and sl.check(90) == 90
    for bad in (84, 91, 95):
        with pytest.raises(SystemExit):
            sl.check(bad)


def test_every_builder_defaults_to_85_and_takes_90_but_never_more():
    cells = bfb.pre_full_cells("1e-3")[:2]
    assert _lines(bfb.spec("x-raunav", cells, "mtx-s2.87", 1, "48Gi")) == {85}
    assert _lines(bfb.spec("x-raunav", cells, "mtx-s2.87", 1, "48Gi", line=90)) == {90}
    assert _lines(btj.spec("a" * 40, "full", [], True)[1]) == {85}
    assert _lines(btj.spec("a" * 40, "full", [], True, 90)[1]) == {90}
    for spec in (bgj.SPEC, brj.SPEC, bgd.SPEC):
        assert "-le {line} ]" in spec and "-le 85" not in spec
    with pytest.raises(SystemExit):
        bfb.spec("x-raunav", cells, "mtx-s2.87", 1, "48Gi", line=91)
    with pytest.raises(SystemExit):
        btj.spec("a" * 40, "full", [], True, 91)


def test_the_parameter_changes_nothing_in_the_launched_equal_stopping_job(tmp_path, monkeypatch):
    """Regenerated at the default line, the launched spec is byte-identical: the line is the only knob."""
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    bfb.main(["--tag", "mtx-s2.87", "--lr-pre", "1e-3", "--lr-scratch", "1e-3", "--mem", "48Gi", "--cpu", "3",
              "--parallelism", "8", "pre-full"])
    assert (tmp_path / "job-ft-budget-pre-full-raunav.yaml").read_text() == \
        (K8S / "job-ft-budget-pre-full-raunav.yaml").read_text()


def _cases(text: str) -> dict:
    return {int(m[1]): m[2] for m in re.finditer(r"^\s+(\d+)\) (TASK=.*;;)$", text, re.M)}


def test_a_retry_of_refused_equal_stopping_cells_is_the_launched_cell_at_90(tmp_path, monkeypatch):
    """A cell the 85% guard refused is retried as itself: the same tag, the same case line, the same script, only the
    line at 90; every committed retry spec is checked the same way."""
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    launched = (K8S / "job-ft-budget-pre-full-raunav.yaml").read_text()
    bfb.main(["--tag", "mtx-s2.87", "--lr-pre", "1e-3", "--lr-scratch", "1e-3", "--mem", "48Gi", "--cpu", "3",
              "--parallelism", "8", "--storage-line", "90", "--indices", "30,35", "pre-full"])
    specs = [tmp_path / "job-ft-budget-pre-full-retry-raunav.yaml", *K8S.glob("job-ft-budget-pre-full-retry*-raunav.yaml")]
    old = _cases(launched)

    def script(text):       # the pod's script without its case lines and its guard
        s = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
        return re.sub(r"^\s*\d+\) TASK=.*;;\n", "", LINE.sub("", s), flags=re.M)

    for p in specs:
        t = p.read_text()
        assert _lines(t) == {90} and 'value: "mtx-s2.87"' in t
        new = _cases(t)
        assert set(new.values()) <= set(old.values()) and len(new) == int(re.search(r"completions: (\d+)", t)[1])
        assert script(t) == script(launched)


@pytest.mark.parametrize("task,n,count", [("top", 1000, 22), ("top", 10000, 22), ("qg", 1000, 21), ("jc1", 10000, 21),
                                          ("jc2b", 10000, 10)])
def test_the_rerun_is_every_other_pretrained_cell_as_launched(task, n, count):
    """Label sets, random partitions, flavor boundaries, mass models, kinematic-only (top): every pretrained cell of the
    (task, size) but the six already in the equal-stopping job, each the launched cell but for its tree and
    patience_from."""
    cells = bfb.pre_full_rerun_cells("1e-3", [(task, n)])
    assert len(cells) == count
    launched = {(c["task"], c["init"], c["n"]): c for c in bfb.bench_cells("1e-3", "1e-3", ("top", "qg", "jc1"))
                + bfb.jc2b_cells("1e-3", "1e-3") + bfb.kin_cells("1e-3") if c["init"]}
    done = {(c["task"], c["init"], c["n"]) for c in bfb.pre_full_cells("1e-3")}
    assert {(c["task"], c["init"], c["n"]) for c in cells} == {k for k in launched if k[2] == n and k not in done
                                                              and k[0] in bfb.PRE_FULL_TASK[task]}
    for c in cells:
        o = launched[(c["task"], c["init"], c["n"])]
        assert {k: v for k, v in c.items() if k not in ("out", "patience_from")} == {k: v for k, v in o.items() if k != "out"}
        assert c["out"] == o["out"].replace("/ft_v2/budget/", "/ft_v2/budget_prefull/") and c["patience_from"] == c["epochs"]
    if task == "top":
        assert any(c["task"] == "topkin" for c in cells)
    with pytest.raises(SystemExit):
        bfb.pre_full_rerun_cells("1e-3", [("top", 100000)])


def test_the_equal_stopping_readout_waits_for_every_cell_at_90(tmp_path, monkeypatch):
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    bfb.main(["--tag", "mtx-s2.88", "--lr-pre", "1e-3", "--storage-line", "90", "--rerun", "top:1000",
              "readout-pre-full"])
    t = (tmp_path / "job-budget-readout-prefull-raunav.yaml").read_text()
    doc = yaml.safe_load(t)
    script = doc["spec"]["template"]["spec"]["containers"][0]["args"][0]
    cells = re.search(r"for c in (.*?); do\n", script)[1].split()
    want = bfb.pre_full_cells("1e-3") + bfb.pre_full_rerun_cells("1e-3", [("top", 1000)])
    assert sorted(cells) == sorted(c["out"] for c in want) and len(cells) == 58
    assert _lines(t) == {90} and f"n=={len(want)} and not d['missing']" in script
    assert f"OUT={bfb.PRE_FULL_METRICS}" in script and 'if [ -e ${OUT} ]; then echo "skip' in script
    for p in K8S.glob("job-budget-readout-prefull*-raunav.yaml"):
        assert _lines(p.read_text()) == {90}
    for p in K8S.glob("job-ft-budget-pre-full-rerun*-raunav.yaml"):
        assert _lines(p.read_text()) == {90} and 'value: "mtx-s2.87"' in p.read_text()


def test_the_real_sample_xbb_xbs_chain_takes_the_finishing_line_and_the_stand_in_keeps_85():
    """PI 2026-11-09 ("increase the line"): every job-xbx-ihep-* spec has its one guard at 90, every other spec at 85."""
    bx = _load("build_xbx_jobs", "scripts/build_xbx_jobs.py")
    assert bx.IHEP_LINE == sl.FINISHING == 90
    n = 0
    for name, text in bx.build(tuple(bx.SIGNALS), bx.MX).items():
        if name.startswith("job-xbx-ihep-"):
            assert text.count("-le 90 ]") == 1 and bx.GUARD not in text
            n += 1
        else:
            assert text.count(bx.GUARD) == 1 and "-le 90 ]" not in text
    assert n > 100


def test_the_real_sample_specs_avoid_the_nodes_whose_gpus_failed():
    bx = _load("build_xbx_jobs", "scripts/build_xbx_jobs.py")
    for name, text in bx.build(tuple(bx.SIGNALS), bx.MX).items():
        for node in ("k8s-haosu-12.sdsc.optiputer.net", "k8s-haosu-09.sdsc.optiputer.net"):
            assert (f'"{node}"' in text) == name.startswith("job-xbx-ihep-"), name
        assert len(bx.HOST_NOTIN.findall(text)) == 1


def test_the_real_sample_extractions_and_fits_request_what_they_were_measured_to_use():
    bx = _load("build_xbx_jobs", "scripts/build_xbx_jobs.py")
    for name, text in bx.build(tuple(bx.SIGNALS), bx.MX).items():
        x = bool(re.match(r"job-xbx-ihep-(xbb|xbs)-x-", name))
        fit = bool(re.match(r"job-xbx-ihep-(xbb|xbs)-fit-", name))
        assert (text.count('memory: "8Gi", cpu: "2"') == 2) == x, name
        assert (text.count('memory: "24Gi", cpu: "2"') == 2) == fit, name
