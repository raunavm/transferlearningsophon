"""PI 2026-11-23, LEDGER FG-25: "tab:sets is 121.7 pt too wide. Fix it, and make overfull boxes over 10 pt in any float
stop the paper build" (scripts/overfull_floats.py, run by scripts/build_render_job.py's PAPER mode)."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


O = _load("overfull_floats_t", "scripts/overfull_floats.py")

# tectonic's form of the warnings (render 10c97a1-preview's tectonic.txt), each pass repeating them
LOG = r"""note: downloading cmbx10.tfm
warning: tables/mlst_sets:26: Overfull \hbox (121.6568pt too wide) in alignment at lines 5--26
warning: sections/methods:234: Overfull \hbox (8.97865pt too wide) in paragraph at lines 231--234
warning: sections/results_jc:12: Overfull \hbox (14.5pt too wide) in paragraph at lines 10--12
warning: sections/results_jc:23: Overfull \hbox (12.25pt too wide) in paragraph at lines 21--23
warning: tables/mlst_tests:255: Overfull \hbox (3.09714pt too wide) in alignment at lines 16--255
warning: main.bbl:361: Underfull \hbox (badness 2080) in paragraph at lines 357--361
warning: tables/mlst_sets:26: Overfull \hbox (121.6568pt too wide) in alignment at lines 5--26
"""
SECTION = "\n".join(["text"] * 9 + [r"\begin{figure}[t]", r"\centering", r"\caption{a caption that runs over}",
                                    r"\end{figure}"] + ["text"] * 10) + "\n"


def _dir(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "results_jc.tex").write_text(SECTION)
    (tmp_path / "sections" / "methods.tex").write_text("\n".join(["text"] * 240))
    return tmp_path


def test_each_overfull_box_is_read_once_and_attributed_to_its_source_and_float(tmp_path):
    rows = O.overfull(LOG, _dir(tmp_path))
    got = [(r["source"], r["lines"], r["pt"], r["float"]) for r in rows]
    assert got == [("tables/mlst_sets.tex", (5, 26), 121.6568, "a generated table"),
                   ("sections/methods.tex", (231, 234), 8.97865, None),
                   ("sections/results_jc.tex", (10, 12), 14.5, "the float at lines 10--13"),
                   ("sections/results_jc.tex", (21, 23), 12.25, None),
                   ("tables/mlst_tests.tex", (16, 255), 3.09714, "a generated table")]
    assert O.float_lines(SECTION) == [(10, 13)]


def test_an_overfull_float_over_ten_points_stops_the_build_and_the_rest_are_listed(tmp_path, capsys):
    d = _dir(tmp_path)
    log = tmp_path / "tectonic.txt"
    log.write_text(LOG)
    assert O.main([str(log), str(d)]) == 4
    out = capsys.readouterr().out
    assert "FATAL: 2 overfull boxes over 10 pt in floats" in out
    assert "tables/mlst_sets.tex lines 5--26, 121.7 pt" in out and "sections/results_jc.tex lines 10--12, 14.5 pt" in out
    # outside a float (12.25 pt in a paragraph) or at most 10 pt in one: listed, never fatal
    assert "overfull 12.2 pt: sections/results_jc.tex lines 21--23\n" in out and "overfull 3.1 pt: tables/mlst_tests.tex" in out
    # the fixed tab:sets: the rest are listed and the build goes on
    log.write_text("\n".join(ln for ln in LOG.splitlines() if "mlst_sets" not in ln and "results_jc:12" not in ln))
    assert O.main([str(log), str(d)]) == 0
    assert "overfull boxes: 3, none over 10 pt in a float" in capsys.readouterr().out


def test_the_paper_render_runs_the_stop_after_the_compile_and_the_other_modes_do_not():
    R = _load("build_render_job_overfull_t", "scripts/build_render_job.py")
    commit = "abcdef1" + "0" * 33
    spec = lambda mode: R.SPEC.format(short=commit[:7], commit=commit, image=R.IMAGE, pins=R.PINS,  # noqa: E731
                                      tectonic=R.TECTONIC, line=90, **mode)
    paper = spec(R.PAPER)
    stop = "python3 scripts/overfull_floats.py ${OUT}/tectonic.txt paper/journal"
    assert paper.index("tectonic -X compile main.tex") < paper.index(stop) < paper.index("RENDER COMPLETE")
    assert 'exit 4; }' in paper
    assert all(stop not in spec(m) for m in (R.MOCKUPS, R.BOOK, R.RELEASE))


def test_tab_sets_wraps_its_merges_and_runs_columns_and_never_its_row_labels():
    M = _load("mlst_results_overfull_t", "experiments/FIGS/mlst_results.py")
    body = M.table_sets(M._paper(ROOT))
    spec = body.splitlines()[0]
    assert spec == r"\begin{tabular}{ll" + "".join(rf">{{\raggedright\arraybackslash}}p{{{w:g}cm}}"
                                                     for w in M.SETS_WRAP_CM) + "}"
    # the row labels' column stays l (TN-6) and the wrapped columns fit the 15 cm text block beside them
    assert spec.startswith(r"\begin{tabular}{ll>") and sum(M.SETS_WRAP_CM) < 15.0 / 2
