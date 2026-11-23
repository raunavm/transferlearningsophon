"""The data release (experiments/FIGS/data_release.py, PI 2026-11-22): every number the paper prints matches the
release. Each number of the generated tables (paper/journal/tables/*.tex) and macros (results_generated.tex), as
make_tables.build() writes them, equals its release value rounded at its printed precision, read here from the
release's CSV text; the means and SDs recompute from the runs the release holds; items.csv and the README map every
figure and table of gallery_placement to its files; and a printed number that is not its value stops the build."""
import collections
import csv
import importlib.util
import io
import json
import math
import pathlib
import re
import statistics

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("data_release_t", "experiments/FIGS/data_release.py")
GP = _load("gallery_placement_release_t", "experiments/FIGS/figstyle/gallery_placement.py")
PLAIN = re.compile(r"([-+]?)(\d[\d,]*)(?:\.(\d+))?")
SCI = re.compile(r"(?:(\d+)(?:\.(\d+))?\s*(?:\\times|×)\s*)?10\s*\^?\s*\{?\s*([-+]?)\s*(\d+)\s*\}?")


def _rounds(value: float, printed: str, rule: str) -> bool:
    """The test's own reading of a printed number: round, |value - printed| at most half a unit of its last printed
    digit (of an integer, its last nonzero digit); at_most, value <= printed; exact, value == printed."""
    if m := PLAIN.fullmatch(printed):
        digits = m.group(2).replace(",", "")
        dec = len(m.group(3) or "")
        p = float(digits + ("." + m.group(3) if dec else "")) * (-1 if m.group(1) == "-" else 1)
        q = 10.0 ** -dec if dec else 10.0 ** (len(digits) - len(digits.rstrip("0")) if digits.strip("0") else 0)
    else:
        m = SCI.fullmatch(printed)
        assert m, printed
        dec = len(m.group(2) or "")
        e = int(m.group(4)) * (-1 if m.group(3) == "-" else 1)
        p = float((m.group(1) or "1") + ("." + m.group(2) if dec else "")) * 10.0 ** e
        q = 10.0 ** (e - dec)
    eps = 1e-9 * max(abs(p), abs(value), q)
    if rule == "at_most":
        return value <= p + eps
    if rule == "exact":
        return abs(value - p) <= eps
    return abs(value - p) <= q / 2 + eps


@pytest.fixture(scope="module")
def release():
    """One build, the release's table, macro and map files written from it (no figures, no per-run files: they need
    TeX and the render job; scripts/build_render_job.py --release writes the whole release)."""
    MT = _load("make_tables_release_t", "experiments/FIGS/make_tables.py")
    with pytest.MonkeyPatch.context() as mp:
        try:
            built, _, _ = MT.build(ROOT)          # the recorder runs inside, and the build's own check at its end
        except SystemExit as e:
            # while fine-tuning reruns are running the paper builds only as the preview (MLST_FT_PREVIEW=1, the
            # render job's --preview); its numbers are checked the same way
            if "rate search is not settled" not in str(e):
                raise
            mp.setenv("MLST_FT_PREVIEW", "1")
            built, _, _ = MT.build(ROOT)
        files, report = R.release_files(built, MT._RELEASE, commit="test")
    return built, files


def _rows(text: str) -> list:
    return list(csv.DictReader(io.StringIO(text)))


def test_every_number_of_every_generated_table_is_its_release_value_rounded(release):
    built, files = release
    n, tables = 0, 0
    tex = {rel: text for rel, text in built.items() if rel.startswith("tables/") and rel.endswith(".tex")}
    names = R.table_csv_names({rel: R.table_label(text) for rel, text in tex.items()})
    for rel, text in tex.items():
        printed = [c for c in R.cells(text) if c.nums]
        name = names[rel]
        assert name in files, rel
        rows = [r for r in _rows(files[name]) if r["file"] == "paper/journal/" + rel]
        assert len(rows) == len(printed), rel
        tables += bool(rows)
        for c, r in zip(printed, rows):
            assert r["printed"] == c.text and r["row"] == c.row and r["column"] == c.column, (rel, r["cell"])
            nums = r["numbers"].split(";")
            assert nums == [x.s for x in c.nums], (rel, c.text)
            for s, v, rule in zip(nums, r["values"].split(";"), r["rules"].split(";"), strict=True):
                assert _rounds(float(v), s, rule), (rel, c.row, c.column, s, v, rule)
                n += 1
    assert tables > 40 and n > 5000


def test_every_number_of_every_macro_is_its_release_value_rounded(release):
    built, files = release
    rows = collections.defaultdict(list)
    for r in _rows(files["macros.csv"]):
        rows[r["macro"]].append(r)
    prov = json.loads(built["provenance.json"])
    macros = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  %", built["results_generated.tex"])
    assert len(macros) == len(prov)
    n = 0
    for name, body in macros:
        assert body == prov[name]["value"]
        nums = R.numbers(body)
        got = [r for r in rows[name] if r["number"] != ""]
        assert len(got) == len(nums), name
        for x, r in zip(nums, got):
            assert r["body"] == body and r["printed"] == x.s, name
            assert _rounds(float(r["value"]), x.s, r["rule"]), (name, body, r["value"], r["rule"])
            n += 1
    assert n > 1000


def test_each_mean_and_sd_recompute_from_the_runs_the_release_holds(release):
    _, files = release
    n = 0
    for name, text in files.items():
        if not name.startswith("tables/"):
            continue
        for r in _rows(text):
            if r["kind"] == "p" and r["run_values"]:
                v = [float(x) for x in r["run_values"].split(";")]
                assert int(r["n_runs"]) == len(v) >= 2
                assert statistics.fmean(v) == pytest.approx(float(r["mean"]), rel=1e-9, abs=1e-12), (name, r["cell"])
                assert statistics.stdev(v) == pytest.approx(float(r["sd"]), rel=1e-9, abs=1e-12), (name, r["cell"])
                n += 1
    assert n > 200


def test_the_readme_and_items_map_every_figure_and_table_the_paper_places(release):
    built, files = release
    items = {r["id"]: r for r in _rows(files["items.csv"])}
    readme = files["README.txt"]
    for key, tier in GP.TIER.items():
        assert key in items and items[key]["placement"] == tier, key
        assert f" {key} (" in readme, key
    for key, _, _ in GP.TABLES:
        listed = items[key]["files"].split()
        assert listed, key                  # every placed table has a generated file, its cells in the release
        assert all(f in files for f in listed), key
    assert not [f for f in files if f.endswith(".md")]
    assert {"README.txt", "items.csv", "macros.csv", "runs.csv"} <= set(files)


def test_the_readme_holds_the_run_records_that_left_the_appendix(release):
    """PI 2026-11-23 (LEDGER SP-13, step 5): the training-order and checkpoint hashes and the fine-tuning attempts'
    GPU-hour accounting moved from the appendix's training settings to the README, their numbers the paper's macros;
    the README names the partitions' label maps, which the class map no longer prints (step 1, R4)."""
    built, files = release
    records = files["README.txt"].split("RUN RECORDS\n", 1)[1].split("ROUNDING AND THE CHECK", 1)[0]
    gen = built["results_generated.tex"]
    for name in ("MlstFtPromFrom", "MlstFtGpuHoursBound", "MlstFtAttemptsBound", "MlstFtAttemptsUnknown"):
        body = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*?)\}\s*(?:%|$)", gen, re.M).group(1)
        assert R.detex(body) in records, name
    assert "{" not in records and "hash" in records and "upper bound" in records
    assert "rand_label_map.v2.csv" in records and "flavour_pair_map.v2.csv" in records


def test_a_printed_number_that_is_not_its_value_stops_the_build():
    tok = R.TOK_OPEN + "p 0.43476 0.00035" + R.TOK_CLOSE
    tab = ("\\begin{tabular}{lc}\n\\toprule\n & accuracy \\\\\n\\midrule\n\\Kfull{} classes & " + tok
           + " \\\\\n\\bottomrule\n\\end{tabular}")
    printed = tab.replace(tok, r"0.43476\,$\pm$\,0.00035")
    for shown, bad in ((printed, False), (printed.replace("0.43476", "0.43500"), True)):
        rec = R.Recorder()
        rec.positional(tab, printed)
        rec.claim("tab:x")
        out = {"tables/x.tex": "\\label{tab:x}\n" + shown}
        assert bool(rec.match(out)) is bad
        row = rec.result["tables"]["tables/x.tex"]["rows"][0]
        assert row["basis"] == "position" and row["row"] == "188 classes" and row["column"] == "accuracy"
    with pytest.raises(SystemExit, match="not their value"):
        R.finish(out, {"_RELEASE": rec})
    # a macro number a formatter printed from a value it does not round to
    rec = R.Recorder()
    rec.log.append(["fmt", "0.44", [0.4349], ["round"], None, False])
    rec.macros.append(("MlstX", "0.44", 0, 1))
    assert rec.match({}) and rec.result["macros"][0]["basis"] == "formatter"


def test_numbers_read_as_the_paper_prints_them():
    s = lambda t: [x.s for x in R.numbers(t)]       # noqa: E731
    assert s(r"{\fontseries{b}\selectfont 1{,}800\,$\pm$\,60}$^{\S}$") == ["1,800", "60"]
    assert s(r"+0.0006~[\ensuremath{-}0.0004, +0.0015]") == ["+0.0006", "-0.0004", "+0.0015"]
    assert s(r"128{,}833--135{,}245") == ["128,833", "135,245"] and s("51--52\\%") == ["51", "52"]
    assert s(r"0.43$^{\S}$ \pending{2 of 3} $R_{30}$") == ["0.43"]
    assert s("L4, RTX 3090, RTX 4090 and V100") == ["3090", "4090"]
    assert math.isclose(R.numbers(r"\ensuremath{8.192\times10^{8}}")[0].value, 8.192e8)
    assert math.isclose(R.numbers(r"\ensuremath{10^{-5}}")[0].value, 1e-5)
    assert math.isclose(R.numbers("5 × 10\u22124")[0].value, 5e-4)
    one = R.numbers("1{,}800")[0]
    assert R.close(1803, one) and not R.close(1851, one)
    assert R.close(0.43476, R.numbers("0.4348")[0]) and not R.close(0.4349, R.numbers("0.4350")[0])
    assert R.close(2.7, R.numbers("3")[0], "at_most") and not R.close(3.2, R.numbers("3")[0], "at_most")


def test_cells_read_rows_columns_and_headings_of_a_split_longtable():
    t = ("\\begin{longtable}{lcc}\n\\caption{x}\\label{tab:y}\\\\\n\\toprule\n & \\multicolumn{2}{c}{JetClass} \\\\\n"
         "\\cmidrule(lr){2-3}\n & accuracy & AUC \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n & a & b \\\\\n\\midrule\n"
         "\\endhead\n\\multicolumn{3}{@{}l}{\\textit{fine-tuned}} \\\\*\n\\mbox{\\Kres{} classes,} \\mbox{seed 2} & "
         "0.7703 & -- \\\\\n\\noalign{\\nobreak\\vskip\\aboverulesep\\hrule}\n\\quad random partitions & "
         "0.75\\,$\\pm$\\,0.01 & 0.98 \\\\\n\\end{longtable}")
    cs = R.cells(t)
    assert [(c.block, c.row, c.column, [x.s for x in c.nums]) for c in cs] == [
        ("fine-tuned", "162 classes, seed 2", "JetClass / accuracy", ["0.7703"]),
        ("fine-tuned", "162 classes, seed 2", "JetClass / AUC", []),
        ("fine-tuned", "random partitions", "JetClass / accuracy", ["0.75", "0.01"]),
        ("fine-tuned", "random partitions", "JetClass / AUC", ["0.98"])]


def test_the_per_run_values_are_the_files_read_and_long_curves_thin_to_a_coarser_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DATA", tmp_path)
    d = tmp_path / "v2" / "curves_v2" / "fam"
    d.mkdir(parents=True)
    (d / "m-s1.json").write_text(json.dumps({"model": "m-s1", "search": {"C": [1, 2]},
                                             "curves": {"a": {"auc": 0.9, "eps": list(range(400)),
                                                              "rej": [1.0 / (k + 1) for k in range(400)]}}}))
    files, index = R.per_run_files([d / "m-s1.json"], stride=1)
    eps = next(r for r in _rows(files["per_run/v2__curves_v2__fam__arrays.csv"]) if r["key"] == "eps")
    assert eps["path"] == "curves.a" and eps["n"] == "400" and eps["stride"] == "1"
    assert eps["values"].split(";")[3] == "3" and len(eps["values"].split(";")) == 400
    assert not any("search" in t for t in files.values())
    thin, _ = R.per_run_files([d / "m-s1.json"], stride=4)
    eps = next(r for r in _rows(thin["per_run/v2__curves_v2__fam__arrays.csv"]) if r["key"] == "eps")
    assert eps["stride"] == "4" and len(eps["values"].split(";")) == 101        # every 4th and the last


def test_the_release_holds_what_the_paper_points_to(release):
    """PI 2026-11-23 (LEDGER DA-2): the two pointers the paper keeps hold in the final release. Table 8's caption points
    to the data release for each run's hours, GPU types and last validation accuracy: tab:computefull's CSV has one row
    per pretraining run of the paper with those columns filled. Appendix A points to the code release for the label maps:
    the repository's maps give a group to every JetClass-II class in every label set and partition."""
    built, files = release
    M = _load("mlst_results_da2_t", "experiments/FIGS/mlst_results.py")
    runs = M._paper(ROOT)
    name = "tables/tab_computefull.csv"
    assert name in files, sorted(f for f in files if f.startswith("tables/"))[:20]
    rows = _rows(files[name])
    cols = {r["column"] for r in rows}
    for want in ("validation", "GPU types (hours)", "GPU-hours"):
        assert any(want in c for c in cols), (want, cols)
    labelled = {r["row"] for r in rows}
    pretrained = [m for a, ms in runs.items() if a in M.ALL_ROWS for m in ms]
    assert len(labelled) >= len(pretrained), (len(labelled), len(pretrained))
    maps = ROOT / "configs/labelmaps"
    rung = list(csv.DictReader((maps / "rung_label_maps.v1.csv").open()))
    rand = list(csv.DictReader((maps / "rand_label_map.v2.csv").open()))
    flav = list(csv.DictReader((maps / "flavour_pair_map.v2.csv").open()))
    for rows_, columns in ((rung, ("L188", "L162", "R42_Q1", "R16_Q1")), (rand, [a for a in M.RAND]),
                           (flav, M.FLAV)):
        assert len(rows_) == 188, len(rows_)
        for c in columns:
            assert all(r[c] != "" for r in rows_), c
