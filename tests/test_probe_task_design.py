"""The probe's task set must form the 2x2 the paper's claim rests on.

If the contraction tree is ever regenerated, a task could silently change which
arm collapses it. Nothing would error: the probe would still fit, still report
an AUC, and the paper would still have a table -- measuring something else.
These tests read the committed label map and assert the design directly.
"""
import csv
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "probe", REPO / "experiments" / "EVAL" / "probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"


def _rows():
    with MAP.open() as f:
        return {int(r["jet_label"]): r for r in csv.DictReader(f)}


def _collapses(rows, arm, a, b):
    """True if `arm` maps native labels a and b to the SAME group."""
    return rows[a][arm] == rows[b][arm]


def test_bvc_resonant_is_collapsed_by_r16q1_and_kept_by_l162():
    rows = _rows()
    sig, bkg = probe.TASKS["bvc_resonant"]["signal"][0], probe.TASKS["bvc_resonant"]["background"][0]
    assert _collapses(rows, "R16_Q1", sig, bkg), "R16_Q1 must collapse the probed axis"
    assert not _collapses(rows, "L162", sig, bkg), "L162 must keep it"


def test_retained_topology_is_kept_by_both_arms():
    """The control leg. If R16_Q1 collapsed this too, the task could not
    distinguish 'this axis was erased' from 'this arm is worse at everything'."""
    rows = _rows()
    t = probe.TASKS["retained_topology"]
    sig, bkg = t["signal"][0], t["background"][0]
    assert not _collapses(rows, "R16_Q1", sig, bkg), "R16_Q1 must KEEP the control axis"
    assert not _collapses(rows, "L162", sig, bkg), "L162 must keep the control axis"
    assert t["collapsed_at"] == [], "control axis is collapsed at no studied rung"


def test_retained_topology_holds_flavour_fixed():
    """It is the complement of bvc_resonant: that task fixes topology and varies
    flavour, this one fixes flavour and varies topology. Both endpoints all-b."""
    rows = _rows()
    t = probe.TASKS["retained_topology"]
    names = [rows[n]["class_name"] for n in (t["signal"][0], t["background"][0])]
    assert names == ["label_X_bb", "label_X_YY_bbbb"], names
    for n in names:
        assert "c" not in n.split("_")[-1].replace("bb", ""), f"{n} is not all-b"


def test_bvc_qcd_is_collapsed_by_both_arms():
    """The null-vs-null leg: neither arm should win, and a gap indicts the method."""
    rows = _rows()
    sig, bkg = probe.TASKS["bvc_qcd"]["signal"][0], probe.TASKS["bvc_qcd"]["background"][0]
    assert _collapses(rows, "R16_Q1", sig, bkg)
    assert _collapses(rows, "L162", sig, bkg)


def test_every_task_endpoint_exists_in_the_map():
    rows = _rows()
    for name, t in probe.TASKS.items():
        for n in t["signal"] + t["background"]:
            assert n in rows, f"task {name} references native label {n}, not in the map"
