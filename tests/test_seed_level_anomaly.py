"""§5 in experiments/STATS/seed_level.py on a synthetic anomaly_merge.py file."""
import contextlib
import importlib.util
import io
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level", REPO / "experiments/STATS/seed_level.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

STEMS = {"l188": 188, "l162": 162, "r42q1": 43, "r16q1": 17}
JITTER = {"l188": [0.010, -0.012, 0.004, 0.008, -0.006], "l162": [-0.004, 0.009, 0.006, -0.008, 0.004],
          "r42q1": [0.006, 0.004, -0.010, 0.001, 0.008], "r16q1": [0.012, -0.008, 0.015, -0.004, 0.0]}


def doc(sigma):
    arms = {}
    for stem in STEMS:
        for s in range(1, 6):
            init = "l162-s1b" if (stem, s) == ("l162", 1) else f"{stem}-s{s}"
            sigs = {}
            for sig in S.AD_B_SIGNALS + S.AD_Q_SIGNALS:
                sigs[sig] = {n: {f: {"sigma_min": sigma(stem, sig, f) * math.exp(JITTER[stem][s - 1]),
                                     "at_ceiling": False, "max_sic": 2.0} for f in S.AD_FAMILIES}
                             for n in ("1000", "4000")}
            arms[init] = {"rung": stem, "signals": sigs}
    return {"arms": arms, "row_alignment_sha256": "ab" * 32, "trainings": 10,
            "signals_with_unequal_seeds_per_rung": []}


def run(tmp_path, sigma):
    p = tmp_path / "ad.json"
    p.write_text(json.dumps(doc(sigma)))
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main(["--anomaly", str(p), "--out", str(tmp_path / "o")]) == 0
    return json.loads((tmp_path / "o" / "anomaly_s5.json").read_text())["section5"]


def test_both_clauses_confirmed_when_coarse_labels_lose_more_on_b_signals(tmp_path):
    def sigma(stem, sig, fam):
        step = {"l188": 0, "l162": 0.05, "r42q1": 0.1, "r16q1": 0.3}[stem]
        return 2.0 * math.exp(step * (2 if sig in S.AD_B_SIGNALS else 1))
    r = run(tmp_path, sigma)
    assert r["clause1"] == "confirmed" and r["clause2"] == "confirmed"
    assert r["tests"]["knn|label_X_bb"]["gap_17_minus_188"] > r["tests"]["knn|label_X_qq"]["gap_17_minus_188"]


def test_no_label_effect_confirms_neither(tmp_path):
    r = run(tmp_path, lambda stem, sig, fam: 2.0)
    assert r["clause1"] == "not confirmed"
    assert all(not v["confirmed"] for v in r["clause1_per_family"].values())


def test_a_signal_skipped_at_the_fixed_injection_makes_clause_two_not_evaluable(tmp_path):
    """Too few jets of one signal in the test sample to inject 4,000 is not evidence
    against the clause; the comparison on the signals that could be tested is shown."""
    d = doc(lambda stem, sig, fam: 2.0)
    for arm in d["arms"].values():
        arm["signals"]["label_X_YY_bbb"]["4000"] = {"skipped": "insufficient jets"}
    p = tmp_path / "ad.json"
    p.write_text(json.dumps(d))
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main(["--anomaly", str(p), "--out", str(tmp_path / "o")]) == 0
    r = json.loads((tmp_path / "o" / "anomaly_s5.json").read_text())["section5"]
    assert r["clause2"] == "not evaluable" and r["skipped_at_injection"] == ["label_X_YY_bbb"]
    assert r["clause2_on_testable_signals"]["knn"]["signals_b"] == ["label_X_bb", "label_X_YY_bbbb"]
