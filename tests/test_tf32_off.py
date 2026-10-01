"""TF32 is off in every EVAL path that runs the pretrained network, before the first
model call, and each output records the setting.

The TF32 check (experiments/FIGS/data/tf32_check, one RTX A4000) measured torch's
defaults -- cuDNN convolutions in TF32 -- moving the resonance-vs-QCD log-odds by up
to 0.045 against TF32 off, while TF32-off GPU scores agree with float32 on the CPU
to 2e-4. That is the cross-GPU difference the real-data rescore saw (L4/L40 against
V100, up to 0.10)."""
import ast
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "EVAL"
# the check that measures TF32 switches it on and off on purpose
EXEMPT = {"tf32_check.py"}
MODEL_CALLS = {"build_model", "load_trunk_or_die", "resolve_checkpoints", "self_check"}


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@pytest.fixture
def tf32_on():
    was = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = was


def _off():
    return not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def test_strict_fp32_switches_both_off_and_reports_it(tf32_on):
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    assert ex.strict_fp32() == {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    assert _off()


def _network_scripts():
    out = []
    for p in sorted(EVAL.glob("*.py")):
        src = p.read_text()
        if p.name not in EXEMPT and ("cuda" in src or "build_model(" in src or "load_trunk_or_die(" in src):
            out.append(p)
    return out


@pytest.mark.parametrize("path", _network_scripts(), ids=lambda p: p.name)
def test_every_script_that_runs_the_network_switches_tf32_off_first_and_records_it(path):
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [(n.lineno, n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
             for n in ast.walk(main) if isinstance(n, ast.Call)]
    first_off = min((l for l, f in calls if f == "strict_fp32"), default=None)
    first_model = min((l for l, f in calls if f in MODEL_CALLS), default=None)
    assert first_off is not None, f"{path.name}: main() never calls strict_fp32()"
    assert first_model is None or first_off < first_model, f"{path.name}: a model call precedes it"
    assert '"tf32": tf32' in path.read_text(), f"{path.name}: the output does not record the setting"


def test_the_scope_is_not_vacuous():
    assert {p.name for p in _network_scripts()} >= {"extract_features.py", "extract_v2.py",
                                                   "epoch_accuracy.py"}


def test_extract_features_turns_it_off_before_its_self_check(tf32_on, tmp_path, monkeypatch):
    pytest.importorskip("weaver")
    from weaver.utils.data.config import DataConfig
    ex = _load("extract_features", "experiments/EVAL/extract_features.py")
    dc = DataConfig.load(str(ROOT / "configs/data/JetClassII_base.yaml"), load_observers=True)
    ck = tmp_path / "k17.pt"
    torch.save(ex.build_model(dc, 17).state_dict(), ck)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    monkeypatch.setattr(sys, "argv", ["extract_features.py", "--checkpoint", str(ck), "--num-classes",
                                      "17", "--arm", "T", "--data-test", "none.parquet",
                                      "--out", str(tmp_path / "o"), "--self-check-only"])
    assert ex.main() == 0 and _off()


def test_extract_v2_turns_it_off_before_it_resolves_a_checkpoint(tf32_on, tmp_path):
    pytest.importorskip("weaver")
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    with pytest.raises(SystemExit, match="checkpoints missing"):
        xv.main(["--run-dir", str(tmp_path), "--rung", "R16_Q1", "--num-classes", "17",
                 "--checkpoints", "79", "--data-test", "none.parquet", "--out", str(tmp_path / "o")])
    assert _off()
