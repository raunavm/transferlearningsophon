"""The frozen-representation extractor's two silent failure modes.

Both produce a plausible AUC from a meaningless representation, which is the
worst kind of bug this project can have: nothing errors, the number looks
publishable, and it is wrong. So both guards are tested here rather than
trusted, and each is tested by CONSTRUCTING the failure, not by asserting the
happy path.

  trunk incomplete   `load_state_dict(strict=False)` is required (the head's
                     shape is K-dependent) and will silently leave the entire
                     trunk at its random initialisation.
  K mislabelled      the trunk is K-independent, so a K=17 checkpoint declared
                     as K=43 loads cleanly and yields valid features attributed
                     to the wrong arm.

Also pins the hook itself: `fc`'s input is x_cls only because weaver 0.4.17's
forward ends `x_cls = self.norm(...); output = self.fc(x_cls)`. If that ever
changes, `fc(tapped) == model(...)` stops holding and this catches it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("weaver")

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs" / "data" / "JetClassII_base.yaml"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "extract_features", REPO / "experiments" / "EVAL" / "extract_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ctx():
    from weaver.utils.data.config import DataConfig
    if not BASE_CFG.exists():
        pytest.skip("base data config absent")
    ex = _mod()
    dc = DataConfig.load(str(BASE_CFG), load_observers=True)
    return ex, dc


@pytest.fixture(scope="module")
def ckpt43(ctx, tmp_path_factory):
    ex, dc = ctx
    model = ex.build_model(dc, 43)
    p = tmp_path_factory.mktemp("ck") / "k43.pt"
    torch.save(model.state_dict(), p)
    return p


def test_fc_input_is_the_representation(ctx, ckpt43):
    """fc(tapped) == model(...) -- the hook captures x_cls and not something else."""
    ex, dc = ctx
    model = ex.build_model(dc, 43)
    ex.load_trunk_or_die(model, ckpt43, 43)
    model.eval()
    tap = ex.ClsTap(model)
    args = ex.synthetic_batch(dc, 4, torch.device("cpu"))
    with torch.no_grad():
        out = model(*args)
    assert torch.isfinite(out).all(), "synthetic fixture is invalid, not the hook"
    assert tap.buf is not None, "forward pre-hook never fired"
    assert tap.buf.shape == (4, ex.EMBED_DIM)
    with torch.no_grad():
        replay = tap.fc(tap.buf)
    assert torch.allclose(replay.float(), out.float(), atol=1e-4, rtol=1e-3)
    tap.close()


def test_incomplete_trunk_is_refused(ctx, ckpt43, tmp_path):
    """Deleting one transformer block must be fatal, not silently tolerated."""
    ex, dc = ctx
    state = torch.load(ckpt43, map_location="cpu")
    holed = {k: v for k, v in state.items() if not k.startswith("mod.blocks.0")}
    assert len(holed) < len(state), "fixture removed nothing"
    p = tmp_path / "holed.pt"
    torch.save(holed, p)
    model = ex.build_model(dc, 43)
    with pytest.raises(SystemExit) as e:
        ex.load_trunk_or_die(model, p, 43)
    assert e.value.code == 2


def test_mislabelled_num_classes_is_refused(ctx, ckpt43):
    """A K=43 checkpoint declared as K=17 would give valid features, wrong arm."""
    ex, dc = ctx
    model = ex.build_model(dc, 17)
    with pytest.raises(SystemExit) as e:
        ex.load_trunk_or_die(model, ckpt43, 17)
    assert e.value.code == 3


def test_correct_checkpoint_loads_clean(ctx, ckpt43):
    ex, dc = ctx
    model = ex.build_model(dc, 43)
    prov = ex.load_trunk_or_die(model, ckpt43, 43)
    assert prov["checkpoint_num_classes"] == 43
    assert prov["trunk_tensors_loaded"] > 200
    assert prov["head_missing"] == 0 and prov["head_unexpected"] == 0


def test_head_width_selector_picks_the_output_layer(ctx, ckpt43):
    """fc.0 is (512,128) and fc.1 is (K,512); selecting by WIDTH reports 512 for
    every arm, which is the bug this pins."""
    ex, dc = ctx
    model = ex.build_model(dc, 43)
    prov = ex.load_trunk_or_die(model, ckpt43, 43)
    assert prov["checkpoint_num_classes"] == 43, "selector picked the hidden layer"
