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
import json
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


# ---------------------------------------------------------------------------
# The checkpoint guard. Both halves of this were broken: the guard ran AFTER
# every np.save (so it reported a mixture it had already created), and it read
# only `checkpoint_sha256`, a key the guard itself introduced -- leaving it
# inert on every cache written before it existed, i.e. exactly the caches it
# was added to protect.
# ---------------------------------------------------------------------------

def _guard():
    import importlib.util, pathlib as _p
    spec = importlib.util.spec_from_file_location(
        "_xf", _p.Path(__file__).resolve().parents[1]
        / "experiments" / "EVAL" / "extract_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.refuse_foreign_checkpoint


def test_guard_passes_on_a_fresh_directory(tmp_path):
    _guard()(tmp_path / "extract_manifest.json", "a" * 64)


def test_guard_passes_on_the_same_checkpoint(tmp_path):
    m = tmp_path / "extract_manifest.json"
    m.write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
    _guard()(m, "a" * 64)


def test_guard_refuses_a_different_checkpoint(tmp_path):
    m = tmp_path / "extract_manifest.json"
    m.write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
    with pytest.raises(SystemExit):
        _guard()(m, "b" * 64)


def test_guard_reads_the_legacy_sha256_key(tmp_path):
    """Pre-fix manifests carry the digest under `sha256`, not
    `checkpoint_sha256`. Without the fallback the guard silently passes and the
    arrays are replaced."""
    m = tmp_path / "extract_manifest.json"
    m.write_text(json.dumps({"sha256": "a" * 64}))
    with pytest.raises(SystemExit):
        _guard()(m, "b" * 64)


def test_guard_is_called_before_any_array_is_written():
    """Ordering IS the fix. After the writes the guard leaves the directory
    strictly worse than no guard: new features under the old manifest."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "experiments" / "EVAL"
           / "extract_features.py").read_text()
    call = src.index("refuse_foreign_checkpoint(prior_manifest")
    first_save = min(src.index("np.save("), src.index("np.savez("))
    assert call < first_save, "guard must precede the first array write"


def _build_mod():
    import importlib.util, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_extract_jobs", root / "scripts/build_extract_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_interleaved_prefix_stops_spanning_every_family():
    """The docstring used to promise 'every prefix spans all three'. It cannot:
    round-robin exhausts the SHORTEST family first. Res2P 50, Res34P 215, QCD 70
    means only the first 150 of 335 files carry all three, and past ~210 the
    list is Res34P alone. extract_features.py's keep() head-slices to --max-jets
    BEFORE striding, so that promise is exactly what its class balance rests on."""
    m = _build_mod()
    per = {f: hi - lo + 1 for f, lo, hi in m.FAMILIES}
    assert len(set(per.values())) > 1, (
        "this test is only meaningful while the families are unequal; if they "
        "are ever equalised the prefix really does span all three throughout")
    files = m.interleaved_files().split()
    fam_of = lambda p: p.rsplit("/", 1)[1].rsplit("_", 1)[0]
    balanced_files = min(per.values()) * len(per)
    assert len(set(map(fam_of, files[:balanced_files]))) == len(per)
    assert len(set(map(fam_of, files))) == len(per)
    # the tail is a single family -- the concrete form of the broken promise
    assert len(set(map(fam_of, files[-50:]))) == 1


def test_the_balanced_prefix_covers_the_cap_actually_in_use():
    m = _build_mod()
    assert m.balanced_prefix_jets() > 2_000_000, (
        "the matrix extractions cap at 2,000,000; if the balanced prefix ever "
        "falls below that, every cached feature matrix is class-biased")


def test_a_cap_past_the_balanced_prefix_is_refused():
    """A head slice beyond that point is class-biased while still looking like
    a uniform cut -- it must fail loudly at build time, not quietly at analysis."""
    m = _build_mod()
    too_big = m.balanced_prefix_jets() + 1
    with pytest.raises(SystemExit, match="balanced prefix"):
        m.build("mtx-l188-s1", "L188", 188, "/ckpt", gpu=False, max_jets=too_big)


def test_window_mode_is_exempt_from_the_prefix_guard():
    """Window mode applies the selection at load, so --max-jets counts SURVIVORS
    drawn from the whole stream, not leading rows of it. Guarding it would refuse
    a cut that is not a head slice at all."""
    m = _build_mod()
    m.build("mtx-l162-s1b", "L162", 162, "/ckpt", gpu=False,
            max_jets=m.balanced_prefix_jets() + 1, window=True)
