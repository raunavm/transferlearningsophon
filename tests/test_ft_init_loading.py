"""Every fine-tuning init must load into the fine-tuning architecture through
the code path weaver 0.4.17 uses, on CPU, before a GPU is spent on it.

weaver 0.4.17's --load-model-weights (train.py:569-585, read from the PyPI
wheel; the locally installed package is 0.4.16 and differs) is: torch.load,
drop every key matching an --exclude-model-weights pattern with re.match, then
model.load_state_dict(state, strict=False) and LOG the missing and unexpected
keys. Unexpected keys are tolerated, missing keys are tolerated, and neither
stops the run: only experiments/FT/smoke_checks.py load-log turns the log into
a hard stop. A shape mismatch on a matched key raises. That logic is
reproduced here verbatim (_weaver_0417_load) so the three inits the 2026-09-18
plan adds can be checked exactly the way the pod will load them:

  (a) a MASS-OUTPUT checkpoint (ParT_sophon_arch_mass.py, K+1-wide head):
      every non-head tensor loads, the only missing keys are the four mod.fc.*
      the exclusion drops on purpose, nothing is unexpected;
  (b) a MASKED-PARTICLE-MODELLING checkpoint (MPMNet: `trunk.mod.*` +
      `decoder.*`) offered RAW loads NOTHING -- every key unexpected, every
      model key missing -- which is why experiments/FT/mpm_init.py exists;
  (c) the CONVERTED self-supervised init loads the embedding, pair embedding
      and 8 particle-attention blocks (194 tensors) and leaves exactly the 39
      class-attention tensors plus the head fresh, which load-log
      --fresh-prefix ... --expect-fresh 39 then requires.
"""
import importlib.util
import os
import pathlib
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("weaver")

REPO = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE = r"mod\.fc\..*"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _weaver_0417_load(model, model_state, exclude=EXCLUDE):
    """train.py:569-585 of weaver 0.4.17, verbatim logic."""
    key_state = {}
    for k in model_state.keys():
        key_state[k] = True
        for pattern in exclude.split(','):
            if re.match(pattern, k):
                key_state[k] = False
                break
    model_state = {k: v for k, v in model_state.items() if key_state[k]}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    return list(missing), list(unexpected), len(model_state)


@pytest.fixture(scope="module")
def ctx():
    from weaver.utils.data.config import DataConfig
    os.environ["HYBRID_MASS_INSTALLED"] = "1"        # the mass arch refuses to build otherwise
    mtx = _load("mtx_arch", "experiments/MTX/ParT_sophon_arch_mtx.py")
    mass = _load("mass_arch", "experiments/MTX/ParT_sophon_arch_mass.py")
    dc_ft = DataConfig.load(str(REPO / "configs/finetune/TopReference.yaml"), load_observers=False)
    dc_pre = DataConfig.load(str(REPO / "configs/data/JetClassII_base.yaml"), load_observers=False)
    return mtx, mass, dc_ft, dc_pre


def _ft_model(ctx, k=2):
    mtx, _, dc_ft, _ = ctx
    model, _ = mtx.get_model(dc_ft, num_classes=k, fc_params=[(512, 0.1)])
    return model


def _fake_mpm_state(ctx) -> dict:
    """An MPMNet state_dict without building MPMNet (its decoder needs a
    weaver symbol the local 0.4.16 lacks). MPMNet holds the arms' trunk as
    `self.trunk` -- built by the same mtx.get_model with num_classes=None and
    fc_params=None -- so its keys are `trunk.` + the headless trunk's keys,
    plus `decoder.*`. That layout was measured on the 0.4.17 wheel
    (trunk 233 tensors, decoder 63) on 2026-09-18."""
    mtx, _, _, dc_pre = ctx
    trunk, _ = mtx.get_model(dc_pre, num_classes=None, fc_params=None)
    state = {"trunk." + k: v for k, v in trunk.state_dict().items()}
    state["decoder.proj.weight"] = torch.zeros(32, 128)
    state["decoder.head_id.weight"] = torch.zeros(8, 32)
    return state


def test_the_fine_tuning_model_has_the_tensor_counts_the_converter_and_load_log_assume(ctx):
    sd = _ft_model(ctx).state_dict()
    by = {}
    for k in sd:
        by.setdefault(k.split(".")[1], 0)
        by[k.split(".")[1]] += 1
    assert by == {"cls_token": 1, "embed": 17, "pair_embed": 33, "blocks": 144,
                  "cls_blocks": 36, "norm": 2, "fc": 4}
    mi = _load("mpm_init", "experiments/FT/mpm_init.py")
    assert mi.N_TRUNK == 17 + 33 + 144 == 194
    assert mi.N_FRESH == 1 + 36 + 2 == 39


def test_a_mass_output_checkpoint_loads_its_whole_trunk_with_only_the_head_missing(ctx):
    _, mass, _, dc_pre = ctx
    torch.manual_seed(1)
    src, _ = mass.get_model(dc_pre, num_classes=162, fc_params=[(512, 0.1)])
    state = src.state_dict()
    assert state["mod.fc.1.weight"].shape[0] == 163, "K + 1 outputs: the mass node"
    model = _ft_model(ctx)
    missing, unexpected, offered = _weaver_0417_load(model, state)
    assert offered == 233
    assert sorted(missing) == ["mod.fc.0.0.bias", "mod.fc.0.0.weight", "mod.fc.1.bias", "mod.fc.1.weight"]
    assert unexpected == []
    got = model.state_dict()
    assert all(torch.equal(got[k], v) for k, v in state.items() if not k.startswith("mod.fc."))
    # and the same load through the mass twin's OWN K would have failed on shape
    with pytest.raises(RuntimeError, match="size mismatch"):
        _ft_model(ctx).load_state_dict(state, strict=False)


def test_a_raw_self_supervised_checkpoint_loads_nothing_which_is_why_it_is_converted(ctx):
    state = _fake_mpm_state(ctx)
    model = _ft_model(ctx)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    missing, unexpected, offered = _weaver_0417_load(model, state)
    assert len(unexpected) == offered == len(state), "every key is unexpected"
    assert all(k.startswith(("trunk.", "decoder.")) for k in unexpected)
    assert {k for k in before if not k.endswith("num_batches_tracked")} <= set(missing)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "nothing loaded"


def test_the_converted_self_supervised_init_loads_the_trunk_and_leaves_39_fresh(ctx):
    mi = _load("mpm_init", "experiments/FT/mpm_init.py")
    state = _fake_mpm_state(ctx)
    init = mi.convert(state)
    assert len(init) == mi.N_TRUNK == 194
    assert all(k.startswith(mi.KEPT) for k in init)
    assert not any(k.startswith(mi.FRESH) or k.startswith("decoder.") for k in init)
    model = _ft_model(ctx)
    missing, unexpected, offered = _weaver_0417_load(model, init)
    assert offered == 194 and unexpected == []
    fresh = [k for k in missing if k.startswith(mi.FRESH)]
    head = [k for k in missing if k.startswith("mod.fc.")]
    assert len(fresh) == mi.N_FRESH == 39 and len(head) == 4
    assert len(missing) == 43, "nothing else is missing: the whole trunk arrived"
    got = model.state_dict()
    assert all(torch.equal(got[k], v) for k, v in init.items())


def test_the_converter_refuses_a_supervised_checkpoint_and_an_incomplete_trunk(ctx):
    mi = _load("mpm_init", "experiments/FT/mpm_init.py")
    with pytest.raises(SystemExit, match="not a masked-particle"):
        mi.convert({"trunk." + k: v for k, v in _ft_model(ctx).state_dict().items()})
    state = _fake_mpm_state(ctx)
    short = {k: v for k, v in state.items() if not k.startswith("trunk.mod.blocks.7.")}
    with pytest.raises(SystemExit, match="Refusing to write"):
        mi.convert(short)
    state["trunk.mod.registers"] = torch.zeros(8, 128)
    with pytest.raises(SystemExit, match="neither kept nor known-untrained"):
        mi.convert(state)


def test_the_converter_writes_the_init_and_a_provenance_sidecar_the_manifest_embeds(ctx, tmp_path):
    mi = _load("mpm_init", "experiments/FT/mpm_init.py")
    src = tmp_path / "net_epoch-79_state.pt"
    torch.save(_fake_mpm_state(ctx), src)
    out = tmp_path / "mpm-s1_trunk.pt"
    assert mi.main(["--src", str(src), "--out", str(out)]) == 0
    assert len(torch.load(out)) == 194
    sc = _load("smoke_checks", "experiments/FT/smoke_checks.py")
    sc.write_manifest(str(tmp_path / "m.json"), [f"checkpoint={out}", "init=mpm-s1"])
    import json
    rec = json.loads((tmp_path / "m.json").read_text())
    assert rec["checkpoint_provenance"]["source"] == str(src)
    assert rec["checkpoint_provenance"]["kept_tensors"] == 194
    assert len(rec["checkpoint_sha256"]) == 64


# ------------------------------------------------ load-log, the in-pod stop

def _log(missing, unexpected) -> str:
    return (f"Model initialized with weights from x.pt\n ... Missing: {missing!r}\n"
            f" ... Unexpected: {unexpected!r}\n")


def _fresh_keys(ctx, mi):
    return [k for k in _ft_model(ctx).state_dict() if k.startswith(mi.FRESH)]


HEAD = ["mod.fc.0.0.weight", "mod.fc.0.0.bias", "mod.fc.1.weight", "mod.fc.1.bias"]


def test_load_log_passes_the_converted_init_only_with_exactly_39_fresh_tensors(ctx, tmp_path):
    sc = _load("smoke_checks", "experiments/FT/smoke_checks.py")
    mi = _load("mpm_init", "experiments/FT/mpm_init.py")
    fresh = _fresh_keys(ctx, mi)
    assert len(fresh) == 39
    p = tmp_path / "ok.log"
    p.write_text(_log(fresh + HEAD, []))
    sc.check_load_log(str(p), mi.FRESH, 39)
    with pytest.raises(SystemExit, match="trunk keys did not load"):
        sc.check_load_log(str(p))                              # without the flag: strict, as before
    # one trunk tensor short: fatal
    p.write_text(_log(fresh + HEAD + ["mod.blocks.3.attn.in_proj_weight"], []))
    with pytest.raises(SystemExit, match="trunk keys did not load"):
        sc.check_load_log(str(p), mi.FRESH, 39)
    # class-attention tensors were LOADED (fewer fresh than expected): fatal
    p.write_text(_log(fresh[:-2] + HEAD, []))
    with pytest.raises(SystemExit, match="started fresh, expected 39"):
        sc.check_load_log(str(p), mi.FRESH, 39)
    # the raw checkpoint's log: everything unexpected: fatal
    p.write_text(_log(fresh + HEAD + ["mod.embed.input_bn.weight"], ["trunk.mod.cls_token"]))
    with pytest.raises(SystemExit, match="trunk keys did not load"):
        sc.check_load_log(str(p), mi.FRESH, 39)


def test_load_log_cli_requires_the_two_fresh_flags_together(tmp_path):
    sc = _load("smoke_checks", "experiments/FT/smoke_checks.py")
    p = tmp_path / "x.log"
    p.write_text(_log(HEAD, []))
    assert sc.main(["load-log", "--log", str(p)]) == 0
    with pytest.raises(SystemExit, match="go together"):
        sc.main(["load-log", "--log", str(p), "--fresh-prefix", "mod.norm."])
