"""logits.npy computed from a feature cache, without re-extracting.

The load-bearing claim is that features.npy IS `fc`'s input, so fc(features)
reproduces the model's logits EXACTLY rather than approximately. That is
asserted here against a real head, not assumed from the docstring.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _mod():
    s = importlib.util.spec_from_file_location(
        "lff", ROOT / "experiments" / "EVAL" / "logits_from_features.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _ckpt(tmp_path, k, embed=128, hidden=512, name="net.pt"):
    """A checkpoint whose fc mirrors what weaver ACTUALLY builds.

    This fixture used to emit a FLAT Sequential keyed `mod.fc.0` / `mod.fc.3`,
    which is what I assumed the head looked like. weaver's ParT builds one
    nested Sequential per fc_params entry:

        fc = Sequential( Sequential(Linear(embed, hidden), ReLU, Dropout),
                         Linear(hidden, k) )

    so the real keys are `mod.fc.0.0.*` and `mod.fc.1.*` -- confirmed against
    the live model summary in the mtx-makeweight-rand2 job log. The flat
    fixture agreed with the code's single-level regex, so the pair was
    self-consistent and wrong together, and eval-anomaly died on the real
    checkpoint with "features are 128-d, the head expects 512": the regex had
    matched only the second Linear, whose output width is k, so the
    num-classes check still passed.
    """
    torch.manual_seed(0)
    inner = torch.nn.Sequential(
        torch.nn.Linear(embed, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1))
    out = torch.nn.Linear(hidden, k)
    fc = torch.nn.Sequential(inner, out)
    state = {}
    for pn in ("weight", "bias"):
        state[f"mod.fc.0.0.{pn}"] = getattr(inner[0], pn).detach().clone()
        state[f"mod.fc.1.{pn}"] = getattr(out, pn).detach().clone()
    state["mod.blocks.0.attn.in_proj_weight"] = torch.zeros(3 * embed, embed)
    p = tmp_path / name
    torch.save(state, p)
    return p, fc.eval()


def _cache(tmp_path, n, embed, ckpt_sha=None):
    d = tmp_path / "feat"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    F = rng.normal(size=(n, embed)).astype(np.float32)
    np.save(d / "features.npy", F)
    if ckpt_sha:
        (d / "extract_manifest.json").write_text(
            json.dumps({"checkpoint_sha256": ckpt_sha}))
    return d, F


def test_logits_reproduce_the_head_exactly(tmp_path):
    """fc(features) == the model's own output. This is the whole justification
    for not re-extracting: features.npy is fc's INPUT by construction, because
    extract_features takes it with a forward PRE-hook on mod.fc."""
    lff = _mod()
    k, embed, n = 17, 128, 500
    ck, fc = _ckpt(tmp_path, k, embed)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    d, F = _cache(tmp_path, n, embed, sha)
    assert lff.main(["--features", str(d), "--checkpoint", str(ck),
                     "--num-classes", str(k)]) == 0
    got = np.load(d / "logits.npy")
    with torch.no_grad():
        want = fc(torch.from_numpy(F)).numpy()
    assert got.shape == (n, k)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_dropout_is_inert_so_the_result_is_deterministic(tmp_path):
    lff = _mod()
    k, embed, n = 17, 128, 300
    ck, _ = _ckpt(tmp_path, k, embed)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    d, _ = _cache(tmp_path, n, embed, sha)
    lff.main(["--features", str(d), "--checkpoint", str(ck), "--num-classes", str(k)])
    a = np.load(d / "logits.npy").copy()
    lff.main(["--features", str(d), "--checkpoint", str(ck),
              "--num-classes", str(k), "--overwrite"])
    np.testing.assert_array_equal(a, np.load(d / "logits.npy"))


def test_a_mislabelled_k_is_refused(tmp_path):
    """A K=17 head declared as K=43 would write a valid-looking array under the
    wrong arm's name."""
    lff = _mod()
    ck, _ = _ckpt(tmp_path, 17, 128)
    d, _ = _cache(tmp_path, 100, 128,
                  hashlib.sha256(ck.read_bytes()).hexdigest())
    with pytest.raises(SystemExit):
        lff.main(["--features", str(d), "--checkpoint", str(ck), "--num-classes", "43"])


def test_a_foreign_checkpoint_is_refused(tmp_path):
    """Applying one model's head to another's features is the exact confound
    the checkpoint digest exists to catch."""
    lff = _mod()
    ck, _ = _ckpt(tmp_path, 17, 128)
    d, _ = _cache(tmp_path, 100, 128, "d" * 64)     # cache says another checkpoint
    with pytest.raises(SystemExit):
        lff.main(["--features", str(d), "--checkpoint", str(ck), "--num-classes", "17"])


def test_existing_logits_are_not_clobbered(tmp_path):
    lff = _mod()
    ck, _ = _ckpt(tmp_path, 17, 128)
    d, _ = _cache(tmp_path, 100, 128, hashlib.sha256(ck.read_bytes()).hexdigest())
    np.save(d / "logits.npy", np.zeros((100, 17), np.float32))
    with pytest.raises(SystemExit):
        lff.main(["--features", str(d), "--checkpoint", str(ck), "--num-classes", "17"])


def test_a_sidecar_manifest_records_the_provenance(tmp_path):
    """These logits did NOT come from the extraction pass; a reader must be
    able to tell."""
    lff = _mod()
    ck, _ = _ckpt(tmp_path, 17, 128)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    d, _ = _cache(tmp_path, 100, 128, sha)
    lff.main(["--features", str(d), "--checkpoint", str(ck), "--num-classes", "17"])
    side = json.loads((d / "logits_manifest.json").read_text())
    assert side["source"] == "logits_from_features.py"
    assert side["checkpoint_sha256"] == sha
    assert side["num_classes"] == 17 and side["n_jets"] == 100


def test_both_linears_are_recovered_from_the_nested_head(tmp_path):
    """The bug this file missed: a single-level `fc.<i>` pattern finds only the
    OUTER Linear, and the resulting head is k-wide on the output so every other
    guard still passes. Pin the shapes directly."""
    lff = _mod()
    ckpt, fc = _ckpt(tmp_path, k=17)
    net, in_dim = lff.head_from_checkpoint(ckpt, 17)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    assert len(lin) == 2, f"recovered {len(lin)} Linear(s) from a two-Linear head"
    assert in_dim == 128, f"head reports {in_dim}-d input, the trunk emits 128"
    assert lin[0].weight.shape == (512, 128)
    assert lin[1].weight.shape == (17, 512)
    assert sum(isinstance(m, torch.nn.ReLU) for m in net) == 1, "ReLU, not GELU"

    x = torch.randn(64, 128)
    with torch.no_grad():
        assert torch.allclose(net(x), fc.eval()(x), atol=1e-6)
