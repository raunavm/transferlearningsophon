"""CI for experiments/FT/smoke_checks.py, the assertions the fine-tuning jobs
run in-pod.

The load-log check is the one that matters: weaver loads --load-model-weights
with strict=False and only LOGS what did not match, so a checkpoint with a
different key layout leaves a randomly initialised trunk, trains it, and
reports a plausible accuracy. The log line is the only evidence. These tests
run the parser against weaver 0.4.17's verbatim message, rendered through
weaver's own log format with a tqdm carriage-return line in front of it,
because that is what the file it reads actually contains.

Run:  python3 -m pytest tests/test_smoke_checks.py -v
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEAD_KEYS = ["mod.fc.0.0.weight", "mod.fc.0.0.bias", "mod.fc.1.weight", "mod.fc.1.bias"]


@pytest.fixture(scope="module")
def sc():
    spec = importlib.util.spec_from_file_location(
        "smoke_checks_under_test", ROOT / "experiments" / "FT" / "smoke_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(missing, unexpected, prefix=""):
    """weaver 0.4.17 train.py model_setup, through its '[%(asctime)s] %(levelname)s:
    %(message)s' formatter, with tqdm's \\r line in front as it appears in stdout.log."""
    return (prefix +
            "[2026-09-06 02:24:11,003] INFO: Model initialized with weights from "
            "/data/results/mtx/mtx-r16q1-s2/net_best_epoch_state.pt\n"
            f" ... Missing: {missing!r}\n"
            f" ... Unexpected: {unexpected!r}\n"
            "[2026-09-06 02:24:11,120] INFO: Using loss function "
            "CrossEntropyLoss() with options {}\n")


def test_parse_reads_both_key_lists_out_of_weavers_own_message(sc):
    missing, unexpected = sc.parse_load_log(_log(HEAD_KEYS, []))
    assert missing == HEAD_KEYS and unexpected == []


def test_parse_survives_the_tqdm_carriage_returns_around_it(sc):
    missing, _ = sc.parse_load_log(_log(HEAD_KEYS, [], prefix="0it [00:00, ?it/s]\r"))
    assert missing == HEAD_KEYS


def test_a_log_without_the_block_fails_rather_than_passing_silently(sc, tmp_path):
    p = tmp_path / "train.log"
    p.write_text("[2026-09-06 02:24:11,003] INFO: Epoch #0 training\n")
    with pytest.raises(SystemExit, match="did not run"):
        sc.check_load_log(str(p))


def test_the_head_only_case_passes(sc, tmp_path):
    p = tmp_path / "ok.log"
    p.write_text(_log(HEAD_KEYS, []))
    sc.check_load_log(str(p))                      # no SystemExit


def test_a_trunk_key_that_did_not_load_is_a_hard_stop(sc, tmp_path):
    """The failure the check exists for: a checkpoint of a different layout."""
    p = tmp_path / "bad.log"
    p.write_text(_log(HEAD_KEYS + ["mod.blocks.0.attn.in_proj_weight"], ["mod.embed.0.weight"]))
    with pytest.raises(SystemExit, match="trunk keys did not load"):
        sc.check_load_log(str(p))


def test_nothing_excluded_is_also_a_hard_stop(sc, tmp_path):
    """Empty Missing means the head loaded too, so --exclude-model-weights
    did not match and the fine-tune starts from the donor's head."""
    p = tmp_path / "noexcl.log"
    p.write_text(_log([], []))
    with pytest.raises(SystemExit, match="nothing was excluded"):
        sc.check_load_log(str(p))


def test_head_width_reads_the_last_fc_layer(sc, tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "net.pt"
    torch.save({"mod.fc.0.0.weight": torch.zeros(512, 128),
                "mod.fc.0.0.bias": torch.zeros(512),
                "mod.fc.1.weight": torch.zeros(18, 512),      # K + 1 = 17 + 1
                "mod.fc.1.bias": torch.zeros(18)}, ckpt)
    assert sc.head_width(str(ckpt)) == 18


def test_head_width_refuses_a_checkpoint_with_no_head(sc, tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "trunkonly.pt"
    torch.save({"mod.blocks.0.attn.in_proj_weight": torch.zeros(384, 128)}, ckpt)
    with pytest.raises(SystemExit, match="no mod.fc."):
        sc.head_width(str(ckpt))
