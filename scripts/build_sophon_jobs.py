#!/usr/bin/env python3
"""The released Sophon checkpoint's missing inference (PI 2026-11-23, LEDGER SH-13: "the inference and probes for the
released Sophon checkpoint ... queued after part A's last cell starts"; CA-16: inference and probes only, no
fine-tuning). Its stored features already cover jc1, jc2bal, top and quark/gluon (sophon_spec,
build_jc2bal_jobs.py); its features on the TEST2M mass jets exist as sophon-public/features_massreg (rows 0-1,999,999
equal the mass MLP's jets, checked 2026-11-23). What needs a GPU is two feature sets our networks have and it lacks:

  bs3-sophon-x-raunav (one GPU pod)  the checkpoint (sha256 pinned), then
      jc2bal_pairs train and test (build_jc2bal_jobs.extract_spec's call, the pairs of Figs. 5, 10, 11) into
          SOPHON_ROOT/sophon-released/jc2bal_pairs/<split>/released;
      the X->bb and X->bs staged splits and the QCD evaluation jets (build_xbx_jobs.x_spec's call, extract_xbx.py
          --arm L188: features and the 188-class head's scores) into EX_ROOT/sophon-released;
      fit_xbx.py model for each signal (zero-shot, linear probe, the transfer head of three seeds; Table 2, Fig. 7)
          into FITS_IHEP/<sig>, and the v5 linear refit into FITS_V5/<sig> -- in the same pod, because fit_xbx reads
          the head's first layer from the checkpoint the manifest names, which the pod downloaded;
  bs3-sophon-x-smoke-raunav (one GPU pod)  the same preamble and SMOKE_JETS jets of the pairs test set into SMOKE,
      with peak memory and seconds, to size the pod and project the GPU-hours.
A split whose manifest exists is skipped and a fit that exists is never redone, so a retried pod does only what is
missing. Nothing is trained but the transfer heads and probes the protocol fits.

    python3 scripts/build_sophon_jobs.py --tag mtx-sN.NN --mem 24Gi
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "sophon"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R5 = _load("build_refit_v5_jobs", "scripts/build_refit_v5_jobs.py")
BT = _load("build_boot_tests_jobs", "scripts/build_boot_tests_jobs.py")
JB, XB = R5.JB, _load("build_xbx_jobs", "scripts/build_xbx_jobs.py")
MODEL, TAG = JB.SOPHON_MODEL, JB.SOPHON_TAG
CK = f"/workspace/sophon/{TAG}.pt"
PAIRS_OUT = f"{JB.SOPHON_ROOT}/{MODEL}/jc2bal_pairs"
SIGNALS = ("xbb", "xbs")
SMOKE, SMOKE_JETS = "/data/results/eval/bs3_smoke/sophon", 50_000
# the pair sets near the volume (coordinator 2026-11-24, option a): the X->bx pod read 1,350 jets/s on ucm-fiona01, the
# 64Gi pair pod 276 jets/s on chi-dgx-node04; general-GPU nodes at UC Merced and SDSC (Prometheus, 2026-11-24), the
# grid's bad nodes still excluded
NEAR = ("ucm-fiona01.ucmerced.edu", "clu-fiona2.ucmerced.edu") + tuple(
    f"k8s-haosu-{i:02d}.sdsc.optiputer.net" for i in (2, 3, 5, 7, 9, 11, 16, 17, 19, 20, 21, 23, 24)) + tuple(
    f"ry-gpu-{i:02d}.sdsc.optiputer.net" for i in (6, 7, 11, 12, 13, 14, 15, 16))
PAIR_JETS = 1_127_621 + 2_116_561      # pairs_train, pairs_test (RUNS jc2bal-x rows)
LINE = 88


def preamble(tag: str) -> str:
    return f"""set -euo pipefail
{JB._start(tag)}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le {LINE} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{JB.GPU_CHECK}
mkdir -p /workspace/sophon
retry python3 -c "import urllib.request; urllib.request.urlretrieve('{JB.SOPHON_URL}', '{CK}')" || exit 1
GOT=$(sha256sum {CK} | cut -d' ' -f1)
[ "${{GOT}}" = "{JB.SOPHON_SHA256}" ] || {{ echo "FATAL: {CK} sha256 ${{GOT}}"; exit 42; }}
# pairs OUT MAXJETS FILES...: build_jc2bal_jobs.extract_spec's extraction with the released checkpoint (sophon_spec's
# flags); a split with its manifest is done
pairs () {{
  OUT=$1; MAXJ=$2; shift 2
  [ -f ${{OUT}}/{TAG}/manifest.json ] && {{ echo "${{OUT}} done"; return 0; }}
  python3 experiments/EVAL/extract_v2.py --run-dir /workspace/sophon --rung L188 --num-classes 188 \\
    --num-reg 0 --checkpoints {CK} --data-config {JB.CFG} --max-jets ${{MAXJ}} \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test "$@" --out ${{OUT}} || exit 1
}}"""


def xbx() -> str:
    """build_xbx_jobs.x_spec's extractions and fit_spec's fits for the released checkpoint, every signal."""
    need = " ".join([f"{XB.STAGE}/pairs/{s}/DONE" for s in SIGNALS] + [f"{XB.STAGE}/{s}/eval/DONE" for s in SIGNALS]
                    + [f"{XB.STAGE}/qcd/eval/DONE"])
    lines = [f'for F in {need}; do [ -f "${{F}}" ] || {{ echo "FATAL: ${{F}} absent: stage first"; exit 42; }}; done',
             "xbx () {",
             f"  OUT={XB.EX_ROOT}/{MODEL}/$1; shift",
             f"  [ -f ${{OUT}}/{TAG}/manifest.json ] && {{ echo \"${{OUT}} done\"; return 0; }}",
             f"  python3 experiments/XBS/extract_xbx.py --arm L188 --run-dir /workspace/sophon --rung L188 \\",
             f"    --num-classes 188 --num-reg 0 --checkpoints {CK} --data-config {XB.CFG} \\",
             "    --observers jet_pt jet_eta jet_sdmass --no-pooled --data-test \"$@\" --out ${OUT} || exit 1",
             "}"]
    for s in SIGNALS:
        lines += [f"xbx {s}/train $(ls {XB.STAGE}/pairs/{s}/train_*.parquet | sort)",
                  f"xbx {s}/val {XB.STAGE}/pairs/{s}/val.parquet",
                  f"xbx {s}/eval $(ls {XB.STAGE}/{s}/eval/*.parquet | sort)"]
    lines.append(f"xbx qcd/eval $(ls {XB.STAGE}/qcd/eval/*.parquet | sort)")
    window = f"{XB.window_arg(XB.MX)} --mx {XB.MX:g}"
    for s in SIGNALS:
        lines += [f"[ -f {XB.FITS_IHEP}/{s}/{MODEL}.json ] || python3 experiments/XBS/fit_xbx.py model --root {XB.EX_ROOT} "
                  f"--model {MODEL} --signals {s} --checkpoint {TAG} {window} --out {XB.FITS_IHEP}/{s}",
                  f"[ -f {R5.XBX_V5}/{s}/{MODEL}.json ] || python3 experiments/XBS/fit_xbx.py model --root {XB.EX_ROOT} "
                  f"--model {MODEL} --signals {s} --checkpoint {TAG} {window} --linear-only --grid v5 --out {R5.XBX_V5}/{s}"]
    return "\n".join(lines)


def x_spec(tag: str, mem: str, part: str = "all") -> str:
    """part all: one pod for both; xbx or pairs: one of them (bs3-sophon-xbx-raunav, bs3-sophon-pairs-raunav), so
    that each job's GPU-hours are known before the next is launched."""
    bal = JB.BAL
    steps = {"pairs": [BT.peak(f"bash -c 'source /tmp/sophon_fns.sh; pairs {PAIRS_OUT}/train 0 {bal}/pairs_train.parquet'"),
                       BT.peak(f"bash -c 'source /tmp/sophon_fns.sh; pairs {PAIRS_OUT}/test 0 "
                               + " ".join(f"{bal}/{p}" for p in JB.PAIRS_TEST_PARTS) + "'")]
             if part != "all" else [f"pairs {PAIRS_OUT}/train 0 {bal}/pairs_train.parquet",
                                    f"pairs {PAIRS_OUT}/test 0 " + " ".join(f"{bal}/{p}" for p in JB.PAIRS_TEST_PARTS)],
             "xbx": [xbx()]}
    body = "\n".join([preamble(tag)] + (["declare -f pairs > /tmp/sophon_fns.sh"] if part == "pairs" else [])
                      + [s for k in ("pairs", "xbx") if part in ("all", k) for s in steps[k]]
                      + ['echo "SOPHON X COMPLETE"', ""])
    name = {"all": "bs3-sophon-x-raunav", "xbx": "bs3-sophon-xbx-raunav", "pairs": "bs3-sophon-pairs-raunav"}[part]
    what = {"all": "pair and X->bx features, the transfer-head fits", "xbx": "X->bx features and the transfer-head fits",
            "pairs": "the class-pair features"}[part]
    text = R5.indexed(name, body, 1, 1, mem, "4", f"the released Sophon checkpoint: {what} (SH-13)", gpu=True)
    return near(text) if part == "pairs" else text


def near(text: str) -> str:
    """The pod kept to NEAR's nodes (a hostname term beside the region's)."""
    old = '                values: ["us-west"]\n'
    if text.count(old) != 1:
        raise SystemExit("FATAL: the spec's region term changed")
    nodes = ", ".join(f'"{n}"' for n in NEAR)
    return text.replace(old, old + f"              - key: kubernetes.io/hostname\n                operator: In\n"
                                   f"                values: [{nodes}]\n")


def smoke_spec(tag: str, mem: str) -> str:
    cmd = (f"pairs {SMOKE}/jc2bal_pairs_test {SMOKE_JETS} {JB.BAL}/{JB.PAIRS_TEST_PARTS[0]}")
    body = "\n".join([preamble(tag), "T0=$(date +%s)", cmd,
                      f'echo "SMOKE {SMOKE_JETS} jets in $(( $(date +%s) - T0 )) s"', 'echo "SOPHON X SMOKE COMPLETE"', ""])
    return R5.indexed("bs3-sophon-x-smoke-raunav", body, 1, 1, mem, "4",
                      "the released Sophon checkpoint, smoke: a slice of the pairs test set, its time and memory",
                      gpu=True)


# ------------------------------------------------------------------ the CPU probes (experiments/EVAL/probe_sophon.py)
OUT = "/data/results/eval/bs3_sophon"
MASS_VIEW = f"{OUT}/{MODEL}/jc2/test/{TAG}"                  # the TEST2M mass jets' features, v2 layout
OBS_DIR = "/data/results/eval/v2_head/mtx-l188-s1/test/best70"   # build_head_jobs.OBS_MODEL's head outputs
CPU_OF = {"mass": R5.FAMILY_CPU["atlas"], "pairfit": R5.FAMILY_CPU["linear"], "recall": R5.FAMILY_CPU["linear"],
          "mlp": R5.FAMILY_CPU["mlp"], "mlp2": R5.FAMILY_CPU["mlp"]}
THREADS = {"mass": "8", "pairfit": "8", "recall": "1", "mlp": "4", "mlp2": "4"}
PAIRS_DONE = [f"{PAIRS_OUT}/{s}/{TAG}/manifest.json" for s in ("train", "test")]


def cpu_steps(part: str) -> list[str]:
    ps = "python3 experiments/EVAL/probe_sophon.py"
    return {
        # Table 4: the view, then atlas_resolution.py as R5.atlas_spec runs it (grid v2, global centre, 200 epochs), no
        # mass output and no mass_v3.json (Sophon has neither, as the partitions)
        "mass": [BT.peak(f"{ps} massview --out {MASS_VIEW}"),
                 f"[ -f {OUT}/atlas_res_v3/{MODEL}.json ] || " + BT.peak(
                     f"python3 experiments/EVAL/atlas_resolution.py --model {MODEL} --observers {OBS_DIR} "
                     f"--features {MASS_VIEW} --arm {TAG} --out-dir {OUT}/atlas_res_v3 --grid v2 --centre global "
                     f"--mlp-max-epochs {R5.MASS_MLP_EPOCHS}")],
        "mlp": [BT.peak(f"{ps} mlp --datasets jc1 top qg --out {OUT}/fits_mlp_bench")],
        "mlp2": [BT.check_inputs(PAIRS_DONE), BT.peak(f"{ps} mlp --datasets jc2bal --out {OUT}/fits_mlp_jc2bal")],
        "recall": [BT.peak(f"{ps} recall --out {OUT}/perclass_recall")],
        "pairfit": [BT.check_inputs(PAIRS_DONE),
                  BT.peak(f"{ps} pairs --out {OUT}/fits_v5_pairs --probs-out {OUT}/fits_v5_pairs_probs")]}[part]


def cpu_spec(tag: str, part: str, mem: str) -> str:
    """bs3-sophon-<part>-raunav: one CPU pod on its fit family's CPU model (PF-1), its threads the family's."""
    body = f"""set -euo pipefail
{JB._start(tag)}
{R5.threads(THREADS[part])}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le {LINE} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
""" + "\n".join(cpu_steps(part)) + f'\necho "SOPHON {part.upper()} COMPLETE"\n'
    text = R5.indexed(f"bs3-sophon-{part}-raunav", body, 1, 1, mem, THREADS[part] if part != "recall" else "2",
                      f"the released Sophon checkpoint's CPU probes, {part} (SH-13)")
    return R5.pin_cpu(text, CPU_OF[part])


CPU_MEM = {"mass": "8Gi", "mlp": "7Gi", "mlp2": "12Gi", "recall": "4Gi", "pairfit": "8Gi"}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mem", required=True)
    ap.add_argument("--smoke-mem", default="24Gi")
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in (("bs3-sophon-x-smoke-raunav", smoke_spec(a.tag, a.smoke_mem)),
                       ("bs3-sophon-x-raunav", x_spec(a.tag, a.mem)),
                       ("bs3-sophon-xbx-raunav", x_spec(a.tag, a.mem, "xbx")),
                       ("bs3-sophon-pairs-raunav", x_spec(a.tag, a.mem, "pairs")),
                       *((f"bs3-sophon-{p}-raunav", cpu_spec(a.tag, p, CPU_MEM[p])) for p in ("mass", "mlp", "mlp2", "recall", "pairfit"))):
        p = OUT_DIR / f"job-{name}.yaml"
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
