"""Jobs that keep the output layer of the 16 runs the paper fine-tunes, and read the mass
head and the mass sculpting from it (plan 2026-10-29, A6 and job J4: the mass head's own
output was kept for no v2 model, and there are no per-jet class scores for QCD test jets).

Per run, one GPU job head-x-<stem>-raunav, on any product but the RTX 3090 (as
scripts/build_linprobe_jobs.py's):
  1. experiments/EVAL/extract_v2.py --keep-head-outputs at best70 on the first 2,000,000
     TEST2M jets, the linear probes' JetClass-II test jets in their order, through
     configs/data/JetClassII_massreg_obs.yaml (native labels; genjet_sdmass, genjet_pt):
     every jet's K logits, the mass output of the two mass-output label sets, and the
     observers, to HEAD_ROOT/<model>/test/best70/head_outputs.npz. No features: the frozen
     features of these jets are the linear probes' (LP_ROOT/<model>/jc2/test/best70).
  2. experiments/EVAL/mass_sculpting.py: QCD jet_sdmass before and after cuts on
     1 - p_QCD, to HEAD_ROOT/<model>/sculpting.json. First: it needs only the head outputs,
     so a failure of step 3 does not lose it.
  3. experiments/EVAL/mass_resolution.py: the uncorrected soft-drop mass, the frozen-feature
     probes on those features (median-centred, and mean-centred beside them), the mass
     head (mass-output runs) and their response in class groups and pT / m_gen bins, to
     HEAD_ROOT/<model>/mass_v3.json.
Each step leaves an existing output as it is, so a retried pod resumes.

Audit 2026-11-07, M4 (PI: "sculpting for the three flavor-boundary controls, same evaluation
and same figure and table as the label sets"): per flavour-boundary run of
configs/arms/v2_paper_runs.txt (FLAV_RUNS), one GPU job head-x-<stem>-raunav at FLAV_PIN,
steps 1 and 2 only, the extraction at rung 'none' (outside the contraction tree) and the
sculpting with --label-set <arm> (its QCD node from flavour_pair_map.v2.csv).
M2 (signal efficiency and R at each cut, R50 and JSD50): one CPU job head-sculpt-v3-raunav,
mass_sculpting.py at FLAV_PIN on the head outputs of all SCULPT_V3 runs, to
HEAD_ROOT/sculpting_v3/<model>_sculpting.json (the old sculpting.json / curves_v2 stay as they
are). Apply it after the flavour-boundary head jobs have finished; before that it stops with
exit 42.

Per untrained trunk init-s1..s3 (plan job J4: "16 runs + 3 init"), one CPU job
head-mass-init-s<N>-raunav: step 3 alone on its linear-probe features
(LP_ROOT/init-s<N>/jc2/test/init), with the observers of OBS_MODEL's head outputs (the
same jets; mass_resolution checks the label digest). Apply them after head-x-l188-s1
has finished; before that they stop with exit 42.

Plan 2026-11-08 step 2 (decision of 2026-11-08, flagged to the PI: on the class-balanced test set weighted to
the natural QCD mix the fine-tuned sculpting was noise, Kish n_eff ~14,400): the fine-tuned JetClass-II networks
at 1e6 on the same TEST2M jets, for their sculpting (experiments/EVAL/sculpting_ft.py) and class pairs
(experiments/FT/roc_points.py pairs_ft). Two Indexed GPU jobs at FT_HEAD_PIN, one cell of ft_head_cells() per
index: ft-head-x-raunav the 27 pretrained cells (launchable now), ft-head-x-scratch-raunav the three from-scratch
seeds (each index refuses, exit 42, until its cell is DONE; apply it after the reruns); on any 24 GB+ GPU
(build_mtx_launch.V2_FT_GPU, as the fine-tuning jobs):
extract_v2.py --keep-head-outputs of the cell's net_best_epoch_state.pt, in the fine-tuning network
(extract_features.build_model: ParT_sophon_arch_mtx.py, fc_params [(512, 0.1)], 188 outputs) at rung L188,
through CONFIG, whose inputs, preprocessing and selection are those of the fine-tuning configs
(JetClassII_L188_noweight.yaml; JetClassII_base.yaml for its test jets), on the first N_TEST jets, onto the
pod's disk; the labels checked against OBS_MODEL's head outputs (the same jets, or exit 42); head_outputs.npz,
then manifest.json (with the checkpoint's sha256), into FT_HEAD_ROOT/<model>/. A model with its manifest is
skipped, never overwritten; an index whose cell is not DONE refuses (exit 42).

    python3 scripts/build_head_jobs.py [--pin-not-yet-tagged] [--only mtx-l188-s1 ...] [--check-only]
    python3 scripts/build_head_jobs.py --pin-not-yet-tagged --ft-head   # the fine-tuned networks' two jobs alone
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "head"
# mtx-s2.18 (4f92021, the linear-probe refit) was tagged before this builder and the scripts it
# runs were committed, so the jobs pin the next tag
PIN = "mtx-s2.19"
HEAD_ROOT = "/data/results/eval/v2_head"
CONFIG = "configs/data/JetClassII_massreg_obs.yaml"
N_TEST = 2_000_000
# the test files only: the v1 head extractions of 2M TEST2M jets ran at 48Gi
# (build_extract_jobs.build_v1err); the 88Gi of the linear-probe extraction is for the
# one-file 1M-jet training subset
X_MEM = "48Gi"
# the 16 runs the paper fine-tunes: four label sets x 3, the two mass-output sets x 2
RUNS = tuple(f"mtx-{a}-s{s}" for a in ("l188", "l162", "r42q1", "r16q1") for s in (1, 2, 3)) + tuple(
    f"mtx-{a}-s{s}" for a in ("l162mass", "r16q1mass") for s in (1, 2))
# audit 2026-11-07 M2/M4: the flavour-boundary controls (2 runs each, v2_paper_runs.txt) and the
# sculpting re-run; mass_sculpting.py's --label-set and signal efficiency need a new tag
FLAV_PIN = "mtx-s2.80"
FLAV_RUNS = tuple(f"mtx-{a}-s{s}" for a in ("flavf0", "flavf1", "flavf1r") for s in (1, 2))
SCULPT_V3 = RUNS + FLAV_RUNS
SCULPT_V3_DIR = f"{HEAD_ROOT}/sculpting_v3"
# What the tag must carry for what the job runs.
NEEDED = {"experiments/EVAL/extract_v2.py": "--keep-head-outputs",
          "experiments/EVAL/mass_resolution.py": "OUT_JSON, OUT_RESIDUALS = \"mass_v3.json\"",
          "experiments/EVAL/mass_sculpting.py": "WORKING_POINTS = (0.1, 0.01)",
          CONFIG: "   - genjet_pt\n"}
# plan 2026-11-08 step 2: the fine-tuned JetClass-II networks on the same jets
FT_HEAD_PIN = "mtx-s2.86"
FT_HEAD_ROOT = "/data/results/eval/v2_ft_head"
FT_HEAD_NAME = "ft-head-x-raunav"                       # the 27 pretrained cells
FT_HEAD_SCRATCH_NAME = "ft-head-x-scratch-raunav"       # the three from-scratch seeds, after the reruns
FT_HEAD_PARALLELISM = 5
# the head outputs refused unless the whole output layer loads (extract_v2.head_loaded)
FT_HEAD_NEEDED = {"experiments/EVAL/extract_v2.py": "def head_loaded(", CONFIG: "   - genjet_pt\n"}
FLAV_NEEDED = {"experiments/EVAL/extract_v2.py": "--keep-head-outputs",
               "experiments/EVAL/mass_sculpting.py": '"--label-set"',
               "configs/labelmaps/flavour_pair_map.v2.csv": "FLAV_F1R", CONFIG: "   - genjet_pt\n"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")
BX, BF = LP.BX, LP.BF
BJ = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")       # the Indexed GPU template
BFB = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")

HEAD = """          OUT={root}/{model}/test
          python3 experiments/EVAL/extract_v2.py \\
              --run-dir {run_dir} --rung {rung} --num-classes {k} --num-reg {reg} \\
              --checkpoints best70 --data-config {config} --max-jets {n} \\
              --observers --feature-classes --prefix-features 0 \\
              --head-prefix 0 --no-anomaly-rows --diag-stride 0 --no-pooled --keep-head-outputs \\
              --data-test {files} \\
              --out ${{OUT}} || halt
          python3 experiments/EVAL/mass_sculpting.py --head ${{OUT}}/best70 \\
            --out {root}/{model}/sculpting.json{label_set} || halt
"""
FEAT = """          FEAT={lp_root}/{model}/jc2/test/best70
          [ -f ${{FEAT}}/manifest.json ] || {{ echo "FATAL: no linear-probe features at ${{FEAT}}"; exit 42; }}
"""
BODY = FEAT + HEAD + """          python3 experiments/EVAL/mass_resolution.py --observers ${{OUT}}/best70 \\
            --features best70=${{FEAT}}{head} --out {root}/{model} || halt
          echo "head outputs, mass and sculpting of {model} done"
"""
FLAV_BODY = HEAD + """          echo "head outputs and sculpting of {model} done"
"""
# the untrained trunks have no output layer: the frozen-feature probes alone
INIT_BODY = """          FEAT={lp_root}/{model}/jc2/test/init
          OBS={root}/{obs}/test/best70
          [ -f ${{FEAT}}/manifest.json ] || {{ echo "FATAL: no linear-probe features at ${{FEAT}}"; exit 42; }}
          [ -f ${{OBS}}/manifest.json ] || {{ echo "FATAL: no head outputs at ${{OBS}}: apply after its head job"; exit 42; }}
          python3 experiments/EVAL/mass_resolution.py --observers ${{OBS}} \\
            --features init=${{FEAT}} --out {root}/{model} || halt
          echo "mass of {model} done"
"""
OBS_MODEL = "mtx-l188-s1"
INIT_MEM = "32Gi"     # what the v2 fit jobs ran mass_resolution in (build_linprobe_jobs.fit_spec)


def models() -> list[tuple[str, str, str, int, int]]:
    """(model, run directory, rung, classes, regression outputs) of the 16 runs."""
    out = [(m, run, rung, k, reg) for m, run, rung, k, reg, _c in LP.models() if m in RUNS]
    assert [m for m, *_ in out] == list(RUNS), out
    return out


def flav_models() -> list[tuple[str, str, str, int, int]]:
    """(model, run directory, arm, classes, regression outputs) of the flavour-boundary runs."""
    arm = {run: a for run, a, *_ in BX.v2_runs()}
    out = [(m, run, rung, k, reg) for m, run, rung, k, reg, _c in LP.models() if m in FLAV_RUNS]
    # outside the contraction tree, so extract_v2.py runs them at rung 'none'
    assert [m for m, *_ in out] == list(FLAV_RUNS) and {r for _m, _u, r, _k, _g in out} == {"none"}, out
    return [(m, run, arm[m], k, reg) for m, run, _r, k, reg in out]


def planned_bytes(ms=None) -> int:
    """What every job writes to /data: head_outputs.npz of N_TEST jets -- K float16
    logits, the float32 mass output, an int64 row, an int16 label and five float32
    observers per jet (the JSON files are kilobytes)."""
    return sum(N_TEST * (2 * k + 4 * reg + 8 + 2 + 5 * 4) for _m, _r, _g, k, reg in (ms or models()))


def ft_planned_bytes() -> int:
    """head_outputs.npz of every fine-tuned cell (188 float16 logits, no mass output)."""
    return len(ft_head_cells()) * N_TEST * (2 * 188 + 8 + 2 + 5 * 4)


def spec(model, run, rung, k, reg) -> str:
    body = BODY.format(lp_root=LP.LP_ROOT, model=model, root=HEAD_ROOT, run_dir=f"{BX.V2_ROOT}/{run}",
                       rung=rung, k=k, reg=reg, config=CONFIG, n=N_TEST, files=BF.test2m_list(),
                       head=" --head best70=${OUT}/best70" if reg else "", label_set="")
    return LP.off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name=f"head-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=PIN, body=body,
        mem=X_MEM, cpu="4", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def flav_spec(model, run, arm, k, reg) -> str:
    body = FLAV_BODY.format(model=model, root=HEAD_ROOT, run_dir=f"{BX.V2_ROOT}/{run}", rung="none", k=k,
                            reg=reg, config=CONFIG, n=N_TEST, files=BF.test2m_list(),
                            label_set=f" --label-set {arm}")
    return LP.off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name=f"head-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=FLAV_PIN, body=body,
        mem=X_MEM, cpu="4", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def sculpt_v3_spec() -> str:
    """mass_sculpting.py on every SCULPT_V3 run's head outputs, into SCULPT_V3_DIR."""
    arm = {m: a for m, _r, a, _k, _g in flav_models()}
    body = "".join(f'          [ -f {HEAD_ROOT}/{m}/test/best70/head_outputs.npz ] || {{ echo "FATAL: no head '
                   f'outputs for {m}: apply after its head job"; exit 42; }}\n' for m in SCULPT_V3)
    body += "".join(f"          python3 experiments/EVAL/mass_sculpting.py --head {HEAD_ROOT}/{m}/test/best70 \\\n"
                    f"            --out {SCULPT_V3_DIR}/{m}_sculpting.json"
                    + (f" --label-set {arm[m]}" if m in arm else "") + " || halt\n" for m in SCULPT_V3)
    body += f'          echo "sculpting of the {len(SCULPT_V3)} runs done"\n'
    return BX.V1ERR_TEMPLATE.format(name="head-sculpt-v3-raunav", image=BX.IMAGE, pin=FLAV_PIN, body=body,
                                    mem=INIT_MEM, cpu="4", gpu_req="", gpu_check="", node_exclude="")


def init_models() -> list[str]:
    return [m for m, *_ in LP.models() if m.startswith("init-")]


def init_spec(model) -> str:
    body = INIT_BODY.format(lp_root=LP.LP_ROOT, model=model, root=HEAD_ROOT, obs=OBS_MODEL)
    return BX.V1ERR_TEMPLATE.format(name=f"head-mass-{model}-raunav", image=BX.IMAGE, pin=PIN,
                                    body=body, mem=INIT_MEM, cpu="8", gpu_req="", gpu_check="",
                                    node_exclude="")


def ft_head_cells() -> list[tuple[str, str]]:
    """(model, cell) of the JetClass-II fine-tuning at 1e6: each paper run (v2_paper_runs.txt order) from the long
    budget where it has a cell (build_ft_budget_jobs.long_cells, long_ctrl_cells; mlst_results.merge_budget's
    rule), else the fixed budget (jc2b_cells); then the three from-scratch seeds of the full schedule
    (long_scratch_full_cells, budget_long_fulldecay)."""
    pick = lambda cells: {c["init"]: c["out"] for c in cells if c["init"] and c["n"] == BFB.LONG_N}  # noqa: E731
    cell = pick(BFB.jc2b_cells("1e-3", "1e-3")) | pick(BFB.long_cells("jc2b", "1e-3", "1e-3")
                                                        + BFB.long_ctrl_cells("1e-3"))
    runs = BFB.paper_runs()
    assert set(runs) <= set(cell) and len(runs) == 27, sorted(set(runs) - set(cell))
    return [(f"mtx-{r}", cell[r]) for r in runs] + [
        (f"scratch-s{c['s']}", c["out"]) for c in BFB.long_scratch_full_cells("jc2b", "1e-3")]


FT_HEAD_BODY = """set -euo pipefail
{start}
IDX=${{JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}}
case ${{IDX}} in
{cases}  *) echo "FATAL: no cell for index ${{IDX}}"; exit 42;;
esac
OUT={root}/${{MODEL}}
REF={ref}/manifest.json
[ -f ${{OUT}}/manifest.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f ${{CELL}}/DONE ] && [ -f ${{CELL}}/net_best_epoch_state.pt ] || {{ echo "FATAL: ${{CELL}} not DONE"; exit 42; }}
[ -f ${{REF}} ] || {{ echo "FATAL: no ${{REF}}"; exit 42; }}
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${{USED}}" -le 85 ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}
{gpu_check}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={cpu} OPENBLAS_NUM_THREADS={cpu} MKL_NUM_THREADS={cpu}
X=/workspace/x; rm -rf ${{X}}
python3 experiments/EVAL/extract_v2.py --run-dir ${{CELL}} --rung L188 --num-classes 188 --num-reg 0 \\
  --checkpoints ${{CELL}}/net_best_epoch_state.pt --data-config {config} --max-jets {n} \\
  --observers --feature-classes --prefix-features 0 --head-prefix 0 --no-anomaly-rows --diag-stride 0 --no-pooled \\
  --keep-head-outputs --data-test {files} --out ${{X}} || exit 1
H=${{X}}/net_best_epoch_state
python3 -c "import json, sys; a, b = (json.load(open(f))['head_outputs'] for f in sys.argv[1:]); sys.exit(a['n'] != {n} or a['label188_sha256'] != b['label188_sha256'])" ${{H}}/manifest.json ${{REF}} \\
  || {{ echo "FATAL: ${{MODEL}}: not the jets of ${{REF}}"; exit 42; }}
mkdir -p ${{OUT}}
cp ${{H}}/head_outputs.npz ${{OUT}}/head_outputs.npz.tmp && mv ${{OUT}}/head_outputs.npz.tmp ${{OUT}}/head_outputs.npz
cp ${{H}}/manifest.json ${{OUT}}/manifest.json.tmp && mv ${{OUT}}/manifest.json.tmp ${{OUT}}/manifest.json
echo "FT HEAD ${{MODEL}} DONE"
"""


def ft_head_jobs() -> dict[str, list[tuple[str, str]]]:
    """{job name: its (model, cell)}: the pretrained cells, and the from-scratch ones apart, which wait for the reruns."""
    cells = ft_head_cells()
    return {FT_HEAD_NAME: [c for c in cells if c[0].startswith("mtx-")],
            FT_HEAD_SCRATCH_NAME: [c for c in cells if c[0].startswith("scratch-")]}


def ft_head_spec(name: str = FT_HEAD_NAME, tag: str = FT_HEAD_PIN, parallelism: int = FT_HEAD_PARALLELISM) -> str:
    """One Indexed GPU job of the module docstring (plan 2026-11-08 step 2), on the cells ft_head_jobs gives it."""
    cells = ft_head_jobs()[name]
    cases = "".join(f"  {i}) MODEL={m}; CELL={c};;\n" for i, (m, c) in enumerate(cells))
    body = FT_HEAD_BODY.format(start=BJ._start(tag), cases=cases, root=FT_HEAD_ROOT,
                               ref=f"{HEAD_ROOT}/{OBS_MODEL}/test/best70", gpu_check=BJ.GPU_CHECK, cpu=4,
                               config=CONFIG, n=N_TEST, files=BF.test2m_list())
    which = "from scratch" if name == FT_HEAD_SCRATCH_NAME else "pretrained"
    text = BJ._indexed(name, body, len(cells), min(parallelism, len(cells)), True, X_MEM, "4",
                       f"the fine-tuned JetClass-II networks at 1e6 ({which}) on the TEST2M jets of the head "
                       "sculpting (plan 2026-11-08 step 2). GPU")
    launch = BJ._launch()
    for old, new in (("# GENERATED by scripts/build_jc2bal_jobs.py", "# GENERATED by scripts/build_head_jobs.py"),
                     ("        - { name: data, mountPath: /data }\n",
                      "        - { name: jc2,  mountPath: /jc2, readOnly: true }\n        - { name: data, mountPath: /data }\n"),
                     ("\n      volumes:\n",
                      "\n              - key: nvidia.com/gpu.product\n                operator: In\n"
                      "                values: [" + ", ".join(f'"{g}"' for g in launch.V2_FT_GPU) + "]"
                      "\n      volumes:\n      - name: jc2\n        persistentVolumeClaim:\n"
                      "          claimName: tn-pvc-base-jetclass2\n          readOnly: true\n")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def build(only=None) -> dict[str, str]:
    out = {f"job-head-x-{m.removeprefix('mtx-')}-raunav.yaml": spec(m, run, rung, k, reg)
           for m, run, rung, k, reg in models() if not only or m in only}
    return out | {f"job-head-mass-{m}-raunav.yaml": init_spec(m)
                  for m in init_models() if not only or m in only}


def build_v3(only=None) -> dict[str, str]:
    """The audit 2026-11-07 M2/M4 jobs: the flavour-boundary head outputs and the sculpting re-run."""
    out = {f"job-head-x-{m.removeprefix('mtx-')}-raunav.yaml": flav_spec(m, run, arm, k, reg)
           for m, run, arm, k, reg in flav_models() if not only or m in only}
    return out | ({"job-head-sculpt-v3-raunav.yaml": sculpt_v3_spec()} if not only else {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--ft-head", action="store_true", help="only the fine-tuned networks' job (FT_HEAD_PIN)")
    a = ap.parse_args(argv)
    if a.ft_head:
        BX.verify_pin(FT_HEAD_PIN, list(FT_HEAD_NEEDED), a.pin_not_yet_tagged, FT_HEAD_NEEDED)
        for name, cells in ft_head_jobs().items():
            out = OUT_DIR / f"job-{name}.yaml"
            if not a.check_only:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                out.write_text(ft_head_spec(name))
            print(f"{out.relative_to(ROOT)}: {len(cells)} cells")
        print(f"together they write {ft_planned_bytes() / 1e9:.1f} GB to /data")
        return 0
    BX.verify_pin(PIN, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    BX.verify_pin(FLAV_PIN, list(FLAV_NEEDED), a.pin_not_yet_tagged, FLAV_NEEDED)
    specs = build(a.only) | build_v3(a.only)
    if not a.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (OUT_DIR / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(OUT_DIR.relative_to(ROOT))}; "
          f"they write {planned_bytes() / 1e9:.1f} GB (+ {planned_bytes(flav_models()) / 1e9:.1f} GB "
          f"flavour-boundary) to /data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
