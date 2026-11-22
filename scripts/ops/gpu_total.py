# GPU-hours of the rate search + reruns since 2026-11-20 23:00 UTC (PI's ceiling count), JetClass-II 1e6 apart.
# Prometheus Running time of ft-budget-fixed-(sweep|edge|rate|smoke|head) pods, 1 GPU per pod. Reads only.
import json, re, sys, collections, pathlib, subprocess, datetime
K = pathlib.Path("/Users/raunavmendiratta/transferlearningsophon/experiments/FT/k8s/budget")
t0 = datetime.datetime(2026, 11, 20, 23, 0, tzinfo=datetime.timezone.utc)
mins = int((datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds() // 60)
q = ('sum by (pod) (sum_over_time(kube_pod_status_phase{namespace="cms-ml",phase="Running",'
     f'pod=~"ft-budget-fixed-(sweep|edge|rate|smoke|head).*"}}[{mins}m:5m]))')
r = json.loads(subprocess.run(["curl", "-s", "--max-time", "120", "https://prometheus.nrp-nautilus.io/api/v1/query", "--data-urlencode", f"query={q}"], capture_output=True, text=True).stdout)
cells = {}
for f in K.glob("job-ft-budget-fixed-*-raunav.yaml"):
    job = f.name.removeprefix("job-").removesuffix(".yaml")
    for m in re.finditer(r"^ *(\d+)\) TASK=(\w+); INIT=([\w-]*);.*?N=(\d+);", f.read_text(), re.M):
        cells[(job, int(m.group(1)))] = (m.group(2), int(m.group(4)))
tot = collections.defaultdict(float)
for x in r["data"]["result"]:
    p = x["metric"]["pod"]; h = float(x["value"][1]) * 5 / 60
    mm = re.match(r"(.*-raunav)-(\d+)-[a-z0-9]{5}$", p)
    c = cells.get((mm.group(1), int(mm.group(2))), ("?", 0))
    tot["jc2b 1e6" if c == ("jc2b", 1_000_000) else ("rerun" if "-rate" in mm.group(1) else "search")] += h
print(f"since 20 Nov 23:00 ({mins/60:.1f} h):", {k: round(v, 1) for k, v in tot.items()},
      "search+rerun", round(tot["search"] + tot["rerun"], 1))
