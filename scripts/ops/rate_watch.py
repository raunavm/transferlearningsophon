"""Compare a fresh rate-search scan with the last reported state; print the rows that need action (exit 1 if any)."""
import json, pathlib, sys
sys.path.insert(0, "/Users/raunavmendiratta/transferlearningsophon/experiments/FT")
import rate_search as RS
scan, state_f = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
seen = json.loads(state_f.read_text()) if state_f.exists() else {}
act = []
for r in RS.evaluate(scan):
    key = f"{r['task']} {r['n']} {r['kind']}"
    need = (r["status"].startswith("settled") and r["best"] != r["recorded"]) or r["status"].startswith("edge") \
        or r["status"] == "every rate diverged" or r["rerun_diverged"] or r["issues"]
    sig = json.dumps([r["status"], r["best"], r["recorded"], r["rerun_diverged"], r["issues"]])
    if need and seen.get(key) != sig:
        act.append(RS.fmt(r) + "".join(f"\n    ISSUE {e}" for e in r["issues"]))
    seen[key] = sig
state_f.write_text(json.dumps(seen, indent=1))
print("\n".join(act))
sys.exit(1 if act else 0)
