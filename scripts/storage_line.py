"""The /data fill line at which a job's in-pod guard refuses to write (`[ "${USED}" -le <line> ] || exit 42`).

GENERAL is the standing rule (~/.claude/CLAUDE.md: above 85% no write over 1 GB without the PI). FINISHING is the
PI's line of 2026-11-09 for the finishing work only: the remaining equal-stopping cells and any rerun they trigger,
the read-outs, the test suite, and the gallery and paper renders. No builder emits a line above MAX; at MAX the PI
decides (PI 2026-11-09: "If /data reaches 90%, stop and ask me").
"""
GENERAL, FINISHING, MAX = 85, 90, 90


def check(line: int) -> int:
    if not GENERAL <= line <= MAX:
        raise SystemExit(f"FATAL: storage line {line}%: between {GENERAL}% and {MAX}% (PI 2026-11-09)")
    return line


def add_argument(ap) -> None:
    ap.add_argument("--storage-line", type=int, default=GENERAL,
                    help=f"the in-pod storage guard's line in percent (default {GENERAL}; {FINISHING} for the "
                         f"finishing work, PI 2026-11-09; never above {MAX})")
