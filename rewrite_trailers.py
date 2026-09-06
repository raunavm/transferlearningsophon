#!/usr/bin/env python3
"""Rewrite every commit from 4f164c7..main that carries a Co-Authored-By
trailer: same tree, same author/committer/dates, same (rewritten) parent,
message without the trailer. The 14 tags pointing into that range move to
the rewritten commits (annotated tag objects are re-created with the same
tagger and message). Pure object surgery: the working tree and index are
never touched, so the unstaged deletions under figures/ etc. are safe.

    python3 rewrite_trailers.py            # dry run: prints the plan, moves nothing
    python3 rewrite_trailers.py --apply    # moves main and the tags, writes
                                           # experiments/commit_rewrite_2026-09-06.csv
Then:
    git push --force origin main
    git push --force origin --tags
The old main tip is kept at refs/backup/main-before-trailer-rewrite.
"""
import csv
import pathlib
import re
import subprocess
import sys

APPLY = "--apply" in sys.argv
OLDEST_TRAILER_COMMIT = "747730a5f1a6c840aea74e1dccb5a40f834c36a3"


def git(*args, inp=None):
    return subprocess.run(["git", *args], input=inp, capture_output=True, check=True).stdout


base = git("rev-parse", f"{OLDEST_TRAILER_COMMIT}~1").decode().strip()
old_tip = git("rev-parse", "refs/heads/main").decode().strip()
commits = git("rev-list", "--reverse", "--topo-order", f"{base}..refs/heads/main").decode().split()

new_of = {}
changed = 0
for c in commits:
    raw = git("cat-file", "commit", c)
    head, _, msg = raw.partition(b"\n\n")
    lines = head.split(b"\n")
    parents = [l.split()[1].decode() for l in lines if l.startswith(b"parent ")]
    assert len(parents) == 1, (c, parents)  # linear history only
    assert not any(l.startswith(b"gpgsig") for l in lines), c
    new_lines = [b"parent " + new_of.get(parents[0], parents[0]).encode() if l.startswith(b"parent ") else l
                 for l in lines]
    kept = [l for l in msg.split(b"\n") if not l.startswith(b"Co-Authored-By:")]
    new_msg = b"\n".join(kept).rstrip(b"\n") + b"\n"
    changed += new_msg != msg
    obj = b"\n".join(new_lines) + b"\n\n" + new_msg
    new_of[c] = git("hash-object", "-t", "commit", "-w", "--stdin", inp=obj).decode().strip()

for c, n in new_of.items():  # trees must be untouched
    assert git("rev-parse", f"{c}^{{tree}}") == git("rev-parse", f"{n}^{{tree}}"), c

tag_moves = []
for t in git("tag").decode().split():
    target = git("rev-list", "-n1", t).decode().strip()
    if target not in new_of:
        continue
    kind = git("cat-file", "-t", t).decode().strip()
    if kind == "tag":
        raw = git("cat-file", "tag", t)
        raw = re.sub(rb"^object [0-9a-f]{40}", b"object " + new_of[target].encode(), raw, count=1, flags=re.M)
        newref = git("hash-object", "-t", "tag", "-w", "--stdin", inp=raw).decode().strip()
    else:
        newref = new_of[target]
    tag_moves.append((t, kind, git("rev-parse", t).decode().strip(), newref))

new_tip = new_of[old_tip]
print(f"{base[:7]}..main: {len(commits)} commits, {changed} messages change; new tip {new_tip[:7]}")
print(f"{len(tag_moves)} tags move: " + " ".join(f"{t}({k})" for t, k, _, _ in tag_moves))

if not APPLY:
    print("dry run: no ref moved (objects are written but unreachable; git gc removes them)")
    sys.exit(0)

git("update-ref", "refs/backup/main-before-trailer-rewrite", old_tip)
git("update-ref", "refs/heads/main", new_tip, old_tip)
for t, _, oldref, newref in tag_moves:
    git("update-ref", f"refs/tags/{t}", newref, oldref)
out = pathlib.Path("experiments/commit_rewrite_2026-09-06.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["old_commit", "new_commit", "subject"])
    for c in commits:
        w.writerow([c, new_of[c], git("log", "-1", "--format=%s", c).decode().strip()])
print(f"applied: main -> {new_tip[:7]}; mapping written to {out} (commit it, then force-push main and --tags)")
