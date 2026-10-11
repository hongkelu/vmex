"""Pick, for every force-front run family (runs/batch/ff<name>[-cN]), the saved step with the lowest driver QS error
across all its continuations, symlink its coils to coils/ours_ff<name>.json and write runs/ours_eval_list.txt
(G families first).  Tied/stalled continuations that only went uphill are thereby ignored.
    python prep_ours_list.py [--latest]   (--latest: newest step of the newest continuation; --fam <regex>: families, default the figure families)"""
import json, os, re, sys
from pathlib import Path
P = Path(__file__).resolve().parent
latest = "--latest" in sys.argv
FAM = re.compile(sys.argv[sys.argv.index("--fam") + 1] if "--fam" in sys.argv else r"^ff(25GA|25GBf|30GA|25SC)-")
fams = {}
for d in (P / "runs/batch").glob("ff*"):
    if d.is_dir() and not d.name.startswith("restart_"):
        fams.setdefault(re.sub(r"-c\d+$", "", d.name), []).append(d)
names = []
for fam, dirs in sorted(fams.items()):
    if not FAM.match(fam):
        continue
    feasible_start = "G" in fam or "SC" in fam   # G/SC families start from feasible sets; the older box families do not
    cands = []
    for d in dirs:
        mm = d / "metrics.jsonl"
        if not mm.exists():
            continue
        for l in mm.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l); f = d / f"coils.step{r['step']}.json"
            if f.exists() and (r["step"] > 0 or feasible_start):
                cn = int(m.group(1)) if (m := re.search(r"-c(\d+)$", d.name)) else 0
                cands.append((r["qa"], cn, r["step"], f))
    if not cands:
        continue
    best = max(cands, key=lambda c: (c[1], c[2])) if latest else min(cands, key=lambda c: c[0])
    link = P / "coils" / f"ours_{fam}.json"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(best[3])
    old = P / "runs/lpqa_vacuum_eval_12x12" / f"ours_{fam}"
    if old.exists():
        import shutil; shutil.rmtree(old)   # always re-solve the chosen step
    names.append(f"ours_{fam}"); print(f"ours_{fam:16s} qa={best[0]:.3e} {best[3].parent.name} step {best[2]}")
names.sort(key=lambda n: (0 if re.match(r"ours_ff(25|30)G", n) else 1, n))
(P / "runs/ours_eval_list.txt").write_text("\n".join(names) + "\n"); print(len(names), "sets")
