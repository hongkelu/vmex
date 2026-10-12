"""One-pass status (few metadata calls): evaluation coverage, run progress from log tails, then the figure."""
import os, re, sys, time, subprocess
from pathlib import Path
P = Path(__file__).resolve().parents[1]
t0 = time.time()
def rows(d):
    base = P / f"runs/lpqa_vacuum_eval_{d}"
    return {e.name for e in os.scandir(base) if e.is_dir() and os.path.exists(os.path.join(e.path, "row.json"))} if base.exists() else set()
r12, r16 = rows("12x12"), rows("16x16")
cand = [l.strip() for l in open(P / "runs/envelope_candidates.txt")]
print(time.strftime("%H:%M"), f"12x12 archive rows {sum(n.startswith('hurfront') for n in r12)}, candidates done {sum(n in r12 for n in cand)}/{len(cand)}, ours rows {sum(n.startswith('ours_') for n in r12)}, 16x16 rows {len(r16)}  [{time.time()-t0:.0f}s]")
runs = {}
for e in os.scandir(P / "runs/batch"):
    m = re.fullmatch(r"(ff(?:25|30)G[A-Za-z]*-F[0-9.]+w?|ff25SC-F12)(-c\d+)?", e.name)
    if m and e.is_dir():
        fam = m.group(1); cn = int(m.group(2)[2:]) if m.group(2) else 0
        if fam not in runs or cn > runs[fam][0]: runs[fam] = (cn, e.name)
for fam, (cn, name) in sorted(runs.items()):
    log = P / "runs/batch" / f"{name}.log"
    if not log.exists(): continue
    with open(log, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4000)); tail = f.read().decode(errors="ignore")
    steps = re.findall(r"\[step (\d+)\] QA=([0-9.e+-]+).*?aspect=([0-9.]+)", tail)
    msg = re.findall(r'"message": "([^"]*)"', tail)
    if steps: s = steps[-1]; print(f"  {name:18s} step {s[0]:>3s} QA={float(s[1]):.3e} A={s[2]}" + (f"  [{msg[-1]}]" if msg else ""))
    else: print(f"  {name:18s} (no step yet)" + (f"  [{msg[-1]}]" if msg else ""))
print(f"[{time.time()-t0:.0f}s]")
if "--fig" in sys.argv:
    out = subprocess.run([sys.executable, str(P / "analysis/fig_force_qs.py"), str(P / "analysis/fig_force_qs.pdf"), "--scan"], capture_output=True, text=True)
    print([l for l in out.stdout.splitlines() if l.startswith("wrote")] or out.stderr[-500:])
print(f"[{time.time()-t0:.0f}s total]")
