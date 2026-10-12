"""Phase-1 overnight job lists (2026-10-11): every run under the full filter at converged resolution, best feasible start.

    python build_phase1.py          -> jobs_P1_A.txt, jobs_P1_B.txt, prints the start table

Constraints (driver defaults now FORCE_POINTS 1024, segment distances >= 512): kappa < 12, MSC <= 6.0, cc >= 0.083,
cs >= 0.166, each coil <= 6.5 m, total length <= 25 / 30 m, max force <= threshold (driver units = N/m at <|B|> =
0.2271 T x 19.856). Plasma pinned (R0, A in [5.99, 6.01], iota >= 0.41) or aspect free (A in [2, 10]).
Start = lowest 12x12 QS error among candidates (12x12-verified cloud sets, our best steps coils/ours_ff*.json) that
satisfy every row at 1024 points (runs/force_front_metrics_1024.json); a force-only excess <= FORCE_SLACK is allowed
(our old steps were pinned at the 128-point force bound). OBJECTIVE_SCALE = 1/QS(start), ftol 1e-12.
"""
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
M = json.load(open(P / "runs/force_front_metrics_1024.json"))
if (P / "runs/force_front_metrics_1024_ours.json").exists():
    M.update(json.load(open(P / "runs/force_front_metrics_1024_ours.json")))
OURS = json.load(open(P / "runs/ours_best_qa.json")) if (P / "runs/ours_best_qa.json").exists() else {}
EV = P / "runs/lpqa_vacuum_eval_12x12"
FSCALE = 19.856
FORCE_SLACK = 0.03
LIM = dict(kappa=12.0, msc=6.0, cc=0.083, cs=0.166, coil=6.5)
tpl = next(l for l in (P / "jobs_batch14.txt").read_text().splitlines() if l.startswith("ff25GA-F10|")).split("|")
base_ovr = json.loads(tpl[3])
for k in ("FORCE_LIMIT", "OBJECTIVE_SCALE"):
    base_ovr.pop(k, None)
base_ovr.update(CURVATURE_LIMIT=LIM["kappa"], MSC_LIMIT=LIM["msc"], COIL_DISTANCE_LIMIT=LIM["cc"],
                COIL_SURFACE_DISTANCE_LIMIT=LIM["cs"], CONDUCTOR_RADIUS=0.05, LENGTH_LIMIT=LIM["coil"],
                OPTIMIZER_FTOL=1e-12, CURVATURE_MARGIN=0.0, MSC_MARGIN=0.0, DISTANCE_MARGIN=0.0, LENGTH_MARGIN=0.0)
extra = re.sub(r"--coils \S+", "", tpl[4]); extra = re.sub(r"--restart \S+", "", extra)
extra = re.sub(r"--design-step-scale \S+", "--design-step-scale 0.5", extra).strip()


def qa(name):
    f = EV / name / "row.json"
    if not f.exists():
        return OURS.get(name, {}).get("qa")   # our best steps: driver QS until their 12x12 row exists
    r = json.load(open(f))
    return r["qa_total"] if r["bn"] < r["bn_initial"] else None


def feasible(m, L, thr):
    if (m["max_kappa"] > LIM["kappa"] or m["max_msc"] > LIM["msc"] + 1e-9 or m["min_cc"] < LIM["cc"] or m["min_cs"] < LIM["cs"]
            or max(m["lengths"]) > LIM["coil"] or m["total_length"] > L):
        return False
    return m["max_force"] / 1e3 <= thr * (1 + FORCE_SLACK)


def start_for(L, thr):
    best = None
    for n, m in M.items():
        if n.startswith("gilpar"):
            continue
        q = qa(n)
        if q is not None and feasible(m, L, thr) and (best is None or q < best[1]):
            best = (n, q, m)
    return best


def line(name, L, thr, aspect, start):
    n, q, m = start
    o = dict(base_ovr, TOTAL_LENGTH_LIMIT=float(L), FORCE_LIMIT=round(thr * 1e3 * FSCALE, 1),
             OBJECTIVE_SCALE=float(f"{1 / q:.3g}"), ASPECT_RANGE=aspect)
    return "|".join([name, tpl[1], tpl[2], json.dumps(o), f"{extra} --coils coils/{n}.json", "-"])


A, B, rows = [], [], []
for thr in (8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12):
    s = start_for(25, thr)
    if s: A.append(line(f"P25A-F{thr}", 25, thr, [5.99, 6.01], s)); rows.append(("P25A", thr, s))
for thr in (8, 9, 10, 11, 12, 13, 14):
    s = start_for(30, thr)
    if s: B.append(line(f"P30A-F{thr}", 30, thr, [5.99, 6.01], s)); rows.append(("P30A", thr, s))
for thr in (10, 11, 12):
    s = start_for(25, thr)
    if s: B.append(line(f"P25F-F{thr}", 25, thr, [2.0, 10.0], s)); rows.append(("P25F", thr, s))
hdr = "# Phase 1 (build_phase1.py): full filter at converged resolution, best feasible start, OBJECTIVE_SCALE=1/QS0\n"
(P / "jobs_P1_A.txt").write_text(hdr + "\n".join(A) + "\n")
(P / "jobs_P1_B.txt").write_text(hdr + "\n".join(B) + "\n")
for fam, thr, (n, q, m) in rows:
    print(f"{fam} F{thr:<5} start {n:22s} QS {q:.2e}  F {m['max_force']/1e3:6.2f}  L {m['total_length']:5.2f}  "
          f"maxl {max(m['lengths']):4.2f}  kappa {m['max_kappa']:5.2f}  msc {m['max_msc']:4.2f}  cc {m['min_cc']:.4f}")
print(f"A: {len(A)} lines, B: {len(B)} lines")
