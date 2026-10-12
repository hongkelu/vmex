"""Preflight for a jobs file: does every run START feasible under its own hard coil rows?

    python preflight.py jobs_X.txt [jobs_Y.txt ...]        (uwplasma-env, CPU, login node is fine)

For each active line, the starting coils are the --coils file (fresh start, src "-") or the newest coils.stepN.json of
the restart source directory. Their metrics (force_front_metrics.py, SIMSOPT convention, cached in
runs/preflight_metrics.json) are compared with the line's COIL_CASE_OVERRIDES: total length, curvature, MSC,
coil-coil and coil-surface distance, force (driver units = N/m at <B> = 0.2271 T x 19.86). An infeasible start makes
SLSQP run feasibility restoration first, which on 2026-10-10/11 cost 2-50x in QS error, so such lines are reported.
Exit status 1 if any line starts infeasible.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
CACHE = P / "runs/preflight_metrics.json"
FORCE_SCALE = 19.86   # driver force units per (N/m at <B> = 0.2271 T): (1.012 T / 0.2271 T)^2
ROWS = [  # (label, metric key, override key, sense, metric -> driver units)
    ("L", "total_length", "TOTAL_LENGTH_LIMIT", "<=", lambda v: v),
    ("kappa", "max_kappa", "CURVATURE_LIMIT", "<=", lambda v: v),
    ("msc", "max_msc", "MSC_LIMIT", "<=", lambda v: v),
    ("cc", "min_cc", "COIL_DISTANCE_LIMIT", ">=", lambda v: v),
    ("cs", "min_cs", "COIL_SURFACE_DISTANCE_LIMIT", ">=", lambda v: v),
    ("F", "max_force", "FORCE_LIMIT", "<=", lambda v: v * FORCE_SCALE),
]


def start_coils(extra, src):
    if src not in ("-", ""):
        d = P / src
        steps = sorted(d.glob("coils.step*.json"), key=lambda f: int(re.search(r"step(\d+)", f.name).group(1)))
        if steps:
            return steps[-1]
        if (d / "coils.json").exists():
            return d / "coils.json"
    m = re.search(r"--coils\s+(\S+)", extra)
    return (P / m.group(1)) if m else None


def main(files):
    lines = []
    for f in files:
        for l in Path(f).read_text().splitlines():
            if l.strip() and not l.startswith("#") and l.count("|") == 5:
                lines.append(l.split("|"))
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = {}
    for name, _, _, _, extra, src in lines:
        c = start_coils(extra, src)
        if c is not None and str(c) not in cache:
            todo[str(c)] = c
    if todo:
        tmp = P / "runs/preflight_tmp.json"
        tmp.unlink(missing_ok=True)
        args = [f"k{i}={c}" for i, c in enumerate(todo.values())]
        subprocess.run([sys.executable, str(P / "force_front_metrics.py"), str(tmp), *args], check=True,
                       stdout=subprocess.DEVNULL)
        got = json.load(open(tmp))
        for i, c in enumerate(todo):
            cache[c] = got[f"k{i}"]
        json.dump(cache, open(CACHE, "w"), indent=1)
    bad = 0
    print(f"{'run':16s} {'start':34s} violations (value vs limit, relative)")
    for name, _, _, ovr, extra, src in lines:
        c = start_coils(extra, src)
        if c is None:
            print(f"{name:16s} {'(no coils: seed fit)':34s} -"); continue
        m, o = cache[str(c)], (json.loads(ovr) if ovr not in ("-", "") else {})
        viol = []
        for lab, mk, ok, sense, conv in ROWS:
            if ok not in o or o[ok] is None:
                continue
            v, lim = conv(m[mk]), float(o[ok])
            rel = (v - lim) / abs(lim) if sense == "<=" else (lim - v) / abs(lim)
            if rel > (0.03 if lab == "F" else 1e-6):   # F: fixed 19.86 conversion is good to ~1-2 %, the driver row is the authority
                viol.append(f"{lab} {v:.4g} vs {lim:.4g} (+{100 * rel:.1f}%)")
        bad += bool(viol)
        print(f"{name:16s} {str(c.relative_to(P))[-34:]:34s} {'; '.join(viol) if viol else 'ok'}")
    print(f"{bad} of {len(lines)} lines start infeasible")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
