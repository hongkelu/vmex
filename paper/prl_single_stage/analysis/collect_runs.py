"""Status table of every run under paper/runs/batch (and runs/ss_vac): latest accepted step from metrics.jsonl.

    python collect_runs.py [--json out.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs"


def latest(run):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        return None
    r = rows[-1]
    r0 = rows[0]
    return dict(run=str(run.relative_to(ROOT)), steps=len(rows) - 1, qa0=r0.get("qa"), qa=r.get("qa"),
                qa_best=min(x.get("qa", float("inf")) for x in rows),
                iota_min=r.get("min_abs_iota"), aspect=r.get("aspect"), beta=r.get("beta"), R0=r.get("major_radius_m"),
                bn=r.get("bn"), pb=r.get("pressure_balance"), K=r.get("sheet_current"), redl=r.get("redl_mismatch"),
                cs=r.get("coil_surface_distance_m"), slack=r.get("coil_minimum_scaled_slack"),
                step_s=r.get("step_seconds"), elapsed_h=(r.get("elapsed_seconds") or 0) / 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    out = []
    for run in sorted(list((ROOT / "batch").glob("*/")) + list((ROOT / "ss_vac").glob("*/"))):
        if (run / "metrics.jsonl").exists():
            row = latest(run)
            if row:
                out.append(row)
    fmt = "{run:28s} {steps:>5} {qa0:>9.2e} {qa:>9.2e} {qa_best:>9.2e} {iota_min:>6.3f} {aspect:>6.3f} {beta:>6.4f} {bn:>8.1e} {K:>8.1e} {slack:>8.4f} {step_s:>6.0f}s {elapsed_h:>5.2f}h"
    print(f"{'run':28s} {'steps':>5} {'qa0':>9} {'qa':>9} {'qa_best':>9} {'iota':>6} {'A':>6} {'beta':>6} {'B.n':>8} {'K':>8} {'slack':>8} {'t/step':>7} {'elapsed':>6}")
    for r in out:
        safe = {k: (v if v is not None else float("nan")) for k, v in r.items()}
        print(fmt.format(**safe))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
