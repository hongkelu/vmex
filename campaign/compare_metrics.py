"""Compare a reproduced run's metrics.jsonl with the reference, step by step.

    python <repo>/campaign/compare_metrics.py <new metrics.jsonl> <reference metrics.jsonl> [--steps N]

Prints the relative difference of every numeric metric (timings excluded) at each common step, and the worst.
"""
import argparse
import json


def rows(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a run still writing its last line
            continue
        out[r["step"]] = r
    return out


p = argparse.ArgumentParser()
p.add_argument("new")
p.add_argument("reference")
p.add_argument("--steps", type=int, default=5)
args = p.parse_args()
new, ref = rows(args.new), rows(args.reference)
skip = ("seconds", "elapsed", "peak_gpu", "trials", "failed", "evaluations", "linearizations", "stage_")
for step in sorted(set(new) & set(ref))[: args.steps]:
    a, b = new[step], ref[step]
    diffs = {k: abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in b
             if k in a and isinstance(b[k], (int, float)) and not isinstance(b[k], bool) and k != "step"
             and not any(s in k for s in skip)}
    worst = max(diffs, key=diffs.get)
    print(f"step {step}: qa {a.get('qa'):.10e} vs {b.get('qa'):.10e}, worst relative difference {diffs[worst]:.2e} ({worst})")
