"""Two converged interior states of one deck (cold, and warm from the deck with its boundary scaled by 1 + 1e-3):
separate roots, or one weak direction?

usage: python chord.py OUT.json   (cwd: a vmex tree; a GPU; env DECK, FTOL as interior_gauge.py)

- fsq (fsqr, fsqz, fsql) along the chord x(alpha) = (1 - alpha) x_cold + alpha x_warm, alpha in [-0.5, 1.5];
- re-solves started at alpha = 0.25, 0.5, 0.75: the distance of each result to the two ends (state max-abs and
  iota), i.e. which state a start between them converges to;
- a repeat cold solve (determinism).
"""
import json
import sys
from pathlib import Path

import numpy as np

import jax

from vmex.core.solver import evaluate_forces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interior_gauge import FIELDS, deck, leaves, solve  # noqa: E402


def gap(a, b):
    """Largest coefficient difference between two states."""
    la, lb = leaves(a), leaves(b)
    return max(float(np.max(np.abs(la[f] - lb[f]))) for f in FIELDS)


def main():
    """Run the chord test and write OUT.json."""
    import dataclasses

    inp = deck()
    cold = solve(inp)
    near = solve(dataclasses.replace(inp, rbc=np.asarray(inp.rbc) * (1 + 1e-3))).state
    warm = solve(inp, near)
    again = solve(inp)
    rt, dev = cold.solver_context, jax.devices()[0]
    out = dict(cold_vs_warm=gap(cold.solution, warm.solution), cold_repeat=gap(cold.solution, again.solution),
               iota_cold_vs_warm=float(np.max(np.abs(np.asarray(cold.wout.iotaf) - np.asarray(warm.wout.iotaf)))),
               chord=[], restarts=[])
    for alpha in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        x = jax.device_put(jax.tree.map(lambda a, b, al=alpha: (1 - al) * np.asarray(a) + al * np.asarray(b),
                                        cold.solution, warm.solution), dev)
        _, res, _ = evaluate_forces(x, rt)
        out["chord"].append(dict(alpha=alpha, fsqr=float(res.fsqr), fsqz=float(res.fsqz), fsql=float(res.fsql)))
        print(f"alpha {alpha:5.2f}: fsq {float(res.fsqr):.3e} {float(res.fsqz):.3e} {float(res.fsql):.3e}", flush=True)
    for alpha in (0.25, 0.5, 0.75):
        start = jax.tree.map(lambda a, b, al=alpha: (1 - al) * np.asarray(a) + al * np.asarray(b),
                             cold.state, warm.state)
        eq = solve(inp, start)
        iota = np.asarray(eq.wout.iotaf)
        row = dict(alpha=alpha, to_cold=gap(eq.solution, cold.solution), to_warm=gap(eq.solution, warm.solution),
                   iota_to_cold=float(np.max(np.abs(iota - np.asarray(cold.wout.iotaf)))),
                   iota_to_warm=float(np.max(np.abs(iota - np.asarray(warm.wout.iotaf)))))
        out["restarts"].append(row)
        print(f"restart from alpha {alpha}: to cold {row['to_cold']:.2e} (iota {row['iota_to_cold']:.1e}), "
              f"to warm {row['to_warm']:.2e} (iota {row['iota_to_warm']:.1e})", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(f"cold vs warm {out['cold_vs_warm']:.2e} (iota {out['iota_cold_vs_warm']:.1e}); "
          f"cold repeat {out['cold_repeat']:.2e}", flush=True)


if __name__ == "__main__":
    main()
