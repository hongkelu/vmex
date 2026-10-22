"""Does the model's tight, Newton-refined solve (or a tighter FTOL) remove the cold/warm interior difference?

usage: python refine_check.py OUT.json   (cwd: a vmex tree; a GPU; env DECK, FTOL as interior_gauge.py)

For FTOL in (deck, 1e-15, 1e-16): plain solves cold and warm (gap, iota gap); then the implicit layer's host solve
with Newton refinement (``_host_solve_and_mask_status``, the path ThreeTermFreeBoundaryModel.solve(tight=True) uses),
seeded from each plain state with every cache cleared: the gap between the two refined states, and how far the
refinement moved each.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax

from vmex.core import implicit as im

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interior_gauge import FIELDS, deck, leaves, solve  # noqa: E402


def gap(a, b):
    """Largest coefficient difference between two states."""
    la, lb = leaves(a), leaves(b)
    return max(float(np.max(np.abs(la[f] - lb[f]))) for f in FIELDS)


def refined(inp, seed):
    """The implicit layer's tight host solve (with Newton refinement) seeded from ``seed``, caches cleared."""
    cfg = im._canonical_config(im.make_config(inp, multigrid=True, hot_restart=True))
    for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
        cache.pop(cfg, None)
    im._PERTURB_SEED[cfg] = seed
    params = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), im.params_from_input(inp))
    state, _, status, _, _ = im._host_solve_and_mask_status(cfg, params)
    return state, int(status)


def main():
    """Run the comparison and write OUT.json."""
    base = deck()
    out = []
    for ftol in (float(np.atleast_1d(base.ftol_array)[-1]), 1e-15, 1e-16):
        inp = dataclasses.replace(base, ftol_array=np.array([ftol]), niter_array=np.array([100000]))
        cold = solve(inp)
        near = solve(dataclasses.replace(inp, rbc=np.asarray(inp.rbc) * (1 + 1e-3))).state
        warm = solve(inp, near)
        row = dict(ftol=ftol, plain_gap=gap(cold.solution, warm.solution),
                   plain_iota_gap=float(np.max(np.abs(np.asarray(cold.wout.iotaf) - np.asarray(warm.wout.iotaf)))),
                   fsq_cold=[float(cold.wout.fsqr), float(cold.wout.fsqz), float(cold.wout.fsql)],
                   fsq_warm=[float(warm.wout.fsqr), float(warm.wout.fsqz), float(warm.wout.fsql)])
        try:
            rc, sc = refined(inp, cold.state)
            rw, sw = refined(inp, warm.state)
            row.update(refined_gap=gap(rc, rw), refine_moved_cold=gap(rc, cold.solution),
                       refine_moved_warm=gap(rw, warm.solution), status=[sc, sw])
        except Exception as error:  # reported, not hidden: the comparison is the result
            row.update(refine_error=f"{type(error).__name__}: {error}")
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    main()
