"""Single-stage design gradient through the three-term free boundary against FD, with tighter boundary fits.

usage: python design_gradient_fd.py OUT.json   (cwd: a vmex tree; a GPU)

test_plasma_map_derivatives_at_nonzero_offsets' problem (small vacuum deck; coil current x0, PHIEDGE map x1, x2)
built with boundary_ftol 1e-3 (the test's), 1e-6 and 1e-9 (POLISH_FTOL lowered to match, 20 rounds).  For each: the adjoint design gradient at
x = (0, 0.1, 0.3) and central differences of problem.fun at h = 1e-2, 1e-3, 1e-4 for every coordinate.  A gap that
shrinks with the fits' tolerance is FD noise; one that stays is a gradient error.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core.freeboundary_problem import FreeBoundaryProblem  # noqa: E402


def build(ftol):
    """The test's problem with the boundary fits converged to ``ftol`` (the certificate's polish target too)."""
    fvc.POLISH_FTOL, fvc.POLISH_ROUNDS = min(1e-4, ftol), 20
    inp, field = t._small_vacuum()
    base = field.currents
    return FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(3),
        field_from_parameters=lambda x: dataclasses.replace(field, currents=base * (1 + 0.05 * x[0])),
        plasma_from_parameters=lambda params, x: dataclasses.replace(
            params, phiedge=params.phiedge * (1 + 0.5 * x[1]) + 0.01 * x[2] ** 2),
        boundary_condition="three_term",
        three_term_options=dict(t.SMALL, boundary_ftol=ftol, boundary_max_nfev=400))


def main():
    """Gradient against FD at three fit tolerances; write OUT.json."""
    x = np.array([0.0, 0.1, 0.3])
    out = []
    for ftol in (1e-3, 1e-6, 1e-9):
        problem = build(ftol)
        _, grad = problem.value_and_grad(x)
        row = dict(boundary_ftol=ftol, grad=[float(g) for g in grad], differences={})
        for h in (1e-2, 1e-3, 1e-4):
            diff = []
            for k in range(3):
                e = np.eye(3)[k] * h
                diff.append(float((problem.fun(x + e) - problem.fun(x - e)) / (2 * h)))
            row["differences"][str(h)] = diff
            rel = np.abs(np.array(diff) - grad) / np.maximum(np.abs(grad), 1e-12)
            print(ftol, h, "grad", np.round(grad, 6), "central", np.round(diff, 6), "rel", np.round(rel, 5),
                  flush=True)
        out.append(row)
        problem.close()
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
