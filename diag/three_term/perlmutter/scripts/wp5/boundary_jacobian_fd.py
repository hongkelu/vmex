"""Is the boundary Jacobian the polish uses the derivative of the tight rows?  (WP5 qualification, small deck)

usage: python boundary_jacobian_fd.py OUT.json   (cwd: a vmex tree; a GPU; env MPOL, NS, BETA (pressure scale))

At the deck's boundary x: J = model.linearize(x) (the implicit Jacobian the Levenberg-Marquardt polish steps with)
against central differences of model.evaluate(x +- h d, tight=True) along random unit directions d (in the
x_scale-scaled coordinates), h = 1e-2, 1e-3, 1e-4.  Reported per direction and h: |FD - J d| / |J d| for the
whole row vector and per row block.  An exact Jacobian shows a relative error falling as h^2 to the solve's
noise floor.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex.core import freeboundary_three_term as fvc  # noqa: E402


def main():
    """Compare J d with central differences; write OUT.json."""
    inp, field = t._small_vacuum()
    if os.environ.get("MPOL") or os.environ.get("NS"):
        inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", int(os.environ.get("MPOL", 2)),
                                          int(os.environ.get("NS", 7))), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    model = fvc.ThreeTermFreeBoundaryModel(dataclasses.replace(inp, lfreeb=False), **t.SMALL)
    x, params = np.asarray(model.x0, dtype=float), model.params0
    rows0, aux = model.evaluate(x, params, field, tight=True)
    J = np.asarray(model.linearize(x, aux, field)[0])
    scale = np.asarray(model.x_scale)
    rng = np.random.default_rng(0)
    out = dict(resolution=model.resolution(), cases=[])
    for k in range(3):
        d = rng.standard_normal(x.size) * scale
        d /= np.linalg.norm(d / scale)
        Jd = J[:, : x.size] @ d
        for h in (1e-2, 1e-3, 1e-4):
            plus = np.asarray(model.evaluate(x + h * d, params, field, tight=True)[0])
            minus = np.asarray(model.evaluate(x - h * d, params, field, tight=True)[0])
            fd = (plus - minus) / (2 * h)
            row = dict(direction=k, h=h, rel=float(np.linalg.norm(fd - Jd) / np.linalg.norm(Jd)))
            for block, b0, b1 in model.row_blocks:
                nrm = np.linalg.norm(Jd[b0:b1])
                if nrm > 0:
                    row[block] = float(np.linalg.norm(fd[b0:b1] - Jd[b0:b1]) / nrm)
            out["cases"].append(row)
            print(json.dumps({k2: (f"{v:.2e}" if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
