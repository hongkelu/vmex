"""Qualify ThreeTermFreeBoundaryProblem.exact_design_gradient against the converged re-solved FD (design_grad.json).

usage: RESOLVED=design_grad.json python exact_gradient_check.py OUT.json   (cwd: a vmex tree; a GPU)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_gradient_fd import build  # noqa: E402


def main():
    """Exact and Gauss-Newton gradients at the tight fit against the re-solved FD; write OUT.json."""
    problem = build(1e-9)
    x = np.array([0.0, 0.1, 0.3])
    got = problem.exact_design_gradient(x)
    rows = json.loads(Path(os.environ["RESOLVED"]).read_text())
    ref = np.asarray([r for r in rows if r["boundary_ftol"] == 1e-9][0]["differences"]["0.001"])
    out = dict(exact=got["gradient"].tolist(), gauss_newton=got["gauss_newton"].tolist(), resolved_fd=ref.tolist(),
               exact_rel=(np.abs(got["gradient"] - ref) / np.abs(ref)).tolist(),
               gauss_newton_rel=(np.abs(got["gauss_newton"] - ref) / np.abs(ref)).tolist(),
               cg_iterations=got["cg_iterations"], cg_converged=got["cg_converged"], cg_residual=got["cg_residual"])
    print(json.dumps(out, indent=1), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
