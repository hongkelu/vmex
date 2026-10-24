"""Two identical cold fixed-boundary solves of the small LP QA deck: bitwise equal?  (GPU nondeterminism check)"""
import os
import sys
from dataclasses import replace

import numpy as np
import jax

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402

m, ns = int(os.environ.get("MPOL", 4)), int(os.environ.get("NS", 15))
inp = replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
              lfreeb=False, ftol_array=np.array([1e-13]), niter_array=np.array([50000]))
a, b = opt.solve_equilibrium(inp), opt.solve_equilibrium(inp)
d = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(jax.tree.leaves(a.state),
                                                                           jax.tree.leaves(b.state)))
print(f"{jax.devices()[0].platform}: repeat cold solves differ by {d:.3e} (max over the state)", flush=True)
