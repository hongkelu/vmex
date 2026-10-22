"""Force residuals (fsqr, fsqz, fsql) along the segment between the cold and the perturbed-start converged states of
the small LP QA deck: a flat valley (gauge-like near-null direction) or two separate roots?"""
import os
import sys
from dataclasses import replace

import numpy as np
import jax

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core.solver import evaluate_forces  # noqa: E402

m, ns = int(os.environ.get("MPOL", 4)), int(os.environ.get("NS", 15))
inp = replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
              lfreeb=False, ftol_array=np.array([1e-13]), niter_array=np.array([50000]))
cold = opt.solve_equilibrium(inp)
seed = jax.tree.map(lambda a: a * (1 + 1e-3 * jax.numpy.cos(jax.numpy.arange(a.size).reshape(a.shape)))
                    if jax.numpy.issubdtype(a.dtype, jax.numpy.inexact) else a, cold.state)
pert = opt.solve_equilibrium(inp, initial_state=seed)
rt = cold.solver_context
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 1.5, 3.0):
    state = jax.device_put(jax.tree.map(lambda a, b: (1 - alpha) * a + alpha * b, cold.solution, pert.solution),
                           jax.devices()[0])
    _, res, diag = evaluate_forces(state, rt)
    print(f"alpha {alpha:5.2f}: fsqr {float(res.fsqr):.3e} fsqz {float(res.fsqz):.3e} fsql {float(res.fsql):.3e}; "
          f"preconditioned {float(diag.preconditioned.fsqr1) if hasattr(diag.preconditioned, 'fsqr1') else 'n/a'}",
          flush=True)
