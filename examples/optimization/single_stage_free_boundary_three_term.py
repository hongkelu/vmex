#!/usr/bin/env python
"""Single-stage optimization of coil currents on the three-term free boundary.

Every design point is the free boundary on which the three plasma-vacuum
interface conditions hold -- ``B . n = 0``, pressure balance and no sheet
current -- with the plasma field from virtual casing
(:mod:`vmex.core.freeboundary_three_term`).  The design is the currents of the
Landreman & Paul (2021) precise-QA coils, held in ESSOS; SLSQP lowers the
quasisymmetry residual of the free boundary while keeping its rotational
transform and aspect ratio in range.  The gradients are those of the free
boundary itself: the implicit function theorem of the boundary fit, through the
equilibrium and the coils' field.

The resolution is the smallest at which the free boundary is meaningful
(``MPOL = NTOR = 3``, ``NS = 11``, a 16 x 16 interface grid).  On 8 CPU cores
the first design point takes about 7 minutes, nearly all of it compilation,
and each SLSQP step about one more: new coils recompile nothing.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402

# The ESSOS coils and the deck that seeds the free boundary:
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_lowres"

# Plasma resolution, vacuum, and the toroidal flux that matches the coils [Wb]:
MPOL = NTOR = 3
NS = 11
PHIEDGE = -0.025

# The interface grid, the boundary modes the free boundary solves for, and the
# fixed singular quadrature of virtual casing, (4 nfp nphi, 2 ntheta):
THREE_TERM = dict(nphi=16, ntheta=16, max_mode=2, quadrature=(128, 32))

# Design: each base coil's current is I_0 (1 + CURRENT_STEP x).  The start detunes
# the optimized currents by up to 10%, which the optimizer then recovers:
CURRENT_STEP = 0.05
X0 = [2.0, -2.0, 1.0, -1.0]

# Quasisymmetry (M, N) on these surfaces is the loss; the bounds below keep the
# mean rotational transform and the aspect ratio where they start:
HELICITY = (1, 0)
SURFACES = [0.25, 0.5, 0.75, 1.0]
IOTA_RANGE = (0.40, 0.46)
ASPECT_RANGE = (5.5, 6.5)

# SLSQP steps, and its relative tolerance (the loss starts small, so SciPy's 1e-6 would stop at once):
MAX_STEPS = 5
FTOL = 1e-12

###############################################################################
# End of input parameters.
###############################################################################

### Set up the coils and the free-boundary deck ###############################

from essos.coils import Coils  # noqa: E402 (optional heavy import)

coils = Coils.from_json(str(COILS_FILE))
n_curves = coils.dofs_curves.size
n_currents = coils.dofs_currents.size
print(f"ESSOS coils: {n_currents} base coils, {coils.currents.shape[0]} after symmetry")


def field_from_parameters(x):
    """Design vector -> the coils' Biot-Savart field (anything with ``b_cyl`` or ``xyz -> B``)."""
    from essos.fields import BiotSavart

    currents = coils.dofs_currents * (1 + CURRENT_STEP * x)
    return BiotSavart(coils.with_dofs(jnp.concatenate([coils.dofs[:n_curves], currents])))


inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", phiedge=PHIEDGE, curtor=0.0,
              am=np.zeros(21), ns_array=np.array([NS]), niter_array=np.array([5000]),
              ftol_array=np.array([1e-10]))

### Build the problem: the free boundary of the starting coils ################

qs = opt.QuasisymmetryRatioResidual(np.asarray(SURFACES), *HELICITY)


def loss(state, runtime, x):
    residuals = qs.residuals_state(state, runtime)
    return 0.5 * jnp.vdot(residuals, residuals)


problem = vj.ThreeTermFreeBoundaryProblem.from_loss(
    inp, loss, np.asarray(X0), field_from_parameters=field_from_parameters,
    quantities=(opt.mean_iota, opt.aspect_ratio), config=vj.ThreeTermConfig(**THREE_TERM))
x0 = problem.x0
start, loss0 = problem.boundary_residual(x0), problem.fun(x0)
print(f"start: loss {loss0:.4e}, iota {problem.constraint_values(x0)[0]:.4f}, "
      f"aspect {problem.constraint_values(x0)[1]:.3f}; interface residuals B.n {start.normal:.1e}, "
      f"pressure {start.pressure:.1e}, sheet current {start.sheet_current:.1e}")

### Optimize ##################################################################

bounds = problem.nonlinear_constraint([IOTA_RANGE[0], ASPECT_RANGE[0]], [IOTA_RANGE[1], ASPECT_RANGE[1]])


def report(intermediate_result):
    x = intermediate_result.x
    iota, aspect = problem.constraint_values(x)
    print(f"step {intermediate_result.nit}: loss {intermediate_result.fun:.4e}, iota {iota:.4f}, "
          f"aspect {aspect:.3f}, currents x {np.array2string(x, precision=3, suppress_small=True)}")


result = opt.minimize(problem, x0=x0, method="SLSQP", constraints=[bounds], callback=report,
                      options=dict(maxiter=MAX_STEPS, ftol=FTOL))

### Print and save ############################################################

x = result.x
end = problem.boundary_residual(x)
print(f"\n{result.accepted_steps} accepted steps ({result.stop_reason or result.message}): "
      f"loss {loss0:.4e} -> {result.fun:.4e}")
print(f"final interface residuals B.n {end.normal:.1e}, pressure {end.pressure:.1e}, "
      f"sheet current {end.sheet_current:.1e}")
print(f"trials {problem.stats['trials']}, boundary evaluations {problem.stats['boundary_evaluations']}")
print(f"Wrote {vj.write_wout('wout_single_stage_three_term.nc', problem.equilibrium_from_x(x).wout)}")
