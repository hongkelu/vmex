#!/usr/bin/env python
r"""Single-stage optimization of coil shapes and plasma on the three-term free boundary.

The design is the shapes and currents of the Landreman & Paul (2021) precise-QA
coils (held in ESSOS).  As in ``single_stage_free_boundary_optimization_three_terms.py``
every design point is the free boundary on which the three plasma-vacuum
interface conditions hold, Newton-converged on the exact gauge, with exact
implicit design gradients; SLSQP (``opt.minimize``) lowers the quasisymmetry
residual with the mean rotational transform and the aspect ratio in bands.

Moving the coils' shapes brings their engineering limits in.  They come from
:mod:`vmex.core.single_stage_targets` and enter
``ThreeTermsFreeBoundaryProblem.from_tuples`` as one entry each:

- ``coil_constraint``: length, peak and mean-squared curvature of every coil
  and the coil-coil distance (a ``design_constraints`` entry: a function of the
  design alone, no equilibrium);
- ``coil_plasma_distance``: the distance from the coils to the plasma boundary
  (a ``parameter_constraints`` entry: a function of the equilibrium and the
  design).

The limits here are those of the starting coils, a little relaxed, so the run
shows the mechanics; a study sets its own.

Needs ``vmex[coils,freeb]`` (ESSOS and virtual-casing-jax) and a GPU.  Run it
with ``VMEX_EXAMPLES_CI=1`` for a short smoke pass that reports and exits 0;
otherwise a run that misses a target says which and exits 1.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import single_stage_targets as targets  # noqa: E402

# The ESSOS coils and the deck that seeds the free boundary:
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_lowres"

# Plasma resolution, vacuum, and the toroidal flux that matches the coils [Wb]:
MPOL = NTOR = 4
NS = 15
FTOL = 1e-12
PHIEDGE = -0.025

# The three-term free boundary: interface grid per field period, its fixed
# singular quadrature (4 nfp nphi, 2 ntheta) and the trials' interior tolerance:
GRID = 16
TRIAL_FTOL = 1e-11

# Design: every curve degree of freedom moves by CURVE_STEP [m] per unit of x,
# every base coil's current by CURRENT_STEP of itself; the start is the coils:
CURVE_STEP = 1e-3
CURRENT_STEP = 0.05

# Quasisymmetry (M, N) on these surfaces is the loss; the bands as the
# currents-only example:
HELICITY = (1, 0)
SURFACES = [0.25, 0.5, 0.75, 1.0]
IOTA_RANGE = (0.40, 0.46)
ASPECT_RANGE = (5.5, 6.5)

# Coil limits relative to the starting coils (length, peak curvature and
# mean-squared curvature at most these multiples of the largest, coil-coil and
# coil-plasma distances at least these fractions of the starting ones):
LENGTH_FACTOR, CURVATURE_FACTOR, MSC_FACTOR = 1.05, 1.10, 1.10
COIL_DISTANCE_FRACTION, PLASMA_DISTANCE_FRACTION = 0.9, 0.9

MAX_INTERFACE_RESIDUAL = 1e-2
MAX_STEPS = 4
SLSQP_FTOL = 1e-12
OUTPUT_DIR = Path("output_single_stage_three_terms_coils")

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MPOL = NTOR = 3
    NS, MAX_STEPS, MAX_INTERFACE_RESIDUAL = 11, 1, 5e-2

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
### Set up the coils, their limits and the free-boundary deck #################
###############################################################################

from essos.coils import Coils  # noqa: E402 (optional heavy import)
from essos.fields import BiotSavart  # noqa: E402

coils = Coils.from_json(str(COILS_FILE))
n_curves, n_currents = int(np.size(coils.dofs_curves)), int(np.size(coils.dofs_currents))
curves0, currents0 = jnp.asarray(coils.dofs)[:n_curves], jnp.asarray(coils.dofs_currents)  # as coils.dofs orders them


def coils_from_x(x):
    """Design vector -> the ESSOS coils: curve dofs, then base currents."""
    curves = curves0 + CURVE_STEP * x[:n_curves]
    currents = currents0 * (1 + CURRENT_STEP * x[n_curves:])
    return coils.with_dofs(jnp.concatenate([curves, currents]))


def field_from_parameters(x):
    return BiotSavart(coils_from_x(x))


start = targets.coil_metrics(coils)
limits = targets.CoilLimits(length=LENGTH_FACTOR * float(np.max(start["length"])),
                            curvature=CURVATURE_FACTOR * float(np.max(start["peak"])),
                            msc=MSC_FACTOR * float(np.max(start["msc"])),
                            coil_distance=COIL_DISTANCE_FRACTION * float(start["coil_distance"]))
print(f"ESSOS coils: {n_currents} base coils, {n_curves} curve dofs; limits: length {limits.length:.3f} m, "
      f"curvature {limits.curvature:.2f} 1/m, msc {limits.msc:.2f} 1/m^2, coil distance {limits.coil_distance:.3f} m")

inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", phiedge=PHIEDGE, curtor=0.0,
              am=np.zeros(21), ns_array=np.array([NS]), niter_array=np.array([5000]),
              ftol_array=np.array([FTOL]), delt=0.5)
nfp = int(inp.nfp)

###############################################################################
### Build the problem: objective tuples, constraints beside their bounds ######
###############################################################################

qs = opt.QuasisymmetryRatioResidual(np.asarray(SURFACES), *HELICITY)
plasma_distance = targets.coil_plasma_distance(coils_from_x, nfp)
config = vj.ThreeTermsConfig(nphi=GRID, ntheta=GRID, quadrature=(4 * nfp * GRID, 2 * GRID), trial_ftol=TRIAL_FTOL)
x0 = np.zeros(n_curves + n_currents)


def build(distance_floor):
    return vj.ThreeTermsFreeBoundaryProblem.from_tuples(
        inp, [(qs.residuals_state, 0.0, 1.0)], x0,
        constraints=[(opt.mean_iota, *IOTA_RANGE), (opt.aspect_ratio, *ASPECT_RANGE)],
        parameter_constraints=[(plasma_distance, distance_floor, np.inf)],
        design_constraints=[targets.coil_constraint(coils_from_x, limits)],
        field_from_parameters=field_from_parameters, boundary_max_residual=MAX_INTERFACE_RESIDUAL,
        config=config, semantic_identity=dict(example="single_stage_free_boundary_three_terms_coils", version=1))


# The coil-plasma floor is a fraction of the starting distance, which needs the starting free boundary: a first
# problem measures it (its constraint rows' third value), the run's problem holds it.
probe = build(0.0)
distance0 = float(probe.constraint_values(x0)[2])
probe.close()
problem = build(PLASMA_DISTANCE_FRACTION * distance0)
loss_start = problem.fun(x0)
iota, aspect, distance = problem.constraint_values(x0)
res = problem.boundary_residual(x0)
print(f"start ({time.perf_counter() - started:.0f} s): loss {loss_start:.4e}, iota {iota:.4f}, aspect {aspect:.3f}, "
      f"coil-plasma distance {distance:.4f} m (floor {PLASMA_DISTANCE_FRACTION * distance0:.4f}); interface "
      f"residuals B.n {res.normal:.1e}, pressure {res.pressure:.1e}, sheet current {res.sheet_current:.1e}")

###############################################################################
### Optimize ##################################################################
###############################################################################

history = []


def report(intermediate_result):
    x = intermediate_result.x
    iota, aspect, distance = problem.constraint_values(x)
    slack = float(np.min(problem.constraints()[-1].fun(x)))
    history.append(dict(step=int(intermediate_result.nit), loss=float(intermediate_result.fun), iota=float(iota),
                        aspect=float(aspect), plasma_distance=float(distance), coil_slack=slack))
    print(f"step {intermediate_result.nit}: loss {intermediate_result.fun:.4e}, iota {iota:.4f}, aspect "
          f"{aspect:.3f}, coil-plasma distance {distance:.4f} m, coil slack {slack:.2e} "
          f"({time.perf_counter() - started:.0f} s)", flush=True)


result = opt.minimize(problem, x0=x0, method="SLSQP", constraints=problem.constraints(), callback=report,
                      options=dict(maxiter=MAX_STEPS, ftol=SLSQP_FTOL))

###############################################################################
### Check, print and save #####################################################
###############################################################################

x = result.x
end = problem.boundary_residual(x)
iota, aspect, distance = problem.constraint_values(x)
slack = float(np.min(problem.constraints()[-1].fun(x)))
print(f"\n{result.accepted_steps} accepted steps ({result.stop_reason or result.message}): "
      f"loss {loss_start:.4e} -> {result.fun:.4e}")
print(f"final: iota {iota:.4f}, aspect {aspect:.3f}, coil-plasma distance {distance:.4f} m, coil slack {slack:.2e}; "
      f"interface residuals B.n {end.normal:.1e}, pressure {end.pressure:.1e}, sheet current {end.sheet_current:.1e}")
coils_from_x(jnp.asarray(x)).to_json(str(OUTPUT_DIR / "coils_optimized.json"))
vj.write_wout(OUTPUT_DIR / "wout_single_stage_three_terms_coils.nc", problem.equilibrium_from_x(x).wout)

misses = []
if not IOTA_RANGE[0] - 1e-4 <= iota <= IOTA_RANGE[1] + 1e-4:
    misses.append(f"mean iota {iota:.4f} outside {IOTA_RANGE}")
if not ASPECT_RANGE[0] - 1e-3 <= aspect <= ASPECT_RANGE[1] + 1e-3:
    misses.append(f"aspect ratio {aspect:.3f} outside {ASPECT_RANGE}")
if distance < PLASMA_DISTANCE_FRACTION * distance0 - 1e-4:
    misses.append(f"coil-plasma distance {distance:.4f} below its floor")
if slack < -1e-6:
    misses.append(f"coil limits violated (slack {slack:.2e})")
if max(end.normal, end.pressure, end.sheet_current) > MAX_INTERFACE_RESIDUAL:
    misses.append(f"interface residual above {MAX_INTERFACE_RESIDUAL:g}")
if not result.fun < loss_start:
    misses.append("the loss did not decrease")
summary = dict(loss_start=float(loss_start), loss=float(result.fun), accepted_steps=int(result.accepted_steps),
               iota=float(iota), aspect=float(aspect), plasma_distance=float(distance), coil_slack=slack,
               interface_residual=vars(end), history=history, misses=misses,
               seconds=time.perf_counter() - started)
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(f"{'smoke pass done' if ci_smoke else 'targets met' if not misses else 'MISSED: ' + '; '.join(misses)} "
      f"({time.perf_counter() - started:.0f} s)")
if misses and not ci_smoke:
    raise SystemExit(1)
