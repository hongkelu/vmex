#!/usr/bin/env python
r"""Single-stage optimization of coils and plasma on the three-term free boundary.

The design is the coil currents of the Landreman & Paul (2021) precise-QA
coils (held in ESSOS) and the toroidal flux PHIEDGE.  Every design point is
the free boundary on which the three plasma-vacuum interface conditions hold
-- ``B . n = 0``, pressure balance and no sheet current, with the plasma's own
field from virtual casing -- not a VMEC + NESTOR solve, which balances only
``|B|`` and leaves the tangential field jump free
(``single_stage_free_boundary_optimization.py`` is the NESTOR version).

The problem is ``vj.ThreeTermsFreeBoundaryProblem``.  Each trial boundary is
fitted from the accepted one and Newton-converged on the exact gauge of the
boundary's poloidal angle (VMEC's spectral condensation), and the design
gradients are the exact implicit derivatives of that converged point, through
the equilibrium, the coils' Biot-Savart field and the plasma parameters.  New
coils or a new PHIEDGE recompile nothing.  SLSQP (``opt.minimize``) lowers the
quasisymmetry residual

    J = (1/2) |r_QS|^2

with the mean rotational transform and the aspect ratio held in bands, starting
from currents detuned by up to 10% and a 2% flux error, which it recovers.

The run saves its accepted point (``problem.save_checkpoint``) and SLSQP's
whole state (``opt.save_slsqp_state``: quasi-Newton Hessian, multipliers) after
every accepted step; with ``RESUME = True`` a second run continues from them
as if it had never stopped, which is how a long run is chained over batch jobs.

Needs ``vmex[coils,freeb]`` (ESSOS and virtual-casing-jax) and a GPU: the seed
point takes about 3 minutes on one A100, most of it compilation, and each
SLSQP step about 1 minute.  Run it with ``VMEX_EXAMPLES_CI=1`` for a short
smoke pass that reports and exits 0; otherwise a run that misses a target says
which and exits 1.
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
# singular quadrature (4 nfp nphi, 2 ntheta), the boundary modes solved for,
# (None: all; a truncated boundary on the gauge keeps a larger residual),
# and the trials' looser interior tolerance (each record is re-solved tight):
GRID = 16
MAX_MODE = None
TRIAL_FTOL = 1e-11

# Design: each base coil's current is I_0 (1 + CURRENT_STEP x_k) and PHIEDGE is
# PHIEDGE_0 (1 + PHIEDGE_STEP x_last).  The start detunes the optimized currents
# by up to 10% and the flux by 2%:
CURRENT_STEP = 0.05
PHIEDGE_STEP = 0.02
X0 = [2.0, -2.0, 1.0, -1.0, 1.0]

# Quasisymmetry (M, N) on these surfaces is the loss; the bands hold the mean
# rotational transform and the aspect ratio where the design should end:
HELICITY = (1, 0)
SURFACES = [0.25, 0.5, 0.75, 1.0]
IOTA_RANGE = (0.40, 0.46)
ASPECT_RANGE = (5.5, 6.5)

# Every accepted point must be a free boundary to this RMS interface residual
# (relative to |B|; the resolution's floor is about 1e-3 here):
MAX_INTERFACE_RESIDUAL = 1e-2

# SLSQP: accepted steps and the relative tolerance (the loss starts small, so
# SciPy's 1e-6 would stop at once):
MAX_STEPS = 6
SLSQP_FTOL = 1e-12

# Output: the checkpoint and SLSQP state written after every accepted step,
# and whether to continue from them:
OUTPUT_DIR = Path("output_single_stage_three_terms")
RESUME = False

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    # Coarser plasma, and the interface floor of that resolution:
    MPOL = NTOR = 3
    NS, MAX_STEPS, MAX_INTERFACE_RESIDUAL = 11, 1, 5e-2

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_file, slsqp_file = OUTPUT_DIR / "checkpoint.npz", OUTPUT_DIR / "slsqp_state.npz"

###############################################################################
### Set up the coils and the free-boundary deck ###############################
###############################################################################

from essos.coils import Coils  # noqa: E402 (optional heavy import)
from essos.fields import BiotSavart  # noqa: E402

coils = Coils.from_json(str(COILS_FILE))
n_curves = coils.dofs_curves.size
n_currents = coils.dofs_currents.size
print(f"ESSOS coils: {n_currents} base coils, {coils.currents.shape[0]} after symmetry")


def field_from_parameters(x):
    """Design vector -> the coils' Biot-Savart field (anything with ``b_cyl`` or ``xyz -> B``)."""
    currents = coils.dofs_currents * (1 + CURRENT_STEP * x[:n_currents])
    return BiotSavart(coils.with_dofs(jnp.concatenate([coils.dofs[:n_curves], currents])))


def plasma_from_parameters(params, x):
    """Design vector -> the plasma parameters (``ImplicitParams``): here PHIEDGE."""
    return replace(params, phiedge=params.phiedge * (1 + PHIEDGE_STEP * x[n_currents]))


inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", phiedge=PHIEDGE, curtor=0.0,
              am=np.zeros(21), ns_array=np.array([NS]), niter_array=np.array([5000]),
              ftol_array=np.array([FTOL]), delt=0.5)
nfp = int(inp.nfp)

###############################################################################
### Build the problem: the free boundary of the starting design ###############
###############################################################################

qs = opt.QuasisymmetryRatioResidual(np.asarray(SURFACES), *HELICITY)
# Objective terms as upstream's (function, target, weight); each constraint keeps its bounds beside it.
terms = [(qs.residuals_state, 0.0, 1.0)]
constraints = [(opt.mean_iota, *IOTA_RANGE), (opt.aspect_ratio, *ASPECT_RANGE)]


config = vj.ThreeTermsConfig(nphi=GRID, ntheta=GRID, quadrature=(4 * nfp * GRID, 2 * GRID), max_mode=MAX_MODE,
                            trial_ftol=TRIAL_FTOL)
resume = RESUME and checkpoint_file.exists()
problem = vj.ThreeTermsFreeBoundaryProblem.from_tuples(
    inp, terms, np.asarray(X0, dtype=float), constraints=constraints, field_from_parameters=field_from_parameters,
    plasma_from_parameters=plasma_from_parameters,
    boundary_max_residual=MAX_INTERFACE_RESIDUAL, config=config,
    checkpoint=checkpoint_file if resume else None,
    semantic_identity=dict(example="single_stage_free_boundary_optimization_three_terms", version=1))
x_start = problem.accepted.parameters
loss_start = problem.fun(x_start)
start = problem.boundary_residual(x_start)
iota, aspect = problem.constraint_values(x_start)
print(f"{'resumed' if resume else 'start'} ({time.perf_counter() - started:.0f} s): loss {loss_start:.4e}, "
      f"iota {iota:.4f}, aspect {aspect:.3f}; interface residuals B.n {start.normal:.1e}, pressure "
      f"{start.pressure:.1e}, sheet current {start.sheet_current:.1e}")

###############################################################################
### Optimize ##################################################################
###############################################################################

history = []


def report(intermediate_result):
    """After each accepted step: print it, and save the accepted point."""
    x = intermediate_result.x
    iota, aspect = problem.constraint_values(x)
    history.append(dict(step=int(intermediate_result.nit), loss=float(intermediate_result.fun), iota=float(iota),
                        aspect=float(aspect), x=np.asarray(x).tolist()))
    print(f"step {intermediate_result.nit}: loss {intermediate_result.fun:.4e}, iota {iota:.4f}, aspect "
          f"{aspect:.3f}, x {np.array2string(np.asarray(x), precision=3, suppress_small=True)} "
          f"({time.perf_counter() - started:.0f} s)", flush=True)
    problem.save_checkpoint(checkpoint_file)


options = dict(maxiter=MAX_STEPS, ftol=SLSQP_FTOL,
               state_callback=lambda state: opt.save_slsqp_state(slsqp_file, state))
if resume and slsqp_file.exists():
    options["resume"] = opt.load_slsqp_state(slsqp_file)  # SLSQP continues with its Hessian
result = opt.minimize(problem, x0=x_start, method="SLSQP", constraints=[problem.constraints()], callback=report, options=options)

###############################################################################
### Check, print and save #####################################################
###############################################################################

x = result.x
end = problem.boundary_residual(x)
iota, aspect = problem.constraint_values(x)
print(f"\n{result.accepted_steps} accepted steps ({result.stop_reason or result.message}): "
      f"loss {loss_start:.4e} -> {result.fun:.4e}")
print(f"final: iota {iota:.4f}, aspect {aspect:.3f}; interface residuals B.n {end.normal:.1e}, "
      f"pressure {end.pressure:.1e}, sheet current {end.sheet_current:.1e}")
print(f"trials {problem.stats['trials']} ({problem.stats['failed']} rejected), boundary evaluations "
      f"{problem.stats['boundary_evaluations']}, Newton evaluations {problem.stats.get('newton_evaluations', 0)}")
wout_file = vj.write_wout(OUTPUT_DIR / "wout_single_stage_three_terms.nc", problem.equilibrium_from_x(x).wout)
print(f"Wrote {wout_file}")

misses = []
if not IOTA_RANGE[0] - 1e-4 <= iota <= IOTA_RANGE[1] + 1e-4:
    misses.append(f"mean iota {iota:.4f} outside {IOTA_RANGE}")
if not ASPECT_RANGE[0] - 1e-3 <= aspect <= ASPECT_RANGE[1] + 1e-3:
    misses.append(f"aspect ratio {aspect:.3f} outside {ASPECT_RANGE}")
if max(end.normal, end.pressure, end.sheet_current) > MAX_INTERFACE_RESIDUAL:
    misses.append(f"interface residual above {MAX_INTERFACE_RESIDUAL:g}")
if not result.fun < loss_start:
    misses.append("the loss did not decrease")
summary = dict(loss_start=float(loss_start), loss=float(result.fun), accepted_steps=int(result.accepted_steps),
               stop_reason=result.stop_reason, iota=float(iota), aspect=float(aspect), x=np.asarray(x).tolist(),
               interface_residual=vars(end), history=history, misses=misses,
               seconds=time.perf_counter() - started)
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(f"{'smoke pass done' if ci_smoke else 'targets met' if not misses else 'MISSED: ' + '; '.join(misses)} "
      f"({time.perf_counter() - started:.0f} s)")
if misses and not ci_smoke:
    raise SystemExit(1)
