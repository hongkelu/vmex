#!/usr/bin/env python
r"""Finite-beta single-stage optimization at the three-term free boundary, with a self-consistent bootstrap current.

The free-boundary design problem of ``single_stage_free_boundary_optimization.py`` (same seed, coils and coil
pre-fit, limits near the seed's) at volume-average beta ``TARGET_BETA`` with a Redl-self-consistent bootstrap current, every trial's
plasma the three-term free boundary of its coils: the boundary on which ``B . n = 0``, pressure balance and
continuity of the tangential field hold together (:class:`vmex.ThreeTermsConfig`), with the plasma field from
virtual casing (``pip install vmex[freeb]``).

Pressure and current are inputs of every free-boundary solve, as in VMEC and in Landreman, Buller & Drevlak (2022).
The kinetic profiles ``n_e = n0 (1 - s^5)``, ``T_e = T_i = T0 (1 - s)`` (hydrogen) give both the deck's pressure
``p = 2 e n_e T_e`` and the Redl bootstrap ``<J.B>``.  The design variables are the coil Fourier coefficients and
the current profile (its first ``N_CURRENT`` AC coefficients and CURTOR), and SciPy's SLSQP holds the Redl
self-consistency as a constraint row, beside the rotational-transform window and the aspect limit:

    minimize   (1/2) |r_QS|^2 + coil length, curvature, separation and clearance penalties
    subject to IOTA_CONSTRAINT <= min |iota|,  max |iota| <= IOTA_CEILING_CONSTRAINT,
               aspect <= ASPECT_CONSTRAINT,  sum_j ((<J.B>_j - <J.B>_Redl,j) / ...)^2 <= REDL_TOLERANCE

with the clearance measured to each trial's free boundary.  As in ``single_stage_free_boundary_optimization.py``
the script owns the optimization: every trial is ``vj.solve_free_boundary_implicit_status`` on a configuration
built with ``three_terms=``, whose state is the fitted free boundary's equilibrium and whose custom VJP
differentiates the boundary fit implicitly; ``jax.value_and_grad`` gives the objective's gradient and
``jax.jacrev`` the constraint Jacobian (one batched pullback), and ``FunctionProblem.from_functions`` hands them to
SciPy.  The coil field is a filament field whose arrays are pytree leaves, so new coils never recompile.  The
seed's current comes from a fixed-boundary Picard iteration to Redl's
(:func:`vmex.core.bootstrap.self_consistent_bootstrap`).

Run it with ``VMEX_EXAMPLES_CI=1`` for a short smoke pass that reports and exits 0.  Otherwise a run that misses a
target says which and exits 1.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im
from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch, self_consistent_bootstrap

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

jax.config.update("jax_enable_x64", True)

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.05, 1.0, 6)

# Seed boundary the coils are fitted to: the rotating ellipse of
# single_stage_optimization.py.
SEED_MINOR_RADIUS = 0.195
SEED_ELLIPSE = 0.10

# Volume-average beta, and the kinetic profiles both the pressure and the Redl current come from:
# n_e = n0 (1 - s^5), T_e = T_i = T0 (1 - s), hydrogen (Z_eff = 1, n_i = n_e), so p = 2 e n_e T_e,
# whose power series in s is PRESSURE_SHAPE times 2 e n0 T0, and the edge pressure vanishes.  n0 is set
# so that the fixed-boundary seed has TARGET_BETA.
TARGET_BETA = 0.005
T0 = 2.0e3                        # eV
PRESSURE_SHAPE = [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0]
BETA_TOLERANCE = 0.1              # relative, for the final check

# The current profile: its first N_CURRENT AC coefficients and CURTOR are design variables (relative
# steps of CURRENT_SCALE), held Redl-consistent by a constraint row, the sum of squared relative <J.B>
# mismatches on REDL_SURFACES, at most REDL_TOLERANCE (REDL_LIMIT in the final check).  The seed's
# current is the fixed-boundary Picard iteration's, PICARD_ITERATIONS at most.
N_CURRENT = 4
CURRENT_SCALE = 0.1
REDL_SURFACES = np.linspace(0.1, 0.9, 9)
REDL_TOLERANCE = 1.0e-3
REDL_LIMIT = 2.0e-3
PICARD_ITERATIONS = 8

# Targets the finished design is checked against (near the seed's values: the demonstration holds the
# constraints while it improves quasisymmetry):
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
IOTA_CEILING = 0.6                # maximum |iota|: the bootstrap current raises iota by ~0.1;
                                  # stay below the 2/3 resonance (NFP = 2)
ASPECT_LIMIT = 6.1                # maximum aspect ratio
COIL_SURFACE_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_LIMIT = 0.17
CURVATURE_LIMIT = 7.0

# Values the optimizer is given instead, a little inside the limits above: SLSQP
# holds its constraint rows only to its tolerance. The normal-field value is used
# only by the coil pre-fit: on a free boundary B.n = 0 holds to the fit's floor.
IOTA_CONSTRAINT = 0.425
IOTA_CEILING_CONSTRAINT = 0.59
ASPECT_CONSTRAINT = 6.05
NORMAL_FIELD_CONSTRAINT = 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT = 0.19
COIL_SURFACE_DISTANCE_CONSTRAINT = 0.16

# Coils: number of unique shapes, Fourier order, and the circle they start on.
N_COILS = 3
COIL_ORDER = 4
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

# Weights, and the longest coil the optimizer may build. The length term is
# one-sided: coils shorter than LENGTH_TARGET cost nothing, so it never pulls
# against quasisymmetry or the normal field.
LENGTH_TARGET = 5.5
LENGTH_WEIGHT = 0.5
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_WEIGHT = 1.0e4
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4
CONSTRAINT_WEIGHT = 1.0e3

# Bound on each design variable, in units of its scale, and the factor on those scales that SLSQP steps
# in: its first step has an identity Hessian, so each scaled variable moves about as far as its gradient,
# and smaller scales keep the steps where the constraints' linearization holds.
PARAMETER_BOUND = 3.0
DESIGN_STEP_SCALE = 0.2

# A low-resolution demonstration: the plasma's Fourier resolution (the boundary's modes are
# solved up to MPOL - 1 = NTOR), and the equilibria of every trial: radial surfaces, force
# tolerance and iteration cap; the check at the end re-solves the final coils' free boundary
# on the same grid.
MPOL = NTOR = 3
NS = 11
FTOL = 1.0e-10
NITER = 4000

# The three-term free boundary of every trial: the labelling ("equal_arc", "anchored" or
# "none") and the weight of its rows (None: the labelling's default, 1e-3 for equal
# arc and 1e-2 for anchored), the interface grid over one field period, and its fixed
# singular quadrature (4 nfp N, 2 N).
LABELLING = "equal_arc"
LABEL_WEIGHT = None
N_INTERFACE = 24

# Budgets: accepted SLSQP steps (each trial is one certified free boundary and,
# when SLSQP asks, one gradient), and the coil-only pre-fit.
MAXITER = 5
COIL_FIT_MAXITER = 200            # coil-only pre-fit, no equilibrium solves

# Surface grid the coil terms use:
NPHI, NTHETA = 37, 32

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    # Only the optimizer budget shrinks: two coil-fit iterations on a coarser
    # grid do not give coils that hold a converged free boundary.
    MAXITER = 1

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()

###############################################################################
### Set up the seed boundary and fit the coils to it ##########################
###############################################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
am = np.zeros(21)
am[:len(PRESSURE_SHAPE)] = PRESSURE_SHAPE
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am, pres_scale=1.0,
              ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 6)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
x_circular = np.asarray(curves0.dofs).ravel()
scales = np.full(x_circular.size, 0.05)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of B.n/|B| on a surface."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    normal_field = jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normal_field**2))


def hinge(constraints):
    """Quadratic penalty for constraints written as c >= 0, normalized by their limits."""
    return 0.5 * CONSTRAINT_WEIGHT * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


def coil_costs(coils, surface):
    """Coil length, curvature, separation and clearance to ``surface``."""
    length = jnp.sqrt(LENGTH_WEIGHT) * jnp.maximum(coils.length[:N_COILS] - LENGTH_TARGET, 0.0)
    curvature = jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
        coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)
    return jnp.asarray([
        0.5 * jnp.vdot(length, length),
        0.5 * jnp.vdot(curvature, curvature),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_CONSTRAINT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_CONSTRAINT, block_size=32),
    ])


def coils_from_x(x):
    return coils_circular.with_dofs(jnp.concatenate((x, coils_circular.dofs_currents)))


def coil_fit_objective(u):
    """The coil half of single_stage_optimization.py with the boundary frozen."""
    coils = coils_from_x(jnp.asarray(x_circular) + jnp.asarray(scales) * u)
    normal_field = 1.0 - normal_field_rms(coils, surface_seed) / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(coil_costs(coils, surface_seed)) + hinge(normal_field[None])


print("Running single_stage_free_boundary_optimization.py")
coil_fit_value_and_grad = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_value_and_grad(jnp.asarray(u)))),
    np.zeros_like(x_circular), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
x0 = x_circular + scales * coil_fit.x
coils0 = coils_from_x(jnp.asarray(x0))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils0, surface_seed)):.3f}% on the seed")

###############################################################################
### Check that the fitted coils confine the seed ##############################
###############################################################################

# The plasma a free-boundary solve finds depends on the toroidal flux it is
# told to enclose. Take it from the fitted field itself, through the seed's
# phi = 0 cross-section (Gauss-Legendre in radius, uniform in angle), so the
# seed boundary is the one the coils are asked to confine.
rho, rho_weights = np.polynomial.legendre.leggauss(24)
rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
# At phi = 0 every toroidal harmonic has phase m*theta, so sum over n.
m = np.arange(inp.rbc.shape[1])
rbc_phi0, zbs_phi0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
R_edge = np.cos(np.outer(theta, m)) @ rbc_phi0
Z_edge = np.sin(np.outer(theta, m)) @ zbs_phi0
dR_edge = -np.sin(np.outer(theta, m)) @ (m * rbc_phi0)
dZ_edge = np.cos(np.outer(theta, m)) @ (m * zbs_phi0)
R_center = rbc_phi0[0]
R = R_center + rho[:, None] * (R_edge - R_center)
Z = rho[:, None] * Z_edge
area_element = rho[:, None] * ((R_edge - R_center) * dZ_edge - Z_edge * dR_edge)
points = np.stack([R, np.zeros_like(R), Z], axis=-1).reshape(-1, 3)
field0 = BiotSavart(coils0)
B_phi = np.asarray(jax.vmap(field0.B)(jnp.asarray(points)))[:, 1].reshape(R.shape)
toroidal_flux = float(np.sum(rho_weights[:, None] * B_phi * area_element)
                      * (2.0 * np.pi / theta.size))
# The free-boundary solve takes the external field along the same +phi, and
# stops (vacuum.f, ier_flag 7) unless PHIEDGE has the sign of that field.
inp = replace(inp, phiedge=toroidal_flux)
print(f"[flux] PHIEDGE = {inp.phiedge:.6f} Wb, the fitted field's flux through the seed")

# Beta is proportional to the pressure at fixed boundary and flux, so one fixed-boundary solve of the
# seed at PRES_SCALE = 1 gives the scale, 2 e n0 T0, and with it n0.
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)
n0 = inp.pres_scale / (2.0 * ELEMENTARY_CHARGE * T0)
profiles = KineticProfiles(n0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]), T0 * np.array([1.0, -1.0]),
                           T0 * np.array([1.0, -1.0]))
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=2 if ci_smoke else PICARD_ITERATIONS)
inp = replace(picard.input, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array([NS]),
              ftol_array=np.array([FTOL]), niter_array=np.array([NITER]))
print(f"[bootstrap] n0 = {n0:.3e} m^-3 at T0 = {T0:.0f} eV for beta {100 * TARGET_BETA:.2f}%; seed current "
      f"CURTOR = {inp.curtor:.4e} A after {picard.iterations} Picard iterations (converged: {picard.converged})")

###############################################################################
### The three-term free-boundary solve ########################################
###############################################################################

# The design u: the coil coefficients, then the current variables, in steps of their scales from the seed.
N_COIL = x0.size
ac_reference = float(np.max(np.abs(np.asarray(inp.ac, dtype=float))))
design0 = np.r_[x0, np.zeros(N_CURRENT + 1)]
bound_scales = np.r_[scales, np.full(N_CURRENT + 1, CURRENT_SCALE)]
design_scales = DESIGN_STEP_SCALE * bound_scales


def x_from_u(u):
    return jnp.asarray(design0) + jnp.asarray(design_scales) * u


def field_from_u(u):
    return BiotSavart(coils_from_x(x_from_u(u)[:N_COIL]))


def plasma_from_u(params, u):
    """The current profile of the design: AC coefficients and CURTOR moved from the seed's."""
    current = x_from_u(u)[N_COIL:]
    return replace(params, ac=params.ac.at[:N_CURRENT].add(ac_reference * current[:N_CURRENT]),
                   curtor=params.curtor * (1.0 + current[N_CURRENT]))


def input_from_x(x):
    """The deck of the design ``x`` (its current profile), for the verification solve."""
    current = np.asarray(x[N_COIL:], dtype=float)
    ac = np.asarray(inp.ac, dtype=float).copy()
    ac[:N_CURRENT] += ac_reference * current[:N_CURRENT]
    return replace(inp, ac=ac, curtor=float(inp.curtor) * (1.0 + current[N_CURRENT]))


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
redl = RedlBootstrapMismatch(profiles, 0, surfaces=REDL_SURFACES)


def redl_mismatch(equilibrium_state, runtime):
    return redl.total_state(equilibrium_state, runtime)


def max_abs_iota(equilibrium_state, runtime):
    return jnp.max(opt.abs_iota_profile(equilibrium_state, runtime))


def boundary_surface(equilibrium_state, runtime, nphi=NPHI, ntheta=NTHETA):
    """The free boundary of a solved state, as an ESSOS surface."""
    rmnc, _, _, zmns = im._edge_physical(equilibrium_state, runtime)
    modes = runtime.modes
    ntor = int(np.max(np.abs(modes.n)))
    rows = jnp.asarray(np.asarray(modes.n) + ntor)
    cols = jnp.asarray(np.asarray(modes.m))
    shape = (2 * ntor + 1, int(np.max(modes.m)) + 1)
    rbc = jnp.zeros(shape).at[rows, cols].set(rmnc)
    zbs = jnp.zeros(shape).at[rows, cols].set(zmns)
    return surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=nphi, ntheta=ntheta)


def plasma_values(equilibrium_state, runtime):
    return {"QA total": float(qs.total_state(equilibrium_state, runtime)),
            "aspect": float(opt.aspect_ratio(equilibrium_state, runtime)),
            "mean iota": float(opt.mean_iota(equilibrium_state, runtime)),
            "min |iota|": float(opt.min_abs_iota(equilibrium_state, runtime)),
            "max |iota|": float(max_abs_iota(equilibrium_state, runtime)),
            "beta": float(opt.volume_average_beta(equilibrium_state, runtime)),
            "Redl mismatch": float(redl_mismatch(equilibrium_state, runtime))}


# The three-term lane of solve_free_boundary_implicit_status. Every trial fits its boundary warm from the
# latest differentiated one, to a relative cost change of 1e-3 in at most 40 evaluations, and counts as
# solved (status 0) when every RMS interface residual is within max_residual: at this low resolution
# (MPOL = 3) the residuals' floor is ~1e-2 (the seed's sheet current is 1.03e-2), so 2e-2 here.
three_terms = vj.ThreeTermsConfig(labelling=LABELLING, label_weight=LABEL_WEIGHT, nphi=N_INTERFACE,
                                  ntheta=N_INTERFACE, quadrature=(4 * NFP * N_INTERFACE, 2 * N_INTERFACE),
                                  ftol=1e-3, max_nfev=40, max_residual=2e-2)
config = vj.make_free_boundary_config(inp, field0, field_from_parameters=field_from_u, three_terms=three_terms)
params = im.params_from_input(inp)


def solve(u):
    """The design's three-term free boundary: its state, plasma parameters, status and largest residual."""
    plasma = plasma_from_u(params, u)
    equilibrium_state, status, worst, _ = vj.solve_free_boundary_implicit_status(plasma, u, config)
    return equilibrium_state, plasma, status, worst


u0 = np.zeros(design0.size)
state0, plasma0, status0, worst0 = solve(jnp.asarray(u0))
if int(status0) != 0:
    raise SystemExit(f"The fitted coils hold no three-term free boundary (largest residual {float(worst0):.2e}).")
runtime0 = im.runtime_from_params(plasma0, config.implicit)
free_seed = boundary_surface(state0, runtime0)
boundary_shift = float(jnp.max(jnp.linalg.norm(free_seed.gamma - surface_seed.gamma, axis=-1)))
seed_values = plasma_values(state0, runtime0)
print("[seed] " + ", ".join(f"{k} = {v:.4g}" for k, v in seed_values.items())
      + f"; free boundary within {boundary_shift:.4f} m of the fitted seed; largest interface residual "
      f"{float(worst0):.2e}")

###############################################################################
### Set up the objective and constraints ######################################
###############################################################################

COIL_TERMS = ("coil length", "coil curvature", "coil separation", "coil-surface separation")


@jax.jit
def objective_terms(equilibrium_state, plasma, u):
    """Quasisymmetry plus the coil penalties, with the clearance to this trial's free boundary."""
    runtime = im.runtime_from_params(plasma, config.implicit)
    qs_rows = qs.residuals_state(equilibrium_state, runtime)
    costs = coil_costs(coils_from_x(x_from_u(u)[:N_COIL]), boundary_surface(equilibrium_state, runtime))
    return 0.5 * jnp.vdot(qs_rows, qs_rows) + jnp.sum(costs), (qs_rows, costs)


@jax.jit
def constraint_terms(equilibrium_state, plasma):
    """min |iota|, max |iota|, aspect ratio and Redl mismatch of a solved state."""
    runtime = im.runtime_from_params(plasma, config.implicit)
    return jnp.stack([opt.min_abs_iota(equilibrium_state, runtime), max_abs_iota(equilibrium_state, runtime),
                      opt.aspect_ratio(equilibrium_state, runtime), redl_mismatch(equilibrium_state, runtime)])


def objective(u):
    """One three-term free boundary and its pullback; the fit runs on the host.

    As in single_stage_free_boundary_optimization.py, the branch on the status is plain Python, outside
    jit: a trial without a free boundary returns a smooth, finite wall whose derivative is explicit.
    """
    equilibrium_state, plasma, status, _ = solve(u)
    value, aux = objective_terms(equilibrium_state, plasma, u)
    if int(status) == 0:
        return value, (*aux, status)
    wall = 1.0e3 * (1.0 + jnp.sqrt(1.0e-12 + jnp.vdot(u, u)))**2
    return wall, (*map(jnp.zeros_like, aux), status)


def constraints(u):
    equilibrium_state, plasma, _, _ = solve(u)
    return constraint_terms(equilibrium_state, plasma)


value_and_grad_jax = jax.value_and_grad(objective, has_aux=True)
# Every row's gradient in one pass: jax.jacrev batches the state cotangents through one block factorization.
constraint_jacobian = jax.jacrev(constraints)
monitor = opt.OptimizationMonitor()
counts = {"trials": 0, "rejected": 0, "gradients": 0}


def value(u):
    """A trial's value alone: its boundary fit, without the linearization a gradient needs."""
    cost, (_, _, status) = objective(jnp.asarray(u))
    counts["trials"] += 1
    counts["rejected"] += int(status) != 0
    return cost


def value_and_grad(u):
    counts["gradients"] += 1
    (value, (qs_rows, costs, status)), gradient = value_and_grad_jax(jnp.asarray(u))
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows)}
    terms.update(zip(COIL_TERMS, map(float, np.asarray(costs))))
    terms["rejected trial"] = float(value) if int(status) else 0.0
    return monitor.cache_evaluation(u, value, gradient, terms)


# Each row is scaled by its own limit, so SLSQP weighs a violation of the Redl row (values ~1e-3) like one of
# the iota rows (~0.4).
lower = np.array([IOTA_CONSTRAINT, -np.inf, -np.inf, -np.inf])
upper = np.array([np.inf, IOTA_CEILING_CONSTRAINT, ASPECT_CONSTRAINT, REDL_TOLERANCE])
row_scales = np.array([IOTA_CONSTRAINT, IOTA_CEILING_CONSTRAINT, 0.1, REDL_TOLERANCE])
nonlinear = NonlinearConstraint(
    lambda u: np.asarray(constraints(jnp.asarray(u))) / row_scales, lower / row_scales, upper / row_scales,
    jac=lambda u: np.asarray(constraint_jacobian(jnp.asarray(u))) / row_scales[:, None])

###############################################################################
### Run the optimization ######################################################
###############################################################################

print(f"Three-term free boundary + ESSOS at beta {100 * TARGET_BETA:.2f}%: {N_COIL} coil and {N_CURRENT + 1} "
      "current variables, no boundary variables and no mgrid file, SLSQP")
# SLSQP asks for gradients only at its major iterates, so its line-search trials pass fun (a fit alone) and
# only an iterate is linearized and becomes the warm start of the next fits.
free_problem = vj.FunctionProblem.from_functions(
    u0, fun=value, value_and_grad=value_and_grad, evaluation_progress=not ci_smoke)
first = free_problem.compile_value_and_gradient(progress=not ci_smoke, report_interval=10.0)
initial_value = float(first.value)
steps = {"accepted": 0}


def report(u):
    steps["accepted"] += 1  # SLSQP reports an iterate before it asks for its gradient: values only here
    rows = np.asarray(constraints(jnp.asarray(u)))
    print(f"[step {steps['accepted']}] loss {float(objective(jnp.asarray(u))[0]):.6e}; min |iota| {rows[0]:.4f}, "
          f"max |iota| {rows[1]:.4f}, aspect {rows[2]:.4f}, Redl mismatch {rows[3]:.3e}", flush=True)


bound = PARAMETER_BOUND / DESIGN_STEP_SCALE
result = minimize(free_problem.fun, u0, jac=free_problem.grad, method="SLSQP", constraints=[nonlinear],
                  bounds=Bounds(-bound, bound), callback=report, options={"maxiter": MAXITER, "ftol": 1e-12})
u = np.asarray(result.x)
state_x, plasma_x, status_x, worst_x = solve(jnp.asarray(u))
if int(status_x) != 0:
    raise RuntimeError("SLSQP ended on a design without a three-term free boundary; see the trial log")
x = np.asarray(x_from_u(jnp.asarray(u)))
final_value = float(objective(jnp.asarray(u))[0])
lane = config.three_terms.stats
print(f"[solve] {steps['accepted']} SLSQP steps ({result.nit} iterations), {counts['trials']} trials "
      f"({counts['rejected']} rejected) and {counts['gradients']} gradients, {lane['solves']} boundary fits with {lane['boundary_evaluations']} "
      f"evaluations and {lane['linearizations']} linearizations: {result.message}", flush=True)
optimization_seconds = time.perf_counter() - started
runtime_x = im.runtime_from_params(plasma_x, config.implicit)
optimizer_values = plasma_values(state_x, runtime_x)
print("[optimizer] " + ", ".join(f"{k} = {v:.4g}" for k, v in optimizer_values.items())
      + f"; largest interface residual {float(worst_x):.2e}")

###############################################################################
### Check the result against the targets ######################################
###############################################################################

# Re-solve the optimized coils' free boundary, with the optimized current profile, independently
# of the optimizer, cold from the seed boundary, and check every target there, the interface
# residuals and the Redl self-consistency included.
coils_final = coils_from_x(jnp.asarray(x[:N_COIL]))
final_input = input_from_x(x)
final = vj.solve_free_boundary_multigrid(final_input, external_field=field_from_u(jnp.asarray(u)),
                                         three_terms=replace(three_terms, start="deck"))
final_deck = final.three_terms.input  # the free boundary as a fixed-boundary deck
final_context = im.runtime_from_params(im.params_from_input(final_deck), im.make_config(final_deck))
final_values = plasma_values(final.state, final_context)
final_residual = final.three_terms.boundary_residual
wout = vj.wout_from_result(final_input, final)
surface_final = SurfaceRZFourier.from_wout_file(
    vj.write_wout("wout_single_stage_free_boundary_three_terms.nc", wout), nphi=61, ntheta=64)
gamma = np.asarray(coils_final.gamma)
points = np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
coil_distance = min(float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
                    for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota, final_aspect = final_values["min |iota|"], final_values["aspect"]

print("[final] " + ", ".join(f"{k} = {v:.4g}" for k, v in final_values.items()))
optimizer_surface = boundary_surface(state_x, runtime_x)
verification_surface = boundary_surface(final.state, final_context)
surface_gap = float(np.max(np.linalg.norm(np.asarray(optimizer_surface.gamma) - np.asarray(verification_surface.gamma),
                                          axis=-1)))
print(f"The verification's free boundary is within {surface_gap:.4f} m of the optimizer's")
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after {steps['accepted']} SLSQP steps")
print("Interface residuals of the final free boundary: " + ", ".join(
    f"{k} {v:.2e}" for k, v in vars(final_residual).items()))
print(f"Volume-average beta = {100 * final_values['beta']:.4f}% (target {100 * TARGET_BETA:.2f}% "
      f"+- {100 * BETA_TOLERANCE:.0f}%)")
print(f"Redl mismatch = {final_values['Redl mismatch']:.3e} (target <= {REDL_LIMIT:.1e}); CURTOR "
      f"{float(inp.curtor):.4e} -> {float(final_input.curtor):.4e} A")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target <= {ASPECT_LIMIT:.4f})")

checks = (
    ("minimum |iota|", minimum_iota, IOTA_FLOOR, "below"),
    ("maximum |iota|", final_values["max |iota|"], IOTA_CEILING, "above"),
    ("aspect ratio", final_aspect, ASPECT_LIMIT, "above"),
    ("beta", final_values["beta"], TARGET_BETA * (1 - BETA_TOLERANCE), "below"),
    ("beta", final_values["beta"], TARGET_BETA * (1 + BETA_TOLERANCE), "above"),
    ("Redl mismatch", final_values["Redl mismatch"], REDL_LIMIT, "above"),
    ("interface sheet current", final_residual.sheet_current, final.three_terms.config.max_residual, "above"),
    ("minimum coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
    ("minimum coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
    ("maximum curvature", maximum_curvature, CURVATURE_LIMIT, "above"),
)
unmet = [f"{name} {value:.4g} {side} the {limit:.4g} limit"
         for name, value, limit, side in checks
         if (value < limit if side == "below" else value > limit)]
if not final.three_terms.success:
    unmet.append(f"the verification free boundary is not solved: {final.three_terms.message}")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    if ci_smoke:
        print("Smoke mode caps the budget far below what the targets need; exit status 0.")
else:
    print("\nAll stated targets met.")

summary = {
    "example": "single_stage_free_boundary_optimization_three_terms.py", "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1), "coil_fit_iterations": int(coil_fit.nit),
    "accepted_steps": steps["accepted"], "slsqp_iterations": int(result.nit), "stop_reason": result.message,
    "trials": counts["trials"], "rejected_trials": counts["rejected"], "gradients": counts["gradients"],
    "boundary_fits": lane,
    "labelling": LABELLING, "label_weight": three_terms.resolved_label_weight, "phiedge": float(inp.phiedge),
    "target beta": TARGET_BETA, "n0": n0, "T0": T0, "picard iterations": int(picard.iterations),
    "seed curtor": float(inp.curtor), "final curtor": float(final_input.curtor),
    "seed boundary shift": boundary_shift, "seed": seed_values, "optimizer": optimizer_values,
    "verification surface gap": surface_gap,
    "final": {**final_values, "interface residuals": vars(final_residual),
              "coil-surface distance": coil_surface_distance,
              "coil-coil distance": coil_distance, "maximum curvature": maximum_curvature},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_free_boundary_three_terms_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
coils_final.to_json("coils_single_stage_free_boundary_three_terms.json")
print("Wrote wout_single_stage_free_boundary_three_terms.nc and coils_single_stage_free_boundary_three_terms.json")
if unmet and not ci_smoke:
    raise SystemExit(1)
